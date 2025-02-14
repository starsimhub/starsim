"""
Define random-number-safe distributions.
"""
import sciris as sc
import numpy as np
import numba as nb
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['link_dists', 'make_dist', 'dist_list', 'Dists', 'Dist']


def str2int(string, modulo=1_000_000_000):
    """
    Convert a string to an int to use as a random seed; not for the user

    Cannot use Python's built-in hash() since it's randomized for strings. Hashlib
    (sc.sha) is 5x slower than int.from_bytes(string.encode(), byteorder='big'), but should
    only add a couple milliseconds to a typical sim.
    """
    integer = sc.sha(string, asint=True) # Hash the string to an integer
    seed = integer % modulo # Ensure a user-friendly representation user-friendly
    return seed


def link_dists(obj, sim, module=None, overwrite=False, init=False, **kwargs): # TODO: actually link the distributions to the modules! Currently this only does the opposite, but should have mod.dists as well
    """ Link distributions to the sim and the module; used in module.init() and people.init() """
    if module is None and isinstance(obj, ss.Module):
        module = obj
    dists = sc.search(obj, type=Dist, **kwargs)
    for dist in dists.values():
        dist.link_sim(sim, overwrite=overwrite)
        dist.link_module(module, overwrite=overwrite)
        if init: # Usually this is false since usually these are initialized centrally by the sim
            dist.init()
    return


def make_dist(pars=None, **kwargs):
    """ Make a distribution from a dictionary """
    pars = sc.mergedicts(pars, kwargs)
    if 'type' not in pars:
        errormsg = 'To make a distribution from a dict, one of the keys must be "type" to specify the distribution'
        raise ValueError(errormsg)
    dist_type = pars.pop('type')
    if dist_type not in dist_list:
        errormsg = f'"{dist_type}" is not a valid distribution; valid types are {dist_list}'
        raise AttributeError(errormsg)
    dist_class = getattr(ss, dist_type)
    dist = dist_class(**pars) # Actually create the distribution
    return dist


class Dists(sc.prettyobj):
    """ Class for managing a collection of Dist objects """
    def __init__(self, obj=None, *args, base_seed=None, sim=None):
        if len(args): obj = [obj] + list(args)
        self.obj = obj
        self.dists = None
        self.base_seed = base_seed
        if sim is None and obj is not None and isinstance(obj, ss.Sim):
            sim = obj
        self.sim = sim
        self.initialized = False
        return

    def init(self, obj=None, base_seed=None, sim=None, force=False):
        """
        Set the base seed, find and initialize all distributions in an object

        In practice, the object is usually a Sim, but can be anything.
        """
        if base_seed:
            self.base_seed = base_seed
        sim = sim if (sim is not None) else self.sim
        obj = obj if (obj is not None) else self.obj
        if sim is None and obj is not None and isinstance(obj, ss.Sim):
            sim = obj
        if obj is None and sim is not None:
            obj = sim
        if obj is None:
            errormsg = 'Must supply a container that contains one or more Dist objects, typically the sim'
            raise ValueError(errormsg)

        # Do not look for distributions in the people states, since they shadow the "real" states
        skip = dict(
            ids=id(sim.people._states) if sim is not None else None,
            keys='module',
        )

        # Find and initialize the distributions
        self.dists = sc.search(obj, type=Dist, skip=skip, flatten=True)
        for trace,dist in self.dists.items():
            if not dist.initialized or force:
                dist.init(trace=trace, seed=base_seed, sim=sim, force=force)

        # Confirm the seeds are unique
        self.check_seeds()
        self.initialized = True
        return self

    def check_seeds(self):
        """ Check that no two distributions share the same seed """
        checked = dict()
        for dist in self.dists.values():
            seed = dist.seed
            if seed in checked.keys():
                raise DistSeedRepeatError(checked[seed], dist)
            else:
                checked[seed] = dist
        return

    def jump(self, to=None, delta=1, force=False):
        """ Advance all RNGs, e.g. to call "to", by jumping """
        out = sc.autolist()

        # Do not jump if centralized
        if ss.options._centralized:
            return out

        for dist in self.dists.values():
            out += dist.jump(to=to, delta=delta, force=force)
        return out

    def jump_dt(self, ti=None, force=False): # Could this be simplified with jump(), or nice to have the parallel with Dist?
        """
        Advance all RNGs to the next timestep

        Args:
            ti (int): if specified, jump to this timestep (default: current sim timestep)
        """
        out = sc.autolist()

        # Do not jump if centralized
        if ss.options._centralized:
            return out

        for dist in self.dists.values():
            out += dist.jump_dt(ti=ti, force=force)
        return out

    def reset(self):
        """ Reset each RNG """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.reset()
        return out

    def copy_to_module(self, module):
        """ Copy the Sim's Dists object to the specified module """
        matches = {key:dist for key,dist in self.dists.items() if id(dist.module) == id(module)} # Find which dists belong to this module
        if len(matches):
            new = Dists() # Create an empty Dists object
            new.__dict__.update(self.__dict__) # Shallow-copy all values over
            new.obj = module # Replace the module
            new.dists = sc.objdict(matches) # Replace the dists with a shallow copy the matching dists
            module.dists = new # Copy to the module
        else:
            new = None
        return new

class Dist:
    """
    Base class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.

    See ss.dist_list for a full list of supported distributions.

    Although it's possible in theory to define a custom distribution (i.e., not
    one from NumPy or SciPy), in practice this is difficult. The distribution needs
    to have both a way to return random variates (easy), as well as the probability
    point function (inverse CDF). In addition, the distribution must be able to
    take a NumPy RNG as its bit generator. It's easier to just use a default Dist
    (e.g., ss.random()), and then take its output as input (i.e., quantiles) for
    whatever custom distribution you want to create.

    Args:
        dist (rv_generic): optional; a scipy.stats distribution (frozen or not) to get the ppf from
        distname (str): the name for this class of distribution (e.g. "uniform")
        name (str): the name for this particular distribution (e.g. "age_at_death")
        seed (int): the user-chosen random seed (e.g. 3)
        offset (int): the seed offset; will be automatically assigned (based on hashing the name) if None
        strict (bool): if True, require initialization and invalidate after each call to rvs()
        auto (bool): whether to auto-reset the state after each draw
        sim (Sim): usually determined on initialization; the sim to use as input to callable parameters
        module (Module): usually determined on initialization; the module to use as input to callable parameters
        kwargs (dict): parameters of the distribution

    **Examples**::

        dist = ss.Dist(sps.norm, loc=3)
        dist.rvs(10) # Return 10 normally distributed random numbers
    """
    def __init__(self, dist=None, distname=None, name=None, seed=None, offset=None,
                 strict=True, auto=True, sim=None, module=None, debug=False, **kwargs):
        # If a string is provided as "dist" but there's no distname, swap the dist and the distname
        if isinstance(dist, str) and distname is None:
            distname = dist
            dist = None
        self.dist = dist # The type of distribution
        self.distname = distname
        self.name = name
        self.pars = sc.objdict(kwargs) # The user-defined kwargs
        self.seed = seed # Usually determined once added to the container
        self.offset = offset
        self.module = module
        self.sim = sim
        self.slots = None # Created on initialization with a sim
        self.strict = strict
        self.auto = auto
        self.debug = debug

        # Auto-generated
        self.rvs_func = None # The default function to call in make_rvs() to generate the random numbers
        self.dynamic_pars = None # Whether or not the distribution has array or callable parameters
        self._pars = None # Validated and transformed (if necessary) parameters
        self._n = None # Internal variable to keep track of "n" argument (usually size)
        self._size = None # Internal variable to keep track of actual number of random variates asked for
        self._uids = None # Internal variable to track currently-in-use UIDs
        self._slots = None # Internal variable to track currently-in-use slots

        # History and random state
        self.dt_jump_size = 1000 # How much to advance the RNG for each timestep (must be larger than the number times dist is called per timestep)
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self.ind = 0 # The index of the RNG (usually updated on each timestep)
        self.called = 0 # The number of times the distribution has been called
        self.history = [] # Previous states
        self.ready = True
        self.initialized = False
        if not strict: # Otherwise, wait for a sim
            self.init()
        return

    def __repr__(self):
        """ Custom display to show state of object """
        j = sc.dictobj(self.to_json())
        string = f'ss.{j.classname}({j.tracestr}, {j.diststr}pars={j.pars})'
        return string

    def disp(self):
        """ Return full display of object """
        return sc.pr(self)

    def show_state(self, output=False):
        """ Show the state of the object """
        keys = ['pars', 'trace', 'offset', 'seed', 'ind', 'called', 'ready', 'state_int']
        data = {key:getattr(self,key) for key in keys}
        s = f'{self}'
        for key,val in data.items():
            s += f'\n{key:9s} = {val}'
        if output:
            return data
        else:
            print(s)
            return

    def __call__(self, n=1):
        """ Alias to self.rvs() """
        return self.rvs(n=n)

    def set(self, *args, dist=None, **kwargs):
        """ Set (change) the distribution type, or one or more parameters of the distribution """
        if dist:
            self.dist = dist
            self.process_dist()
        if args:
            if kwargs:
                errormsg = f'You can supply args or kwargs, but not both (args={len(args)}, kwargs={len(kwargs)})'
                raise ValueError(errormsg)
            else: # Convert from args to kwargs
                parkeys = list(self.pars.keys())
                for i,arg in enumerate(args):
                    kwargs[parkeys[i]] = arg
        if kwargs:
            self.pars.update(kwargs)
            self.process_pars(call=False)
        return

    @property
    def bitgen(self):
        try:    return self.rng._bit_generator
        except: return None

    @property
    def state(self):
        """ Get the current state """
        try:    return self.bitgen.state
        except: return None

    @property
    def state_int(self):
        """ Get the integer corresponding to the current state """
        try:    return self.state['state']['state']
        except: return None

    def get_state(self):
        """ Return a copy of the state """
        return self.state.copy()

    def make_history(self, reset=False):
        """ Store the current state in history """
        if reset:
            self.history = []
        self.history.append(self.get_state()) # Store the initial state
        return

    def reset(self, state=0):
        """
        Restore state, allowing the same numbers to be resampled

        Use 0 for original state, -1 for most recent state.

        **Example**::

            dist = ss.random(seed=5).init()
            r1 = dist(5)
            r2 = dist(5)
            dist.reset(-1)
            r3 = dist(5)
            dist.reset(0)
            r4 = dist(5)
            assert all(r1 != r2)
            assert all(r2 == r3)
            assert all(r4 == r1)
        """
        if not isinstance(state, dict):
            state = self.history[state]
        self.rng._bit_generator.state = state.copy()
        self.ready = True
        return self.state

    def jump(self, to=None, delta=1, force=False):
        """ Advance the RNG, e.g. to timestep "to", by jumping """

        # Do not jump if centralized # TODO: remove
        if ss.options._centralized:
            return self.state

        # Validation
        jumps = to if (to is not None) else self.ind + delta
        if self.ind >= jumps and not force:
            errormsg = f'You tried to jump the distribution "{self}" to state {jumps}, but the ' \
                       f'RNG state is already at state {self.ind}, meaning you will draw the same ' \
                        'random numbers twice. If you are sure you want to do this, set force=True.'
            raise DistSeedRepeatError(msg=errormsg)

        # Do the jumping
        self.ind = jumps
        self.reset() # First reset back to the initial state (used in case of different numbers of calls)
        if jumps: # Seems to randomize state if jumps=0
            self.bitgen.state = self.bitgen.jumped(jumps=jumps).state # Now take "jumps" number of jumps
        return self.state

    def jump_dt(self, ti=None, force=False):
        """
        Automatically jump on the next value of dt

        Args:
            ti (int): if specified, jump to this timestep (default: current module timestep plus one)
        """
        if ti is None:
            ti = self.module.t.ti + 1
        to = self.dt_jump_size*ti
        return self.jump(to=to, force=force)

    def init(self, trace=None, seed=None, module=None, sim=None, slots=None, force=False):
        """ Calculate the starting seed and create the RNG """

        if self.initialized is True and not force: # Don't warn if we have a partially initialized distribution
            msg = f'Distribution {self} is already initialized, use force=True if intentional'
            ss.warn(msg)

        # Calculate the offset (starting seed)
        self.process_seed(trace, seed)

        # Create the actual RNG
        if ss.options._centralized:
            self.rng = np.random.mtrand._rand # If _centralized, return the centralized numpy random number instance
        else:
            self.rng = np.random.default_rng(seed=self.seed)
        self.make_history(reset=True)

        # Handle the sim, module, and slots
        self.link_sim(sim)
        self.link_module(module)
        if slots is None and self.sim is not None:
            try:
                slots = self.sim.people.slot
            except Exception as E:
                warnmsg = f'Could not extract slots from sim object, is this an error?\n{E}'
                ss.warn(warnmsg)
        if slots is not None:
            self.slots = slots

        # Initialize the distribution and finalize
        self.process_dist()
        self.process_pars(call=False)
        self.ready = True
        none_dict = dict(trace=self.trace, sim=self.sim, slots=self.slots)
        none_dict = {k:v for k,v in none_dict.items() if v is None}
        if len(none_dict):
            self.initialized = 'partial'
            if self.strict:
                errormsg = f'Distribution {self} is not fully initialized, the following inputs are None:\n{none_dict.keys()}\nThis distribution may not produce valid random numbers.'
                raise RuntimeError(errormsg)
        else:
            self.initialized = True
        return self

    def link_sim(self, sim=None, overwrite=False):
        """ Shortcut for linking the sim, only overwriting an existing one if overwrite=True; not for the user """
        if (not self.sim or overwrite) and sim is not None:
            self.sim = sim
        return

    def link_module(self, module=None, overwrite=False):
        """ Shortcut for linking the module """
        if (not self.module or overwrite) and module is not None:
            self.module = module
        return

    def process_seed(self, trace=None, seed=None):
        """ Obtain the seed offset by hashing the path to this distribution; not for the user """
        unique_name = trace or self.trace or self.name
        if unique_name:
            if not self.name:
                self.name = unique_name
            self.trace = unique_name
            self.offset = str2int(unique_name) # Key step: hash the path to the distribution
        else:
            self.offset = self.offset or 0
        self.seed = self.offset + (seed or self.seed or 0)
        return

    def process_dist(self):
        """ Ensure the distribution works; not for the user """

        # Handle a SciPy distribution, if provided
        if self.dist is not None:

            # Pull out parameters of an already-frozen distribution
            if isinstance(self.dist, sps._distn_infrastructure.rv_frozen):
                if not self.initialized: # Don't do this more than once
                    self.pars = sc.dictobj(sc.mergedicts(self.pars, self.dist.kwds))

            # Convert to a frozen distribution
            if isinstance(self.dist, sps._distn_infrastructure.rv_generic):
                spars = self.process_pars(call=False)
                spars.pop('dtype', None) # Not a valid arg for SciPy distributions
                self.dist = self.dist(**spars)

            # Override the default random state with the correct one
            self.dist.random_state = self.rng

        # Set the default function for getting the rvs
        if self.distname is not None and hasattr(self.rng, self.distname): # Don't worry if it doesn't, it's probably being manually overridden
            self.rvs_func = self.distname # e.g. self.rng.uniform -- can't use the actual function because can become linked to the wrong RNG

        return

    def process_size(self, n=1):
        """ Handle an input of either size or UIDs and calculate size, UIDs, and slots; not for the user """
        if np.isscalar(n) or isinstance(n, tuple):  # If passing a non-scalar size, interpret as dimension rather than UIDs iff a tuple
            uids = None
            slots = None
            size = n
        else:
            uids = ss.uids(n)
            if len(uids):
                if self.slots is None:
                    errormsg = f'Could not find any slots in {self}. Did you remember to initialize the distribution with the sim?'
                    raise ValueError(errormsg)
                slots = self.slots[uids]
                if len(slots): # Handle case where uids is boolean
                    size = slots.max() + 1
                else:
                    size = 0
            else:
                slots = np.array([])
                size = 0

        self._n = n
        self._size = size
        self._uids = uids
        self._slots = slots
        return size, slots

    def process_pars(self, call=True):
        """ Ensure the supplied dist and parameters are valid, and initialize them; not for the user """
        self._timepar = None # Time rescalings need to be done after distributions are calculated; store the correction factor here
        self._pars = sc.cp(self.pars) # The actual keywords; shallow copy, modified below for special cases
        if call:
            self.call_pars() # Convert from function to values if needed
        spars = self.sync_pars() # Synchronize parameters between the NumPy and SciPy distributions
        return spars

    def preprocess_timepar(self, key, timepar):
        """ Method to handle how timepars are processed; not for the user. By default, scales the output of the distribution. """
        if self._timepar is None: # Store this here for later use
            self._timepar = sc.dcp(timepar) # Make a copy to avoid modifying the original
        elif timepar.factor != self._timepar.factor:
            errormsg = f'Cannot have time parameters in the same distribution with inconsistent unit/dt values: {self._pars}'
            raise ValueError(errormsg)
        self._pars[key] = timepar.v # Use the raw value, since it could be anything (including a function)
        return timepar.v # Also use this for the rest of the loop

    def convert_callable(self, key, val, size, uids):
        """ Method to handle how callable parameters are processed; not for the user """
        size_par = sc.ifelse(uids, size, ss.uids()) # Allow none size
        out = val(self.module, self.sim, size_par)
        val = np.asarray(out) # Necessary since FloatArrs don't allow slicing # TODO: check if this is correct
        self._pars[key] = val
        return val

    def call_par(self, key, val, size, uids):
        """ Check if this parameter needs to be called to be turned into an array; not for the user """
        if isinstance(val, ss.TimePar): # If it's a time parameter, transform it to a float now
            val = self.preprocess_timepar(key, val)
        if callable(val) and not isinstance(val, type): # If the parameter is callable, then call it (types can appear as callable)
            val = self.convert_callable(key, val, size, uids)
        return val

    def call_pars(self):
        """ Check if any parameters need to be called to be turned into arrays; not for the user """

        # Initialize
        size, uids = self._size, self._uids
        if self.dynamic_pars != False: # Allow "False" to prevent ever using dynamic pars (used in ss.choice())
            self.dynamic_pars = None

        # Check each parameter
        for key,val in self._pars.items():
            val = self.call_par(key, val, size, uids)

            # If it's iterable and UIDs are provided, then we need to use array-parameter logic
            if self.dynamic_pars is None and np.iterable(val) and uids is not None:
                self.dynamic_pars = True
        return

    def sync_pars(self):
        """ Perform any necessary synchronizations or transformations on distribution parameters; not for the user """
        self.update_dist_pars()
        return self._pars

    def update_dist_pars(self, pars=None):
        """ Update SciPy distribution parameters; not for the user """
        if self.dist is not None:
            pars = pars if pars is not None else self._pars
            self.dist.kwds = pars
        return

    def rand(self, size):
        """ Simple way to get simple random numbers """
        return self.rng.random(size, dtype=ss.dtypes.float) # Up to 2x faster with float32

    def make_rvs(self):
        """ Return default random numbers for scalar parameters; not for the user """
        if self.rvs_func is not None:
            rvs_func = getattr(self.rng, self.rvs_func) # Can't store this because then it references the wrong RNG after copy
            rvs = rvs_func(size=self._size, **self._pars)
        elif self.dist is not None:
            rvs = self.dist.rvs(self._size)
        else:
            errormsg = 'Dist.rvs() failed: no valid NumPy/SciPy function found in this Dist. Has it been created and initialized correctly?'
            raise ValueError(errormsg)
        return rvs

    def ppf(self, rands):
        """ Return default random numbers for array parameters; not for the user """
        rvs = self.dist.ppf(rands)
        return rvs

    def postprocess_timepar(self, rvs):
        """ Scale random variates after generation; not for the user """
        timepar = self._timepar # Shorten
        self._timepar = None # Remove the timepar which is no longer needed
        timepar.v = rvs # Replace the base value with the random variates
        timepar.update_cached() # Recalculate the factor and values with the time scaling
        rvs = timepar.values # Replace the rvs with the scaled version
        if isinstance(rvs, np.ndarray): # This can be false when converting values for a Bernoulli distribution (in which case rvs are actually dist parameters)
            rvs = rvs.astype(rvs.dtype) # Replace the random variates with the scaled version, and preserve type
        return rvs

    def rvs(self, n=1, reset=False):
        """
        Get random variates -- use this!

        Args:
            n (int/tuple/arr): if an int or tuple, return this many random variates; if an array, treat as UIDs
            reset (bool): whether to automatically reset the random number distribution state after being called
        """
        # Check for readiness
        if not self.initialized:
            raise DistNotInitializedError(self)
        if not self.ready and self.strict and not ss.options._centralized:
            raise DistNotReadyError(self)

        # Figure out size, UIDs, and slots
        size, slots = self.process_size(n)

        # Check if size is 0, then we can return
        if size == 0:
            return np.array([], dtype=ss.dtypes.int) # int dtype allows use as index, e.g. when filtering
        elif isinstance(size, ss.uids) and self.initialized == 'partial': # This point can be reached if and only if strict=False and UIDs are used as input
            errormsg = f'Distribution {self} is only partially initialized; cannot generate random numbers to match UIDs'
            raise ValueError(errormsg)

        # Store the state
        self.make_history() # Store the pre-call state

        # Check if any keywords are callable -- parameters shouldn't need to be reprocessed otherwise
        self.process_pars()

        # Actually get the random numbers
        if self.dynamic_pars:
            rands = self.rand(size)[slots] # Get random values
            rvs = self.ppf(rands) # Convert to actual values via the PPF
        else:
            rvs = self.make_rvs() # Or, just get regular values
            if self._slots is not None:
                rvs = rvs[self._slots]

        # Scale by time if needed
        if self._timepar is not None:
            rvs = self.postprocess_timepar(rvs)

        # Tidy up
        self.called += 1
        if reset:
            self.reset(-1)
        elif self.auto: # TODO: check
            self.jump()
        elif self.strict:
            self.ready = False
        if self.debug:
            tistr   = f'on ti={self.module.ti} ' if self.module else ''
            sizestr = f'with size={self._size}, '
            slotstr = f'Σ(slots)={self._slots.sum()}, ' if self._slots else '<no slots>, '
            rvstr   = f'Σ(rvs)={rvs.sum():0.2f}, |rvs|={rvs.mean():0.4f}'
            pre_state = str(self.history[-1]['state']['state'])[-5:]
            post_state = str(self.state_int)[-5:]
            statestr = f"state {pre_state}→{post_state}"
            print(f'Debug: {self} called {tistr}{sizestr}{slotstr}{rvstr}, {statestr}')
            assert pre_state != post_state # Always an error if the state doesn't change after drawing random numbers

        return rvs

    def to_json(self):
        """ Return a dictionary representation of the Dist """
        tracestr = '<no trace>' if self.trace is None else f"{self.trace}"
        classname = self.__class__.__name__
        diststr = ''
        if classname == 'Dist':
            if self.dist is not None:
                try:
                    diststr = f'dist={self.dist.name}, '
                except:
                    try: # What is wrong with you, SciPy -- after initialization, it moves here
                        diststr = f'dist={self.dist.dist.name}, '
                    except:
                        diststr = f'dist={type(self.dist)}'
            elif self.distname is not None:
                diststr = f'dist={self.distname}, '
        out = dict(
            type = 'Dist',
            classname = classname,
            tracestr = tracestr,
            diststr = diststr,
            pars = dict(self.pars),
        )
        return out


    def plot_hist(self, n=1000, bins=None, fig_kw=None, hist_kw=None):
        """ Plot the current state of the RNG as a histogram """
        plt.figure(**sc.mergedicts(fig_kw))
        rvs = self.rvs(n)
        self.reset(-1) # As if nothing ever happened
        plt.hist(rvs, bins=bins, **sc.mergedicts(hist_kw))
        plt.title(str(self))
        plt.xlabel('Value')
        plt.ylabel(f'Count ({n} total)')
        return rvs


#%% Specific distributions

# Add common distributions so they can be imported directly; assigned to a variable since used in help messages
dist_list = ['random', 'uniform', 'normal', 'lognorm_ex', 'lognorm_im', 'expon', 'poisson', 'nbinom',
             'weibull', 'gamma', 'constant', 'randint', 'rand_raw', 'bernoulli', 'choice', 'histogram']
__all__ += dist_list
__all__ += ['multi_random'] # Not a dist in the same sense as the others (e.g. same tests would fail)


class random(Dist):
    """ Random distribution, with values on the interval (0, 1) """
    def __init__(self, **kwargs):
        super().__init__(distname='random', dtype=ss.dtypes.float, **kwargs)
        return

    def ppf(self, rands):
        return rands


class uniform(Dist):
    """
    Uniform distribution, values on interval (low, high)

    Args:
        low (float): the lower bound of the distribution (default 0.0)
        high (float): the upper bound of the distribution (default 1.0)
    """
    def __init__(self, low=None, high=None, **kwargs):
        if high is None and low is not None: # One argument, swap
            high = low
            low = 0.0
        if low is None:
            low = 0.0
        if high is None:
            high = 1.0
        super().__init__(distname='uniform', low=low, high=high, **kwargs)
        return

    def make_rvs(self):
        """ Specified here because uniform() doesn't take a dtype argument """
        p = self._pars
        rvs = self.rand(self._size) * (p.high - p.low) + p.low
        return rvs

    def ppf(self, rands):
        p = self._pars
        rvs = rands * (p.high - p.low) + p.low
        return rvs


class normal(Dist):
    """
    Normal distribution

    Args:
        loc (float): the mean of the distribution (default 0.0)
        scale (float) the standard deviation of the distribution (default 1.0)

    """
    def __init__(self, loc=0.0, scale=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='normal', dist=sps.norm, loc=loc, scale=scale, **kwargs)
        return


class lognorm_im(Dist):
    """
    Lognormal distribution, parameterized in terms of the "implicit" (normal)
    distribution, with mean=loc and std=scale (see lognorm_ex for comparison).

    Note: the "loc" parameter here does *not* correspond to the mean of the resulting
    random variates!

    Args:
        mean (float): the mean of the underlying normal distribution (not this distribution) (default 0.0)
        sigma (float): the standard deviation of the underlying normal distribution (not this distribution) (default 1.0)

    **Example**::

        ss.lognorm_im(mean=2, sigma=1, strict=False).rvs(1000).mean() # Should be roughly 10
    """
    def __init__(self, mean=0.0, sigma=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='lognormal', dist=sps.lognorm, mean=mean, sigma=sigma, **kwargs)
        return

    def sync_pars(self, call=True):
        """ Translate between NumPy and SciPy parameters """
        if call:
            self.call_pars()
        p = self._pars
        spars = sc.dictobj()
        spars.s = p.sigma
        spars.scale = np.exp(p.mean)
        spars.loc = 0
        self.update_dist_pars(spars)
        return spars

    def preprocess_timepar(self, key, val):
        """ Not valid since incorrect time units """
        errormsg = f'Cannot use timepars with a lognorm_im distribution ({self}) since its units are not time. Use lognorm_ex instead.'
        raise NotImplementedError(errormsg)


class lognorm_ex(Dist):
    """
    Lognormal distribution, parameterized in terms of the "explicit" (lognormal)
    distribution, with mean=mean and std=std for this distribution (see lognorm_im for comparison).
    Note that a mean ≤ 0.0 is impossible, since this is the parameter of the distribution
    after the log transform.

    Args:
        mean (float): the mean of this distribution (not the underlying distribution) (default 1.0)
        std (float): the standard deviation of this distribution (not the underlying distribution) (default 1.0)

    **Example**::

        ss.lognorm_ex(mean=2, std=1, strict=False).rvs(1000).mean() # Should be close to 2
    """
    def __init__(self, mean=1.0, std=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='lognormal', dist=sps.lognorm, mean=mean, std=std, **kwargs)
        return

    def convert_ex_to_im(self):
        """
        Lognormal distributions can be specified in terms of the mean and standard
        deviation of the "explicit" lognormal distribution, or the "implicit" normal distribution.
        This function converts the parameters from the lognormal distribution to the
        parameters of the underlying (implicit) distribution, which are the form expected by NumPy's
        and SciPy's lognorm() distributions.
        """
        self.call_pars() # Since can't work with functions
        p = self._pars
        mean = p.pop('mean')
        std = p.pop('std')
        if np.isscalar(mean) and mean <= 0:
            errormsg = f'Cannot create a lognorm_ex distribution with mean≤0 (mean={mean}); did you mean to use lognorm_im instead?'
            raise ValueError(errormsg)
        std2 = std**2
        mean2 = mean**2
        sigma_im = np.sqrt(np.log(std2/mean2 + 1)) # Computes std for the underlying normal distribution
        mean_im  = np.log(mean2 / np.sqrt(std2 + mean2)) # Computes the mean of the underlying normal distribution
        p.mean = mean_im
        p.sigma = sigma_im
        return mean_im, sigma_im

    def sync_pars(self):
        """ Convert from overlying to underlying parameters, then translate to SciPy """
        self.convert_ex_to_im()
        spars = lognorm_im.sync_pars(self, call=False) # Borrow sync_pars from lognorm_im
        return spars


class expon(Dist):
    """
    Exponential distribution

    Args:
        scale (float): the scale of the distribution (default 1.0)

    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(distname='exponential', dist=sps.expon, scale=scale, **kwargs)
        return


class poisson(Dist): # TODO: does not currently scale correctly with dt
    """
    Poisson distribution

    Args:
        lam (float): the scale of the distribution (default 1.0)
    """
    def __init__(self, lam=1.0, **kwargs):
        super().__init__(distname='poisson', dist=sps.poisson, lam=lam, **kwargs)
        return

    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
        spars = dict(mu=self._pars.lam)
        self.update_dist_pars(spars)
        return spars

    def preprocess_timepar(self, key, timepar):
        """ Try to update the timepar before calculating array parameters, but raise an exception if this isn't possible """
        try:
            timepar.update_cached()
        except Exception as E:
            errormsg = f'Could not process timepar {timepar} for {self}. Note that Poisson distributions are not compatible with both callable parameters and timepars, since this would change the shape in an unknowable way.'
            raise ValueError(errormsg) from E

        self._pars[key] = timepar.values # Use the raw value, since it could be anything (including a function)
        return timepar.values # Also use this for the rest of the loop


class nbinom(Dist):
    """
    Negative binomial distribution

    Args:
        n (float): the number of successes, > 1 (default 1.0)
        p (float): the probability of success in [0,1], (default 0.5)

    """
    def __init__(self, n=1, p=0.5, **kwargs):
        super().__init__(distname='negative_binomial', dist=sps.nbinom, n=n, p=p, **kwargs)
        return


class randint(Dist):
    """
    Random integer distribution, on the interval [low, high)

    Args:
        low (int): the lower bound of the distribution (default 0)
        high (int): the upper bound of the distribution (default of maximum integer size: 9,223,372,036,854,775,807)
        allow_time (bool): allow time parameters to be specified as high/low values (disabled by default since introduces rounding error)
    """
    def __init__(self, *args, low=None, high=None, dtype=ss.dtypes.rand_int, allow_time=False, **kwargs):
        # Handle input arguments # TODO: reconcile with how this is handled in uniform()
        self.allow_time = allow_time
        if len(args):
            if len(args) == 1:
                high = args[0]
            elif len(args) == 2:
                low,high = args
            else:
                errormsg = f'ss.randint() takes one or two arguments, not {len(args)}'
                raise ValueError(errormsg)
        if low is None:
            low = 0
        if high is None:
            high = np.iinfo(ss.dtypes.rand_int).max

        if ss.options._centralized: # randint because we're accessing via numpy.random
            super().__init__(distname='randint', low=low, high=high, dtype=dtype, **kwargs)
        else: # integers instead of randint because interfacing a numpy.random.Generator
            super().__init__(distname='integers', low=low, high=high, dtype=dtype, **kwargs)
        return

    def ppf(self, rands):
        p = self._pars
        rvs = rands * (p.high + 1 - p.low) + p.low
        rvs = rvs.astype(self.dtype)
        return rvs

    def preprocess_timepar(self, key, timepar):
        """ Not valid due to a rounding error """
        if self.allow_time:
            return super().preprocess_timepar(key, timepar)
        else:
            errormsg = f'Cannot use timepars with a randint distribution ({self}) since the values may be rounded incorrectly. Use uniform() instead and convert to int yourself, or set allow_time=True if you really want to do this.'
            raise NotImplementedError(errormsg)


class rand_raw(Dist):
    """
    Directly sample raw integers (uint64) from the random number generator.
    Typicaly only used with ss.combine_rands().
    """
    def make_rvs(self):
        if ss.options._centralized:
            return self.rng.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=self._size)
        else:
            return self.bitgen.random_raw(self._size) # TODO: figure out how to make accept dtype, or check speed


class weibull(Dist):
    """
    Weibull distribution (specifically, scipy.stats.weibull_min)

    Args:
        c (float): the shape parameter, sometimes called k (default 1.0)
        loc (float): the location parameter, which shifts the position of the distribution (default 0.0)
        scale (float): the scale parameter, sometimes called λ (default 1.0)
    """
    def __init__(self, c=1.0, loc=0.0, scale=1.0, **kwargs):
        super().__init__(distname='weibull', dist=sps.weibull_min, c=c, loc=loc, scale=scale, **kwargs)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


class gamma(Dist):
    """
    Gamma distribution (specifically, scipy.stats.gamma)

    Args:
        a (float): the shape parameter, sometimes called k (default 1.0)
        loc (float): the location parameter, which shifts the position of the distribution (default 0.0)
        scale (float): the scale parameter, sometimes called θ (default 1.0)
    """
    def __init__(self, a=1.0, loc=0.0, scale=1.0, **kwargs):
        super().__init__(distname='gamma', dist=sps.gamma, a=a, loc=loc, scale=scale, **kwargs)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


class constant(Dist):
    """
    Constant (delta) distribution: equivalent to np.full()

    Args:
        v (float): the value to return
    """
    def __init__(self, v=0.0, **kwargs):
        super().__init__(distname='const', v=v, **kwargs)
        return

    def make_rvs(self):
        return np.full(self._size, self._pars.v)

    def ppf(self, rands): # NB: don't actually need to use random numbers here, but not worth the complexity of avoiding this
        return np.full(rands.shape, self._pars.v)


class bernoulli(Dist):
    """
    Bernoulli distribution: return True or False with the specified probability (which can be an array)

    Unlike other distributions, Bernoulli distributions have a filter() method,
    which returns elements of the array that return True.

    Args:
        p (float): the probability of returning True (default 0.5)
    """
    def __init__(self, p=0.5, **kwargs):
        super().__init__(distname='bernoulli', p=p, **kwargs)
        return

    def make_rvs(self):
        rvs = self.rand(self._size) < self._pars.p # 3x faster than using rng.binomial(1, p, size)
        return rvs

    def ppf(self, rands):
        rvs = rands < self._pars.p
        return rvs

    def filter(self, uids=None, both=False):
        """ Return UIDs that correspond to True, or optionally return both True and False """
        if uids is None:
            uids = self.sim.people.auids # All active UIDs
        elif isinstance(uids, (ss.BoolArr, ss.IndexArr)):
            uids = uids.uids

        bools = self.rvs(uids)
        if both:
            return uids[bools], uids[~bools]
        else:
            return uids[bools]

    def split(self, uids=None):
        """ Alias to filter(uids, both=True) """
        return self.filter(uids=uids, both=True)

    def call_par(self, key, val, size, uids):
        """ Reverse the usual order of processing so callable is processed first, and then the timepar conversion """
        is_timepar = isinstance(val, ss.TimePar)

        if is_timepar: # If it's a time parameter, pull out the value
            timepar = sc.dcp(val) # Rename to make more sense within the context of this method
            val = timepar.v # Pull out the base value; we'll deal with the transformation later
            self._timepar = timepar # This is used, then destroyed, by postprocess_timepar() below
            if isinstance(timepar, ss.dur): # Validation
                errormsg = f'Bernoulli distributions can only be used with ss.time_prob() or ss.rate(), not {timepar}'
                raise TypeError(errormsg)

        # As normal: if the parameter is callable, then call it (types can appear as callable)
        if callable(val) and not isinstance(val, type):
            val = self.convert_callable(key, val, size, uids)

        # Process as a timepar
        if is_timepar:
            val = self.postprocess_timepar(val) # Note: this is processing the parameter rather than the rvs as usual

        # Store in the parameters and return
        self._pars[key] = val
        return val


class choice(Dist):
    """
    Random choice between discrete options (note: dynamic parameters not supported)

    Args:
        a (int or array): the number of choices, or the choices themselves (default 2)
        p (array): if supplied, the probability of each choice (default, 1/a for a choices)

    **Examples**::

        # Simulate 10 die rolls
        ss.choice(6, strict=False)(10) + 1

        # Choose between specified options each with a specified probability (must sum to 1)
        ss.choice(a=[30, 70], p=[0.3, 0.7], strict=False)(10)

    Note: although Bernoulli trials can be generated using a=2, it is much faster
    to use ss.bernoulli() instead.
    """
    def __init__(self, a=2, p=None, **kwargs):
        super().__init__(distname='choice', a=a, p=p, **kwargs)
        self.dynamic_pars = False # Set to false since array arguments don't imply dynamic pars here
        return

    def ppf(self, rands):
        """ Shouldn't actually be needed since dynamic pars not supported """
        pars = self._pars
        if np.isscalar(pars.a):
            pars.a = np.arange(pars.a)
        pcum = np.cumsum(pars.p)
        inds = np.searchsorted(pcum, rands)
        rvs = pars.a[inds]
        return rvs

    def preprocess_timepar(self, key, timepar):
        """ Not valid since does not scale with time """
        errormsg = f'Cannot use timepars with a choice distribution ({self}) since its units are not time. Convert output to time units instead.'
        raise NotImplementedError(errormsg)


class histogram(Dist):
    """
    Sample from a histogram with defined bins

    Note: unlike other distributions, the parameters of this distribution can't
    be modified after creation.

    Args:
        values (array): the probability (or count) of each bin
        bins (array): the edges of each bin
        density (bool): treat the histogram as a density instead of counts; only matters with unequal bin widths, see numpy.histogram and scipy.stats.rv_histogram for more information
        data (array): if supplied, compute the values and bin edges using this data and np.histogram() instead

    Note: if the length of bins is equal to the length of values, they will be
    interpreted as left bin edges, and one additional right-bin edge will be added
    based on the difference between the last two bins (e.g. if the last two bins are
    40 and 50, the final right edge will be added at 60). If no bins are supplied,
    then they will be created as integers matching the length of the values.

    The values can be supplied in either normalized (sum to 1) or un-normalized
    format.

    **Examples**::

        # Sample from an age distribution
        age_bins = [0,    10,  20,  40,  65, 100]
        age_vals = [0.1, 0.1, 0.3, 0.3, 0.2]
        h1 = ss.histogram(values=age_vals, bins=age_bins, strict=False)
        h1.plot_hist()

        # Create a histogram from data
        data = np.random.randn(10_000)*2+5
        h2 = ss.histogram(data=data, strict=False)
        h2.plot_hist(bins=100)
    """
    def __init__(self, values=None, bins=None, density=False, data=None, **kwargs):
        if data is not None:
            if values is not None:
                errormsg = 'You can supply values or data, but not both'
                raise ValueError(errormsg)
            kw = {'bins':bins} if bins is not None else {} # Hack to not overwrite default bins value
            values, bins = np.histogram(data, **kw)
        else:
            if values is None:
                values = [1.0] # Uniform distribution
            values = sc.toarray(values)
            if bins is None:
                bins = np.arange(len(values)+1)
            bins = sc.toarray(bins)
        if len(bins) == len(values): # Append a final bin, if necessary
            delta = bins[-1] - bins[-2]
            bins = np.append(bins, bins[-1]+delta)
        vsum = values.sum()
        if vsum != 1.0:
            values = values / vsum
        dist = sps.rv_histogram((values, bins), density=density) # Create the SciPy distribution
        super().__init__(dist=dist, distname='histogram', **kwargs)
        self.dynamic_pars = False # Set to false since array arguments don't imply dynamic pars here
        return


class multi_random(sc.prettyobj):
    """
    A class for holding two or more ss.random() distributions, and generating
    random numbers linked to each of them. Useful for e.g. pairwise transmission
    probabilities.

    See ss.combine_rands() for the manual version; in almost all cases this class
    should be used instead.

    Usage:
        multi = ss.multi_random('source', 'target')
        rvs = multi.rvs(source_uids, target_uids)
    """
    def __init__(self, names, *args, **kwargs):
        names = sc.mergelists(names, args)
        self.dists = [ss.random(name=name, **kwargs) for name in names]
        return

    def __len__(self):
        return len(self.dists)

    def init(self, *args, **kwargs):
        """ Not usually needed since each dist will handle this automatically; for completeness only """
        for dist in self.dists: dist.init(*args, **kwargs)
        return

    def reset(self, *args, **kwargs):
        """ Not usually needed since each dist will handle this automatically; for completeness only """
        for dist in self.dists: dist.reset(*args, **kwargs)
        return

    def jump(self, *args, **kwargs):
        """ Not usually needed since each dist will handle this automatically; for completeness only """
        for dist in self.dists: dist.jump(*args, **kwargs)
        return

    @staticmethod
    @nb.njit(fastmath=True, parallel=False, cache=True) # Numba is 3x faster, but disabling parallel for efficiency
    def combine_rvs(rvs_list, int_type, int_max):
        """ Combine inputs into one number """
        # Combine using bitwise-or
        rand_ints = rvs_list[0].view(int_type)
        for rand_floats in rvs_list[1:]:
            rand_ints2 = rand_floats.view(int_type)
            rand_ints = np.bitwise_xor(rand_ints*rand_ints2, rand_ints-rand_ints2)

        # Normalize
        rvs = rand_ints / int_max
        return rvs

    def rvs(self, *args):
        """ Get random variates from each of the underlying distributions and combine them efficiently """
        # Validation
        n_args = len(args)
        n_dists = len(self)
        if n_args != len(self):
            errormsg = f'Number of UID lists supplied ({n_args}) does not match number of distributions ({n_dists})'
            raise ValueError(errormsg)

        rvs_list = [dist.rvs(arg) for dist,arg in zip(self.dists, args)]
        int_type = ss.dtypes.rand_uint
        int_max = np.iinfo(int_type).max
        rvs = self.combine_rvs(rvs_list, int_type, int_max)
        return rvs




#%% Dist exceptions

class DistNotInitializedError(RuntimeError):
    """ Raised when Dist object is called when not initialized. """
    def __init__(self, dist=None, msg=None):
        if msg is None:
            msg = f'{dist} has not been initialized; please set strict=False when creating the distribution, or call dist.init()'
        super().__init__(msg)
        return


class DistNotReadyError(RuntimeError):
    """ Raised when a Dist object is called without being ready. """
    def __init__(self, dist=None, msg=None):
        if msg is None:
            msg = f'{dist} is not ready. This is likely caused by calling a distribution multiple times in a single step. Call dist.jump() to reset.'
        super().__init__(msg)
        return


class DistSeedRepeatError(RuntimeError):
    """ Raised when a Dist object shares a seed with another """
    def __init__(self, dist1=None, dist2=None, msg=None):
        if msg is None:
            msg = f'A common seed was found between {dist1} and {dist2}. This is likely caused by incorrect initialization of the parent Dists object.'
        super().__init__(msg)
        return
