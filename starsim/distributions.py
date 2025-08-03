"""
Define random-number-safe distributions.
"""
import inspect
import sciris as sc
import numpy as np
import numba as nb
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as plt
ss_int_ = ss.dtypes.int

__all__ = ['link_dists', 'make_dist', 'dist_list', 'scale_types', 'Dists', 'Dist']


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
    return dists


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
        if ss.options.single_rng:
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
        if ss.options.single_rng:
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
            module.setattribute('dists', new) # Copy to the module
        else:
            new = None
        return new


class scale_types(sc.prettyobj):
    """ Define how distributions scale

    Distributions scale in different ways, such as converting between time units.
    Some distributions can't be scaled at all (e.g. `ss.beta_dist()` or `ss.choice()`).
    For distributions that can be scaled, some distributions can only be (linearly)
    scaled *before* the random numbers are generated (called "predraw"), some can only be
    scaled after (called "postdraw"), and some can be scaled in either way ("both").

    For example, a normal distribution is "both" since 2*Normal(a, b) = Normal(2*a, 2*b).
    A Poisson distribution is "predraw" since 2*Poisson(λ) ≠ Poisson(2*λ), and
    there is no way to get the correct shape of a different Poisson distribution
    once the numbers have been drawn. Finally, distributions with unitless shape
    parameters as well as parameter that can have units (e.g. a gamma distribution
    with shape and scale parameters) are referred to as "postdraw" since scaling
    all input parameters is invalid (i.e. 2*Gamma(shape, scale) ≠ Gamma(2*shape, 2*scale)),
    but they can still be scaled (i.e. 2*Gamma(shape, scale) = Gamma(1*shape, 2*scale)).

    To summarize, options for `dist.scaling` are:

        - 'postdraw' (after the random numbers are drawn, e.g. `ss.weibull()`)
        - 'predraw' (before the draw, e.g. `ss.poisson()`)
        - 'both' (either pre or post draw, e.g. `ss.normal()`)
        - False (not at all, e.g. `ss.beta_dist()`)

    Use `ss.distributions.scale_types.show()` to show how each distribution scales
    with time.
    """
    both = ['predraw', 'postdraw']
    postdraw = 'postdraw'
    predraw = 'predraw'
    false = None

    @classmethod
    def check_predraw(cls, dist):
        """ Check if the supplied distribution supports pre-draw (parameter) scaling """
        return cls.predraw in sc.tolist(dist.scaling)

    @classmethod
    def check_postdraw(cls, dist):
        """ Check if the supplied distribution supports post-draw (results) scaling """
        return cls.postdraw in sc.tolist(dist.scaling)

    @classmethod
    def show(cls, to_df=False):
        """ Show which distributions have which scale types """
        data = []
        for distname in ss.distributions.dist_list:
            dist = getattr(ss.distributions, distname)
            predraw = cls.check_predraw(dist)
            postdraw = cls.check_postdraw(dist)
            both = predraw and postdraw
            none = not predraw and not postdraw
            row = dict(name=distname, predraw=predraw, postdraw=postdraw, both=both, none=none)
            data.append(row)
        df = sc.dataframe(data)
        if to_df:
            return df
        else:
            print('Distribution scale types (see the ss.distributions.scale_types docstring for more info):')
            df.disp()
        return


class Dist:
    """
    Base class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.

    See `ss.dist_list` for a full list of supported distributions. Parameter inputs
    tend to follow SciPy's, rather than NumPy's, definitions (although in most
    cases they're the same). See also `ss.distributions.scale_types` for more information
    on how different distributions scale by time.

    Note: by default, `ss.Dist` is initialized with an `ss.Sim` object to ensure
    random number reproducibility. You can override this with either `ss.Dist(strict=False)`
    on creation, or `dist.init(force=True)` after creation.

    Although it's possible in theory to define a custom distribution (i.e., not
    one from NumPy or SciPy), in practice this is difficult. The distribution needs
    to have both a way to return random variates (easy), as well as the probability
    point function (inverse CDF). In addition, the distribution must be able to
    take a NumPy RNG as its bit generator. It's easier to just use a default Dist
    (e.g., ss.random()), and then take its output as input (i.e., quantiles) for
    whatever custom distribution you want to create.

    Args:
        dist (rv_generic): optional; a `scipy.stats` distribution (frozen or not) to get the ppf from
        distname (str): the name for this class of distribution (e.g. "uniform")
        name (str): the name for this particular distribution (e.g. "age_at_death")
        unit (str/`ss.TimePar`): if provided, convert the output of the distribution to a timepar (e.g. rate or duration); can also be inferred from distribution parameters (see examples below)
        seed (int): the user-chosen random seed (e.g. 3)
        offset (int): the seed offset; will be automatically assigned (based on hashing the name) if None
        strict (bool): if True, require initialization and invalidate after each call to rvs()
        auto (bool): whether to auto-reset the state after each draw
        sim (Sim): usually determined on initialization; the sim to use as input to callable parameters
        module (Module): usually determined on initialization; the module to use as input to callable parameters
        mock (int): if provided, then initialize with a mock Sim object (of size `mock`) for debugging purposes
        debug (bool): print out additional detail
        kwargs (dict): parameters of the distribution

    **Examples**:

        # Create a Bernoulli distribution
        p_death = ss.bernoulli(p=0.1).init(force=True)
        p_death.rvs(50) # Create 50 draws

        # Create a normal distribution that's also a timepar
        dur_infection = ss.normal(loc=12, scale=2, unit='years')
        dur_infection = ss.years(ss.normal(loc=12, scale=2)) # Same as above
        dur_infection = ss.normal(loc=ss.years(12), scale=2)) # Same as above
        dur_infection = ss.normal(loc=ss.years(12), scale=ss.months(24)) # Same as above, perform time unit conversion internally
        dur_infection.init(force=True).plot_hist() # Show results

        # Create a distribution manually
        dist = ss.Dist(dist=sps.norm, loc=3).init(force=True)
        dist.rvs(10) # Return 10 normally distributed random numbers
    """
    valid_pars = None
    scaling = None # See "scale_types" above

    def __init__(self, dist=None, distname=None, name=None, unit=None, seed=None, offset=None,
                 strict=True, auto=True, sim=None, module=None, mock=False, debug=False, **kwargs):
        # If a string is provided as "dist" but there's no distname, swap the dist and the distname
        if isinstance(dist, str) and distname is None:
            distname = dist
            dist = None
        self.dist = dist # The type of distribution
        self.distname = distname
        self.name = name
        self.pars = sc.objdict(kwargs) # The user-defined kwargs
        self.unit = ss.time.get_timepar_class(unit) # The timepar class -- can be None
        self.seed = seed # Usually determined once added to the container
        self.offset = offset
        self.module = module
        self.sim = sim
        self.slots = None # Created on initialization with a sim
        self.strict = strict
        self.auto = auto
        self.debug = debug
        self.scaling = sc.tolist(self.scaling) # Convert e.g. 'predraw' to ['predraw'] so can use "if x in y" later

        # Auto-generated
        self.rvs_func = None # The default function to call in make_rvs() to generate the random numbers
        self._use_ppf = None # If True, use the PPF to generate random values. Otherwise, use make_rvs()
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
        if self.valid_pars is not None:
            self.validate_pars()

        # Finalize
        if self.unit is not None:
            if scale_types.postdraw not in self.scaling:
                errormsg = 'You can only specify a distribution-level time unit for distributions that can be scaled post-draw, e.g. ss.normal(). '
                if scale_types.predraw in self.scaling:
                    errormsg += f'{self} can only be scaled pre-draw, meaning you must convert the input parameters to TimePars.'
                else:
                    errormsg += f'Distributions ({type(self)} cannot be scaled.'
                raise ValueError(errormsg)

        if mock:
            if mock is True: # Convert from boolean to a reasonable int
                mock = 100
            strict = False
            self.sim = ss.mock_sim(mock)
            self.module = ss.mock_module()
            self.trace = 'mock_dist'

        if not strict: # Otherwise, wait for a sim
            self.init()

        return

    def __repr__(self):
        """ Custom display to show state of object """
        j = sc.dictobj(self.to_json())
        parstr = [f'{k}={v}' for k,v in j.pars.items()]
        string = f'ss.{j.classname}({j.diststr}{sc.strjoin(parstr)})'
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

        self._callable_keys = None # Reset this in case a function was provided and its signature changed (unlikely, but better safe than sorry!)
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
            if self.initialized:
                # If initialized, re-process the pars to update self._pars
                # If not initialized, the module may not be available to do this conversion - but in any case,
                # initialization will cause the pars to be processed again. So skip this step here
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

        **Example**:

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
        if ss.options.single_rng:
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
        """
        Calculate the starting seed and create the RNG

        Typically this is not invoked by the user, although the user can call it with force=True
        to initialize a distribution manually independently of a `ss.Sim` object
        (which is equivalent to setting strict=False when creating the dist).

        Args:
            trace (str): the distribution's location within the sim
            seed (int): the base random number seed that other random number seeds will be generated from
            module (`ss.Module`): the parent module
            sim (`ss.Sim`): the parent sim
            slots (array): the agent slots of the parent sim
            force (bool): whether to skip validation (if the dist has already been initialized, and if any inputs are None)
        """

        if self.initialized is True and not force: # Don't warn if we have a partially initialized distribution
            msg = f'Distribution {self} is already initialized, use force=True if intentional'
            ss.warn(msg)

        # Calculate the offset (starting seed)
        self.process_seed(trace, seed)

        # Create the actual RNG
        if ss.options.single_rng:
            self.rng = np.random.mtrand._rand # If single_rng, return the centralized numpy random number instance
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
            if self.strict and not force:
                errormsg = f'Distribution {self} is not fully initialized, the following inputs are None:\n{none_dict.keys()}\n'
                errormsg += 'This distribution may not produce valid random numbers; set force=True if this is intentional (e.g. for testing it independently of a sim).'
                raise RuntimeError(errormsg)
        else:
            self.initialized = True
        return self

    def mock(self, trace='mock', **kwargs):
        """ Create a distribution using a mock sim for testing purposes

        Args:
            trace (str): the "trace" of the distribution (normally, where it would be located in the sim)
            **kwargs (dict): passed to `ss.mock_sim()` as well as `ss.mock_module()` (typically time args, e.g. dt)

        **Example**:

            dist = ss.normal(3, 2, unit='years').mock(dt=ss.days(1))
            dist.rvs(10)
        """
        mock_sim = ss.mock_sim(**kwargs)
        mock_mod = ss.mock_module(**kwargs)
        self.init(trace=trace, module=mock_mod, sim=mock_sim)
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

    def validate_pars(self):
        """ Check if parameters are valid; only used for non-SciPy distributions """
        valid = set(self.valid_pars)
        user = set(self.pars.keys())
        unmatched = user - valid
        if unmatched:
            errormsg = f'Mismatch for {type(self)} between valid parameters {valid} and user-provided parameters {user}'
            raise ValueError(errormsg)
        return

    def process_pars(self, call=True):
        """ Ensure the supplied dist and parameters are valid, and initialize them; not for the user """
        # self._timepar = None # Time rescalings need to be done after distributions are calculated; store the correction factor here
        self._pars = sc.cp(self.pars) # The actual keywords; shallow copy, modified below for special cases
        if call:
            self.call_pars() # Convert from function to values if needed
        self.convert_timepars()
        spars = self.sync_pars() # Synchronize parameters between the NumPy and SciPy distributions
        return spars

    def convert_timepars(self):
        """
        Convert time parameters (durations and rates) to scalars

        This function converts time parameters into bare numbers that will be returned by rvs() depending
        on the timestep of the parent module for this `Dist`. The conversion for these types is

        - Durations are divided by `dt` (so the result will be a number of timesteps)
        - Rates are multiplied by `dt` (so the result will be a number of events, or else the equivalent multiplicate value for the timestep)
        """
        # Check through and see if anything is a timepar
        timepar_dict = dict()
        for key,val in self._pars.items():
            if isinstance(val, ss.TimePar):
                timepar_type = type(val)
                timepar_dict[key] = timepar_type

        # Check for e.g. ss.normal(ss.years(3), ss.years(2), unit=ss.days)
        if self.unit is not None and len(timepar_dict):
            errormsg = f'You have provided timepars both for the distribution itself ({self.unit}) and one or more parameters:\n{timepar_dict}.\nPlease use one or the other, not both!'
            raise ValueError(errormsg)

        # Check for e.g. ss.poisson(3, unit=ss.years) (valid, warning) or ss.gamma(0.5, 1, unit=ss.years) (invalid, error)
        if self.unit is not None and not ss.distributions.scale_types.check_postdraw(self):
            msg = f'You have supplied a distribution-level timepar {self.unit}, but {self} can only be scaled pre-draw. '
            if len(self._pars) == 1:
                self._pars[0] = self.unit(self._pars[0]) # Try to convert to a time unit (NB, may fail for functions)
                self.unit = None
                if ss.options.warn_convert:
                    msg += f'Since ss.{self.__name__} only takes one input parameter, this has been automatically converted to predrawn scaling. '
                    msg += 'To avoid this warning, use e.g. ss.poisson(ss.years(3)) instead of ss.years(ss.poisson(3)), or set ss.options.warn_convert=False.'
                    ss.warnmsg(msg)
            else:
                msg += f'Since ss.{self.__name__} has more than one input parameter, the parameters cannot be scaled by time in this way. '
                msg += 'Use e.g. ss.weibull(3, ss.years(5), ss.years(2)) instead of ss.weibull(3, 5, 2, unit=ss.years).'
                raise ValueError(msg)

        # Do predraw scaling
        for key, v in self._pars.items():
            is_timepar = False
            if isinstance(v, ss.TimePar):
                is_timepar = True
                timepar_type = type(v)
            elif isinstance(v, np.ndarray) and v.size and isinstance(v.flat[0], ss.TimePar):
                is_timepar = True
                timepar_type = type(v.flat[0])
                if v.size == 1:
                    v = v.flat[0] # Case where we have an array wrapping a dur wrapping an array # TODO: refactor
                else:
                    warnmsg = f'Operating on an array of timepars is very slow; timepars should wrap arrays, not vice versa. Values:\n{v}'
                    ss.warn(warnmsg)

            if is_timepar:
                if self.module is not None:
                    dt = self.module.dt
                elif self.sim is not None:
                    dt = self.sim.dt
                else:
                    errormsg = 'Cannot do predraw scaling of timepar {v} when both module and sim are None. Consider e.g. ss.normal(3, 2, unit=ss.year) instead of ss.normal(ss.years(3), ss.years(2))'
                    raise RuntimeError(errormsg)

                if issubclass(timepar_type, ss.dur):
                    self._pars[key] = v/dt
                elif issubclass(timepar_type, ss.Rate):
                    self._pars[key] = v*dt # Usually to_prob, but may be n_events
                else:
                    errormsg = f'Unknown timepar type {v}'
                    raise NotImplementedError(errormsg)

                try:
                    self._pars[key] = self._pars[key].astype(float)
                except:
                    pass

    def convert_callable(self, parkey, func, size, uids):
        """ Method to handle how callable parameters are processed; not for the user """
        size_par = sc.ifelse(uids, size, ss.uids()) # Allow none size
        if not getattr(self, '_callable_keys', None): # Inspect isn't that fast, so only do this once
            keys = list(inspect.signature(func).parameters.keys()) # Get the input arguments  of the function
            module = self.module
            mapping = {
                'sim': self.sim,
                'self': module,
                'module': module,
                'uids': size_par,
                'size': size_par,
            }
            if isinstance(module, ss.Module): # If it's an actual module, we can add a couple more
                mapping['pars'] = module.pars
                mapping['states'] = module.state_dict
            self._callable_keys = keys
            self._callable_args = mapping
        else:
            keys = self._callable_keys
            mapping = self._callable_args

        # We have to update this every time
        for key in ['uids', 'size']:
            mapping[key] = size_par

        # Assemble the arguments
        args = []
        for key in keys:
            try:
                args.append(mapping[key])
            except KeyError as e:
                valid = mapping.keys()
                errormsg = f'Valid distribution function arguments are 1-3 of {sc.strjoin(valid)}, not "{key}"'
                raise sc.KeyNotFoundError(errormsg) from e

        out = func(*args) # Actually calculate the function
        val = np.asarray(out) # Necessary since FloatArrs don't allow slicing # TODO: check if this is correct
        self._pars[parkey] = val
        return val

    def call_par(self, key, val, size, uids):
        """ Check if this parameter needs to be called to be turned into an array; not for the user """
        if callable(val) and not isinstance(val, type): # If the parameter is callable, then call it (types can appear as callable)
            val = self.convert_callable(key, val, size, uids)
        return val

    def call_pars(self):
        """ Check if any parameters need to be called to be turned into arrays; not for the user """

        # Initialize
        size, uids = self._size, self._uids
        if self._use_ppf != False: # Allow "False" to prevent ever using dynamic pars (used in ss.choice())
            self._use_ppf = None

        # Check each parameter
        for key,val in self._pars.items():
            val = self.call_par(key, val, size, uids)

            # If it's iterable and UIDs are provided, then we need to use array-parameter logic
            if self._use_ppf is None and np.iterable(val) and uids is not None:
                self._use_ppf = True
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
        if ss.options.single_rng:
            return np.random.mtrand._rand.random(size).astype(ss.dtypes.float)
        return self.rng.random(size, dtype=ss.dtypes.float) # Up to 2x faster with float32

    def make_rvs(self):
        """ Return default random numbers for scalar parameters; not for the user """
        if self.rvs_func is not None:
            rvs_func = getattr(self.rng, self.rvs_func) # Can't store this because then it references the wrong RNG after copy
            if ss.options.single_rng:
                dtype = self._pars.pop('dtype', None)
            rvs = rvs_func(size=self._size, **self._pars)
            if ss.options.single_rng and dtype:
                rvs = rvs.astype(dtype)
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

    def rvs(self, n=1, round=False, reset=False):
        """
        Get random variates -- use this!

        Args:
            n (int/tuple/arr): if an int or tuple, return this many random variates; if an array, treat as UIDs
            round (bool): if True, randomly round up or down based on how close the value is
            reset (bool): whether to automatically reset the random number distribution state after being called
        """
        # Check for readiness
        if not self.initialized:
            raise DistNotInitializedError(self)
        if not self.ready and self.strict and not ss.options.single_rng:
            raise DistNotReadyError(self)

        # Figure out size, UIDs, and slots
        if ss.options.single_rng and not (np.isscalar(n) or isinstance(n, tuple)):
            n = len(n) # If centralized, treat n as a size
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
        if self._use_ppf:
            rands = self.rand(size)[slots] # Get random values
            rvs = self.ppf(rands) # Convert to actual values via the PPF
        else:
            rvs = self.make_rvs() # Or, just get regular values
            if self._slots is not None:
                rvs = rvs[self._slots]

        # Handle unit if provided
        if self.unit is not None:
            if scale_types.check_postdraw(self): # It can be scaled post-draw, proceed
                rvs = self.unit(rvs)
                if isinstance(rvs, ss.dur):
                    rvs = rvs/self.module.dt
                elif isinstance(rvs, ss.Rate):
                    rvs = rvs*self.module.dt
                else:
                    errormsg = f'Unexpected timepar type {type(rvs)}: expecting subclass of ss.dur or ss.Rate'
                    raise NotImplementedError(errormsg)
            else:  # It can't, raise an error
                errormsg = f'You have provided a timepar {self.unit} to scale {self} by, but this distribution cannot be scaled postdraw. '
                if scale_types.check_predraw(self):
                    errormsg += 'Since this distribution supports pre-draw scaling, please scale the individual input parameters instead. '
                else:
                    errormsg += 'Not all distributions can be scaled; see ss.distributions.scale_types for information how different distributions scale. '
                raise ValueError(errormsg)

        # Round if needed
        if round:
            rvs = self.randround(rvs)

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

    def randround(self, rvs):
        """ Round the values up or down to an integer stochastically; usually called via `dist.rvs(round=True)` """
        rvs = np.array(np.floor(rvs+self.rand(rvs.shape)), dtype=ss_int_) # Unsure whether the dtype should be int or rand_int, but the former is safer performance-wise
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

    def shrink(self):
        """ Shrink the size of the module for saving to disk """
        shrunk = ss.utils.shrink()
        self.slots = shrunk
        self._slots = shrunk
        self.module = shrunk
        self.sim = shrunk
        self._n = shrunk
        self._uids = shrunk
        self.history = shrunk
        self._callable_args = shrunk
        return

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
dist_list = ['random', 'uniform', 'normal', 'lognorm_ex', 'lognorm_im', 'expon',
             'poisson', 'nbinom', 'beta_dist', 'beta_mean', 'weibull', 'gamma', 'constant',
             'randint', 'rand_raw', 'bernoulli', 'choice', 'histogram']
__all__ += dist_list
__all__ += ['multi_random'] # Not a dist in the same sense as the others (e.g. same tests would fail)


class random(Dist):
    """ Random distribution, with values on the interval (0, 1) """
    valid_pars = ['dtype']
    scaling = scale_types.false # Always (0,1), can't be scaled
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
    valid_pars = ['low', 'high']
    scaling = scale_types.both

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
    scaling = scale_types.both
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

    **Example**:

        ss.lognorm_im(mean=2, sigma=1, strict=False).rvs(1000).mean() # Should be roughly 10
    """
    scaling = scale_types.postdraw

    def __init__(self, mean=0.0, sigma=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='lognormal', dist=sps.lognorm, mean=mean, sigma=sigma, **kwargs)
        return

    def convert_timepars(self):
        for key, v in self._pars.items():
            if isinstance(v, ss.dur) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.dur):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')
            if isinstance(v, ss.Rate) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.Rate):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')

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


class lognorm_ex(Dist):
    """
    Lognormal distribution, parameterized in terms of the "explicit" (lognormal)
    distribution, with mean=mean and std=std for this distribution (see lognorm_im for comparison).
    Note that a mean ≤ 0.0 is impossible, since this is the parameter of the distribution
    after the log transform.

    Args:
        mean (float): the mean of this distribution (not the underlying distribution) (default 1.0)
        std (float): the standard deviation of this distribution (not the underlying distribution) (default 1.0)

    **Example**:

        ss.lognorm_ex(mean=2, std=1, strict=False).rvs(1000).mean() # Should be close to 2
    """
    scaling = scale_types.both

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
    scaling = scale_types.both
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(distname='exponential', dist=sps.expon, scale=scale, **kwargs)
        return


class poisson(Dist): # TODO: does not currently scale correctly with dt
    """
    Poisson distribution

    Args:
        lam (float): the scale of the distribution (default 1.0)
    """
    scaling = scale_types.predraw
    def __init__(self, lam=1.0, **kwargs):
        super().__init__(distname='poisson', dist=sps.poisson, lam=lam, **kwargs)
        return

    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
        spars = dict(mu=self._pars.lam)
        self.update_dist_pars(spars)
        return spars


class beta_dist(Dist):
    """
    Beta distribution

    Args:
        a (float): shape parameter, must be > 0 (default 1.0)
        b (float): shape parameter, must be > 0 (default 1.0)
    """
    scaling = scale_types.false
    def __init__(self, a=1.0, b=1.0, **kwargs):
        super().__init__(distname="beta", dist=sps.beta, a=a, b=b, **kwargs)
        return


class beta_mean(Dist):
    """
    Beta distribution paramterized by the mean

    Note: the variance of a beta distribution must be 0 < var < mean*(1-mean)

    Args:
        mean (float): mean of distribution, must be 0 < a < 1 (default 0.5)
        var (float): variance of distribution, must be > 0 (default 0.05)
        force (bool): if True, scale the parameters to the valid range
    """
    scaling = scale_types.false
    def __init__(self, mean=0.5, var=0.05, force=False, **kwargs):  # Does not accept dtype
        # Validation
        max_var = mean*(1-mean)
        if not (0 < mean < 1):
            if force:
                if ss.options.warn_convert:
                    warnmsg = f'Clipping the mean from {mean:n} to 0 < mean < 1.'
                    ss.warn(warnmsg)
                mean = np.clip(mean, 0, 1)
            else:
                errormsg = f'The mean of a beta distribution must be 0 < mean < 1, not {mean:n}'
                raise ValueError(errormsg)
        if not (0 < var < max_var):
            if force:
                if ss.options.warn_convert:
                    warnmsg = f'Clipping the variance from {var} to 0 < var < {max_var:n}.'
                    ss.warn(warnmsg)
                var = np.clip(var, 0, max_var)
            else:
                errormsg = f'The variance of a beta distribution must be 0 < var < mean*(1-mean). For your {mean=}, {var=} is invalid; the maximum variance is {max_var:n}.'
                raise ValueError(errormsg)

        # We're good to go
        a = ((1 - mean)/var - 1/mean) * mean**2
        b = a * (1 / mean - 1)
        super().__init__(distname="beta", dist=sps.beta, a=a, b=b, **kwargs)
        return


class nbinom(Dist):
    """
    Negative binomial distribution

    Args:
        n (float): the number of successes, > 1 (default 1.0)
        p (float): the probability of success in [0,1], (default 0.5)

    """
    scaling = scale_types.predraw
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
    scaling = scale_types.predraw
    valid_pars = ['low', 'high', 'dtype']

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

        if ss.options.single_rng: # randint because we're accessing via numpy.random
            super().__init__(distname='randint', low=low, high=high, dtype=dtype, **kwargs)
        else: # integers instead of randint because interfacing a numpy.random.Generator
            super().__init__(distname='integers', low=low, high=high, dtype=dtype, **kwargs)
        return

    def convert_timepars(self):
        for key, v in self._pars.items():
            if isinstance(v, ss.dur) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.dur):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')
            if isinstance(v, ss.Rate) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.Rate):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')

    def ppf(self, rands):
        p = self._pars
        rvs = rands * (p.high + 1 - p.low) + p.low
        rvs = rvs.astype(self.dtype)
        return rvs

class rand_raw(Dist):
    """
    Directly sample raw integers (uint64) from the random number generator.
    Typicaly only used with ss.combine_rands().
    """
    valid_pars = []
    scaling = scale_types.false

    def make_rvs(self):
        if ss.options.single_rng:
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
    scaling = scale_types.postdraw
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
    scaling = scale_types.postdraw
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
    valid_pars = ['v']
    scaling = scale_types.both

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
    valid_pars = ['p']
    scaling = scale_types.predraw

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


class choice(Dist):
    """
    Random choice between discrete options (note: dynamic parameters not supported)

    Args:
        a (int or array): the number of choices, or the choices themselves (default 2)
        p (array): if supplied, the probability of each choice (default, 1/a for a choices)

    **Examples**:

        # Simulate 10 die rolls
        ss.choice(6, strict=False)(10) + 1

        # Choose between specified options each with a specified probability (must sum to 1)
        ss.choice(a=[30, 70], p=[0.3, 0.7], strict=False)(10)

    Note: although Bernoulli trials can be generated using a=2, it is much faster
    to use ss.bernoulli() instead.
    """
    valid_pars = ['a', 'p', 'replace', 'dtype']
    scaling = scale_types.false

    def __init__(self, a=2, p=None, replace=True, **kwargs):
        super().__init__(distname='choice', a=a, p=p, replace=replace, **kwargs)
        self._use_ppf = False # Set to false since array arguments don't imply dynamic pars here
        return

    def convert_timepars(self):
        for key, v in self._pars.items():
            if isinstance(v, ss.dur) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.dur):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')
            if isinstance(v, ss.Rate) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.Rate):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')

    def ppf(self, rands):
        """ Shouldn't actually be needed since dynamic pars not supported """
        pars = self._pars
        if np.isscalar(pars.a):
            pars.a = np.arange(pars.a)
        pcum = np.cumsum(pars.p)
        inds = np.searchsorted(pcum, rands)
        rvs = pars.a[inds]
        return rvs


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

    **Examples**:

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
    valid_pars = ['values', 'bins', 'density', 'data']
    scaling = scale_types.false

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

        # Validate inputs
        values = values.astype(float)
        vsum = values.sum()
        if vsum != 1.0:
            values = values / vsum
        dist = sps.rv_histogram((values, bins), density=density) # Create the SciPy distribution
        super().__init__(dist=dist, distname='histogram', **kwargs)
        self._use_ppf = False # Set to false since array arguments correspond to bins, not UIDs
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
        rvs_list = nb.typed.List(rvs_list) # See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
        int_type = ss.dtypes.rand_uint
        int_max = np.iinfo(int_type).max
        rvs = self.combine_rvs(rvs_list, int_type, int_max)
        return rvs


#%% Dist exceptions

class DistNotInitializedError(RuntimeError):
    """ Raised when Dist object is called when not initialized. """
    def __init__(self, dist=None, msg=None):
        if msg is None:
            msg = f'{dist} has not been initialized; please set strict=False when creating the distribution, or call dist.mock() or dist.init().'
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
