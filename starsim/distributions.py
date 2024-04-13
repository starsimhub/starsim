"""
Define random-number-safe distributions.
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as pl

__all__ = ['find_dists', 'dist_list', 'Dists', 'Dist']


def str2int(string, modulo=1_000_000):
    """
    Convert a string to an int
    
    Cannot use Python's built-in hash() since it's randomized for strings, but
    this is almost as fast (and 5x faster than hashlib).
    """
    return int.from_bytes(string.encode(), byteorder='big') % modulo


def find_dists(obj, verbose=False):
    """ Find all Dist objects in a parent object """
    out = sc.objdict()
    tree = sc.iterobj(obj, depthfirst=False, flatten=True)
    if verbose: print(f'Found {len(tree)} objects')
    for trace,val in tree.items():
        if isinstance(val, Dist):
            out[trace] = val
            if verbose: print(f'  {trace} is a dist ({len(out)})')
    return out


class Dists(sc.prettyobj):
    """ Class for managing a collection of Dist objects """

    def __init__(self, obj=None, *args, base_seed=None, sim=None):
        if len(args): obj = [obj] + list(args)
        self.obj = obj
        self.dists = None
        self.base_seed = base_seed
        self.sim = sim
        self.initialized = False
        if self.obj is not None:
            self.initialize()
        return

    def initialize(self, obj=None, base_seed=None, sim=None, force=False):
        """
        Set the base seed, find and initialize all distributions in an object
        
        In practice, the object is usually a Sim, but can be anything.
        """
        if base_seed:
            self.base_seed = base_seed
        sim = sim if (sim is not None) else self.sim
        obj = obj if (obj is not None) else self.obj
        if obj is None:
            errormsg = 'Must supply a container that contains one or more Dist objects, typically the sim'
            raise ValueError(errormsg)
        self.dists = find_dists(obj)
        for trace,dist in self.dists.items():
            if not dist.initialized or force:
                dist.initialize(trace=trace, seed=base_seed, sim=sim, force=force)
        self.check_seeds()
        self.initialized = True
        return self
    
    def check_seeds(self):
        """ Check that no two distributions share the same seed """
        checked = dict()
        for trace,dist in self.dists.items():
            seed = dist.seed
            if seed in checked.keys():
                raise DistSeedRepeatError(checked[seed], dist)
            else:
                checked[seed] = dist
        return

    def jump(self, to=None, delta=1):
        """ Advance all RNGs, e.g. to timestep "to", by jumping """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.jump(to=to, delta=delta)
        return out

    def reset(self):
        """ Reset each RNG """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.reset()
        return out


class Dist: # TODO: figure out why subclassing sc.prettyobj breaks isinstance
    """
    Base class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.
    
    See ss.dist_list for a full list of supported distributions.
    
    Although it's possible in theory to define a custom distribution (i.e., not
    one from NumPy or SciPy), in practice this is difficult. The distribution needs
    to have both a way to return random variables (easy), as well as the probability
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
        self.pars = sc.dictobj(kwargs) # The user-defined kwargs
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
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self.ind = 0 # The index of the RNG (usually updated on each timestep)
        self.called = 0 # The number of times the distribution has been called
        self.history = [] # Previous states
        self.ready = True
        self.initialized = False
        if not strict: # Otherwise, wait for a sim
            self.initialize()
        return
    
    def __repr__(self):
        """ Custom display to show state of object """
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
        string = f'ss.{classname}({tracestr}, {diststr}pars={dict(self.pars)})'
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
    
    def set(self, dist=None, **kwargs):
        """ Set (change) the distribution type, or one or more parameters of the distribution """
        if dist:
            self.dist = dist
            self.process_dist()
        if kwargs:
            self.pars.update(kwargs)
            self.process_pars(call=False)
        return

    @property
    def bitgen(self):
        try:    return self.rng.bit_generator
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
            
            dist = ss.random(seed=5).initialize()
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
        self.rng.bit_generator.state = state.copy()
        self.ready = True
        return self.state

    def jump(self, to=None, delta=1):
        """ Advance the RNG, e.g. to timestep "to", by jumping """
        jumps = to if (to is not None) else self.ind + delta
        self.ind = jumps
        self.reset() # First reset back to the initial state (used in case of different numbers of calls)
        if jumps: # Seems to randomize state if jumps=0
            self.bitgen.state = self.bitgen.jumped(jumps=jumps).state # Now take "jumps" number of jumps
        return self.state
    
    def initialize(self, trace=None, seed=None, module=None, sim=None, slots=None, force=False):
        """ Calculate the starting seed and create the RNG """
        
        if self.initialized is True and not force: # Don't warn if we have a partially initialized distribution
            msg = f'Distribution {self} is already initialized, use force=True if intentional'
            ss.warn(msg)
        
        # Calculate the offset (starting seed)
        self.process_seed(trace, seed)
        
        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.make_history(reset=True)
        
        # Handle the sim, module, and slots
        self.sim = sim if (sim is not None) else self.sim
        self.module = module if (module is not None) else self.module
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
        if self.trace is None or self.sim is None or self.slots is None:
            self.initialized = 'partial' # Initialized enough to produce random numbers, but not fully initialized
        else:
            self.initialized = True
        return self
    
    def process_seed(self, trace=None, seed=None):
        """ Obtain the seed offset by hashing the path to this distribution; called automatically """
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
        """ Ensure the distribution works """
        
        # Handle a SciPy distribution, if provided
        if self.dist is not None:
            
            # Pull out parameters of an already-frozen distribution
            if isinstance(self.dist, sps._distn_infrastructure.rv_frozen):
                if not self.initialized: # Don't do this more than once
                    self.pars = sc.dictobj(sc.mergedicts(self.pars, self.dist.kwds))
            
            # Convert to a frozen distribution
            if isinstance(self.dist, sps._distn_infrastructure.rv_generic):
                spars = self.process_pars(call=False)
                self.dist = self.dist(**spars) 
                
            # Override the default random state with the correct one
            self.dist.random_state = self.rng 
            
        # Set the default function for getting the rvs
        if self.distname is not None and hasattr(self.rng, self.distname): # Don't worry if it doesn't, it's probably being manually overridden
            self.rvs_func = self.distname # e.g. self.rng.uniform -- can't use the actual function because can become linked to the wrong RNG
        
        return
    
    def process_size(self, n=1):
        """ Handle an input of either size or UIDs and calculate size, UIDs, and slots """
        if np.isscalar(n) or isinstance(n, tuple):  # If passing a non-scalar size, interpret as dimension rather than UIDs iff a tuple
            uids = None
            slots = None
            size = n
        else:
            uids = np.asarray(n)
            if len(uids):
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
        """ Ensure the supplied dist and parameters are valid, and initialize them; called automatically """
        self._pars = sc.cp(self.pars) # The actual keywords; shallow copy, modified below for special cases
        if call:
            self.call_pars()
        spars = self.sync_pars()
        return spars
    
    def call_pars(self):
        """ Check if any parameters need to be called to be turned into arrays """
        
        # Initialize
        size, uids = self._size, self._uids
        if self.dynamic_pars != False: # Allow "False" to prevent ever using dynamic pars (used in ss.choice())
            self.dynamic_pars = None
        
        # Check each parameter
        for key,val in self._pars.items():
            
            # If the parameter is callable, then call it
            if callable(val): 
                size_par = uids if uids is not None else size
                out = val(self.module, self.sim, size_par) # TODO: swap order to sim, module, size?
                val = np.asarray(out) # Necessary since UIDArrays don't allow slicing # TODO: check if this is correct
                self._pars[key] = val
            
            # If it's iterable and UIDs are provided, then we need to use array-parameter logic
            if self.dynamic_pars is None and np.iterable(val) and uids is not None:
                self.dynamic_pars = True
        return
    
    def sync_pars(self):
        """ Perform any necessary synchronizations or transformations on distribution parameters """
        self.update_dist_pars()
        return self._pars
    
    def update_dist_pars(self, pars=None):
        """ Update SciPy distribution parameters """
        if self.dist is not None:
            pars = pars if pars is not None else self._pars
            self.dist.kwds = pars
        return
    
    def rand(self, size):
        """ Simple way to get simple random numbers """
        return self.rng.random(size)
    
    def make_rvs(self):
        """ Return default random numbers for scalar parameters """
        if self.rvs_func is not None:
            rvs_func = getattr(self.rng, self.rvs_func) # Can't store this because then it references the wrong RNG after copy
            rvs = rvs_func(**self._pars, size=self._size)
        elif self.dist is not None:
            rvs = self.dist.rvs(self._size)
        else:
            errormsg = 'Dist.rvs() failed: no valid NumPy/SciPy function found in this Dist. Has it been created and initialized correctly?'
            raise ValueError(errormsg)
        return rvs
    
    def ppf(self, rands):
        """ Return default random numbers for array parameters """
        rvs = self.dist.ppf(rands)
        return rvs
    
    def rvs(self, n=1, reset=False):
        """
        Get random variables
        
        Args:
            n (int/tuple/arr): if an int or tuple, return this many random variables; if an array, treat as UIDs
        """
        # Check for readiness
        if not self.initialized:
            raise DistNotInitializedError(self)
        if not self.ready and self.strict:
            raise DistNotReadyError(self)
        
        # Figure out size, UIDs, and slots
        size, slots = self.process_size(n)
        
        # Check if size is 0, then we can return
        if size == 0:
            return np.array([], dtype=int) # int dtype allows use as index, e.g. when filtering
        
        # Store the state
        self.make_history() # Store the pre-call state
        
        # Check if any keywords are callable -- parameters shouldn't need to be reprocessed otherwise
        if True: # self.dynamic_pars: # TODO: fix!!!!
            self.process_pars()
        
        # Actually get the random numbers
        if self.dynamic_pars:
            rands = self.rand(size)[slots] # Get random values 
            rvs = self.ppf(rands) # Convert to actual values via the PPF
        else:
            rvs = self.make_rvs() # Or, just get regular values
            if self._slots is not None:
                rvs = rvs[self._slots]
        
        # Tidy up
        self.called += 1
        if reset:
            self.reset(-1)
        elif self.auto: # TODO: check
            self.jump()
        elif self.strict:
            self.ready = False
        if self.debug:
            simstr = f'on ti={self.sim.ti} ' if self.sim else ''
            sizestr = f'with size={self._size}, '
            slotstr = f'Σ(slots)={self._slots.sum()}, ' if self._slots else '<no slots>, '
            rvstr = f'Σ(rvs)={rvs.sum():0.2f}, |rvs|={rvs.mean():0.4f}'
            pre_state = str(self.history[-1]['state']['state'])[-5:]
            post_state = str(self.state_int)[-5:]
            statestr = f"state {pre_state}→{post_state}"
            print(f'Debug: {self} called {simstr}{sizestr}{slotstr}{rvstr}, {statestr}')
            assert pre_state != post_state # Always an error if the state doesn't change after drawing random numbers
            
        return rvs


    def plot_hist(self, n=1000, bins=None, fig_kw=None, hist_kw=None):
        """ Plot the current state of the RNG as a histogram """
        pl.figure(**sc.mergedicts(fig_kw))
        rvs = self.rvs(n)
        self.reset(-1) # As if nothing ever happened
        pl.hist(rvs, bins=bins, **sc.mergedicts(hist_kw))
        pl.title(str(self))
        pl.xlabel('Value')
        pl.ylabel(f'Count ({n} total)')
        return rvs
        

#%% Specific distributions

# Add common distributions so they can be imported directly; assigned to a variable since used in help messages
dist_list = ['random', 'uniform', 'normal', 'lognorm_ex', 'lognorm_im', 'expon',
             'poisson', 'weibull', 'delta', 'randint', 'bernoulli', 'choice']
__all__ += dist_list


class random(Dist):
    """ Random distribution, values on interval (0,1) """
    def __init__(self, **kwargs):
        super().__init__(distname='random', **kwargs)
        return
    
    def ppf(self, rands):
        return rands


class uniform(Dist):
    """ Uniform distribution, values on interval (low, high) """
    def __init__(self, low=0.0, high=1.0, **kwargs):
        super().__init__(distname='uniform', low=low, high=high, **kwargs)
        return
    
    def ppf(self, rands):
        p = self._pars
        rvs = rands * (p.high - p.low) + p.low
        return rvs


class normal(Dist):
    """ Normal distribution, with mean=loc and stdev=scale """
    def __init__(self, loc=0.0, scale=1.0, **kwargs):
        super().__init__(distname='normal', dist=sps.norm, loc=loc, scale=scale, **kwargs)
        return


class lognorm_im(Dist):
    """
    Lognormal distribution, parameterized in terms of the "implicit" (normal)
    distribution, with mean=loc and stdev=scale (see lognorm_ex for comparison).
    
    Note: the "loc" parameter here does *not* correspond to the mean of the resulting
    random variates!
    
    **Example**::
        
        ss.lognorm_im(mean=2, sigma=1).rvs(1000).mean() # Should be roughly 10
    """
    def __init__(self, mean=0.0, sigma=1.0, **kwargs):
        super().__init__(distname='lognormal', dist=sps.lognorm, mean=mean, sigma=sigma, **kwargs)
        return
    
    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
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
    distribution, with mean=mean and stdev=stdev (see lognorm_im for comparison).
    
    **Example**::
        
        ss.lognorm_ex(mean=2, stdev=1).rvs(1000).mean() # Should be close to 2
    """
    def __init__(self, mean=1.0, stdev=1.0, **kwargs):
        super().__init__(distname='lognormal', dist=sps.lognorm, mean=mean, stdev=stdev, **kwargs)
        return
    
    def convert_ex_to_im(self):
        """
        Lognormal distributions can be specified in terms of the mean and standard
        deviation of the "explicit" lognormal distribution, or the "implicit" normal distribution.
        This function converts the parameters from the lognormal distribution to the
        parameters of the underlying (implicit) distribution, which are the form expected by NumPy's
        and SciPy's lognorm() distributions.
        """
        p = self._pars
        mean = p.pop('mean')
        stdev = p.pop('stdev')
        if mean <= 0:
            errormsg = f'Cannot create a lognorm_ex distribution with mean≤0 (mean={mean}); did you mean to use lognorm_im instead?'
            raise ValueError(errormsg)
        std2 = stdev**2
        mean2 = mean**2
        sigma_im = np.sqrt(np.log(std2/mean2 + 1)) # Computes stdev for the underlying normal distribution
        mean_im  = np.log(mean2 / np.sqrt(std2 + mean2)) # Computes the mean of the underlying normal distribution
        p.mean = mean_im
        p.sigma = sigma_im
        return mean_im, sigma_im
    
    def sync_pars(self):
        """ Convert from overlying to underlying parameters, then translate to SciPy """
        self.convert_ex_to_im()
        spars = lognorm_im.sync_pars(self) # Borrow sync_pars from lognorm_im
        return spars
    

class expon(Dist):
    """ Exponential distribution """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(distname='exponential', dist=sps.expon, scale=scale, **kwargs)
        return


class poisson(Dist):
    """ Exponential distribution """
    def __init__(self, lam=1.0, **kwargs):
        super().__init__(distname='poisson', dist=sps.poisson, lam=lam, **kwargs)
        return
    
    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
        spars = dict(mu=self._pars.lam)
        self.update_dist_pars(spars)
        return spars


class randint(Dist):
    """ Random integers, values on the interval [low, high-1] (i.e. "high" is excluded) """
    def __init__(self, low=0, high=2,  **kwargs):
        super().__init__(distname='integers', dist=sps.randint, low=low, high=high, **kwargs)
        return
    

class weibull(Dist):
    """ Weibull distribution -- NB, uses SciPy rather than NumPy """
    def __init__(self, c=1, loc=0, scale=1,  **kwargs):
        super().__init__(distname='weibull', dist=sps.weibull_min, c=c, loc=loc, scale=scale, **kwargs)
        return
    
    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


class delta(Dist):
    """ Delta distribution: equivalent to np.full() """
    def __init__(self, v=0, **kwargs):
        super().__init__(distname='delta', v=v, **kwargs)
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
    """
    def __init__(self, p=0.5, **kwargs):
        super().__init__(distname='bernoulli', p=p, **kwargs)
        return
    
    def make_rvs(self):
        rvs = self.rng.random(self._size) < self._pars.p # 3x faster than using rng.binomial(1, p, size)
        return rvs
    
    def ppf(self, rands):
        rvs = rands < self._pars.p
        return rvs
    
    def filter(self, uids, both=False):
        """ Return UIDs that correspond to True, or optionally return both True and False """
        bools = self.rvs(uids)
        if both:
            return uids[bools], uids[~bools]
        else:
            return uids[bools]


class choice(Dist):
    """
    Random choice between discrete options
    
    **Examples**::
        
        # Simulate 10 die rolls
        ss.choice(6)(10) + 1 
        
        # Choose between specified options each with a specified probability (must sum to 1)
        ss.choice(a=[30, 70], p=[0.3, 0.7])(10)
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


#%% Dist exceptions

class DistNotInitializedError(RuntimeError):
    """ Raised when Dist object is called when not initialized. """
    def __init__(self, dist):
        msg = f'{dist} has not been initialized; please call dist.initialize()'
        super().__init__(msg)

class DistNotReadyError(RuntimeError):
    """ Raised when a Dist object is called without being ready. """
    def __init__(self, dist):
        msg = f'{dist} is not ready. This is likely caused by calling a distribution multiple times in a single step. Call dist.jump() to reset.'
        super().__init__(msg)
        
class DistSeedRepeatError(RuntimeError):
    """ Raised when a Dist object shares a seed with another """
    def __init__(self, dist1, dist2):
        msg = f'A common seed was found between {dist1} and {dist2}. This is likely caused by incorrect initialization of the parent Dists object.'
        super().__init__(msg)
