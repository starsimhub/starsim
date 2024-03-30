"""
Define random-number-safe distributions.
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import starsim as ss
import pylab as pl

__all__ = ['find_dists', 'dist_list', 'Dists', 'Dist']


def str2int(string, modulo=10_000_000):
    """
    Convert a string to an int
    
    Cannot use Python's built-in hash() since it's randomized for strings, but
    this is almost as fast (and 5x faster than hashlib).
    """
    return int.from_bytes(string.encode(), byteorder='big') % modulo


def find_dists(obj, verbose=False):
    """ Find all Dist objects in a parent object """
    out = sc.objdict()
    tree = sc.iterobj(obj) # Temporary copy of sc.iterobj(obj) until Sciris 3.1.5 is released
    if verbose: print(f'Found {len(tree)} objects')
    for trace,val in tree.items():
        if isinstance(val, Dist):
            key = str(trace)
            out[key] = val
            if verbose: print(f'  {key} is a dist ({len(out)})')
    return out


class Dists:
    """ Class for managing a collection of Dist objects """

    def __init__(self, obj=None, base_seed=None, sim=None):
        self.obj = obj
        self.dists = None
        self.base_seed = base_seed
        self.sim = sim
        self.initialized = False
        if self.obj is not None:
            self.initialize()
        return

    def initialize(self, obj=None, base_seed=None, sim=None, force=True):
        """
        Set the base seed, find and initialize all distributions in an object
        
        In practice, the object is usually a Sim, but can be anything.
        """
        if base_seed:
            self.base_seed = base_seed
        sim = sim if (sim is not None) else self.sim
        obj = obj if (obj is not None) else self.obj
        if obj is None:
            errormsg = 'Must supply a container that contains one or more Dist objects'
            raise ValueError(errormsg)
        self.dists = find_dists(obj)
        for trace,dist in self.dists.items():
            if not dist.initialized or force:
                dist.initialize(trace=trace, seed=base_seed, sim=sim)
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
    
    Args:
        dist (rv_generic): optional; a scipy.stats distribution (frozen or not) to get the ppf from
        name (str): the name for this distribution
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
    def __init__(self, dist=None, name=None, seed=None, offset=None, strict=False, auto=True, sim=None, module=None, **kwargs): # TODO: switch back to strict=True
        self.dist = dist # The type of distribution
        self.name = name
        self.pars = sc.dictobj(kwargs) # The user-defined kwargs
        self.seed = seed # Usually determined once added to the container
        self.offset = offset
        self.module = module
        self.sim = sim
        self.slots = None # Created on initialization with a sim
        self.strict = strict
        self.auto = auto
        
        # Auto-generated 
        self._pars = None # Validated and transformed (if necessary) parameters
        self.dynamic_pars = None # Whether or not the distribution has array or callable parameters
        
        # Internal state
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self.ind = 0 # The index of the RNG (usually updated on each timestep)
        self.called = 0 # The number of times the distribution has been called
        self.history = [] # Previous states
        self.ready = True
        self.initialized = False
        if not strict:
            self.initialize()
        return
    
    def __repr__(self):
        """ Custom display to show state of object """
        tracestr = '<no trace>' if self.trace is None else "{self.trace}"
        diststr = '' if self.dist is None else f'dist={self.dist}, '
        string = f'ss.{self.__class__.__name__}({tracestr}, {diststr}pars={dict(self.pars)})'
        return string
    
    def disp(self):
        """ Return full display of object """
        return sc.pr(self)
    
    def show_state(self):
        """ Show the state of the object """
        s = sc.autolist()
        s += f'  dist = {self.dist}'
        s += f'  pars = {self.pars}'
        s += f' trace = {self.trace}'
        s += f'offset = {self.offset}'
        s += f'  seed = {self.seed}'
        s += f'   ind = {self.ind}'
        s += f'called = {self.called}'
        s += f' ready = {self.ready}'
        s += f' state = {self.state}'
        string = sc.newlinejoin(s)
        print(string)
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
            self.process_pars()
        return

    @property
    def bitgen(self):
        try:
            return self.rng.bit_generator
        except:
            return None
    
    @property
    def state(self):
        """ Get the current state """
        try:
            return self.bitgen.state
        except:
            return None
    
    def get_state(self):
        """ Return a copy of the state """
        return self.state.copy()
    
    def make_history(self):
        """ Store the current state in history """
        self.history.append(self.get_state()) # Store the initial state
        return

    def reset(self, state=0):
        """ Restore state: use 0 for initial, -1 for most recent """
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
    
    def initialize(self, trace=None, seed=None, module=None, sim=None, slots=None):
        """ Calculate the starting seed and create the RNG """
        
        # Calculate the offset (starting seed)
        self.process_seed(trace, seed)
        
        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.make_history()
        
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
        self.process_pars()
        self.ready = True
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
        if self.dist is not None:
            if isinstance(self.dist, sps._distn_infrastructure.rv_generic):
                pars = self.translate_pars(self._pars or self.pars)
                self.dist = self.dist(**pars) # Convert to a frozen distribution
            self.dist.random_state = self.rng # Override the default random state with the correct one
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
                size = slots.max() + 1
            else:
                size = 0
        
        self._n = n
        self._size = size
        self._uids = uids
        self._slots = slots
        return size, uids, slots
    
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
                out = val(self.sim, self.module, size_par)
                val = np.asarray(out) # Necessary since UIDArrays don't allow slicing # TODO: check if this is correct
                self._pars[key] = val
            
            # If it's iterable and UIDs are provided, then we need to use array-parameter logic
            if self.dynamic_pars is None and np.iterable(val) and uids is not None:
                self.dynamic_pars = True
        return
    
    def sync_pars(self):
        """ Perform any necessary synchronizations or transformations on distribution parameters """
        self.update_dist_pars()
        return
    
    def update_dist_pars(self, pars=None):
        """ Update SciPy distribution parameters """
        if self.dist is not None:
            pars = pars if pars is not None else self._pars
            self.dist.kwds = pars
        return
    
    def process_pars(self):
        """ Ensure the supplied dist and parameters are valid, and initialize them; called automatically """
        self._pars = self.pars.copy() # The actual keywords; shallow copy, modified below for special cases
        self.call_pars()
        self.sync_pars()
        return
    
    def rand(self, size=1):
        """ Simple way to get simple random numbers """
        return self.rng.random(size)
    
    def make_rvs(self):
        """ Return default random numbers for scalar parameters """
        rvs = self.dist.rvs(self._size)
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
        size, uids, slots = self.process_size(n)
        
        # Check if size is 0, then we can return
        if size == 0:
            return np.array([], dtype=int) # int dtype allows use as index, e.g. when filtering
        
        # Check if any keywords are callable
        if self.dynamic_pars:
            self.process_pars()
        
        # Store the state
        self.make_history() # Store the pre-call state
        
        # Actually get the random numbers
        if self.dynamic_pars:
            rands = self.rand(size)[slots] # Get random values 
            rvs = self.ppf(rands)
        else:
            rvs = self.make_rvs()
            if slots is not None:
                rvs = rvs[slots]
        
        # Tidy up
        self.called += 1
        if reset:
            self.reset(-1)
        elif self.auto: # TODO: check
            self.jump()
        elif self.strict:
            self.ready = False
            
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
        super().__init__(**kwargs)
        return
    
    def make_rvs(self):
        rvs = self.rng.random(size=self._size)
        return rvs
    
    def ppf(self, rands):
        return rands


class uniform(Dist):
    """ Uniform distribution, values on interval (low, high) """
    def __init__(self, low=0.0, high=1.0, **kwargs):
        super().__init__(low=low, high=high, **kwargs)
        return
    
    def make_rvs(self):
        p = self._pars
        rvs = self.rng.uniform(low=p.low, high=p.high, size=self._size)
        return rvs
    
    def ppf(self, rands):
        p = self._pars
        rvs = rands * (p.high - p.low) + p.low
        return rvs


class normal(Dist):
    """ Normal distribution, with mean=loc and stdev=scale """
    def __init__(self, loc=0.0, scale=1.0, **kwargs):
        super().__init__(dist=sps.norm, loc=loc, scale=scale, **kwargs)
        return
    
    def make_rvs(self):
        p = self._pars
        rvs = self.rng.normal(loc=p.loc, scale=p.scale, size=self._size)
        return rvs


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
        super().__init__(dist=sps.lognorm, mean=mean, sigma=sigma, **kwargs)
        return
    
    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
        p = self._pars
        spars = sc.dictobj()
        spars.s = p.sigma
        spars.scale = np.exp(p.mean)
        spars.loc = 0
        self.update_dist_pars(spars)
        return
    
    def make_rvs(self):
        p = self._pars
        rvs = self.rng.lognormal(mean=p.mean, sigma=p.sigma, size=self._size)
        return rvs


class lognorm_ex(lognorm_im):
    """
    Lognormal distribution, parameterized in terms of the "explicit" (lognormal)
    distribution, with mean=mean and stdev=stdev (see lognorm_im for comparison).
    
    **Example**::
        
        ss.lognorm_ex(mean=2, stdev=1).rvs(1000).mean() # Should be close to 2
    """
    def __init__(self, mean=1.0, stdev=1.0, **kwargs):
        super().__init__(dist=sps.lognorm, mean=mean, stdev=stdev, **kwargs)
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
        super().sync_pars()
        return
    

class expon(Dist):
    """ Exponential distribution """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(dist=sps.expon, scale=scale, **kwargs)
        return
    
    def make_rvs(self):
        rvs = self.rng.exponential(scale=self._pars.scale, size=self._size)
        return rvs


class poisson(Dist):
    """ Exponential distribution """
    def __init__(self, lam=1.0, **kwargs):
        super().__init__(dist=sps.poisson, lam=lam, **kwargs)
        return
    
    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
        spars = dict(mu=self._pars.lam)
        self.update_dist_pars(spars)
        return
    
    def make_rvs(self):
        rvs = self.rng.poisson(lam=self._pars.lam, size=self._size)
        return rvs


class randint(Dist):
    """ Random integers, values on the interval [low, high-1] (i.e. "high" is excluded) """
    def __init__(self, low=0, high=2,  **kwargs):
        super().__init__(dist=sps.randint, low=low, high=high **kwargs)
        return
    
    def make_rvs(self):
        p = self._pars
        rvs = self.rng.integers(low=p.low, high=p.high, size=self._size)
        return rvs


class weibull(Dist):
    """ Weibull distribution -- NB, uses SciPy rather than NumPy """
    def __init__(self, a=1, loc=0, scale=1,  **kwargs):
        super().__init__(dist=sps.weibull_min, a=a, loc=loc, scale=scale, **kwargs)
        return
    
    def make_rvs(self):
        rvs = self.dist.rvs(self._size)
        return rvs


class delta(Dist):
    """ Delta distribution: equivalent to np.full() """
    def __init__(self, v=0, **kwargs):
        super().__init__(v=v, **kwargs)
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
    def __init__(self, p=5, **kwargs):
        super().__init__(p=p, **kwargs)
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
        super().__init__(a=a, p=p, **kwargs)
        self.dynamic_pars = False # Set to false since array arguments don't imply dynamic pars here
        return
    
    def make_rvs(self):
        rvs = self.rng.choice(**self._pars, size=self._size)
        return rvs
    
    def ppf(self, rands):
        """ Shouldn't actually be needed """
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