"""
Define random-number-safe distributions.
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import pylab as pl

__all__ = ['find_dists', 'dist_list', 'Dists', 'Dist']


def str2int(string, modulo=10_000_000):
    """
    Convert a string to an int
    
    Cannot use Python's built-in hash() since it's randomized for strings, but
    this is almost as fast (and 5x faster than hashlib).
    """
    return int.from_bytes(string.encode(), byteorder='big') % modulo


def lognorm_convert(mean, stdev):
    """
    Lognormal distributions can be specified in terms of the mean and standard
    deviation of the "overlying" lognormal distribution, or the "underlying" normal distribution.
    This function converts the parameters from the lognormal distribution to the
    parameters of the underlying distribution, which are the form expected by NumPy's
    and SciPy's lognorm() distributions.
    """
    if mean <= 0:
        errormsg = f'Cannot create a lognorm_o distribution with mean≤0 (mean={mean}); did you mean to use lognorm_u instead?'
        raise ValueError(errormsg)
    std2 = stdev**2
    mean2 = mean**2
    under_std = np.sqrt(np.log(std2/mean2 + 1)) # Computes stdev for the underlying normal distribution
    under_mean  = np.log(mean2 / np.sqrt(std2 + mean2)) # Computes the mean of the underlying normal distribution
    return under_mean, under_std


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
        self._spars = None # If needed, set the scipy.stats parameters
        
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
    
    # def preprocess_pars(self):
    #     """ Do any preprocessing on keywords """
    #     pass

    def call_pars(self):
        """ Check if any parameters need to be called to be turned into arrays """
        size, uids = self._size, self._uids
        
        self.array_pars = False
        for key,val in self._pars.items():
            
            # If the parameter is callable, then call it
            if callable(val): 
                size_par = uids if uids is not None else size
                out = val(self.sim, self.module, size_par)
                val = np.asarray(out) # Necessary since UIDArrays don't allow slicing # TODO: check if this is correct
                self._pars[key] = val
            
            # If it's iterable, check the size and pad with zeros if it's the wrong shape
            if np.iterable(val) and uids is not None and self.dist != 'choice': # TODO: figure out logic
                self.array_pars = True
        return
    
    def transform_pars(self):
        """ Perform any necessary transformations on distribution parameters """
        pass
    
    def translate_pars(self):
        """ Translate keywords from Numpy into Scipy """
        pass
    
    
    
    def process_pars(self):
        """ Ensure the supplied dist and parameters are valid, and initialize them; called automatically """
        # dist = self.dist # The name of the distribution (a string, usually)
        self._pars = self.pars.copy() # The actual keywords; shallow copy, modified below for special cases
        self.call_pars()
        self.transform_pars()
        self.translate_pars()
        # self.preprocess_pars()
        
        
        
        self._pars = pars
        
        # Main use case: handle strings, including special cases
        if isinstance(dist, str):
            
            # Handle lognormal distributions
            if dist == 'lognorm_o': # Convert parameters for a lognormal
                pars['mean'], pars['sigma'] = lognorm_convert(pars.pop('mean'), pars.pop('stdev'))
                dist = 'lognormal'
            elif dist == 'lognorm_u':
                pars['mean'] = pars.pop('loc') # Rename parameters
                pars['sigma'] = pars.pop('scale')
                dist = 'lognormal' # For the underlying distribution
            
            # # Create the actual distribution -- first the special cases of Bernoulli and delta
            # if dist == 'bernoulli': # Special case, predefine the distribution here
            #     dist = lambda p, size: 

        
        # # It wasn't a string, so assume it's a SciPy distribution
        # else:
        #     self.method = 'scipy' if callable(dist) else 'frozen' # Need to handle regular and frozen distributions differently
        #     if hasattr(dist, 'random_state'): # For SciPy distributions # TODO: Check if safe with non-frozen (probably not?)
        #         dist.random_state = self.rng # Override the default random state with the correct one
        #     else:
        #         errormsg = f'Unknown distribution {type(dist)}: must be string or scipy.stats distribution, or another distribution with a random_state attribute'
        #         raise TypeError(errormsg)
        
        # Now that we have the dist function, process the keywords for callable and array inputs
        
            
        return dist, pars
    
    def rand(self, size=None):
        """ Simple way to get simple random numbers """
        if size is None: size = self._size
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
        self.process_pars()
        
        # Store the state
        self.make_history() # Store the pre-call state
        
        # Actually get the random numbers
        if self.is_array:
            rands = self.rand(size)[slots] # Get random values 
            rvs = self.ppf(rands)
        else:
            rvs = self.make_rvs()
            if slots is not None:
                rvs = rvs[slots]
        
        # # Actually get the random numbers # TODO: tidy up!
        # if not self.has_array_pars:
        #     if self.method == 'numpy':
        #         if isinstance(dist, str): # Main use case: get the distribution
        #             dist = getattr(self.rng, dist)
        #         rvs = dist(size=size, **pars)
        #     elif self.method == 'scipy':
        #         rvs = dist.rvs(size=size, **pars)
        #     elif self.method == 'frozen': 
        #         rvs = dist.rvs(size=size) # Frozen distributions don't take keyword arguments
        #     else:
        #         raise ValueError(f'Unknown method: {self.method}') # Should not happen
        #     if slots is not None:
        #         rvs = rvs[slots]
        # else:
        #     rands = self.rand(size)[slots]
        #     if self.method == 'numpy':
        #         mapping = dict(normal='norm', lognormal='lognorm')
        #         dname = mapping[self.dist] if self.dist in mapping else self.dist # TODO: refactor
        #         if dname == 'uniform': # TODO: hack to get uniform to work since different args for SciPy
        #             pars['loc'] = pars.pop('low')
        #             pars['scale'] = pars.pop('high') - pars['loc']
        #         spdist = getattr(sps, dname)(**pars) # TODO: make it work better, not actually numpy
        #     elif self.method == 'scipy':
        #         spdist = dist(**pars)
        #     elif self.method == 'frozen': 
        #         spdist = dist
        #     else:
        #         raise ValueError(f'Unknown method: {self.method}') # Should not happen
            
        #     rvs = spdist.ppf(rands) # Use the PPF to convert the quantiles to the actual random variates
        
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
dist_list = ['random', 'uniform', 'normal', 'lognorm_o', 'lognorm_u', 'expon',
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
    
    # def translate_pars(self, pars):
    #     """ Translate keywords from default (numpy) versions into scipy.stats versions """
    #     self._spars = dict(
    #         loc = pars.low,
    #         scale = pars.high - pars.low,
    #     )
    #     return
    
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


class lognorm_u(Dist):
    """
    Lognormal distribution, parameterized in terms of the "underlying" (normal)
    distribution, with mean=loc and stdev=scale (see lognorm_o for comparison).
    
    Note: the "loc" parameter here does *not* correspond to the mean of the resulting
    random variates!
    
    **Example**::
        
        ss.lognorm_u(loc=2, scale=1).rvs(1000).mean() # Should be roughly 10
    """
    def __init__(self, loc=0.0, scale=1.0, **kwargs):
        super().__init__(spsdist=sps.lognorm, loc=loc, scale=scale, **kwargs)
        return
    
    def make_rvs(self):
        p = self._pars
        return self.rng.normal(loc=p.loc, scale=p.scale, size=self._size)


def lognorm_o(mean=1.0, stdev=1.0, **kwargs):
    """
    Lognormal distribution, parameterized in terms of the "overlying" (lognormal)
    distribution, with mean=mean and stdev=stdev (see lognorm_u for comparison).
    
    **Example**::
        
        ss.lognorm_o(mean=2, stdev=1).rvs(1000).mean() # Should be close to 2
    """
    return Dist(dist='lognorm_o', mean=mean, stdev=stdev, **kwargs)


def expon(scale=1.0, **kwargs):
    """ Exponential distribution """
    return Dist(dist='exponential', scale=scale, **kwargs)

def poisson(lam=1.0, **kwargs):
    """ Poisson distribution """
    return Dist(dist='poisson', lam=lam, **kwargs)

def randint(low=None, high=None, **kwargs):
    """ Random integers, values on the interval [low, high-1] (i.e. "high" is excluded) """
    if low is None and high is None: # Ugly, but gets the default of acting like a Bernoulli trial with no input
        low = 2 # Note that the endpoint is excluded, so this is [0,1]
    return Dist(dist='integers', low=low, high=high, **kwargs)

def weibull(a=1.0, **kwargs):
    """ Weibull distribution (note: there is no scale parameter) """
    return Dist(dist='weibull', a=a, **kwargs)


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


def choice(a=2, p=None, **kwargs):
    """
    Random choice between discrete options
    
    **Examples**::
        
        # Simulate 10 die rolls
        ss.choice(6)(10) + 1 
        
        # Choose between specified options each with a specified probability (must sum to 1)
        ss.choice(a=[30, 70], p=[0.3, 0.7])(10)
    
    """
    return Dist(dist='choice', a=a, p=p, **kwargs)



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