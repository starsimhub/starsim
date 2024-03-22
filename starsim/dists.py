"""
Define random-number-safe distributions.
"""

import hashlib
import numpy as np
import sciris as sc
import pylab as pl

__all__ = ['find_dists', 'Dists', 'Dist']


def str2int(string, modulo=1e8):
    """
    Convert a string to an int via hashing
    
    Cannot use Python's built-in hash() since it's randomized for strings. While
    hashlib is slower, this function is only used at initialization, so makes no
    appreciable difference to runtime.
    """
    return int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % int(modulo)


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
    Class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.
    
    Args:
        dist (str/dist): the type of distribution to use; can be any attribute of ``np.random.default_rng()``, or any distribution from ``scipy.stats``
        name (str): the name for this distribution
        seed (int): the user-chosen random seed (e.g. 3)
        offset (int): usually calculated on initialization; the unique identifier of this distribution
        strict (bool): whether to prevent multiple draws without resetting
        module (Module): usually calculated on initialization; the module to use for lambda functions
        sim (Sim): usually calculated on initialization; the sim to use for lambda functions
        kwargs (dict): the parameters of the distribution to be called
    
    **Examples**::
        
        # Create and use a simple distribution
        dist = ss.Dist('random')
        dist(5)
        
        # Create a more complex distribution
        ss.Dist('lognorm_o', seed=3, mean=2, stdev=5).rvs(5)
        
        # Draw using UIDs rather than a fixed size
        uids = np.array([1,2,4,9])
        ss.Dist('bernoulli', p=0.5).urvs(uids)
    """
    def __init__(self, dist=None, name=None, seed=None, offset=None, strict=False, module=None, sim=None, **kwargs): # TODO: switch back to strict=True
        """
        Create a random number generator
        
        Args:
            dist (str): the name of the (default) distribution to draw random numbers from, or a SciPy distribution
            name (str): the unique name of this distribution, e.g. "coin_flip" (in practice, usually generated automatically)
            seed (int): if supplied, the seed to use for this distribution
            offset (int): the seed offset; will be automatically assigned (based on hashing the name) if None
            module (ss.Module): if provided, used when supplying lambda-function arguments as parameters
            sim (ss.Sim): if provided, used when supplying lambda-function arguments as parameters
            strict (bool): if True, require initialization and invalidate after each call to rvs()
            kwargs (dict): (default) parameters of the distribution
            
        **Examples**::
            
            dist = ss.Dist('normal', loc=3)
            dist.rvs(10) # Return 10 normally distributed random numbers
        """
        self.dist = dist
        self.name = name
        self.kwds = sc.dictobj(kwargs)
        self._seed = seed # Usually determined once added to the container
        self.seed = None
        self.offset = offset
        self.module = module
        self.sim = sim
        self.slots = None # Created on initialization with a sim
        self.strict = strict
        
        if dist is None:
            errormsg = 'You must supply the name of a distribution, or a SciPy distribution'
            raise ValueError(errormsg)
            
        # Internal state
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self.ind = 0 # The index of the RNG (usually updated on each timestep)
        self.called = 0 # The number of times the distribution has been called
        self.method = 'numpy' # Flag whether the method to call the distribution is 'numpy' (default), 'scipy', or 'frozen'
        self.ready = True
        self.initialized = False
        if not strict:
            self.initialize()
        return
    
    def __repr__(self):
        """ Custom display to show state of object """
        string = f'ss.Dist("{self.trace}", dist={self.dist}, kwds={dict(self.kwds)})'
        return string
    
    def disp(self):
        """ Return full display of object """
        return sc.pr(self)
    
    def show_state(self):
        """ Show the state of the object """
        s = sc.autolist()
        s += f'  dist = {self.dist}'
        s += f'  kwds = {self.kwds}'
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

    def __setitem__(self, key, value):
        if key not in self.kwds:
            raise Exception(f'Cannot set {key} for distribution of type {self.dist}.')
        # Set a parameter value
        self.kwds[key] = value
        return
    
    def __getattr__(self, attr):
        """ Make it behave like a random number generator mostly -- enables things like uniform(), normal(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self.rng, attr)
        
    def __call__(self, size=1):
        """ Alias to self.rvs() """
        return self.rvs(size=size)

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

    def reset(self, state=None):
        """ Restore initial state """
        state = state if (state is not None) else self.init_state
        self.rng.bit_generator.state = state.copy()
        self.ready = True
        return self.bitgen.state

    def jump(self, to=None, delta=1):
        """ Advance the RNG, e.g. to timestep "to", by jumping """
        jumps = to if (to is not None) else self.ind + delta
        self.ind = jumps
        self.reset() # First reset back to the initial state (used in case of different numbers of calls)
        if jumps: # Seems to randomize state if jumps=0
            self.bitgen.state = self.bitgen.jumped(jumps=jumps).state # Now take "jumps" number of jumps
        return self.bitgen.state
    
    def initialize(self, trace=None, seed=0, module=None, sim=None, slots=None):
        """ Calculate the starting seed and create the RNG """
        
        # Calculate the offset (starting seed)
        self.set_offset(trace, seed)
        
        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.init_state = sc.dcp(self.bitgen.state) # Store the initial state
        
        # Finalize
        self.module = module if (module is not None) else self.module
        self.sim = sim if (sim is not None) else self.sim
        if self.slots is None:
            if slots is None:
                if self.sim is not None and hasattr(self.sim, 'people') and hasattr(self.sim.people, 'slot'):
                    slots = self.sim.people.slot
            self.slots = slots 
        self.ready = True
        self.initialized = True
        return self
    
    def set_offset(self, trace=None, seed=0):
        """ Obtain the seed offset by hashing the path to this distribution; called automatically """
        unique_name = trace or self.trace or self.name
        if unique_name:
            if not self.name:
                self.name = unique_name
            self.trace = unique_name
            self.offset = str2int(unique_name) # Key step: hash the path to the distribution
        else:
            self.offset = self.offset or 0
        self.seed = self.offset + (seed or self._seed or 0)
        return
    
    def set(self, dist=None, **kwargs):
        """ Set (change) the distribution type, or one or more parameters of the distribution """
        if dist:
            self.dist = dist
        self.kwds = sc.mergedicts(self.kwds, kwargs)
        return
    
    def make_dist(self, size, uids=None):
        """ Ensure the supplied dist and parameters are valid, and initialize them; called automatically """
        dist = self.dist # The name of the distribution (a string, usually)
        kwds = self.kwds.copy() # The actual keywords; shallow copy, modified below for special cases
        
        # Main use case: handle strings, including special cases
        if isinstance(dist, str):
            
            # Handle lognormal distributions
            if dist == 'lognorm_o': # Convert parameters for a lognormal
                kwds['mean'], kwds['sigma'] = lognorm_convert(kwds.pop('mean'), kwds.pop('stdev'))
                dist = 'lognormal'
            elif dist == 'lognorm_u':
                kwds['mean'] = kwds.pop('loc') # Rename parameters
                kwds['sigma'] = kwds.pop('scale')
                dist = 'lognormal' # For the underlying distribution
            
            # Create the actual distribution -- first the special cases of Bernoulli and delta
            if dist == 'bernoulli': # Special case, predefine the distribution here
                dist = lambda p, size, self=self: self.rng.random(size) < p # 3x faster than using rng.binomial(1, p, size)
            elif dist == 'delta': # Special case, predefine the distribution here
                dist = lambda v, size: np.full(size, fill_value=v)
            else: # It's still a string, so try getting the function from the generator
                try:
                    dist = getattr(self.rng, dist) # Main use case: replace the string with the actual distribution
                except Exception as E:
                    errormsg = f'Could not interpret "{dist}", are you sure this is a valid distribution? (i.e., an attribute of np.random.default_rng())'
                    raise ValueError(errormsg) from E
        
        # It wasn't a string, so assume it's a SciPy distribution
        else:
            self.method = 'scipy' if callable(dist) else 'frozen' # Need to handle regular and frozen distributions differently
            if hasattr(dist, 'random_state'): # For SciPy distributions # TODO: Check if safe with non-frozen (probably not?)
                dist.random_state = self.rng # Override the default random state with the correct one
            else:
                errormsg = f'Unknown distribution {type(dist)}: must be string or scipy.stats distribution, or another distribution with a random_state attribute'
                raise TypeError(errormsg)
        
        # Now that we have the dist function, process the keywords for callable and array inputs
        for key,val in kwds.items():
            
            # If the parameter is callable, then call it
            if callable(val): 
                size_par = uids if uids is not None else size
                out = val(self.module, self.sim, size_par) # DJK, this needs to be UIDs, not slots!
                val = np.asarray(out) # Necessary since UIDArrays don't allow slicing # TODO: check if this is correct
                kwds[key] = val
            
            # If it's iterable, check the size and pad with zeros if it's the wrong shape
            if np.iterable(val) and uids is not None and (len(val) == len(uids)) and self.dist != 'choice':
                resized = np.zeros(size, dtype=val.dtype) # TODO: fix, problem from when there happen to be uid entries, but it's not slots
                resized[uids] = val[:len(uids)] # TODO: check if slicing is ok here
                val = resized
                kwds[key] = val # Replace 
            
        return dist, kwds
    
    def rvs(self, size=1, uids=None):
        """ Main method for getting random variables """
        
        # Check for readiness
        if not self.initialized:
            raise DistNotInitializedError(self)
        if not self.ready and self.strict:
            raise DistNotReadyError(self)
        
        # Shortcut if nothing to return
        if not np.isscalar(size):
            errormsg = f'Expecting a scalar size, not {size}; multidimensional output not supported and for UIDs, use urvs() instead'
            raise ValueError(errormsg)
        if size == 0:
            return np.array([], dtype=int) # int dtype allows use as index, e.g. when filtering
            
        # Actually get the random numbers
        dist, kwds = self.make_dist(size, uids)
        if self.method == 'numpy':
            if isinstance(dist, str): # Main use case: get the distribution
                dist = getattr(self.rng, dist)
            rvs = dist(size=size, **kwds)
        elif self.method == 'scipy':
            rvs = dist.rvs(size=size, **kwds)
        elif self.method == 'frozen': 
            rvs = dist.rvs(size=size) # Frozen distributions don't take keyword arguments
        else:
            raise ValueError(f'Unknown method: {self.method}') # Should not happen
        
        # Tidy up
        self.called += 1
        if self.strict:
            self.ready = False
        if uids is not None:
            rvs = rvs[uids]
            
        return rvs
    
    def urvs(self, uids):
        """ Like rvs(), but get based on a list of unique identifiers (UIDs or slots) instead """
        uids = np.asarray(uids)
        if not len(uids):
            return np.array([], dtype=int) # int dtype allows use as index, e.g. when filtering
        if self.slots is not None: # Use slots if available
            uids = self.slots[uids]
        maxval = uids.max() + 1 # Since UIDs are inclusive
        urvs = self.rvs(size=maxval, uids=uids)
        return urvs

    def filter(self, uids, **kwargs): # TODO: should this only be valid for Bernoulli distribution types?
        """ Filter UIDs by a binomial array """
        return uids[self.urvs(uids, **kwargs).astype(bool)] # TODO: tidy up
    
    def plot_hist(self, size=1000, bins=None, fig_kw=None, hist_kw=None):
        """ Plot the current state of the RNG as a histogram """
        pl.figure(**sc.mergedicts(fig_kw))
        state = self.get_state()
        rvs = self.rvs(size)
        self.reset(state=state) # As if nothing ever happened
        pl.hist(rvs, bins=bins, **sc.mergedicts(hist_kw))
        pl.title(str(self))
        pl.xlabel('Value')
        pl.ylabel(f'Count ({size} total)')
        return rvs
        

#%% Specific distributions

# Add common distributions so they can be imported directly; assigned to a variable since used in help messages
dist_list = ['random', 'uniform', 'normal', 'lognorm_o', 'lognorm_u', 'expon',
             'poisson', 'weibull', 'delta', 'randint', 'bernoulli', 'choice']
__all__ += dist_list


def random(**kwargs):
    """ Random distribution, values on interval (0,1) """
    return Dist(dist='random', **kwargs)

def uniform(low=0.0, high=1.0, **kwargs):
    """ Uniform distribution, values on interval (low, high) """
    return Dist(dist='uniform', low=low, high=high, **kwargs)

def normal(loc=0.0, scale=1.0, **kwargs):
    """ Normal distribution, with mean=loc and stdev=scale """
    return Dist(dist='normal', loc=loc, scale=scale, **kwargs)

def lognorm_o(mean=1.0, stdev=1.0, **kwargs):
    """
    Lognormal distribution, parameterized in terms of the "overlying" (lognormal)
    distribution, with mean=mean and stdev=stdev (see lognorm_u for comparison).
    
    **Example**::
        
        ss.lognorm_o(mean=2, stdev=1).rvs(1000).mean() # Should be close to 2
    """
    return Dist(dist='lognorm_o', mean=mean, stdev=stdev, **kwargs)

def lognorm_u(loc=0.0, scale=1.0, **kwargs):
    """
    Lognormal distribution, parameterized in terms of the "underlying" (normal)
    distribution, with mean=loc and stdev=scale (see lognorm_o for comparison).
    
    **Example**::
        
        ss.lognorm_u(loc=2, scale=1).rvs(1000).mean() # Should be roughly 10
    """
    return Dist(dist='lognorm_u', loc=loc, scale=scale, **kwargs)

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

def delta(v=0, **kwargs):
    """ Delta distribution: equivalent to np.full() """
    return Dist(dist='delta', v=v, **kwargs)

def bernoulli(p=0.5, **kwargs):
    """ Bernoulli distribution: return True or False with the specified probability (which can be an array) """
    return Dist(dist='bernoulli', p=p, **kwargs)

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