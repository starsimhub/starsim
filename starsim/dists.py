"""
Define random-number-safe distributions.
"""

import hashlib
import numpy as np
import sciris as sc

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
    tree = sc.iterobj(obj)
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
        sim = sim if sim else self.sim
        obj = obj if obj else self.obj
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
    
    def find_dists(self, obj):
        """
        Find all distributions in an object
        
        In practice, the object is usually a Sim. This function returns a 
        """
        self.dists = find_dists(obj)
        return self.dists
    
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

    def jump(self, delta=1, to=None):
        """ Advance all RNGs, e.g. to timestep "to", by jumping """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.jump(delta=delta, to=to)
        return out

    def reset(self):
        """ Reset each RNG """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.reset()
        return out


class Dist(sc.prettyobj):
    """
    Class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.

    The main use case is to sample random numbers from various distributions
    that are specific to each agent (per decision and timestep) so as to enable
    variance reduction between simulations through the use of common random
    numbers. For example, the user might create a random number generator called
    rng and ultimately ask for randomly distributed random numbers for agents
    with UIDs 1 and 4:

    >>> import starsim as ss
    >>> import numpy as np
    >>> dist = ss.Dist('random', name='test') # The hashed name determines the seed offset.
    >>> dist.initialize(slots=5) # In practice, slots will be sim.people.slots. When scalar (for testing), an np.arange will be used.
    >>> uids = np.array([1,4])
    >>> dist.random(uids)
    array([0.88110549, 0.86915719])

    In theory, what this is doing is drawing 5 random numbers and returning the
    draws at positions 1 and 4.

    In practice, using UIDs as "slots" (the indices into the larger draw) falls
    apart when new agents are born.  The issue is that one simulation might have
    more births than another, so an agent born in one simulation may not
    get the same UID as that same agent in a comparison simulation.
    
    The solution applied here is for each agent to have a property called "slot"
    that is precisely the index used when selecting from an array of random
    numbers.  When new agents are born, the mother uses her UID to sample a
    random integer for the newborn that is used as the "slot".  With this
    approach, newborns will be identical between two different simulations,
    unless an intervention mechanistically drove a change.

    The slot-based approach is not without challenges.
    * Two newborn agents may received the same "slot," and thus will receive the
      same random draws.
    * The chance of overlapping slots can be reduced by
      allowing mothers to choose from a larger number of possible slots (say up
      to one-million). However, as slots are used as indices, the number of
      random variables drawn for each query must number the maximum slot. So if
      one agent has a slot of 1M, then 1M random numbers will be drawn,
      consuming more time than would be necessary if the maximum slot was
      smaller.
    * The maximum slot is now determined by a new configure parameter named
      "slot_scale". A value of 5 will mean that new agents will be assigned
      slots between 1*N and 5*N, where N is sim.pars['n_agents'].
    """
    
    def __init__(self, dist=None, name=None, seed=None, offset=None, module=None, sim=None, strict=False, **kwargs): # TODO: switch back to strict=True
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
        self.seed = seed # Usually determined once added to the container
        self.offset = offset
        self.module = module
        self.sim = sim
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
    
    def disp_state(self):
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
        string = sc.newlinejoin(s)
        print(string)
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
        
    def initialize(self, trace=None, seed=0, module=None, sim=None):
        """ Calculate the starting seed and create the RNG """
        
        # Calculate the offset (starting seed)
        self.set_offset(trace, seed)
        
        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.init_state = self.bitgen.state # Store the initial state
        self.make_dist() # Convert the inputs into an actual validated distribution
        
        # Finalize
        self.module = module if module else self.module
        self.sim = sim if sim else self.sim
        self.ready = True
        self.initialized = True
        return self
    
    def set_offset(self, trace=None, seed=0):
        """ Obtain the seed offset by hashing the path to this distribution """
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
    
    def set(self, dist=None, **kwargs):
        """ Set (change) the distribution type, or one or more parameters of the distribution """
        if dist:
            self.dist = dist
        self.kwds = sc.mergedicts(self.kwds, kwargs)
        self.make_dist()
        return
    
    def make_dist(self):
        """ Ensure the supplied dist is valid, and initialize it if needed """
        dist = self.dist # The name of the distribution (a string, usually)
        self._dist = None # The actual distribution function
        self._kwds = sc.dcp(self.kwds) # The actual keywords; modified below for special cases
        
        # Handle strings, including special cases
        if isinstance(dist, str):
            # Handle special cases
            if dist == 'lognorm_o': # Convert parameters for a lognormal
                self._kwds.mean, self._kwds.sigma = lognorm_convert(self._kwds.pop('mean'), self._kwds.pop('stdev'))
                dist = 'lognormal'
            elif dist == 'lognorm_u':
                self._kwds.mean = self._kwds.pop('loc') # Rename parameters
                self._kwds.sigma = self._kwds.pop('scale')
                dist = 'lognormal' # For the underlying distribution
            
            # Create the actual distribution
            if dist == 'bernoulli': # Special case, predefine the distribution here
                self._dist = lambda size, p: self.rng.random(size) < p # 3x faster than using rng.binomial(1, p, size)
            elif dist == 'delta': # Special case, predefine the distribution here
                self._dist = lambda size, v: np.full(size, fill_value=v)
            else:
                try:
                    self._dist = getattr(self.rng, dist) # Main use case; replace the string with the actual distribution
                except Exception as E:
                    errormsg = f'Could not interpret "{dist}", are you sure this is a valid distribution? (i.e., an attribute of np.random.default_rng())'
                    raise ValueError(errormsg) from E
        
        # It's not a string, so assume it's a SciPy distribution
        else:
            self._dist = dist # Use directly
            self.method = 'scipy' if callable(dist) else 'frozen' # Need to handle regular and frozen distributions differently
                
            if hasattr(self._dist, 'random_state'): # For SciPy distributions # TODO: Check if safe with non-frozen (probably not?)
                self._dist.random_state = self.rng # Override the default random state with the correct one
            else:
                errormsg = f'Unknown distribution {type(dist)}: must be string or scipy.stats distribution, or another distribution with a random_state attribute'
                raise TypeError(errormsg)
        return

    def reset(self):
        """ Restore initial state """
        self.bitgen.state = self.init_state
        self.ready = True
        return self.bitgen.state

    def jump(self, delta=1, to=None):
        """ Advance the RNG, e.g. to timestep "to", by jumping """
        jumps = to if to else self.ind + delta
        self.ind = jumps
        self.reset() # First reset back to the initial state (used in case of different numbers of calls)
        if jumps: # Seems to randomize state if jumps=0
            self.bitgen.state = self.bitgen.jumped(jumps=jumps).state # Now take "jumps" number of jumps
        return self.bitgen.state
    
    def process_kwds(self, size, uids=None):
        """ Handle array and callable keyword arguments """
        kwds = dict() # Loop over parameters (keywords) and modify any that are callable or arrays of the wrong shape
        for key,val in self._kwds.items(): 
            if callable(val): # If the parameter is callable, then call it
                size_par = uids if uids is not None else size
                out = val(self.module, self.sim, size_par)
                val = np.asarray(out) # Necessary since UIDArrays don't allow slicing # TODO: check if this is correct
            if np.iterable(val): # If it's iterable, check the size and pad with zeros if it's the wrong shape
                if uids is not None and (len(val) == len(uids)):
                    resized = np.zeros(size, dtype=val.dtype)
                    resized[uids] = val[:len(uids)]
                    val = resized
                if len(val) != size: # TODO: handle multidimensional?
                    errormsg = f'Shape mismatch: dist parameter has length {len(val)}, but {size} elements are needed'
                    raise ValueError(errormsg)
            kwds[key] = val # Replace 
        return kwds
    
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
        kwds = self.process_kwds(size, uids)
        if self.method == 'numpy':
            rvs = self._dist(size=size, **kwds)
        elif self.method == 'scipy':
            rvs = self._dist.rvs(size=size, **kwds)
        elif self.method == 'frozen': 
            rvs = self._dist.rvs(size=size) # Frozen distributions don't take keyword arguments
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
        maxval = uids.max() + 1 # Since UIDs are inclusive
        urvs = self.rvs(size=maxval, uids=uids)
        return urvs

    def filter(self, uids, **kwargs):
        return uids[self.urvs(uids, **kwargs).astype(bool)] # TODO: tidy up



#%% Specific distributions

# Add common distributions so they can be imported directly
dist_list = ['random', 'uniform', 'normal', 'lognorm_o', 'lognorm_u', 'expon',
             'poisson', 'weibull', 'delta', 'randint', 'bernoulli']
__all__ += dist_list

def random(**kwargs):
    return Dist(dist='random', **kwargs)

def uniform(low=0.0, high=1.0, **kwargs):
    return Dist(dist='uniform', low=low, high=high, **kwargs)

def normal(loc=0.0, scale=1.0, **kwargs):
    return Dist(dist='normal', loc=loc, scale=scale, **kwargs)

def lognorm_o(mean=1.0, stdev=1.0, **kwargs):
    return Dist(dist='lognorm_o', mean=mean, stdev=stdev, **kwargs)

def lognorm_u(loc=0.0, scale=1.0, **kwargs):
    return Dist(dist='lognorm_u', loc=loc, scale=scale, **kwargs)

def expon(scale=1.0, **kwargs):
    return Dist(dist='exponential', scale=scale, **kwargs)

def poisson(lam=1.0, **kwargs):
    return Dist(dist='poisson', lam=lam, **kwargs)

def randint(low=None, high=None, **kwargs):
    if low is None and high is None: # Ugly, but gets the default of acting like a Bernoulli trial with no input
        low = 2 # Note that the endpoint is excluded, so this is [0,1]
    return Dist(dist='integers', low=low, high=high, **kwargs)

def weibull(a=1.0, **kwargs):
    return Dist(dist='weibull', a=a, **kwargs)

def delta(v=0, **kwargs):
    return Dist(dist='delta', v=v, **kwargs)

def bernoulli(p=0.5, **kwargs):
    return Dist(dist='bernoulli', p=p, **kwargs)



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