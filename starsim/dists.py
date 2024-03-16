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


def lognormal_params(mean, stdev):
    """
    Lognormal distributions can be specified in terms of the mean and standard
    deviation of the lognormal distribution, or the underlying normal distribution.
    This function converts the parameters from the lognormal distribution to the
    parameters of the underlying distribution, which are the form expected by NumPy's
    and SciPy's lognorm() distributions.
    """
    std2 = stdev**2
    mean2 = mean**2
    under_mean = np.sqrt(np.log(std2/mean2 + 1))
    mu = np.log(mean2 / np.sqrt(std2+mean2))
    under_std = np.exp(mu)
    return under_mean, under_std


def find_dists(obj):
    """ Find all Dist objects in a parent object """
    out = sc.objdict()
    tree = sc.iterobj(obj)
    for trace,val in tree.items():
        if isinstance(val, Dist):
            out[str(trace)] = val
    return out


class Dists(sc.prettyobj):
    """ Class for managing a collection of Dist objects """

    def __init__(self, obj=None, base_seed=None):
        self.obj = obj
        self.dists = None
        self.base_seed = base_seed
        self.initialized = False
        if self.obj is not None:
            self.initialize()
        return

    def initialize(self, obj=None, base_seed=None):
        """
        Set the base seed, find and initialize all distributions in an object
        
        In practice, the object is usually a Sim, but can be anything.
        """
        if base_seed:
            self.base_seed = base_seed
        obj = obj if obj else self.obj
        self.dists = find_dists(obj)
        for trace,dist in self.dists.items():
            dist.initialize(trace=trace, seed=base_seed)
        self.initialized = True
        return self
    
    def find_dists(self, obj):
        """
        Find all distributions in an object
        
        In practice, the object is usually a Sim. This function returns a 
        """
        self.dists = find_dists(obj)
        return self.dists

    def jump(self, delta=1, ti=None):
        """ Advance all RNGs, e.g. to timestep ti, by jumping """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.jump(delta=delta, ti=ti)
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
    
    def __init__(self, dist=None, name=None, seed=None, offset=None, strict=True, **kwargs):
        """
        Create a random number generator
        
        Args:
            dist (str): the name of the (default) distribution to draw random numbers from, or a SciPy distribution
            name (str): the unique name of this distribution, e.g. "coin_flip" (in practice, usually generated automatically)
            seed (int): if supplied, the seed to use for this distribution
            offset (int): the seed offset; will be automatically assigned (based on hashing the name) if None
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
        self.strict = strict
        
        if dist is None:
            errormsg = 'You must supply the name of a distribution, or a SciPy distribution'
            raise ValueError(errormsg)
            
        # Internal state
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self.ind = 0 # The index of the RNG (usually updated on each timestep)
        self.is_scipy = False # Need a flag because rvs logic is different
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
        s += f'   ind = {self.ind}'
        s += f'offset = {self.offset}'
        s += f'  seed = {self.seed}'
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
        
    def initialize(self, trace=None, seed=0):
        """ Calculate the starting seed and create the RNG """
        
        # Calculate the offset (starting seed)
        self.set_offset(trace, seed)
        
        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.init_state = self.bitgen.state # Store the initial state
        self.make_dist() # Convert the inputs into an actual validated distribution
        
        # Finalize
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
        dist = self.dist
        self._dist = None # The processed distribution
        self._kwds = sc.dcp(self.kwds) # The actual keywords
        
        # Handle special cases for strings
        if isinstance(dist, str):
            if dist == 'bernoulli': # Special case for a Bernoulli distribution: a binomial distribution with n=1
                dist = 'binomial' # TODO: check performance; Covasim uses np.random.random(len(prob_arr)) < prob_arr
                self._kwds['n'] = 1
            elif dist == 'lognormal': # Convert parameters for a lognormal
                self._kwds.mean, self._kwds.sigma = lognormal_params(self._kwds.pop('loc'), self._kwds.pop('scale'))
                
            try:
                if dist == 'delta': # Special case, predefine the distribution here
                    self._dist = lambda size, v: np.full(size, fill_value=v)
                else:
                    self._dist = getattr(self.rng, dist) # Replace the string with the actual distribution
            except Exception as E:
                errormsg = f'Could not interpret "{dist}", are you sure this is a valid distribution? (i.e., an attribute of np.random.default_rng())'
                raise ValueError(errormsg) from E
        
        # It's not a string, so assume it's a SciPy distribution
        else:
            if callable(dist):
                self._dist = dist(**self._kwds) # Create the frozen distribution
            else:
                self._dist = dist # Assume it already is a frozen distribution
                
            if hasattr(self._dist, 'random_state'): # For SciPy distributions
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

    def jump(self, delta=1, ti=None):
        """ Advance the RNG, e.g. to timestep ti, by jumping """
        jumps = ti if ti else self.ind + delta
        self.ind = jumps
        self.reset() # First reset back to the initial state (used in case of different numbers of calls)
        if jumps: # Seems to randomize state if jumps=0
            self.bitgen.state = self.bitgen.jumped(jumps=jumps).state # Now take ti jumps
        return self.bitgen.state
    
    def rvs(self, size=1):
        """ Main method for getting random variables """
        
        # Check for readiness
        if not self.ready and self.strict:
            raise DistNotReady(self)
            
        # Actually get the random numbers
        if self.is_scipy:
            rvs = self._dist.rvs(size=size) 
        else:
            rvs = self._dist(size=size, **self._kwds) # Actually get the random numbers
        
        # Tidy up
        if self.strict:
            self.ready = False
        return rvs
    
    def urvs(self, uids):
        """ Like rvs(), but get based on a list of unique identifiers (UIDs or slots) instead """
        maxval = uids.max()
        rvs = self.rvs(size=maxval)
        return rvs[uids]

    def filter(self, size, **kwargs):
        return size[self.rvs(size, **kwargs)]



#%% Specific distributions

# Add common distributions so they can be imported directly
dist_list = ['random', 'uniform', 'normal', 'lognormal', 'expon', 'poisson', 'weibull', 'delta', 'randint', 'bernoulli']
__all__ += dist_list

def random(**kwargs):
    return Dist(dist='random', **kwargs)

def uniform(low=0.0, high=1.0, **kwargs):
    return Dist(dist='uniform', low=low, high=high, **kwargs)

def normal(loc=0.0, scale=1.0, **kwargs):
    return Dist(dist='normal', loc=loc, scale=scale, **kwargs)

def lognormal(loc=1.0, scale=1.0, **kwargs):
    return Dist(dist='lognormal', loc=loc, scale=scale, **kwargs)

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

class DistNotInitialized(RuntimeError):
    "Raised when a random number generator or a RNGContainer object is called when not initialized."
    def __init__(self, obj_name=None):
        if obj_name is None: 
            msg = 'An RNG is being used without proper initialization; please initialize first'
        else:
            msg = f'The RNG "{obj_name}" is being used prior to initialization.'
        super().__init__(msg)

class DistNotReady(RuntimeError):
    "Raised when a random generator is called without being ready."
    def __init__(self, dist):
        msg = f'{dist} is not ready. This is likely caused by calling a distribution multiple times in a single step. Call dist.jump() to reset.'
        super().__init__(msg)

# class SeedRepeatException(ValueError):
#     "Raised when two random number generators have the same seed."
#     def __init__(self, rng_name, seed_offset):
#         msg = f'Requested seed offset {seed_offset} for the random number generator named {rng_name} has already been used.'
#         super().__init__(msg)

# class RepeatNameException(ValueError):
#     "Raised when adding a random number generator to a RNGContainer when the rng name has already been used."
#     def __init__(self, rng_name):
#         msg = f'A random number generator with name {rng_name} has already been added.'
#         super().__init__(msg)