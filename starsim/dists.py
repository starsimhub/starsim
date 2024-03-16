import hashlib
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Dists', 'Dist']


def str2int(string, modulo=1e8):
    """
    Convert a string to an int via hashing
    
    Cannot use Python's built-in hash() since it's randomized for strings. While
    hashlib is slower, this function is only used at initialization, so makes no
    appreciable difference to runtime.
    """
    return int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % int(modulo)


class Dists(sc.prettyobj):
    """ Class for managing a collection of Dist objects """

    def __init__(self):
        self.dists = ss.ndict()
        self.used_offsets = set()
        self.base_seed = None
        self.current_seed = None
        self.initialized = False
        return

    def initialize(self, base_seed=0):
        self.base_seed = base_seed
        self.initialized = True
        return
    
    def find_dists(self, obj):
        """
        Find all distributions in an object
        
        In practice, the object is usually a Sim. This function returns a 
        """
        
    
    def add(self, dist, check_repeats=True):
        """
        Keep track of a new Dist
        
        Can request an offset, will check for overlap
        Otherwise, return value will be used as the seed offset for this rng
        """
        if not self.initialized:
            raise NotInitializedException()

        if dist.name in self.dists:
            raise RepeatNameException(dist.name)

        if check_repeats:
            if dist.offset in self.used_offsets:
                raise SeedRepeatException(dist.name, dist.offset)
            self.used_offsets.add(dist.offset)

        self.dists.append(dist)
        self.current_seed = self.base_seed + dist.offset # Add in the base seed

        return self.current_seed

    def step(self, ti):
        """ Step each RNG forward, for each sim timestep """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.step(ti)
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
    >>> dist = ss.Dist('Test') # The hashed name determines the seed offset.
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
    
    def __init__(self, dist=None, name=None, offset=None, **kwargs):
        """
        Create a random number generator
        
        Args:
            dist (str): the name of the (default) distribution to draw random numbers from, or a SciPy distribution
            name (str): the unique name of this distribution, e.g. "coin_flip" (in practice, usually generated automatically)
            offset (int): the seed offset; will be automatically assigned (based on hashing the name) if None
            kwargs (dict): (default) parameters of the distribution
            
        **Examples**::
            
            dist = ss.Dist('normal', loc=3)
            dist.rvs(10) # Return 10 normally distributed random numbers
        """
        self.dist = dist
        self.name = name
        self.kwds = kwargs
        self.offset = offset
        
        if dist is None:
            errormsg = 'You must supply the name of a distribution, or a SciPy distribution'
            raise ValueError(errormsg)
            
        self.seed = None # Will be determined once added to the container
        self.rng = None # The actual RNG generator for generating random numbers
        self.ready = True
        self.initialized = False
        return
    
    def __getattr__(self, attr):
        """ Make it behave like a random number generator mostly -- enables things like uniform(), normal(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self.rng, attr)
        
    @property
    def bitgen(self):
        try:
            return self.rng.bit_generator
        except:
            return None
        
    def initialize(self, path=None):
        """ Calculate the starting seed and create the RNG """
        
        if self.offset is None: # Obtain the seed offset by hashing the path to this distribution
            name = path or self.name
            self.offset = str2int(name)
        self.seed = self.offset # Initialize these to be the same # TODO: is this needed?
        
        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.init_state = self.bitgen.state # Store the initial state
        
        # Initialize the distribution, if not a string
        if isinstance(self.dist, str):
            if self.dist == 'bernoulli': # Special case for a Bernoulli distribution: a binomial distribution with n=1
                self.dist = 'binomial' # TODO: check performance; Covasim uses np.random.random(len(prob_arr)) < prob_arr
                self.kwds['n'] = 1
        else:
            if callable(self.dist):
                self.dist = self.dist(**self.kwds)
            if hasattr(self.dist, 'random_state'):
                self.dist.random_state = self.rng # Override the default random state with the correct one
            else:
                errormsg = f'Unknown distribution type {type(self.dist)}: must be string or scipy.stats distribution'
                raise TypeError(errormsg)
        
        # Finalize
        self.ready = True
        self.initialized = True
        return

    def reset(self):
        """ Restore initial state """
        self.bitgen.state = self.init_state
        self.ready = True
        return self.bitgen.state

    def step(self, ti):
        """ Advance to time ti step by jumping """
        self.reset() # First reset back to the initial state
        self.bitgen.state = self.bitgen.jumped(jumps=ti).state # Now take ti jumps
        return self.bitgen.state
    
    def rvs(self, size=1, **kwargs):
        """ Main method for getting random numbers """
        kwds = sc.mergedicts(dict(size=size), self.kwds, kwargs)
        if isinstance(self.dist, str):
            dist = getattr(self.rng, self.dist)
            rvs = dist(**kwds)
        else:
            if not np.isscalar(size):
                print('WARNING, not random number safe yet!')
                size = len(size)
            rvs = self.dist.rvs(size=size, **kwargs) # TODO: CHECK!!!!!
        if ss.options.multirng:
            self.ready = False # Needs to reset before being called again
        return rvs

    def filter(self, size, **kwargs):
        return size[self.rvs(size, **kwargs)]



#%% Specific distributions

# Add common distributions so they can be imported directly
__all__ += ['random', 'uniform', 'normal', 'lognormal', 'expon', 'poisson', 'randint', 'weibull', 'bernoulli']

def random(*args, **kwargs):    return Dist(dist='random', *args, **kwargs)
def uniform(*args, **kwargs):   return Dist(dist='uniform', *args, **kwargs)
def normal(*args, **kwargs):    return Dist(dist='normal', *args, **kwargs)
def lognormal(*args, **kwargs): return Dist(dist='lognormal', *args, **kwargs)
def expon(*args, **kwargs):     return Dist(dist='exponential', *args, **kwargs)
def poisson(*args, **kwargs):   return Dist(dist='poisson', *args, **kwargs)
def randint(*args, **kwargs):   return Dist(dist='integer', *args, **kwargs)
def weibull(*args, **kwargs):   return Dist(dist='weibull', *args, **kwargs)
def bernoulli(*args, **kwargs): return Dist(dist='bernoulli', *args, **kwargs)




#%% Dist exceptions

class NotInitializedException(RuntimeError):
    "Raised when a random number generator or a RNGContainer object is called when not initialized."
    def __init__(self, obj_name=None):
        if obj_name is None: 
            msg = 'An RNG is being used without proper initialization; please initialize first'
        else:
            msg = f'The RNG "{obj_name}" is being used prior to initialization.'
        super().__init__(msg)

class NotReadyException(RuntimeError):
    "Raised when a random generator is called without being ready."
    def __init__(self, rng_name):
        msg = f'The random generator named "{rng_name}" was not ready when called. This error is likely caused by calling a distribution or underlying MultiRNG generator two or more times in a single step.'
        super().__init__(msg)

class SeedRepeatException(ValueError):
    "Raised when two random number generators have the same seed."
    def __init__(self, rng_name, seed_offset):
        msg = f'Requested seed offset {seed_offset} for the random number generator named {rng_name} has already been used.'
        super().__init__(msg)

class RepeatNameException(ValueError):
    "Raised when adding a random number generator to a RNGContainer when the rng name has already been used."
    def __init__(self, rng_name):
        msg = f'A random number generator with name {rng_name} has already been added.'
        super().__init__(msg)