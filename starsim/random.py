import hashlib
from copy import deepcopy
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Dists', 'Dist', 'RNGs', 'RNG']


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

    def initialize(self, base_seed):
        self.base_seed = base_seed
        self.initialized = True
        return
    
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
    
    def __init__(self, dist='random', name=None, offset=None, **kwargs):
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

        # Get the offset
        if offset is None: # Obtain the seed offset by hashing the class name
            self.offset = str2int(self.name)
        else: # Use user-provided offset (unlikely)
            self.offset = offset

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
        
    def initialize(self, sim=None, container=None, slots=None):
        """ Calculate the starting seed and create the RNG """
        
        if sim: # TODO: can probably remove
            container = sim.dists
            slots = sim.people.slot
        
        # Set the seed, usually from within a container
        if container is not None:
            self.seed = container.add(self) # base_seed + offset
        else: # Enable use of Dist without a container
            self.seed = self.offset

        # Set up the slots (corresponding to agents)
        if isinstance(slots, int): # Handle edge case in which the user wants n sequential slots, as used in testing
            self.slots = np.arange(slots)
        else:
            self.slots = slots # E.g. sim.people.slots (instead of using uid as the slots directly)

        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.init_state = self.bitgen.state # Store the initial state
        
        # Initialize the distribution, if not a string
        if not isinstance(self.dist, str):
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



# TODO: define custom distributions like ss.lognormal


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












#%% Archive

class RNGs(sc.prettyobj):
    """
    Class for managing a collection MultiRNG random number generators
    """

    def __init__(self):
        self.rngs = ss.ndict()
        self.used_offsets = set()
        self.base_seed = None
        self.current_seed = None
        self.initialized = False
        return

    def initialize(self, base_seed):
        self.base_seed = base_seed
        self.initialized = True
        return
    
    def add(self, rng, check_repeats=True):
        """
        Add a random number generator
        
        Can request an offset, will check for overlap
        Otherwise, return value will be used as the seed offset for this rng
        """
        if not self.initialized:
            raise NotInitializedException()

        if rng.name in self.rngs:
            raise RepeatNameException(rng.name)

        if check_repeats:
            if rng.seed_offset in self.used_offsets:
                raise SeedRepeatException(rng.name, rng.seed_offset)
            self.used_offsets.add(rng.seed_offset)

        self.rngs.append(rng)
        self.current_seed = self.base_seed + rng.seed_offset # Add in the base seed

        return self.current_seed

    def step(self, ti):
        """ Step each RNG forward, for each sim timestep """
        for rng in self.rngs.values():
            rng.step(ti)
        return

    def reset(self):
        """ Reset each RNG """
        for rng in self.rngs.values():
            rng.reset()
        return

class RNG(np.random.Generator):
    """
    Class for tracking one random number generators associated with one decision per timestep.

    The main use case is to sample random numbers from various distributions
    that are specific to each agent (per decision and timestep) so as to enable
    variance reduction between simulations through the use of common random
    numbers. For example, the user might create a random number generator called
    rng and ultimately ask for randomly distributed random numbers for agents
    with UIDs 1 and 4:

    >>> import starsim as ss
    >>> import numpy as np
    >>> rng = ss.MultiRNG('Test') # The hashed name determines the seed offset.
    >>> rng.initialize(container=None, slots=5) # In practice, slots will be sim.people.slots. When scalar (for testing), an np.arange will be used.
    >>> uids = np.array([1,4])
    >>> rng.random(uids)
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
    
    def __init__(self, name, seed_offset=None, **kwargs):
        """
        Create a random number generator

        seed_offset will be automatically assigned (based on hashing the name) if None
        
        name: a name for this random number generator, like "coin_flip"
        """

        self.name = name
        self.kwargs = kwargs

        if seed_offset is None:
            # Obtain the seed offset by hashing the class name. Don't use python's hash because it is randomized.
            self.seed_offset = int(hashlib.sha256(self.name.encode('utf-8')).hexdigest(), 16) % 10**8
        else:
            # Use user-provided seed_offset (unlikely)
            self.seed_offset = seed_offset

        self.seed = None # Will be determined once added to the RNG Container
        self.initialized = False
        self.ready = True
        return

    def initialize(self, container, slots):
        if self.initialized:
            return

        if container is not None:
            self.seed = container.add(self) # base_seed + seed_offset
        else:
            # Enable use of MultiRNG without a container
            self.seed = self.seed_offset

        if isinstance(slots, int):
            # Handle edge case in which the user wants n sequential slots, as used in testing.
            self.slots = np.arange(slots)
        else:
            self.slots = slots # E.g. sim.people.slots (instead of using uid as the slots directly)

        if 'bit_generator' not in self.kwargs:
            self.kwargs['bit_generator'] = np.random.PCG64DXSM(seed=self.seed)
        super().__init__(**self.kwargs)

        self._init_state = self.bit_generator.state # Store the initial state

        self.initialized = True
        self.ready = True
        return

    def reset(self):
        """ Restore initial state """
        self.bit_generator.state = self._init_state
        self.ready = True
        return

    def step(self, ti):
        """ Advance to time ti step by jumping """
        
        # First reset back to the initial state
        self.reset()

        # Now take ti jumps
        # jumped returns a new bit_generator, use directly instead of setting state?
        self.bit_generator.state = self.bit_generator.jumped(jumps=ti).state
        return

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        #'bit_generator' kwarg has changed
        super(MultiRNG, result).__init__(**result.kwargs)

        return result
