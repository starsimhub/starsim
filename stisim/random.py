import hashlib
import numpy as np
import stisim as ss

__all__ = ['RNGContainer', 'MultiRNG', 'SingleRNG', 'RNG']


class RNGContainer:
    """
    Class for managing a collection random number generators (MultiRNG or SingleRNG instances)
    """

    def __init__(self):
        self._rngs = ss.ndict()
        self.used_seed_offsets = set()
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

        if rng.name in self._rngs:
            raise RepeatNameException(rng.name)

        if check_repeats:
            if rng.seed_offset in self.used_seed_offsets:
                raise SeedRepeatException(rng.name, rng.seed_offset)
            self.used_seed_offsets.add(rng.seed_offset)

        self._rngs.append(rng)

        return self.base_seed + rng.seed_offset # Add in the base seed

    def step(self, ti):
        for rng in self._rngs.dict_values():
            rng.step(ti)
        return

    def reset(self):
        for rng in self._rngs.dict_values():
            rng.reset()
        return


class NotResetException(Exception):
    "Raised when an object is called twice in one timestep."
    def __init__(self, rng_name):
        msg = f'The random number generator with name {rng_name} has already been sampled on this timestep!'
        super().__init__(msg)
        return


class NotInitializedException(Exception):
    "Raised when a random number generator or a RNGContainer object is called when not initialized."
    def __init__(self, obj_name=None):
        if obj_name is None: 
            msg = f'An object is being used without proper initialization.'
        else:
            msg = f'The object named {obj_name} is being used prior to initialization.'
        super().__init__(msg)
        return


class SeedRepeatException(Exception):
    "Raised when two random number generators have the same seed."
    def __init__(self, rng_name, seed_offset):
        msg = f'Requested seed offset {seed_offset} for the random number generator named {rng_name} has already been used.'
        super().__init__(msg)
        return


class RepeatNameException(Exception):
    "Raised when adding a random number generator to a RNGContainer when the rng name has already been used."
    def __init__(self, rng_name):
        msg = f'A random number generator with name {rng_name} has already been added.'
        super().__init__(msg)
        return


def RNG(*args, **kwargs):
    """
    Class to choose a random number generator class.
    
    Parameters:
    set_for (Distribution): Connect this RNG instance to the provided Distribution [Optional]

    Returns:
    SingleRNG or MultiRNG: Instance of a random number generator
    """
    if ss.options.multirng:
        rng = MultiRNG(*args, **kwargs)
    else:
        rng = SingleRNG(*args, **kwargs)
    return rng


class MultiRNG(np.random.Generator):
    """
    Class for tracking one random number generators associated with one decision per timestep.

    The main use case is to sample random numbers from various distributions
    that are specific to each agent (per decision and timestep) so as to enable
    variance reduction between simulations through the use of common random
    numbers. For example, the user might create a random number generator called
    rng and ultimately ask for randomly distributed random numbers for agents
    with UIDs 1 and 4:

    >>> import stisim as ss
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

    def sample(self, distribution, size=None, **kwargs):
        """
        Sample from a ss.Distribution

        :param size: can be
            - Integer (returns fixed number of samples)
            - Boolean array (returns samples for True values)
            - Array of UIDs (returns samples for each UID using slots)    <---- If sampling per-agent, this is likely the desired option
        :return:
        """

        if not self.ready:
            raise NotResetException(self.name)

        # Work out how many samples to draw. If sampling by UID, this depends on the number of slots
        if np.isscalar(size):
            if size < 0:
                raise Exception('Input "size" cannot be negative')
            elif size == 0:
                return np.array([], dtype=int)
            else:
                n_samples = size
        elif len(size) == 0:
            return np.array([], dtype=int)  # int dtype allows use as index, e.g. bernoulli_filter
        elif size.dtype == bool:
            n_samples = len(size)
        elif size.dtype == int:
            v = size.__array__() # TODO - check if this works without calling __array__()?
            max_slot = self.slots[v].__array__().max()
            if max_slot == ss.INT_NAN:
                raise Exception('Attempted to sample from an INT_NAN slot')
            n_samples = max_slot + 1
        else:
            raise Exception("Unrecognized input type")

        if isinstance(distribution, ss.choice):
            raise Exception('The "choice" function is not MultiRNG-safe.')

        vals = distribution.sample(n_samples, rng=self, **kwargs)
        self.ready = False

        if np.isscalar(size):
            return vals
        elif size.dtype == bool:
            return vals[size]
        else:
            slots = self.slots[size].__array__()
            return vals[slots]

    def random(self, size=1):
        return self.sample(ss.uniform(), size=size)

    def bernoulli_filter(self, p, uids):
        """
        Bernoulli filtering of UIDs in an RNG-safe way

        :param p: Probability of success. Supported types are
            - A float in the range [0,1]
            - An array the same length as UIDs
            - A distribution instance, where the probability of success is given by the mean of the distribution
        :param uids: Array of UIDs for the agents to filter
        :return: An array of UIDs that 'succeeded'
        """

        vals = self.random(size=uids) # Draw RNG-safe samples

        if isinstance(p, ss.Distribution):
            return uids[vals < p.mean()] # Use the mean of the distribution
        else:
            return uids[vals < p]


class SingleRNG():
    """
    Class to imitate the behavior of a centralized random number generator
    """

    def __init__(self, name, seed_offset=None, **kwargs):
        """
        Create a random number generator

        seed_offset will be automatically assigned (sequentially in first-come order) if None
        
        name: a name for this random number generator, like "coin_flip"
        """
        self.name = name
        self.initialized = False
        self.seed_offset = 0 # For compatibility with MultiRNG
        self.seed = None
        self.ready = False
        return

    def initialize(self, container, slots=None):
        """
        Slots are not used by the SingleRNG, but here for compatibility with the MultiRNG
        """
        if self.initialized:
            return

        if container is not None:
            container.add(self, check_repeats=False) # Seed is returned, but not used here as we're using the global np.random generator which has been seeded elsewhere

        self.initialized = True
        self.reset()
        return

    def reset(self):
        self.ready = True


    def step(self, ti):
        self.reset()


    def sample(self, distribution, size, **kwargs):
        """
        Sample from a ss.Distribution
        """

        if not self.ready:
            raise NotResetException(self.name)
        self.ready = False

        # Check for zero length size
        if isinstance(size, int):
            # size-based
            if not isinstance(size, int):
                raise Exception('Input "size" must be an integer')

            if size < 0:
                raise Exception('Input "size" cannot be negative')

            if size == 0:
                return np.array([], dtype=int)  # int dtype allows use as index, e.g. bernoulli_filter

        else:
            # UID-based (size should be an array)
            uids = size

            if len(uids) == 0:
                return np.array([], dtype=int)  # int dtype allows use as index, e.g. bernoulli_filter

            if uids.dtype == bool:
                size = uids.sum()
            else:
                size = len(uids)

        vals = distribution.sample(size, rng=None)
        self.ready = False
        return vals

    def random(self, size=1):
        return self.sample(ss.uniform(), size=size)

    def bernoulli_filter(self, p, uids):

        vals = ss.uniform().sample(len(uids))

        if isinstance(p, ss.Distribution):
            return uids[vals < p.mean()] # Use the mean of the distribution
        else:
            return uids[vals < p]