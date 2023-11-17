import hashlib
import numpy as np
import stisim as ss

__all__ = ['RNGContainer', 'MultiRNG', 'SingleRNG', 'RNG']


SIZE  = 0
UIDS  = 1
BOOLS = 2

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
    Class to choose a random number generator class
    """
    if ss.options.multirng:
        return MultiRNG(*args, **kwargs)
    
    return SingleRNG(*args, **kwargs)


def _pre_draw_multi(func):
    """
    Decorator function that does quite a bit to empower calls to the sampling distributions in the MultiRNG class.

    The size parameter, from either kwargs or args[0], is used to determine if the user is seeking a fixed number of samples (if size is an integer), or instead if the user is providing an array.
    If size is an array, it could contain UIDs or be of boolean type. If UIDS, assume the user wants random numbers for these specific agents. If boolean, select random numbers based on the provided array. N.b. the approach based on UIDs is likely to be more "common random number safe".
    """
    def check_ready(self, *args, **kwargs):
        """ Validation before drawing """

        if 'size' in kwargs:
            size = kwargs.pop('size')
        elif len(args) == 1:
            size = args[0]
        elif len(args) > 1:
            raise Exception('Only one argument is allowed, please use key=value pairs for inputs other than size.')
        else:
            raise Exception('Could not assess the size of the random draw.')

        if isinstance(size, int):
            # Size-based
            if size < 0:
                raise Exception('Input "size" cannot be negative')

            if size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            basis = SIZE

        else:
            # UID-based (size should be an array)
            uids = size
            kwargs['uids'] = uids

            if len(uids) == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            v = uids.__array__()
            if v.dtype == bool:
                size = len(uids)
                basis = BOOLS
            else:
                size = self.slots[v].__array__().max() + 1
                basis = UIDS

        if not self.initialized:
            raise NotInitializedException(self.name)
        if not self.ready:
            raise NotResetException(self.name)
        self.ready = False

        return func(self, size=size, basis=basis, **kwargs)

    return check_ready


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
            self.kwargs['bit_generator'] = np.random.PCG64(seed=self.seed)
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

    def _select(self, vals, basis, uids):
        """ Select from the values given the basis and uids """
        if basis==SIZE:
            return vals
        elif basis == UIDS:
            slots = self.slots[uids].__array__()
            return vals[slots]
        elif basis == BOOLS:
            return vals[uids]
        else:
            raise Exception(f'Invalid basis: {basis}. Valid choices are [{SIZE}, {UIDS}, {BOOLS}]')

    @_pre_draw_multi
    def random(self, size, basis, uids=None):
        vals = super(MultiRNG, self).random(size=size)
        return self._select(vals, basis, uids)

    @_pre_draw_multi
    def uniform(self, size, basis, low, high, uids=None):
        vals = super(MultiRNG, self).uniform(size=size, low=low, high=high)
        return self._select(vals, basis, uids)

    @_pre_draw_multi
    def integers(self, size, basis, low, high, uids=None, **kwargs):
        vals = super(MultiRNG, self).integers(size=size, low=low, high=high, **kwargs)
        return self._select(vals, basis, uids)

    @_pre_draw_multi
    def poisson(self, size, basis, lam, uids=None):
        vals = super(MultiRNG, self).poisson(size=size, lam=lam)
        return self._select(vals, basis, uids)

    @_pre_draw_multi
    def normal(self, size, basis, loc=0, scale=1, uids=None):
        vals = super(MultiRNG, self).normal(size=size)
        return loc + scale*self._select(vals, basis, uids)

    @_pre_draw_multi
    def lognormal(self, size, basis, mean=0, sigma=1, uids=None):
        vals = super(MultiRNG, self).lognormal(size=size, mean=mean, sigma=sigma)
        return self._select(vals, basis, uids)

    @_pre_draw_multi
    def negative_binomial(self, size, basis, n, p, uids=None):
        vals = super(MultiRNG, self).negative_binomial(size=size, n=n, p=p)
        return self._select(vals, basis, uids)

    @_pre_draw_multi
    def bernoulli(self, size, prob, basis, uids=None):
        vals = super(MultiRNG, self).random(size=size)
        return self._select(vals, basis, uids) < prob

    # @_pre_draw_multi <-- handled by call to self.bernoullli
    def bernoulli_filter(self, uids, prob):
        return uids[self.bernoulli(size=uids, prob=prob)]

    def choice(self, size, basis, a, **kwargs):
        # Consider raising a warning instead?
        raise Exception('The "choice" function is not MultiRNG-safe.')


def _pre_draw_single(func):
    """
    Decorator for SingleRNG
    """
    def check_ready(*args, **kwargs):
        """ Validation before drawing """
        if 'size' in kwargs:
            size = kwargs.pop('size')
        elif len(args) == 1:
            size = args[0]
        elif len(args) > 1:
            raise Exception('Only one argument is allowed, please use key=value pairs for inputs other than size.')
        else:
            raise Exception('Could not assess the size of the random draw.')

        # Check for zero length size
        if isinstance(size, int):
            # size-based
            if not isinstance(size, int):
                raise Exception('Input "size" must be an integer')

            if size < 0:
                raise Exception('Input "size" cannot be negative')

            if size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

        else:
            # UID-based (size should be an array)
            uids = size

            if len(uids) == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            if uids.dtype == bool:
                size = uids.sum()
            else:
                size = len(uids)

        return func(size=size, **kwargs)

    return check_ready


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
        return

    def reset(self):
        pass

    def step(self, ti):
        pass

    def __getattr__(self, attr):
        # Returns wrapped numpy.random.(attr) if not a property
        try:
            return self.__getattribute__(attr)
        except Exception:
            try:
                numpy_func = getattr(np.random, attr)
                return _pre_draw_single(numpy_func)
            except Exception:
                errormsg = f'"{attr}" is not a member of this class or numpy.random'
                raise Exception(errormsg)

    @staticmethod
    @_pre_draw_single
    def integers(size, low, high, **kwargs):
        # provide integers via random_integers
        return np.random.random_integers(size=size, low=low, high=high)

    @staticmethod
    @_pre_draw_single
    def bernoulli(size, prob, **kwargs):
        return np.random.random(size=size, **kwargs) < prob

    @staticmethod
    def bernoulli_filter(uids, prob):
        return uids[SingleRNG.bernoulli(size=uids, prob=prob)]