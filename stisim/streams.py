import hashlib
import numpy as np
import stisim as ss

__all__ = ['Streams', 'MultiStream', 'CentralizedStream', 'Stream']


SIZE  = 0
UIDS  = 1
BOOLS = 2

class Streams:
    """
    Class for managing a collection random number streams
    """

    def __init__(self):
        self._streams = ss.ndict()
        self.used_seed_offsets = []
        self.initialized = False
        return

    def initialize(self, base_seed):
        self.base_seed = base_seed
        self.initialized = True
        return
    
    def add(self, stream, check_repeats=True):
        """
        Add a stream
        
        Can request an offset, will check for overlap
        Otherwise, return value will be used as the seed offset for this stream
        """
        if not self.initialized:
            raise Exception('Please call initialize before adding a stream to Streams.')

        if stream.name in self._streams:
            raise Exception(f'A Stream with name {stream.name} has already been added.')

        if check_repeats:
            if stream.seed_offset in self.used_seed_offsets:
                raise Exception(f'Requested seed offset {stream.seed_offset} for stream {stream} has already been used.')
            self.used_seed_offsets.append(stream.seed_offset)

        self._streams.append(stream)

        return self.base_seed + stream.seed_offset # Add in the base seed

    def step(self, ti):
        for stream in self._streams.dict_values():
            stream.step(ti)
        return

    def reset(self):
        for stream in self._streams.dict_values():
            stream.reset()
        return


def Stream(*args, **kwargs):
    """
    Class to choose a stream
    """
    if ss.options.multistream:
        return MultiStream(*args, **kwargs)
    
    return CentralizedStream(*args, **kwargs)


def _pre_draw(func):
    def check_ready(self, **kwargs):
        """ Validation before drawing """

        uids = None
        if 'uids' in kwargs:
            uids = kwargs['uids']

        size = None
        if 'size' in kwargs:
            size = kwargs.pop('size')

        if not ((size is None) ^ (uids is None)):
            raise Exception('Specify either "uids" or "size", but not both.')

        if size is not None:
            # Size-based
            if not isinstance(size, int):
                raise Exception('Input "size" must be an integer')

            if size < 0:
                raise Exception('Input "size" cannot be negative')

            if size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            basis = SIZE

        else:
            # UID-based
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
            msg = f'Stream {self.name} has not been initialized!'
            raise Exception(msg)
        if not self.ready:
            msg = f'Stream {self.name} has already been sampled on this timestep!'
            raise Exception(msg)
        self.ready = False

        return func(self, basis=basis, size=size, **kwargs)

    return check_ready


class MultiStream(np.random.Generator):
    """
    Class for tracking one random number stream associated with one decision per timestep
    """
    
    def __init__(self, name, seed_offset=None, **kwargs):
        """
        Create a random number stream

        seed_offset will be automatically assigned (based on hashing the name) if None
        
        name: a name for this Stream, like "coin_flip"
        """

        self.name = name
        self.kwargs = kwargs

        if seed_offset is None:
            # Obtain the seed offset by hashing the class name. Don't use python's hash because it is randomized.
            self.seed_offset = int(hashlib.sha256(self.name.encode('utf-8')).hexdigest(), 16) % 10**8
        else:
            # Use user-provided seed_offset (unlikely)
            self.seed_offset = seed_offset

        self.seed = None # Will be determined once added to Streams
        self.initialized = False
        self.ready = True
        return

    def initialize(self, streams, slots):
        if self.initialized:
            return

        self.seed = streams.add(self) # base_seed + seed_offset

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

    @_pre_draw
    def random(self, size, basis, uids=None):
        vals = super(MultiStream, self).random(size=size)
        return self._select(vals, basis, uids)

    @_pre_draw
    def uniform(self, size, basis, low, high, uids=None):
        vals = super(MultiStream, self).uniform(size=size, low=low, high=high)
        return self._select(vals, basis, uids)

    @_pre_draw
    def integers(self, size, basis, low, high, uids=None, **kwargs):
        vals = super(MultiStream, self).integers(size=size, low=low, high=high, **kwargs)
        return self._select(vals, basis, uids)

    @_pre_draw
    def poisson(self, size, basis, lam, uids=None):
        vals = super(MultiStream, self).poisson(size=size, lam=lam)
        return self._select(vals, basis, uids)

    @_pre_draw
    def normal(self, size, basis, mu=0, std=1, uids=None):
        vals = mu + std*super(MultiStream, self).normal(size=size)
        return self._select(vals, basis, uids)

    @_pre_draw
    def lognormal(self, size, basis, mean=0, sigma=1, uids=None):
        vals = super(MultiStream, self).lognormal(size=size, mean=mean, sigma=sigma)
        return self._select(vals, basis, uids)

    @_pre_draw
    def negative_binomial(self, size, basis, n, p, uids=None):
        vals = super(MultiStream, self).negative_binomial(size=size, n=n, p=p)
        return self._select(vals, basis, uids)

    @_pre_draw
    def bernoulli(self, prob, size, basis, uids=None):
        vals = super(MultiStream, self).random(size=size)
        return self._select(vals, basis, uids) < prob

    # @_pre_draw <-- handled by call to self.bernoullli
    def bernoulli_filter(self, uids, prob):
        return uids[self.bernoulli(uids=uids, prob=prob)]

    def choice(self, size, basis, a, **kwargs):
        # Consider raising a warning instead?
        raise Exception('The "choice" function is not MultiStream-safe.')


def _pre_draw_centralized(func):
    def check_ready(self, **kwargs):
        """ Validation before drawing """

        uids = None
        if 'uids' in kwargs:
            uids = kwargs.pop('uids')

        size = None
        if 'size' in kwargs:
            size = kwargs.pop('size')

        if not ((size is None) ^ (uids is None)):
            raise Exception('Specify either "uids" or "size", but not both.')

        # Check for zero length size
        if size is not None:
            # size-based
            if not isinstance(size, int):
                raise Exception('Input "size" must be an integer')

            if size < 0:
                raise Exception('Input "size" cannot be negative')

            if size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

        else:
            # uid-based
            if len(uids) == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            if uids.dtype == bool:
                size = uids.sum()
            else:
                size = len(uids)

        if not self.initialized:
            msg = f'Stream {self.name} has not been initialized!'
            raise Exception(msg)

        return func(self, size=size, **kwargs)

    return check_ready


class CentralizedStream():
    """
    Class to imitate the behavior of a centralized random number generator
    """

    def __init__(self, name, seed_offset=None, **kwargs):
        """
        Create a random number stream

        seed_offset will be automatically assigned (sequentially in first-come order) if None
        
        name: a name for this Stream, like "coin_flip"
        """
        self.name = name
        self.initialized = False
        self.seed_offset = 0 # For compatibility with MultiStream
        return

    def initialize(self, streams, slots=None):
        if self.initialized:
            return

        streams.add(self, check_repeats=False) # Seed is returned, but not used here as we're using the global np.random stream which has been seeded elsewhere
        self.initialized = True
        return

    def reset(self):
        pass

    def step(self, ti):
        pass

    @_pre_draw_centralized
    def random(self, size, **kwargs):
        return np.random.random(size=size, **kwargs)

    @_pre_draw_centralized
    def uniform(self, size, low, high, **kwargs):
        return np.random.uniform(size=size, low=low, high=high, **kwargs)

    @_pre_draw_centralized
    def integers(self, size, low, high, **kwargs):
        return np.random.random_integers(size=size, low=low, high=high)

    @_pre_draw_centralized
    def poisson(self, size, lam, **kwargs):
        return np.random.poisson(lam=lam, size=size, **kwargs)

    @_pre_draw_centralized
    def normal(self, size, mu=0, std=1, **kwargs):
        return mu + std*np.random.normal(size=size, loc=mu, scale=std)

    @_pre_draw_centralized
    def lognormal(self, size, mean=0, sigma=1, **kwargs):
        return np.random.lognormal(size=size, mean=mean, sigma=sigma)

    @_pre_draw_centralized
    def negative_binomial(self, size, n, p, **kwargs):
        return np.random.negative_binomial(size=size, n=n, p=p, **kwargs)

    @_pre_draw_centralized
    def bernoulli(self, prob, size, **kwargs):
        return np.random.random(size=size, **kwargs) < prob

    def bernoulli_filter(self, uids, prob):
        return uids[self.bernoulli(uids=uids, prob=prob)]

    @_pre_draw_centralized
    def choice(self, size, a, **kwargs):
        return np.random.choice(a, size=size, **kwargs)