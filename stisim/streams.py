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
        self.used_seeds = []
        self.initialized = False
        return

    def initialize(self, base_seed):
        self.base_seed = base_seed
        self.initialized = True
        return
    
    def add(self, stream):
        """
        Add a stream
        
        Can request an offset, will check for overlap
        Otherwise, return value will be used as the seed offset for this stream
        """
        if not self.initialized:
            raise NotInitializedException('Please call initialize before adding a stream to Streams.')

        if stream.name in self._streams:
            raise RepeatNameException(f'A Stream with name {stream.name} has already been added.')

        if stream.seed_offset is None:
            seed = int(hashlib.sha256(stream.name.encode('utf-8')).hexdigest(), 16) % 10**8 #abs(hash(stream.name))
        elif stream.seed_offset in self.used_seeds:
            raise SeedRepeatException(f'Requested seed offset {stream.seed_offset} for stream {stream} has already been used.')
        else:
            seed = stream.seed_offset
        self.used_seeds.append(seed)

        self._streams.append(stream)

        return self.base_seed + seed

    def step(self, ti):
        for stream in self._streams.dict_values():
            stream.step(ti)
        return

    def reset(self):
        for stream in self._streams.dict_values():
            stream.reset()
        return


class NotResetException(Exception):
    "Raised when stream is called when not ready."
    pass


class NotInitializedException(Exception):
    "Raised when stream is called when not initialized."
    pass


class RepeatNameException(Exception):
    "Raised when adding a stream to streams when the stream name has already been used."
    pass


class SeedRepeatException(Exception):
    "Raised when stream is called when not initialized."
    pass


class NotStreamSafeException(Exception):
    "Raised when an unsafe-for-streams function is called."
    pass


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

        if 'size' in kwargs and 'uids' in kwargs and not ((kwargs['size'] is None) ^ (kwargs['uids'] is None)):
            raise Exception('Specify either "uids" or "size", but not both.')

        # Check for zero length size
        if 'size' in kwargs.keys():
            # size-based
            size = kwargs.pop('size')

            if not isinstance(size, int):
                raise Exception('Input "size" must be an integer')

            if size < 0:
                raise Exception('Input "size" cannot be negative')

            if size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            basis = SIZE

        else:
            # uid-based
            uids = kwargs['uids']

            if len(uids) == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

            if isinstance(uids, ss.states.FusedArray):
                v = uids.values
            elif isinstance(uids, ss.states.DynamicView):
                v = uids._view
            else:
                v = uids

            if v.dtype == bool:
                size = len(uids)
                basis = BOOLS
            else:
                #size = self.slots.values[v.max()] + 1
                #size = self.slots.values[v].max() + 1
                size = self.slots[v].values.max() + 1
                basis = UIDS

        if not self.initialized:
            msg = f'Stream {self.name} has not been initialized!'
            raise NotInitializedException(msg)
        if not self.ready:
            msg = f'Stream {self.name} has already been sampled on this timestep!'
            raise NotResetException(msg)
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

        seed_offset will be automatically assigned (sequentially in first-come order) if None
        
        name: a name for this Stream, like "coin_flip"
        """

        '''
        if 'bit_generator' not in kwargs:
            kwargs['bit_generator'] = np.random.PCG64(seed=self.seed + self.seed_offset)
        super().__init__(bit_generator=np.random.PCG64())
        '''

        self.name = name
        self.seed_offset = seed_offset
        self.kwargs = kwargs
        
        self.seed = None

        self.initialized = False
        self.ready = True
        return

    def initialize(self, streams, slots):
        if self.initialized:
            # TODO: Raise warning
            assert not self.initialized
            return

        self.seed = streams.add(self)
        self.slots = slots # E.g. sim.people.slots (instead of using uid as the slots directly)

        if 'bit_generator' not in self.kwargs:
            self.kwargs['bit_generator'] = np.random.PCG64(seed=self.seed)
        super().__init__(**self.kwargs)

        #self.rng = np.random.default_rng(seed=self.seed + self.seed_offset)
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

    @_pre_draw
    def random(self, size, basis, uids=None):
        if basis==SIZE:
            return super(MultiStream, self).random(size=size)
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return super(MultiStream, self).random(size=size)[slots]
        elif basis == BOOLS:
            return super(MultiStream, self).random(size=size)[uids]
        else:
            raise Exception('TODO BASISEXCPETION')

    @_pre_draw
    def uniform(self, size, basis, low, high, uids=None):
        if basis == SIZE:
            return super(MultiStream, self).uniform(size=size, low=low, high=high)
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return super(MultiStream, self).uniform(size=size, low=low, high=high)[slots]
        elif basis == BOOLS:
            return super(MultiStream, self).uniform(size=size, low=low, high=high)[uids]
        else:
            raise Exception('TODO BASISEXCPETION')

    @_pre_draw
    def integers(self, size, basis, low, high, uids=None, **kwargs):
        if basis == SIZE:
            return super(MultiStream, self).integers(size=size, low=low, high=high, **kwargs)
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return super(MultiStream, self).integers(size=size, low=low, high=high, **kwargs)[slots]
        elif basis == BOOLS:
            return super(MultiStream, self).integers(size=size, low=low, high=high, **kwargs)[uids]
        else:
            raise Exception('TODO BASISEXCPETION')

    @_pre_draw
    def poisson(self, size, basis, lam, uids=None):
        if basis == SIZE:
            return super(MultiStream, self).poisson(size=size, lam=lam)
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return super(MultiStream, self).poisson(size=size, lam=lam)[slots]
        elif basis == BOOLS:
            return super(MultiStream, self).poisson(size=size, lam=lam)[uids]
        else:
            raise Exception('TODO BASISEXCPETION')

    @_pre_draw
    def normal(self, size, basis, mu=0, std=1, uids=None):
        if basis == SIZE:
            return mu + std*super(MultiStream, self).normal(size=size)
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return mu + std*super(MultiStream, self).normal(size=size)[slots]
        elif basis == BOOLS:
            return mu + std*super(MultiStream, self).normal(size=size)[uids]
        else:
            raise Exception('TODO BASISEXCPETION')

    @_pre_draw
    def negative_binomial(self, size, basis, n, p, uids=None): #n=nbn_n, p=nbn_p, size=n)
        if basis == SIZE:
            return super(MultiStream, self).negative_binomial(size=size, n=n, p=p)
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return super(MultiStream, self).negative_binomial(size=size, n=n, p=p)[slots]
        elif basis == BOOLS:
            return super(MultiStream, self).negative_binomial(size=size, n=n, p=p)[uids]
        else:
            raise Exception('TODO BASISEXCPETION')

    @_pre_draw
    def bernoulli(self, prob, size, basis, uids=None):
        #return super(MultiStream, self).choice([True, False], size=size.max()+1) # very slow
        #return (super(MultiStream, self).binomial(n=1, p=prob, size=size.max()+1))[size].astype(bool) # pretty fast
        if basis == SIZE:
            return super(MultiStream, self).random(size=size) < prob # fastest
        elif basis == UIDS:
            #slots = self.slots.values[uids]
            slots = self.slots[uids].values
            return super(MultiStream, self).random(size=size)[slots] < prob # fastest
        elif basis == BOOLS:
            return super(MultiStream, self).random(size=size)[uids] < prob # fastest
        else:
            raise Exception('TODO BASISEXCPETION')

    # @_pre_draw <-- handled by call to self.bernoullli
    def bernoulli_filter(self, uids, prob):
        return uids[self.bernoulli(uids=uids, prob=prob)]

    def choice(self, size, basis, a, **kwargs):
        # Consider raising a warning instead?
        raise NotStreamSafeException('The "choice" function is not MultiStream-safe.')


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

            size = len(uids)

        if not self.initialized:
            msg = f'Stream {self.name} has not been initialized!'
            raise NotInitializedException(msg)

        return func(self, size=size, **kwargs)

    return check_ready

class CentralizedStream(np.random.Generator):
    """
    Class to imitate the behavior of a centralized random number generator
    """

    def __init__(self, name, seed_offset=None, **kwargs):
        """
        Create a random number stream

        seed_offset will be automatically assigned (sequentially in first-come order) if None
        
        name: a name for this Stream, like "coin_flip"
        """
        super().__init__(bit_generator=np.random.PCG64())
        self.name = name
        self.initialized = False
        self.seed_offset = None # Not used, so override to avoid potential seed collisions in Streams.
        return

    def initialize(self, streams, slots=None):
        if self.initialized:
            # TODO: Raise warning
            assert not self.initialized
            return

        streams.add(self) # Seed is returned, but not used here
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
    def normal(self, size, mu=0, std=1):
        return mu + std*np.random.normal(size=size, loc=mu, scale=std)

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