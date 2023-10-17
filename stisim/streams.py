import hashlib
import numpy as np
import stisim as ss

__all__ = ['Streams', 'MultiStream', 'CentralizedStream', 'Stream']


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
    def check_ready(self, *args, **kwargs):
        """ Validation before drawing """

        # Check for zero length size
        if 'size' in kwargs.keys():
            size = kwargs['size']
        else:
            size = args[0]
        if isinstance(size, int):
            # If an integer, the user wants "n" samples
            if size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter
            else:
                kwargs['size'] = np.arange(size)
        elif len(size) == 0:
            return np.array([], dtype=int) # int dtype allows use as index, e.g. bernoulli_filter

        if not self.initialized:
            msg = f'Stream {self.name} has not been initialized!'
            raise NotInitializedException(msg)
        if not self.ready:
            msg = f'Stream {self.name} has already been sampled on this timestep!'
            raise NotResetException(msg)
        self.ready = False
        return func(self, *args, **kwargs)
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

    def draw_size(self, size):
        """ Determine how many random numbers to draw for a given size """

        if isinstance(size, ss.states.FusedArray):
            v = size.values
        elif isinstance(size, ss.states.DynamicView):
            v = size._view
        else:
            v = size

        if v.dtype == bool:
            return len(size)

        return v.max()+1

    @_pre_draw
    def random(self, size):
        slots = self.slots.values[size]
        return super(MultiStream, self).random(size=self.draw_size(slots))[slots]

    @_pre_draw
    def uniform(self, size, **kwargs):
        slots = self.slots.values[size]
        return super(MultiStream, self).uniform(size=self.draw_size(slots), **kwargs)[slots]

    @_pre_draw
    def poisson(self, size, lam):
        slots = self.slots.values[size]
        return super(MultiStream, self).poisson(size=self.draw_size(slots), lam=lam)[slots]

    @_pre_draw
    def normal(self, size, mu=0, std=1):
        slots = self.slots.values[size]
        return mu + std*super(MultiStream, self).normal(size=self.draw_size(slots))[slots]

    @_pre_draw
    def negative_binomial(self, size, **kwargs): #n=nbn_n, p=nbn_p, size=n)
        slots = self.slots.values[size]
        return super(MultiStream, self).negative_binomial(size=self.draw_size(slots), **kwargs)[slots]

    @_pre_draw
    def bernoulli(self, size, prob):
        #return super(MultiStream, self).choice([True, False], size=size.max()+1) # very slow
        #return (super(MultiStream, self).binomial(n=1, p=prob, size=size.max()+1))[size].astype(bool) # pretty fast
        slots = self.slots.values[size]
        return super(MultiStream, self).random(size=self.draw_size(slots))[slots] < prob # fastest

    # @_pre_draw <-- handled by call to self.bernoullli
    def bernoulli_filter(self, size, prob):
        #slots = self.slots[size[:]]
        return size[self.bernoulli(size, prob)] # Slightly faster on my machine for bernoulli to typecast

    def choice(self, size, a, **kwargs):
        # Consider raising a warning instead?
        raise NotStreamSafeException('The "choice" function is not MultiStream-safe.')


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

    def initialize(self, streams):
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

    def draw_size(self, size):
        """ Determine how many random numbers to draw for a given size """

        if isinstance(size, int):
            return size
        elif isinstance(size, ss.states.FusedArray):
            v = size.values
        elif isinstance(size, ss.states.DynamicView):
            v = size._view
        else:
            v = size

        if v.dtype == bool:
            return size.sum()

        return len(size)

    def random(self, size):
        return np.random.random(self.draw_size(size))

    def uniform(self, size, **kwargs):
        return np.random.uniform(size=self.draw_size(size), **kwargs)

    def poisson(self, size, lam):
        return np.random.poisson(lam=lam, size=self.draw_size(size))

    def normal(self, size, mu=0, std=1):
        return mu + std*np.random.normal(size=self.draw_size(size), loc=mu, scale=std)

    def negative_binomial(self, size, **kwargs): #n=nbn_n, p=nbn_p, size=n)
        return np.random.negative_binomial(**kwargs, size=self.draw_size(size))

    def bernoulli(self, size, prob):
        return np.random.random(self.draw_size(size)) < prob

    def bernoulli_filter(self, size, prob):
        return size[self.bernoulli(size, prob)] # Slightly faster on my machine for bernoulli to typecast

    def choice(self, size, a, **kwargs):
        return np.random.choice(a, size=self.draw_size(size), **kwargs)