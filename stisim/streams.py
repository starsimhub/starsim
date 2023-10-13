import numpy as np
import sciris as sc
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
            seed = abs(hash(stream.name))
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

        # Check for zero length arr
        if 'arr' in kwargs.keys():
            arr = kwargs['arr']
        else:
            arr = args[0]
        if isinstance(arr, int):
            # If an integer, the user wants "n" samples
            kwargs['arr'] = np.arange(arr)
        elif len(arr) == 0:
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

    def initialize(self, streams):
        if self.initialized:
            # TODO: Raise warning
            assert not self.initialized
            return

        self.seed = streams.add(self)

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

    def draw_size(self, arr):
        """ Determine how many random numbers to draw for a given arr """

        if isinstance(arr, ss.states.FusedArray):
            v = arr.values
        elif isinstance(arr, ss.states.DynamicView):
            v = arr._view
        else:
            v = arr

        if v.dtype == bool:
            return len(arr)

        return v.max()+1

    @_pre_draw
    def random(self, arr):
        return super(MultiStream, self).random(self.draw_size(arr))[arr]

    @_pre_draw
    def uniform(self, arr, **kwargs):
        return super(MultiStream, self).uniform(self.draw_size(arr), **kwargs)[arr]

    @_pre_draw
    def poisson(self, arr, lam):
        return super(MultiStream, self).poisson(lam=lam, size=self.draw_size(arr))[arr]

    @_pre_draw
    def normal(self, arr, mu=0, std=1):
        return mu + std*super(MultiStream, self).normal(size=self.draw_size(arr))[arr]

    @_pre_draw
    def negative_binomial(self, arr, **kwargs): #n=nbn_n, p=nbn_p, size=n)
        return super(MultiStream, self).negative_binomial(**kwargs, size=self.draw_size(arr))[arr]

    @_pre_draw
    def bernoulli(self, arr, prob):
        #return super(MultiStream, self).choice([True, False], size=arr.max()+1) # very slow
        #return (super(MultiStream, self).binomial(n=1, p=prob, size=arr.max()+1))[arr].astype(bool) # pretty fast
        return super(MultiStream, self).random(self.draw_size(arr))[arr] < prob # fastest

    # @_pre_draw <-- handled by call to self.bernoullli
    def bernoulli_filter(self, arr, prob):
        return arr[self.bernoulli(arr, prob)] # Slightly faster on my machine for bernoulli to typecast

    def choice(self, arr, prob):
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

    def draw_size(self, arr):
        """ Determine how many random numbers to draw for a given arr """

        if isinstance(arr, int):
            return arr
        elif isinstance(arr, ss.states.FusedArray):
            v = arr.values
        elif isinstance(arr, ss.states.DynamicView):
            v = arr._view
        else:
            v = arr

        if v.dtype == bool:
            return arr.sum()

        return len(arr)

    def random(self, arr):
        return np.random.random(self.draw_size(arr))

    def uniform(self, arr, **kwargs):
        return np.random.uniform(self.draw_size(arr), **kwargs)

    def poisson(self, arr, lam):
        return np.random.poisson(lam=lam, size=self.draw_size(arr))

    def normal(self, arr, mu=0, std=1):
        return mu + std*np.random.normal(size=self.draw_size(arr), loc=mu, scale=std)

    def negative_binomial(self, arr, **kwargs): #n=nbn_n, p=nbn_p, size=n)
        return np.random.negative_binomial(**kwargs, size=self.draw_size(arr))

    def bernoulli(self, arr, prob):
        return np.random.random(self.draw_size(arr)) < prob

    def bernoulli_filter(self, arr, prob):
        return arr[self.bernoulli(arr, prob)] # Slightly faster on my machine for bernoulli to typecast

    def choice(self, arr, **kwargs):
        return self.stream.choice(size=self.draw_size(arr), **kwargs)