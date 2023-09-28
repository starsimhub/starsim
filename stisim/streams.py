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


def Stream(multistream=True):
    """
    Class to choose a stream
    """
    if multistream:
        return MultiStream
    
    return CentralizedStream


def _pre_draw(func):
    def check_ready(self, *args, **kwargs):
        """ Validation before drawing """

        # Check for zero length arr
        if 'arr' in kwargs.keys():
            arr = kwargs['arr']
        else:
            arr = args[0]
        if len(arr) == 0:
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
        uid: an identifier added to the name to make it uniquely identifiable, for example the name or id of the calling class
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
    def poisson(self, arr, lam):
        return super(MultiStream, self).poisson(lam=lam, size=self.draw_size(arr))[arr]

    @_pre_draw
    def normal(self, arr, mu=0, std=1):
        return mu + std*super(MultiStream, self).normal(size=self.draw_size(arr))[arr]

    @_pre_draw
    def bernoulli(self, arr, prob):
        #return super(MultiStream, self).choice([True, False], size=arr.max()+1) # very slow
        #return (super(MultiStream, self).binomial(n=1, p=prob, size=arr.max()+1))[arr].astype(bool) # pretty fast
        return super(MultiStream, self).random(self.draw_size(arr))[arr] < prob # fastest

    # @_pre_draw <-- handled by call to self.bernoullli
    def bernoulli_filter(self, arr, prob):
        return arr[self.bernoulli(arr, prob)] # Slightly faster on my machine for bernoulli to typecast

    def sample(self, dist=None, par1=None, par2=None, size=None, **kwargs):
        """
        Draw a sample from the distribution specified by the input. The available
        distributions are:

        - 'uniform'       : uniform from low=par1 to high=par2; mean is equal to (par1+par2)/2
        - 'choice'        : par1=array of choices, par2=probability of each choice
        - 'normal'        : normal with mean=par1 and std=par2
        - 'lognormal'     : lognormal with mean=par1, std=par2 (parameters are for the lognormal, not the underlying normal)
        - 'normal_pos'    : right-sided normal (i.e. only +ve values), with mean=par1, std=par2 of the underlying normal
        - 'normal_int'    : normal distribution with mean=par1 and std=par2, returns only integer values
        - 'lognormal_int' : lognormal distribution with mean=par1 and std=par2, returns only integer values
        - 'poisson'       : Poisson distribution with rate=par1 (par2 is not used); mean and variance are equal to par1
        - 'neg_binomial'  : negative binomial distribution with mean=par1 and k=par2; converges to Poisson with k=∞
        - 'beta'          : beta distribution with alpha=par1 and beta=par2;
        - 'gamma'         : gamma distribution with shape=par1 and scale=par2;

        Args:
            self (Stream) : the random number generator stream
            dist (str)    : the distribution to sample from
            par1 (float)  : the "main" distribution parameter (e.g. mean)
            par2 (float)  : the "secondary" distribution parameter (e.g. std)
            size (int)    : the number of samples (default=1)
            kwargs (dict) : passed to individual sampling functions

        Returns:
            A length N array of samples

        **Examples**::

            ss.sample() # returns Unif(0,1)
            ss.sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)
            ss.sample(dist='lognormal_int', par1=5, par2=3) # returns lognormally distributed values with mean 5 and std 3

        Notes:
            Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
            https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
            function assumes the user wants to specify the mean and std of the lognormal distribution.

            Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
            (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
            distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
            of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
            large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
            the mean.
        """

        # Some of these have aliases, but these are the "official" names
        choices = [
            'uniform',
            'normal',
            'choice',
            'normal_pos',
            'normal_int',
            'lognormal',
            'lognormal_int',
            'poisson',
            'neg_binomial',
            'beta',
            'gamma',
        ]

        # Ensure it's an integer
        if size is not None and not isinstance(size, tuple):
            size = int(size)

        # Compute distribution parameters and draw samples
        # NB, if adding a new distribution, also add to choices above
        if dist in ['unif', 'uniform']:
            samples = self.uniform(low=par1, high=par2, size=size)
        elif dist in ['choice']:
            samples = self.choice(a=par1, p=par2, size=size, **kwargs)
        elif dist in ['norm', 'normal']:
            samples = self.normal(loc=par1, scale=par2, size=size)
        elif dist == 'normal_pos':
            samples = np.abs(self.normal(loc=par1, scale=par2, size=size))
        elif dist == 'normal_int':
            samples = np.round(np.abs(self.normal(loc=par1, scale=par2, size=size)))
        elif dist == 'poisson':
            samples = self.poisson(rate=par1, n=size)  # Use Numba version below for speed
        elif dist == 'beta':
            samples = self.beta(a=par1, b=par2, size=size)
        elif dist == 'gamma':
            samples = self.gamma(shape=par1, scale=par2, size=size)
        elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
            if (sc.isnumber(par1) and par1 > 0) or (sc.checktype(par1, 'arraylike') and (par1 > 0).all()):
                mean = np.log(
                    par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
                sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution
                samples = self.lognormal(mean=mean, sigma=sigma, size=size)
            else:
                samples = np.zeros(size)
            if '_int' in dist:
                samples = np.round(samples)
        # Calculate a and b using mean (par1) and variance (par2)
        # https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
        elif dist == 'beta_mean':
            a = ((1 - par1) / par2 - 1 / par1) * par1 ** 2
            b = a * (1 / par1 - 1)
            samples = self.beta(a=a, b=b, size=size)
        else:
            errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
            raise NotImplementedError(errormsg)

        return samples


class CentralizedStream(np.random.Generator):
    """
    Class to imitate the behavior of a centralized random number generator
    """
    
    def __init__(self, name, seed_offset=None, **kwargs):
        """
        Create a random number stream

        seed_offset will be automatically assigned (sequentially in first-come order) if None
        
        name: a name for this Stream, like "coin_flip"
        uid: an identifier added to the name to make it uniquely identifiable, for example the name or id of the calling class
        """
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

        if isinstance(arr, ss.states.FusedArray):
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

    def poisson(self, arr, lam):
        return np.random.poisson(lam=lam, size=self.draw_size(arr))

    def normal(self, arr, mu=0, std=1):
        return mu + std*np.random.normal(size=self.draw_size(arr), loc=mu, scale=std)

    def bernoulli(self, arr, prob):
        return np.random.random(self.draw_size(arr)) < prob

    def bernoulli_filter(self, arr, prob):
        return arr[self.bernoulli(arr, prob)] # Slightly faster on my machine for bernoulli to typecast

    def sample(self, dist=None, par1=None, par2=None, size=None, **kwargs):
        """
        Draw a sample from the distribution specified by the input. The available
        distributions are:

        - 'uniform'       : uniform from low=par1 to high=par2; mean is equal to (par1+par2)/2
        - 'choice'        : par1=array of choices, par2=probability of each choice
        - 'normal'        : normal with mean=par1 and std=par2
        - 'lognormal'     : lognormal with mean=par1, std=par2 (parameters are for the lognormal, not the underlying normal)
        - 'normal_pos'    : right-sided normal (i.e. only +ve values), with mean=par1, std=par2 of the underlying normal
        - 'normal_int'    : normal distribution with mean=par1 and std=par2, returns only integer values
        - 'lognormal_int' : lognormal distribution with mean=par1 and std=par2, returns only integer values
        - 'poisson'       : Poisson distribution with rate=par1 (par2 is not used); mean and variance are equal to par1
        - 'neg_binomial'  : negative binomial distribution with mean=par1 and k=par2; converges to Poisson with k=∞
        - 'beta'          : beta distribution with alpha=par1 and beta=par2;
        - 'gamma'         : gamma distribution with shape=par1 and scale=par2;

        Args:
            self (Stream) : the random number generator stream
            dist (str)    : the distribution to sample from
            par1 (float)  : the "main" distribution parameter (e.g. mean)
            par2 (float)  : the "secondary" distribution parameter (e.g. std)
            size (int)    : the number of samples (default=1)
            kwargs (dict) : passed to individual sampling functions

        Returns:
            A length N array of samples

        **Examples**::

            ss.sample() # returns Unif(0,1)
            ss.sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)
            ss.sample(dist='lognormal_int', par1=5, par2=3) # returns lognormally distributed values with mean 5 and std 3

        Notes:
            Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
            https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
            function assumes the user wants to specify the mean and std of the lognormal distribution.

            Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
            (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
            distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
            of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
            large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
            the mean.
        """

        # Some of these have aliases, but these are the "official" names
        choices = [
            'uniform',
            'normal',
            'choice',
            'normal_pos',
            'normal_int',
            'lognormal',
            'lognormal_int',
            'poisson',
            'neg_binomial',
            'beta',
            'gamma',
        ]

        # Ensure it's an integer
        if size is not None and not isinstance(size, tuple):
            size = int(size)

        # Compute distribution parameters and draw samples
        # NB, if adding a new distribution, also add to choices above
        if dist in ['unif', 'uniform']:
            samples = np.random.uniform(low=par1, high=par2, size=size)
        elif dist in ['choice']:
            samples = np.random.choice(a=par1, p=par2, size=size, **kwargs)
        elif dist in ['norm', 'normal']:
            samples = np.random.normal(loc=par1, scale=par2, size=size)
        elif dist == 'normal_pos':
            samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size))
        elif dist == 'normal_int':
            samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size)))
        elif dist == 'poisson':
            samples = np.random.poisson(rate=par1, n=size)  # Use Numba version below for speed
        elif dist == 'beta':
            samples = np.random.beta(a=par1, b=par2, size=size)
        elif dist == 'gamma':
            samples = np.random.gamma(shape=par1, scale=par2, size=size)
        elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
            if (sc.isnumber(par1) and par1 > 0) or (sc.checktype(par1, 'arraylike') and (par1 > 0).all()):
                mean = np.log(
                    par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
                sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution
                samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
            else:
                samples = np.zeros(size)
            if '_int' in dist:
                samples = np.round(samples)
        # Calculate a and b using mean (par1) and variance (par2)
        # https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
        elif dist == 'beta_mean':
            a = ((1 - par1) / par2 - 1 / par1) * par1 ** 2
            b = a * (1 / par1 - 1)
            samples = np.random.beta(a=a, b=b, size=size)
        else:
            errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
            raise NotImplementedError(errormsg)

        return samples