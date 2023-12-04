""" 
stisim distributions

Example usage

>>> dist = stisim.normal(1,1) # Make a distribution
>>> dist()  # Draw a sample
>>> dist(10) # Draw several samples
>>> dist.sample(10) # Same as above
>>> stisim.State('foo', float, fill_value=dist)  # Use distribution as the fill value for a state
>>> disease.pars['immunity'] = dist  # Store the distribution as a parameter
>>> disease.pars['immunity'].sample(5)  # Draw some samples from the parameter
>>> stisim.poisson(rate=1).sample(n=10)  # Sample from a temporary distribution
"""

import numpy as np
import sciris as sc
from .utils import get_subclasses
from stisim.utils import INT_NAN
from stisim.random import SingleRNG, MultiRNG, RNG
from stisim import options

__all__ = [
    'Distribution', 'bernoulli', 'gamma', 'uniform', 'normal', 'poisson', 'rate', 'weibull', 
] #  , 'uniform_int', 'choice', 'normal_pos', 'normal_int', 'lognormal', 'lognormal_int', 'neg_binomial', 'beta', 'data_dist', 'delta',


_default_rng = np.random.default_rng()

class Distribution():

    def __init__(self, rng=None):
        """
        Create a Distribution object.

        :param rng: A random number generator instance or string used to create
        a random generator. A centralized random number generator will be used
        by default.
        :return:
        """
        self.sim = None # Only needed if using callable parameters.
        
        # Often distribution parameters are scalars so that the same values are
        # used for all samples.  But a user may want different parameters for
        # each agent.
        # A simple example is a binomial representing the probability of
        # infection - each agent might have a different probability of
        # infection given their susceptibiltiy and network connections.
        # But a binomial is easy because we can simply draw the right number of
        # random numbers between 0 and 1 and apply heterogeneous parameters
        # thereafter.
        # For other distributions, there may not be a convenient "standard"
        # distribution from which the right number of samples can be drawn and
        # later transformed according per-agent.
        # For this reason, we provide "needs_full_pars," which defaults to
        # False. When set to True, the parameter array will be expanded to match
        # the length of the number of samples requested from the distribution.
        # The "desired" samples can be selected thereafter.
        # The fill_value is used only when creating a full-length parameter array.
        # N.B. it's faster without needs_full_pars, so try to avoid if possible.
        self.needs_full_pars = False
        self.fill_value = 1 # Default fill value, only used if the Distribution needs_full_pars

        if rng is None:
            self.rng = _default_rng # Default to a single centralized random number generator
        else:
            if isinstance(rng, (MultiRNG, SingleRNG)):
                self.rng = rng
            elif isinstance(rng, str):
                self.rng = RNG(rng)
        return

    def initialize(self, sim):
        """
        Initialize the Distribution object. If a string name was provided as the
        rng, also initialize the rng.

        :param sim: The simulation object.
        :return:
        """
        self.sim = sim
        if not self.rng.initialized:
            self.rng.initialize(sim.rng_container, sim.people.slot)
        return

    def set_rng(self, rng):
        """
        Switch to a user-supplied random number generator
        """
        self.rng = rng
        return

    def __call__(self, size):
        return self.sample(size=size)

    @property
    def name(self):
        return self.__class__.__name__

    def sample(self, size=1, **kwargs):
        """
        Return a specified number of samples from the distribution
        """
        # Work out how many samples to draw. If sampling by UID, this depends on the slots assigned to agents.
        if np.isscalar(size):
            if not isinstance(size, (int, np.int64)):
                raise Exception('Input "size" must be an integer')
            if size < 0:
                raise Exception('Input "size" cannot be negative')
            elif size == 0:
                return np.array([], dtype=int)
            else:
                n_samples = size
        elif len(size) == 0:
            return np.array([], dtype=int)  # int dtype allows use as index, e.g. bernoulli_filter
        elif size.dtype == bool:
            n_samples = len(size) if options.multirng else size.sum()
        elif size.dtype == int:
            if not options.multirng:
                n_samples = len(size)
            else:
                v = size.__array__() # TODO - check if this works without calling __array__()?
                max_slot = self.rng.slots[v].__array__().max()
                if max_slot == INT_NAN:
                    raise Exception('Attempted to sample from an INT_NAN slot')
                n_samples = max_slot + 1
        else:
            raise Exception("Unrecognized input type")

        # Now handle distribution arguments
        pars = {}
        for k, v in kwargs.items():
            if np.isscalar(v):
                pars[k] = v
            else:
                if callable(v):
                    vals = v(self.sim, size)
                    pars[k] = vals
                else:
                    pars[k] = v
                
                if self.needs_full_pars and options.multirng:
                    if len(pars[k]) not in [len(size), sum(size)]: # Could handle uid and bool separately? len(size) for uid and sum(size) for bool
                        raise Exception('When providing an array of parameters, the length of the parameters must match the number of agents for the selected size (uids).')
                    vals_all = np.full(n_samples, fill_value=self.fill_value)
                    vals_all[size] = pars[k]
                    pars[k] = vals_all

        return n_samples, pars

    def filter(self, size, **kwargs):
        """
        Return the indices where the sample is True
        """
        return size[self.sample(size, **kwargs)]

    def _select(self, vals, size):
        if not options.multirng:
            return vals

        if np.isscalar(size):
            return vals
        elif size.dtype == bool:
            return vals[size]
        else:
            slots = self.rng.slots[size].__array__()
            return vals[slots]

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create a distribution instance by name

        :param name: A string with the name of a distribution class e.g., 'normal'
        :param args:
        :param kwargs:
        :return:
        """
        for subcls in get_subclasses(cls):
            if subcls.__name__ == name:
                return subcls(*args, **kwargs)
        else:
            raise KeyError(f'Distribution "{name}" did not match any known distributions')


class uniform(Distribution):
    """ Uniform distribution """
    def __init__(self, low=0, high=1, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, low=self.low, high=self.high)
        vals = self.rng.random(size=n_samples)
        vals = self._select(vals, size)
        return pars['low'] + (pars['high'] - pars['low'])*vals


class gamma(Distribution):
    """ Gamma distribution """
    def __init__(self, shape, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.needs_full_pars = True
        self.shape = shape
        self.scale = scale
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, shape=self.shape, scale=self.scale)
        vals = self.rng.gamma(size=n_samples, **pars) # Because needs_full_pars, individual pars will be scalars or have length equal to n_samples
        return self._select(vals, size)


class bernoulli(Distribution):
    """
    Bernoulli distribution, returns sequence of True or False from independent trials
    """
    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        return

    def sample(self, size=1):
        n_samples, pars = super().sample(size, p=self.p)
        vals = self.rng.random(size=n_samples)
        vals = self._select(vals, size)
        return vals < pars['p']

'''
class bernoulli_filter(Distribution):
    """
    Returns uids of True in a bernoulli draw
    """
    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        return

    def sample(self, size=1):
        n_samples, pars = super().sample(size, p=self.p)
        vals = self.rng.random(size=n_samples)
        vals = self._select(vals, size)
        return size[vals < pars['p']]
'''

class normal(Distribution):
    """ Normal distribution """
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        return

    def sample(self, size=1):
        n_samples, pars = super().sample(size, mean=self.mean, std=self.std)
        vals = self.rng.normal(size=n_samples)
        vals = self._select(vals, size)
        return pars['mean'] + self['std'] * vals


class poisson(Distribution):
    """ Poisson distribution """
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.needs_full_pars = True
        self.rate = rate
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, rate=self.rate)
        vals = self.rng.poisson(size=n_samples, lam=pars['rate']) # Use full pars array here, before _select
        vals = self._select(vals, size)
        return vals

class exponential(Distribution):
    """ Exponentials distribution """
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, scale=self.scale)
        vals = self.rng.exponential(size=n_samples, scale=pars['scale']) # Use full pars array here, before _select
        vals = self._select(vals, size)
        vals /= self.pars['rate']
        return vals


class rate(Distribution):
    """
    Exponentially distributed, accounts for dt.
    Assumes the rate is constant over each dt interval.
    """
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.dt = None
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.dt = sim.dt
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, rate=self.rate)
        prob = 1 - np.exp(-pars['rate'] * self.dt)
        vals = self.rng.random(size=n_samples)
        vals = self._select(vals, size)
        return vals < prob


class weibull(Distribution):
    """
    Weibull distribution parameterized in terms of:
    - shape (k)
    - scale (lambda)

    """
    def __init__(self, shape, scale, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape # k
        self.scale = scale # lambda
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, shape=self.shape, scale=self.scale)
        vals = self.rng.random(size=n_samples)
        vals = self._select(vals, size)
        # If U ~ uniform(0,1), then W ~ scale * (-ln(U))^(1/shape) is Weibull with shape and scale
        return pars['scale'] * (-np.log(vals))**(1/pars['shape'])

######################################################

'''
class delta(Distribution):
    """
    Delta function at specified value
    """

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _sample(self, size=1):
        return np.full(size, fill_value=self.value)


class data_dist(Distribution):
    """ Sample from data """

    def __init__(self, vals, bins, **kwargs):
        super().__init__(**kwargs)
        self.vals = vals
        self.bins = bins
        return

    def mean(self):
        return

    def _sample(self, size=None, rng=None, **kwargs):
        """ Sample using CDF """
        rng = rng or _default_rng
        bin_midpoints = self.bins[:-1] + np.diff(self.bins) / 2
        cdf = np.cumsum(self.vals)
        cdf = cdf / cdf[-1]
        values = rng.random(size=size)
        value_bins = np.searchsorted(cdf, values)
        return bin_midpoints[value_bins]


class uniform_int(uniform):
    """
    Uniform distribution returning only integer values
    Note that like its continuous counterpart, the upper endpoint is not included in the range.
    """

    def _sample(self, size=None, rng=None, **kwargs):
        return super()._sample(size, rng, **kwargs).astype(int)

class choice(Distribution):
    """
    Choose from samples, optionally with specific probabilities
    """

    def __init__(self, choices, probabilities=None, replace=True, **kwargs):
        super().__init__(**kwargs)
        self.choices = choices
        self.probabilities = probabilities
        self.replace = replace
        return

    def _sample(self, size=None, rng=None, **kwargs):
        rng = rng or _default_rng
        return rng.choice(size=size, a=self.choices, p=self.probabilities, replace=self.replace, **kwargs)


class normal_pos(normal):
    """
    right-sided normal (i.e. only +ve values), with mean=par1, std=par2 of the underlying normal

    WARNING - this function came from hpvsim but confirm that the implementation is correct?
    """

    def _sample(self, size=None, rng=None, **kwargs):
        return np.abs(super()._sample(size, rng, **kwargs))


class normal_int(normal):
    """
    Normal distribution returning only integer values
    """

    def _sample(self, size=None, rng=None, **kwargs):
        return np.round(super()._sample(size, rng, **kwargs))


class lognormal(Distribution):
    """
    lognormal with mean=par1, std=par2 (parameters are for the lognormal, not the underlying normal)

    Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
    function assumes the user wants to specify the mean and std of the lognormal distribution.
    """

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.check()

        self.underlying_mean = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))  # Computes the mean of the underlying normal distribution
        self.underlying_std = np.sqrt(np.log(std ** 2 / mean ** 2 + 1))  # Computes sigma for the underlying normal distribution


        return
    
    def check(self):
        if (sc.isnumber(self.mean) and self.mean > 0) or (sc.checktype(self.mean, 'arraylike') and (self.mean > 0).all()):
            return True
        raise Exception('The mean parameter passed to the lognormal distribution must be a positive number or array with all positive values.')

    def _sample(self, size=1, rng=None, **kwargs):
        rng = rng or _default_rng
        return rng.lognormal(size=size, mean=self.underlying_mean, sigma=self.underlying_std, **kwargs)


class lognormal_int(lognormal):
    """
    Lognormal returning only integer values
    """

    def _sample(self, size=None, rng=None, **kwargs):
        return np.round(super()._sample(size, rng, **kwargs))


class neg_binomial(Distribution):
    """
    Negative binomial distribution

    Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
    (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
    distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
    of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
    large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
    the mean.
    """

    def __init__(self, mean, dispersion, **kwargs):
        """
        mean (float): the rate of the process (same as Poisson)
        dispersion (float):  dispersion parameter; lower is more dispersion, i.e. 0 = infinite, ∞ = Poisson
        n (int): number of trials
        """
        super().__init__(**kwargs)
        self.mean = mean
        self.dispersion = dispersion
        return

    def _sample(self, size=None, rng=None):
        rng = rng or _default_rng
        nbn_n = self.dispersion
        nbn_p = self.dispersion / (self.mean + self.dispersion)
        return rng.negative_binomial(size=size, n=nbn_n, p=nbn_p)


class beta(Distribution):
    """
    Beta distribution
    """

    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        return

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def _sample(self,size,rng=None, **kwargs):
        rng = rng or _default_rng
        return rng.beta(size=size, a=self.alpha, b=self.beta, **kwargs)

'''