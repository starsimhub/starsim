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

__all__ = [
    'Distribution', 'uniform', 'choice', 'normal', 'normal_pos', 'normal_int', 'lognormal', 'lognormal_int',
    'poisson', 'neg_binomial', 'beta', 'gamma', 'from_data'
]


class Distribution():

    def __init__(self):
        return

    def mean(self):
        """
        Return the mean value
        """
        raise NotImplementedError

    def __call__(self, n=1, **kwargs):
        return self.sample(n, **kwargs)

    def sample(cls, n=1, **kwargs):
        """
        Return a specified number of samples from the distribution
        """
        raise NotImplementedError


class from_data(Distribution):
    """ Sample from data """

    def __init__(self, vals, bins):
        self.vals = vals
        self.bins = bins

    def mean(self):
        return

    def sample(self, n=1):
        """ Sample using CDF """
        bin_midpoints = self.bins[:-1] + np.diff(self.bins) / 2
        cdf = np.cumsum(self.vals)
        cdf = cdf / cdf[-1]
        values = np.random.rand(n)
        value_bins = np.searchsorted(cdf, values)
        return bin_midpoints[value_bins]


class uniform(Distribution):
    """
    Uniform distribution
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def mean(self):
        return (self.low + self.high) / 2

    def sample(self, n=1):
        return np.random.uniform(low=self.low, high=self.high, size=n)


class choice(Distribution):
    """
    Choose from samples, optionally with specific probabilities
    """

    def __init__(self, choices, probabilities=None, replace=True):
        self.choices = choices
        self.probabilities = probabilities
        self.replace = replace

    def sample(self, n, replace=True):
        return np.random.choice(a=self.choices, p=self.probabilities, replace=self.replace, size=n)


class normal(Distribution):
    """
    Normal distribution
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, n=1):
        return np.random.normal(loc=self.mean, scale=self.std, size=n)


class normal_pos(normal):
    """
    right-sided normal (i.e. only +ve values), with mean=par1, std=par2 of the underlying normal

    WARNING - this function came from hpvsim but confirm that the implementation is correct?
    """

    def sample(self, n=1):
        return np.abs(super().sample(n))


class normal_int(Distribution):
    """
    Normal distribution returning only integer values
    """

    def sample(self, n=1):
        return np.round(super().sample(n))


class lognormal(Distribution):
    """
    lognormal with mean=par1, std=par2 (parameters are for the lognormal, not the underlying normal)

    Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
    function assumes the user wants to specify the mean and std of the lognormal distribution.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.underlying_mean = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))  # Computes the mean of the underlying normal distribution
        self.underlying_std = np.sqrt(np.log(std ** 2 / mean ** 2 + 1))  # Computes sigma for the underlying normal distribution

    def sample(self, n=1):

        if (sc.isnumber(self.mean) and self.mean > 0) or (sc.checktype(self.mean, 'arraylike') and (self.mean > 0).all()):
            return np.random.lognormal(mean=self.underlying_mean, sigma=self.underlying_std, size=n)
        else:
            return np.zeros(n)


class lognormal_int(lognormal):
    """
    Lognormal returning only integer values
    """

    def sample(self, n=1):
        return np.round(super().sample(n))


class poisson(Distribution):
    """
    Poisson distribution
    """

    def __init__(self, rate):
        self.rate = rate

    def mean(self):
        return self.rate

    def sample(self, n=1):
        return np.random.poisson(self.rate, n)


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

    def __init__(self, mean, dispersion, step=1):
        """
        mean (float): the rate of the process (same as Poisson)
        dispersion (float):  dispersion parameter; lower is more dispersion, i.e. 0 = infinite, ∞ = Poisson
        n (int): number of trials
        step (float): the step size to use if non-integer outputs are desired
        """
        self.mean = mean
        self.dispersion = dispersion
        self.step = step

    def sample(self, n=1):
        nbn_n = self.dispersion
        nbn_p = self.dispersion / (self.mean / self.step + self.dispersion)
        return np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n) * self.step


class beta(Distribution):
    """
    Beta distribution
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def sample(self, n=1):
        return np.random.beta(a=self.alpha, b=self.beta, size=n)


class gamma(Distribution):
    """
    Gamma distribution
    """

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def mean(self):
        return self.shape * self.scale

    def sample(self, n=1):
        return np.random.gamma(shape=self.shape, scale=self.scale, size=n)
