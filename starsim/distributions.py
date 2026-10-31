"""
The distributions to choose from: `ss.uniform`, `ss.normal`, `ss.poisson`, and so on, each a random-number-safe `ss.Dist`

These exist so that a model can use any distribution for a parameter or a draw, with the same random numbers for the same agent across scenarios. How they work is in `ss.dists`. As agents, they are drawn for each agent; in a compartmental run, most are read for what they say about the group, such as their mean.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import numpy as np
import numba as nb
import scipy.special as sp_special
import sciris as sc
import starsim as ss
from .internals.dists import Dist, every, inverse, positive, some, sps
ss_int = ss.dtypes.int


def combine_rands(a, b):
    """
    Efficient algorithm for combining two arrays of random integers into an array
    of floats.

    See ss.multi_random() for the user-facing version.

    Inputs must be uint64: signed integers would produce output in [-0.5, 0.5] rather than
    [0, 1], so they raise an error rather than being cast, since casting would silently copy
    the array. To reinterpret existing bits for free, use `arr.view(np.uint64)`.

    Args:
        a (array): array of random uint64 integers between 0 and np.iinfo(np.uint64).max, as from ss.randint(low=0, high=np.iinfo('uint64').max, dtype=np.uint64)
        b (array): ditto, same size as a

    Returns:
        A new array of uniformly distributed random numbers in [0, 1], the same size as a and b
    """
    a_dtype = getattr(a, 'dtype', type(a).__name__)
    b_dtype = getattr(b, 'dtype', type(b).__name__)
    if a_dtype != np.uint64 or b_dtype != np.uint64:
        errormsg = f'combine_rands() requires uint64 arrays, not a={a_dtype} and b={b_dtype}. Signed integers give output in [-0.5, 0.5] instead of [0, 1]. Use ss.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64), or arr.view(np.uint64) to reinterpret the bits without copying.'
        raise TypeError(errormsg)
    c = np.bitwise_xor(a*b, a-b)
    u = c / np.iinfo(np.uint64).max
    return u


# Common distributions, assigned to a variable since used in help messages
dist_list = ['random', 'uniform', 'normal', 'truncnorm', 'lognorm_ex', 'lognorm_im', 'expon',
             'poisson', 'nbinom', 'binom', 'beta_dist', 'beta_mean', 'weibull', 'gamma', 'constant',
             'randint', 'bernoulli', 'choose_n', 'choice', 'histogram']



class random(Dist):
    """ Random distribution, with values on the interval (0, 1)

    Args:
        dtype (type): the data type of the values (default: `ss.dtypes.float`)
        kwargs (dict): passed to `ss.Dist`
    """
    valid_pars = ['dtype']
    scaling = None # Always (0,1), can't be scaled
    def __init__(self, dtype=None, **kwargs):
        super().__init__(distname='random', dtype=ss.dtypes.float if dtype is None else dtype, **kwargs)
        return

    def ppf(self, rands):
        return rands


class uniform(Dist):
    """
    Uniform distribution, values on interval (low, high)

    Args:
        low (float): the lower bound of the distribution (default 0.0)
        high (float): the upper bound of the distribution (default 1.0)
    """
    valid_pars = ['low', 'high']
    scaling = ['predraw', 'postdraw']
    unit_pars = ('low', 'high')

    def __init__(self, low=None, high=None, **kwargs):
        if high is None and low is not None: # One argument, swap
            high = low
            low = 0.0
        if low is None:
            low = 0.0
        if high is None:
            high = 1.0
        super().__init__(distname='uniform', low=low, high=high, **kwargs)
        return

    def make_rvs(self):
        """ Specified here because uniform() doesn't take a dtype argument """
        p = self._pars
        rvs = self.rand(self._size) * (p.high - p.low) + p.low
        return rvs

    def ppf(self, rands):
        p = self._pars
        rvs = rands * (p.high - p.low) + p.low
        return rvs


class normal(Dist):
    """
    Normal distribution

    Args:
        loc (float): the mean of the distribution (default 0.0)
        scale (float) the standard deviation of the distribution (default 1.0)

    """
    scaling = ['predraw', 'postdraw']
    unit_pars = ('loc', 'scale')
    def __init__(self, loc=0.0, scale=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='normal', dist='norm', loc=loc, scale=scale, **kwargs)
        return

    def ppf(self, rands):
        """ The inverse CDF: the same arithmetic as SciPy's, without its checking of the arguments, which is most of the cost of a small draw """
        kwds = self.dist.kwds
        scale = kwds['scale']
        if not positive(scale): # Leave the unusual cases to SciPy
            return super().ppf(rands)
        return sp_special.ndtri(rands)*np.asarray(scale) + kwds['loc']


class truncnorm(Dist):
    """
    Normal distribution truncated to an interval

    Args:
        loc (float): the mean of the normal distribution before truncation (default 0.0)
        scale (float): its standard deviation (default 1.0)
        low (float): the lowest value (default -inf)
        high (float): the highest value (default inf)

    Examples:
        ```python
        ss.truncnorm(loc=2.5, scale=0.5, low=1.0, high=4.0, strict=False).rvs(1000)
        ```
    """
    scaling = ['predraw', 'postdraw']
    unit_pars = ('loc', 'scale', 'low', 'high')

    def __init__(self, loc=0.0, scale=1.0, low=-np.inf, high=np.inf, **kwargs):
        super().__init__(distname='truncnorm', loc=loc, scale=scale, low=low, high=high, **kwargs)
        return

    def ppf(self, rands):
        """ The inverse CDF: a uniform draw between the normal's CDF at the two ends """
        p = self._pars
        a = sp_special.ndtr((p.low - p.loc)/p.scale)
        b = sp_special.ndtr((p.high - p.loc)/p.scale)
        return p.loc + p.scale*sp_special.ndtri(a + rands*(b - a))

    def make_rvs(self):
        return self.ppf(self.rand(self._size))


class lognorm_im(Dist):
    """
    Lognormal distribution, parameterized in terms of the "implicit" (normal)
    distribution, with mean=loc and std=scale (see lognorm_ex for comparison).

    Note: the "loc" parameter here does *not* correspond to the mean of the resulting
    random variates!

    Args:
        mean (float): the mean of the underlying normal distribution (not this distribution) (default 0.0)
        sigma (float): the standard deviation of the underlying normal distribution (not this distribution) (default 1.0)

    Examples:
        ```python
        ss.lognorm_im(mean=2, sigma=1, strict=False).rvs(1000).mean() # Should be roughly 10
        ```
    """
    scaling = 'postdraw'
    unitless_pars = ('mean', 'sigma')

    def __init__(self, mean=0.0, sigma=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='lognormal', dist='lognorm', mean=mean, sigma=sigma, **kwargs)
        return

    @staticmethod
    def scaled_pars(pars, factor):
        """ Parameters (as plain numbers, or functions of the agents) whose draws are those of `pars` times `factor`: the mean of the log moves by log(factor); used for `unit=` in a declared model """
        shift = np.log(factor)
        mean = pars['mean']
        if callable(mean):
            func = mean
            mean = lambda *args: func(*args) + shift
            mean.__signature__ = inspect.signature(func) # The distribution passes arguments by their names
        else:
            mean = mean + shift
        return pars | dict(mean=mean)

    def convert_timepars(self):
        for key, v in self._pars.items():
            if isinstance(v, ss.dur) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.dur):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')
            if isinstance(v, ss.Rate) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.Rate):
                raise NotImplementedError('lognormal_im parameters must be nondimensional')

    def sync_pars(self, call=True):
        """ Translate between NumPy and SciPy parameters """
        if call:
            self.call_pars()
        p = self._pars
        spars = dict(s=p.sigma, scale=np.exp(p.mean), loc=0)
        self.update_dist_pars(spars)
        return spars

    def ppf(self, rands):
        kwds = self.dist.kwds
        s, scale = kwds['s'], kwds['scale']
        if positive(s) and positive(scale): # The same arithmetic as SciPy's, without its checking of the arguments, which is most of the cost of a small draw
            return np.exp(np.asarray(s)*sp_special.ndtri(rands))*np.asarray(scale) + kwds['loc']
        s = np.asarray(s)
        rvs = Dist.ppf(self, rands)
        if some(s == 0): # SciPy gives NaN for sigma=0, but NumPy (and the limit) gives exp(mean)
            rvs = np.where(s == 0, self.dist.kwds['scale'], rvs)
        return rvs


class lognorm_ex(Dist):
    """
    Lognormal distribution, parameterized in terms of the "explicit" (lognormal)
    distribution, with mean=mean and std=std for this distribution (see lognorm_im for comparison).
    Note that a mean ≤ 0.0 is impossible, since this is the parameter of the distribution
    after the log transform.

    Args:
        mean (float): the mean of this distribution (not the underlying distribution) (default 1.0)
        std (float): the standard deviation of this distribution (not the underlying distribution) (default 1.0)

    Examples:
        ```python
        ss.lognorm_ex(mean=2, std=1, strict=False).rvs(1000).mean() # Should be close to 2
        ```
    """
    scaling = ['predraw', 'postdraw']
    unit_pars = ('mean', 'std')

    def __init__(self, mean=1.0, std=1.0, **kwargs): # Does not accept dtype
        super().__init__(distname='lognormal', dist='lognorm', mean=mean, std=std, **kwargs)
        return

    def convert_ex_to_im(self):
        """
        Lognormal distributions can be specified in terms of the mean and standard
        deviation of the "explicit" lognormal distribution, or the "implicit" normal distribution.
        This function converts the parameters from the lognormal distribution to the
        parameters of the underlying (implicit) distribution, which are the form expected by NumPy's
        and SciPy's lognorm() distributions.
        """
        p = self._pars
        if any(callable(v) and not isinstance(v, type) for v in p.values()): # Since can't work with functions (process_pars() has called them already, when drawing)
            self.call_pars()
        mean = p.pop('mean')
        std = p.pop('std')
        if np.isscalar(mean) and mean <= 0:
            errormsg = f'Cannot create a lognorm_ex distribution with mean≤0 (mean={mean}); did you mean to use lognorm_im instead?'
            raise ValueError(errormsg)
        std2 = std**2
        mean2 = mean**2
        sigma_im = np.sqrt(np.log(std2/mean2 + 1)) # Computes std for the underlying normal distribution
        mean_im  = np.log(mean2 / np.sqrt(std2 + mean2)) # Computes the mean of the underlying normal distribution
        p.mean = mean_im
        p.sigma = sigma_im
        return mean_im, sigma_im

    def sync_pars(self):
        """ Convert from overlying to underlying parameters, then translate to SciPy """
        self.convert_ex_to_im()
        spars = lognorm_im.sync_pars(self, call=False) # Borrow sync_pars from lognorm_im
        return spars

    def ppf(self, rands):
        return lognorm_im.ppf(self, rands) # Borrow ppf from lognorm_im


class expon(Dist):
    """
    Exponential distribution

    Args:
        scale (float): the scale of the distribution (default 1.0)
    """
    scaling = ['predraw', 'postdraw']
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(distname='exponential', dist='expon', scale=scale, **kwargs)
        return

    def ppf(self, rands):
        """ The inverse CDF, compiled (see `ss.solvers.backends.draws`); SciPy's for unusual parameters """
        scale, loc = self.dist.kwds.get('scale', 1.0), self.dist.kwds.get('loc', 0.0)
        if not positive(scale):
            return super().ppf(rands)
        rvs = inverse('expon', rands, scale)
        return rvs + loc if some(loc) else rvs


class poisson(Dist):
    """
    Poisson distribution

    Note: unlike rate-based distributions, the Poisson lambda parameter is not
    automatically scaled by dt. If using this distribution in a context where
    dt != 1, you may need to scale lambda manually (e.g., lam*dt).

    Args:
        lam (float): the scale of the distribution (default 1.0)
    """
    scaling = 'predraw'
    def __init__(self, lam=1.0, **kwargs):
        super().__init__(distname='poisson', dist='poisson', lam=lam, **kwargs)
        self._ppf_lam = None # Cached lambda for the fast ppf (see ppf())
        self._ppf_cdf = None # Cached CDF for the fast ppf
        return

    def sync_pars(self):
        """ Translate between NumPy and SciPy parameters """
        spars = dict(mu=self._pars.lam)
        self.update_dist_pars(spars)
        return spars

    def ppf(self, rands):
        """
        Fast inverse-CDF sampling for the common-random-number (ppf) path.

        SciPy's generic discrete ``ppf`` is very slow (it root-finds per element),
        and for CRN draws it is called every time an ``ss.poisson`` is sampled by
        UID. For a scalar ``lam`` we instead precompute the CDF once and map each
        uniform via ``np.searchsorted``, which is 30-60x faster and produces
        bitwise-identical results. Array-valued ``lam`` is inverted by `ss.solvers.backends.draws`
        (SciPy's above a mean of 30, and for unusual values).
        """
        lam = self._pars.lam
        if np.isscalar(lam) or (isinstance(lam, np.ndarray) and lam.ndim == 0):
            lam = float(lam)
            if self._ppf_lam != lam: # Rebuild the CDF only when lambda changes
                kmax = int(lam + 10.0*np.sqrt(lam) + 30) # P(X > kmax) is negligible, so the CDF saturates to 1.0
                self._ppf_cdf = sps.poisson.cdf(np.arange(kmax + 1), lam)
                self._ppf_lam = lam
            rvs = np.searchsorted(self._ppf_cdf, rands, side='left') # side='left' matches sps.poisson.ppf
        elif every(lam >= 0):
            rvs = inverse('poisson', rands, lam, big=lam > ss.solvers.backends.draws.INV_MAX, fallback=lambda u, i: sps.poisson.ppf(u, lam[i]))
        else:
            rvs = self.dist.ppf(rands)
        return rvs.astype(ss.dtypes.int) # Poisson yields integer counts, matching the native rvs path (SciPy's ppf returns float)


class beta_dist(Dist):
    """
    Beta distribution

    Args:
        a (float): shape parameter, must be > 0 (default 1.0)
        b (float): shape parameter, must be > 0 (default 1.0)
    """
    scaling = None
    unitless_pars = ('a', 'b')
    def __init__(self, a=1.0, b=1.0, **kwargs):
        super().__init__(distname="beta", dist='beta', a=a, b=b, **kwargs)
        return


class beta_mean(Dist):
    """
    Beta distribution paramterized by the mean

    Note: the variance of a beta distribution must be 0 < var < mean*(1-mean)

    Args:
        mean (float): mean of distribution, must be 0 < a < 1 (default 0.5)
        var (float): variance of distribution, must be > 0 (default 0.05)
        force (bool): if True, scale the parameters to the valid range
    """
    scaling = None
    unitless_pars = ('mean', 'var')
    def __init__(self, mean=0.5, var=0.05, force=False, **kwargs):  # Does not accept dtype
        # Validation
        eps = 1e-6 # With force=True, clip to strictly inside the valid range, since the bounds themselves are invalid
        if not (0 < mean < 1):
            if force:
                ss.warn(f'Clipping the mean from {mean:n} to 0 < mean < 1.')
                mean = np.clip(mean, eps, 1-eps)
            else:
                errormsg = f'The mean of a beta distribution must be 0 < mean < 1, not {mean:n}'
                raise ValueError(errormsg)
        max_var = mean*(1-mean)
        if not (0 < var < max_var):
            if force:
                ss.warn(f'Clipping the variance from {var} to 0 < var < {max_var:n}.')
                var = np.clip(var, eps*max_var, (1-eps)*max_var)
            else:
                errormsg = f'The variance of a beta distribution must be 0 < var < mean*(1-mean). For your {mean=}, {var=} is invalid; the maximum variance is {max_var:n}.'
                raise ValueError(errormsg)

        # We're good to go
        a = ((1 - mean)/var - 1/mean) * mean**2
        b = a * (1 / mean - 1)
        super().__init__(distname="beta", dist='beta', a=a, b=b, **kwargs)
        return


class nbinom(Dist):
    """
    Negative binomial distribution

    Args:
        n (float): the number of successes, > 1 (default 1.0)
        p (float): the probability of success in [0,1], (default 0.5)

    """
    scaling = 'predraw'
    unitless_pars = ('n',)
    def __init__(self, n=1, p=0.5, **kwargs):
        super().__init__(distname='negative_binomial', dist='nbinom', n=n, p=p, **kwargs)
        return

    def ppf(self, rands):
        """ The inverse CDF, compiled (see `ss.solvers.backends.draws`); SciPy's above a mean of 30, and for unusual parameters """
        n, p = np.asarray(self.dist.kwds['n']), np.asarray(self.dist.kwds['p'])
        if not (every(n > 0) and every(p > 0) and every(p <= 1)):
            return super().ppf(rands)
        big = n*(1 - p)/p > ss.solvers.backends.draws.INV_MAX
        return inverse('nbinom', rands, n, p, big=big, fallback=lambda u, i: sps.nbinom.ppf(u, n if n.ndim == 0 else n[i], p if p.ndim == 0 else p[i]))


class binom(Dist):
    """
    Binomial distribution: the number of successes in n trials; as a likelihood (see `sim.fit()`), the model's value is the probability, e.g. the positives among n tested

    Args:
        n (int): the number of trials (default 1)
        p (float): the probability of success in [0,1], (default 0.5)
    """
    scaling = 'predraw'
    unitless_pars = ('n',)
    def __init__(self, n=1, p=0.5, **kwargs):
        super().__init__(distname='binomial', dist='binom', n=n, p=p, **kwargs)
        return

    def ppf(self, rands):
        """ The inverse CDF, compiled (see `ss.solvers.backends.draws`); SciPy's above a mean of 30 (on the smaller side of 1/2), and for unusual parameters """
        n, p = np.asarray(self.dist.kwds['n']), np.asarray(self.dist.kwds['p'])
        if not (every(n >= 0) and every(n == np.floor(n)) and every(p >= 0) and every(p <= 1)):
            return super().ppf(rands)
        big = n*np.minimum(p, 1 - p) > ss.solvers.backends.draws.INV_MAX
        return inverse('binomial', rands, n, p, big=big, fallback=lambda u, i: sps.binom.ppf(u, n if n.ndim == 0 else n[i], p if p.ndim == 0 else p[i]))


class randint(Dist):
    """
    Random integer distribution, on the interval [low, high)

    Args:
        low (int): the lower bound of the distribution (default 0)
        high (int): the upper bound of the distribution (default of maximum integer size: 9,223,372,036,854,775,807)
        dtype (type): the data type of the values
    """
    scaling = 'predraw'
    unit_pars = ('low', 'high')
    valid_pars = ['low', 'high', 'dtype']

    def __init__(self, *args, low=None, high=None, dtype=None, **kwargs):
        # Handle input arguments # TODO: reconcile with how this is handled in uniform()
        if len(args):
            if len(args) == 1:
                high = args[0]
            elif len(args) == 2:
                low,high = args
            else:
                errormsg = f'ss.randint() takes one or two arguments, not {len(args)}'
                raise ValueError(errormsg)
        if dtype is None:
            dtype = ss.dtypes.rand_int # Not a default argument, so it follows ss.options.precision
        if low is None:
            low = 0
        if high is None:
            high = np.iinfo(ss.dtypes.rand_int).max

        # integers (rather than randint) because we interface a numpy.random.Generator
        super().__init__(distname='integers', low=low, high=high, dtype=dtype, **kwargs)
        return

    def convert_timepars(self):
        for key, v in self._pars.items():
            if isinstance(v, ss.dur) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.dur):
                raise NotImplementedError('randint parameters must be nondimensional')
            if isinstance(v, ss.Rate) or isinstance(v, np.ndarray) and v.shape and isinstance(v[0], ss.Rate):
                raise NotImplementedError('randint parameters must be nondimensional')

    @property
    def hash_dtype(self):
        """ Use float64 uniforms for wide ranges, since float32 only gives 2^24 distinct values """
        p = self._pars
        return np.float64 if np.any(p.high - p.low > 2**24) else None

    def ppf(self, rands):
        p = self._pars
        rvs = (rands * (p.high - p.low)).astype(p.dtype) + p.low  # Map [0,1) to [low, high), matching np.random.integers (exclusive high); convert before adding low so it floors rather than truncates
        rvs = np.minimum(rvs, p.high - 1)  # Clamp guards against rands rounding up to high
        return rvs


class weibull(Dist):
    """
    Weibull distribution (specifically, scipy.stats.weibull_min)

    Args:
        c (float): the shape parameter, sometimes called k (default 1.0)
        scale (float): the scale parameter, sometimes called λ (default 1.0)
        loc (float): the location parameter, which shifts the position of the distribution (default 0.0); by keyword only, so `ss.weibull(2, 5)` is a shape and a scale

    Examples:
        ```python
        ss.weibull(c=1.4, scale=ss.days(8))
        ```
    """
    scaling = 'postdraw'
    unitless_pars = ('c',)
    unit_pars = ('scale', 'loc')
    def __init__(self, c=1.0, scale=1.0, *, loc=0.0, **kwargs):
        super().__init__(distname='weibull', dist='weibull_min', c=c, scale=scale, loc=loc, **kwargs)
        return

    def ppf(self, rands):
        """ The inverse CDF, compiled (see `ss.solvers.backends.draws`); SciPy's for unusual parameters """
        kw = self.dist.kwds
        c, scale, loc = kw['c'], kw.get('scale', 1.0), kw.get('loc', 0.0)
        if not (every(np.asarray(c) > 0) and every(np.asarray(scale) > 0)):
            return super().ppf(rands)
        rvs = inverse('weibull', rands, c, scale)
        return rvs + loc if some(loc) else rvs

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


class gamma(Dist):
    """
    Gamma distribution (specifically, scipy.stats.gamma)

    Given by its shape and scale, or by its mean and standard deviation.

    Args:
        a (float): the shape parameter, sometimes called k (default 1.0)
        scale (float): the scale parameter, sometimes called θ (default 1.0)
        loc (float): the location parameter, which shifts the position of the distribution (default 0.0); by keyword only, so `ss.gamma(2, 5)` is a shape and a scale
        mean (float): the mean, instead of `a` and `scale` (with `std`)
        std (float): the standard deviation, with `mean`

    Examples:
        ```python
        ss.gamma(a=2, scale=ss.days(5)) # A mean of 10 days
        ss.gamma(mean=ss.days(10), std=ss.days(7)) # About the same
        ss.par(1800, prior=ss.gamma(mean=1800, std=400))
        ```
    """
    scaling = 'postdraw'
    unitless_pars = ('a',)
    unit_pars = ('scale', 'loc')
    def __init__(self, a=None, scale=None, *, loc=0.0, mean=None, std=None, **kwargs):
        if mean is not None or std is not None:
            if mean is None or std is None or a is not None or scale is not None:
                errormsg = f'ss.gamma() is given by a and scale, or by mean and std, not by {sc.strjoin([k for k,v in dict(a=a, scale=scale, mean=mean, std=std).items() if v is not None])}'
                raise ValueError(errormsg)
            given = dict(mean=mean, std=std) | (dict(loc=loc) if loc else {})
            a = float((mean/std)**2) # A mean of a×scale and a variance of a×scale²
            scale = mean/a
        a = sc.ifelse(a, 1.0)
        scale = sc.ifelse(scale, 1.0)
        super().__init__(distname='gamma', dist='gamma', a=a, scale=scale, loc=loc, **kwargs)
        if mean is not None:
            self.given = ('gamma', given)
        return

    def make_rvs(self):
        """ Use SciPy rather than NumPy to include the scale parameter """
        rvs = self.dist.rvs(self._size)
        return rvs


def erlang(k=1, mean=1.0, **kwargs):
    """
    Erlang distribution: the sum of `k` exponential stages, given by its number of stages and its mean

    This is `ss.gamma` with an integer shape: `ss.erlang(k=3, mean=10)` is `ss.gamma(a=3, scale=10/3)`.
    As a dwell time, it has an exact compartmental form (`k` sub-stages), so runs the same under every solver.

    Args:
        k (int): the number of stages (the shape parameter)
        mean (float): the mean of the distribution (not of each stage)
        kwargs (dict): passed to `ss.gamma()`

    Examples:
        ```python
        ss.erlang(k=3, mean=ss.days(10))
        ```
    """
    if k != int(k) or k < 1:
        errormsg = f'The number of stages of an Erlang distribution must be a positive whole number, not {k}; for other shapes, use ss.gamma()'
        raise ValueError(errormsg)
    dist = gamma(a=int(k), scale=mean/k, **kwargs)
    dist.given = ('erlang', dict(k=int(k), mean=mean) | ({'loc':kwargs['loc']} if kwargs.get('loc') else {}))
    return dist


class constant(Dist):
    """
    Constant (delta) distribution: equivalent to np.full()

    Args:
        v (float): the value to return
    """
    valid_pars = ['v']
    scaling = ['predraw', 'postdraw']

    def __init__(self, v=0.0, **kwargs):
        super().__init__(distname='const', v=v, **kwargs)
        return

    def make_rvs(self):
        return np.full(self._size, self._pars.v)

    def ppf(self, rands): # NB: don't actually need to use random numbers here, but not worth the complexity of avoiding this
        return np.full(rands.shape, self._pars.v)


class bernoulli(Dist):
    """
    Bernoulli distribution: return True or False with the specified probability (which can be an array)

    Unlike other distributions, Bernoulli distributions have a filter() method,
    which returns elements of the array that return True.

    Args:
        p (float): the probability of returning True (default 0.5)
    """
    valid_pars = ['p']
    scaling = 'predraw'

    def __init__(self, p=0.5, **kwargs):
        super().__init__(distname='bernoulli', p=p, **kwargs)
        return

    def __bool__(self):
        if callable(self.pars.p) or np.any(sc.promotetoarray(self.pars.p) > 0):
            return True
        else:
            return False

    def make_rvs(self):
        rvs = self.rand(self._size) < self._pars.p # 3x faster than using rng.binomial(1, p, size)
        return rvs

    def ppf(self, rands):
        rvs = rands < self._pars.p
        return rvs

    def filter(self, uids=None, both=False):
        """ Return UIDs that correspond to True, or optionally return both True and False

        Args:
            uids (array): the agents to choose among (default: all active agents)
            both (bool): if True, return the UIDs for both True and False
        """
        if uids is None:
            uids = self.sim.people.auids # All active UIDs
        elif isinstance(uids, (ss.BoolArr, ss.IndexArr)):
            uids = uids.uids

        bools = self.rvs(uids)
        if both:
            return uids[bools], uids[~bools]
        else:
            return uids[bools]

    def split(self, uids=None):
        """ Alias to filter(uids, both=True)

        Args:
            uids (array): the agents to choose among (default: all active agents)
        """
        return self.filter(uids=uids, both=True)


class choose_n(Dist):
    """
    Choose exactly n agents (without replacement) from a set of candidate UIDs

    Like `ss.bernoulli`, use the filter() method, which returns the chosen UIDs.
    One random number is drawn per candidate (based on its slot), and the n
    candidates with the smallest numbers are chosen. This means that if the
    candidates change slightly (e.g. between two scenarios), most of the same
    agents are still chosen (unlike e.g. `ss.choice(replace=False)`).

    If weights are supplied, candidates are chosen with probability proportional
    to their weight, via the Efraimidis–Spirakis key `(1-u)**(1/w)`. This is computed
    as `-log(1-u)/w`, so the n smallest are chosen as before, and equal weights
    give the same agents as no weights. Candidates with zero weight are never
    chosen, so fewer than n UIDs are returned if fewer than n candidates have
    nonzero weight.

    Args:
        n (int/func): the number of agents to choose (or a function returning it); if more than the number of candidates, all are chosen
        weights (array/func): if supplied, the relative weight of each candidate (or a function returning them), matching the UIDs passed to filter()
        die (bool): if True, raise an exception if fewer than n agents can be chosen (default: False)

    Examples:
        ```python
        # Seed exactly 20 infections of a disease written as code
        sir = ssl.AgentSIR(init_prev=ss.choose_n(20))

        # Choose 5 agents, weighted by age
        chooser = ss.choose_n(5, weights=lambda self, sim, uids: sim.people.age[uids])
        ```
    """
    valid_pars = ['n', 'weights']
    scaling = None
    unitless_pars = ('n', 'weights')

    def __init__(self, n=1, weights=None, die=False, **kwargs):
        super().__init__(distname='choose_n', n=n, weights=weights, **kwargs)
        self.die = die
        return

    def make_rvs(self):
        return self.rand(self._size)

    def ppf(self, rands):
        return rands

    def rvs(self, n=1, round=None, reset=False):
        """ Return a boolean array of whether each agent is chosen (n can be UIDs or a size), so ss.choose_n() can be used in place of ss.bernoulli()

        Args:
            n (int/array): the UIDs, or a size
            round (bool): not used; as in `ss.Dist.rvs()`
            reset (bool): not used; as in `ss.Dist.rvs()`
        """
        chosen = np.zeros(n if np.isscalar(n) else len(n), dtype=bool)
        chosen[self._choose(n)] = True
        return chosen

    def filter(self, uids=None, both=False):
        """ Return the chosen UIDs (sorted), i.e. min(n, len(uids)) of them; if both=True, also return the UIDs not chosen (as ss.bernoulli.filter())

        Args:
            uids (array): the agents to choose among (default: all active agents)
            both (bool): if True, also return the UIDs not chosen
        """
        if uids is None:
            uids = self.sim.people.auids # All active UIDs
        elif isinstance(uids, (ss.BoolArr, ss.IndexArr)):
            uids = uids.uids
        chosen = ss.uids(np.sort(uids[self._choose(uids)]))
        if both:
            return chosen, ss.uids(uids).remove(chosen)
        return chosen

    def split(self, uids=None):
        """ Alias to filter(uids, both=True)

        Args:
            uids (array): the agents to choose among (default: all active agents)
        """
        return self.filter(uids=uids, both=True)

    def _choose(self, n):
        """ Return the indices of the chosen agents (n can be UIDs or a size); see filter() """
        keys = super().rvs(n) # One random number per candidate; also evaluates the parameters
        if not len(keys): # No candidates: rvs() returns early without evaluating the parameters, so don't use them
            if self.die:
                errormsg = f'{self} could not choose any agents since there are no candidates'
                raise ValueError(errormsg)
            return np.array([], dtype=ss_int)
        n_choose = min(int(self._pars.n), len(keys))
        weights = self._pars.weights
        if weights is not None:
            weights = np.asarray(weights)
            if np.any(weights < 0):
                errormsg = f'Weights for {self} must be non-negative'
                raise ValueError(errormsg)
            with np.errstate(divide='ignore'): # Zero weights give infinite keys, so are never chosen
                keys = -np.log1p(-keys)/weights
            n_choose = min(n_choose, np.count_nonzero(keys < np.inf))
        if self.die and n_choose < int(self._pars.n):
            errormsg = f'{self} could only choose {n_choose} of the {int(self._pars.n)} requested agents'
            raise ValueError(errormsg)
        if n_choose <= 0:
            return np.array([], dtype=ss_int)
        inds = np.argpartition(keys, n_choose-1)[:n_choose] # Indices of the n smallest keys
        return inds


class choice(Dist):
    """
    Random choice between discrete options

    The probabilities can differ by agent: give `p` as a 2D array with one row
    per agent drawn for (or a function returning one), and one column per choice.
    Each row is normalized to sum to 1. Each agent's choice is found from their
    own random number by the inverse CDF of their own row, so the draw is one
    vectorized call, and keeps common random numbers: an agent's choice doesn't
    depend on who else is drawn for, or on other agents' probabilities.

    Args:
        a (int or array): the number of choices, or the choices themselves (default 2)
        p (array/func): if supplied, the probability of each choice (default, 1/a for a choices); or a 2D array (or a function returning one) with a row of probabilities (or weights) for each agent
        replace (bool): whether an option can be chosen more than once (default True)

    Examples:
        ```python
        # Simulate 10 die rolls
        ss.choice(6, strict=False)(10) + 1

        # Choose between specified options each with a specified probability (must sum to 1)
        ss.choice(a=[30, 70], p=[0.3, 0.7], strict=False)(10)

        # A row of probabilities per agent: here by age group, from a table of age groups × choices
        table = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
        def p_by_age(self, sim, uids):
            return table[(sim.people.age[uids] >= 18).astype(int)]
        method = ss.choice(a=['none', 'pill', 'iud'], p=p_by_age) # In a module's __init__()
        chosen = self.method.rvs(uids) # In its step()
        ```
    Note: although Bernoulli trials can be generated using a=2, it is much faster
    to use ss.bernoulli() instead.
    """
    valid_pars = ['a', 'p', 'replace', 'dtype']
    scaling = 'postdraw' # A choice is one of its values, so values with a unit (unit=, or typed values) are the numbers times the unit
    unitless_pars = ('p',)

    def __init__(self, a=2, p=None, replace=True, **kwargs):
        items = [a] if isinstance(a, ss.TimePar) else sc.tolist(a) if not np.isscalar(a) else []
        if items and all(isinstance(v, ss.TimePar) for v in items): # e.g. a=[ss.months(9), ss.months(10)] or ss.years([1, 2]): the numbers, in the unit of the first
            unit = type(items[0])
            if kwargs.get('unit') is not None:
                raise ValueError(f'ss.choice() has typed values ({items[0]!r}, ...) and unit={kwargs["unit"]!r}; give one or the other')
            a = np.concatenate([np.atleast_1d(np.asarray((v if type(v) is unit else unit(v)).value, dtype=float)) for v in items])
            kwargs['unit'] = unit
        super().__init__(distname='choice', a=a, p=p, replace=replace, **kwargs)
        if not replace:
            # Sampling without replacement is a joint draw over all elements, so it
            # cannot be expressed as an independent per-slot inverse-CDF. Keep the
            # native (make_rvs) path in that case, but draw exactly len(uids) values
            # rather than slots.max()+1 (see process_size below).
            self._use_ppf = False
        # For the (default) replace=True case, leave _use_ppf as None so that the
        # ppf path is used for UID draws. This maps each slot's uniform through the
        # inverse CDF (below), which is CRN-safe and, crucially, avoids the
        # make_rvs "slot blowup" (drawing slots.max()+1 values and discarding most)
        # -- the same issue hash-based CRN already fixed for bernoulli/poisson/etc.
        return

    def process_size(self, n=1):
        """ For replace=False, draw one value per UID (no slot blowup); not for the user """
        size, slots = super().process_size(n)
        if not self.pars.replace and slots is not None:
            # Without replacement can't be slot-indexed CRN (it's a joint draw), and
            # requesting slots.max()+1 distinct items both blows up and errors once
            # slots.max() >= len(a). Draw exactly one value per UID instead. Like
            # ss.options.crn=False, this is statistically valid but not CRN-safe
            # across scenarios -- which is inherent to sampling without replacement.
            self._slots = None
            size = self._size = len(self._uids)
            slots = None
        return size, slots

    def ppf(self, rands):
        """ Map uniform values to choices via the inverse CDF (one draw per slot). """
        pars = self._pars
        a = pars.a
        if np.isscalar(a):
            a = np.arange(a)
        else:
            a = np.asarray(a) # Ensure array indexing works below (e.g. a=[30, 70] passed as a list)
        if pars.p is None: # Uniform over the choices
            n = len(a)
            inds = np.minimum((rands*n).astype(int), n - 1) # minimum guards against rands == 1.0
        elif np.ndim(pars.p) == 2: # A row of probabilities per agent: each agent's uniform through their own row's CDF
            pcum = np.cumsum(pars.p, axis=1)
            if pcum.shape != (len(rands), len(a)):
                errormsg = f'{self} has p with shape {pcum.shape}, but needs one row per agent drawn for and one column per choice, i.e. {(len(rands), len(a))}'
                raise ValueError(errormsg)
            total = pcum[:, -1:]
            if np.any(total <= 0):
                errormsg = f'{self} has a row of probabilities that sums to 0; each agent needs at least one choice with a positive probability'
                raise ValueError(errormsg)
            inds = np.minimum((pcum <= rands[:, None]*total).sum(axis=1), len(a) - 1) # Normalizing each row by its total; the first choice whose cumulative probability exceeds the uniform
        else:
            pcum = np.cumsum(pars.p)
            inds = np.minimum(np.searchsorted(pcum, rands, side='right'), len(a) - 1)
        return a[inds]

    def make_rvs(self):
        """ Draws without UIDs (a size, or replace=False) when p is a row per draw: the same inverse CDF as ppf(); not for the user """
        if np.ndim(self._pars.p) == 2:
            if not self._pars.replace:
                errormsg = f'{self} has a row of probabilities per agent, which needs replace=True'
                raise ValueError(errormsg)
            return self.ppf(self.rand(self._size))
        return super().make_rvs()


class histogram(Dist):
    """
    Sample from a histogram with defined bins

    Note: unlike other distributions, the parameters of this distribution can't
    be modified after creation.

    Args:
        values (array): the probability (or count) of each bin
        bins (array): the edges of each bin
        density (bool): treat the histogram as a density instead of counts; only matters with unequal bin widths, see numpy.histogram and scipy.stats.rv_histogram for more information
        data (array): if supplied, compute the values and bin edges using this data and np.histogram() instead

    Note: if the length of bins is equal to the length of values, they will be
    interpreted as left bin edges, and one additional right-bin edge will be added
    based on the difference between the last two bins (e.g. if the last two bins are
    40 and 50, the final right edge will be added at 60). If no bins are supplied,
    then they will be created as integers matching the length of the values.

    The values can be supplied in either normalized (sum to 1) or un-normalized
    format.

    Examples:
        ```python
        # Sample from an age distribution
        age_bins = [0,    10,  20,  40,  65, 100]
        age_vals = [0.1, 0.1, 0.3, 0.3, 0.2]
        h1 = ss.histogram(values=age_vals, bins=age_bins, strict=False)
        h1.plot_hist()

        # Create a histogram from data
        data = np.random.randn(10_000)*2+5
        h2 = ss.histogram(data=data, strict=False)
        h2.plot_hist(bins=100)
        ```
    """
    valid_pars = ['values', 'bins', 'density', 'data', 'group']
    scaling = 'postdraw' # Its draws are within its bins, so bins with a unit (unit=) are the numbers times the unit

    def __init__(self, values=None, bins=None, density=False, data=None, **kwargs):
        if data is not None:
            if values is not None:
                errormsg = 'You can supply values or data, but not both'
                raise ValueError(errormsg)
            kw = {'bins':bins} if bins is not None else {} # Hack to not overwrite default bins value
            values, bins = np.histogram(data, **kw)
        else:
            if values is None:
                values = [1.0] # Uniform distribution
            values = sc.toarray(values)
            if bins is None:
                bins = np.arange(len(values)+1)
            bins = sc.toarray(bins)
        if len(bins) == len(values): # Append a final bin, if necessary
            delta = bins[-1] - bins[-2]
            bins = np.append(bins, bins[-1]+delta)

        # Validate inputs
        values = values.astype(float)
        vsum = values.sum()
        if vsum != 1.0:
            values = values / vsum
        widths = np.diff(bins)
        pdf = values if density or (density is None and not np.allclose(widths, widths[0])) else values/widths # As scipy.stats.rv_histogram, which takes unequal bins as a density
        pdf = pdf/np.sum(pdf*widths)
        super().__init__(distname='histogram', **kwargs)
        self.cdf_bins = (np.hstack([0.0, np.cumsum(pdf*widths)]), bins.astype(float)) # For ppf()
        self.hist = (values, bins, density) # As given (not in pars): for copies, see ss.transitions.numeric_dist()
        self.parts = None # One histogram for each group, see by_group()
        # Leave _use_ppf as None (the default) so UID draws go through the hash-based
        # CRN path: one uniform per slot mapped through the histogram's inverse
        # CDF (ppf()). This is CRN-safe and draws exactly len(uids) numbers,
        # avoiding the slot_scale "blowup" (drawing slots.max()+1 and discarding most).
        # The values/bins arrays are not treated as per-UID pars.
        return

    @classmethod
    def by_group(cls, parts, group, **kwargs):
        """
        One histogram for each group: each person is drawn from the histogram of their group

        Args:
            parts (list): the histograms (`ss.histogram`), one for each group
            group: each person's group, as an index into `parts` (e.g. `ss.by(urban={False: 0, True: 1})`, or a function of the agents)
            kwargs (dict): passed to `ss.histogram`
        """
        values, bins, density = parts[0].hist
        dist = cls(values=values, bins=bins, density=density, group=group, **kwargs)
        dist.parts = [part.hist for part in parts]
        dist.cdf_parts = [part.cdf_bins for part in parts]
        return dist

    def make_rvs(self):
        """ Draws without UIDs: the same inverse CDF as ppf(); not for the user """
        return self.ppf(self.rand(self._size))

    def ppf(self, rands):
        """ The inverse of the CDF, which is piecewise linear """
        def one(rands, cdf, bins):
            return np.where(rands == 0, bins[0], np.interp(rands, cdf, bins)) # SciPy gives the lowest bin edge for 0, even if the first bins are empty
        if self.parts is None:
            return one(rands, *self.cdf_bins)
        group = np.broadcast_to(np.asarray(self._pars['group']), np.shape(rands)) # Each person's histogram
        out = np.empty(np.shape(rands))
        for i,(cdf,bins) in enumerate(self.cdf_parts):
            these = group == i
            out[these] = one(rands[these], cdf, bins)
        return out


class multi_random(sc.prettyobj):
    """
    A class for holding two or more ss.random() distributions, and generating
    random numbers linked to each of them. Useful for e.g. pairwise transmission
    probabilities.

    See ss.distributions.combine_rands() for the manual version; in almost all cases this class
    should be used instead.

    Args:
        names (str/list): name(s) for each internal random distribution
        *args: additional names (shorthand)
        crn (bool): whether to use common random numbers; if None (default), follow `ss.options.crn` (note: only `crn=False` can override it, since `crn=True` has no effect if `ss.options.crn=False`)
        **kwargs: passed to each `ss.random()` instance

    Usage:
        multi = ss.multi_random('source', 'target')
        rvs = multi.rvs(source_uids, target_uids)
    """
    def __init__(self, names, *args, crn=None, **kwargs):
        names = sc.mergelists(names, *args)
        self.dists = [ss.random(name=name, **kwargs) for name in names]
        self.crn = crn # If None, follow ss.options.crn at draw time
        return

    def __len__(self):
        return len(self.dists)

    def init(self, *args, **kwargs):
        """ Not usually needed since each dist will handle this automatically; for completeness only

        Args:
            args (list): passed to the `init()` of each distribution
            kwargs (dict): passed to the `init()` of each distribution
        """
        for dist in self.dists: dist.init(*args, **kwargs)
        return

    def reset(self, *args, **kwargs):
        """ Not usually needed since each dist will handle this automatically; for completeness only

        Args:
            args (list): passed to the `reset()` of each distribution
            kwargs (dict): passed to the `reset()` of each distribution
        """
        for dist in self.dists: dist.reset(*args, **kwargs)
        return

    def jump(self, *args, **kwargs):
        """ Not usually needed since each dist will handle this automatically; for completeness only

        Args:
            args (list): passed to the `jump()` of each distribution
            kwargs (dict): passed to the `jump()` of each distribution
        """
        for dist in self.dists: dist.jump(*args, **kwargs)
        return

    @staticmethod
    @nb.jit(fastmath=True, parallel=False, cache=True) # Numba is 3x faster, but disabling parallel for efficiency
    def combine_rvs(rvs_list, int_type, int_max):
        """ Combine inputs into one number """
        # Combine using bitwise-xor
        rand_ints = rvs_list[0].view(int_type)
        for rand_floats in rvs_list[1:]:
            rand_ints2 = rand_floats.view(int_type)
            rand_ints = np.bitwise_xor(rand_ints*rand_ints2, rand_ints-rand_ints2)

        # Normalize
        rvs = rand_ints / int_max
        return rvs

    @staticmethod
    @nb.jit(fastmath=True, parallel=False, cache=True)
    def combine2_rvs(a, b, int_type, int_max):
        """ Fast path for the common two-distribution case (avoids nb.typed.List, ~550us/call) """
        ra = a.view(int_type)
        rb = b.view(int_type)
        rand_ints = np.bitwise_xor(ra*rb, ra-rb)
        return rand_ints / int_max

    def rvs(self, *args):
        """ Get random variates from each of the underlying distributions and combine them efficiently

        Args:
            args (list): the UIDs for each distribution, in order
        """
        # Validation
        n_args = len(args)
        n_dists = len(self)
        if n_args != len(self):
            errormsg = f'Number of UID lists supplied ({n_args}) does not match number of distributions ({n_dists})'
            raise ValueError(errormsg)

        # Fast non-CRN path: a single plain uniform draw per element, skipping the
        # slot gathers and XOR combine. Statistically valid but not CRN-safe.
        crn = ss.options.crn if self.crn is None else self.crn
        if not crn:
            dist = self.dists[0]
            rvs = dist.rng.random(len(args[0]))
            dist.jump() # Advance the RNG manually, since we bypassed dist.rvs() above
            return rvs

        rvs_list = [dist.rvs(arg) for dist,arg in zip(self.dists, args)]

        # The XOR-combine reinterprets the float bytes as unsigned ints, so the int width must
        # match the float width - we need to base it on the width of ss.dtypes.float, not ss.dtypes.rand_uint
        itemsize = np.dtype(ss.dtypes.float).itemsize
        if itemsize == 4:
            int_type = np.uint32
        elif itemsize == 8:
            int_type = np.uint64
        else:
            errormsg = f'Unexpected float itemsize {itemsize}; expected 4 (float32) or 8 (float64)'
            raise ValueError(errormsg)
        int_max = np.iinfo(int_type).max
        if n_dists == 2: # Common case (e.g. pairwise transmission): skip the costly nb.typed.List build
            rvs = self.combine2_rvs(rvs_list[0], rvs_list[1], int_type, int_max)
        else:
            nb_list = nb.typed.List(rvs_list) # See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
            rvs = self.combine_rvs(nb_list, int_type, int_max)
        return rvs
