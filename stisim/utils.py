"""
Numerical utilities
"""

# %% Housekeeping

import warnings
import numpy as np
import sciris as sc
from . import settings as sss


# What functions are externally visible -- note, this gets populated in each section below
__all__ = []


# %% Helper functions
__all__ += ['ndict', 'omerge']


class ndict(sc.objdict):
    """
    A dictionary-like class that provides additional functionalities for handling named items.

    Args:
        _name (str): The items' attribute to use as keys.
        _type (type): The expected type of items.
        _strict (bool): If True, only items with the specified attribute will be accepted.

    **Examples**::

        networks = ss.ndict(ss.simple_sexual(), ss.maternal())
        networks = ss.ndict([ss.simple_sexual(), ss.maternal()])
        networks = ss.ndict({'simple_sexual':ss.simple_sexual(), 'maternal':ss.maternal()})

    """

    def __init__(self, *args, _name='name', _type=None, _strict=True, **kwargs):
        self.setattribute('_name', _name) # Since otherwise treated as keys
        self.setattribute('_type', _type)
        self.setattribute('_strict', _strict)
        argdict = self._validate(*args)
        argdict.update(kwargs)
        super().__init__(argdict)
        return
        
            
    def _validate(self, *args):
        args = sc.mergelists(*args)
        _name   = self.getattribute('_name')
        _type   = self.getattribute('_type')
        _strict = self.getattribute('_strict')
        failed = []
        argdict = {}
        for i,arg in enumerate(args):
            if arg is None:
                pass
            elif hasattr(arg, _name) or not _strict:
                try:
                    argdict[getattr(arg, _name)] = arg
                except:
                    i = 0
                    make_key = lambda i: f'item{i}'
                    while make_key(i) in argdict:
                        i += 1
                    argdict[make_key(i)] = arg
            elif isinstance(arg, dict):
                argdict.update(arg)
            else:
                failed.append(i)
        
        # Check types
        if _type is not None:
            wrong = {}
            for k,v in argdict.items():
                if not isinstance(v, _type):
                    wrong[k] = type(v)
            if len(wrong):
                errormsg = f'The following items do not have the expected type {self._type}:\n{wrong}'
                raise TypeError(errormsg)
        
        if len(failed):
            errormsg = f'Could not interpret arguments {failed}: does not have expected attribute "{self._name}"'
            raise ValueError(errormsg)
        else:
            return argdict
        
    
    def append(self, *args):
        ''' Allow being used like a list '''
        argdict = self._validate(*args)
        self.update(argdict)
        return
    
    
    def __add__(self, dict2):
        ''' Allow two dictionaries to be added (merged) '''
        return sc.mergedicts(self, self._validate(dict2))



def omerge(*args, **kwargs):
    """ Merge things into an objdict """
    return sc.objdict(sc.mergedicts(*args, **kwargs))


def warn(msg, category=None, verbose=None, die=None):
    """ Helper function to handle warnings -- not for the user """

    # Handle inputs
    warnopt = sss.options.warnings if not die else 'error'
    if category is None:
        category = RuntimeWarning
    if verbose is None:
        verbose = sss.options.verbose

    # Handle the different options
    if warnopt in ['error', 'errors']:  # Include alias since hard to remember
        raise category(msg)
    elif warnopt == 'warn':
        msg = '\n' + msg
        warnings.warn(msg, category=category, stacklevel=2)
    elif warnopt == 'print':
        if verbose:
            msg = 'Warning: ' + msg
            print(msg)
    elif warnopt == 'ignore':
        pass
    else:
        options = ['error', 'warn', 'print', 'ignore']
        errormsg = f'Could not understand "{warnopt}": should be one of {options}'
        raise ValueError(errormsg)

    return


# %% The core functions
def unique(arr):
    """
    Find the unique elements and counts in an array.
    Equivalent to np.unique(return_counts=True) but ~5x faster, and
    only works for arrays of positive integers.
    """
    counts = np.bincount(arr.ravel())
    unique = np.flatnonzero(counts)
    counts = counts[unique]
    return unique, counts


def find_contacts(p1, p2, inds):  # pragma: no cover
    """
    Variation on Network.find_contacts() that avoids sorting.

    A set is returned here rather than a sorted array so that custom tracing interventions can efficiently
    add extra people. For a version with sorting by default, see Network.find_contacts(). Indices must be
    an int64 array since this is what's returned by true() etc. functions by default.
    """
    pairing_partners = set()
    inds = set(inds)
    for i in range(len(p1)):
        if p1[i] in inds:
            pairing_partners.add(p2[i])
        if p2[i] in inds:
            pairing_partners.add(p1[i])
    return pairing_partners


# %% Sampling and seed methods

__all__ += ['sample', 'set_seed']


def sample(dist=None, par1=None, par2=None, size=None, **kwargs):
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
        dist (str):   the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int):   the number of samples (default=1)
        kwargs (dict): passed to individual sampling functions

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
        samples = n_poisson(rate=par1, n=size)  # Use Numba version below for speed
    elif dist == 'neg_binomial':
        samples = n_neg_binomial(rate=par1, dispersion=par2, n=size, **kwargs)  # Use custom version below
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


def set_seed(seed=None):
    """
    Reset the random seed. This function also resets Python's built-in random
    number generated.

    Args:
        seed (int): the random seed
    """
    # Dies if a float is given
    if seed is not None:
        seed = int(seed)
    np.random.seed(seed)  # If None, reinitializes it
    return


# %% Probabilities -- mostly not jitted since performance gain is minimal

__all__ += ['binomial_filter', 'n_poisson', 'n_neg_binomial']



def binomial_filter(prob, arr):
    """
    Binomial "filter" -- the same as n_binomial, except return
    the elements of arr that succeeded.

    Args:
        prob (float): probability of each trial succeeding
        arr (array): the array to be filtered

    Returns:
        Subset of array for which trials succeeded

    **Example**::

        inds = ss.binomial_filter(0.5, np.arange(20)**2) # Which values in the (arbitrary) array passed the coin flip
    """
    return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]



def n_poisson(rate, n):
    """
    An array of Poisson trials.

    Args:
        rate (float): the rate of the Poisson process (mean)
        n (int): number of trials

    **Example**::

        outcomes = ss.n_poisson(100, 20) # 20 Poisson trials with mean 100
    """
    return np.random.poisson(rate, n)


def n_neg_binomial(rate, dispersion, n, step=1):  # Numba not used due to incompatible implementation
    """
    An array of negative binomial trials. See ss.sample() for more explanation.

    Args:
        rate (float): the rate of the process (mean, same as Poisson)
        dispersion (float):  dispersion parameter; lower is more dispersion, i.e. 0 = infinite, ∞ = Poisson
        n (int): number of trials
        step (float): the step size to use if non-integer outputs are desired

    **Example**::

        outcomes = ss.n_neg_binomial(100, 1, 50) # 50 negative binomial trials with mean 100 and dispersion roughly equal to mean (large-mean limit)
        outcomes = ss.n_neg_binomial(1, 100, 20) # 20 negative binomial trials with mean 1 and dispersion still roughly equal to mean (approximately Poisson)
    """
    nbn_n = dispersion
    nbn_p = dispersion / (rate / step + dispersion)
    samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n) * step
    return samples


# %% Simple array operations

__all__ += ['true', 'false', 'defined', 'undefined']


def true(arr):
    """
    Returns the indices of the values of the array that are true: just an alias
    for arr.nonzero()[0].

    Args:
        arr (array): any array

    **Example**::

        inds = ss.true(np.array([1,0,0,1,1,0,1])) # Returns array([0, 3, 4, 6])
    """
    return arr.nonzero()[-1]


def false(arr):
    """
    Returns the indices of the values of the array that are false.

    Args:
        arr (array): any array

    **Example**::

        inds = ss.false(np.array([1,0,0,1,1,0,1]))
    """
    return np.logical_not(arr).nonzero()[-1]


def defined(arr):
    """
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = ss.defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    """
    return (~np.isnan(arr)).nonzero()[-1]


def undefined(arr):
    """
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = ss.defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    """
    return np.isnan(arr).nonzero()[-1]