"""
Numerical utilities
"""

# %% Housekeeping

import warnings
import numpy as np
import sciris as sc
import stisim as ss
import numba as nb

# What functions are externally visible -- note, this gets populated in each section below
__all__ = []

# System constants
__all__ += ['INT_NAN']

INT_NAN = np.iinfo(np.int32).max  # Value to use to flag invalid content (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)


# %% Helper functions
__all__ += ['ndict', 'omerge', 'warn', 'unique', 'find_contacts', 'get_subclasses']


class ndict(sc.objdict):
    """
    A dictionary-like class that provides additional functionalities for handling named items.

    Args:
        name (str): The items' attribute to use as keys.
        type (type): The expected type of items.
        strict (bool): If True, only items with the specified attribute will be accepted.

    **Examples**::

        networks = ss.ndict(ss.simple_sexual(), ss.maternal())
        networks = ss.ndict([ss.simple_sexual(), ss.maternal()])
        networks = ss.ndict({'simple_sexual':ss.simple_sexual(), 'maternal':ss.maternal()})

    """

    def __init__(self, *args, name='name', type=None, strict=True, **kwargs):
        self.setattribute('_name', name)  # Since otherwise treated as keys
        self.setattribute('_type', type)
        self.setattribute('_strict', strict)
        self._initialize(*args, **kwargs)
        return
    
    def append(self, arg, key=None):
        valid = False
        if arg is None:
            return # Nothing to do
        elif hasattr(arg, self._name):
            key = key or getattr(arg, self._name)
            valid = True
        elif isinstance(arg, dict):
            if self._name in arg:
                key = key or arg[self._name]
                valid = True
            else:
                for k,v in arg.items():
                    self.append(v, key=k)
                valid = None # Skip final processing
        elif not self._strict:
            key = key or f'item{len(self)+1}'
            valid = True
        else:
            valid = False
        
        if valid is True:
            self._check_type(arg)
            self[key] = arg
        elif valid is None:
            pass # Nothing to do
        else:
            errormsg = f'Could not interpret argument {arg}: does not have expected attribute "{self._name}"'
            raise ValueError(errormsg)
            
        return
        
    def _check_type(self, arg):
        """ Check types """
        if self._type is not None:
            if not isinstance(arg, self._type):
                errormsg = f'The following item does not have the expected type {self._type}:\n{arg}'
                raise TypeError(errormsg)
        return
    
    def _initialize(self, *args, **kwargs):
        args = sc.mergelists(*args)
        for arg in args:
            self.append(arg)
        for key,arg in kwargs.items():
            self.append(arg, key=key)
        return
    
    def copy(self):
        new = self.__class__.__new__(name=self._name, type=self._type, strict=self._strict)
        new.update(self)
        return new
    
    def __add__(self, dict2):
        """ Allow c = a + b """
        new = self.copy()
        new.append(dict2)
        return new

    def __iadd__(self, dict2):
        """ Allow a += b """
        self.append(dict2)
        return self


def omerge(*args, **kwargs):
    """ Merge things into an objdict """
    return sc.objdict(sc.mergedicts(*args, **kwargs))


def warn(msg, category=None, verbose=None, die=None):
    """ Helper function to handle warnings -- not for the user """

    # Handle inputs
    warnopt = ss.options.warnings if not die else 'error'
    if category is None:
        category = RuntimeWarning
    if verbose is None:
        verbose = ss.options.verbose

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


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass

# %% Seed methods

__all__ += ['set_seed']


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

__all__ += ['binomial_arr', 'binomial_filter', 'n_multinomial', 'n_poisson', 'n_neg_binomial']


def binomial_arr(prob_arr):
    '''
    Binomial (Bernoulli) trials each with different probabilities.

    Args:
        prob_arr (array): array of probabilities

    Returns:
         Boolean array of which trials on the input array succeeded

    **Example**::

        outcomes = ss.binomial_arr([0.1, 0.1, 0.2, 0.2, 0.8, 0.8]) # Perform 6 trials with different probabilities
    '''
    return np.random.random(prob_arr.shape) < prob_arr


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


def n_multinomial(probs, n): # No speed gain from Numba
    '''
    An array of multinomial trials.

    Args:
        probs (array): probability of each outcome, which usually should sum to 1
        n (int): number of trials

    Returns:
        Array of integer outcomes

    **Example**::

        outcomes = hpv.n_multinomial(np.ones(6)/6.0, 50)+1 # Return 50 die-rolls
    '''
    return np.searchsorted(np.cumsum(probs), np.random.random(n))


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

@nb.njit
def _true(uids, values):
    """
    Returns the UIDs for indices where the value evaluates as True
    """
    out = np.empty(len(uids), dtype=uids.dtype)
    j = 0
    for i in range(len(values)):
        out[j] = uids[i]
        if values[i]:
            j += 1
    out = out[0:j]
    return out

@nb.njit
def _false(uids, values):
    """
    Returns the UIDs for indices where the value evaluates as False
    """
    out = np.empty(len(uids), dtype=uids.dtype)
    j = 0
    for i in range(len(values)):
        out[j] = uids[i]
        if not values[i]:
            j += 1
    out = out[0:j]
    return out


def true(state):
    """
    Returns the UIDs of the values of the array that are true

    Args:
        state (State, FusedArray)

    **Example**::

        inds = ss.true(people.alive) # Returns array of UIDs of alive agents
    """
    return _true(state.uid.__array__(), state.__array__())


def false(state):
    """
    Returns the indices of the values of the array that are false.

    Args:
        state (State, FusedArray)

    **Example**::

        inds = ss.false(people.alive) # Returns array of UIDs of dead agents
    """
    return _false(state.uid.__array__(), state.__array__())


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
