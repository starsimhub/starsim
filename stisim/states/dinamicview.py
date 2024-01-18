import pandas as pd
import numpy as np
import sciris as sc
import numba as nb
from stisim.utils.ndict import INT_NAN
from stisim.distributions import ScipyDistribution
from stisim.utils.ndict import warn
from numpy.lib.mixins import NDArrayOperatorsMixin  # Inherit from this to automatically gain operators like +, -, ==, <, etc.
from scipy.stats._distn_infrastructure import rv_frozen

__all__ = ['DynamicView']

class DynamicView(NDArrayOperatorsMixin):
    def __init__(self, dtype, fill_value=None):
        """
        Args:
            name: name of the result as used in the model
            dtype: datatype
            fill_value: default value for this state upon model initialization. If not provided, it will use the default value for the dtype
            shape: If not none, set to match a string in `pars` containing the dimensionality
            label: text used to construct labels for the result for displaying on plots and other outputs
        """
        self.fill_value = fill_value if fill_value is not None else dtype()
        self.n = 0  # Number of agents currently in use
        self._data = np.empty(0, dtype=dtype)  # The underlying memory array (length at least equal to n)
        self._view = None  # The view corresponding to what is actually accessible (length equal to n)
        self._map_arrays()
        return

    @property
    def _s(self):
        # Return the size of the underlying array (maximum number of agents that can be stored without reallocation)
        return len(self._data)

    @property
    def dtype(self):
        # The specified dtype and the underlying array dtype can be different. For instance, the user might pass in
        # DynamicView(dtype=int) but the underlying array's dtype will be np.dtype('int32'). This distinction is important
        # because the numpy dtype has attributes like 'kind' that the input dtype may not have. We need the DynamicView's
        # dtype to match that of the underlying array so that it can be more seamlessly exchanged with direct numpy arrays
        # Therefore, we retain the original dtype in DynamicView._dtype() and use
        return self._data.dtype

    def __len__(self):
        # Return the number of active elements
        return self.n

    def __repr__(self):
        # Print out the numpy view directly
        return self._view.__repr__()

    def grow(self, n):
        # If the total number of agents exceeds the array size, extend the underlying arrays
        if self.n + n > self._s:
            n_new = max(n, int(self._s / 2))  # Minimum 50% growth
            self._data = np.concatenate([self._data, np.full(n_new, dtype=self.dtype, fill_value=self.fill_value)], axis=0)
        self.n += n  # Increase the count of the number of agents by `n` (the requested number of new agents)
        self._map_arrays()

    def _trim(self, inds):
        # Keep only specified indices
        # Note that these are indices, not UIDs!
        n = len(inds)
        self._data[:n] = self._data[inds]
        self._data[n:self.n] = self.fill_value
        self.n = n
        self._map_arrays()

    def _map_arrays(self):
        """
        Set main simulation attributes to be views of the underlying data

        This method should be called whenever the number of agents required changes
        (regardless of whether or not the underlying arrays have been resized)
        """
        self._view = self._data[:self.n]

    def __getitem__(self, key):
        return self._view.__getitem__(key)

    def __setitem__(self, key, value):
        self._view.__setitem__(key, value)

    @property
    def __array_interface__(self):
        return self._view.__array_interface__

    def __array__(self):
        return self._view

    def __array_ufunc__(self, *args, **kwargs):
        args = [(x if x is not self else self._view) for x in args]
        kwargs = {k: v if v is not self else self._view for k, v in kwargs.items()}
        return self._view.__array_ufunc__(*args, **kwargs)
