import pandas as pd
import numpy as np
import sciris as sc
import numba as nb
from numpy.lib.mixins import NDArrayOperatorsMixin # Inherit from this to automatically gain operators like +, -, ==, <, etc.

__all__ = ['State', 'DynamicView', 'INT_NAN']

INT_NAN = np.iinfo(int).max  # Value to use to flag removed UIDs (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)

class FusedArray(NDArrayOperatorsMixin):
    # This is a class that allows indexing by UID but does not support dynamic growth
    # It's kind of like a Pandas series but one that only supports a monotonically increasing
    # unique integer index, and that we can customize and optimize indexing for.
    #
    # We explictly do NOT support slicing (except for `[:]`), as these arrays are indexed by UID and slicing
    # by UID can be confusing/ambiguous when there are missing values. Indexing with a list/array returns
    # another FusedArray instance which enables chained filtering

    __slots__ = ('values','_uid_map', 'uid')

    def __init__(self, values, uid, uid_map=None):

        self.values = values
        self.uid = uid

        if uid_map is None and uid is not None:
            # Construct a local UID map as opposed to using a shared one (i.e., the one for all agents contained in the People instance)
            self.uid_map = np.full(np.max(uid) + 1, fill_value=INT_NAN, dtype=int)
            self.uid_map[uid] = np.arange(len(uid))
        else:
            self._uid_map = uid_map

    def __repr__(self):
        # TODO - optimize? Don't really need to create a dataframe just to print it, but on the other hand, it's fast enough and very easy
        df = pd.DataFrame(self.values.T, index=self.uid, columns=['Quantity'])
        df.index.name = 'UID'
        return df.__repr__()

    @property
    def dtype(self):
        return self.values.dtype

    @staticmethod
    @nb.njit
    def _get_vals_uids(vals, key, uid_map):
        """
        Extract valued from a collection of UIDs

        :param vals: A 1D np.ndarray containing the values
        :param key: A 1D np.ndnarray of integers containing the UIDs to query
        :param uid_map: A 1D np.ndarray of integers mapping UID to array position in ``vals``
        :return: A tuple of (values, uids, new_uid_map) suitable for passing into the FusedArray constructor
        """
        out = np.empty(len(key), dtype=vals.dtype)
        new_uid_map = np.full(uid_map.shape[0], fill_value=INT_NAN, dtype=np.int64)
        for i in range(len(key)):
            out[i] = vals[uid_map[key[i]]]
            new_uid_map[key[i]] = i
        return out, key, new_uid_map

    @staticmethod
    @nb.njit
    def _set_vals_uids_multiple(vals, key, uid_map, value):
        """
        Insert an array of values based on UID

        :param vals: A reference to a 1D np.ndarray in which to insert the values
        :param key: A 1D np.ndnarray of integers containing the UIDs to add values for
        :param uid_map:  A 1D np.ndarray of integers mapping UID to array position in ``vals``
        :param value: A 1D np.ndarray the same length as ``key`` containing values to insert
        :return:
        """
        for i in range(len(key)):
            vals[uid_map[key[i]]] = value[i]

    @staticmethod
    @nb.njit
    def _set_vals_uids_single(vals, key, uid_map, value):
        """
        Insert a single value into multiple UIDs

        :param vals: A reference to a 1D np.ndarray in which to insert the values
        :param key: A 1D np.ndnarray of integers containing the UIDs to add values for
        :param uid_map:  A 1D np.ndarray of integers mapping UID to array position in ``vals``
        :param value: A scalar value to insert at every position specified by ``key``
        :return:
        """
        for i in range(len(key)):
            vals[uid_map[key[i]]] = value

    def __getitem__(self, key):
        try:
            if isinstance(key, (int, np.integer)):
                # Handle getting a single item by UID
                return self.values[self._uid_map[key]]
            elif isinstance(key, (np.ndarray, FusedArray)):
                if key.dtype.kind == 'b':
                    # Handle accessing items with a logical array. Surprisingly, it seems faster to use nonzero() to convert
                    # it to indices first. Also, the pure Python implementation is difficult to improve upon using numba
                    mapped_key = key.__array__().nonzero()[0]
                    uids = self.uid.__array__()[mapped_key]
                    new_uid_map = np.full(len(self._uid_map), fill_value=INT_NAN, dtype=int)
                    new_uid_map[uids] = np.arange(len(uids))
                    values = self.values[mapped_key]
                else:
                    # Access items by an array of integers. We do get a decent performance boost from using numba here
                    values, uids, new_uid_map = self._get_vals_uids(self.values, key, self._uid_map.__array__())
            elif isinstance(key, slice):
                if key.start is None and key.stop is None and key.step is None:
                    return sc.dcp(self)
                else:
                    raise Exception('Slicing not supported - slice the .values attribute by index instead e.g., x.values[0:5], not x[0:5]')
            else:
                # This branch is specifically handling the user passing in a list of integers instead of an array, therefore
                # we need an additional conversion to an array first using np.fromiter to improve numba performance
                values, uids, new_uid_map = self._get_vals_uids(self.values, np.fromiter(key, dtype=int), self._uid_map.__array__())
            return FusedArray(values=values, uid=uids, uid_map=new_uid_map)
        except IndexError as e:
            if str(INT_NAN) in str(e):
                raise IndexError(f'UID not present in array')
            else:
                raise e

    def __setitem__(self, key, value):
        # nb. the use of .__array__() calls is to access the array interface and thereby treat both np.ndarray and DynamicView instances
        # in the same way without needing an additional type check. This is also why the FusedArray.dtype property is defined. Noting
        # that for a State, the uid_map is a dynamic view attached to the People, but after an indexing operation, it will be a bare
        # FusedArray that has an ordinary numpy array as the uid_map
        try:
            if isinstance(key, (int, np.integer)):
                return self.values.__setitem__(self._uid_map[key], value)
            elif isinstance(key, (np.ndarray, FusedArray)):
                if key.dtype.kind == 'b':
                    self.values.__setitem__(key.__array__().nonzero()[0], value)
                else:
                    if isinstance(value, (np.ndarray, FusedArray)):
                        return self._set_vals_uids_multiple(self.values, key, self._uid_map.__array__(), value.__array__())
                    else:
                        return self._set_vals_uids_single(self.values, key, self._uid_map.__array__(), value)
            elif isinstance(key, slice):
                if key.start is None and key.stop is None and key.step is None:
                    return self.values.__setitem__(key, value)
                else:
                    raise Exception('Slicing not supported - slice the .values attribute by index instead e.g., x.values[0:5], not x[0:5]')
            else:
                if isinstance(value, (np.ndarray, FusedArray)):
                    return self._set_vals_uids_multiple(self.values, np.fromiter(key, dtype=int), self._uid_map.__array__(), value.__array__())
                else:
                    return self._set_vals_uids_single(self.values, np.fromiter(key, dtype=int), self._uid_map.__array__(), value)
        except IndexError as e:
            if str(INT_NAN) in str(e):
                raise IndexError(f'UID not present in array')
            else:
                raise e

    # Make it behave like a regular array mostly
    def __len__(self):
        return len(self.values)

    def __contains__(self, *args, **kwargs):
        return self.values.__contains__(*args, **kwargs)

    def astype(self, *args, **kwargs):
        return self.values.astype(*args, **kwargs)

    @property
    def shape(self):
        return self.values.shape

    @property
    def __array_interface__(self):
        return self.values.__array_interface__

    def __array__(self):
        return self.values

    def __array_ufunc__(self, *args, **kwargs):
        if args[1] != '__call__':
            # This is a generic catch-all for ufuncs that are not being applied with '__call__' (e.g., operations returning a scalar like 'np.sum()' use reduce instead)
            args = [(x if x is not self else self.values) for x in args]
            kwargs = {k: v if v is not self else self.values for k, v in kwargs.items()}
            return self.values.__array_ufunc__(*args, **kwargs)

        args = [(x if x is not self else self.values) for x in args]
        if 'out' in kwargs and kwargs['out'][0] is self:
            # In-place operations like += use this branch
            kwargs['out'] = self.values
            args[0](*args[2:], **kwargs)
            return self
        else:
            out = args[0](*args[2:], **kwargs)
            if isinstance(out, FusedArray):
                # For some operations (e.g., those involving two FusedArrays) the result of the ufunc will already be a FusedArray
                # In particular, operating on two states will result in a FusedArray where the references to the original People uid_map and uids
                # are still intact. In such cases, we can return the resulting FusedArray directly
                return out
            else:
                # Otherwise, if the result of the ufunc is an array (e.g., because one of the arguments was an array) then
                # we need to wrap it up in a new FusedArray. With '__call__' the dimensions should hopefully be the same and we should
                # be able to reuse the UID arrays directly
                return FusedArray(values=out, uid=self.uid, uid_map=self._uid_map)

    def __array_wrap__(self, out_arr, context=None):
        # This allows numpy operations addition etc. to return instances of FusedArray
        if out_arr.ndim == 0:
            return out_arr.item()
        return FusedArray(values=out_arr, uid=self.uid, uid_map=self._uid_map) # Hardcoding class means State can inherit from FusedArray but return FusedArray base instances


class DynamicView(NDArrayOperatorsMixin):
    def __init__(self, dtype, fill_value=0, label=None):
        """
        Args:
            name: name of the result as used in the model
            dtype: datatype
            fill_value: default value for this state upon model initialization. If callable, it is called with a single argument for the number of samples
            shape: If not none, set to match a string in `pars` containing the dimensionality
            label: text used to construct labels for the result for displaying on plots and other outputs
        """
        self.dtype = dtype
        self.fill_value = fill_value

        self.n = 0  # Number of agents currently in use

        self._data = None  # The underlying memory array (length at least equal to n)
        self._view = None  # The view corresponding to what is actually accessible (length equal to n)
        return

    @property
    def _s(self):
        # Return the size of the underlying array (maximum number of agents that can be stored without reallocation)
        return len(self._data)

    def __len__(self):
        # Return the number of active elements
        return self.n

    def __repr__(self):
        # Print out the numpy view directly
        return self._view.__repr__()

    def _new_items(self, n):
        # Create new arrays of the correct dtype and fill them based on the (optionally callable) fill value
        if callable(self.fill_value):
            new = np.empty(n, dtype=self.dtype)
            new[:] = self.fill_value(n)
        else:
            new = np.full(n, dtype=self.dtype, fill_value=self.fill_value)
        return new

    def initialize(self, n):
        self._data = self._new_items(n)
        self.n = n
        self._map_arrays()

    def grow(self, n):

        if self.n + n > self._s:
            n_new = max(n, int(self._s / 2))  # Minimum 50% growth
            self._data = np.concatenate([self._data, self._new_items(n)], axis=0)

        self.n += n
        self._map_arrays()

    def _trim(self, inds):
        # Keep only specified indices
        # Note that these are indices, not UIDs!
        n = len(inds)
        self._data[:n] = self._data[inds]
        self._data[n:self.n] = self.fill_value(self.n-n) if callable(self.fill_value) else self.fill_value
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


class State(FusedArray):

    def __init__(self, name, dtype, fill_value=0, label=None):
        super().__init__(values=None, uid=None, uid_map=None) # Call the FusedArray constructor
        self._data = DynamicView(dtype=dtype, fill_value=fill_value)
        self.name = name
        self.label = label or name
        self.values = self._data._view
        self._initialized = False

    def __repr__(self):
        if not self._initialized:
            return f'<State {self.name} (uninitialized)>'
        else:
            return FusedArray.__repr__(self)

    def initialize(self, people):
        if self._initialized:
            return

        people.add_state(self)
        self._uid_map = people._uid_map
        self.uid = people.uid
        self._data.initialize(len(self.uid))
        self.values = self._data._view
        self._initialized = True

    def grow(self, n):
        self._data.grow(n)
        self.values = self._data._view

    def _trim(self, inds):
        self._data._trim(inds)
        self.values = self._data._view

