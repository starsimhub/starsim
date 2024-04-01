"""
Define array-handling classes, including agent states
"""

import numpy as np
import sciris as sc
import numba as nb
import starsim as ss
from starsim.settings import INT_NAN
from starsim.settings import dtypes as sdt
from numpy.lib.mixins import NDArrayOperatorsMixin  # Inherit from this to automatically gain operators like +, -, ==, <, etc.


__all__ = ['check_dtype', 'UIDArray', 'State', 'ArrayView']


def check_dtype(dtype, default=None):
    """ Check that the supplied dtype is one of the supported options """
    
    # Handle dtype
    if dtype is None:
        if default is None:
            errormsg = 'Must supply either a dtype or a default value'
            raise ValueError(errormsg)
        else:
            dtype = type(default)
    
    if dtype in ['float', float, np.float64, np.float32]:
        dtype = sdt.float
    elif dtype in ['int', int, np.int64, np.int32]:
        dtype = sdt.int
    elif dtype in ['bool', bool, np.bool_]:
        dtype = sdt.bool
    else:
        warnmsg = f'Data type {type(default)} not a supported data type; set warn=False to suppress warning'
        ss.warn(warnmsg)
    
    return dtype


class UIDArray(NDArrayOperatorsMixin):
    """
    This is a class that allows indexing by UID but does not support dynamic growth
    It's kind of like a Pandas series but one that only supports a monotonically increasing
    unique integer index, and that we can customize and optimize indexing for.
    
    We explictly do NOT support slicing (except for `[:]`), as these arrays are indexed by UID and slicing
    by UID can be confusing/ambiguous when there are missing values. Indexing with a list/array returns
    another UIDArray instance which enables chained filtering
    """

    __slots__ = ('values', '_uid_map', 'uid')

    def __init__(self, values=None, uid=None, uid_map=None):

        self.values = values
        self.uid = uid

        if uid_map is None and uid is not None:
            # Construct a local UID map as opposed to using a shared one (i.e., the one for all agents contained in the People instance)
            self._uid_map = np.full(np.max(uid) + 1, fill_value=ss.INT_NAN, dtype=sdt.int)
            self._uid_map[uid] = np.arange(len(uid))
        else:
            self._uid_map = uid_map
        return

    def __repr__(self):
        if len(self) == 0:
            return f'<{self.__class__.__name__} (empty)>'
        df = self.to_df()
        return df.__repr__()
    
    def to_df(self):
        """ Convert to a dataframe """
        df = sc.dataframe({'Quantity':self.values.T}, index=self.uid)
        df.index.name = 'UID'
        return df

    @property
    def dtype(self):
        return self.values.dtype

    @staticmethod
    @nb.njit(cache=True)
    def _get_vals_uids(vals, key, uid_map):
        """
        Extract values from a collection of UIDs

        This function is used to retrieve values based on UID. As indexing a UIDArray returns a new UIDArray,
        this method also populates the new UID map for use in the subsequently created UIDArray, avoiding the
        need to re-compute it separately.

        Args:
            vals: A 1D np.ndarray containing the values
            key: A 1D np.ndnarray of integers containing the UIDs to query
            uid_map: A 1D np.ndarray of integers mapping UID to array position in ``vals``
        
        Returns:
            A tuple of (values, uids, new_uid_map) suitable for passing into the UIDArray constructor
        """
        out = np.empty(len(key), dtype=vals.dtype)
        new_uid_map = np.full(uid_map.shape[0], fill_value=INT_NAN, dtype=np.int64)

        for i,kv in enumerate(key):
            idx = uid_map[kv]
            if idx == INT_NAN:
                raise IndexError('UID not present in array')
            out[i] = vals[idx]
            new_uid_map[kv] = i
        return out, key, new_uid_map
    
    @staticmethod
    @nb.njit(cache=True, parallel=True)
    def _get_vals_uids_par(vals, key, uid_map):
        """
        Parallelized version of _get_vals_uids() -- need both because this is much slower for small arrays
        """
        len_key = len(key)
        out = np.empty(len_key, dtype=vals.dtype)
        new_uid_map = np.full(uid_map.shape[0], fill_value=INT_NAN, dtype=np.int64)

        for i in nb.prange(len_key):
            kv = key[i]
            idx = uid_map[kv]
            out[i] = vals[idx]
            new_uid_map[kv] = i
        return out, key, new_uid_map

    @staticmethod
    @nb.njit(cache=True)
    def _set_vals_uids_multiple(vals, key, uid_map, value):
        """
        Insert an array of values based on UID

        Args:
            vals: A reference to a 1D np.ndarray in which to insert the values
            key: A 1D np.ndnarray of integers containing the UIDs to add values for
            uid_map:  A 1D np.ndarray of integers mapping UID to array position in ``vals``
            value: A 1D np.ndarray the same length as ``key`` containing values to insert
        """

        for i,kv in enumerate(key):
            if kv >= len(uid_map):
                errormsg = f'UID not present in array (requested UID ({key[i]}) is larger than the maximum UID in use ({len(uid_map)}))'
                raise IndexError(errormsg)
            idx = uid_map[kv]
            if idx == INT_NAN:
                raise IndexError('UID not present in array')
            elif idx >= len(vals):
                errormsg = f'Attempted to write to a non-existant index ({idx}) - this can happen if attempting to write to new entries that have not yet been allocated via grow()'
                raise IndexError(errormsg)
            vals[idx] = value[i]
        return

    @staticmethod
    @nb.njit(cache=True)
    def _set_vals_uids_single(vals, key, uid_map, value):
        """
        Insert a single value into multiple UIDs

        Args:
            vals: A reference to a 1D np.ndarray in which to insert the values
            key: A 1D np.ndnarray of integers containing the UIDs to add values for
            uid_map:  A 1D np.ndarray of integers mapping UID to array position in ``vals``
            value: A scalar value to insert at every position specified by ``key``
        """
        for i,kv in enumerate(key):
            if kv >= len(uid_map):
                raise IndexError('UID not present in array (requested UID is larger than the maximum UID in use)')
            idx = uid_map[kv]
            if idx == INT_NAN:
                raise IndexError('UID not present in array')
            elif idx >= len(vals):
                raise Exception('Attempted to write to a non-existant index - this can happen if attempting to write to new entries that have not yet been allocated via grow()')
            vals[idx] = value

    def __getitem__(self, key):
        if np.iterable(key) and len(key) > 10_000: # Approximate cutoff for when the overhead becomes worthwhile
            gvu_func = self._get_vals_uids_par
        else:
            gvu_func = self._get_vals_uids
            
        try:
            if isinstance(key, (int, np.integer)):
                # Handle getting a single item by UID
                return self.values[self._uid_map[key]]
            elif isinstance(key, (np.ndarray, UIDArray, ArrayView)):
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
                    values, uids, new_uid_map = gvu_func(self.values, key.__array__(), self._uid_map.__array__())
            elif isinstance(key, slice):
                if key.start is None and key.stop is None and key.step is None:
                    return sc.dcp(self)
                else:
                    raise Exception('Slicing not supported - slice the .values attribute by index instead e.g., x.values[0:5], not x[0:5]')
            else:
                # This branch is specifically handling the user passing in a list of integers instead of an array, therefore
                # we need an additional conversion to an array first using np.fromiter to improve numba performance
                values, uids, new_uid_map = gvu_func(self.values, np.fromiter(key, dtype=int), self._uid_map.__array__())
            return UIDArray(values=values, uid=uids, uid_map=new_uid_map)
        except IndexError as e:
            if str(INT_NAN) in str(e):
                raise IndexError(f'UID {key} not present in array')
            else:
                raise e

    def __setitem__(self, key, value):
        """
        NB: the use of .__array__() calls is to access the array interface and thereby treat both np.ndarray and ArrayView instances
        in the same way without needing an additional type check. This is also why the UIDArray.dtype property is defined. Noting
        that for a State, the uid_map is a dynamic view attached to the People, but after an indexing operation, it will be a bare
        UIDArray that has an ordinary numpy array as the uid_map.
        """
        try:
            if isinstance(key, (int, np.integer)):
                return self.values.__setitem__(self._uid_map[key], value)
            elif isinstance(key, (np.ndarray, UIDArray)):
                if key.dtype.kind == 'b':
                    self.values.__setitem__(key.__array__().nonzero()[0], value)
                else:
                    if isinstance(value, (np.ndarray, UIDArray)):
                        return self._set_vals_uids_multiple(self.values, key, self._uid_map.__array__(), value.__array__())
                    else:
                        return self._set_vals_uids_single(self.values, key, self._uid_map.__array__(), value)
            elif isinstance(key, slice):
                if key.start is None and key.stop is None and key.step is None:
                    return self.values.__setitem__(key, value)
                else:
                    raise Exception('Slicing not supported - slice the .values attribute by index instead e.g., x.values[0:5], not x[0:5]')
            else:
                if isinstance(value, (np.ndarray, UIDArray)):
                    return self._set_vals_uids_multiple(self.values, np.fromiter(key, dtype=int), self._uid_map.__array__(), value.__array__())
                else:
                    return self._set_vals_uids_single(self.values, np.fromiter(key, dtype=int), self._uid_map.__array__(), value)
        except IndexError as e:
            if str(INT_NAN) in str(e):
                raise IndexError(f'UID {key} not present in array')
            else:
                raise e

    def __getattr__(self, attr):
        """ Make it behave like a regular array mostly -- enables things like sum(), mean(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self.values, attr)

    # Make it behave like a regular array mostly
    def __len__(self):
        return len(self.values)

    def __contains__(self, *args, **kwargs):
        return self.values.__contains__(*args, **kwargs)

    def astype(self, *args, **kwargs):
        return UIDArray(values=self.values.astype(*args, **kwargs), uid=self.uid, uid_map=self._uid_map)

    def sum(self, *args, **kwargs):
        return self.values.sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.values.mean(*args, **kwargs)

    def any(self, *args, **kwargs):
        return self.values.any(*args, **kwargs)

    def all(self, *args, **kwargs):
        return self.values.all(*args, **kwargs)

    def count(self, *args, **kwargs):
        return np.count_nonzero(self.values, *args, **kwargs)

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
            if isinstance(out, UIDArray):
                # For some operations (e.g., those involving two UIDArrays) the result of the ufunc will already be a UIDArray
                # In particular, operating on two states will result in a UIDArray where the references to the original People uid_map and uids
                # are still intact. In such cases, we can return the resulting UIDArray directly
                return out
            else:
                # Otherwise, if the result of the ufunc is an array (e.g., because one of the arguments was an array) then
                # we need to wrap it up in a new UIDArray. With '__call__' the dimensions should hopefully be the same and we should
                # be able to reuse the UID arrays directly
                return UIDArray(values=out, uid=self.uid, uid_map=self._uid_map)

    def __array_wrap__(self, out_arr, context=None):
        # This allows numpy operations addition etc. to return instances of UIDArray
        if out_arr.ndim == 0:
            return out_arr.item()
        return UIDArray(values=out_arr, uid=self.uid, uid_map=self._uid_map) # Hardcoding class means State can inherit from UIDArray but return UIDArray base instances


class ArrayView(NDArrayOperatorsMixin):

    __slots__ = ('_data', '_view', 'n', 'default')

    def __init__(self, dtype, default=None, coerce=True):
        """
        Args:
            dtype (class): The dtype to use for this instance (if None, infer from value)
            default (any): Specify default value for new agents. This can be
            coerce (bool): Whether to ensure the the data is one of the supported data types
        """
        if coerce:
            dtype = check_dtype(dtype, default)
        self.default = default if default is not None else dtype()
        self.n = 0  # Number of agents currently in use
        self._data = np.empty(0, dtype=dtype)  # The underlying memory array (length at least equal to n)
        self._view = None  # The view corresponding to what is actually accessible (length equal to n)
        self._map_arrays()
        return

    @property
    def _s(self):
        """ Return the size of the underlying array (maximum number of agents that can be stored without reallocation) """
        return len(self._data)

    @property
    def dtype(self):
        """
        The specified dtype and the underlying array dtype can be different. For instance, the user might pass in
        ArrayView(dtype=int) but the underlying array's dtype will be np.dtype('int32'). This distinction is important
        because the numpy dtype has attributes like 'kind' that the input dtype may not have. We need the ArrayView's
        dtype to match that of the underlying array so that it can be more seamlessly exchanged with direct numpy arrays
        Therefore, we retain the original dtype in ArrayView._dtype() and use
        """
        return self._data.dtype

    def __len__(self):
        """ Return the number of active elements """
        return self.n

    def __repr__(self):
        """ Print out the numpy view directly """
        return self._view.__repr__()

    def grow(self, n):
        """ If the total number of agents exceeds the array size, extend the underlying arrays """
        if self.n + n > self._s:
            n_new = max(n, int(self._s / 2))  # Minimum 50% growth
            self._data = np.concatenate([self._data, np.full(n_new, dtype=self.dtype, fill_value=self.default)], axis=0)
        self.n += n  # Increase the count of the number of agents by `n` (the requested number of new agents)
        self._map_arrays()
        return

    def _trim(self, inds):
        """ Keep only specified indices """
        # Note that these are indices, not UIDs!
        n = len(inds)
        self._data[:n] = self._data[inds]
        self._data[n:self.n] = self.default
        self.n = n
        self._map_arrays()
        return

    def __getstate__(self):
        """ When pickling, skip storing the `._view` attribute, which should be re-linked after unpickling """
        return {k:getattr(self, k) for k in self.__slots__ if k != '_view'}

    def __setstate__(self, state):
        """ Re-map arrays after unpickling so that `.view` is a reference to the correct array in-memory """
        for k,v in state.items():
            setattr(self, k, v)
        self._map_arrays()
        return

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
        
    def __getattr__(self, attr):
        """ Make it behave like a regular array mostly -- enables things like sum(), mean(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self._view, attr)

    @property
    def __array_interface__(self):
        return self._view.__array_interface__

    def __array__(self):
        return self._view

    def __array_ufunc__(self, *args, **kwargs):
        args = [(x if x is not self else self._view) for x in args]
        kwargs = {k: v if v is not self else self._view for k, v in kwargs.items()}
        return self._view.__array_ufunc__(*args, **kwargs)


class State(UIDArray):

    __slots__ = ('values', '_uid_map', 'uid', 'default', 'name', 'label', '_data', 'values', '_initialized')

    def __init__(self, name, dtype=None, default=None, label=None, coerce=True):
        """
        Store a state of the agents (e.g. age, infection status, etc.)

        Args: 
            name (str): The name for the state (also used as the dictionary key, so should not have spaces etc.)
            dtype (class): The dtype to use for this instance (if None, infer from value)
            default (any): Specify default value for new agents. This can be
            - A scalar with the same dtype (or castable to the same dtype) as the State
            - A callable, with a single argument for the number of values to produce
            - A ``ss.Dist`` instance
            label (str): The human-readable name for the state
            coerce (bool): Whether to ensure the the data is one of the supported data types
        """
        super().__init__()  # Call the UIDArray constructor
        
        if coerce:
            dtype = check_dtype(dtype, default)
        
        if default is None:
            default = dtype()
        
        # Set attributes
        self.default = default
        self.name = name
        self.label = label or name
        self._data = ArrayView(dtype=dtype)
        self.values = self._data._view
        self._initialized = False
        return

    def __repr__(self):
        if not self._initialized:
            return f'<State {self.name} (uninitialized)>'
        else:
            return UIDArray.__repr__(self)

    def _new_vals(self, uids):
        if isinstance(self.default, ss.Dist):
            new_vals = self.default.rvs(uids)
        elif callable(self.default):
            new_vals = self.default(len(uids))
        else:
            new_vals = self.default
        return new_vals

    def initialize(self, sim):
        """
        Initialize state

        This method should be called as part of initialization of the parent class containing the state -
        specifically, `People.initialize()` and `Module.initialize()`. Initialization of a State object
        involves two processes:

        - Converting any distribution objects to a Dist instance and linking it to RNGs stored in a `Sim`
        - Establishing a bidirectional connection with a `People` object for the purpose of UID indexing and resizing

        Since State objects can be stored in `People` or in a `Module` and the collection of all states in a `Sim` should
        be connected to RNGs within that same `Sim`, the states must necessarily be linked to the same `People` object that
        is inside a `Sim`. Initializing States outside of a `Sim` is not possible because of this RNG dependency, particularly
        because the states in a `People` object cannot be initialized without a `Sim` and therefore it would not be possible to
        have an initialized `People` object outside of a `Sim`.
        
        Args:
            sim: A `Sim` instance that contains an initialized `People` object
        """

        if self._initialized:
            return

        people = sim.people
        assert people.initialized, 'People must be initialized before initializing states'
        
        # Connect any distributions in the default to RNGs in the Sim
        if isinstance(self.default, ss.Dist):
            self.default.initialize(module=self, sim=sim)

        # Establish connection with the People object
        people.register_state(self)
        self._uid_map = people._uid_map
        self.uid = people.uid

        # Populate initial values
        self._data.grow(len(self.uid))
        self._data[:len(self.uid)] = self._new_vals(self.uid)
        self.values = self._data._view

        self._initialized = True
        return

    def grow(self, uids):
        """
        Add state for new agents

        This method is normally only called via `People.grow()`.

        Args:
            uids: Numpy array of UIDs for the new agents being added This array should have length n
        """

        n = len(uids)
        self._data.grow(n)
        self.values = self._data._view
        self._data[-n:] = self._new_vals(uids)
        return

    def _trim(self, inds):
        # Trim arrays to remove agents - should only be called via `People.remove()`
        self._data._trim(inds)
        self.values = self._data._view
        return

    def __getstate__(self):
        # When pickling, skip storing the `.values` attribute, which should be re-linked after unpickling
        return {k:getattr(self, k) for k in self.__slots__ if k != 'values'}

    def __setstate__(self, state):
        # When unpickling, re-link the `.values` attribute
        for k,v in state.items():
            setattr(self, k, v)
        self.values = self._data._view
        return