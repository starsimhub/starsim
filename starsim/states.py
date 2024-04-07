"""
Define array-handling classes, including agent states
"""

import numpy as np
import sciris as sc
import starsim as ss
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

    __slots__ = ('values', 'uid')

    def __init__(self, values=None, uid=None):

        self.values = values
        self.uid = uid
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


    def __getitem__(self, key):
        out = UIDArray(values=self.values[key], uid=key)
        return out

    def __setitem__(self, key, value):
        """
        NB: the use of .__array__() calls is to access the array interface and thereby treat both np.ndarray and ArrayView instances
        in the same way without needing an additional type check. This is also why the UIDArray.dtype property is defined. Noting
        that for a State, the uid_map is a dynamic view attached to the People, but after an indexing operation, it will be a bare
        UIDArray that has an ordinary numpy array as the uid_map.
        """
        self.values[key] = value
        return

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
        return UIDArray(values=self.values.astype(*args, **kwargs), uid=self.uid)

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
                return UIDArray(values=out, uid=self.uid)

    def __array_wrap__(self, out_arr, context=None):
        # This allows numpy operations addition etc. to return instances of UIDArray
        if out_arr.ndim == 0:
            return out_arr.item()
        return UIDArray(values=out_arr, uid=self.uid) # Hardcoding class means State can inherit from UIDArray but return UIDArray base instances


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

    # def _trim(self, inds):
    #     """ Keep only specified indices """
    #     # Note that these are indices, not UIDs!
    #     n = len(inds)
    #     self._data[:n] = self._data[inds]
    #     self._data[n:self.n] = self.default
    #     self.n = n
    #     self._map_arrays()
    #     return

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
        return self._data[key]

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

    __slots__ = ('values', 'uid', 'default', 'name', 'label', '_data', 'values', '_initialized')

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

    # def _trim(self, inds):
    #     # Trim arrays to remove agents - should only be called via `People.remove()`
    #     self._data._trim(inds)
    #     self.values = self._data._view
    #     return

    def __getstate__(self):
        # When pickling, skip storing the `.values` attribute, which should be re-linked after unpickling
        return {k:getattr(self, k) for k in self.__slots__ if k != 'values'}

    def __setstate__(self, state):
        # When unpickling, re-link the `.values` attribute
        for k,v in state.items():
            setattr(self, k, v)
        self.values = self._data._view
        return