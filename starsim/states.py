"""
Define array-handling classes, including agent states
"""

import numpy as np
import starsim as ss

# Shorten these for performance
ss_float = ss.dtypes.float
ss_int   = ss.dtypes.int
ss_bool  = ss.dtypes.bool

__all__ = ['check_dtype', 'Arr', 'FloatArr', 'IntArr', 'BoolArr', 'IndexArr', 'uids']


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
        dtype = ss_float
    elif dtype in ['int', int, np.int64, np.int32]:
        dtype = ss_int
    elif dtype in ['bool', bool, np.bool_]:
        dtype = ss_bool
    else:
        warnmsg = f'Data type {type(default)} not a supported data type; set warn=False to suppress warning'
        ss.warn(warnmsg)
    
    return dtype


class Arr(np.lib.mixins.NDArrayOperatorsMixin):

    # __slots__ = ('values', 'uid', 'default', 'name', 'label', 'raw', 'values', 'initialized') # TODO: reinstate for speed later

    def __init__(self, name, dtype=None, default=None, nan=None, label=None, coerce=True, skip_init=False):
        """
        Store a state of the agents (e.g. age, infection status, etc.) as an array

        Args: 
            name (str): The name for the state (also used as the dictionary key, so should not have spaces etc.)
            dtype (class): The dtype to use for this instance (if None, infer from value)
            default (any): Specify default value for new agents. This can be
            - A scalar with the same dtype (or castable to the same dtype) as the State
            - A callable, with a single argument for the number of values to produce
            - A ``ss.Dist`` instance
            nan (any): the value to use to represent NaN (not a number); also used as the default value if not supplied
            label (str): The human-readable name for the state
            coerce (bool): Whether to ensure the the data is one of the supported data types
            skip_init (bool): Whether to skip initialization with the People object (used for uid and slot states)
        """
        if coerce:
            dtype = check_dtype(dtype, default)
        
        # Set attributes
        self.name = name
        self.label = label or name
        self.default = default
        self.nan = nan
        self.dtype = dtype
        
        # Properties that are initialized later
        self.raw = np.empty(0, dtype=dtype)
        self.people = None
        self.len_used = 0
        self.len_tot = 0
        self.initialized = skip_init
        return
    
    def __repr__(self):
        arr_str = np.array2string(self.values, max_line_width=200)
        string = f'<{self.__class__.__name__} "{str(self.name)}", len={len(self)}, {arr_str}>'
        return string
    
    def __len__(self):
        try:
            return len(self.aliveinds)
        except:
            return len(self.raw)
    
    def __getitem__(self, key):
        if isinstance(key, uids): # Check that it's UIDs
            return self.raw[key]
        elif isinstance(key, np.ndarray) and key.dtype == np.int64:
            errormsg = f'Indexing an Arr ({self.name}) by an int array ({key}) is not allowed. Use ss.uids()` instead.'
            raise Exception(errormsg)
        else:
            return self.values[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, np.ndarray) and key.dtype == ss_int:
            self.raw[key] = value
        else:
            newkey = self.aliveinds[key]
            self.raw[newkey] = value
            
    def __getattr__(self, attr):
        """ Make it behave like a regular array mostly -- enables things like sum(), mean(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self.values, attr)
        
    def dup(self, arr=None):
        """ Duplicate, optionally resetting the array """
        return arr
        # new = object.__new__(self.__class__) # Create a new Arr instance
        # new.__dict__ = self.__dict__.copy() # Copy pointers
        # if arr is not None:
        #     new.raw = arr
        # return new
    
    def __gt__(self, other): return self.notnan(self.values > other)
    def __lt__(self, other): return self.notnan(self.values < other)
    def __ge__(self, other): return self.notnan(self.values >= other)
    def __le__(self, other): return self.notnan(self.values <= other)
    def __eq__(self, other): return self.notnan(self.values == other)
    def __ne__(self, other): return self.notnan(self.values != other)
    
    def __and__(self, other): raise BooleanOperationError(self)
    def __or__(self, other):  raise BooleanOperationError(self)
    def __xor__(self, other): raise BooleanOperationError(self)
    def __invert__(self):     raise BooleanOperationError(self)
    
    # The mixin class delegates the operations to the corresponding numpy functions
    def _raway_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [x.values if isinstance(x, Arr) else x for x in inputs]
        return getattr(ufunc, method)(*inputs, **kwargs)
    
    def count(self):
        return np.count_nonzero(self.values)

    @property
    def values(self):
        return self.raw[self.aliveinds] # TODO: think about if this makes sense for uids
    
    @property
    def aliveinds(self):
        try:
            return self.people.aliveinds
        except:
            print('TEMP: Could not return aliveinds!')
            return np.arange(len(self.raw))
        
    def isnan(self):
        return self.values == self.nan

    def notnan(self, mask=None):
        valid = self.values != self.nan
        if mask is not None:
            valid = valid*mask
        return valid

    def set_new(self, uids, new_vals=None):
        if new_vals is None: 
            if isinstance(self.default, ss.Dist):
                new_vals = self.default.rvs(uids)
            elif callable(self.default):
                new_vals = self.default(len(uids))
            elif self.default is not None:
                new_vals = self.default
            else:
                new_vals = self.nan
        self.raw[uids] = new_vals
        return new_vals
    
    def set_nan(self, uids):
        self.raw[uids] = self.nan
        return
    
    def grow(self, new_uids=None, new_vals=None):
        """
        Add new agents to an Arr

        This method is normally only called via `People.grow()`.

        Args:
            new_uids: Numpy array of UIDs for the new agents being added
            new_vals: If provided, assign these state values to the new UIDs
        """
        if new_uids is None and new_vals is not None: # Used as a shortcut to avoid needing to supply twice
            new_uids = new_vals
        orig_len = self.len_used
        n_new = len(new_uids)
        self.len_used += n_new  # Increase the count of the number of agents by `n` (the requested number of new agents)
        
        # Physically reshape the arrays, if needed
        if orig_len + n_new > self.len_tot:
            n_grow = max(n_new, self.len_tot//2)  # Minimum 50% growth, since growing arrays is slow
            new_empty = np.empty(n_grow, dtype=self.dtype) # 10x faster than np.zeros()
            self.raw = np.concatenate([self.raw, new_empty], axis=0)
            self.len_tot = len(self.raw)
            if n_grow > n_new: # We added extra space at the end, set to NaN
                nan_uids = np.arange(self.len_used, self.len_tot)
                self.set_nan(nan_uids)
        
        # Set new values, and NaN if needed
        self.set_new(new_uids, new_vals=new_vals) # Assign new default values to those agents
        return
    
    def set_people(self, people):
        """ Reset the people object associated with this state """
        assert people.initialized, 'People must be initialized before initializing states'
        self.people = people
        return

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
        # Skip if already initialized
        if self.initialized:
            return

        # Establish connection with the People object
        people = sim.people
        self.set_people(people)
        people.register_state(self)
        
        # Connect any distributions in the default to RNGs in the Sim
        if isinstance(self.default, ss.Dist):
            self.default.initialize(module=self, sim=sim)

        # Populate initial values
        self.grow(people.uid)
        self.initialized = True
        return


class BooleanOperationError(NotImplementedError):
    """ Raised when a logical operation is performed on a non-logical array """
    def __init__(self, arr):
        msg = f'Logical operations are only valid on Boolean arrays, not {arr.dtype}'
        super().__init__(msg)

class BooleanNaNError(NotImplementedError):
    """ Raised when NaN is called on a Boolean array """
    def __init__(self):
        msg = 'NaN is not defined for boolean arrays, you do not want to do this!'
        super().__init__(msg)


class FloatArr(Arr):
    """ Subclass of Arr with defaults for floats """
    def __init__(self, name, default=None, nan=np.nan, label=None, skip_init=False):
        super().__init__(name=name, dtype=ss_float, default=default, nan=nan, label=label, coerce=False, skip_init=skip_init)
        return
    
    def isnan(self):
        return np.isnan(self.values)

    def notnan(self, mask=None):
        valid = ~np.isnan(self.values)
        if mask is not None:
            valid = valid*mask
        return valid

    
class IntArr(Arr):
    """ Subclass of Arr with defaults for integers """
    def __init__(self, name, default=None, nan=ss.intnan, label=None, skip_init=False):
        super().__init__(name=name, dtype=ss_int, default=default, nan=nan, label=label, coerce=False, skip_init=skip_init)
        return


class BoolArr(Arr):
    """ Subclass of Arr with defaults for booleans """
    def __init__(self, name, default=None, nan=False, label=None, skip_init=False): # No good NaN equivalent for bool arrays
        super().__init__(name=name, dtype=ss_bool, default=default, nan=nan, label=label, coerce=False, skip_init=skip_init)
        return
    
    def __and__(self, other): return self.dup(self.values & other)
    def __or__(self, other):  return self.dup(self.values | other)
    def __xor__(self, other): return self.dup(self.values ^ other)
    def __invert__(self):     return self.dup(~self.values)
    
    def true(self):
        return self.aliveinds[np.nonzero(self.values)[0]] # TODO: think if can be simplified
    
    def false(self):
        return self.aliveinds[np.nonzero(~self.values)[0]]
    
    def isnan(self):
        raise BooleanNaNError()

    def notnan(self, mask=None):
        if mask is None:
            raise BooleanNaNError()
        else:
            return mask
    
    
class IndexArr(IntArr):
    """ A special class of IndexArr used for UIDs and RNG IDs """
    def __init__(self, name, label=None):
        super().__init__(name=name, label=label, skip_init=True)
        self.raw = uids(self.raw)
        return
    
    def grow(self, new_uids=None, new_vals=None):
        super().grow(new_uids=new_uids, new_vals=new_vals)
        self.raw = uids(self.raw)
        return
    
    
class uids(np.ndarray):
    """ Special class to keep track of UIDs: just a wrapped NumPy array """
    def __new__(cls, arr):
        return np.asarray(arr).view(cls)
    
    def concat(self, other, **kw): # TODO: why can't they both be called cat()?
        """ Equivalent to np.concatenate(), but return correct type """
        return np.concatenate([self, other], **kw).view(self.__class__)
    
    @classmethod
    def cat(cls, *args, **kw):
        """ Equivalent to np.concatenate(), but return correct type """
        arrs = args[0] if len(args) == 1 else args
        return np.concatenate(arrs, **kw).view(cls)
    
    def remove(self, other, **kw):
        return np.setdiff1d(self, other, assume_unique=True, **kw).view(self.__class__)
    
    def intersect(self, other, **kw):
        return np.intersect1d(self, other, assume_unique=True, **kw).view(self.__class__)