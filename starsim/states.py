"""
Define array-handling classes, including agent states
"""

import numpy as np
import starsim as ss

# Shorten these for performance
ss_float = ss.dtypes.float
ss_int   = ss.dtypes.int
ss_bool  = ss.dtypes.bool

__all__ = ['check_dtype', 'Arr', 'FloatArr', 'BoolArr', 'IndexArr', 'uids']


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
    """
    Store a state of the agents (e.g. age, infection status, etc.) as an array.
    
    In practice, ``Arr`` objects can be used interchangeably with NumPy arrays.
    They have two main data interfaces: ``Arr.raw`` contains the "raw", underlying
    NumPy array of the data. ``Arr.values`` contains the "active" values, which
    usually corresponds to agents who are alive.
    
    By default, operations are performed on active agents only (specified by ``Arr.auids``,
    which is a pointer to ``sim.people.auids``). For example, ``sim.people.age.mean()``
    will only use the ages of active agents. Thus, ``sim.people.age.mean()``
    is equal to ``sim.people.age.values.mean()``, not ``sim.people.age.raw.mean()``.
    
    If indexing by an int or slice, ``Arr.values`` is used. If indexing by an
    ``ss.uids`` object, ``Arr.raw`` is used. ``Arr`` objects can't be directly
    indexed by a list or array of ints, as this would be ambiguous about whether
    ``values`` or ``raw`` is intended. For example, if there are 1000 people in a 
    simulation and 100 of them have died, ``sim.people.age[999]`` will return
    an ``IndexError`` (since ``sim.people.age[899]`` is the last active agent),
    whereas ``sim.people.age[ss.uids(999)]`` is valid.

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
        people (ss.People): Optionally specify an initialized People object, used to construct temporary Arr instances
    """
    def __init__(self, name=None, dtype=None, default=None, nan=None, label=None, coerce=True, skip_init=False, people=None):
        if coerce:
            dtype = check_dtype(dtype, default)
        
        # Set attributes
        self.name = name
        self.label = label or name
        self.default = default
        self.nan = nan
        self.dtype = dtype
        self.people = people # Used solely for accessing people.auids

        if self.people is None:
            # This Arr is being defined in advance (e.g., as a module state) and we want a bidirectional link
            # with a People instance for dynamic growth. These properties will be initialized later when the
            # People/Sim are initialized
            self.len_used = 0
            self.len_tot = 0
            self.initialized = skip_init
            self.raw = np.empty(0, dtype=dtype)
        else:
            # This Arr is a temporary object used for intermediate calculations when we want to index an array
            # by UID (e.g., inside an update() method). We allow this state to reference an existing, initialized
            # People object, but do not register it for dynamic growth
            self.len_used = self.people.uid.len_used
            self.len_tot = self.people.uid.len_tot
            self.initialized = True
            self.raw = np.full(self.len_tot, dtype=self.dtype, fill_value=self.nan)

        return

    def __repr__(self):
        arr_str = np.array2string(self.values, max_line_width=200)
        if self.name:
            string = f'<{self.__class__.__name__} "{str(self.name)}", len={len(self)}, {arr_str}>'
        else:
            string = f'<{self.__class__.__name__}, len={len(self)}, {arr_str}>'
        return string

    def __len__(self):
        return len(self.auids)

    def _convert_key(self, key):
        """
        Used for getitem and setitem to determine whether the key is indexing
        the raw array (``raw``) or the active agents (``values``), and to convert
        the key to array indices if needed.
        """
        if isinstance(key, (uids, int, ss_int)):
            return key
        elif isinstance(key, (BoolArr, IndexArr)):
            return key.uids
        elif isinstance(key, slice):
            return self.auids[key]
        elif not np.isscalar(key) and len(key) == 0: # Handle [], np.array([]), etc.
            return uids()
        else:
            errormsg = f'Indexing an Arr ({self.name}) by ({key}) is ambiguous or not supported. Use ss.uids() instead, or index Arr.raw or Arr.values.'
            raise Exception(errormsg)
    
    def __getitem__(self, key):
        key = self._convert_key(key)
        return self.raw[key]
    
    def __setitem__(self, key, value):
        key = self._convert_key(key)
        self.raw[key] = value
        return
            
    def __getattr__(self, attr):
        """ Make it behave like a regular array mostly -- enables things like sum(), mean(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self.values, attr)
        
    def __gt__(self, other): return self.asnew(self.values > other,  cls=BoolArr)
    def __lt__(self, other): return self.asnew(self.values < other,  cls=BoolArr)
    def __ge__(self, other): return self.asnew(self.values >= other, cls=BoolArr)
    def __le__(self, other): return self.asnew(self.values <= other, cls=BoolArr)
    def __eq__(self, other): return self.asnew(self.values == other, cls=BoolArr)
    def __ne__(self, other): return self.asnew(self.values != other, cls=BoolArr)
    
    def __and__(self, other): raise BooleanOperationError(self)
    def __or__(self, other):  raise BooleanOperationError(self)
    def __xor__(self, other): raise BooleanOperationError(self)
    def __invert__(self):     raise BooleanOperationError(self)

    def __array_ufunc__(self, *args, **kwargs):
        if args[1] != '__call__':
            # This is a catch-all for ufuncs that are not being applied with '__call__' (e.g., operations returning a scalar like 'np.sum()' use reduce instead)
            args = [(x if x is not self else self.values) for x in args]
            kwargs = {k: v if v is not self else self.values for k, v in kwargs.items()}
            return self.values.__array_ufunc__(*args, **kwargs)
        else:
            args = [(x if x is not self else self.values) for x in args] # Convert any operands that are Arr instances to their value arrays
            if 'out' in kwargs and kwargs['out'][0] is self:
                # In-place operations like += applied to the entire Arr instance
                # use this branch. Therefore, we perform our computation on a new
                # array with the same size as self.values, and then write it back
                # to the appropriate entries in `self.raw` via `self[:]`
                del kwargs['out']
                self[:] = args[0](*args[2:], **kwargs)
                return self
            else:
                # Otherwise, just run the ufunc
                return args[0](*args[2:], **kwargs)

    @property
    def auids(self):
        """ Link to the indices of active agents -- sim.people.auids """
        try:
            return self.people.auids
        except:
            if not self.initialized:
                ss.warn('Trying to access non-initialized Arr object; in most cases, Arr objects need to be initialized with a Sim object, but set skip_init=True if this is intentional.')
            return uids(np.arange(len(self.raw)))
    
    def count(self):
        return np.count_nonzero(self.values)

    @property
    def values(self):
        """ Return the values of the active agents """
        return self.raw[self.auids]

    def set(self, uids, new_vals=None):
        """ Set the values for the specified UIDs"""
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
        """ Shortcut function to set values to NaN """
        self.raw[uids] = self.nan
        return

    @property
    def isnan(self):
        return self.asnew(self.values == self.nan, cls=BoolArr)

    @property
    def notnan(self):
        return self.asnew(self.values != self.nan, cls=BoolArr)

    def grow(self, new_uids=None, new_vals=None):
        """
        Add new agents to an Arr

        This method is normally only called via `People.grow()`.

        Args:
            new_uids: Numpy array of UIDs for the new agents being added
            new_vals: If provided, assign these state values to the new UIDs
        """
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
        self.set(new_uids, new_vals=new_vals) # Assign new default values to those agents
        return
    
    def link_people(self, people):
        """ Link a People object to this state, for access auids """
        self.people = people # Link the people object to this state
        people._link_state(self) # Ensure the state is linked to the People object as well
        return
    
    def init_vals(self):
        """ Actually populate the initial values and mark as initialized; only to be used on initialization """
        if self.initialized:
            errormsg = f'Cannot re-initialize state {self}; use set() instead'
            raise RuntimeError(errormsg)
        self.grow(self.people.uid)
        self.initialized = True
        return

    def asnew(self, arr=None, cls=None, name=None):
        """ Duplicate and copy (rather than link) data, optionally resetting the array """
        if cls is None:
            cls = self.__class__
        if arr is None:
            arr = self.values
        new = object.__new__(cls) # Create a new Arr instance
        new.__dict__ = self.__dict__.copy() # Copy pointers
        new.dtype = arr.dtype # Set to correct dtype
        new.name = name # In most cases, the asnew Arr has different values to the original Arr so the original name no longer makes sense
        new.raw = np.empty(new.raw.shape, dtype=new.dtype) # Copy values, breaking reference
        new.raw[new.auids] = arr
        return new

    def true(self):
        """ Efficiently convert truthy values to UIDs """
        return self.auids[self.values.astype(bool)]

    def false(self):
        """ Reverse of true(); return UIDs of falsy values """
        return self.auids[~self.values.astype(bool)]


class FloatArr(Arr):
    """
    Subclass of Arr with defaults for floats and ints.
    
    Note: Starsim does not support integer arrays by default since they introduce
    ambiguity in dealing with NaNs, and float arrays are suitable for most purposes.
    If you really want an integer array, you can use the default Arr class instead.    
    """
    def __init__(self, name=None, nan=np.nan, **kwargs):
        super().__init__(name=name, dtype=ss_float, nan=nan, coerce=False, **kwargs)
        return

    @property
    def isnan(self):
        """ Return BoolArr for NaN values """
        return self.asnew(np.isnan(self.values), cls=BoolArr)

    @property
    def notnan(self):
        """ Return BoolArr for non-NaN values """
        return self.asnew(~np.isnan(self.values), cls=BoolArr)
    
    @property
    def notnanvals(self):
        """ Return values that are not-NaN """
        vals = self.values # Shorten and avoid double indexing
        out = vals[np.nonzero(~np.isnan(vals))[0]]
        return out


class BoolArr(Arr):
    """ Subclass of Arr with defaults for booleans """
    def __init__(self, name=None, nan=False, **kwargs): # No good NaN equivalent for bool arrays
        super().__init__(name=name, dtype=ss_bool, nan=nan, coerce=False, **kwargs)
        return
    
    def __and__(self, other): return self.asnew(self.values & other)
    def __or__(self, other):  return self.asnew(self.values | other)
    def __xor__(self, other): return self.asnew(self.values ^ other)
    def __invert__(self):     return self.asnew(~self.values)

    # BoolArr cannot store NaNs so report all entries as being not-NaN
    @property
    def isnan(self):
        return self.asnew(np.full_like(self.values, fill_value=False), cls=BoolArr)

    @property
    def notnan(self):
        return self.asnew(np.full_like(self.values, fill_value=True), cls=BoolArr)

    @property
    def uids(self):
        """ Alias to Arr.true """
        return self.true()

    def split(self):
        """ Return UIDs of values that are true and false as separate arrays """
        t_uids = self.true()
        f_uids = self.false()
        return t_uids, f_uids

    
class IndexArr(Arr):
    """ A special class of Arr used for UIDs and RNG IDs; not to be used as an integer array (for that, use FloatArr) """
    def __init__(self, name=None, label=None):
        super().__init__(name=name, dtype=ss_int, default=None, nan=-1, label=label, coerce=False, skip_init=True)
        self.raw = uids(self.raw)
        return
    
    @property
    def uids(self):
        """ Alias to self.values, to allow Arr.uids like BoolArr """
        return self.values

    
    def grow(self, new_uids=None, new_vals=None):
        """ Change the size of the array """
        if new_uids is None and new_vals is not None: # Used as a shortcut to avoid needing to supply twice
            new_uids = new_vals
        super().grow(new_uids=new_uids, new_vals=new_vals)
        self.raw = uids(self.raw)
        return
    
    
class uids(np.ndarray):
    """
    Class to specify that integers should be interpreted as UIDs.
    
    For all practical purposes, behaves like a NumPy integer array. However,
    has additional methods ``uids.concat()`` (instance method), ``ss.uids.cat()``
    (class method), ``uids.remove()``, and ``uids.intersect()`` to simplify common
    UID operations.    
    """
    def __new__(cls, arr=None):
        if isinstance(arr, np.ndarray): # Shortcut to typical use case, where the input is an array
            return arr.astype(ss_int).view(cls)
        elif isinstance(arr, BoolArr): # Shortcut for arr.uids
            return arr.uids
        elif isinstance(arr, set):
            return np.fromiter(arr, dtype=ss_int).view(cls)
        elif arr is None: # Shortcut to return empty
            return np.empty(0, dtype=ss_int).view(cls)
        elif isinstance(arr, int): # Convert e.g. ss.uids(0) to ss.uids([0])
            arr = [arr]
        return np.asarray(arr, dtype=ss_int).view(cls) # Handle everything else

    def concat(self, other, **kw): # TODO: why can't they both be called cat()?
        """ Equivalent to np.concatenate(), but return correct type; see ss.uids.cat() for the class method """
        return np.concatenate([self, other], **kw).view(self.__class__)

    @classmethod
    def cat(cls, *args, **kw):
        """ Equivalent to np.concatenate(), but return correct type; see ss.uids.concat() for the instance method """
        arrs = args[0] if len(args) == 1 else args
        return np.concatenate(arrs, **kw).view(cls)

    def remove(self, other, **kw):
        """ Remove provided UIDs from current array"""
        if isinstance(other, BoolArr):
            other = other.uids
        return np.setdiff1d(self, other, **kw).view(self.__class__)

    def intersect(self, other, **kw):
        """ Keep only UIDs that are also present in the other array """
        if isinstance(other, BoolArr):
            other = other.uids
        return np.intersect1d(self, other, **kw).view(self.__class__)

    def union(self, other, **kw):
        """ Return all UIDs present in both arrays """
        if isinstance(other, BoolArr):
            other = other.uids
        return np.union1d(self, other, **kw).view(self.__class__)

    def xor(self, other, **kw):
        """ Return UIDs present in only one of the arrays """
        if isinstance(other, BoolArr):
            other = other.uids
        return np.setxor1d(self, other, **kw).view(self.__class__)

    def to_numpy(self):
        """ Return a view as a standard NumPy array """
        return self.view(np.ndarray)
    
    def unique(self, return_index=False):
        """ Return unique UIDs; equivalent to np.unique() """
        if return_index:
            arr, index = np.unique(self, return_index=True)
            return arr.view(self.__class__), index
        else:
            arr = np.unique(self).view(self.__class__)
            return arr

    # Implement collection of operators
    def __and__(self, other): return self.intersect(other)
    def __or__(self, other) : return self.union(other)
    def __sub__(self, other): return self.remove(other)
    def __xor__(self, other): return self.xor(other)
    def __invert__(self)    : raise Exception(f"Cannot invert an instance of {self.__class__.__name__}. One possible cause is attempting `~x.uids` - use `x.false()` or `(~x).uids` instead")


class BooleanOperationError(NotImplementedError):
    """ Raised when a logical operation is performed on a non-logical array """
    def __init__(self, arr):
        msg = f'Logical operations are only valid on Boolean arrays, not {arr.dtype}'
        super().__init__(msg)