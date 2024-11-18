"""
Define array-handling classes, including agent states
"""
import numpy as np
import sciris as sc
import starsim as ss

# Shorten these for performance
ss_float = ss.dtypes.float
ss_int   = ss.dtypes.int
ss_bool  = ss.dtypes.bool
type_def = {
    ss_float: ('float', float, np.float64, np.float32),
    ss_int: ('int', int, np.int64, np.int32),
    ss_bool: ('bool', bool, np.bool_),
}
type_map = {v:k for k,vlist in type_def.items() for v in vlist} # Invert into a full dictionary

__all__ = ['BaseArr', 'Arr', 'FloatArr', 'BoolArr', 'State', 'IndexArr', 'uids']


class BaseArr(np.lib.mixins.NDArrayOperatorsMixin):
    """
    An object that acts exactly like a NumPy array, except stores the values in self.values.
    """
    def __init__(self, values, *args, **kwargs):
        self.values = np.array(values, *args, **kwargs)
        return

    def __getattr__(self, attr):
        """ Make it behave like a regular array mostly -- enables things like sum(), mean(), etc. """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return object.__getattribute__(self, 'values').__getattribute__(attr) # Be explicit to avoid possible recurison

    # Define more base methods
    def __len__(self):   return self.values.__len__()
    def __bool__(self):  return self.values.__bool__()
    def __int__(self):   return self.values.__int__()
    def __float__(self): return self.values.__float__()
    def __contains__(self, key): return self.values.__contains__(key)

    def convert(self, obj):
        """ Check if an object is an array, and convert if so """
        if isinstance(obj, np.ndarray):
            return self.asnew(obj)
        elif isinstance(obj, BaseArr):
            return self.asnew(obj.values)
        return obj

    @staticmethod
    def _arr(obj):
        """ Helper function to efficiently extract values from a BaseArr, or return the original object """
        if isinstance(obj, BaseArr):
            return obj.values
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ To handle all numpy operations, e.g. arr1*arr2 """
        # Convert all inputs to their .values if they are BaseArr, otherwise leave unchanged
        inputs = [self._arr(x) for x in inputs]
        kwargs = {k:self._arr(v) for k,v in kwargs.items()}
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # If result is a tuple (e.g., for divmod or ufuncs that return multiple values), convert all results to BaseArr
        if isinstance(result, tuple):
            return tuple(self.convert(x) for x in result)

        result = self.convert(result)
        return result

    def __iter__(self):
        """ For iterating correctly, e.g. for sum() """
        return iter(self.values)

    def __getitem__(self, index):
        """ For indexing and slicing, e.g. arr[inds] """
        return self.values[index]

    def __setitem__(self, index, value):
        """ Assign values, e.g. arr1[inds] = arr2 """
        self.values[index] = value

    def __array__(self, *args, **kwargs):
        """ To ensure isinstance(arr, BaseArr) passes when creating new instances """
        return np.array(self.values, *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values})"

    def disp(self):
        """ Full display of object """
        return sc.pr(self)

    def asnew(self, values=None, cls=None, **kwargs):
        """ Duplicate and copy (rather than link) data """
        if cls is None: # Use the current class if none is provided
            cls = self.__class__
        new = object.__new__(cls) # Create a new Arr instance
        new.__dict__ = self.__dict__.copy() # Copy pointers
        new.__dict__.update(kwargs) # Update any keyword arguments provided # TODO: add validation?
        if values is not None:
            new.values = values # Replace data with new data
        else:
            new.values = sc.cp(new.values) # TODO: is this needed?
        return new

    def update(self, skip_none=True, overwrite=True, force=False, **kwargs):
        """ Update the attributes, skipping None values and raising an error if extra keys are added """
        if skip_none: # Remove None inputs
            kwargs = {k:v for k,v in kwargs.items() if v is not None}
        if not overwrite: # Don't overwrite non-None values
            kwargs = {k:v for k,v in kwargs.items() if getattr(self, k, None) is None}
        if not force: # Check if we'd be creating any new keys
            kw_keys = set(kwargs.keys())
            self_keys = set(self.__dict__.keys())
            diff = kw_keys - self_keys
            if diff:
                errormsg = f'Invalid arguments to {self}: {diff}'
                raise ValueError(errormsg)
        self.__dict__.update(kwargs) # Actually perform the update
        return self

    def to_json(self):
        """ Return a dictionary representation of the Arr """
        out = dict(
            classname = self.__class__.__name__,
            values = sc.jsonify(self.values),
        )
        return out


class Arr(BaseArr):
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
        skip_init (bool): Whether to skip initialization with the People object (used for uid and slot states)
        people (ss.People): Optionally specify an initialized People object, used to construct temporary Arr instances
    """
    def __init__(self, name=None, dtype=None, default=None, nan=None, label=None, skip_init=False, people=None):
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
        elif isinstance(key, np.ndarray) and ss.options.reticulate: # TODO: fix ss.uids
            return key.astype(int)
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

    def to_json(self):
        """ Export to JSON """
        out = dict(
            classname = self.__class__.__name__,
            name = self.name,
            label = self.label,
            default = self.default,
            nan = self.nan,
            dtype = self.dtype,
            values = sc.jsonify(self.values),
        )
        return out


class FloatArr(Arr):
    """
    Subclass of Arr with defaults for floats and ints.

    Note: Starsim does not support integer arrays by default since they introduce
    ambiguity in dealing with NaNs, and float arrays are suitable for most purposes.
    If you really want an integer array, you can use the default Arr class instead.
    """
    def __init__(self, name=None, nan=np.nan, **kwargs):
        super().__init__(name=name, dtype=ss_float, nan=nan, **kwargs)
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
        super().__init__(name=name, dtype=ss_bool, nan=nan, **kwargs)
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


class State(BoolArr):
    """
    A boolean array being used as a state.

    Although functionally identical to BoolArr, a State is handled differently in
    terms of automation: specifically, results are automatically generated from a
    State (but not a BoolArr).

    States are typically used to keep track of externally-facing variables (e.g.
    disease.susceptible), while BoolArrs can be used to keep track of internal
    ones (e.g. disease.has_immunity).
    """
    pass


class IndexArr(Arr):
    """ A special class of Arr used for UIDs and RNG IDs; not to be used as an integer array (for that, use FloatArr) """
    def __init__(self, name=None, label=None):
        super().__init__(name=name, dtype=ss_int, default=None, nan=-1, label=label, skip_init=True)
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

    def concat(self, other, **kw): # Class and instance methods can't share a name
        """ Equivalent to np.concatenate(), but return correct type """
        return np.concatenate([self, other], **kw).view(self.__class__)

    @classmethod
    def cat(cls, *args, **kw):
        """ Equivalent to np.concatenate(), but return correct type """
        if len(args) == 0 or (len(args) == 1 and (args[0] is None or not len(args[0]))):
            return uids()
        arrs = args[0] if len(args) == 1 else args # TODO: handle one-array case
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
