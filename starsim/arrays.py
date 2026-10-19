"""
Define array-handling classes, including agent states
"""
import itertools
import numbers
import numpy as np
import numba as nb
import sciris as sc
import starsim as ss

# Shorten these for performance -- reset with ss.options.refresh_references(), which is called automatically
ss_float   = ss.dtypes.float
ss_nbfloat = ss.dtypes.nbfloat
ss_int     = ss.dtypes.int
ss_uid     = ss.dtypes.uid
ss_nbint   = ss.dtypes.nbint
ss_bool    = ss.dtypes.bool
int_nan    = ss.dtypes.int_nan
numba_indexing = ss.options.numba_indexing


def np_indexer(arr, inds):
    """ Much faster than Numba for small numbers of indices (<1k) """
    return arr[inds]

@nb.jit(fastmath=True, parallel=False, cache=True)
def nb_indexer(arr, inds):
    """ Roughly 30% faster than NumPy for large numbers of indices (>10k)

    NB: in some versions of Numba (>0.62), the explicit loop is much faster than array indexing.
    """
    out = np.empty(inds.size, dtype=arr.dtype)
    for i in range(inds.size):
        out[i] = arr[inds[i]]
    return out

@nb.jit(cache=True) # No fastmath: it assumes no NaNs, which would break the truthiness test for NaN floats
def nb_true(auids, vals, want):
    """ Return the UIDs where `bool(vals[k])` equals `want` (True for true(), False for false()).

    ~4-5x faster than `auids[vals.astype(bool)]` for large arrays: a single branchless pass
    with no intermediate mask allocation. `vals[k] != 0` matches `.astype(bool)` for every dtype,
    including NaN floats (NaN != 0 is True, i.e. NaN is truthy), so results are identical.
    """
    n = auids.shape[0]
    out = np.empty(n, dtype=auids.dtype)
    m = 0
    for k in range(n):
        out[m] = auids[k] # Written every iteration (branchless); only kept if m advances
        m += (vals[k] != 0) == want
    return out[:m]

@nb.jit(cache=True)
def nb_true_raw(auids, raw, want):
    """ As `nb_true()`, reading each agent's value from the full array, so the values of the active agents are not gathered first """
    n = auids.shape[0]
    out = np.empty(n, dtype=auids.dtype)
    m = 0
    for k in range(n):
        uid = auids[k]
        out[m] = uid
        m += (raw[uid] != 0) == want
    return out[:m]

@nb.jit(cache=True)
def nb_count(auids, raw):
    """ The number of active agents whose value is truthy (as `np.count_nonzero()`), without gathering the values first """
    m = 0
    for k in range(auids.shape[0]):
        m += raw[auids[k]] != 0
    return m


class BaseArr(np.lib.mixins.NDArrayOperatorsMixin):
    """
    An object that acts exactly like a NumPy array, except stores the values in self.values.
    """
    columns = None # Column names for 2D arrays, e.g. ['wild', 'alpha']

    def __init__(self, values, *args, **kwargs):
        self.values = np.array(values, *args, **kwargs)
        return

    def __getattr__(self, attr):
        """ Make it behave like a regular array mostly -- enables things like sum(), mean(), etc. -- and allow e.g. arr.alpha for a named column """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        elif self.columns is not None and attr in self.columns:
            return self.col(attr)
        else:
            return object.__getattribute__(self, 'values').__getattribute__(attr) # Be explicit to avoid possible recurison

    def __setattr__(self, attr, value):
        """ Set the values of a named column, e.g. arr.alpha = 0; also needed so arr.alpha += 1 doesn't store the column as an attribute """
        if self.columns is not None and attr in self.columns:
            self.col(attr)[:] = value
        else:
            object.__setattr__(self, attr, value)
        return

    def validate_columns(self):
        """ Check that column names are unique strings that don't clash with attributes, e.g. "low" """
        if self.columns is not None:
            self.columns = list(self.columns)
            if len(set(self.columns)) != len(self.columns):
                errormsg = f'Column names for "{self.name}" must be unique: {self.columns}'
                raise ValueError(errormsg)
            for col in self.columns:
                if not isinstance(col, str) or col in self.__dict__ or hasattr(type(self), col) or hasattr(np.ndarray, col):
                    errormsg = f'Invalid column name "{col}" for "{self.name}": column names must be strings that are not attributes'
                    raise ValueError(errormsg)
        return

    def _col_index(self, v):
        """ Convert a column name to its index, e.g. 'alpha' → 1; other values are returned unchanged """
        if not isinstance(v, str):
            return v
        if self.columns is None:
            errormsg = f'Cannot get column "{v}" since "{self.name}" does not have named columns; use e.g. columns=["a", "b"] to name them'
            raise KeyError(errormsg)
        elif v not in self.columns:
            errormsg = f'Column "{v}" not found in "{self.name}"; available columns are {self.columns}'
            raise KeyError(errormsg)
        return self.columns.index(v)

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
        for x in itertools.chain(inputs, kwargs.values()):
            if isinstance(x, ss.TimePar):
                return NotImplemented # Operations involving a TimePar and a BaseArr won't return a BaseArr, and will be handled by a TimePar operator
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
        return

    def __array__(self, *args, **kwargs):
        """ To ensure isinstance(arr, BaseArr) passes when creating new instances """
        return np.array(self.values, *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values})"

    def disp(self):
        """ Full display of object """
        return sc.pr(self)

    def asnew(self, values=None, cls=None, **kwargs):
        """
        Duplicate and copy (rather than link) data.

        Args:
            values (array): replacement data; if None, copies existing values
            cls (type): class to use for the new instance; defaults to the current class
            kwargs: additional attributes to set on the new instance
        """
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

    In practice, `Arr` objects can be used interchangeably with NumPy arrays.
    They have two main data interfaces: `Arr.raw` contains the "raw", underlying
    NumPy array of the data. `Arr.values` contains the "active" values, which
    usually corresponds to agents who are alive.

    By default, operations are performed on active agents only (specified by `Arr.auids`,
    which is a pointer to `sim.people.auids`). For example, `sim.people.age.mean()`
    will only use the ages of active agents. Thus, `sim.people.age.mean()`
    is equal to `sim.people.age.values.mean()`, not `sim.people.age.raw.mean()`.

    If indexing by an int, integer array, or `ss.uids` object, `Arr.raw` is used
    (i.e. the ints are treated as UIDs). If indexing by a full slice (`[:]`) or a
    boolean array, `Arr.values` is used. `Arr` objects can't be directly indexed
    by a list, as this would be ambiguous about whether `values` or `raw` is
    intended. For example, if there are 1000 people in a simulation and
    100 of them have died, `sim.people.age[999]` and `sim.people.age[ss.uids(999)]`
    both return the age of the agent with UID 999, whereas `sim.people.age.values[899]`
    returns the age of the last active agent.

    Note on terminology: the term "states" is often used to refer to *all* `ss.Arr` objects,
    (e.g. in `module.define_states()`, whether or not they are BoolStates.

    Args:
        name (str): The name for the state (also used as the dictionary key, so should not have spaces etc.)
        dtype (class): The dtype to use for this instance (if None, infer from value)
        default (any): Specify default value for new agents. This can be:

            * A scalar with the same dtype (or castable to the same dtype) as the Arr;
            * A callable, with a single argument for the number of values to produce;
            * A [`ss.Dist`](`starsim.distributions.Dist`) instance.
        nan (any): the value to use to represent NaN (not a number); also used as the default value if not supplied
        label (str): The human-readable name for the state
        raw (array): If provided, initialize the array with these raw values
        skip_init (bool): Whether to skip initialization with the People object (used for uid and slot states)
        people (`ss.People`): Optionally specify an initialized People object, used to construct temporary Arr instances
        mock (int): if provided, create a mock People object (of length `mock`, unless `raw` is provided) to initialize the array (for debugging purposes)
        columns (int/list): if provided, create a 2D array with this many columns (e.g. one per variant), or a list of column names (e.g. `['wildtype', 'mutant']`)
        shared (bool/Arr): if True, the array takes no memory per agent until a value is set (other than by its default, which must be a scalar): until then every agent has the default, and `raw` is a read-only view of it; if an `Arr`, the same with that array's values (as `people.slot` reads `people.uid`)

    For 2D arrays (i.e. if `columns` is supplied), `arr[uids]` returns an array of shape `(len(uids), ncols)`,
    `arr[uids, v]` or `arr[:, v]` returns column `v`, and `arr.col(v)` returns column `v` as a 1D `Arr`. If the columns
    are named, `v` can also be a name, e.g. `arr.col('mutant')`, `arr['mutant']`, or `arr.mutant`. A scalar default fills every column, and a callable
    default should return an array of shape `(n, ncols)`. Arrays with more dimensions are also supported via a tuple,
    e.g. `columns=(2,3)` gives a raw array of shape `(len_tot, 2, 3)`.

    Examples:
        ```python
        # Create a standalone Arr for quick testing
        age = ss.Arr('age', default=0, mock=5) # 5 = length if not supplying a real People object
        age[:] = [20, 30, 40, 50, 60]
        age.mean()  # Returns 40.0

        # Use within a simulation
        sim = ss.Sim(n_agents=100).init()
        sim.people.age.mean()  # Mean age of active agents

        # Create a 2D array, e.g. immunity to each of 3 variants
        imm = ss.FloatArr('imm', default=0, columns=3, mock=5)
        imm[ss.uids([0, 1]), 2] = 0.8 # Set immunity to variant 2 for agents 0 and 1
        imm[ss.uids([0, 1])] # Returns an array of shape (2, 3)
        (imm.col(2) > 0.5).uids  # Returns ss.uids([0, 1])

        # Create a 2D array with named columns
        imm = ss.FloatArr('imm', default=0, columns=['wildtype', 'mutant'], mock=5)
        imm[ss.uids([0, 1]), 'mutant'] = 0.8 # Same as imm[ss.uids([0, 1]), 1] = 0.8
        (imm.col('mutant') > 0.5).uids  # Returns ss.uids([0, 1])
        ```
    """
    mod = None # The module whose state this is, if any; set by Module.define_states(), so a condition on it (e.g. hiv.cd4 < 200) can find it
    shared = None # Whether it has no memory of its own yet: see own()

    def __init__(self, name=None, dtype=None, default=None, nan=None, label=None, raw=None, skip_init=False, people=None, mock=None, columns=None, shared=False):
        # Set attributes
        self.name = name
        self.shared = None if shared is False else shared # True or an Arr until a value is set: see own()
        self.label = label or name
        self.default = default
        if nan is not None: self.nan = nan
        if dtype is None: dtype = np.asarray(default).dtype if np.isscalar(default) else ss_float # Infer from the default value
        self.dtype = dtype
        self.nan_eq = (nan == nan) # Distinguish between NaN placeholder values (e.g. int), and ones where equality is impossible (e.g. float)
        self.people = people # Used solely for accessing people.auids

        # Handle 2D (or higher) arrays; item_shape is the shape of each agent's values, e.g. (3,) for columns=3
        colnames = None # Set at the end, since column names are treated as attributes
        if isinstance(columns, (list, tuple)) and all(isinstance(col, str) for col in columns): # Named columns, e.g. columns=['wildtype', 'mutant']
            colnames = columns
            columns = len(columns)
        self.item_shape = () if columns is None else tuple(np.atleast_1d(columns))
        self.ndim = 1 + len(self.item_shape)
        if self.ndim > 1:
            self._index = np_indexer # Numba indexing is 1D only; set here so there's no extra check on each access
            if isinstance(default, ss.Dist):
                errormsg = f'A distribution cannot be used as the default for 2D array "{name}"; use a callable returning an array of shape (n, ncols) instead'
                raise TypeError(errormsg)

        if self.people is None:
            # This Arr is being defined in advance (e.g., as a module state) and we want a bidirectional link
            # with a People instance for dynamic growth. These properties will be initialized later when the
            # People/Sim are initialized
            self.len_used = 0
            self.len_tot = 0
            self.initialized = skip_init
            if raw is None:
                self.raw = np.empty((0, *self.item_shape), dtype=self.dtype)
            else:
                self.raw = raw # Can't check length since we don't have the People object yet
        else:
            # This Arr is a temporary object used for intermediate calculations when we want to index an array
            # by UID (e.g., inside an update() method). We allow this state to reference an existing, initialized
            # People object, but do not register it for dynamic growth
            self.len_used = self.people.uid.len_used
            self.len_tot = self.people.uid.len_tot
            self.initialized = True
            if raw is None:
                self.raw = np.full((self.len_tot, *self.item_shape), dtype=self.dtype, fill_value=self.nan)
            else:
                if len(raw) == self.len_tot:
                    self.raw = raw # Do not coerce dtype
                else:
                    errormsg = f'Cannot populate array of length {self.len_tot} with values of length {len(raw)}'
                    raise ValueError(errormsg)

        # If we have a mock People object, initialize the values
        if mock:
            n_agents = len(raw) if raw is not None else mock
            self.people = ss.mock_people(n_agents)
            if raw is None:
                self.init_vals()
            else:
                self.initialized = True # Its values are given, so arithmetic on it is not a lazy condition

        self.columns = colnames
        self.validate_columns() # Do this last, so column names can be checked against all attributes
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
        the raw array (`raw`) or the active agents (`values`), and to convert
        the key to array indices if needed.

        Note that values[key] = raw[self.auids[key]]

        In short:
            - ss.uids, integer, or integer array: raw
            - BoolArr, boolean array, or full slice: values
            - tuple (for 2D arrays, e.g. `arr[uids, v]`): the first element is converted as above, and the second can be a column name
        """
        if isinstance(key, (uids, int, ss_int, ss_uid)) or (isinstance(key, np.ndarray) and key.dtype in (ss_int, ss_uid)): # Catch immediately for speed; 32-bit, as people.auids are, too
            return key
        elif isinstance(key, tuple):
            rows = self._convert_key(key[0])
            cols = tuple(self._col_index(k) for k in key[1:])
            if isinstance(key[0], slice) and any(np.ndim(c) for c in cols): # E.g. arr[:, [0,2]]: select the columns for every row, rather than pairing the rows and columns
                rows = rows[:, None]
            return (rows,) + cols
        elif isinstance(key, (BoolArr, IndexArr)):
            return key.uids
        elif isinstance(key, slice):
            if key.start is None and key.stop is None:
                return self.auids[key]
            elif isinstance(key.start, uids) and isinstance(key.stop, uids):
                return key
            else:
                errormsg = f'Using {key} on an array is ambiguous. Use arr.values[slice] for alive agents, or arr.raw[slice] for all agents.'
                raise ValueError(errormsg)
        elif not np.isscalar(key) and len(key) == 0: # Handle [], np.array([]), etc.
            return uids()
        elif isinstance(key, np.ndarray) and key.dtype == bool: # Not commonly used since would usually use BoolArr, which is handled separately
            return self.auids[key]
        elif isinstance(key, np.ndarray) and ss.options.reticulate: # TODO: fix ss.uids so it works from R/reticulate
            return key.astype(int)
        else:
            errormsg = f'Indexing an Arr ({self.name}) by ({key}) is ambiguous or not supported. Use ss.uids() instead, or index Arr.raw or Arr.values.'
            raise TypeError(errormsg)

    def _index(self, arr, inds):
        """ Index the array using the most efficient method for the current array size """
        if isinstance(inds, np.ndarray) and inds.size >= numba_indexing:
            return nb_indexer(arr, inds)
        else:
            return np_indexer(arr, inds)

    def __getitem__(self, key):
        if not isinstance(key, uids): # Shortcut since main pathway
            if isinstance(key, str): # Named column, e.g. arr['mutant']
                return self.col(key)
            key = self._convert_key(key)
        return self._index(self.raw, key)

    def __setitem__(self, key, value):
        if not isinstance(key, uids):
            if isinstance(key, str): # Named column for all active agents, e.g. arr['mutant'] = 0.5
                key = (slice(None), key)
            key = self._convert_key(key)
        if self.shared is not None:
            self.own()
        self.raw[key] = value
        return

    def _boolmath(self, op, other=None, inplace=False):
        """ Helper function for boolean operations.

        Args:
            op: A string with the operator e.g., ">", "&" etc.
            other: The object operating on the BoolArr e.g., Arr, np.ndarray.
            inplace: If True, overwrite the current object rather than returning a new one.

        Returns:
            A BoolArr with the result of the operation.
        """
        if not self.initialized and not inplace: # Before the sim exists, this builds a lazy condition instead: see ss.conditions.Cond
            return ss.conditions.lazy(op, self, other)
        self_raw = self.raw # Always size N
        both_raw = True # Assume this by default
        if isinstance(other, Arr):
            other_raw = other.raw # Also always size N
        elif isinstance(other, (numbers.Number, type(None))):
            other_raw = other # If it's a scalar, we don't have to worry about array size
        elif isinstance(other, ss.date):
            other_raw = other.years
        elif isinstance(other, np.ndarray): # It's a NumPy array, we have to check the size
            both_raw = self_raw.size == other.size # It's raw if it's the same size, values otherwise
            other_raw = other
        else:
            self._type_error(other)

        # Figure out the two arrays to operate on
        if both_raw:
            a = self_raw
            b = other_raw
        else:
            a = self.values
            b = other
            inds = self.auids

        # Perform operations, listed in rough order of how frequently used they are; Numba doesn't make these operations faster
        if   op == '==': c = a == b
        elif op == '>':  c = a > b
        elif op == '<':  c = a < b
        elif op == '>=': c = a >= b
        elif op == '<=': c = a <= b
        elif op == '!=': c = a != b
        elif op == '&':  c = a & b
        elif op == '|':  c = a | b
        elif op == '^':  c = a ^ b
        elif op == '~':  c = ~a
        else:
            errormsg = f'Unrecognized operator "{op}"'
            raise ValueError(errormsg)

        if both_raw:
            result_raw = c
        else:
            result_raw = np.empty(self_raw.shape, dtype=np.bool_)
            result_raw[inds] = c

        if inplace:
            self.raw[:] = result_raw
            return self
        else:
            return self.asnew(result_raw, cls=BoolArr, copy=False)

    def __gt__(self, other): return self._boolmath('>', other)
    def __lt__(self, other): return self._boolmath('<', other)
    def __ge__(self, other): return self._boolmath('>=', other)
    def __le__(self, other): return self._boolmath('<=', other)
    def __eq__(self, other): return self._boolmath('==', other)
    def __ne__(self, other): return self._boolmath('!=', other)

    def __and__(self, other): raise BooleanOperationError(self)
    def __or__(self, other):  raise BooleanOperationError(self)
    def __xor__(self, other): raise BooleanOperationError(self)
    def __invert__(self):     raise BooleanOperationError(self)

    def isin(self, values):
        """
        Return a `BoolArr` that is True wherever the array's value is in `values`.

        An `Arr`-aware analogue of `numpy.isin` that also picks the faster algorithm:
        `values` may be a scalar or any array-like of test values. For a small number
        of test values the equality comparisons are OR-ed together (which avoids the
        sort/hash overhead of `np.isin` and is faster for the common
        handful-of-categories case); for larger sets it falls back to `np.isin`. The
        result is a `BoolArr`, so `.uids`, `[uids]`, and boolean combination all work.

        Args:
            values: a scalar, or an array-like (list/tuple/array) of values to test against.

        Returns:
            BoolArr: True wherever the array's value equals one of `values`.

        **Example**:

            active = disease.state.isin([State.MILD, State.SEVERE]).uids
        """
        if not self.initialized: # Before the sim exists, this builds a lazy condition instead: see ss.conditions.Cond
            return ss.conditions.from_arr(self).isin(sc.tolist(values))
        values = np.atleast_1d(sc.toarray(values)) # Normalize scalars/lists/arrays to 1D (matches np.isin for other inputs)
        if values.dtype == object: # A set, dict, or generator is wrapped as a single object, which np.isin would silently treat as no-match; be louder
            errormsg = f'isin() expects a scalar or array-like object (list/tuple/array), not {values}'
            raise TypeError(errormsg)
        raw = self.raw # Always size N, so the result has correct full-size raw
        n = len(values)
        if n == 0:
            result = np.zeros(raw.shape, dtype=np.bool_)
        elif n <= 10: # OR-of-equality beats np.isin for small sets (measured crossover ~11)
            result = (raw == values[0])
            for v in values[1:]:
                result |= (raw == v)
        else:
            result = np.isin(raw, values)
        return self.asnew(result, cls=BoolArr, copy=False)

    def _math(self, op, other=None, inplace=False, reverse=False):
        """
        Helper function for numeric operations

        The main purpose of this function is to handle in-place operations where the
        result of the operation needs to be applied to active agents only.
        """

        if isinstance(other, (ss.TimePar, ss.date)):
            return NotImplemented # Not implemented here, at least for now. But allow falling back to the other classes if they have an implementation
        if not self.initialized and not inplace and op in ss.conditions.ops: # Before the sim exists, this builds a lazy expression instead, as a comparison does: see ss.conditions.Cond
            arr = ss.conditions.from_arr(self)
            return ss.conditions.Cond(op, other, arr) if reverse else ss.conditions.Cond(op, arr, other)

        inds = self.auids
        a = self._index(self.raw, inds)

        if isinstance(other, Arr):
            b = other._index(other.raw, inds)
        elif isinstance(other, BaseArr):
            b = other.values
        else:
            b = other

        if reverse:
            a, b = b, a

        if op == '+': ufunc = np.add
        elif op == '-': ufunc = np.subtract
        elif op == '*': ufunc = np.multiply
        elif op == '/': ufunc = np.divide
        elif op == '//': ufunc = np.floor_divide
        elif op == '**': ufunc = np.power
        elif op == '%': ufunc = np.mod
        else:
            errormsg = f'Unrecognized operator "{op}"'
            raise ValueError(errormsg)

        if inplace:
            out = np.empty(a.shape, dtype=self.raw.dtype) # In-place operations cannot change the dtype
            ufunc(a, b, out=out) # Calling the ufunc this way ensures in-place operations will raise the same error for Arr as for np.array
            if self.shared is not None:
                self.own()
            self.raw[inds] = out # Write the values back to the existing raw array in-place
            return self
        else:
            c = ufunc(a, b) # Evaluate the ufunc. Unlike the in-place operation above, the dtype is now allowed to change
            result_dtype = np.asarray(c).dtype # The resulting dtype will dictate the return class e.g., BoolArr * FloatArr = FloatArr, with np.bool * np.float = np.float
            result_raw = self.raw.astype(result_dtype, copy=True) # Cast the existing values to the new dtype
            result_raw[inds] = c # Write the new values into the raw array (with the updated dtype)
            result_cls, result_nan = self._math_result_type(result_dtype) # Work out which Arr class the new dtype corresponds to
            result = self.asnew(result_raw, cls=result_cls, copy=False) # Wrap the new result array in an Arr subclass of the appropriate type e.g., a raw `float` array gets wrapped in FloatArr
            result.nan = result_nan # Since asnew() does not set the .nan or .nan_eq attributes because it does not call __init__, set them now
            result.nan_eq = (result.nan == result.nan)
            return result

    @staticmethod
    def _math_result_type(dtype):
        """
        Choose the Arr subclass and nan sentinel for a numeric result dtype.

        The NaN sentinel value is required because the NaN value is set in the constructor, which
        does not get called when the new instance is created via `asnew()`. Therefore it is necessary
        to explicitly set it. Because the default value is only defined as a function default value or
        otherwise accessible only after creating an instance, we have to explicitly re-declare the NaN
        values here. This could be avoided in the future if the NaN value were stored as a class variable
        """
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.bool_):
            return BoolArr, False
        elif np.issubdtype(dtype, np.integer):
            return IntArr, int_nan
        elif np.issubdtype(dtype, np.floating):
            return FloatArr, np.nan
        else:
            errormsg = f'Unsupported arithmetic result dtype {dtype}'
            raise TypeError(errormsg)

    def __add__(self, other): return self._math('+', other)
    def __radd__(self, other): return self._math('+', other, reverse=True)
    def __sub__(self, other): return self._math('-', other)
    def __rsub__(self, other): return self._math('-', other, reverse=True)
    def __mul__(self, other): return self._math('*', other)
    def __rmul__(self, other): return self._math('*', other, reverse=True)
    def __truediv__(self, other): return self._math('/', other)
    def __rtruediv__(self, other): return self._math('/', other, reverse=True)
    def __floordiv__(self, other): return self._math('//', other)
    def __rfloordiv__(self, other): return self._math('//', other, reverse=True)
    def __pow__(self, other): return self._math('**', other)
    def __rpow__(self, other): return self._math('**', other, reverse=True)
    def __mod__(self, other): return self._math('%', other)
    def __rmod__(self, other): return self._math('%', other, reverse=True)

    def __iadd__(self, other): return self._math('+', other, inplace=True)
    def __isub__(self, other): return self._math('-', other, inplace=True)
    def __imul__(self, other): return self._math('*', other, inplace=True)
    def __itruediv__(self, other): return self._math('/', other, inplace=True)
    def __ifloordiv__(self, other): return self._math('//', other, inplace=True)
    def __ipow__(self, other): return self._math('**', other, inplace=True)
    def __imod__(self, other): return self._math('%', other, inplace=True)

    @property
    def auids(self):
        """ Link to the indices of active agents -- sim.people.auids """
        try:
            return self.people.auids
        except AttributeError:
            if not self.initialized:
                ss.warn('Trying to access non-initialized Arr object; in most cases, Arr objects need to be initialized with a Sim object, but set skip_init=True if this is intentional.')
            return uids(np.arange(len(self.raw)))

    def count(self, axis=None):
        """ Count the number of nonzero (truthy) values among active agents (for 2D arrays, use `axis=0` to count each column) """
        auids = self.auids
        if axis is None and self.raw.ndim == 1 and auids.size >= numba_indexing and len(self.raw) != auids.size:
            return int(nb_count(auids.view(np.ndarray), self.raw))
        return np.count_nonzero(self.values, axis=axis)

    @property
    def values(self):
        """ Return the values of the active agents """
        if len(self.raw) == self.auids.size:
            return self.raw
        else:
            try:
                return self._index(self.raw, self.auids) # Optimized for speed
            except Exception as E:
                warnmsg = f'Trying to access an uninitialized array; use arr.raw to see the underlying values (if any).\n{E}'
                ss.warn(warnmsg)
                return np.array([])

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
        if self.shared is not None:
            self.own()
        self.raw[uids] = new_vals
        return new_vals

    def set_nan(self, uids):
        """ Shortcut function to set values to NaN """
        if self.shared is not None:
            self.own()
        self.raw[uids] = self.nan
        return

    def shared_raw(self):
        """ The read-only values of an array that has no memory of its own yet (see `shared` in `ss.Arr`): its default, or the values of the array it reads """
        if isinstance(self.shared, Arr):
            raw = self.shared.raw.view()
            raw.flags.writeable = False
            return raw
        value = self.nan if self.default is None else self.default
        return np.broadcast_to(np.asarray(value, dtype=self.dtype), (self.len_tot, *self.item_shape))

    def own(self):
        """ Give an array that has no memory of its own yet (see `shared` in `ss.Arr`) its own values, before one is set """
        raw = np.array(self.raw) # A copy, which can be written
        if self.shared is True: # The space for agents not yet made is NaN, as for any array
            raw[self.len_used:] = self.nan
        self.shared = None
        self.raw = raw
        return

    @property
    def isnan(self):
        """ Return BoolArr for NaN values """
        out = self.values == self.nan if self.nan_eq else np.isnan(self.values)
        return self.asnew(out, cls=BoolArr)

    @property
    def notnan(self):
        """ Return BoolArr for non-NaN values """
        out = self.values != self.nan if self.nan_eq else ~np.isnan(self.values)
        return self.asnew(out, cls=BoolArr)

    @property
    def notnanvals(self):
        """ Return values that are not-NaN """
        vals = self.values # Shorten and avoid double indexing
        if self.ndim > 1: # For 2D (or higher) arrays, return a flat array of all non-NaN values
            return vals[vals != self.nan] if self.nan_eq else vals[~np.isnan(vals)]
        elif self.nan_eq:
            return vals[np.nonzero(vals != self.nan)[0]]
        else:
            return vals[np.nonzero(~np.isnan(vals))[0]]

    def grow(self, new_uids=None, new_vals=None):
        """ Add new agents to an Arr

        This method is normally only called via `People.grow()`.

        Args:
            new_uids (array): Numpy array of UIDs for the new agents being added
            new_vals (array): If provided, assign these state values to the new UIDs
        """
        orig_len = self.len_used
        n_new = len(new_uids)
        if self.shared is not None and new_vals is None: # Every agent still has the default (or the values of the array it reads), so nothing to store
            self.len_used += n_new
            if self.len_used > self.len_tot:
                self.len_tot += max(n_new, self.len_tot//5) # The same room as below, so its length is that of the other arrays
            self.raw = self.shared_raw()
            return
        self.len_used += n_new  # Increase the count of the number of agents by `n` (the requested number of new agents)

        # Physically reshape the arrays, if needed
        if orig_len + n_new > self.len_tot:
            n_grow = max(n_new, self.len_tot//5)  # Minimum 20% growth, since growing arrays is slow (but each agent's arrays take a fifth more memory at most)
            new_empty = np.empty((n_grow, *self.item_shape), dtype=self.dtype) # 10x faster than np.zeros()
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

    @staticmethod
    def _type_error(obj):
        errormsg = f'Can only handle NumPy arrays and Arr objects, not {type(obj)}'
        raise TypeError(errormsg)

    def convert(self, obj, copy=None):
        """
        Check if an object is an array-like, and convert it to this array's type if so.

        Args:
            obj: object to convert (BaseArr, np.ndarray, or raises TypeError)
            copy (bool): whether to copy the data; passed through to asnew()
        """
        if isinstance(obj, BaseArr):
            return obj
        elif isinstance(obj, np.ndarray):
            if obj.dtype == self.dtype:
                return self.asnew(obj, copy=copy)
            cls, nan = self._math_result_type(obj.dtype) # The dtype has changed, e.g. np.float64(30) < age gives a BoolArr
            new = self.asnew(obj, cls=cls, copy=copy)
            new.nan = nan
            new.nan_eq = (nan == nan)
            return new
        else:
            self._type_error(obj)
        return obj

    def _to_raw(self, obj):
        """ Convert an Arr object to raw values, leaving NumPy arrays unchanged (for argument normalization) """
        if isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, Arr):
            return obj.raw
        else:
            self._type_error()
        return obj

    def _to_values(self, obj):
        """ Convert an Arr object to indexed values, leaving NumPy arrays unchanged (for argument normalization) """
        if isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, Arr):
            return obj.values
        else:
            self._type_error(obj)
        return obj

    def asnew(self, arr=None, cls=None, name=None, copy=None):
        """ Duplicate and copy (rather than link) data, optionally resetting the array """
        if cls is None:
            cls = self.__class__
        if copy is None:
            copy = True if (arr is None) or (arr.size != self.raw.size) else False
        if arr is None:
            arr = self.values
        new = object.__new__(cls) # Create a new Arr instance
        new.__dict__ = self.__dict__.copy() # Copy pointers
        new.dtype = arr.dtype # Set to correct dtype
        new.name = name # In most cases, the asnew Arr has different values to the original Arr so the original name no longer makes sense
        new.shared = None # It has values of its own
        new.raw = self.raw # Stored, if it was made when read (see IndexArr.raw)
        if copy:
            new.raw = np.empty(new.raw.shape, dtype=new.dtype) # Copy values, breaking reference
            new.raw[new.auids] = arr
        else:
            if isinstance(arr, np.ndarray):
                new.raw = arr
            elif isinstance(arr, Arr):
                new.raw = arr.raw
            else:
                self._type_error(arr)
        return new

    def astype(self, cls, copy=False):
        """ Convert the Arr type, e.g. `arr.astype(ss.IntArr)`; other types (e.g. `int`) return a NumPy array """
        if not (isinstance(cls, type) and issubclass(cls, Arr)):
            return self.values.astype(cls)

        # Create the new object
        new = object.__new__(cls) # Create a new Arr instance
        new.__dict__ = self.__dict__.copy() # Copy pointers
        new.raw = self.raw # Stored, if it was made when read (see IndexArr.raw)

        # Reset Arr properties
        tmp = cls() # dtype and nan are only defined on instances
        new.dtype = tmp.dtype # Set to correct dtype
        new.nan = tmp.nan
        new.nan_eq = (new.nan == new.nan)

        # Optionally copy the array values (slow), and update the dtype
        new.raw = new.raw.astype(new.dtype, copy=copy)
        return new

    def true(self):
        """ Efficiently convert truthy values to UIDs """
        if self.ndim > 1: self._uids_error()
        auids = self.auids
        if auids.size >= numba_indexing: # Branchless Numba compaction wins above ~1k elements
            return uids(nb_true_raw(auids.view(np.ndarray), self.raw, True))
        return auids[self.values.astype(bool)]

    def false(self):
        """ Reverse of true(); return UIDs of falsy values """
        if self.ndim > 1: self._uids_error()
        auids = self.auids
        if auids.size >= numba_indexing:
            return uids(nb_true_raw(auids.view(np.ndarray), self.raw, False))
        return auids[~self.values.astype(bool)]

    def _uids_error(self):
        errormsg = f'Cannot convert 2D array "{self.name}" to UIDs; use e.g. arr.col(v).uids instead'
        raise ValueError(errormsg)

    def col(self, v):
        """
        Return column `v` of a 2D array as a 1D `Arr`, e.g. `(imm.col(2) > 0.5).uids` or `imm.col('mutant')`

        For arrays with more dimensions, returns an `Arr` with one fewer dimension.
        The new `Arr` is a view, so setting its values also changes the original array.
        It is not linked to `People`, so should not be stored (since it will not grow).
        """
        new = self.asnew(self.raw[:, self._col_index(v)], name=f'{self.name}[{v}]', copy=False)
        new.columns = None
        new.item_shape = self.item_shape[1:]
        new.ndim = self.ndim - 1
        if new.ndim == 1:
            del new._index # Revert to the class method
        return new

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
    """ Subclass of `ss.Arr` with defaults for floats and ints.

    Note: Starsim does not support integer arrays by default since they introduce
    ambiguity in dealing with NaNs, and float arrays are suitable for most purposes.
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, dtype=ss_float, nan=np.nan, **kwargs) # Do not allow dtype or nan to be overwritten
        return


class ModifiableArr(FloatArr):
    """
    A float array that modifiers act on: a declared disease's `rel_sus` and `rel_trans`, as agents

    Reading it gives the base values, which a modifier never changes; `.effective` gives
    the values with the modifiers that are active now applied, which is what the model
    uses. `sim.what_modifies()` lists the modifiers.
    """
    engine = None # The engine that applies the modifiers; set by it

    @property
    def effective(self):
        """ The values with the modifiers that are active now applied """
        mult = 1.0 if self.engine is None else self.engine.mult_raw(self.name)
        return self.asnew(self.raw*mult, cls=FloatArr, copy=False)


class StepArr(Arr):
    """
    Step numbers (time indices), stored compactly: a declared disease's `ti_X` and `ti_exit_X`, as agents, and `people.ti_dead` and `people.ti_removed`

    The values are stored as 16-bit integers when the module's run has fewer than
    32,000 steps (and 32-bit otherwise), which is a quarter of the memory of a
    float time and half of a float32. Reading them gives float32 step numbers, with
    NaN for "never" (not set) and `inf` for a time after the end of the run, so
    `ti_I == sim.ti`, `ti_I > 0`, and `(sim.ti - ti_I)*sim.t.dt` all give the right
    answer. A time between two steps is stored as the first step at or after it.
    `.years` converts them to years; `.raw` has the stored integers, where `.nan`
    means not set and `.never` means after the end of the run.

    With `late=True` (as for `people.ti_dead`), "not set" is stored as the largest
    integer, the same as after the end of the run, so `raw <= ti` is whether the time
    has come, with no special case; both read as NaN.

    Example:
        ```python
        sim = ss.Sim(ss.SIR(), unit='day', stop=100).run()
        sir = sim.diseases.sir
        infected_today = (sir.ti_I == sim.ti).uids
        years = sir.ti_I.years # The year of infection of each agent (NaN if never infected)
        ```
    """
    def __init__(self, name=None, dtype=np.int32, late=False, **kwargs):
        self.late = late
        super().__init__(name=name, dtype=dtype, nan=self.sentinels(dtype)[0], **kwargs)
        self.lowest, self.never = self.sentinels(dtype)[1:]
        self.nan_eq = False # Read as floats, so NaN is found with np.isnan()
        self.t = None # The module's timeline, for .years; set by the engine
        return

    def sentinels(self, dtype):
        """ The stored integers for not set, the lowest step, and after the end of the run """
        info = np.iinfo(dtype)
        return (info.max if self.late else info.min), info.min + (0 if self.late else 1), info.max

    def fit(self, t):
        """ Use 16-bit integers if the module's run (on timeline `t`) fits in them """
        self.t = t
        dtype = np.int16 if t.npts < 32_000 else np.int32
        if dtype != self.dtype:
            nan = self.raw == self.nan
            self.dtype = dtype
            self.nan, self.lowest, self.never = self.sentinels(dtype)
            self.raw = self.raw.astype(dtype)
            self.raw[nan] = self.nan
        return

    def to_float(self, vals):
        """ Convert stored step numbers to float32: NaN where not set, and inf after the end of the run """
        vals = np.asarray(vals)
        out = vals.astype(ss_float)
        out[vals == self.never] = np.inf
        out[vals == self.nan] = np.nan # After inf, since with late=True they are the same
        return out if out.ndim else out[()]

    def from_float(self, vals):
        """ Convert step numbers to the stored integers: NaN to not set, a time between steps to the next step, and inf (or beyond the integers) to after the end of the run """
        if isinstance(vals, numbers.Integral): # The usual case, e.g. the step now
            return min(max(int(vals), self.lowest), self.never)
        elif isinstance(vals, float) and vals != vals: # NaN, e.g. a scheduled exit forgotten
            return self.nan
        vals = np.asarray(vals)
        if vals.dtype.kind in 'iub':
            return np.clip(vals, self.lowest, self.never)
        out = np.ceil(vals, out=np.empty(vals.shape)) # An array, even for one value
        np.clip(out, self.lowest, self.never, out=out) # NaN stays NaN here
        out[np.isnan(out)] = self.nan
        return out.astype(self.dtype)

    def floats(self):
        """ The values as a `FloatArr` of step numbers, for arithmetic and comparisons """
        new = self.asnew(self.to_float(self.raw), cls=FloatArr)
        new.dtype, new.nan, new.nan_eq = ss_float, np.nan, False
        return new

    @property
    def years(self):
        """ The values as years (e.g. 2020.5), with NaN where not set """
        steps = self.to_float(self.values).astype(float)
        return self.t.yearvec[0] + steps*self.t.dt_year

    def __getitem__(self, key):
        return self.to_float(super().__getitem__(key))

    def __setitem__(self, key, value):
        super().__setitem__(key, self.from_float(value))
        return

    @property
    def values(self):
        return self.to_float(super().values)

    def set(self, uids, new_vals=None):
        new_vals = self.nan if new_vals is None and self.default is None else self.from_float(new_vals if new_vals is not None else self.default)
        self.raw[uids] = new_vals
        return new_vals

    def _boolmath(self, op, other=None, inplace=False):
        if inplace:
            errormsg = f'"{self.name}" holds step numbers, which cannot be changed in place by a comparison'
            raise TypeError(errormsg)
        if not self.initialized:
            return super()._boolmath(op, other)
        return self.floats()._boolmath(op, other)

    def _math(self, op, other, reverse=False, inplace=False):
        if not self.initialized and not inplace:
            return super()._math(op, other, reverse=reverse)
        result = self.floats()._math(op, other, reverse=reverse)
        if inplace:
            self.raw[:] = self.from_float(result.raw)
            return self
        return result


class Unkept(BaseArr):
    """
    A declared disease's time of entering a state (`ti_X`) that is not kept, since nothing in the sim reads it (see `ss.Disease.times_read()`)

    Setting it does nothing, and reading it raises `ss.TimeNotKept`, which says how to keep it.
    It takes no memory, and is not one of the module's states.
    """
    def __init__(self, name, mod):
        self.name = name
        self.mod = mod
        return

    @property
    def values(self):
        errormsg = (f'{self.mod.name}.{self.name} is not kept, since nothing in the sim read it when it was initialized (to save memory).\n'
                    f'     Set ss.options.keep_times = True before the sim is initialized to keep the time of entering every state.')
        raise ss.TimeNotKept(errormsg)

    raw = values

    def __getattr__(self, attr):
        if attr.startswith('_'): # E.g. __deepcopy__ or __array__, which are looked for
            raise AttributeError(attr)
        return self.values

    def __setitem__(self, key, value):
        return # Not kept

    def __repr__(self):
        return f'<{self.name}: not kept>'


class GroupAges(FloatArr):
    """
    `people.age` in a compartmental run: one row per group, at the middle of its age group

    Comparing it with an age inside a group (`sim.people.age > 50` with groups [0, 15, 65])
    raises, since the answer would be the same for everyone in the group; an edge works.
    """
    edges = None # The edges of the age groups; set by the sim

    def _boolmath(self, op, other=None, inplace=False):
        if op not in ['>', '<', '>=', '<='] or not isinstance(other, numbers.Number):
            return super()._boolmath(op, other, inplace)
        inside = other > self.edges[0] and other not in self.edges # Not an edge, and above the lowest one
        if inside:
            edges = sorted(set(self.edges.tolist() + [other]))
            edges = [int(e) if e == int(e) else e for e in edges]
            raise ss.CapabilityError(f'a function compares people.age with {other:n}, but the model\'s age groups are {[int(e) if e == int(e) else e for e in self.edges]} and {other:n} falls inside a group (each group has the age of its middle).\n'
                                     f'     Fix: add {other:n} as a bin edge, stratify(age={edges}), or run with solver="abm".')
        return super()._boolmath(op, other, inplace)


class IntArr(Arr):
    """ Subclass of `ss.Arr` with defaults for ints.

    Note: Because integer arrays do not handle NaN values natively, users are
    recommended to use `ss.FloatArr()` in most cases instead.
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, dtype=ss_int, nan=int_nan, **kwargs)
        return


class BoolArr(Arr):
    """
    Subclass of `ss.Arr` with defaults for booleans.

    Examples:
        ```python
        # Create a standalone BoolArr
        infected = ss.BoolArr('infected', mock=5)
        infected[ss.uids([0, 2, 4])] = True
        infected.count()  # Returns 3
        infected.uids     # Returns ss.uids([0, 2, 4])
        ```
    """
    def __init__(self, name=None, **kwargs): # No good NaN equivalent for bool arrays
        super().__init__(name=name, dtype=ss_bool, nan=False, **kwargs)
        return

    def __eq__(self, other):
        if isinstance(other, uids):
            out = ~self.raw
            out[other] = self.raw[other]
            return self.asnew(out, cls=BoolArr, copy=False)
        return self._boolmath('==', other)

    def __ne__(self, other):
        if isinstance(other, uids):
            return self.__xor__(other)
        return self._boolmath('!=', other)

    def __and__(self, other):
        """ Bitwise AND

        Supports BoolArr and uids as inputs. If input is uids, it is treated as a BoolArr where
        the provided indices are True and all other elements are False.
        """
        if isinstance(other, uids):
            out = np.zeros_like(self.raw, dtype=bool)
            out[other] = self.raw[other]
            return self.asnew(out, cls=BoolArr, copy=False)
        return self._boolmath('&', other)

    def __or__(self, other):
        """ Bitwise OR"""
        if isinstance(other, uids):
            out = self.raw.copy()
            out[other] = True
            return self.asnew(out, cls=BoolArr, copy=False)
        return self._boolmath('|', other)

    def __xor__(self, other):
        """ Bitwise XOR: returns BoolArr when operating on arrays, or uids when operating on uids (set symmetric difference) """
        if isinstance(other, uids):
            out = self.raw.copy()
            out[other] = ~out[other]
            return self.asnew(out, cls=BoolArr, copy=False)
        return self._boolmath('^', other)

    def __invert__(self):
        """ Bitwise NOT """
        if not self.initialized:
            return ss.conditions.lazy('~', self)
        return self.asnew(~self.raw, cls=BoolArr, copy=False)

    # For BoolArr instances implement true in-place operators that reuse the memory from the underlying array
    # This allows in-place operations to be used directly on `BoolState` instances without breaking references

    def __iand__(self, other):
        """ In-place AND """
        if isinstance(other, uids):
            tmp = self.raw[other]
            self.raw[:] = False
            self.raw[other] = tmp
            return self
        else:
            return self._boolmath('&', other, inplace=True)

    def __ior__(self, other):
        """ In-place OR """
        if isinstance(other, uids):
            self.raw[other] = True
            return self
        else:
            return self._boolmath('|', other, inplace=True)

    def __ixor__(self, other):
        """ In-place XOR """
        if isinstance(other, uids):
            self.raw[other] = ~self.raw[other]
            return self
        else:
            return self._boolmath('^', other, inplace=True)

    # BoolArr cannot store NaNs so report all entries as being not-NaN
    @property
    def isnan(self):
        """ Always False for BoolArr (booleans cannot be NaN). """
        return self.asnew(np.full_like(self.values, fill_value=False), cls=BoolArr)

    @property
    def notnan(self):
        """ Always True for BoolArr (booleans cannot be NaN). """
        return self.asnew(np.full_like(self.values, fill_value=True), cls=BoolArr)

    @property
    def uids(self):
        """ Alias to Arr.true """
        return self.true()

    def split(self):
        """ Return UIDs of values that are true and false as separate arrays """
        t_uids = self.true() # Could be made more performant by doing the loop once, but not heavily used
        f_uids = self.false()
        return t_uids, f_uids


class BoolState(BoolArr):
    """ A boolean array being used as a state.

    Although functionally identical to `BoolArr`, a `BoolState` is handled differently in
    terms of automation: specifically, results are automatically generated from a
    `BoolState` (but not a `BoolArr`).

    BoolStates are typically used to keep track of externally-facing variables (e.g.
    `disease.susceptible`), while BoolArrs can be used to keep track of internal
    ones (e.g. `disease.has_immunity`). A BoolState named "susceptible" will automatically
    generate a result named "n_susceptible", for example.

    Note on terminology: the term "states" is often used to refer to *all* `ss.Arr` objects,
    (e.g. in `module.define_states()`, whether or not they are BoolStates.
    """
    watch = None # What keeps count of the agents in the state (for a state of a declared disease run as agents, its engine), which is told of every change: see `Watch`

    @property
    def n(self):
        """ The number of people in the state: its agents, each counted by its weight (`people.scale`), as `I.n` in a rate counts them, and as a compartmental run's state gives it """
        uids = self.uids
        engine = getattr(self.watch, 'engine', None)
        if engine is not None: # A declared disease's engine counts people, with each agent's weight and the population's scale
            return float(engine.n_of(uids))*engine.scale
        return float(len(uids))

    def __init__(self, name=None, **kwargs):
        if kwargs.get('columns') is not None: # Automatic results are 1D, so don't allow 2D states
            errormsg = f'BoolState "{name}" cannot be 2D since it automatically generates results; use ss.BoolArr instead'
            raise ValueError(errormsg)
        super().__init__(name=name, **kwargs)
        return

    def asnew(self, arr=None, cls=None, name=None, copy=None):
        """ As `Arr.asnew()`; a copy whose values are not the state's own is not counted (see `watch`) """
        new = super().asnew(arr=arr, cls=cls, name=name, copy=copy)
        if new.raw is not self.raw and new.__dict__.get('watch') is not None:
            new.__dict__['watch'] = None
        return new

    def _counter(self):
        """ The engine keeping count of who is in the state, if it is counting now """
        watch = self.watch
        if watch is None or watch.engine.live is None:
            return None
        return watch.engine

    def __setitem__(self, key, value):
        engine = self._counter()
        if engine is None: # The usual case
            return super().__setitem__(key, value)
        if not isinstance(key, uids):
            key = self._convert_key(key)
        inds = np.atleast_1d(np.asarray(key))
        if len(inds) > 1 and not np.all(inds[1:] > inds[:-1]): # Each agent once, so a change is counted once
            inds = np.unique(inds)
        old = self.raw[inds]
        self.raw[key] = value
        engine.wrote(self.name, inds, old, self.raw[inds])
        return

    def set(self, uids, new_vals=None):
        """ Set the values for the specified UIDs, telling whatever keeps count of the state (see `__setitem__()`) """
        engine = self._counter()
        if engine is None:
            return super().set(uids, new_vals=new_vals)
        inds = np.atleast_1d(np.asarray(uids))
        old = self.raw[inds]
        out = super().set(uids, new_vals=new_vals)
        engine.wrote(self.name, inds, old, self.raw[inds])
        return out

    def grow(self, new_uids=None, new_vals=None):
        """ Add new agents (see `Arr.grow()`); they were not counted before, so whatever keeps count of the state counts those who start in it """
        engine = self._counter()
        if engine is None:
            return super().grow(new_uids, new_vals=new_vals)
        watch = self.watch
        self.watch = None # The values the new agents' rows had (if they existed already) are not a change
        try:
            super().grow(new_uids, new_vals=new_vals)
        finally:
            self.watch = watch
        if new_vals is not None or self.default: # Otherwise they start out of the state (the usual case)
            inds = np.atleast_1d(np.asarray(new_uids))
            engine.wrote(self.name, inds, np.zeros(len(inds), dtype=bool), self.raw[inds])
        return

    def _changed_all(self):
        """ After a change to the whole array (an in-place operation), whatever keeps count of the state counts it again """
        if self.watch is not None:
            self.watch.engine.regroup()
        return

    def __iand__(self, other):
        out = super().__iand__(other)
        self._changed_all()
        return out

    def __ior__(self, other):
        out = super().__ior__(other)
        self._changed_all()
        return out

    def __ixor__(self, other):
        out = super().__ixor__(other)
        self._changed_all()
        return out


class Watch:
    """
    The link from a state to the engine that keeps count of who is in it (`BoolState.watch`), which is told of
    every change to it (see `ss.solvers.Agents.wrote()`). It is never searched for distributions (see `ss.distributions.no_dists()`),
    so the link does not change where they are found, which gives each its random numbers.
    """
    __slots__ = ['engine']

    def __init__(self, engine):
        self.engine = engine
        return


class IndexArr(Arr):
    """
    A special class of Arr used for UIDs and RNG IDs; not to be used as a general integer array (for that, use IntArr)

    The values are stored as 32-bit integers, to save memory (so there can be up to 2^31 agents), and read as
    UIDs (`ss.uids`, which keep 32-bit integers as they are).
    """
    def __init__(self, name=None, label=None, shared=False):
        super().__init__(name=name, dtype=np.int32, default=None, nan=-1, label=label, skip_init=True, shared=shared)
        return

    @property
    def raw(self):
        """ The stored values; while they are those of `people.uid` (see `shared` in `ss.Arr`), the UIDs, made when read """
        if isinstance(self.shared, UIDArr):
            return self.shared.raw
        return self._raw

    @raw.setter
    def raw(self, raw):
        self._raw = None if isinstance(self.shared, UIDArr) else raw
        return

    def shared_raw(self):
        return None if isinstance(self.shared, UIDArr) else super().shared_raw() # Nothing to store: see raw

    def __getitem__(self, key):
        out = super().__getitem__(key)
        return uids(out) if isinstance(out, np.ndarray) else ss_int(out)

    @property
    def values(self):
        """ The values of the active agents, as UIDs """
        return uids(super().values)

    @property
    def uids(self):
        """ Alias to self.values, to allow Arr.uids like BoolArr """
        return self.values

    def grow(self, new_uids=None, new_vals=None):
        """ Change the size of the array """
        if new_uids is None and new_vals is not None: # Used as a shortcut to avoid needing to supply twice
            new_uids = new_vals
        if new_vals is not None and np.size(new_vals) and np.max(new_vals) >= 2**31:
            errormsg = f'"{self.name}" is stored as 32-bit integers, so it cannot hold {np.max(new_vals):n}: there can be at most 2^31 agents'
            raise ValueError(errormsg)
        super().grow(new_uids=new_uids, new_vals=new_vals)
        return


class UIDArr(IndexArr):
    """
    The UIDs of all agents (`people.uid`), which take no memory: a UID is the agent's index, so `raw` is made when read

    It is `np.arange()` up to the number of agents made, and -1 (not set) in the room for
    agents not yet made, as a stored `IndexArr` would be. Code that only needs its length
    reads `len_used` or `len_tot` instead.
    """
    @property
    def raw(self):
        raw = np.arange(self.len_tot, dtype=np.int32)
        raw[self.len_used:] = self.nan
        raw.flags.writeable = False # Nothing is stored, so nothing can be set
        return raw

    @raw.setter
    def raw(self, raw):
        return # Nothing is stored

    def grow(self, new_uids=None, new_vals=None):
        """ Add new agents, whose UIDs are the next ones """
        if new_uids is None:
            new_uids = new_vals
        n_new = len(new_uids)
        if self.len_used + n_new >= 2**31:
            errormsg = f'"{self.name}" is stored as 32-bit integers, so there can be at most 2^31 agents'
            raise ValueError(errormsg)
        self.len_used += n_new
        if self.len_used > self.len_tot:
            self.len_tot += max(n_new, self.len_tot//5) # The same room as the other arrays: see Arr.grow()
        return


class uids(np.ndarray):
    """ Class to specify that integers should be interpreted as UIDs.

    For all practical purposes, behaves like a NumPy integer array. However,
    has additional methods `uids.concatenate()`, `uids.remove()`, and
    `uids.intersect()` to simplify common UID operations.

    The following operators are supported:

    - `+` / `+=`: if the RHS is a `uids`, concatenation (equivalent to `uids.concatenate()`); otherwise element-wise addition (e.g. `uids([1,2]) + 1` → `uids([2,3])`). Concatenation preserves duplicates.
    - `-` / `-=`: set difference — equivalent to `uids.remove()`. Duplicate entries in the original will also be removed (uses `np.setdiff1d` internally).
    - `|` / `|=`: union (unique elements from both arrays). Note that duplicate entries in the original are **removed** (use `+` instead to keep them).
    - `&` / `&=`: intersection
    - `^` / `^=`: symmetric difference

    Note: because `uids` is a subclass of `np.ndarray`, set-like in-place operators
    (`+=` with a `uids` RHS, `-=`, `|=`, `&=`, `^=`) must return a new array and rebind
    the variable. However, `+=` with a scalar or array RHS modifies the array in-place
    and preserves `id(self)`.

    Examples:
        ```python
        a = ss.uids([1, 2, 3])
        b = ss.uids([3, 4, 5])
        a + b   # Concatenate: uids([1, 2, 3, 3, 4, 5])
        a | b   # Union:       uids([1, 2, 3, 4, 5])
        a & b   # Intersect:   uids([3])
        a - b   # Difference:  uids([1, 2])
        ```
    """
    def __new__(cls, arr=None):
        if isinstance(arr, np.ndarray): # Shortcut to typical use case, where the input is an array
            return cls._ensure_int(arr)
        elif isinstance(arr, BoolArr): # Shortcut for arr.uids
            return arr.uids
        elif isinstance(arr, set):
            return cls._ensure_int(np.array(list(arr)))
        elif arr is None: # Shortcut to return empty
            return np.empty(0, dtype=ss_int).view(cls)
        elif isinstance(arr, (int, np.integer)): # Convert e.g. ss.uids(0) to ss.uids([0])
            arr = [arr]
        return cls._ensure_int(np.asarray(arr)) # Handle everything else

    def concatenate(*args):  # pylint: disable=no-self-argument  # intentionally no `self` so it works as both instance and class method
        """
        Concatenate uids into a single instance

        Can be called as either an instance method or an unbound class method::

            x.concatenate(y)               # instance method
            x.concatenate(None)            # None is treated as empty
            ss.uids.concatenate(x, y)      # unbound; identical to instance call
            ss.uids.concatenate([x, y])    # single iterable of arrays
            ss.uids.concatenate([None, x]) # None elements are filtered out
            ss.uids.concatenate()          # returns empty ss.uids()
        """
        # Handle a single input argument: my_uids.concatenate([uids1, uids2])
        if len(args) == 1 and np.iterable(args[0]) and (not isinstance(args[0], (uids, np.ndarray))):
            arrays = list(args[0]) if not isinstance(args[0], list) else args[0] # Handle corner case of [[]]
        
        # Handle multiple input arguments: my_uids.concatenate(uids1, uids2)
        else:
            arrays = list(args)
        
        # Remove any None values
        valid = [a for a in arrays if a is not None]

        # If non-empty arrays remain, concatenate, always using int type
        if valid:
            out = uids._ensure_int(np.concatenate(valid))
        else:
            out = uids()
        return out

    def concat(self, other):
        """ Deprecated — use `uids.concatenate()` instead """
        ss.warn('uids.concat() is deprecated; use uids.concatenate() instead', category=DeprecationWarning)
        return self.concatenate(other)

    @classmethod
    def cat(cls, *args):
        """ Deprecated — use `uids.concatenate()` instead """
        ss.warn('uids.cat() is deprecated; use uids.concatenate() instead', category=DeprecationWarning)
        return uids.concatenate(*args)

    @classmethod
    def _ensure_int(cls, arr):
        """
        Return an integer-typed uids view, validating that the values are valid UIDs.

        UIDs are integer identifiers used for array indexing, so they must be integer-valued.
        This is called both by the constructor and by operations that can change the dtype:
        numpy may promote the return type (e.g. `np.intersect1d(uids, [])` returns a `float`),
        and the constructor may receive float input directly (e.g. `ss.uids([1.5])`).

        Integer input of the correct type is returned as a view with no copy. Other input is
        accepted only if every value is finite and integer-valued (e.g. an empty list, or `2.0`);
        genuinely fractional or non-finite values (`1.5`, `nan`, `inf`) are rejected, since they
        can never index an array.
        """
        if arr.dtype == ss_int or arr.dtype == ss_uid:
            return arr.view(cls) # Fast path: already the correct type (64-bit, or 32-bit as people.auids are), no copy needed
        if arr.dtype.kind not in ('i', 'u'):
            # Non-integer dtype: isfinite rejects nan/inf, floor rejects fractions
            if not (np.isfinite(arr).all() and (arr == np.floor(arr)).all()):
                raise TypeError(f'UIDs must be integers, but got non-integer values: {arr}')
        return arr.astype(ss_int).view(cls) # Normalize integer width, or cast validated floats

    def remove(self, other, **kw):
        """ Remove provided UIDs from current array"""
        if isinstance(other, BoolArr):
            other = other.uids
        return self._ensure_int(np.setdiff1d(self, other, **kw))

    def intersect(self, other, **kw):
        """ Keep only UIDs that are also present in the other array """
        if isinstance(other, BoolArr):
            other = other.uids
        return self._ensure_int(np.intersect1d(self, other, **kw))

    def union(self, other, **kw):
        """ Return all UIDs present in both arrays """
        if isinstance(other, BoolArr):
            other = other.uids
        return self._ensure_int(np.union1d(self, other, **kw))

    def xor(self, other, **kw):
        """ Return UIDs present in only one of the arrays """
        if isinstance(other, BoolArr):
            other = other.uids
        return self._ensure_int(np.setxor1d(self, other, **kw))

    def to_numpy(self):
        """ Return a view as a standard NumPy array """
        return self.view(np.ndarray)

    def unique(self, return_index=False):
        """ Return unique UIDs; equivalent to np.unique() """
        if return_index:
            arr, index = np.unique(self, return_index=True)
            return self._ensure_int(arr), index # index is a position array, not a uids
        else:
            return self._ensure_int(np.unique(self))

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        # Guard 1: non-integer dtype → return a plain ndarray.
        # Example: `uids_a == uids_b` produces a bool array. A boolean uids should
        # never exist — uids are integer indices only — and returning one would cause
        # `~(a == b)` to hit our __invert__ guard instead of doing a normal bitwise NOT.
        if out_arr.dtype.kind not in ('i', 'u'):
            return out_arr.view(np.ndarray)
        # Guard 2: scalar reductions → return a plain Python int.
        # Example: `uids.max()` should behave like `uids[0]` (a bare integer), not
        # return a 0-d uids whose subsequent use in arithmetic (e.g. `max + 1`) would
        # incorrectly route through our __add__ concatenation operator.
        if return_scalar:
            return out_arr.item()
        return super().__array_wrap__(out_arr, context, return_scalar)

    def _uid_op_error(self, name):
        msg = f'{name}() is not meaningful for {self.__class__.__name__}: UIDs are identifiers, not numeric values'
        raise TypeError(msg)

    def sum(self, *args, **kwargs):     self._uid_op_error('sum')
    def mean(self, *args, **kwargs):    self._uid_op_error('mean')
    def std(self, *args, **kwargs):     self._uid_op_error('std')
    def var(self, *args, **kwargs):     self._uid_op_error('var')
    def prod(self, *args, **kwargs):    self._uid_op_error('prod')
    def cumsum(self, *args, **kwargs):  self._uid_op_error('cumsum')
    def cumprod(self, *args, **kwargs): self._uid_op_error('cumprod')

    # Implement collection of operators
    def __add__(self, other):
        if isinstance(other, uids):
            return self.concatenate(other)   # concatenation: uids([1,2]) + uids([3]) → uids([1,2,3])
        return super().__add__(other)   # element-wise:  uids([1,2]) + 1        → uids([2,3])
    def __and__(self, other) : return self.intersect(other)
    def __or__(self, other)  : return self.union(other)
    def __sub__(self, other) : return self.remove(other)
    def __xor__(self, other) : return self.xor(other)
    def __invert__(self)     : raise Exception(f"Cannot invert an instance of {self.__class__.__name__}. One possible cause is attempting `~x.uids` - use `x.false()` or `(~x).uids` instead")

    # For set-like operations (uids RHS) the size changes, so in-place must return a new object
    # and rebind the variable — same semantics as in-place operations on an immutable type.
    # For scalar/array RHS (element-wise arithmetic) the size is unchanged, so we can modify
    # in-place and preserve id(self).
    def __iadd__(self, other):
        if isinstance(other, uids):
            return self.__add__(other)   # new object — size changes
        np.add(self, other, out=self)    # in-place — size unchanged, id preserved
        return self
    def __iand__(self, other): return self.__and__(other)
    def __ior__(self, other) : return self.__or__(other)
    def __isub__(self, other): return self.__sub__(other)
    def __ixor__(self, other): return self.__xor__(other)


class BooleanOperationError(NotImplementedError):
    """ Raised when a logical operation is performed on a non-logical array """
    def __init__(self, arr):
        msg = f'Logical operations are only valid on Boolean arrays, not {arr.dtype}'
        super().__init__(msg)
