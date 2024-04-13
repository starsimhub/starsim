"""
Define array-handling classes, including agent states
"""

import numpy as np
import starsim as ss

# Shorten these for performance
ss_float = ss.dtypes.float
ss_int   = ss.dtypes.int
ss_bool  = ss.dtypes.bool

__all__ = ['check_dtype', 'Arr', 'FloatArr', 'IntArr', 'BoolArr', 'IndexArr']


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


class Arr:

    # __slots__ = ('values', 'uid', 'default', 'name', 'label', '_data', 'values', 'initialized')

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
            skip_init (bool): Whether to skip initialization with the People object (used for uid and rngid states)
        """
        if coerce:
            dtype = check_dtype(dtype, default)
        
        # Set attributes
        self.name = name
        self.label = label or name
        self.default = default
        self.nan = nan
        
        # Properties that are initialized later
        self._data = np.empty(0, dtype=dtype)
        self.people = None
        self.len_used = 0
        self.len_tot = 0
        self.initialized = skip_init
        return
    
    def __repr__(self):
        string = f'<State {str(self.name)}, dtype={self.dtype}, len={len(self)}>\n'
        string += self._data.__repr__()
        return string
    
    @property
    def values(self):
        return self._data[self.aliveinds]
    
    @property
    def aliveinds(self):
        try:
            return self.people.aliveinds
        except:
            print('TEMP: Could not return aliveinds!')
            return np.arange(len(self._data))
    
    def __len__(self):
        try:
            return len(self.aliveinds)
        except:
            return len(self._data)
    
    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == ss_int: # Check that it's (likely) UIDs
            return self._data[key]
        else:
            return self.values[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, np.ndarray) and key.dtype == ss_int:
            self._data[key] = value
        else:
            newkey = self.aliveinds[key]
            self._data[newkey] = value
            
    def __getattr__(self, attr):
        return getattr(self.values, attr)

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
        self._data[uids] = new_vals
        return new_vals
    
    def set_nan(self, uids):
        self._data[uids] = self.nan
        return
    
    def grow(self, uids, new_vals=None):
        """
        Add new agents to an Arr

        This method is normally only called via `People.grow()`.

        Args:
            uids: Numpy array of UIDs for the new agents being added
        """
        orig_len = self.len_used
        n_new = len(uids)
        if orig_len + n_new > self.len_tot:
            n_grow = max(n_new, self.len_tot//2)  # Minimum 50% growth, since growing arrays is slow
            new_empty = np.empty(n_grow, dtype=self.dtype) # 10x faster than np.zeros()
            self._data = np.concatenate([self._data, new_empty], axis=0)
            self.len_tot = len(self._data)
        
        self.len_used += n_new  # Increase the count of the number of agents by `n` (the requested number of new agents)
        self.set_new(uids, new_vals=new_vals) # Assign new default values to those agents
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
        if self.initialized:
            return

        people = sim.people
        assert people.initialized, 'People must be initialized before initializing states'
        
        # Connect any distributions in the default to RNGs in the Sim
        if isinstance(self.default, ss.Dist):
            self.default.initialize(module=self, sim=sim)

        # Establish connection with the People object
        people.register_state(self)
        self.people = people

        # Populate initial values
        self.grow(len(people))
        self.initialized = True
        return


class FloatArr(Arr):
    """ Subclass of Arr with defaults for floats """
    def __init__(self, name, default=None, nan=np.nan, label=None, skip_init=False):
        super().__init__(name=name, dtype=ss_float, default=default, nan=nan, label=label, coerce=False, skip_init=skip_init)
        return
    
    
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
    
    
class IndexArr(IntArr):
    """ A special class of IndexArr used for UIDs and RNG IDs """
    def __init__(self, name, label=None):
        super().__init__(name=name, dtype=ss_int, default=None, nan=ss.intnan, label=label, coerce=False, skip_init=True)
        return