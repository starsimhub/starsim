"""
Base classes for *sim models
"""

import numpy as np
import sciris as sc
import functools
from . import utils as ssu
from . import misc as ssm
from . import settings as sss
from .version import __version__

# Specify all externally visible classes this file defines
__all__ = ['State', 'BasePeople']

# Default object getter/setter
obj_set = object.__setattr__
base_key = 'uid'  # Define the key used by default for getting length, etc.


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


# %% Define simulation classes


def set_metadata(obj, **kwargs):
    """ Set standard metadata for an object """
    obj.created = kwargs.get('created', sc.now())
    obj.version = kwargs.get('version', __version__)
    obj.git_info = kwargs.get('git_info', ssm.git_info())
    return




# %% Define people classes

class State(sc.prettyobj):
    def __init__(self, name, dtype, fill_value=0, shape=None, distdict=None, label=None):
        """
        Args:
            name: name of the result as used in the model
            dtype: datatype
            fill_value: default value for this state upon model initialization
            shape: If not none, set to match a string in `pars` containing the dimensionality
            label: text used to construct labels for the result for displaying on plots and other outputs
        """
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value
        self.shape = shape
        self.distdict = distdict
        self.is_dist = distdict is not None # Set this by default, but allow it to be overridden
        self.label = label or name
        return

    @property
    def ndim(self):
        return len(sc.tolist(self.shape)) + 1
    
    def new(self, n):
        if self.is_dist:
            return self.new_dist(n)
        else:
            return self.new_scalar(n)

    def new_scalar(self, n):
        shape = sc.tolist(self.shape)
        shape.append(n)
        out = np.full(shape, dtype=self.dtype, fill_value=self.fill_value)
        return out
    
    def new_dist(self, n):
        shape = sc.tolist(self.shape)
        shape.append(n)
        out = ssu.sample(**self.distdict, size=tuple(shape))
        return out


base_states = ssu.named_dict(
    State('uid', sss.default_int),
    State('age', sss.default_float),
    State('female', bool, False),
    State('debut', sss.default_float),
    State('dead', bool, False),
    State('ti_dead', sss.default_float, np.nan),  # Time index for death
    State('scale', sss.default_float, 1.0),
)


class BasePeople(sc.prettyobj):
    """
    A class to handle all the boilerplate for people -- note that as with the
    BaseSim vs Sim classes, everything interesting happens in the People class,
    whereas this class exists to handle the less interesting implementation details.
    """

    def __init__(self, n, states=None, *args, **kwargs):
        """ Initialize essential attributes """

        super().__init__(*args, **kwargs)
        self.initialized = False
        self.version = __version__  # Store version info

        # Initialize states, networks, modules
        self.states = sc.mergedicts(base_states, states)
        self.networks = ssu.named_dict()
        self._modules = sc.autolist()

        # Private variables relating to dynamic allocation
        self._data = dict()
        self._n = n  # Number of agents (initial)
        self._s = self._n  # Underlying array sizes
        self._inds = None  # No filtering indices

        # Initialize underlying storage and map arrays
        for state_name, state in self.states.items():
            self._data[state_name] = state.new(self._n)
        self._map_arrays()
        self['uid'][:] = np.arange(self._n)

        # Define lock attribute here, since BasePeople.lock()/unlock() requires it
        self._lock = False  # Prevent further modification of keys

        return

    def initialize(self, popdict=None):
        """ Initialize people by setting their attributes """
        if popdict is None:
            self['age'][:] = np.random.random(size=self.n) * 100
            self['female'][:] = np.random.choice([False, True], size=self.n)
        else:
            # Use random defaults
            self['age'][:] = popdict['age']
            self['female'][:] = popdict['female']
        self.initialized = True
        return

    def __len__(self):
        """ Length of people """
        try:
            arr = getattr(self, base_key)
            return len(arr)
        except Exception as E:
            print(f'Warning: could not get length of People (could not get self.{base_key}: {E})')
            return 0

    @property
    def n(self):
        return len(self)

    def _len_arrays(self):
        """ Length of underlying arrays """
        return len(self._data[base_key])

    def lock(self):
        """ Lock the people object to prevent keys from being added """
        self._lock = True
        return

    def unlock(self):
        """ Unlock the people object to allow keys to be added """
        self._lock = False
        return

    def _grow(self, n):
        """
        Increase the number of agents stored

        Automatically reallocate underlying arrays if required
        
        Args:
            n (int): Number of new agents to add
        """
        orig_n = self._n
        new_total = orig_n + n
        if new_total > self._s:
            n_new = max(n, int(self._s / 2))  # Minimum 50% growth
            for state_name, state in self.states.items():
                self._data[state_name] = np.concatenate([self._data[state_name], state.new(n_new)],
                                                        axis=self._data[state_name].ndim - 1)
            self._s += n_new
        self._n += n
        self._map_arrays()
        new_inds = np.arange(orig_n, self._n)
        return new_inds

    def _map_arrays(self, keys=None):
        """
        Set main simulation attributes to be views of the underlying data

        This method should be called whenever the number of agents required changes
        (regardless of whether the underlying arrays have been resized)
        """
        row_inds = slice(None, self._n)

        # Handle keys
        if keys is None: keys = self.states.keys()
        keys = sc.tolist(keys)

        # Map arrays for selected keys
        for k in keys:
            arr = self._data[k]
            if arr.ndim == 1:
                rsetattr(self, k, arr[row_inds])
            elif arr.ndim == 2:
                rsetattr(self, k, arr[:, row_inds])
            else:
                errormsg = 'Can only operate on 1D or 2D arrays'
                raise TypeError(errormsg)

        return

    def __getitem__(self, key):
        """ Allow people['attr'] instead of getattr(people, 'attr')
            If the key is an integer, alias `people.person()` to return a `Person` instance
        """
        if isinstance(key, int):
            return self.person(key)
        else:
            return self.__getattribute__(key)

    def __setitem__(self, key, value):
        """ Ditto """
        if self._lock and key not in self.__dict__:  # pragma: no cover
            errormsg = f'Key "{key}" is not an attribute of people and the people object is locked; see people.unlock()'
            raise AttributeError(errormsg)
        return self.__setattr__(key, value)

    def __iter__(self):
        """ Iterate over people """
        for i in range(len(self)):
            yield self[i]

    def _brief(self):
        """
        Return a one-line description of the people -- used internally and by repr();
        see people.brief() for the user version.
        """
        try:
            string = f'People(n={len(self):0n})'
        except Exception as E:  # pragma: no cover
            string = sc.objectid(self)
            string += f'Warning, sim appears to be malformed:\n{str(E)}'
        return string

    def set(self, key, value):
        """
        Set values. Note that this will raise an exception the shapes don't match,
        and will automatically cast the value to the existing type
        """
        self[key][:] = value[:]

    def get(self, key):
        """ Convenience method -- key can be string or list of strings """
        if isinstance(key, str):
            return self[key]
        elif isinstance(key, list):
            arr = np.zeros((len(self), len(key)))
            for k, ky in enumerate(key):
                arr[:, k] = self[ky]
            return arr

    @property
    def alive(self):
        """ Alive boolean """
        return ~self.dead

    @property
    def f_inds(self):
        """ Indices of everyone female """
        return self.true('female')

    @property
    def m_inds(self):
        """ Indices of everyone male """
        return self.false('female')

    @property
    def active(self):
        """ Indices of everyone sexually active  """
        return (self.age >= self.debut) & self.alive

    @property
    def int_age(self):
        """ Return ages as an integer """
        return np.array(self.age, dtype=sss.default_int)

    @property
    def round_age(self):
        """ Rounds age up to the next highest integer"""
        return np.array(np.ceil(self.age))

    @property
    def alive_inds(self):
        """ Indices of everyone alive """
        return self.true('alive')

    @property
    def n_alive(self):
        """ Number of people alive """
        return len(self.alive_inds)

    def true(self, key):
        """ Return indices matching the condition """
        return self[key].nonzero()[-1]

    def false(self, key):
        """ Return indices not matching the condition """
        return (~self[key]).nonzero()[-1]

    def defined(self, key):
        """ Return indices of people who are not-nan """
        return (~np.isnan(self[key])).nonzero()[0]

    def undefined(self, key):
        """ Return indices of people who are nan """
        return np.isnan(self[key]).nonzero()[0]

    def count(self, key, weighted=True):
        """ Count the number of people for a given key """
        inds = self[key].nonzero()[0]
        if weighted:
            out = self.scale[inds].sum()
        else:
            out = len(inds)
        return out

    def count_any(self, key, weighted=True):
        """ Count the number of people for a given key for a 2D array if any value matches """
        inds = self[key].sum(axis=0).nonzero()[0]
        if weighted:
            out = self.scale[inds].sum()
        else:
            out = len(inds)
        return out

    def keys(self):
        """ Returns keys for all non-derived properties of the people object """
        return [state.name for state in self.states]

    def indices(self):
        """ The indices of each people array """
        return np.arange(len(self))

    def to_arr(self):
        """ Return as numpy array """
        arr = np.empty((len(self), len(self.keys())), dtype=sss.default_float)
        for k, key in enumerate(self.keys()):
            if key == 'uid':
                arr[:, k] = np.arange(len(self))
            else:
                arr[:, k] = self[key]
        return arr

    def to_list(self):
        """ Return all people as a list """
        return list(self)