import pandas as pd
import numpy as np
import sciris as sc

INT_NAN = np.iinfo(int).max  # Value to use to flag removed UIDs (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)

class DynamicPeople():
    def __init__(self, states):

        self._uid_map = State('uid',int) # This state tracks *ALL* uids ever created

        self.uids = State('uid',int) # This state tracks *ALL* uids ever created
        self.dead = State('dead',bool)
        self.states = states

        self._dynamic_states = [] # List of references to dynamic states that should be updated. These should be local to a Sim but could be contained in interventions, connectors or analyzers

    def __len__(self):
        return len(self.uids)

    def initialize(self, n):
        self._uid_map.initialize(n)
        self._uid_map[:] = np.arange(0, len(self._uid_map))

        self.uids.initialize(uid_map=self._uid_map)
        self.uids[:] = np.arange(0, len(self._uid_map))

        self.dead.initialize(uid_map=self._uid_map)
        for state in self.states:
            state.initialize(uid_map=self._uid_map)

    def __setattr__(self, attr, value):
        if hasattr(self, attr) and isinstance(getattr(self, attr), State):
            raise Exception('Cannot assign directly to a state - must index into the view instead e.g. `people.uid[:]=`')
        else:   # If not initialized, rely on the default behavior
            object.__setattr__(self, attr, value)


    def grow(self, n):
        # Add agents
        start_uid = len(self._uid_map)
        start_idx = len(self.uids)

        new_uids = np.arange(start_uid, start_uid+n)
        new_inds = np.arange(start_idx, start_idx+n)

        self._uid_map.grow(n)
        self._uid_map[new_uids] = new_inds

        self.uids.grow(n)
        self.uids[new_uids] = new_uids

        self.dead.grow(n)
        for state in self.states:
            state.grow(n)

    def remove(self, uids):
        # Remove specified indices

        # After removing the requested UIDs, what are the remaining UIDs and their array positions?
        remaining_uids = self.uids.values[~np.in1d(self.uids, uids)]

        # Calculate indices to keep
        keep_inds = self._uid_map[remaining_uids]

        self.uids._trim(keep_inds)
        self.dead._trim(keep_inds)
        for state in self.states:
            state._trim(keep_inds)

        # Update the UID map
        self._uid_map[:] = INT_NAN
        self._uid_map[remaining_uids] = np.arange(0, len(remaining_uids))

        print()

from numpy.lib.mixins import NDArrayOperatorsMixin # Inherit from this to automatically gain operators like +, -, ==, <, etc.
class State(NDArrayOperatorsMixin):
    def __init__(self, name, dtype, fill_value=0, shape=None, label=None):
        """
        Args:
            name: name of the result as used in the model
            dtype: datatype
            fill_value: default value for this state upon model initialization. If callable, it is called with a single argument for the number of samples
            shape: If not none, set to match a string in `pars` containing the dimensionality
            label: text used to construct labels for the result for displaying on plots and other outputs
        """
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value
        self.shape = shape
        self.label = label or name

        self.n = 0  # Number of agents currently in use
        self._uid_map = None  # Array of length equal to number of agents ever created - this would be stored at the People level normally
        self._data = None  # The underlying memory array (length at least equal to n)
        self.values = None  # The view corresponding to what is actually accessible (length equal to n)
        return

    @property
    def _s(self):
        # Return the size of the underlying array (maximum number of agents that can be stored without reallocation)
        return self._data.shape[-1]

    def __len__(self):
        return self.n

    def __repr__(self):
        # Quick/rough implementation to get it to print both the UIDs and the values
        # It's important to print the UID as well so that it's obvious to users that they
        # cannot index it like a normal array i.e., by position
        if self._uid_map is not None:
            uids = pd.Index(np.where(self._uid_map != INT_NAN)[0], name='UID')
        else:
            uids = None

        df = pd.DataFrame(self.values.T, index=uids)
        df.columns.name = self.name
        return df.__repr__()

    @property
    def indexed_by_uid(self):
        # Returns True if this object is indexed by UID rather than array position
        return self._uid_map is not None

    def initialize(self, n=None, uid_map=None):
        """

        :param n:
        :param uid_map: Optionally specify a State containing UIDs - if specified, this State is indexed
        :return:
        """

        # Either specify an initial size, or the UIDs to reference
        assert (n is None) != (uid_map is None), 'Must specify either the initial number of agents or a reference to UIDs'

        if uid_map is not None:
            self.n = len(uid_map)
        else:
            self.n = n

        self._uid_map = uid_map

        # Calculate the number of rows
        if sc.isstring(self.shape):
            raise Exception('TODO: Come up with a better way to link a state to a parameter like n_genotypes')
        elif self.shape is None:
            shape = (self.n, )
        else:
            shape = (self.shape, self.n) # Just leave it as an (assumed) integer

        if callable(self.fill_value):
            self._data = np.empty(shape, dtype=self.dtype)
            self._data[:] = self.fill_value(self.n)
        else:
            self._data = np.full(shape, dtype=self.dtype, fill_value=self.fill_value)

        self._map_arrays()

    def grow(self, n):

        if self.n + n > self._s:
            n_new = max(n, int(self._s / 2))  # Minimum 50% growth

            if self._data.ndim == 1:
                shape = (n_new,)
            else:
                shape = (self._data.shape[0], n_new)

            if callable(self.fill_value):
                new = np.empty(shape, dtype=self.dtype)
                new[:] = self.fill_value(self.n)
            else:
                new = np.full(shape, dtype=self.dtype, fill_value=self.fill_value)

            self._data = np.concatenate([self._data, new], axis=self._data.ndim - 1)

        self.n += n

        self._map_arrays()

    def _trim(self, inds):
        # Keep only specified indices
        # Should only be called via people.remove() so that the UID map can be updated too
        self.n = len(inds)
        if self._data.ndim == 1:
            self._data[:self.n] = self._data[inds]
        else:
            self._data[:,:self.n] = self._data[:,inds]

        self._map_arrays()

    def _map_arrays(self):
        """
        Set main simulation attributes to be views of the underlying data

        This method should be called whenever the number of agents required changes
        (regardless of whether or not the underlying arrays have been resized)
        """

        if self._data.ndim == 1:
            self.values = self._data[:self.n]
        elif self._data.ndim == 2:
            self.values = self._data[:, :self.n]
        else:
            errormsg = 'Can only operate on 1D or 2D arrays'
            raise TypeError(errormsg)
        return

    def _map_slice(self, key):
        return slice(None if key.start is None else self._uid_map[key.start],None if key.stop is None else self._uid_map[key.stop],key.step)

    def __getitem__(self, key):
        if self._uid_map is None:
            return self.values.__getitem__(key)

        if isinstance(key, slice):
            return self.values.__getitem__(self._map_slice(key))
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            return self.values.__getitem__(key)
        elif isinstance(key, tuple):
            # If it's a tuple, then map the second index and not the first
            if isinstance(key[1], slice):
                new_key = (key[0], self._map_slice(key[1]))
            elif isinstance(key[1], np.ndarray) and key[1].dtype == bool:
                new_key = key
            else:
                new_key = (key[0], self._uid_map[key[1]])
            return self.values.__getitem__(new_key)
        else:
            # The key is expected to be the UID
            # Translate to indices
            return self.values.__getitem__(self._uid_map[key])

    def __setitem__(self, key, value):
        if self._uid_map is None:
            self.values.__setitem__(key, value)
            return

        if isinstance(key, slice):
            self.values.__setitem__(self._map_slice(key), value)
        elif isinstance(key, np.ndarray) and key.dtype == bool:
                self.values.__setitem__(key, value)
        elif isinstance(key, tuple):
            # If it's a tuple, then map the second index and not the first
            if isinstance(key[1], slice):
                new_key = self._map_slice(key[1])
            else:
                new_key = (key[0], self._uid_map[key[1]])
            self.values.__setitem__(new_key, value)
        else:
            # The key is expected to be the UID
            # Translate to indices
            self.values.__setitem__(self._uid_map[key], value)

    # These methods allow operators and numpy functions like sum(), mean() etc.
    # to be used directly on the State object, and with reasonable performance

    @property
    def __array_interface__(self):
        return self.values.__array_interface__

    def __array__(self):
        return self.values

    def __array_ufunc__(self, *args, **kwargs):
        args = [(x if x is not self else self.values) for x in args]
        kwargs = {k: v if v is not self else self.values for k, v in kwargs.items()}
        return self.values.__array_ufunc__(*args, **kwargs)


x = State('foo', int, 0)
y = State('imm', float, 0, shape=2)
z = State('bar', int, lambda n: np.random.randint(1,3,n))

p = DynamicPeople(states=[x,y,z])
p.initialize(3)
p.grow(3)

x[2] = 10
x[3] = 20

y[[0,1],2] = 10

y[0,2] = 10
y[0,3] = 20
y[1,2] = 15
y[1,3] = 25

p.remove([0, 1])
p.remove(2)

print(x)
print(y)
print(z)




x = State('foo', int, 0)
y = State('imm', float, 0, shape=2)
z = State('bar', int, lambda n: np.random.randint(1,3,n))

p = DynamicPeople(states=[x,y,z])
p.initialize(100000)

# To remove
remove = np.random.choice(np.arange(len(p)), 50000, replace=False)
p.remove(remove)


s = pd.Series(z.values, p.uids)

# %timeit s[99990]
#
# %timeit z[99990]

def test():
    for i in range(1000000):
        z[99990]

sc.profile(run=test, follow=[State.__getitem__])
