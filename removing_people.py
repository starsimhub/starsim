import pandas as pd
import numpy as np
import sciris as sc
from numpy.lib.mixins import NDArrayOperatorsMixin # Inherit from this to automatically gain operators like +, -, ==, <, etc.

INT_NAN = np.iinfo(int).max  # Value to use to flag removed UIDs (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)
import numba as nb

# Some gotchas
# - If two FusedArrays have the same number of entries but have different UIDs (or even a different order of UIDs) then operations like addition will work
#   but they will produce incorrect output. States all reference the same UIDs so can be safely operated on




class FusedArray(NDArrayOperatorsMixin):
    # This is a class that allows indexing by UID but does not support dynamic growth
    # It's kind of like a Pandas series but one that only supports a monotonically increasing
    # unique integer index, and that we can customize and optimize indexing for. We also
    # support indexing 2D arrays, with the second dimension (columns) being mapped by UID
    # (this functionality is used to store vector states like genotype-specific quantities)
    #
    # We explictly do NOT support slicing, as these arrays are indexed by UID and slicing
    # by UID can be confusing/ambiguous when there are missing values. Indexing returns
    # another FusedArray instance

    __slots__ = ('values','_uid_map','uids')

    def __init__(self, values, uids, uid_map=None):

        self.values = values
        self.uids = uids

        if uid_map is None and uids is not None:
            # Construct a local UID map as opposed to using a shared one (i.e., the one for all agents contained in the People instance)
            self.uid_map = np.full(np.max(uids)+1, fill_value=INT_NAN, dtype=int)
            self.uid_map[uids] = np.arange(len(uids))
        else:
            self._uid_map = uid_map

    def __repr__(self):
        # TODO - optimize (obviously won't want to make a full dataframe just to print it!)
        df = pd.DataFrame(self.values.T, index=self.uids, columns=['Quantity'])
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
                    uids = self.uids.__array__()[mapped_key]
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
            return FusedArray(values=values, uids=uids, uid_map=new_uid_map)
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

    @property
    def shape(self):
        return self.values.shape

    @property
    def __array_interface__(self):
        return self.values.__array_interface__

    def __array__(self):
        return self.values

    def __array_ufunc__(self, *args, **kwargs):
        # Does this function really work?! This is to support using +=
        if args[1] != '__call__':
            args = [(x if x is not self else self.values) for x in args]
            kwargs = {k: v if v is not self else self.values for k, v in kwargs.items()}
            return self.values.__array_ufunc__(*args, **kwargs)

        args = [(x if x is not self else self.values) for x in args]
        if 'out' in kwargs and kwargs['out'][0] is self:
            kwargs['out'] = self.values
            args[0](*args[2:], **kwargs)
            return self
        else:
            out_arr = args[0](*args[2:], **kwargs)
            return FusedArray(values=out_arr, uids=self.uids, uid_map=self._uid_map)

    def __array_wrap__(self, out_arr, context=None):
        # This allows numpy operations addition etc. to return instances of FusedArray
        if out_arr.ndim == 0:
            return out_arr.item()
        return FusedArray(values=out_arr, uids=self.uids, uid_map=self._uid_map) # Hardcoding class means State can inherit from FusedArray but return FusedArray base instances


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
        super().__init__(values=None, uids=None, uid_map=None) # Call the FusedArray constructor
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
        self.uids = people.uids
        self._data.initialize(len(self.uids))
        self.values = self._data._view
        self._initialized = True

    def grow(self, n):
        self._data.grow(n)
        self.values = self._data._view

    def _trim(self, inds):
        self._data._trim(inds)
        self.values = self._data._view


class DynamicPeople():
    def __init__(self, states=None):

        self._uid_map = DynamicView(int, fill_value=INT_NAN) # This variable tracks all UIDs ever created
        self.uids = DynamicView(int, fill_value=INT_NAN) # This variable tracks all UIDs currently in use

        self.sex = State('sex',bool)
        self.dead = State('dead',bool)

        self.states = [self.sex, self.dead] # All state objects linked to this People instance. Note that the states above are not yet linked to the People because they haven't been initialized

        # In reality, these states might be added elsewhere (e.g., in modules)
        if states is not None:
            for state in states:
                self.add_state(state)

    def add_state(self, state):
        if id(state) not in {id(x) for x in self.states}:
            self.states.append(state)

    def __len__(self):
        return len(self.uids)

    def initialize(self, n):

        self._uid_map.initialize(n)
        self._uid_map[:] = np.arange(0, n)
        self.uids.initialize(n)
        self.uids[:] = np.arange(0, n)

        for state in self.states:
            state.initialize(self)

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
        self.uids[new_inds] = new_uids

        for state in self.states:
            state.grow(n)

    def remove(self, uids_to_remove):
        # Calculate the *indices* to keep
        keep_uids = self.uids[~np.in1d(self.uids, uids_to_remove)] # Calculate UIDs to keep
        keep_inds = self._uid_map[keep_uids] # Calculate indices to keep

        # Trim the UIDs and states
        self.uids._trim(keep_inds)
        for state in self.states:
            state._trim(keep_inds)

        # Update the UID map
        self._uid_map[:] = INT_NAN # Clear out all previously used UIDs
        self._uid_map[keep_uids] = np.arange(0, len(keep_uids)) # Assign the array indices for all of the current UIDs

# TEST INDEXING PERFORMANCE



# x = State('foo', int, 0)
# y = State('imm', float, 0, shape=2)
# z = State('bar', int, lambda n: np.random.randint(1,3,n))
#
# p = DynamicPeople(states=[x,y,z])
# p.initialize(3)
# p.grow(3)
# p.remove([0, 1])
# p.remove(2)
# p.grow(10)
#
# p.grow(8)
# p.grow(8)

# print(x)
# #
# # z + 1
# # z += 1
# # z[z==2]
# #
# x[2] = 10
# x[3] = 20
# #
# y[[0,1],2] = 10
#
# y[0,2] = 10
# y[0,3] = 20
# y[1,2] = 15
# y[1,3] = 25
# #
# p.remove([0, 1])
# p.remove(2)
# #
# print(x)
# print(y)
# print(z)
# x[x==20]

#
#
x = State('foo', int, 0)
# y = State('imm', float, 0, shape=2)
z = State('bar', int, lambda n: np.random.randint(1,3,n))

p = DynamicPeople(states=[x,z])
p.initialize(200000)

remove = np.random.choice(np.arange(len(p)), 100000, replace=False)
p.remove(remove)

# Single item
# Multiple items
multiple_items_uid = np.random.choice(p.uids, 50000, replace=False)
multiple_items_ind = p._uid_map[multiple_items_uid]
multiple_items_few_uid = np.random.choice(p.uids, 1000, replace=False)
multiple_items_few_ind = p._uid_map[multiple_items_few_uid]
single_item_uid = multiple_items_uid[5000]
single_item_ind = multiple_items_ind[5000]
boolean = np.random.randint(0,2, size=len(p)).astype(bool)
#
# z[single_item_uid]
z[multiple_items_few_uid]
# #
def index():
    for i in range(50000):
        z[multiple_items_few_uid]

# sc.profile(index, follow=[FusedArray.__getitem__, FusedArray._process_key, FusedArray.__init__])


a = z.values
s = pd.Series(z.values, p.uids)
uid_map = p._uid_map._view

# print('SINGLE ITEM LOOKUP')
# print('Array (direct): ', end='')
# %timeit a[single_item_ind]
# print('Array (mapped): ', end='')
# %timeit a[uid_map[single_item_uid]]
# print('State: ', end='')
# %timeit z[single_item_uid]
# print('Series: ', end='')
# %timeit s[single_item_uid]
# print('Series (loc): ', end='')
# %timeit s.loc[single_item_uid]
# print()
# print('MULTIPLE ITEM LOOKUP')
# print('Array (direct): ', end='')
# %timeit a[multiple_items_ind]
# print('Array (mapped): ', end='')
# %timeit a[uid_map[multiple_items_uid]]
# print('State: ', end='')
# %timeit z[multiple_items_uid]
# print('Series: ', end='')
# %timeit s[multiple_items_uid]
# print('Series (loc): ', end='')
# %timeit s.loc[multiple_items_uid]
# print()
# print('MULTIPLE ITEM LOOKUP (FEW)')
# print('Array (direct): ', end='')
# %timeit a[multiple_items_few_ind]
# print('Array (mapped): ', end='')
# %timeit a[uid_map[multiple_items_few_uid]]
# print('State: ', end='')
# %timeit z[multiple_items_few_uid]
# print('Series: ', end='')
# %timeit s[multiple_items_few_uid]
# print('Series (loc): ', end='')
# %timeit s.loc[multiple_items_few_uid]
# print()
# print('BOOLEAN ARRAY LOOKUP')
# print('Array (direct): ', end='')
# %timeit a[boolean]
# print('Array (nonzero): ', end='')
# %timeit a[boolean.nonzero()[0]]
# print('State: ', end='')
# %timeit z[boolean]
# print('Series: ', end='')
# %timeit s[boolean]
# print('Series (loc): ', end='')
# %timeit s.loc[boolean]
# print()
# print('SINGLE ITEM ASSIGNMENT')
# print('Array (direct): ', end='')
# %timeit a[single_item_ind] = 1
# print('Array (mapped): ', end='')
# %timeit a[uid_map[single_item_uid]] = 1
# print('State: ', end='')
# %timeit z[single_item_uid] = 1
# print('Series: ', end='')
# %timeit s[single_item_uid] = 1
# print('Series (loc): ', end='')
# %timeit s.loc[single_item_uid] = 1
# print()
# print('MULTIPLE ITEM ASSIGNMENT')
# print('Array (direct): ', end='')
# %timeit a[multiple_items_ind] = 1
# print('Array (mapped): ', end='')
# %timeit a[uid_map[multiple_items_uid]] = 1
# print('State: ', end='')
# %timeit z[multiple_items_uid] = 1
# print('Series: ', end='')
# %timeit s[multiple_items_uid] = 1
# print('Series (loc): ', end='')
# %timeit s.loc[multiple_items_uid] = 1
# print()
# print('MULTIPLE ITEM ASSIGNMENT (FEW)')
# print('Array (direct): ', end='')
# %timeit a[multiple_items_few_ind] = 1
# print('Array (mapped): ', end='')
# %timeit a[uid_map[multiple_items_few_uid]] = 1
# print('State: ', end='')
# %timeit z[multiple_items_few_uid] = 1
# print('Series: ', end='')
# %timeit s[multiple_items_few_uid] = 1
# print('Series (loc): ', end='')
# %timeit s.loc[multiple_items_few_uid] = 1
# print()
# print('MULTIPLE ITEM ARRAY ASSIGNMENT')
# print('Array (direct): ', end='')
# %timeit a[multiple_items_ind] = multiple_items_ind
# print('Array (mapped): ', end='')
# %timeit a[uid_map[multiple_items_uid]] = multiple_items_ind
# print('State: ', end='')
# %timeit z[multiple_items_uid] = multiple_items_ind
# print('Series: ', end='')
# %timeit s[multiple_items_uid] = multiple_items_ind
# print('Series (loc): ', end='')
# %timeit s.loc[multiple_items_uid] = multiple_items_ind
# print()
# print('MULTIPLE ITEM ARRAY ASSIGNMENT (FEW)')
# print('Array (direct): ', end='')
# %timeit a[multiple_items_few_ind] = multiple_items_few_ind
# print('Array (mapped): ', end='')
# %timeit a[uid_map[multiple_items_few_uid]] = multiple_items_few_ind
# print('State: ', end='')
# %timeit z[multiple_items_few_uid] = multiple_items_few_ind
# print('Series: ', end='')
# %timeit s[multiple_items_few_uid] = multiple_items_few_ind
# print('Series (loc): ', end='')
# %timeit s.loc[multiple_items_few_uid] = multiple_items_few_ind
# print()
# print('BOOLEAN ARRAY ASSIGNMENT')
# print('Array (direct): ', end='')
# %timeit a[boolean] = 1
# print('Array (nonzero): ', end='')
# %timeit a[boolean.nonzero()[0]] = 1
# print('State: ', end='')
# %timeit z[boolean] = 1
# print('Series: ', end='')
# %timeit s[boolean] = 1
# print('Series (loc): ', end='')
# %timeit s.loc[boolean]
