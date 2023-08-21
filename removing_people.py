import pandas as pd
import numpy as np
import sciris as sc

from numpy.lib.mixins import NDArrayOperatorsMixin # Inherit from this to automatically gain operators like +, -, ==, <, etc.


INT_NAN = np.iinfo(int).max  # Value to use to flag removed UIDs (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)


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
    #
    # Note that we do NOT use the

    __slots__ = ('values','_uid_map','_uids')

    def __init__(self, values, uids, uid_map=None):

        self.values = values

        if uid_map is None and uids is not None:
            # Construct a local UID map as opposed to using a shared one (i.e., the one for all agents contained in the People instance)
            uid_map = np.full(np.max(uids)+1, fill_value=INT_NAN, dtype=int)
            uid_map[uids] = np.arange(len(uids))

        self._uids = uids
        self._uid_map = uid_map

    def __repr__(self):
        # TODO - optimize (obviously won't want to make a full dataframe just to print it!)
        if self.values.ndim == 1:
            df = pd.DataFrame(self.values.T, index=self._uids, columns=['Quantity'])
            df.index.name = 'UID'
        else:
            df = pd.DataFrame(self.values.T, index=self._uids).T
            df.index.name='Quantity'
            df.columns.name='UID'
        return df.__repr__()

    # Overload indexing
    #
    # Supported indexing operations on the last dimension (max dimension of 2)
    # - Integer value (mapped)
    # - List/array of integers (mapped)
    # - Boolean array (used directly)
    # Unsupported operations
    # - Slices
    def _process_key(self, key):
        if (isinstance(key, np.ndarray) and key.dtype == bool) or (isinstance(key, FusedArray) and key.values.dtype == bool):
            # If we pass in a boolean array, apply it directly
            return key
        elif isinstance(key, tuple):
            if (isinstance(key[1], np.ndarray) and key[1].dtype == bool) or (isinstance(key[1], FusedArray) and key[1].values.dtype == bool):
                return key
            elif isinstance(key[1], slice):
                if key[1].start is None and key[1].stop is None and key[1].step is None:
                    return key
                else:
                    raise Exception('Slicing not supported - slice the .values attribute by index instead e.g., x.values[0:5], not x[0:5]')
            else:
                return (key[0], self._uid_map[key[1]])
        elif isinstance(key, slice):
            if key.start is None and key.stop is None and key.step is None:
                return key
            else:
                raise Exception('Slicing not supported - slice the .values attribute by index instead e.g., x.values[0:5], not x[0:5]')
        else:
            return self._uid_map[key]


    def __getitem__(self, key):
        try:
            if isinstance(key, (int, np.integer)):
                return self.values.__getitem__(self._process_key(key))
            else:
                key = self._process_key(key)
                if isinstance(key, tuple):
                    if isinstance(key[1],(int, np.integer)):
                        uids = [self._uids[key[1]]]
                        values = self.values.__getitem__((key[0], key[1], None))
                        if isinstance(key[0],(int, np.integer)):
                            return values[0]
                    else:
                        uids = self._uids[key[1]]
                        values = self.values.__getitem__((key[0], key[1]))
                else:
                    uids = self._uids[key]
                    values = self.values.__getitem__(key)
                return FusedArray(values=values, uids=uids)
        except IndexError as e:
            if str(INT_NAN) in str(e):
                raise IndexError(f'UID not present in array')
            else:
                raise e

    def __setitem__(self, key, value):
        try:
            return self.values.__setitem__(self._process_key(key), value)
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
    def ndim(self):
        return self.values.ndim

    @property
    def __array_interface__(self):
        return self.values.__array_interface__

    def __array__(self):
        return self.values

    def __array_ufunc__(self, *args, **kwargs):
        # Does this function really work?!
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
            return FusedArray(values=out_arr, uids=self._uids, uid_map=self._uid_map)

    def __array_wrap__(self, out_arr, context=None):
        # This allows numpy operations addition etc. to return instances of FusedArray
        if out_arr.ndim == 0:
            return out_arr.item()
        return FusedArray(values=out_arr, uids=self._uids, uid_map=self._uid_map) # Hardcoding class means State can inherit from FusedArray but return FusedArray base instances


class DynamicView(NDArrayOperatorsMixin):
    def __init__(self, dtype, fill_value=0, shape=None, label=None):
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
        self.shape = shape

        self.n = 0  # Number of agents currently in use

        self._data = None  # The underlying memory array (length at least equal to n)
        self._view = None  # The view corresponding to what is actually accessible (length equal to n)
        return

    @property
    def _s(self):
        # Return the size of the underlying array (maximum number of agents that can be stored without reallocation)
        return self._data.shape[-1]

    def __len__(self):
        return self.n

    def __repr__(self):
        return self._view.__repr__()

    def initialize(self, n):
        self.n = n

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
                new[:] = self.fill_value(n_new) # TODO - make this work properly for 2d arrays
            else:
                new = np.full(shape, dtype=self.dtype, fill_value=self.fill_value)

            self._data = np.concatenate([self._data, new], axis=self._data.ndim - 1)

        self.n += n
        self._map_arrays()

    def _trim(self, inds):
        # Keep only specified indices
        # Note that these are indices, not UIDs!
        n = len(inds)
        if self._data.ndim == 1:
            self._data[:n] = self._data[inds]
            self._data[n:self.n] = self.fill_value(self.n-n) if callable(self.fill_value) else self.fill_value
        else:
            self._data[:,:n] = self._data[:,inds]
            self._data[:,n:self.n] = self.fill_value(self.n-n) if callable(self.fill_value) else self.fill_value  # TODO - make this work properly for 2D arrays
        self.n = n
        self._map_arrays()

    def _map_arrays(self):
        """
        Set main simulation attributes to be views of the underlying data

        This method should be called whenever the number of agents required changes
        (regardless of whether or not the underlying arrays have been resized)
        """

        if self._data.ndim == 1:
            self._view = self._data[:self.n]
        elif self._data.ndim == 2:
            self._view = self._data[:, :self.n]
        else:
            errormsg = 'Can only operate on 1D or 2D arrays'
            raise TypeError(errormsg)
        return

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

    def __init__(self, name, dtype, fill_value=0, shape=None, label=None):
        super().__init__(values=None, uids=None, uid_map=None) # Call the FusedArray constructor
        self._data = DynamicView(dtype=dtype, fill_value=fill_value,shape=shape)
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
        self._uids = people.uids
        self._data.initialize(len(self._uids))
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


x = State('foo', int, 0)
y = State('imm', float, 0, shape=2)
z = State('bar', int, lambda n: np.random.randint(1,3,n))

p = DynamicPeople(states=[x,y,z])
p.initialize(3)
p.grow(3)
p.remove([0, 1])
p.remove(2)
p.grow(10)

p.grow(8)
p.grow(8)

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
# x = State('foo', int, 0)
# y = State('imm', float, 0, shape=2)
# z = State('bar', int, lambda n: np.random.randint(1,3,n))
#
# p = DynamicPeople(states=[x,y,z])
# p.initialize(100000)
#
#
# def true(x):
#     return x._uids[np.nonzero(x)[-1]]

# x = State('foo', int, 0)
# y = State('imm', float, 0, shape=2)
# z = State('bar', int, lambda n: np.random.randint(1,3,n))
#
# p = DynamicPeople(states=[x,y,z])
# p.initialize(100000)
#
# # To remove
# remove = np.random.choice(np.arange(len(p)), 50000, replace=False)
# p.remove(remove)
#
# # How should cv.true()/hp.true() get used?
# # Essentially we want to return UIDs of people rather than actual IDs
#
# def true(v):
#     if isinstance(v):
#
# # hpvsim
# # filter_inds = people.true('hiv')  # indices fo people with HIV
# # if len(filter_inds):
# #     art_inds = filter_inds[hpu.true(people.art[filter_inds])]  # Indices of people on ART
# #     not_art_inds = filter_inds[hpu.false(people.art[filter_inds])]
# #     cd4_remaining_inds = hpu.itrue(((people.t - people.date_hiv[not_art_inds]) * dt) < people.dur_hiv[not_art_inds], not_art_inds)  # Indices of people not on ART who have an active infection
# #
#
# # IN AN IDEAL WORLD
# # Essentilly, we require true() to return the UIDs that go with each variable
# hiv_uids = true(hiv)
# art_uids = true(art[hiv_uids]])
# not_art_inds = false(art[hiv_uids])

# TODO - return something that tracks the UIDs if the getitem argument was a list/array

# Support things like `x = x + 1`
# x += 1 might be sensible
# x[hiv_uids] += 1
# x[:] += 1 works

#
# s = pd.Series(z.values, p.uids)
# v = z.values
# d = z._data
# l = list(z.values)
# #
# # %timeit z[99999]
# # %timeit s[99999]
# # %timeit v[49999]
# # %timeit l[49999]
# # %timeit d[49999]
#
# # TODO - remove slicing
#
#
# def test():
#     for i in range(100000):
#         z[99999]
#
# sc.profile(run=test, follow=[State.__getitem__])
#
