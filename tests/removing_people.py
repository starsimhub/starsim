import pandas as pd
import numpy as np
import sciris as sc
from starsim.states import State, ArrayView

INT_NAN = np.iinfo(int).max  # Value to use to flag removed UIDs (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)

# Some gotchas
# - If two FusedArrays have the same number of entries but have different UIDs (or even a different order of UIDs) then operations like addition will work
#   but they will produce incorrect output. States all reference the same UIDs so can be safely operated on


class DynamicPeople():
    def __init__(self, states=None):

        self._uid_map = ArrayView(int, fill_value=INT_NAN) # This variable tracks all UIDs ever created
        self.uid = ArrayView(int, fill_value=INT_NAN) # This variable tracks all UIDs currently in use

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
        return len(self.uid)

    def initialize(self, n):

        self._uid_map.grow(n)
        self._uid_map[:] = np.arange(0, n)
        self.uid.grow(n)
        self.uid[:] = np.arange(0, n)

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
        start_idx = len(self.uid)

        new_uids = np.arange(start_uid, start_uid+n)
        new_inds = np.arange(start_idx, start_idx+n)

        self._uid_map.grow(n)
        self._uid_map[new_uids] = new_inds

        self.uid.grow(n)
        self.uid[new_inds] = new_uids

        for state in self.states:
            state.grow(n)

    def remove(self, uids_to_remove):
        # Calculate the *indices* to keep
        keep_uids = self.uid[~np.in1d(self.uid, uids_to_remove)] # Calculate UIDs to keep
        keep_inds = self._uid_map[keep_uids] # Calculate indices to keep

        # Trim the UIDs and states
        self.uid._trim(keep_inds)
        for state in self.states:
            state._trim(keep_inds)

        # Update the UID map
        self._uid_map[:] = INT_NAN # Clear out all previously used UIDs
        self._uid_map[keep_uids] = np.arange(0, len(keep_uids)) # Assign the array indices for all of the current UIDs

if __name__ == '__main__':
    x = State('foo', int, 0)
    z = State('bar', int, lambda n: np.random.randint(1,3,n))

    p = DynamicPeople(states=[x,z])
    p.initialize(200000)

    remove = np.random.choice(np.arange(len(p)), 100000, replace=False)
    p.remove(remove)

    # Define uids and indices
    multiple_items_uid = np.random.choice(p.uid, 50000, replace=False)
    multiple_items_ind = p._uid_map[multiple_items_uid]
    multiple_items_few_uid = np.random.choice(p.uid, 1000, replace=False)
    multiple_items_few_ind = p._uid_map[multiple_items_few_uid]
    single_item_uid = multiple_items_uid[5000]
    single_item_ind = multiple_items_ind[5000]
    boolean = np.random.randint(0,2, size=len(p)).astype(bool)

    a = z.values
    s = pd.Series(z.values, p.uid)
    uid_map = p._uid_map._view

    def lookup(lbl, uids, n=2_000):
        print('\n' + lbl)
        with sc.timer('State: '):
            for i in range(n): z[uids]
        with sc.timer('Series: '):
            for i in range(n): s[uids]
        with sc.timer('Series (loc): '):
            for i in range(n): s.loc[uids]

        with sc.timer('Series (hack): '):
            for i in range(n): 
                if uids.dtype == np.bool_:
                    s.values[uids]
                else:
                    s.values[uid_map[uids]]

    def assign(lbl, uids, vals, n=2_000):
        print('\n' + lbl)
        with sc.timer('State: '):
            for i in range(n): z[uids] = vals
        with sc.timer('Series: '):
            for i in range(n): s[uids] = vals
        with sc.timer('Series (loc): '):
            for i in range(n): s.loc[uids] = vals

        with sc.timer('Series (hack): '):
            for i in range(n): 
                if uids.dtype == np.bool_:
                    s.values[uids] = vals
                else:
                    s.values[uid_map[uids]] = vals

    lookup('SINGLE ITEM LOOKUP', single_item_uid)
    lookup('MULTIPLE ITEM LOOKUP (state lookup also includes construction of a FusedArray)', multiple_items_uid)
    lookup('MULTIPLE ITEM LOOKUP (FEW) (state lookup also includes construction of a FusedArray)', multiple_items_few_uid)
    lookup('BOOLEAN ARRAY LOOKUP', boolean)

    assign('SINGLE ITEM ASSIGNMENT', single_item_uid, 1)
    assign('MULTIPLE ITEM ASSIGNMENT', multiple_items_uid, 1)
    assign('MULTIPLE ITEM ASSIGNMENT (FEW)', multiple_items_few_uid, 1)
    
    assign('MULTIPLE ITEM ARRAY ASSIGNMENT', multiple_items_uid, multiple_items_ind)
    assign('MULTIPLE ITEM ARRAY ASSIGNMENT (FEW)', multiple_items_few_uid, multiple_items_few_ind)
    assign('BOOLEAN ARRAY ASSIGNMENT', boolean, 1)