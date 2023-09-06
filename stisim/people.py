"""
Defines the People class and functions associated with making people
"""

# %% Imports
import numpy as np
import pandas as pd
import sciris as sc
import stisim as ss

__all__ = ['BasePeople', 'People']

# %% Main people class
class BasePeople(sc.prettyobj):
    """
    A class to handle all the boilerplate for people -- everything interesting 
    happens in the People class, whereas this class exists to handle the less 
    interesting implementation details.
    """

    def __init__(self, n):

        self.initialized = False
        self._uid_map = ss.DynamicView(int, fill_value=ss.INT_NAN)  # This variable tracks all UIDs ever created
        self.uid = ss.DynamicView(int, fill_value=ss.INT_NAN)  # This variable tracks all UIDs currently in use

        self._uid_map.initialize(n)
        self._uid_map[:] = np.arange(0, n)
        self.uid.initialize(n)
        self.uid[:] = np.arange(0, n)

        # We internally store states in a dict keyed by the memory ID of the state, so that we can have colliding names
        # e.g., across modules, but we will never change the size of a State multiple times in the same iteration over
        # _states. This is a hidden variable because it is internally used to synchronize the size of all States contained
        # within the sim, regardless of where they are. In contrast, `People.states` offers a more user-friendly way to access
        # a selection of the states e.g., module states could be added in there, while intervention states might not
        self._states = {}

    def __len__(self):
        """ Length of people """
        return len(self.uid)

    def add_state(self, state):
        if id(state) not in self._states:
            self._states[id(state)] = state
        return

    def grow(self, n):
        """
        Increase the number of agents

        :param n: Integer number of agents to add
        """
        start_uid = len(self._uid_map)
        start_idx = len(self.uid)

        new_uids = np.arange(start_uid, start_uid + n)
        new_inds = np.arange(start_idx, start_idx + n)

        self._uid_map.grow(n)
        self._uid_map[new_uids] = new_inds

        self.uid.grow(n)
        self.uid[new_inds] = new_uids

        for state in self._states.values():
            state.grow(n)

        return new_uids

    def remove(self, uids_to_remove):
        """
        Reduce the number of agents

        :param uids_to_remove: An int/list/array containing the UID(s) to remove
        """
        # Calculate the *indices* to keep
        keep_uids = self.uid[~np.in1d(self.uid, uids_to_remove)]  # Calculate UIDs to keep
        keep_inds = self._uid_map[keep_uids]  # Calculate indices to keep

        # Trim the UIDs and states
        self.uid._trim(keep_inds)
        for state in self._states.values():
            state._trim(keep_inds)

        # Update the UID map
        self._uid_map[:] = ss.INT_NAN  # Clear out all previously used UIDs
        self._uid_map[keep_uids] = np.arange(0, len(keep_uids))  # Assign the array indices for all of the current UIDs

        # Remove the UIDs from the network too
        for network in self.networks.values():
            network.remove_uids(uids_to_remove)

        return

    def __getitem__(self, key):
        """
        Allow people['attr'] instead of getattr(people, 'attr')
        If the key is an integer, alias `people.person()` to return a `Person` instance
        """
        if isinstance(key, int):
            return self.person(key)  # TODO: need to re-implement
        else:
            return self.__getattribute__(key)

    def __setitem__(self, key, value):
        """ Ditto """
        return self.__setattr__(key, value)

    def __iter__(self):
        """ Iterate over people """
        for i in range(len(self)):
            yield self[i]


class People(BasePeople):
    """
    A class to perform all the operations on the people
    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized.

    Note that this class handles the mechanics of updating the actual people, while
    ``ss.BasePeople`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        strict (bool): whether to only create keys that are already in self.meta.person; otherwise, let any key be set
        pop_trend (dataframe): a dataframe of years and population sizes, if available
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::
        ppl = ss.People(2000)
    """

    # %% Basic methods

    def __init__(self, n, age_data=None, extra_states=None, networks=None):
        """ Initialize """

        super().__init__(n)

        self.initialized = False
        self.version = ss.__version__  # Store version info

        states = [
            ss.State('age', float, 0),
            ss.State('female', bool, ss.choice([True, False])),
            ss.State('debut', float),
            ss.State('alive', bool, True),
            ss.State('ti_dead', int, ss.INT_NAN),  # Time index for death
            ss.State('scale', float, 1.0),
        ]
        states.extend(sc.promotetolist(extra_states))

        self.states = ss.ndict()
        self._initialize_states(states)
        self.networks = ss.ndict(networks)

        # Set initial age distribution - likely move this somewhere else later
        age_data_dist = self.validate_age_data(age_data)
        self.age[:] = age_data_dist.sample(len(self))

        return

    @staticmethod
    def validate_age_data(age_data):
        """ Validate age data """
        if age_data is None: return ss.uniform(0, 100)
        if sc.checktype(age_data, pd.DataFrame):
            return ss.from_data(vals=age_data['value'].values, bins=age_data['age'].values)

    def initialize(self):
        """ Initialization - TBC what needs to go here """
        self.initialized = True
        return

    def _initialize_states(self, states):
        for state in states:
            self.add_state(state)  # Register the state internally for dynamic growth
            self.states.append(state)  # Expose these states with their original names
            state.initialize(self)  # Connect the state to this people instance
            setattr(self, state.name, state)
        return

    def add_module(self, module, force=False):
        # Map the module's states into the People state ndict
        if hasattr(self, module.name) and not force:
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())

        # The entries created below make it possible to do `sim.people.hiv.susceptible` or
        # `sim.people.states['hiv.susceptible']` and have both of them work
        module_states = sc.objdict()
        setattr(self, module.name, module_states)
        self._register_module_states(module, module_states)

    def _register_module_states(self, module, module_states):
        """Enable dot notation for module specific states:
         - `sim.people.hiv.susceptible` or
         - `sim.people.states['hiv.susceptible']`
        """

        for state_name, state in module.states.items():
            combined_name = module.name + '.' + state_name  # We will have to resolve how this works with multiple instances of the same module (e.g., for strains). The underlying machinery should be fine though, with People._states being flat and keyed by ID
            self.states[combined_name] = state  # Register the state on the user-facing side using the combined name. Within the original module, it can still be referenced by its original name
            pre, _, post = combined_name.rpartition('.')
            setattr(module_states, state_name, state)

        return

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def update(self, sim):
        """ Update demographics and networks """
        self.update_demographics(sim.dt, sim.ti)

        if sim.pars.remove_dead:
            self.remove(self.uid[self.dead])

        self.update_networks()

        return

    def update_demographics(self, dt, ti):
        """ Perform vital dynamic updates at the current timestep """
        death_uids = ss.true(self.ti_dead <= ti)
        self.alive[death_uids] = False
        self.age += dt
        return

    def update_networks(self):
        """
        Update networks
        """
        for network in self.networks.values():
            network.update(self)

    @property
    def active(self):
        """ Indices of everyone sexually active  """
        return (self.age >= self.debut) & self.alive

    @property
    def dead(self):
        """ Dead boolean """
        return ~self.alive

    @property
    def male(self):
        """ Male boolean """
        return ~self.female
