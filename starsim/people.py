"""
Defines the People class and functions associated with making people
"""

# %% Imports
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['BasePeople', 'People']

# %% Main people class
class BasePeople(sc.prettyobj):
    """
    A class to handle all the boilerplate for people -- everything interesting 
    happens in the People class, whereas this class exists to handle the less 
    interesting implementation details.
    """

    def __init__(self, n_agents):

        self.initialized = False
        self._uid_map = ss.ArrayView(int, default=ss.INT_NAN)  # This variable tracks all UIDs ever created
        self.uid = ss.ArrayView(int, default=ss.INT_NAN)  # This variable tracks all UIDs currently in use

        n = int(n_agents)

        self._uid_map.grow(n)
        self._uid_map[:] = np.arange(0, n)
        self.uid.grow(n)
        self.uid[:] = np.arange(0, n)

        # A slot is a special state managed internally by BasePeople
        # This is because it needs to be updated separately from any other states, as other states
        # might have fill_values that depend on the slot
        self.slot = ss.State('slot', int, ss.INT_NAN)

        self.ti = None  # Track simulation time index
        self.dt = np.nan  # Track simulation time step

        # User-facing collection of states
        self.states = ss.ndict(type=ss.State)

        # We also internally store states in a dict keyed by the memory ID of the state, so that we can have colliding names
        # e.g., across modules, but we will never change the size of a State multiple times in the same iteration over
        # _states. This is a hidden variable because it is internally used to synchronize the size of all States contained
        # within the sim, regardless of where they are. In contrast, `People.states` offers a more user-friendly way to access
        # a selection of the states e.g., module states could be added in there, while intervention states might not
        self._states = {}

    def __len__(self):
        """ Length of people """
        return len(self.uid)

    def register_state(self, state, die=True):
        """
        Register a state with the People instance for dynamic resizing

        All states should be registered by this function for the purpose of connecting them to the
        People's UIDs and to have them be automatically resized when the number of agents changes.
        This operation is normally triggered as part of initializing the state (via `State.initialize()`)
        """
        if id(state) not in self._states:
            self._states[id(state)] = state
        elif die:
            errormsg = f'Cannot add state {state} since already added'
            raise ValueError(errormsg)
        return

    def grow(self, n, new_slots=None):
        """
        Increase the number of agents

        :param n: Integer number of agents to add
        :param new_slots: Optionally specify the slots to assign for the new agents. Otherwise, it will default to the new UIDs
        """

        if n == 0:
            return np.array([], dtype=ss.dtypes.int)

        start_uid = len(self._uid_map)
        start_idx = len(self.uid)

        new_uids = np.arange(start_uid, start_uid + n)
        new_inds = np.arange(start_idx, start_idx + n)

        self._uid_map.grow(n)
        self._uid_map[new_uids] = new_inds

        self.uid.grow(n)
        self.uid[new_inds] = new_uids

        # We need to grow the slots as well
        self.slot.grow(new_uids)
        self.slot[new_uids] = new_slots if new_slots is not None else new_uids

        for state in self._states.values():
            state.grow(new_uids)

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
        for state in self._states.values(): # includes self.slot
            state._trim(keep_inds)

        # Update the UID map
        self._uid_map[:] = ss.INT_NAN  # Clear out all previously used UIDs
        self._uid_map[keep_uids] = np.arange(0, len(keep_uids))  # Assign the array indices for all of the current UIDs

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

    def __setstate__(self, state):
        """
        Set the state upon unpickling/deepcopying

        If a People instance is copied (by any mechanism) then the keys in the `_states`
        registry will no longer match the memory addresses of the new copied states. Therefore,
        after copying, we need to re-create the states registry with the new object IDs
        """
        state['_states'] =  {id(v):v for v in state['_states'].values()}
        self.__dict__ = state


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

    def __init__(self, n_agents, age_data=None, extra_states=None, rand_seed=0):
        """ Initialize """

        super().__init__(n_agents)

        self.initialized = False
        self.version = ss.__version__  # Store version info

        # Handle states
        states = [
            ss.State('age', float, np.nan), # NaN until conceived
            ss.State('female', bool, ss.bernoulli(name='female', p=0.5)),
            ss.State('ti_dead', int, ss.INT_NAN),  # Time index for death
            ss.State('alive', bool, True),  # Time index for death
            ss.State('scale', float, 1.0),
        ]
        states.extend(sc.promotetolist(extra_states))

        # Expose states with their original names directly as people attributes (e.g., `People.age`) and nested under states
        # (e.g., `People.states.age`)
        for state in states:
            self.states.append(state, overwrite=False)
            setattr(self, state.name, state)

        # Set initial age distribution - likely move this somewhere else later
        self.age_data_dist = self.get_age_dist(age_data)

        return

    @staticmethod
    def get_age_dist(age_data):
        """ Return an age distribution based on provided data """
        if age_data is None:
            dist = ss.uniform(low=0, high=100, name='Age distribution')
            return dist

        if sc.checktype(age_data, pd.DataFrame):
            age_bins = age_data['age'].values
            age_props = age_data['value'].values
            age_props = age_props / age_props.sum()
            return ss.choice(a=age_bins, p=age_props)


    def initialize(self, sim):
        """ Initialization """

        if self.initialized:
            return
        else:
            self.initialized = True # Expected by state.initialize()
        
        # For People initialization, first initialize slots, then initialize RNGs, then initialize remaining states
        # This is because some states may depend on RNGs being initialized to generate initial values
        self.slot.initialize(sim)
        self.slot[:] = self.uid

        # Initialize states
        # Age is handled separately because the default value for new agents is NaN until they are concieved/born whereas
        # the initial values need to depend on the current age distribution for the setting. In contrast, the sex for new
        # agents can be sampled from the same distribution used to initialize the population
        for state in self.states.values():
            state.initialize(sim)

        # Assign initial ages based on the current age distribution
        self.age_data_dist.initialize(module=self, sim=sim)
        self.age[:] = self.age_data_dist.rvs(self.uid)
        return

    def add_module(self, module, force=False):
        """
        Add a Module to the People instance

        This method is used to add a module to the People. It will register any module states with this
        people instance for dynamic resizing, and expose the states contained in the module to the user
        via `People.states.<module_name>.<state_name>`
        :param module:
        :param force:
        :return:
        """
        # Map the module's states into the People state ndict
        if hasattr(self, module.name) and not force:
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())

        # The entries created below make it possible to do `sim.people.hiv.susceptible` or
        # `sim.people.states['hiv.susceptible']` and have both of them work

        module_states = sc.objdict()
        setattr(self, module.name, module_states)
        self._register_module_states(module, module_states)
        return

    def _register_module_states(self, module, module_states):
        """Enable dot notation for module specific states:
         - `sim.people.hiv.susceptible` or
         - `sim.people.states['hiv.susceptible']`
        """

        for state in module.states:
            combined_name = module.name + '.' + state.name  # We will have to resolve how this works with multiple instances of the same module (e.g., for strains). The underlying machinery should be fine though, with People._states being flat and keyed by ID
            self.states[combined_name] = state  # Register the state on the user-facing side using the combined name. Within the original module, it can still be referenced by its original name
            pre, _, post = combined_name.rpartition('.')
            setattr(module_states, state.name, state)

        return

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def remove_dead(self, sim):
        """
        Remove dead agents
        """
        uids_to_remove = ss.true(self.dead)
        if len(uids_to_remove):
            self.remove(uids_to_remove)

        # Remove the UIDs from the network too
        for network in sim.networks.values():
            network.remove_uids(uids_to_remove)

        return

    def update_post(self, sim):
        """
        Final updates at the very end of the timestep

        :param sim:
        :return:
        """
        self.age[self.alive] += self.dt
        return

    def resolve_deaths(self):
        """
        Carry out any deaths that took place this timestep

        :return:
        """
        death_uids = ss.true(self.ti_dead <= self.ti)
        self.alive[death_uids] = False
        return death_uids


    @property
    def dead(self):
        """ Dead boolean """
        return ~self.alive

    @property
    def male(self):
        """ Male boolean """
        return ~self.female

    @property
    def f(self):
        """ Shorthand for female """
        return self.female

    @property
    def m(self):
        """ Shorthand for male """
        return self.male

    def init_results(self, sim):
        sim.results += [
            ss.Result(None, 'n_alive', sim.npts, ss.dtypes.int, scale=True),
            ss.Result(None, 'new_deaths', sim.npts, ss.dtypes.int, scale=True),
            ss.Result(None, 'cum_deaths', sim.npts, ss.dtypes.int, scale=True),
        ]
        return

    def update_results(self, sim):
        res = sim.results
        res.n_alive[self.ti] = np.count_nonzero(self.alive)
        res.new_deaths[self.ti] = np.count_nonzero(self.ti_dead == self.ti)
        res.cum_deaths[self.ti] = np.sum(res.new_deaths[:sim.ti])
        return

    def request_death(self, uids):
        """
        External-facing function to request an agent die at the current timestep

        In general, users should not directly interact with `People.ti_dead` to minimize
        interactions between modules (e.g., if a module requesting a future death, overwrites
        death due to a different module taking place at the current timestep).

        Modules that have a future time of death (e.g., due to disease duration) should keep
        track of that internally. When the module is ready to cause the agent to die, it should
        call this method, and can update its own results for the cause of death. This way, if
        multiple modules request death on the same day, they can each record a death due to their
        own cause.

        The actual deaths are resolved after modules have all run, but before analyzers. That way,
        regardless of whether removing dead agents is enabled or not, analyzers will be able to
        see and record outcomes for agents that died this timestep.

        **WARNING** - this function allows multiple modules to each independently carry out and
        record state changes associated with death. It is therefore important that they can
        guarantee that after requesting death, the death is guaranteed to occur.

        :param uids: Agent IDs to request deaths for
        :return: UIDs of agents that have been scheduled to die on this timestep
        """

        # Only update the time of death for agents that are currently alive. This way modules cannot
        # modify the time of death for agents that have already died. Noting that if remove_people is
        # enabled then often such agents would not be present in the simulation anyway
        uids = ss.true(self.alive[uids])
        self.ti_dead[uids] = self.ti
        return
