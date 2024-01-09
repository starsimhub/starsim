"""
Defines the People class and functions associated with making people
"""

# %% Imports
import numpy as np
import pandas as pd
import sciris as sc
import stisim as ss
import scipy.stats as sps

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

    @property
    def rngs(self):
        return [x for x in self.__dict__.values() if isinstance(x, (ss.MultiRNG, ss.SingleRNG))]

    def __len__(self):
        """ Length of people """
        return len(self.uid)

    def add_state(self, state, die=True):
        if id(state) not in self._states:
            self._states[id(state)] = state
            self.states.append(state)  # Expose these states with their original names
            setattr(self, state.name, state)
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
            return np.array([], dtype=ss.int_)

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

    def __init__(self, n, age_data=None, extra_states=None, networks=None, rand_seed=0):
        """ Initialize """

        super().__init__(n)

        self.initialized = False
        self.version = ss.__version__  # Store version info

        # Handle states
        states = [
            ss.State('age', float, np.nan), # NaN until conceived
            ss.State('female', bool, sps.bernoulli(p=0.5)),
            ss.State('debut', float),
            ss.State('ti_dead', int, ss.INT_NAN),  # Time index for death
            ss.State('alive', bool, True),  # Time index for death
            ss.State('scale', float, 1.0),
        ]
        states.extend(sc.promotetolist(extra_states))
        for state in states:
            self.add_state(state)
        self._initialize_states(sim=None) # No sim yet, but initialize what we can
        
        self.networks = ss.Networks(networks)

        # Set initial age distribution - likely move this somewhere else later
        self.age_data_dist = self.get_age_dist(age_data)

        return

    @staticmethod
    def get_age_dist(age_data):
        """ Return an age distribution based on provided data """
        if age_data is None:
            dist = sps.uniform(loc=0, scale=100)  # loc and width
            return ss.ScipyDistribution(dist, 'Age distribution')

        if sc.checktype(age_data, pd.DataFrame):
            bb = np.append(age_data['age'].values, age_data['age'].values[-1] + 1)
            vv = age_data['value'].values
            #dist = sps.rv_histogram((vv, bb), density=False)
            return ss.ScipyHistogram((vv, bb), density=False, rng='Age distribution')

    def _initialize_states(self, sim=None):
        for state in self.states.values():
            state.initialize(sim=sim, people=self)  # Connect the state to this people instance
        return

    def initialize(self, sim):
        """ Initialization """
    
        # For People initialization, first initialize slots, then initialize RNGs, then initialize remaining states
        # This is because some states may depend on RNGs being initialized to generate initial values
        self.slot.initialize(sim)
        self.slot[:] = self.uid
    
        # Initialize all RNGs (noting that includes those that are declared in child classes)
        for rng in self.rngs:
            rng.initialize(sim.rng_container, self.slot)

        self.age_data_dist.initialize(sim, self)
            
        # Define age (CK: why is age handled differently than sex?)
        self._initialize_states(sim=sim)  # Now initialize with the sim
        self.age[:] = self.age_data_dist.rvs(size=self.uid)
        
        self.initialized = True
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

    def update_networks(self):
        """
        Update networks
        """
        return self.networks.update(self)

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

    @property
    def f(self):
        """ Shorthand for female """
        return self.female

    @property
    def m(self):
        """ Shorthand for male """
        return self.male

    def init_results(self, sim):
        sim.results += ss.Result(None, 'n_alive', sim.npts, ss.int_, scale=True)
        sim.results += ss.Result(None, 'new_deaths', sim.npts, ss.int_, scale=True)
        return

    def update_results(self, sim):
        sim.results.n_alive[self.ti] = np.count_nonzero(self.alive)
        sim.results.new_deaths[self.ti] = np.count_nonzero(self.ti_dead == self.ti)
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
