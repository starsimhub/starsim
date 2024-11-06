"""
Defines the People class and functions associated with making people
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from pathlib import Path

__all__ = ['People', 'Person']


class People(sc.prettyobj):
    """
    A class to perform all the operations on the people
    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized.

    Note that this class handles the mechanics of updating the actual people,
    as well as the additional housekeeping methods (saving, loading, exporting, etc.).

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        age_data (dataframe): a dataframe of years and population sizes, if available
        extra_states (list): non-default states to initialize

    **Examples**::
        ppl = ss.People(2000)
    """

    def __init__(self, n_agents, age_data=None, extra_states=None):
        """ Initialize """

        # We internally store states in a dict keyed by the memory ID of the state, so that we can have colliding names
        # e.g., across modules, but we will never change the size of a State multiple times in the same iteration over
        # _states. This is a hidden variable because it is internally used to synchronize the size of all States contained
        # within the sim, regardless of where they are. In contrast, `People.states` offers a more user-friendly way to access
        # a selection of the states e.g., module states could be added in there, while intervention states might not
        self._states = {}
        self.version = ss.__version__  # Store version info
        self.initialized = False

        # Handle the three fundamental arrays: UIDs for tracking agents, slots for
        # tracking random numbers, and AUIDs for tracking alive agents
        n = int(n_agents)
        uids = ss.uids(np.arange(n))
        self.auids = uids.copy() # This tracks all active UIDs (in practice, agents who are alive)
        self.uid = ss.IndexArr('uid')  # This variable tracks all UIDs
        self.slot = ss.IndexArr('slot') # A slot is a special state managed internally
        self.parent = ss.IndexArr('parent', label='UID of parent')  # UID of parent, if any, IndexArray?
        self.uid.grow(new_vals=uids)
        self.slot.grow(new_vals=uids)
        self.parent.grow(new_uids=uids, new_vals=np.full(len(uids), self.parent.nan))
        for state in [self.uid, self.slot]:
            state.people = self # Manually link to people since we don't want to link to states

        # Handle additional states
        extra_states = sc.promotetolist(extra_states)
        states = [
            ss.State('alive', default=True),  # Time index for death
            ss.State('female', default=ss.bernoulli(name='female', p=0.5)),
            ss.FloatArr('age', default=self.get_age_dist(age_data)), # NaN until conceived
            ss.FloatArr('ti_dead'),  # Time index for death
            ss.FloatArr('scale', default=1.0), # The scale factor for the agents (multiplied for making results)
        ]
        states.extend(extra_states)
        self.states = ss.ndict(type=ss.Arr)
        for state in states:
            self.states.append(state, overwrite=False)
            setattr(self, state.name, state)
            state.link_people(self)

        return

    @staticmethod
    def get_age_dist(age_data):
        """
        Return an age distribution based on provided data

        The data should be provided in the form of either an Nx2 array, a pandas series
        with age as the index and counts/probability as the value, or a pandas DataFrame
        with "age" and "value" as columns. Each of these should look like e.g.:

            age      value
            0      220.548
            1      206.188
            2      195.792
            3      187.442

        The ages will be interpreted as lower bin edges. An upper bin edge will
        automatically be added based on the final age plus the difference of the
        last two bins. To explicitly control the width of the upper age bin, add
        an extra entry to the ``age_data`` with a value of 0 and an age value
        corresponding to the desired upper age bound.

        Args:
            age_data: An array/series/dataframe with an index corresponding to age values, and a value corresponding to histogram counts
                         or relative proportions. A distribution will be estimated based on the histogram. The histogram will be
                         assumed to correspond to probability densitiy if the sum of the histogram values is equal to 1, otherwise
                         it will be assumed to correspond to counts.

        Note: age_data can also be provided as a string

        Returns:
            An ``ss.Dist`` instance that returns an age for newly created agents
        """
        if age_data is None:
            dist = ss.uniform(low=0, high=100, name='Age distribution')
        else:
            # Try loading from file
            if isinstance(age_data, str) or isinstance(age_data, Path):
                age_data = pd.read_csv(age_data)

            # Process
            if isinstance(age_data, np.ndarray): # TODO: accept output of np.histogram()
                age_bins = age_data[:,0]
                age_props = age_data[:,1]
            elif isinstance(age_data, pd.Series):
                age_bins = age_data.index
                age_props = age_data.values
            elif isinstance(age_data, pd.DataFrame):
                age_bins = age_data['age'].values
                age_props = age_data['value'].values

            # Convert to a histogram
            dist = ss.histogram(values=age_props, bins=age_bins, name='Age distribution')

        return dist

    def link_sim(self, sim):
        """ Initialization """
        if self.initialized:
            errormsg = 'Cannot re-initialize a People object directly; use sim.init(reset=True)'
            raise RuntimeError(errormsg)
        self.sim = sim # Store the sim
        ss.link_dists(obj=self.states, sim=sim, module=self, skip=[ss.Sim, ss.Module])
        return

    def add_module(self, module, force=False):
        """
        Add a Module to the People instance

        This method is used to add a module to the People. It will register any module states with this
        people instance for dynamic resizing, and expose the states contained in the module to the user
        via `People.states.<module_name>.<state_name>`

        The entries created below make it possible to do `sim.people.hiv.susceptible` or
        `sim.people.states['hiv.susceptible']` and have both of them work
        """
        # Map the module's states into the People state ndict
        if hasattr(self, module.name) and not force:
            raise Exception(f'Module {module.name} already added')

        if len(module.states):
            module_states = sc.objdict()
            setattr(self, module.name, module_states)
            for state in module.states:
                state.link_people(self)
                combined_name = module.name + '.' + state.name  # We will have to resolve how this works with multiple instances of the same module (e.g., for strains). The underlying machinery should be fine though, with People._states being flat and keyed by ID
                self.states[combined_name] = state # Register the state on the user-facing side using the combined name. Within the original module, it can still be referenced by its original name
                module_states[state.name] = state
        return

    def init_vals(self):
        """ Populate states with initial values, the final step of initialization """
        for state in self.states():
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def __bool__(self):
        """ Ensure that zero-length people are still truthy """
        return True

    def __len__(self):
        """ Length of people """
        return len(self.auids)

    @property
    def n_uids(self):
        return self.uid.len_used

    def _link_state(self, state, die=True):
        """
        Link a state with the People instance for dynamic resizing; usually called by
        state.link_people()

        All states should be registered by this function for the purpose of connecting them to the
        People's UIDs and to have them be automatically resized when the number of agents changes.
        This operation is normally triggered as part of initializing the state (via `State.init()`)
        """
        if id(state) not in self._states:
            self._states[id(state)] = state
        elif die:
            errormsg = f'Cannot add state {state} since already added'
            raise ValueError(errormsg)
        return

    def grow(self, n=None, new_slots=None):
        """
        Increase the number of agents

        Args:
            n: Integer number of agents to add
            new_slots: Optionally specify the slots to assign for the new agents. Otherwise, it will default to the new UIDs
        """
        if n is None:
            if new_slots is None:
                errormsg = 'Must supply either n or new_slots'
                raise ValueError(errormsg)
            else:
                n = len(new_slots)

        if n == 0:
            return np.array([], dtype=ss.dtypes.int)

        start_uid = self.uid.len_used
        stop_uid = start_uid + n
        new_uids = ss.uids(np.arange(start_uid, stop_uid))
        self.uid.grow(new_uids, new_vals=new_uids)

        # We need to grow the slots as well
        new_slots = new_slots if new_slots is not None else new_uids
        self.slot.grow(new_uids, new_vals=new_slots)

        self.parent.grow(new_uids, new_vals=self.parent.nan) # Grow parent array

        # Grow the states
        for state in self._states.values():
            state.grow(new_uids)

        # Finally, update the alive indices
        self.auids = self.auids.concat(new_uids)
        return new_uids

    def __getitem__(self, key):
        """
        Allow people['attr'] instead of getattr(people, 'attr')
        If the key is an integer, alias `people.person()` to return a `Person` instance
        """
        if isinstance(key, int):
            return self.person(key)
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        """ Ditto """
        return setattr(self, key, value)

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

        return

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def update_post(self):
        """ Final updates at the very end of the timestep """
        sim = self.sim
        if sim.pars.use_aging:
            self.age[self.alive.uids] += sim.t.dt_year
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
        self.ti_dead[uids] = self.sim.ti
        return

    def step_die(self):
        """ Carry out any deaths that took place this timestep """
        death_uids = (self.ti_dead <= self.sim.ti).uids
        self.alive[death_uids] = False

        # Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed
        # before analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep
        for disease in self.sim.diseases():
            if isinstance(disease, ss.Disease):
                disease.step_die(death_uids)

        return death_uids

    def remove_dead(self):
        """
        Remove dead agents
        """
        uids = self.dead.uids
        if len(uids):

            # Remove the UIDs from the networks too
            for network in self.sim.networks.values():
                network.remove_uids(uids) # TODO: only run once every nth timestep

            # Calculate the indices to keep
            self.auids = self.auids[np.isin(self.auids, np.unique(uids), assume_unique=True, invert=True, kind='sort')]

        return

    @property
    def dead(self):
        """ Dead boolean """
        return ~self.alive

    @property
    def male(self):
        """ Male boolean """
        return ~self.female

    def update_results(self):
        ti = self.sim.ti
        res = self.sim.results
        res.n_alive[ti] = np.count_nonzero(self.alive)
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res.new_deaths[:ti]) # TODO: inefficient to compute the cumulative sum on every timestep!
        return

    def finish_step(self):
        # self.update_results() # This is called separately
        self.remove_dead()
        self.update_post()
        return

    def person(self, ind):
        """ Get all the properties for a single person """
        person = Person()
        for key in ['uid', 'slot']:
            person[key] = self[key][ind]
        for key in self.states.keys():
            person[key] = self.states[key][ind]
        return person


class Person(sc.objdict):
    """ A simple class to hold all attributes of a person """
    def to_df(self):
        """ Convert to a dataframe """
        df = sc.dataframe.from_dict(self, orient='index', columns=['value'])
        df.index.name = 'key'
        return df
