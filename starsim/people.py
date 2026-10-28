"""
Defines the People class and functions associated with making people
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from pathlib import Path
import matplotlib.pyplot as plt
import numba as nb



class LazyPeople:
    """
    The people's properties, before there are any people: `ss.ppl.<name>` is a lazy reference to the property `<name>` of each person (e.g. `ss.ppl.age`, `ss.ppl.urban`), checked when the sim is built

    It can be used in a condition (`ss.ppl.age < 15`, for `where=`) and, as the person's value,
    in a rate, a branch probability, or a result (`expit(b0 + b1*ss.ppl.urban)`); a condition in a
    rate is 1 for the people it holds for and 0 for the others (`0.3*(ss.ppl.age > 65)`).

    Examples:
        ```python
        p_use = expit(b0 + b_age*ss.ppl.age + b_urban*ss.ppl.urban)
        kids = ss.ppl.age < 15
        ```
    """
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return ss.conditions.Cond('people', name)

    def get(self, name, default):
        """
        `ss.ppl.<name>`, or `default` for everyone if nothing adds the property `<name>` to the people, e.g. one that a
        module adds (`people.add_state()`), read by a rate whether or not that module is in the sim

        **Example**::

            onset = S >> D @ ss.peryear(0.01)*(1 + 0.5*ss.ppl.get('unstable_housing', 0))
        """
        return ss.conditions.Cond('people', name, default)

    def __repr__(self):
        return 'ss.ppl'

ppl = LazyPeople()


class AgentScale(ss.FloatArr):
    """ The number of people each agent stands for relative to the others (`people.scale`); setting it other than 1 makes results, and the counts in a declared model's rates (`I.n`, `I.frac`, and the share infectious in a transmission), add up the agents' weights """
    weighted = False # Set once any agent's scale is set other than 1, so that results are worked out as before until then

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not self.weighted and np.any(np.asarray(value) != 1):
            self.weighted = True
        return


class MadeOnUse:
    """
    One of the four fundamental arrays of `ss.People`, made on first use (see `People.make_uids()`)

    A descriptor rather than a branch of `People.__getattr__()`, so that a subclass with a `__getattr__()` of its
    own can't hide it. It defines only `__get__()`, so once the arrays are made, the attribute in the instance's
    `__dict__` is found first and this costs nothing.
    """
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if 'n_agents_init' not in obj.__dict__ or 'uid' in obj.__dict__: # Not yet built (e.g. while unpickling), or made and since removed
            raise AttributeError(f'People object has no attribute "{self.name}"')
        obj.make_uids()
        return obj.__dict__[self.name]


class People:
    """
    A class to perform all the operations on the people
    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized.

    Note that this class handles the mechanics of updating the actual people,
    as well as the additional housekeeping methods (saving, loading, exporting, etc.).

    Args:
        n_agents (int): the initial number of agents in the model
        age_data (dataframe/series/array/str/`ss.Dist`): the age distribution: a dataframe with columns `age` (the lower edge of each age bin) and `value` (the number or share of people in it), a series of values indexed by age, a two-column array, a CSV file with the dataframe's columns, or a distribution of ages. This gives agents ages; for age groups in a compartmental model, use `disease.stratify(age=[0, 15, 65], split=[0.3, 0.6, 0.1])` instead
        extra_states (list): non-default states to initialize
        p_female (float): the share of agents who are female
        mock (bool): if True, initialize the People object with a mock Sim object (for debugging only)

    Examples:
        ```python
        ppl = ss.People(2000)
        kids = ss.ppl.age < 15 # A lazy condition, for a sim run as agents or as compartments (see ss.ppl)
        ```
    """
    auids  = MadeOnUse() # The four fundamental arrays, made on first use: see make_uids()
    uid    = MadeOnUse()
    slot   = MadeOnUse()
    parent = MadeOnUse()

    def __init__(self, n_agents, age_data=None, extra_states=None, p_female=0.5, mock=False):
        """ Initialize """

        # We internally store states in a dict keyed by the memory ID of the state, so that we can have colliding names
        # e.g., across modules, but we will never change the size of a state multiple times in the same iteration over
        # _states. This is a hidden variable because it is internally used to synchronize the size of all states contained
        # within the sim, regardless of where they are. In contrast, `People.states` offers a more user-friendly way to access
        # a selection of the states e.g., module states could be added in there, while intervention states might not.
        self._states = {}
        self.version = ss.__version__  # Store version info
        self.untraited = [] # Ranges of UIDs made by code that sets their properties afterwards (grow() without age=), whose traits are drawn at the next step: see grow()
        self.has_unborn = False # Whether a module makes agents before they are born, with a negative age (e.g. ss.Pregnancy's fetuses), who are not counted among the people: see unborn_uids()
        self.initialized = False
        self.n_agents_init = round(n_agents) # The nearest whole number, as ss.SimPars takes it (e.g. a number a fit chose)
        self.named = None # For a named population (ss.people('Kenya')), where it came from: see ss.people()

        # The four fundamental arrays (auids, uid, slot, and parent) are made on first use: see make_uids()

        # Handle additional states
        extra_states = sc.promotetolist(extra_states)
        states = [
            ss.BoolState('alive', default=True),  # Time index for death
            ss.FloatArr('age', default=self.get_age_dist(age_data)),  # NaN until conceived
            ss.BoolState('female', default=ss.bernoulli(name='female', p=p_female)),
            ss.TimeArr('ti_dead'),  # Time index for death
            ss.TimeArr('ti_removed'),  # Time index for removal (e.g. emigration)
            AgentScale('scale', default=1.0, shared=True), # The scale factor for the agents (multiplied for making results); no memory until one is set
            ss.Arr('cause_of_death', dtype=np.int16, default=-1, nan=-1, label='Cause of death', shared=True), # Its position in people.causes (-1 for none); no memory until someone dies
        ]
        states.extend(extra_states)
        self.states = ss.ndict(type=ss.Arr)
        for state in states:
            self.states.append(state)
            setattr(self, state.name, state)
            state.link_people(self)
        self.causes = [] # The causes of death, by name (see request_death()), in the order first seen: people.cause_of_death is the position in this list
        self._claims = [] # The deaths requested in this step, with their causes: see step_die()
        self._died_of = {} # Who died of each cause in this step
        self._cause_dist = ss.random(name='cause_of_death') # To choose one cause for someone whose death several modules request in a step

        if mock:
            self.init_mock()

        return

    def make_uids(self):
        """
        Make the four fundamental arrays: UIDs for tracking agents, slots for tracking random
        numbers, AUIDs for tracking alive agents, and agents' parents

        They are made on first use rather than in `__init__()`, so that a People not yet in
        a sim (such as the copy a sim keeps of itself before it is initialized, for `sim.new()`) holds no arrays per agent.
        The slots read the UIDs, and the parents are all none, until one is set (see `shared`
        in `ss.Arr`), so they take no memory unless something (e.g. `ss.Pregnancy`, with fetuses as agents) sets them.
        """
        uids = ss.uids(np.arange(int(self.n_agents_init)))
        self.auids = uids.astype(ss.dtypes.uid).view(ss.uids) # This tracks all active UIDs (in practice, agents who are alive), as ss.dtypes.uid (64-bit, which index arrays fastest)
        self.uid    = ss.arrays.UIDArr('uid')  # This variable tracks all UIDs; it takes no memory, since a UID is the agent's index
        self.slot   = ss.IndexArr('slot', shared=self.uid) # A slot is a special state managed internally to generate random numbers
        self.parent = ss.IndexArr('parent', label='UID of parent', shared=True)  # UID of parent, if any (used with pregnancy)
        for state in [self.uid, self.slot, self.parent]:
            state.people = self # Manually link to people since we need to set these up before using state.link_people()
            state.grow(new_uids=uids, new_vals=uids if state is self.uid else None)
        return

    def slot_raw(self):
        """ The slot of every agent, by UID, for a kernel; or None while the slots are the UIDs (see `make_uids()`), which have no array of their own """
        return None if isinstance(self.slot.shared, ss.arrays.UIDArr) else self.slot.raw

    def phase(self, uids=None):
        """
        Each agent's phase, in [0, 1), by UID: a time between steps, t, is taken at step floor(t + phase), so it is rounded up
        or down at random in proportion (2.3 steps is 2 for 70% of agents and 3 for the rest), keeping its mean, with no draws:
        both a time due in a disease written as code (`ss.TimeArr.due()`) and a dwell time of the agent engine. Spread evenly
        over the slots (so an agent has the same phase in every scenario), and shifted by the sim's seed

        Args:
            uids (uids): the agents to give the phases of (worked out for them alone); by default, every agent's, kept until agents are made
        """
        if getattr(self, '_phase_shift', None) is None:
            seed = getattr(getattr(self, 'sim', None), 'pars', {}).get('rand_seed', 0)
            self._phase_shift = np.random.default_rng(seed).random()
        of = lambda slot: (slot*0.6180339887498949 + self._phase_shift) % 1 # The golden ratio's fractional part spreads consecutive slots evenly
        slots = self.slot_raw()
        if uids is not None:
            return of(uids if slots is None else slots[uids])
        n, used = self.uid.len_tot, self.uid.len_used
        phase, done = getattr(self, '_phase', (None, 0))
        if phase is None or len(phase) != n: # The arrays have grown: keep the phases worked out so far
            old, phase = phase, np.zeros(n, dtype=np.float32)
            if old is not None:
                phase[:done] = old[:done]
        if used > done: # Slots are set when agents are made, so only the new agents' phases are worked out
            phase[done:used] = of(np.arange(done, used) if slots is None else slots[done:used])
        self._phase = (phase, used)
        return phase

    def _link_state(self, state, die=True):
        """
        Link a state with the People instance for dynamic resizing; usually called by
        `state.link_people()`

        All states should be registered by this function for the purpose of connecting them to the
        People's UIDs and to have them be automatically resized when the number of agents changes.
        This operation is normally triggered as part of initializing the state (via `Arr.init()`)

        Args:
            state (Arr): the state to register
            die (bool): if True (default), raise an error if the state has already been registered
        """
        if id(state) not in self._states:
            self._states[id(state)] = state
        elif die:
            errormsg = f'Cannot add state {state} since already added'
            raise ValueError(errormsg)
        return

    def __getattr__(self, attr):
        """ Called only if the attribute isn't found: in a compartmental run, per-agent attributes that are not stratifications do not exist """
        missing = self.__dict__.get('_no_groups', [])
        if attr in missing:
            raise ss.CapabilityError(f'people.{attr} was asked for, but it is a per-agent attribute and not a stratification, so it does not exist in a compartmental run.\n'
                                     f'     Stratify by it (e.g. sim.stratify({attr}=[...])), or run with solver="abm".')
        if '_no_groups' in self.__dict__ and not attr.startswith('_'): # A compartmental run, whose people lack what networks and modules written as code add to agents
            raise AttributeError(f'People object has no attribute "{attr}".\n     In a compartmental run, people are one row per group, without the attributes that networks and modules written as code give agents; if it is one of those, run with solver="abm"')
        raise AttributeError(f'People object has no attribute "{attr}"')

    def __getitem__(self, key):
        """
        Allow people['attr'] instead of getattr(people, 'attr')
        If the key is an integer, alias `people.person()` to return a `Person` instance
        """
        if isinstance(key, (int, np.integer)):
            return self.person(key)
        elif isinstance(key, str):
            return getattr(self, key)
        else:
            errormsg = f"Cannot handle {key} ({type(key)}: must be str to access an attribute, e.g. people['age'], or int to access a person, e.g. people[14]"
            raise TypeError(errormsg)

    def __setitem__(self, key, value):
        """ Allow people['attr'] = value as an alias for setattr """
        return setattr(self, key, value)

    def __iter__(self):
        """ Iterate over (alive) people """
        for uid in self.auids:
            yield self.person(uid)

    def __contains__(self, key):
        """ Whether people have a property by this name, e.g. `'patch' in sim.people` (which would otherwise go through every person) """
        if isinstance(key, str):
            return key in self.keys()
        return any(person is key for person in self)

    def __setstate__(self, state):
        """
        Set the state upon unpickling/deepcopying

        If a People instance is copied (by any mechanism) then the keys in the `_states`
        registry will no longer match the memory addresses of the new copied states. Therefore,
        after copying, we need to re-create the states registry with the new object IDs
        """
        state['_states'] = {id(v):v for v in state['_states'].values()}
        self.__dict__ = state
        return
    
    def __bool__(self):
        """ Ensure that zero-length people are still truthy """
        return True

    def __len__(self):
        """ Length of people """
        return len(self.auids)

    @property
    def n_alive(self):
        """
        The number of living agents, which is what `results.alive` records (with each agent counted by its weight if agents' weights differ: see `people.scale`)

        This is not the same as `len(people)`: agents who died on the current timestep are
        not removed from `auids` until `finish_step()`, so that analyzers can still inspect
        them, and the unborn (agents made at conception by `ss.Pregnancy`) are not counted. Use
        this rather than `len(people)` for any per-timestep denominator.
        """
        unborn = self.unborn_uids()
        return np.count_nonzero(self.alive) - np.count_nonzero(self.alive.raw[unborn])
    
    def __repr__(self):
        return self.brief(output=True)

    def __str__(self):
        out = self.brief(output=True)[:-1] # Remove trailing parenthesis
        out += f'; states=[{sc.strjoin(self.states.keys())}])'
        return out

    def disp(self):
        """ Full object information """
        return sc.pr(self)

    @property
    def grouped(self):
        """ Whether these are the groups of a compartmental run (one row per group), not agents """
        sim = getattr(self, 'sim', None)
        return sim is not None and getattr(sim, 'solver', 'abm') != 'abm' and bool(getattr(sim, 'declared', None))

    @property
    def n(self):
        """ In a compartmental run, the number of people in each group now, from the compartments (one row per group); as agents, the number alive """
        if not self.grouped:
            return len(self)
        engine = self.sim.declared[0].engine
        return engine.n_people(total=False) if hasattr(engine, 'n_people') else np.full(self.n_agents_init, np.nan)

    def brief(self, output=False):
        n = self.n_agents_init
        if self.grouped and self.initialized: # A compartmental run: the rows are groups of people
            out = f'People({n:n} groups; {float(np.sum(self.n)):n} people; age={self.age.mean():0.1f}±{self.age.std():0.1f} over the groups)'
            return out if output else print(out)
        if not self.initialized: # Age and other states hold uninitialized memory until init_vals() runs
            out = f'People({n=:n}; uninitialized)'
        else:
            alive = len(self)
            alive_str = f'{alive=:n}; ' if alive != n else ''
            age_mean = self.age.mean()
            age_std = self.age.std()
            out = f'People({n=:n}; {alive_str}age={age_mean:0.1f}±{age_std:0.1f})'
        return out if output else print(out)
    
    def keys(self):
        """ Return keys of the states stored in the People object """
        return self.states.keys()

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
        an extra entry to the `age_data` with a value of 0 and an age value
        corresponding to the desired upper age bound.

        Args:
            age_data: An array/series/dataframe with an index corresponding to age values, and a value corresponding to histogram counts
                         or relative proportions. A distribution will be estimated based on the histogram. The histogram will be
                         assumed to correspond to probability densitiy if the sum of the histogram values is equal to 1, otherwise
                         it will be assumed to correspond to counts.

        Note: `age_data` can also be provided as a string (interpreted as a filename).

        If no value is provided, the ages are the world's (`ss.data.age('world')`), as for a sim
        with no named population.

        Returns:
            An [`ss.Dist`](`starsim.distributions.Dist`) instance that returns an age for newly created agents
        """
        if age_data is None:
            age_data = ss.data.age('world')
        if isinstance(age_data, ss.Dist): # A distribution of ages, used as is
            dist = age_data
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
            else:
                errormsg = f'age_data must be a file, an array, a Series, a dataframe, or an ss.Dist, not {type(age_data)}'
                raise TypeError(errormsg)

            # Convert to a histogram
            dist = ss.histogram(values=age_props, bins=age_bins, name='Age distribution')

        return dist

    def link_sim(self, sim, init=False):
        """ Initialization """
        if self.initialized:
            errormsg = 'Cannot re-initialize a People object directly; use sim.init(reset=True)'
            raise RuntimeError(errormsg)
        self.sim = sim # Store the sim
        for arr in [self.ti_dead, self.ti_removed]:
            arr.t = sim.t
        ss.distributions.link_dists(obj=self.states, sim=sim, module=self, skip=[ss.Sim, ss.Module], init=init)
        return

    @property
    def auto_state_list(self):
        """
        List of "automatic" states with boolean type (`ss.BoolState`)

        BoolStates automatically generate results of the same names (alive, female, etc.), the number in each.
        This only includes BoolStates that belong directly to People, not those
        from modules (which have their own result generation).
        For a list of all states, see `People.states`.
        """
        # Only include BoolStates that belong directly to People, not module states
        # Module states have combined names like "sir.susceptible" with a dot
        return [state for key, state in self.states.items()
                if isinstance(state, ss.BoolState) and '.' not in key]

    def init_results(self):
        """
        Initialize results for People; called during `Sim.init_results()`

        By default, People reports on counts for any `ss.BoolState` instances, e.g. if
        People contains a `ss.BoolState` called 'alive' it will automatically contain a
        Result for 'alive' (the number alive). For identical behavior that does not automatically
        generate results, use `ss.BoolArr` instead of `ss.BoolState`.

        This method also creates the standard people-related results like new_deaths,
        new_emigrants, and cum_deaths.

        If `sim.pars.people_results` is False, People-level result collection is disabled
        (no results are created here and `People.update_results` is skipped in the loop).
        """
        if not self.sim.pars.get('people_results', True): # Opt-out: skip People-level results entirely
            return
        kw = dict(module='People', shape=self.sim.t.npts, timevec=self.sim.t.timevec, dtype=int, scale=True)
        results = []

        # Create results for all BoolStates
        for state in self.auto_state_list:
            results.append(ss.Result(state.name, label=state.label, summarize_by='last', **kw))

        # Add standard people-related results
        results.extend([
            ss.Result('new_deaths', label='Deaths', **kw),
            ss.Result('new_emigrants', label='Emigrants', **kw),
            ss.Result('cum_deaths', label='Cumulative deaths', **kw),
        ])

        # Add results to the sim
        self.sim.results += results
        return

    def add_module(self, module):
        """
        Add a Module to the People instance

        This method is used to add a module to the People. It will register any module states with this
        people instance for dynamic resizing, and expose the module via People, in two ways

        - `People.<module>` will be a reference to the module itself, enabling access via `people.hiv.susceptible`
        - `People.states['<module>.<state>']` will be a reference to the state, enabling access via `people.states['hiv.susceptible']`

        """

        # Connect the module as People attribute
        if hasattr(self, module.name):
            existing = getattr(self, module.name)
            what = 'a module of that name is already in the sim' if isinstance(existing, ss.Module) else f'people already have a property called "{module.name}"'
            raise AttributeError(f'The {type(module).__name__} "{module.name}" cannot be added to the people, since {what}; give it another name, e.g. name="{module.name}_mod"')
        else:
            setattr(self, module.name, module)

        if len(module.state_list):
            for state in module.state_list:
                state.link_people(self)
                if isinstance(state, ss.TimeArr) and state.t is None: # Times as steps of the module's timeline (see due())
                    state.t = module.t
                combined_name = module.name + '.' + state.name  # We will have to resolve how this works with multiple instances of the same module (e.g., for strains). The underlying machinery should be fine though, with People._states being flat and keyed by ID
                self.states[combined_name] = state # Register the state on the user-facing side using the combined name. Within the original module, it can still be referenced by its original name
        return

    def add_state(self, state):
        """
        Add a property of each person, from a module (e.g. a social determinant of health that several modules read, as `ss.ppl.<name>`)

        Idempotent: if the people have a property of that name already (added by another module, say), it is
        returned as it is, so each module that reads it can add it. Call it before the people's values are made,
        e.g. in a module's `init_pre()`; added later, its values are made at once.

        Args:
            state (Arr): the array, e.g. `ss.BoolArr('housing', default=False)` or `ss.FloatArr('adherence', default=ss.uniform())`

        Returns:
            The property's array (the one given, or the one added before)

        **Example**::

            def init_pre(self, sim):
                super().init_pre(sim)
                self.housing = sim.people.add_state(ss.BoolArr('housing', default=ss.bernoulli(0.8)))
        """
        name = state.name
        if name in self.states:
            return self.states[name]
        if name in self.__dict__ or name in dir(type(self)):
            errormsg = f'The people cannot have a property called "{name}": it is already {"a module" if isinstance(getattr(self, name), ss.Module) else "an attribute"} of the people. Give the array another name.'
            raise AttributeError(errormsg)
        self.states.append(state)
        setattr(self, name, state)
        state.link_people(self)
        sim = self.__dict__.get('sim')
        if sim is not None:
            dists = ss.distributions.link_dists(obj=state, sim=sim, module=self, skip=[ss.Sim, ss.Module, People]) # Its own (not those of the people it links to)
            if self.initialized: # Initialized with the others otherwise (see ss.Sim.init())
                for trace,dist in dists.items():
                    dist.init(trace=f'people_{name}_{trace}', seed=sim.dists.base_seed, sim=sim)
        if self.initialized:
            state.init_vals()
        return state

    def init_vals(self):
        """ Populate states with initial values, the final step of initialization """
        for state in self.states():
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def init_mock(self):
        """ Initialize with a mock simulation (for debugging purposes only) """
        sim = ss.debugtools.mock_sim(n_agents=self.n_agents)
        self.link_sim(sim, init=True)
        self.init_vals()
        return

    @property
    def n_agents(self):
        """ Alias for len(people) """
        return len(self)

    @property
    def n_uids(self):
        """ Number of UIDs used in People """
        return self.uid.len_used
    
    @property
    def dead(self):
        """
        Whether each agent has died

        Unlike other arrays, this covers every agent ever created (indexed by UID),
        since dead agents are removed from the active agents. Agents who were removed
        for other reasons (e.g. emigration) are not counted as dead.

        Examples:
            ```python
            sim = ss.Sim(diseases=dict(type='sir', p_death=0.5), networks='random').run()
            n_deaths = sim.people.dead.sum()
            dead_uids = sim.people.dead.uids
            ```
        """
        n = self.n_uids
        raw = np.zeros(self.uid.len_tot, dtype=bool)
        raw[:n] = ~self.alive.raw[:n] & ~np.isnan(self.ti_dead.raw[:n])
        dead = self.alive.asnew(raw, cls=ss.BoolArr, name='dead', copy=False)
        dead.people = sc.objdict(auids=ss.uids(np.arange(n))) # Cover all agents, not just active ones
        return dead

    @property
    def born(self):
        """ Born boolean """
        return self.age >= 0

    @property
    def male(self):
        """ Male boolean  (female is a state) """
        return ~self.female

    def grow(self, n=None, new_slots=None, placed=False, parent=None, age=None):
        """
        Increase the number of agents

        Args:
            n: Integer number of agents to add
            new_slots: Optionally specify the slots to assign for the new agents. Otherwise, it will default to the new UIDs
            placed: whether the caller gives them their levels of the sim's stratifications; if not, they are drawn from each one's split
            parent: the parent of each (e.g. for newborns), set before their levels are drawn, so they can be born where their parent is
            age: their age (e.g. 0 for newborns), likewise
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

        # We need to grow the slots as well; unless they are given, they are the UIDs
        if new_slots is None and self.slot.shared is None:
            new_slots = new_uids
        self.slot.grow(new_uids, new_vals=new_slots)

        self.parent.grow(new_uids) # Grow parent array: none, until set

        # Grow the states
        for state in self._states.values():
            state.grow(new_uids)
        if parent is not None:
            self.parent[new_uids] = parent
        if age is not None:
            self.age[new_uids] = age

        # Finally, update the alive indices
        self.add_active(new_uids)
        sim = getattr(self, 'sim', None)
        if getattr(sim, 'solver', None) == 'abm': # A dimension called sex is people.female
            sim.groups.set_sex(new_uids)
            if not placed and getattr(sim.groups, 'vals_set', False) and sim.groups.strats: # E.g. newborns from ss.Births (or ss.Pregnancy's before the start), who would otherwise all be at the first level
                sim.groups.place_new(new_uids)
            if not placed and age is None: # Made by code that sets their age (and the rest) itself afterwards, as Starsim 3's pregnancy did: their traits are drawn at the next step, from what it set (FP91: a fetus's trait by age was drawn from a default adult age)
                self.untraited.append((start_uid, stop_uid))
            elif not placed: # Their traits (ss.attr(dist)), as states get their defaults; people placed by a declared module get them there (see ss.solvers.Agents.add_people())
                for mod in sim.declared:
                    if getattr(mod.engine, '_started', False):
                        mod.engine.draw_traits(new_uids)
        return new_uids

    def add_active(self, new_uids):
        """ Add agents to the active agents, in place where there is room (so a few births do not copy all of them) """
        auids = self.auids
        n = len(auids)
        k = len(new_uids)
        room = getattr(self, '_auids_room', None)
        if room is not None and auids is self._auids_view and n + k <= len(room): # The active agents are still the first part of the array with room
            room[n:n+k] = new_uids
        else:
            room = np.empty(n + k + max(1000, (n + k)//10), dtype=ss.dtypes.uid)
            room[:n] = auids
            room[n:n+k] = new_uids
            self._auids_room = room
        self.auids = self._auids_view = room[:n+k].view(ss.uids)
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
        if sim.pars.use_aging and sim.solver == 'abm': # In a compartmental run, each row is a group, whose age stays the middle of the group (people move between groups as they age)
            self.age[self.alive.uids] += sim.t.dt_year
        return

    def request_death(self, uids, cause=None, hazard=None):
        """
        External-facing function to request an agent die at the current timestep

        In general, users should not directly interact with `People.ti_dead` to minimize
        interactions between modules (e.g., if a module requesting a future death, overwrites
        death due to a different module taking place at the current timestep).

        Modules that have a future time of death (e.g., due to disease duration) should keep
        track of that internally. When the module is ready to cause the agent to die, it should
        call this method, with itself as the cause.

        The actual deaths are resolved after modules have all run, but before analyzers. That way,
        regardless of whether removing dead agents is enabled or not, analyzers will be able to
        see and record outcomes for agents that died this timestep.

        Each person dies once, of one cause: if several modules request the death of the same person
        in the same step, one of them is chosen at random (whatever the order of the modules, with
        common random numbers): in proportion to the hazards of the claims if each gave one (`hazard=`),
        and otherwise each equally likely. The cause is kept in `people.cause_of_death`
        (see `people.causes`). A module counts its deaths after they are resolved, with
        `people.died_of(module)`, so a death is counted once.

        Args:
            uids: Agent IDs to request deaths for
            cause (Module/str): what they die of: a module (by its name) or a name (default: none)
            hazard (float/array): the hazard of this death, one number or one per agent (in any unit, the same for every module, e.g. per year), to weight the choice of cause among the claims on the same person in the step (e.g. `dis.death.hazard(uids)` for a declared transition)

        Returns:
            UIDs of agents that have been scheduled to die on this timestep
        """
        self.ti_dead[uids] = self.sim.ti
        if len(uids):
            name = cause.name if isinstance(cause, ss.Module) else cause
            if name not in self.causes:
                self.causes.append(name)
            hazard = np.full(len(uids), np.nan) if hazard is None else np.broadcast_to(np.asarray(hazard, dtype=float), len(uids))
            self._claims.append((ss.uids(uids), self.causes.index(name), hazard))
        return

    def died_of(self, cause):
        """ The agents who died in this step of a cause (a module or a name), once each death is resolved to one cause (see `request_death()`) """
        name = cause.name if isinstance(cause, ss.Module) else cause
        return self._died_of.get(name, ss.uids())

    def resolve_causes(self, death_uids):
        """ Give each agent who dies in this step one cause, from the deaths requested (see `request_death()`); called by `step_die()` """
        claims, self._claims = self._claims, []
        self._died_of = {}
        if not claims:
            return
        uids = np.concatenate([u for u,c,h in claims]).astype(np.int64)
        codes = np.concatenate([np.full(len(u), c) for u,c,h in claims])
        hazards = np.concatenate([h for u,c,h in claims])
        keep = np.isin(uids, death_uids) # E.g. not someone who left instead
        uids, codes, hazards = uids[keep], codes[keep], hazards[keep]
        rank = np.argsort(np.argsort([str(name) for name in self.causes]))[codes] # By the name of the cause, so the order of the modules does not matter
        order = np.lexsort((rank, uids))
        uids, codes, rank, hazards = uids[order], codes[order], rank[order], hazards[order]
        new = np.ones(len(uids), dtype=bool)
        new[1:] = (uids[1:] != uids[:-1]) | (rank[1:] != rank[:-1]) # One claim per cause per person
        uids, codes, hazards = uids[new], codes[new], hazards[new]
        people, start, counts = np.unique(uids, return_index=True, return_counts=True)
        pick = start.copy()
        several = counts > 1
        if several.any(): # One of the causes: in proportion to their hazards if every claim on the person gave one (MG8), else each equally likely
            u = self._cause_dist.rvs(ss.uids(people[several]))
            if np.isnan(hazards).all():
                pick[several] += np.minimum((u*counts[several]).astype(int), counts[several] - 1)
            else:
                group = np.repeat(np.arange(len(people)), counts)
                w = np.maximum(hazards, 0)
                equal = np.logical_or.reduceat(np.isnan(w), start) | (np.add.reduceat(np.nan_to_num(w), start) <= 0)
                w = np.where(equal[group], 1.0, w)
                cum = np.cumsum(w)
                within = cum - np.repeat(cum[start] - w[start], counts) # The running total within each person's claims
                target = np.zeros(len(people))
                target[several] = u*within[start + counts - 1][several]
                below = np.add.reduceat((within <= target[group]).astype(int), start)
                pick[several] += np.minimum(below[several], counts[several] - 1)
        people, codes = ss.uids(people), codes[pick]
        self.cause_of_death[people] = codes
        for code in np.unique(codes):
            self._died_of[self.causes[code]] = people[codes == code]
        return

    def request_removal(self, uids):
        """
        Request an agent to be removed from the simulation
        Similar to request_death, but in this case the agent will be recorded as removed instead of dead
        """
        self.ti_removed[uids] = self.sim.ti
        return

    def step_die(self):
        """ Carry out any deaths or removals that took place this timestep """
        ti = self.sim.ti
        deaths, n_alive = ss.solvers.backends.passes.due_alive(self.alive.raw[:self.uid.len_used], self.ti_dead.raw, self.ti_removed.raw, ti) # As ((self.ti_dead <= ti) | (self.ti_removed <= ti)).uids, in one pass by UID, since the active agents are those alive...
        if n_alive != len(self.auids): # ...unless something else has changed who is alive: then from the active agents themselves
            deaths = ss.solvers.backends.passes.due(self.auids.view(np.ndarray), self.ti_dead.raw, self.ti_removed.raw, ti)
        death_uids = ss.uids(deaths)
        self.resolve_causes(death_uids[self.ti_dead.raw[death_uids] <= ti])
        self.alive[death_uids] = False  # Whilst not dead, removed agents should not be included in alive totals
        self._dying = death_uids # Still active until the end of the step: see update_results() and remove_dead()

        # Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed
        # before analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep
        diseases = [disease for disease in self.sim.diseases() if isinstance(disease, ss.Disease)]
        diseases += [mod for mod in getattr(self.sim, 'declared', []) if all(mod is not disease for disease in diseases)] # A declared disease given as another kind of module: its agents leave its states as well
        for disease in diseases:
            disease.step_die(death_uids)

        return death_uids

    def remove_dead(self, aging=False):
        """
        Remove agents who are exiting the population (death, migration, etc)

        Args:
            aging (bool): also age the agents who remain (as `update_post()` does), in the same pass over them
        """
        auids = self.auids
        dying = self.dying()
        self._dying = None
        age = self.age.raw if aging else np.zeros(0, dtype=ss.dtypes.float) # In a compartmental run, people have no ages
        dt = age.dtype.type(self.sim.t.dt_year if aging else 0) # Added in the array's own precision, as NumPy adds a Python float
        expected = len(auids) - len(dying)
        if not len(dying) and auids is getattr(self, '_auids_view', None): # Nobody leaves (as in most steps): the agents kept, in order, are written over themselves, so no copy is made
            room = self._auids_room
        else:
            room = np.empty(expected + max(1000, expected//10), dtype=ss.dtypes.uid) # With room for births: see add_active()
        n_kept = ss.solvers.backends.passes.keep_raw(self.alive.raw[:self.uid.len_used], age, dt, aging, room, expected) # By UID, if the active agents are those alive and those who died in this step...
        if n_kept >= 0:
            gone = dying.view(np.ndarray)
        else: # ...or else from the active agents themselves
            room = np.empty(len(auids) + max(1000, len(auids)//10), dtype=ss.dtypes.uid)
            n_kept, gone = ss.solvers.backends.passes.keep_alive(auids.view(np.ndarray), self.alive.raw, age, dt, aging, room)
        if len(gone): # Agents who are still active but no longer alive

            # Remove the UIDs from the networks too
            uids = ss.uids(gone)
            for network in self.sim.networks.values():
                if not network.one_step: # Edges that end at the next step anyway are left
                    network.remove_uids(uids)

            # Keep the others, in order
            self._auids_room = room
            self.auids = self._auids_view = room[:n_kept].view(ss.uids)

        return

    def dying(self):
        """ The agents who died or left in this step (see `step_die()`), who are active until the end of it """
        dying = getattr(self, '_dying', None)
        return dying if dying is not None else ss.uids()

    def update_results(self):
        """ Record per-timestep population counts into the simulation results """
        ti = self.sim.ti
        res = self.sim.results
        states = [getattr(self, state.name) for state in self.auto_state_list] # Each auto-generated BoolState result, e.g. alive, female
        dying = self.dying()
        counts = None
        if self.scale.weighted: # Each agent counted by its weight (see AgentScale)
            auids = self.auids
            w = self.scale.raw[auids]
            gone = dying[self.ti_dead.raw[dying] <= ti], dying[self.ti_removed.raw[dying] <= ti]
            counts = [float(w[state.raw[auids]].sum()) for state in states] + [float(self.scale.raw[uids].sum()) for uids in gone]
        elif all(state.raw.ndim == 1 for state in states):
            counts = ss.solvers.backends.passes.count_alive(tuple(state.raw for state in states), self.alive.raw[:self.uid.len_used]) # In one pass by UID, if the active agents are those alive and those who died in this step
            if counts[-1] + len(dying) == len(self.auids):
                counts = [count + np.count_nonzero(state.raw[dying]) for state,count in zip(states, counts)] + [np.count_nonzero(self.ti_dead.raw[dying] <= ti), np.count_nonzero(self.ti_removed.raw[dying] <= ti)]
            else: # From the active agents themselves
                counts = ss.solvers.backends.passes.count_people(self.auids.view(np.ndarray), tuple(state.raw for state in states), self.alive.raw, self.ti_dead.raw, self.ti_removed.raw, ti)
        if self.scale.weighted: # Agents of different weights make fractions of people
            for key in [state.name for state in states] + ['new_deaths', 'new_emigrants', 'cum_deaths']:
                if res[key].values.dtype.kind in 'iu':
                    res[key].init_values(dtype=float, force=True)
        unborn = self.unborn_uids()
        if len(unborn): # Agents made at conception (by ss.Pregnancy) are not yet people, so are not counted (ss.Pregnancy counts them: pregnancy.n_unborn)
            if counts is None:
                counts = [state.count() for state in states] + list(count_gone(self.auids.view(np.ndarray), self.alive.raw, self.ti_dead.raw, self.ti_removed.raw, ti))
            w = self.scale.raw[unborn] if self.scale.weighted else np.ones(len(unborn))
            gone = ~self.alive.raw[unborn]
            counts = [count - w[state.raw[unborn] != 0].sum() for state,count in zip(states, counts)] + [counts[-2] - w[gone & (self.ti_dead.raw[unborn] <= ti)].sum(), counts[-1] - w[gone & (self.ti_removed.raw[unborn] <= ti)].sum()]
        if counts is not None:
            for state,count in zip(states, counts):
                res[state.name][ti] = count
            # Those who died or left in this step are no longer alive, and are taken out at the end of it (see finish_step()); whenever in the step it was requested
            res.new_deaths[ti], res.new_emigrants[ti] = counts[-2:]
        else:
            for state in states:
                res[state.name][ti] = state.count()
            res.new_deaths[ti], res.new_emigrants[ti] = count_gone(self.auids.view(np.ndarray), self.alive.raw, self.ti_dead.raw, self.ti_removed.raw, ti)
        res.cum_deaths[ti] = res.new_deaths[ti] if ti == 0 else res.cum_deaths[ti-1] + res.new_deaths[ti]
        return

    def unborn_uids(self):
        """ The active agents not yet born, with a negative age (e.g. made at conception by `ss.Pregnancy`), who are not counted among the people (e.g. in the result `alive`) """
        if not self.has_unborn:
            return ss.uids()
        auids = self.auids
        return auids[self.age.raw[auids] < 0]

    def finish_step(self):
        """ Remove dead agents and run post-step updates (in one pass over the agents, unless update_post() is changed) """
        sim = self.sim
        if type(self).update_post is People.update_post:
            self.remove_dead(aging=bool(sim.pars.use_aging and sim.solver == 'abm'))
        else:
            self.remove_dead()
            self.update_post()
        return

    def to_df(self):
        """ Export to dataframe; 2D states have one column per value, e.g. `sus_imm_0` (or `sus_imm_wild` if the columns are named) """
        cols = dict(uid=self.uid, slot=self.slot)
        for key,state in self.states.items():
            if state.ndim > 1:
                vals = state.values.reshape(len(state), int(np.prod(state.item_shape))).T
                names = state.columns if state.columns is not None else range(len(vals))
                cols.update({f'{key}_{name}':col for name,col in zip(names, vals)})
            else:
                cols[key] = state
        df = sc.dataframe(**cols)
        return df

    def plot(self, key=None, alive=True, bins=50, hist_kw=None, max_plots=20, figsize=(16,12), **kwargs):
        """
        Plot all the properties of the People object

        Args:
            key (str/list): the state(s) to plot
            alive (bool): if false, use all agents in the simulation instead of active (alive) ones
            bins (int): the number of bins to use in `plt.hist()`
            hist_kw (dict): passed to `plt.hist()`
            max_plots (int): the maximum number of plots to create
            figsize (tuple): passed to `plt.figure()`; by default large since the plots are crowded
            **kwargs (dict): passed to `sc.getrowscols()`, including `fig_kw`
        """
        # Handle figsize
        if 'figsize' not in kwargs:
            kwargs['figsize'] = figsize

        # Decide what to plot
        if key is None:
            key = self.states.keys()
        key = sc.tolist(key)
        if max_plots:
            key = key[:max_plots]
        n_keys = len(key)

        # Create figure and plot
        fig,axs = sc.getrowscols(n_keys, make=True, **kwargs)
        for i,k in enumerate(key):
            ax = fig.axes[i]
            state = self.states[k]
            if alive:
                data = state.values
            else:
                data = state.raw

            # It's a float array, plot a histogram
            if isinstance(state, ss.FloatArr):
                data = sc.rmnans(data)
                if not len(data):
                    print(f'Not plotting "{k}" since all values are NaN')
                ax.hist(data, bins=bins, **sc.mergedicts(hist_kw))

            # It's a boolean array, plot the proportions
            elif isinstance(state, ss.BoolArr):
                true,false = data.sum(), (~data).sum()
                labels = ['False', 'True']
                ax.bar(labels, [false,true], color=['#E0D3DE', '#3C88A3'])

            # Give up
            else:
                warnmsg = f'Cannot understand state of {type(state)} for plotting; skipping'
                ss.warn(warnmsg)

            # Tidy up
            ax.set_title(f'{state.name} (mean={data.mean():n})')
            ax.set_ylabel('Count')
            sc.commaticks(ax)
            sc.boxoff(ax)

        plt.tight_layout()
        return ss.return_fig(fig)

    def plot_ages(self, bins=None, absolute=False, fig_kw=None):
        """
        Plot the age pyramid for males and females.

        Args:
            bins (list/int): optional list or int for binning age groups (default: 5-year bins up to max age)
            absolute (bool): whether to show absolute numbers or percentage of the population
            fig_kw (dict): passed to `plt.subplots()`

        Examples:
            ```python
            sim = ss.demo(plot=False)
            sim.people.plot_ages()
            ```
        """
        # Preliminaries
        age = self.age
        female = self.female
        f_age = age[female]
        m_age = age[~female]
        max_age = age.max()

        # Define age bins
        if np.iterable(bins):
            width = None # Already defined, don't need
        elif np.isscalar(bins):
            width = bins
        else:
            width = 5
        if width is not None:
            max_bin = (np.ceil(max_age / width) + 1) * width
            bins = np.arange(0, max_bin, width)
        y_pos = np.arange(len(bins)-1)

        # Histogram counts
        f_vals, _ = np.histogram(f_age, bins=bins)
        m_vals, _ = np.histogram(m_age, bins=bins)

        # Optionally convert to percentages
        if absolute:
            xlabel = 'Number of people'
        else:
            total = f_vals.sum() + m_vals.sum()
            f_vals = f_vals / total * 100
            m_vals = m_vals / total * 100
            xlabel = 'Percentage of population'

        # Plot
        fig,ax = plt.subplots(**sc.mergedicts(fig_kw))
        ax.barh(y_pos, -f_vals, color='salmon', label='Female')  # Negative = left
        ax.barh(y_pos,  m_vals, color='skyblue', label='Male')

        # Annotate
        y_labels = [f'{bins[i]:n}–{bins[i+1]-1:n}' if i < len(bins)-1 else f'{bins[i]:n}+' for i in range(len(bins)-1)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Age group')
        ax.legend(loc='lower right')
        max_val = max(f_vals.max(), m_vals.max()) * 1.1
        ax.set_xlim([-max_val, max_val])
        ax.axvline(0, color='black', linewidth=0.8)

        # Tidy up
        sc.boxoff()
        plt.tight_layout()
        return ss.return_fig(fig)

    def person(self, ind):
        """
        Get all the properties for a single person.

        Examples:
            ```python
            sim = ss.Sim(diseases='sir', networks='random', n_agents=100).run()
            print(sim.people.person(5)) # The 5th agent in the simulation
            ```
        """
        person = Person()
        for key in ['uid', 'slot']:
            person[key] = self[key][ind]
        for key in self.states.keys():
            person[key] = self.states[key][ind]
        return person


def people(place, year=None, n_agents=None, revision=None):
    """
    A named population: the size, age structure, and sex ratio of a country or a US state

    The data are the UN World Population Prospects, bundled with `starsim_data` (for a US state, the US Census for 2023, downloaded on first use: see `ss.data.population()`). The
    sim takes its population size (`total_pop`) from them, unless it is given one, and
    the split of an age stratification that is not given one, so a compartmental model
    can use a named population as well as agents can. Where the population came from is
    shown by `sim.describe()`.

    Args:
        place (str): a country, by name or code ('Kenya', 'KE', 'KEN'), or a US state ('United States:Massachusetts')
        year (int): the year (default: the year the sim starts, if it has dates; otherwise the year of the WPP revision, e.g. 2024); years between the decadal values are interpolated
        n_agents (int): the number of agents (default: the sim's); giving it makes the model agent-based, as `ss.People(n_agents)` does
        revision (str): the WPP revision (default: the latest bundled)

    Returns:
        An `ss.People`, with `ppl.named` saying where it came from

    **Examples**:

        sim = ss.Sim(sir, people=ss.people('Kenya', year=2024), networks=ss.contacts('Kenya'))
        ss.people('NG').named.total # The population of Nigeria in 2024
    """
    ages, female, male, ref = ss.data.population(place, year=year, revision=revision)
    total = female + male
    ppl = People(n_agents=sc.ifelse(n_agents, ss.SimPars().n_agents), age_data=ss.data.to_age_data(ages, total), p_female=female.sum()/total.sum())
    ppl.named = sc.objdict(place=place, year=year, n_agents=n_agents, revision=revision, total=float(total.sum()), ages=ages, counts=total, ref=ref)
    return ppl


class Person(sc.objdict):
    """
    A simple class to hold all attributes of a person

    Examples:
        ```python
        sim = ss.Sim(diseases='sir', networks='random', n_agents=100).run()
        print(sim.people.person(5)) # The 5th agent in the simulation
        ```
    """
    def to_df(self):
        """ Convert to a dataframe """
        df = sc.dataframe.from_dict(self, orient='index', columns=['value'])
        df.index.name = 'key'
        return df


@nb.jit(cache=True)
def count_gone(auids, alive, ti_dead, ti_removed, ti):
    """ The number of active agents who are no longer alive, because they died or left by step ti, in one pass """
    deaths = 0
    removed = 0
    for k in range(auids.shape[0]):
        uid = auids[k]
        if not alive[uid]:
            deaths += ti_dead[uid] <= ti
            removed += ti_removed[uid] <= ti
    return deaths, removed
