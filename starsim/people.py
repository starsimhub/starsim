"""
The people in a sim (`ss.People`), lazy references to their properties (`ss.person`), and how they are made

People hold what every module shares, such as age, sex, and who is alive. In a compartmental run, each row is a group of people rather than an agent.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from pathlib import Path
plt = sc.lazyimport('matplotlib.pyplot') # Imported when first used, since most runs never plot


class LazyPeople:
    """
    The people's properties, before there are any people: `ss.person.<name>` is a lazy reference to the property `<name>` of each person (e.g. `ss.person.age`, `ss.person.urban`), checked when the sim is built

    It can be used in a condition (`ss.person.age < 15`, for `who=`) and, as the person's value,
    in a rate, a branch probability, or a result (`expit(b0 + b1*ss.person.urban)`); a condition in a
    rate is 1 for the people it holds for and 0 for the others (`0.3*(ss.person.age > 65)`).

    Examples:
        ```python
        p_use = expit(b0 + b_age*ss.person.age + b_urban*ss.person.urban)
        kids = ss.person.age < 15
        ```
    """
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return ss.conditions.Cond('people', name)

    def get(self, name, default):
        """
        `ss.person.<name>`, or `default` for everyone if nothing adds the property `<name>` to the people, e.g. one that a
        module adds (`people.add_state()`), read by a rate whether or not that module is in the sim

        Args:
            name (str): the name of the property
            default (float): the value for everyone if nothing adds the property

        Example:
            ```python
            onset = S >> D @ ss.peryear(0.01)*(1 + 0.5*ss.person.get('unstable_housing', 0))
            ```
        """
        return ss.conditions.Cond('people', name, default)

    def __repr__(self):
        return 'ss.person'

person = LazyPeople()


class AgentScale(ss.FloatArr):
    """ The number of people each agent stands for relative to the others (`people.scale`); setting it other than 1 makes results, and the counts in a declared model's rates (`I.n`, `I.frac`, and the share infectious in a transmission), add up the agents' weights """
    weighted = False # Set once any agent's scale is set other than 1, so that results are unweighted until then

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
    The people in a sim: their ages, sexes, who is alive, and the states that modules add to them; also how agents are born, die, and are counted

    The sim usually makes this itself, from `n_agents=` (and `total_pop=`, the people they stand for). Give one
    yourself to give agents ages from data of your own (`ss.People(n_agents, age_data=...)`) or extra states; for
    the population of a place, use `ss.population('Kenya')`. In a compartmental run, each row is a group of people
    rather than an agent.

    Args:
        n_agents (int): the initial number of agents in the model
        age_data (dataframe/series/array/str/`ss.Dist`): the age distribution: a dataframe with columns `age` (the lower edge of each age bin) and `value` (the number or share of people in it), a series of values indexed by age, a two-column array, a CSV file with the dataframe's columns, or a distribution of ages (default: the world's ages). This gives agents ages; for age groups in a compartmental model, use `disease.stratify(age={0: 0.3, 15: 0.6, 65: 0.1})` instead
        extra_states (list): more properties of each person, e.g. `[ss.FloatArr('income')]` (modules can also add them, with `people.add_state()`)
        p_female (float): the share of agents who are female
        mock (bool): if True, link the people to a mock sim, so they can be used without one (for testing and debugging)

    Examples:
        ```python
        ppl = ss.People(2000)
        kids = ss.person.age < 15 # A lazy condition, for a sim run as agents or as compartments (see ss.person)
        ```
    """
    _taken = None # See taken_names()
    auids  = MadeOnUse() # The four fundamental arrays, made on first use: see make_uids()
    uid    = MadeOnUse()
    slot   = MadeOnUse()
    parent = MadeOnUse()

    def __init__(self, n_agents, age_data=None, extra_states=None, p_female=0.5, mock=False):
        # We internally store states in a dict keyed by the memory ID of the state, so that we can have colliding names
        # e.g., across modules, but we will never change the size of a state multiple times in the same iteration over
        # _states. This is a hidden variable because it is internally used to synchronize the size of all states contained
        # within the sim, regardless of where they are. In contrast, `People.states` offers a more user-friendly way to access
        # a selection of the states e.g., module states could be added in there, while intervention states might not.
        self._states = {}
        self.version = ss.__version__
        self.untraited = [] # Ranges of UIDs made by code that sets their properties afterwards (grow() without age=), whose traits are drawn at the next step: see grow()
        self._age_bins = {} # The bin of each agent by age, for each set of edges: see age_bins()
        self.has_unborn = False # Whether a module makes agents before they are born, with a negative age (e.g. ss.Pregnancy's fetuses), who are not counted among the people: see unborn_uids()
        self.initialized = False
        self.n_agents_init = round(n_agents) # The nearest whole number, as ss.SimPars takes it (e.g. a number a fit chose)
        self.named = None # For a named population (ss.population('Kenya')), where it came from: see ss.population()

        # The four fundamental arrays (auids, uid, slot, and parent) are made on first use: see make_uids()

        states = [
            ss.BoolState('alive', default=True),  # Whether the agent is alive
            ss.FloatArr('age', default=self.get_age_dist(age_data)),  # NaN until conceived
            ss.BoolState('female', default=ss.bernoulli(name='female', p=p_female)),
            ss.TimeArr('ti_dead'),  # The step of death
            ss.TimeArr('ti_removed'),  # The step of leaving for another reason (e.g. emigration)
            AgentScale('scale', default=1.0, shared=True), # The scale factor for the agents (multiplied for making results); no memory until one is set
            ss.Arr('cause_of_death', dtype=np.int16, default=-1, nan=-1, label='Cause of death', shared=True), # Its position in people.causes (-1 for none); no memory until someone dies
        ] + sc.tolist(extra_states)
        self.states = ss.ndict(type=ss.Arr)
        for state in states:
            self.states.append(state)
            setattr(self, state.name, state)
            state.link_people(self)
        self.causes = [] # The causes of death, by name (see request_death()), in the order first seen: people.cause_of_death is the position in this list
        self._claims = [] # The deaths requested in this step, with their causes: see step_die()
        self._died_of = {} # Who died of each cause in this step
        self._knock_on = None # The deaths requested by modules' on_death(), while step_die() calls it, so they happen in the same step
        self._warned_late = set() # The causes of the deaths requested too late in a step, warned about once each: see request_death()
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
        self.slot   = ss.arrays.IndexArr('slot', shared=self.uid) # A slot is a special state managed internally to generate random numbers
        self.parent = ss.arrays.IndexArr('parent', label='UID of parent', shared=True)  # UID of parent, if any (used with pregnancy)
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
        Register a state with the people, so that it grows when agents are added; called by `state.link_people()`

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
        ss.v3_deprecations.check_attr(ss.People, attr)
        missing = self.__dict__.get('_no_groups', [])
        if attr == 'auids' and '_no_groups' in self.__dict__:
            raise ss.errors.NoAgentAttribute('people.auids was asked for, but this run has no agents: it has one row for each group of people.\n'
                                             '     Use the number of people alive (people.n) or the number in a disease\'s states (e.g. sir.I.n), or run with solver="abm".')
        if attr in missing:
            raise ss.errors.NoAgentAttribute(f'people.{attr} was asked for, but it is a per-agent attribute and not a stratification, so it does not exist in a compartmental run.\n'
                                             f'     Stratify by it (e.g. sim.stratify({attr}=[...])), or run with solver="abm".')
        if '_no_groups' in self.__dict__ and not attr.startswith('_'): # A compartmental run, whose people lack what networks and modules written as code add to agents
            raise AttributeError(f'People object has no attribute "{attr}".\n     In a compartmental run, people are one row per group, without the attributes that networks and modules written as code give agents; if it is one of those, run with solver="abm"')
        raise AttributeError(f'People object has no attribute "{attr}"')

    def __getitem__(self, key):
        """ `people['age']` for an attribute, or `people[14]` for one person (see `get_person()`) """
        if isinstance(key, (int, np.integer)):
            return self.get_person(key)
        elif isinstance(key, str):
            return getattr(self, key)
        errormsg = f"Cannot handle {key} ({type(key)}: must be str to access an attribute, e.g. people['age'], or int to access a person, e.g. people[14]"
        raise TypeError(errormsg)

    def __setitem__(self, key, value):
        """ `people['attr'] = value`, as `setattr()` """
        return setattr(self, key, value)

    def __iter__(self):
        """ Iterate over (alive) people """
        for uid in self.auids:
            yield self.get_person(uid)

    def __contains__(self, key):
        """ Whether people have a property by this name, e.g. `'patch' in sim.people` (which would otherwise go through every person) """
        if isinstance(key, str):
            return key in self.keys()
        return any(person is key for person in self)

    def __setstate__(self, state):
        """
        Rebuild the registry of states when unpickled or copied, since it is keyed by the ids of the states, which a copy changes
        """
        state['_states'] = {id(v):v for v in state['_states'].values()}
        self.__dict__ = state
        return

    def __bool__(self):
        """ People with no agents are still true (otherwise `len()` would make them false) """
        return True

    def __len__(self):
        """
        The number of rows: the active agents, or in a compartmental run, the groups of people. Agents who die in a step stay
        among the active agents until the step ends, so that analyzers can still look at them; the number of people alive is `people.n`
        """
        return self.n_agents_init if '_no_groups' in self.__dict__ else len(self.auids)

    @property
    def n(self):
        """
        The number of people alive now, under every solver, as `sir.I.n` counts the people in a state (the same as `people.alive.n`):
        as agents, each agent counted by the people it stands for (its weight and the sim's scale), not counting those not yet born
        (made at conception by `ss.Pregnancy`); in a compartmental run, the people in the compartments. `people.alive.count()` counts agents.
        """
        return self.alive.n

    def _n_agents_alive(self):
        """ The number of agents alive, not counting those not yet born: `results.alive` before it is scaled (with agents all of the same weight) """
        unborn = self.unborn_uids()
        return np.count_nonzero(self.alive) - np.count_nonzero(self.alive.raw[unborn])

    def __repr__(self):
        return self.brief(output=True)

    def __str__(self):
        return self.brief(output=True)[:-1] + f'; states=[{sc.strjoin(self.states.keys())}])' # Inside the closing parenthesis

    def disp(self):
        """ Print every attribute of the people """
        return sc.pr(self)

    @property
    def grouped(self):
        """ Whether these are the groups of a compartmental run (one row per group), not agents """
        sim = getattr(self, 'sim', None)
        return sim is not None and getattr(sim, 'solver', None) is not None and not isinstance(sim.solver, ss.solvers.abm) and bool(getattr(sim, 'declared', None))

    def brief(self, output=False):
        """ Print a one-line summary of the people

        Args:
            output (bool): if True, return the text rather than printing it
        """
        n = self.n_agents_init
        if self.grouped and self.initialized: # A compartmental run: the rows are groups of people
            age = '' if 'age' in self.__dict__.get('_no_groups', []) else f'; age={self.age.mean():0.1f}±{self.age.std():0.1f} over the groups' # Not when unstratified by age
            out = f'People({n:n} group{"s"*(n != 1)}; {self.n:n} people{age})'
            return out if output else print(out)
        if not self.initialized: # Age and other states hold uninitialized memory until init_vals() runs
            out = f'People({n=:n}; uninitialized)'
        else:
            alive = len(self)
            alive_str = f'{alive=:n}; ' if alive != n else ''
            out = f'People({n=:n}; {alive_str}age={self.age.mean():0.1f}±{self.age.std():0.1f})'
        return out if output else print(out)

    def keys(self):
        """ The names of the people's states, including those of modules (e.g. 'sir.S') """
        return self.states.keys()

    @staticmethod
    def get_age_dist(age_data):
        """
        The distribution that new agents' ages are drawn from, made from age data

        The data are a two-column array, a pandas series with age as the index and the
        number (or share) of people as the value, a dataframe with "age" and "value"
        columns, or the name of a CSV file with those columns, e.g.:

            age      value
            0      220.548
            1      206.188
            2      195.792
            3      187.442

        The ages are the lower edges of the bins, so ages within a bin are spread evenly.
        The last bin is as wide as the one before it; to set its upper edge, add a row with
        that age and a value of 0. Counts and shares give the same distribution.

        Args:
            age_data (array/series/dataframe/str/`ss.Dist`): the data, as above, or a distribution of ages, used as it is (default: the world's ages, `ss.data.age('world')`, as for a sim with no named population)

        Returns:
            An `ss.Dist` of ages
        """
        if age_data is None:
            age_data = ss.data.age('world')
        if isinstance(age_data, ss.Dist):
            return age_data
        if isinstance(age_data, (str, Path)):
            age_data = pd.read_csv(age_data)
        if isinstance(age_data, np.ndarray):
            age_bins, age_props = age_data[:,0], age_data[:,1]
        elif isinstance(age_data, pd.Series):
            age_bins, age_props = age_data.index, age_data.values
        elif isinstance(age_data, pd.DataFrame):
            age_bins, age_props = age_data['age'].values, age_data['value'].values
        else:
            errormsg = f'age_data must be a file, an array, a Series, a dataframe, or an ss.Dist, not {type(age_data)}'
            raise TypeError(errormsg)
        return ss.histogram(values=age_props, bins=age_bins, name='Age distribution')

    def link_sim(self, sim, init=False):
        """ Link the people to the sim, and their distributions (e.g. of ages) to its random numbers; called by `sim.init()` """
        if self.initialized:
            errormsg = 'Cannot re-initialize a People object directly; use sim.new() to make the sim again'
            raise RuntimeError(errormsg)
        self.sim = sim
        for arr in [self.ti_dead, self.ti_removed]:
            arr.t = sim.t
        ss.dists.link_dists(obj=self.states, sim=sim, module=self, skip=[ss.Sim, ss.Module], init=init)
        return

    @property
    def auto_state_list(self):
        """
        The people's own `ss.BoolState` states (e.g. alive, female), each of which makes a result of the same name, the number of people in it

        Not those of modules (named like "sir.S"), which make their own results; for all states, see `People.states`.
        """
        return [state for key, state in self.states.items()
                if isinstance(state, ss.BoolState) and '.' not in key]

    def init_results(self):
        """
        Make the results of the people; called by `Sim.init_results()`

        Each of the people's own `ss.BoolState` states makes a result of the number in it (e.g. `alive`, the
        number alive); for a state that makes no result, use `ss.BoolArr` instead. There are also the flows of
        people who died and who left (`died` and `emigrated`; `results.died.cum` is the cumulative total).

        With `sim.pars.record_people=False`, there are none of these (and `record_stocks()` and
        `record_flows()` are skipped in the loop), which saves a little time and memory.
        """
        if not self.sim.pars.get('record_people', True):
            return
        kw = dict(module='People', shape=self.sim.t.npts, timevec=self.sim.t.timevec, dtype=int, scale=True)
        results = [ss.Result(state.name, label=state.label, kind='stock', **kw) for state in self.auto_state_list]
        results += [
            ss.Result('died', label='Deaths', kind='flow', **kw),
            ss.Result('emigrated', label='Emigrants', kind='flow', **kw),
        ]
        self.sim.results += results
        return

    @classmethod
    def taken_names(cls):
        """ The names of the attributes every People has (e.g. "age", "version"), which a module can't take from its variable """
        if cls._taken is None:
            cls._taken = set(dir(cls)) | set(vars(cls(n_agents=1)))
        return cls._taken

    def add_module(self, module):
        """
        Add a module to the people: its states are registered, so that they grow with the agents, and the module is found in two ways

        - `People.<module>` will be a reference to the module itself, enabling access via `people.sir.S`
        - `People.states['<module>.<state>']` will be a reference to the state, enabling access via `people.states['sir.S']`

        Args:
            module (Module): the module to add
        """
        if hasattr(self, module.name):
            existing = getattr(self, module.name)
            what = 'a module of that name is already in the sim' if isinstance(existing, ss.Module) else f'people already have a property called "{module.name}"'
            raise AttributeError(f'The {type(module).__name__} "{module.name}" cannot be added to the people, since {what}; give it another name, e.g. name="{module.name}_mod"')
        setattr(self, module.name, module)
        for state in module.state_list:
            state.link_people(self)
            if isinstance(state, ss.TimeArr) and state.t is None: # Times as steps of the module's timeline (see due())
                state.t = module.t
            self.states[f'{module.name}.{state.name}'] = state # By the combined name here; in the module, by its own name
        return

    def add_state(self, state):
        """
        Add a property of each person, from a module (e.g. a social determinant of health that several modules read, as `ss.person.<name>`)

        Idempotent: if the people have a property of that name already (added by another module, say), it is
        returned as it is, so each module that reads it can add it. Call it before the people's values are made,
        e.g. in a module's `init_pre()`; added later, its values are made at once.

        Args:
            state (Arr): the array, e.g. `ss.BoolArr('housing', default=False)` or `ss.FloatArr('adherence', default=ss.uniform())`

        Returns:
            The property's array (the one given, or the one added before)

        Example:
            ```python
            def init_pre(self, sim):
                super().init_pre(sim)
                self.housing = sim.people.add_state(ss.BoolArr('housing', default=ss.bernoulli(0.8)))
            ```
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
            dists = ss.dists.link_dists(obj=state, sim=sim, module=self, skip=[ss.Sim, ss.Module, People]) # Its own (not those of the people it links to)
            if self.initialized: # Initialized with the others otherwise (see ss.Sim.init())
                for trace,dist in dists.items():
                    dist.init(trace=f'people_{name}_{trace}', seed=sim.dists.base_seed, sim=sim)
        if self.initialized:
            state.init_vals()
        return state

    def init_vals(self):
        """ Give each state its initial values (e.g. agents' ages), the final step of initialization """
        for state in self.states():
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def init_mock(self):
        """ Link the people to a mock sim and give them their values, so they can be used without a sim (for testing and debugging) """
        sim = ss.debugtools.mock_sim(n_agents=self.n_agents)
        self.link_sim(sim, init=True)
        self.init_vals()
        return

    @property
    def n_agents(self):
        """ The number of active agents, as `len(people)` (for the number of people alive, see `people.n`) """
        return len(self)

    @property
    def n_uids(self):
        """ The number of agents ever made, alive or not (each has its own UID) """
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
        dead.people = sc.obj(auids=ss.uids(np.arange(n))) # Cover all agents, not just active ones
        return dead

    @property
    def male(self):
        """ Whether each agent is male (`~people.female`; female is a state) """
        return ~self.female

    def grow(self, n=None, new_slots=None, placed=False, parent=None, age=None):
        """
        Add agents, e.g. newborns or immigrants, and return their UIDs

        Args:
            n (int): the number of agents to add (default: one for each of `new_slots`)
            new_slots (array): the slots of the new agents, which say which random numbers each draws (default: their UIDs)
            placed (bool): whether the caller gives them their levels of the sim's stratifications; if not, they are drawn from each one's split
            parent (array): the parent of each (e.g. for newborns), set before their levels are drawn, so they can be born where their parent is
            age (array): their age (e.g. 0 for newborns), likewise
        """
        if n is None:
            if new_slots is None:
                errormsg = 'Must supply either n or new_slots'
                raise ValueError(errormsg)
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

        self.parent.grow(new_uids) # None, until set
        for state in self._states.values():
            state.grow(new_uids)
        if parent is not None:
            self.parent[new_uids] = parent
        if age is not None:
            self.age[new_uids] = age
        self.add_active(new_uids)
        sim = getattr(self, 'sim', None)
        if isinstance(getattr(sim, 'solver', None), ss.solvers.abm): # A dimension called sex is people.female
            sim.groups.set_sex(new_uids)
            if not placed and getattr(sim.groups, 'vals_set', False) and sim.groups.strats: # E.g. newborns from ss.Births (or ss.Pregnancy's before the start), who would otherwise all be at the first level
                sim.groups.place_new(new_uids)
            if not placed and age is None: # Made by code that sets their age (and the rest) itself afterwards, so their traits are drawn at the next step, from what it set (rather than, say, a fetus's trait by age from a default adult age)
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
        The number of people some agents stand for, by their weights (`people.scale`), e.g. for a result that counts people rather than agents

        Args:
            inds (array): the agents to count
        """
        return self.scale[inds].sum()

    def update_post(self):
        """ Updates at the end of each step: agents get older (see `finish_step()`) """
        sim = self.sim
        if sim.pars.aging and isinstance(sim.solver, ss.solvers.abm): # In a compartmental run, each row is a group, whose age stays the middle of the group (people move between groups as they age)
            self.age[self.alive.uids] += sim.t.dt_year
        return

    def request_death(self, uids, cause=None, hazard=None):
        """
        Request that agents die at this step; how a module kills agents

        Use this rather than setting `People.ti_dead` directly, so that modules don't overwrite each
        other's deaths (e.g. a death set for the future overwriting one due now from another module).
        A module with a future time of death (e.g. after the course of a disease) keeps it itself, and
        calls this when the time comes, with itself as the cause.

        The deaths happen after every module has stepped, but before the analyzers, so that analyzers
        can see and record the agents who died in this step.

        Each person dies once, of one cause: if several modules request the death of the same person
        in the same step, one of them is chosen at random (whatever the order of the modules, with
        common random numbers): in proportion to the hazards of the claims if each gave one (`hazard=`),
        and otherwise each equally likely. The cause is kept in `people.cause_of_death`
        (see `people.causes`). A module counts its deaths after they are resolved, with
        `people.died_of(module)`, so a death is counted once.

        Args:
            uids (array): the agents who die
            cause (Module/str): what they die of: a module (by its name) or a name (default: none)
            hazard (float/array): the hazard of this death, one number or one per agent (in any unit, the same for every module, e.g. per year), to weight the choice of cause among the claims on the same person in the step (e.g. `dis.death.hazard(uids)` for a declared transition)
        """
        self.ti_dead[uids] = self.sim.ti
        if len(uids):
            name = cause.name if isinstance(cause, ss.Module) else cause
            if getattr(self, '_knock_on', None) is not None: # During step_die(), from a module's on_death(), so it happens in this step
                self._knock_on.append(ss.uids(uids))
            elif getattr(self, '_resolved_ti', None) == self.sim.ti and name not in self._warned_late: # E.g. in finish_step()
                self._warned_late.add(name)
                ss.warn(f"A death (cause: {name}) was requested after this step's deaths were resolved (by people.step_die(), which comes after every module's step()), so it happens at the next step; request it in step() for it to happen in this one")
            if name not in self.causes:
                self.causes.append(name)
            hazard = np.full(len(uids), np.nan) if hazard is None else np.broadcast_to(np.asarray(hazard, dtype=float), len(uids))
            self._claims.append((ss.uids(uids), self.causes.index(name), hazard))
        return

    def died_of(self, cause):
        """ The agents who died in this step of a cause (a module or a name), once each death is resolved to one cause (see `request_death()`)

        Args:
            cause (Module/str): the module, or its name
        """
        name = cause.name if isinstance(cause, ss.Module) else cause
        return self._died_of.get(name, ss.uids())

    def resolve_causes(self, death_uids):
        """ Give each agent who dies in this step one cause, from the deaths requested (see `request_death()`); called by `step_die()` """
        claims, self._claims = self._claims, []
        if not claims:
            return
        uids = np.concatenate([u for u,c,h in claims]).astype(np.int64)
        codes = np.concatenate([np.full(len(u), c) for u,c,h in claims])
        hazards = np.concatenate([h for u,c,h in claims])
        keep = np.isin(uids, death_uids, kind='sort') # E.g. not someone who left instead; by default, a table the size of the range of UIDs
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
        if several.any(): # One of the causes: in proportion to their hazards if every claim on the person gave one, else each equally likely
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
            name = self.causes[code]
            these = people[codes == code]
            self._died_of[name] = self._died_of[name].concatenate(these) if name in self._died_of else these # Added to, for the deaths that on_death() requests (see step_die())
        return

    def request_removal(self, uids):
        """
        Request that agents leave the sim at this step for a reason other than death (e.g. emigration), so they are counted as having left (`results.emigrated`) rather than as having died

        Args:
            uids (array): the agents to remove
        """
        self.ti_removed[uids] = self.sim.ti
        return

    def step_die(self):
        """ Carry out the deaths and removals of this step (those requested in it, or set for it before), and return who died or left """
        ti = self.sim.ti
        deaths, n_alive = ss.solvers.backends.passes.due_alive(self.alive.raw[:self.uid.len_used], self.ti_dead.raw, self.ti_removed.raw, ti) # As ((self.ti_dead <= ti) | (self.ti_removed <= ti)).uids, in one pass by UID, since the active agents are those alive...
        if n_alive != len(self.auids): # ...unless something else has changed who is alive: then from the active agents themselves
            deaths = ss.solvers.backends.passes.due(self.auids.view(np.ndarray), self.ti_dead.raw, self.ti_removed.raw, ti)
        death_uids = ss.uids(deaths)
        dying = death_uids
        self._died_of = {}

        # Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed
        # before analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep.
        # A death that a module's on_death() requests (e.g. of the fetuses of a mother who dies) happens in this step too
        while len(dying):
            self.resolve_causes(dying[self.ti_dead.raw[dying] <= ti])
            self.alive[dying] = False  # While not dead, removed agents should not be included in alive totals
            self._dying = death_uids # Still active until the end of the step: see record_flows() and remove_dead()
            self._knock_on = [] # Filled by request_death() during on_death()
            try:
                for mod in self.death_modules():
                    mod.on_death(dying)
                knock_on = self._knock_on
            finally:
                self._knock_on = None
            dying = ss.uids.concatenate(knock_on) if knock_on else ss.uids()
            dying = ss.uids(np.unique(dying[self.alive.raw[dying]]))
            if len(dying):
                death_uids = death_uids.concatenate(dying)

        self._resolved_ti = ti if self.sim.initialized else None # A death requested later in this step happens at the next one: see request_death() (not during the initialization, whose step 0 is the first step's too)
        return death_uids

    def death_modules(self):
        """ The modules whose `on_death()` is called at a death (see `ss.Module.on_death()`): the diseases, then any other declared module, then any other module that defines it """
        mods = getattr(self, '_death_mods', None)
        if mods is None:
            sim = self.sim
            mods = [disease for disease in sim.diseases() if isinstance(disease, ss.Disease)]
            mods += [mod for mod in getattr(sim, 'declared', []) if all(mod is not m for m in mods)] # A declared disease given as another kind of module: its agents leave its states as well
            mods += [mod for mod in sim.modules if type(mod).on_death is not ss.Module.on_death and all(mod is not m for m in mods)]
            if sim.initialized: # Kept once the modules are all there
                self._death_mods = mods
        return mods

    def born(self, children):
        """ Tell each module that defines `on_birth()` about these newborns (see `ss.Module.on_birth()`); called wherever children are born """
        if len(children):
            parents = None
            for mod in self.sim.modules:
                if type(mod).on_birth is not ss.Module.on_birth:
                    if parents is None:
                        parents = ss.uids(self.parent.raw[children])
                    mod.on_birth(parents, children)
        return

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

    def age_bins(self, edges):
        """ The bin of each agent's age, as np.digitize(age, edges), by UID: everyone gets older at each step, but few leave their bin, so only those who did are worked out again """
        age = self.age.raw
        key = edges.tobytes()
        bins = self._age_bins.get(key)
        if bins is None or len(bins) > len(age):
            bins = np.digitize(age, edges)
        else:
            if len(bins) < len(age): # The arrays have grown
                bins = np.concatenate([bins, np.digitize(age[len(bins):], edges)])
            moved = ss.solvers.backends.passes.crossed(age, bins, np.append(-np.inf, edges), np.append(edges, np.inf))
            bins[moved] = np.digitize(age[moved], edges)
        self._age_bins[key] = bins
        return bins

    def dying(self):
        """ The agents who died or left in this step (see `step_die()`), who are active until the end of it """
        dying = getattr(self, '_dying', None)
        return dying if dying is not None else ss.uids()

    def record_stocks(self):
        """ Record the population counts at the start of the step (each `ss.BoolState`, e.g. alive, female), at its index (see `ss.loop.Loop`) """
        ti = self.sim.ti
        res = self.sim.results
        states = [getattr(self, state.name) for state in self.auto_state_list] # Each auto-generated BoolState result, e.g. alive, female
        counts = None
        if self.scale.weighted: # Each agent counted by its weight (see AgentScale)
            w = self.scale.raw[self.auids]
            counts = [float(w[state.raw[self.auids]].sum()) for state in states]
            for state in states: # Agents of different weights make fractions of people
                if res[state.name].values.dtype.kind in 'iu':
                    res[state.name]._init_values(dtype=float, force=True)
        elif all(state.raw.ndim == 1 for state in states):
            counts = ss.solvers.backends.passes.count_alive(tuple(state.raw for state in states), self.alive.raw[:self.uid.len_used]) # In one pass by UID, if the active agents are those alive (as they are at the start of a step, the dead having been taken out at the end of the last)
            counts = counts[:-1] if counts[-1] == len(self.auids) else [state.count() for state in states]
        unborn = self.unborn_uids()
        if len(unborn): # Agents made at conception (by ss.Pregnancy) are not yet people, so are not counted (ss.Pregnancy counts them: pregnancy.unborn)
            if counts is None:
                counts = [state.count() for state in states]
            w = self.scale.raw[unborn] if self.scale.weighted else np.ones(len(unborn))
            counts = [count - w[state.raw[unborn] != 0].sum() for state,count in zip(states, counts)]
        for i,state in enumerate(states):
            res[state.name][ti] = counts[i] if counts is not None else state.count()
        return

    def record_flows(self):
        """ Record the people who died or left in the step, at its index; they are taken out at the end of it (see `finish_step()`), whenever in the step it was requested """
        ti = self.sim.ti
        res = self.sim.results
        dying = self.dying() # Those who died or left in the step, who are all the active agents not alive
        if self.scale.weighted:
            gone = [float(self.scale.raw[dying[arr.raw[dying] <= ti]].sum()) for arr in [self.ti_dead, self.ti_removed]]
            for key in ['died', 'emigrated']:
                if res[key].values.dtype.kind in 'iu':
                    res[key]._init_values(dtype=float, force=True)
        else:
            gone = [np.count_nonzero(arr.raw[dying] <= ti) for arr in [self.ti_dead, self.ti_removed]]
        unborn = self.unborn_uids()
        if len(unborn):
            w = self.scale.raw[unborn] if self.scale.weighted else np.ones(len(unborn))
            out = ~self.alive.raw[unborn]
            gone = [gone[0] - w[out & (self.ti_dead.raw[unborn] <= ti)].sum(), gone[1] - w[out & (self.ti_removed.raw[unborn] <= ti)].sum()]
        res.died[ti], res.emigrated[ti] = gone
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
            self.remove_dead(aging=bool(sim.pars.aging and isinstance(sim.solver, ss.solvers.abm)))
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
        Plot the distribution of each of the people's states: a histogram for numbers, and a bar chart for true/false

        Args:
            key (str/list): the state(s) to plot (default: all, up to `max_plots`)
            alive (bool): if False, include every agent ever made, not only those active
            bins (int): the number of bins to use in `plt.hist()`
            hist_kw (dict): passed to `plt.hist()`
            max_plots (int): the maximum number of plots to create
            figsize (tuple): passed to `plt.figure()`; by default large since the plots are crowded
            **kwargs (dict): passed to `sc.getrowscols()`, including `fig_kw`
        """
        kwargs.setdefault('figsize', figsize)
        key = sc.tolist(self.states.keys() if key is None else key)
        if max_plots:
            key = key[:max_plots]

        fig,axs = sc.getrowscols(len(key), make=True, **kwargs)
        for i,k in enumerate(key):
            ax = fig.axes[i]
            state = self.states[k]
            data = state.values if alive else state.raw
            if isinstance(state, ss.FloatArr):
                data = sc.rmnans(data)
                if not len(data):
                    print(f'Not plotting "{k}" since all values are NaN')
                ax.hist(data, bins=bins, **sc.mergedicts(hist_kw))
            elif isinstance(state, ss.BoolArr):
                ax.bar(['False', 'True'], [(~data).sum(), data.sum()], color=['#E0D3DE', '#3C88A3'])
            else:
                warnmsg = f'Cannot understand state of {type(state)} for plotting; skipping'
                ss.warn(warnmsg)
            ax.set_title(f'{state.name} (mean={data.mean():n})')
            ax.set_ylabel('Count')
            sc.commaticks(ax)
            sc.boxoff(ax)

        plt.tight_layout()
        return ss.return_fig(fig)

    def plot_ages(self, bins=None, absolute=False, fig_kw=None):
        """
        Plot the age pyramid: the number or share of females and males in each age group

        Args:
            bins (list/int): the edges of the age groups, or the width of each (default: 5-year groups up to the oldest age)
            absolute (bool): whether to show absolute numbers or percentage of the population
            fig_kw (dict): passed to `plt.subplots()`

        Examples:
            ```python
            sim = ss.demo(plot=False)
            sim.people.plot_ages()
            ```
        """
        age = self.age
        female = self.female
        if not np.iterable(bins): # A width, by default 5 years
            width = bins if np.isscalar(bins) else 5
            bins = np.arange(0, (np.ceil(age.max()/width) + 1)*width, width)
        y_pos = np.arange(len(bins)-1)
        f_vals, _ = np.histogram(age[female], bins=bins)
        m_vals, _ = np.histogram(age[~female], bins=bins)

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
        y_labels = [f'{bins[i]:n}–{bins[i+1]-1:n}' for i in range(len(bins)-1)]
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

    def get_person(self, ind):
        """
        All the properties of one agent, as an `ss.people.Person`

        Args:
            ind (int): the UID of the agent

        Examples:
            ```python
            sim = ss.Sim(diseases='sir', networks='random', n_agents=100).run()
            print(sim.people.get_person(5)) # The agent with UID 5
            ```
        """
        person = Person()
        for key in ['uid', 'slot']:
            person[key] = self[key][ind]
        for key in self.states.keys():
            person[key] = self.states[key][ind]
        return person


def population(place, year=None, n_agents=None, revision=None):
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

    Examples:
        ```python
        sim = ss.Sim(sir, people=ss.population('Kenya', year=2024), networks=ss.contacts('Kenya'))
        ss.population('NG').named.total # The population of Nigeria in 2024
        ```
    """
    pop = ss.data.population(place, year=year, revision=revision)
    ages = np.array(ss.data.age_edges(pop.index))
    total = (pop.female + pop.male).values
    ppl = People(n_agents=sc.ifelse(n_agents, ss.SimPars().n_agents), age_data=ss.data.to_age_data(ages, total), p_female=pop.female.sum()/total.sum())
    ppl.named = sc.obj(place=place, year=year, n_agents=n_agents, revision=revision, total=float(total.sum()), ages=ages, counts=total, ref=pop.attrs['ref'])
    return ppl


class Person(sc.obj):
    """
    All the properties of one agent (see `People.get_person()`), as a dict that can be turned into a dataframe

    Examples:
        ```python
        sim = ss.Sim(diseases='sir', networks='random', n_agents=100).run()
        print(sim.people.get_person(5)) # The agent with UID 5
        ```
    """
    def to_df(self):
        """ The properties as a dataframe, one row each """
        df = sc.dataframe.from_dict(self, orient='index', columns=['value'])
        df.index.name = 'key'
        return df
