"""
Base classes for diseases
"""
import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

ss_int = ss.dtypes.int
ss_float = ss.dtypes.float
_ = None # For function signatures



class Disease(ss.Module):
    """
    A disease declared from its states and the transitions between them

    No methods need to be defined, and the disease can be run with any solver
    (see `ss.Sim`):

        sir = ss.Disease(
            infection = ss.transmission('S -> I', beta=0.3),
            recovery  = ss.progression('I -> R', dur=10),
            init_prev = 0.01,
        )

    The same disease can be declared in a class body (the class form), which can also
    add behavior by defining methods; `ss.SIR`, `ss.SIS`, and `ss.SEIR` are written
    this way. A disease written as code instead, with `define_states()` and `step()`
    and per-agent arrays, subclasses `ss.CodedDisease` (Starsim 3's `ss.Disease`), or `ss.Infection`
    if it is transmitted between agents.

    Args:
        args (list): transitions and outcomes, named from the variables they are assigned to
        pars (dict): parameter values by name
        transitions (list): transitions, as a list (for longer models)
        results (list): extra results: sums and ratios of states
        outcomes (list): `ss.outcome` objects
        init_prev (float): the initial prevalence, seeded through the transmission
        init (dict): the initial fraction or count in each state, e.g. `dict(S=0.90, I=0.01, R=0.09)`
        kwargs (dict): one keyword per state, transition, parameter, result, or function; also `name`, `label`, and `dt`

    The results are the number in each state (`n_I`), the flow along each transition by its
    name (`infection`, with its cumulative total as `res.infection.cum`), and the results and
    outcomes declared. The library's `ss.SIR`, `ss.SIS`, and `ss.SEIR` also have Starsim's
    standard `new_infections`, `cum_infections`, and `prevalence` (`standard_results = True`).

    Examples:
        ```python
        # Operator form: declare the states, then the transitions
        S, I, R   = ss.states(3)
        infection = S >> I @ 0.3 * I
        recovery  = I >> R @ ss.days(10)
        sir = ss.Disease(infection, recovery, init_prev=0.01)
        ss.Sim(sir, stop=100).run().plot()

        # Class form: for when there is behavior that is not a transition
        class SIR(ss.Disease):
            S, I, R   = ss.states(3)
            infection = S >> I @ 0.3 * I
            recovery  = I >> R @ ss.days(10)
        ```
    """
    graph = None # The declared states and transitions; defined on the class so it always exists
    standard_results = False # Whether to add new_infections, cum_infections, and prevalence (for a disease with a transmission), as the library's diseases do

    def __init__(self, *args, pars=None, **kwargs):
        super().__init__()
        self.infection_log = None
        self.engine = None # Runs the graph; created when the sim is initialized
        if len(args) and (args[0] is None or isinstance(args[0], dict)): # Parameters can also be the first argument
            pars, args = args[0], args[1:]
        members = ss.transitions.class_members(type(self))
        declared = (ss.State, ss.Transition, ss.outcome, ss.transitions.Expr, ss.transitions.FlowSum)
        if not args and not members and not any(isinstance(v, declared) or key in ss.transitions.Graph.grouped for key,v in kwargs.items()):
            raise ss.CapabilityError(f'"{type(self).__name__}" declares no states or transitions. A disease written as code (with define_states() and step()) subclasses ss.CodedDisease\n'
                             f'     (Starsim 3\'s ss.Disease), or ss.Infection if it is transmitted between agents;\n'
                             f'     a declared disease gives its transitions, e.g. ss.Disease(infection=ss.transmission("S -> I", beta=0.1), recovery=ss.progression("I -> R", dur=10)).')
        self.init_graph(args, members, pars, kwargs, frame=ss.utils.getframe(1))
        return

    @staticmethod
    def reuse(args, members, frame=None):
        """ States and transitions that already belong to another disease are copied (together, so they refer to each other), so one set can build several diseases; parameters stay shared """
        owners = {}
        pars = {}
        endpoints = set() # States that the transitions given move people between, which this disease will own
        counted = {} # States counted in a rate (I.n), which may be another disease's
        def walk(obj, depth=0): # The states, transitions, and parameters in what was given
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item, depth)
            elif isinstance(obj, ss.par):
                pars[id(obj)] = obj
            elif isinstance(obj, (ss.State, ss.Transition, ss.transitions.Expr, ss.outcome, ss.transitions.FlowSum)) and depth < 6:
                if getattr(obj, 'module', None) is not None and isinstance(obj, (ss.State, ss.Transition)):
                    owners[id(obj.module)] = obj.module
                if isinstance(obj, ss.Transition):
                    endpoints.update(id(st) for st in (obj.src or []) + (obj.dst or []) if isinstance(st, ss.State))
                elif isinstance(obj, ss.State) and depth == 0:
                    endpoints.add(id(obj))
                elif isinstance(obj, ss.transitions.Count):
                    counted.update({id(st):st for st in obj.states})
                for key,val in vars(obj).items():
                    if key != 'module':
                        walk(val, depth + 1)
        walk(list(args) + list(members.values()))
        others = {key:st for key,st in counted.items() if key not in endpoints and getattr(st, 'module', None) is not None} # Another disease's state in a rate, e.g. syph.I.n: a reference to it, not a copy
        if not owners:
            return args, members
        members = dict(members)
        grouped = [obj for key in ss.transitions.Graph.grouped for obj in sc.tolist(members.pop(key, None))]
        positional = []
        for obj in list(args) + grouped: # Named first, as keywords, since a copy is not bound to the variable
            name = getattr(obj, 'name', None) or (None if isinstance(obj, ss.State) else ss.transitions.find_name(obj, frame))
            if name is None or isinstance(obj, ss.State):
                positional.append(obj)
            else:
                members[name] = obj
        args = positional
        memo = {key:None for key in owners} | {key:p for key,p in pars.items()} | others # The copies belong to no module yet, and parameters stay shared, e.g. one R0 for several models
        return sc.dcp((args, members), memo=memo)

    def init_graph(self, args, members, pars, kwargs, frame=None):
        """ Build the graph of declared states and transitions; see `ss.transitions.Graph` """
        meta = {key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs}
        declared = (ss.State, ss.Transition, ss.outcome, ss.transitions.Expr, ss.transitions.FlowSum, ss.transitions.Transfer)
        timepars = {key:kwargs.pop(key) for key in ss.Timeline.time_args if key in kwargs and not isinstance(kwargs[key], declared)} # A transition can be called "stop"
        init_prev = kwargs.pop('init_prev', None)
        init = kwargs.pop('init', None)
        self.set_metadata(**meta)
        self.t.update(**timepars)
        members = sc.dcp(members) # Copy class members (together, so they still refer to each other) so instances do not share them
        for key in [key for key,val in kwargs.items() if isinstance(val, ss.par) and isinstance(members.get(key), ss.par)]: # e.g. ss.SIS(beta=ss.par(0.1, bounds=[0.05, 1])): the class's parameter, which its transitions use, takes the value and bounds
            members[key].merge(kwargs.pop(key))
        members = sc.mergedicts(members, kwargs)
        args, members = self.reuse(args, members, frame)
        self.graph = ss.transitions.Graph(self, args, members, init_prev=init_prev, init=init, frame=frame)
        for name,obj in list(self.graph.states.items()) + list(self.graph.transitions.items()):
            self.setattribute(name, obj) # e.g. sir.S, sir.infection
        for tr in list(self.graph.transitions.values()) + list(self.graph.transfers.values()):
            tr.module = self
        for name,tf in self.graph.transfers.items(): # e.g. sir.vaccination, so it can be the target of ss.modify()
            self.setattribute(name, tf)
        for key in ['rel_sus', 'rel_trans']: # So they can be the target of ss.modify() before the sim exists
            self.setattribute(key, ss.connectors.Target(self, key))
        self.define_pars(**self.graph.pars) # The parameters themselves, so e.g. sir.pars.beta can be the target of ss.modify()
        for key in sc.mergedicts(pars):
            if key not in self.pars:
                raise ss.UnknownParameter(f'"{self.name}" has no parameter named "{key}". Its parameters are {sc.strjoin(self.pars.keys())}{ss.transitions.did_you_mean(key, self.pars.keys())}')
        self.pars.update(pars) # A declared parameter keeps its bounds and name: see ss.Pars
        return

    def stratify(self, inplace=True, **kwargs):
        """
        Stratify the disease by a property of the people in it (age, risk, place, ...), in place, and return it

        Args:
            split (list): the share of the population in each level (if not given: from a named population or geography; for age groups, the world's ages; for a boolean dimension, or one that people are moved along, everyone at the first level; otherwise equal)
            adjust (dict): multipliers on named transitions, one per level, e.g. `{'recovery': [1.5, 1.0, 0.5]}`
            kind (str): 'age', 'space', 'strain', or 'plain'; if not given, from the value: 'space' for an `ss.geo` or with `mobility=`, 'strain' with `immunity=`, 'age' for `age=`, and otherwise 'plain' (whatever the name)
            states (list): the states the dimension applies to (default: every state that is people; for a strain, every state reached only through infection)
            where (dict): the groups the dimension applies to, e.g. `dict(sex='female')`; people in other groups have no level of it, so selecting a level of it leaves them out (default: every group)
            mobility (gravity/Mobility/array): for a place (e.g. `patch=`), the movement between its levels: `ss.gravity()`, `ss.Mobility()`, or a matrix
            inplace (bool): if False, return a stratified copy and leave this disease unchanged
            kwargs (dict): the dimension and its levels (bin edges or labels), e.g. `age=[0, 15, 65]`;
                an `ss.transfer` passed as a keyword is added under that name

        Examples:
            ```python
            sir.stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
            sir.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
            sir.stratify(parity=['0', '1-2', '3+'], where=dict(sex='female')) # Only women have a parity
            ```
        """
        mod = self if inplace else self.copy()
        mod.graph.stratify(frame=ss.utils.getframe(1), **kwargs)
        for strat in mod.graph.strats: # So a dimension can be used in a condition, e.g. where=~sir.vaccinated
            if not hasattr(mod, strat.name):
                mod.setattribute(strat.name, ss.conditions.Cond('dim', strat.name))
        return mod

    @property
    def transitions(self):
        """ The table of declared transitions; printing it also shows the matrix of sources by destinations """
        return self.graph.transitions

    def describe(self, output=False):
        """
        Print the resolved model as text, with every guessed default marked

        Args:
            output (bool): if True, return the text rather than printing it
        """
        text = '\n'.join(self.graph.describe(self.sim))
        return text if output else print(text)

    def equations(self, output=False):
        """
        Print the ODE system of the declared transitions, with symbolic parameter names

        Args:
            output (bool): if True, return the text rather than printing it
        """
        text = '\n'.join(self.graph.equations(self.engine))
        return text if output else print(text)

    def diagram(self, format='plot', **kwargs):
        """
        Draw the state-transition diagram

        States are placed in columns by how many transitions they are from the first
        state, with counts (states that are not people) in a row below. Transmissions
        are dashed and red, infectious states are shaded, and a dotted grey line shows
        a state that a rate depends on (the source of an infection, or a count).

        Args:
            format (str): 'plot' (a Matplotlib figure), or 'mermaid' (the text of a Mermaid flowchart)
            kwargs (dict): passed to `plt.figure()`

        **Example**:

            sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10))
            sir.diagram()
            print(sir.diagram('mermaid'))
        """
        return self.graph.diagram(format=format, **kwargs)

    def solvers(self, output=False, **kwargs):
        """ Which solvers can run this disease, before anything runs: see `ss.Sim.solvers()`; keyword arguments go to the sim """
        return ss.Sim(self, verbose=0, **kwargs).solvers(output=output)

    def R0(self, matrix=False):
        """
        The basic reproduction number, from the next-generation matrix at the disease-free state

        The model is laid out as an ODE: if the disease is in a sim, a copy of that sim
        is made with `solver='ode'`; otherwise a sim with the disease alone, under
        homogeneous mixing (so use `sim.R0()` for a disease with a contact matrix or
        places). Values that vary in time are taken at the start.

        This is the largest eigenvalue of the next-generation matrix. Where infection
        passes through a vector (people infect mosquitoes, which infect people), each
        passage is a generation, so R0 is the square root of the classical Ross–Macdonald
        value. Shedding into a reservoir that is a count (such as water) is not an infection,
        so there R0 is the number of people infected through the reservoir by one person.

        Args:
            matrix (bool): if True, return the next-generation matrix (over the infected compartments of every group) rather than its largest eigenvalue

        **Example**:

            sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10))
            sir.R0() # 3.0
        """
        if self.sim is not None and self.sim.initialized:
            sim = self.sim.new(solver='ode', verbose=0).init()
            dis = sim.diseases[self.name]
        else: # This disease alone, whatever the sim names it
            sim = ss.Sim(sc.dcp(self), solver='ode', verbose=0).init()
            dis = sim.diseases[0]
        ngm = dis.engine.next_generation()
        if ngm is None:
            errormsg = f'Nothing in "{self.name}" moves people into an infected state, so it has no R0'
            raise ValueError(errormsg)
        return ngm if matrix else float(np.max(np.abs(np.linalg.eigvals(ngm))))

    def to_spec(self, filename=None, **kwargs):
        """
        The model spec: the states, parameters, and transitions as plain data (as `sim.to_spec()` holds it, for `sim.hash()` and `ss.diff()`)

        Args:
            filename (str): if provided, save to this file
            kwargs (dict): passed to `sc.savejson()`
        """
        out = self.graph.to_json(self.sim)
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

    def to_json(self, filename=None, **kwargs):
        """ The model spec, as in `sim.to_json()`; see `to_spec()` """
        return self.to_spec(filename=filename, **kwargs)

    def init_pre(self, sim):
        """ Link the disease to the sim, create objects, and initialize results; see Module.init_pre() for details """
        self.engine = ss.solvers.make_engine(self, sim)
        self.engine.define()
        super().init_pre(sim)
        self.engine.init_pre()
        if any(isinstance(a, ss.infection_log) for a in sim.analyzers.values()):
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys())
        return

    def init_post(self):
        """ Initialize the values of the states, and set the initial conditions of the declared states """
        super().init_post()
        self.engine.init_post()
        return

    def start_step(self):
        super().start_step()
        self.engine.start_step()
        return

    def step_state(self):
        """
        Nothing, for a declared model: all of its transitions are taken together in `step()`, after the
        interventions, so a delivery sees the same people under every solver (as agents and as compartments)
        """
        return

    def step_die(self, uids):
        """ Take the agents who died out of every state (as agents; called by `people.step_die()`) """
        if isinstance(self.engine, ss.solvers.Agents):
            self.engine.die(uids)
        return

    def not_naive(self):
        """ The agents who have left the first state (e.g. who are no longer susceptible), for dynamic rescaling (see `ss.Sim.rescale()`) """
        return (~self[self.graph.first.name]).uids

    def make_naive(self, uids):
        """ Put agents back in the first state, as if they had never had the disease; for dynamic rescaling (see `ss.Sim.rescale()`) """
        states = self.graph.people_states
        for state in states:
            self[state.name][uids] = False
            self.engine.clear_exit(state.name, uids)
        self[states[0].name][uids] = True
        self[f'ti_{states[0].name}'][uids] = self.ti
        return

    def step(self):
        """ Take the transitions of the step, those that are not contact-mediated and then those that are, followed by any custom functions """
        self.engine.step_state()
        self.engine.step()
        for func in self.graph.funcs.values():
            func(self.sim)
        return

    def update_results(self):
        self.engine.update_results()
        return

    def finalize_results(self):
        super().finalize_results()
        self.engine.finalize_results()
        return

    def shrink(self):
        super().shrink()
        if self.engine is not None:
            self.engine.shrink()
        return


class CodedDisease(Disease):
    """
    Base class for a disease written as code, with no states of its own

    A disease that its own code moves along, with no built-in infection states and no
    transmission: it defines its states (`define_states()`), results (`init_results()`),
    and what happens each step (`step_state()`, `step()`), as for any other module, and is
    listed with the sim's diseases. It is Starsim 3's `ss.Disease`. For a disease
    written as code that is transmitted between agents, subclass `ss.Infection`, which
    adds the susceptible and infected states, `rel_sus` and `rel_trans`, and transmission
    over networks; a disease that can be written as states and transitions is declared
    instead, with `ss.Disease` (see `ss.SIR`).

    Args:
        pars (dict): parameter values by name, as for `define_pars()`
        kwargs (dict): merged with `pars`

    **Example**:

        class Anemia(ss.CodedDisease):
            def __init__(self, **kwargs):
                super().__init__()
                self.define_pars(p_onset=ss.bernoulli(0.01))
                self.update_pars(**kwargs)
                self.define_states(ss.BoolState('anemic', label='Anemic'))

            def step_state(self):
                self.anemic[self.pars.p_onset.filter()] = True
    """

    def __init__(self, pars=None, **kwargs):
        ss.Module.__init__(self) # Written as code, so there is no graph of declared states to build (see ss.Disease)
        self.infection_log = None
        self.engine = None
        self.update_pars(pars, **kwargs)
        return

    @property
    def transitions(self):
        """ A disease written as code has no table of declared transitions """
        return None

    def needs_declared(self, what):
        """ Raise `E804` for something that needs the structure of a declared disease """
        raise ss.CapabilityError(f'{what} needs a disease declared from states and transitions, but "{self.name}" ({type(self).__name__}) is written as code.\n'
                         f'     Declare it with ss.Disease (e.g. ss.SIR, or ss.Disease(infection=ss.transmission("S -> I", beta=0.1), recovery=ss.progression("I -> R", dur=10))).')

    def describe(self, output=False):
        """ Print a one-line summary: a disease written as code has no declared states and transitions to show """
        text = f"{type(self).__name__} '{self.name}': written as code (states: {sc.strjoin(self.state_dict.keys())})"
        return text if output else print(text)

    def equations(self, output=False):
        return self.needs_declared('equations()')

    def diagram(self, format='plot', **kwargs):
        return self.needs_declared('diagram()')

    def R0(self, matrix=False):
        return self.needs_declared('R0()')

    def stratify(self, inplace=True, **kwargs):
        return self.needs_declared('stratify()')

    def to_json(self, filename=None, **kwargs):
        out = ss.Module.to_json(self)
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim and initialize results; see `ss.Module.init_pre()` """
        ss.Module.init_pre(self, sim)
        if any(isinstance(a, ss.infection_log) for a in sim.analyzers.values()):
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys())
        return

    def init_post(self):
        ss.Module.init_post(self)
        return

    def start_step(self):
        ss.Module.start_step(self)
        return

    def step_state(self):
        pass

    def step_die(self, uids):
        pass

    def step(self):
        pass

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses upon acquiring the disease (by default, only adding to the log if there is an `ss.infection_log` analyzer) """
        if self.infection_log:
            self.infection_log.add_entries(uids, sources, self.now)
        return

    def make_naive(self, uids, skip_states=None):
        """
        Reset agents to the state of never having had the disease; used for dynamic rescaling (see `ss.Sim.rescale()`)

        By default, this resets every state of the disease to its default value, except
        `rel_sus` and `rel_trans`, which are often modified by other modules (e.g. vaccines).
        Override this method if the disease needs something different.

        Args:
            uids (`ss.uids`): the agents to make naive
            skip_states (list): names of states not to reset (default `['rel_sus', 'rel_trans']`)
        """
        if skip_states is None:
            skip_states = ['rel_sus', 'rel_trans']
        skip_states = sc.tolist(skip_states)
        for state in self.state_list:
            if state.name not in skip_states:
                state.set(uids)
        return

    def update_results(self):
        ss.Module.update_results(self)
        return

    def finalize_results(self):
        ss.Module.finalize_results(self)
        return

    def shrink(self):
        ss.Module.shrink(self)
        return


class Infection(CodedDisease):
    """
    Base class for diseases written as code

    A disease written as code defines its own per-agent states (`define_states()`)
    and how they change (`step_state()`, `step()`, `set_prognoses()`), as for any other
    module, so it can hold anything an agent can have: a CD4 count, an immunity level
    that wanes, a time since infection. It runs as agents. This class implements
    network-based transmission with directional beta values, and defines the attributes
    (`rel_sus`, `rel_trans`) that other modules act on; a disease that is not transmitted
    (such as `ss.NCD`) defines no `beta` and its own `step()`.

    A disease that can be written as states and transitions is declared instead, with
    `ss.Disease` (see `ss.SIR`), which gives it every solver and `describe()`, `equations()`,
    and `R0()`. The v4 verbs that act on arrays work on both: `ss.modify(hiv.rel_trans, ...)`,
    conditions such as `hiv.cd4 < 200`, and fitting the values in `pars`.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infected'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            ss.FloatArr('ti_infected', label='Time of infection' ),
            infectious = 'infected', # Only infectious agents transmit; by default, everyone infected is infectious
        )

        self.define_pars(
            init_prev = None, # Replace None with a ss.bernoulli to seed infections
        )
        self.update_pars(pars, **kwargs)

        # Define random number generator for determining transmission
        self.trans_rng = ss.multi_random('source', 'target')
        return

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim and initialize results; see `ss.Module.init_pre()` """
        super().init_pre(sim)
        if 'beta' in self.pars: # A disease that is not transmitted, such as ss.NCD, has no beta
            self.validate_beta()
        self.validate_init_prev()
        return

    def validate_init_prev(self):
        """ Check that init_prev is a probability, not a count (which should use `ss.choose_n()`) """
        init_prev = self.pars.init_prev
        p = init_prev.pars.p if isinstance(init_prev, ss.bernoulli) else init_prev
        if sc.isnumber(p) and p > 1:
            errormsg = f'init_prev={p} is not a valid probability; to infect exactly {p} agents, use init_prev=ss.choose_n({p})'
            raise ValueError(errormsg)
        if sc.isnumber(init_prev): # A share given where the class has no distribution to set it on (the base class's default is None)
            self.pars.init_prev = ss.bernoulli(init_prev)
        return

    def init_post(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the Arr objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        ss.Module.init_post(self)
        if self.pars.init_prev is None:
            return

        initial_cases = self.pars.init_prev.filter()
        if len(initial_cases):
            self.set_prognoses(initial_cases, sources=-1)  # -1 = externally seeded infection with no source agent
        return initial_cases

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        self.define_results(
            ss.Result('prevalence',     dtype=float, scale=False, label='Prevalence'),
            ss.Result('new_infections', dtype=int,   scale=True,  label='New infections'),
            ss.Result('cum_infections', dtype=int,   scale=True,  label='Cumulative infections'),
        )
        return

    def validate_beta(self):
        """ Validate beta and return as a map to match the networks """
        sim = self.sim
        β = self.pars.beta

        def scalar_beta(β):
            return isinstance(β, ss.Rate) or sc.isnumber(β)

        # If beta is a scalar, apply this bi-directionally to all networks
        if scalar_beta(β):
            betamap = {ss.standardize_netkey(k):[β,β] for k in sim.networks.keys()}

        # If beta is a dict, check all entries are bi-directional
        elif isinstance(β, dict):
            betamap = dict()
            for k,thisbeta in β.items():
                nkey = ss.standardize_netkey(k)
                if scalar_beta(thisbeta):
                    betamap[nkey] = [thisbeta, thisbeta]
                else:
                    betamap[nkey] = thisbeta

        else:
            errormsg = f'Invalid type {type(β)} for beta'
            raise TypeError(errormsg)

        # Check that it matches the network
        netkeys = [ss.standardize_netkey(k) for k in list(sim.networks.keys())]
        if set(betamap.keys()) != set(netkeys):
            missing = sorted(set(netkeys) - set(betamap.keys()))
            extra = sorted(set(betamap.keys()) - set(netkeys))
            errormsg = f'Network keys ({netkeys}) and beta keys ({list(betamap.keys())}) do not match for disease "{self.name}"'
            if missing: errormsg += f'; missing beta for network(s) {missing}'
            if extra:   errormsg += f'; no network(s) found matching {extra}'
            raise ValueError(errormsg)

        return betamap

    def step_state(self):
        """
        Carry out updates at the start of the timestep (prior to transmission);
        these are typically state changes, such as recovery
        """
        pass

    def step_die(self, uids):
        """
        Carry out state changes upon death

        This function is triggered after deaths are resolved, and before analyzers are run.
        See `ssl.CodedSIR` for a typical use case - deaths are requested as an autonomous
        update, to take effect after transmission on the same timestep. State changes that occur
        upon death (e.g., clearing an `infected` flag) are executed in this function. That also
        allows an intervention to avert a death scheduled on the same timestep, without having
        to undo any state changes that have already been applied (because they only run via this
        function if the death actually occurs).

        Unlike other methods during the integration loop, this method is not called directly
        by the sim; instead, it is called by people.step_die(), which reconciles the UIDs of
        the agents who will die.
        """
        pass

    def not_naive(self):
        """ The agents who have ever been infected, for dynamic rescaling (see `ss.Sim.rescale()`) """
        return self.ti_infected.notnan.uids

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses upon infection: make the agents non-susceptible, and record the new infections

        Subclasses extend this to assign state values upon infection (calling `super().set_prognoses()`).
        Infections are added to the log here if there is an `ss.infection_log` analyzer.

        Args:
            uids (array): UIDs for agents to assign disease prognoses to
            sources (array): Optionally specify the infecting agent
        """
        if self.infection_log:
            self.infection_log.add_entries(uids, sources, self.now)
        self.susceptible[uids] = False
        if self.sim.initialized: # Infections seeded by init_post() are prevalent, not incident, so don't count them
            self.results.new_infections[self.ti + 1] += len(uids) # Count infections when they happen, rather than inferring them from ti_infected; at the end of the step (self.ti is its start)
        return

    def step(self):
        """
        Perform key infection updates, including infection and setting prognoses
        """
        # Create new cases
        new_cases, sources, networks = self.infect() # TODO: store outputs in self or use objdict rather than 3 returns

        # Set prognoses
        if len(new_cases):
            self.set_outcomes(new_cases, sources)
            if self.infection_log:
                self.infection_log.add_data(new_cases, network=networks)

        return new_cases, sources, networks

    @staticmethod
    @nb.jit(cache=True) # No fastmath: a stray NaN must compare False (no transmission), not be assumed absent
    def _nb_transmit(src, trg, rel_trans, rel_sus, beta_per_dt, randvals):
        """ Optimized transmission kernel: returns the (source, target) UIDs of transmitting edges.

        Fuses the gather/multiply/compare *and* the UID extraction into a single branchless pass.
        This avoids the full-length boolean temporary and the two separate gather passes
        (`trg[mask]`, `src[mask]`) of the previous approach -- ~1.4x faster at typical transmission
        rates, more when transmission is common. UIDs are emitted in edge order, preserving CRN behavior.
        """
        n = src.shape[0]
        src_out = np.empty(n, dtype=np.int64) # int64 = uid dtype, so the caller can .view(uids) without a copy
        trg_out = np.empty(n, dtype=np.int64)
        m = 0
        for i in range(n):
            transmitted = rel_trans[src[i]] * rel_sus[trg[i]] * beta_per_dt[i] > randvals[i]
            src_out[m] = src[i] # Written every iteration (branchless); kept only if m advances
            trg_out[m] = trg[i]
            m += transmitted
        return src_out[:m], trg_out[:m]

    @staticmethod
    @nb.jit(cache=True) # No fastmath: a stray NaN must compare False (edge dropped), as in _nb_transmit()
    def _nb_can_transmit(src, trg, rel_trans, rel_sus, beta_per_dt):
        """ Return the indices of the edges with a nonzero transmission probability, in edge order """
        n = src.shape[0]
        inds = np.empty(n, dtype=np.int64)
        m = 0
        for i in range(n):
            can_transmit = rel_trans[src[i]] * rel_sus[trg[i]] * beta_per_dt[i] > 0 # Same expression as _nb_transmit()
            inds[m] = i # Written every iteration (branchless); kept only if m advances
            m += can_transmit
        return inds[:m]

    def compute_transmission(self, src, trg, rel_trans, rel_sus, beta_per_dt, randvals):
        """ Compute the probability of a->b transmission for networks (for other routes, the Route handles this) """
        if np.ndim(beta_per_dt) == 0: # net_beta returns a per-edge array, but tolerate a scalar
            beta_per_dt = np.full(len(src), beta_per_dt, dtype=ss_float)
        source_arr, target_arr = self._nb_transmit(np.asarray(src), np.asarray(trg), rel_trans.raw, rel_sus.raw, beta_per_dt, randvals)
        return target_arr.view(ss.uids), source_arr.view(ss.uids) # view (no copy): kernel output is uniquely owned int64; concatenate() compacts later

    def infect(self):
        """
        Determine who gets infected on this timestep via transmission on the network

        Computes the effective transmissibility and susceptibility, calls `infect_route()`
        for each network (or other route), and then removes duplicate infections with
        `finalize_infections()`. A multi-strain disease can override this method to loop
        over strains, calling `infect_route()` with per-strain values.
        """
        if 'beta' not in self.pars: # A disease that is not transmitted, such as ss.NCD, has no beta
            return self.finalize_infections([], [], [])
        betamap = self.validate_beta()

        # Compute effective transmissibility and susceptibility directly on the raw
        # (full-length) arrays. This avoids the gather/scatter, full-length astype copy,
        # and extra wrapper allocations of the Arr math operators; edges only ever index
        # living agents, so stale raw values for inactive agents are never used.
        rel_trans = self.rel_trans.asnew(self.infectious.raw * self.modified('rel_trans'), copy=False)
        rel_sus   = self.rel_sus.asnew(self.susceptible.raw * self.modified('rel_sus'), copy=False)

        new_cases = []
        sources = []
        networks = []
        for i, (nkey,route) in enumerate(self.sim.networks.items()):
            betas = betamap[ss.standardize_netkey(nkey)]
            target_uids, source_uids, network_ids = self.infect_route(i, route, betas, rel_trans, rel_sus)
            new_cases.append(target_uids)
            sources.append(source_uids)
            networks.append(network_ids)

        return self.finalize_infections(new_cases, sources, networks)

    def modified(self, key):
        """ The raw values of `rel_sus` or `rel_trans`, with any modifiers of them applied (see `ss.modify()`) """
        arr = self[key]
        if isinstance(arr, ss.ModifiableArr) and arr.engine is not None:
            return arr.raw*arr.engine.mult_raw(key)
        return arr.raw

    def infect_route(self, i, route, betas, rel_trans, rel_sus):
        """
        Compute the transmission along a single network (or other route)

        Args:
            i (int): the index of the route in `sim.networks`, stored as the network ID of each infection
            route (`ss.networks.Route`): the network or other route (e.g. a mixing pool)
            betas (list): the pair of betas for the route (p1→p2 and p2→p1); routes other than networks only use the first
            rel_trans (`ss.FloatArr`): the effective transmissibility of each agent (0 if not infectious)
            rel_sus (`ss.FloatArr`): the effective susceptibility of each agent (0 if not susceptible)

        Returns:
            A tuple of the target UIDs, the source UIDs, and the network IDs of the new infections (possibly with duplicates)
        """
        new_cases = []
        sources = []

        # Main use case: networks
        if isinstance(route, ss.Network):
            if len(route): # Skip networks with no edges
                edges = route.edges
                p1_to_p2 = [edges.p1, edges.p2, betas[0]]  # p1→p2 direction, beta 0
                p2_to_p1 = [edges.p2, edges.p1, betas[1]]  # p2→p1 direction, beta 1
                for src, trg, beta in [p1_to_p2, p2_to_p1]:
                    if beta: # Skip networks with no transmission
                        disease_beta = beta.to_prob(self.t.dt) if isinstance(beta, ss.Rate) else beta
                        beta_per_dt = route.net_beta(disease_beta=disease_beta, disease=self) # Compute beta for this network and timestep
                        if np.ndim(beta_per_dt) == 0: # net_beta returns a per-edge array, but tolerate a scalar
                            beta_per_dt = np.full(len(src), beta_per_dt, dtype=ss_float)

                        # Only draw random numbers for edges that can transmit (infectious source, susceptible target). With CRN, each
                        # edge's random number depends only on its source and target, so this gives identical results to drawing for every edge.
                        inds = self._nb_can_transmit(src, trg, rel_trans.raw, rel_sus.raw, beta_per_dt)
                        src_can = src[inds]
                        trg_can = trg[inds]
                        beta_can = beta_per_dt[inds]
                        randvals = self.trans_rng.rvs(src_can, trg_can) # Generate a new random number based on the two other random numbers (called even if empty, to keep CRN draws in sync)
                        args = (src_can, trg_can, rel_trans, rel_sus, beta_can, randvals) # Set up the arguments to calculate transmission
                        target_uids, source_uids = self.compute_transmission(*args) # Actually calculate it
                        new_cases.append(target_uids)
                        sources.append(source_uids)

        # Handle everything else: mixing pools, environmental transmission, etc.
        elif isinstance(route, ss.networks.Route):
            # Mixing pools are unidirectional, only use the first beta value
            disease_beta = betas[0].to_prob(self.t.dt) if isinstance(betas[0], ss.Rate) else betas[0]
            target_uids = route.compute_transmission(rel_sus, rel_trans, disease_beta, disease=self)
            new_cases.append(target_uids)
            sources.append(np.full(len(target_uids), dtype=ss_int, fill_value=ss.dtypes.int_nan))
        else:
            errormsg = f'Cannot compute transmission via route {type(route)}; please subclass ss.networks.Route and define a compute_transmission() method'
            raise TypeError(errormsg)

        new_cases = ss.uids.concatenate(new_cases)
        sources = ss.uids.concatenate(sources)
        networks = np.full(len(new_cases), dtype=ss_int, fill_value=i)
        return new_cases, sources, networks

    def finalize_infections(self, new_cases, sources, networks):
        """
        Combine the infections from each route, keeping only the first infection of each agent

        Args:
            new_cases (list): the target UIDs from each call to `infect_route()`
            sources (list): the corresponding source UIDs
            networks (list): the corresponding network IDs

        Returns:
            A tuple of the unique target UIDs, and their source UIDs and network IDs
        """
        if len(new_cases):
            new_cases = ss.uids.concatenate(new_cases)
            new_cases, inds = new_cases.unique(return_index=True)
            sources = ss.uids.concatenate(sources)[inds]
            networks = np.concatenate(networks)[inds]
        else:
            new_cases = ss.uids()
            sources = ss.uids()
            networks = np.empty(0, dtype=ss_int)
        return new_cases, sources, networks

    def set_outcomes(self, uids, sources=None):
        """
        Route newly infected agents to congenital or postnatal prognosis
        assignment based on age (age <= 0 is treated as in-utero).

        Args:
            uids (UIDs):    UIDs of newly infected agents.
            sources (UIDs): UIDs of the agents who transmitted to them (optional).
        """
        sim = self.sim
        congenital = sim.people.age[uids] <= 0
        if np.count_nonzero(congenital):
            src_c = sources[congenital] if sources is not None else None
            self.set_congenital(uids[congenital], src_c)
        src_p = sources[~congenital] if sources is not None else None
        self.set_prognoses(uids[~congenital], src_p)
        return

    # Birth outcomes that trigger request_death rather than setting a bool state.
    # Prenatal outcomes are fetal losses and must fire while the agent is still unborn;
    # postnatal outcomes happen to a live newborn. See set_congenital() for why this matters.
    prenatal_death_keys = {'miscarriage', 'stillborn'}
    postnatal_death_keys = {'neonatal_deaths'}
    congenital_death_keys = prenatal_death_keys | postnatal_death_keys

    def set_congenital(self, target_uids, source_uids=None):
        """
        Default implementation for assigning congenital outcomes during in-utero
        infection (called when transmission occurs via PrenatalNet).

        Does nothing unless the disease defines `birth_outcome_keys` and
        `birth_outcomes` in its pars. Diseases that need fully custom logic
        (e.g. syphilis, which has stage-dependent outcomes) can override this
        method entirely.

        To use the default implementation, define in the disease's `__init__`:

            self.define_pars(
                birth_outcome_keys = ['stillborn', 'congenital', 'normal'],
                birth_outcomes     = sc.objdict(default=ss.choice(a=3, p=[0.3, 0.4, 0.3])),  # Illustrative placeholder
            )

        Each outcome name needs a matching `ti_<name>` FloatArr state; non-lethal
        outcomes also need a BoolArr of the same name. Death outcomes ('miscarriage',
        'neonatal_deaths', 'stillborn') fire via `request_death`; others set a bool state.

        Outcomes are scheduled for the mother's delivery timestep, except for fetal
        losses ('miscarriage', 'stillborn'), which are scheduled one timestep earlier so
        that they occur before the fetus is delivered and are classified as fetal losses
        rather than neonatal deaths. A fetus infected within one timestep of delivery is
        therefore born alive, and a lethal outcome for it counts as a neonatal death.

        For state- or GA-dependent probabilities, provide multiple keyed
        distributions in `birth_outcomes` and override
        `_assign_congenital_outcomes`.

        Call `step_congenital` from the disease's `step_state()` to
        execute the scheduled events each timestep.
        """
        if 'birth_outcomes' not in self.pars or self.pars.birth_outcomes is None:
            return

        # Prevent repeated MTC transmission each timestep of pregnancy
        self.susceptible[target_uids] = False

        # Assign outcomes
        outcomes = self.pars.birth_outcomes
        if len(outcomes) == 1 and 'default' in outcomes:
            assigned = outcomes['default'].rvs(target_uids)
        else:
            assigned = self._assign_congenital_outcomes(target_uids, source_uids)

        # Store outcome index if the disease tracks it
        if hasattr(self, 'cs_outcome'):
            self.cs_outcome[target_uids] = assigned

        # Schedule events at delivery time
        preg = self.sim.demographics.pregnancy
        dt_delivery = preg.ti_delivery - self.ti  # Full array, indexed by UID below
        for oi, key in enumerate(self.pars.birth_outcome_keys):
            o_uids = target_uids[assigned == oi]
            s_uids = source_uids[assigned == oi]
            if len(o_uids):
                ti_key = f'ti_{key}'
                if hasattr(self, ti_key):
                    ti_event = self.ti + dt_delivery[s_uids]
                    if key in self.prenatal_death_keys:
                        # Fetal losses must fire while the agent is still unborn. Pregnancy.step()
                        # runs before diseases' step_state() in the integration loop, so an event
                        # scheduled at ti_delivery would kill an already-delivered newborn: it would
                        # be counted as a live birth and then as a neonatal death, rather than as a
                        # fetal loss. Firing a timestep early also matches the biology, since a
                        # stillborn fetus dies in utero before labor.
                        ti_event = np.maximum(ti_event - 1, self.ti)
                    getattr(self, ti_key)[o_uids] = ti_event
        return

    def _assign_congenital_outcomes(self, target_uids, source_uids):
        """
        Override point for diseases with state- or GA-dependent outcome probabilities.
        Must return an integer array of outcome indices (one per target_uid),
        corresponding to `self.pars.birth_outcome_keys`.
        """
        errormsg = 'Subclass must implement _assign_congenital_outcomes or use a single "default" distribution in birth_outcomes'
        raise NotImplementedError(errormsg)

    def step_congenital(self):
        """
        Execute scheduled congenital events whose `ti_<key>` has arrived.

        Does nothing unless the disease defines `birth_outcome_keys` in its
        pars. Call from the disease's `step_state()`; see `set_congenital`
        for setup details.
        """
        if 'birth_outcome_keys' not in self.pars:
            return
        death_keys = self.congenital_death_keys
        for key in self.pars.birth_outcome_keys:
            ti_key = f'ti_{key}'
            if not hasattr(self, ti_key):
                continue
            vals = getattr(self, ti_key)
            due = (vals.notnan & (vals <= self.ti)).uids
            if len(due):
                if key in death_keys:
                    self.sim.people.request_death(due)
                elif hasattr(self, key):
                    getattr(self, key)[due] = True
                vals[due] = np.nan  # Clear after firing
        return

    def update_results(self):
        """ Update prevalence; new_infections is recorded by set_prognoses() as infections happen """
        ss.Module.update_results(self)
        res = self.results
        ti = self.ti
        res.prevalence[ti] = res.n_infected[ti] / self.sim.people.n_alive
        return

    def finalize_results(self):
        """ Compute cumulative infections from the new-infections timeseries. """
        ss.Module.finalize_results(self) # Called first to scale the results
        res = self.results
        res.cum_infections[:] = np.cumsum(res.new_infections[:]) # Computed after scaling, since the scale can vary over time (see ss.Sim.rescale())
        ti_infected = getattr(self, 'ti_infected', None) # A subclass can remove it (define_states(reset=...))
        if ti_infected is not None and not res.new_infections.values.any() and np.any(ti_infected.raw > 0): # People were infected during the run, but none were counted
            ss.warn(f'"{self.name}": people were infected during the run, but new_infections is 0 at every time point. Infection.set_prognoses() counts them, '
                    f'so an override of it ({type(self).__name__}.set_prognoses()) should call super().set_prognoses(uids, sources)')
        return


class InfectionLog:
    """
    Record infections

    The infection log records transmission events and optionally other data
    associated with each transmission. Entries are stored as one chunk of arrays
    per call, so logging is cheap; they are combined when the log is read. Basic
    functionality is to track transmission with

    >>> Disease.infection_log.add_entries(targets, sources, t)

    or, for a single infection,

    >>> Disease.infection_log.append(source, target, t)

    Seed infections can be recorded with a source of `None` (or NaN), although all
    infections should have a target and a time. Other data can be captured in the
    log, either at the time of creation, or later on. For example

    >>> Disease.infection_log.add_entries(targets, sources, t, variant=variants)

    records extra data for each infection (`ss.Infection` records the network this way).
    Modules can optionally add per-infection outcomes later as well, for example

    >>> Disease.infection_log.add_data(uids, t_dead=2024.25)

    This would be equivalent to having specified the data at the original time the log
    entry was created - however, it is more useful for tracking events that may or may
    not occur after the infection and could be modified by interventions (e.g., tracking
    diagnosis, treatment, notification etc.)

    A table of outcomes can be returned using `InfectionLog.to_df()`, and a NetworkX
    graph with `InfectionLog.to_graph()`.

    Args:
        disease (str): the name of the disease being logged
        networks (list): the names of the networks, used to convert the network IDs recorded by `ss.Infection` to names in `to_df()`
    """
    def __init__(self, disease=None, networks=None):
        self.disease = disease
        self.networks = sc.tolist(networks)
        self.chunks = [] # One dict of arrays (t, source, target, and any extra data) per call to add_entries()
        self.updates = [] # Data added with add_data(), applied when the log is read
        self.n = 0 # Number of entries
        return

    def __len__(self):
        return self.n

    def __bool__(self):
        """ Ensure that zero-length infection logs are still truthy """
        return True

    def __repr__(self):
        """ Brief summary of the log, without building the dataframe """
        string = f'InfectionLog(disease={self.disease!r}, n={self.n}'
        if self.n:
            string += f', t={self.chunks[0]["t"][0]}–{self.chunks[-1]["t"][0]}'
        string += ')'
        return string

    def disp(self, **kwargs):
        """ Full display of the infection log """
        return sc.pr(self, **kwargs)

    def add_entries(self, uids, sources=None, time=np.nan, **kwargs):
        """
        Record new infections

        Args:
            uids (array): the UIDs of the infected agents (targets)
            sources (array/int): the UIDs of the infecting agents; None or NaN for seed infections
            time (any): the time of infection (all entries in the chunk share it)
            kwargs (dict): extra data to store, either a scalar or one value per UID
        """
        n = len(uids)
        if n:
            sources = np.nan if sources is None else sources
            chunk = dict(t=np.full(n, time, dtype=object), source=sources, target=uids, **kwargs)
            self.chunks.append({k:np.broadcast_to(v, n) if np.ndim(v) == 0 else np.array(v) for k,v in chunk.items()}) # Copy arrays, so later changes to them don't change the log
            self.n += n
        return

    def append(self, source, target, t, **kwargs):
        """ Record a single infection """
        self.add_entries([target], [source], t, **kwargs)
        return

    def add_data(self, uids, **kwargs):
        """
        Record extra infection data

        This method can be used to add data to an existing transmission event.
        The most recent transmission event for each agent will be used.

        Args:
            uids (array): The UIDs of the target nodes (the agents that were infected)
            kwargs (dict): Remaining arguments are stored as data for each entry, either a scalar or one value per UID
        """
        uids = np.array(uids) # Copy arrays, so later changes to them don't change the log
        if len(uids):
            kwargs = {k:np.array(v) if np.ndim(v) else v for k,v in kwargs.items()}
            self.updates.append((self.n, uids, kwargs)) # Store the number of entries so far, so only earlier entries are updated
        return

    def to_df(self):
        """
        Return a tabular representation of the log as a line list dataframe

        This function returns a dataframe containing columns for all quantities
        recorded in the log. Note that the log will contain `NaN` for quantities
        that are defined for some edges and not others (and which are missing for
        a particular entry)
        """
        if self.n == 0:
            return sc.dataframe(columns=['t', 'source', 'target'])

        df = pd.concat([pd.DataFrame(chunk) for chunk in self.chunks], ignore_index=True)

        # Apply data added later: for each UID, update its most recent entry at the time of the call
        for n, uids, kwargs in self.updates:
            targets = df.target.values[:n]
            inds = np.flatnonzero(np.isin(targets, uids))
            last = pd.Series(inds, index=targets[inds]).groupby(level=0).last() # Most recent entry for each UID
            for k,v in kwargs.items():
                if k not in df.columns:
                    df[k] = None # Object dtype, so any value can be stored
                if np.ndim(v):
                    v = pd.Series(v, index=uids)[last.index].values # Match per-UID values to the entries
                df.loc[last.values, k] = v

        # Convert network IDs to names, leaving any other values (e.g. names) as-is
        if self.networks and 'network' in df.columns:
            names = dict(enumerate(self.networks))
            df['network'] = df['network'].map(lambda v: names.get(v, v))

        df = df.sort_values(['t', 'source', 'target'], kind='stable')
        df = df.reset_index(drop=True)

        # Use Pandas "Int64" type to allow nullable integers. This allows the 'source' column
        # to have an integer type corresponding to UIDs while simultaneously supporting the use
        # of null values to represent exogenous/seed infections
        df = df.fillna(pd.NA)
        df['source'] = df['source'].astype("Int64")
        df['target'] = df['target'].astype("Int64")
        return sc.dataframe(df)

    def to_graph(self):
        """ Return the log as a NetworkX `MultiDiGraph`, with sources and targets as nodes and the time as the edge key """
        import networkx as nx # Lazy import since slow
        graph = nx.MultiDiGraph()
        for row in self.to_df().to_dict('records'):
            source, target, t = row.pop('source'), row.pop('target'), row.pop('t')
            graph.add_edge(np.nan if pd.isna(source) else source, target, key=t, **row)
        return graph


class NCD(Infection):
    """
    Example non-communicable disease

    This class implements a basic NCD model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having the condition, and associated
    mortality. It is written as code (see `ss.Infection`), and is not transmitted.

    Args:
        initial_risk (float/`ss.bernoulli`): initial prevalence of risk factors
        dur_risk (float/`ss.dur`/`ss.Dist`): how long a person is at risk for
        prognosis (float/`ss.dur`/`ss.Dist`): time in years between first becoming affected and death
    """
    def __init__(self, pars=None, initial_risk=_, dur_risk=_, prognosis=_, **kwargs):
        super().__init__()
        self.define_pars(
            initial_risk = ss.bernoulli(p=0.3), # Initial prevalence of risk factors
            dur_risk = ss.expon(scale=ss.years(10)),
            prognosis = ss.weibull(c=2, scale=ss.years(5)), # Time between first becoming affected and death; c is the (dimensionless) shape parameter
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('at_risk', label='At risk'),
            ss.BoolState('affected', label='Affected'),
            ss.FloatArr('ti_affected', label='Time of becoming affected'),
            ss.FloatArr('ti_dead', label='Time of death'),
            reset = True, # Not an infection, so without the states of ss.Infection
        )
        return

    @property
    def not_at_risk(self):
        """ Boolean array of agents not currently at risk. """
        return ~self.at_risk

    def init_post(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        super().init_post()
        initial_risk = self.pars['initial_risk'].filter()
        self.at_risk[initial_risk] = True
        self.ti_affected[initial_risk] = self.ti + self.pars['dur_risk'].rvs(initial_risk, round=True)
        return initial_risk

    def step_state(self):
        ti = self.ti
        deaths = (self.ti_dead <= ti).uids # <= since a prognosis that rounds to 0 is only caught on the next step
        self.sim.people.request_death(deaths)
        if self.infection_log:
            self.infection_log.add_data(deaths, died=True)
        self.results.new_deaths[ti + 1] = len(deaths) # Log deaths attributable to this module, at the end of the step (ti is its start)
        return

    def step(self):
        ti = self.ti
        new_cases = (self.ti_affected == ti).uids
        self.affected[new_cases] = True
        dur_prog = self.pars.prognosis.rvs(new_cases, round=True)
        self.ti_dead[new_cases] = ti + dur_prog
        if self.infection_log:
            self.infection_log.add_entries(new_cases, None, self.now)
        return new_cases

    def init_results(self):
        """
        Initialize results (not those of an infection)
        """
        ss.Module.init_results(self)
        self.define_results(
            ss.Result('n_not_at_risk', dtype=int,   label='Not at risk'),
            ss.Result('prevalence',    dtype=float, label='Prevalence'),
            ss.Result('new_deaths',    dtype=int,   label='Deaths'),
        )
        return

    def update_results(self):
        ss.Module.update_results(self)
        ti = self.ti
        self.results.n_not_at_risk[ti] = np.count_nonzero(self.not_at_risk)
        self.results.prevalence[ti]    = np.count_nonzero(self.affected)/self.sim.people.n_alive
        return

    def finalize_results(self):
        ss.Module.finalize_results(self)
        return


class SIR(Disease):
    """
    An SIR model: susceptible, infected, and recovered, with deaths from infection

    Declared from states and transitions (see `ss.Disease`), so it runs with any
    solver: as agents over networks, or as compartments. The keywords are Starsim 3's.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected (or a number of people)
        dur_inf (float/`ss.dur`/`ss.Dist`): how long people are infected for
        p_death (float/`ss.bernoulli`): the probability of death from infection, at the end of the infection
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    The Starsim 3 names `susceptible`, `infected`, and `recovered` are aliases of
    `S`, `I`, and `R` for this release. Starsim 3's `ss.SIR`, written as code, is
    `ssl.CodedSIR`.

    **Example**:

        sim = ss.Sim(diseases=ss.SIR(beta=0.1, dur_inf=ss.years(6)), networks='random', n_agents=5000)
        sim.run().plot()
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_inf   = ss.par(ss.lognorm_ex(mean=ss.years(6)))
    p_death   = ss.par(0.01)
    S, I, R   = ss.states('S I R')
    infection = ss.transmission('S -> I', beta=beta)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.null, dur=dur_inf, p=p_death)
    standard_results = True

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        return

    susceptible = property(lambda self: self.S)
    infected    = property(lambda self: self.I)
    recovered   = property(lambda self: self.R)


class SEIR(Disease):
    """
    An SEIR model: as `ss.SIR`, with an exposed (latent) state before people become infectious

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected, who start exposed (or a number of people)
        dur_exp (float/`ss.dur`/`ss.Dist`): how long people are exposed (latent) for
        dur_inf (float/`ss.dur`/`ss.Dist`): how long people are infectious for
        p_death (float/`ss.bernoulli`): the probability of death from infection, at the end of the infection
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    The Starsim 3 names `susceptible`, `exposed`, `infectious`, and `recovered` are
    aliases of `S`, `E`, `I`, and `R` for this release, and `infected` is `E | I`.
    Starsim 3's `ss.SEIR`, written as code, is `ssl.CodedSEIR`.
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_exp   = ss.par(ss.lognorm_ex(mean=ss.years(0.5)))
    dur_inf   = ss.par(ss.lognorm_ex(mean=ss.years(6)))
    p_death   = ss.par(0.01)
    S, E, I, R = ss.states('S E I R')
    infection = ss.transmission('S -> E', beta=beta, source='I')
    latency   = ss.progression('E -> I', dur=dur_exp)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.null, dur=dur_inf, p=p_death)
    standard_results = True

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        return

    susceptible = property(lambda self: self.S)
    exposed     = property(lambda self: self.E)
    infectious  = property(lambda self: self.I)
    recovered   = property(lambda self: self.R)
    infected    = property(lambda self: self.E | self.I)


class SIS(Disease):
    """
    An SIS model: people recover from infection and are susceptible again

    Declared from states and transitions (see `ss.Disease`), so it runs with any solver.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected (or a number of people)
        dur_inf (float/`ss.dur`/`ss.Dist`): how long people are infected for
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    The Starsim 3 names `susceptible` and `infected` are aliases of `S` and `I` for
    this release. Starsim 3's `ss.SIS` also had a per-agent immunity level, boosted by
    each infection and waning over time (`waning=`, `imm_boost=`), which has no declared
    form; that model, written as code, is `ssl.CodedSIS`.
    """
    beta      = ss.par(ss.peryear(0.05))
    dur_inf   = ss.par(ss.lognorm_ex(mean=ss.years(10)))
    S, I      = ss.states('S I')
    infection = ss.transmission('S -> I', beta=beta)
    recovery  = ss.progression('I -> S', dur=dur_inf)
    standard_results = True

    def __init__(self, pars=None, **kwargs):
        immunity = [key for key in ['waning', 'imm_boost'] if key in kwargs or key in (pars or {})]
        if immunity:
            raise ss.UnknownParameter(f'ss.SIS has no {sc.strjoin(immunity)}: Starsim 3\'s SIS had a per-agent immunity level that waned, which has no declared form.\n'
                             f'     For that model, written as code, use ssl.CodedSIS({immunity[0]}=...); for immunity as a state, declare an SIRS, e.g. ss.Disease(infection=ss.transmission("S -> I", beta=0.05), recovery=ss.progression("I -> R", dur=10), waning=ss.progression("R -> S", dur=20)).')
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        return

    susceptible = property(lambda self: self.S)
    infected    = property(lambda self: self.I)


def library_kwargs(cls, kwargs):
    """
    The keywords of `ss.SIR`, `ss.SIS`, and `ss.SEIR`, as Starsim 3 gave them: 1% start infected unless
    said otherwise, as a parameter; a probability may be an `ss.bernoulli`; and a number, list, or dict for a duration
    sets the values of its distribution (e.g. `dur_inf=10` is a lognormal with a mean of 10, as by default)
    """
    if 'init' not in kwargs: # A parameter, so it can be set (sir.pars.init_prev = 0.05) or fitted like the others
        init_prev = kwargs.get('init_prev', 0.01)
        kwargs['init_prev'] = init_prev if isinstance(init_prev, ss.par) else ss.par(init_prev)
    if isinstance(kwargs.get('p_death'), ss.bernoulli):
        kwargs['p_death'] = kwargs['p_death'].pars.p
    for key in ['dur_inf', 'dur_exp']:
        val = kwargs.get(key)
        default = getattr(cls, key, None)
        if isinstance(default, ss.par) and val is not None and not isinstance(val, (ss.Dist, ss.par)):
            pars = ss.Pars({key: sc.dcp(default.value)})
            pars.update({key: val})
            kwargs[key] = pars[key]
    return kwargs
