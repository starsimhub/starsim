"""
Base classes for diseases
"""
import re
import inspect
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

ss_int = ss.dtypes.int
_ = None # For function signatures



class Disease(ss.Module):
    """
    A disease: declared from its states and the transitions between them, or written as code

    No methods need to be defined, and the disease can be run with any solver
    (see `ss.Sim`):

        sir = ss.Disease(
            infection = ss.transmission('S -> I', beta=0.3),
            recovery  = ss.progression('I -> R', dur=10),
            init_prev = 0.01,
        )

    The same disease can be declared in a class body (the class form), which can also
    add behavior by defining methods; `ss.SIR`, `ss.SIS`, and `ss.SEIR` are written
    this way.

    A subclass that declares nothing is a disease written as code, as in Starsim 3:
    its `__init__()` calls `super().__init__()` and then `define_pars()` and
    `define_states()`, and it defines what happens each step (`step_state()`, `step()`),
    as for any other module. It runs as agents. A disease written as code that is
    transmitted between agents subclasses `ss.Infection`, which adds the susceptible
    and infected states, `rel_sus` and `rel_trans`, and transmission over networks.

    Args:
        args (list): transitions and outcomes, named from the variables they are assigned to
        pars (dict): parameter values by name
        transitions (list): transitions, as a list (for longer models)
        results (list): extra results: sums and ratios of states
        outcomes (list): `ss.outcome` objects
        init_prev (float): the initial prevalence, seeded through the transmission
        init (dict): the initial fraction or count in each state, e.g. `dict(S=0.90, I=0.01, R=0.09)`
        first (str): the state that everyone not put elsewhere starts in; by default, the one that no transition moves people into (see `ss.transitions.Graph.anchor()`)
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

        # Written as code: Starsim 3's ss.Disease
        class Anemia(ss.Disease):
            def __init__(self, **kwargs):
                super().__init__()
                self.define_pars(p_onset=ss.bernoulli(0.01))
                self.update_pars(**kwargs)
                self.define_states(ss.BoolState('anemic', label='Anemic'))

            def step_state(self):
                self.anemic[self.pars.p_onset.filter()] = True
        ```
    """
    graph = None # The declared states and transitions; None for a disease written as code
    standard_results = False # Whether to add new_infections, cum_infections, and prevalence (for a disease with a transmission), as the library's diseases do
    code_pars = () # Parameters of a declared module that its own code reads rather than its graph (e.g. ss.Pregnancy's fertility rates)

    _coded = False # Whether the class is written as code (ss.Infection and its subclasses), so that the check in __init_subclass__() does not apply
    _coded_methods = ['set_prognoses', 'infect', 'set_outcomes', 'finalize_infections', 'compute_transmission', 'infect_route', 'set_congenital'] # Called only on a disease written as code

    def __init_subclass__(cls, **kwargs):
        """ A declared class never calls the methods of a disease written as code, so overriding one (e.g. set_prognoses() on a subclass of ss.SIS) would be silently ignored """
        super().__init_subclass__(**kwargs)
        overridden = [name for name in cls._coded_methods if name in vars(cls)]
        declared = any(isinstance(val, (ss.State, ss.Transition)) for b in cls.__mro__ for val in vars(b).values()) # Its class body, or a base's, declares states or transitions (e.g. ss.SIS)
        if overridden and declared and not cls._coded:
            base = next((f'ssl.Agent{b.__name__}' for b in cls.__mro__ if b.__module__ == __name__ and b.__name__ in ['SIR', 'SIS', 'SEIR']), 'ss.Infection')
            errormsg = (f'{cls.__name__} defines {sc.strjoin(overridden)}, which a declared disease never calls, so it would be silently ignored.\n'
                        f'     A disease written as code subclasses {base} instead (import starsim.library as ssl); a declared one changes its transitions.')
            raise ss.CapabilityError(errormsg)
        return

    def __init__(self, *args, pars=None, **kwargs):
        super().__init__()
        self.infection_log = None
        self.engine = None # Runs the graph; created when the sim is initialized
        if len(args) and (args[0] is None or isinstance(args[0], dict)): # Parameters can also be the first argument
            pars, args = args[0], args[1:]
        members = ss.transitions.class_members(type(self))
        declared = (ss.State, ss.Transition, ss.outcome, ss.transitions.Expr, ss.transitions.FlowSum)
        if args or members or any(isinstance(v, declared) or key in ss.transitions.Graph.grouped for key,v in kwargs.items()):
            self.init_graph(args, members, pars, kwargs, names=ss.utils.caller_names(1))
        else: # Nothing declared, so written as code: the subclass defines its parameters and states after calling this
            self.update_pars(pars, **kwargs)
        return

    def define_pars(self, inherit=True, **kwargs):
        """
        Create or merge parameters; see `ss.Module.define_pars()`

        For a declared disease, a parameter defined here that is not one of its graph's (e.g. by a subclass, after
        `super().__init__()`, for its own code to read) is added to `code_pars`, so the sim does not reject it as one
        that nothing reads. A parameter set any other way (e.g. a misspelling in `pars=`) is still rejected.
        """
        out = super().define_pars(inherit=inherit, **kwargs)
        if self.graph is not None:
            self.code_pars = list(self.code_pars) + [k for k in kwargs if k not in self.graph.pars and k not in self.code_pars]
        return out

    @staticmethod
    def reuse(args, members, names=None):
        """ States and transitions that already belong to another disease are copied (together, so they refer to each other), so one set can build several diseases; parameters stay shared """
        owners = {}
        pars = {}
        endpoints = set() # States that the transitions given move people between, which this disease will own
        counted = {} # States counted in a rate (I.n), which may be another disease's
        conds = [] # Conditions in a where=, which may be on another disease's states
        def walk(obj, depth=0, where=False): # The states, transitions, and parameters in what was given
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item, depth, where)
            elif isinstance(obj, ss.conditions.Cond):
                conds.append(obj)
            elif isinstance(obj, ss.State) and where: # A state in a where=, e.g. where=hpv16.Ca
                counted[id(obj)] = obj
            elif isinstance(obj, ss.par):
                pars[id(obj)] = obj
            elif isinstance(obj, (ss.State, ss.Transition, ss.transitions.Expr, ss.outcome, ss.transitions.FlowSum)) and depth < 6:
                if getattr(obj, 'module', None) is not None and isinstance(obj, (ss.State, ss.Transition)):
                    owners[id(obj.module)] = obj.module
                if isinstance(obj, ss.Transition):
                    ends = [st for st in (obj.src or []) + (obj.dst or []) if isinstance(st, ss.State)]
                    endpoints.update(id(st) for st in ends)
                    for st in ends:
                        ss.transitions.Graph.check_read(st)
                elif isinstance(obj, ss.State) and depth == 0:
                    endpoints.add(id(obj))
                elif isinstance(obj, ss.transitions.Count):
                    counted.update({id(st):st for st in obj.states})
                for key,val in vars(obj).items():
                    if key != 'module':
                        walk(val, depth + 1, where or key == 'where')
        walk(list(args) + list(members.values()))
        others = {key:st for key,st in counted.items() if key not in endpoints and getattr(st, 'module', None) is not None} # Another disease's state in a rate, e.g. syph.I.n: a reference to it, not a copy
        for cond in conds: # A condition on another disease's states is kept as it is, unless it is on a state being copied
            if not any(id(mod.graph.states.get(name)) in endpoints for mod,name in ss.conditions.leaf_states(cond) if getattr(mod, 'graph', None) is not None):
                others[id(cond)] = cond
        if not owners:
            return args, members
        members = dict(members)
        grouped = [obj for key in ss.transitions.Graph.grouped for obj in sc.tolist(members.pop(key, None))]
        positional = []
        for obj in list(args) + grouped: # Named first, as keywords, since a copy is not bound to the variable
            name = getattr(obj, 'name', None) or (None if isinstance(obj, ss.State) else (names or {}).get(id(obj)))
            if name is None or isinstance(obj, ss.State):
                positional.append(obj)
            else:
                members[name] = obj
        args = positional
        memo = {key:None for key in owners} | {key:p for key,p in pars.items()} | others # The copies belong to no module yet, and parameters stay shared, e.g. one R0 for several models
        return sc.dcp((args, members), memo=memo)

    def init_graph(self, args, members, pars, kwargs, names=None):
        """ Build the graph of declared states and transitions; see `ss.transitions.Graph` """
        meta = {key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs}
        declared = (ss.State, ss.Transition, ss.outcome, ss.transitions.Expr, ss.transitions.FlowSum, ss.transitions.Transfer)
        timepars = {key:kwargs.pop(key) for key in ss.Timeline.time_args if key in kwargs and not isinstance(kwargs[key], declared)} # A transition can be called "stop"
        init_prev = kwargs.pop('init_prev', None)
        init = kwargs.pop('init', None)
        first = kwargs.pop('first') if isinstance(kwargs.get('first'), str) else None # A state named "first" can still be declared by keyword
        self.set_metadata(**meta)
        self.t.update(**timepars)
        memo = {id(leaf):leaf for val in members.values() if isinstance(val, ss.Transition) for expr in [val.rate, val.dur] + sc.tolist(getattr(val, 'p', None)) if isinstance(expr, ss.transitions.Expr) for leaf in expr.leaves if isinstance(leaf, ss.par) and not any(leaf is m for m in members.values())}
        members = sc.dcp(members, memo=memo) # Copy class members (together, so they still refer to each other) so instances do not share them; a parameter defined outside the class (rate = ss.par(...) at module level) stays shared, as in the operator form, so it keeps its name
        kwargs = {k:ss.expressions.rebind(v, memo) for k,v in kwargs.items()} # A result given here that reads the class's states reads the copies
        for key in [key for key,val in kwargs.items() if isinstance(val, ss.par) and isinstance(members.get(key), ss.par)]: # e.g. ss.SIS(beta=ss.par(0.1, bounds=[0.05, 1])): the class's parameter, which its transitions use, takes the value and bounds
            members[key].merge(kwargs.pop(key))
        members = sc.mergedicts(members, kwargs)
        args, members = self.reuse(args, members, names)
        self.graph = ss.transitions.Graph(self, args, members, init_prev=init_prev, init=init, first=first, names=names)
        for name,obj in list(self.graph.states.items()) + list(self.graph.transitions.items()) + list(self.graph.attrs.items()):
            self.setattribute(name, obj) # e.g. sir.S, sir.infection, hiv.cd4
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

    def value(self, name):
        """
        The current value of a parameter, with the modifiers active on it applied

        For a declared parameter (an `ss.par`), a number in the sim's unit once the sim is
        initialized (before that, the value it holds); for any other parameter, the value itself.
        Use this from module code (e.g. a network's `net_beta()`) rather than `self.pars[name]`,
        which for a declared module is the `ss.par`.

        Args:
            name (str): the name of the parameter

        Examples:
            ```python
            eff = disease.value('eff_condom')
            ```
        """
        p = self.pars[name]
        if not isinstance(p, ss.par):
            return p
        if self.engine is None:
            return p.raw
        return self.engine.value(p)

    def stratify(self, inplace=True, **kwargs):
        """
        Stratify the disease by a property of the people in it (age, risk, place, ...), in place, and return it

        Args:
            split (list): the share of the population in each level (if not given: from a named population or geography; for age groups, the world's ages; for a boolean dimension, or one that people are moved along, everyone at the first level; otherwise equal)
            adjust (dict): multipliers on named transitions, one per level, e.g. `{'recovery': [1.5, 1.0, 0.5]}`: on its rate (a duration is divided), or on its probability for a branch chosen with p=
            kind (str): 'age', 'space', 'strain', 'clock' (the time since an event, e.g. `since_birth=ss.months([0, 3, 12]), kind='clock'`), or 'plain'; if not given, from the value: 'space' for an `ss.geo` or with `mobility=`, 'strain' with `immunity=`, 'age' for `age=`, and otherwise 'plain' (whatever the name)
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
        self.check_declared('stratify()')
        mod = self if inplace else self.copy()
        mod.graph.stratify(**kwargs)
        for strat in mod.graph.strats: # So a dimension can be used in a condition, e.g. where=~sir.vaccinated
            if not hasattr(mod, strat.name):
                mod.setattribute(strat.name, ss.conditions.Cond('dim', strat.name))
        return mod

    @property
    def transitions(self):
        """ The table of declared transitions (None for a disease written as code); printing it also shows the matrix of sources by destinations """
        return self.graph.transitions if self.graph is not None else None

    def check_declared(self, what):
        """ Raise `ss.CapabilityError` for something that needs the structure of a declared disease, if this one is written as code """
        if self.graph is None:
            raise ss.CapabilityError(f'{what} needs a disease declared from states and transitions, but "{self.name}" ({type(self).__name__}) is written as code.\n'
                                     f'     Declare it with ss.Disease (e.g. ss.SIR, or ss.Disease(infection=ss.transmission("S -> I", beta=0.1), recovery=ss.progression("I -> R", dur=10))).')
        return

    def describe(self, output=False):
        """
        Print the resolved model as text, with every guessed default marked (for a disease written as code, a one-line summary)

        Args:
            output (bool): if True, return the text rather than printing it
        """
        if self.graph is None:
            text = f"{type(self).__name__} '{self.name}': written as code (states: {sc.strjoin(self.state_dict.keys())})"
        else:
            text = '\n'.join(self.graph.describe(self.sim))
        return text if output else print(text)

    def equations(self, output=False):
        """
        Print the ODE system of the declared transitions, with symbolic parameter names

        Args:
            output (bool): if True, return the text rather than printing it
        """
        self.check_declared('equations()')
        text = '\n'.join(self.graph.equations(self.engine))
        return text if output else print(text)

    def diagram(self):
        """
        The state-transition diagram, as the text of a Mermaid flowchart (https://mermaid.js.org)

        People states are rounded and counts slanted; transmissions are dashed; each
        arrow is labelled with its transition's name and rate. `ss.save('sir.md', sim, format='mermaid')`
        writes the diagram of every disease in a sim.

        **Example**:

            sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10))
            print(sir.diagram())
        """
        self.check_declared('diagram()')
        return '\n'.join(['flowchart LR'] + self.graph.diagram())

    def R0(self, matrix=False):
        """
        The basic reproduction number, from the next-generation matrix at the disease-free state

        The model is laid out as an ODE: if the disease is in a sim, a copy of that sim
        is made with `solver='ode'`, with its mixing and stratification; otherwise a sim with
        the disease alone, under homogeneous mixing. Values that vary in time are taken at the start.

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
        self.check_declared('R0()')
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
        The disease's part of the model spec (`sim.to_spec()`): its states, parameters, and transitions as plain data, for `sim.hash()`, `ss.diff()`, and `ss.to_code()`; for a disease written as code, see `ss.Module.to_spec()`

        Args:
            filename (str): if provided, save to this file
            kwargs (dict): passed to `sc.savejson()`
        """
        out = self.graph.to_spec(self.sim) if self.graph is not None else super().to_spec()
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim, create objects, and initialize results; see Module.init_pre() for details """
        if self.graph is not None:
            unknown = [key for key in self.pars if key not in self.graph.pars and key not in self.code_pars] # Added after the disease was made, so nothing reads it (TL58: pars.init was ignored)
            if unknown:
                hint = ' The starting values are given when the disease is made, e.g. ss.Disease(..., init=dict(I=100)), and cannot be set through pars.' if 'init' in unknown else ''
                raise ss.UnknownParameter(f'"{self.name}" has no parameter named {sc.strjoin(unknown)}, so nothing would read it. Its parameters are {sc.strjoin(self.graph.pars.keys())}.{hint}')
            self.engine = ss.solvers.make_engine(self, sim)
            self.engine.define()
        elif not self.state_list and type(self).step is Disease.step and type(self).step_state is Disease.step_state:
            raise ss.CapabilityError(f'"{type(self).__name__}" declares no states or transitions, and defines no states or step(). A declared disease gives its transitions,\n'
                                     f'     e.g. ss.Disease(infection=ss.transmission("S -> I", beta=0.1), recovery=ss.progression("I -> R", dur=10)); a disease written as code\n'
                                     f'     defines its states (define_states()) and what happens each step (step_state(), step()), or subclasses ss.Infection if it is transmitted between agents.')
        super().init_pre(sim)
        if self.engine is not None:
            self.engine.init_pre()
        if any(isinstance(a, ss.infection_log) for a in sim.analyzers.values()):
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys())
        return

    def init_post(self):
        """ Initialize the values of the states, and set the initial conditions of the declared states """
        super().init_post()
        if self.engine is not None:
            self.engine.init_post()
        return

    def start_step(self):
        super().start_step()
        if self.engine is not None:
            self.engine.start_step()
        return

    def step_state(self):
        """
        Nothing, for a declared model: all of its transitions are taken together in `step()`, after the
        interventions, so a delivery sees the same people under every solver (as agents and as compartments).
        A disease written as code changes its states here, before transmission (e.g. recovery).
        """
        return

    def step_die(self, uids):
        """ Take the agents who died out of every state (as agents; called by `people.step_die()`) """
        if isinstance(self.engine, ss.solvers.Agents):
            self.engine.die(uids)
        return

    def not_naive(self):
        """ The agents who have left the first state (e.g. who are no longer susceptible), for dynamic rescaling (see `ss.Sim.rescale()`); none, for a disease written as code """
        if self.graph is None:
            return ss.uids()
        return (~self[self.graph.first.name]).uids

    def times_read(self, sim=None):
        """
        The states whose time of entering (`ti_X`) code in the sim reads, for a declared disease run as agents: a set of names, or None for all

        The code of the sim's modules, other than Starsim's own, and of this disease's class is
        searched for `ti_X` (e.g. a Starsim 3-style analyzer that reads `sir.ti_R`). A name made
        from pieces (e.g. `f'ti_{state}'`), or code whose source cannot be found, counts as reading
        them all, as does `ss.options.keep_times`. The engine keeps these, as well as the ones it
        needs itself (see `ss.solvers.Agents.define()`); the others are `ss.arrays.Unkept`.
        """
        if ss.options.keep_times:
            return None
        names = set()
        for mod in (sim or self.sim).modules:
            code = [cls for cls in type(mod).__mro__ if cls.__module__.split('.')[0] not in ['starsim', 'sciris', 'builtins']] # Starsim's own modules read none
            code += [f for f in vars(mod).values() if inspect.isfunction(f) or inspect.ismethod(f)] # E.g. a function as an analyzer (see ss.Module.from_func())
            for obj in code:
                try:
                    source = inspect.getsource(obj)
                except (OSError, TypeError): # E.g. defined at the prompt
                    return None
                if re.search(r"""ti_\{|['"]ti_['"]""", source): # A name made from pieces
                    return None
                names.update(re.findall(r'\bti_(\w+)', source))
        return names

    @property
    def ti_entry(self):
        """
        The time point that an infection in this step is dated at: the end of the step (`self.ti + 1`), since it has happened by then, or the start of the run for seeds

        A declared disease dates infections this way (`ti_I`); a disease written as code dates them the same way by
        setting e.g. `self.ti_infected[uids] = self.ti_entry` in `set_prognoses()`, and schedules what follows from it.
        So in `update_results()` and analyzers, where `self.ti` is the end of the step, this step's infections are `ti_infected == self.ti`.
        """
        return self.ti + 1 if self.sim.initialized else self.ti

    @property
    def now_entry(self):
        """ The time (e.g. a date) of `ti_entry`, which an infection log records """
        ti, tvec = self.ti_entry, self.t.tvec
        return tvec[ti] if ti < len(tvec) else tvec[0] + self.t.dt*ti # E.g. an infection after the last step

    def set_prognoses(self, uids, sources=None):
        """ For a disease written as code: set prognoses upon acquiring the disease (by default, only adding to the log if there is an `ss.infection_log` analyzer) """
        if self.infection_log:
            self.infection_log.add_entries(uids, sources, self.now_entry)
        return

    def move_to(self, uids, to, src=None):
        """
        Move some agents of a declared disease into a state, as a one-off move along no transition (e.g. from code: a cure, or seeding)

        Args:
            uids (uids): the agents to move
            to (str/State): the state to move them into
            src (str/list): the states to move them from (by default, any state of people other than `to`); agents in none of them stay where they are

        Returns:
            The agents moved

        **Example**::

            sir.move_to(uids, 'R', src='I') # A cure, from code
        """
        graph = getattr(self, 'graph', None)
        if graph is None or self.engine is None:
            raise ss.CapabilityError(f'"{self.name}" is not a declared disease (or the sim is not initialized), so it has no states to move people between; set its arrays directly')
        if self.sim.solver != 'abm':
            raise ss.CapabilityError(f'disease.move_to() moves agents, but "{self.name}" is run with solver="{self.sim.solver}"; use an intervention (e.g. ss.Delivery(move=...)) for a compartmental run')
        name = lambda s: s if isinstance(s, str) else s.name
        dst = graph.states[name(to)]
        srcs = [graph.states[name(s)] for s in sc.tolist(src)] if src is not None else [s for s in graph.people_states if s is not dst]
        return self.engine.move_states(ss.uids(uids), srcs, dst)

    def current_state(self, states=None):
        """
        The state each agent is in, as a read-only `ss.IntArr` (by UID) of its index among `states`, or -1 for none

        The names of the states, in order, are its `.labels`, so `arr.labels[i]` is the name of index `i`.

        Args:
            states (list): the states to look in, as names or states (default: the module's states of people, in order)

        **Example**::

            cs = sir.current_state() # 0 for S, 1 for I, 2 for R
            cs.labels[cs[uid]] # The name of an agent's state
        """
        graph = getattr(self, 'graph', None)
        if graph is None or self.engine is None:
            raise ss.CapabilityError(f'"{self.name}" is not a declared disease (or the sim is not initialized), so it has no states to look in')
        if self.sim.solver != 'abm':
            raise ss.CapabilityError(f'disease.current_state() is the state of each of the agents, but "{self.name}" is run with solver="{self.sim.solver}"; use the counts of its states (e.g. sim.results.{self.name}.n_S) instead')
        names = [s if isinstance(s, str) else s.name for s in sc.tolist(states)] if states is not None else [s.name for s in graph.people_states]
        unknown = [name for name in names if name not in graph.states]
        if unknown:
            raise ss.UnknownState(f'"{self.name}" has no state {sc.strjoin(unknown)}; its states are {sc.strjoin(graph.states.keys())}')
        age = self.sim.people.age # A plain array of the people, to copy
        raw = np.full(len(age.raw), -1, dtype=ss.dtypes.int)
        for i,name in enumerate(names):
            raw[self[name].raw] = i
        raw.flags.writeable = False
        out = age.asnew(raw, cls=ss.IntArr, name='current_state', copy=False)
        out.label = f'Current state of {self.name}'
        out.nan = out.default = -1 # In none of the states
        out.labels = names
        return out

    def make_naive(self, uids, skip_states=None):
        """
        Reset agents to the state of never having had the disease; for dynamic rescaling (see `ss.Sim.rescale()`)

        A declared disease puts them back in its first state. A disease written as code
        resets every state to its default value, except `rel_sus` and `rel_trans`, which
        are often modified by other modules (e.g. vaccines); override this method if the
        disease needs something different.

        Args:
            uids (`ss.uids`): the agents to make naive
            skip_states (list): for a disease written as code, the names of states not to reset (default `['rel_sus', 'rel_trans']`)
        """
        if self.graph is None:
            skip_states = sc.tolist(skip_states) if skip_states is not None else ['rel_sus', 'rel_trans']
            for state in self.state_list:
                if state.name not in skip_states:
                    state.set(uids)
            return
        states = self.graph.people_states
        for state in states:
            self[state.name][uids] = False
            self.engine.clear_exit(state.name, uids)
        self[states[0].name][uids] = True
        self[f'ti_{states[0].name}'][uids] = self.ti
        return

    def step(self):
        """ Take the transitions of the step, those that are not contact-mediated and then those that are, followed by any custom functions (nothing, for a disease written as code that does not define it) """
        if self.engine is None:
            return
        self.engine.step_state()
        self.engine.step()
        for func in self.graph.funcs.values():
            func(self.sim)
        return

    def update_results(self):
        if self.engine is None: # Written as code: the counts of its states
            return super().update_results()
        self.engine.update_results()
        return

    def finalize_results(self):
        super().finalize_results()
        if self.engine is not None:
            self.engine.finalize_results()
        return

    def shrink(self):
        super().shrink()
        if self.engine is not None:
            self.engine.shrink()
        return


class Infection(Disease):
    """
    Base class for diseases written as code

    A disease written as code defines its own per-agent states (`define_states()`)
    and how they change (`step_state()`, `step()`, `set_prognoses()`), as for any other
    module, so it can hold anything an agent can have: a CD4 count, an immunity level
    that wanes, a time since infection. It runs as agents. This class implements
    network-based transmission with directional beta values, and defines the attributes
    (`rel_sus`, `rel_trans`) that other modules act on; a disease that is not transmitted
    (such as `ssl.NCD`) defines no `beta` and its own `step()`.

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
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility', shared=True), # No memory until something sets them
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission', shared=True),
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
        if 'beta' in self.pars: # A disease that is not transmitted, such as ssl.NCD, has no beta
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

        people = self.sim.people
        unborn = people.unborn_uids() if isinstance(people, ss.People) else ss.uids() # E.g. the fetuses of pregnancies that ss.Pregnancy starts before the start, who are not seeded
        initial_cases = self.pars.init_prev.filter() if not len(unborn) else self.pars.init_prev.filter(people.auids[people.age.raw[people.auids] >= 0])
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

        nets = {ss.standardize_netkey(k):net for k,net in sim.networks.items()}
        def both_ways(k, β): # A single beta applies in both directions, except on a maternal network: from mother to child only
            net = nets.get(ss.standardize_netkey(k))
            return [β, β*0] if getattr(net, 'one_way', False) else [β, β]

        def as_pair(β, nkey):
            """ A probability per act is one value for both [p1 to p2, p2 to p1], as Starsim 3's pair; with directions, an array of a value for each of the network's (see `infect_route()`) """
            if not isinstance(β, ss.peract):
                return β
            net = nets.get(nkey)
            vals = β.resolve(net) if net is not None else β.values
            if len(vals) == 1:
                return vals*2
            return [np.asarray(vals, dtype=float)]*2

        # If beta is a scalar, apply this bi-directionally to all networks
        if scalar_beta(β) or isinstance(β, ss.peract):
            betamap = {ss.standardize_netkey(k):both_ways(k, β) if scalar_beta(β) else as_pair(β, ss.standardize_netkey(k)) for k in sim.networks.keys()}

        # If beta is a dict, check all entries are bi-directional; a default is the beta of every network it does not name, and the names that are not networks are not used
        elif isinstance(β, dict):
            betamap = dict()
            if 'default' in β:
                unknown = [k for k in β if k != 'default' and ss.standardize_netkey(k) not in nets]
                if unknown:
                    ss.note(f'the beta of "{self.name}" names {sc.strjoin(unknown)}, which {"is not a network" if len(unknown) == 1 else "are not networks"} of the sim, so {"it is" if len(unknown) == 1 else "they are"} not used; the networks not named take the default', sim=sim)
                β = {k:ss.utils.route_beta(β, k) for k in sim.networks.keys()}
            for k,thisbeta in β.items():
                nkey = ss.standardize_netkey(k)
                if scalar_beta(thisbeta):
                    betamap[nkey] = both_ways(nkey, thisbeta)
                else:
                    betamap[nkey] = as_pair(thisbeta, nkey)

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
        See `ssl.AgentSIR` for a typical use case - deaths are requested as an autonomous
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
            self.infection_log.add_entries(uids, sources, self.now_entry) # Dated at the end of the step, as the infection is (see ti_entry)
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

    def infect(self):
        """
        Determine who gets infected on this timestep via transmission on the network

        Computes the effective transmissibility and susceptibility, calls `infect_route()`
        for each network (or other route), and then removes duplicate infections with
        `finalize_infections()`. A multi-strain disease can override this method to loop
        over strains, calling `infect_route()` with per-strain values.
        """
        if 'beta' not in self.pars: # A disease that is not transmitted, such as ssl.NCD, has no beta
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

        On a network, each edge transmits with probability `rel_trans·rel_sus·net_beta()` (see `ss.Network.transmit()`), times the modifiers
        of the network's contacts and of `rel_sus` on the route (`ss.modify(..., route=)`), as for a declared disease run as agents.

        Args:
            i (int): the index of the route in `sim.networks`, stored as the network ID of each infection
            route (`ss.networks.Route`): the network or other route (e.g. a mixing pool)
            betas (list): the pair of betas for the route (p1→p2 and p2→p1), each a number or rate, or an array of a probability per act for each direction of the network; routes other than networks only use the first
            rel_trans (`ss.FloatArr`): the effective transmissibility of each agent (0 if not infectious)
            rel_sus (`ss.FloatArr`): the effective susceptibility of each agent (0 if not susceptible)

        Returns:
            A tuple of the target UIDs, the source UIDs, and the network IDs of the new infections (possibly with duplicates)
        """
        new_cases = []
        sources = []
        mods = self.sim.modifiers
        if mods.get((route.name, 'par', 'n_contacts')) or mods.get((self.name, 'route', f'rel_sus@{route.name}')): # Modifiers of its contacts, or on this route
            engine = ss.connectors.AgentModifiers.engine(self, self.sim)
            mult = lambda inds, trg: ss.connectors.route_mult(engine, 'rel_sus', route, inds, trg)
        else:
            mult = lambda inds, trg: 1.0

        # Main use case: networks
        if isinstance(route, ss.Network):
            def edge_beta(d, src, trg):
                beta = betas[d]
                if isinstance(beta, np.ndarray): # A probability per act for each direction of the network
                    beta = beta[route.edge_directions(src, trg)]
                elif not beta: # Skip directions with no transmission
                    return None
                disease_beta = beta.to_prob(self.t.dt) if isinstance(beta, ss.Rate) else beta
                beta_per_dt = route.net_beta(disease_beta=disease_beta, disease=self) # Compute beta for this network and timestep
                m = mult(Ellipsis, trg)
                return beta_per_dt*m if np.ndim(m) or m != 1 else beta_per_dt # Bit for bit as before, with no modifiers
            for source_uids, target_uids in route.transmit(rel_trans.raw, rel_sus.raw, edge_beta, self.trans_rng):
                new_cases.append(target_uids)
                sources.append(source_uids)

        # Handle everything else: mixing pools, environmental transmission, etc.
        elif isinstance(route, ss.networks.Route):
            # Mixing pools are unidirectional, only use the first beta value
            disease_beta = betas[0].to_prob(self.t.dt) if isinstance(betas[0], ss.Rate) else betas[0]
            m = mult(None, slice(None))
            if np.ndim(m) or m != 1:
                rel_sus = rel_sus.asnew(rel_sus.raw*m, copy=False)
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

    def set_congenital(self, uids, sources=None):
        """
        Set the prognoses of people infected before birth (over `ss.PrenatalNet`; see `set_outcomes()`); by default nothing happens to them

        A disease with outcomes at birth overrides this, e.g. `ssl.CongenitalDisease` (`import starsim.library as ssl`),
        which schedules a birth outcome (a stillbirth, a congenital infection) for each.
        """
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


class SIR(Disease):
    """
    An SIR model: susceptible, infected, and recovered, optionally with deaths from infection

    Declared from states and transitions (see `ss.Disease`), so it runs with any
    solver: as agents over networks, or as compartments. It is the textbook SIR: the
    infectious period is exponential, as any declared duration is, and nobody dies of
    the infection unless `p_death` is given. The keywords are Starsim 3's.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected (or a number of people)
        dur_inf (float/`ss.dur`/`ss.Dist`): the mean time that people are infected for (by default 6 years; a number is in years), as an exponential; give a distribution (e.g. `ss.lognorm_ex(mean=ss.years(6))`) for another shape
        p_death (float/`ss.bernoulli`): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    Starsim 3's `ss.SIR`, with its lognormal
    `dur_inf` and `p_death=0.01`, is `ssl.AgentSIR`, written as code.

    **Example**:

        sim = ss.Sim(diseases=ss.SIR(beta=0.1, dur_inf=ss.years(6)), networks='random', n_agents=5000)
        sim.run().plot()
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_inf   = ss.par(ss.years(6))
    p_death   = ss.par(0.0)
    S, I, R   = ss.states('S I R')
    infection = ss.transmission('S -> I', beta=beta)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.null, dur=dur_inf, p=p_death)
    standard_results = True

    def __init__(self, pars=None, **kwargs):
        default_beta = 'beta' not in kwargs and 'beta' not in (pars or {})
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        self.graph.pars['beta'].library_default = default_beta # Read as a probability per act on a network with acts, as in Starsim 3 (see ss.solvers.check_betas())
        return


class SEIR(Disease):
    """
    An SEIR model: as `ss.SIR`, with an exposed (latent) state before people become infectious

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected, who start exposed, in `E` (or a number of people); to start people infectious, give `init=dict(I=10)` instead
        dur_exp (float/`ss.dur`/`ss.Dist`): the mean time that people are exposed (latent) for (by default 0.5 years; a number is in years), as an exponential
        dur_inf (float/`ss.dur`/`ss.Dist`): the mean time that people are infectious for (by default 6 years), as an exponential
        p_death (float/`ss.bernoulli`): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    As in `ss.SIR`, each duration is exponential unless a distribution is given, and
    nobody dies of the infection unless `p_death` is given.

    Starsim 3's `ss.SEIR`, with its lognormal durations and `p_death=0.01`, is
    `ssl.AgentSEIR`, written as code.
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_exp   = ss.par(ss.years(0.5))
    dur_inf   = ss.par(ss.years(6))
    p_death   = ss.par(0.0)
    S, E, I, R = ss.states('S E I R')
    infection = ss.transmission('S -> E', beta=beta, source='I')
    latency   = ss.progression('E -> I', dur=dur_exp)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.null, dur=dur_inf, p=p_death)
    standard_results = True

    def __init__(self, pars=None, **kwargs):
        default_beta = 'beta' not in kwargs and 'beta' not in (pars or {})
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        self.graph.pars['beta'].library_default = default_beta # Read as a probability per act on a network with acts, as in Starsim 3 (see ss.solvers.check_betas())
        return



class SIS(Disease):
    """
    An SIS model: people recover from infection and are susceptible again

    Declared from states and transitions (see `ss.Disease`), so it runs with any solver.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected (or a number of people)
        dur_inf (float/`ss.dur`/`ss.Dist`): the mean time that people are infected for (by default 10 years; a number is in years), as an exponential unless a distribution is given
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    Starsim 3's `ss.SIS` had a lognormal `dur_inf` and a per-agent
    immunity level, boosted by each infection and waning over time (`waning=`,
    `imm_boost=`), which has no declared form; that model, written as code, is
    `ssl.AgentSIS`.
    """
    beta      = ss.par(ss.peryear(0.05))
    dur_inf   = ss.par(ss.years(10))
    S, I      = ss.states('S I')
    infection = ss.transmission('S -> I', beta=beta)
    recovery  = ss.progression('I -> S', dur=dur_inf)
    standard_results = True

    def __init__(self, pars=None, **kwargs):
        default_beta = 'beta' not in kwargs and 'beta' not in (pars or {})
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        self.graph.pars['beta'].library_default = default_beta # Read as a probability per act on a network with acts, as in Starsim 3 (see ss.solvers.check_betas())
        return


def library_kwargs(cls, kwargs):
    """
    The keywords of `ss.SIR`, `ss.SIS`, and `ss.SEIR`, as Starsim 3 gave them: 1% start infected unless
    said otherwise, as a parameter; a probability may be an `ss.bernoulli`; a number for a duration is in the
    unit of its default (e.g. `dur_inf=10` is 10 years, as the default of 6 years is); and a dict with a `type`
    is a distribution (e.g. `dur_inf=dict(type='expon', scale=ss.years(6))`)
    """
    if 'init' not in kwargs: # A parameter, so it can be set (sir.pars.init_prev = 0.05) or fitted like the others
        init_prev = kwargs.get('init_prev', 0.01)
        kwargs['init_prev'] = init_prev if isinstance(init_prev, ss.par) else ss.par(init_prev)
    if isinstance(kwargs.get('p_death'), ss.bernoulli):
        kwargs['p_death'] = kwargs['p_death'].pars.p
    for key in ['dur_inf', 'dur_exp']:
        val = kwargs.get(key)
        default = getattr(cls, key, None)
        if isinstance(default, ss.par) and sc.isnumber(val):
            pars = ss.Pars({key: sc.dcp(default.value)})
            pars.update({key: val})
            kwargs[key] = pars[key]
        elif isinstance(val, dict) and 'type' in val:
            kwargs[key] = ss.distributions.make_dist(val)
        elif isinstance(val, (list, dict)): # Starsim 3's values of its default lognormal, e.g. dur_inf=[10, 2]
            raise TypeError(f'{key}={val} set the values of the lognormal that was Starsim 3\'s default; {key} is now the mean of an exponential.\n'
                            f'     Give a number (in years) for that, or the distribution, e.g. {key}=ss.lognorm_ex(mean=ss.years(10), std=ss.years(2)); ssl.Agent{cls.__name__} is Starsim 3\'s model')
    return kwargs
