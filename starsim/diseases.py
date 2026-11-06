"""
Diseases: the base class (`ss.Disease`), diseases with transmission written as code (`ss.AgentDisease`), and the textbook models (`ss.SIR`, `ss.SIS`, `ss.SEIR`)

`ss.Disease` exists so that one class can hold a disease declared from states and transitions, written as code, or both. A declared disease runs with any solver; `ss.AgentDisease` and other code that works on individual agents runs as agents.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

ss_int = ss.dtypes.int



class BuiltAfterInit(type):
    """ Build what a disease declares in its `__init__()` (`define_states()`, `define_transitions()`) once that has finished """
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls) # Called self, as a constructor of it: see ss.utils.assigned_name()
        self.__init__(*args, **kwargs)
        self.build_graph()
        return self


class Disease(ss.Module, metaclass=BuiltAfterInit):
    """
    A disease: states people are in and the transitions between them (declared), code, or both

    Declared from its states and transitions, no methods need to be defined, and the
    disease can be run with any solver (see `ss.Sim`):

        S, I, R = ss.states(3)
        sir = ss.Disease(
            infection = S >> I @ 0.3 * I,
            recovery  = ss.Transition(I, R, dur=10),
            init_prev = 0.01,
        )

    The same disease can be declared in a class body (the class form), which can also
    add behavior by defining methods; `ss.SIR`, `ss.SIS`, and `ss.SEIR` are written
    this way.

    A disease can also be written as code: a subclass that declares
    nothing, defines its per-agent arrays (`define_states()`), and changes them in its
    own `step_state()` and `step()`; it runs as agents. One class can mix the two, e.g. a
    declared natural history and a coded array that its `step()` updates after
    `super().step()` has taken the declared transitions. Each lifecycle method runs the
    declared part through the disease's engine (which, for a disease that declares
    nothing, has nothing to run), so an override calls `super()` as for any module.
    `ss.AgentDisease` adds transmission written as code.

    Note: CRN safe as agents.

    Args:
        args (list): transitions and outcomes, named from the variables they are assigned to (one by one, or in a list)
        pars (dict): parameter values by name
        transitions (list): transitions, as a list (for longer models)
        results (list): extra results: sums and ratios of states
        outcomes (list): `ss.outcome` objects
        init_prev (float): the initial prevalence, seeded through the transmission (by default 0.01, unless a state is given an initial value)
        init (dict): the share of people who start in each state, e.g. `dict(S=0.90, I=0.01, R=0.09)`, and, for a state that is not people, its amount, which can be read from the people (e.g. `dict(Sv=m*N)`, with `N = (S + I).n`); each value can be an `ss.by` or a function of the sim (in the class form, `init = dict(I=0.01)` in the class body; likewise `init_n` and `init_prev`)
        init_n (dict): instead, the number of people who start in each state, e.g. `dict(I=1)` for one person
        default_state (str): the state that everyone not put elsewhere starts in, and is born into; by default, the one named `S` or `susceptible`, else the first state of people made (in the class form, `default_state = 'N'` in the class body)
        strains (Strains): its strains (`ss.Strains`), which share its states; then `dis.strains`
        stratify (dict/list): as `stratify()`, e.g. `dict(age={0: 0.3, 15: 0.6, 65: 0.1})`; a list of dicts for one call each
        solver (str): its own solver in a sim of agents, instead of the sim's: `'abm-midstep'` (by steps, with those who arrive in a state during a step facing its rates for the rest of it), `'abm-wholestep'` (by steps, with rates over whole steps only, for those in the state at the start of the step, as code with `ss.bernoulli()` takes them: faster, and a little less exact at a coarse step), `'abm-events'` (event by event, in continuous time), or `'abm'` (its `suggested_solver`, whatever the sim's); a class can instead suggest one, `suggested_solver = 'abm-wholestep'` in the class body, which the sim's `solver='abm'` (or `'auto'`) follows and any other agent solver overrides (see `ss.solvers.agent_solver()`)
        kwargs (dict): one keyword per state, transition, parameter, result, function, or modifier (`ss.modify()`, of its own quantities, e.g. `fading=ss.modify('rel_trans', multiply=0.5, where=I)`); also `name`, `label`, and `dt`

    The results are the number in each state (`I`), the flow along each transition by its
    name (`infection`, with its cumulative total as `res.infection.cum`), and the results and
    outcomes declared (e.g. `active=(E + I).frac`). States marked infected (`ss.State(infected=True)`,
    or `ss.states(2, infected=True)`) also give the results `infected`, the number in them, and
    `prevalence`, their share of everyone in the disease's states, unless it defines its own.

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

        # Written as code: a disease that is not transmitted
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
    graph = None # Its declared states and transitions (`ss.graph.Graph`); None for a disease written as code
    engine = None # Runs the declared part (see ss.solvers.make_engine()); created when the sim is initialized
    infection_log = None # An InfectionLog, if the sim has an ss.infection_log analyzer
    suggested_solver = None # For a declared module, how it suggests running as agents, one of ss.solvers.agent_solvers (e.g. 'abm-wholestep'), which the sim's solver='abm' (or 'auto') follows: see ss.solvers.agent_solver()
    code_pars = () # Parameters of a declared module that its own code reads rather than its graph (e.g. ss.Pregnancy's fertility rates)
    allow_modifiers = () # For a disease written as code: which of rel_sus, rel_trans, and its parameters its own code reads, with current_value(), so ss.modify() of them is applied (an ss.AgentDisease with a beta reads both)

    _infection_methods = ['set_prognoses', 'transmit', 'transmissibility', 'susceptibility', 'route_betas', 'set_outcomes', 'finalize_infections', 'compute_transmission', 'transmit_route', 'set_congenital'] # Methods of ss.AgentDisease's transmission, which a declared transition never calls

    def __init_subclass__(cls, **kwargs):
        """ A class that declares states or transitions takes its transitions through its engine, which never calls the methods of ss.AgentDisease's transmission, so overriding one (e.g. set_prognoses() on a subclass of ss.SIS) would be silently ignored """
        super().__init_subclass__(**kwargs)
        overridden = [name for name in cls._infection_methods if name in vars(cls) and name != 'transmit'] # A declared disease's own transmit() is called for its transmissions (see transmit())
        infection = any(b.__name__ == 'AgentDisease' and b.__module__ == __name__ for b in cls.__mro__)
        if overridden and not infection and ss.transitions.class_members(cls):
            base = next((f'ssl.Agent{b.__name__}' for b in cls.__mro__ if b.__module__ == __name__ and b.__name__ in ['SIR', 'SIS', 'SEIR']), 'ss.AgentDisease')
            errormsg = (f'{cls.__name__} declares states or transitions and defines {sc.strjoin(overridden)}, which a declared transition never calls, so it would be silently ignored.\n'
                        f'     To write a declared transmission as code, override transmit(self, transition), which gives who is infected (see ss.Disease.transmit()); a disease written as code from the start subclasses {base} (import starsim.library as ssl).\n'
                        f'     If this was written for Starsim 3, see https://docs.starsim.org/v3v4 for help migrating.')
            raise ss.CapabilityError(errormsg)
        rebound = [name for name in dict.fromkeys(k for base in cls.__mro__[1:] for k,v in vars(base).items() if isinstance(v, ss.par)) if name in vars(cls) and not isinstance(vars(cls)[name], (ss.par, ss.expressions.Expr))]
        if rebound: # A declared parameter rebound to something else (a method, a number) is dropped from the model, and its transitions keep the base class's parameter, so it would be silently ignored
            name = rebound[0]
            how = (f'give the parameter a function as its value instead, e.g. self.pars.update({name}=self.my_{name}) in __init__(), where my_{name}(sim, uids) returns the value for those agents' if callable(vars(cls)[name]) else
                   f'to change its default, give the value in __init__(), e.g. super().__init__(**({{"{name}": {vars(cls)[name]!r}}} | kwargs))')
            errormsg = f'{cls.__name__} defines {sc.strjoin(rebound)}, a parameter of {cls.__mro__[1].__name__}, as {"a method" if callable(vars(cls)[name]) else "a value"}, which its transitions never read; {how}'
            raise ss.CapabilityError(errormsg)
        infect = vars(cls).get('infect')
        if infect is not None and infection and len(inspect.signature(infect).parameters) == 1: # Starsim 3's transmission pass, infect(self), which is transmit() now: it would never be called
            errormsg = f'{cls.__name__} defines infect(self), the transmission pass, which is called transmit() now (infect(uids) infects people from code); rename it to transmit(), and infect_route() to transmit_route()'
            raise ss.CapabilityError(errormsg)
        return

    def __init__(self, *args, pars=None, strains=None, stratify=None, solver=None, _names=None, **kwargs):
        super().__init__()
        self.strains = strains # Its ss.Strains, if it has strains: set now, so a subclass's __init__() can read it before the disease is built (see ss.strains.attach())
        if solver not in [None, 'abm'] + ss.solvers.agent_solvers:
            raise ValueError(f'solver must be one of {sc.strjoin(ss.solvers.agent_solvers)} (a module\'s own solver, as agents), "abm" (its suggested one), or None to follow the sim\'s, not {solver!r}')
        self.solver = solver # Its own solver, as agents (see ss.solvers.agent_solver()); None follows the sim's
        self.callbacks = [] # Functions called with the agents an event happened to: see add_callback()
        if len(args) and (args[0] is None or isinstance(args[0], dict)): # Parameters can also be the first argument
            pars, args = args[0], args[1:]
        args = tuple(x for arg in args for x in (arg if isinstance(arg, (list, tuple)) else [arg])) # A list of transitions and outcomes is the same as giving them one by one
        members = ss.transitions.class_members(type(self))
        if members and all(isinstance(val, ss.modify) for val in members.values()):
            errormsg = f'{sc.strjoin(members)} of {type(self).__name__} is a modifier, which is a member of a declared disease; for a disease written as code, give it to the sim, in connectors='
            raise ss.CapabilityError(errormsg)
        given = args or members or self.has_structure(kwargs) # Its structure; a parameter's value (e.g. with bounds) is not
        later = strains is not None or stratify is not None or any(key in kwargs for key in ['init', 'init_n', 'init_prev', 'default_state']) or any(isinstance(v, ss.par) for v in kwargs.values()) # Only for a declared disease, so its states and transitions follow, from define_states() and define_transitions()
        if given or later:
            names = (_names or ss.utils.caller_names(1, skip=BuiltAfterInit.__call__)) if given else {} # The names of the variables of the code that made it; or, made by the sim from the transitions it was given, of the code that made the sim
            if not given: # A parameter's value given by keyword (e.g. with bounds) sets the parameter of that name that the transitions declared next use, as pars= does
                pars = sc.mergedicts(pars, {key:kwargs.pop(key) for key in list(kwargs) if isinstance(kwargs[key], ss.par)})
            self._declared = sc.obj(args=list(args), members=members, pars=pars, kwargs=kwargs, names=names, strains=strains, stratify=stratify, code_pars=[])
            if given: # Built now, so a subclass's __init__() can use it (e.g. self.pars) after super().__init__()
                self.build_graph()
            else: # Built once the subclass's __init__() has declared the rest
                self.set_metadata(**{key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs})
        else: # Written as code: its own define_pars() and define_states() follow; or declared by define_states() and define_transitions()
            self.update_pars(pars, **kwargs)
        return

    def __setattr__(self, attr, value):
        """ A state or transition assigned once the disease is built would never be part of it, so it raises (see `declare()`) """
        if isinstance(value, (ss.State, ss.Transition)) and self.__dict__.get('graph') is not None and value is not getattr(self, attr, None):
            errormsg = (f'"{self.name}" was built when it was made, so the {type(value).__name__.lower()} "{attr}" would not be part of it. Declare its states and transitions '
                        f'in its __init__() or class body: for a variant of a disease, subclass it and call self.define_transitions() in __init__() after super().__init__()')
            raise ss.CapabilityError(errormsg)
        return super().__setattr__(attr, value)

    @staticmethod
    def has_structure(kwargs):
        """ Whether these keywords declare part of a model, not just a parameter's value """
        declared = (ss.State, ss.Transition, ss.outcome, ss.expressions.Expr, ss.transitions.FlowSum, ss.modify)
        return any((isinstance(v, declared) and not isinstance(v, ss.par)) or key in ss.graph.Graph.grouped for key,v in kwargs.items())

    def build_graph(self):
        """ Build the graph from what was declared: when the disease is made, or for one declared in its __init__(), after that (see `BuiltAfterInit`) """
        d = self.__dict__.pop('_declared', None)
        if d is None:
            return
        if not (d.args or d.members or d.strains is not None or d.stratify is not None or self.has_structure(d.kwargs)): # Nothing was declared, only parameters given: written as code
            return self.update_pars(d.pars, **d.kwargs)
        if isinstance(d.strains, ss.Strains): # An outcome for each strain carried, before the transitions are built
            ss.strains.prepare(d.strains, [(x.name, x) for x in d.args + sc.tolist(d.kwargs.get('transitions'))] + list(d.kwargs.items()) + list(d.members.items()))
        self.init_graph(d.args, d.members, d.pars, d.kwargs, names=d.names)
        self.add_code_pars(d.code_pars)
        if d.strains is not None:
            ss.strains.attach(self, d.strains)
        for call in sc.tolist(d.stratify):
            self.stratify(**call)
        return

    def declare(self, args=(), kwargs=None, names=None, overwrite=False):
        """ Add declared members (states, transitions, results), to be built with the rest; see `define_transitions()`

        Args:
            args (list): the members to add, without names
            kwargs (dict): the members to add, by name
            names (dict): the names of the variables the members were assigned to, if known
            overwrite (bool): whether a member can replace one of the same name
        """
        if self.graph is not None:
            errormsg = (f'"{self.name}" was built when it was made, so nothing more can be declared in it now: its states, transitions, and results must be declared before that, '
                        f'in its __init__() or class body (or given to ss.Disease() with the rest). A plain ss.Result can still be added, as to any module')
            raise ss.CapabilityError(errormsg)
        d = self.__dict__.get('_declared')
        if d is None: # The first declaration of a disease that called super().__init__() with none
            d = self._declared = sc.obj(args=[], members=ss.transitions.class_members(type(self)), pars=None, kwargs={}, names={}, strains=None, stratify=None, code_pars=[])
        names = names or {}
        kwargs = kwargs or {}
        for obj in args: # Positional members, named by name= or by the variable they were assigned to
            name = getattr(obj, 'name', None) or names.get(id(obj))
            if name is None:
                d.args.append(obj)
            else:
                kwargs = {name:obj} | kwargs # By name, so it can replace one
        for name,obj in kwargs.items():
            if not overwrite and (name in d.kwargs or name in d.members):
                raise ValueError(f'"{name}" is already declared in "{self.name}"; names must be unique within a module (or give overwrite=True to replace it)')
            d.kwargs[name] = obj
            self.setattribute(name, obj) # So the declarations that follow can use it, e.g. self.S >> self.I
        d.names.update(names)
        return

    def define_states(self, *args, overwrite=False, **kwargs):
        """
        Define its states: declared states (`ss.State`), or for a disease written as code, arrays (see `ss.Module.define_states()`)

        Declared states are given in its `__init__()`, as for `define_transitions()`, and each is an attribute at once (`self.S`).

        Args:
            args (list): the states (`ss.State('S')`, named), or lists of them; or arrays (`ss.BoolState`, ...)
            overwrite (bool): replace a state of the same name, rather than raising
            kwargs (dict): states by name (`S=ss.State(label='Susceptible')`); or for arrays, aliases
        """
        flat = [x for arg in args for x in (arg if isinstance(arg, list) and all(isinstance(y, ss.State) for y in arg) else [arg])]
        states = [x for x in flat if isinstance(x, ss.State)]
        named = {k:v for k,v in kwargs.items() if isinstance(v, ss.State)}
        if not states and not named: # Arrays, for a disease written as code
            return super().define_states(*args, overwrite=overwrite, **kwargs)
        if len(states) + len(named) < len(flat) + len(kwargs):
            errormsg = f'define_states() of "{self.name}" takes declared states (ss.State) or arrays (ss.BoolState, ...), not both in one call'
            raise TypeError(errormsg)
        unnamed = [st for st in states if st.name is None]
        if unnamed:
            raise ValueError(f'a state given to define_states() of "{self.name}" has no name: give it one, e.g. ss.State("S"), or give it by keyword, e.g. define_states(S=ss.State())')
        self.declare(states, named, overwrite=overwrite)
        return

    def define_transitions(self, *args, overwrite=False, **kwargs):
        """
        Define its transitions, in its `__init__()`, after its states

        For a declared disease whose structure depends on its arguments, so that it cannot be written
        in a class body: the states (`define_states()`), transitions, and results (`define_results()`)
        are built together once its `__init__()` has finished. Each is an attribute at once (`self.infection`).

        Args:
            args (list): transitions and outcomes, or lists of them, named by `name=` or by the variable they were assigned to
            overwrite (bool): replace a transition of the same name (e.g. one whose rate is changed), rather than raising
            kwargs (dict): transitions by name

        Examples:
            ```python
            class SIRS(ss.Disease):
                def __init__(self, waning=True, **kwargs):
                    super().__init__(init_prev=0.01, **kwargs)
                    self.define_states(ss.states('S I R'))
                    beta = ss.par(0.3)
                    self.define_transitions(
                        infection = self.S >> self.I @ beta*self.I,
                        recovery  = self.I >> self.R @ ss.days(10),
                    )
                    if waning:
                        self.define_transitions(waning=self.R >> self.S @ ss.days(100))
            ```
        """
        flat = [x for arg in args for x in (arg if isinstance(arg, (list, tuple)) else [arg])]
        self.declare(flat, kwargs, names=ss.utils.caller_names(1), overwrite=overwrite)
        return

    def define_results(self, *args, **kwargs):
        """
        Define results: for a declared disease, sums and ratios of states by name (e.g. `prevalence=I/(S + I + R)`, or `active=(E + I).frac`); otherwise `ss.Result` objects (see `ss.Module.define_results()`)

        Args:
            args (list): the results, as `ss.Result` objects or expressions
            kwargs (dict): the results of a declared disease, by name
        """
        declared = (ss.expressions.Expr, ss.outcome, ss.transitions.FlowSum)
        if not kwargs and not any(isinstance(arg, declared) for arg in args):
            return super().define_results(*args)
        if self.graph is not None: # Too late to declare (its states are no longer expressions, so the check below would blame the wrong thing)
            return self.declare(args, kwargs)
        if not all(isinstance(arg, declared) for arg in list(args) + list(kwargs.values())):
            errormsg = f'define_results() of "{self.name}" takes expressions of states (e.g. prevalence=I/(S + I + R)) or ss.Result objects, not both in one call'
            raise TypeError(errormsg)
        names = ss.utils.caller_names(1)
        for arg in args:
            if names.get(id(arg)) is None:
                errormsg = f'a result given to define_results() of "{self.name}" has no name: give it by keyword, e.g. define_results(prevalence=I/N)'
                raise ValueError(errormsg)
        self.declare(args, kwargs, names=names)
        return

    def define_pars(self, inherit=True, **kwargs):
        """
        Create or merge parameters; see `ss.Module.define_pars()`

        For a declared disease, a parameter defined here that is not one of its graph's (e.g. by a subclass, after
        `super().__init__()`, for its own code to read) is added to `code_pars`, so the sim does not reject it as one
        that nothing reads. A parameter set any other way (e.g. a misspelling in `pars=`) is still rejected.

        Args:
            inherit (bool): whether to add to the parameters that exist, rather than replace them
            kwargs (dict): the parameters, by name
        """
        out = super().define_pars(inherit=inherit, **kwargs)
        if '_declared' in vars(self): # Being declared: sorted once it is built, so more can still be declared after this
            self._declared.code_pars += list(kwargs)
        elif self.graph is not None:
            self.add_code_pars(kwargs)
        return out

    def add_code_pars(self, keys):
        """ Parameters its own code reads, not its graph's: see `define_pars()`

        Args:
            keys (list): the names of the parameters
        """
        self.code_pars = list(self.code_pars) + [k for k in keys if k not in self.graph.pars and k not in self.code_pars]
        return

    @staticmethod
    def reuse(args, members, names=None):
        """ States and transitions that already belong to another disease are copied (together, so they refer to each other, and with their parameters), so one set can build several independent diseases """
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
            elif isinstance(obj, (ss.State, ss.Transition, ss.expressions.Expr, ss.outcome, ss.transitions.FlowSum)) and depth < 6:
                if getattr(obj, 'module', None) is not None and isinstance(obj, (ss.State, ss.Transition)):
                    owners[id(obj.module)] = obj.module
                if isinstance(obj, ss.Transition):
                    ends = [st for st in (obj.source or []) + (obj.dest or []) if isinstance(st, ss.State)]
                    endpoints.update(id(st) for st in ends)
                    for st in ends:
                        ss.graph.Graph.check_read(st)
                elif isinstance(obj, ss.State) and depth == 0:
                    endpoints.add(id(obj))
                elif isinstance(obj, ss.expressions.Count):
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
        grouped = [obj for key in ss.graph.Graph.grouped for obj in sc.tolist(members.pop(key, None))]
        positional = []
        for obj in list(args) + grouped: # Named first, as keywords, since a copy is not bound to the variable
            name = getattr(obj, 'name', None) or (None if isinstance(obj, ss.State) else (names or {}).get(id(obj)))
            if name is None or isinstance(obj, ss.State):
                positional.append(obj)
            else:
                members[name] = obj
        args = positional
        memo = {key:None for key in owners} | {id(p.module):None for p in pars.values() if p.module is not None} # The copies belong to no module yet
        memo |= {key:p for key,p in pars.items() if p.module is None} | others # Parameters that came with the copied transitions are copied too, so each disease has its own; one not yet registered stays shared, e.g. one R0 for several models
        return sc.dcp((args, members), memo=memo)

    def check_par_members(self, args, members, kwargs):
        """ A parameter given by keyword under the name it already has (name=), and read by a transition, as if it were a result (e.g. eir=eir) is an error: a result needs a name of its own """
        for key,val in kwargs.items():
            if isinstance(val, ss.par) and val.name_given and val.name == key:
                exprs = [e for obj in list(args) + list(members.values()) + list(kwargs.values()) if isinstance(obj, ss.Transition) for e in [obj.rate, obj.dur] + sc.tolist(obj.p)]
                exprs += [obj for obj in list(members.values()) + list(kwargs.values()) if isinstance(obj, ss.expressions.Expr) and obj is not val]
                if any(part is val for part in ss.expressions.nodes(exprs)):
                    errormsg = (f'"{key}" of "{self.name}" is given the parameter "{key}" itself, which its transitions already read, so it is not a result: a result needs a name of its own. '
                                f'For a result of a parameter, write an expression of it under another name, e.g. {key}_now = 1*{key}')
                    raise ValueError(errormsg)
        return

    def init_graph(self, args, members, pars, kwargs, names=None):
        """ Build the graph of declared states and transitions; see `ss.graph.Graph` """
        meta = {key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs}
        declared = (ss.State, ss.Transition, ss.outcome, ss.expressions.Expr, ss.transitions.FlowSum, ss.dimensions.Transfer)
        timepars = {key:kwargs.pop(key) for key in ss.Timeline.time_args if key in kwargs and not isinstance(kwargs[key], declared)} # A transition can be called "stop"
        pars = sc.mergedicts(pars)
        for key in ['init_prev', 'init', 'init_n']: # The initial conditions can also be given in pars=, as the parameters are
            if key in pars:
                kwargs[key] = pars.pop(key)
        for key in ['init_prev', 'init', 'init_n']: # Or in the class body, e.g. init = dict(I=0.01), as for default_state below
            given = getattr(type(self), key, None)
            if key not in kwargs and (isinstance(given, dict) or sc.isnumber(given)):
                kwargs[key] = dict(given) if isinstance(given, dict) else given
        init_prev = kwargs.pop('init_prev', None)
        if sc.isnumber(init_prev): # A parameter, so it can be set (sir.pars.init_prev = 0.05) or fitted like the others
            init_prev = ss.par(init_prev)
        init = kwargs.pop('init', None)
        init_n = kwargs.pop('init_n', None)
        default_state = kwargs.pop('default_state') if isinstance(kwargs.get('default_state'), str) else getattr(type(self), 'default_state', None) # A state named "default_state" can still be declared by keyword; in the class form, default_state = 'N' in the class body
        default_state = default_state if isinstance(default_state, str) else None
        self.set_metadata(**meta)
        self.t.update(**timepars)
        holds_members = lambda leaf: any(v is m for v in ss.graph.inner_pars(leaf.raw) for m in members.values()) # A value holding the class's parameters, copied with them: by level (ss.by(sex=dict(female=p_f, male=p_m))), a distribution (ss.constant(dur_inc)), or one that varies in time
        def leaves(expr): # Including those inside a parameter that holds an expression, as a dwell time written as an expression is held (e.g. dur=d*ss.days(1), with d = ss.par(ss.lognormal(log_mean=..., log_std=...)))
            for leaf in expr.leaves:
                yield leaf
                if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.expressions.Expr):
                    yield from leaves(leaf.raw)
        holds_expr = lambda leaf: isinstance(leaf.raw, ss.expressions.Expr) and any(inner is m for inner in leaves(leaf.raw) for m in members.values()) # One holding the class's parameters, copied with them
        memo = {id(leaf):leaf for val in members.values() if isinstance(val, ss.Transition) for expr in [val.rate, val.dur, getattr(val, 'p_die', None), getattr(val, 'p_stay', None)] + sc.tolist(getattr(val, 'p', None)) if isinstance(expr, ss.expressions.Expr) for leaf in leaves(expr) if isinstance(leaf, ss.par) and not any(leaf is m for m in members.values()) and not holds_members(leaf) and not holds_expr(leaf)}
        members = sc.dcp(members, memo=memo) # Copy class members (together, so they still refer to each other) so instances do not share them; a parameter defined outside the class (rate = ss.par(...) at module level) stays shared, as in the operator form, so it keeps its name
        for tr in [val for val in members.values() if isinstance(val, ss.Transition)]: # A sibling as an event (vax = ss.Transition(R, V, on=recovery)) is the copy
            tr.where = ss.conditions.rebind(tr.where, memo)
            if tr.trigger is not None and (x := ss.conditions.rebind(tr.trigger.x, memo)) is not tr.trigger.x:
                delay, tr.trigger = tr.trigger.delay, ss.on(x)
                tr.trigger.delay = delay
        kwargs = {k:ss.expressions.rebind(v, memo) for k,v in kwargs.items()} # A result given here that reads the class's states reads the copies
        init, init_n = [{k:ss.expressions.rebind(v, memo) for k,v in d.items()} if isinstance(d, dict) else d for d in [init, init_n]] # Likewise a starting value, e.g. init=dict(S=1 - (A - age)/(k*age + A)) with the class's parameters
        for key in [key for key,val in kwargs.items() if isinstance(val, ss.par) and isinstance(members.get(key), ss.par)]: # e.g. ss.SIS(beta=ss.par(0.1, bounds=[0.05, 1])): the class's parameter, which its transitions use, takes the value and bounds
            members[key].merge(kwargs.pop(key))
        self.check_par_members(args, members, kwargs)
        for key,val in kwargs.items(): # By keyword, an expression is a result, so one with a value for each person cannot be: it would be dropped (in the class body, it can be a helper that other members use)
            if isinstance(val, ss.expressions.Expr) and not isinstance(val, (ss.par, ss.State, ss.expressions.attr, ss.Transition, ss.outcome)) and ss.expressions.per_person_in(val):
                errormsg = (f'{key}={val} was given to "{self.name}" as a result, but it has a value for each person (it reads a property of people, a clock, or a value by person), so it cannot be one. '
                            f'For a share of the people in some states, write e.g. {key}=(E + I).frac')
                raise ValueError(errormsg)
        members = sc.mergedicts(members, kwargs)
        args, members = self.reuse(args, members, names)
        self.graph = ss.graph.Graph(self, args, members, init_prev=init_prev, init=init, init_n=init_n, default_state=default_state, names=names)
        for name,obj in list(self.graph.states.items()) + list(self.graph.transitions.items()) + list(self.graph.attrs.items()):
            self.setattribute(name, obj) # e.g. sir.S, sir.infection, hiv.cd4
        for tr in list(self.graph.transitions.values()) + list(self.graph.transfers.values()):
            tr.module = self
        for name,tf in self.graph.transfers.items(): # e.g. sir.vaccination, so it can be the target of ss.modify()
            self.setattribute(name, tf)
        for key in ['rel_sus', 'rel_trans']: # So they can be the target of ss.modify() before the sim exists
            self.setattribute(key, ss.connectors.Target(self, key))
        self.define_pars(**self.graph.pars) # The parameters themselves, so e.g. sir.pars.beta can be the target of ss.modify()
        for name,par in self.pars.items(): # So sir.beta is this module's parameter, not the one of the class it was copied from
            if isinstance(par, ss.par) and isinstance(getattr(type(self), name, par), ss.par):
                self.setattribute(name, par)
        for name,modifier in self.graph.modifiers.items():
            modifier.own(self, name)
        for key in sc.mergedicts(pars):
            if key not in self.pars:
                raise sc.KeyNotFoundError(f'"{self.name}" has no parameter named "{key}". Its parameters are {sc.strjoin(self.pars.keys())}{ss.expressions.did_you_mean(key, self.pars.keys())}{ss.errors.starsim3(keys=key, call=f"ss.{type(self).__name__}")}')
        self.pars.update(pars) # A declared parameter keeps its bounds and name: see ss.Pars
        return

    def stratify(self, inplace=True, **kwargs):
        """
        Stratify the disease by a property of the people in it (age, risk, place, ...), in place, and return it

        Args:
            fractions (list): the share of the population in each level, unless the levels are a dict of them (if not given: from a named population or geography; for age groups, the world's ages; for a boolean dimension, or one that people are moved along, everyone at the first level; otherwise equal)
            adjust (dict): multipliers on named transitions, one per level, e.g. `{'recovery': [1.5, 1.0, 0.5]}`: on its rate (a duration is divided), or on its probability for a branch chosen with p=
            kind (str): 'age', 'space', or 'plain'; if not given, from the value: 'space' for an `ss.geo.geography` or with `mobility=`, 'age' for `age=`, and otherwise 'plain' (whatever the name); strains are `strains=ss.Strains(...)` on the disease
            states (list): the states the dimension applies to (default: every state that is people)
            where (dict): the groups the dimension applies to, e.g. `dict(sex='female')`; people in other groups have no level of it, so selecting a level of it leaves them out (default: every group)
            mobility (gravity/radiation/Mobility/array): for a place (e.g. `patch=`), the movement between its levels: `ss.geo.gravity()`, `ss.geo.radiation()`, `ss.geo.Mobility()`, or a matrix
            inplace (bool): if False, return a stratified copy and leave this disease unchanged
            kwargs (dict): the dimension and its levels (bin edges or labels), e.g. `age=[0, 15, 65]`;
                an `ss.transfer` passed as a keyword is added under that name

        Examples:
            ```python
            sir.stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
            sir.stratify(age={0: 0.3, 15: 0.6, 65: 0.1}) # With the share of people in each age group
            sir.stratify(risk=['low', 'high'], fractions=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
            sir.stratify(parity=['0', '1-2', '3+'], where=dict(sex='female')) # Only women have a parity
            ```
        """
        if self.graph is None:
            errormsg = f'"{self.name}" is written as code ({type(self).__name__}), so it has no states for stratify() to divide; its own code can read a property of people instead, or a sim stratified with sim.stratify() gives people the levels (people.<name>) for a network or a declared module to use'
            raise ValueError(errormsg)
        mod = self if inplace else self.copy()
        mod.graph.stratify(**kwargs)
        for strat in mod.graph.strats: # So a dimension can be used in a condition, e.g. where=~sir.vaccinated
            if not hasattr(mod, strat.name):
                cond = ss.conditions.Cond('dim', strat.name)
                cond.owner = mod # So a delivery whose condition reads it is for this disease (see ss.conditions.state_refs())
                mod.setattribute(strat.name, cond)
        return mod

    @property
    def abm(self):
        """ Whether it runs as agents """
        return self.engine is not None and self.engine.agents

    @property
    def transitions(self):
        """ The table of declared transitions; printing it also shows the matrix of sources by destinations """
        return self.graph.transitions

    @property
    def states(self):
        """ Its declared states (`ss.State`) by name, e.g. `'second_line' in tb.states` or `[s for s in tb.states.values() if s.people]`; empty for a disease written as code. (The arrays behind them are `state_dict`.) """
        return sc.obj(self.graph.states) if self.graph is not None else sc.obj()

    @property
    def strats(self):
        """ Its stratifications by name, e.g. `tb.strats.age.edges` or `tb.strats.risk.labels` (as a condition, a level is `tb.age`) """
        return sc.obj({strat.name:strat for strat in self.graph.strats}) if self.graph is not None else sc.obj()

    def describe(self, output=False, out=None):
        """
        Print the resolved model, with every guessed default marked

        Args:
            output (bool): if True, return the text rather than printing it
            out (`ss.display.Description`): if given, add the description to it and return it
        """
        desc = ss.display.Description() if out is None else out
        if self.graph is not None:
            ss.display.describe_graph(self.graph, self.sim, out=desc)
            if self.strains is not None:
                self.strains.describe(self.graph, out=desc)
        else: # Written as code: its class, arrays, and parameters
            desc.section(f"{type(self).__name__} '{self.name}'", f'written as code, with {len(self.state_list)} arrays')
            desc.add('arrays', ', '.join(s.name for s in self.state_list))
            desc.add('parameters', ', '.join(self.pars.keys()))
            if self.strains is not None:
                desc.add('strains', f'{sc.strjoin(self.strains.names)}, one carried at a time')
        if out is not None:
            return desc
        return str(desc) if output else desc.show()

    def equations(self, output=False):
        """
        Print the ODE system of the declared transitions, with symbolic parameter names

        Args:
            output (bool): if True, return the text rather than printing it
        """
        text = '\n'.join(ss.display.equations(self.graph, self.engine))
        return text if output else print(text)

    def diagram(self):
        """
        The state-transition diagram, as a Mermaid flowchart (https://mermaid.js.org), which a notebook draws (see `ss.display.Diagram`)

        People states are rounded and counts slanted; transmissions are dashed; each
        arrow is labeled with its transition's name and value (`recovery: 10 days`), and
        the share that takes it for a transition with `p=`. `print(sir.diagram())` gives the text.

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=ss.days(10)))
            sir.diagram()
            ```
        """
        return ss.display.Diagram('\n'.join(['flowchart LR'] + ss.display.diagram(self.graph)))

    def R0(self, matrix=False):
        """
        The basic reproduction number, from the next-generation matrix at the disease-free state

        By definition it is in a population with no infection and no immunity: the people infected at the start are left
        out, and everyone else is taken to be in the first state (e.g. S), each group keeping its size, whoever starts
        immune (`init=dict(R=0.9)`). So where transmission is density-dependent (`beta*I.n`), R0 is for the population less
        the people seeded, which matters only when many are (half the people seeded halves it). If no transition links some states to the first (a chain of states for each host
        species), their people are in the state their own chain starts from. People filled by a constant inflow (births of a
        number a day) and counts that are not people (mosquitoes) are at their equilibrium, which the inflow and the rates out
        of them set (Λ/μ). The reproduction number at the start, with any immunity, is in the line a run prints as it starts,
        when it differs (`R0 = 15.2 (R = 1.52 at the start, with 10% susceptible)`).

        The model is laid out as an ODE: if the disease is in a sim, a copy of that sim
        is made with `solver='ode'`, with its mixing and stratification; otherwise a sim with
        the disease alone, under homogeneous mixing, with the default number of people. Values that vary in time
        (a seasonal β) are taken at the start, so this is R0 at the start time.
        A sim runs a copy of the disease it is given, so the R0 of a sim's model is `sim.diseases.<name>.R0()`;
        this matters where a count is a number rather than a share of the people (mosquitoes per person).

        This is the largest eigenvalue of the next-generation matrix. Where infection
        passes through a vector (people infect mosquitoes, which infect people), each
        passage is a generation, so R0 is the square root of the classical Ross–Macdonald
        value. Shedding into a reservoir that is a count (such as water) is not an infection,
        so there R0 is the number of people infected through the reservoir by one person.

        Args:
            matrix (bool): if True, return the next-generation matrix (over the infected compartments of every group) rather than its largest eigenvalue

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10))
            sir.R0() # 3.0
            ```
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

    def Rt(self):
        """
        The reproduction number over the course of a run, as a result: the number one infectious person infects at each time point

        This is R0 at the state of the run at each time point: the largest eigenvalue of the next-generation matrix with the
        people in each state of each group as they were then (so fewer when some are immune), and the rates as they were then
        (so lower while an `ss.modify()` halves β). It falls below 1 as an outbreak peaks. As in `R0()`, the model is laid out
        as an ODE, a copy of the sim with `solver='ode'`, stepped alongside the run; a run as agents is taken by the counts in
        each state of each group. A state with several compartments (the stages of a dwell time) is split among them as in the
        copy; and the copy is what an intervention triggered by the state of the run sees.

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), name='sir')
            sim = ss.Sim(sir, total_pop=10_000, stop=150).run()
            sim.diseases.sir.Rt().plot() # From 3 at the start, below 1 after the peak
            ```
        """
        sim = self.sim
        if sim is None or not sim.results_ready:
            errormsg = f'"{self.name}" has no run to take R over time from: run the sim first, e.g. sim.run(), then sim.diseases.{self.name}.Rt() (for the start alone, see R0())'
            raise RuntimeError(errormsg)
        copy = sim.new(solver='ode', verbose=0).init()
        engine = copy.diseases[self.name].engine
        results = sim.results[self.name]
        stored = {} # The people in each state of each group at each kept time point
        group_ti = None
        for name in engine.state_comps.keys():
            res = results.get(name) if name in results else None
            if res is None:
                continue
            elif getattr(res, 'strata', None) is not None:
                stored[name], group_ti = res.strata, res.group_ti
            elif engine.n_groups == 1:
                stored[name] = np.asarray(res.values, dtype=float)[:, None]
        if not stored:
            errormsg = f'"{self.name}" keeps no counts of its states by group, so its R over time cannot be worked out'
            raise ValueError(errormsg)
        tis = np.arange(copy.t.npts) if group_ti is None else np.asarray(group_ti)
        values = np.full(len(tis), np.nan)
        taken = 0 # The steps the copy has taken
        for row,ti in enumerate(tis):
            while taken <= ti: # Through the step that starts at this time point, so its rates are those of the run's
                copy.run_one_step()
                taken += 1
            y = engine.y.copy()
            for name,vals in stored.items():
                comps = engine.state_comps[name]
                now = y[comps]
                total = now.sum(axis=0)
                share = np.divide(now, total, out=np.full_like(now, 1/len(comps)), where=total > 0)
                y[comps] = share*vals[row]
            ngm = engine.next_generation(y=y)
            if ngm is None:
                errormsg = f'Nothing in "{self.name}" moves people into an infected state, so it has no R'
                raise ValueError(errormsg)
            values[row] = float(np.max(np.abs(np.linalg.eigvals(ngm))))
        timevec = sim.results.timevec[tis]
        return ss.Result(name='Rt', label=f'R over time ({self.name})', module=self.name, scale=False, timevec=timevec, values=values)

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim, create its engine and objects, initialize results, and start its infection log if there is an `ss.infection_log` analyzer; see Module.init_pre() for details

        Args:
            sim (Sim): the sim being initialized
        """
        if self.graph is not None and any(st.given_init for st in self.graph.states.values()): # A starting value set after the disease was made (e.g. by an importer) turns off the default seeding
            self.graph.unset_default_init_prev()
        unknown = [key for key in self.pars if key not in self.graph.pars and key not in self.code_pars] if self.graph is not None else [] # Added after the disease was made, so nothing reads it
        if unknown:
            hint = ' The starting values are given when the disease is made, e.g. ss.Disease(..., init=dict(I=0.01)) or init_n=dict(I=100), and cannot be set through pars.' if 'init' in unknown or 'init_n' in unknown else ''
            raise sc.KeyNotFoundError(f'"{self.name}" has no parameter named {sc.strjoin(unknown)}, so nothing would read it. Its parameters are {sc.strjoin(self.graph.pars.keys())}{ss.expressions.did_you_mean(unknown[0], self.graph.pars.keys())}{hint}{ss.errors.starsim3(keys=unknown, call=f"ss.{type(self).__name__}")}')
        self.engine = ss.solvers.make_engine(self, sim)
        self.engine.define()
        needs = ss.interventions.memory_needs(sim) if isinstance(sim, ss.Sim) and self.graph is not None and self.graph.transitions else {} # Not a mock sim
        self.remembered = [name for mod,name in needs if mod == self.name and name in self.graph.transitions] # Transitions whose memory something reads, e.g. tb.diagnosis.since
        if self.abm and (sim.pars.keep_times if isinstance(sim, ss.Sim) else ss.options.keep_times): # Or every transition's, to read after the run
            self.remembered += [name for name in self.graph.transitions if name not in self.remembered]
        for name in self.remembered:
            self.define_states(ss.FloatArr(f'last_{name}', label=f'Time index of the last {name}'), ss.FloatArr(f'count_{name}', default=0, label=f'Number of times: {name}'))
        super().init_pre(sim)
        if any(isinstance(a, ss.infection_log) for a in sim.analyzers.values()):
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys(), timeline=self.t)
        self.engine.init_pre()
        return

    def init_post(self):
        """ Initialize the values of the states, and set the initial conditions of the declared states """
        super().init_post()
        self.engine.init_post()
        if self.callbacks and not self.abm:
            raise ss.CapabilityError(f'"{self.name}" has callbacks (add_callback()), which are called with the agents an event happened to, so it cannot run with solver="{self.sim.solver}". Run with solver="abm", or use ss.on() in a declared module.')
        return

    # The hooks the solvers call on a declared module, for one that needs more than its states and transitions (e.g. ss.Pregnancy)
    def compartmental_form(self, sim):
        """ A module that runs only as agents unless it defines this: make its small compartmental form, used only when a compartmental solver is named, and return the note that says what it leaves out

        Args:
            sim (Sim): the sim being initialized
        """
        return None

    def burnin_plan(self, engine):
        """ People part way through at the start: None for none (the default), or the plan for them (see `ss.Pregnancy.burnin_plan()` and `ss.solvers.Agents.backfill_burnin()`)

        Args:
            engine (Engine): the engine running this module
        """
        return None

    def make_children(self, tr, parents, ti):
        """ As agents, the children of the parents who went through a transition with births (`ss.birth(on=)`); returns their UIDs

        Args:
            tr (Transition): the transition with births
            parents (array): the parents
            ti (int): the step of the births
        """
        return self.engine.grow_children(tr, parents, ti)

    def also_happened(self, name, uids):
        """ As agents, others an event (a transition, by name) also happens to, as UIDs (e.g. the fetuses of a mother at a delivery); None for nobody else

        Args:
            name (str): the name of the transition
            uids (array): the agents it happened to
        """
        return None

    def add_callback(self, event, fn):
        """
        Call `fn(uids)` at the end of each step with the agents who went through a declared transition or event of this disease in it, from a module written as code (agents only)

        A declared module reacts to the same events with `ss.on()` instead. Pass a bound method (e.g. `self.on_delivery`), or register
        from `init_pre()`: the sim runs its own copy of each module, and a lambda made in `__init__()` would still refer to the original.

        Args:
            event (Transition/str): the transition or event, e.g. `preg.stillbirth`, `preg.loss`, or its name
            fn (func): called with the agents' UIDs (once for each time, so an agent can appear twice), only on a step when there are some

        Example:
            ```python
            preg.add_callback(preg.stillbirth, lambda uids: self.n_stillbirths.__setitem__(uids, self.n_stillbirths[uids] + 1))
            ```
        """
        self.check_not_given('add_callback()')
        if isinstance(event, ss.Transition):
            name = event.name
        elif isinstance(event, ss.conditions.Event) and event.args[0] == 'tr': # E.g. preg.loss, several events
            name = event.args[1].name
        elif isinstance(event, (str, tuple)):
            name = event
        else:
            errormsg = f'add_callback() takes a transition or event of "{self.name}" (e.g. {self.name}.{next(iter(self.graph.transitions), "infection") if self.graph else "infection"}) or its name, not {event!r}; for entering or leaving states, use ss.on() in a declared module'
            raise TypeError(errormsg)
        self.callbacks.append((name, fn))
        return

    def run_callbacks(self):
        """ Call each function of add_callback() with the agents its event happened to since it was last called """
        for i,(name,fn) in enumerate(self.callbacks):
            uids = self.engine.fresh(f'callback{i}', name, unique=False)
            if len(uids):
                fn(uids)
        return

    def start_step(self):
        super().start_step()
        self.engine.start_step()
        return

    def finish_step(self):
        """ Once the step's flows are recorded (and the analyzers have run), its engine starts counting them afresh """
        super().finish_step()
        self.engine.clear_flows()
        return

    def step_state(self):
        """
        State changes at the start of the step, before the networks and interventions, by a disease written as code (e.g. recovery)

        The declared transitions are all taken together in `step()`, after the interventions,
        so a delivery sees the same people under every solver (as agents and as compartments).
        """
        return

    def on_death(self, uids):
        """
        Carry out state changes upon death: the agents who died leave every declared state (as agents)

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

        Args:
            uids (array): the agents who died
        """
        if self.abm:
            self.engine.die(uids)
        return

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses upon acquisition, for a disease written as code: by default, add the agents to the infection log (if there is an `ss.infection_log` analyzer)

        A subclass extends this to assign state values (calling `super().set_prognoses()`); a declared disease never calls it.

        Args:
            uids (array): UIDs for agents to assign disease prognoses to
            sources (array): Optionally specify the infecting agent
        """
        if self.infection_log:
            self.infection_log.add_entries(uids, sources, self.t.tvec[max(self.ti, 0)]) # At the step's time point, as ti_infected is (the seeds at the start)
        return

    def current_value(self, key, uids=None):
        """
        `rel_sus` or `rel_trans` (`key`) of each agent with the modifiers and products active on it applied (`ss.modify(dis.rel_sus, ...)`), as agents:
        what transmission uses, whereas reading `dis.rel_sus` gives the base values, which a modifier never changes. Or a parameter (`key` is its
        name) with the modifiers active on it, in its own type, declared or written as code: one value, or one for each agent if they differ between agents.

        A disease written as code that reads one in its own code (other than in an `ss.AgentDisease`'s transmission) says so in its class,
        `allow_modifiers = ('rel_sus',)`, so that modifiers of it are applied rather than refused. A parameter it doesn't say this of is
        read from `self.pars` with the modifiers applied, which is one value, so a modifier of it can't differ between agents.

        Args:
            key (str): 'rel_sus', 'rel_trans', or the name of a parameter
            uids (uids): the agents to give the values of (default: an array of every agent, by UID)

        Example:
            ```python
            sim.diseases.sir.current_value('rel_sus') # With a vaccine's efficacy applied

            class Diabetes(ss.Disease):
                allow_modifiers = ('rel_sus',) # ss.modify(diabetes.rel_sus, ...) scales its onset
                def step(self):
                    uids = self.at_risk.uids
                    onset = self.pars.p_onset.rvs(uids)*self.current_value('rel_sus', uids) # ...
            ```
        """
        if key not in ['rel_sus', 'rel_trans']:
            if self.graph is not None and not self.abm and uids is not None:
                raise ss.CapabilityError(f'{self.name}.current_value() of some agents needs agents, which a compartmental run does not have; leave out uids, or use sim.what_modifies().')
            return self.engine.current_par(key, uids)
        if not isinstance(self.engine, ss.solvers.Transmission):
            raise ss.CapabilityError(f'{self.name}.current_value() is the value of each agent with the modifiers applied, which only a run as agents has; in a compartmental run, sim.what_modifies({self.name}.{key}) lists the modifiers')
        arr = self[key]
        vals = arr.asnew(arr.raw*self.engine.mult_raw(key), cls=ss.FloatArr, copy=False)
        return vals if uids is None else vals[uids]

    def not_naive(self):
        """ The agents who have left the first state (e.g. who are no longer susceptible), for dynamic rescaling (see `ss.Sim.rescale()`) """
        return (~self[self.graph.default_state.name]).uids

    def move(self, *args, **kwargs):
        """
        Move people between this disease's states from code, the same way under every solver: `ss.move()` with the disease given

        Args:
            args (list): passed to `ss.move()`: what to move, from where, and to where
            kwargs (dict): passed to `ss.move()`, e.g. `p` or `uids`

        Examples:
            ```python
            sir.move(sir.I, sir.R, p=0.5) # Cure half of those infectious
            sir.move(uids, sir.R) # These agents, from wherever they are (or sir.move(uids, 'R'))
            self.move(self.I, self.R, uids=uids) # In the disease's own code: these agents, if they are infectious
            ```
        """
        return ss.move(self, *args, **kwargs)

    def infect(self, uids, strain=None):
        """
        Infect some agents now, from code (e.g. a strain introduced on a date): those in the first state move where its transmission puts them, counted in its flow and dated as its infections are (`ti_I.isclose(sim.ti)`), with a source of -1 in an `ss.infection_log`

        Args:
            uids (uids): the agents; those not in the first state are left where they are
            strain (str/Strain): for a disease with strains in shared states, the strain they are infected with (by default, as the split says)

        Returns:
            The agents infected

        Example:
            ```python
            covid.infect(sim.people.auids[:20], strain='alpha') # 20 imported cases of alpha
            ```
        """
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" infects people once the sim is initialized (sim.init())')
        if self.sim.solver != 'abm':
            raise ss.CapabilityError(f'disease.infect() infects agents, but "{self.name}" is run with solver="{self.sim.solver}"; use ss.Delivery(move=S >> I) for a compartmental run')
        return self.engine.infect_now(ss.uids(uids), strain)

    def transmit(self, transition=None):
        """
        Who is infected in this step by one of the disease's transmissions (its transitions whose rate depends on contact with infectious people)

        Override this to write a transmission as code, and call `super().transmit(transition)` for what the solver would do, e.g. to add
        a route of its own or remove some of the cases. The engine calls it once a step for each transmission, and moves those it gives
        along that transition, counted in its flow. A disease with one transmission can leave out the argument (`def transmit(self):`);
        with several, `transition` says which (e.g. `if transition.name == 'infection':`).

        As agents, it gives the new cases' UIDs (those not in one of the transition's source states are not moved); as compartments, the
        number infected in each group (or one number), taken from the source states in proportion. Under the ODE, it is taken once a step,
        at the start, as a move of that many (as `ss.move()` from code is), so the ODE no longer integrates that transition within the
        step: give a rate (e.g. `S >> I @ (lambda self, sim: ...)`) to keep it exact. Anything that needs individuals (UIDs) raises an
        error in a compartmental run; `if self.sim.solver == 'ode':` can give each its own.

        Args:
            transition (`ss.Transition`): the transmission; may be left out for a disease with only one

        Returns:
            As agents, the UIDs of the new cases (`ss.uids`); as compartments, the number of them in each group

        Example:
            ```python
            class Imported(ss.SIR):
                def transmit(self, transition):
                    new = super().transmit(transition)
                    if self.sim.solver == 'abm': # And 2 imported cases a step, as agents
                        return new | self.sim.people.auids[self.S.raw[self.sim.people.auids]][:2]
                    return new + 2/self.sim.groups.n_groups
            ```
        """
        return self.engine.transmitted(self.transmission(transition))

    def transmission(self, transition=None):
        """ One of the disease's transmissions, by transition (or name); with none given, its only one """
        trs = self.graph.transmissions if self.graph is not None else []
        if transition is None:
            if len(trs) != 1:
                errormsg = f'"{self.name}" has {len(trs)} transmissions ({sc.strjoin(tr.name for tr in trs) or "none"}), so transmit() is given which one: def transmit(self, transition), with transition.name one of them'
                raise ValueError(errormsg)
            return trs[0]
        name = transition if isinstance(transition, str) else transition.name
        found = [tr for tr in trs if tr.name == name]
        if not found:
            raise sc.KeyNotFoundError(f'"{name}" is not a transmission of "{self.name}"; its transmissions are {sc.strjoin(tr.name for tr in trs)}')
        return found[0]

    @property
    def custom_transmit(self):
        """ Whether the disease is declared and has its own `transmit()`, which the engine calls for its transmissions """
        cls = type(self)
        return self.graph is not None and cls.transmit is not Disease.transmit and not issubclass(cls, AgentDisease)

    def call_transmit(self, tr):
        """ The disease's own `transmit()` for one of its transmissions, with or without the transition as its argument (see `transmit()`) """
        if len(inspect.signature(self.transmit).parameters):
            return self.transmit(tr)
        elif len(self.graph.transmissions) > 1:
            errormsg = f'"{self.name}" has {len(self.graph.transmissions)} transmissions ({sc.strjoin(tr.name for tr in self.graph.transmissions)}), so its transmit() is given which one: def transmit(self, transition), with transition.name one of them'
            raise TypeError(errormsg)
        return self.transmit()

    def current_state(self, states=None):
        """
        The state each agent is in, as a read-only `ss.IntArr` (by UID) of its index among `states`, or -1 for none

        The names of the states, in order, are its `.labels`, so `arr.labels[i]` is the name of index `i`.

        Args:
            states (list): the states to look in, as names or states (default: the module's states of people, in order)

        Example:
            ```python
            cs = sir.current_state() # 0 for S, 1 for I, 2 for R
            cs.labels[cs[uid]] # The name of an agent's state
            ```
        """
        graph = self.graph
        if self.engine is None and getattr(self, '_given', False):
            raise RuntimeError(f'"{self.name}" was given to a sim, which runs a copy of each module it is given (so it can be given to other sims); read the one that ran from the sim, e.g. sim.{self.name}.current_state()')
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" has the state of each agent once the sim is initialized (sim.init())')
        if isinstance(self.sim, ss.utils.Shrunk):
            raise RuntimeError(f'the state of each agent of "{self.name}" was dropped when the sim was shrunk; run it with shrink=False to keep it')
        if self.sim.solver != 'abm':
            raise ss.CapabilityError(f'disease.current_state() is the state of each of the agents, but "{self.name}" is run with solver="{self.sim.solver}"; use the counts of its states (e.g. sim.results.{self.name}.S) instead')
        names = [s if isinstance(s, str) else s.name for s in sc.tolist(states)] if states is not None else [s.name for s in graph.people_states]
        unknown = [name for name in names if name not in graph.states]
        if unknown:
            raise sc.KeyNotFoundError(f'"{self.name}" has no state {sc.strjoin(unknown)}; its states are {sc.strjoin(graph.states.keys())}')
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

    def make_naive(self, uids):
        """
        Put agents back in the first state, as if they had never had the disease; for dynamic rescaling (see `ss.Sim.rescale()`)

        Args:
            uids (`ss.uids`): the agents to make naive
        """
        states = self.graph.people_states
        for state in states:
            self[state.name][uids] = False
            self.engine.clear_exit(state.name, uids)
        self[states[0].name][uids] = True
        self[f'ti_{states[0].name}'][uids] = self.ti
        return

    def step(self):
        """ Take the declared transitions of the step, those that are not contact-mediated and then those that are, followed by any custom functions; a disease written as code overrides this """
        self.engine.step_state()
        self.engine.step()
        self.engine.step_funcs()
        if self.callbacks:
            self.run_callbacks()
        return

    def record_stocks(self):
        """ The counts of the states at the start of the step; for a disease written as code, the count of each `ss.BoolState` (see `ss.Module.record_stocks()`) """
        self.engine.record_stocks()
        return

    def record_flows(self):
        """ The flows of the step, and the results worked out from them """
        self.engine.record_flows()
        return

    def finalize_results(self):
        super().finalize_results()
        self.engine.finalize_results()
        return

    def shrink(self):
        super().shrink()
        self.engine.shrink()
        return


class AgentDisease(Disease):
    """
    A disease written as code for agents, with transmission over the sim's networks

    An `ss.Disease` that runs only as agents, and adds what a disease written as code needs to
    be transmitted: the `susceptible` and `infected` states, the attributes that other modules
    act on (`rel_sus`, `rel_trans`), seeding (`init_prev`), transmission with directional beta
    values (`transmit()`), `set_prognoses()` for the people infected, and the `prevalence` and
    `infection` results. The transmission itself is done by the solver (`ss.solvers.Transmission`,
    the same code that transmits a declared disease as agents); each piece of it is a method
    that can be overridden on its own: `transmissibility()`, `susceptibility()`, `route_betas()`,
    `transmit_route()`, and `finalize_infections()`. The rest of the disease is written as for
    any other module: its own per-agent states (`define_states()`) and how they change
    (`step_state()`, `set_prognoses()`), so it can hold anything an agent can have: a CD4 count,
    an immunity level that wanes, a time since infection. A disease with no `beta` is not
    transmitted; one that has no use for these states at all (such as `ssl.NCD`) is a plain
    `ss.Disease`.

    An infection in a step happens at its end, so a course scheduled from it starts at the next step: schedule a move
    `dur` steps after an infection at `self.ti + 1 + dur`, in an `ss.TimeArr`, and take it when `due()` (as in the
    example below). Each case is then infectious for `dur` steps on average, as a declared disease's is; the seeds are
    dated by the step before the first (`self.ti` is -1 while they are set), so they are too.

    A disease that can be written as states and transitions is declared instead, with
    `ss.Disease` (see `ss.SIR`), which gives it every solver and `describe()`, `equations()`,
    and `R0()`; a declared disease can still write any of its transmissions as code, with
    its own `transmit(transition)` (see `ss.Disease.transmit()`), and add per-agent arrays
    of its own. The verbs that act on arrays work on both: `ss.modify(hiv.rel_trans, ...)`,
    conditions such as `hiv.cd4 < 200`, and fitting the values in `pars`.

    With `strains=ss.Strains(...)`, each person carries one strain at a time (`self.strain`, the index of the
    strain of their current or last infection, -1 if none), which the disease's code reads (e.g. in
    `set_prognoses()`, and `strain_value()` for a strain's values). The strains are transmitted in one pass, each
    by those infectious with it at its `rel_trans`; each new case is infected by one of those who reached them in
    the step, at random, and takes their strain, unless protected against it (`strain_sus()`: protection from
    `ss.immunity(due_to=<state>, protection=matrix)` by the strain they had, or the disease's own), as a declared
    disease's strains are passed on, by the same code (`ss.solvers.Transmission`).

    Note: CRN safe.

    Args:
        pars (dict): parameters
        strains (`ss.Strains`): the strains, carried one at a time, sharing the disease's states
        kwargs (dict): parameters

    Examples:
        ```python
        class SIRS(ss.AgentDisease):
            def __init__(self, **kwargs):
                super().__init__()
                self.define_pars(beta=ss.perday(0.05), init_prev=ss.bernoulli(0.01), dur_inf=10)
                self.define_states(ss.BoolState('recovered'), ss.TimeArr('ti_recovered'))
                self.update_pars(**kwargs)

            def step_state(self):
                done = (self.infected & self.ti_recovered.due()).uids
                self.infected[done] = False
                self.recovered[done] = True
                self.susceptible[done] = True # Protected only by the immunity of their strain, if any

            def set_prognoses(self, uids, sources=None):
                super().set_prognoses(uids, sources)
                self.infected[uids] = True
                self.recovered[uids] = False
                self.ti_infected[uids] = self.ti
                self.ti_recovered[uids] = self.ti + 1 + self.pars.dur_inf # Infected at the end of this step, so infectious from the next

        # Two strains, the second more transmissible; recovering from one protects fully against it, and partly against the other
        strains = ss.Strains(['H1', 'H3'], rel_trans=[1, 1.5], immunity=ss.immunity(due_to='recovered', protection=[[1, 0.5], [0.5, 1]]))
        sim = ss.Sim(diseases=SIRS(strains=strains, name='flu'), networks='random', n_agents=2000, unit='day', stop=100).run()
        sim.diseases.flu.strain # The strain of each person's current or last infection
        ```
    """

    def __init__(self, pars=None, strains=None, **kwargs):
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
        self.update_pars(pars, strains=strains, **kwargs)

        # Define random number generator for determining transmission
        self.trans_rng = ss.multi_random('source', 'target')
        return

    def define_strains(self, strains):
        """ Give the disease strains (`ss.Strains`, also as `strains=` to the disease), carried one at a time: the strain of each person (`self.strain`, added when the sim is initialized, set for those infected by transmission, at the start, and by `infect()`; code that infects people itself sets it), and the strain of those infected at the start, by its split

        Args:
            strains (Strains): the strains
        """
        if strains.max_strains > 1:
            raise ss.CapabilityError(f'"{self.name}" is written as code, so its strains are carried one at a time, sharing its states, with protection on rel_sus due to its own states; but max_strains={strains.max_strains}. Declare the disease (see ss.Disease) for more.')
        self.strains = strains # Its arrays are added when the sim is initialized (see init_pre()), so a subclass may define its own states after this, even with reset=True
        return

    def update_pars(self, pars=None, **kwargs):
        """ As `ss.Module.update_pars()`, and `strains=` for `define_strains()`, so a subclass that passes its keywords on here takes it too (e.g. `ssl.AgentSIR(strains=...)`)

        Args:
            pars (dict): the parameters to update
            kwargs (dict): more parameters to update, by name
        """
        if isinstance(pars, dict) and 'strains' in pars: # Also in a dict of parameters
            pars = dict(pars)
            kwargs['strains'] = pars.pop('strains')
        strains = kwargs.pop('strains', None)
        if strains is not None:
            self.define_strains(strains)
        return super().update_pars(pars, **kwargs)

    def strain_value(self, key, uids):
        """ The value of `key` (given to `ss.Strains`, e.g. `rel_trans=[1, 1.5]`) for the strain each of some agents carries (1 if it has none)

        Args:
            key (str): the name of the value
            uids (array): the agents
        """
        vals = self.strains.values[key] if key in self.strains.values else np.ones(len(self.strains))
        strain = self.strain[uids]
        return np.where(strain >= 0, vals[np.maximum(strain, 0)], 1.0)

    def strain_sus(self, k):
        """
        The share of their susceptibility to strain `k` that each agent (by UID) keeps: `1 − protection` from each `ss.immunity` of the
        strains, for those in its state (e.g. `due_to='recovered'`) by the strain they had; a disease with protection of its own (e.g.
        from antibodies) overrides this

        Args:
            k (int): the index of the strain
        """
        out = 1.0
        strain = self.strain.raw
        for imm in self.strains.immunity:
            held = np.logical_or.reduce([self[name].raw for name in imm.names]) & (strain >= 0)
            out = out*np.where(held, 1 - imm.matrix(len(self.strains))[np.maximum(strain, 0), k], 1.0)
        return out

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim and initialize results; see `ss.Module.init_pre()`

        Args:
            sim (Sim): the sim being initialized
        """
        if self.strains is not None and not isinstance(getattr(self, 'strain', None), ss.IntArr): # The strain of each person, and of those infected at the start, by the split
            n = len(self.strains)
            self.define_states(ss.IntArr('strain', default=-1, label='Strain of the current or last infection'))
            self.strain_dist = ss.choice(a=n, p=self.strains.fractions if self.strains.fractions is not None else np.full(n, 1/n))
        super().init_pre(sim)
        if 'beta' in self.pars: # A disease that is not transmitted, such as ssl.NCD, has no beta
            self.route_betas()
            self.check_routes()
        self.validate_init_prev()
        unknown = [name for imm in (self.strains.immunity if self.strains is not None else []) for name in imm.names if not isinstance(getattr(self, name, None), ss.BoolArr)]
        if unknown:
            raise sc.KeyNotFoundError(f'the strains of "{self.name}" are protected by ss.immunity(due_to={unknown[0]!r}), which is not one of its states (a boolean array, e.g. ss.BoolState("recovered"))')
        return

    def check_routes(self):
        """ Raise an error if the disease has a beta but nothing to transmit it over: no network or mixing pool, and no `transmit()` of its own (e.g. cholera's, from the environment) """
        if len(self.sim.networks) or type(self).transmit is not AgentDisease.transmit or not isinstance(self.sim, ss.Sim): # A mock sim (init_mock()) has no networks
            return
        β = self.pars.beta
        values = β.values() if isinstance(β, dict) else [β]
        value = lambda v: v.values if isinstance(v, ss.peract) else v.value if isinstance(v, ss.Rate) else v
        if not any(np.any(value(v)) for v in values): # Not transmitted by design (beta=0)
            return
        errormsg = (f'"{self.name}" is transmitted over networks and mixing pools, but the sim has none, so it would infect no one: '
                     'add one (e.g. networks="random"), give it beta=0 if it is not transmitted, or override transmit() for a route of its own. '
                     '(A declared disease, such as ss.SIR, mixes homogeneously with no networks.)')
        raise ValueError(errormsg)

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
        Set initial values for states: by default, infect `init_prev` of the people (`set_prognoses()`)

        The initial infections are dated by the step before the first (`self.ti` is -1 while they are set), as if
        infected at its end, which is the start of the run, as a declared disease dates its seeds and an infection in a
        step is dated by that step: so a disease that schedules a move from `self.ti + 1` (e.g. `self.ti_recovered[uids] =
        self.ti + 1 + dur`, read with `ss.TimeArr.due()`) gives the seeds the same course as later cases.
        """
        super().init_post()
        if self.pars.init_prev is None:
            return

        people = self.sim.people
        unborn = people.unborn_uids() if isinstance(people, ss.People) else ss.uids() # E.g. the fetuses of pregnancies that ss.Pregnancy starts before the start, who are not seeded
        initial_cases = self.pars.init_prev.filter() if not len(unborn) else self.pars.init_prev.filter(people.auids[people.age.raw[people.auids] >= 0])
        if len(initial_cases):
            if self.strains is not None:
                self.strain[initial_cases] = self.strain_dist.rvs(initial_cases)
            saved, self.t.ti = self.t.ti, -1 # Dated by the step before the first (see above)
            try:
                self.set_prognoses(initial_cases, sources=-1)  # -1 = externally seeded infection with no source agent
            finally:
                self.t.ti = saved
        return initial_cases

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        self.define_results(
            ss.Result('prevalence', dtype=float, scale=False, label='Prevalence'),
            ss.Result('infection',  dtype=int,   scale=True,  label='New infections', kind='flow'), # As a declared disease's flow: its cumulative total is res.infection.cum
        )
        return

    def route_betas(self):
        """ The beta of each route, as a pair (p1 to p2, p2 to p1) for each of the sim's networks by name, checked against them (see `ss.solvers.Transmission.code_betas()`) """
        return self.engine.code_betas()

    def not_naive(self):
        """ The agents who have ever been infected, for dynamic rescaling (see `ss.Sim.rescale()`) """
        return self.ti_infected.notnan.uids

    def make_naive(self, uids, skip_states=None):
        """
        Reset agents to the state of never having had the disease; for dynamic rescaling (see `ss.Sim.rescale()`)

        Every state is reset to its default value, except `rel_sus` and `rel_trans`, which
        are often modified by other modules (e.g. vaccines); override this method if the
        disease needs something different.

        Args:
            uids (`ss.uids`): the agents to make naive
            skip_states (list): the names of states not to reset (default `['rel_sus', 'rel_trans']`)
        """
        skip_states = sc.tolist(skip_states) if skip_states is not None else ['rel_sus', 'rel_trans']
        for state in self.state_list:
            if state.name not in skip_states:
                state.set(uids)
        return

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses upon infection: make the agents non-susceptible, and record the new infections

        Subclasses extend this to assign state values upon infection (calling `super().set_prognoses()`).
        Infections are added to the log here if there is an `ss.infection_log` analyzer.

        Args:
            uids (array): UIDs for agents to assign disease prognoses to
            sources (array): Optionally specify the infecting agent
        """
        super().set_prognoses(uids, sources) # The infection log
        self.susceptible[uids] = False
        if self.sim.initialized: # Infections seeded by init_post() are prevalent, not incident, so don't count them
            weights = ss.results.agent_weights(self.sim) # Each agent counted by its weight (people.scale), if they differ
            if weights is None:
                self.results.infection[self.ti] += len(uids) # Count infections when they happen, rather than inferring them from ti_infected
            else:
                ss.results.as_float(self.results.infection)[self.ti] += weights[uids].sum()
        return

    def infect(self, uids, strain=None):
        """
        Infect some agents now, from code (e.g. a strain introduced on a date): those susceptible get the outcomes that transmission gives (`set_outcomes()`), counted in `infection`, with a source of -1 in an `ss.infection_log`

        A disease that declares its transmission infects them along it instead (see `ss.Disease.infect()`).

        Args:
            uids (uids): the agents; those not susceptible are left as they are
            strain (str/int): with strains, the strain they are infected with (its name or index; by default, as the split says)

        Returns:
            The agents infected

        Example:
            ```python
            hiv.infect(sim.people.auids[:20]) # 20 imported cases
            ```
        """
        if self.graph is not None and self.graph.transmissions:
            return super().infect(uids, strain)
        if not self.sim or not self.sim.initialized:
            raise RuntimeError(f'"{self.name}" infects people once the sim is initialized (sim.init())')
        if strain is not None and self.strains is None:
            raise ss.CapabilityError(f'"{self.name}" has no strains (define_strains()), so infect() takes no strain=')
        uids = ss.uids(uids)
        uids = uids[self.susceptible[uids]]
        if len(uids):
            if self.strains is not None:
                self.strain[uids] = self.strain_dist.rvs(uids) if strain is None else self.strains.names.index(strain) if isinstance(strain, str) else int(strain)
            self.set_outcomes(uids, sources=np.full(len(uids), -1))
        return uids

    def step(self):
        """
        Take any declared transitions, then the transmission (`transmit()`), and set the outcomes of those infected
        """
        super().step()

        # Create new cases
        new_cases, sources, networks = self.transmit() # TODO: store outputs in self or use objdict rather than 3 returns

        # Set prognoses
        if len(new_cases):
            self.set_outcomes(new_cases, sources)
            if self.infection_log:
                self.infection_log.add_data(new_cases, network=networks)

        return new_cases, sources, networks

    def transmit(self):
        """
        Who is infected in this step by transmission over the sim's networks and other routes (see `ss.solvers.Transmission.code_transmit()`)

        Each piece can be overridden on its own: the effective transmissibility and susceptibility (`transmissibility()`, `susceptibility()`),
        the betas (`route_betas()`), the transmission over each route (`transmit_route()`), and how the routes' infections are combined
        (`finalize_infections()`); or this whole step, e.g. to add a route of its own (as `ssl.Cholera` adds the environment).

        Returns:
            A tuple of the UIDs of the new cases, and of their sources and network IDs
        """
        return self.engine.code_transmit()

    def transmissibility(self):
        """ The effective transmissibility of each agent (`rel_trans`, as a `ss.FloatArr` over all of them), 0 for those not infectious; the modifiers are applied in `transmit_route()` """
        return self.rel_trans.asnew(self.infectious.raw*self.rel_trans.raw, copy=False)

    def susceptibility(self):
        """ The effective susceptibility of each agent (`rel_sus`, as a `ss.FloatArr` over all of them), 0 for those not susceptible; the modifiers are applied in `transmit_route()` """
        return self.rel_sus.asnew(self.susceptible.raw*self.rel_sus.raw, copy=False)

    def transmit_route(self, i, route, betas, rel_trans, rel_sus):
        """
        The transmission along one network (or other route), with the modifiers of `rel_trans` and `rel_sus` (including products) applied (see `ss.solvers.Transmission.code_route()`)

        Args:
            i (int): the index of the route in `sim.networks`, stored as the network ID of each infection
            route (`ss.networks.Route`): the network or other route (e.g. a mixing pool)
            betas (list): the pair of betas for the route (p1→p2 and p2→p1), each a number or rate, or an array of a probability per act for each direction of the network; routes other than networks only use the first
            rel_trans (`ss.FloatArr`): the effective transmissibility of each agent (0 if not infectious)
            rel_sus (`ss.FloatArr`): the effective susceptibility of each agent (0 if not susceptible)

        Returns:
            A tuple of the target UIDs, the source UIDs, and the network IDs of the new infections (possibly with duplicates)
        """
        return self.engine.code_route(i, route, betas, rel_trans, rel_sus)

    def finalize_infections(self, new_cases, sources, networks):
        """
        Combine the infections from each route, keeping only the first infection of each agent (see `ss.solvers.Transmission.first_infections()`)

        Args:
            new_cases (list): the target UIDs from each call to `transmit_route()`
            sources (list): the corresponding source UIDs
            networks (list): the corresponding network IDs

        Returns:
            A tuple of the unique target UIDs, and their source UIDs and network IDs
        """
        return ss.solvers.Transmission.first_infections(list(zip(sources, new_cases, networks)))

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

        Args:
            uids (array): the people infected before birth
            sources (array): the people who infected them
        """
        return

    def record_stocks(self):
        """ Update prevalence, at the start of the step; the result infection is recorded by set_prognoses() as infections happen """
        super().record_stocks()
        res = self.results
        alive = self.sim.people.n_alive if ss.results.agent_weights(self.sim) is None else self.sim.results.alive[self.sim.ti] # With agents of different weights, people alive as results count them (recorded just before)
        res.prevalence[self.ti] = res.infected[self.ti] / alive
        return

    def finalize_results(self):
        """ Check that the infections of the run were counted """
        super().finalize_results()
        res = self.results
        ti_infected = getattr(self, 'ti_infected', None) # A subclass can remove it (define_states(reset=...))
        if ti_infected is not None and not res.infection.values.any() and np.any(ti_infected.raw > 0): # People were infected during the run, but none were counted (not those seeded, which are dated by the step before the first: see init_post())
            ss.warn(f'"{self.name}": people were infected during the run, but the result infection is 0 at every time point. AgentDisease.set_prognoses() counts them, '
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

    records extra data for each infection (`ss.AgentDisease` records the network this way).
    Modules can optionally add per-infection outcomes later as well, for example

    >>> Disease.infection_log.add_data(uids, t_dead=2024.25)

    This would be equivalent to having specified the data at the original time the log
    entry was created - however, it is more useful for tracking events that may or may
    not occur after the infection and could be modified by interventions (e.g., tracking
    diagnosis, treatment, notification etc.)

    A table of outcomes can be returned using `InfectionLog.to_df()`, and a NetworkX
    graph with `InfectionLog.to_graph()`. An analyzer that follows transmission as the
    sim runs reads each step's entries with `to_df(ti=self.ti)`:

    >>> new = sim.diseases.sir.infection_log.to_df(ti=self.ti) # In an analyzer's step()

    Args:
        disease (str): the name of the disease being logged
        networks (list): the names of the networks, used to convert the network IDs recorded by `ss.AgentDisease` to names in `to_df()`
        timeline (`ss.Timeline`): the disease's timeline, so that each entry has the time index it was logged at, for `to_df(ti=...)`
    """
    def __init__(self, disease=None, networks=None, timeline=None):
        self.disease = disease
        self.networks = sc.tolist(networks)
        self.timeline = timeline
        self.chunks = [] # One dict of arrays (t, source, target, and any extra data) per call to add_entries()
        self.chunk_ti = [] # The time index each chunk was logged at (None without a timeline), for to_df(ti=...)
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
        """ Full display of the infection log

        Args:
            kwargs (dict): passed to `sc.pr()`
        """
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
            self.chunk_ti.append(None if self.timeline is None else self.timeline.ti)
            self.n += n
        return

    def append(self, source, target, t, **kwargs):
        """ Record a single infection

        Args:
            source (int): who did the infecting
            target (int): who was infected
            t (int): the time of the infection
            kwargs (dict): other data to store with the entry
        """
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

    def to_df(self, ti=None):
        """
        Return a tabular representation of the log as a line list dataframe

        This function returns a dataframe containing columns for all quantities
        recorded in the log. Note that the log will contain `NaN` for quantities
        that are defined for some edges and not others (and which are missing for
        a particular entry)

        Args:
            ti (int): if given, only the entries logged at this time index of the disease (e.g. `ti=self.ti` in an analyzer's `step()`, for the infections of the step just taken; at `ti=0`, the seeds); it reads only those entries, so is quick to call every step

        Example:
            ```python
            class new_infections(ss.Analyzer):
                def init_results(self):
                    super().init_results()
                    self.define_results(ss.Result('known', label='Infections from a known source'))

                def step(self):
                    df = self.sim.diseases.sir.infection_log.to_df(ti=self.ti)
                    self.results.known[self.ti] = (df.source >= 0).sum() # Not seeds (-1), nor those with no source (NA)

            sim = ss.Sim(diseases='sir', networks='random', analyzers=[ss.infection_log(), new_infections()]).run()
            ```
        """
        columns = ['t', 'source', 'target']
        if ti is None:
            keep = list(range(len(self.chunks)))
        else:
            if self.timeline is None:
                raise ValueError(f'The infection log of "{self.disease}" has no timeline, so it cannot find the entries of a time index; give InfectionLog(timeline=...) (a disease gives its own)')
            ti = int(ti) if ti >= 0 else self.timeline.npts + int(ti)
            keep = [i for i,cti in enumerate(self.chunk_ti) if cti == ti]
        if not keep:
            return sc.dataframe(columns=columns)

        # Only the entries from the first chunk kept on are needed: data added later can only change those, or later ones
        sizes = np.array([len(chunk['target']) for chunk in self.chunks])
        starts = np.concatenate([[0], np.cumsum(sizes)])
        offset = starts[keep[0]]
        df = pd.concat([pd.DataFrame(self.chunks[i]) for i in keep], ignore_index=True)

        # Apply data added later: for each UID, update its most recent entry at the time of the call
        if self.updates:
            targets_all = np.concatenate([chunk['target'] for chunk in self.chunks[keep[0]:]]) # From the first kept entry on
            row = np.full(len(targets_all), -1) # The row of the dataframe of each of those entries, if kept
            r = 0
            for i in keep:
                row[starts[i] - offset:starts[i+1] - offset] = np.arange(r, r + sizes[i])
                r += sizes[i]
            for n, uids, kwargs in self.updates:
                if n <= offset:
                    continue
                targets = targets_all[:n - offset]
                inds = np.flatnonzero(np.isin(targets, uids))
                last = pd.Series(inds, index=targets[inds]).groupby(level=0).last() # Most recent entry for each UID
                last = last[row[last.values] >= 0] # Of those kept
                if not len(last):
                    continue
                for k,v in kwargs.items():
                    if k not in df.columns:
                        df[k] = None # Object dtype, so any value can be stored
                    if np.ndim(v):
                        v = pd.Series(v, index=uids)[last.index].values # Match per-UID values to the entries
                    df.loc[row[last.values], k] = v

        # Convert network IDs to names, leaving any other values (e.g. names) as-is
        if self.networks and 'network' in df.columns:
            names = dict(enumerate(self.networks))
            df['network'] = df['network'].map(lambda v: names.get(v, v))
        if 'network' in df.columns and df['network'].isna().all(): # No route is known (e.g. homogeneous mixing, with no networks): no column of blanks
            df = df.drop(columns='network')
        df['t'] = [float(t.value) if isinstance(t, ss.Duration) else t for t in df['t']] # Plain numbers (on the disease's own axis), or dates, rather than durations, so times can be subtracted

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
    the infection unless `p_death` is given.

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks, over each step of the contact, so `ss.probperday(p)` gives each contact a chance p in a day; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float): the fraction of people who start infected (or, above 1, a number of people)
        dur_inf (float/`ss.Duration`/`ss.Dist`): the mean time that people are infected for (by default 6 years), as an exponential; give a distribution (e.g. `ss.lognormal(mean=ss.years(6))`) for another shape
        p_death (float): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`
        pars (dict): the same parameters, as a dict

    An SIR model written as code, with a lognormal `dur_inf` and `p_death=0.01`,
    is `ssl.AgentSIR`.

    Example:
        ```python
        sim = ss.Sim(diseases=ss.SIR(beta=0.1, dur_inf=ss.years(6)), networks='random', n_agents=5000)
        sim.run().plot()
        ```
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_inf   = ss.par(ss.years(6))
    p_death   = ss.par(0.0)
    S         = ss.State('S')
    I         = ss.State('I', infected=True) # Counted in the results infected and prevalence
    R         = ss.State('R')
    infection = ss.Transition(S, I, rate=beta * I)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.dead, dur=dur_inf, p=p_death)


class SEIR(Disease):
    """
    An SEIR model: as `ss.SIR`, with an exposed (latent) state before people become infectious

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks, over each step of the contact, so `ss.probperday(p)` gives each contact a chance p in a day; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float): the fraction of people who start infected, who start exposed, in `E` (or, above 1, a number of people); to start people infectious, give `init=dict(I=0.01)` or `init_n=dict(I=10)` instead
        dur_exp (float/`ss.Duration`/`ss.Dist`): the mean time that people are exposed (latent) for (by default 0.5 years; a number is in the model's unit), as an exponential
        dur_inf (float/`ss.Duration`/`ss.Dist`): the mean time that people are infectious for (by default 6 years), as an exponential
        p_death (float): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`
        pars (dict): the same parameters, as a dict

    As in `ss.SIR`, each duration is exponential unless a distribution is given, and
    nobody dies of the infection unless `p_death` is given.

    An SEIR model written as code, with lognormal durations and `p_death=0.01`, is
    `ssl.AgentSEIR`.
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_exp   = ss.par(ss.years(0.5))
    dur_inf   = ss.par(ss.years(6))
    p_death   = ss.par(0.0)
    S         = ss.State('S')
    E, I      = ss.states('E I', infected=True) # Counted in the results infected and prevalence
    R         = ss.State('R')
    infection = ss.Transition(S, E, rate=beta * I)
    latency   = ss.Transition(E, I, dur=dur_exp)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.dead, dur=dur_inf, p=p_death)


class SIS(Disease):
    """
    An SIS model: people recover from infection and are susceptible again

    Declared from states and transitions (see `ss.Disease`), so it runs with any solver.

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks, over each step of the contact, so `ss.probperday(p)` gives each contact a chance p in a day; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float): the fraction of people who start infected (or, above 1, a number of people)
        dur_inf (float/`ss.Duration`/`ss.Dist`): the mean time that people are infected for (by default 10 years), as an exponential unless a distribution is given
        kwargs (dict): passed to `ss.Disease`, e.g. `name`
        pars (dict): the same parameters, as a dict

    An SIS model with a per-agent immunity level, boosted by each infection and
    waning over time (`waning=`, `imm_boost=`), which has no declared form, is
    `ssl.AgentSIS`, written as code.
    """
    beta      = ss.par(ss.peryear(0.05))
    dur_inf   = ss.par(ss.years(10))
    S         = ss.State('S')
    I         = ss.State('I', infected=True) # Counted in the results infected and prevalence
    infection = ss.Transition(S, I, rate=beta * I)
    recovery  = ss.Transition(I, S, dur=dur_inf)
