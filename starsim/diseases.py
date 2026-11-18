"""
Diseases: the base class (`ss.Disease`), diseases with transmission written as code (`ss.AgentDisease`), the record of who infected whom (`InfectionLog`), and the textbook models (`ss.SIR`, `ss.SIS`, `ss.SEIR`)

`ss.Disease` exists so that one class can hold a disease declared from states and transitions, written as code, or both. A declared disease runs with any solver; `ss.AgentDisease` and other code that works on individual agents runs as agents.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .internals import declaring



class BuiltAfterInit(type):
    """ Build what a disease declares in its `__init__()` (`define_states()`, `define_transitions()`) once that has finished """
    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls) # Called self, as a constructor of it: see ss.utils.assigned_name()
        self.__init__(*args, **kwargs)
        self.build_graph()
        return self


def check_stratify(stratify, where):
    """ The `stratify=` of a disease or a sim: one dict of dimensions, as `stratify()` takes them """
    if isinstance(stratify, (list, tuple)):
        merged = {k:v for call in stratify if isinstance(call, dict) for k,v in call.items()}
        errormsg = f'{where}(stratify=) takes one dict of dimensions, e.g. stratify={merged or dict(age=[0, 15])!r}, not a list of them (options of a dimension are ss.levels())'
        raise TypeError(errormsg)
    if not isinstance(stratify, dict):
        errormsg = f'{where}(stratify=) takes a dict of dimensions, e.g. stratify=dict(age=[0, 15, 65]), not {type(stratify).__name__}'
        raise TypeError(errormsg)
    return stratify


def spectral_radius(matrix):
    """ The largest absolute eigenvalue of a matrix: the reproduction number of a next-generation matrix (see `ss.Disease.R0()`) """
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


class Disease(ss.Module, metaclass=BuiltAfterInit):
    """
    A disease: states people are in and the transitions between them (declared), code, or both

    Declared from its states and transitions (see the examples), no methods need to be defined, and the
    disease can be run with any solver (see `ss.Sim`). The same disease can be declared in a class body
    (the class form), which can also add behavior by defining methods; `ss.SIR`, `ss.SIS`, and `ss.SEIR`
    are written this way.

    A disease can also be written as code: a subclass that declares
    nothing, defines its per-agent arrays (`define_states()`), and changes them in its
    own `step_state()` and `step()`; it runs as agents. One class can mix the two, e.g. a
    declared natural history and a coded array that its `step()` updates after
    `super().step()` has taken the declared transitions. Each lifecycle method runs the
    declared part through the disease's engine (which, for a disease that declares
    nothing, has nothing to run), so an override calls `super()` as for any module.
    `ss.AgentDisease` adds transmission written as code: `transmit(rel_trans=..., rel_sus=...)`,
    the same array operation as a declared transmission's (see `transmit()`).

    Note: CRN safe as agents.

    Args:
        args (list): transitions and outcomes, named from the variables they are assigned to (one by one, or in a list)
        pars (dict): parameter values by name
        transitions (list): transitions, as a list (for longer models)
        results (list): extra results: sums and ratios of states
        outcomes (list): `ss.outcome` objects
        init_prev (float): the initial prevalence, a fraction of people (up to 1), seeded through the transmission (by default 0.01, unless a state is given an initial value)
        init (dict): the fraction of people who start in each state, e.g. `dict(S=0.90, I=0.01, R=0.09)`, and, for a state that is not people, its amount, which can be read from the people (e.g. `dict(Sv=m*N)`, with `N = (S + I).n`); each value can be an `ss.by` or a function of the sim (in the class form, `init = dict(I=0.01)` in the class body; likewise `init_n` and `init_prev`)
        init_n (int/dict): instead, a number of people: one number is seeded through the transmission, as `init_prev` is (e.g. `init_n=10`, a parameter, like `init_prev`); a dict is the number who start in each state, e.g. `dict(I=1)` for one person
        default_state (str/State): the state that everyone not put elsewhere starts in, and is born into (its name, or the state): by default, the one named `S` or `susceptible`, else the first state of people made (in the class form, `default_state = 'N'` in the class body). A state with a name of its own given under another keyword (e.g. `first=S`) raises, rather than being renamed
        strains (Strains): its strains (`ss.Strains`), which share its states; then `dis.strains`
        stratify (dict): the dimensions, as `stratify()` takes them, e.g. `dict(age={0: 0.3, 15: 0.6, 65: 0.1}, risk=ss.levels(['low', 'high'], states=[I]))`
        rel_sus (Dist): a distribution that each person's susceptibility is drawn from once, e.g. `ss.lognormal(mean=1, std=0.5)` (also in `ss.SIR` and the other ready-made diseases); it needs agents, so `solver='auto'` runs them, and a compartmental solver uses its mean, with a note. For a value by group, give `ss.levels(..., rel_sus=[...])` to `stratify()`
        rel_trans (Dist): likewise, each person's infectiousness, e.g. `ss.gamma(shape=0.2, scale=5)` (a mean of 1) for superspreading
        solver (abm): how this disease runs as agents, whatever the sim's solver says, e.g. `ss.solvers.abm(step='wholestep')` (see below); an error for a compartmental run
        kwargs (dict): one keyword per state, transition, parameter, result, function, or modifier (`ss.modify()`, of its own quantities, e.g. `fading=ss.modify('rel_trans', multiply=0.5, who=I)`); also `name`, `label`, and `dt`

    A disease's own `solver=` is useful when one disease of a sim needs a different way of running as agents from the others:

    - `ss.solvers.abm(step='wholestep')`: by steps, with rates applied over whole steps to those in the state at the start of the step, as code with `ss.bernoulli()` does; faster, and a little less exact when the step is coarse
    - `ss.solvers.abm(step='events')`: event by event, in continuous time
    - by default: the sim's, if it names a step; else the class's `suggested_solver` (e.g. `suggested_solver = ss.solvers.abm(step='wholestep')` in the class body); else `'midstep'` (see `ss.solvers.abm()` and `ss.solvers.agent_solver()`)

    The results are the number in each state (`I`), the flow along each transition by its
    name (`infection`, with its cumulative total as `res.infection.cum`), and the results and
    outcomes declared (e.g. `active=(E + I).frac`). Its infected states also give the results `infected`,
    the number in them, and `prevalence`, their fraction of everyone in the disease's states, unless it
    defines its own: those of its force of infection (`E` and `I` in `S >> E @ beta*I`), or those it says
    are infected (`infected = E + I + H`, `ss.State(infected=True)`, or `ss.states('S E I H R', infected='E I H')`).

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
            p_onset = ss.par(ss.bernoulli(0.01))
            anemic  = ss.BoolState(label='Anemic')

            def step_state(self):
                self.anemic[self.pars.p_onset.filter()] = True
        ```
    """
    graph = None # Its declared states and transitions (`ss.graph.Graph`); None for a disease written as code
    engine = None # Runs the declared part (see ss.solvers.make_engine()); created when the sim is initialized
    infection_log = None # An InfectionLog, if an analyzer needs it (e.g. ss.infection_log; see ss.Analyzer)
    suggested_solver = None # How a declared module's class suggests it runs as agents, e.g. ss.solvers.abm(step='wholestep') (or the step, 'wholestep'): followed unless the sim's solver names a step, or the module has its own solver= (see ss.solvers.agent_solver())
    code_pars = () # Parameters of a declared module that its own code reads rather than its graph (e.g. ss.Pregnancy's fertility rates)
    remembered = () # The transitions whose memory something reads, by name (see _init_memory())
    remembered_events = {} # Other events whose memory something reads, by name: what the engine asks of each move (see _init_memory())
    trans_rng = None # The draws of transmission over each edge (ss.multi_random('source', 'target')), for transmit(rel_trans=..., rel_sus=...): a disease written as code defines it in __init__()

    _infection_methods = ['set_prognoses', 'transmissibility', 'susceptibility', 'route_betas', 'set_outcomes', 'finalize_infections', 'compute_transmission', 'transmit_route', 'set_congenital'] # Methods of ss.AgentDisease's transmission, which a declared transition never calls

    def __init_subclass__(cls, **kwargs):
        """ A class that declares states or transitions takes its transitions through its engine, which never calls the methods of ss.AgentDisease's transmission, so overriding one (e.g. set_prognoses() on a subclass of ss.SIS) would be silently ignored """
        super().__init_subclass__(**kwargs)
        overridden = [name for name in cls._infection_methods if name in vars(cls)]
        if overridden and any(not isinstance(val, ss.par) for val in ss.transitions.class_members(cls).values()): # Parameters alone may be those of a disease written as code
            base = next((f'ssl.Agent{b.__name__}' for b in cls.__mro__ if b.__module__ == __name__ and b.__name__ in ['SIR', 'SIS', 'SEIR']), None)
            base = f'is an ss.AgentDisease, as {base} is (import starsim.library as ssl)' if base else 'is an ss.AgentDisease'
            errormsg = (f'{cls.__name__} declares states or transitions and defines {sc.strjoin(overridden)}, which a declared transition never calls, so it would be silently ignored.\n'
                        f'     To write a declared transmission as code, override transmit(self, transition), which gives who is infected (see ss.Disease.transmit()); a disease written as code from the start {base}.\n'
                        f'     If this was written for Starsim 3, see {ss.v3_deprecations.guide} for help migrating.')
            raise ss.CapabilityError(errormsg)
        rebound = [name for name in dict.fromkeys(k for base in cls.__mro__[1:] for k,v in vars(base).items() if isinstance(v, ss.par)) if name in vars(cls) and not isinstance(vars(cls)[name], (ss.par, ss.expressions.Expr))]
        if rebound: # A declared parameter rebound to something else (a method, a number) is dropped from the model, and its transitions keep the base class's parameter, so it would be silently ignored
            name = rebound[0]
            how = (f'give the parameter a function as its value instead, e.g. self.pars.update({name}=self.my_{name}) in __init__(), where my_{name}(sim, uids) returns the value for those agents' if callable(vars(cls)[name]) else
                   f'to change its default, give the value in __init__(), e.g. super().__init__(**({{"{name}": {vars(cls)[name]!r}}} | kwargs))')
            errormsg = f'{cls.__name__} defines {sc.strjoin(rebound)}, a parameter of {cls.__mro__[1].__name__}, as {"a method" if callable(vars(cls)[name]) else "a value"}, which its transitions never read; {how}'
            raise ss.CapabilityError(errormsg)
        infect = vars(cls).get('infect')
        if infect is not None and len(inspect.signature(infect).parameters) == 1: # Starsim 3's transmission pass, infect(self), which is transmit() now: it would never be called
            errormsg = f'{cls.__name__} defines infect(self), the transmission pass, which is called transmit() now (infect(uids) infects people from code); in ss.AgentDisease, which returns the UIDs of the new cases (see ss.AgentDisease.transmit())'
            raise ss.CapabilityError(errormsg)
        return

    def __init__(self, *args, pars=None, strains=None, stratify=None, solver=None, _names=None, **kwargs):
        super().__init__()
        self.strains = strains # Its ss.Strains, if it has strains: set now, so a subclass's __init__() can read it before the disease is built (see ss.strains.attach())
        if solver is not None:
            try:
                solver = ss.solvers.as_solver(solver)
            except ValueError as e:
                raise ValueError(f'{e} (a module\'s own solver is a way of running as agents, e.g. ss.solvers.abm(step="wholestep"))') from None
            if not isinstance(solver, ss.solvers.abm):
                raise ValueError(f'A module\'s own solver is a way of running as agents, ss.solvers.abm(step=...), not {solver!r}: the sim\'s solver= says how the whole model runs')
        self.solver = solver # Its own way of running as agents (see ss.solvers.agent_solver()); None follows the sim's
        self.callbacks = [] # Functions called with the agents an event happened to: see add_callback()
        if len(args) and (args[0] is None or isinstance(args[0], dict)): # Parameters can also be the first argument
            pars, args = args[0], args[1:]
        args = tuple(x for arg in args for x in (arg if isinstance(arg, (list, tuple)) else [arg])) # A list of transitions and outcomes is the same as giving them one by one
        members = ss.transitions.class_members(type(self))
        if isinstance(self, AgentDisease): # Written as code, so the parameters in its class body are its own (see ss.Module.define_class_parts())
            members = {key:val for key,val in members.items() if not isinstance(val, ss.par)}
        if members and all(isinstance(val, ss.modify) for val in members.values()):
            errormsg = f'{sc.strjoin(members)} of {type(self).__name__} is a modifier, which is a member of a declared disease; for a disease written as code, give it to the sim, in connectors='
            raise ss.CapabilityError(errormsg)
        structure = declaring.structure_of(members) # Parameters alone in the class body are a disease written as code, unless its __init__() declares states and transitions that use them
        given = args or structure or declaring.has_structure(kwargs) # Its structure; a parameter's value (e.g. with bounds) is not
        later = members or strains is not None or stratify is not None or any(key in kwargs for key in ['init', 'init_n', 'init_prev', 'default_state']) or any(isinstance(v, ss.par) for v in kwargs.values()) # Only for a declared disease, so its states and transitions follow, from define_states() and define_transitions(); or for parameters in the class body, which are set once that is known (see build_graph())
        if given or later:
            names = (_names or ss.utils.caller_names(1, skip=BuiltAfterInit.__call__)) if given else {} # The names of the variables of the code that made it; or, made by the sim from the transitions it was given, of the code that made the sim
            if not given: # A parameter's value given by keyword (e.g. with bounds) sets the parameter of that name that the transitions declared next use, as pars= does
                pars = sc.mergedicts(pars, {key:kwargs.pop(key) for key in list(kwargs) if isinstance(kwargs[key], ss.par)})
            self._declared = declaring.declaration(members, args, pars=pars, kwargs=kwargs, names=names, strains=strains, stratify=stratify)
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

    def build_graph(self):
        """ Build the graph from what was declared: when the disease is made, or for one declared in its __init__(), after that (see `BuiltAfterInit`) """
        d = self.__dict__.pop('_declared', None)
        if d is None:
            return
        if not (d.args or declaring.structure_of(d.members) or d.strains is not None or d.stratify is not None or declaring.has_structure(d.kwargs)): # Nothing was declared, only parameters given (or in the class body): written as code
            self.define_class_parts({key:val for key,val in d.members.items() if isinstance(val, ss.par)}, {})
            return self.update_pars(d.pars, **d.kwargs)
        declaring.build(self, d.args, d.members, d.pars, d.kwargs, names=d.names, strains=d.strains)
        self.add_code_pars(d.code_pars)
        if d.strains is not None:
            ss.strains.attach(self, d.strains)
        if d.stratify is not None:
            self.stratify(**check_stratify(d.stratify, 'ss.Disease'))
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
            d = self._declared = declaring.declaration(ss.transitions.class_members(type(self)))
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
        declared = {id(obj) for obj in d.kwargs.values()}
        d.names.update({i:n for i,n in names.items() if i not in declared}) # A local variable that holds a member already named by keyword (e.g. src = self.I) does not rename it
        return

    def define_states(self, *args, overwrite=False, **kwargs):
        """
        Define its states: declared states (`ss.State`), or for a disease written as code, arrays (see `ss.Module.define_states()`)

        Declared states are given in its `__init__()`, as for `define_transitions()`, and each is an attribute at once (`self.S`).

        Args:
            args (list): the states (`ss.states('S I R')`, or held in variables), or lists of them; or arrays (`ss.BoolArr`, ...)
            overwrite (bool): replace a state of the same name, rather than raising
            kwargs (dict): states by name (`S=ss.State(label='Susceptible')`); or for arrays, aliases
        """
        flat = [x for arg in args for x in (arg if isinstance(arg, list) and all(isinstance(y, ss.State) for y in arg) else [arg])]
        states = [x for x in flat if isinstance(x, ss.State)]
        named = {k:v for k,v in kwargs.items() if isinstance(v, ss.State)}
        if not states and not named: # Arrays, for a disease written as code
            return super().define_states(*args, overwrite=overwrite, **kwargs)
        if len(states) + len(named) < len(flat) + len(kwargs):
            errormsg = f'define_states() of "{self.name}" takes declared states (ss.State) or arrays (ss.BoolArr, ...), not both in one call'
            raise TypeError(errormsg)
        names = ss.utils.caller_names(1) # A state held in a variable takes its name, e.g. V = ss.State('Vaccinated'); self.define_states(V)
        for st in states:
            st.name = st.name or names.get(id(st)) or (st.label if isinstance(st.label, str) and st.label.isidentifier() else None) # Its variable, else its label if no variable holds it (ss.State('V'))
        unnamed = [st for st in states if st.name is None]
        if unnamed:
            raise ValueError(f'a state given to define_states() of "{self.name}" has no name: hold it in a variable, give it by keyword (define_states(V=ss.State())), or name it (ss.State(\'V\'), or ss.states(\'V W\'))')
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

    def stratify(self, inplace=True, **kwargs):
        """
        Stratify the disease by a property of the people in it (age, risk, place, ...), in place, and return it

        Every keyword is a dimension, with its levels: bin edges (for a continuous property such as age) or labels,
        or a dict of the fraction of the population in each level (if not given: from a named population or geography;
        for age groups, the world's ages; for a boolean dimension, or one that people are moved along, everyone at
        the first level; otherwise equal). `ss.levels()` adds the options that are needed less often: the states it
        applies to (`states=`), the people it applies to (`who=`), its kind, and the movement between places.
        A value by level, e.g. a rate, is `ss.by()`: `S >> I @ beta*ss.by(risk=[1, 3])*I`, or for a disease
        already written, `ss.modify(sir.infection, multiply=ss.by(risk=[1, 3]))`.

        Args:
            inplace (bool): if False, return a stratified copy and leave this disease unchanged
            kwargs (dict): each dimension and its levels, e.g. `age=[0, 15, 65]`, `risk={'low': 0.8, 'high': 0.2}`, or `severity=ss.levels(['mild', 'severe'], states=[I])`;
                an `ss.transfer` passed as a keyword is added under that name

        Examples:
            ```python
            sir.stratify(age={0: 0.3, 15: 0.6, 65: 0.1}) # With the fraction of people in each age group
            sir.stratify(age=[0, 15, 65], risk={'low': 0.8, 'high': 0.2})
            sir.stratify(sex=['female', 'male'], parity=ss.levels(['0', '1-2', '3+'], who=ss.person.female)) # Only women have a parity
            sir.stratify(severity=ss.levels({'mild': 0.9, 'severe': 0.1}, states=[sir.I]))
            ```
        """
        if self.graph is None:
            errormsg = f'"{self.name}" is written as code ({type(self).__name__}), so it has no states for stratify() to divide; its own code can read a property of people instead, or a sim stratified with sim.stratify() gives people the levels (people.<name>) for a network or a declared module to use'
            raise ValueError(errormsg)
        mod = self if inplace else self.copy()
        mod.graph.stratify(**kwargs)
        for strat in mod.graph.strats: # So a dimension can be used in a condition, e.g. who=~sir.vaccinated
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
        """ The table of declared transitions; printing it also shows the matrix of sources by destinations; empty for a disease written as code """
        return self.graph.transitions if self.graph is not None else ss.graph.TransitionTable()

    @property
    def states(self):
        """ Its declared states (`ss.State`) by name, e.g. `'second_line' in tb.states` or `[s for s in tb.states.values() if s.people]`; empty for a disease written as code. (The arrays behind them are `state_dict`.) """
        return sc.obj(self.graph.states) if self.graph is not None else sc.obj()

    @property
    def strats(self):
        """ Its stratifications by name, e.g. `tb.strats.age.edges` or `tb.strats.risk.labels` (as a condition, a level is `tb.age`) """
        return sc.obj({strat.name:strat for strat in self.graph.strats}) if self.graph is not None else sc.obj()

    def describe(self, out=None):
        """
        The resolved model, with every guessed default marked, as text that shows itself (`ss.display.Text`), as `diagram()` is

        Args:
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
        return desc if out is not None else desc.as_text()

    def equations(self):
        """
        The ODE system of the declared transitions, with symbolic parameter names, as text that shows itself (`ss.display.Text`; `print()` it in a script)
        """
        if self.graph is None:
            errormsg = f'"{self.name}" is written as code ({type(self).__name__}), so it has no declared states and transitions to write as equations; only a declared disease has them (describe() shows its arrays and parameters)'
            raise ValueError(errormsg)
        return ss.display.Text('\n'.join(ss.display.equations(self.graph, self.engine)))

    def diagram(self, kind='plot', **kwargs):
        """
        The state-transition diagram: drawn as a Matplotlib figure, or as the text of a Mermaid flowchart (https://mermaid.js.org)

        People states are rounded and counts square (slanted in the text); transmissions are dashed; each
        arrow is labeled with its transition's name and value (`recovery: 10 days`), and
        the fraction that takes it for a transition with `p=`. For every disease of a sim, and how they act on each other, see `sim.diagram()`.

        Args:
            kind (str): 'plot' to draw it, or 'text' for the Mermaid text, which a notebook draws and `print()` shows (see `ss.display.Diagram`)
            kwargs (dict): for a plot, passed to `ss.return_fig()` (e.g. `filename=`, `do_show=`)

        Example:
            ```python
            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=ss.days(10)))
            sir.diagram()
            print(sir.diagram('text'))
            ```
        """
        return ss.display.show_diagram([self.graph], [self.name], kind=kind, **kwargs)

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
        this matters where a count is a number rather than a fraction of the people (mosquitoes per person).

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
            sim = ss.solvers.as_ode(self.sim, verbose=0).init()
            dis = sim.diseases[self.name]
        else: # This disease alone, whatever the sim names it
            sim = ss.Sim(sc.dcp(self), solver='ode', verbose=0).init()
            dis = sim.diseases[0]
        ngm = self._check_ngm(dis.engine.next_generation(), 'R0')
        return ngm if matrix else spectral_radius(ngm)

    def _check_ngm(self, ngm, what):
        """ The next-generation matrix of `R0()` or `Rt()`, which is None for a disease that infects nobody """
        if ngm is None:
            errormsg = f'Nothing in "{self.name}" moves people into an infected state, so it has no {what}'
            raise ValueError(errormsg)
        return ngm

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
        copy = ss.solvers.as_ode(sim, verbose=0).init()
        engine = copy.diseases[self.name].engine
        results = sim.results[self.name]
        stored = {} # The people in each state of each group at each kept time point
        group_ti = None
        for name in engine.state_comps.keys():
            res = results[name] if name in results else None # Not get(), which ignores case
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
                frac = np.divide(now, total, out=np.full_like(now, 1/len(comps)), where=total > 0)
                y[comps] = frac*vals[row]
            values[row] = spectral_radius(self._check_ngm(engine.next_generation(y=y), 'R'))
        timevec = sim.results.timevec[tis]
        return ss.Result(name='Rt', label=f'R over time ({self.name})', module=self.name, kind='ratio', scale=False, timevec=timevec, values=values)

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim, create its engine and objects, initialize results, and start its infection log if an analyzer needs it (e.g. `ss.infection_log`); see `ss.Module.init_pre()`

        Args:
            sim (Sim): the sim being initialized
        """
        if self.graph is not None and any(st.given_init for st in self.graph.states.values()): # A starting value set after the disease was made (e.g. by an importer) turns off the default seeding
            self.graph.unset_default_init_prev()
        unknown = [key for key in self.pars if key not in self.graph.pars and key not in self.code_pars] if self.graph is not None else [] # Added after the disease was made, so nothing reads it
        if unknown:
            hint = ' The starting values are given when the disease is made, e.g. ss.Disease(..., init=dict(I=0.01)) or init_n=dict(I=100), and cannot be set through pars.' if 'init' in unknown or 'init_n' in unknown else ''
            raise sc.KeyNotFoundError(f'"{self.name}" has no parameter named {sc.strjoin(unknown)}, so nothing would read it. Its parameters are {sc.strjoin(self.graph.pars.keys())}{ss.expressions.did_you_mean(unknown[0], self.graph.pars.keys())}{hint}{ss.v3_deprecations.hint(keys=unknown, call=f"ss.{type(self).__name__}")}')
        self.engine = ss.solvers.make_engine(self, sim)
        self.engine.define()
        self._init_memory(sim)
        super().init_pre(sim)
        if any(getattr(a, 'needs_infection_log', False) for a in sim.analyzers.values()): # e.g. ss.infection_log
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys(), timeline=self.t)
        self.engine.init_pre()
        return

    def _init_memory(self, sim):
        """ The transitions and other events whose memory something reads (e.g. `tb.diagnosis.since`, `sir.I.enter.ever`), each with an array of when it last happened and one of how often (see `ss.solvers.deliveries.memory_needs()`)

        Args:
            sim (Sim): the sim being initialized
        """
        self.remembered, self.remembered_events = [], {}
        if not isinstance(sim, ss.Sim) or self.graph is None or not self.graph.transitions: # A mock sim, or nothing declared to remember
            return
        needs = ss.solvers.deliveries.memory_needs(sim)
        self.remembered = [name for mod,name in needs if mod == self.name and name in self.graph.transitions]
        if self.abm and sim.solver.keep_times: # Or every transition's, to read after the run
            self.remembered += [name for name in self.graph.transitions if name not in self.remembered]
        events = sim.__dict__.get('_memory_events', {}) if needs else {}
        self.remembered_events = {key:events[(mod, key)].spec() for mod,key in needs if mod == self.name and (mod, key) in events} # What the engine asks of each move (see ss.solvers.Agents.record())
        for name in self.remembered + list(self.remembered_events):
            self.define_states(ss.FloatArr(f'last_{name}', label=f'Time index of the last {name}'), ss.FloatArr(f'count_{name}', default=0, label=f'Number of times: {name}'))
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
        """ As agents, the children of the parents who went through a transition with births (`ss.birth(when=)`); returns their UIDs

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
        What happens to agents when they die: as agents, they leave every declared state

        It is called once the step's deaths are settled, before the analyzers run. A disease written as code overrides it
        for what changes at death (e.g. clearing `infected`). A death is only requested during the step (e.g. in
        `step_state()`, as `ssl.AgentSIR` does), so an intervention on the same step can still avert it, and nothing
        needs undoing, since these changes are made only if the death happens. The sim does not call it directly:
        `people.step_die()` does, once it knows who died.

        Args:
            uids (array): the agents who died
        """
        if self.abm:
            self.engine.die(uids)
        return

    def current_value(self, key, uids=None):
        """
        `rel_sus` or `rel_trans` (`key`) of each agent with the modifiers and products active on it applied (`ss.modify(dis.rel_sus, ...)`), as agents:
        what transmission uses, whereas reading `dis.rel_sus` gives the base values, which a modifier never changes. Or a parameter (`key` is its
        name) with the modifiers active on it, in its own type, declared or written as code: one value, or one for each agent if they differ between agents.

        A disease written as code that reads one in its own code (including by `transmit(rel_trans=..., rel_sus=...)`) says so in its class,
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
            return super().current_value(key, uids)
        if not isinstance(self.engine, ss.solvers.Transmission):
            raise ss.CapabilityError(f'{self.name}.current_value() is the value of each agent with the modifiers applied, which only a run as agents has; in a compartmental run, sim.what_modifies({self.name}.{key}) lists the modifiers')
        arr = self[key]
        vals = arr.asnew(arr.raw*self.engine.mult_raw(key), cls=ss.FloatArr, copy=False)
        return vals if uids is None else vals[uids]

    def not_naive(self):
        """ The agents who have left the default state (e.g. who are no longer susceptible), for dynamic rescaling (see `ss.Sim.rescale()`) """
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
        if self.graph is None or not self.graph.transmissions:
            raise ss.CapabilityError(f'disease.infect() infects people along the disease\'s declared transmission, but "{self.name}" declares none; a disease written as code infects people with its own code (e.g. ss.AgentDisease.infect())')
        if not self.abm:
            raise ss.CapabilityError(f'disease.infect() infects agents, but "{self.name}" is run with solver="{self.sim.solver}"; use ss.Event(S >> I) for a compartmental run')
        return self.engine.infect_now(ss.uids(uids), strain)

    def transmit(self, transition=None, rel_trans=None, rel_sus=None, beta=None):
        """
        Who is infected in this step by transmission: by one of the disease's declared transmissions (`transition`), or, as agents, from the
        transmissibility and susceptibility of each agent (`rel_trans` and `rel_sus`), over the sim's networks and other routes

        **Declared.** A declared disease's transmissions are its transitions whose rate depends on contact with infectious people (e.g. `S >> I @ beta*I`).
        Override this to write one as code, and call `super().transmit(transition)` for what the solver would do, e.g. to add a route of its own
        or remove some of the cases. The engine calls it once a step for each transmission, and moves those it gives along that transition,
        counted in its flow. A disease with one transmission can leave out the argument (`def transmit(self):`); with several, `transition`
        says which (e.g. `if transition.name == 'infection':`). As agents, it gives the new cases' UIDs (those not in one of the transition's
        source states are not moved); as compartments, the number infected in each group (or one number), taken from the source states in
        proportion. Under the ODE, it is taken once a step, at the start, as a move of that many (as `ss.move()` from code is), so the ODE no
        longer integrates that transition within the step: give a rate (e.g. `S >> I @ (lambda self, sim: ...)`) to keep it exact. Anything that
        needs individuals (UIDs) raises an error in a compartmental run; `if isinstance(self.sim.solver, ss.solvers.ode):` can give each its own.

        **As array operations.** With `rel_trans` and `rel_sus`, it is the same transmission as an operation that code calls, as agents (in a
        disease written as code, or in a declared one's own code): each edge of a network from `u` to `v` transmits with probability
        1 − exp(−`rel_trans[u]·rel_sus[v]`·H), where H is the hazard of the network's chance for the step (see `ss.Network.transmit()`), with the modifiers and products on
        `rel_trans` and `rel_sus` applied (so the disease says `allow_modifiers = ('rel_sus', 'rel_trans')`), and the draws of each edge from the
        disease's `trans_rng`; other routes (e.g. a mixing pool) give their own (`compute_transmission()`). Someone reached by several is infected
        by the first, which is their source, recorded with the route in an `ss.infection_log`. Nobody is moved: the code that calls it says what
        happens to those infected.

        Args:
            transition (`ss.Transition`): the declared transmission; may be left out for a disease with only one
            rel_trans (array): the transmissibility of each agent, by UID, 0 for those not infectious (e.g. `self.rel_trans.raw*self.I.raw`)
            rel_sus (array): the susceptibility of each agent, by UID, 0 for those not susceptible
            beta (float/`ss.Rate`/dict): with `rel_trans` and `rel_sus`, the beta (a rate, or a number in the sim's unit; a dict gives one per network); by default `self.pars.beta`

        Returns:
            As agents, the UIDs of the new cases (`ss.uids`); as compartments, the number of them in each group

        Examples:
            ```python
            class Imported(ss.SIR): # A declared disease, with 2 imported cases a step added to its transmission
                def transmit(self, transition):
                    new = super().transmit(transition)
                    if isinstance(self.sim.solver, ss.solvers.abm): # And 2 imported cases a step, as agents
                        return new | self.sim.people.auids[self.S.raw[self.sim.people.auids]][:2]
                    return new + 2/self.sim.groups.n_groups

            class SIR(ss.Disease): # Written as code: the S >> I @ beta*I of ss.SIR, as agents
                allow_modifiers = ('rel_sus', 'rel_trans') # Read here, by transmit()
                ...
                def step(self):
                    new = self.transmit(rel_trans=self.rel_trans.raw*self.I.raw, rel_sus=self.rel_sus.raw*self.S.raw)
                    self.S[new] = False
                    self.I[new] = True
            ```
        """
        if rel_trans is not None or rel_sus is not None:
            return self.transmit_arrays(rel_trans, rel_sus, beta)
        trs = self.graph.transmissions if self.graph is not None else []
        if transition is None:
            if len(trs) != 1:
                errormsg = f'"{self.name}" has {len(trs)} transmissions ({sc.strjoin(tr.name for tr in trs) or "none"}), so transmit() is given which one (def transmit(self, transition), with transition.name one of them), or the arrays to transmit from (transmit(rel_trans=..., rel_sus=...))'
                raise ValueError(errormsg)
            return self.engine.transmitted(trs[0])
        name = transition if isinstance(transition, str) else transition.name
        found = [tr for tr in trs if tr.name == name]
        if not found:
            raise sc.KeyNotFoundError(f'"{name}" is not a transmission of "{self.name}"; its transmissions are {sc.strjoin(tr.name for tr in trs)}')
        return self.engine.transmitted(found[0])

    def transmit_arrays(self, rel_trans, rel_sus, beta=None):
        """ The transmission of `transmit(rel_trans=..., rel_sus=...)`: the new cases, with their sources and routes recorded in an `ss.infection_log` """
        if rel_trans is None or rel_sus is None:
            raise ValueError(f'"{self.name}".transmit() transmits from both rel_trans and rel_sus (one value for each agent, by UID); give both')
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" transmits once the sim is initialized (sim.init())')
        solver = getattr(self.sim, 'solver', None) # A mock sim (init_mock()) has none, and is agents
        if solver is not None and not solver.agents:
            raise ss.CapabilityError(f'transmit(rel_trans=..., rel_sus=...) infects agents, but "{self.name}" is run with solver="{self.sim.solver}"; declare the transmission (e.g. S >> I @ beta*I) to run it as compartments')
        if self.trans_rng is None: # The draws of each edge, keyed by its source and target
            raise AttributeError(f'"{self.name}" has no trans_rng, which transmit() draws from; define it in __init__(): self.trans_rng = ss.multi_random(\'source\', \'target\')')
        if not len(self.sim.networks) and isinstance(self.sim, ss.Sim): # A mock sim (init_mock()) has no networks
            β = self.pars.beta if beta is None else beta
            value = lambda v: v.values if isinstance(v, ss.peract) else v.value if isinstance(v, ss.Rate) else v
            if any(np.any(value(v)) for v in (β.values() if isinstance(β, dict) else [β])): # Not an error if not transmitted by design (beta=0)
                errormsg = (f'"{self.name}" transmits over networks and mixing pools, but the sim has none, so it would infect no one: add one (e.g. networks="random"), or give it beta=0 if it is not transmitted. '
                             '(A declared disease, such as ss.SIR, mixes homogeneously with no networks.)')
                raise ValueError(errormsg)
        engine = self.engine
        betamap = self.route_betas() if beta is None else engine.code_betas(beta)
        if self.strains is None or self.graph is not None: # A declared disease's strains are a dimension of its states, which its own transmissions carry
            out = engine.over_routes(rel_trans, rel_sus, betamap)
            new, sources, routes = engine.first_infections([(srcs, trgs, nets) for trgs,srcs,nets in out])
        else: # Written as code with strains (ss.AgentDisease): each new case takes the strain of their source
            new, sources, routes, strain = engine.strain_cases(rel_trans, rel_sus, betamap)
            self.strain[new] = strain
        if self.infection_log is not None and len(new):
            self.infection_log.add_entries(new, sources, self.t.tvec[self.ti], network=routes)
        return new

    def route_betas(self):
        """
        The beta of each of the sim's networks (by name), as a pair (p1 to p2, p2 to p1), that `transmit(rel_trans=..., rel_sus=...)` uses when not given `beta=`: by default `pars.beta`, as a number, rate, or dict by network, checked against them

        Override it for betas worked out by code, e.g. by sex or genotype, starting from `super().route_betas()`.

        **Example**:
        ```python
        class STI(ss.AgentDisease):
            def route_betas(self):
                betamap = super().route_betas()
                betamap['sexual'] = [self.pars.beta_m2f, self.pars.beta_f2m] # Male to female and female to male, on ss.SexualNet
                return betamap
        ```
        """
        return self.engine.code_betas()

    @property
    def custom_transmit(self):
        """ Whether the disease is declared and has its own `transmit()`, which the engine calls for its transmissions """
        return self.graph is not None and type(self).transmit is not Disease.transmit

    def call_transmit(self, tr):
        """ The disease's own `transmit()` for one of its transmissions, with or without the transition as its argument (see `transmit()`) """
        if len(inspect.signature(self.transmit).parameters):
            return self.transmit(tr)
        elif len(self.graph.transmissions) > 1:
            errormsg = f'"{self.name}" has {len(self.graph.transmissions)} transmissions ({sc.strjoin(tr.name for tr in self.graph.transmissions)}), so its transmit() is given which one: def transmit(self, transition), with transition.name one of them'
            raise TypeError(errormsg)
        return self.transmit()

    def count(self, who=None, by=None):
        """
        The number of people now in a state of this disease (or for whom a condition holds), under every solver: as agents,
        each counted by their weight and the sim's scale, as its results count them

        Args:
            who (State/Cond/str/list): the state (as code sees it, `sir.I`, or by name, `'I'`), several (a list, counted together), or a condition (e.g. `sir.I & (ss.person.age < 15)`); default: all of its people
            by (str/bool): instead, the number in each level of a dimension the sim is stratified by (its name, e.g. `'age'`), or in each group (`True`), as an array

        Examples:
            ```python
            sir.count('I') # The number infectious
            sir.count([sir.I, sir.R], by='age') # The number infectious or recovered in each age group
            ```
        """
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" counts its people once the sim is initialized (sim.init())')
        if isinstance(who, (list, tuple)):
            parts = [self[x] if isinstance(x, str) else x for x in who]
            who = parts[0]
            for part in parts[1:]:
                who = who | part
        elif isinstance(who, str):
            who = self[who]
        return ss.conditions.count_now(self.sim, who, self, by=by)

    def current_state(self, states=None, uids=None):
        """
        The state each agent is in, as a read-only `ss.IntArr` (by UID) of its index among `states`, or -1 for none

        The names of the states, in order, are its `.labels`, so `arr.labels[i]` is the name of index `i`.

        Args:
            states (list): the states to look in, as names or states (default: the module's states of people, in order)
            uids (uids): instead, only these agents, as an array of their indices (in the order given), without making one for everyone

        Example:
            ```python
            cs = sir.current_state() # 0 for S, 1 for I, 2 for R
            cs.labels[cs[uid]] # The name of an agent's state
            sir.current_state(uids=uids) # The index for some agents only
            ```
        """
        graph = self.graph
        if self.engine is None and getattr(self, '_given', False):
            raise RuntimeError(f'"{self.name}" was given to a sim, which runs a copy of each module it is given (so it can be given to other sims); read the one that ran from the sim, e.g. sim.{self.name}.current_state()')
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" has the state of each agent once the sim is initialized (sim.init())')
        if isinstance(self.sim, ss.utils.Shrunk):
            raise RuntimeError(f'the state of each agent of "{self.name}" was dropped when the sim was shrunk; run it with shrink=False to keep it')
        if not self.abm:
            raise ss.CapabilityError(f'disease.current_state() is the state of each of the agents, but "{self.name}" is run with solver="{self.sim.solver}"; use the counts of its states (e.g. sim.results.{self.name}.S) instead')
        names = [s if isinstance(s, str) else s.name for s in sc.tolist(states)] if states is not None else [s.name for s in graph.people_states]
        unknown = [name for name in names if name not in graph.states]
        if unknown:
            raise sc.KeyNotFoundError(f'"{self.name}" has no state {sc.strjoin(unknown)}; its states are {sc.strjoin(graph.states.keys())}')
        rows = Ellipsis if uids is None else np.asarray(uids, dtype=np.int64) # Every slot of the arrays, or these agents
        age = self.sim.people.age # A plain array of the people, to copy
        raw = np.full(len(age.raw) if uids is None else len(rows), -1, dtype=ss.dtypes.int)
        for i,name in enumerate(names):
            raw[self[name].raw[rows]] = i
        if uids is not None:
            return raw
        raw.flags.writeable = False
        out = age.asnew(raw, cls=ss.IntArr, name='current_state', copy=False)
        out.label = f'Current state of {self.name}'
        out.nan = out.default = -1 # In none of the states
        out.labels = names
        return out

    def make_naive(self, uids):
        """
        Put agents back in the default state (e.g. S), as if they had never had the disease; for dynamic rescaling (see `ss.Sim.rescale()`)

        Args:
            uids (`ss.uids`): the agents to make naive
        """
        for state in self.graph.people_states:
            self[state.name][uids] = False
            self.engine.clear_exit(state.name, uids)
        naive = self.graph.default_state.name # As not_naive() reads it, which need not be the first state (e.g. V, S, I, R = ss.states(4) with default_state='S')
        self[naive][uids] = True
        self[f'ti_{naive}'][uids] = self.ti
        return

    def step(self):
        """ Take the declared transitions of the step, those that are not contact-mediated and then those that are, followed by any custom functions; a disease written as code overrides this """
        self.engine.step_state()
        self.engine.step()
        self.engine.step_funcs()
        if self.callbacks:
            self.run_callbacks()
        return

    def record_states(self):
        """ The counts of the states at the start of the step; for a disease written as code, the count of each `ss.BoolState` (see `ss.Module.record_states()`) """
        self.engine.record_states()
        return

    def record_events(self):
        """ The flows of the step, and the results worked out from them """
        self.engine.record_events()
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
    A transmitted disease written as code, as agents: the plain array operations of an infection, each a method to override

    Gives the `susceptible`, `infected` (and `infectious`, by default the same), `rel_sus`, `rel_trans`, and `ti_infected`
    states; seeds `init_prev` of the people at the start; and each step transmits from `transmissibility()` to
    `susceptibility()` over the sim's networks (`transmit()`), counts the new cases in the `infection` result, and gives
    them their prognoses (`set_prognoses()`). A subclass overrides only what it needs: usually `set_prognoses()`, for what
    happens to those infected, and `step_state()`, for how their states change; those infected before birth (over
    `ss.PrenatalNet`) go to `set_congenital()` instead, which a disease with outcomes at birth overrides. Anything else
    is plain code on the arrays. The betas of the networks are `route_betas()`, which can be overridden too (e.g. by sex).

    With `strains=ss.Strains(...)`, each person carries one strain at a time (`self.strain`, the index of the strain of
    their current or last infection, -1 if none), which the disease's code reads (e.g. in `set_prognoses()`, with
    `strain_value()` for a strain's values). The strains are transmitted in one pass, each by those infectious with it
    at its `rel_trans`; each new case is infected by one of those who reached them in the step, at random, and takes
    their strain, unless protected against it (`strain_sus()`: protection from `ss.immunity(due_to=<state>,
    protection=matrix)` by the strain they had, or the disease's own), as a declared disease's strains are passed on.

    An infection in a step happens at its end, so a course scheduled from it starts at the next step: schedule a move
    `dur` steps after it at `self.ti + 1 + dur`, in an `ss.TimeArr`, and take it when `due()`. The seeds are dated by the
    step before the first (`self.ti` is -1 while they are set), so they have the same course.

    A disease that can be written as states and transitions is shorter declared (see `ss.SIR`), and runs with every solver.

    Args:
        pars (dict): parameters, e.g. `beta` and `init_prev` (a fraction, or an `ss.bernoulli`)
        strains (`ss.Strains`): the strains, carried one at a time, sharing the disease's states (also `define_strains()`)
        kwargs (dict): parameters; also `rel_sus=` and `rel_trans=`, a distribution each person's value is drawn from when they are made (e.g. `rel_trans=ss.gamma(shape=0.2, scale=5)`)

    Example:
        ```python
        class SIR(ss.AgentDisease):
            beta         = ss.par(ss.perday(0.05))
            init_prev    = ss.par(0.01)
            dur_inf      = ss.par(10)
            recovered    = ss.BoolState()
            ti_recovered = ss.TimeArr()

            def step_state(self):
                done = (self.infected & self.ti_recovered.due()).uids
                self.infected[done] = False
                self.recovered[done] = True

            def set_prognoses(self, uids, sources=None):
                super().set_prognoses(uids, sources)
                self.ti_recovered[uids] = self.ti + 1 + self.pars.dur_inf

        sim = ss.Sim(diseases=SIR(), networks='random', unit='day', stop=100).run()

        # With two strains, the second more transmissible; recovering from one protects fully against it, and partly against the other
        class SIRS(SIR):
            def step_state(self):
                super().step_state()
                self.susceptible[self.recovered] = True # Protected only by the immunity of their strain

            def set_prognoses(self, uids, sources=None):
                super().set_prognoses(uids, sources)
                self.recovered[uids] = False

        strains = ss.Strains(['H1', 'H3'], rel_trans=[1, 1.5], immunity=ss.immunity(due_to='recovered', protection=[[1, 0.5], [0.5, 1]]))
        sim = ss.Sim(diseases=SIRS(strains=strains, name='flu'), networks='random', unit='day', stop=100).run()
        sim.diseases.flu.strain # The strain of each person's current or last infection
        ```
    """
    allow_modifiers = ('rel_sus', 'rel_trans') # Read by transmit(), so ss.modify() of them and products (e.g. a vaccine) are applied
    strain_dist = None # With strains, the strain of those infected with none given (see init_pre())

    def __init__(self, pars=None, strains=None, **kwargs):
        super().__init__()
        self.define_pars(init_prev=None) # None for no seeds
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infected'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'), # Or drawn from a distribution given as rel_sus= (see update_pars())
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            infectious = 'infected', # Those who transmit; by default, everyone infected
        )
        self.trans_rng = ss.multi_random('source', 'target') # The draws of transmission over each edge, keyed by its source and target
        self.update_pars(pars, strains=strains, **kwargs)
        return

    def define_strains(self, strains):
        """
        Give the disease strains (`ss.Strains`, also as `strains=`), carried one at a time: the strain of each person (`self.strain`, added when the
        sim is initialized, and set for those infected by transmission, at the start, and by `infect()`; code that infects people itself sets it)

        Args:
            strains (Strains): the strains; those infected at the start, or by `infect()` with no strain, have one drawn by its `fractions`
        """
        if not isinstance(strains, ss.Strains):
            raise TypeError(f'strains= of "{self.name}" is an ss.Strains, e.g. ss.Strains(["a", "b"], rel_trans=[1, 2]); not {type(strains).__name__}')
        if strains.max_strains > 1:
            raise ss.CapabilityError(f'"{self.name}" is written as code, so its strains are carried one at a time, sharing its states, with protection on rel_sus due to its own states; but max_strains={strains.max_strains}. Declare the disease (see ss.Disease) for more.')
        self.strains = strains # Its arrays are added when the sim is initialized (see init_pre()), so a subclass may define its own states after this
        return

    def update_pars(self, pars=None, **kwargs):
        """ As `ss.Module.update_pars()`, and `strains=` for `define_strains()`, so a subclass that passes its keywords on here takes it too (e.g. `ssl.AgentSIR(strains=...)`) """
        if isinstance(pars, dict) and 'strains' in pars: # Also in a dict of parameters
            pars = dict(pars)
            kwargs['strains'] = pars.pop('strains')
        strains = kwargs.pop('strains', None)
        if strains is not None:
            self.define_strains(strains)
        for key in ['rel_sus', 'rel_trans']: # Each person's own, drawn from a distribution when they are made, e.g. ssl.AgentSIR(rel_trans=ss.gamma(shape=0.2, scale=5))
            val = kwargs.pop(key, None)
            if isinstance(pars, dict) and isinstance(pars.get(key), ss.Dist):
                pars = dict(pars)
                val = pars.pop(key)
            if val is None:
                continue
            elif not isinstance(val, ss.Dist):
                errormsg = f'{key}= of "{self.name}" is a distribution that each person\'s {key} is drawn from, e.g. {key}=ss.gamma(shape=0.2, scale=5) (a mean of 1); not {val!r}. To scale it for everyone, scale beta instead'
                raise TypeError(errormsg)
            getattr(self, key).default = val # Its array, defined in __init__()
        return super().update_pars(pars, **kwargs)

    def strain_value(self, key, uids):
        """
        The value of `key` (given to `ss.Strains`, e.g. `dur=[0.5, 1]`) for the strain each of some agents carries (1 if they have none)

        Args:
            key (str): the name of the value
            uids (array): the agents
        """
        vals = np.asarray(self.strains.values[key], dtype=float) if key in self.strains.values else np.ones(len(self.strains))
        strain = self.strain[uids]
        return np.where(strain >= 0, vals[np.maximum(strain, 0)], 1.0)

    def strain_sus(self, k):
        """
        The fraction of their susceptibility to strain `k` that each agent (by UID) keeps: `1 − protection` from each `ss.immunity` of the strains,
        for those in its state (e.g. `due_to='recovered'`) by the strain they had; a disease with protection of its own (e.g. from antibodies) overrides this

        Args:
            k (int): the index of the strain
        """
        out = 1.0
        strain = self.strain.raw
        for imm in self.strains.immunity:
            held = np.logical_or.reduce([self[name].raw for name in imm.names]) & (strain >= 0)
            out = out*np.where(held, 1 - imm.matrix(len(self.strains))[np.maximum(strain, 0), k], 1.0)
        return out

    def init_pre(self, sim):
        if sc.isnumber(self.pars.init_prev): # A fraction, e.g. init_prev=0.01
            self.pars.init_prev = ss.bernoulli(self.pars.init_prev)
        if self.strains is not None and not isinstance(getattr(self, 'strain', None), ss.IntArr): # The strain of each person, and of those infected with none given, by the split
            n = len(self.strains)
            self.define_states(ss.IntArr('strain', default=-1, label='Strain of the current or last infection'))
            self.strain_dist = ss.choice(a=n, p=self.strains.fractions if self.strains.fractions is not None else np.full(n, 1/n))
        out = super().init_pre(sim)
        unknown = [name for imm in (self.strains.immunity if self.strains is not None else []) for name in imm.names if not isinstance(getattr(self, name, None), ss.BoolArr)]
        if unknown:
            raise sc.KeyNotFoundError(f'the strains of "{self.name}" are protected by ss.immunity(due_to={unknown[0]!r}), which is not one of its states (a boolean array, e.g. ss.BoolArr("recovered"))')
        return out

    def init_post(self):
        """ Seed `init_prev` of the people born by the start (not the fetuses of pregnancies begun before it), dated by the step before the first """
        super().init_post()
        if self.pars.init_prev is None:
            return ss.uids()
        people = self.sim.people
        seeds = self.pars.init_prev.filter(people.auids[people.age[people.auids] >= 0])
        if self.infection_log is not None: # With no source (-1)
            self.infection_log.add_entries(seeds, -1)
        saved, self.t.ti = self.t.ti, -1
        try:
            self.infect(seeds, log=False)
        finally:
            self.t.ti = saved
        return seeds

    def init_results(self):
        """ The counts of the states (one for each `ss.BoolState`), and the prevalence and new infections """
        super().init_results()
        self.define_results(
            ss.Result('prevalence', dtype=float, scale=False, label='Prevalence', kind='ratio'),
            ss.Result('infection',  dtype=int,   scale=True,  label='New infections', kind='flow'),
        )
        return

    def step(self):
        """ Transmit (`transmit()`), and infect the new cases """
        super().step() # The modifiers worked out again, once the interventions have run
        new = self.transmit()
        return self.infect(new, log=False, strain=self.strain[new] if self.strains is not None else None) # transmit() logs them, with their sources, and gives each the strain of their source

    def transmissibility(self):
        """ The transmissibility of each agent, by UID: `rel_trans`, 0 for those not infectious """
        return self.rel_trans.raw*self.infectious.raw

    def susceptibility(self):
        """ The susceptibility of each agent, by UID: `rel_sus`, 0 for those not susceptible """
        return self.rel_sus.raw*self.susceptible.raw

    def transmit(self, rel_trans=None, rel_sus=None, beta=None):
        """ Who is infected in this step: by default, from `transmissibility()` to `susceptibility()` (see `ss.Disease.transmit()`); nobody, for a disease with no `beta` (e.g. one infected only from code, with `infect()`) """
        if beta is None and 'beta' not in self.pars:
            return ss.uids()
        rel_trans = self.transmissibility() if rel_trans is None else rel_trans
        rel_sus = self.susceptibility() if rel_sus is None else rel_sus
        return super().transmit(rel_trans=rel_trans, rel_sus=rel_sus, beta=beta)

    def infect(self, uids, log=True, strain=None):
        """
        Infect some agents now: those susceptible, counted in `infection` (except the seeds, which are prevalent), and given their prognoses

        Args:
            uids (uids): the agents (e.g. imported cases); those not susceptible are left as they are
            log (bool): record them in an `ss.infection_log`, with no source (-1)
            strain (str/int/array): with strains, the strain they are infected with: its name or index, or one for each agent; by default drawn by the strains' `fractions`

        Returns:
            The agents infected

        Example:
            ```python
            flu.infect(sim.people.auids[:20], strain='H3') # 20 imported cases of H3
            ```
        """
        uids = ss.uids(uids)
        if strain is not None and self.strains is None:
            raise ss.CapabilityError(f'"{self.name}" has no strains (strains=ss.Strains(...)), so infect() takes no strain=')
        ok = self.susceptible[uids]
        uids = uids[ok]
        if self.strains is not None: # Each one's strain, before their prognoses, which may read it (strain_value())
            if strain is None:
                strain = self.strain_dist.rvs(uids)
            elif isinstance(strain, str):
                if strain not in self.strains.names:
                    raise sc.KeyNotFoundError(f'"{strain}" is not a strain of "{self.name}"; its strains are {sc.strjoin(self.strains.names)}')
                strain = self.strains.names.index(strain)
            elif not sc.isnumber(strain):
                strain = np.asarray(strain)[ok]
            self.strain[uids] = strain
        if log and self.infection_log is not None:
            self.infection_log.add_entries(uids, -1)
        if self.sim.initialized:
            ss.results.add_count(self.results.infection, self.ti, uids, self.sim)
        self.set_outcomes(uids)
        return uids

    def set_outcomes(self, uids, sources=None):
        """
        Give those just infected their outcomes: those infected before birth (age 0 or less, e.g. over `ss.PrenatalNet`) by `set_congenital()`, the rest by `set_prognoses()`

        Args:
            uids (uids): the agents infected
            sources (array): who infected them, if known
        """
        unborn = self.sim.people.age[uids] <= 0
        if np.any(unborn):
            self.set_congenital(uids[unborn], None if sources is None else sources[unborn])
            uids, sources = uids[~unborn], None if sources is None else sources[~unborn]
        self.set_prognoses(uids, sources)
        return

    def set_congenital(self, uids, sources=None):
        """
        What happens to those infected before birth (over `ss.PrenatalNet`; see `set_outcomes()`): by default, the same as anyone infected (`set_prognoses()`)

        A disease with outcomes at birth overrides this, e.g. `ssl.CongenitalDisease` (`import starsim.library as ssl`), which schedules a
        birth outcome (a stillbirth, a congenital infection) for each.

        Args:
            uids (uids): the people infected before birth
            sources (array): who infected them, if known
        """
        return self.set_prognoses(uids, sources)

    def set_prognoses(self, uids, sources=None):
        """
        What happens to those infected, at the end of this step: by default, they are infected and no longer susceptible; override this for more

        Args:
            uids (uids): the agents infected
            sources (array): who infected them, if known
        """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        return

    def record_states(self):
        """ The counts of the states, and the prevalence, at the start of the step """
        super().record_states()
        res = self.results
        alive = self.sim.people._n_agents_alive() if ss.results.agent_weights(self.sim) is None else self.sim.results.alive[self.sim.ti] # With agents of different weights, as the people alive are counted
        res.prevalence[self.ti] = res.infected[self.ti] / alive
        return

    def not_naive(self):
        """ The agents who have ever been infected, for dynamic rescaling (see `ss.Sim.rescale()`) """
        return self.ti_infected.notnan.uids

    def make_naive(self, uids, skip_states=('rel_sus', 'rel_trans')):
        """
        Reset agents to never having had the disease, for dynamic rescaling: every state to its default, except `skip_states`

        Args:
            uids (uids): the agents
            skip_states (list): the names of the states to keep (by default `rel_sus` and `rel_trans`, which other modules, e.g. vaccines, may have set)
        """
        for state in self.state_list:
            if state.name not in skip_states:
                state.set(uids)
        return


class InfectionLog:
    """
    A record of who infected whom, and when: kept by a disease when an analyzer needs it (e.g. `ss.infection_log`), to trace chains of transmission

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

    records extra data for each infection (`ss.Disease.transmit()` records the network this way).
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
        networks (list): the names of the networks, used to convert the network IDs recorded by `ss.Disease.transmit()` to names in `to_df()`
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
        times = f', t={self.chunks[0]["t"][0]}–{self.chunks[-1]["t"][0]}' if self.n else ''
        return f'InfectionLog(disease={self.disease!r}, n={self.n}{times})'

    def disp(self, **kwargs):
        """ Full display of the infection log

        Args:
            kwargs (dict): passed to `sc.pr()`
        """
        return sc.pr(self, **kwargs)

    def add_entries(self, uids, sources=None, time=None, **kwargs):
        """
        Record new infections

        Args:
            uids (array): the UIDs of the infected agents (targets)
            sources (array/int): the UIDs of the infecting agents; None or NaN for seed infections, or -1 (as a declared disease records its seeds)
            time (any): the time of infection (all entries in the chunk share it); by default the time point of the current step (the start, for seeds set before the first)
            kwargs (dict): extra data to store, either a scalar or one value per UID
        """
        n = len(uids)
        if n:
            if time is None:
                time = np.nan if self.timeline is None else self.timeline.tvec[max(self.timeline.ti, 0)]
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
                    self.define_results(ss.Result('known', label='Infections from a known source', kind='flow'))

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
                    if k not in df.columns or df[k].dtype != object:
                        df[k] = df[k].astype(object) if k in df.columns else None # Object dtype, so any value can be stored (e.g. a name, such as 'seed_infection', in a column of network IDs)
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
    the infection unless `p_death` is given. Its defaults are those of a respiratory
    infection: infectious for 10 days, with an R0 of 3 without a network (β × the
    duration); with a network, β is per contact, so R0 is that many times the number
    of contacts.

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks, over each step of the contact, so `ss.probperday(p)` gives each contact a chance p in a day; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float): the fraction of people who start infected (for a number of people, give `init_n=` instead, e.g. `init_n=10`)
        dur_inf (float/`ss.Duration`/`ss.Dist`): the mean time that people are infected for (by default 10 days), as an exponential; give a distribution (e.g. `ss.lognormal(mean=ss.days(10))`) for another shape
        p_death (float): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`
        pars (dict): the same parameters, as a dict

    An SIR model written as code, with a lognormal `dur_inf` and `p_death=0.01`,
    is `ssl.AgentSIR`.

    Example:
        ```python
        sim = ss.Sim(diseases=ss.SIR(beta=0.03, dur_inf=ss.days(10)), networks='random', n_agents=5000) # R0 of 3 with 10 contacts each
        sim.run().plot()
        ```
    """
    beta      = ss.par(ss.perday(0.3))
    dur_inf   = ss.par(ss.days(10))
    p_death   = ss.par(0.0)
    S         = ss.State()
    I         = ss.State(infected=True) # Counted in the results infected and prevalence
    R         = ss.State()
    infection = ss.Transition(S, I, rate=beta * I)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.dead, dur=dur_inf, p=p_death)


class SEIR(Disease):
    """
    An SEIR model: as `ss.SIR`, with an exposed (latent) state before people become infectious

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks, over each step of the contact, so `ss.probperday(p)` gives each contact a chance p in a day; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float): the fraction of people who start infected, who start exposed, in `E` (for a number of people, give `init_n=` instead); to start people infectious, give `init=dict(I=0.01)` or `init_n=dict(I=10)` instead
        dur_exp (float/`ss.Duration`/`ss.Dist`): the mean time that people are exposed (latent) for (by default 3 days; a number is in the model's unit), as an exponential
        dur_inf (float/`ss.Duration`/`ss.Dist`): the mean time that people are infectious for (by default 10 days), as an exponential
        p_death (float): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`
        pars (dict): the same parameters, as a dict

    As in `ss.SIR`, the defaults are those of a respiratory infection, each duration is
    exponential unless a distribution is given, and nobody dies of the infection unless
    `p_death` is given.

    An SEIR model written as code, with lognormal durations and `p_death=0.01`, is
    `ssl.AgentSEIR`.
    """
    beta      = ss.par(ss.perday(0.3))
    dur_exp   = ss.par(ss.days(3))
    dur_inf   = ss.par(ss.days(10))
    p_death   = ss.par(0.0)
    S         = ss.State()
    E, I      = ss.states('E I', infected=True) # Counted in the results infected and prevalence
    R         = ss.State()
    infection = ss.Transition(S, E, rate=beta * I)
    latency   = ss.Transition(E, I, dur=dur_exp)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.dead, dur=dur_inf, p=p_death)


class SIS(Disease):
    """
    An SIS model: people recover from infection and are susceptible again

    Declared from states and transitions (see `ss.Disease`), so it runs with any solver.
    Its defaults are those of a respiratory infection that leaves no immunity: infectious
    for 10 days, with an R0 of 2 without a network, so that half of people are infected
    once it is endemic.

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks, over each step of the contact, so `ss.probperday(p)` gives each contact a chance p in a day; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float): the fraction of people who start infected (for a number of people, give `init_n=` instead, e.g. `init_n=10`)
        dur_inf (float/`ss.Duration`/`ss.Dist`): the mean time that people are infected for (by default 10 days), as an exponential unless a distribution is given
        kwargs (dict): passed to `ss.Disease`, e.g. `name`
        pars (dict): the same parameters, as a dict

    An SIS model with a per-agent immunity level, boosted by each infection and
    waning over time (`waning=`, `imm_boost=`), which has no declared form, is
    `ssl.AgentSIS`, written as code.
    """
    beta      = ss.par(ss.perday(0.2))
    dur_inf   = ss.par(ss.days(10))
    S         = ss.State()
    I         = ss.State(infected=True) # Counted in the results infected and prevalence
    infection = ss.Transition(S, I, rate=beta * I)
    recovery  = ss.Transition(I, S, dur=dur_inf)
