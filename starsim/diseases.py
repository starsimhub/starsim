"""
Base classes for diseases
"""
import functools
import inspect
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

ss_int = ss.dtypes.int
_ = None # For function signatures



def built_after(init):
    """ Wrap the __init__() of a subclass of ss.Disease, so that what it declares (`define_states()`, `define_transitions()`) is built once the outermost __init__() has finished """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        depth = self.__dict__.get('_init_depth', 0) # How many __init__()s deep: a subclass's calls its parent's
        self.__dict__['_init_depth'] = depth + 1
        try:
            init(self, *args, **kwargs)
        finally:
            self.__dict__['_init_depth'] = depth
        if depth == 0:
            del self.__dict__['_init_depth']
            if '_declared' in self.__dict__:
                self.build_graph()
        return
    return __init__


class Disease(ss.Module):
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
    `ss.Infection` adds transmission written as code.

    Note: CRN safe as agents.

    Args:
        args (list): transitions and outcomes, named from the variables they are assigned to
        pars (dict): parameter values by name
        transitions (list): transitions, as a list (for longer models)
        results (list): extra results: sums and ratios of states
        outcomes (list): `ss.outcome` objects
        init_prev (float): the initial prevalence, seeded through the transmission (by default 0.01, unless a state is given an initial value)
        init (dict): the share of people who start in each state, e.g. `dict(S=0.90, I=0.01, R=0.09)`; each value can be an `ss.by` or a function of the sim
        init_n (dict): instead, the number of people who start in each state, e.g. `dict(I=1)` for one person
        first (str): the state that everyone not put elsewhere starts in, and is born into; by default, the one named `S` or `susceptible`, else the first state of people made (in the class form, `first = 'N'` in the class body)
        strains (Strains): its strains (`ss.Strains`): in its states, which they share, or each with a course of its own; then `dis.strains`
        stratify (dict/list): as `stratify()`, e.g. `dict(age={0: 0.3, 15: 0.6, 65: 0.1})`; a list of dicts for one call each
        kwargs (dict): one keyword per state, transition, parameter, result, or function; also `name`, `label`, and `dt`

    The results are the number in each state (`I`), the flow along each transition by its
    name (`infection`, with its cumulative total as `res.infection.cum`), and the results and
    outcomes declared (e.g. `prevalence=I/(S + I + R)`, as `ss.SIR` declares).

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
    engine = None # Runs the declared part (see ss.solvers.make_engine()); created when the sim is initialized
    infection_log = None # An InfectionLog, if the sim has an ss.infection_log analyzer
    code_pars = () # Parameters of a declared module that its own code reads rather than its graph (e.g. ss.Pregnancy's fertility rates)
    reads_modifiers = () # For a disease written as code: which of rel_sus and rel_trans its own code reads, with modified(), so ss.modify() of them is applied (an ss.Infection with a beta reads both)

    _infection_methods = ['set_prognoses', 'transmit', 'set_outcomes', 'finalize_infections', 'compute_transmission', 'transmit_route', 'set_congenital'] # Methods of ss.Infection's transmission, which a declared transition never calls

    def __init_subclass__(cls, **kwargs):
        """ A class that declares states or transitions takes its transitions through its engine, which never calls the methods of ss.Infection's transmission, so overriding one (e.g. set_prognoses() on a subclass of ss.SIS) would be silently ignored """
        super().__init_subclass__(**kwargs)
        overridden = [name for name in cls._infection_methods if name in vars(cls)]
        infection = any(b.__name__ == 'Infection' and b.__module__ == __name__ for b in cls.__mro__)
        if overridden and not infection and ss.transitions.class_members(cls):
            base = next((f'ssl.Agent{b.__name__}' for b in cls.__mro__ if b.__module__ == __name__ and b.__name__ in ['SIR', 'SIS', 'SEIR']), 'ss.Infection')
            errormsg = (f'{cls.__name__} declares states or transitions and defines {sc.strjoin(overridden)}, which a declared transition never calls, so it would be silently ignored.\n'
                        f'     A disease whose transmission is written as code subclasses {base} (import starsim.library as ssl); a declared one changes its transitions.')
            raise ss.CapabilityError(errormsg)
        rebound = [name for name in dict.fromkeys(k for base in cls.__mro__[1:] for k,v in vars(base).items() if isinstance(v, ss.par)) if name in vars(cls) and not isinstance(vars(cls)[name], (ss.par, ss.expressions.Expr))]
        if rebound: # A declared parameter rebound to something else (a method, a number) is dropped from the model, and its transitions keep the base class's parameter, so it would be silently ignored
            name = rebound[0]
            how = (f'give the parameter a function as its value instead, e.g. self.pars.update({name}=self.my_{name}) in __init__(), where my_{name}(sim, uids) returns the value for those agents' if callable(vars(cls)[name]) else
                   f'to change its default, give the value in __init__(), e.g. super().__init__(**({{"{name}": {vars(cls)[name]!r}}} | kwargs))')
            errormsg = f'{cls.__name__} defines {sc.strjoin(rebound)}, a parameter of {cls.__mro__[1].__name__}, as {"a method" if callable(vars(cls)[name]) else "a value"}, which its transitions never read; {how}'
            raise ss.CapabilityError(errormsg)
        init = vars(cls).get('__init__')
        if init is not None: # Its states and transitions, declared in __init__() (see define_transitions()), are built once the outermost __init__() has finished
            cls.__init__ = built_after(init)
        infect = vars(cls).get('infect')
        if infect is not None and infection and len(inspect.signature(infect).parameters) == 1: # Starsim 3's transmission pass, infect(self), which is transmit() now: it would never be called
            errormsg = f'{cls.__name__} defines infect(self), the transmission pass, which is called transmit() now (infect(uids) infects people from code); rename it to transmit(), and infect_route() to transmit_route()'
            raise ss.CapabilityError(errormsg)
        return

    def __init__(self, *args, pars=None, strains=None, stratify=None, _names=None, **kwargs):
        super().__init__()
        self.strains = None # Its ss.Strains, if it has strains
        self.family = None # For strains with courses of their own: the module that runs them (see ss.strains.Family)
        self.callbacks = [] # Functions called with the agents an event happened to: see add_callback()
        if len(args) and (args[0] is None or isinstance(args[0], dict)): # Parameters can also be the first argument
            pars, args = args[0], args[1:]
        if strains is not None and getattr(strains, 'members', None) is not None: # Strains written as diseases of their own
            return ss.strains.attach_members(self, strains, args, pars, kwargs)
        members = ss.transitions.class_members(type(self))
        declared = (ss.State, ss.Transition, ss.outcome, ss.expressions.Expr, ss.transitions.FlowSum)
        given = args or members or any((isinstance(v, declared) and not isinstance(v, ss.par)) or key in ss.graph.Graph.grouped for key,v in kwargs.items()) # Its structure; a parameter's value (e.g. with bounds) is not
        later = strains is not None or stratify is not None or any(key in kwargs for key in ['init', 'init_n', 'init_prev', 'first']) or any(isinstance(v, ss.par) for v in kwargs.values()) # Only for a declared disease, so its states and transitions follow, from define_states() and define_transitions()
        if given or later:
            names = (_names or ss.utils.caller_names(1)) if given else {} # The names of the variables of the code that made it; or, made by the sim from the transitions it was given, of the code that made the sim
            if not given: # A parameter's value given by keyword (e.g. with bounds) sets the parameter of that name that the transitions declared next use, as pars= does
                pars = sc.mergedicts(pars, {key:kwargs.pop(key) for key in list(kwargs) if isinstance(kwargs[key], ss.par)})
            self._declared = sc.objdict(args=list(args), members=members, pars=pars, kwargs=kwargs, names=names, strains=strains, stratify=stratify, code_pars=[])
            if given: # Built now, so a subclass's __init__() can use it (e.g. self.pars) after super().__init__()
                self.build_graph()
            else: # Built once the subclass's __init__() has declared the rest
                self.set_metadata(**{key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs})
        else: # Written as code: its own define_pars() and define_states() follow, as in Starsim 3; or declared by define_states() and define_transitions()
            self.update_pars(pars, **kwargs)
        return

    @property
    def graph(self): # Built when the disease is made (see built_after()); here too, in case something reads it while it is being made
        """ Its declared states and transitions (`ss.graph.Graph`), built from what was declared when first needed; None for a disease written as code """
        if '_declared' in self.__dict__:
            self.build_graph()
        return self.__dict__.get('_graph')

    @graph.setter
    def graph(self, graph):
        self.__dict__['_graph'] = graph

    def build_graph(self):
        """ Build the graph from what was declared: when the disease is made, or for one declared in its __init__() (see `define_transitions()`), when the graph is first needed """
        d = self.__dict__.pop('_declared')
        if d.strains is not None and getattr(d.strains, 'shared', None) is True: # An outcome for each strain carried, before the transitions are built
            ss.strains.prepare(d.strains, [(x.name, x) for x in d.args + sc.tolist(d.kwargs.get('transitions'))] + list(d.kwargs.items()) + list(d.members.items()))
        self.init_graph(d.args, d.members, d.pars, d.kwargs, names=d.names)
        self.add_code_pars(d.code_pars)
        if d.strains is not None:
            ss.strains.attach(self, d.strains)
        for call in sc.tolist(d.stratify):
            self.stratify(**call)
        return

    def declare(self, args=(), kwargs=None, names=None, overwrite=False):
        """ Add declared members (states, transitions, results), to be built with the rest; see `define_transitions()` """
        if self.__dict__.get('_graph') is not None:
            errormsg = f'"{self.name}" has already been built, so nothing more can be declared in it: declare its states, transitions, and results while it is being made, in __init__()'
            raise ss.CapabilityError(errormsg)
        d = self.__dict__.get('_declared')
        if d is None: # The first declaration of a disease that called super().__init__() with none
            d = self._declared = sc.objdict(args=[], members=ss.transitions.class_members(type(self)), pars=None, kwargs={}, names={}, strains=None, stratify=None, code_pars=[])
        names = names or {}
        kwargs = kwargs or {}
        for obj in args: # Positional members, named by name= or by the variable they were assigned to
            name = getattr(obj, 'name', None) or names.get(id(obj))
            if name is None:
                d.args.append(obj)
            else:
                kwargs = {name:obj} | kwargs # By name, so it can replace one
        reserved = {key for cls in type(self).__mro__ for key,val in vars(cls).items() if callable(val) or isinstance(val, property)} | set(dir(ss.Module))
        for name,obj in kwargs.items():
            if name in reserved:
                raise ValueError(f'"{name}" cannot be used as the name of a state, transition, or result of "{self.name}", since it is an attribute of every module or of {type(self).__name__}. Please choose another name.')
            if not overwrite and (name in d.kwargs or name in d.members):
                raise ValueError(f'"{name}" is already declared in "{self.name}"; names must be unique within a module (or give overwrite=True to replace it)')
            d.kwargs[name] = obj
            self.setattribute(name, obj) # So the declarations that follow can use it, e.g. self.S >> self.I
        d.names.update(names)
        return

    def define_states(self, *args, overwrite=False, **kwargs):
        """
        Define its states: declared states (`ss.State`), or for a disease written as code, arrays (see `ss.Module.define_states()`)

        Declared states are given while the disease is being made, in its `__init__()`, together with
        its transitions (`define_transitions()`), so that which states it has can depend on its
        arguments. Each is an attribute at once (`self.susceptible`), for the transitions to use.

        Args:
            args (list): the states (`ss.State('S')`, named), or lists of them; or arrays (`ss.BoolState`, ...)
            overwrite (bool): replace a state of the same name, rather than raising
            kwargs (dict): states by name (`S=ss.State(label='Susceptible')`); or for arrays, aliases

        Examples:
            ```python
            class TB(ss.Disease):
                def __init__(self, relapse=False, **kwargs):
                    super().__init__(**kwargs)
                    self.define_states(
                        ss.State('susceptible', label='Susceptible'),
                        ss.State('infection',   label='Latent infection'),
                        ss.State('active',      label='Active TB'),
                    )
                    if relapse: # A state only some versions of the model have
                        self.define_states(ss.State('relapsing', label='Will relapse'))
            ```
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
        Define its transitions, while the disease is being made (in its `__init__()`), after its states

        For a declared disease whose structure depends on its arguments, so that it cannot be written
        in a class body: the states (`define_states()`), transitions, and results (`define_results()`)
        are declared on `self`, and built together when the disease is first used. Each is an
        attribute at once (`self.infection`).

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
        """
        declared = (ss.expressions.Expr, ss.outcome, ss.transitions.FlowSum)
        if not kwargs and not any(isinstance(arg, declared) for arg in args):
            return super().define_results(*args)
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

    def __getattr__(self, attr):
        """ For strains with courses of their own, what is over all of them, e.g. `where=hpv.infected` (see `ss.strains.Family`) """
        family = self.__dict__.get('family')
        if family is not None and not attr.startswith('_'):
            return getattr(family, attr)
        return super().__getattr__(attr)

    def expand(self):
        """ The modules the sim takes for the disease: itself, or for strains with courses of their own, one disease per strain and the module that runs them together """
        return [self] if self.family is None else self.family.expand()

    def define_pars(self, inherit=True, **kwargs):
        """
        Create or merge parameters; see `ss.Module.define_pars()`

        For a declared disease, a parameter defined here that is not one of its graph's (e.g. by a subclass, after
        `super().__init__()`, for its own code to read) is added to `code_pars`, so the sim does not reject it as one
        that nothing reads. A parameter set any other way (e.g. a misspelling in `pars=`) is still rejected.
        """
        out = super().define_pars(inherit=inherit, **kwargs)
        if '_declared' in vars(self): # Being declared: sorted once it is built, so more can still be declared after this
            self._declared.code_pars += list(kwargs)
        elif self.graph is not None:
            self.add_code_pars(kwargs)
        return out

    def add_code_pars(self, keys):
        """ Parameters its own code reads, not its graph's: see `define_pars()` """
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
                    ends = [st for st in (obj.src or []) + (obj.dst or []) if isinstance(st, ss.State)]
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
        """ A parameter given by keyword under the name it already has (name=), and read by a transition, as if it were a result (e.g. eir=eir) is an error: a result needs a name of its own (MS11) """
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
        init_prev = kwargs.pop('init_prev', None)
        init = kwargs.pop('init', None)
        init_n = kwargs.pop('init_n', None)
        first = kwargs.pop('first') if isinstance(kwargs.get('first'), str) else getattr(type(self), 'first', None) # A state named "first" can still be declared by keyword; in the class form, first = 'N' in the class body
        first = first if isinstance(first, str) else None
        self.set_metadata(**meta)
        self.t.update(**timepars)
        holds_members = lambda leaf: isinstance(leaf.raw, ss.by) and any(v is m for v in leaf.raw.items for m in members.values()) # A value by level holding the class's parameters (ss.by(sex=dict(female=p_f, male=p_m))), copied with them
        def leaves(expr): # Including those inside a parameter that holds an expression, as a dwell time written as an expression is held (e.g. dur=d*ss.days(1), with d = ss.par(ss.lognorm_im(...)))
            for leaf in expr.leaves:
                yield leaf
                if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.expressions.Expr):
                    yield from leaves(leaf.raw)
        holds_expr = lambda leaf: isinstance(leaf.raw, ss.expressions.Expr) and any(inner is m for inner in leaves(leaf.raw) for m in members.values()) # One holding the class's parameters, copied with them
        memo = {id(leaf):leaf for val in members.values() if isinstance(val, ss.Transition) for expr in [val.rate, val.dur, getattr(val, 'p_die', None), getattr(val, 'p_stay', None)] + sc.tolist(getattr(val, 'p', None)) if isinstance(expr, ss.expressions.Expr) for leaf in leaves(expr) if isinstance(leaf, ss.par) and not any(leaf is m for m in members.values()) and not holds_members(leaf) and not holds_expr(leaf)}
        members = sc.dcp(members, memo=memo) # Copy class members (together, so they still refer to each other) so instances do not share them; a parameter defined outside the class (rate = ss.par(...) at module level) stays shared, as in the operator form, so it keeps its name
        kwargs = {k:ss.expressions.rebind(v, memo) for k,v in kwargs.items()} # A result given here that reads the class's states reads the copies
        init, init_n = [{k:ss.expressions.rebind(v, memo) for k,v in d.items()} if isinstance(d, dict) else d for d in [init, init_n]] # Likewise a starting value, e.g. init=dict(S=1 - (A - age)/(k*age + A)) with the class's parameters
        for key in [key for key,val in kwargs.items() if isinstance(val, ss.par) and isinstance(members.get(key), ss.par)]: # e.g. ss.SIS(beta=ss.par(0.1, bounds=[0.05, 1])): the class's parameter, which its transitions use, takes the value and bounds
            members[key].merge(kwargs.pop(key))
        self.check_par_members(args, members, kwargs)
        members = sc.mergedicts(members, kwargs)
        args, members = self.reuse(args, members, names)
        self.graph = ss.graph.Graph(self, args, members, init_prev=init_prev, init=init, init_n=init_n, first=first, names=names)
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
                raise sc.KeyNotFoundError(f'"{self.name}" has no parameter named "{key}". Its parameters are {sc.strjoin(self.pars.keys())}{ss.expressions.did_you_mean(key, self.pars.keys())}')
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
            split (list): the share of the population in each level, unless the levels are a dict of them (if not given: from a named population or geography; for age groups, the world's ages; for a boolean dimension, or one that people are moved along, everyone at the first level; otherwise equal)
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
            sir.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust={'infection': [1.0, 3.0]})
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
                mod.setattribute(strat.name, ss.conditions.Cond('dim', strat.name))
        return mod

    @property
    def transitions(self):
        """ The table of declared transitions; printing it also shows the matrix of sources by destinations """
        return self.graph.transitions

    @property
    def states(self):
        """ Its declared states (`ss.State`) by name, e.g. `'second_line' in tb.states` or `[s for s in tb.states.values() if s.people]`; empty for a disease written as code. (The arrays behind them are `state_dict`.) """
        return sc.objdict(self.graph.states) if self.graph is not None else sc.objdict()

    def count(self, where=None):
        """
        The number of people now for whom a condition holds, under every solver (as agents, each counted by their weight and the sim's scale)

        For a function of the sim that needs a current number, e.g. a delivery's capacity toward a target:
        `capacity=lambda sim: max(0.3*hiv.count(hiv.infected) - hiv.count(hiv.care == 'art'), 0)`.

        Args:
            where (Cond/State): the condition, e.g. `hiv.L | hiv.A` or `hiv.care == 'art'` (default: everyone)
        """
        selector = self.engine.selector
        sel = selector.select(where)
        if selector.abm:
            sel = np.asarray(sel, dtype=bool) & self.sim.people.alive.raw
            weights = ss.results.agent_weights(self.sim)
            n = np.sum(weights[sel]) if weights is not None else np.count_nonzero(sel)
            return float(n*self.sim.current_scale)
        return float(np.sum(sel*self.engine.state_counts()))

    @property
    def strats(self):
        """ Its stratifications by name, e.g. `tb.strats.age.edges` or `tb.strats.risk.labels` (as a condition, a level is `tb.age`) """
        return sc.objdict({strat.name:strat for strat in self.graph.strats}) if self.graph is not None else sc.objdict()

    def describe(self, output=False, out=None):
        """
        Print the resolved model, with every guessed default marked

        Args:
            output (bool): if True, return the text rather than printing it
            out (`ss.presentation.Description`): if given, add the description to it and return it
        """
        desc = ss.presentation.Description() if out is None else out
        if self.family is not None: # Strains with courses of their own
            self.family.describe(out=desc)
        elif self.graph is not None:
            self.graph.describe(self.sim, out=desc)
            if self.strains is not None:
                self.strains.describe(self.graph, out=desc)
        else: # Written as code: its class, arrays, and parameters
            desc.section(f"{type(self).__name__} '{self.name}'", f'written as code, with {len(self.state_list)} arrays')
            desc.add('arrays', ', '.join(s.name for s in self.state_list))
            desc.add('parameters', ', '.join(self.pars.keys()))
        if out is not None:
            return desc
        return str(desc) if output else desc.show()

    def equations(self, output=False):
        """
        Print the ODE system of the declared transitions, with symbolic parameter names

        Args:
            output (bool): if True, return the text rather than printing it
        """
        text = '\n'.join(self.graph.equations(self.engine))
        return text if output else print(text)

    def diagram(self):
        """
        The state-transition diagram, as a Mermaid flowchart (https://mermaid.js.org), which a notebook draws (see `ss.presentation.Diagram`)

        People states are rounded and counts slanted; transmissions are dashed; each
        arrow is labelled with its transition's name and value (`recovery: 10 days`), and
        the share that takes it for a transition with `p=`. `print(sir.diagram())` gives the text.

        **Example**:

            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=ss.days(10)))
            sir.diagram()
        """
        return ss.presentation.Diagram('\n'.join(['flowchart LR'] + self.graph.diagram()))

    def R0(self, matrix=False):
        """
        The basic reproduction number, from the next-generation matrix at the disease-free state

        By definition it is in a population with no immunity: everyone is taken to be in the first
        state (e.g. S), each group keeping its size, whoever starts immune (`init=dict(R=0.9)`).
        The reproduction number at the start, with that immunity, is in the line a run prints
        as it starts, when it differs (`R0 = 15.2 (R = 1.52 at the start, with 10% susceptible)`).

        The model is laid out as an ODE: if the disease is in a sim, a copy of that sim
        is made with `solver='ode'`, with its mixing and stratification; otherwise a sim with
        the disease alone, under homogeneous mixing, with the default number of people. Values that vary in time are taken at the start.
        A sim runs a copy of the disease it is given, so the R0 of a sim's model is `sim.diseases.<name>.R0()`;
        this matters where a count is a number rather than a share of the people (mosquitoes per person).

        This is the largest eigenvalue of the next-generation matrix. Where infection
        passes through a vector (people infect mosquitoes, which infect people), each
        passage is a generation, so R0 is the square root of the classical Ross–Macdonald
        value. Shedding into a reservoir that is a count (such as water) is not an infection,
        so there R0 is the number of people infected through the reservoir by one person.

        Args:
            matrix (bool): if True, return the next-generation matrix (over the infected compartments of every group) rather than its largest eigenvalue

        **Example**:

            S, I, R = ss.states(3)
            sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10))
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
        The disease's part of the model spec (`sim.to_spec()`): its states, parameters, and transitions as plain data, for `sim.hash()`, `ss.diff()`, and `sim.to_code()`

        Args:
            filename (str): if provided, save to this file
            kwargs (dict): passed to `sc.savejson()`
        """
        out = self.graph.to_spec(self.sim) if self.graph is not None else super().to_spec() # Written as code: see ss.Module.to_spec()
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim, create its engine and objects, initialize results, and start its infection log if there is an `ss.infection_log` analyzer; see Module.init_pre() for details """
        if self.graph is not None and any(st.given_init for st in self.graph.states.values()): # A starting value set after the disease was made (e.g. by an importer) turns off the default seeding
            self.graph.unset_default_init_prev()
        unknown = [key for key in self.pars if key not in self.graph.pars and key not in self.code_pars] if self.graph is not None else [] # Added after the disease was made, so nothing reads it (TL58: pars.init was ignored)
        if unknown:
            hint = ' The starting values are given when the disease is made, e.g. ss.Disease(..., init=dict(I=0.01)) or init_n=dict(I=100), and cannot be set through pars.' if 'init' in unknown or 'init_n' in unknown else ''
            raise sc.KeyNotFoundError(f'"{self.name}" has no parameter named {sc.strjoin(unknown)}, so nothing would read it. Its parameters are {sc.strjoin(self.graph.pars.keys())}{ss.expressions.did_you_mean(unknown[0], self.graph.pars.keys())}{hint}')
        self.engine = ss.solvers.make_engine(self, sim)
        self.engine.define()
        needs = ss.interventions.memory_needs(sim) if isinstance(sim, ss.Sim) and self.graph is not None and self.graph.transitions else {} # Not a mock sim
        self.remembered = [name for mod,name in needs if mod == self.name and name in self.graph.transitions] # Transitions whose memory something reads, e.g. tb.diagnosis.since
        for name in self.remembered:
            self.define_states(ss.FloatArr(f'last_{name}', label=f'Time index of the last {name}'), ss.FloatArr(f'count_{name}', default=0, label=f'Number of times: {name}'))
        super().init_pre(sim)
        if any(isinstance(a, ss.infection_log) for a in sim.analyzers.values()):
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys())
        self.engine.init_pre()
        return

    def init_post(self):
        """ Initialize the values of the states, and set the initial conditions of the declared states """
        super().init_post()
        self.engine.init_post()
        if self.callbacks and not isinstance(self.engine, ss.solvers.Agents):
            raise ss.CapabilityError(f'"{self.name}" has callbacks (add_callback()), which are called with the agents an event happened to, so it cannot run with solver="{self.sim.solver}". Run with solver="abm", or use ss.on() in a declared module.')
        return

    # The hooks the solvers call on a declared module, for one that needs more than its states and transitions (e.g. ss.Pregnancy)
    def compartmental_form(self, sim):
        """ A module that runs only as agents unless it defines this: make its small compartmental form, used only when a compartmental solver is named, and return the note that says what it leaves out """
        return None

    def burnin_plan(self, engine):
        """ People part way through at the start: None for none (the default), or the plan for them (see `ss.Pregnancy.burnin_plan()` and `ss.solvers.Agents.backfill_burnin()`) """
        return None

    def make_children(self, tr, parents, ti):
        """ As agents, the children of the parents who went through a transition with births (`ss.birth(per=)`); returns their UIDs """
        return self.engine.grow_children(tr, parents, ti)

    def also_happened(self, name, uids):
        """ As agents, others an event (a transition, by name) also happens to, as UIDs (e.g. the fetuses of a mother at a delivery); None for nobody else """
        return None

    def add_callback(self, event, fn):
        """
        Call `fn(uids)` at the end of each step with the agents who went through a declared transition or event of this disease in it, from a module written as code (agents only)

        A declared module reacts to the same events with `ss.on()` instead. Pass a bound method (e.g. `self.on_delivery`), or register
        from `init_pre()`: the sim runs its own copy of each module, and a lambda made in `__init__()` would still refer to the original.

        Args:
            event (Transition/str): the transition or event, e.g. `preg.stillbirth`, `preg.loss`, or its name
            fn (func): called with the agents' UIDs (once for each time, so an agent can appear twice), only on a step when there are some

        **Example**:

            preg.add_callback(preg.stillbirth, lambda uids: self.n_stillbirths.__setitem__(uids, self.n_stillbirths[uids] + 1))
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

    def step_state(self):
        """
        State changes at the start of the step, before the networks and interventions, by a disease written as code (e.g. recovery)

        The declared transitions are all taken together in `step()`, after the interventions,
        so a delivery sees the same people under every solver (as agents and as compartments).
        """
        return

    def step_die(self, uids):
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
        """
        if isinstance(self.engine, ss.solvers.Agents):
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
            self.infection_log.add_entries(uids, sources, self.t.tvec[self.ti]) # At the step's time point, as ti_infected is
        return

    def modified(self, key, uids=None, vals=None):
        """
        `rel_sus` or `rel_trans` (`key`) times the modifiers active on it (`ss.modify(dis.rel_sus, ...)`, and products), for a disease written as code that reads it

        An `ss.Infection` with a `beta` reads both in its transmission (`transmit_route()`). Any other disease written as
        code that reads one says so in its class, `reads_modifiers = ('rel_sus',)`, so that modifiers of it are applied
        rather than refused.

        Args:
            key (str): 'rel_sus' or 'rel_trans'
            uids (uids): the agents to give the values of (default: an array of every agent, by UID)
            vals (Arr): the values to multiply (default: the disease's own `rel_sus` or `rel_trans`)

        **Example**::

            class Diabetes(ss.Disease):
                reads_modifiers = ('rel_sus',) # ss.modify(diabetes.rel_sus, ...) scales its onset
                def step(self):
                    uids = self.at_risk.uids
                    onset = self.pars.p_onset.rvs(uids)*self.modified('rel_sus', uids) # ...
        """
        arr = self[key]
        vals = arr if vals is None else vals
        if isinstance(arr, ss.arrays.ModifiableArr) and arr.engine is not None:
            vals = vals.asnew(vals.raw*arr.engine.mult_raw(key), copy=False)
        return vals if uids is None else vals[uids]

    def not_naive(self):
        """ The agents who have left the first state (e.g. who are no longer susceptible), for dynamic rescaling (see `ss.Sim.rescale()`) """
        return (~self[self.graph.first.name]).uids

    def move(self, *args, **kwargs):
        """
        Move people between this disease's states from code, the same way under every solver: `ss.move()` with the disease given

        **Examples**::

            sir.move(sir.I, sir.R, p=0.5) # Cure half of those infectious
            sir.move(uids, sir.R) # These agents, from wherever they are (or sir.move(uids, 'R'))
            self.move(self.I, self.R, uids=uids) # In the disease's own code: these agents, if they are infectious
        """
        return ss.move(self, *args, **kwargs)

    def infect(self, uids, strain=None):
        """
        Infect some agents now, from code (e.g. a strain introduced on a date): those in the first state move where its transmission puts them, counted in its flow and dated as its infections are (`ti_I == sim.ti`), with a source of -1 in an `ss.infection_log`

        Args:
            uids (uids): the agents; those not in the first state are left where they are
            strain (str/Strain): for a disease with strains in shared states, the strain they are infected with (by default, as the split says)

        Returns:
            The agents infected

        **Example**::

            covid.infect(sim.people.auids[:20], strain='alpha') # 20 imported cases of alpha
        """
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" infects people once the sim is initialized (sim.init())')
        if self.sim.solver != 'abm':
            raise ss.CapabilityError(f'disease.infect() infects agents, but "{self.name}" is run with solver="{self.sim.solver}"; use ss.Delivery(move=S >> I) for a compartmental run')
        return self.engine.infect_now(ss.uids(uids), strain)

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
        graph = self.graph
        if self.engine is None:
            raise RuntimeError(f'"{self.name}" has the state of each agent once the sim is initialized (sim.init())')
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

    def update_results(self):
        """ The counts of the states and the flows; for a disease written as code, the count of each `ss.BoolState` (see `ss.Module.update_results()`) """
        self.engine.update_results()
        return

    def finalize_results(self):
        super().finalize_results()
        self.engine.finalize_results()
        return

    def shrink(self):
        super().shrink()
        self.engine.shrink()
        return


class Infection(Disease):
    """
    A disease with transmission written as code

    An `ss.Disease` that adds network-based transmission with directional beta values
    (`transmit()`, `transmit_route()`), the `susceptible` and `infected` states, the attributes
    that other modules act on (`rel_sus`, `rel_trans`), `set_prognoses()` for the people
    infected, and the `prevalence` and `new_infections` results. The rest of the disease is
    written as for any other module: its own per-agent states (`define_states()`) and how
    they change (`step_state()`, `set_prognoses()`), so it can hold anything an agent can
    have: a CD4 count, an immunity level that wanes, a time since infection. It runs as
    agents. A disease with no `beta` (such as `ssl.NCD`) is not transmitted; one that has no
    use for these states at all is a plain `ss.Disease`.

    A disease that can be written as states and transitions is declared instead, with
    `ss.Disease` (see `ss.SIR`), which gives it every solver and `describe()`, `equations()`,
    and `R0()`. The v4 verbs that act on arrays work on both: `ss.modify(hiv.rel_trans, ...)`,
    conditions such as `hiv.cd4 < 200`, and fitting the values in `pars`.

    Note: CRN safe.

    Examples:
        ```python
        class SI(ss.Infection):
            def __init__(self, **kwargs):
                super().__init__()
                self.define_pars(beta=ss.peryear(0.5), init_prev=ss.bernoulli(0.01))
                self.update_pars(**kwargs)
        ```
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
        Set initial values for states: by default, infect `init_prev` of the people (`set_prognoses()`)

        The initial infections are dated by the first step: in `set_prognoses()`, `self.ti` is 1 (not 0, the
        start of the run), as it is for those infected in the first step, so a disease that schedules from
        `self.ti` (e.g. `self.ti_recovered[uids] = self.ti + dur`) gives them the same course, as Starsim 3 did.
        """
        super().init_post()
        if self.pars.init_prev is None:
            return

        people = self.sim.people
        unborn = people.unborn_uids() if isinstance(people, ss.People) else ss.uids() # E.g. the fetuses of pregnancies that ss.Pregnancy starts before the start, who are not seeded
        initial_cases = self.pars.init_prev.filter() if not len(unborn) else self.pars.init_prev.filter(people.auids[people.age.raw[people.auids] >= 0])
        if len(initial_cases):
            ti = self.t.ti
            self.t.ti = min(1, self.t.npts-1) # Dated by the first step, as an infection in it is, so code that schedules from self.ti (e.g. ti_recovered = self.ti + dur) gives them as long as Starsim 3 did, not a step less (CV2)
            try:
                self.set_prognoses(initial_cases, sources=-1)  # -1 = externally seeded infection with no source agent
            finally:
                self.t.ti = ti
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

        nets = {ss.utils.standardize_netkey(k):net for k,net in sim.networks.items()}
        def both_ways(k, β): # A single beta applies in both directions, except on a maternal network: from mother to child only
            net = nets.get(ss.utils.standardize_netkey(k))
            return [β, β*0] if getattr(net, 'one_way', False) else [β, β]

        def as_pair(β, nkey):
            """ A probability per act is one value for both [p1 to p2, p2 to p1], as Starsim 3's pair; with directions, an array of a value for each of the network's (see `transmit_route()`) """
            if not isinstance(β, ss.peract):
                return β
            net = nets.get(nkey)
            vals = β.resolve(net) if net is not None else β.values
            if len(vals) == 1:
                return vals*2
            return [np.asarray(vals, dtype=float)]*2

        # If beta is a scalar, apply this bi-directionally to all networks
        if scalar_beta(β) or isinstance(β, ss.peract):
            betamap = {ss.utils.standardize_netkey(k):both_ways(k, β) if scalar_beta(β) else as_pair(β, ss.utils.standardize_netkey(k)) for k in sim.networks.keys()}

        # If beta is a dict, check all entries are bi-directional; a default is the beta of every network it does not name, and the names that are not networks are not used
        elif isinstance(β, dict):
            betamap = dict()
            if 'default' in β:
                unknown = [k for k in β if k != 'default' and ss.utils.standardize_netkey(k) not in nets]
                if unknown:
                    ss.note(f'the beta of "{self.name}" names {sc.strjoin(unknown)}, which {"is not a network" if len(unknown) == 1 else "are not networks"} of the sim, so {"it is" if len(unknown) == 1 else "they are"} not used; the networks not named take the default', sim=sim)
                β = {k:ss.utils.route_beta(β, k) for k in sim.networks.keys()}
            for k,thisbeta in β.items():
                nkey = ss.utils.standardize_netkey(k)
                if scalar_beta(thisbeta):
                    betamap[nkey] = both_ways(nkey, thisbeta)
                else:
                    betamap[nkey] = as_pair(thisbeta, nkey)

        else:
            errormsg = f'Invalid type {type(β)} for beta'
            raise TypeError(errormsg)

        # Check that it matches the network
        netkeys = [ss.utils.standardize_netkey(k) for k in list(sim.networks.keys())]
        if set(betamap.keys()) != set(netkeys):
            missing = sorted(set(netkeys) - set(betamap.keys()))
            extra = sorted(set(betamap.keys()) - set(netkeys))
            errormsg = f'Network keys ({netkeys}) and beta keys ({list(betamap.keys())}) do not match for disease "{self.name}"'
            if missing: errormsg += f'; missing beta for network(s) {missing}'
            if extra:   errormsg += f'; no network(s) found matching {extra}'
            raise ValueError(errormsg)

        return betamap

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
                self.results.new_infections[self.ti] += len(uids) # Count infections when they happen, rather than inferring them from ti_infected
            else:
                ss.results.as_float(self.results.new_infections)[self.ti] += weights[uids].sum()
        return

    def infect(self, uids, strain=None):
        """
        Infect some agents now, from code (e.g. a strain introduced on a date): those susceptible get the outcomes that transmission gives (`set_outcomes()`), counted in `new_infections`, with a source of -1 in an `ss.infection_log`

        A disease that declares its transmission infects them along it instead (see `ss.Disease.infect()`).

        Args:
            uids (uids): the agents; those not susceptible are left as they are
            strain (str): for a declared disease with strains only

        Returns:
            The agents infected

        **Example**::

            hiv.infect(sim.people.auids[:20]) # 20 imported cases
        """
        if self.graph is not None and self.graph.transmissions:
            return super().infect(uids, strain)
        if not self.sim or not self.sim.initialized:
            raise RuntimeError(f'"{self.name}" infects people once the sim is initialized (sim.init())')
        if strain is not None:
            raise ss.CapabilityError(f'"{self.name}" is written as code, so infect() takes no strain=; set the strain in its own states')
        uids = ss.uids(uids)
        uids = uids[self.susceptible[uids]]
        if len(uids):
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
        Determine who gets infected on this timestep via transmission on the network

        Computes the effective transmissibility and susceptibility, calls `transmit_route()`
        for each network (or other route), and then removes duplicate infections with
        `finalize_infections()`. A multi-strain disease can override this method to loop
        over strains, calling `transmit_route()` with per-strain values.
        """
        if 'beta' not in self.pars: # A disease that is not transmitted, such as ssl.NCD, has no beta
            return self.finalize_infections([], [], [])
        betamap = self.validate_beta()

        # Compute effective transmissibility and susceptibility directly on the raw
        # (full-length) arrays. This avoids the gather/scatter, full-length astype copy,
        # and extra wrapper allocations of the Arr math operators; edges only ever index
        # living agents, so stale raw values for inactive agents are never used.
        rel_trans = self.rel_trans.asnew(self.infectious.raw * self.rel_trans.raw, copy=False) # The modifiers are applied by transmit_route()
        rel_sus   = self.rel_sus.asnew(self.susceptible.raw * self.rel_sus.raw, copy=False)

        new_cases = []
        sources = []
        networks = []
        for i, (nkey,route) in enumerate(self.sim.networks.items()):
            betas = betamap[ss.utils.standardize_netkey(nkey)]
            target_uids, source_uids, network_ids = self.transmit_route(i, route, betas, rel_trans, rel_sus)
            new_cases.append(target_uids)
            sources.append(source_uids)
            networks.append(network_ids)

        return self.finalize_infections(new_cases, sources, networks)

    def transmit_route(self, i, route, betas, rel_trans, rel_sus):
        """
        Compute the transmission along a single network (or other route)

        On a network, each edge transmits with probability `rel_trans·rel_sus·net_beta()` (see `ss.Network.transmit()`), times the modifiers
        of the network's contacts and of `rel_sus` on the route (`ss.modify(..., route=)`), as for a declared disease run as agents. The modifiers
        of `rel_trans` and `rel_sus` (including products) are applied here, so a disease that overrides `transmit()` has them too.

        Args:
            i (int): the index of the route in `sim.networks`, stored as the network ID of each infection
            route (`ss.networks.Route`): the network or other route (e.g. a mixing pool)
            betas (list): the pair of betas for the route (p1→p2 and p2→p1), each a number or rate, or an array of a probability per act for each direction of the network; routes other than networks only use the first
            rel_trans (`ss.FloatArr`): the effective transmissibility of each agent (0 if not infectious)
            rel_sus (`ss.FloatArr`): the effective susceptibility of each agent (0 if not susceptible)

        Returns:
            A tuple of the target UIDs, the source UIDs, and the network IDs of the new infections (possibly with duplicates)
        """
        rel_trans = self.modified('rel_trans', vals=rel_trans)
        rel_sus = self.modified('rel_sus', vals=rel_sus)
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
            def edge_beta(d, inds, src, trg): # For the edges that might transmit (see ss.Network.transmit())
                beta = betas[d]
                if isinstance(beta, np.ndarray): # A probability per act for each direction of the network
                    beta = beta[route.edge_directions(*(route.ends(d) if route.custom_beta else (src, trg)))] # For every edge, for a net_beta() of its own
                elif not beta: # Skip directions with no transmission
                    return None
                disease_beta = beta.to_prob(self.t.dt) if isinstance(beta, ss.Rate) else beta
                beta_per_dt = route.edge_net_beta(inds, disease_beta=disease_beta, disease=self) # Compute beta for this network and timestep
                m = mult(inds, trg)
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
            new_cases (list): the target UIDs from each call to `transmit_route()`
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
        super().update_results()
        res = self.results
        ti = self.ti
        alive = self.sim.people.n_alive if ss.results.agent_weights(self.sim) is None else self.sim.results.alive[self.sim.ti] # With agents of different weights, people alive as results count them (recorded just before)
        res.prevalence[ti] = res.infected[ti] / alive
        return

    def finalize_results(self):
        """ Compute cumulative infections from the new-infections timeseries. """
        super().finalize_results() # Called first to scale the results
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
        if 'network' in df.columns and df['network'].isna().all(): # No route is known (e.g. homogeneous mixing, with no networks): no column of blanks
            df = df.drop(columns='network')
        df['t'] = [float(t.value) if isinstance(t, ss.dur) else t for t in df['t']] # Plain numbers (on the disease's own axis), or dates, rather than durations, so times can be subtracted

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
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected (or a number of people)
        dur_inf (float/`ss.dur`/`ss.Dist`): the mean time that people are infected for (by default 6 years; a number is in years), as an exponential; give a distribution (e.g. `ss.lognorm_ex(mean=ss.years(6))`) for another shape
        p_death (float/`ss.bernoulli`): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    An SIR model written as code, with a lognormal `dur_inf` and `p_death=0.01`,
    is `ssl.AgentSIR`.

    **Example**:

        sim = ss.Sim(diseases=ss.SIR(beta=0.1, dur_inf=ss.years(6)), networks='random', n_agents=5000)
        sim.run().plot()
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_inf   = ss.par(ss.years(6))
    p_death   = ss.par(0.0)
    S, I, R   = ss.states('S I R')
    infection = ss.Transition(S, I, rate=beta * I)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.null, dur=dur_inf, p=p_death)
    prevalence = I/(S + I + R)

    def __init__(self, pars=None, **kwargs):
        default_beta = 'beta' not in kwargs and 'beta' not in (pars or {})
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        self.graph.pars['beta'].library_default = default_beta # Read as a probability per act on a network with acts, as in Starsim 3 (see ss.solvers.check_betas())
        return


class SEIR(Disease):
    """
    An SEIR model: as `ss.SIR`, with an exposed (latent) state before people become infectious

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected, who start exposed, in `E` (or a number of people); to start people infectious, give `init=dict(I=0.01)` or `init_n=dict(I=10)` instead
        dur_exp (float/`ss.dur`/`ss.Dist`): the mean time that people are exposed (latent) for (by default 0.5 years; a number is in years), as an exponential
        dur_inf (float/`ss.dur`/`ss.Dist`): the mean time that people are infectious for (by default 6 years), as an exponential
        p_death (float/`ss.bernoulli`): the probability of death from infection, at the end of the infection (by default 0)
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    As in `ss.SIR`, each duration is exponential unless a distribution is given, and
    nobody dies of the infection unless `p_death` is given.

    An SEIR model written as code, with lognormal durations and `p_death=0.01`, is
    `ssl.AgentSEIR`.
    """
    beta      = ss.par(ss.peryear(0.1))
    dur_exp   = ss.par(ss.years(0.5))
    dur_inf   = ss.par(ss.years(6))
    p_death   = ss.par(0.0)
    S, E, I, R = ss.states('S E I R')
    infection = ss.Transition(S, E, rate=beta * I)
    latency   = ss.Transition(E, I, dur=dur_exp)
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.null, dur=dur_inf, p=p_death)
    prevalence = (E + I)/(S + E + I + R)

    def __init__(self, pars=None, **kwargs):
        default_beta = 'beta' not in kwargs and 'beta' not in (pars or {})
        super().__init__(pars, **library_kwargs(type(self), kwargs))
        self.graph.pars['beta'].library_default = default_beta # Read as a probability per act on a network with acts, as in Starsim 3 (see ss.solvers.check_betas())
        return



class SIS(Disease):
    """
    An SIS model: people recover from infection and are susceptible again

    Declared from states and transitions (see `ss.Disease`), so it runs with any solver.

    Note: CRN safe as agents.

    Args:
        beta (float/`ss.per`/dict): the transmission rate (per contact, if there are networks; a dict gives one per network, and `default=` for the networks it does not name)
        init_prev (float/`ss.bernoulli`/`ss.choose_n`): the fraction of people who start infected (or a number of people)
        dur_inf (float/`ss.dur`/`ss.Dist`): the mean time that people are infected for (by default 10 years; a number is in years), as an exponential unless a distribution is given
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    An SIS model with a per-agent immunity level, boosted by each infection and
    waning over time (`waning=`, `imm_boost=`), which has no declared form, is
    `ssl.AgentSIS`, written as code.
    """
    beta      = ss.par(ss.peryear(0.05))
    dur_inf   = ss.par(ss.years(10))
    S, I      = ss.states('S I')
    infection = ss.Transition(S, I, rate=beta * I)
    recovery  = ss.Transition(I, S, dur=dur_inf)
    prevalence = I/(S + I)

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
    if 'init' not in kwargs and 'init_n' not in kwargs: # A parameter, so it can be set (sir.pars.init_prev = 0.05) or fitted like the others
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
