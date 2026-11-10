"""
Not for the user: the graph of one module: its states, transitions, and parameters, collected, named, completed with defaults (each guess recorded), and checked

It exists as the one description of a declared model, which every solver runs and `describe()` shows, so that a model can be written once.

Used with: agents and compartments; declared modules
"""
import types
import itertools
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import as_expr, as_share, BinOp, Count, count_power, dependence, did_you_mean, factors, is_total, Expr, Func, is_one, linear, par, per_person, unheld
from .parameters import by, TimeValue
from ..transitions import FlowSum, outcome, State, Transition

__all__ = ['Graph']


def inner_pars(val):
    """ The parameters inside a value: of a value that varies in time (`ss.seasonal(mean=beta)`), a probability per act (`ss.peract(m2f=b)`), a value by level (`ss.by(sex=dict(female=p_f, male=p_m))`), or a distribution (`ss.constant(dur_inc)`), also within a beta for each route """
    if isinstance(val, (TimeValue, ss.peract)):
        return val.pars
    elif isinstance(val, ss.Dist):
        return [q for v in (val.pars or {}).values() for q in ([v] if isinstance(v, par) else inner_pars(v))]
    elif isinstance(val, ss.by):
        return [q for v in val.items for q in ([v] if isinstance(v, par) else inner_pars(v))]
    elif isinstance(val, dict):
        return [q for v in val.values() for q in inner_pars(v)]
    return []


def bad_end(end, side):
    """ The message for a source or destination of a transition that is not a state: what to write instead """
    stay = ss.transitions.stay
    outside = isinstance(end, ss.transitions.Outside)
    if outside and side == 'source':
        return f'{end} cannot be one of several sources of a transition: for an inflow, give it alone, e.g. ss.Transition({end}, S, rate=0.1)'
    elif end is stay and side == 'source':
        return 'ss.stay is a destination, not a source: give it as one destination of a split, e.g. ss.Transition(I, {R: 0.3, ss.stay: None}, dur=10)'
    elif outside or end is stay:
        return f'{end} is one destination of a split, with at least one other, e.g. ss.Transition(I, {{R: 0.9, {end}: 0.1}}, dur=10)'
    if isinstance(end, str):
        return f'the {side} of a transition is a state, not its name "{end}": declare the states and give them, e.g. S, I, R = ss.states(3) and ss.Transition(S, I, rate=beta * I)'
    if isinstance(end, ss.conditions.Cond):
        names = [name for _,name in ss.conditions.leaf_states(end)]
    else:
        names = [leaf.name for leaf in getattr(end, 'leaves', []) if isinstance(leaf, State)]
    names = list(dict.fromkeys(name for name in names if name is not None))
    if not names:
        return f'the {side} of a transition is a state or a list of states, not {end}'
    elif side == 'source':
        return f'the source of a transition is a state or a list of states, not {end}; for people moving out of any of them, write [{", ".join(names)}] or the sum {" + ".join(names)}'
    return f'the destination of a transition is a state, or a dict of states and their probabilities (a split), not {end}; for a split between them, write {{{", ".join(f"{n}: ..." for n in names)}}}'


class TransitionTable(sc.obj):
    """ The transitions of a module; printing it shows one row per transition, and the matrix of sources by destinations """
    def __repr__(self):
        if not len(self):
            return 'No transitions'
        rows = [['name', 'from', 'to', 'driven by', 'kind']]
        for name,tr in self.items():
            src = ' | '.join(s.name for s in tr.source) or '(outside)'
            dst = ' | '.join(s.name for s in tr.dest) or '(outside)'
            counted = tr.module.graph.counted(tr) if getattr(getattr(tr, 'module', None), 'graph', None) is not None else []
            kind = f'transmission, from {" + ".join(s.name for s in tr.terms)}' if tr.contact else f'transmission, from {" + ".join(s.name for s in counted)} ({tr.module.graph.dependence(tr)}-dependent)' if counted else 'spontaneous'
            rows.append([name, src, dst, tr._driver_str, kind])
        widths = [max(len(row[i]) for row in rows) + 2 for i in range(5)]
        out = '\n'.join(''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in rows)

        # The matrix
        names = list(dict.fromkeys(s.name for tr in self.values() for s in tr.source + tr.dest))
        grid = {(a,b):[] for a in names for b in names}
        for name,tr in self.items():
            for a in tr.source:
                for b in tr.dest:
                    grid[(a.name, b.name)].append(name)
        cells = [[''] + names] + [[a] + [', '.join(grid[(a,b)]) or '.' for b in names] for a in names]
        widths = [max(len(row[i]) for row in cells) + 2 for i in range(len(names) + 1)]
        out += '\n\n' + '\n'.join(''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in cells)
        return out


def is_zero(rate):
    """ Whether a transition's rate is a 0 written as a number (`I >> R @ 0`), not a parameter, which marks it as moved only by products, deliveries, or code """
    return isinstance(rate, ss.expressions.par) and rate.literal and sc.isnumber(rate.value) and rate.value == 0


class Graph:
    """
    The states and transitions of one module

    The graph is built when the module is constructed. It names everything, works out
    which states are people and which transitions are transmissions (from their rates),
    turns inline values into named parameters, and checks the result. Every guess is recorded, in `graph.guessed` and in the `guessed` of each state and transition.

    This class is not usually used directly: see `ss.Disease`.

    Args:
        module (`ss.Module`): the module the graph belongs to
        args (list): transitions and outcomes passed positionally
        members (dict): states, transitions, parameters, results, and functions, by name
        init_prev (float): the initial prevalence, seeded through the transmission (by default 0.01, unless a state is given an initial value)
        init (dict): the share of people who start in each state, by name
        init_n (dict): the number of people who start in each state, by name
        seed_n (bool): whether `init_prev` is a number of people (given as `init_n=10`) rather than a share
        default_state (str): the state people are in unless they are put elsewhere (by default, the one named `S` or `susceptible`, else the first made; see `resolve_people()`)
        names (dict): the names of the variables of the code that made the module, by the id of what they hold (see `ss.utils.caller_names()`), which name the states, transitions, and parameters given without one
    """
    grouped = ['transitions', 'outcomes', 'results'] # The keywords that each take several members, the keyword form of giving them positionally: ss.Disease(transitions=[infection, recovery]) is ss.Disease(infection, recovery)

    def __init__(self, module, args=None, members=None, init_prev=None, init=None, init_n=None, default_state=None, names=None, seed_n=False):
        self.module = module
        self.names = names or {}
        self.default_given = default_state # The state people are in unless they are put elsewhere, if given
        self.init_prev = init_prev # A share of people seeded through the transmission, or a number of them (init_n=10), if seed_n
        self.seed_n = seed_n
        if isinstance(init_prev, par): # So it can be fitted
            init_prev.name = init_prev.name or ('init_n' if seed_n else 'init_prev')
        name = getattr(module, 'name', None) or type(module).__name__
        if init is not None and not isinstance(init, dict):
            errormsg = f'init= of "{name}" is a dict of the starting value of each state, e.g. init=dict(I=0.01), not {init!r}; a share of people seeded through the transmission is init_prev={init!r}'
            raise TypeError(errormsg)
        if init_n is not None and not isinstance(init_n, dict): # A number is a seeding, turned into init_prev before here (see ss.Disease.init_graph())
            errormsg = f'init_n= of "{name}" is a number of people seeded through the transmission, or a dict of the number in each state, e.g. init_n=dict(I=10), not {init_n!r}'
            raise TypeError(errormsg)
        self.init = dict(init or {})
        self.init_n = dict(init_n or {})
        self.states = sc.obj()
        self.transitions = TransitionTable()
        self.pars = sc.obj()
        self.outcomes = sc.obj()
        self.modifiers = sc.obj() # What modifies its quantities: see ss.modify()
        self.exprs = sc.obj() # Results that are sums or ratios of states
        self.funcs = sc.obj() # Custom per-step behavior
        self.attrs = sc.obj() # Values for each person: see ss.attr
        self.derived = [] # Stratifications by a derived attribute, which add no groups: see stratify()
        self.guessed = sc.obj()
        self.strats = [] # Stratifications, filled by stratify()
        self.transfers = sc.obj() # Movement between the levels of a stratification
        self.marked_results = set() # The results in exprs worked out from the infected states (infected, prevalence): see resolve_derived()

        self.collect(list(args or []), dict(members or {}))
        if isinstance(init_prev, par):
            self._add_par(init_prev, name=init_prev.name)
        self.resolve_states()
        self.resolve_transitions()
        self.resolve_init_prev()
        self.resolve_people()
        self.resolve_attrs()
        self.resolve_acquisition()
        self.resolve_pars()
        self.set_keywords()
        self.resolve_exits()
        self.resolve_derived()
        self.check()
        return

    def taken(self, name):
        """ Whether a member of the module has this name """
        return any(name in other for other in [self.states, self.transitions, self.pars, self.outcomes, self.exprs, self.funcs, self.transfers, self.attrs, self.modifiers])

    def resolve_derived(self):
        """
        Which states are infected, and their results. The infected states are those the model says are infected (marked
        `ss.State(infected=True)`, or in its own result `infected = E + I`), exactly; or, if it says none are, those of a
        force of infection: the states people are infected into, and the sources (`E` and `I` in `S >> E @ beta*I`). With
        neither, as in a staged condition such as a cancer, no state is infected. Each state's `infected` is set to match,
        and the infected states give the results `infected`, the number in them, and `prevalence`, their share of the
        people in every state, unless the module defines its own of that name, or has a state called `infected` (whose
        count has the name: it can define its own prevalence)
        """
        if self.marked:
            found = {id(s) for s in self.marked if s.people} # As the model says, and nothing else
        elif self.infection_sources:
            found = {id(s) for s in self.infected if s.people} # The states people are infected into and the sources (see infected)
        else:
            found = set() # Not the states a staged condition reaches (see infected): it is not an infection unless the model says so
        for state in self.states.values():
            state.infected = id(state) in found
        marked = [state for state in self.people_states if state.infected]
        if not marked or 'infected' in self.states:
            return
        cls = type(self.module)
        if not self.taken('infected') and not hasattr(cls, 'infected'):
            self.add(self.exprs, 'infected', ss.expressions.as_stocks(sum(marked[1:], marked[0])))
            self.marked_results.add('infected')
        if not self.taken('prevalence') and not hasattr(cls, 'prevalence'):
            self.add(self.exprs, 'prevalence', ss.expressions.as_stocks(sum(marked[1:], marked[0])/sum(self.people_states[1:], self.people_states[0])))
            self.marked_results.add('prevalence')
        return

    def __repr__(self):
        return f'Graph(states={list(self.states.keys())}, transitions={list(self.transitions.keys())})'

    def collect(self, args, members):
        """ Sort the members by what they are, and name them """
        module = self.module
        named = [] # (name, obj) pairs
        for key in self.grouped:
            args = args + sc.tolist(members.pop(key, None))
        for arg in args:
            named.append((None, arg))
        named += list(members.items())

        self._declared = [] # States that were passed in directly
        self._keywords = dict() # Values given as keywords, e.g. ss.Disease(..., beta=0.3)
        unnamed = sc.autolist()
        given = {name for name,_ in named if name is not None}
        for name,obj in named:
            if isinstance(obj, FlowSum): # A sum or a share of flows, e.g. recovery + death or 0.5 * infection, is an outcome
                obj = outcome(obj)
            if isinstance(obj, ss.modify): # Part of the model, given to the sim with it (see ss.solvers.choose.init_where())
                self.add(self.modifiers, name or self.names.get(id(obj)) or obj.name, obj)
                continue
            if isinstance(obj, ss.expressions.attr):
                if name is not None:
                    obj.name = name
                elif obj.name is None or obj.name == '<lambda>':
                    obj.name = self.names.get(id(obj))
                if obj.name is None:
                    errormsg = f'Could not find a name for {obj}; pass it as a keyword, e.g. ss.Disease(..., cd4=ss.attr(...))'
                    raise ValueError(errormsg)
                self.add(self.attrs, obj.name, obj)
                continue
            if isinstance(obj, ss.conditions.Cond) and obj.op == '|' and (name or self.names.get(id(obj))) is not None and ss.conditions.state_objects(obj) is not None: # A named set of states, e.g. infected = E | I, is a result: the number in them, as for E + I
                name = name or self.names.get(id(obj))
                states = ss.conditions.state_objects(obj)
                self.add(self.exprs, name, ss.expressions.as_stocks(sum(states[1:], states[0])))
                continue
            if isinstance(obj, State) and None not in (name, obj.name) and name != obj.name: # A named state under another name, e.g. n_infected = I, is a result: the number in the state
                self.add(self.exprs, name, ss.expressions.as_stocks(obj))
                continue
            if isinstance(obj, par) and name is not None and obj.name not in (None, name) and (any(p is obj for p in self.pars.values()) or obj.name in given): # Another name for a parameter declared under its own name (which a keyword may give a value), e.g. A = saturation_age in a class body: the same parameter, not a second one
                continue
            if isinstance(obj, (Transition, outcome, par, State)):
                if name is not None:
                    obj.name = name
                elif obj.name is None:
                    obj.name = self.names.get(id(obj))
            if isinstance(obj, ss.dimensions.Transfer):
                obj.name = name or obj.name or self.names.get(id(obj))
                self.add(self.transfers, obj.name, obj)
            elif isinstance(obj, State):
                self._declared.append(obj)
            elif isinstance(obj, Transition):
                unnamed += obj
            elif isinstance(obj, outcome):
                if obj.name is None:
                    errormsg = 'Could not find a name for an outcome; pass it as a keyword, e.g. ss.Disease(..., cases=ss.outcome(...))'
                    raise ValueError(errormsg)
                self.add(self.outcomes, obj.name, obj)
            elif isinstance(obj, par):
                self._add_par(obj, name=obj.name)
            elif isinstance(obj, Expr) and ss.expressions.per_person_in(obj): # A helper with a value for each person, e.g. gain = cd4_potential - cd4_preart, age = ss.person.age/A, or duration = affected.time_in, used through the objects; not a result
                continue
            elif isinstance(obj, Expr): # A sum or ratio of states is a result
                name = name or self.names.get(id(obj))
                if name is None:
                    errormsg = f'Could not find a name for the result "{obj}"; pass it as a keyword, e.g. ss.Disease(..., prevalence=I/N)'
                    raise ValueError(errormsg)
                self.add(self.exprs, name, ss.expressions.as_stocks(obj))
            elif callable(obj) and not isinstance(obj, (ss.Dist, type, TimeValue)): # A value that varies in time is a value, e.g. beta=ss.seasonal(...)
                self.add(self.funcs, name or obj.__name__, obj)
            elif name is not None: # A number, typed quantity, or distribution sets the value of a parameter, once they are known
                self._keywords[name] = obj
            else:
                errormsg = f'Could not understand the argument {obj} of type {type(obj)}; expecting a transition or an outcome'
                raise TypeError(errormsg)

        # Name the transitions: explicit names first, so fallback names cannot take them
        self._unnamed = [tr for tr in unnamed if tr.name is None]
        for tr in unnamed:
            if tr.name is not None:
                self.add(self.transitions, tr.name, tr)
        return

    def add(self, container, name, obj):
        """ Add a named member, checking the name """
        reserved = set(self.grouped) | set(dir(ss.Module)) | {'sim', 'pars', 't', 'dists', 'name', 'label', 'graph', 'engine'} # The attributes of every module
        if container is not self.pars: # And this module's methods, e.g. a transition named "describe" would replace the method (parameters are kept in pars, so may be named "R0")
            reserved |= {key for cls in type(self.module).__mro__ for key,val in vars(cls).items() if isinstance(val, (types.FunctionType, property, classmethod, staticmethod))}
        if name in reserved:
            raise ValueError(f'"{name}" cannot be used as the name of a state, transition, parameter, or result, since it is an attribute of every module. Please choose another name.')
        for other in [self.states, self.transitions, self.pars, self.outcomes, self.exprs, self.funcs, self.transfers, self.attrs, self.modifiers]:
            if name in other and other[name] is not obj:
                raise ValueError(f'two members of "{self.module.name}" are both named "{name}"; names must be unique within a module.')
        container[name] = obj
        return

    def _add_par(self, obj, name):
        """ Register a parameter with the module """
        obj.name = name
        obj.literal = False
        if obj.module is None: # A parameter shared by several modules belongs to the first, and is registered with each
            obj.module = self.module
        self.add(self.pars, name, obj)
        return

    @staticmethod
    def check_read(state):
        """ Refuse a state that an earlier module only read in a rate, so took as a state of its own that nothing fills, e.g. `Rb.frac` in disease `a`, declared before the disease `b` that `Rb` was meant to belong to """
        graph = getattr(state.module, 'graph', None)
        if graph is not None and any(ref is state for ref in getattr(graph, '_refs', [])):
            owner = state.module.name
            errormsg = (f'the state "{state.name}" is read in a rate of "{owner}", which was declared first, so "{owner}" took it as a state of its own, which none of its transitions fill.\n'
                        f'     Declare this disease first, and then "{owner}" with a rate that reads its state, e.g. "beta * I * (1 - 0.5*b.{state.name}.frac)" if this disease is b.')
            raise ValueError(errormsg)
        return

    def resolve_states(self):
        """ Find the states: the ones declared, and those the transitions use """
        trs = list(self.transitions.values()) + self._unnamed
        found = {id(s):s for s in self._declared}
        self._refs = [] # States used only inside a rate
        for tr in trs:
            if tr.rate is not None:
                tr.rate = unheld(tr.rate) # A rate held in a parameter, e.g. ss.par(beta*I), is that rate
            for side,ends in [('source', tr.source or []), ('destination', tr.dest or [])]:
                for end in ends:
                    if not isinstance(end, State):
                        raise TypeError(bad_end(end, side))
                    found[id(end)] = end
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p):
                if expr is not None:
                    for leaf in expr.leaves:
                        for s in ([leaf] if isinstance(leaf, State) else leaf.states if isinstance(leaf, Count) else []):
                            if id(s) not in found:
                                self._refs.append(s)
        self._refs = [s for s in self._refs if id(s) not in found] # A state read in a rate before the transition that fills it (e.g. the I in S >> E @ beta*I, with E >> I after) is not only read
        for s in self._refs: # A state used only in a rate still belongs to this module unless another module has claimed it
            if s.module is None or s.module is self.module:
                found[id(s)] = s
        statelist = sorted(found.values(), key=lambda s: s.order)

        # Name the states
        for i,state in enumerate(statelist):
            if state.name is None:
                state.name = self.names.get(id(state)) or (state.label if isinstance(state.label, str) and state.label.isidentifier() else None) # Its variable, else its label if no variable holds it (ss.Disease(ss.State('S'), ...))
            if state.name is None:
                used = [tr.name or tr._arrow for tr in trs if any(end is state for end in (tr.source or []) + (tr.dest or []))]
                raise ValueError(f'state number {i+1} of the module has no name. It is used in {used}, and no variable or keyword holding it was found.\n'
                                      f"     Hold it in a variable, or give the names up front: S, I, R = ss.states('S I R').")
        for state in statelist:
            if state.module is not None and state.module is not self.module:
                raise ValueError(f'state "{state.name}" is used by transitions in two modules, "{state.module.name}" and "{self.module.name}"; a state belongs to one module.')
            if state.module is None and getattr(state, 'init_from_module', False): # A copy of another disease's state (see ss.Disease.reuse()): that disease's init= was its own
                state.init, state.init_n = state.declared_init
                state.init_from_module = False
            state.module = self.module
            self.add(self.states, state.name, state)

        # Initial conditions given on the module
        for key,given in [('init', self.init), ('init_n', self.init_n)]:
            for name,val in given.items():
                if name not in self.states:
                    raise sc.KeyNotFoundError(f'{key}= names a state "{name}" that "{self.module.name}" does not have. States are {sc.strjoin(self.states.keys())}.')
                setattr(self.states[name], key, val)
                self.states[name].init_from_module = True
        for state in statelist:
            if state.init is not None and state.init_n is not None:
                raise ValueError(f'"{state.name}" of "{self.module.name}" is given both init= (a share of people) and init_n= (a number of people); give one or the other')
        return

    def resolve_transitions(self):
        """ Give every transition a name, a source, and a rate or a dwell time """
        for tr in list(self.transitions.values()) + self._unnamed:
            name = tr.name or tr._arrow # Not named yet: see below
            if tr.dest is None and not (tr._inflow and not tr.state_given): # A birth with no state is into the first state (see resolve_people())
                errormsg = f'The transition "{name}" has no destination'
                raise ValueError(errormsg)

            undriven = tr.dur is None and tr.p is None and tr.trigger is None and getattr(tr, 'birth_on', None) is None and not tr.every_state
            if undriven and tr.rate is None and not tr.undriven: # What would it mean? A transition that nothing moves people along is written @ 0
                arrow = tr._arrow.replace(' -> ', ' >> ')
                if not tr.source:
                    raise ValueError(f'the inflow "{name}" ({tr._arrow}) has no rate. Give it one, e.g. {arrow} @ 0.1')
                raise ValueError(f'the transition "{name}" ({arrow}) has no rate, duration, p=, or trigger. Give it one, e.g. {arrow} @ ss.days(10); '
                                 f'or, if only a product (e.g. a treatment\'s cure), a delivery, or code moves people along it, a rate of 0: {arrow} @ 0')
            if tr.undriven or (undriven and is_zero(tr.rate)): # A rate of 0 as written (or built before, in another module): moved only by products, deliveries, or code, at the rate of 0 set again in resolve_exits()
                tr.rate = None
                tr.undriven = True

            # Derive the kind from the rate, written per person (a rate written as the whole flow, e.g. beta*S*I/N, is divided by its source)
            if tr.rate is not None:
                people = [st for st in self.states.values() if st.people]
                rate = per_person(tr.rate, tr.source, people)
                if rate is not tr.rate:
                    tr.flow_given, tr.rate = tr.rate, rate
                tr.rate = as_share(tr.rate, people) # beta*I/N is beta*I
            tr.const, tr.terms = (None, {}) if tr.rate is None else linear(tr.rate)
            tr.contact = bool(tr.terms)
            for state in tr.terms:
                state.infectious = True

        # Fallback names: the only transmission (by contact, or density-dependent) is the infection (as in ss.SIR), the rest are named after their endpoints
        contacts = [tr for tr in list(self.transitions.values()) + self._unnamed if tr.contact or self.counted(tr)]
        for tr in self._unnamed:
            src = '_'.join([s.name for s in tr.source or []]) or 'in'
            dst = '_'.join([s.name for s in tr.dest or []]) or 'out'
            if contacts == [tr] and not self.taken('infection'):
                base, why = 'infection', 'given to the only transmission'
            else:
                if tr.dest is None: # A birth with no state given is into the first state, found later
                    base = 'immigration' if tr.migration else 'births'
                elif src == 'in' and dst == 'out': # Out of every state (ss.death() with no state), as add_flow() names it
                    base = 'emigration' if tr.migration else 'deaths'
                else:
                    base = f'{src}_to_{dst}'
                why = 'built from its endpoints'
            name = base
            count = 1
            while name in self.transitions:
                count += 1
                name = f'{base}_{count}'
            tr.name = name
            tr.guessed.setdefault('name', why)
            self.add(self.transitions, name, tr)
        return

    def counted(self, tr):
        """
        The states of people of this module whose number or share a transition reads in its rate, other than its source, if the rate grows with them:
        `I` in `S >> I @ beta*I.n` (density-dependent transmission) or `S >> I @ beta*I.frac` (frequency-dependent, by the share of people in `I`; see
        `dependence()`); none for births at a rate on counts (`ss.unborn >> N @ b*N.n`)
        """
        if tr._inflow or tr.source is None or self.dependence(tr) is None: # No source is everyone (ss.death(rate=...)), so whatever it counts is its source
            return []
        out = {id(st):st for leaf in self.counts_in(tr) for st in leaf.states if st.people and st.module is self.module and not any(st is src for src in tr.source)}
        return list(out.values())

    def counts_in(self, tr):
        """ The counts (`I.n`, `I.frac`) in the rate of a transition, other than the total of the people it is divided by, e.g. `I.n` and not `N.n` in `beta*I.n/N.n` """
        people = [st for st in self.states.values() if st.people]
        total = {id(leaf) for f in factors(tr.rate)[1] if is_total(f, people) for leaf in f.leaves}
        return [leaf for leaf in tr.rate.leaves if isinstance(leaf, Count) and id(leaf) not in total]

    def dependence(self, tr):
        """ How the rate of a transition grows with the people in other states it counts: 'density' (`beta*I.n`), 'frequency' (`beta*I.frac`, or `beta*I.n/N.n`), or None (see `ss.expressions.dependence()`) """
        return dependence(tr.rate, [st for st in self.states.values() if st.people])

    def resolve_init_prev(self):
        """ By default, 1% of people start infected, seeded through the transmission (frequency- or density-dependent), unless a state is given a starting value """
        given = any(state.given_init for state in self.states.values())
        transmitted = any(tr.contact or self.counted(tr) for tr in self.transitions.values())
        if self.init_prev is None and not given and transmitted:
            self.init_prev = par(0.01)
            self._add_par(self.init_prev, name='init_prev')
            self.guessed['init_prev'] = 'the default, 1%'
        return

    def resolve_starting(self):
        """ Let the starting values (init_prev, and each state's init= and init_n=) leave levels out of an `ss.by` by label, which start at 0 (see `ss.parameters.starting()`); done when the sim is built, after values by name are set and before anything checks or reads them """
        for state in self.states.values():
            state.init, state.init_n = [ss.parameters.starting(v) for v in [state.init, state.init_n]]
        self.init_prev = ss.parameters.starting(self.init_prev)
        raw = self.init_prev.raw if isinstance(self.init_prev, par) else self.init_prev
        if isinstance(raw, ss.bernoulli): # The same as its p, e.g. from a coded disease: each person starts infected with that chance anyway
            if isinstance(self.init_prev, par):
                self.init_prev.value = raw.pars.p
            else:
                self.init_prev = raw.pars.p
        return

    def unset_default_init_prev(self):
        """ Remove the default init_prev """
        if self.guessed.pop('init_prev', None) is not None:
            self.init_prev = None
            self.pars.pop('init_prev')
            self.module.pars.pop('init_prev', None)
        return

    def transmitted_from(self):
        """ The states of people that transmission (by contact, or at a rate on counts of others) takes people from """
        out = {}
        for tr in self.transitions.values():
            if tr.source and not tr._inflow and (tr.contact or self.counted(tr)):
                out.update({id(s):s for s in tr.source if s.people})
        return list(out.values())

    def resolve_people(self):
        """
        Which states are people: every state, unless it is declared a count (`ss.State(people=False)`, e.g. mosquitoes or
        bacteria in the water); and the first state, where people are unless they are put elsewhere: `default_state=` if given,
        else the state named `S` or `susceptible` (in any case), else the one state transmission infects people from, else the first state of people made (states are kept in the
        order they were created: see `resolve_states()`), which `describe()` shows as a guess. Not the state a birth goes to
        (`ss.birth(NB)` says where births go, not where everyone starts), nor the order of the transitions
        """
        states = list(self.states.values())
        self.people_states = [s for s in states if s.people]
        self.count_states = [s for s in states if not s.people]
        if self.default_given is not None:
            name = self.default_given.name if isinstance(self.default_given, State) else self.default_given
            if name not in self.states:
                raise sc.KeyNotFoundError(f'default_state="{name}" is not a state of "{self.module.name}". Its states are {sc.strjoin(self.states.keys())}{did_you_mean(name, list(self.states.keys()))}')
            self.default_state = self.states[name]
        elif any(s.name.lower() in ['s', 'susceptible'] for s in self.people_states):
            self.default_state = next(s for s in self.people_states if s.name.lower() in ['s', 'susceptible'])
        elif len(sources := self.transmitted_from()) == 1: # The one state transmission infects people from, e.g. Ca in Ca >> P @ beta*P
            self.default_state = sources[0]
            self.guessed['default_state'] = f'{self.default_state.name} is where people start, the state infected from (or give default_state=)'
        elif self.people_states:
            self.default_state = self.people_states[0]
            self.guessed['default_state'] = f'{self.default_state.name} is where people start, the first state made (or give default_state=)'
        else:
            self.default_state = None # Nothing is people (e.g. a model of mosquitoes alone)
        for tr in self.transitions.values(): # A birth or immigration with no state given
            if tr.dest is None and self.default_state is not None:
                tr.dest = [self.default_state]
        self.resolve_rest()

        # Now that the people are known, fill in the things that depend on them
        for name,tr in self.transitions.items():
            self.fill_people(tr)
            for state in tr.terms:
                if not state.people:
                    raise ValueError(f'the rate for "{name}" uses the bare state {state.name}, and {state.name} is not people.\n'
                                             f'       rate = {tr.rate}\n'
                                             f'     A bare state means contact with people in that state, and there is no contact with a count.\n'
                                             f'     Write {state.name}.n for the number itself, or leave out people=False if {state.name} is people.')
            ends = (tr.source or []) + (tr.dest or [])
            tr.people = all(s.people for s in ends)
            if not tr.people and any(s.people for s in ends):
                counts = [s.name for s in ends if not s.people]
                raise ValueError(f'"{name}" moves between people ({sc.strjoin(s.name for s in ends if s.people)}) and counts ({sc.strjoin(counts)}, declared with people=False). '
                                         f'If {sc.strjoin(counts)} {"is" if len(counts) == 1 else "are"} people, leave out people=False.')
        self.check_counts()
        return

    def check_counts(self):
        """
        Refuse a forgotten `people=False`: a group of states of people that no transition links to the first state
        and that only an inflow (not a birth) fills, as mosquitoes are, would otherwise silently become agents
        """
        if not any(s is self.default_state for s in self.people_states): # No first state given (sim.init() raises), or not one of people
            return
        group = {id(s) for s in next(g for g in self.linked() if any(s is self.default_state for s in g))} # The states linked to the first
        for tr in self.transitions.values():
            dst = [s for s in tr.dest or [] if id(s) not in group]
            if tr._inflow and tr.people and dst and not getattr(tr, 'per_capita', False) and tr.birth_on is None and not any(s.people_given for s in dst):
                others = [s.name for s in self.people_states if id(s) not in group]
                raise ValueError(f'"{tr.name}" ({tr._arrow}) fills {sc.strjoin(others)}, which no transition links to {self.default_state.name}, where people start, so {"it looks like" if len(others) == 1 else "they look like"} a count of something else (e.g. mosquitoes). '
                                                f'Every state is people unless declared a count: if {"it is" if len(others) == 1 else "they are"} not people, declare {sc.strjoin(others)} with ss.State(people=False) (or ss.states(n, people=False)); if {"it is" if len(others) == 1 else "they are"} people, say so with people=True.')
        return

    def linked(self):
        """ The groups of states of people that transitions link, in either direction: one, unless the people are in separate chains of states (e.g. a chain for each host species) """
        links = {id(s):{id(s)} for s in self.people_states}
        for tr in self.transitions.values():
            ends = [s for s in (tr.source or []) + (tr.dest or []) if s.people]
            for s in ends:
                links[id(s)].update(id(e) for e in ends)
        groups, seen = [], set()
        for state in self.people_states:
            group, todo = set(), [id(state)]
            while todo:
                i = todo.pop()
                if i not in group and i not in seen:
                    group.add(i)
                    todo += links[i]
            if group:
                seen |= group
                groups.append([s for s in self.people_states if id(s) in group])
        return groups

    def resolve_rest(self):
        """
        Work out where the people not given an initial state start: the first state, unless it is given an initial value of
        its own, which is then kept (`init=dict(S=0.1, I=1e-4)` starts 10% susceptible). Then it is the one state of people
        with no initial value, with a note in `guessed`; with none or several, there is no such state, and the initial values
        must sum to 1 (see `Engine.init_fractions()`)
        """
        seeded = {id(st) for tr in self.transitions.values() if tr.contact for st in tr.dest or []} if self.init_prev is not None else set() # Where init_prev puts people
        both = [s for s in self.people_states if id(s) in seeded and s.given_init]
        if both:
            keys = sc.strjoin(sorted({'init_n=' if s.init_n is not None else 'init=' for s in both}), sep=' or ')
            errormsg = f'"{self.module.name}" is given both init_prev= and {keys} for {sc.strjoin(s.name for s in both)}, where init_prev= puts people; give one or the other'
            raise ValueError(errormsg)
        self.rest = self.default_state
        if self.default_state is None or not self.default_state.given_init:
            return
        unset = [s for s in self.people_states if not s.given_init and id(s) not in seeded]
        if len(unset) == 1:
            self.rest = unset[0]
            self.guessed['rest'] = f'{self.rest.name}, the only state with no initial value, since {self.default_state.name} has one'
        else:
            self.rest = None
        return

    def resolve_acquisition(self):
        """
        Work out which transitions are an acquisition of the disease, which `rel_sus`, a product's efficacy against acquisition, and `ss.treat` act on

        A contact-mediated transition is one. So is any other transition of people out of the first state that is not an outflow,
        whatever its rate (a bite rate, an EIR, a count of bacteria), unless a transition out of that state is contact-mediated, which
        is then the acquisition (so `S -> V` alongside `S -> I @ beta * I` is a vaccination). `acquisition=` on the transition says it outright.
        """
        first = self.default_state
        out_of_first = [tr for tr in self.transitions.values() if any(src is first for src in tr.source or [])]
        by_contact = [tr.name for tr in out_of_first if tr.contact]
        for name,tr in self.transitions.items():
            given = getattr(tr, 'acquisition_given', None)
            if given is not None:
                tr.acquisition = bool(given)
            elif tr.contact:
                tr.acquisition = True
            elif tr in out_of_first and tr.people and not tr._outflow:
                tr.acquisition = not by_contact
                tr.guessed['acquisition'] = f'contact-mediated {sc.strjoin(by_contact)} is' if by_contact else f'out of {first.name}'
            else:
                tr.acquisition = False
        return

    def fill_people(self, tr):
        """ Fill in what depends on which states are people: the sources of the deaths of everyone (`ss.death()` with no state), and the people a birth rate is per person of (`ss.birth()`) """
        if tr.every_state:
            tr.source = list(self.people_states)
        if tr.per_capita:
            tr.per_person = tr.rate # Kept for a rate by level, whose births are from the people at each level (see ss.solvers.Engine.per_level())
            tr.rate = tr.rate * Count(list(self.people_states), kind='n')
            tr.const, tr.terms = linear(tr.rate)
        return

    def set_keywords(self):
        """ A value given as a keyword sets the parameter of that name; there is nothing else for it to be """
        for name,value in self._keywords.items():
            if name not in self.pars and name.startswith(('init', 'seed')): # e.g. init_infected=100
                state = self.infectious[0].name if self.infectious else 'I'
                raise sc.KeyNotFoundError(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}".\n'
                                          f'     To start with people in a state, use init=dict({state}=0.01) for a share of people, or init_n=dict({state}=10) for a number; init_prev= seeds the state a new infection enters (by default 1% of people), as a share or a number of people.')
            elif name not in self.pars:
                raise sc.KeyNotFoundError(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}"; its parameters are {sc.strjoin(self.pars.keys())}.{ss.errors.starsim3(keys=name, call=f"ss.{type(self.module).__name__}")}\n'
                                          f'     To add a parameter, use it in a transition, e.g. infection = S >> I @ {name} * I with {name} = ss.par({value}).\n'
                                          f'     If another module reads it from this disease, add it after making the disease: {self.module.name}.define_pars({name}={value!r}).' +
                                          (f'\n     Condoms (or other protection used in some acts) are an intervention on the transmission instead, e.g. ss.modify({self.module.name}.infection, efficacy={value!r}, share=0.3, network="sexualnet").' if 'condom' in name else ''))
            ss.parameters.replaced_dist(name, self.pars[name].value, value)
            self.pars[name].value = value
        return

    @staticmethod
    def set_slots(tr, density=False):
        """
        Record the kind of value each parameter of a transition is, as `par.slot` (which describe() uses):
        'rate' or 'dur', or None for a plain number (e.g. a factor of a rate); returns the standard slots it fills, as (slot name, parameter).
        `density` says the transition is a transmission at a rate on the number or share of people in a state (`beta*I.n`, `beta*I.frac`), whose coefficient is beta too
        """
        slots = []
        if tr.dur is not None:
            tr.dur.slot = 'dur'
            slots.append(('dur', tr.dur))
        if tr.rate is not None:
            for leaf in tr.rate.leaves:
                if isinstance(leaf, par):
                    leaf.slot = 'rate' if isinstance(leaf.raw, (ss.TimePar, ss.Dist)) else None
            if isinstance(tr.rate, par):
                tr.rate.slot = 'rate'
                slots.append(('rate', tr.rate))
            elif (tr.contact and tr.const is None) or density: # The coefficient of the source of infection is beta, by contact (beta*I), density-dependent (beta*I.n), or frequency-dependent (beta*I.frac)
                coefs = list(tr.terms.values())
                rate = tr.rate
                beta = lambda leaf: isinstance(leaf, par) and (not leaf.literal or isinstance(leaf.raw, (dict, ss.peract)) or ss.expressions.value_dimension(leaf.raw) is not None) # A parameter, a beta for each route, or a typed value
                alone = isinstance(rate, BinOp) and not any(beta(leaf) for leaf in rate.b.leaves) # Only the rate as a whole has a unit, so a factor of a product of parameters (or of a typed beta) is a number: R0 in R0 * (gamma * I), or 0.275 in 0.275*(dict(sexualnet=ss.peract(0.05))*I)
                if isinstance(rate, BinOp) and rate.op == '*' and isinstance(rate.a, par) and alone:
                    rate.a.slot = 'rate'
                    slots.append(('beta', rate.a))
                elif not density and len(coefs) == 1 and isinstance(coefs[0], par) and not is_one(coefs[0]):
                    coefs[0].slot = 'rate'
                    slots.append(('beta', coefs[0]))
                elif density: # The only parameter it is multiplied by, e.g. 0.3 in 0.3*I.n/N.n
                    num, den = factors(rate)
                    pars = [f for f in num if isinstance(f, par)]
                    if len(pars) == 1 and not any(isinstance(leaf, par) for f in den for leaf in f.leaves) and not any(isinstance(leaf, par) for f in num if f is not pars[0] for leaf in f.leaves):
                        pars[0].slot = 'rate'
                        slots.append(('beta', pars[0]))
        for p in tr.p or []:
            if isinstance(p, par) and len(tr.p) == 1:
                slots.append(('p', p))
        if isinstance(tr.noise, par): # Sigma of the noise on the rate, named noise_<transition> if not named
            slots.append(('noise', tr.noise))
        return slots

    def resolve_pars(self):
        """ Turn inline values in the standard slots into named parameters (beta, dur, rate, p) """
        slots = [(tr, slot, p) for tr in self.transitions.values() for slot,p in self.set_slots(tr, bool(self.counted(tr)))] # (transition, slot name, parameter)

        # Name the anonymous ones after their slot: a duration by its state and a rate by its transition, and any other qualified by the transition where two would clash
        anon = [(tr, slot, p) for tr,slot,p in slots if p.literal or getattr(p, 'slot_named', False) or (p.name is None and self.names.get(id(p)) is None)] # Including an ss.par written in the slot itself, e.g. beta=ss.par(0.3, bounds=[0.1, 1]), and one named after its slot before (in a class this one subclasses), which is named again
        anon = [(tr, slot, p) for i,(tr,slot,p) in enumerate(anon) if not any(q is p for _,_,q in anon[:i])] # One used in several transitions (e.g. beta = ss.seasonal(...)) is one parameter, named after the first
        counts = sc.obj()
        for tr,slot,p in anon:
            counts[slot] = counts.get(slot, 0) + 1
        inner = {q.name or self.names.get(id(q)) for tr,slot,p in slots for q in inner_pars(p.value)} # The names of parameters inside values that vary in time, e.g. ss.seasonal(mean=beta), or inside a probability per act, which are registered below
        inner |= {q.name or self.names.get(id(q)) for tr,slot,p in slots if isinstance(p.value, Expr) for q in p.value.leaves if isinstance(q, par) and q is not p} # And inside an expression in a slot, e.g. dur=dur_rec*k
        inner |= {p.name or self.names.get(id(p)) for tr,slot,p in slots if not any(p is q for _,_,q in anon)} # And the explicit ones in the slots, registered below, e.g. dur = ss.par(ss.days(6)) in one transition and a bare ss.days(100) in another
        taken = lambda name: name in self.pars or name in self.states or name in self.transitions or name in inner
        timed = lambda state: sum(tr.dur is not None and any(s is state for s in tr.source or []) for tr in self.transitions.values()) # The number of timed exits from a state
        for tr,slot,p in anon:
            qual = tr.source[0].name if slot == 'dur' and len(tr.source or []) == 1 and timed(tr.source[0]) == 1 else tr.name # A dwell time is the mean time in its state (dur_E), unless the state has several timed exits
            name = f'{slot}_{qual}' if slot in ['dur', 'rate', 'noise'] or counts[slot] > 1 or taken(slot) else slot
            if qual != tr.name and taken(name):
                name = f'{slot}_{tr.name}'
            if taken(name): # e.g. a parameter of the user's called dur_rec, inside the dwell time of a transition called rec
                name = f'{slot}_of_{tr.name}'
            self._add_par(p, name=name)
            p.slot_named = True
        for tr,slot,p in slots: # A parameter inside a value that varies in time, given without a name or a variable of its own, is named after the parameter and its field, e.g. beta_mean in beta = ss.seasonal(mean=ss.par(...))
            val = p.value
            for field in getattr(val, 'fields', []) if isinstance(val, TimeValue) else []:
                for q in sc.tolist(getattr(val, field, None)):
                    if isinstance(q, par) and not q.literal and q.name is None and self.names.get(id(q)) is None:
                        q.name = f'{p.name}_{field}'

        # Register the explicit ss.par objects used in the transitions, and in the weights and likelihoods of the outcomes
        used = [(tr.name, expr) for tr in self.transitions.values() for expr in [tr.rate, tr.dur, tr.noise] + sc.tolist(tr.p) + [getattr(tr, 'p_die', None), getattr(tr, 'p_stay', None)]] # Including the shares of ss.dead and ss.stay among the destinations
        used += [(oc.name, weight) for oc in self.outcomes.values() for weight in oc.weights]
        used += [(oc.name, val) for oc in self.outcomes.values() for val in oc.likelihood_pars.values() if isinstance(val, par)]
        for oc in self.outcomes.values(): # A parameter written inside a likelihood, e.g. ss.nbinom(n=ss.par(10, bounds=[1, 100])), is named after the outcome and its keyword
            for key,val in oc.likelihood_pars.items():
                if isinstance(val, par) and not val.literal and val.name is None and self.names.get(id(val)) is None:
                    val.name = f'{oc.name}_{key}'
        used += [(f'init of {state.name}', init) for state in self.states.values() for init in [state.init, state.init_n] if isinstance(init, Expr)] # So an initial count can be fitted
        used += [(name, expr) for name,tf in self.transfers.items() for expr in [tf.rate, tf.dur] + sc.tolist(tf.p)] # And a parameter in the rate of a transfer
        used += [(name, expr) for name,a in self.attrs.items() for expr in ([e for names,e in a.cases] if a.cases is not None else [e for _,e in a.by_event] if a.by_event is not None else [a.value]) if isinstance(expr, Expr) and not isinstance(expr, ss.expressions.attr)] # And in an attribute
        used += [(name, val) for name,a in self.attrs.items() for _,v in (a.by_event or []) for p in ss.expressions.dists_in(v) for val in p.value.pars.values() if isinstance(val, par)]
        used += [(name, val) for name,a in self.attrs.items() if a.kind in ['drawn', 'trait'] for p in ss.expressions.dists_in(a.value) for val in p.value.pars.values() if isinstance(val, par)] # And a parameter of a distribution an attribute is drawn from, e.g. ss.bernoulli(ss.par(0.6, name='sero'))
        used += [(name, a.init_value) for name,a in self.attrs.items() if isinstance(a.init_value, Expr)] # And in its value before any event
        used += [(name, val) for name,a in self.attrs.items() if a.init_value is not None for p in ss.expressions.dists_in(a.init_value) for val in p.value.pars.values() if isinstance(val, par)]
        used += [(name, leaf.value) for name,expr in list(used) if expr is not None for leaf in expr.leaves if isinstance(leaf, par) and isinstance(leaf.value, Expr)] # And inside a dwell time given as an expression (of attributes)
        used += [(name, w) for name,expr in list(used) if expr is not None for w in count_weights(expr)] # And in the weights of a weighted count, e.g. (ss.by(biting=w)*I).n
        used += [(name, inner) for name,expr in list(used) if expr is not None for leaf in expr.leaves if isinstance(leaf, par) for inner in inner_pars(leaf.value)] # And a parameter inside a value that varies in time, a probability per act, or a beta for each route
        for name,expr in used:
            if expr is not None:
                for leaf in expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        if leaf.name is None:
                            leaf.name = self.names.get(id(leaf))
                        if leaf.name is None:
                            errormsg = f'Could not find a name for the parameter {leaf.value} used in "{name}"; give it one, e.g. ss.par({leaf.value}, name="{name}_{leaf.slot or "p"}")'
                            raise ValueError(errormsg)
                        self._add_par(leaf, name=leaf.name)
        return

    def resolve_exits(self):
        """ Group the transitions out of each state: those with p= split on entry, the rest compete """
        self.exits = sc.obj()
        for state in self.states.values():
            self.exits[state.name] = sc.obj(branches=[], timed=[], hazards=[], contact=[], rest=[], triggered=[])
        self.inflows = []
        for tr in self.transitions.values():
            if tr.undriven: # To the engines, a rate of 0 that nothing modifies; a delivery moves people along it (see ss.Treatment)
                tr.rate = par.inline(0.0)
                tr.const, tr.terms = linear(tr.rate)
            if tr._inflow:
                self.inflows.append(tr)
            if tr._immediate and tr.where is not None and tr.trigger is None and (tr.p is None or tr._split): # dur=0 with who=: as soon as the condition holds, as ss.on() does (a branch with p= waits in its source until who= holds instead)
                tr.trigger = ss.on(tr.where)
            for state in tr.source:
                exits = self.exits[state.name]
                if tr.contact:
                    exits.contact.append(tr)
                elif tr.trigger is not None: # Moved as soon as its condition holds, or its event happens: see ss.on
                    exits.triggered.append(tr)
                elif tr.p is not None and not tr._split: # One branch of several out of this state
                    exits.branches.append(tr)
                elif tr.dur is not None and not tr._immediate:
                    exits.timed.append(tr)
                elif tr.rate is None and len(tr.dest) > 1 and not tr._immediate: # Several destinations and no wait: say how to write an immediate split
                    names = [d.name for d in tr.dest]
                    raise ValueError(f'the transition "{tr.name}" ({state.name} -> {sc.strjoin(names)}) has p= but no rate and no duration (dur=), so nothing says when people leave. '
                                                f'For a split as soon as people arrive, give it dur=0, e.g. ss.Transition({state.name}, {{{", ".join(f"{n}: {None if v is None else v.raw if isinstance(v, par) else v}" for n,v in zip(names, tr.p))}}}, dur=0); or give it a duration.')
                elif tr.rate is None: # Immediate: dur=0, one destination and ss.stay, or what the branches leave over
                    exits.rest.append(tr)
                else:
                    exits.hazards.append(tr)

        # A single timed transition with no p= (or an immediate split) takes whatever the branches leave over: the only one, or the only one for the same people (the same who=) as the branches; one with no driver and no p= is undriven, not the remainder
        for name,exits in self.exits.items():
            exits.remainder = None
            rest = exits.pop('rest')
            if len(rest) > 1:
                raise ValueError(f'{sc.strjoin(tr.name for tr in rest)} out of state "{name}" have no p=, rate, or duration; only one transition can take what the branches leave over.')
            elif rest:
                exits.remainder = rest[0]
            apart = [] # Timed exits for other people (another who=), which compete as before
            if exits.branches and len(exits.timed) > 1:
                wheres = {id(getattr(tr, 'where', None)) for tr in exits.branches}
                same = [tr for tr in exits.timed if len(wheres) == 1 and id(getattr(tr, 'where', None)) in wheres]
                if len(same) == 1: # e.g. women split between CIN and clearance, and men only clear: the clearance of women is the remainder
                    apart = [tr for tr in exits.timed if tr is not same[0]]
                    exits.timed = same
            if exits.branches:
                ps = [tr.p[0] for tr in exits.branches]
                const = all(isinstance(p, par) and sc.isnumber(p.raw) for p in ps)
                total = sum(p.raw for p in ps) if const else None
                if not const and exits.remainder is None and len(exits.timed) == 1: # A share that is worked out as the sim runs (e.g. a function): the timed exit with no p= takes the rest, then
                    exits.remainder = exits.timed.pop()
                    exits.remainder.guessed['p'] = f'1 - {" - ".join(tr.name for tr in exits.branches)}'
                elif const and (bad := self.bad_splits(exits)):
                    below = all(t < 1 for _,t in bad)
                    if below and len(exits.timed) == 1:
                        exits.remainder = exits.timed.pop()
                        exits.remainder.guessed['p'] = f'{1 - bad[0][1]:n}'
                    else:
                        group, total = bad[0]
                        undriven = [tr.name for tr in exits.hazards if tr.undriven]
                        driven = [tr for tr in exits.hazards if not tr.undriven]
                        why = f' {sc.strjoin(undriven)} {"has" if len(undriven) == 1 else "have"} a rate of 0 (@ 0) and no p=, so {"is" if len(undriven) == 1 else "are"} moved only by products or code, not the remainder; for an immediate split, give the destinations together, e.g. ss.Transition({name}, {{A: {total:n}, B: None}}, dur=0).' if undriven else ''
                        why += f' {sc.strjoin(tr.name for tr in driven)} {"has" if len(driven) == 1 else "have"} a rate and no p=, so {"it competes" if len(driven) == 1 else "they compete"} with the branches rather than taking the rest: give it p={1 - total:n}, or a duration (dur=), which takes the rest.' if total < 1 and driven and not exits.timed else ''
                        why += f' For the rest to stay in {name}, give the destinations as one split with ss.stay, e.g. [A, ss.stay] with p=[{total:n}, None].' if total < 1 else ''
                        raise ValueError(f'the branch probabilities out of state "{name}" sum to {total:n}, not 1 '
                                                f'({sc.strjoin([f"{tr.name}: p={tr.p[0]}" for tr in group])}).{why}')
            exits.timed += apart
        for name,exits in self.exits.items():
            outs = exits.branches + exits.timed + exits.hazards + exits.triggered + sc.tolist(exits.remainder)
            exits.held = any(tr.stays for tr in outs) # Whether some people stay with no exit of their own (ss.stay): see ss.solvers.Compartmental.build_layout()
        return

    @staticmethod
    def bad_splits(exits):
        """ The groups of branches with constant p= whose shares do not sum to 1: those that apply to the people at each level named by a move= arrow are summed within that level; returns [(branches, total)] """
        bad = []
        dims = sorted({dim for tr in exits.branches for dim in tr.move_from})
        levels = [sorted({lev for tr in exits.branches for lev in tr.move_from.get(dim, [])}) for dim in dims]
        for cell in itertools.product(*levels):
            group = [tr for tr in exits.branches if all(lev in tr.move_from.get(dim, [lev]) for dim,lev in zip(dims, cell))]
            total = sum(tr.p[0].raw for tr in group)
            if group and not np.isclose(total, 1) and not (exits.remainder is not None and total < 1):
                bad.append((group, total))
        return bad

    def check(self):
        """ Checks that do not need the sim """
        for name,tr in self.transitions.items():
            tr._check_dimensions()
            if tr.trigger is not None and (tr.contact or tr._inflow or (tr.p is not None and not tr._split and not tr.trigger.outcome)): # On an outcome, p= is the share of its people who move; a split with ss.stay or ss.dead is a split, though neither is a state
                raise ValueError(f'"{name}" is triggered by {tr.trigger}, so it cannot be {"a transmission" if tr.contact else "an inflow" if tr._inflow else "one branch of several with p="}; '
                                        f'give the destinations of a triggered split together, e.g. "{tr._arrow} | B" with p=[...].')
            for inc in [c for c in ss.expressions.nodes([tr.rate, tr.dur] + sc.tolist(tr.p)) if isinstance(c, ss.conditions.Cond) and c.op == 'incidence']: # A flow over a window, in a rate, would be a delay equation
                flow = inc.args[0]
                mod = ss.conditions.modname(getattr(flow, 'module', None)) or getattr(self.module, 'name', None)
                path = f'{mod}.{flow.name}' if mod else flow.name
                dst = (flow.dest or [None])[0]
                raise ss.CapabilityError(f'"{name}" reads the incidence of "{flow.name}" in its rate, which is not supported yet: a flow over a window, in a rate, needs the history of the flow in every solver.\n'
                                         f'     Use a count of a state instead, e.g. {getattr(dst, "name", "X")}.n, which is about the flow into it times the time people spend there. '
                                         f'The incidence can be used in who= of ss.modify(), e.g. ss.modify(..., multiply=0.6, who={path}.incidence(over=ss.days(7)) > 0.001)')
            for clock in ss.expressions.clocks_in([tr.rate, tr.dur] + sc.tolist(tr.p)): # A clock has a value only inside its states, and progress needs a scheduled dwell
                if clock.module not in [None, self.module] or not all(n in self.states for n in clock.names):
                    continue # Another module's clock: checked when it runs
                outside = [src.name for src in tr.source or [] if src.name not in clock.names]
                if outside:
                    raise ValueError(f'"{name}" reads {clock}, which has a value only for people in {sc.strjoin(clock.names)}, but moves people from {sc.strjoin(outside)}.')
                if clock.kind == 'progress':
                    ex = self.exits[clock.names[0]]
                    if not (ex.branches or ex.remainder or ex.timed):
                        raise ValueError(f'"{name}" reads {clock}, but {clock.names[0]} has only rates out of it, so no dwell time is scheduled when people enter it.')
                if len(tr.dest or []) <= 1 and any(c is clock for c in ss.expressions.clocks_in(sc.tolist(tr.p))):
                    raise ss.CapabilityError(f'"{name}" has p= with {clock}, but a branch with p= is decided when people enter {sc.strjoin(clock.names)}, when no time has passed. '
                                             f'Write one transition with several destinations, decided when they leave, e.g. ss.Transition({tr.source[0].name}, {{{tr.dest[0].name}: f({clock}), ...: None}}, dur=...)')
            if tr.stays and not tr.dest and not tr.may_die:
                raise ValueError(f'"{name}" has ss.stay as its only destination; give it with another, e.g. [B, ss.stay] with p=[0.3, None].')
            if tr.p is not None and tr._split:
                ps = tr.p
                if len(ps) == 1 and len(tr.dest) == 2 and not tr.stays: # With two destinations, a single value is the first one's probability
                    ps = tr.p = [ps[0], None]
                ends = ps + [tr.p_stay]*tr.stays + [tr.p_die]*tr.may_die # One entry for each destination, with ss.stay and ss.dead
                if len(ps) != len(tr.dest):
                    raise ValueError(f'"{name}" has {len(tr.dest) + tr.stays + tr.may_die} destinations but {len(ps) + (tr.p_stay is not None) + (tr.p_die is not None)} branch probabilities; give one for each (or None for the one that takes the rest).')
                given = [p for p in ends if p is not None]
                if len(given) < len(ends) - 1:
                    which = [label for label,p in zip([d.name for d in tr.dest] + ['ss.stay']*tr.stays + ['ss.dead']*tr.may_die, ends) if p is None]
                    raise ValueError(f'"{name}" leaves the rest to {sc.strjoin(which)}; only one destination can take the rest, so give the others a p=.')
                if all(isinstance(p, par) and sc.isnumber(p.raw) for p in given):
                    total = sum(p.raw for p in given)
                    if (len(given) == len(ends) and not np.isclose(total, 1)) or total > 1 + 1e-9:
                        raise ValueError(f'the branch probabilities of "{name}" sum to {total:n}, not 1.')
            elif tr.p is None and tr._split:
                raise ValueError(f'"{name}" has {len(tr.dest)} destinations but no branch probabilities; give them with p=[...].')
            if tr.dest is not None and len(tr.dest) > 1: # Each destination's flow is res.<transition>.<state>, so it cannot be named like an attribute of a result
                clash = [d.name for d in tr.dest if d.name in ss.Result.reserved()]
                if clash:
                    raise ValueError(f'the flow of "{name}" to {sc.strjoin(clash)} is a result read as res.{name}.{clash[0]}, which is already an attribute of every result; rename the state {sc.strjoin(clash)}')
        for name,a in self.attrs.items(): # Each case of a derived attribute reads clocks only of states it is for
            for names,expr in (a.cases or []):
                for clock in ss.expressions.clocks_in(expr):
                    if not all(any(c is s for s in self.states.values()) for c in clock.states):
                        continue
                    outside = [n for n in names if n not in [c.name for c in clock.states]]
                    if outside:
                        raise ValueError(f'the attribute "{name}" reads {clock} for people in {sc.strjoin(outside)}, but it has a value only for people in {sc.strjoin(c.name for c in clock.states)}.')
        self.dwell_trs = {name for name,tr in self.transitions.items() if len(tr.dest or []) > 1 and ss.expressions.clocks_in(sc.tolist(tr.p))} # Split as people leave, by their time in the state
        return

    def resolve_attrs(self):
        """ Resolve each attribute: another attribute given by name, the cases of a derived one by state (a state, a sum of states, or a | of them), and its events """
        def names_of(key):
            if isinstance(key, State):
                states = [key]
            elif isinstance(key, Expr): # A sum of states, e.g. latent + falling
                states = [leaf for leaf in key.leaves if isinstance(leaf, State)]
            elif isinstance(key, ss.conditions.Cond): # A | B
                states = [s[1] for s in (ss.conditions.set_states(key) or [])]
            else:
                states = sc.tolist(key)
            out = [s if isinstance(s, str) else s.name for s in states]
            for name in out:
                if name not in self.states:
                    raise sc.KeyNotFoundError(f'an attribute of "{self.module.name}" names a state "{name}" that it does not have. Its states are {sc.strjoin(self.states.keys())}.')
            return out
        for tr in self.transitions.values(): # One written in a transition rather than passed to the module, e.g. rate=b*ss.attr(ss.uniform(0.5, 1.5), name='frailty')*I, is the module's
            for expr in [tr.rate, tr.dur] + list(tr.p or []):
                for leaf in (expr.leaves if isinstance(expr, Expr) else []):
                    if isinstance(leaf, ss.expressions.attr) and leaf.module is None and not any(a is leaf for a in self.attrs.values()):
                        if leaf.name is None:
                            raise ValueError(f'Could not find a name for {leaf} in "{tr.name}"; give it one, e.g. ss.attr(..., name="frailty"), or pass it as a keyword, e.g. ss.Disease(..., frailty=ss.attr(...))')
                        self.add(self.attrs, leaf.name, leaf)
        for name,a in self.attrs.items():
            if a.module is not None and a.module is not self.module: # It would be drawn for each module separately, not shared
                raise ValueError(f'the attribute "{name}" is given to two modules, "{a.module.name}" and "{self.module.name}"; an attribute belongs to one module, and each would draw its own values. '
                                                     f'For one value per person read by both, give it to one and read it from the other in a function of the sim.')
            a.module = self.module
            if isinstance(a.value, str):
                if a.value not in self.attrs:
                    errormsg = f'The attribute "{name}" of "{self.module.name}" records "{a.value}", which is not one of its attributes ({sc.strjoin(self.attrs.keys())})'
                    raise ValueError(errormsg)
                a.value = self.attrs[a.value]
            if isinstance(a.value, dict) and all(ss.expressions.is_event(key) for key in a.value): # A value for each event
                a.by_event = [(ss.expressions.events_in(key, self), val if isinstance(val, ss.Dist) else as_expr(val)) for key,val in a.value.items()]
            elif isinstance(a.value, dict):
                if a.on is not None:
                    errormsg = f'The attribute "{name}" is a dict by state, which is derived (worked out when it is read), so it takes no when='
                    raise ValueError(errormsg)
                a.cases = [(names_of(key), as_expr(val)) for key,val in a.value.items()]
                seen = [n for names,expr in a.cases for n in names]
                if len(seen) != len(set(seen)):
                    errormsg = f'The attribute "{name}" gives more than one value for {sc.strjoin(sorted({n for n in seen if seen.count(n) > 1}))}'
                    raise ValueError(errormsg)
            elif a.kind in ['recorded', 'derived'] and not isinstance(a.value, ss.Dist):
                a.value = as_expr(a.value)
            a.events = [] if a.on is None else ss.expressions.events_in(a.on, self)
        for obj in self.attrs.values():
            for e in obj.events:
                kind, target, dst = e.args
                if kind in ['tr', 'to'] and target.name not in self.transitions:
                    raise sc.KeyNotFoundError(f'"{obj.name}" of "{self.module.name}" is set on {e.label}, which is not one of its transitions ({sc.strjoin(self.transitions.keys())}).')
                elif kind in ['enter', 'exit']:
                    names_of([s[1] if isinstance(s, tuple) else s for s in target])
        return

    def stratify(self, **kwargs):
        """ Stratify the graph by one or more dimensions; see `ss.Disease.stratify()` """
        by_attr = [key for key in kwargs if key in self.attrs] # By a derived attribute: its level is worked out from the attribute, so it adds no groups
        if by_attr:
            others = [key for key in kwargs if key not in by_attr]
            if others:
                errormsg = f'A stratification by the attribute "{by_attr[0]}" is worked out from it, so is given in a stratify() of its own, not with {sc.strjoin(others)}'
                raise ValueError(errormsg)
            for name in by_attr:
                a, levels = self.attrs[name], kwargs[name]
                states = None
                if isinstance(levels, ss.dimensions.levels):
                    extra = [key for key,val in levels.options.items() if val is not None and key != 'states']
                    if extra:
                        errormsg = f'A stratification by the attribute "{name}" is worked out from it, so takes only states=; give a value by level in a rate with ss.by({name}=[...]), not {sc.strjoin(extra)}'
                        raise ValueError(errormsg)
                    levels, states = levels.levels, levels.options['states']
                if states is not None:
                    states = [leaf.name for leaf in states.leaves if isinstance(leaf, State)] if isinstance(states, Expr) and not isinstance(states, State) else [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
                if a.kind != 'derived' or not all(sc.isnumber(v) for v in levels):
                    errormsg = f'A stratification by the attribute "{name}" takes bin edges, and the attribute must be derived (worked out when it is read); "{name}" is {a.kind}'
                    raise ValueError(errormsg)
                if any(s.name == name for s in self.strats + self.derived):
                    raise ValueError(f'"{self.module.name}" is already stratified by "{name}". stratify() works in place, so this is what running the same line (or notebook cell) twice gives.')
                strat = ss.dimensions.Strat(name, levels, kind='derived', states=states or [s.name for s in self.people_states])
                strat.attr_of = a
                self.derived.append(strat)
            return self.module
        strats, transfers = ss.dimensions.parse_stratify(kwargs, graphs=[self])
        old = {s.name:s for s in self.strats}
        if strats and all(strat.name in old and strat.same_as(old[strat.name]) for strat in strats) and all(name in self.transfers for name in transfers):
            ss.note(f'"{self.module.name}" is already stratified by {sc.strjoin(strat.name for strat in strats)} in the same way, so this stratify() does nothing (stratify() works in place)')
            return self.module
        for strat in strats:
            if strat.name in old:
                raise ValueError(f'"{self.module.name}" is already stratified by "{strat.name}", in another way. stratify() works in place, so the stratification can be given once; to change it, build the model again.')
            self.strats.append(strat)
            for key,leaf in ({} if strat.mobility is None else strat.mobility.fitted()).items(): # A parameter of the movement between places, e.g. ss.geo.Mobility(..., away=ss.par(0.1, bounds=[0.01, 0.5])), so it can be fitted
                if not any(p is leaf for p in self.pars.values()):
                    self._add_par(leaf, name=leaf.name or key)
                    self.module.define_pars(**{leaf.name:leaf})
        for name,tf in transfers.items():
            tf.name = name
            tf.module = self.module
            self.add(self.transfers, name, tf)
            self.module.setattribute(name, tf) # So it can be the target of ss.modify()
            for expr in [tf.rate, tf.dur]: # Register its parameters, as for those of the transitions
                for leaf in [] if expr is None else expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        self._add_par(leaf, name=leaf.name or f'rate_{name}')
                        self.module.define_pars(**{leaf.name:leaf})
        return self.module

    def add_flow(self, tr):
        """ Add a birth, death, or migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...)])`), after the graph is built; see `ss.solvers.choose.init_flows()` """
        name = tr.name or ('births' if tr.per_capita else 'immigration' if tr._inflow else 'emigration' if tr.migration else 'deaths')
        if name in self.transitions:
            raise ValueError(f'"{self.module.name}" already has a transition named "{name}"; give the one in demographics= another name, e.g. ss.birth(..., name="{name}2")')
        tr.name = name
        if tr._inflow: # Born into (or arriving in) the state given, or else the state people are in unless they are put elsewhere
            dst = tr.dest[0] if tr.state_given else self.default_state
            if not any(s is dst for s in self.people_states):
                raise sc.KeyNotFoundError(f'"{name}" in demographics= adds people to "{getattr(dst, "name", None)}", which is not a state of people of "{self.module.name}"; its states are {sc.strjoin(s.name for s in self.people_states)}')
            tr.dest = [dst]
        if tr.rate is None and tr.dur is None:
            raise ValueError(f'"{name}" in demographics= has no rate; give it one, e.g. ss.birth(rate=ss.peryear(0.02))')
        tr.const, tr.terms = linear(tr.rate) if tr.rate is not None else (None, {})
        self.fill_people(tr)
        tr.contact = False
        tr.acquisition = False
        tr.people = True
        tr.module = self.module
        slots = {id(p) for slot,p in self.set_slots(tr)}
        self.add(self.transitions, name, tr)
        self.module.setattribute(name, tr) # e.g. sir.births, so it can be the target of ss.modify()
        for expr in [tr.rate, tr.dur]: # Register its parameters, as for those of the transitions
            for leaf in [] if expr is None else expr.leaves:
                if not isinstance(leaf, par) or any(p is leaf for p in self.pars.values()):
                    continue
                if leaf.literal and id(leaf) not in slots: # A value inside a rate (e.g. the trend in ss.peryear(0.01) * ts), which stays a number, as in a transition (see resolve_pars())
                    continue
                self._add_par(leaf, name=leaf.name if not leaf.literal and leaf.name else f'rate_{name}')
                self.module.define_pars(**{leaf.name:leaf})
        self.resolve_exits()
        return tr

    @property
    def infectious(self):
        """ The states that are a source of infection """
        return [s for s in self.states.values() if s.infectious]

    @property
    def count_sources(self):
        """ The states whose numbers or shares drive an acquisition at a rate on counts of them, which are the sources of infection without a transmission: `I` in `S >> I @ beta*I.n` (density-dependent), or `beta*I.frac` """
        out = {}
        for tr in self.acquisitions:
            if tr.rate is not None and not tr.contact:
                out.update({id(leaf.states[0]):leaf.states[0] for leaf in self.counts_in(tr) if len(leaf.states) == 1
                            and leaf.states[0].people and leaf.states[0].module is self.module and not any(leaf.states[0] is src for src in tr.source or [])})
        return list(out.values())

    @property
    def met_counts(self):
        """
        The counts in a rate that a contact matrix acts on (see `ss.solvers.Mixing.count_terms()`): in the rate of an acquisition, the number
        (`I.n`, or weighted: `(ss.by(w)*I).n`, `I.n_infectious`) or share (`I.frac`) of the people in states of this disease that it does not take
        people from, as `I` in `S >> I @ beta*I.n`; not one inside a function of the rate (e.g. `np.clip(I.n/K, 0, 1)`), which modifies the rate by
        the total rather than being met
        """
        def outside(expr): # The leaves of a rate outside any function in it
            if isinstance(expr, Func):
                return []
            if isinstance(expr, BinOp):
                return outside(expr.a) + outside(expr.b)
            return expr.leaves
        out = {}
        for tr in self.acquisitions:
            for leaf in (outside(tr.rate) if tr.rate is not None else []):
                if isinstance(leaf, Count) and leaf.kind in ['n', 'frac'] and all(s.people and s.module is self.module and not any(s is src for src in tr.source or []) for s in leaf.states):
                    out[id(leaf)] = leaf
        return list(out.values())

    @property
    def transmissions(self):
        return [tr for tr in self.transitions.values() if tr.contact]

    @property
    def acquisitions(self):
        """ The transitions that are an acquisition of the disease: see `resolve_acquisition()` """
        return [tr for tr in self.transitions.values() if tr.acquisition]

    @property
    def uninfected(self):
        """ The names of the states people can be in without ever having been infected: those reached from the first state by transitions that are not an acquisition """
        seen = []
        todo = [self.default_state] if self.default_state is not None else []
        while todo:
            state = todo.pop()
            if state.name not in seen and state.people:
                seen.append(state.name)
                todo += [dst for tr in self.transitions.values() if not tr.acquisition and state in tr.source for dst in tr.dest]
        return seen

    @property
    def infection_sources(self):
        """ The sources of infection: of people by contact (`I` in `beta*I`), density-dependent (`beta*I.n`), or of a count (`Ih` in a bite rate `Ih.n / N` of mosquitoes) """
        sources = {id(s):s for s in self.infectious + self.count_sources}
        for tr in self.transitions.values():
            if not tr.people and tr.rate is not None:
                sources.update({id(leaf.states[0]):leaf.states[0] for leaf in tr.rate.leaves if isinstance(leaf, Count) and len(leaf.states) == 1 and leaf.states[0].people})
        return list(sources.values())

    @property
    def marked(self):
        """ The states the model says are infected: marked (`ss.State(infected=True)`), or in its own result `infected` if that is a sum or set of its states (`infected = E + I`, or `E | I`) """
        out = {id(s):s for s in self.states.values() if s.infected_given}
        out.update({id(s):s for s in summed_states(self.exprs.get('infected')) or []})
        return list(out.values())

    @property
    def infected(self):
        """
        The states people carry an infection in, for what needs them (e.g. the states strains are carried in, or the number infected at the start): those
        the model says are infected (see `marked`), if any, with the sources of infection (a state that infects others carries it, said or not).
        Otherwise, as worked out from the transitions: the states people are infected into, and the sources of infection (see `infection_sources`).
        The results `infected` and `prevalence`, and each state's `infected`, are these without a source the model left out (see `resolve_derived()`)

        In a model where nobody is a source of infection (a staged condition such as a cancer), also every state that people reach
        from those without passing through a state they can be in without having been infected; these are not infected for the results,
        which such a model does not have unless it says which states are infected
        """
        sources = {id(s):s for s in self.infection_sources}
        marked = self.marked
        if marked:
            return sorted({**{id(s):s for s in marked}, **sources}.values(), key=lambda s: s.order)
        out = {id(s):s for tr in self.acquisitions for s in tr.dest}
        if not sources:
            stop = set(self.uninfected)
            todo = list(out.values())
            while todo:
                state = todo.pop()
                for tr in self.transitions.values():
                    if state in (tr.source or []) and tr.people and not tr.acquisition:
                        for dst in tr.dest or []:
                            if dst.people and dst.name not in stop and id(dst) not in out:
                                out[id(dst)] = dst
                                todo.append(dst)
        out.update(sources)
        return sorted(out.values(), key=lambda s: s.order)

    def unit_values(self):
        """ All the typed values (durations and rates) in the graph, used to infer the sim's unit """
        vals = [p.raw for p in self.all_pars()] + [val for oc in self.outcomes.values() for val in [oc.delay, oc.dur]]
        return unit_values(vals)

    def modifier_kind(self, name):
        """ What a modifier on a transition acts on: its probability if it is one branch of several with p= (chosen on entry) or one destination of a transition with several (`'recovery.R'`), and otherwise its rate """
        tr = self.transitions.get(name)
        if tr is not None and tr.p is not None and not tr._split and not tr.contact:
            return 'branch'
        if self.destination(name) is not None:
            return 'branch'
        return 'transition'

    def dwell_not_branch(self, name, how, modname):
        """ For a multiplier (`how`, e.g. 'the modifier "m"') on a transition with several destinations chosen with p= and a dwell time: a note that it changes the dwell, not a branch, naming the destinations that would; otherwise None """
        tr = self.transitions.get(name)
        if tr is None or tr.p is None or tr.dest is None or len(tr.dest) < 2 or tr.dur is None:
            return None
        dsts = sc.strjoin([f"'{modname}.{name}.{dst.name}'" for dst in tr.dest])
        return f'{how} on "{name}" changes how long people take to leave {sc.strjoin(s.name for s in tr.source)}, not which destination they go to; to change the probability of one destination, name it instead: {dsts}'

    def destination(self, name):
        """ For the name of one destination of a transition with several, `'<transition>.<state>'` (e.g. `'progression.C'`, or `'outcome.dead'` and `'outcome.stay'` for ss.dead and ss.stay): the transition; otherwise None """
        trname, _, dst = name.rpartition('.')
        tr = self.transitions.get(trname)
        if tr is None or not tr._split or dst not in [d.name for d in tr.dest] + ['dead']*tr.may_die + ['stay']*tr.stays:
            return None
        return tr

    def destinations(self):
        """ The names of the destinations of the transitions with several, as `destination()` takes them """
        return [f'{name}.{end}' for name,tr in self.transitions.items() for end in tr._ends]

    def leaves(self):
        """ The leaves of the rates and branch probabilities of its transitions """
        exprs = [expr for tr in self.transitions.values() for expr in [tr.rate] + sc.tolist(tr.p)]
        return [leaf for expr in exprs if isinstance(expr, Expr) for leaf in expr.leaves]

    def all_pars(self):
        """ Every parameter in the graph, named or inline """
        out = {id(p):p for p in self.pars.values()}
        exprs = [expr for tr in self.transitions.values() for expr in [tr.rate, tr.dur, tr.noise] + sc.tolist(tr.p)]
        exprs += [w for expr in exprs if expr is not None for w in count_weights(expr)]
        for expr in exprs:
            if expr is not None:
                out.update({id(leaf):leaf for leaf in expr.leaves if isinstance(leaf, par)})
        return list(out.values())

    def spread_pars(self):
        """ The parameters whose value is a distribution across people (e.g. `beta=ss.lognormal(0.3, 0.1)`), other than dwell times, which are stages or are sampled on entry """
        dwells = [tr.dur for tr in self.transitions.values()]
        dwells += [scaled[0] for tr in self.transitions.values() if tr.dur is not None and isinstance(tr.dur.raw, Expr) and (scaled := ss.transitions.scaled_dist(tr.dur.raw)) is not None] # A distribution times a factor, e.g. dur=d*ss.weeks(1), is drawn on entry too
        return [p for p in self.all_pars() if isinstance(p.raw, ss.Dist) and not any(p is dur for dur in dwells)]


def summed_states(expr):
    """ The states a result is the number in (`E + I`, or `E | I`, as a result), or None if it is anything else """
    if isinstance(expr, ss.expressions.Stock):
        return [expr.state]
    if isinstance(expr, BinOp) and expr.op == '+':
        a, b = summed_states(expr.a), summed_states(expr.b)
        return None if a is None or b is None else a + b
    return None


def count_weights(expr):
    """ The weights of the weighted counts in an expression (`w` in `(w*I).n`), whose parameters are the module's too """
    return [w for leaf in expr.leaves if isinstance(leaf, ss.expressions.Count) and leaf.weights for w in leaf.weights.values()]


def unit_values(vals):
    """ The units of the typed values (durations and rates) among some values, including those inside distributions and values by level or in time """
    out = []
    def check(val):
        if isinstance(val, ss.Duration) and not isinstance(val, ss.dates.datedur):
            out.append(type(val).base)
        elif isinstance(val, ss.Rate) and val.unit is not None:
            out.append(type(val.unit).base)
        elif isinstance(val, ss.Dist):
            if val.unit is not None:
                out.append(val.unit.base)
            for v in val.pars.values():
                check(v)
        elif isinstance(val, TimeValue):
            for v in val.typed:
                check(v)
        elif isinstance(val, by):
            for v in val.items:
                check(v)
    for val in vals:
        check(val)
    return out
