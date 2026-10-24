"""
The graph of one module: its states, transitions, and parameters, collected, named, completed with defaults (each guess recorded), and checked
"""
import types
import itertools
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import as_expr, BinOp, Count, count_power, did_you_mean, Expr, is_one, linear, par, per_person, Row, Share, to_pars
from .parameters import by, TimeValue
from .transitions import FlowSum, outcome, State, Transition
from .presentation import Presentation, TransitionTable

__all__ = ['Graph']


def inner_pars(val):
    """ The parameters inside a value: of a value that varies in time (`ss.seasonal(mean=beta)`) or a probability per act (`ss.peract(m2f=b)`), also within a beta for each route """
    if isinstance(val, (TimeValue, ss.peract)):
        return val.pars
    elif isinstance(val, dict):
        return [q for v in val.values() for q in inner_pars(v)]
    return []


def bad_end(end, side):
    """ The message for a source or destination of a transition that is not a state: what to write instead """
    null, stay = ss.transitions.null, ss.transitions.stay
    if end is null and side == 'source':
        return 'ss.null cannot be one of several sources of a transition: for an inflow, give it alone, e.g. ss.Transition(ss.null, S, rate=0.1)'
    elif end is stay and side == 'source':
        return 'ss.stay is a destination, not a source: give it as one destination of a split, e.g. ss.Transition(I, [R, ss.stay], p=[0.3, None], dur=10)'
    elif end is null or end is stay:
        return f'{end} is one destination of a split, with at least one other, e.g. ss.Transition(I, [R, {end}], p=[0.9, 0.1], dur=10)'
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
    return f'the destination of a transition is a state or a list of states (a split, with p=), not {end}; for a split between them, write [{", ".join(names)}] with p=[...]'


class Graph(Presentation):
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
        first (str): the state people are in unless they are put elsewhere (by default, the one named `S` or `susceptible`; see `resolve_people()`)
        names (dict): the names of the variables of the code that made the module, by the id of what they hold (see `ss.utils.caller_names()`), which name the states, transitions, and parameters given without one
    """
    grouped = ['transitions', 'results', 'outcomes'] # The keywords that each take several members, e.g. ss.Disease(transitions=[...])

    def __init__(self, module, args=None, members=None, init_prev=None, init=None, init_n=None, first=None, names=None):
        self.module = module
        self.names = names or {}
        self.first_given = first # The state people are in unless they are put elsewhere, if given
        self.init_prev = init_prev
        if isinstance(init_prev, par): # So it can be fitted
            init_prev.name = init_prev.name or 'init_prev'
        self.init = dict(init or {})
        self.init_n = dict(init_n or {})
        self.states = sc.objdict()
        self.transitions = TransitionTable()
        self.pars = sc.objdict()
        self.outcomes = sc.objdict()
        self.exprs = sc.objdict() # Results that are sums or ratios of states
        self.funcs = sc.objdict() # Custom per-step behavior
        self.attrs = sc.objdict() # Values for each person: see ss.attr
        self.derived = [] # Stratifications by a derived attribute, which add no groups: see stratify()
        self.guessed = sc.objdict()
        self.strats = [] # Stratifications, filled by stratify()
        self.transfers = sc.objdict() # Movement between the levels of a stratification

        self.collect(list(args or []), dict(members or {}))
        if isinstance(init_prev, par):
            self._add_par(init_prev, name=init_prev.name)
        self.resolve_states()
        self.resolve_transitions()
        self.resolve_init_prev()
        self.expand_rows()
        self.resolve_people()
        self.resolve_attrs()
        self.resolve_acquisition()
        self.resolve_pars()
        self.set_keywords()
        self.resolve_exits()
        self.check()
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
        for name,obj in named:
            if isinstance(obj, FlowSum): # A sum or a share of flows, e.g. recovery + death or 0.5 * infection, is an outcome
                obj = outcome(obj)
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
            if isinstance(obj, State) and None not in (name, obj.name) and name != obj.name: # A named state under another name, e.g. n_infected = I, is a result: the number in the state
                self.add(self.exprs, name, ss.expressions.as_stocks(obj))
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
            elif isinstance(obj, Expr) and any(isinstance(leaf, ss.expressions.attr) for leaf in obj.leaves): # A helper of attributes, e.g. gain = cd4_potential - cd4_preart, used through the objects
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
            raise ss.errors.ReservedName(f'"{name}" cannot be used as the name of a state, transition, parameter, or result, since it is an attribute of every module. Please choose another name.')
        for other in [self.states, self.transitions, self.pars, self.outcomes, self.exprs, self.funcs, self.transfers, self.attrs]:
            if name in other and other[name] is not obj:
                raise ss.errors.DuplicateDeclaration(f'two members of "{self.module.name}" are both named "{name}"; names must be unique within a module.')
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
            raise ss.errors.DuplicateDeclaration(errormsg)
        return

    def resolve_states(self):
        """ Find the states: the ones declared, and those the transitions use """
        trs = list(self.transitions.values()) + self._unnamed
        found = {id(s):s for s in self._declared}
        self._refs = [] # States used only inside a rate
        for tr in trs:
            for side,ends in [('source', tr.src or []), ('destination', tr.dst or [])]:
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
        for s in self._refs: # A state used only in a rate still belongs to this module unless another module has claimed it
            if s.module is None or s.module is self.module:
                found[id(s)] = s
        statelist = sorted(found.values(), key=lambda s: s.order)

        # Name the states
        for i,state in enumerate(statelist):
            if state.name is None:
                state.name = self.names.get(id(state))
            if state.name is None:
                used = [tr.name or tr.arrow for tr in trs if any(end is state for end in (tr.src or []) + (tr.dst or []))]
                raise ss.errors.UnnamedState(f'state number {i+1} of the module has no name. It is used in {used}, and no variable or keyword holding it was found.\n'
                                      f"     Give the names up front: S, I, R = ss.states('S I R').")
        for state in statelist:
            if state.module is not None and state.module is not self.module:
                raise ss.errors.DuplicateDeclaration(f'state "{state.name}" is used by transitions in two modules, "{state.module.name}" and "{self.module.name}"; a state belongs to one module.')
            state.module = self.module
            self.add(self.states, state.name, state)

        # Initial conditions given on the module
        for key,given in [('init', self.init), ('init_n', self.init_n)]:
            for name,val in given.items():
                if name not in self.states:
                    raise ss.errors.UnknownState(f'{key}= names a state "{name}" that "{self.module.name}" does not have. States are {sc.strjoin(self.states.keys())}.')
                setattr(self.states[name], key, val)
        for state in statelist:
            if state.init is not None and state.init_n is not None:
                raise ValueError(f'"{state.name}" of "{self.module.name}" is given both init= (a share of people) and init_n= (a number of people); give one or the other')
        return

    def resolve_transitions(self):
        """ Give every transition a name, a source, and a rate or a dwell time """
        # Fallback names
        for tr in self._unnamed:
            src = '_'.join([s.name for s in tr.src or []]) or 'in'
            dst = '_'.join([s.name for s in tr.dst or []]) or 'out'
            base = f'{src}_to_{dst}' if tr.dst is not None else 'immigration' if tr.migration else 'births' # A birth with no state given is into the first state, found later
            name = base
            count = 1
            while name in self.transitions:
                count += 1
                name = f'{base}_{count}'
            tr.name = name
            tr.guessed.setdefault('name', 'built from its endpoints')
            self.add(self.transitions, name, tr)

        for name,tr in self.transitions.items():
            if tr.dst is None and not (tr.inflow and not tr.state_given): # A birth with no state is into the first state (see resolve_people())
                errormsg = f'The transition "{name}" has no destination'
                raise ValueError(errormsg)

            if tr.undriven: # Built before (the same transition in another module): its rate of 0 is set again in resolve_exits()
                tr.rate = None
            if tr.rate is None and tr.dur is None and tr.p is None and tr.trigger is None and getattr(tr, 'per', None) is None and not tr.every_state:
                if not tr.src:
                    raise ss.errors.UndrivenTransition(f'the inflow "{name}" ({tr.arrow}) has no rate. Give it one, e.g. {tr.arrow.replace(" -> ", " >> ")} @ 0.1')
                tr.undriven = True

            # Derive the kind from the rate, written per person (a rate written as the whole flow, e.g. beta*S*I/N, is divided by its source)
            if tr.rate is not None:
                rate = per_person(tr.rate, tr.src, [st for st in self.states.values() if st.people])
                if rate is not tr.rate:
                    tr.flow_given, tr.rate = tr.rate, rate
            tr.const, tr.terms = (None, {}) if tr.rate is None else linear(tr.rate)
            tr.contact = bool(tr.terms)
            for state in tr.terms:
                state.infectious = True
        return

    def resolve_init_prev(self):
        """ By default, 1% of people start infected, seeded through the transmission (frequency- or density-dependent), unless a state is given a starting value """
        given = any(state.given_init for state in self.states.values())
        counted = lambda tr: {id(st) for leaf in tr.rate.leaves if isinstance(leaf, Count) for st in leaf.states if st.module is self.module} - {id(st) for st in tr.src or []} # The module's states whose numbers the rate reads, other than its source: I in beta*I.n
        transmitted = any(tr.contact or (tr.rate is not None and count_power(tr.rate) > 0 and counted(tr)) for tr in self.transitions.values())
        if self.init_prev is None and not given and transmitted:
            self.init_prev = par(0.01)
            self._add_par(self.init_prev, name='init_prev')
            self.guessed['init_prev'] = 'the default, 1%'
        return

    def unset_default_init_prev(self):
        """ Remove the default init_prev, e.g. of a strain that its disease seeds """
        if self.guessed.pop('init_prev', None) is not None:
            self.init_prev = None
            self.pars.pop('init_prev')
            self.module.pars.pop('init_prev', None)
        return

    def expand_rows(self):
        """
        Entries of `p=` with a `to` axis (an `ss.by` over `to`, whose labels are destination states): one `Share` of the entry for each destination it covers

        Entries without one are one destination each, in order, skipping the destinations that a `to` axis names; an entry whose
        `to` axis gives values by position only covers the destinations left. The entries as given are kept as `tr.p_given`.
        """
        for name,tr in self.transitions.items():
            if not tr.p or len(tr.dst or []) < 2 or not any(to_pars(p) for p in tr.p):
                continue
            names = [s.name for s in tr.dst]
            rows = {i:[lab for q in to_pars(p) for lab in (q.raw.to_labels() or [])] for i,p in enumerate(tr.p) if to_pars(p)}
            named = [lab for labs in rows.values() for lab in labs]
            for lab in named:
                if lab not in names:
                    raise ss.errors.UnknownState(f'p= of "{name}" has a value over "to" for "{lab}", which is not one of its destinations ({sc.strjoin(names)}){did_you_mean(lab, names)}')
            plain = [i for i in range(len(tr.p)) if i not in rows]
            free = [n for n in names if n not in named]
            unlabeled = [i for i,labs in rows.items() if not labs]
            if len(unlabeled) > 1 or (not unlabeled and len(free) != len(plain)) or len(free) < len(plain):
                raise ss.errors.BadBranchSplit(f'"{name}" has {len(names)} destinations, but its p= has {len(plain)} entries of one destination each and values over "to" for {sc.strjoin(named) or "the others"}; give one entry for each destination that no value over "to" names')
            for i in unlabeled:
                rows[i] = free[len(plain):]
            covered = {lab:i for i,labs in rows.items() for lab in dict.fromkeys(labs)}
            built = {i:Row(tr.p[i], list(dict.fromkeys(labs))) for i,labs in rows.items()}
            entries = iter(plain)
            tr.p_given = list(tr.p)
            tr.p = [Share(built[covered[n]], n) if n in covered else tr.p[next(entries)] for n in names]
        return

    def resolve_people(self):
        """
        Which states are people: every state, unless it is declared a count (`ss.State(people=False)`, e.g. mosquitoes or
        bacteria in the water); and the first state, where people are unless they are put elsewhere: `first=` if given,
        else the state named `S` or `susceptible` (in any case). Nothing else is guessed: with neither, `check_first()` raises
        when the sim is initialized (a birth into a state of its own, `ss.birth(X)`, needs no first state, but the people who
        start in no state do)
        """
        states = list(self.states.values())
        self.people_states = [s for s in states if s.people]
        self.count_states = [s for s in states if not s.people]
        if self.first_given is not None:
            name = self.first_given.name if isinstance(self.first_given, State) else self.first_given
            if name not in self.states:
                raise ss.errors.UnknownState(f'first="{name}" is not a state of "{self.module.name}". Its states are {sc.strjoin(self.states.keys())}{did_you_mean(name, list(self.states.keys()))}')
            self.first = self.states[name]
        elif any(s.name.lower() in ['s', 'susceptible'] for s in self.people_states):
            self.first = next(s for s in self.people_states if s.name.lower() in ['s', 'susceptible'])
        else:
            self.first = None # Raises when the sim is initialized: see check_first()
        for tr in self.transitions.values(): # A birth or immigration with no state given
            if tr.dst is None and self.first is not None:
                tr.dst = [self.first]
        self.resolve_rest()

        # Now that the people are known, fill in the things that depend on them
        for name,tr in self.transitions.items():
            self.fill_people(tr)
            for state in tr.terms:
                if not state.people:
                    raise ss.errors.ExposureToCount(f'the rate for "{name}" uses the bare state {state.name}, and {state.name} is not people.\n'
                                             f'       rate = {tr.rate}\n'
                                             f'     A bare state means contact with people in that state, and there is no contact with a count.\n'
                                             f'     Write {state.name}.n for the number itself, or leave out people=False if {state.name} is people.')
            ends = (tr.src or []) + (tr.dst or [])
            tr.people = all(s.people for s in ends)
            if not tr.people and any(s.people for s in ends):
                counts = [s.name for s in ends if not s.people]
                raise ss.errors.PeopleAndCounts(f'"{name}" moves between people ({sc.strjoin(s.name for s in ends if s.people)}) and counts ({sc.strjoin(counts)}, declared with people=False). '
                                         f'If {sc.strjoin(counts)} {"is" if len(counts) == 1 else "are"} people, leave out people=False.')
        self.check_counts()
        return

    def check_counts(self):
        """
        Refuse a forgotten `people=False` (MA35): a group of states of people that no transition links to the first state
        and that only an inflow (not a birth) fills, as mosquitoes are, would otherwise silently become agents
        """
        links = {id(s):set() for s in self.people_states}
        if id(self.first) not in links: # No first state given (sim.init() raises), or not one of people
            return
        for tr in self.transitions.values():
            ends = [s for s in (tr.src or []) + (tr.dst or []) if s.people]
            for s in ends:
                links[id(s)].update(id(e) for e in ends)
        group, todo = set(), [id(self.first)]
        while todo: # The states linked to the first, in either direction
            i = todo.pop()
            if i not in group:
                group.add(i)
                todo += links[i]
        for tr in self.transitions.values():
            dst = [s for s in tr.dst or [] if id(s) not in group]
            if tr.inflow and tr.people and dst and not getattr(tr, 'per_capita', False) and tr.per is None and not any(s.people_given for s in dst):
                others = [s.name for s in self.people_states if id(s) not in group]
                raise ss.errors.PeopleAndCounts(f'"{tr.name}" ({tr.arrow}) fills {sc.strjoin(others)}, which no transition links to {self.first.name}, where people start, so {"it looks like" if len(others) == 1 else "they look like"} a count of something else (e.g. mosquitoes). '
                                                f'Every state is people unless declared a count: if {"it is" if len(others) == 1 else "they are"} not people, declare {sc.strjoin(others)} with ss.State(people=False) (or ss.states(n, people=False)); if {"it is" if len(others) == 1 else "they are"} people, say so with people=True.')
        return

    def check_first(self, why=None):
        """ Raise if the module has states of people but no first state (see `resolve_people()`), which every solver needs: it is where people start unless they are put elsewhere, and where those added later (born, or seeded from) come from """
        if self.first is None and self.people_states:
            names = [s.name for s in self.people_states]
            errormsg = (f'"{self.module.name}" has no first state, the state people {why or "start in unless they are put elsewhere, and are born into"}. It is the state named S or susceptible, '
                        f'and "{self.module.name}" has neither (its states are {sc.strjoin(names)}).\n'
                        f'     Say which with first=, e.g. ss.Disease(..., first="{names[0]}")')
            raise ss.errors.UnknownState(errormsg)
        return

    def resolve_rest(self):
        """
        Work out where the people not given an initial state start: the first state, unless it is given an initial value of
        its own, which is then kept (`init=dict(S=0.1, I=1e-4)` starts 10% susceptible). Then it is the one state of people
        with no initial value, with a note in `guessed`; with none or several, there is no such state, and the initial values
        must sum to 1 (see `Engine.init_fractions()`)
        """
        seeded = {id(st) for tr in self.transitions.values() if tr.contact for st in tr.dst or []} if self.init_prev is not None else set() # Where init_prev puts people
        both = [s.name for s in self.people_states if id(s) in seeded and s.given_init]
        if both:
            errormsg = f'"{self.module.name}" is given both init_prev= and init= for {sc.strjoin(both)}, where init_prev= puts people; give one or the other'
            raise ValueError(errormsg)
        self.rest = self.first
        if self.first is None or not self.first.given_init:
            return
        unset = [s for s in self.people_states if not s.given_init and id(s) not in seeded]
        if len(unset) == 1:
            self.rest = unset[0]
            self.guessed['rest'] = f'{self.rest.name}, the only state with no initial value, since {self.first.name} has one'
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
        first = self.first
        out_of_first = [tr for tr in self.transitions.values() if any(src is first for src in tr.src or [])]
        by_contact = [tr.name for tr in out_of_first if tr.contact]
        for name,tr in self.transitions.items():
            given = getattr(tr, 'acquisition_given', None)
            if given is not None:
                tr.acquisition = bool(given)
            elif tr.contact:
                tr.acquisition = True
            elif tr in out_of_first and tr.people and not tr.outflow:
                tr.acquisition = not by_contact
                tr.guessed['acquisition'] = f'contact-mediated {sc.strjoin(by_contact)} is' if by_contact else f'out of {first.name}'
            else:
                tr.acquisition = False
        return

    def fill_people(self, tr):
        """ Fill in what depends on which states are people: the sources of the deaths of everyone (`ss.death()` with no state), and the people a birth rate is per person of (`ss.birth()`) """
        if tr.every_state:
            tr.src = list(self.people_states)
        if tr.per_capita:
            tr.rate = tr.rate * Count(list(self.people_states), kind='n')
            tr.const, tr.terms = linear(tr.rate)
        return

    def set_keywords(self):
        """ A value given as a keyword sets the parameter of that name; there is nothing else for it to be """
        for name,value in self._keywords.items():
            if name not in self.pars and name.startswith(('init', 'seed')): # e.g. init_infected=100
                state = self.infectious[0].name if self.infectious else 'I'
                raise ss.errors.UnknownParameter(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}".\n'
                                          f'     To start with people in a state, use init=dict({state}=0.01) for a share of people, or init_n=dict({state}=10) for a number; init_prev= seeds the state a new infection enters (by default 1% of people), as a share or a number of people.')
            elif name not in self.pars:
                raise ss.errors.UnknownParameter(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}"; its parameters are {sc.strjoin(self.pars.keys())}.\n'
                                          f'     To add a parameter, use it in a transition, e.g. infection = S >> I @ {name} * I with {name} = ss.par({value}).\n'
                                          f'     If another module reads it from this disease, add it after making the disease: {self.module.name}.define_pars({name}={value!r}).' +
                                          (f'\n     Condoms (or other protection used in some acts) are an intervention on the transmission instead, e.g. ss.modify({self.module.name}.infection, efficacy={value!r}, share=0.3, route="sexualnet").' if 'condom' in name else ''))
            self.pars[name].value = value
        return

    @staticmethod
    def set_slots(tr):
        """
        Record the kind of value each parameter of a transition is, as `par.slot` (which describe() and the model spec use):
        'rate' or 'dur', or None for a plain number (e.g. a factor of a rate); returns the standard slots it fills, as (slot name, parameter)
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
            elif tr.contact and tr.const is None: # The coefficient of the source of infection is beta
                coefs = list(tr.terms.values())
                rate = tr.rate
                beta = lambda leaf: isinstance(leaf, par) and (not leaf.literal or isinstance(leaf.raw, (dict, ss.peract)) or ss.expressions.value_dimension(leaf.raw) is not None) # A parameter, a beta for each route, or a typed value
                alone = isinstance(rate, BinOp) and not any(beta(leaf) for leaf in rate.b.leaves) # Only the rate as a whole has a unit, so a factor of a product of parameters (or of a typed beta) is a number: R0 in R0 * (gamma * I), or 0.275 in 0.275*(dict(sexualnet=ss.peract(0.05))*I)
                if isinstance(rate, BinOp) and rate.op == '*' and isinstance(rate.a, par) and alone:
                    rate.a.slot = 'rate'
                    slots.append(('beta', rate.a))
                elif len(coefs) == 1 and isinstance(coefs[0], par) and not is_one(coefs[0]):
                    coefs[0].slot = 'rate'
                    slots.append(('beta', coefs[0]))
        for p in tr.p or []:
            if isinstance(p, par) and len(tr.p) == 1:
                slots.append(('p', p))
        return slots

    def resolve_pars(self):
        """ Turn inline values in the standard slots into named parameters (beta, dur, rate, p) """
        slots = [(tr, slot, p) for tr in self.transitions.values() for slot,p in self.set_slots(tr)] # (transition, slot name, parameter)

        # Name the anonymous ones after their slot, qualified by the transition where two would clash
        anon = [(tr, slot, p) for tr,slot,p in slots if p.literal or getattr(p, 'slot_named', False) or (p.name is None and self.names.get(id(p)) is None)] # Including an ss.par written in the slot itself, e.g. beta=ss.par(0.3, bounds=[0.1, 1]), and one named after its slot before (in a class this one subclasses), which is named again
        anon = [(tr, slot, p) for i,(tr,slot,p) in enumerate(anon) if not any(q is p for _,_,q in anon[:i])] # One used in several transitions (e.g. beta = ss.seasonal(...)) is one parameter, named after the first
        counts = sc.objdict()
        for tr,slot,p in anon:
            counts[slot] = counts.get(slot, 0) + 1
        inner = {q.name or self.names.get(id(q)) for tr,slot,p in slots for q in inner_pars(p.value)} # The names of parameters inside values that vary in time, e.g. ss.seasonal(mean=beta), or inside a probability per act, which are registered below
        inner |= {q.name or self.names.get(id(q)) for tr,slot,p in slots if isinstance(p.value, Expr) for q in p.value.leaves if isinstance(q, par) and q is not p} # And inside an expression in a slot, e.g. dur=dur_rec*k
        taken = lambda name: name in self.pars or name in self.states or name in self.transitions or name in inner
        for tr,slot,p in anon:
            name = f'{slot}_{tr.name}' if counts[slot] > 1 or taken(slot) else slot
            if taken(name): # e.g. a parameter of the user's called dur_rec, inside the dwell time of a transition called rec
                name = f'{slot}_of_{tr.name}'
            self._add_par(p, name=name)
            p.slot_named = True

        # Register the explicit ss.par objects used in the transitions, and in the weights and likelihoods of the outcomes
        used = [(tr.name, expr) for tr in self.transitions.values() for expr in [tr.rate, tr.dur] + sc.tolist(tr.p) + sc.tolist(getattr(tr, 'p_given', None))]
        used += [(oc.name, weight) for oc in self.outcomes.values() for weight in oc.weights]
        used += [(oc.name, val) for oc in self.outcomes.values() if oc.likelihood is not None for val in oc.likelihood.pars.values() if isinstance(val, par)]
        for oc in self.outcomes.values(): # A parameter written inside a likelihood, e.g. ss.nbinom(n=ss.par(10, bounds=[1, 100])), is named after the outcome and its keyword
            for key,val in (oc.likelihood.pars.items() if oc.likelihood is not None else []):
                if isinstance(val, par) and not val.literal and val.name is None and self.names.get(id(val)) is None:
                    val.name = f'{oc.name}_{key}'
        used += [(f'init of {state.name}', init) for state in self.states.values() for init in [state.init, state.init_n] if isinstance(init, Expr)] # So an initial count can be fitted
        used += [(name, expr) for name,tf in self.transfers.items() for expr in [tf.rate, tf.dur] + sc.tolist(tf.p)] # And a parameter in the rate of a transfer
        used += [(name, expr) for name,a in self.attrs.items() for expr in ([e for names,e in a.cases] if a.cases is not None else [a.value]) if isinstance(expr, Expr) and not isinstance(expr, ss.expressions.attr)] # And in an attribute
        used += [(name, val) for name,a in self.attrs.items() if a.kind in ['drawn', 'trait'] for p in ss.expressions.dists_in(a.value) for val in p.value.pars.values() if isinstance(val, par)] # And a parameter of a distribution an attribute is drawn from, e.g. ss.bernoulli(ss.par(0.6, name='sero'))
        used += [(name, leaf.value) for name,expr in list(used) if expr is not None for leaf in expr.leaves if isinstance(leaf, par) and isinstance(leaf.value, Expr)] # And inside a dwell time given as an expression (of attributes)
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
        self.exits = sc.objdict()
        for state in self.states.values():
            self.exits[state.name] = sc.objdict(branches=[], timed=[], hazards=[], contact=[], rest=[], triggered=[])
        self.inflows = []
        for tr in self.transitions.values():
            if tr.undriven: # To the engines, a rate of 0 that nothing modifies; a delivery moves people along it (see ss.Treatment)
                tr.rate = par(0.0, literal=True)
                tr.const, tr.terms = linear(tr.rate)
            if tr.inflow:
                self.inflows.append(tr)
            if tr.immediate and tr.where is not None and tr.trigger is None and (tr.p is None or tr.split): # dur=0 with where=: as soon as the condition holds, as ss.on() does (a branch with p= waits in its source until where= holds instead)
                tr.trigger = ss.on(tr.where)
            for state in tr.src:
                exits = self.exits[state.name]
                if tr.contact:
                    exits.contact.append(tr)
                elif tr.trigger is not None: # Moved as soon as its condition holds, or its event happens: see ss.on
                    exits.triggered.append(tr)
                elif tr.p is not None and not tr.split: # One branch of several out of this state
                    exits.branches.append(tr)
                elif tr.dur is not None and not tr.immediate:
                    exits.timed.append(tr)
                elif tr.rate is None and len(tr.dst) > 1 and not tr.immediate: # Several destinations and no wait: say how to write an immediate split
                    names = [d.name for d in tr.dst]
                    raise ss.errors.UndrivenTransition(f'the transition "{tr.name}" ({state.name} -> {sc.strjoin(names)}) has p= but no rate and no duration (dur=), so nothing says when people leave. '
                                                f'For a split as soon as people arrive, give it dur=0, e.g. ss.Transition({state.name}, [{", ".join(names)}], dur=0, p={[None if v is None else v.raw if isinstance(v, par) else v for v in tr.p]}); or give it a duration.')
                elif tr.rate is None: # Immediate: dur=0, one destination and ss.stay, or what the branches leave over
                    exits.rest.append(tr)
                else:
                    exits.hazards.append(tr)

        # A single timed transition with no p= (or an immediate split) takes whatever the branches leave over: the only one, or the only one for the same people (the same where=) as the branches; one with no driver and no p= is undriven, not the remainder
        for name,exits in self.exits.items():
            exits.remainder = None
            rest = exits.pop('rest')
            if len(rest) > 1:
                raise ss.errors.BadBranchSplit(f'{sc.strjoin(tr.name for tr in rest)} out of state "{name}" have no p=, rate, or duration; only one transition can take what the branches leave over.')
            elif rest:
                exits.remainder = rest[0]
            apart = [] # Timed exits for other people (another where=), which compete as before
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
                        why = f' {sc.strjoin(undriven)} {"has" if len(undriven) == 1 else "have"} no rate, duration, or p=, so {"is" if len(undriven) == 1 else "are"} undriven, not the remainder; for an immediate split, give the destinations together, e.g. ss.Transition({name}, [A, B], dur=0, p=[{total:n}, None]).' if undriven else ''
                        why += f' {sc.strjoin(tr.name for tr in driven)} {"has" if len(driven) == 1 else "have"} a rate and no p=, so {"it competes" if len(driven) == 1 else "they compete"} with the branches rather than taking the rest: give it p={1 - total:n}, or a duration (dur=), which takes the rest.' if total < 1 and driven and not exits.timed else ''
                        why += f' For the rest to stay in {name}, give the destinations as one split with ss.stay, e.g. [A, ss.stay] with p=[{total:n}, None].' if total < 1 else ''
                        raise ss.errors.BadBranchSplit(f'the branch probabilities out of state "{name}" sum to {total:n}, not 1 '
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
            tr.check_dimensions()
            if tr.trigger is not None and (tr.contact or tr.inflow or (tr.p is not None and not tr.split and not tr.trigger.outcome)): # On an outcome, p= is the share of its people who move; a split with ss.stay or ss.null is a split, though neither is a state
                raise ss.errors.BadBranchSplit(f'"{name}" is triggered by {tr.trigger}, so it cannot be {"a transmission" if tr.contact else "an inflow" if tr.inflow else "one branch of several with p="}; '
                                        f'give the destinations of a triggered split together, e.g. "{tr.arrow} | B" with p=[...].')
            for inc in [c for c in ss.expressions.nodes([tr.rate, tr.dur] + sc.tolist(tr.p)) if isinstance(c, ss.conditions.Cond) and c.op == 'incidence']: # A flow over a window, in a rate, would be a delay equation
                flow = inc.args[0]
                mod = ss.conditions.modname(getattr(flow, 'module', None)) or getattr(self.module, 'name', None)
                path = f'{mod}.{flow.name}' if mod else flow.name
                dst = (flow.dst or [None])[0]
                raise ss.CapabilityError(f'"{name}" reads the incidence of "{flow.name}" in its rate, which is not supported yet: a flow over a window, in a rate, needs the history of the flow in every solver.\n'
                                         f'     Use a count of a state instead, e.g. {getattr(dst, "name", "X")}.n, which is about the flow into it times the time people spend there. '
                                         f'The incidence can be used in where= of ss.modify(), e.g. ss.modify(..., multiply=0.6, where={path}.incidence(over=ss.days(7)) > 0.001)')
            for clock in ss.expressions.clocks_in([tr.rate, tr.dur] + sc.tolist(tr.p)): # A clock has a value only inside its states, and progress needs a scheduled dwell
                if clock.module not in [None, self.module] or not all(n in self.states for n in clock.names):
                    continue # Another module's clock: checked when it runs
                outside = [src.name for src in tr.src or [] if src.name not in clock.names]
                if outside:
                    raise ss.errors.ClockOutsideDomain(f'"{name}" reads {clock}, which has a value only for people in {sc.strjoin(clock.names)}, but moves people from {sc.strjoin(outside)}.')
                if clock.kind == 'progress':
                    ex = self.exits[clock.names[0]]
                    if not (ex.branches or ex.remainder or ex.timed):
                        raise ss.errors.NoScheduledDwell(f'"{name}" reads {clock}, but {clock.names[0]} has only rates out of it, so no dwell time is scheduled when people enter it.')
                if len(tr.dst or []) <= 1 and any(c is clock for c in ss.expressions.clocks_in(sc.tolist(tr.p))):
                    raise ss.CapabilityError(f'"{name}" has p= with {clock}, but a branch with p= is decided when people enter {sc.strjoin(clock.names)}, when no time has passed. '
                                             f'Write one transition with several destinations, decided when they leave, e.g. ss.Transition({tr.src[0].name}, [{tr.dst[0].name}, ...], dur=..., p=[f({clock}), None])')
            if tr.stays and not tr.dst and not tr.dies:
                raise ss.errors.BadBranchSplit(f'"{name}" has ss.stay as its only destination; give it with another, e.g. [B, ss.stay] with p=[0.3, None].')
            if tr.p is not None and tr.split:
                ps = tr.p
                if len(ps) == 1 and len(tr.dst) == 2 and not tr.stays: # With two destinations, a single value is the first one's probability
                    ps = tr.p = [ps[0], None]
                ends = ps + [tr.p_stay]*tr.stays + [tr.p_die]*tr.dies # One entry for each destination, with ss.stay and ss.null
                if len(ps) != len(tr.dst):
                    raise ss.errors.BadBranchSplit(f'"{name}" has {len(tr.dst) + tr.stays + tr.dies} destinations but {len(ps) + (tr.p_stay is not None) + (tr.p_die is not None)} branch probabilities; give one for each (or None for the one that takes the rest).')
                given = [p for p in ends if p is not None]
                if len(given) < len(ends) - 1:
                    which = [label for label,p in zip([d.name for d in tr.dst] + ['ss.stay']*tr.stays + ['ss.null']*tr.dies, ends) if p is None]
                    raise ss.errors.BadBranchSplit(f'"{name}" leaves the rest to {sc.strjoin(which)}; only one destination can take the rest, so give the others a p=.')
                if all(isinstance(p, par) and sc.isnumber(p.raw) for p in given):
                    total = sum(p.raw for p in given)
                    if (len(given) == len(ends) and not np.isclose(total, 1)) or total > 1 + 1e-9:
                        raise ss.errors.BadBranchSplit(f'the branch probabilities of "{name}" sum to {total:n}, not 1.')
            elif tr.p is None and tr.split:
                raise ss.errors.BadBranchSplit(f'"{name}" has {len(tr.dst)} destinations but no branch probabilities; give them with p=[...].')
            if tr.dst is not None and len(tr.dst) > 1: # Each destination's flow is res.<transition>.<state>, so it cannot be named like an attribute of a result
                clash = [d.name for d in tr.dst if d.name in ss.Result.reserved()]
                if clash:
                    raise ss.errors.DuplicateDeclaration(f'the flow of "{name}" to {sc.strjoin(clash)} is a result read as res.{name}.{clash[0]}, which is already an attribute of every result; rename the state {sc.strjoin(clash)}')
        for name,a in self.attrs.items(): # Each case of a derived attribute reads clocks only of states it is for
            for names,expr in (a.cases or []):
                for clock in ss.expressions.clocks_in(expr):
                    if not all(any(c is s for s in self.states.values()) for c in clock.states):
                        continue
                    outside = [n for n in names if n not in [c.name for c in clock.states]]
                    if outside:
                        raise ss.errors.ClockOutsideDomain(f'the attribute "{name}" reads {clock} for people in {sc.strjoin(outside)}, but it has a value only for people in {sc.strjoin(c.name for c in clock.states)}.')
        self.dwell_trs = {name for name,tr in self.transitions.items() if len(tr.dst or []) > 1 and ss.expressions.clocks_in(sc.tolist(tr.p))} # Split as people leave, by their time in the state
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
                    raise ss.errors.UnknownState(f'an attribute of "{self.module.name}" names a state "{name}" that it does not have. Its states are {sc.strjoin(self.states.keys())}.')
            return out
        for tr in self.transitions.values(): # One written in a transition rather than passed to the module, e.g. rate=b*ss.attr(ss.uniform(0.5, 1.5), name='frailty')*I, is the module's
            for expr in [tr.rate, tr.dur] + list(tr.p or []):
                for leaf in (expr.leaves if isinstance(expr, Expr) else []):
                    if isinstance(leaf, ss.expressions.attr) and leaf.module is None and not any(a is leaf for a in self.attrs.values()):
                        if leaf.name is None:
                            raise ValueError(f'Could not find a name for {leaf} in "{tr.name}"; give it one, e.g. ss.attr(..., name="frailty"), or pass it as a keyword, e.g. ss.Disease(..., frailty=ss.attr(...))')
                        self.add(self.attrs, leaf.name, leaf)
        for name,a in self.attrs.items():
            if a.module is not None and a.module is not self.module: # HP43: it would be drawn for each module separately, not shared
                raise ss.errors.DuplicateDeclaration(f'the attribute "{name}" is given to two modules, "{a.module.name}" and "{self.module.name}"; an attribute belongs to one module, and each would draw its own values. '
                                                     f'For one value per person read by both, give it to one and read it from the other in a function of the sim.')
            a.module = self.module
            if isinstance(a.value, str):
                if a.value not in self.attrs:
                    errormsg = f'The attribute "{name}" of "{self.module.name}" records "{a.value}", which is not one of its attributes ({sc.strjoin(self.attrs.keys())})'
                    raise ValueError(errormsg)
                a.value = self.attrs[a.value]
            if isinstance(a.value, dict):
                if a.on is not None:
                    errormsg = f'The attribute "{name}" is a dict by state, which is derived (worked out when it is read), so it takes no on='
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
                    raise ss.errors.UnknownTransition(f'"{obj.name}" of "{self.module.name}" is set on {e.label}, which is not one of its transitions ({sc.strjoin(self.transitions.keys())}).')
                elif kind in ['enter', 'exit']:
                    names_of([s[1] if isinstance(s, tuple) else s for s in target])
        return

    def stratify(self, **kwargs):
        """ Stratify the graph by one or more dimensions; see `ss.Disease.stratify()` """
        by_attr = [key for key in kwargs if key in self.attrs] # By a derived attribute: its level is worked out from the attribute, so it adds no groups
        if by_attr:
            others = [key for key in kwargs if key not in by_attr and key != 'states']
            if others:
                errormsg = f'A stratification by the attribute "{by_attr[0]}" is worked out from it, so takes only states=; give a value by level in a rate with ss.by({by_attr[0]}=[...]), not {sc.strjoin(others)}'
                raise ValueError(errormsg)
            states = kwargs.get('states')
            if states is not None:
                states = [leaf.name for leaf in states.leaves if isinstance(leaf, State)] if isinstance(states, Expr) and not isinstance(states, State) else [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
            for name in by_attr:
                a, levels = self.attrs[name], kwargs[name]
                if a.kind != 'derived' or not all(sc.isnumber(v) for v in levels):
                    errormsg = f'A stratification by the attribute "{name}" takes bin edges, and the attribute must be derived (worked out when it is read); "{name}" is {a.kind}'
                    raise ValueError(errormsg)
                if any(s.name == name for s in self.strats + self.derived):
                    raise ss.errors.DuplicateStructuralDimension(f'"{self.module.name}" is already stratified by "{name}". stratify() works in place, so this is what running the same line (or notebook cell) twice gives.')
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
                raise ss.errors.DuplicateStructuralDimension(f'"{self.module.name}" is already stratified by "{strat.name}", in another way. stratify() works in place, so the stratification can be given once; to change it, build the model again.')
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
        """ Add a birth, death, or migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...)])`), after the graph is built; see `ss.Sim._init_flows()` """
        name = tr.name or ('births' if tr.per_capita else 'immigration' if tr.inflow else 'emigration' if tr.migration else 'deaths')
        if name in self.transitions:
            raise ss.errors.DuplicateDeclaration(f'"{self.module.name}" already has a transition named "{name}"; give the one in demographics= another name, e.g. ss.birth(..., name="{name}2")')
        tr.name = name
        if tr.inflow: # Born into (or arriving in) the state given, or else the state people are in unless they are put elsewhere
            if not tr.state_given:
                self.check_first(f'added by "{name}" in demographics= are put in')
            dst = tr.dst[0] if tr.state_given else self.first
            if not any(s is dst for s in self.people_states):
                raise ss.errors.UnknownState(f'"{name}" in demographics= adds people to "{dst.name}", which is not a state of people of "{self.module.name}"; its states are {sc.strjoin(s.name for s in self.people_states)}')
            tr.dst = [dst]
        if tr.rate is None and tr.dur is None:
            raise ss.errors.UndrivenTransition(f'"{name}" in demographics= has no rate; give it one, e.g. ss.birth(rate=ss.peryear(0.02))')
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
        todo = [self.first] if self.first is not None else []
        while todo:
            state = todo.pop()
            if state.name not in seen and state.people:
                seen.append(state.name)
                todo += [dst for tr in self.transitions.values() if not tr.acquisition and state in tr.src for dst in tr.dst]
        return seen

    @property
    def infected(self):
        """
        The states counted in prevalence: the states people are infected into, and the sources of infection (of people, or of a count, as `Ih` in a bite rate `Ih.n / N` of mosquitoes)

        In a model where nobody is a source of infection (a staged condition such as a cancer), also every state that
        people reach from those without passing through a state they can be in without having been infected (TL52: only
        the first stage was counted, so a test found 649 of 1,637 people with cancer)
        """
        sources = {id(s):s for s in self.infectious}
        for tr in self.transitions.values():
            if not tr.people and tr.rate is not None:
                sources.update({id(leaf.states[0]):leaf.states[0] for leaf in tr.rate.leaves if isinstance(leaf, Count) and len(leaf.states) == 1 and leaf.states[0].people})
        out = {id(s):s for tr in self.acquisitions for s in tr.dst}
        if not sources:
            stop = set(self.uninfected)
            todo = list(out.values())
            while todo:
                state = todo.pop()
                for tr in self.transitions.values():
                    if state in (tr.src or []) and tr.people and not tr.acquisition:
                        for dst in tr.dst or []:
                            if dst.people and dst.name not in stop and id(dst) not in out:
                                out[id(dst)] = dst
                                todo.append(dst)
        out.update(sources)
        return sorted(out.values(), key=lambda s: s.order)

    def unit_values(self):
        """ All the typed values (durations and rates) in the graph, used to infer the sim's unit """
        vals = [p.raw for p in self.all_pars()] + [val for oc in self.outcomes.values() for val in [oc.delay, oc.duration]]
        return unit_values(vals)


    def modifier_kind(self, name):
        """ What a modifier on a transition acts on: its probability if it is one branch of several with p= (chosen on entry) or one destination of a transition with several (`'recovery.R'`), and otherwise its rate """
        tr = self.transitions.get(name)
        if tr is not None and tr.p is not None and not tr.split and not tr.contact:
            return 'branch'
        if self.destination(name) is not None:
            return 'branch'
        return 'transition'

    def dwell_not_branch(self, name, how, modname):
        """ For a multiplier (`how`, e.g. 'the modifier "m"') on a transition with several destinations chosen with p= and a dwell time: a note that it changes the dwell, not a branch (TL16), naming the destinations that would; otherwise None """
        tr = self.transitions.get(name)
        if tr is None or tr.p is None or tr.dst is None or len(tr.dst) < 2 or tr.dur is None:
            return None
        dsts = sc.strjoin([f"'{modname}.{name}.{dst.name}'" for dst in tr.dst])
        return f'{how} on "{name}" changes how long people take to leave {sc.strjoin(s.name for s in tr.src)}, not which destination they go to; to change the probability of one destination, name it instead: {dsts}'

    def destination(self, name):
        """ For the name of one destination of a transition with several, `'<transition>.<state>'` (e.g. `'progression.C'`): the transition; otherwise None """
        trname, _, dst = name.rpartition('.')
        tr = self.transitions.get(trname)
        if tr is None or tr.dst is None or len(tr.dst) < 2 or dst not in [d.name for d in tr.dst]:
            return None
        return tr

    def destinations(self):
        """ The names of the destinations of the transitions with several, as `destination()` takes them """
        return [f'{name}.{end}' for name,tr in self.transitions.items() for end in tr.ends]

    def leaves(self):
        """ The leaves of the rates and branch probabilities of its transitions """
        exprs = [expr for tr in self.transitions.values() for expr in [tr.rate] + sc.tolist(tr.p)]
        return [leaf for expr in exprs if isinstance(expr, Expr) for leaf in expr.leaves]

    def all_pars(self):
        """ Every parameter in the graph, named or inline """
        out = {id(p):p for p in self.pars.values()}
        exprs = [expr for tr in self.transitions.values() for expr in [tr.rate, tr.dur] + sc.tolist(tr.p)]
        for expr in exprs:
            if expr is not None:
                out.update({id(leaf):leaf for leaf in expr.leaves if isinstance(leaf, par)})
        return list(out.values())

    def spread_pars(self):
        """ The parameters whose value is a distribution across people (e.g. `beta=ss.lognorm_ex(0.3, 0.1)`), other than dwell times, which are stages or are sampled on entry """
        dwells = [tr.dur for tr in self.transitions.values()]
        return [p for p in self.all_pars() if isinstance(p.raw, ss.Dist) and not any(p is dur for dur in dwells)]


def unit_values(vals):
    """ The units of the typed values (durations and rates) among some values, including those inside distributions and values by level or in time """
    out = []
    def check(val):
        if isinstance(val, ss.dur) and not isinstance(val, ss.datedur):
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
