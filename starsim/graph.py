"""
The graph of one module: its states, transitions, and parameters, collected, named, completed with defaults (each guess recorded), and checked
"""
import types
import itertools
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import as_expr, BinOp, Column, Count, did_you_mean, Dwell, Expr, is_one, linear, par, Row, Share, to_pars
from .parameters import by, TimeValue
from .transitions import bound_names, find_name, FlowSum, outcome, State, Transition, update
from .presentation import Presentation, TransitionTable

__all__ = ['Graph']


def inner_pars(val):
    """ The parameters inside a value: of a value that varies in time (`ss.seasonal(mean=beta)`) or a probability per act (`ss.peract(m2f=b)`), also within a beta for each route """
    if isinstance(val, (TimeValue, ss.peract)):
        return val.pars
    elif isinstance(val, dict):
        return [q for v in val.values() for q in inner_pars(v)]
    return []


class Graph(Presentation):
    """
    The states and transitions of one module

    The graph is built when the module is constructed. It names everything, infers
    states from arrow strings, works out the source of infection of each transmission
    and which states are people, turns inline values into named parameters, and checks
    the result. Every guess is recorded, in `graph.guessed` and in the `guessed` of each state and transition.

    This class is not usually used directly: see `ss.Disease`.

    Args:
        module (`ss.Module`): the module the graph belongs to
        args (list): transitions and outcomes passed positionally
        members (dict): states, transitions, parameters, results, and functions, by name
        init_prev (float): the initial prevalence, seeded through the transmission
        init (dict): the initial fraction or count in each state, by name
        first (str): the state people are in unless they are put elsewhere (by default, the one nothing moves people into; see `anchor()`)
        frame (frame): the calling frame, used to find the names of variables
    """
    grouped = ['transitions', 'results', 'outcomes'] # The keywords that each take several members, e.g. ss.Disease(transitions=[...])

    def __init__(self, module, args=None, members=None, init_prev=None, init=None, first=None, frame=None):
        self.module = module
        self.first_given = first # The state people are in unless they are put elsewhere, if given
        self.init_prev = init_prev
        if isinstance(init_prev, par): # So it can be fitted
            init_prev.name = init_prev.name or 'init_prev'
        self.init = dict(init or {})
        self.states = sc.objdict()
        self.transitions = TransitionTable()
        self.pars = sc.objdict()
        self.outcomes = sc.objdict()
        self.exprs = sc.objdict() # Results that are sums or ratios of states
        self.updates = sc.objdict() # Counts set to a value each step (ss.update)
        self.funcs = sc.objdict() # Custom per-step behavior
        self.guessed = sc.objdict()
        self.inferred = False # Whether the states were inferred from arrow strings
        self.strats = [] # Stratifications, filled by stratify()
        self.transfers = sc.objdict() # Movement between the levels of a stratification

        self.collect(list(args or []), dict(members or {}), frame)
        if isinstance(init_prev, par):
            self._add_par(init_prev, name=init_prev.name)
        self.resolve_states(frame)
        self.resolve_transitions()
        self.expand_rows()
        self.resolve_people()
        self.resolve_acquisition()
        self.resolve_pars(frame)
        self.set_keywords()
        self.resolve_exits()
        self.check()
        return

    def guess_init(self):
        """ When the sim is built: with no initial values at all, 1% start infected, since nobody infected would mean no epidemic """
        if self.init_prev is None and not self.init and self.transmissions and all(s.init is None for s in self.states.values()) and getattr(self, 'seeded_by', None) is None: # A member of a family that seeds it (ss.Strains) is not seeded on its own
            self.init_prev = 0.01
            self.guessed['init'] = 'nothing given'
        return

    def __repr__(self):
        return f'Graph(states={list(self.states.keys())}, transitions={list(self.transitions.keys())})'

    def collect(self, args, members, frame):
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
            if isinstance(obj, (Transition, outcome, par, State, update)):
                if name is not None:
                    obj.name = name
                elif obj.name is None:
                    obj.name = find_name(obj, frame)
                    names = bound_names(obj, frame) if obj.name is None and isinstance(obj, Transition) else []
                    if len(names) > 1: # Which one is meant is not known, so the name is built from its endpoints
                        obj.guessed['name'] = f'built from its endpoints, since it is bound to {sc.strjoin(names, sep=" and ")}'
            if isinstance(obj, ss.dimensions.Transfer):
                obj.name = name or obj.name or find_name(obj, frame)
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
            elif isinstance(obj, update):
                if obj.name is None:
                    errormsg = 'Could not find a name for an update; pass it as a keyword, e.g. ss.Disease(..., track=ss.update(...))'
                    raise ValueError(errormsg)
                self.add(self.updates, obj.name, obj)
            elif isinstance(obj, par):
                self._add_par(obj, name=obj.name)
            elif isinstance(obj, Expr): # A sum or ratio of states is a result
                name = name or find_name(obj, frame)
                if name is None:
                    errormsg = f'Could not find a name for the result "{obj}"; pass it as a keyword, e.g. ss.Disease(..., prevalence=I/N)'
                    raise ValueError(errormsg)
                self.add(self.exprs, name, obj)
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
            raise ss.ReservedName(f'"{name}" cannot be used as the name of a state, transition, parameter, or result, since it is an attribute of every module. Please choose another name.')
        for other in [self.states, self.transitions, self.pars, self.outcomes, self.exprs, self.updates, self.funcs, self.transfers]:
            if name in other and other[name] is not obj:
                raise ss.DuplicateDeclaration(f'two members of "{self.module.name}" are both named "{name}"; names must be unique within a module.')
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
            raise ss.DuplicateDeclaration(errormsg)
        return

    def resolve_states(self, frame):
        """ Find the states: the ones declared, or else every name in an arrow string """
        # Find all the state objects used in the transitions
        trs = list(self.transitions.values()) + self._unnamed
        found = {id(s):s for s in self._declared}
        strings = sc.autolist()
        self._refs = [] # States used only inside a rate
        for tr in trs:
            for end in (tr.src or []) + (tr.dst or []):
                if isinstance(end, State): found[id(end)] = end
                else:                      strings += end
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p):
                if expr is not None:
                    for leaf in expr.leaves:
                        for s in ([leaf] if isinstance(leaf, State) else leaf.states if isinstance(leaf, Count) else []):
                            if id(s) not in found:
                                self._refs.append(s)
            if tr._pending is not None: # The source of a transmission, if given
                strings += [s for s in (list(tr._pending[1]) if isinstance(tr._pending[1], dict) else sc.tolist(tr._pending[1])) if isinstance(s, str)]
            if getattr(tr, 'offspring', None) is not None: # The state newborns go into, which may be in no arrow
                kid = tr.offspring[0]
                if isinstance(kid, State): found[id(kid)] = kid
                else:                      strings += kid
        for upd in self.updates.values(): # The count an update sets, which may be in no arrow, and the states its value uses
            if isinstance(upd.state, State): found[id(upd.state)] = upd.state
            else:                            strings += upd.state
            self._refs += [s for leaf in upd.value.leaves for s in ([leaf] if isinstance(leaf, State) else leaf.states if isinstance(leaf, Count) else []) if id(s) not in found]
        for s in self._refs: # A state used only in a rate still belongs to this module unless another module has claimed it
            if s.module is None or s.module is self.module:
                found[id(s)] = s
        if found and frame is not None: # A string can also name a declared state that is not used as an object, e.g. ss.Transition('I', 'R') after S, I, R = ss.states(3)
            for string in strings:
                var = frame.f_locals.get(string, frame.f_globals.get(string))
                if isinstance(var, State) and (var.name is None or var.name == string) and string not in [s.name for s in found.values()]:
                    if var.module not in (None, self.module): # A state of another disease: this one gets its own
                        self.check_read(var)
                        var = sc.cp(var)
                        var.module = None
                        var.guessed = sc.objdict()
                    found[id(var)] = var
        declared = sorted(found.values(), key=lambda s: s.order)

        # Name the declared states
        for i,state in enumerate(declared):
            if state.name is None:
                state.name = find_name(state, frame)
            if state.name is None:
                used = [tr.name or tr.arrow for tr in trs if any(end is state for end in (tr.src or []) + (tr.dst or []))]
                raise ss.UnnamedState(f'state number {i+1} of the module has no name. It is used in {used}, and no variable or keyword holding it was found.\n'
                                      f"     Give the names up front: S, I, R = ss.states('S I R').")

        # Either check the strings against the declared states, or infer the states from them
        names = [s.name for s in declared]
        for string in strings:
            if not isinstance(string, str): # e.g. ss.Transition(A + B, C): a sum of states is a count, for a rate, not where people move from
                states = sc.strjoin(leaf.name for leaf in getattr(string, 'leaves', []) if isinstance(leaf, State)) or str(string)
                raise TypeError(f'the source or destination of a transition is a state or a list of states, not {string}; for people moving out of either, write [{states}]')
        if declared:
            for string in strings:
                if string not in names:
                    raise ss.UnknownState(f'no state named "{string}" was declared in "{self.module.name}". Declared states are {sc.strjoin(names)}{did_you_mean(string, names)}')
            statelist = declared
        else:
            self.inferred = True
            self.guessed['states'] = 'inferred from the arrows'
            for string in strings:
                if not string.isidentifier():
                    raise ss.UnknownState(f'"{string}" in an arrow string of "{self.module.name}" is not a state name. Branches are written with "|", e.g. "I -> R | D".')
                var = None if frame is None else frame.f_locals.get(string, frame.f_globals.get(string))
                if isinstance(var, State) and var.module not in (None, self.module): # e.g. 'I -> Rb' after an earlier disease read Rb.frac
                    self.check_read(var)
            statelist = [State(name) for name in dict.fromkeys(strings)] # Unique names, in order of first appearance
        for state in statelist:
            if state.module is not None and state.module is not self.module:
                raise ss.DuplicateDeclaration(f'state "{state.name}" is used by transitions in two modules, "{state.module.name}" and "{self.module.name}"; a state belongs to one module.')
            state.module = self.module
            self.add(self.states, state.name, state)

        # Replace strings with the states
        def lookup(end):
            return self.states[end] if isinstance(end, str) else end
        for tr in trs:
            if tr.src is not None: tr.src = [lookup(end) for end in tr.src]
            if tr.dst is not None: tr.dst = [lookup(end) for end in tr.dst]
            if tr._pending is not None and tr._pending[1] is not None:
                beta, source = tr._pending
                tr._pending = (beta, {lookup(end):w for end,w in source.items()} if isinstance(source, dict) else [lookup(end) for end in sc.tolist(source)]) # A dict gives a weight for each state
        for upd in self.updates.values():
            upd.state = lookup(upd.state)

        # Initial conditions given on the module
        for name,val in self.init.items():
            if name not in self.states:
                raise ss.UnknownState(f'init= names a state "{name}" that "{self.module.name}" does not have. States are {sc.strjoin(self.states.keys())}.')
            self.states[name].init = val
        return

    def resolve_transitions(self):
        """ Give every transition a name, a source, and a rate or a dwell time """
        # Fallback names
        for tr in self._unnamed:
            src = '_'.join([s.name for s in tr.src or []]) or 'in'
            dst = '_'.join([s.name for s in tr.dst or []]) or 'out'
            base = f'{src}_to_{dst}'
            name = base
            count = 1
            while name in self.transitions:
                count += 1
                name = f'{base}_{count}'
            tr.name = name
            tr.guessed.setdefault('name', 'built from its endpoints')
            self.add(self.transitions, name, tr)

        for name,tr in self.transitions.items():
            if tr.dst is None:
                errormsg = f'The transition "{name}" has no destination'
                raise ValueError(errormsg)

            # ss.transmission(): the rate is beta times the source of infection
            if tr._pending is not None:
                beta, sources = tr._pending
                if sources is None:
                    infectious = [s for s in self.states.values() if s.infectious]
                    if infectious:
                        sources = infectious
                        tr.guessed['source'] = 'the states declared infectious'
                    else:
                        sources = tr.dst[:1]
                        tr.guessed['source'] = 'source is the destination'
                        latent = self.latent_states(tr)
                        if latent:
                            raise ss.SourceNeeded(f'"{name}" ({tr.arrow}) leads to {tr.dst[0].name}, from which people move on to {sc.strjoin(s.name for s in latent)} and then on again, so whether they infect others in {tr.dst[0].name} (as in an SIR with hospital care) or in {latent[0].name} (as in an SEIR) cannot be guessed.\n'
                                                  f'     Give the source, e.g. ss.transmission("{tr.arrow}", beta=..., source="{latent[0].name}").')
                weights = sources if isinstance(sources, dict) else {s:1 for s in sources} # source=dict(P=1, EL=0.55) is beta * (P + 0.55*EL)
                terms = [s if (sc.isnumber(w) and w == 1) else as_expr(w)*s for s,w in weights.items()]
                source = terms[0]
                for term in terms[1:]:
                    source = source + term
                tr.rate = as_expr(beta) * source
                tr._pending = None

            if tr.undriven: # Built before (the same transition in another module): its rate of 0 is set again in resolve_exits()
                tr.rate = None
            if tr.rate is None and tr.dur is None and tr.p is None and tr.trigger is None and not tr.every_state:
                if not tr.src:
                    raise ss.UndrivenTransition(f'the inflow "{name}" ({tr.arrow}) has no rate. Give it one, e.g. {tr.arrow.replace(" -> ", " >> ")} @ 0.1')
                tr.undriven = True

            # Derive the kind from the rate
            tr.const, tr.terms = (None, {}) if tr.rate is None else linear(tr.rate)
            tr.contact = bool(tr.terms)
            for state in tr.terms:
                if not state.infectious: # Not declared, so a guess
                    state.guessed['infectious'] = name
                state.infectious = True
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
                    raise ss.UnknownState(f'p= of "{name}" has a value over "to" for "{lab}", which is not one of its destinations ({sc.strjoin(names)}){did_you_mean(lab, names)}')
            plain = [i for i in range(len(tr.p)) if i not in rows]
            free = [n for n in names if n not in named]
            unlabeled = [i for i,labs in rows.items() if not labs]
            if len(unlabeled) > 1 or (not unlabeled and len(free) != len(plain)) or len(free) < len(plain):
                raise ss.BadBranchSplit(f'"{name}" has {len(names)} destinations, but its p= has {len(plain)} entries of one destination each and values over "to" for {sc.strjoin(named) or "the others"}; give one entry for each destination that no value over "to" names')
            for i in unlabeled:
                rows[i] = free[len(plain):]
            covered = {lab:i for i,labs in rows.items() for lab in dict.fromkeys(labs)}
            built = {i:Row(tr.p[i], list(dict.fromkeys(labs))) for i,labs in rows.items()}
            entries = iter(plain)
            tr.p_given = list(tr.p)
            tr.p = [Share(built[covered[n]], n) if n in covered else tr.p[next(entries)] for n in names]
        return

    def anchor(self, states, connected):
        """
        The state that people are in unless they are put elsewhere (`first=`), which the other people states are connected to

        Unless it is given, it is found among the group of states connected to the ones seeded (given initial values, or
        entered by the contact-mediated transition that init_prev seeds through; else the largest group of connected
        states), leaving out a group that only an inflow fills (`ss.null >> Sv @ emergence`, not a birth), which is a count
        when there is another group (so `init=dict(Sv=3000)` on mosquitoes does not make them the people). With contact-mediated transitions, it is a source of one that people do not reach from another (S, not V
        after S -> V); of several, the one that no transition moves people into, then the earliest declared, with a note in
        `guessed`. With none, it is the earliest state that no transition moves people into, or else the earliest state.
        The order in which the transitions are given does not matter.
        """
        if self.first_given is not None:
            name = self.first_given.name if isinstance(self.first_given, State) else self.first_given
            if name not in self.states:
                raise ss.UnknownState(f'first="{name}" is not a state of "{self.module.name}". Its states are {sc.strjoin(self.states.keys())}{did_you_mean(name, list(self.states.keys()))}')
            return self.states[name]
        given = [s for s in states if s.init is not None]
        seeded = given + ([st for tr in self.transitions.values() if tr.contact for st in tr.dst or []] if self.init_prev is not None else [])
        inflowed = {id(st) for tr in self.transitions.values() if tr.inflow and not tr.per_capita and not tr.migration for st in tr.dst or []} # Filled from nothing in total per unit time (mosquitoes emerging, bacteria shed): a count, unless joined to people
        groups = [connected([s]) for s in states if s.people is not False]
        allowed = set().union(*([g for g in groups if not inflowed & g] or groups)) # Never a group filled by an inflow alone, when there is another
        group = connected(seeded) & allowed
        if not group: # Only counts are seeded (e.g. init=dict(Sv=3000)), or nothing is: the largest group
            group = max(groups, key=lambda g: (bool(g & allowed), len(g)), default=set())
        members = [s for s in states if id(s) in group] or states
        entered = {id(st) for tr in self.transitions.values() for st in (tr.dst or []) if isinstance(st, State)}
        entered |= {id(self.states[name]) for name in self.__dict__.get('delivered', []) if name in self.states} # Moved into by a delivery (move='Sev -> H'): see delivered_to()
        entered |= {id(self.states[kid] if isinstance(kid, str) else kid) for kid in [tr.offspring[0] for tr in self.transitions.values() if getattr(tr, 'offspring', None) is not None]} # Newborns enter theirs
        sources = [s for s in members if any(tr.contact and any(src is s for src in tr.src or []) for tr in self.transitions.values())] # In declared order
        if sources:
            fed = {id(st) for tr in self.transitions.values() if any(src is s for src in tr.src or [] for s in sources) for st in tr.dst or []} # Reached from a source, e.g. by vaccination
            candidates = [s for s in sources if id(s) not in fed] or sources
            first = next((s for s in candidates if id(s) not in entered), candidates[0])
            if len(candidates) > 1:
                self.guessed['first'] = f'{first.name}, of the sources of contact-mediated transitions {sc.strjoin(s.name for s in candidates)}'
            return first
        return next((s for s in members if id(s) not in entered), members[0])

    def delivered_to(self, names):
        """ When the sim is built: the states that its deliveries move people into (`ss.Delivery(move='Sev -> H')`), which are not the first state unless given as it """
        if self.first_given is not None or self.first.name not in names:
            return
        self.delivered = list(names)
        self.resolve_people()
        self.resolve_acquisition()
        return

    def resolve_people(self):
        """ Work out which states are people: the ones connected by transitions to the first state """
        states = list(self.states.values())
        links = {id(s):set() for s in states}
        for tr in self.transitions.values():
            kid = [self.states[tr.offspring[0]] if isinstance(tr.offspring[0], str) else tr.offspring[0]] if getattr(tr, 'offspring', None) is not None else [] # Newborns are people, as their parents are
            for a in tr.src or []:
                for b in (tr.dst or []) + kid:
                    links[id(a)].add(id(b))
                    links[id(b)].add(id(a))
            if not tr.src: # An inflow split between several states: its destinations are connected too
                for a in tr.dst or []:
                    links[id(a)].update(id(b) for b in tr.dst if b is not a)

        # Start from the anchor, and from everything declared to be people
        lookup = {id(s):s for s in states}
        def connected(start):
            todo = list(start)
            seen = set()
            while todo:
                state = todo.pop()
                if id(state) not in seen and state.people is not False:
                    seen.add(id(state))
                    todo += [lookup[i] for i in links[id(state)]]
            return seen
        first = self.anchor(states, connected)
        joined = connected([first])
        seen = connected([first] + [s for s in states if s.people is True])
        self.separate = [s for s in states if s.people is True and id(s) not in joined] # People with no transition to or from the first state's, which are said when the sim is built
        for state in states:
            if state.people is None:
                state.people = id(state) in seen
                if not state.people:
                    state.guessed['people'] = f'no transition connects {state.name} to {first.name}'
        self.people_states = [s for s in states if s.people]
        self.count_states = [s for s in states if not s.people]
        self.first = first
        self.resolve_rest()

        # Now that the people are known, fill in the things that depend on them
        for name,tr in self.transitions.items():
            self.fill_people(tr)
            for state in tr.terms:
                if not state.people:
                    raise ss.ExposureToCount(f'the rate for "{name}" uses the bare state {state.name}, and {state.name} is not people.\n'
                                             f'       rate = {tr.rate}\n'
                                             f'     A bare state means contact with people in that state, and there is no contact with a count.\n'
                                             f'     Write {state.name}.n for the number itself. If {state.name} is people, declare it with ss.State(people=True).')
            ends = (tr.src or []) + (tr.dst or [])
            tr.people = all(s.people for s in ends)
            if not tr.people and any(s.people for s in ends):
                counts = [s.name for s in ends if not s.people]
                raise ss.PeopleAndCounts(f'"{name}" moves between people ({sc.strjoin(s.name for s in ends if s.people)}) and counts ({sc.strjoin(counts)}). '
                                         f'If {sc.strjoin(counts)} {"is" if len(counts) == 1 else "are"} people, declare it with ss.State(people=True).')
        return

    def resolve_rest(self):
        """
        Work out where the people not given an initial state start: the first state, unless it is given an initial value of
        its own, which is then kept (`init=dict(S=0.1, I=1e-4)` starts 10% susceptible). Then it is the one state of people
        with no initial value, with a note in `guessed`; with none or several, there is no such state, and the initial values
        must sum to 1 (see `Engine.init_fractions()`)
        """
        self.rest = self.first
        if self.first.init is None:
            return
        seeded = {id(st) for tr in self.transitions.values() if tr.contact for st in tr.dst or []} if self.init_prev is not None else set() # Where init_prev puts people
        unset = [s for s in self.people_states if s.init is None and id(s) not in seeded]
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
        """ Fill in what depends on which states are people: the sources of the deaths of everyone (`ss.death()` with no arrow), and the people a birth rate is per person of (`ss.birth()`) """
        if tr.every_state:
            tr.src = list(self.people_states)
        if tr.per_capita:
            tr.rate = tr.rate * Count(list(self.people_states), kind='n')
            tr.const, tr.terms = linear(tr.rate)
        return

    def latent_states(self, tr):
        """ For a transmission whose source would be guessed as its destination: the states that people move on to from there and then leave for somewhere other than the susceptible state, as in an SEIR (in an SIR, SIS, or SIRS, there are none) """
        def onward(state):
            return [dst for t in self.transitions.values() if t is not tr and any(src is state for src in t.src or []) for dst in t.dst or []]
        susceptible = tr.src or []
        out = []
        for state in onward(tr.dst[0]):
            if any(dst is not state and not any(dst is s for s in susceptible) for dst in onward(state)):
                out.append(state)
        return out

    def set_keywords(self):
        """ A value given as a keyword sets the parameter of that name; there is nothing else for it to be """
        for name,value in self._keywords.items():
            if name not in self.pars and name.startswith(('init', 'seed')): # e.g. init_infected=100
                state = self.infectious[0].name if self.infectious else 'I'
                raise ss.UnknownParameter(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}".\n'
                                          f'     To start with people in a state, use init=dict({state}={value!r}); init_prev= seeds the state a new infection enters (by default 1% of people), as a share or a number of people.')
            elif name not in self.pars:
                raise ss.UnknownParameter(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}"; its parameters are {sc.strjoin(self.pars.keys())}.\n'
                                          f'     To add a parameter, use it in a transition, e.g. infection = S >> I @ {name} * I with {name} = ss.par({value}).\n'
                                          f'     If another module reads it from this disease, add it after making the disease: {self.module.name}.define_pars({name}={value!r}).' +
                                          (f'\n     Condoms (or other protection used in some acts) are an intervention on the transmission instead, e.g. ss.modify({self.module.name}.infection, efficacy={value!r}, share=0.3, route="mfnet").' if 'condom' in name else ''))
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
                alone = isinstance(rate, BinOp) and not any(isinstance(leaf, par) and not leaf.literal for leaf in rate.b.leaves) # Only the rate as a whole has a unit, so a factor of a product of parameters is a number: R0 in R0 * (gamma * I)
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

    def resolve_pars(self, frame):
        """ Turn inline values in the standard slots into named parameters (beta, dur, rate, p) """
        slots = [(tr, slot, p) for tr in self.transitions.values() for slot,p in self.set_slots(tr)] # (transition, slot name, parameter)

        # Name the anonymous ones after their slot, qualified by the transition where two would clash
        anon = [(tr, slot, p) for tr,slot,p in slots if p.literal or getattr(p, 'slot_named', False) or (p.name is None and find_name(p, frame) is None)] # Including an ss.par written in the slot itself, e.g. beta=ss.par(0.3, bounds=[0.1, 1]), and one named after its slot before (in a class this one subclasses), which is named again
        counts = sc.objdict()
        for tr,slot,p in anon:
            counts[slot] = counts.get(slot, 0) + 1
        inner = {q.name or find_name(q, frame) for tr,slot,p in slots for q in inner_pars(p.value)} # The names of parameters inside values that vary in time, e.g. ss.seasonal(mean=beta), or inside a probability per act, which are registered below
        for tr,slot,p in anon:
            clash = counts[slot] > 1 or slot in self.pars or slot in self.states or slot in self.transitions or slot in inner
            self._add_par(p, name=f'{slot}_{tr.name}' if clash else slot)
            p.slot_named = True

        # Register the explicit ss.par objects used in the transitions, and in the weights and likelihoods of the outcomes
        used = [(tr.name, expr) for tr in self.transitions.values() for expr in [tr.rate, tr.dur] + sc.tolist(tr.p) + sc.tolist(getattr(tr, 'p_given', None))]
        used += [(oc.name, weight) for oc in self.outcomes.values() for weight in oc.weights]
        used += [(oc.name, val) for oc in self.outcomes.values() if oc.likelihood is not None for val in oc.likelihood.pars.values() if isinstance(val, par)]
        for oc in self.outcomes.values(): # A parameter written inside a likelihood, e.g. ss.nbinom(n=ss.par(10, bounds=[1, 100])), is named after the outcome and its keyword
            for key,val in (oc.likelihood.pars.items() if oc.likelihood is not None else []):
                if isinstance(val, par) and not val.literal and val.name is None and find_name(val, frame) is None:
                    val.name = f'{oc.name}_{key}'
        used += [(f'init of {state.name}', state.init) for state in self.states.values() if isinstance(state.init, Expr)] # So an initial count can be fitted
        used += [(name, expr) for name,tf in self.transfers.items() for expr in [tf.rate, tf.dur]] # And a parameter in the rate of a transfer
        used += [(name, upd.value) for name,upd in self.updates.items()] # And in the value of an update
        used += [(name, inner) for name,expr in list(used) if expr is not None for leaf in expr.leaves if isinstance(leaf, par) for inner in inner_pars(leaf.value)] # And a parameter inside a value that varies in time, a probability per act, or a beta for each route
        for name,expr in used:
            if expr is not None:
                for leaf in expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        if leaf.name is None:
                            leaf.name = find_name(leaf, frame)
                        if leaf.name is None:
                            errormsg = f'Could not find a name for the parameter {leaf.value} used in "{name}"; give it one, e.g. ss.par({leaf.value}, name="{name}_{leaf.slot or "p"}")'
                            raise ValueError(errormsg)
                        self._add_par(leaf, name=leaf.name)
        return

    def resolve_exits(self):
        """ Group the transitions out of each state: those with p= split on entry, the rest compete """
        self.exits = sc.objdict()
        for state in self.states.values():
            self.exits[state.name] = sc.objdict(branches=[], timed=[], hazards=[], contact=[], rest=[], triggered=[], stay=0)
        self.inflows = []
        for tr in self.transitions.values():
            if tr.undriven: # To the engines, a rate of 0 that nothing modifies; a delivery moves people along it (see ss.Treatment)
                tr.rate = par(0.0, literal=True)
                tr.const, tr.terms = linear(tr.rate)
            if tr.inflow:
                self.inflows.append(tr)
            for state in tr.src:
                exits = self.exits[state.name]
                if tr.contact:
                    exits.contact.append(tr)
                elif tr.trigger is not None: # Moved as soon as its condition holds, or its event happens: see ss.on
                    exits.triggered.append(tr)
                elif tr.p is not None and not tr.split: # One branch of several out of this state
                    exits.branches.append(tr)
                elif tr.dur is not None:
                    exits.timed.append(tr)
                elif tr.rate is None and len(tr.dst) > 1: # Several destinations and no wait: say how to write an immediate split
                    names = [d.name for d in tr.dst]
                    example = ', '.join(f'{tr.name}_{n}=ss.Transition({state.name}, {n}, p={p})' for n,p in zip(names, [str(v) if v is not None else 'None' for v in sc.tolist(tr.p)]))
                    raise ss.UndrivenTransition(f'the transition "{tr.name}" ({state.name} -> {sc.strjoin(names)}) has p= but no rate and no duration (dur=), so nothing says when people leave. '
                                                f'For a split as soon as people arrive, write one transition for each destination: {example}; or give it dur=.')
                elif tr.rate is None: # Immediate: one destination and ss.stay, or what the branches leave over
                    exits.rest.append(tr)
                else:
                    exits.hazards.append(tr)

        # A single timed transition with no p= (or an immediate split) takes whatever the branches leave over: the only one, or the only one for the same people (the same where=) as the branches; one with no driver and no p= is undriven, not the remainder
        for name,exits in self.exits.items():
            exits.remainder = None
            rest = exits.pop('rest')
            if len(rest) > 1:
                raise ss.BadBranchSplit(f'{sc.strjoin(tr.name for tr in rest)} out of state "{name}" have no p=, rate, or duration; only one transition can take what the branches leave over.')
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
                    elif below and ss.transitions.rest_stay and not exits.timed and exits.remainder is None and len(bad) == 1 and len(bad[0][0]) == len(exits.branches): # The rest stay (pending a decision: see ss.transitions.rest_stay)
                        exits.stay = 1 - bad[0][1]
                        self.guessed[f'states.{name}.stay'] = f'the other {exits.stay:.0%} stay'
                    else:
                        group, total = bad[0]
                        undriven = [tr.name for tr in exits.hazards if tr.undriven]
                        driven = [tr for tr in exits.hazards if not tr.undriven]
                        why = f' {sc.strjoin(undriven)} {"has" if len(undriven) == 1 else "have"} no rate, duration, or p=, so {"is" if len(undriven) == 1 else "are"} undriven, not the remainder; for an immediate split, give the destinations together, e.g. "{name} -> A | B" with p=[{total:n}, None].' if undriven else ''
                        why += f' {sc.strjoin(tr.name for tr in driven)} {"has" if len(driven) == 1 else "have"} a rate and no p=, so {"it competes" if len(driven) == 1 else "they compete"} with the branches rather than taking the rest: give it p={1 - total:n}, or a duration (dur=), which takes the rest.' if total < 1 and driven and not exits.timed else ''
                        why += f' For the rest to stay in {name}, give the destinations as one split with ss.stay, e.g. [A, ss.stay] with p=[{total:n}, None].' if total < 1 else ''
                        raise ss.BadBranchSplit(f'the branch probabilities out of state "{name}" sum to {total:n}, not 1 '
                                                f'({sc.strjoin([f"{tr.name}: p={tr.p[0]}" for tr in group])}).{why}')
            exits.timed += apart
        for name,exits in self.exits.items():
            outs = exits.branches + exits.timed + exits.hazards + exits.triggered + sc.tolist(exits.remainder)
            exits.held = bool(exits.stay) or any(tr.stays for tr in outs) # Whether some people stay with no exit of their own: see ss.solvers.Compartmental.build_layout()
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
            if tr.trigger is not None and (tr.contact or tr.inflow or (tr.p is not None and len(tr.dst) <= 1 and not tr.trigger.outcome)): # On an outcome, p= is the share of its people who move
                raise ss.BadBranchSplit(f'"{name}" is triggered by {tr.trigger}, so it cannot be {"a transmission" if tr.contact else "an inflow" if tr.inflow else "one branch of several with p="}; '
                                        f'give the destinations of a triggered split together, e.g. "{tr.arrow} | B" with p=[...].')
            for clock in ss.expressions.clocks_in([tr.rate, tr.dur] + sc.tolist(tr.p)): # A clock has a value only inside its states, and progress needs a scheduled dwell
                if clock.module not in [None, self.module] or not all(n in self.states for n in clock.names):
                    continue # Another module's clock: checked when it runs
                outside = [src.name for src in tr.src or [] if src.name not in clock.names]
                if outside:
                    raise ss.ClockOutsideDomain(f'"{name}" reads {clock}, which has a value only for people in {sc.strjoin(clock.names)}, but moves people from {sc.strjoin(outside)}.')
                if clock.kind == 'progress':
                    ex = self.exits[clock.names[0]]
                    if not (ex.branches or ex.remainder or ex.timed):
                        raise ss.NoScheduledDwell(f'"{name}" reads {clock}, but {clock.names[0]} has only rates out of it, so no dwell time is scheduled when people enter it.')
            if tr.stays and not tr.dst:
                raise ss.BadBranchSplit(f'"{name}" has ss.stay as its only destination; give it with another, e.g. [B, ss.stay] with p=[0.3, None].')
            if tr.p is not None and tr.split:
                ps = tr.p
                if len(ps) == 1 and len(tr.dst) == 2 and not tr.stays: # With two destinations, a single value is the first one's probability
                    ps = tr.p = [ps[0], None]
                elif len(ps) == 1 and isinstance(ps[0], par) and callable(ps[0].raw) and not isinstance(ps[0].raw, (ss.Dist, TimeValue)): # A function giving a row, one column per destination
                    tr.p_given = list(ps)
                    ps = tr.p = [Column(ps[0], i, len(tr.dst)) for i in range(len(tr.dst))]
                if len(ps) != len(tr.dst):
                    raise ss.BadBranchSplit(f'"{name}" has {len(tr.dst) + tr.stays} destinations but {len(ps) + (tr.p_stay is not None)} branch probabilities.')
                given = [p for p in ps + [tr.p_stay] if p is not None]
                if tr.stays and len(given) < len(ps):
                    raise ss.BadBranchSplit(f'"{name}" leaves the rest to both ss.stay and another destination; give one of them a p=.')
                if all(isinstance(p, par) and sc.isnumber(p.raw) for p in given):
                    total = sum(p.raw for p in given)
                    if (len(given) == len(ps) + tr.stays and not np.isclose(total, 1)) or total > 1 + 1e-9:
                        raise ss.BadBranchSplit(f'the branch probabilities of "{name}" sum to {total:n}, not 1.')
            elif tr.p is None and tr.split:
                raise ss.BadBranchSplit(f'"{name}" has {len(tr.dst)} destinations but no branch probabilities; give them with p=[...].')
            if tr.dst is not None and len(tr.dst) > 1: # Each destination's flow is res.<transition>.<state>, so it cannot be named like an attribute of a result
                clash = [d.name for d in tr.dst if d.name in ss.Result.reserved()]
                if clash:
                    raise ss.DuplicateDeclaration(f'the flow of "{name}" to {sc.strjoin(clash)} is a result read as res.{name}.{clash[0]}, which is already an attribute of every result; rename the state {sc.strjoin(clash)}')
        updated = set()
        for name,upd in self.updates.items(): # A count with one value, set by one update and nothing else
            state = upd.state
            moved = [tr.name for tr in self.transitions.values() if any(s is state for s in (tr.src or []) + (tr.dst or []))]
            if state.people:
                raise ss.CapabilityError(f'the update "{name}" sets "{state.name}", which is people; an update sets a count (ss.State(people=False)), and people move only along transitions')
            elif moved or state.name in updated:
                raise ss.DuplicateDeclaration(f'"{state.name}" is set by the update "{name}" and also by {sc.strjoin(moved) if moved else "another update"}; a count that is updated has no transitions and one update')
            bare = [leaf.name for leaf in upd.value.leaves if isinstance(leaf, State)]
            if bare:
                raise ss.ExposureToCount(f'the value of the update "{name}" uses the bare state {bare[0]}, which means contact; write {bare[0]}.n for the number, or {bare[0]}.frac for the share')
            updated.add(state.name)
        self.check_dwell()
        return

    def check_dwell(self):
        """ The time in a state (`P.dwell`) is known only for people in it: allowed in `p=` of a transition with several destinations out of it (decided when they leave), and in the rate of a transition out of it """
        self.dwell_states = set() # The states whose time in state is read
        self.dwell_trs = set() # The transitions that read it
        where = 'P.dwell can be used in p= of a transition with several destinations out of P (decided when people leave), e.g. ss.Transition(P, [C, R], dur=..., p=[f(P.dwell), None]), or in the rate of a transition out of P'
        dwells = lambda expr: [leaf.state for leaf in (expr.leaves if isinstance(expr, Expr) else []) if isinstance(leaf, Dwell)]
        for name,tr in self.transitions.items():
            own = tr.src[0] if len(tr.src or []) == 1 and not tr.contact else None
            for state in dwells(tr.rate) + [s for p in (sc.tolist(tr.p) if len(tr.dst) > 1 else []) for s in dwells(p)]:
                if state is not own:
                    raise ss.CapabilityError(f'"{name}" ({tr.arrow}) reads {state.name}.dwell, the time in {state.name}, which only people leaving {state.name} have. {where}')
                self.dwell_states.add(state.name)
                self.dwell_trs.add(name)
            if len(tr.dst) <= 1 and any(dwells(p) for p in sc.tolist(tr.p)):
                state = dwells(tr.p[0])[0]
                raise ss.CapabilityError(f'"{name}" has p= with {state.name}.dwell, but a branch with p= is decided when people enter {state.name}, when no time has passed. '
                                         f'Write one transition with several destinations, decided when they leave, e.g. ss.Transition({state.name}, [{tr.dst[0].name}, ...], dur=..., p=[f({state.name}.dwell), None])')
            if dwells(tr.dur):
                raise ss.CapabilityError(f'the dwell time of "{name}" reads {dwells(tr.dur)[0].name}.dwell; {where}')
        others = [(name, upd.value) for name,upd in self.updates.items()] + [(name, expr) for name,tf in self.transfers.items() for expr in [tf.rate, tf.dur]] + list(self.exprs.items())
        others += [(oc.name, w) for oc in self.outcomes.values() for w in oc.weights] + [(f'init of {s.name}', s.init) for s in self.states.values()]
        for name,expr in others:
            if dwells(expr):
                raise ss.CapabilityError(f'"{name}" reads {dwells(expr)[0].name}.dwell, the time in a state, which is not a number of people; {where}')
        return

    def stratify(self, frame=None, **kwargs):
        """ Stratify the graph by one or more dimensions; see `ss.Disease.stratify()` """
        strats, transfers = ss.dimensions.parse_stratify(kwargs, graphs=[self])
        old = {s.name:s for s in self.strats}
        if strats and all(strat.name in old and strat.same_as(old[strat.name]) for strat in strats) and all(name in self.transfers for name in transfers):
            ss.note(f'"{self.module.name}" is already stratified by {sc.strjoin(strat.name for strat in strats)} in the same way, so this stratify() does nothing (stratify() works in place)')
            return self.module
        for strat in strats:
            if strat.name in old:
                raise ss.DuplicateStructuralDimension(f'"{self.module.name}" is already stratified by "{strat.name}", in another way. stratify() works in place, so the stratification can be given once; to change it, build the model again.')
            self.strats.append(strat)
            for key,leaf in ({} if strat.mobility is None else strat.mobility.fitted()).items(): # A parameter of the movement between places, e.g. ss.Mobility(..., away=ss.par(0.1, bounds=[0.01, 0.5])), so it can be fitted
                if not any(p is leaf for p in self.pars.values()):
                    self._add_par(leaf, name=leaf.name or find_name(leaf, frame) or key)
                    self.module.define_pars(**{leaf.name:leaf})
        for name,tf in transfers.items():
            tf.name = name
            tf.module = self.module
            self.add(self.transfers, name, tf)
            self.module.setattribute(name, tf) # So it can be the target of ss.modify()
            for expr in [tf.rate, tf.dur]: # Register its parameters, as for those of the transitions
                for leaf in [] if expr is None else expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        self._add_par(leaf, name=leaf.name or find_name(leaf, frame) or f'rate_{name}')
                        self.module.define_pars(**{leaf.name:leaf})
        return self.module

    def add_flow(self, tr):
        """ Add a birth, death, or migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...)])`), after the graph is built; see `ss.Sim._init_flows()` """
        name = tr.name or ('births' if tr.per_capita else 'immigration' if tr.inflow else 'emigration' if tr.migration else 'deaths')
        if name in self.transitions:
            raise ss.DuplicateDeclaration(f'"{self.module.name}" already has a transition named "{name}"; give the one in demographics= another name, e.g. ss.birth(..., name="{name}2")')
        tr.name = name
        if tr.inflow: # Born into (or arriving in) the state named, or else the state people are in unless they are put elsewhere
            dst = tr.dst[0] if tr.arrow_given else self.first.name
            if dst not in self.states or not self.states[dst].people:
                raise ss.UnknownState(f'"{name}" in demographics= adds people to "{dst}", which is not a state of people of "{self.module.name}"; its states are {sc.strjoin(s.name for s in self.people_states)}')
            tr.dst = [self.states[dst]]
        if tr.rate is None and tr.dur is None:
            raise ss.UndrivenTransition(f'"{name}" in demographics= has no rate; give it one, e.g. ss.birth(rate=ss.peryear(0.02))')
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
        todo = [self.first]
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
        for p in self.all_pars():
            check(p.raw)
        for oc in self.outcomes.values():
            for val in [oc.delay, oc.duration]:
                check(val)
        return out

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
        return [f'{name}.{dst.name}' for name,tr in self.transitions.items() if tr.dst is not None and len(tr.dst) > 1 for dst in tr.dst]

    def leaves(self):
        """ The leaves of the rates and branch probabilities of its transitions """
        exprs = [expr for tr in self.transitions.values() for expr in [tr.rate] + sc.tolist(tr.p)]
        return [leaf for expr in exprs if isinstance(expr, Expr) for leaf in expr.leaves]

    def all_pars(self):
        """ Every parameter in the graph, named or inline """
        out = {id(p):p for p in self.pars.values()}
        exprs = [expr for tr in self.transitions.values() for expr in [tr.rate, tr.dur] + sc.tolist(tr.p)] + [upd.value for upd in self.updates.values()]
        for expr in exprs:
            if expr is not None:
                out.update({id(leaf):leaf for leaf in expr.leaves if isinstance(leaf, par)})
        return list(out.values())

    def unentered(self):
        """ The states inferred from arrow strings that nothing moves people into and that start empty, which are usually typos """
        if not self.inferred:
            return []
        first = next(iter(self.states.values()))
        entered = {id(first)} | {id(dst) for tr in self.transitions.values() for dst in tr.dst or []}
        entered |= {id(self.states[key]) for key in self.init if key in self.states}
        return [state for state in self.states.values() if id(state) not in entered]

    def spread_pars(self):
        """ The parameters whose value is a distribution across people (e.g. `beta=ss.lognorm_ex(0.3, 0.1)`), other than dwell times, which are stages or are sampled on entry """
        dwells = [tr.dur for tr in self.transitions.values()]
        return [p for p in self.all_pars() if isinstance(p.raw, ss.Dist) and not any(p is dur for dur in dwells)]
