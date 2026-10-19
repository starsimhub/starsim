"""
The graph of one module: its states, transitions, and parameters, collected, named, completed with defaults (each guess recorded), and checked
"""
import types
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import as_expr, BinOp, Count, did_you_mean, Expr, is_one, linear, par
from .parameters import by, TimeValue
from .transitions import bound_names, find_name, FlowSum, outcome, State, Transition, update
from .presentation import Presentation, TransitionTable

__all__ = ['Graph']


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
        self.resolve_people()
        self.resolve_pars(frame)
        self.set_keywords()
        self.resolve_exits()
        self.check()
        return

    def guess_init(self):
        """ When the sim is built: with no initial values at all, 1% start infected, since nobody infected would mean no epidemic """
        if self.init_prev is None and not self.init and self.transmissions and all(s.init is None for s in self.states.values()):
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
                strings += [s for s in sc.tolist(tr._pending[1]) if isinstance(s, str)]
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
                tr._pending = (beta, [lookup(end) for end in sc.tolist(source)])
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
                source = sources[0]
                for s in sources[1:]:
                    source = source + s
                tr.rate = as_expr(beta) * source
                tr._pending = None

            if tr.rate is None and tr.dur is None and tr.p is None:
                raise ss.UndrivenTransition(f'the transition "{name}" ({tr.arrow}) has no rate, no duration, and no p=. Give it one, e.g. {tr.arrow.replace(" -> ", " >> ")} @ 0.1')

            # Derive the kind from the rate
            tr.const, tr.terms = (None, {}) if tr.rate is None else linear(tr.rate)
            tr.contact = bool(tr.terms)
            for state in tr.terms:
                if not state.infectious: # Not declared, so a guess
                    state.guessed['infectious'] = name
                state.infectious = True
        return

    def anchor(self, states, connected):
        """
        The state that people are in unless they are put elsewhere (`first=`), which the other people states are connected to

        Unless it is given, it is found among the group of states connected to the ones seeded (given initial values, or
        entered by the contact-mediated transition that init_prev seeds through; else the largest group of connected
        states). With contact-mediated transitions, it is a source of one that people do not reach from another (S, not V
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
        group = connected(seeded) if seeded else max((connected([s]) for s in states), key=len)
        members = [s for s in states if id(s) in group]
        entered = {id(st) for tr in self.transitions.values() for st in (tr.dst or []) if isinstance(st, State)}
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
        seen = connected([first] + [s for s in states if s.people is True])
        for state in states:
            if state.people is None:
                state.people = id(state) in seen
                if not state.people:
                    state.guessed['people'] = f'no transition connects {state.name} to {first.name}'
        self.people_states = [s for s in states if s.people]
        self.count_states = [s for s in states if not s.people]
        self.first = first

        # Now that the people are known, fill in the things that depend on them
        for name,tr in self.transitions.items():
            self.fill_people(tr)
            for state in tr.terms:
                if not state.people:
                    raise ss.ExposureToCount(f'the rate for "{name}" uses the bare state {state.name}, and {state.name} is not people.\n'
                                             f'       rate = {tr.rate}\n'
                                             f'     A bare state means contact with people in that state, and there is no contact with a count.\n'
                                             f'     Write {state.name}.n for the number itself. If {state.name} is people, declare it with ss.State(people=True).')
            tr.people = all(s.people for s in (tr.src or []) + (tr.dst or []))
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
            if name not in self.pars:
                raise ss.UnknownParameter(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}"; its parameters are {sc.strjoin(self.pars.keys())}.\n'
                                          f'     To add a parameter, use it in a transition, e.g. infection = S >> I @ {name} * I with {name} = ss.par({value}).\n'
                                          f'     If another module reads it from this disease (as STIsim\'s networks read disease.pars.eff_condom), add it after making the disease: {self.module.name}.define_pars({name}={value!r}).')
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
        anon = [(tr, slot, p) for tr,slot,p in slots if p.literal or (p.name is None and find_name(p, frame) is None)] # Including an ss.par written in the slot itself, e.g. beta=ss.par(0.3, bounds=[0.1, 1])
        counts = sc.objdict()
        for tr,slot,p in anon:
            counts[slot] = counts.get(slot, 0) + 1
        inner = {q.name or find_name(q, frame) for tr,slot,p in slots if isinstance(p.value, TimeValue) for q in p.value.pars} # The names of parameters inside values that vary in time, e.g. ss.seasonal(mean=beta), which are registered below
        for tr,slot,p in anon:
            clash = counts[slot] > 1 or slot in self.pars or slot in self.states or slot in self.transitions or slot in inner
            self._add_par(p, name=f'{slot}_{tr.name}' if clash else slot)

        # Register the explicit ss.par objects used in the transitions, and in the weights and likelihoods of the outcomes
        used = [(tr.name, expr) for tr in self.transitions.values() for expr in [tr.rate, tr.dur] + sc.tolist(tr.p)]
        used += [(oc.name, weight) for oc in self.outcomes.values() for weight in oc.weights]
        used += [(oc.name, val) for oc in self.outcomes.values() if oc.likelihood is not None for val in oc.likelihood.pars.values() if isinstance(val, par)]
        for oc in self.outcomes.values(): # A parameter written inside a likelihood, e.g. ss.nbinom(n=ss.par(10, bounds=[1, 100])), is named after the outcome and its keyword
            for key,val in (oc.likelihood.pars.items() if oc.likelihood is not None else []):
                if isinstance(val, par) and not val.literal and val.name is None and find_name(val, frame) is None:
                    val.name = f'{oc.name}_{key}'
        used += [(f'init of {state.name}', state.init) for state in self.states.values() if isinstance(state.init, Expr)] # So an initial count can be fitted
        used += [(name, expr) for name,tf in self.transfers.items() for expr in [tf.rate, tf.dur]] # And a parameter in the rate of a transfer
        used += [(name, upd.value) for name,upd in self.updates.items()] # And in the value of an update
        used += [(name, inner) for name,expr in list(used) if expr is not None for leaf in expr.leaves if isinstance(leaf, par) and isinstance(leaf.value, TimeValue) for inner in leaf.value.pars] # And a parameter inside a value that varies in time
        for name,expr in used:
            if expr is not None:
                for leaf in expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        if leaf.name is None:
                            leaf.name = find_name(leaf, frame)
                        if leaf.name is None:
                            errormsg = f'Could not find a name for the parameter {leaf.value} used in "{name}"; give it one with ss.par(..., name="beta")'
                            raise ValueError(errormsg)
                        self._add_par(leaf, name=leaf.name)
        return

    def resolve_exits(self):
        """ Group the transitions out of each state: those with p= split on entry, the rest compete """
        self.exits = sc.objdict()
        for state in self.states.values():
            self.exits[state.name] = sc.objdict(branches=[], timed=[], hazards=[], contact=[])
        self.inflows = []
        for tr in self.transitions.values():
            if tr.inflow:
                self.inflows.append(tr)
            for state in tr.src:
                exits = self.exits[state.name]
                if tr.contact:
                    exits.contact.append(tr)
                elif tr.p is not None and len(tr.dst) <= 1: # One branch of several out of this state
                    exits.branches.append(tr)
                elif tr.dur is not None:
                    exits.timed.append(tr)
                else:
                    exits.hazards.append(tr)

        # A single timed transition with no p= takes whatever the branches leave over
        for name,exits in self.exits.items():
            exits.remainder = None
            if exits.branches:
                ps = [tr.p[0] for tr in exits.branches]
                const = all(isinstance(p, par) and sc.isnumber(p.raw) for p in ps)
                total = sum(p.raw for p in ps) if const else None
                if not const and len(exits.timed) == 1: # A share that is worked out as the sim runs (e.g. a function): the timed exit with no p= takes the rest, then
                    exits.remainder = exits.timed.pop()
                    exits.remainder.guessed['p'] = f'1 - {" - ".join(tr.name for tr in exits.branches)}'
                elif const and not np.isclose(total, 1):
                    if total < 1 and len(exits.timed) == 1:
                        exits.remainder = exits.timed.pop()
                        exits.remainder.guessed['p'] = f'{1 - total:n}'
                    else:
                        raise ss.BadBranchSplit(f'the branch probabilities out of state "{name}" sum to {total:n}, not 1 '
                                                f'({sc.strjoin([f"{tr.name}: p={tr.p[0]}" for tr in exits.branches])}).')
        return

    def check(self):
        """ Checks that do not need the sim """
        for name,tr in self.transitions.items():
            tr.check_dimensions()
            if tr.p is not None and len(tr.dst) > 1:
                ps = tr.p
                if len(ps) == 1 and len(tr.dst) == 2: # With two destinations, a single value is the first one's probability
                    ps = tr.p = [ps[0], None]
                if len(ps) != len(tr.dst):
                    raise ss.BadBranchSplit(f'"{name}" has {len(tr.dst)} destinations but {len(ps)} branch probabilities.')
                given = [p for p in ps if p is not None]
                if all(isinstance(p, par) and sc.isnumber(p.raw) for p in given):
                    total = sum(p.raw for p in given)
                    if (len(given) == len(ps) and not np.isclose(total, 1)) or total > 1 + 1e-9:
                        raise ss.BadBranchSplit(f'the branch probabilities of "{name}" sum to {total:n}, not 1.')
            elif tr.p is None and len(tr.dst) > 1:
                raise ss.BadBranchSplit(f'"{name}" has {len(tr.dst)} destinations but no branch probabilities; give them with p=[...].')
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
        tr.people = True
        tr.module = self.module
        self.set_slots(tr)
        self.add(self.transitions, name, tr)
        self.module.setattribute(name, tr) # e.g. sir.births, so it can be the target of ss.modify()
        for expr in [tr.rate, tr.dur]: # Register its parameters, as for those of the transitions
            for leaf in [] if expr is None else expr.leaves:
                if isinstance(leaf, par) and not any(p is leaf for p in self.pars.values()):
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
    def uninfected(self):
        """ The names of the states people can be in without ever having been infected: those reached from the first state by transitions that are not contact-mediated """
        seen = []
        todo = [self.first]
        while todo:
            state = todo.pop()
            if state.name not in seen and state.people:
                seen.append(state.name)
                todo += [dst for tr in self.transitions.values() if not tr.contact and state in tr.src for dst in tr.dst]
        return seen

    @property
    def infected(self):
        """ The states counted in prevalence: the sources of infection and the states people are infected into """
        out = {id(s):s for tr in self.transmissions for s in tr.dst}
        out.update({id(s):s for s in self.infectious})
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
        """ What a modifier on a transition acts on: its probability if it is one branch of several with p= (chosen on entry), and otherwise its rate """
        tr = self.transitions.get(name)
        if tr is not None and tr.p is not None and len(tr.dst) <= 1 and not tr.contact:
            return 'branch'
        return 'transition'

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
