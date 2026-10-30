"""
Not for the user: the agent engine: a declared model run as agents (`abm`)

It also holds the transmission that diseases written as code use, so that both give the same numbers.

Used with: agents; declared and coded modules
"""
import functools
import operator
import numpy as np
import sciris as sc
import starsim as ss
from .engine import Bunch, CountState, Effects, Engine, times
from .compartmental import Compartmental

__all__ = ['Transmission', 'Agents']

check_counts = sc.parse_env('STARSIM_CHECK_COUNTS', False, bool) # Whether the agent engine checks the counts it keeps against a new count at each step: see ss.solvers.Agents.check_counts()
check_draws = sc.parse_env('STARSIM_CHECK_DRAWS', False, bool) # Whether the agent engine checks that no agent is drawn twice from a stream in a round: see ss.solvers.Agents.draw()
ROUNDS = 2**52 # The index of the engine's draws is ROUNDS + ti*ROUND_MAX + round, apart from each distribution's own (see Agents.draw())
ROUND_MAX = 2**20


def power(a, b):
    """
    `a**b` in an expression: a number to the power of whole numbers for many agents (e.g. a protection per past infection,
    `(1 - p)**n_infections`) is taken from a table of its powers, the same numbers with one power worked out for each whole number
    """
    if np.ndim(a) == 0 and np.ndim(b) == 1 and len(b) > 100 and b.dtype.kind == 'f':
        lo, hi = b.min(), b.max() # NaN if any is, which is worked out as usual
        if 0 <= lo and hi < 1000:
            whole = b.astype(np.int64)
            if (whole == b).all():
                return np.power(a, np.arange(int(hi) + 1, dtype=b.dtype))[whole]
    return a ** b


def member(a, b):
    """ Which of the UIDs `a` are in `b`, as np.isin(), for the few agents of a round of moves (whose fixed cost np.isin() is several times) """
    if not len(b):
        return np.zeros(len(a), dtype=bool)
    b = np.sort(b)
    return b[np.minimum(np.searchsorted(b, a), len(b) - 1)] == a


class Transmission(Effects):
    """
    Transmission as agents, the same for a declared disease (`Agents`, for its transmissions) and one written as code (`ss.Infection.transmit_route()`,
    through its engine, `ss.solvers.Undeclared`): who can transmit and be infected, with the modifiers and products on `rel_trans` and `rel_sus`
    applied (`current_value()`); who is infected over a network (`over_network()`); and who infected whom (`first_infections()`)

    Where the two differ, the caller says how:

    - **Hazard or probability.** `over_network()` takes the chance of an edge in the step. A declared disease gives 1 − exp(−β·dt) from its rate,
      or, on a network with acts, works out a hazard per act (`Agents.transmit_acts()`); one written as code gives its beta as the probability in
      the step, converting a rate the same way (`ss.Rate.to_prob()`).
    - **Order and attribution.** The routes are taken in the order of `sim.networks`, and on each, p1 to p2 and then p2 to p1; someone reached by
      several is infected by the first (`first_infections()`), which is the source in the infection log. With strains, every strain is passed on
      in one pass, and someone reached by several is infected by one of them drawn at random, whose strain they take (`pick_sources()`), and
      then only if they are not protected against it (`unprotected()`): declared or written as code, by the same code.
    - **Random numbers.** Each edge's is drawn by its source and target from the disease's `trans_rng`, so the same disease, by the same name,
      gives the same edges the same numbers, declared or written as code; and so are the draws of who infects whom, and of protection, which
      are keyed by the step alone (`step_draw()`).
    """
    def current_value(self, kind, *factors):
        """ `rel_sus` or `rel_trans` (`kind`) of every agent (by UID), times the factors (e.g. whether each is susceptible), with the modifiers and products active on it applied; may be the array itself, so it is not to be changed """
        return times(self.mod[kind].raw, *factors, self.mult_raw(kind))

    def route_mult(self, name, route):
        """ The multiplier on a transmission over a route, as a function of the edges and of the people who would be infected (see `ss.connectors.route_mult()`); `name` is that of a declared disease's transmission, or `rel_sus` for a disease written as code """
        mods = self.sim.modifiers
        if mods.get((route.name, 'par', 'n_contacts')) or mods.get((self.mod.name, 'route', f'{name}@{route.name}')):
            return lambda inds, trg: ss.connectors.route_mult(self, name, route, inds, trg)
        return lambda inds, trg: 1.0

    def over_network(self, i, route, rel_trans, rel_sus, betas, name):
        """
        Who is infected over a network in the step: an edge from `u` to `v` transmits with probability `rel_trans[u]·rel_sus[v]·net_beta(beta)`,
        times the modifiers on the route for `v` (see `ss.Network.transmit()`)

        Args:
            i (int): the index of the route in `sim.networks`, recorded for who infected whom
            route (`ss.Network`): the network
            rel_trans (array): the effective transmissibility of each agent, by UID (0 if not infectious)
            rel_sus (array): the effective susceptibility of each agent, by UID (0 if not susceptible)
            betas (list): the probability of transmission over an edge in the step, p1 to p2 and p2 to p1: each a number (0 for none, so no random
                numbers are drawn), a rate (`ss.Rate.to_prob()`), or an array of one for each of the network's directions by sides (see `ss.peract`)
            name (str): what the modifiers on the route are under (see `route_mult()`)

        Returns:
            A list of (sources, targets, i), for each direction that transmits
        """
        mult = self.route_mult(name, route)
        def edge_beta(d, inds, src, trg): # For the edges that might transmit
            beta = betas[d]
            if isinstance(beta, np.ndarray): # A probability per act for each direction of the network
                beta = beta[route.edge_directions(*(route.ends(d) if route.custom_beta else (src, trg)))] # For every edge, for a net_beta() of its own
            elif not beta: # No random numbers are drawn
                return None
            if isinstance(beta, ss.Rate): # A disease written as code can give its beta as a rate
                beta = beta.to_prob(self.mod.t.dt)
            return route.edge_net_beta(inds, disease_beta=beta, disease=self.mod)*mult(inds, trg)
        return [(srcs, trgs, i) for srcs,trgs in route.transmit(rel_trans, rel_sus, edge_beta, self.mod.trans_rng)]

    @staticmethod
    def first_infections(pairs):
        """
        Each agent infected, once, with who infected them and over which route: the first of `pairs` to reach them. Each pair is the sources, the
        targets, and the index of the route (one, or one for each), and any more values (e.g. a strain); returns the targets (sorted), and for
        each, the source (`ss.dtypes.int_nan` where not known), the route, and the more values, of the first pair to reach them
        """
        if not pairs:
            return ss.uids(), ss.uids(), np.empty(0, dtype=ss.dtypes.int)
        trgs, first = ss.uids.concatenate([pair[1] for pair in pairs]).unique(return_index=True)
        srcs = ss.uids.concatenate([pair[0] for pair in pairs])[first]
        rest = [np.concatenate([np.broadcast_to(np.asarray(pair[c], dtype=ss.dtypes.int), len(pair[1])) for pair in pairs])[first] for c in range(2, len(pairs[0]))]
        return trgs, srcs, *rest

    def step_draw(self, key, uids):
        """ Uniforms for some agents from one of the transmission's distributions (`self.dists[key]`), keyed by the step alone, so the same agents get the same numbers, declared or written as code; without common random numbers, its next draws """
        dist = self.dists[key]
        if not len(uids):
            return np.empty(0)
        elif not ss.options.crn:
            return dist.rvs(uids)
        slots = self.sim.people.slot_raw()
        return ss.dists.hash_uniforms(dist.seed, ROUNDS + self.mod.t.ti*ROUND_MAX, uids if slots is None else slots[uids], dtype=dist.hash_dtype)

    def pick_sources(self, pairs, name):
        """
        With strains, each agent reached in the step, once, with who infected them and the rest of that pair's values: one of those who reached
        them, each as likely (the first to happen is as likely to be any of them), drawn from `pick_<name>`. Each pair is the sources, the
        targets, the index of the route (one, or one for each), and any more values (e.g. a strain), as for `first_infections()`
        """
        trgs = ss.uids.concatenate([pair[1] for pair in pairs])
        cols = [ss.uids.concatenate([pair[0] for pair in pairs])] + [np.concatenate([np.broadcast_to(np.asarray(pair[c], dtype=ss.dtypes.int), len(pair[1])) for pair in pairs]) for c in range(2, len(pairs[0]))]
        order = np.argsort(trgs, kind='stable')
        trgs, first, counts = np.unique(trgs[order], return_index=True, return_counts=True)
        if np.any(counts > 1): # Reached by several in the step
            first = first + np.minimum((self.step_draw(f'pick_{name}', ss.uids(trgs))*counts).astype(int), counts - 1)
        return ss.uids(trgs), *[col[order][first] for col in cols]

    def unprotected(self, name, uids, kept):
        """ Which of some agents reached by a strain are infected by it: each with the chance `kept`, the share of the force of infection of the strain that reaches them (1 − their protection against it), drawn from `protect_<name>` """
        return self.step_draw(f'protect_{name}', uids) < kept


class Agents(Engine, Transmission):
    """
    Runs a graph as agents

    It exists so that a declared model can run with individuals, for what compartments cannot hold.

    Each state that holds people is a boolean array over agents. A dwell time is
    sampled when an agent enters a state and the exit is scheduled; rates that depend
    on the state of the model are drawn each step, with competing rates out of one
    state resolved in a single draw; and contact-mediated transitions are drawn over
    the sim's networks. States that are not people stay plain counts.
    """
    def define(self):
        """ Create the per-agent states and the distributions """
        mod = self.mod
        graph = self.graph
        self.init_dims()
        for obj in list(graph.states.values()) + list(graph.attrs.values()): # Remove the declarations, which are replaced by their run-time form
            if obj.name in mod.__dict__:
                delattr(mod, obj.name)
        for key in ['rel_sus', 'rel_trans']:
            mod.__dict__.pop(key, None)
        self.events = dict() # The agents who went through each transition this step
        self.also = type(mod).also_happened is not ss.Disease.also_happened # Whether the module says whom else an event happens to: see record()
        self.live = None # The number of agents in each state by group, kept up to date from the census: see census() and wrote()
        self.unscheduled = None # The states that people have been put in other than by the engine's moves (by code, or at the start) since the last step, so that may have people with no exit scheduled; None for any state
        self.live_total = None # The number alive in each group, likewise
        self.n_counted = 0 # The agents (by UID) that the number alive has been kept for: see alive_counts()
        self.start = None # The number in each state by group at the start of the step
        self.start_weighted = None # The same, with each agent counted by its weight, if agents' weights differ (people.scale)
        self._members = None # The agents in each state at the start of the step (those that step_state() needs), while its rates are worked out
        self.forced = dict() # The agents moved along a transition by an intervention (see force()), who are events from the next step
        self._kept_exits = {} # For each split with ss.stay triggered by on=, the exits of those it moved, for those who stay (see stay())
        self._step_pairs = [] # Who infected whom over the networks in this step's transmissions: see sources()
        self.held = None # As a branching process: the number in each state of the uninfected, by group, which are counts rather than agents (see init_branching())
        self._drawn_ti = -1 # The step whose competing rates have been drawn: see enter()
        self.batches = [] # Who entered a state, from where, and when, for the events of branches and of entering and leaving states: see event_uids()
        self.n_batches = 0 # The number of each batch, in the order recorded, so what is new since a number is found at the end: see unseen()
        self.last_batch = dict() # The number of the last batch of each transition, so a trigger on it can tell at once that nothing has happened since it looked: see new_events()
        self._phase = (-1, 0) # Where the engine is in a step: see start_phase()
        self.moved_early = None # Moved by code (ss.move()) before the start, so they have no history to fill in: see fire_placed()
        self.state_index = {state.name:i for i,state in enumerate(graph.people_states)}
        self.triggers = [tr for tr in graph.transitions.values() if tr.trigger is not None and not tr.trigger.outcome] # Moved by ss.on(): see triggered(); those on the outcome of a delivery move as it happens (see ss.interventions.trigger_moves())
        self.trigger_mods = set().union(*[ss.conditions.module_refs(tr.trigger.x) for tr in self.triggers]) # The modules whose changes they react to
        self._seen = dict() # The events each trigger on an event has seen already
        self.lifelong = dict() # Values by sex for everyone, by UID, and the parameters of the distributions that use them, kept for the run: see by_people() and agent_pars()
        self._news = False # Whether anyone has moved since the modules that react to this one last looked: see notify()
        self._subscribers = None
        self._started = False
        self._backfill = None # While people are back-dated: see backfill()
        self.round = (None, 0) # The step and the round of moves within it that the engine's draws are keyed by: see new_round()
        self._drawn = {} # With STARSIM_CHECK_DRAWS, who has been drawn from each stream in this round: see draw()
        self._route_memo = None # The coefficients worked out on each route in a transmission: see route_beta()
        self._draw_how = {} # How each of the engine's distributions is drawn from: see draw()
        self._died = [] # Deaths by its transitions in the step: see uncount_deaths()
        self._fired_seen = {} # For each trigger on a delivery's or product's event, who it has seen it for: see unseen_fired()
        self._row = None # Who has had half a step of the competing rates of the state they entered in the state-change row: see finish_moves()
        self._infecting = False # While the infections of the step, and the moves at once after them, are made: see enter()
        self.halved = set() # The sources of infection that people can enter by an infection, who have half a step of their rates at the first draw: see enter()
        self._junctions = None # The states passed through at once: see junctions
        self._leaving = None # The people leaving a state and the states they leave, while clocks (see clock_values()) are read for them
        self._uids = None # The agents being evaluated: see eval_for()
        self._for_uids = False
        self._memo = None # Values of parts of expressions kept while several are evaluated for the same agents: see compile()
        self.compiled = dict() # Each expression evaluated, as a function: see eval()
        arrs = [ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility', shared=True), # No memory until something sets them; the modifiers on them are applied by effective()
                ss.FloatArr('rel_trans', default=1.0, label='Relative transmission', shared=True)]
        self.exits_of = {state.name: graph.exits[state.name].branches + sc.tolist(graph.exits[state.name].remainder) + graph.exits[state.name].timed for state in graph.people_states} # The exits scheduled when people enter each state: its branches, and its timed exits (the earliest is taken), by index
        self.scheduled = [name for name,exits in self.exits_of.items() if exits] # The states with exits that are scheduled when people enter them
        self.loops_of = {name:np.array([any(getattr(dst, 'name', None) == name for dst in (tr.dst or [])) for tr in exits], dtype=bool) for name,exits in self.exits_of.items()} # Which exits lead back to the state: a decision taken again, e.g. FPsim's choice of method
        self._rescaled = None # Which exits of each state are rescaled when their speed changes: see rescaled_of()
        self.init_dwells()
        modifies = lambda m: isinstance(m, ss.modify) or bool(getattr(getattr(m, 'product', None), 'effects', None)) # A modifier, or a delivery of a product with an efficacy (TL17: these were left out, so a product did nothing against a dwell time)
        rescalable = any(modifies(m) for m in list(self.sim.interventions()) + list(self.sim.connectors())) or any(self.varies(dw) for dw in self.dwells.values()) # Whether a dwell time can change after it is drawn: see rescale_exits()
        rescalable = rescalable or any(isinstance(fam, ss.strains.Family) and any(key.startswith('dur.') for imm in fam.immunity for key in imm.values) for fam in self.sim.diseases()) # Or protection on the time in a state, from a family
        strats = list(graph.strats) + [s for s in self.sim.groups.sim_strats if s not in graph.strats]
        adjusted = {self.sim.groups.dim_index(s.name) for s in strats if any(key.rpartition('.')[2] in graph.transitions for key in s.adjust)}
        rescalable = rescalable or bool(adjusted & self.changing_dims) # Or a multiplier by level, for people who can change level
        read = None if ss.options.keep_times else set() # The times of entering a state kept for code to read (None for all); the engine keeps those of the states it schedules exits from, that a clock reads, or whose exits have after=
        clocked = {name for names in ss.expressions.Clock.sets for name in names} # The states a clock reads (from this module or another), whose times of entering are kept
        clocked |= {name for m in list(self.sim.declared) + list(self.sim.pars.connectors.values()) + list(self.sim.pars.interventions.values()) for clock in ss.solvers.choose.clocks_of(m) for name in clock.names}
        infected = {d.name for tr in graph.transmissions for d in tr.dst if getattr(d, 'people', False)} # The states people can enter at a step's end, by an infection or the moves at once after it
        todo = list(infected)
        while todo:
            exits = graph.exits[todo.pop()]
            onward = exits.branches + sc.tolist(exits.remainder) + exits.timed + self.triggers # A dwell time can round to no steps; a trigger can be anywhere
            for d in {d.name for tr in onward for d in (tr.dst or []) if getattr(d, 'people', False)} - infected:
                infected.add(d)
                todo.append(d)
        self.halved = infected & {s.name for s in graph.infectious if graph.exits[s.name].hazards}
        self.late = set() # The states with a late_ array: see enter()
        unkept = [state.name for state in graph.people_states if read is not None and not state.keep_time and state.name not in read and state.name not in self.scheduled and state.name not in clocked and not any(getattr(tr, 'after', None) is not None for tr in graph.exits[state.name].hazards)]
        exited = {s.name for tr in graph.transitions.values() if any(getattr(d, 'people', False) for d in (tr.dst or [])) for s in (tr.src or []) if getattr(s, 'people', False)}
        self.lefts = {self.state_index[name]:f'ti_left_{name}' for name in self.state_index if name in exited and name not in unkept} # The engine's time each agent last left a state that leads to another and whose time of entering is kept, for its stays: see enter() and ss.BoolState.stays
        for state in graph.people_states:
            name = state.name
            arrs += [ss.BoolState(name, label=state.label or name)]
            if name not in unkept: # Only if something reads it, to save memory
                arrs.append(ss.TimeArr(f'ti_{name}', label=f'Time of entering {name}'))
                arrs.append(ss.Arr(f'shift_{name}', dtype=np.int8, nan=0, default=0, label=f'Steps from the engine\'s time of entering {name} to the step\'s index')) # See stamp()
                if self.state_index[name] in self.lefts:
                    arrs.append(ss.FloatArr(f'ti_left_{name}', label=f'The engine\'s time of last leaving {name}')) # See enter()
            if name in self.scheduled: # Only for a state with scheduled exits, to save memory
                arrs += [ss.TimeArr(f'ti_exit_{name}', label=f'Scheduled time of leaving {name}'),
                         ss.Arr(f'exit_{name}', dtype=np.int8, nan=-1, label=f'Scheduled exit from {name}')] # Which of the exits, as 8-bit integers
                if rescalable: # Only needed if a rate can change after dwell times are drawn: see rescale_exits()
                    arrs.append(ss.FloatArr(f'mult_{name}', default=1.0, label=f'Speed of the scheduled exit from {name}, when it was scheduled'))
            if graph.exits[name].hazards and (name in self.scheduled or name in self.halved): # See enter()
                arrs.append(ss.FloatArr(f'late_{name}', default=0.0, label=f'The share of a step of the competing rates out of {name} missed in the step of entering it (less than 0 if too much)'))
                self.late.add(name)
        self.attrs = graph.attrs # Attributes: an array each (for a derived one, its values when last worked out); see fire() and attr_values()
        self.on_event = [a for a in self.attrs.values() if a.events] # Set at events, in the order declared
        self.traits = [a for a in self.attrs.values() if a.kind == 'trait']
        self.derived = graph.derived
        self._drawing = False
        if self.on_event:
            arrs.append(ss.FloatArr('left_exit', label='The scheduled exit from the state each agent last left'))
        for a in self.attrs.values():
            arrs.append(ss.FloatArr(a.name, default=None if a.kind == 'derived' else a.default, label=a.label or a.name))
            if a.combine == 'first':
                arrs.append(ss.BoolArr(f'had_{a.name}', label=f'Whether {a.name} has been set'))
        self.sets = sc.obj() # The sets of states with a clock, and the name of the array of the time each person entered the set: see ss.expressions.Clock
        people = [state.name for state in graph.people_states]
        for names in ss.expressions.Clock.sets:
            if all(name in people for name in names):
                key = tuple(sorted(set(names), key=people.index))
                if key not in self.sets:
                    self.sets[key] = f'ti_{"_or_".join(key)}'
                    arrs.append(ss.TimeArr(self.sets[key], label=f'Time of entering {" | ".join(key)} from outside it'))
        self.sets_of = {name:[(arr, [self.state_index[k] for k in key]) for key,arr in self.sets.items() if name in key] for name in people} # For each state, the sets with a clock it is in, and their states
        strains = next((strat.strains for strat in graph.strats if strat.strains is not None), None)
        self.spread = dict() # A parameter that is a distribution across people: each agent's own value, drawn when they are made
        for p in graph.spread_pars():
            self.spread[id(p)] = f'par_{p.name}'
            arrs.append(ss.FloatArr(f'par_{p.name}', default=ss.transitions.numeric_dist(p.raw, self.convert), label=f'Value of {p.name}'))
        self.delays = {tr.name:tr.trigger.delay for tr in self.triggers if tr.trigger.delay is not None} # after= an event, with dur=: each person's time after it (see triggered())
        for name in self.delays:
            arrs.append(ss.TimeArr(f'ti_after_{name}', label=f'Scheduled time of {name}, after its event'))
        self.tf_dists = {name:dist for name,tf in graph.transfers.items() if (dist := self.transfer_dist(tf, per_agent=self.agent_values)) is not None} # A transfer whose dur= is a distribution: each person's time is drawn when they can first move, and they move when it is up (see step_transfers())
        for name in self.tf_dists:
            arrs.append(ss.TimeArr(f'ti_move_{name}', label=f'Scheduled time of the transfer {name}'))
        mod.define_states(*arrs, overwrite=True) # Overwrite, since in the class form the declarations are class attributes
        for state in graph.people_states:
            if state.name not in unkept:
                mod[f'ti_{state.name}'].shift = mod[f'shift_{state.name}']
            if state.name in self.scheduled:
                mod[f'ti_exit_{state.name}'].shift = 1 # Read as the step the exit is taken in (exits are taken at a step's start)
        for name in unkept: # Setting it does nothing, and reading it says how to keep it
            mod.setattribute(f'ti_{name}', ss.arrays.Unkept(f'ti_{name}', mod))
        for state in graph.people_states: # Each change to who is in a state is counted: see wrote()
            mod[state.name].watch = ss.arrays.Watch(self)
        self.counts = dict()
        self.count_exprs = dict() # Whether each part of a rate of counts uses them: see count_rate()
        for state in graph.count_states:
            self.counts[state.name] = np.zeros(self.sim.groups.n_groups) # One number per group; all but the first are zero unless the state has dimensions of its own
            mod.setattribute(state.name, CountState(self, state))

        # Distributions: each is called at most once per step
        self.dists = sc.obj()
        self.by_groups = dict() # Whether the rate of each transmission is the same for everyone in a group: see same_in_group()
        add = self.add_dist
        def copy(dist):
            """ A copy of a distribution of the model, for the engine to draw from: without the sim and module it may be linked to already (by a module that reads it, e.g. a modifier of an attribute), so it is this disease's, and steps with it """
            sim, module = dist.sim, dist.module
            dist.sim = dist.module = None
            try:
                return sc.dcp(dist)
            finally:
                dist.sim, dist.module = sim, module
        add('init', ss.random())
        add('seed_who', ss.random()) # Who is seeded, by an init_prev for each agent: see init_post()
        add('seed_where', ss.random()) # Where the seeded go, if a new infection has several destinations: see seed()
        add('move', ss.random()) # Who code moves (ss.move()): see move_some()
        if type(self.mod).burnin_plan is not ss.Disease.burnin_plan:
            add('burnin', ss.random()) # Who starts part way through, and since when: see backfill_burnin()
        for state in graph.people_states:
            init = state.init.raw if isinstance(state.init, ss.par) else state.init
            if isinstance(init, ss.by) or (isinstance(init, ss.expressions.Expr) and ss.expressions.per_person_in(init)): # For a share given for each person's own properties: see init_post()
                add(f'init_{state.name}', ss.random())
        for state in graph.people_states:
            add(f'branch_{state.name}', ss.random())
            add(f'exit_{state.name}', ss.random())
        carriage = any(getattr(strat, 'carries', None) is not None for strat in self.sim.groups.strats)
        self.lifetables = {name:None for name,tr in graph.transitions.items() if getattr(tr, 'lifetable', None) is not None} # Who dies in each step by each life table: see step_lifetables()
        for name in self.lifetables:
            add(f'life_{name}', ss.random())
        self.competing = {state.name: [tr for tr in graph.exits[state.name].hazards if tr.name not in self.lifetables] for state in graph.people_states} # The rates drawn each step: a life table's deaths are drawn once for each person instead, see step_lifetables()
        for name,competing in self.competing.items():
            if competing:
                add(f'entry_{name}', ss.random()) # The competing rates of the step people enter in: see finish_moves()
        self.listed = [state.name for state in graph.people_states if self.competing[state.name] or self.exits_of[state.name]] # The states whose agents step_state() goes through
        for name,tr in graph.transitions.items():
            add(f'dst_{name}', ss.random())
            if tr.trigger is not None and tr.trigger.outcome:
                add(f'trig_{name}', ss.random()) # Who of the people of the outcome move: see ss.interventions.trigger_moves()
            add(f'mix_{name}', ss.random())
            if carriage and tr.contact:
                add(f'pass_{name}', ss.random()) # Which strain someone who carries several passes on
            if any(isinstance(level, tuple) and level[0] == 'matrix' for level in (tr.move or {}).values()):
                add(f'level_{name}', ss.random()) # Which strains are left (see ss.strains.prepare())
            if tr.contact and strains is not None:
                add(f'pick_{name}', ss.random()) # Which of several contacts who reach someone in a step infects them (see transmit())
                add(f'protect_{name}', ss.random()) # Whether someone protected in part against the strain that reaches them over a network is infected
        for name,delay in self.delays.items():
            raw = ss.transitions.by_dist(getattr(delay, 'raw', None))
            add(f'after_{name}', ss.transitions.numeric_dist(raw, self.convert, per_agent=self.agent_values) if isinstance(raw, ss.Dist) else ss.random()) # A scalar is the mean of an exponential, as for any dur=
        for name,tf in graph.transfers.items():
            add(f'transfer_{name}', self.tf_dists.get(name, ss.random()))
            if tf.p is not None:
                add(f'transfer_to_{name}', ss.random()) # Which level they move to
        if self.sim.mixing.migration is not None:
            add('migrate', ss.random())
        adders = [f'born_{tr.name}' for tr in graph.inflows if tr.people] # What adds people: births, and a movement register
        if self.sim.mixing.register is not None:
            add('register', ss.random()) # The order in which people in a place are chosen
            adders.append('register')
        for key in adders: # For the people added: their state, and their level of each dimension other than place and age
            add(f'{key}_state', ss.random())
            for strat in self.sim.groups.strats:
                if strat.kind not in ['space', 'age']:
                    add(f'{key}_{strat.name}', ss.random())
        self.entered_dims = [] # A dimension that only some states have: the level people take when they enter one of those states (see enter_dims())
        for d,strat in enumerate(self.sim.groups.strats):
            if strat.name in [own.name for own in graph.strats] and strat.states is not None:
                add(f'enter_{strat.name}', ss.random())
                self.entered_dims.append((d, strat, f'enter_{strat.name}'))
        for tr in graph.inflows: # Births driven by a flow: the number of children of each parent, and their sex (see step_per())
            if getattr(tr, 'per', None) is not None and tr.people:
                if isinstance(tr.n_born, ss.Dist):
                    add(f'n_{tr.name}', copy(tr.n_born))
                if tr.sex is not None:
                    add(f'sex_{tr.name}', copy(tr.sex) if isinstance(tr.sex, ss.Dist) else ss.bernoulli(p=tr.sex))
        self.attr_dists = dict() # Each distribution an attribute is drawn from, and the key of its copy: see draw_attr()
        for a in self.attrs.values():
            values = [('', a.value)] if a.kind in ['drawn', 'trait'] else [(f'event{j}_', v) for j,(_,v) in enumerate(a.by_event or [])] # Each event's own (TY1)
            values += [('init_', a.init_value)] if a.init_value is not None else [] # And the value before any event, if drawn (TY3)
            for prefix,value in values:
                for i,p in enumerate(ss.expressions.dists_in(value)):
                    key = f'attr_{a.name}_{prefix}{i}'
                    add(key, ss.transitions.numeric_dist(p.value, self.convert, per_agent=self.agent_values)) # In the sim's unit, as a dwell time is (a distribution of typed values would draw them in steps)
                    self.attr_dists[id(p.value)] = key
        mod.setattribute('trans_rng', ss.multi_random('source', 'target'))
        for name,dw in self.dwells.items():
            add(f'dwell_{name}', dw.dist if dw.dist is not None else ss.expon(scale=1.0) if dw.k == 1 else ss.gamma(a=dw.k, scale=1/dw.k)) # A unit-mean exponential (or Erlang), scaled by the mean when used
        return

    def init_pre(self):
        for tr in self.graph.transmissions: # Not yet built for agents either, so not silently ignored
            if getattr(tr, 'after', None) is not None:
                raise ss.CapabilityError(f'"{tr.name}" has a minimum stay (after=), which a transmission cannot have yet; put it on the transition out of the state before, or out of the state infected people enter')
        self.init_immunity()
        self.init_adjust()
        self.init_results()
        self.note_steps()
        for arr in self.mod.state_list:
            if isinstance(arr, ss.TimeArr):
                arr.t = self.mod.t
        return

    def groups(self, uids, state=None):
        """ The group of each agent; given a state, the group they are counted in while in it (which leaves out the dimensions the state does not have) """
        out = self.sim.groups.group_raw()[uids]
        return out if state is None else self.collapse[state.name][out]

    def weights(self, uids):
        """ The number of people each of some agents stands for relative to the others (`people.scale`), or None if every agent stands for the same number """
        people = self.sim.people
        return people.scale.raw[uids] if people.scale.weighted else None

    def n_of(self, uids):
        """ The number of people some agents stand for, before the population scale: their number, or, if agents are weighted (`people.scale`), the sum of their weights """
        return len(uids) if not self.sim.people.scale.weighted else float(self.sim.people.scale.raw[uids].sum())

    def strain_raw(self):
        """ The strain each agent carries, as an array indexed by UID """
        return self.sim.groups.level_raw(self.strain)

    def adjusted(self, name, uids):
        """ The multiplier on a transition for each agent, from the adjust= of the stratifications (1 if there is none), and their susceptibility if it is an acquisition that is not contact-mediated (see `transmit()` for those) """
        adjust = self.adjust[name]
        if adjust is None and not self.sim.modifiers.get((self.mod.name, 'transition', name)) and not self.mult_raw_extra('transition') and not self.graph.transitions[name].acquisition: # The usual case
            return 1.0
        out = 1.0 if adjust is None else adjust[self.groups(uids)]
        tr = self.graph.transitions[name]
        if tr.acquisition and not tr.contact:
            out = out*self.mod.rel_sus[uids]*self.mult('rel_sus', None, uids)
            for src in tr.src:
                if src.rel_sus != 1:
                    out = out*np.where(self.mod[src.name][uids], src.rel_sus, 1.0)
        return out*self.mult('transition', name, uids)

    def members(self, state):
        """
        The agents in a state, as UIDs

        Read from the state's array directly rather than through the living agents
        (`.uids`), which is several times faster for a large population: anyone who
        has died has already left every state (see `die()`).
        """
        return ss.uids(ss.solvers.backends.passes.flatnonzero(self.mod[state.name].raw))

    def adjusted_rel(self, kind, state=None):
        """ The multiplier on rel_trans or rel_sus for each agent (by UID), for those in a state that has the dimension """
        key = ('rel_adjust', kind)
        if key not in self.cache:
            out = 1.0
            for d,vals in self.rel_adjust[kind]:
                inside = np.zeros(len(self.mod.rel_trans.raw), dtype=bool)
                for s in self.graph.people_states:
                    if self.applies[s.name][d]:
                        inside |= self.mod[s.name].raw
                out = out*np.where(inside, self.sim.groups.group_vals(d, vals)[self.sim.groups.group_raw()], 1.0)
            self.cache[key] = out
        return self.cache[key]

    def mult_raw_extra(self, kind):
        """ Whether mult_raw() has a multiplier on a kind of quantity other than its modifiers: the adjust= of the stratifications, or a strain's own value """
        return bool(getattr(self, 'rel_adjust', None) and self.rel_adjust.get(kind)) or kind in self.__dict__.get('rel_scale', ())

    def mult(self, kind, name, uids):
        """ The multiplier on a quantity for some agents, from its modifiers """
        out = self.mult_raw(kind, name)
        return out if np.ndim(out) == 0 else out[uids]

    def move_levels_of(self, tr, dst, uids):
        """ The change of level of a transition with `move=` or `increment=`, for the agents it takes to a state """
        for d,level in (self.moves(tr) or ()):
            strat = self.sim.groups.strats[d]
            arr = self.sim.people[strat.attr]
            if isinstance(level, tuple) and level[0] == 'matrix': # A level drawn from each one's row, for those going where it applies (see ss.strains.prepare())
                if self.applies[dst.name][d] and len(uids):
                    cdf = np.cumsum(np.array(level[1])[np.asarray(arr[uids], dtype=int)], axis=1)
                    arr[uids] = np.minimum((cdf <= self.draw(f'level_{tr.name}', uids)[:, None]).sum(axis=1), strat.n - 1)
            else:
                arr[uids] = strat.moved(arr[uids], level)
            self.sim.groups.groups_changed(uids)
        return

    def force(self, uids, tr, dst, srcs=None, ti=None):
        """ Move agents who are in a state a transition leaves from (or in `srcs`) to a state, counted in the transition's flow if there is one (as when a treatment cures), with its change of level (ST30); dated at the step's start, or at `ti` """
        self.new_round()
        if tr is not None and self.moves(tr): # Their new levels, before they enter (as for a move by the transition itself)
            inside = np.logical_or.reduce([self.mod[s.name].raw[uids] for s in (tr.src if srcs is None else srcs) if s.people]) if len(uids) else np.zeros(0, dtype=bool)
            self.move_levels_of(tr, dst, uids[inside])
        moved = self.move_states(uids, tr.src if srcs is None else srcs, dst, tr=tr, ti=ti)
        if tr is None:
            return len(moved)
        self.flowcounts[tr.name] += self.n_of(moved)
        if self.sim.groups.n_groups > 1 and len(moved):
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        key = f'{tr.name}.{dst.name}'
        if key in self.flowcounts: # One destination of a transition with several
            self.flowcounts[key] += self.n_of(moved)
            if self.sim.groups.n_groups > 1 and len(moved):
                self.flow_groups[key] = self.flow_groups.get(key, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        self.forced[tr.name] = ss.uids.concatenate([self.forced[tr.name], moved]) if tr.name in self.forced else moved
        if self._started and (self.triggers or self.subscribers) and len(moved): # Onward at once, if a trigger holds for them now
            self._moved = []
            self.finish_moves(self.ti)
        return len(moved)

    def move_states(self, uids, srcs, dst, tr=None, ti=None):
        """ Move the agents who are in some states to a state: along a transition (see `force()`), or along none, as a one-off move, dated at the step's start (or at `ti`); returns those moved """
        mod = self.mod
        moved = []
        frm = []
        for state in srcs:
            if state.people:
                inds = uids[mod[state.name][uids]]
                if self.derived_trs and tr is not None:
                    self.derived_flows(tr, state, inds)
                if self.on_event and state.name in self.scheduled:
                    mod.left_exit[inds] = mod[f'ti_exit_{state.name}'].raw[inds]
                mod[state.name][inds] = False
                self.clear_exit(state.name, inds)
                moved.append(inds)
                frm.append(np.full(len(inds), self.state_index[state.name]))
        moved = ss.uids.concatenate(moved) if moved else ss.uids()
        srcs = np.concatenate(frm) if frm else -1
        self.fire(tr, moved, srcs, self.state_index.get(dst.name, -1)) # The attributes the move sets, before the dwell times that can read them
        self.enter(dst, moved, self.ti if ti is None else ti, tr=tr, srcs=srcs)
        if not self._started:
            self.moved_early = moved if self.moved_early is None else ss.uids.concatenate([self.moved_early, moved])
        if tr is None and self.stocks and self.sim.loop.index == 0: # A one-off move before the first step (e.g. ss.move() in init_post(), or after sim.init()): the start, recorded already (see init_post()), is recorded again with them
            self.record_stocks()
        return moved

    def move_some(self, srcs, dst, p=None, n=None, uids=None, where=None):
        """ A one-off move from code (`ss.move()`): a share `p` of the agents in some states (or `n` people, as the nearest whole number of agents; or of `uids`, those given; or of those for whom `where` holds), chosen at random, to a state; or of a count, to and from outside the model too (`dst` or `srcs` empty); returns the number of people moved """
        if not (srcs[0] if srcs else dst).people: # A count: an exact share, as its flows are
            have = sum(self.counts[state.name].sum() for state in srcs)
            if n is not None:
                p = min(n/have, 1.0) if have > 0 else 0.0
            moved = 0.0 if srcs else n*self.state_split[dst.name]
            for state in srcs:
                amount = self.counts[state.name]*p
                self.counts[state.name] = self.counts[state.name] - amount
                moved = moved + amount
            if dst is not None:
                self.counts[dst.name] = self.counts[dst.name] + moved
            return float(np.sum(moved))
        mod = self.mod
        self.new_round()
        cands = ss.uids.concatenate([mod[state.name].uids for state in srcs])
        if uids is not None:
            cands = cands.intersect(uids)
        if where is not None:
            cands = cands[np.asarray(self.selector.select(where), dtype=bool)[cands]]
        if n is None and np.ndim(p) == 0 and p == 1: # Everyone: no random numbers to draw
            return len(self.move_states(cands, srcs, dst))*self.scale
        draws = self.draw('move', cands)
        if n is None:
            chosen = cands[draws < p]
        else:
            k = min(int(round(n/self.scale)), len(cands))
            chosen = cands[np.sort(np.argsort(draws, kind='stable')[:k])]
        moved = self.move_states(chosen, srcs, dst)
        return len(moved)*self.scale

    @property
    def scale(self):
        """ The number of people each agent stands for """
        return self.sim.current_scale

    @property
    def total_pop(self):
        """ The number of people the agents stand for now: with dynamic rescaling (rescale=True), each agent starts as one person, so a number of people to seed (init_n=) is a number of agents """
        return self.sim.pars.total_pop*self.scale/self.sim.pars.pop_scale

    def init_post(self):
        """ Sample the agents into the states """
        mod = self.mod
        self.new_round()
        people = self.sim.people
        for name in sorted({name for leaf in self.graph.leaves() if isinstance(leaf, ss.expressions.Prop) for name in leaf.names}): # The properties of people that its rates read (ss.ppl.<name>)
            if not hasattr(people, name):
                raise sc.KeyNotFoundError(f'"{mod.name}" reads people.{name}, which is not a property of the people{ss.expressions.did_you_mean(name, list(people.states.keys()))}')
        fracs, counts = self.init_fractions()
        for name,count in counts.items():
            self.counts[name] = count*self.state_split[name]
        if fracs and self.sim.pars.branching:
            self.init_branching(fracs)
        elif fracs:
            people = self.sim.people
            uids = people.auids
            unborn = people.unborn_uids() # E.g. the fetuses of pregnancies that ss.Pregnancy starts before the start: they start in the first state, and are not seeded
            if len(unborn):
                uids = uids[people.age.raw[uids] >= 0]
            rvs = self.draw('init', uids)
            names = list(fracs.keys())
            groups = self.groups(uids) if any(np.ndim(frac) for frac in fracs.values()) else np.zeros(len(uids), dtype=int) # Fractions can differ between groups
            wanted, placed = np.zeros(len(names)), np.zeros(len(names))
            plan = {}
            for g in np.unique(groups):
                inds = uids[groups == g]
                inds = inds[np.argsort(rvs[groups == g], kind='stable')] # In a random order
                share = np.array([np.broadcast_to(fracs[name], self.sim.groups.n_groups)[g] for name in names])
                plan[g] = (inds, share*len(inds), Compartmental.round_preserving(share*len(inds))) # The number in each state is exact, as it is in a compartmental run
                wanted += share*len(inds)
                placed += plan[g][2]
            for i in np.flatnonzero((wanted > 0) & (placed == 0)): # Rounded to whole agents, a small seed would be nobody at all: one agent instead, in the group that wanted the most, taken from its largest state
                g = max(plan, key=lambda g: plan[g][1][i])
                n = plan[g][2]
                if n.max() > 1:
                    n[np.argmax(n)] -= 1
                    n[i] += 1
            for g,(inds,_,n) in plan.items():
                edges = np.cumsum(np.append(0, n)).astype(int)
                for i,name in enumerate(names):
                    chosen = inds[edges[i]:edges[i+1]]
                    mod[name][chosen] = True
                    self.stamp(name, chosen, self.seed_ti(name))
            for state in self.graph.people_states: # A share given for each person's own properties (e.g. init=dict(I=ss.by(dict(age=[0, 15], value=[0, 0.1])))): drawn for each agent of the first state
                init = state.init.raw if isinstance(state.init, ss.par) else state.init
                per_person = isinstance(init, ss.expressions.Expr) and ss.expressions.per_person_in(init) # e.g. init=dict(S=1 - (A - age)/(k*age + A)) (TY2)
                if per_person or (isinstance(init, ss.by) and init.on_people(self.sim)):
                    first = self.members(self.graph.first)
                    share = self.values_for(init, first) if per_person else init.lookup(self.sim, self.convert)[first]
                    chosen = first[self.draw(f'init_{state.name}', first) < share]
                    mod[self.graph.first.name][chosen] = False
                    mod[state.name][chosen] = True
                    self.stamp(state.name, chosen, self.seed_ti(state.name))
            seed = self.seed_dist()
            if seed is not None:
                first = self.members(self.graph.first)
                if isinstance(seed, ss.Dist):
                    chosen = seed.filter(first)
                else: # A share for each agent: from their own properties (ss.by), or a function of the sim (and the agents)
                    if not isinstance(seed, ss.by) and self.n_args(seed) > 2:
                        errormsg = f'init_prev of "{mod.name}" is a function of {self.n_args(seed)} arguments; it takes the sim, or the sim and the agents (sim, uids), and returns the share of each that starts infected'
                        raise TypeError(errormsg)
                    p = seed.lookup(self.sim, self.convert)[first] if isinstance(seed, ss.by) else seed(self.sim, first) if self.n_args(seed) == 2 else seed(self.sim)
                    p = np.asarray(p.raw if isinstance(p, ss.Arr) else p, dtype=float)
                    p = p[first] if np.ndim(p) and len(p) == len(self.mod.rel_sus.raw) and len(p) != len(first) else p # One value per agent, by UID
                    chosen = first[self.draw('seed_who', first) < p]
                self.seed(chosen, log=False)
            if len(unborn):
                mod[self.graph.first.name][unborn] = True
                self.stamp(self.graph.first.name, unborn, 0)
            if mod.infection_log is not None: # Whoever starts where a new infection goes is a seed
                infected = [name for name in names if any(state.name == name for tr in self.graph.transmissions for state in tr.dst)]
                for name in infected:
                    self.log_infections(self.members(mod[name]), seeds=True)
            final = sum((n for _,_,n in plan.values()), np.zeros(len(names)))
            for name,want,got,now in zip(names, wanted, placed, final): # Rounded to whole agents, a small seed would be none at all
                if want > 0 and got == 0:
                    start = f'starts with one agent in {name} ({self.scale:n} {"person" if self.scale == 1 else "people"})' if now else f'starts with nobody in {name}'
                    ss.note(f'"{mod.name}" {start}: its initial number is {want:.2g} agents at {self.scale:n} people per agent, which rounds to 0. For the number given, give more agents (n_agents=), or rescale=True', sim=self.sim)
        self.draw_traits(self.sim.people.auids)
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the start, once every module has its people
            self.step_register(0)
            self.sim.people.step_die() # Whoever left is gone before the start is counted
            self.sim.people.remove_dead()
        for name,tr in self.graph.transitions.items(): # A dwell time of counts is a rate out of them, as agents (see step_counts())
            dw = self.dwells.get(name)
            if not tr.people and dw is not None and dw.family != 'exponential':
                ss.note(f'"{mod.name}.{name}" moves counts ({tr.arrow}), which as agents take its dwell time ({dw.family}) as exponential with the same mean; a compartmental run keeps its shape', sim=self.sim)
        for name,tf in self.graph.transfers.items(): # Those who can move by a transfer at the start have been there since it, as if they arrived in the step before
            if name in self.tf_dists:
                self.schedule_transfer(name, self.transfer_uids(name, tf)[1], self.dt, -1)
        self.n_seen = self.sim.people.uid.len_used # Agents made after this (e.g. born of ss.Births) are put in the first state: see place_newcomers()
        self.record_stocks() # The results at the first time point are the start
        return

    def seed_dist(self):
        """
        As agents, an init_prev that chooses who starts infected one by one, as in Starsim 3: `ss.choose_n()` (a number of agents),
        `ss.bernoulli()` with a probability for each agent, a function of the sim (and the agents), or `ss.by` over a property of people that is not a dimension
        """
        init_prev = self.graph.init_prev
        if isinstance(init_prev, ss.expressions.par):
            init_prev = init_prev.raw
        trs = self.seed_transmissions() # With no transmission, the acquisition (e.g. the onset of a condition that is not transmitted)
        if not len(trs):
            return None
        elif isinstance(init_prev, ss.choose_n): # With several destinations, a number of people shared between them as a fraction is
            return init_prev if len(trs[0].dst) == 1 and len(self.seed_transmissions()) == 1 else None
        elif isinstance(init_prev, ss.bernoulli) and callable(init_prev.pars.p):
            return init_prev
        elif isinstance(init_prev, ss.by) and init_prev.on_people(self.sim):
            return init_prev
        elif callable(init_prev) and not isinstance(init_prev, (ss.Dist, ss.by)):
            return init_prev
        elif sc.isnumber(init_prev) and init_prev > 0 and len(self.seed_transmissions()) > 1: # Through several transmissions, each to those it can infect (see seed()): agent by agent
            share = init_prev if init_prev <= 1 else init_prev/self.total_pop
            return lambda sim: share
        return None

    def can_infect(self, tr, uids=None):
        """ Whether a transmission can infect each agent (or the people of each group, without `uids`): over its routes, a coefficient above 0 in a direction whose susceptible side is theirs """
        if uids is None:
            return super().can_infect(tr)
        routes = list(self.sim.networks.items())
        if not routes:
            return np.logical_or.reduce([np.broadcast_to(np.asarray(self.eval_for(coef, uids), dtype=float) > 0, len(uids)) for coef in tr.terms.values()])
        out = np.zeros(len(uids), dtype=bool)
        for coef in tr.terms.values():
            for nkey,route in routes:
                if isinstance(route, ss.Network) and route.has_acts: # A value for each direction
                    h = np.atleast_1d(np.asarray(self.route_beta(coef, nkey, route), dtype=float))
                    h = h.reshape(len(h), -1).max(axis=1)
                    if route.sides and not route.one_way:
                        side, names = route.side_raw()[uids], list(route.sides)
                        for d,key in enumerate(route.directions):
                            if h[d] > 0:
                                out |= side == names.index(key.split('2')[1])
                    elif h.max() > 0:
                        out[:] = True
                else:
                    out |= np.broadcast_to(np.asarray(self.route_beta(coef, nkey, route, uids), dtype=float) > 0, len(uids))
        return out

    def varying_dwell(self, name, tr, raw, dw, to_unit):
        """ A dwell time whose distribution has parameters with a value for each group (`ss.by`): the parameters are looked up for each agent, and one draw is made (see `Compartmental.varying_dwell()`) """
        dw.family = type(raw).__name__
        dw.staged = True
        dw.mean = dw.std = None # Each agent's: see agent_dwell_mean()
        dw.dist = ss.transitions.numeric_dist(raw, to_unit, per_agent=self.agent_values)
        return

    def agent_dwell_mean(self, name, uids):
        """ The mean of a dwell time whose distribution has parameters by group (see `varying_dwell()`), for each of some agents: worked out once for each set of values, e.g. each age group """
        trans = ss.transitions
        raw = self.dwells[name].raw
        varying = trans.varying_pars(raw)
        plain = {k:v for k,v in trans.numeric_pars(raw, self.convert, per_agent=lambda val: None).items() if k not in varying}
        vals = np.column_stack([np.broadcast_to(self.by_agent(by, uids), len(uids)) for by in varying.values()])
        rows, which = np.unique(vals, axis=0, return_inverse=True)
        means = [trans.dwell_stats(type(raw)(**plain, **dict(zip(varying.keys(), row))), lambda v: v)[0] for row in rows]
        return np.asarray(means, dtype=float)[which.ravel()]

    def add_dist(self, key, dist):
        """ A distribution the engine draws from, registered on its module so the sim initializes it """
        dist.name = f'rng_{key}'
        self.mod.setattribute(f'rng_{key}', dist)
        self.dists[key] = dist
        return

    def init_dwell_mods(self):
        """ See `Engine.init_dwell_mods()`; and the distributions each set= draws from, before the sim initializes the distributions """
        super().init_dwell_mods()
        for dw in self.dwells.values():
            for new in getattr(dw, 'sets', {}).values():
                self.add_dist(new.key, new.dist)
        return

    def infect_now(self, uids, strain=None):
        """ Infect some agents now, from code (`ss.Disease.infect()`): those in the first state, along its first transmission (with the strain given), counted in its flow, dated as infections are (at the step's index, its end), and logged as seeds """
        mod = self.mod
        self.new_round()
        uids = uids[mod[self.graph.first.name][uids]]
        tr = next((tr for tr in self.graph.transmissions if not getattr(tr, 'generated', False)), None)
        if tr is None:
            raise ss.CapabilityError(f'"{mod.name}" has no transmission for disease.infect() to infect people along')
        if strain is not None:
            if self.strain is None:
                raise ss.CapabilityError(f'"{mod.name}" has no strains in shared states, so disease.infect() takes no strain=')
            strains = self.sim.groups.strats[self.strain].strains
            self.set_strain(uids, np.full(len(uids), strains.level_of([strains.names.index(getattr(strain, 'name', strain))])))
        which = np.zeros(len(uids), dtype=int)
        if len(tr.dst) > 1: # Where a new infection goes, in its shares
            self._uids = uids
            ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
            which = np.minimum((np.cumsum(ps, axis=0) <= self.draw('seed_where', uids)[None, :]).sum(axis=0), len(tr.dst) - 1)
        before = dict(self.flowcounts)
        for i,dst in enumerate(tr.dst):
            self.force(uids[which == i], tr, dst, ti=mod.ti)
        if getattr(self, 'written_ti', None) == mod.ti: # Between steps (e.g. after sim.run(until=...)), once this step's results are written: the flow is added to them, as it would be lost when the next step starts counting
            for key,val in self.flowcounts.items():
                if val != before[key]:
                    mod.results[key][mod.ti] += val - before[key]
        if mod.infection_log is not None:
            self.log_infections(uids, seeds=True)
        return uids

    def seed_ti(self, name):
        """
        The time of entering a state for someone put in it at the start: as if they had entered it in the step before the
        first, so that a seed with a dwell of d steps is a source for d steps, as anyone who enters later is (HP62).
        Contacts (a network, a mixing pool, or `beta*I` well mixed) are made after the step's state changes, and a rate
        (`beta*I.n`) reads the people at its start, so a source by contact entered at 0 (in the first step's contacts,
        leaving at the start of the second), and anyone else at -1 if they are where an infection puts people (in the first
        step's rates, then counted at the start only, as under the ODE: CR4), else at 0
        """
        state = self.graph.states[name]
        counted = any(s is state for tr in self.graph.transitions.values() if tr.rate is not None
                      for leaf in tr.rate.leaves if isinstance(leaf, ss.expressions.Count) for s in leaf.states) # This module's own state, not another's of the same name (syph.I.n)
        if state.infectious and not counted:
            return 0
        acquired = any(dst.name == name for tr in self.graph.acquisitions for dst in tr.dst)
        return -1 if acquired or counted else 0

    def seed(self, uids, log=True):
        """
        Seed some agents who are in the first state: put them where a new infection puts them, the destinations of the first
        transmission in its shares, as init_prev does (and the strains with courses of their own, for each); returns those seeded

        Args:
            uids (uids): the agents; those not in the first state are left where they are
            log (bool): whether to record them as seeds in an `ss.infection_log`
        """
        mod = self.mod
        self.new_round()
        first = self.graph.first.name
        uids = ss.uids(uids)
        uids = uids[mod[first][uids]]
        trs = self.seed_transmissions()
        if not trs:
            errormsg = f'"{mod.name}" has no transmission or acquisition to seed through (init_prev=); use init= on a state instead'
            raise ValueError(errormsg)
        if len(trs) == 1 and len(trs[0].dst) == 1:
            dsts = [(trs[0].dst[0], uids)]
        else: # Each seed to a destination of a transmission that can infect them, in the shares of its destinations
            self._uids = uids
            ps = [np.broadcast_to(w*p, len(uids)) for tr,w in zip(trs, self.seed_reach(trs, uids)) for p in (self.branch_probs_for(tr, uids) if len(tr.dst) > 1 else [1.0])]
            edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
            rvs = self.draw('seed_where', uids)
            dsts = [(dst, uids[(rvs >= edges[i]) & (rvs < edges[i+1])]) for i,dst in enumerate(dst for tr in trs for dst in tr.dst)]
        mod[first][uids] = False
        self.clear_exit(first, uids)
        for dst,inds in dsts:
            mod[dst.name][inds] = True
            self.stamp(dst.name, inds, self.seed_ti(dst.name))
        if log and mod.infection_log is not None:
            self.log_infections(uids, seeds=True)
        return uids

    def put_first(self, uids):
        """ Put some agents in the first state, and in no other (e.g. fetuses made after the start, whose states would otherwise be their arrays' defaults) """
        if not len(uids) or not self.graph.people_states:
            return
        mod = self.mod
        for state in self.graph.people_states:
            mod[state.name][uids] = False
            self.clear_exit(state.name, uids)
        first = self.graph.first.name
        mod[first][uids] = True
        self.stamp(first, uids, self.ti)
        return

    def place_newcomers(self):
        """ Put agents made by other modules since the last step (e.g. born of `ss.Births` or `ss.Pregnancy`), who are in no state, in the first state """
        n = self.sim.people.uid.len_used
        if n == self.n_seen or not self.graph.people_states:
            return
        mod = self.mod
        new = ss.uids(np.arange(self.n_seen, n))
        for start,stop in self.sim.people.untraited: # Made by code that set their properties after making them: their traits now (see ss.People.grow())
            later = new[(new >= start) & (new < stop)]
            later = later[self.sim.people.alive.raw[later]]
            if len(later):
                self.draw_traits(later)
        self.n_seen = n
        if self.sim.people.untraited: # Forget the ranges every declared module has seen
            seen = min(getattr(m.engine, 'n_seen', 0) for m in self.sim.declared)
            self.sim.people.untraited = [r for r in self.sim.people.untraited if r[1] > seen]
        placed = np.zeros(len(new), dtype=bool)
        for state in self.graph.people_states:
            placed |= mod[state.name].raw[new]
        new = new[~placed & self.sim.people.alive.raw[new]]
        if len(new):
            if self.sim.groups.n_groups > 1:
                self.sim.groups.groups_changed(new) # Before they enter the state, so they are counted in their groups: see wrote()
            first = self.graph.first.name
            mod[first][new] = True
            self.stamp(first, new, self.ti)
            self.fire(None, new, -1, self.state_index[first]) # Placement is entering the state (spec 03)
        return

    #%% The context used to evaluate expressions

    def stock(self, state):
        if state.people:
            counts = self.counted(state)
            if self.sim.groups.n_groups > 1:
                self.stock_groups[state.name] = counts
            return float(counts.sum())
        if self.sim.groups.n_groups > 1:
            self.stock_groups[state.name] = self.counts[state.name].copy()
        return self.counts[state.name].sum()

    #%% Counting the agents in each state

    def census(self, want=()):
        """
        Count the agents in each state of people, by group, and list those in some of the states: one pass over the agents (threaded for a large population)

        The counts are then kept up to date as agents enter and leave the states, whatever moves them (see
        `wrote()`), so they are counted again only when their groups change (see `regroup()`).

        Args:
            want (list): the names of the states whose agents to list

        Returns a dict of the agents in each state in `want`.
        """
        sim = self.sim
        names = [state.name for state in self.graph.people_states]
        ng = sim.groups.n_groups
        groups = sim.groups.group_raw() if ng > 1 else np.zeros(0, dtype=np.int64)
        order = {name:k for k,name in enumerate(want)}
        listed = np.array([order.get(name, -1) for name in names], dtype=np.int64)
        members = tuple(self.mod[name].raw for name in names)
        counts, found, offsets = ss.solvers.backends.passes.census(members, sim.people.alive.raw, groups, ng, listed)
        self.live = dict(zip(names, counts[:-1]))
        self.live_total = counts[-1] # Everyone alive
        self.n_counted = sim.people.uid.len_used
        return {name:ss.uids(found[offsets[k]:offsets[k+1]]) for name,k in order.items()}

    def wrote(self, name, entered, left):
        """ Some agents have been put in a state of people, or taken out of it, by anything (see `ss.BoolState`): keep its count by group (each counted once) """
        if not len(entered) and not len(left):
            return
        unique = lambda u: u if len(u) < 2 or (u[1:] > u[:-1]).all() else np.unique(u)
        entered, left = unique(entered), unique(left)
        if len(entered) and self.unscheduled is not None:
            self.unscheduled.add(name) # Put in it other than by the engine's moves, so perhaps with no exit scheduled: see step_state()
        ng = self.sim.groups.n_groups
        if ng == 1:
            self.live[name] = self.live[name] + (len(entered) - len(left))
        else:
            groups = self.sim.groups.group_raw()
            self.live[name] = self.live[name] + np.bincount(groups[entered], minlength=ng) - np.bincount(groups[left], minlength=ng)
        return

    def regroup(self, uids=None):
        """ The groups of some agents (or of anyone, if uids is None) have changed: if any of them are in a state, the counts by group are taken again when next needed """
        if self.live is None:
            return
        if uids is not None:
            uids = np.asarray(uids, dtype=np.int64)
            if not any(self.mod[state.name].raw[uids].any() for state in self.graph.people_states):
                return # New agents, who are not in a state yet
        self.live = None
        self.unscheduled = None # Writes are not watched while nothing is counted
        self.live_total = None
        return

    def moved_groups(self, uids, before, after):
        """ Some agents' groups have changed by themselves (e.g. they reached the next age group: see `ss.dimensions.Groups.regrouped()`): move them between the counts kept by group """
        if self.live is None:
            return
        ng = self.sim.groups.n_groups
        mod = self.mod
        shift = lambda inside: np.bincount(after[inside], minlength=ng) - np.bincount(before[inside], minlength=ng)
        for state in self.graph.people_states:
            inside = mod[state.name].raw[uids]
            if inside.any():
                self.live[state.name] = self.live[state.name] + shift(inside)
        counted = self.sim.people.alive.raw[uids] & (uids < self.n_counted) # Those made since are counted by alive_counts()
        if counted.any():
            self.live_total = self.live_total + shift(counted)
        return

    def live_counts(self):
        """ The number of agents in each state of people now, by group (not collapsed): kept from the last census (with any changes of group since: see `moved_groups()`), or counted again """
        self.sim.groups.group_raw()
        if self.live is None:
            self.census()
        return self.live

    def alive_counts(self):
        """ The number of agents alive now, by group: kept from the last census, with those made since """
        self.live_counts()
        people = self.sim.people
        n = people.uid.len_used
        if n > self.n_counted:
            new = np.arange(self.n_counted, n)
            new = new[people.alive.raw[new]]
            self.live_total = self.live_total + (np.bincount(self.sim.groups.group_raw()[new], minlength=self.sim.groups.n_groups) if self.sim.groups.n_groups > 1 else len(new))
            self.n_counted = n
        return self.live_total

    def check_counts(self):
        """ Check the counts kept since the last census against a new one (with `STARSIM_CHECK_COUNTS=1`, at every step; for testing) """
        kept = {name:vals.copy() for name,vals in self.live.items()}
        total = self.alive_counts().copy()
        self.census()
        for name,vals in self.live.items():
            if not np.array_equal(kept[name], vals):
                raise RuntimeError(f'The count of "{self.mod.name}.{name}" kept by group, {kept[name].sum()}, differs from a new count, {vals.sum()}')
        if not np.array_equal(total, self.live_total):
            raise RuntimeError(f'The number alive kept by group for "{self.mod.name}", {total.sum()}, differs from a new count, {self.live_total.sum()}')
        return

    def counted(self, state):
        """ The number in a state of people now, by the group it is counted in (see `stock()`) """
        vals = self.live_counts()[state.name]
        if self.sim.groups.n_groups == 1:
            return vals.astype(float)
        return np.bincount(self.collapse[state.name], weights=vals, minlength=self.sim.groups.n_groups)

    def weighted_count(self, state):
        """ As `counted()`, with each agent counted by its weight (`people.scale`) """
        uids = self.mod[state.name].uids
        if self.sim.groups.n_groups == 1:
            return np.array([self.sim.people.scale.raw[uids].sum()], dtype=float)
        return np.bincount(self.groups(uids, state), weights=self.sim.people.scale.raw[uids], minlength=self.sim.groups.n_groups)

    def by_place(self, vals):
        """ Numbers by group as numbers by place, or None if a group does not say where its people are (when it leaves out the dimension of places) """
        sim = self.sim
        if sim.groups.group_fold is not None:
            return None
        return np.bincount(sim.groups.group_coords[:, sim.mixing.space], weights=vals, minlength=sim.groups.strats[sim.mixing.space].n)

    def record_stocks(self):
        """ As for the other engines (see `Engine.record_stocks()`), with the number in each state of people as it has been kept since the last census (see `census()`) """
        people = self.graph.people_states
        counts = {state.name:(self.counted(state) if not self.sim.people.scale.weighted else self.weighted_count(state)) for state in people}
        if self.held is not None: # The uninfected of a branching process, as counts
            counts.update({name:np.broadcast_to(held, self.sim.groups.n_groups).copy() for name,held in self.held.items()})
        for state in self.graph.states.values():
            if state.name in counts:
                if self.sim.groups.n_groups > 1:
                    self.stock_groups[state.name] = counts[state.name]
                self.stocks[state.name] = float(counts[state.name].sum())
            else:
                self.stocks[state.name] = self.stock(state)
        if getattr(self, 'derived_res', None) and self._started: # Once the people placed at the start have their attributes: see settle_start()
            self.derived_stocks()
        return

    def count(self, states, total=True):
        out = 0.0
        weighted = self.sim.people.scale.weighted # Agents of different weights are each counted by their weight, so a rate means the same whatever the weights
        counts = self.start if self._members is not None else self.live # At the start of the step, while its rates are worked out (see step_state()), or now
        if weighted:
            counts = self.start_weighted if self._members is not None else None
        for state in states:
            if self.held is not None and state.name in self.held: # The uninfected of a branching process
                out += self.held[state.name].sum() if total else self.held[state.name]
            elif state.people:
                if counts is not None:
                    n = counts[state.name].sum()
                elif weighted:
                    n = self.weighted_count(state).sum()
                else:
                    n = np.count_nonzero(self.mod[state.name])
                out += n*self.scale
            else:
                out += self.counts[state.name].sum() if total else self.counts[state.name]
        return out

    def set_count(self, state, value):
        """ Set the number in a state that is not people, keeping its split between groups """
        current = self.counts[state.name]
        total = current.sum()
        self.counts[state.name] = current*(value/total) if total > 0 else value*self.state_split[state.name]
        return

    def n_people(self, total=True):
        return self.count(self.graph.people_states)

    def exposure(self, state):
        """ Under homogeneous mixing: the infectiousness-weighted fraction of people in the state """
        mod = self.mod
        uids = self.members(state)
        share = 1.0
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            share = self.strain_weights[self.strain_raw()[uids], self._strain]
            uids, share = uids[share > 0], share[share > 0]
        infectious = mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans*share
        mixing = self.sim.mixing
        people = self.sim.people
        weighted = people.scale.weighted
        if weighted: # Each agent counted by its weight, as in count()
            infectious = infectious*people.scale.raw[uids]
        if mixing.space is None:
            if weighted:
                total = people.scale.raw[people.auids].sum()
                return infectious.sum()/total if total > 0 else 0.0
            n = len(people) if self.held is None else self.held_n().sum() # In a branching process, the people a rate divides by are the uninfected
            return infectious.sum()/n if n > 0 else 0.0
        places = self.place_raw() # People mix within their own place, so there is one value per agent
        n_places = self.sim.groups.strats[mixing.space].n
        if 'place_sizes' not in self.cache:
            sizes = self.by_place(self.alive_counts() if self.held is None else self.held_n()) if not weighted else None
            self.cache['place_sizes'] = sizes if sizes is not None else np.bincount(places[people.auids], weights=people.scale.raw[people.auids] if weighted else None, minlength=n_places).astype(float)
        num = np.bincount(places[uids], weights=infectious, minlength=n_places)
        prev = mixing.by_place(np.arange(n_places), num, self.cache['place_sizes'], n_places)
        return prev[mixing.places] if self._by_group else prev[places[self._uids]] # For each group, or for the agents the rate is being worked out for

    def place_raw(self):
        """ The place of each agent, as an array indexed by UID """
        return self.sim.groups.level_raw(self.sim.mixing.space)

    def local_count(self, states):
        """ In a model with places: the number in the same place as each agent, or, when the counts are being advanced, as each group """
        mixing = self.sim.mixing
        if mixing.space is None:
            return self.count(states)
        n_places = self.sim.groups.strats[mixing.space].n
        places = self.place_raw()
        out = np.zeros(n_places)
        members = self._members
        counts = self.start if members is not None else self.live_counts() # At the start of the step, while its rates are worked out, or now
        weights = self.sim.people.scale.raw if self.sim.people.scale.weighted else None # Agents of different weights: each counted by its weight, as in count()
        for state in states:
            if state.people:
                placed = self.by_place(counts[state.name]) if weights is None else None
                if placed is None: # Counted from the agents themselves (who were all listed by the census, if this is the start of the step; a state that is not listed has had nobody leave it yet)
                    uids = members[state.name] if members is not None and state.name in members else self.members(state)
                    placed = np.bincount(places[uids], weights=None if weights is None else weights[uids], minlength=n_places)
                out += placed*self.scale
            else:
                out += np.bincount(mixing.places, weights=self.counts[state.name], minlength=n_places)
        return out[mixing.places] if self._by_group else self.for_uids(out[places]) # For each group, or for the agents the rate is being worked out for, so it combines with a value by person (e.g. ss.by(patch=...))

    def count_weighted(self, states, expr):
        """ A weighted count (`(ss.by(biting=w)*I).n`, `A.n_trans`): each agent in the states counted by its weight (and infectiousness), in the same place, as `local_count()` gives it """
        sim = self.sim
        mixing = sim.mixing
        people = sim.people
        places = self.place_raw() if mixing.space is not None else None
        n_places = sim.groups.strats[mixing.space].n if places is not None else 1
        members = self._members
        out = np.zeros(n_places)
        for state in states:
            uids = members[state.name] if members is not None and state.name in members else self.members(state)
            if not len(uids):
                continue
            w = np.ones(len(uids))
            if expr.weights is not None and state in expr.weights: # Per agent, even while the counts are advanced by group
                with self.evaluating(uids=uids, for_uids=True, by_group=False):
                    vals = np.asarray(self.eval(expr.weights[state]), dtype=float)
                w = w*(vals[uids] if vals.ndim and len(vals) != len(uids) else np.broadcast_to(vals, len(uids)))
            if expr.trans: # Infectiousness, as in exposure()
                w = w*self.mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans
            if people.scale.weighted:
                w = w*people.scale.raw[uids]
            out += np.bincount(places[uids], weights=w, minlength=n_places) if places is not None else w.sum()
        out = out*self.scale
        if places is None:
            return float(out[0])
        return out[mixing.places] if self._by_group else self.for_uids(out[places])

    def coverage(self, expr):
        """ A product's coverage (`itn.coverage`): the share of the agents in the same place who hold it, each by the protection they have left (or a function of the time since delivery) """
        sim = self.sim
        people = sim.people
        mixing = sim.mixing
        product = self.product_of(expr)
        alive = np.asarray(people.alive.raw, dtype=bool)
        held = (np.asarray(people[product.dim].raw) > 0) & alive
        fn = expr.weight if expr.weight is not None else product.waning if callable(product.waning) else None
        w = product.remaining(fn, held, self) if fn is not None else held*1.0
        n = alive*1.0
        if people.scale.weighted:
            w, n = w*people.scale.raw, n*people.scale.raw
        if mixing.space is None:
            return w.sum()/max(n.sum(), 1e-12)
        places = self.place_raw()
        n_places = sim.groups.strats[mixing.space].n
        auids = people.auids
        share = np.bincount(places[auids], weights=w[auids], minlength=n_places)/np.maximum(np.bincount(places[auids], weights=n[auids], minlength=n_places), 1e-12)
        return share[mixing.places] if self._by_group else self.for_uids(share[places])

    def note_counter(self, strat, what):
        """ Agents add to each person's own value, so a counter's levels need not be one apart (see `Compartmental.note_counter()`) """
        return

    def call_uids(self, func, p=None):
        return func(self.sim, self._uids)

    def weighted_events(self, key, weight, integer):
        """ An outcome weighted per person (e.g. years of life lost by each person's age): the weight of each person who went through the transition this step, summed by group """
        if key not in self.graph.transitions:
            return None
        ng = self.sim.groups.n_groups
        uids = self.events.get(key, ss.uids())
        w = np.broadcast_to(np.asarray(self.eval_for(weight, uids), dtype=float), len(uids)) if len(uids) else np.zeros(0)
        if integer and np.all((0 <= w) & (w <= 1)): # A fraction of whole events is a draw, as for a weight by group
            w = (self.rng.random(len(uids)) < w).astype(float)
        people = self.weights(uids)
        if people is not None:
            w = w*people
        return np.array([w.sum()]) if ng == 1 else np.bincount(self.groups(uids), weights=w, minlength=ng).astype(float)

    def call(self, func, p=None):
        """ As `Engine.call()`; a function of the sim may give one value per living agent (as `sim.people.age` does), which is put in place by UID (TL8: it was indexed by UID, so crashed once anyone had died) """
        out = super().call(func, p)
        if isinstance(out, ss.Arr):
            out = out.raw
        if np.ndim(out) != 1 or isinstance(out, ss.uids) or isinstance(func, ss.parameters.TimeValue) or self.n_args(func) >= 2:
            return out
        people = self.sim.people
        n, alive = people.uid.len_tot, people.auids
        if len(out) == n or len(out) == self.sim.groups.n_groups:
            return out
        elif len(out) == len(alive):
            key = ('by_uid', id(func))
            if key not in self.cache:
                out = np.asarray(out)
                full = np.full(n, np.nan) if out.dtype.kind == 'f' else np.zeros(n, dtype=out.dtype)
                full[alive] = out
                self.cache[key] = full
            return self.cache[key]
        name = getattr(func, '__name__', 'function')
        errormsg = f'the function "{name}" gave {len(out)} values, but there are {len(alive)} people alive ({n} agents in all, by UID); give one value per person alive (e.g. from sim.people.age), one per agent by UID (from .raw), or one number'
        raise ValueError(errormsg)

    def by_group(self, by):
        """ A value for each level of a dimension becomes a value for each of the agents being evaluated (or for each group, when rates are being worked out by group) """
        if not self._by_group:
            return self.by_agent(by, self._uids)
        return super().by_group(by)

    def by_agent(self, by, uids):
        """ A value for each level of a dimension, for some agents; sex needs no stratification, since it is people.female (the first level is women) """
        sim = self.sim
        strat = self.derived_strat(by)
        if strat is not None: # By the level of an attribute
            raw = by.values
            vals = self.convert(raw) if isinstance(raw, ss.TimePar) else [self.convert(v(sim) if isinstance(v, ss.parameters.TimeValue) else v) for v in sc.tolist(raw)]
            vals = np.asarray(vals, dtype=float).ravel()
            if len(vals) != strat.n:
                raise ValueError(f'ss.by({strat.name}=...) has {len(vals)} values, but "{strat.name}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
            return vals[self.derived_level(strat, uids)]
        if by.on_people(sim): # Read from each person's own properties, e.g. ss.by(sex=dict(female=..., male=...))
            return self.by_people(by, uids)
        if by.dim in ['sex', 'female'] and by.dim not in [strat.name for strat in sim.groups.strats]:
            vals = [self.convert(v(sim) if isinstance(v, ss.parameters.TimeValue) else v) for v in sc.tolist(by.values)]
            if len(vals) != 2:
                raise ValueError(f'ss.by({by.dim}=...) has {len(vals)} values; sex has two levels, women and then men.')
            return np.where(sim.people.female.raw[uids], vals[0], vals[1])
        return np.asarray(super().by_group(by))[self.groups(slice(None) if uids is None else uids)] # With no agents being evaluated, for everyone, by UID

    def agent_values(self, by):
        """ A parameter of a distribution with a value for each group, as a function the distribution calls for the agents it draws for """
        def values(sim, uids):
            return self.by_agent(by, uids) if len(uids) else np.ones(0) # None, when the distribution is initialized
        values.by = by # So the engine can tell which values it can work out for everyone at once: see agent_pars()
        return values

    def by_people(self, by, uids=None):
        """ A value read from each person's own properties (see `ss.by.lookup()`), for the agents being evaluated (or for `uids`): looked up for those agents alone, since often few are asked for (e.g. those entering a state), or, for a value by sex, from a table kept for the run """
        if self._by_group and uids is None:
            errormsg = f'{by} is read from each person, so cannot be used where a value is worked out for groups rather than agents (a count that is not people, an inflow, or init_prev)'
            raise ValueError(errormsg)
        if uids is not None and not self.sim.people.initialized: # A distribution's parameter, tried when the distribution is made, before there are people
            return np.ones(np.size(uids))
        uids = self._uids if uids is None else uids
        if self.is_lifelong(by):
            vals = self.lifelong_table(id(by), lambda new: by.lookup(self.sim, self.convert, year=self.year, cache=self.cache, uids=new))
            return vals.copy() if uids is None else vals[uids] # Everyone, or these agents
        return by.lookup(self.sim, self.convert, year=self.year, cache=self.cache, uids=uids)

    def is_lifelong(self, by):
        """ Whether a value by people's properties is the same for each person for the whole run: by sex alone, with values that do not change in time """
        if by.table is not None or not all(dim in ['sex', 'female'] for dim in by.dims):
            return False
        vals = [v for given in by.given.values() for v in (given.values() if isinstance(given, dict) else sc.tolist(given))]
        return not any(isinstance(v, ss.parameters.TimeValue) or (callable(v) and not isinstance(v, ss.TimePar)) for v in vals)

    def is_fixed(self, x):
        """
        Whether a value (an expression of parameters, or a condition) is the same for each person for the whole run, so it can be worked out for
        everyone once (see fixed_edges()): numbers, parameters with no modifiers whose values are numbers or values by sex (see is_lifelong()), and
        conditions on sex
        """
        E = ss.expressions
        for part in E.nodes(x):
            if part is None or isinstance(part, (int, float, np.number, E.BinOp, E.Prop)):
                continue
            elif isinstance(part, E.par):
                v = part.value
                if part.name and not part.literal and self.sim.modifiers.get((self.mod.name, 'par', part.name)):
                    return False
                if isinstance(v, ss.by):
                    if not self.is_lifelong(v):
                        return False
                elif not isinstance(v, (int, float, np.number, E.Expr, ss.conditions.Cond)):
                    return False
            elif isinstance(part, ss.conditions.Cond):
                if part.op in ss.conditions.leaves and not (part.op == 'people' and tuple(part.args) in [('female',), ('male',)]):
                    return False
            elif not (isinstance(part, ss.by) and self.is_lifelong(part)):
                return False
        return True

    def fixed_edges(self, key, fixed, edges, uids):
        """
        Cumulative probabilities (one row for each, then each agent: see dest_edges() and branch_edges()) for some agents: if they depend only
        on fixed values (`fixed`, see is_fixed()), from a table for everyone kept for the run, else from `edges(uids)`
        """
        if not fixed:
            return edges(uids)
        return self.lifelong_table(key, lambda new: edges(new).T)[uids].T

    def unmodified(self, kind, name):
        """ Whether a branch (or one destination of a transition) has no modifiers and no adjust= of the stratifications """
        return not self.sim.modifiers.get((self.mod.name, kind, name)) and self.branch_adjust.get(name) is None

    def lifelong_table(self, key, values):
        """ Values for everyone, by UID, kept for the run (see is_lifelong()), from `values(uids)`; those for agents added since are worked out as they are needed """
        vals = self.lifelong.get(key)
        n = len(self.sim.people.uid.raw)
        if vals is None or len(vals) < n:
            start = 0 if vals is None else len(vals)
            new = values(ss.uids(np.arange(start, n)))
            vals = self.lifelong[key] = new if vals is None else np.concatenate([vals, new])
        return vals

    def value(self, p):
        if self._drawing and id(p.raw) in self.attr_dists: # A distribution an attribute is drawn from: see draw_attr()
            return np.asarray(self.draw(self.attr_dists[id(p.raw)], self._uids), dtype=float)
        if isinstance(p.raw, ss.by) and self.derived_strat(p.raw) is not None: # By the level of a derived attribute, for the agents being evaluated
            return self.by_agent(p.raw, self._uids)
        if isinstance(p.raw, ss.by) and self._route is None: # Depends on which agents are being evaluated, so is not cached (on a route, a beta is worked out by group: see Engine.route_value())
            return self.by_people(p.raw) if p.raw.on_people(self.sim) else self.by_group(p.raw)
        if isinstance(p.raw, ss.parameters.by_func) and self._route is None: # From a function of the sim, one value per level (MS2): as a value by level
            out = self.by_levels(p.raw)
            return self.by_group(out) if isinstance(out, ss.by) else out
        key = id(p)
        if key in self.spread: # Each agent's own value, as an array indexed by UID
            if key not in self.cache:
                self.cache[key] = self.modified(p, self.mod[self.spread[key]].raw)
            return self.for_uids(self.cache[key])
        return self.for_uids(super().value(p))

    def prop(self, cond):
        """ A property of people or a condition, in a rate (see `ss.expressions.Prop`): its value for each agent being evaluated, as it was when first asked for in the step """
        if self._by_group:
            errormsg = f'{cond} is a property of each person, so cannot be used where a value is worked out for groups rather than agents (a count that is not people, an inflow, or init_prev)'
            raise ValueError(errormsg)
        key = ('prop', id(cond))
        n = self.sim.people.n_uids
        old, n_old = self.cache.get(key, (None, 0))
        if old is None or (np.ndim(old) and n_old < n): # Or people born since, who are added to it
            out = np.asarray(cond.eval(self.selector), dtype=float)
            if np.ndim(old):
                out[:n_old] = old[:n_old]
            self.cache[key] = (out, n)
        out = self.cache[key][0]
        return out if np.ndim(out) == 0 else out[self._uids]

    def for_uids(self, val):
        """ Within `eval_for()`, a value for each agent in the population (e.g. a parameter modified for some people) is taken for the agents being evaluated, so it combines with the rest of the expression """
        if not getattr(self, '_for_uids', False) or np.ndim(val) == 0 or len(val) == len(self._uids):
            return val
        return np.asarray(val)[self._uids]

    def eval_for(self, expr, uids):
        """ Evaluate an expression for some agents: a scalar or one value per agent """
        with self.evaluating(uids=uids, for_uids=True):
            out = self.eval(expr)
        if np.ndim(out) and len(out) != len(uids): # One value per agent in the population
            out = np.asarray(out)[uids]
        return out

    #%% Expressions, compiled once

    def __getstate__(self):
        """ Copied or saved without the compiled expressions (functions that hold this engine), which are made again when first used """
        state = self.__dict__.copy()
        state['compiled'] = dict()
        state['_memo'] = None
        return state

    def eval(self, expr):
        """
        Evaluate an expression, as `ss.expressions.evaluate()` does, from a function of it made the first time it is
        evaluated (see `compile()`), so that each step does only the arithmetic and reads the values
        """
        entry = self.compiled.get(id(expr))
        if entry is None or entry[0] is not expr: # Kept with the expression, so its id is not taken by another
            if len(self.compiled) > 10_000: # Expressions made as the sim runs (e.g. by a function of the model's states) are let go of now and then
                self.compiled.clear()
            entry = self.compiled[id(expr)] = (expr, self.compile(expr))
        return entry[1]()

    def compile(self, expr):
        """
        An expression as a function of no arguments that evaluates it for the agents being evaluated, with what to do at each
        part decided once, rather than at each step. It gives the same numbers as `ss.expressions.evaluate()`, and:

        - a part that has one value for each group (made of numbers, values in time, values by the sim's dimensions, and counts:
          see `groupwise()`) is worked out for the groups, and then taken for each agent from their group, which is much less
          arithmetic than for each agent when there are many agents and few groups (e.g. a rate by place that reads counts by place);
        - a product with a condition (e.g. `infant_rate*((ss.ppl.age >= 0) & (ss.ppl.age < 1))`) is worked out only for the agents
          the condition holds for (0 for the others), so a function rate in it is called only for them (see `gated()`);
        - a part evaluated again for the same agents while `self._memo` is kept (e.g. the factor that the terms of a transmission share)
          is taken from it.
        """
        E = ss.expressions
        if isinstance(expr, E.BinOp):
            if expr.op == '*' and self.gated(expr):
                return self.compile_gated(expr)
            if self.groupwise(expr):
                return self.compile_grouped(expr)
            return self.memoized(expr, self.compile_op(expr))
        elif isinstance(expr, E.Func):
            if self.groupwise(expr):
                return self.compile_grouped(expr)
            return self.memoized(expr, self.compile_op(expr))
        return self.compile_leaf(expr)

    def compile_op(self, expr, compile=None):
        """ Arithmetic or a NumPy function of the parts of an expression, each compiled with `compile` (by default `compile()`) """
        E = ss.expressions
        compile = compile or self.compile
        if isinstance(expr, E.Func):
            args = [compile(arg) for arg in expr.args]
            if expr.name == 'expit': # The logistic function, 1/(1 + exp(-x)), which NumPy does not have
                return lambda: 1/(1 + np.exp(-args[0]()))
            func = getattr(np, expr.name)
            if len(args) == 1:
                arg = args[0]
                return lambda: 1.0*func(arg()) # A comparison is 1 or 0
            a, b = args
            return lambda: 1.0*func(a(), b())
        a, b = compile(expr.a), compile(expr.b)
        op = expr.op
        if   op == '+': return lambda: a() + b()
        elif op == '-': return lambda: a() - b()
        elif op == '*': return lambda: a() * b()
        elif op == '/': return lambda: E.divide(a(), b())
        else:           return lambda: power(a(), b())

    def compile_leaf(self, expr):
        """ A part of an expression with no arithmetic in it: a parameter, a count, a property of people, ... """
        E = ss.expressions
        if isinstance(expr, E.par):
            return lambda: self.value(expr)
        elif isinstance(expr, E.Count):
            return lambda: self.count_expr(expr)
        elif isinstance(expr, ss.transitions.State):
            return lambda: self.exposure(expr)
        elif isinstance(expr, E.Prop):
            return lambda: self.prop(expr.cond)
        elif isinstance(expr, ss.conditions.Cond):
            return lambda: self.prop(expr)
        elif isinstance(expr, E.Clock):
            return lambda: self.clock(expr)
        elif isinstance(expr, E.attr):
            return lambda: self.attr(expr)
        return lambda: ss.expressions.evaluate(expr, self) # Anything else (a share of a row, a mother's value) as it evaluates itself

    def memoized(self, expr, func):
        """ A compiled part of an expression whose value is kept in `self._memo`, while it is kept, for the same agents and the same route and strain """
        key = id(expr)
        def memoized():
            memo = self._memo
            if memo is None:
                return func()
            kept = memo.get(key)
            if kept is not None and kept[0] is self._uids and kept[1] == (self._by_group, self._route, self._strain):
                return kept[2]
            out = func()
            memo[key] = (self._uids, (self._by_group, self._route, self._strain), out) # The agents are kept with it, so their id is not taken by others
            return out
        return memoized

    def groupwise(self, expr):
        """
        Whether a part of an expression has one value for each group (or for everyone): made only of numbers, values in time, values by
        the sim's dimensions (`ss.by`, not read from each person), and counts and exposures of this module's own states. Not a property of
        people, an attribute, a clock, a function (which may give a value for each agent), or a parameter spread across people.
        """
        E = ss.expressions
        if isinstance(expr, E.BinOp):
            return self.groupwise(expr.a) and self.groupwise(expr.b)
        elif isinstance(expr, E.Func):
            return all(self.groupwise(arg) for arg in expr.args)
        elif isinstance(expr, E.Count):
            return not self.split_states(expr.states)[1] # Another module's states are counted by its engine
        elif isinstance(expr, ss.transitions.State):
            return True
        elif isinstance(expr, E.par):
            raw = expr.raw
            if id(expr) in self.spread or isinstance(raw, (dict, ss.Dist, ss.peract)):
                return False
            elif isinstance(raw, E.Expr):
                return self.groupwise(raw)
            elif isinstance(raw, ss.by):
                return type(raw) is ss.by and not raw.on_people(self.sim) and self.derived_strat(raw) is None
            elif isinstance(raw, ss.parameters.TimeValue):
                return True
            return not callable(raw)
        return False

    def compile_grouped(self, expr):
        """
        A part of an expression with one value for each group (see `groupwise()`), worked out for the groups and then taken for each
        agent being evaluated from their group: the same numbers, since each agent's is the same arithmetic on the same values. Worked out
        for the agents instead where a group's value is not what an agent's is (on a route, while drawing, or with modifiers of its parameters)
        """
        E = ss.expressions
        plain = self.memoized(expr, self.compile_op(expr, compile=self.compile_plain))
        leaves = expr.leaves
        if not any(isinstance(leaf, (E.Count, ss.transitions.State)) or (isinstance(leaf, E.par) and isinstance(leaf.raw, ss.by)) for leaf in leaves):
            return plain # Numbers for everyone, so nothing is gained
        names = [leaf.name for leaf in leaves if isinstance(leaf, E.par) and leaf.name and not leaf.literal]
        modname = self.mod.name
        def grouped():
            if self._by_group or not self._for_uids or self._route is not None or self._drawing:
                return plain()
            modifiers = self.sim.modifiers
            if names and modifiers and any(modifiers.get((modname, 'par', name)) for name in names): # A modifier may differ between the agents of a group
                return plain()
            with self.evaluating(by_group=True, for_uids=False):
                out = plain()
            if np.ndim(out) == 0:
                return out
            if len(out) != self.sim.groups.n_groups: # Not one value per group after all
                return plain()
            return out[self.groups(self._uids)]
        return grouped

    def compile_plain(self, expr):
        """ As `compile()`, for the parts inside a part worked out by group (see `compile_grouped()`), which are compiled as they are """
        E = ss.expressions
        if isinstance(expr, (E.BinOp, E.Func)):
            return self.compile_op(expr, compile=self.compile_plain)
        return self.compile_leaf(expr)

    def gated(self, expr):
        """ Whether a product has a condition as a factor (true or false for each person) and a function as a parameter in another, so is worked out only where the condition holds (see `compile_gated()`) """
        factors = self.product_factors(expr)
        conds = [f for f in factors if self.is_condition(f)]
        if not conds:
            return False
        funcs = [leaf for f in factors if not self.is_condition(f) for leaf in f.leaves if isinstance(leaf, ss.expressions.par) and callable(leaf.raw) and not isinstance(leaf.raw, (ss.parameters.TimeValue, ss.Dist))]
        return bool(funcs)

    @staticmethod
    def product_factors(expr):
        """ The factors of a product, in order (e.g. `[a, b, c]` for `a*b*c`) """
        if isinstance(expr, ss.expressions.BinOp) and expr.op == '*':
            return Agents.product_factors(expr.a) + Agents.product_factors(expr.b)
        return [expr]

    @staticmethod
    def is_condition(expr):
        """ Whether a part of an expression is a condition, which is 1 or 0 for each person (not a property, e.g. `ss.ppl.age`, which is its value) """
        cond = expr.cond if isinstance(expr, ss.expressions.Prop) else expr
        return isinstance(cond, ss.conditions.Cond) and (cond.op in ss.conditions.comparisons or cond.op in ['&', '|', '~', 'isin', 'state'])

    def compile_gated(self, expr):
        """
        A product with a condition as a factor (see `gated()`): the conditions are evaluated first, and the product only for the agents they
        hold for, so a function in it (e.g. FPsim's infant mortality, by the mother's age) is called only for them. The others' value is 0
        (as the product's would be, unless the function gives them a NaN or an infinity, which is not read).
        """
        conds = [self.compile_leaf(f) for f in self.product_factors(expr) if self.is_condition(f)]
        whole = self.memoized(expr, self.compile_op(expr))
        def gated():
            if self._by_group or not self._for_uids:
                return whole()
            uids = self._uids
            holds = np.ones(len(uids), dtype=bool)
            for cond in conds:
                holds &= np.asarray(cond()) != 0
            if holds.all():
                return whole()
            out = np.zeros(len(uids))
            if holds.any():
                sub = uids[holds]
                with self.evaluating(uids=sub):
                    vals = whole()
                if np.ndim(vals) and len(vals) != len(sub): # One value per agent in the population
                    vals = np.asarray(vals)[sub]
                out[holds] = vals
            return out
        return gated

    def clock(self, clock):
        """ The value of a clock for the agents being evaluated, from the engine of the module its states belong to """
        name = ss.conditions.modname(clock.module)
        engine = self if name in [None, self.mod.name] else next((m.engine for m in self.sim.declared if m.name == name), None)
        if engine is None:
            errormsg = f'{clock} is a clock of the module "{name}", which is not in the sim'
            raise KeyError(errormsg)
        return engine.clock_values(clock, self._uids)

    def mother(self, expr):
        """ `ss.mother(x)`: x evaluated on each unborn agent's mother (by UID; with born=True, each agent's), and 0 for everyone else """
        ppl = self.sim.people
        parent = ppl.parent.raw
        who = parent != ppl.parent.nan
        if not expr.born:
            who &= ppl.age.raw < 0
        out = np.zeros(len(parent))
        if who.any():
            moms = ss.uids(parent[who])
            x = expr.x
            if isinstance(x, ss.conditions.Cond): # A value (e.g. ss.ppl.age) is read as it is, and a condition as 1/0
                vals = x.eval(self.selector)
                numeric = isinstance(vals, np.ndarray) and vals.dtype != bool and vals.shape == self.selector.shape
                vals = vals[moms] if numeric else self.selector.to_selection(vals, x)[moms]
            elif callable(x) and not isinstance(x, ss.expressions.Expr):
                vals = self.selector.select(x)[moms]
            else:
                vals = np.broadcast_to(np.asarray(self.eval_for(x, moms), dtype=float), len(moms))
            out[who] = vals
        return out

    def clock_values(self, clock, uids):
        """ Time in some states (or the share of the dwell time elapsed), for some agents; NaN for those not in them """
        mod = self.mod
        names = clock.names
        inside = np.zeros(len(uids), dtype=bool)
        for name in names:
            inside |= mod[name].raw[uids]
        leaving = getattr(self, '_leaving', None)
        if leaving is not None: # Being split as they leave: still in the state they left
            src = np.full(len(mod.rel_sus.raw), -1)
            src[leaving[0]] = leaving[1]
            inside |= np.isin(src[uids], [self.state_index[name] for name in names])
        if len(set(names)) == 1:
            arr = mod[f'ti_{names[0]}']
        else:
            key = tuple(sorted(set(names), key=list(self.graph.states.keys()).index))
            if key not in self.sets:
                errormsg = f'{clock} was not known when the sim was built, so the time of entering the set was not kept; make it before the sim'
                raise ValueError(errormsg)
            arr = mod[self.sets[key]]
        if isinstance(arr, ss.arrays.Unkept):
            errormsg = f'{clock} reads the time of entering {names[0]}, which was not kept since no clock of it was known when the sim was built; make the clock before the sim, or set ss.options.keep_times = True'
            raise ValueError(errormsg)
        start = np.asarray(arr.raw[uids], dtype=float)
        ti = self.ti
        if clock.kind == 'time':
            out = np.maximum(ti - start, 0)*self.dt # In the sim's unit; someone who entered at the end of this step (by transmission) has had none of it
        else:
            name = names[0]
            exits = self.graph.exits[name]
            if not (exits.branches or exits.remainder or exits.timed):
                raise ValueError(f'{clock} needs a dwell time scheduled when people enter {name}, and {name} has only rates out of it.')
            end = np.asarray(mod[f'ti_exit_{name}'].raw[uids], dtype=float) # The step the exit is due in (from its start)
            if isinstance(leaving, tuple) and 'left_exit' in mod.__dict__: # Leaving it now: the exit they had
                end = np.where(np.isnan(end), mod.left_exit.raw[uids], end)
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.clip(np.where(end > start, (ti - start)/(end - start), 1.0), 0, 1)
        return np.where(inside, out, np.nan)

    #%% Attributes

    def in_states_for(self, names, uids):
        """ Whether each of some agents is in any of some states; those moving (see `fire()`) are still in the state they left """
        mod = self.mod
        out = np.zeros(len(uids), dtype=bool)
        for name in names:
            out |= mod[name].raw[uids]
        leaving = self._leaving
        if isinstance(leaving, tuple):
            src = np.full(len(mod.rel_sus.raw), -1)
            src[leaving[0]] = leaving[1]
            out |= np.isin(src[uids], [self.state_index[name] for name in names])
        return out

    def attr(self, a):
        """ The value of an attribute for the agents being evaluated, from the engine of the module it belongs to """
        name = ss.conditions.modname(a.module)
        engine = self if name in [None, self.mod.name] else next((m.engine for m in self.sim.declared if m.name == name), None)
        if engine is None:
            errormsg = f'The attribute "{a.name}" belongs to the module "{name}", which is not in the sim'
            raise KeyError(errormsg)
        if getattr(self, '_for_uids', False):
            return engine.attr_values(a, self._uids)
        return engine.attr_values(a, ss.uids(np.arange(len(self.mod.rel_sus.raw)))) # For every agent, by UID (e.g. in a modifier's amount)

    def values_for(self, expr, uids):
        """ An expression for some agents, as one value each, leaving the agents being evaluated as they were """
        return np.array(np.broadcast_to(self.eval_for(expr, uids), len(uids)), dtype=float)

    def attr_values(self, a, uids):
        """ The value of an attribute for some agents: as held, or worked out now if it is derived (NaN in a state it has no value for); with its modifiers """
        if a.kind != 'derived':
            out = self.mod[a.name].raw[uids].astype(float)
        elif a.cases is None:
            out = self.values_for(a.value, uids)
        else:
            out = np.full(len(uids), np.nan)
            for names,expr in a.cases:
                inside = self.in_states_for(names, uids)
                if inside.any():
                    out[inside] = self.values_for(expr, uids[inside])
        mods = self.sim.modifiers.get((self.mod.name, 'attr', a.name))
        return self.modify_attr(mods, out, uids) if mods else out

    def modify_attr(self, mods, value, uids):
        """ The values of an attribute for some agents with its active modifiers: set, then multiply, then add """
        with self.evaluating(uids=self._uids, for_uids=self._for_uids): # As they were, after the amounts are worked out
            active = [m for m in mods if m.active]
            for op in ['set', 'multiply', 'add']:
                for m in active:
                    if m.op != op:
                        continue
                    amount = m.current(self)
                    amount = amount[uids] if np.ndim(amount) else amount
                    mask = np.ones(len(uids), dtype=bool) if m.where is None else self.selector.select(m.where)[uids] > 0
                    if   op == 'set':      value = np.where(mask, amount, value)
                    elif op == 'multiply': value = value*np.where(mask, amount, 1.0)
                    elif op == 'add':      value = value + np.where(mask, amount, 0.0)
        return value

    def draw_attr(self, a, uids, value=None):
        """ Draws of an attribute for some agents: from its distribution, or an expression of several, each from a stream of its own (or of `value`, e.g. its value before any event) """
        value = a.value if value is None else value
        if not len(uids):
            return np.zeros(0)
        if isinstance(value, ss.Dist):
            return np.asarray(self.draw(self.attr_dists[id(value)], uids), dtype=float)
        with self.evaluating(drawing=True):
            return self.values_for(ss.expressions.as_expr(value), uids)

    def draw_traits(self, uids):
        """ The traits of agents who have just been made (or placed at the start), and their derived attributes from them """
        if not len(uids):
            return
        for a in getattr(self, 'traits', []):
            self.mod[a.name][uids] = self.draw_attr(a, uids)
        for a in getattr(self, 'attrs', {}).values(): # The value before any event, drawn for each person
            if a.init_value is not None and a.kind not in ['derived', 'trait']:
                self.mod[a.name][uids] = self.draw_attr(a, uids, value=a.init_value)
        self.write_derived(uids)
        return

    def write_derived(self, uids):
        """ Derived attributes as they are now, for code that reads them (they are worked out again when the engine reads them) """
        for a in self.attrs.values():
            if a.kind == 'derived':
                self.mod[a.name][uids] = self.attr_values(a, uids)
        return

    def event_mask(self, events, tr, srcs, dst):
        """ Which of some agents moving along a transition (None if moved some other way) from states (by index) to a state an event happens to """
        out = np.zeros(len(srcs), dtype=bool)
        index = self.state_index
        for e in events:
            kind, obj, to = e.args
            if kind in ['tr', 'to']:
                if tr is not None and tr.name == obj.name and (kind == 'tr' or dst == index.get(getattr(to, 'name', to))):
                    out[:] = True
            else:
                inside = [index[s[1] if isinstance(s, tuple) else s.name] for s in obj]
                table = np.zeros(len(index) + 1, dtype=bool) # Whether each state (by index, and -1 for none, the last) is one of them
                table[inside] = True
                src_in = table[srcs]
                dst_in = np.bool_(dst in inside) # Not a Python bool, whose ~ is an integer
                out |= (~src_in & dst_in) if kind == 'enter' else (src_in & ~dst_in)
        return out

    def fire(self, tr, uids, srcs, dst):
        """
        Set the attributes an event sets, for the agents it happens to, in the order they are declared: inside
        the move, once they have left the state they were in and before they are in the new one, so a recorded
        value (and a clock it reads) is as it was just before the move

        Args:
            tr (Transition): the transition (None for a move of another kind)
            uids (uids): the agents
            srcs (array): the index of the state each came from (-1 for none), or one for all
            dst (int): the index of the state they go to (-1 for none)
        """
        if not getattr(self, 'on_event', None) or not len(uids):
            return
        mod = self.mod
        srcs = np.broadcast_to(srcs, len(uids))
        with self.evaluating(leaving=(uids, srcs)):
            for a in self.on_event:
                mask = self.event_mask(a.events, tr, srcs, dst)
                if not mask.any():
                    continue
                sel = uids if mask.all() else uids[mask]
                if a.by_event is not None: # Each event's own value (TY1)
                    new = np.full(len(sel), np.nan)
                    for events,value in a.by_event:
                        case = self.event_mask(events, tr, srcs[mask], dst)
                        if case.any():
                            new[case] = self.draw_attr(a, sel[case], value=value)
                else:
                    new = self.draw_attr(a, sel) if a.kind == 'drawn' else self.values_for(a.value, sel)
                old = mod[a.name].raw[sel]
                if a.combine == 'first':
                    val = np.where(mod[f'had_{a.name}'].raw[sel], old, new)
                    mod[f'had_{a.name}'][sel] = True
                else:
                    val = a.combined(old, new)
                mod[a.name][sel] = val
        return

    def paths(self):
        """ For each state, the transitions on the shortest path to it from the first state, as (transition, source, destination) """
        graph = self.graph
        out = {graph.first.name: []}
        todo = [graph.first.name]
        while todo:
            name = todo.pop(0)
            for tr in graph.transitions.values():
                if any(s.name == name for s in tr.src or []):
                    for d in tr.dst or []:
                        if d.people and d.name not in out:
                            out[d.name] = out[name] + [(tr, name, d.name)]
                            todo.append(d.name)
        return out

    def fire_placed(self):
        """
        People placed at the start count as having gone through the transitions on the shortest path from the first state to theirs, at the start
        (each source read as just completed), or, with no such path, as having entered their state; not those moved by code (ss.move()), whose move set what it sets
        """
        mod = self.mod
        paths = self.paths()
        for state in self.graph.people_states:
            uids = self.members(state)
            if self.moved_early is not None:
                uids = uids[~np.isin(uids, self.moved_early)]
            if not len(uids):
                continue
            if not paths.get(state.name): # The first state, or one no transition leads to: placement is entering it (spec 03)
                self.fire(None, uids, -1, self.state_index[state.name])
            for tr,src,dst in paths.get(state.name, []):
                self.new_round() # One for each transition, as each can set an attribute again
                arr = mod[f'ti_{src}']
                kept = not isinstance(arr, ss.arrays.Unkept) # Not kept if nothing reads the time of entering it, which then this cannot either
                if kept:
                    before = np.asarray(arr.raw[uids]).copy()
                    arr[uids] = 0
                mod.left_exit[uids] = -0.5 # A dwell time just completed: see clock_values()
                self.fire(tr, uids, self.state_index[src], self.state_index[dst])
                if kept:
                    arr.raw[uids] = before
        return

    def derived_strat(self, by):
        """ The stratification by a derived attribute that a value by level (`ss.by`) is given by, if it is one """
        if not getattr(self, 'derived', None) or by.dim is None:
            return None
        return next((strat for strat in self.derived if strat.name == by.dim), None)

    def derived_level(self, strat, uids):
        """ The level of a stratification by a derived attribute for some agents """
        vals = self.attr_values(strat.attr_of, uids)
        if np.isnan(vals).any():
            raise ValueError(f'"{strat.name}" is read for people for whom {strat.attr_of.name} has no value (it is NaN in {sc.strjoin(sorted({s.name for s in self.graph.people_states if self.mod[s.name].raw[uids[np.isnan(vals)]].any()}))}); give it a value there, or read it only inside the states that have one.')
        return np.clip(np.searchsorted(strat.edges, vals, side='right') - 1, 0, strat.n - 1)

    def derived_flows(self, tr, src, uids):
        """ Count a flow by the level of a derived attribute at its source, before the move """
        strat = self.derived_trs.get(tr.name)
        if strat is not None and len(uids):
            ng = self.sim.groups.n_groups
            groups = self.groups(uids, src) if ng > 1 else 0
            index = self.derived_groups(strat, groups, self.derived_level(strat, uids))
            self.flow_derived[tr.name] = self.flow_derived.get(tr.name, 0) + np.bincount(index, weights=self.weights(uids), minlength=ng*strat.n)
        return

    def derived_stocks(self, uids=None):
        """ The number in each state by the level of a derived attribute; or, given some agents (who died), with them taken out """
        ng = self.sim.groups.n_groups
        for key,strat in self.derived_res.items():
            if key not in self.graph.states:
                continue
            state = self.graph.states[key]
            inds = self.members(state) if uids is None else uids[self.mod[state.name].raw[uids]]
            groups = self.groups(inds, state) if ng > 1 else 0
            counts = np.bincount(self.derived_groups(strat, groups, self.derived_level(strat, inds)), minlength=ng*strat.n) if len(inds) else 0.0
            self.stock_derived[state.name] = counts if uids is None else self.stock_derived.get(state.name, 0) - counts
        return

    #%% Moving agents

    def put(self, name, uids, value):
        """ Put agents in a state of people (value True), or take them out, as the engine moves them (each once), keeping its count: as a write from code does (see `wrote()`), more cheaply """
        arr = self.mod[name].raw
        if self.live is None:
            arr[uids] = value
            return
        ng = self.sim.groups.n_groups
        groups = self.sim.groups.group_raw() if ng > 1 else None # Any changes of group (e.g. by age) are counted first, as for a write from code
        was = arr[uids]
        arr[uids] = value
        if ng == 1:
            n = np.count_nonzero(was)
            self.live[name] = self.live[name] + (len(uids) - n if value else -n)
        else:
            change = np.bincount(groups[uids[~was if value else was]], minlength=ng)
            self.live[name] = self.live[name] + (change if value else -change)
        return

    def clear_exit(self, name, uids):
        """ Forget the scheduled exit of some agents from a state (which they have left), if it has scheduled exits """
        if name in self.scheduled:
            arr = self.mod[f'ti_exit_{name}']
            arr.raw[uids] = arr.nan
        return

    def move(self, tr, src, uids):
        """ Move agents out of a state along a transition """
        if not len(uids):
            return
        mod = self.mod
        ti = self.ti
        if self.derived_trs:
            self.derived_flows(tr, src, uids)
        if self.on_event and src.name in self.scheduled:
            mod.left_exit[uids] = mod[f'ti_exit_{src.name}'].raw[uids]
        if tr.stays and tr.trigger is not None and src.name in self.scheduled: # Those who stay keep the exit they had (see stay())
            kept = mod[f'ti_exit_{src.name}']
            self._kept_exits.setdefault(tr.name, []).append((uids, kept.raw[uids].copy()))
        self.put(src.name, uids, False)
        self.clear_exit(src.name, uids)
        self.flowcounts[tr.name] += self.n_of(uids)
        self.events[tr.name] = uids if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], uids])
        if self.sim.groups.n_groups > 1 and not (self.enters(tr, src) and not (tr.contact and self.strain is not None)): # A move into a dimension is recorded at the level people arrive at, once they have one: see finish_moves()
            groups = self.groups(uids, src)
            if tr.contact and self.strain is not None and not self.applies[src.name][self.strain]: # A new infection is recorded under the strain acquired (which, for someone infected again, is already their strain)
                groups = groups + self.strain_raw()[uids]*int(self.sim.groups.group_mults[self.strain])
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(groups, weights=self.weights(uids), minlength=self.sim.groups.n_groups)
        self._moved.append((tr, src, uids))
        return

    def finish_moves(self, ti):
        """
        The same-step cascade, as one sweep over the transitions in their order (see `sweep()`), as the code of a model would take them: at each,
        those its trigger moves now, then those whose exit by it is due already among those who arrived in its source, leave, and everyone who
        left by it arrives at their destinations. Someone who reaches a transition the sweep has passed (across a loop whose wait rounds to no
        steps) is taken by another sweep, as are the triggers on an event that happened after them, until nobody moves.

        Args:
            ti (int): the time of entry: the start of the step (`self.ti`) for the transitions that are not contact-mediated, and its end (the
                loop's `ti`) for infections, which have happened by then (so an infected person transmits over their whole dwell time, as in the ODE)
        """
        mod = self.mod
        order, position, back = self.sweep()
        sent = {} # Who has left by each transition, to be sent on: (transition, state, uids)
        def collect():
            moved, self._moved = self._moved, []
            for item in moved:
                sent.setdefault(item[0].name, []).append(item)
        collect()
        triggers = {tr.name for tr in self.triggers} if self._started else set()
        limit = len(self.graph.people_states) + 1
        arrived = {} # Who arrived in each state, whose exit may be due already
        for n in range(limit + 1):
            again = False
            for tr in order:
                name = tr.name
                self.new_round()
                if name in triggers:
                    for t,state,uids in self.triggered([tr]):
                        self.move(t, state, uids)
                for src in tr.src: # Those whose exit by it is due already
                    uids = arrived.get(src.name)
                    if uids is not None and tr in self.exits_of[src.name]:
                        uids = uids[mod[src.name].raw[uids]]
                        now = uids[(mod[f'ti_exit_{src.name}'].raw[uids] <= ti) & (mod[f'exit_{src.name}'].raw[uids] == self.exits_of[src.name].index(tr))]
                        if len(now):
                            self.move(tr, src, now)
                collect()
                items = sent.pop(name, None)
                if not items:
                    continue
                got = self.send(items, ti)
                if self._row is not None and self._backfill is None: # Births driven by a flow (e.g. at a delivery) before the arrivals face the rates of where they arrived, so those rates see the event that brought them (its children and records, e.g. ss.Pregnancy's since_birth: FP2)
                    if self._news: # The modules that react to the event first, as they would at the end of the step (e.g. a fetus that dies at the delivery is stillborn, not born)
                        self._news = False
                        self.notify(ti)
                    self.step_per(ti)
                for state,uids in self.arrive(got, ti):
                    arrived[state] = uids if state not in arrived else ss.uids.concatenate([arrived[state], uids]).unique()
                    again |= any(position[e.name] <= position[name] for e in self.exits_of[state]) # Its exits have been passed
                again |= name in back # A trigger on it has been passed
                collect()
            if not (sent or again):
                break
        else:
            items = [item for v in sent.values() for item in v]
            cycle = sc.strjoin(sorted({f'{tr.name} ({state.name} -> {sc.strjoin([d.name for d in tr.dst]) or "out"})' for tr,state,_ in items})) or sc.strjoin(sorted(arrived))
            timed = sorted({tr.name for tr,_,_ in items if tr.name in self.dwells})
            hint = (f'{sc.strjoin(timed)} {"has a dwell time" if len(timed) == 1 else "have dwell times"} shorter than a step, so for many people it rounds to no steps and they go round again within the step (a dwell is rounded to whole steps at random, keeping its mean). Use a smaller dt, or a longer dwell time.' if timed else
                    'Give one of them a rate or a duration, or change the branches so they do not lead back.')
            raise ValueError(f'moves with no wait out of the states of "{mod.name}" did not settle within a step: {cycle}.\n     {hint}')
        if self._news:
            self._news = False
            self.notify(ti)
        return

    def sweep(self):
        """
        The order in which the same-step cascade takes the transitions (see `finish_moves()`), worked out once: each after those that take people
        into its source and, for a trigger on an event of this module, after the transition of the event; and the triggers out of a state ahead
        of its scheduled exits, so a trigger moves people ahead of an exit due at the same time. A loop is broken at the transition declared first.
        Returns the order, each transition's position in it, and the transitions whose event a trigger before them waits for
        """
        if self.__dict__.get('_sweep') is None:
            trs = list(self.graph.transitions.values())
            after = {tr.name:set() for tr in trs} # The transitions each comes after
            events = {tr.name:self.event_transitions(tr) for tr in trs}
            for t in trs:
                srcs = {s.name for s in t.src}
                after[t.name] |= {u.name for u in trs if u is not t and any(getattr(d, 'name', None) in srcs for d in (u.dst or []))}
                after[t.name] |= events[t.name] - {t.name}
                if t.trigger is not None:
                    for s in t.src:
                        after.update({u.name:after[u.name] | {t.name} for u in self.exits_of.get(s.name, []) if u is not t and u.trigger is None})
            order, done = [], set()
            while len(order) < len(trs):
                ready = [t for t in trs if t.name not in done and after[t.name] <= done]
                t = ready[0] if ready else next(t for t in trs if t.name not in done) # A loop: broken at the first declared
                order.append(t)
                done.add(t.name)
            position = {t.name:i for i,t in enumerate(order)}
            back = {u for t in trs for u in events[t.name] if position[u] > position[t.name]}
            self._sweep = (order, position, back)
        return self._sweep

    def event_transitions(self, tr):
        """ The transitions of this module whose moves are the event a trigger is on (none for another module's event, or a condition) """
        if tr.trigger is None or not tr.trigger.event:
            return set()
        event = tr.trigger.x
        if ss.conditions.modname(event.module) not in [None, self.mod.name]:
            return set()
        spec = event.spec()
        if 'tr' in spec:
            return {spec['tr']}
        names = set(spec.get('enter', ()) or spec.get('exit', ()))
        key = 'dst' if 'enter' in spec else 'src'
        return {t.name for t in self.graph.transitions.values() if any(getattr(s, 'name', None) in names for s in (getattr(t, key) or []))}

    def send(self, moved, ti):
        """
        Send the agents who left a state in a round to their destinations, one transition at a time (in the order they first left by it):
        a transition with several destinations takes one draw; an outflow is a death (or a removal). Returns who arrived in each state
        whose exits are scheduled on entry, or, in the state-change row, whose competing rates are drawn on entry.
        """
        mod = self.mod
        index = self.state_index
        ng = self.sim.groups.n_groups
        row = self._row
        by_tr = {}
        for tr,src,uids in moved:
            by_tr.setdefault(tr.name, (tr, [], []))
            by_tr[tr.name][1].append(src)
            by_tr[tr.name][2].append(uids)
        arrived = {}
        for name,(tr,srcs,uidlist) in by_tr.items():
            uids = ss.uids.concatenate(uidlist) if len(uidlist) > 1 else uidlist[0]
            frm = np.concatenate([np.full(len(u), index.get(s.name, -1)) for s,u in zip(srcs, uidlist)]) # The state each came from
            if tr.outflow:
                self.fire(tr, uids, frm, -1)
                self.record(tr, uids, frm, -1) # An event, e.g. for the children of ss.birth(per=) of a death at delivery
                if tr.migration:
                    self.sim.people.request_removal(uids)
                else:
                    self.sim.people.request_death(uids, cause=mod)
                    self._died.append((tr, srcs, uidlist)) # Counted once each death has one cause: see uncount_deaths()
                continue
            entering = ng > 1 and any(self.enters(tr, src) for src in srcs) # Into a dimension: recorded at the level they arrive at
            if not tr.split:
                dsts = [(tr.dst[0], uids, frm)]
            else:
                dsts = self.destinations(tr, srcs, uidlist, uids, frm, entering)
            for dst,inds,came in dsts:
                self.enter_dims(tr, srcs, uidlist, dst, inds)
                self.move_levels_of(tr, dst, inds)
                self.fire(tr, inds, came, index[dst.name]) # The attributes the move sets, before the dwell times that can read them
                self.enter(dst, inds, ti, tr=tr, srcs=came) # Once their levels are known, since dwell times can depend on them
                if entering and not (tr.contact and self.strain is not None):
                    self.add_flow_groups(name, self.groups(inds, dst), inds)
                    if tr.ends:
                        self.add_flow_groups(f'{name}.{dst.name}', self.groups(inds, dst), inds)
                if len(inds) and (dst.name in self.scheduled or (row is not None and self.competing[dst.name])):
                    arrived.setdefault(dst.name, []).append(inds)
        return arrived

    def destinations(self, tr, srcs, uidlist, uids, frm, entering):
        """ The destination of each agent who leaves by a transition with several (in its shares, with one draw), as (state, uids, came from); those who take ss.null die, and those who take ss.stay are put back """
        edges = self.dest_edges(tr, uids, frm)
        rvs = self.draw(f'dst_{tr.name}', uids)
        masks = [(rvs >= edges[i]) & (rvs < edges[i+1]) for i in range(len(edges) - 1)]
        dsts = [(dst, uids[mask], frm[mask]) for dst,mask in zip(tr.dst, masks)]
        if tr.dies: # Those who took ss.null die, as by an outflow
            self.split_deaths(tr, srcs, uidlist, uids[masks[-1]], frm[masks[-1]])
        if tr.stays: # What the others leave over stays: the move is canceled
            stayed = rvs >= edges[-1]
            self.stay(tr, uids[stayed], frm[stayed])
        if tr.ends: # The flow to each destination
            left = np.concatenate([self.groups(u, src) for u,src in zip(uidlist, srcs)]) if self.sim.groups.n_groups > 1 and not entering else None # The groups they left, where the total is recorded (see move()), before any change of level
            for (dst,inds,_),mask in zip(dsts + [(ss.transitions.null, uids[masks[-1]], None)]*tr.dies, masks):
                key = f'{tr.name}.{"null" if dst is ss.transitions.null else dst.name}'
                self.flowcounts[key] += self.n_of(inds)
                if left is not None:
                    self.add_flow_groups(key, left[mask], inds)
        return dsts

    def dest_edges(self, tr, uids, frm):
        """ Where each destination of a transition with several starts, as a cumulative probability, for some agents leaving by it from the states `frm` (one row per destination, ss.null last if it is one, and its end); what is beyond is ss.stay """
        key = ('dest', tr.name)
        fixed = self.__dict__.setdefault('_fixed', {}).get(key)
        if fixed is None:
            fixed = self._fixed[key] = not tr.dies and all(self.is_fixed(p) for p in tr.p) and all(self.unmodified('branch', f'{tr.name}.{dst.name}') for dst in tr.dst)
        return self.fixed_edges(key, fixed, lambda u: self.dest_edges_for(tr, u, frm if u is uids else np.full(len(u), -1)), uids)

    def dest_edges_for(self, tr, uids, frm):
        """ As dest_edges(), worked out for these agents """
        self._uids = uids
        with self.evaluating(leaving=(uids, frm)): # Decided as they leave: the time in a clock is the time they spent there (see clock_values())
            ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
            if tr.dies:
                ps.append(np.broadcast_to(self.die_share(tr, ps, lambda p: self.eval_for(p, uids)), len(uids)))
        return np.cumsum([np.zeros(len(uids))] + ps, axis=0)

    def arrive(self, arrived, ti):
        """
        Of those who arrived in each state (from `send()`), those still there, as (state name, uids); in the state-change row, those who arrived
        in a state with competing rates, and whose exit is not due already, have half a step of them now (those who leave are moved)
        """
        mod = self.mod
        row = self._row
        out = []
        for name,uidlist in arrived.items():
            uids = ss.uids.concatenate(uidlist) if len(uidlist) > 1 else uidlist[0]
            uids = uids[mod[name].raw[uids]] # Those still there
            out.append((name, uids))
            if row is not None and self.competing[name]: # Half a step of the competing rates of the state now, as for people who arrive during a step in a compartmental run (once a step each)
                now = self.due(name, uids, ti)
                stay = uids[~np.isin(uids, now[1])] if now is not None else uids
                hz = stay[~row[stay]]
                row[hz] = True
                if len(hz):
                    self.draw_competing(self.graph.states[name], hz, 0.5*self.dt, key=f'entry_{name}')
        return out

    def due(self, name, uids, ti):
        """ Of some agents in a state, those whose scheduled exit is due by `ti`: (transition, state, uids) for each exit, and all of them; or None if none are """
        if name not in self.scheduled:
            return None
        mod = self.mod
        now = uids[mod[f'ti_exit_{name}'].raw[uids] <= ti]
        if not len(now):
            return None
        state = self.graph.states[name]
        which = mod[f'exit_{name}'].raw[now]
        return [(tr, state, now[which == i]) for i,tr in enumerate(self.exits_of[name])], now

    def add_flow_groups(self, key, groups, uids):
        """ Add some agents to a flow by group (each counted by its weight, if agents' weights differ) """
        self.flow_groups[key] = self.flow_groups.get(key, 0) + np.bincount(groups, weights=self.weights(uids), minlength=self.sim.groups.n_groups)
        return

    def triggered(self, triggers=None):
        """
        The agents that `ss.on()` moves now, as (transition, state, uids): those in its source for whom its condition holds, or to whom its
        event has happened since it last looked; with a delay (`dur=`), once the time drawn for each of them is up, if they are still there.
        For all the triggers, or some (as `finish_moves()` takes them, each in its place)
        """
        if not self.triggers or not self._started:
            return []
        mod = self.mod
        out = []
        taken = None # Whom an earlier trigger moves now, by UID
        for tr in (self.triggers if triggers is None else triggers):
            srcs = [s for s in tr.src if s.people]
            names = [s.name for s in srcs]
            if tr.trigger.event:
                uids = self.new_events(tr) # Seen whether or not anyone is in the source, so an old event does not move people who enter later
            elif getattr(tr.trigger.x, 'op', None) == 'event': # A delivery's or a product's event: news for a step, so moved once
                uids = np.flatnonzero(self.unseen_fired(tr, np.asarray(self.selector.select(tr.trigger.x), dtype=bool)))
            elif any(mod[name].raw.any() for name in names):
                uids = np.flatnonzero(np.asarray(self.selector.select(tr.trigger.x), dtype=bool))
            else:
                continue
            uids = uids[self.in_states_for(names, uids)]
            if tr.name in self.delays: # Not at once, but a time after the event drawn for each person, as long as they stay in the source
                uids = self.delayed(tr, names, uids, taken)
            if tr.where is not None and len(uids):
                uids = uids[np.asarray(self.selector.select(tr.where), dtype=bool)[uids]]
            if taken is not None:
                uids = uids[~taken[uids]]
            if len(uids):
                for s in srcs:
                    out.append((tr, s, ss.uids(uids[mod[s.name].raw[uids]])))
                if taken is None:
                    taken = np.zeros(len(mod.rel_sus.raw), dtype=bool)
                taken[uids] = True
        return out

    def delayed(self, tr, names, uids, taken):
        """ For a trigger with a delay: draw the time after the event for those it has just happened to (`uids`), forget it for those who left the source, and return those whose time is up """
        mod = self.mod
        due = mod[f'ti_after_{tr.name}']
        pending = np.flatnonzero(~np.isnan(due.raw))
        inside = self.in_states_for(names, pending)
        due.raw[pending[~inside]] = due.nan
        pending = pending[inside]
        new = ss.uids(uids[np.isnan(due.raw[uids])])
        if len(new):
            dur = self.draw(f'after_{tr.name}', new)
            if not isinstance(ss.transitions.by_dist(getattr(self.delays[tr.name], 'raw', None)), ss.Dist): # A scalar: an exponential with that mean
                dur = ss.dists.inverse('expon', dur, np.broadcast_to(np.asarray(self.eval_for(self.delays[tr.name], new), dtype=float), len(new)))
            due[new] = self.ti + self.whole_steps(new, dur) # As for a dwell time
            pending = np.sort(np.concatenate([pending, new])) # As np.union1d(), since none of the new ones is pending
        now = pending[due.raw[pending] <= self.ti]
        if taken is not None: # Those another trigger moves now keep their time, so it is taken in the next step if they are still in the source
            now = now[~taken[now]]
        due.raw[now] = due.nan
        return now

    def unseen_fired(self, tr, mask):
        """ Of the people an event is news for, those this trigger has not already seen it for (on this step or the last) """
        ti = self.ti
        last, now, before = self._fired_seen.get(tr.name, (None, None, None))
        pad = lambda m: np.zeros(len(mask), dtype=bool) if m is None else np.pad(m, (0, len(mask) - len(m)))
        if last != ti: # A new step: what was seen now was seen on the last step, if it was the last step
            now, before = None, (now if last == ti - 1 else None) # The old now, as the right side is evaluated first
        seen = pad(now) | pad(before)
        self._fired_seen[tr.name] = (ti, pad(now) | mask, before)
        return mask & ~seen

    def new_events(self, tr):
        """ The agents an event trigger's event has happened to since it last looked """
        event = tr.trigger.x
        name = ss.conditions.modname(event.module)
        engine = self if name in [None, self.mod.name] else self.sim.get_declared(name).engine
        spec = event.spec()
        if len(spec) == 1 and 'tr' in spec and engine.last_batch.get(spec['tr'], 0) <= self._seen.get(tr.name, 0): # Nothing of its transition since it looked
            self._seen[tr.name] = max(self._seen.get(tr.name, 0), engine.batches[-1].n if engine.batches else 0) # As unseen() would mark it
            return ss.uids()
        fresh = engine.unseen(self._seen, tr.name)
        return engine.event_uids(**spec, batches=fresh) if fresh else ss.uids() # Each once, in order (event_uids() gives them so)

    def step_per(self, ti):
        """ Births driven by a flow (`ss.birth(per=)`): children for the people who went through the transition since this last looked, made by the module that owns the transition; then whatever moves at once, including the children's own exits due now (e.g. a `dur=0` split) """
        born = []
        for tr in self.graph.inflows:
            if getattr(tr, 'per', None) is None or not tr.people or ss.conditions.modname(tr.per.module) not in [None, self.mod.name]: # Another module's transition: that module makes the children, who are placed here by this inflow's state (see into())
                continue
            parents = self.fresh(('per', tr.name), tr.per.name, unique=False) # Once for each time, e.g. twice for a parent whose dwell times rounded to no steps
            if not len(parents):
                continue
            new = self.mod.make_children(tr, parents, ti) # E.g. ss.Pregnancy's, which also records events on the mothers
            self.fire_entered(new, tr)
            self.flowcounts[tr.name] += self.n_of(new)
            if len(new):
                self.record(tr, new, -1, self.state_index[tr.dst[0].name]) # An event, e.g. preg.birth
                self.events[tr.name] = new if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], new])
                if self.sim.groups.n_groups > 1:
                    self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(new), weights=self.weights(new), minlength=self.sim.groups.n_groups)
                for mod in self.sim.declared: # Counted in the flow of each module's own ss.birth(per=) of the same transition, and recorded there as its event (for on= and conditions)
                    other = next((o for o in mod.graph.inflows if mod is not self.mod and getattr(o, 'per', None) is not None and o.per.name == tr.per.name and ss.conditions.modname(o.per.module) == self.mod.name), None)
                    if other is not None:
                        engine = mod.engine
                        engine.flowcounts[other.name] += self.n_of(new)
                        engine.record(other, new, -1, engine.state_index[other.dst[0].name])
                        engine.events[other.name] = new if other.name not in engine.events else ss.uids.concatenate([engine.events[other.name], new])
                born.append(new)
        if born:
            born = ss.uids.concatenate(born)
            self.cache.clear() # Values worked out this step before the children and the event's records (e.g. a rate function of the mothers' time since birth), for the draws of those who arrive now (see finish_moves())
            for mod in self.sim.declared: # In each module they were put in
                engine = mod.engine
                if engine is not self and isinstance(engine, Agents) and (engine.move_due(born, ti) or engine._news): # Or a birth recorded there just now, which its triggers and those of the modules that react to it see
                    engine.finish_moves(ti)
            self.move_due(born, ti)
            self.finish_moves(ti)
        return

    def fire_entered(self, uids, tr=None):
        """ The attributes that entering their states sets, for agents just put in them by an inflow (e.g. newborns), in each module, once their ages and sexes are set """
        for mod in self.sim.declared:
            engine = mod.engine
            if not isinstance(engine, Agents) or not getattr(engine, 'on_event', None):
                continue
            for state in mod.graph.people_states:
                inds = uids[mod[state.name].raw[uids]]
                if len(inds):
                    engine.fire(tr if engine is self else None, inds, -1, engine.state_index[state.name])
        return

    def move_due(self, uids, ti):
        """ Agents just put in their states (e.g. newborns) whose exit is due already move on now, as those arriving by a transition do (see finish_moves()); returns whether any did """
        moved = False
        for name in self.scheduled:
            now = self.due(name, uids[self.mod[name].raw[uids]], ti)
            if now is not None:
                for tr,state,inds in now[0]:
                    self.move(tr, state, inds)
                moved = True
        return moved

    def n_children(self, tr, parents):
        """ The number of children of each parent at a birth driven by a flow """
        n = tr.n_born
        if isinstance(n, ss.Dist):
            n = self.draw(f'n_{tr.name}', parents)
        elif callable(n):
            n = n(self.sim, parents)
        return np.broadcast_to(np.asarray(n, dtype=int), len(parents))

    def grow_children(self, tr, parents, ti, counts=None, age=0.0, into=None):
        """
        Make the children of some parents (`counts` of each, default `n_children()`), linked to them, in their parent's place,
        and in each declared module's state for them (see `into()`), or (`into=False`) each module's first state, as for fetuses
        """
        sim = self.sim
        people = sim.people
        counts = self.n_children(tr, parents) if counts is None else counts
        moms = np.repeat(np.asarray(parents), counts)
        if not len(moms):
            return ss.uids()
        places = self.place_raw()[moms] if sim.mixing.space is not None else np.zeros(len(moms), dtype=int)
        states = () if into is False else tuple(self.into(tr))
        born = np.all(np.asarray(age) >= 0) # Or fetuses, aged below 0
        npl = np.bincount(places, minlength=1)
        new = self.add_people([(place, states, int(k)) for place,k in enumerate(npl) if k], ti, f'born_{tr.name}', newborn=born)
        order = np.concatenate([moms[places == place] for place,k in enumerate(npl) if k]) # In the order they were made: by place, then each parent's
        people.parent[new] = order
        if f'sex_{tr.name}' in self.dists:
            people.female[new] = self.draw(f'sex_{tr.name}', new)
            self.sim.groups.set_sex(new) # A dimension called sex follows (it was set from the default when they were made)
            self.sim.groups.groups_changed(new)
            sim.groups.groups_changed(new)
        if not born: # Fetuses, so that they are born at 0
            byplace = np.concatenate([np.flatnonzero(places == place) for place,k in enumerate(npl) if k])
            people.age[new] = np.broadcast_to(age, len(moms))[byplace]
            sim.groups.groups_changed(new)
        return new

    def into(self, tr):
        """ The names of the states that children born by a transition are put in: this inflow's, and that of each declared module's own `ss.birth(per=)` of the same transition """
        out = [d.name for d in tr.dst]
        per = (ss.conditions.modname(tr.per.module) or self.mod.name, tr.per.name)
        for mod in self.sim.declared:
            for other in mod.graph.inflows:
                if getattr(other, 'per', None) is not None and (ss.conditions.modname(other.per.module) or mod.name, other.per.name) == per:
                    out += [d.name for d in other.dst]
        return list(dict.fromkeys(out))

    def fresh(self, key, tr, unique=True):
        """
        The agents who went through a transition (or a recorded event) of this module since `key` last asked: for code that reacts to events once each.
        With `unique=False`, an agent appears once for each time, e.g. a woman who conceived, had an abortion at once, and conceived again in a step
        """
        fresh = [b for b in self.unseen(self._seen, key) if not b.also] # Those the event happened to themselves
        return self.event_uids(tr, batches=fresh, unique=unique) if fresh else ss.uids()

    def unseen(self, seen, key):
        """ The batches recorded since `key` last looked (by the number of the last it saw, in `seen`) """
        last = seen.get(key, 0)
        batches = self.batches
        i = len(batches)
        while i and batches[i-1].n > last:
            i -= 1
        if batches:
            seen[key] = batches[-1].n
        return batches[i:]

    def record_event(self, name, uids):
        """ Record an event that is not a transition (e.g. a live birth, for the mother), so it can be read by ss.on() and conditions """
        if len(uids):
            self.record(ss.conditions.Named(self.mod, name), ss.uids(uids), -1, -1)
        return

    @property
    def subscribers(self):
        """ The engines with a trigger that reads this module's states or events """
        if self._subscribers is None:
            self._subscribers = [m.engine for m in self.sim.declared if m is not self.mod and isinstance(m.engine, Agents) and self.mod.name in m.engine.trigger_mods]
        return self._subscribers

    def notify(self, ti):
        """ After people moved here: the triggers of other modules that react to this one are resolved now, in the same step """
        sim = self.sim
        if not self.subscribers:
            return
        depth = getattr(sim, '_trigger_depth', 0)
        limit = sum(len(m.graph.people_states) for m in sim.declared) + 1
        if depth > limit:
            raise ValueError(f'triggers in {sc.strjoin(m.name for m in sim.declared)} kept moving people back and forth within a step (from "{self.mod.name}").\n'
                                 f'     Change the conditions so they do not lead back.')
        sim._trigger_depth = depth + 1
        try:
            for engine in self.subscribers:
                if engine._started:
                    engine._moved = []
                    engine.finish_moves(ti)
        finally:
            sim._trigger_depth = depth
        return

    def settle_start(self):
        """ Once every module has its people (see `ss.Sim.init()`): those placed in a state nobody waits in move on at once, as they would during a step; these moves are not counted as flows """
        mod = self.mod
        self.new_round()
        burnin = self.mod.burnin_plan(self) # People who start part way through, from a module's own rule (ss.Pregnancy's burnin=)
        if burnin is not None:
            self.backfill_burnin(burnin)
            self.record_stocks() # The start, again
        for key,arr in self.sets.items(): # Those placed in a set entered it when they were placed
            for name in key:
                uids = self.members(self.graph.states[name])
                uids = uids[np.isnan(np.asarray(mod[arr][uids], dtype=float))]
                mod[arr][uids] = mod[f'ti_{name}'].raw[uids]
        if self.on_event:
            self.fire_placed()
        moved = False
        self._moved = []
        self._started = True
        n_batches = len(self.batches)
        for state in self.graph.people_states:
            if state.name not in self.junctions:
                continue
            uids = self.members(state)
            if not len(uids):
                continue
            self.schedule(state, uids)
            which = mod[f'exit_{state.name}'][uids]
            exits = self.graph.exits[state.name]
            for i,tr in enumerate(exits.branches + sc.tolist(exits.remainder)):
                self.move(tr, state, uids[which == i])
            moved = True
        if self.triggers: # Seeds a trigger holds for move on before the start (see finish_moves())
            moved = True
        if moved: # Counted at the start (the results at the first time point), and news to the deliveries of the first step only
            self.finish_moves(self.ti)
            for b in self.batches[n_batches:]:
                b.key = (self.ti - 1, 0)
            self.events = dict()
            self.live = None # Counted again
            self.unscheduled = None # Writes are not watched while nothing is counted
            self.record_stocks()
        return

    @property
    def junctions(self):
        """ The states left only by branches that nobody waits in (no dwell time and no minimum stay) """
        if self._junctions is None:
            self._junctions = set()
            for name,exits in self.graph.exits.items():
                branches = exits.branches + sc.tolist(exits.remainder)
                if branches and not exits.timed and not self.competing.get(name) and all(tr.name not in self.dwells and not self.after_of(tr) for tr in branches):
                    self._junctions.add(name)
            def walk(name, path=()): # People passed through at once must reach a state they wait in
                if name in path:
                    cycle = ' -> '.join(path[path.index(name):] + (name,))
                    raise ValueError(f'moves with no wait out of the states of "{self.mod.name}" form a loop: {cycle}.\n'
                                         f'     Give one of them a rate or a duration, or change the branches so they do not lead back.')
                if name in self._junctions:
                    exits = self.graph.exits[name]
                    for tr in exits.branches + sc.tolist(exits.remainder):
                        for dst in tr.dst:
                            walk(dst.name, path + (name,))
                return
            for name in sorted(self._junctions):
                walk(name)
        return self._junctions

    def split_deaths(self, tr, srcs, uidlist, uids, src_idx):
        """ People who took ss.null in a split: they die, as by an outflow, and are counted in its flow to ss.null (see uncount_deaths()) """
        if not len(uids):
            return
        self.fire(tr, uids, src_idx, -1)
        self.record(tr, uids, src_idx, -1)
        self.sim.people.request_death(uids, cause=self.mod)
        self._died.append((tr, srcs, [u[np.isin(u, uids)] for u in uidlist]))
        return

    def stay(self, tr, uids, src_idx):
        """
        People who took ss.stay in a split: back in the state they left, with no new dwell time, so `X.time` keeps running; not counted in the flow

        After a split taken at the end of a dwell time, they have no exit scheduled (the dwell was this transition's); after one
        triggered by an event or a condition (`on=`), they keep the exit they had, e.g. their clearance (see `move()`).
        """
        kept = self._kept_exits.pop(tr.name, [])
        if not len(uids):
            return
        if tr.name in self.events: # Nor are they its event (e.g. who=dis.seek_care), as they are not its flow
            moved = self.events[tr.name]
            self.events[tr.name] = moved[~member(moved, uids)]
        mod = self.mod
        names = list(self.state_index.keys())
        for i in np.unique(src_idx):
            inds = uids[src_idx == i]
            state = self.graph.states[names[i]]
            self.put(state.name, inds, True)
            if state.name in self.scheduled:
                if tr.trigger is not None:
                    for moved,exits in kept:
                        mine = member(moved, inds)
                        mod[f'ti_exit_{state.name}'].raw[moved[mine]] = exits[mine]
                else:
                    mod[f'ti_exit_{state.name}'][inds] = np.inf
                    mod[f'exit_{state.name}'][inds] = -2 # Stays: never scheduled again (see schedule())
            if self.sim.groups.n_groups > 1 and tr.name in self.flow_groups:
                self.flow_groups[tr.name] = self.flow_groups[tr.name] - np.bincount(self.groups(inds, state), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
        self.flowcounts[tr.name] -= self.n_of(uids)
        return

    def enter_dims(self, tr, srcs, uidlist, dst, uids):
        """ People who enter a state with a dimension that the state they left did not have take a level of it, in proportion to its split """
        sim = self.sim
        for d,strat,key in self.entered_dims:
            if self.applies[dst.name][d] and not (tr.contact and d == self.strain): # A new infection has taken its strain already
                lacked = [inds for src,inds in zip(srcs, uidlist) if not self.applies[src.name][d]]
                if lacked:
                    new = uids[np.isin(uids, ss.uids.concatenate(lacked))]
                    if len(new):
                        sim.people[strat.attr][new] = strat.value(strat.draw(sim, new, self.draw(key, new)))
                        sim.groups.groups_changed(new)
        return

    def branch_probs_for(self, tr, uids):
        self._memo = {} # A part that several entries share (e.g. p_use in 1 - p_use and p_use*share) is evaluated once for all of them (see memoized())
        try:
            ps = [None if p is None else self.eval_for(p, uids) for p in tr.p]
        finally:
            self._memo = None
        if any(np.ndim(p) > 1 for p in ps): # e.g. a function giving a row of shares for each person (a switching matrix): not supported yet
            raise ss.CapabilityError(f'p= of "{tr.name}" gave a row of probabilities for each person, but each entry of p= is one destination\'s probability (one value per person); '
                                     f'give one per destination, e.g. p=[f_{tr.dst[0].name}, None], where each function returns one value per person')

        return self.dest_shares(tr, ps, uids, stay=self.ends_given(tr, lambda p: self.eval_for(p, uids))) # With ss.stay or ss.null, what the states leave over stays or dies

    def whole_steps(self, uids, dur, dt=None):
        """
        Dwell times (in a state, or before a delayed move or a transfer) as whole numbers of steps, rounded up or down at random in proportion
        by each agent's phase, as `ss.TimeArr.due()` rounds a time (see `ss.People.phase()`; FP55: 0.2 steps is 0 steps for 80% of agents and
        1 step for 20%), so their mean is kept; a dwell of 0 (dur=0) stays 0, a move in the step people enter
        """
        return np.floor(np.asarray(dur, dtype=float)/(dt or self.dt) + self.sim.people.phase(uids))

    def new_round(self):
        """
        Start a round of moves: the engine's draws are keyed by the step and the round within it (see `draw()`), so a draw
        does not depend on how the round's moves are batched. Each round of the same-step cascade is one, as is each way in
        from code (e.g. `force()`, `ss.move()`); an agent moves at most once in a round, so each agent's draws stay its own.
        """
        ti = self.mod.t.ti # The loop's step (before the start, the step being back-dated: see backfill())
        step, n = self.round
        self.round = (ti, n + 1 if step == ti else 0)
        if self.round[1] >= ROUND_MAX:
            raise ValueError(f'"{self.mod.name}" took more than {ROUND_MAX} rounds of moves in step {ti}')
        self._drawn = {}
        return

    def draw_ind(self):
        """ The index of the engine's draws in this round (see `new_round()`) """
        ti, n = self.round
        return ROUNDS + ti*ROUND_MAX + n

    def draw(self, key, uids):
        """
        Draws from one of the engine's distributions for some agents, keyed by the step and the round of moves within it
        (see `new_round()`), and the distribution left as it was; without common random numbers, the next of its draws
        """
        dist = self.dists[key]
        if not ss.options.crn:
            return dist.rvs(uids)
        if check_draws:
            self.check_drawn(key, uids)
        how = self._draw_how.get(key)
        if how is None: # The uniforms themselves (most of them), the unit exponential of a dwell time (see sample_dwell()), a distribution with values by sex, drawn from a table kept for the run (see agent_pars()), or any other as it is
            how = self._draw_how[key] = 'uniform' if type(dist) is ss.random else 'expon' if (type(dist) is ss.expon and dist.unit is None and not dist.round and dist.pars.scale == 1) else 'agent' if self.agent_pars(key, dist) is not None else 'own'
        if self.crn_key(key) is not None and (how != 'own' or dist.resolved()): # The uniforms of this round, and the inverse: as rvs() would give them, without its bookkeeping
            if not len(uids):
                return np.array([], dtype=dist.hash_dtype or ss.dtypes.float) # Uniforms, as below
            slots = self.sim.people.slot_raw()
            rvs = ss.dists.hash_uniforms(dist.seed, self.draw_ind(), uids if slots is None else slots[uids], dtype=dist.hash_dtype)
            if how == 'uniform':
                return rvs
            elif how == 'expon':
                return ss.solvers.backends.draws.many['expon'](1.0, np.asarray(rvs, dtype=np.float64), np.empty(len(rvs))) # As dist.ppf(), without its checks
            elif how == 'agent':
                n, pars, kwds = self.agent_pars(key, dist, checked=True)
                pick = lambda d: {k:(v[uids] if isinstance(v, np.ndarray) and v.shape == (n,) else v) for k,v in d.items()} # These agents' values
                dist._pars = type(dist.pars)(pick(pars))
                if kwds is not None:
                    dist.dist.kwds = pick(kwds)
            return dist.ppf(rvs)
        ind = dist.ind
        dist.jump(to=self.draw_ind(), force=True)
        try:
            return dist.rvs(uids)
        finally:
            dist.jump(to=ind, force=True)

    def agent_pars(self, key, dist, checked=False):
        """
        For one of the engine's distributions whose parameters are values by sex (see agent_values() and is_lifelong()), its parameters for
        everyone (by UID), worked out as rvs() would and kept for the run, so a draw is the uniforms and the inverse; None for any other
        (whose draws rvs() makes)
        """
        if not checked: # Once, for draw()
            if dist._use_ppf is False or dist.unit is not None or dist.round:
                return None
            bys = [getattr(v, 'by', None) for v in dist.pars.values() if callable(v) and not isinstance(v, type)]
            if not bys or not all(by is not None and self.is_lifelong(by) for by in bys): # Read from each person or by group (when the sim is stratified by sex), the same for the run
                return None
        def pars(uids):
            dist.process_size(uids)
            dist.process_pars()
            return dict(dist._pars), None if dist.dist is None else dict(dist.dist.kwds)
        n = len(self.sim.people.uid.raw)
        held = self.lifelong.get(('pars', key))
        if held is None or held[0] < n: # Or people have been added since: each value for each agent, appended
            start = 0 if held is None else held[0]
            new = pars(ss.uids(np.arange(start, n)))
            join = lambda old, add: {k:(np.concatenate([old[k], v]) if isinstance(v, np.ndarray) and v.shape == (n - start,) else v) for k,v in add.items()}
            held = self.lifelong[('pars', key)] = (n, *new) if held is None else (n, join(held[1], new[0]), None if new[1] is None else join(held[2], new[1]))
        return held

    def crn_key(self, key, uids=None):
        """ As `ss.Dist.crn_key()`, for a compiled pass that makes the draws of one of the engine's distributions itself (for `uids`), keyed as `draw()` is; None if it cannot """
        dist = self.dists[key]
        diagnostics = self.sim.diagnostics
        if not ss.options.crn or dist.initialized is not True or dist.slots is None or (diagnostics and diagnostics.rvs is not None):
            return None
        if check_draws and uids is not None:
            self.check_drawn(key, uids)
        return (dist.seed, self.draw_ind(), np.dtype(dist.hash_dtype or ss.dtypes.float))

    def check_drawn(self, key, uids):
        """ With STARSIM_CHECK_DRAWS=1 (for testing): that none of some agents has been drawn from a stream already in this round """
        uids = np.unique(np.asarray(uids, dtype=np.int64))
        seen = self._drawn.get(key)
        if seen is not None and len(np.intersect1d(seen, uids)):
            raise RuntimeError(f'"{self.mod.name}" drew {key} twice in round {self.round} for {np.intersect1d(seen, uids)[:5]}')
        self._drawn[key] = uids if seen is None else np.union1d(seen, uids)
        return

    def stamp(self, name, uids, ti):
        """
        Date some agents' entering a state, at the engine's time `ti` (the start of the step for a move made then, `ti`
        of the loop less 1; its end for an infection by contact), and record the steps from it to the loop's index, so
        that `ti_X` reads as the step the move happened in (see `ss.TimeArr`), while `.raw` has the engine's time
        """
        arr = self.mod[f'ti_{name}']
        if isinstance(arr, ss.arrays.Unkept):
            return
        arr.raw[uids] = ti
        arr.shift.raw[uids] = self.index_of(ti) - ti if ti == ti else 0
        return

    def backdate(self, state, uids, entered, exit=None):
        """
        Date when some agents entered the state they are in, and when their scheduled exit from it is due, in the engine's steps (see
        `stamp()`): for people placed in a state from code who were partway through it, e.g. infected before the sim began. Their clocks of
        the state (`X.time`, `X.progress`) and their exit follow; an exit due by now is taken at the next step. As agents only.

        Args:
            state (State/str): the state they are in
            uids (uids): the agents
            entered (float/array): the step each entered it, before now
            exit (float/array): the step each is due to leave it by its scheduled exit (default: as drawn when they were placed, moved back as far as their entry)
        """
        name = state if isinstance(state, str) else state.name
        uids = ss.uids(uids)
        if not len(uids):
            return
        mod = self.mod
        entered = np.broadcast_to(np.asarray(entered, dtype=float), len(uids))
        arr = mod[f'ti_{name}']
        before = None if isinstance(arr, ss.arrays.Unkept) else arr.raw[uids]
        if before is not None:
            arr.raw[uids] = np.ceil(entered) # A time between steps, at the next step
        if name in self.scheduled:
            ti_exit = mod[f'ti_exit_{name}']
            if exit is None:
                start = before if before is not None else self.ti
                exit = ti_exit.raw[uids] - (start - entered)
            exit = np.broadcast_to(np.asarray(exit, dtype=float), len(uids))
            ti_exit.raw[uids] = np.ceil(np.maximum(exit, self.ti))
        return

    def index_of(self, ti):
        """
        The index of the step a move that the engine dates `ti` happens in, which is what `ti_X` reads: during a step,
        the loop's `ti`, both for a move at its start (dated `ti - 1`) and for an infection by contact at its end (dated
        `ti`); 0 for anyone placed at the start, however the engine dates them (see `seed_ti()`); and before the start
        (see `backfill()`), the step after `ti`, since every move there is taken at the start of its step
        """
        if self._backfill is not None:
            return ti + 1
        ti_loop = self.mod.t.ti
        if ti_loop == 0: # Placed at the start
            return 0
        return ti if ti == ti_loop else ti + 1

    def enter(self, state, uids, ti, tr=None, srcs=-1):
        """
        Put agents in a state, and schedule their exit from it, at the moment they enter (so `p=` is read then, and code after the step sees the exit)

        An agent who enters during a step after the state's competing rates were drawn is marked
        `late_<state>` with the share of a step of them it missed: half, for a move in the
        state-change row (which draws the other half then: see `finish_moves()`), and all of it for
        a move by an intervention. The step their scheduled exit is due in draws that share for them
        (see `step_state()`), which gives a rate competing with a dwell time the exposure the dwell
        time says.

        An agent infected in a step (or moved at once after it) into a source of infection is marked
        -0.5: on average they were infected halfway through the step, and infectious for the half
        step after that, which no transmission counts, since the step's has happened. So they have
        half a step of the competing rates at their first draw, in the next step (and, with a
        scheduled exit, the other half in the step it is due), and each stay counts in transmission
        for as many steps as it lasts, to first order. Otherwise each would count for half a step
        less, as if R0 were R0·(1 - γ·dt/2). A state that is not a source (a latent state) is not
        marked: there it would only make the stay half a step longer.
        """
        if not state.people or not len(uids):
            return
        mod = self.mod
        name = state.name
        self.put(name, uids, True)
        self.stamp(name, uids, ti)
        if self.lefts and not np.isscalar(srcs): # Dated as leaving the states they came from, for their stays (see ss.BoolState.stays)
            for i,arr in self.lefts.items():
                came = srcs == i
                if came.any():
                    mod[arr].raw[uids[came]] = ti
        for arr,inside in self.sets_of[name]: # Entering a set from outside it starts its clock
            outside = ~np.isin(srcs, inside)
            mod[arr][uids[np.broadcast_to(outside, len(uids))]] = ti
        if name in self.late:
            late = ti == self.ti and self._drawn_ti == self.ti
            mod[f'late_{name}'][uids] = -0.5 if self._infecting and name in self.halved else 0.0 if not late else 0.5 if self._row is not None else 1.0
        self.record(tr, uids, srcs, self.state_index[name])
        if name in self.scheduled:
            self.schedule(state, uids, entered=ti)
        return

    def start_phase(self, phase):
        """ Where the engine is in a step (0 for its state changes, 2 for its transmission): what happened at this point in the last step is no longer news """
        ti = self.ti
        self.batches = [b for b in self.batches if b.key[0] > ti - 1 or (b.key[0] == ti - 1 and b.key[1] > phase)]
        self._phase = (ti, phase)
        return

    def record(self, tr, uids, srcs, dst):
        """ Record who entered a state, from where (the index of the state each came from, or -1 for none), as an event: true from the moment of the move until the same point in the next step """
        ti = self.ti
        phase = self._phase[1] if self._phase[0] == ti else -1 # Moved by another module before this one's step: see start_phase()
        self.n_batches += 1
        self.batches.append(Bunch(tr=None if tr is None else tr.name, uids=uids, src=srcs, dst=dst, key=(ti, phase), also=False, n=self.n_batches))
        self.last_batch[None if tr is None else tr.name] = self.n_batches
        self._news = True
        if tr is not None and tr.name in self.mod.__dict__.get('remembered', ()): # Its memory, e.g. for tb.diagnosis.since
            self.mod[f'last_{tr.name}'][uids] = ti
            count = self.mod[f'count_{tr.name}']
            np.add.at(count.raw, uids, 1)
        if tr is not None and self.also: # Others an event also happens to, e.g. the fetuses of a mother at a delivery
            others = self.mod.also_happened(tr.name, uids)
            if others is not None and len(others):
                self.n_batches += 1
                self.batches.append(Bunch(tr=tr.name, uids=others, src=-1, dst=-1, key=(ti, phase), also=True, n=self.n_batches))
                self.last_batch[tr.name] = self.n_batches
        return

    def event_uids(self, tr=None, to=None, enter=None, exit=None, batches=None, unique=True):
        """
        The agents an event happened to, as UIDs: those who went through a transition (as `on=` sees them), or one of
        its branches (`to`), or who entered or left a set of states (given as names), from outside it
        """
        if tr is not None and to is None and batches is None and tr in self.graph.transitions: # As on= sees them
            uids = self.events.get(tr)
            return ss.uids() if uids is None else ss.uids(uids).unique() if unique else ss.uids(uids)
        out = []
        index = self.state_index
        trs = tr if isinstance(tr, (tuple, list)) else (tr,) # Several events at once, e.g. preg.loss
        for b in (self.batches if batches is None else batches):
            if tr is not None:
                if b.tr in trs and (to is None or b.dst == index[to]):
                    out.append(b.uids)
                continue
            inside = [index[name] for name in (enter if enter is not None else exit)]
            src_in = np.isin(b.src, inside)
            dst_in = np.bool_(b.dst in inside) # Not a Python bool, whose ~ is an integer
            keep = (dst_in & ~src_in) if enter is not None else (src_in & ~dst_in)
            if np.ndim(keep) == 0:
                if keep:
                    out.append(b.uids)
            elif keep.any():
                out.append(b.uids[keep])
        if not out:
            return ss.uids()
        out = ss.uids.concatenate(out)
        return out.unique() if unique else out

    def schedule(self, state, members, entered=None):
        """
        For the agents in a state who have just entered it (at `entered`, if the caller knows it; otherwise as dated): choose their branch, sample its dwell time, and schedule the exit

        Returns:
            Which of the members entered the state during the last step's moves (by a progression, dated at its start), after that step's
            competing rates were drawn, so that they have one draw of them to catch up on (see step_state()); None if there are none
        """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = self.exits_of[name] # Transitions that can be scheduled
        if not candidates:
            return None
        ti_exit = mod[f'ti_exit_{name}']
        is_new = np.isnan(ti_exit.raw[members])
        new = members[is_new]
        if not len(new):
            return None
        missed = None
        if entered is None:
            entered = mod[f'ti_{name}'].raw[new]
            if self.competing[name]:
                late = entered < self.ti # NaN (undated) is not
                if late.any():
                    missed = np.zeros(len(members), dtype=bool)
                    missed[np.flatnonzero(is_new)[late]] = True
            undated = np.isnan(entered) # Put in the state directly (e.g. sir.I[uids] = True), with no time of entering: they entered it now
            if undated.any():
                entered[undated] = self.ti
                self.stamp(name, new[undated], self.ti)
        best = np.full(len(new), np.inf)
        which = np.full(len(new), -1, dtype=np.int8)

        branches = self.branches_of(name)
        if branches: # Split on entry
            edges = self.branch_edges(state, new)
            rvs = self.draw(f'branch_{name}', new)
            for i,tr in enumerate(branches):
                chosen = (rvs >= edges[i]) & (rvs < edges[i+1])
                if tr.name in self.dwells:
                    best[chosen] = self.sample_dwell(tr, new[chosen]) + self.after_of(tr)
                else: # Nobody waits in this branch, beyond a minimum stay
                    best[chosen] = self.after_of(tr) or -np.inf
                which[chosen] = i
        for i,tr in enumerate(exits.timed, start=len(branches)): # Competing dwell times: the earliest wins
            dur = self.sample_dwell(tr, new) + self.after_of(tr)
            earlier = (dur < best) | (which < 0) # A dwell that does not end until its rate changes (e.g. outside a where=) is still this exit's, so it can be rescaled
            best[earlier] = dur[earlier]
            which[earlier] = i

        steps = self.whole_steps(new, best)
        loops = self.loops_of[name]
        if loops.any(): # Back to this state: taken again no sooner than the next step, so a dwell shorter than a step does not go round within one
            again = (which >= 0) & loops[np.maximum(which, 0)] & (best > 0)
            steps[again] = np.maximum(steps[again], 1)
        ti_exit.raw[new] = entered + steps # Exits happen at the start of a step
        mod[f'exit_{name}'][new] = which
        modified = self.rescaled_of(name)
        if modified and f'mult_{name}' in mod.__dict__: # How fast each exit ran when its dwell time was drawn, so a later change can rescale what is left of it: see rescale_exits()
            mults = np.ones(len(new))
            for i in modified:
                chosen = which == i
                if chosen.any():
                    mults[chosen] = self.speed(candidates[i], new[chosen])
            mod[f'mult_{name}'][new] = mults
        return missed

    def branch_edges(self, state, uids):
        """ Where each branch out of a state starts, as a cumulative probability, for some agents entering it (one row per branch, and its end): with their modifiers, and each branch's where= closed to the others """
        key = ('branch', state.name)
        fixed = self.__dict__.setdefault('_fixed', {}).get(key)
        if fixed is None:
            branches = self.branches_of(state.name)
            fixed = self._fixed[key] = all(tr is None or ((tr.p is None or self.is_fixed(tr.p[0])) and self.unmodified('branch', tr.name) and self.is_fixed(getattr(tr, 'where', None))) for tr in branches)
        return self.fixed_edges(key, fixed, lambda u: self.branch_edges_for(state, u), uids)

    def branch_edges_for(self, state, uids):
        """ As branch_edges(), worked out for these agents """
        branches = self.branches_of(state.name)
        ps = [None if tr.p is None or tr.split else np.broadcast_to(self.eval_for(tr.p[0], uids), len(uids)) for tr in branches] # A split between several destinations takes what the branches leave over
        mults = self.where_mults(branches, state, [self.branch_mult(tr, uids) for tr in branches], uids)
        ps = [np.broadcast_to(p, len(uids)) for p in self.branch_shares(branches, ps, mults)]
        return np.cumsum([np.zeros(len(uids))] + ps, axis=0)

    def rescaled_of(self, name):
        """ The exits of a state (by index) whose dwell times are rescaled when their speed changes (see `rescaled()`), worked out once the sim is initialized """
        if self._rescaled is None or not self.sim.initialized:
            rescaled = {key:[i for i,tr in enumerate(exits) if self.rescaled(tr)] for key,exits in self.exits_of.items()}
            if not self.sim.initialized:
                return rescaled[name]
            self._rescaled = rescaled
        return self._rescaled[name]

    def rescaled(self, tr):
        """ Whether what is left of a transition's dwell times is rescaled when its speed changes: if it has modifiers, or a duration or rate that can change (see `varies()`) """
        if tr.name not in self.dwells:
            return False
        return bool(self.sim.modifiers.get((self.mod.name, 'transition', tr.name))) or self.varies(self.dwells[tr.name]) or bool(self.adjust_dims[tr.name] & self.changing_dims)

    @property
    def changing_dims(self):
        """ The dimensions whose level a person can change during the run: by age, a transfer, a move= on a transition or a delivery, or migration between places (worked out once the mixing is known) """
        if getattr(self, '_changing_dims', None) is not None:
            return self._changing_dims
        sim = self.sim
        out = {d for d,strat in enumerate(sim.groups.strats) if strat.kind == 'age'}
        for mod in sim.declared:
            out |= {sim.groups.dim_index(tf.dim) for tf in mod.graph.transfers.values()}
        out |= {d for tr in self.graph.transitions.values() for d,level in (self.moves(tr) or ())}
        names = [strat.name for strat in sim.groups.strats]
        out |= {names.index(dim) for iv in sim.interventions() for dim in list(getattr(iv, 'move', None) or {}) + list(getattr(iv, 'increment', None) or {}) if dim in names} # A delivery's move= or increment=
        if getattr(sim.mixing, 'migration', None) is not None:
            out.add(sim.mixing.space)
        if getattr(sim, 'initialized', False):
            self._changing_dims = out
        return out

    def varies(self, dw):
        """ Whether the mean of a dwell time can change during the run: a duration or rate that is a function or varies in time, or a parameter with modifiers """
        if dw.dist is not None:
            return bool(getattr(dw, 'mods', None)) # A distribution with modifiers: see dist_dwell()
        p = dw.par
        if callable(p.raw):
            return True
        return bool(p.name and not p.literal and self.sim.modifiers.get((self.mod.name, 'par', p.name)))

    def speed(self, tr, uids):
        """ How fast a scheduled exit runs for some agents: the multiplier on its rate, over its mean dwell time if that can change """
        out = np.broadcast_to(self.adjusted(tr.name, uids), len(uids)).astype(float)
        if self.varies(self.dwells[tr.name]):
            self._uids = uids
            mean = self.dwell_mean(tr.name)
            mean = mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids]
            out = out/np.maximum(mean, 1e-12) # A mean of inf (a rate of 0) is a speed of 0
        return out

    def rescale_exits(self, state, uids):
        """ When the speed of an exit changes after people's dwell times were drawn (a modifier, or a duration or rate that changes), what is left of each is rescaled by the change """
        mod = self.mod
        name = state.name
        candidates = self.exits_of[name]
        modified = self.rescaled_of(name)
        if not modified or not len(uids) or f'mult_{name}' not in mod.__dict__:
            return
        which = mod[f'exit_{name}'].raw[uids]
        ti = self.ti
        for i in modified:
            sel = uids[which == i] if len(modified) > 1 or len(candidates) > 1 else uids
            if not len(sel):
                continue
            raw = self.speed(candidates[i], sel)
            now = np.maximum(raw, 1e-9)
            before = mod[f'mult_{name}'].raw[sel]
            stalled = before <= 0 # Their rate was 0 when they entered (e.g. outside a where=), so none of their dwell has passed
            changed = np.abs(now - before) > 1e-6*np.abs(before) # Not exactly, since the speed is kept in single precision
            if stalled.any():
                changed = np.where(stalled, raw > 0, changed)
            if changed.any():
                sel, now, before, stalled = sel[changed], now[changed], before[changed], stalled[changed]
                ti_exit = mod[f'ti_exit_{name}'].raw[sel]
                new_exit = np.empty(len(sel))
                ok = ~stalled
                new_exit[ok] = ti + np.maximum(ti_exit[ok] - 0.5 - ti, 0)*before[ok]/now[ok] # A scheduled exit is a whole step, so what is left is taken from the middle of it
                if stalled.any(): # Now that their rate is not 0, a dwell is drawn from now
                    new_exit[stalled] = ti + self.whole_steps(sel[stalled], self.sample_dwell(candidates[i], sel[stalled]))
                lost = ok & np.isinf(ti_exit) # Never (a rate of 0, which has changed), so the dwell time is gone: drawn again from now, less what has passed of it since entering (as if it had stalled)
                if lost.any():
                    passed = (ti - mod[f'ti_{name}'].raw[sel[lost]])*before[lost]/now[lost]
                    new_exit[lost] = ti + self.whole_steps(sel[lost], np.maximum(self.sample_dwell(candidates[i], sel[lost]) - passed*self.dt, 0))
                mod[f'ti_exit_{name}'].raw[sel] = np.ceil(new_exit) # The step it falls in
                mod[f'mult_{name}'][sel] = now
        return

    def dist_dwell(self, name):
        """
        For a dwell time that is a distribution whose parameter has modifiers (see `Engine.dwell_mods()`): the key of the distribution drawn
        from now (that of the last set= that is active, else its own) and its mean, and the multiplier and addend from multiply= and add=
        (each a scalar, or one value per agent)
        """
        dw = self.dwells[name]
        active = [m for m in dw.mods if m.active]
        sets = [m for m in active if m.op == 'set']
        new = dw.sets[next(i for i,m in enumerate(dw.mods) if m is sets[-1])] if sets else sc.obj(key=f'dwell_{name}', mean=dw.mean)
        mult = ss.connectors.combine([m for m in active if m.op == 'multiply'], 1.0, self)
        add = ss.connectors.combine([m for m in active if m.op == 'add'], 0.0, self)
        return new.key, new.mean, mult, add

    def dwell_mean(self, name):
        """ The current mean dwell time of a transition, in the sim's unit, including the modifiers of a distribution (see `dist_dwell()`) """
        if getattr(self.dwells[name], 'mods', None):
            key, mean, mult, add = self.dist_dwell(name)
            if mean is None: # By group: for the agents being evaluated (see speed())
                mean = self.agent_dwell_mean(name, self._uids)
            return mean*mult + add
        return super().dwell_mean(name)

    def sample_dwell(self, tr, uids):
        """ Sample a dwell time for each agent, in the sim's unit """
        dw = self.dwells[tr.name]
        if getattr(dw, 'mods', None): # A distribution whose parameter has modifiers
            key, mean, mult, add = self.dist_dwell(tr.name)
            pick = lambda val: val if np.ndim(val) == 0 or len(val) == len(uids) else np.asarray(val)[uids]
            rvs = np.maximum(self.draw(key, uids)*pick(mult) + pick(add), 0)
        else:
            rvs = self.draw(f'dwell_{tr.name}', uids)
        if dw.dist is not None and getattr(dw, 'scale', None) is not None: # A distribution times a factor: each person's draw, scaled by theirs
            rvs = np.asarray(rvs, dtype=float)*self.values_for(dw.scale, uids)
        if dw.dist is None:
            with self.evaluating(uids=uids, for_uids=True): # So each part of an expression is for these agents
                mean = self.dwell_mean(tr.name)
            rvs = rvs*(mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids])
        adjust = self.adjusted(tr.name, uids)
        if np.ndim(adjust) == 0 and adjust == 1.0: # The usual case
            return np.asarray(rvs, dtype=float)
        with np.errstate(divide='ignore'): # A rate of 0 (e.g. outside a where=) is a dwell that does not end until it changes: see rescale_exits()
            return np.asarray(rvs, dtype=float)/adjust # An adjustment multiplies the rate, so divides the duration

    def branches_of(self, name):
        """ The branches out of a state, chosen on entry: those with a share, and the one that takes the rest """
        exits = self.graph.exits[name]
        return exits.branches + sc.tolist(exits.remainder)

    #%% People part way through at the start (ss.Pregnancy's burnin=)

    def backfill_burnin(self, sp):
        """
        A module's people who start part way through (`ss.Pregnancy(burnin=True)`: see its `burnin_plan()`): each agent of
        the first state with their own chance of having entered (`sp.share()`), at a step uniform over the `sp.steps` before
        the start, is run forward to the start by the module's own spontaneous transitions (see `backfill()`); those no
        longer in the target at the start are put back, as if never there, and those who died are removed. One draw for
        each agent, from the module's own common random numbers, says both whether and when
        """
        mod = self.mod
        sim = self.sim
        people = sim.people
        share = np.asarray(sp.share(), dtype=float)
        pool = self.members(self.graph.first)
        p = share[pool] if share.ndim else np.full(len(pool), share)
        u = self.draw('burnin', pool)
        yes = u < p
        chosen = pool[yes]
        t0 = -np.minimum((u[yes]/p[yes]*sp.steps).astype(int), sp.steps - 1) - 1 # Uniform over the steps before the start, given that they entered
        levels = {strat.attr:people[strat.attr].raw[chosen].copy() for strat in sim.groups.strats if strat.attr in people.states} # A counter moved (e.g. parity) is undone for those reset
        self.backfill(chosen, sp.tr, t0)
        inside = np.zeros(len(chosen), dtype=bool)
        for name in sp.target:
            inside |= mod[name].raw[chosen]
        back = ~inside & self.alive_in_states(chosen)
        self.reset_people(chosen[back]) # Left before the start, e.g. delivered: as if never there
        if sp.get('after') is not None: # The module's own records of what happened before the start to those kept, e.g. ss.Pregnancy's parity
            sp.after(chosen[~back], {attr:vals[~back] for attr,vals in levels.items()})
        if levels and back.any():
            for attr,vals in levels.items():
                people[attr][chosen[back]] = vals[back]
            sim.groups.groups_changed(chosen[back])
        ss.note(sp.note, sim=sim)
        people.step_die() # Those who died before the start are gone before it is counted
        people.remove_dead()
        self.live = None # Counted again
        self.unscheduled = None # Writes are not watched while nothing is counted
        return

    def alive_in_states(self, uids):
        """ Whether each of some agents is in a state of this module """
        out = np.zeros(len(uids), dtype=bool)
        for state in self.graph.people_states:
            out |= self.mod[state.name].raw[uids]
        return out

    def reset_people(self, uids):
        """ Put agents back in the first state at the start, with no attributes: as if never seeded """
        if not len(uids):
            return
        mod = self.mod
        self.new_round()
        first = self.graph.first.name
        for state in self.graph.people_states:
            inds = uids[mod[state.name].raw[uids]]
            mod[state.name][inds] = False
            self.clear_exit(state.name, inds)
            mod[f'ti_{state.name}'][inds] = np.nan
        for key,arr in self.sets.items():
            mod[arr][uids] = np.nan
        for a in getattr(self, 'attrs', {}).values():
            if a.kind not in ['derived', 'trait']:
                mod[a.name][uids] = a.default if a.init_value is None else self.draw_attr(a, uids, value=a.init_value)
        mod[first][uids] = True
        self.stamp(first, uids, 0)
        return

    def backfill(self, seeds, entry, t0):
        """
        Run some agents' natural history from before the start: each enters at the step `t0` (negative), along the
        transition `entry` out of the first state, and from then the module's spontaneous transitions move them, a step
        at a time, to the start, with draws of their own (see `draw()`)
        """
        if not len(seeds):
            return
        mod = self.mod
        graph = self.graph
        first = graph.first
        saved = mod.t.ti
        saved_round = self.round # The rounds of the step they are back-dated to (see new_round())
        self._backfill = True
        active = np.zeros(len(mod.rel_sus.raw), dtype=bool)
        self._moved = []
        try:
            for k in range(int(t0.min()), 0):
                mod.t.ti = self._ti = k
                new = seeds[t0 == k]
                if len(new):
                    active[new] = True
                    self.move(entry, first, new)
                    self.finish_moves(k)
                self.new_round() # As at the start of a step
                for state in graph.people_states:
                    name = state.name
                    uids = self.members(state)
                    uids = uids[active[uids]]
                    if not len(uids) or name not in self.listed:
                        continue
                    candidates = self.exits_of[name]
                    if candidates: # Scheduled exits that are due
                        unscheduled = uids[np.isnan(mod[f'ti_exit_{name}'].raw[uids])]
                        if len(unscheduled):
                            self.schedule(state, unscheduled)
                        due = mod[f'ti_exit_{name}'].raw[uids] <= k
                        which = mod[f'exit_{name}'][uids]
                        for i,tr in enumerate(candidates):
                            self.move(tr, state, uids[due & (which == i)])
                        uids = uids[~due]
                    if self.competing[name] and len(uids):
                        self.draw_competing(state, uids, self.dt, key=f'exit_{name}')
                self._drawn_ti = k
                self._row = np.zeros(len(mod.rel_sus.raw), dtype=bool)
                self.finish_moves(k)
                self._row = None
        finally:
            mod.t.ti = saved
            self.round = saved_round
            self._ti = None
            self._backfill = None
            self._row = None
            self._moved = []
        for name in self.flowcounts: # Not part of the run
            self.flowcounts[name] = 0.0
        self.flow_groups = sc.obj()
        self.events = dict()
        self.batches = []
        self._died = []
        return

    #%% Stepping

    def step_state(self):
        """
        Everything that is not contact-mediated, at the start of the step: who leaves each state now (by a scheduled exit that is due,
        or by its competing rates), then the same-step cascade (see `finish_moves()`), births, and transfers; and the counts
        """
        mod = self.mod
        ti = self.ti
        dt = self.dt
        self._moved = []
        self.new_round()
        self.start_phase(0)
        self.place_newcomers()
        self.renew_events()
        self.step_lifetables()
        quick = set() if self.unscheduled is None or self.late or any(self.competing[name] for name in self.listed) else {name for name in self.listed if name not in self.unscheduled and self.exits_of[name] and not self.rescaled_of(name)} # States with only scheduled exits, all scheduled already: who leaves them is found in one pass over the agents, with no need to list who is in them (see leave())
        members = self.start_counts([name for name in self.listed if name not in quick])
        self.step_counts(dt) # States that are not people, from the people at the start of the step, so that a rate of people that uses a count (a bite rate) takes it once advanced
        if self.held is not None:
            self.step_held(dt)
        missed = {name:self.schedule(self.graph.states[name], members[name]) if self.unscheduled is None or name in self.unscheduled else None for name in members if name in self.listed} # Who has a draw of the competing rates to catch up on, by state: see schedule()
        self.unscheduled = set() if self.live is not None else None
        for name in self.listed:
            if name in quick:
                self.leave(self.graph.states[name])
            elif len(members[name]):
                self.leave(self.graph.states[name], members[name], missed[name])
        self._drawn_ti = ti
        self._members = None
        self._row = np.zeros(len(mod.rel_sus.raw), dtype=bool) # Who has had the competing rates of the state they entered in this row: see finish_moves()
        self.finish_moves(ti)
        self._row = None
        self.step_per(ti)
        self.step_births(dt)
        self.step_transfers(dt)
        return

    def renew_events(self):
        """ At the start of the step: who went through a transition in the last step is no longer news, and who an intervention moved since is """
        graph = self.graph
        for name,tr in graph.transitions.items():
            if not tr.contact:
                self.events.pop(name, None)
        for name,uids in self.forced.items(): # Moved by an intervention in this step (before the disease's transitions): news from now on, whatever order the interventions run in
            if not graph.transitions[name].contact: # Those infected (disease.infect()) are news with the step's infections: see step()
                self.events[name] = ss.uids.concatenate([self.events[name], uids]) if name in self.events else uids
                self._news = True
        self.forced = {name:uids for name,uids in self.forced.items() if graph.transitions[name].contact}
        return

    def start_counts(self, listed):
        """ The number in each state at the start of the step, which the rates read whatever order they are taken in; returns who is in each state that is `listed` (some of those stepped through) """
        mod = self.mod
        sim = self.sim
        people_states = self.graph.people_states
        if sim.groups.n_groups > 1:
            sim.groups.group_raw() # Groups that change by themselves (age groups) are moved between the counts now: see moved_groups()
        if sim.groups.group_fold is not None: # Counts by place taken from the agents (see local_count()): counted again at each step
            self.live = None
            self.unscheduled = None # Writes are not watched while nothing is counted
        elif self.live is not None and check_counts:
            self.check_counts()
        if self.live is None:
            listed = self.listed if sim.mixing.space is None or sim.groups.group_fold is None else [state.name for state in people_states]
            members = self.census(listed) # Who is where before anyone moves, in one pass over the agents
        else: # Counted already: only those asked for are listed
            members = {name:ss.uids(ss.solvers.backends.passes.flatnonzero(mod[name].raw)) for name in listed}
        self.start = {name:vals.copy() for name,vals in self.live.items()}
        self.start_weighted = {state.name:self.weighted_count(state) for state in people_states} if sim.people.scale.weighted else None
        self._members = members
        return members

    def leave(self, state, uids=None, missed=None):
        """
        At the start of the step: who of the agents in a state leaves it now, by a scheduled exit that is due, or by its competing rates
        (one draw per agent); `missed` are those who have a draw of the competing rates to catch up on (see `schedule()`). With no `uids`,
        the state has only scheduled exits, all scheduled already (see `step_state()`), so who is due is found in one pass over the agents

        An agent marked `late_<state>` (see `enter()`) entered after the draws of the step it entered in: the share of a step it missed
        is drawn in the step its exit is due, so it faces the competing rates for as long as the dwell time says; one marked as infected
        in the last step (less than 0) has half a step at this draw, and the other half in the step its exit is due.
        """
        mod = self.mod
        name = state.name
        dt = self.dt
        candidates = self.exits_of[name]
        if uids is None:
            exits = mod[f'ti_exit_{name}']
            due = ss.uids(np.flatnonzero(mod[name].raw & ~(exits.raw > self.ti))) # And not set (NaN), so those put in the state through its NumPy array (which is not watched) are among them
            unset = np.isnan(exits.raw[due])
            if unset.any(): # Scheduled now, as at the start of the step (see schedule()), and taken if that is due
                self.schedule(state, due[unset])
                due = due[exits.raw[due] <= self.ti]
            which = mod[f'exit_{name}'].raw[due] if len(candidates) > 1 else None
            for i,tr in enumerate(candidates):
                self.move(tr, state, due if which is None else due[which == i])
            return
        competing = self.competing[name]
        late = missed
        flagged = None
        early = None
        if name in self.late:
            share = mod[f'late_{name}'].raw[uids]
            flagged = share > 0
            early = share < 0 # Infected last step: half a step at this draw, as for an arrival in this row (who, if they leave, have the next state's half step on arriving there)
            early = early if early.any() else None
        if candidates: # Scheduled exits that are due
            self.rescale_exits(state, uids)
            due = mod[f'ti_exit_{name}'].raw[uids] <= self.ti # Everyone here has been scheduled, so none is .nan
            which = mod[f'exit_{name}'][uids]
            if flagged is not None and (flagged & due).any(): # Their missed draw is taken in the step their exit is due
                late = (flagged & due) if late is None else (late | (flagged & due))
            if late is not None or early is not None: # Those placed in the state without a schedule last step face the competing rates for that step too, before their scheduled exit
                late = np.zeros(len(uids), dtype=bool) if late is None else late
                drawn = ~due | late
                left = np.zeros(len(uids), dtype=bool)
                double = late & ~due if flagged is None else late & ~due & ~flagged
                span = np.where(double, 2.0, 1.0) if flagged is None else np.where(double, 2.0, np.where(flagged & due, share, 1.0))
                if early is not None:
                    span[early] = 0.5
                left[np.flatnonzero(drawn)] = self.draw_competing(state, uids[drawn], span[drawn]*dt)
                due &= ~left
                competing = None
                if early is not None: # The other half in the step their exit is due, as for an arrival in the state-change row
                    mod[f'late_{name}'][uids[early]] = np.where(due | left, 0.0, 0.5)[early]
            if flagged is not None and flagged.any():
                mod[f'late_{name}'][uids[flagged & due]] = 0.0 # Caught up; the others keep the flag until their exit is due
            for i,tr in enumerate(candidates):
                self.move(tr, state, uids[due & (which == i)])
            if competing: # Those still here face them
                uids = uids[~due]
        elif early is not None and competing: # With only competing rates
            self.draw_competing(state, uids[early], 0.5*dt, key=f'entry_{name}')
            mod[f'late_{name}'][uids[early]] = 0.0
            uids = uids[~early]
        if competing and len(uids): # Competing rates: one draw per agent
            self.draw_competing(state, uids, dt)
        return

    def draw_competing(self, state, uids, dt, key=None):
        """ The competing rates out of a state, in one draw per agent over a time `dt` (a number, or one for each agent): moves those who leave, and returns which they are """
        with self.evaluating(memo={} if len(self.competing[state.name]) > 1 else None): # A part that several rates share (e.g. the EIR, indoors and outdoors) is worked out once
            rates = [(tr, self.eval_for(tr.rate, uids)*self.adjusted(tr.name, uids)) for tr in self.competing[state.name]]
        rates = [(tr, rate) for tr,rate in rates if np.ndim(rate) or rate > 0] # A rate of 0 for everyone (e.g. ss.Pregnancy's infant_rate by default) moves nobody, so it is left out of the draw
        competing = [tr for tr,rate in rates]
        stays = [self.stayed(tr, state, uids) for tr,rate in rates]
        same = key is None and len(uids) and np.ndim(dt) == 0 and all(np.ndim(rate) == 0 for tr,rate in rates) and all(np.ndim(s) == 0 for s in stays) # The same chances for everyone: worked out once
        if same: # As numbers rather than arrays, since this is most calls, and each is small
            hazards = [(max(float(rate), 0.0)*float(stay) if stay > 0 else 0.0) for (tr,rate),stay in zip(rates, stays)] # (Not a product with stay, which would make an infinite rate NaN for those who can't take it yet)
            total = sum(hazards)
            if np.isinf(total): # An infinite rate (e.g. -log(1 - p) of a probability of 1) is certain within the step, shared among the infinite rates
                hazards = [float(np.isinf(h)) for h in hazards]
                total, p_exit = sum(hazards), 1.0
            else:
                p_exit = -np.expm1(-total*dt)
            edges = np.cumsum([0.0] + hazards)*(p_exit/total if total > 0 else 0.0)
        else:
            n = len(uids)
            hazards = [np.broadcast_to(np.maximum(rate, 0), n)*stay if np.ndim(stay) == 0 else np.where(stay > 0, np.maximum(rate, 0), 0.0) for (tr,rate),stay in zip(rates, stays)] # (Not a product with stay, which would make an infinite rate NaN for those who can't take it yet)
            total = np.sum(hazards, axis=0) if hazards else np.zeros(n)
            p_exit = -np.expm1(-total*dt)
            certain = np.isinf(total)
            if certain.any(): # An infinite rate is certain within the step, shared among the infinite rates
                hazards = [np.where(certain, np.isinf(h), h) for h in hazards]
                total = np.sum(hazards, axis=0)
                p_exit = np.where(certain, 1.0, p_exit)
            share = np.divide(p_exit, total, out=np.zeros(n), where=total>0)
            edges = None # Worked out in the compiled pass below, unless it can't be taken
        if np.isnan(total).any():
            bad = [tr.name for tr,h in zip(competing, hazards) if np.isnan(h).any()]
            raise ValueError(f'the rate of {sc.strjoin(bad)} out of {state.name} is NaN for some people: it reads a clock or an attribute where it has no value.')
        crn = self.crn_key(f'exit_{state.name}', uids) if key is None and len(uids) else None
        if crn is not None: # The usual case: each agent's draw is made in one compiled pass, as rvs() would make it, so only those who leave are listed
            if same:
                pos, which = ss.solvers.backends.passes.competing_exits(uids, self.sim.people.slot_raw(), *crn, p_exit, edges)
            else: # With the edges of each agent's exits worked out in it, as np.cumsum() of the hazards times p_exit/total
                pos, which = ss.solvers.backends.passes.competing_agent_exits(uids, self.sim.people.slot_raw(), *crn, p_exit, share, np.array(hazards, dtype=float).reshape(len(hazards), n))
            leaving = np.zeros(len(uids), dtype=bool)
            leaving[pos] = True
            for i,tr in enumerate(competing):
                self.move(tr, state, uids[pos[which == i]])
            return leaving
        if edges is None:
            edges = np.cumsum([np.zeros(n)] + hazards, axis=0)*share
        rvs = self.draw(f'exit_{state.name}' if key is None else key, uids) # Drawn on entry as well, so perhaps more than once a step
        leaving = rvs < p_exit
        for i,tr in enumerate(competing):
            self.move(tr, state, uids[leaving & (rvs >= edges[i]) & (rvs < edges[i+1])])
        return leaving

    def after_of(self, tr):
        """ A transition's minimum stay (`after=`), in the sim's unit """
        return 0.0 if getattr(tr, 'after', None) is None else float(self.convert(tr.after))

    def stayed(self, tr, state, uids):
        """ Whether a transition out of a state can happen for each agent, given how long they have been in it: after its minimum stay (`after=`) """
        after = self.after_of(tr)
        if not after:
            return 1.0
        stay = (self.ti - self.mod[f'ti_{state.name}'].raw[uids])*self.dt
        return (stay >= after - 1e-9).astype(float)

    def step_lifetables(self):
        """
        Deaths from a life table (`ss.death(lifetable=)`): each person's age at death is drawn once, given their age, the first step they are alive; they die in the step it falls in

        Each person is filed under the step they die in when it is drawn, so a step takes only those who die
        in it, rather than going over everyone.
        """
        if not self.lifetables:
            return
        mod = self.mod
        people = self.sim.people
        ti = self.ti
        n = people.uid.len_used
        alive = people.alive.raw
        dt = self.mod.t.dt.to('year').value if isinstance(self.mod.t.dt, ss.TimePar) else self.dt
        for name,table in self.lifetables.items():
            tr = self.graph.transitions[name]
            if table is None: # The agents drawn so far (by UID); most of those still to die, in order of the step they die in (and how many have been taken); those drawn since, in no order; those due who have not died yet; and those to draw again
                empty, empty32 = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32) # Kept as 32-bit integers, to save memory
                table = self.lifetables[name] = Bunch(n=0, uids=empty32, steps=empty32, done=0, new_uids=empty32, new_steps=empty32, carry=empty, redraw=empty)
            new = np.concatenate([table.redraw, np.arange(table.n, n)]) # Those with no age at death yet: made since the last step, or drawn as not a number
            table.n = n
            new = ss.uids(new[alive[new]])
            table.redraw = new[:0]
            if len(new):
                age = np.asarray(people.age.raw[new], dtype=float)
                times = ti + (tr.lifetable.draw(age, self.draw(f'life_{name}', new)) - age)/dt
                drawn = ~np.isnan(times)
                table.redraw = new[~drawn] # A time of death that is not a number is drawn again next step, as before
                table.new_uids = np.concatenate([table.new_uids, new[drawn].astype(np.int32)])
                table.new_steps = np.concatenate([table.new_steps, np.floor(np.minimum(times[drawn], 2**31 - 1)).astype(np.int32)]) # The step their time of death falls in
                if len(table.new_uids) > max(10_000, (len(table.uids) - table.done)//10): # Put in order with the rest from time to time
                    uids = np.concatenate([table.uids[table.done:], table.new_uids])
                    steps = np.concatenate([table.steps[table.done:], table.new_steps])
                    order = np.argsort(steps, kind='stable')
                    table.uids, table.steps, table.done = uids[order], steps[order], 0
                    table.new_uids, table.new_steps = uids[:0], steps[:0]
            end = np.searchsorted(table.steps, ti, side='right')
            now = table.new_steps <= ti
            due = np.sort(np.concatenate([table.carry, table.uids[table.done:end], table.new_uids[now]]))
            table.done = end
            if now.any():
                table.new_uids, table.new_steps = table.new_uids[~now], table.new_steps[~now]
            due = ss.uids(due[alive[due]]) # Those alive whose time of death falls in this step or before it
            placed = np.zeros(len(due), dtype=bool)
            for state in self.graph.people_states:
                inside = mod[state.name].raw[due]
                placed |= inside
                self.move(tr, state, due[inside])
            table.carry = due[~placed] # Anyone due who is in no state does not die by it yet, and is due again next step, as before
        return

    def uncount_deaths(self):
        """ Take out of this step's death flows anyone who died of something else, when another module requested their death in the same step too (see `People.request_death()`) """
        died, self._died = getattr(self, '_died', []), []
        people = self.sim.people
        mine = people.causes.index(self.mod.name) if self.mod.name in people.causes else -1
        for tr,srcs,uidlist in died:
            for src,uids in zip(srcs, uidlist):
                other = uids[people.cause_of_death.raw[uids] != mine]
                if len(other):
                    for key in dict.fromkeys([tr.name, tr.deaths]): # A split with a state counts its deaths in its flow to ss.null too
                        self.flowcounts[key] -= self.n_of(other)
                        if key in self.flow_groups:
                            self.flow_groups[key] = self.flow_groups[key] - np.bincount(self.groups(other, src), weights=self.weights(other), minlength=self.sim.groups.n_groups)
        return

    def update_results(self):
        self.uncount_deaths()
        return super().update_results()

    def die(self, uids):
        """ People who died this step (of anything, in any module) leave every state, and are taken out of the counts recorded for it """
        if not len(uids):
            return
        if getattr(self, 'derived_res', None):
            self.derived_stocks(uids=uids)
        if self.live_total is not None: # Counted as alive no longer
            counted = uids[uids < self.n_counted]
            self.live_total = self.live_total - (np.bincount(self.sim.groups.group_raw()[counted], minlength=self.sim.groups.n_groups) if self.sim.groups.n_groups > 1 else len(counted))
        for state in self.graph.people_states:
            inds = uids[self.mod[state.name][uids]]
            if len(inds): # Each by their weight, as record_stocks() counted them
                self.stocks[state.name] = self.stocks.get(state.name, 0) - self.n_of(inds)
                if state.name in self.stock_groups:
                    self.stock_groups[state.name] = self.stock_groups[state.name] - np.bincount(self.groups(inds, state), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
            self.mod[state.name][uids] = False
        return

    def step_births(self, dt):
        """ Inflows of people (`ss.birth`): a number of new agents in each place, drawn from the expected number """
        sim = self.sim
        d = sim.mixing.space
        for tr in self.graph.inflows:
            if not tr.people or getattr(tr, 'per', None) is not None: # Driven by a flow: see step_per()
                continue
            with self.evaluating(by_group=True):
                adjust = 1.0 if self.adjust[tr.name] is None else self.adjust[tr.name]
                amount = np.maximum(self.eval(tr.rate)*adjust*dt*self.inflow_split(tr), 0)/self.scale # Agents, by group
                ps = [np.broadcast_to(p, sim.groups.n_groups) for p in self.branch_probs(tr)] # An inflow split between several states, e.g. ss.null >> (0.9*S + 0.1*X)
            amount = np.broadcast_to(amount, sim.groups.n_groups)
            places = sim.groups.group_coords[:, d] if d is not None else np.zeros(sim.groups.n_groups, dtype=int)
            expected = np.bincount(places, weights=amount)
            born = self.rng.poisson(expected) # By place; the other levels are drawn as people are born into them
            enter = []
            for place,k in enumerate(born):
                if k:
                    split = [np.bincount(places, weights=amount*p)[place]/expected[place] for p in ps]
                    enter += [(place, (dst.name,), n) for dst,n in zip(tr.dst, self.rng.multinomial(k, split) if len(ps) > 1 else [k]) if n]
            new = self.add_people(enter, self.ti, f'born_{tr.name}', newborn=not tr.migration)
            self.fire_entered(new, tr)
            self.flowcounts[tr.name] += self.n_of(new)
            if len(new): # An event, e.g. on=measles.births, as for births driven by a flow (see step_per())
                for dst in tr.dst:
                    inds = new[self.mod[dst.name].raw[new]]
                    if len(inds):
                        self.record(tr, inds, -1, self.state_index[dst.name])
                self.events[tr.name] = new if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], new])
            if sim.groups.n_groups > 1:
                self.flow_groups[tr.name] = np.bincount(self.groups(new), weights=self.weights(new), minlength=sim.groups.n_groups)
        return

    def transfer_uids(self, name, tf):
        """ The agents who can move by a transfer now (at a level it moves people from, and in its states), and their levels """
        sim = self.sim
        mod = self.mod
        strat = sim.groups.strats[sim.groups.dim_index(tf.dim)]
        if strat.kind == 'age':
            errormsg = f'The transfer "{name}" cannot move agents between age groups: agents get older by themselves'
            raise ValueError(errormsg)
        d = sim.groups.dim_index(tf.dim)
        uids = sim.people.auids
        level = sim.groups.level_raw(d)[uids] # By level, not value: people of a numeric dimension hold values within each level
        if tf.increment is not None: # A counter: everyone who can move up it (or down)
            keep = strat.shifted(tf.increment)[level] != level
        else:
            keep = (level == strat.index(tf.src)) if tf.src is not None else (level != strat.index(tf.dst))
        states = [name for name in (tf.states or [s.name for s in self.graph.people_states]) if self.applies[name][d]] # Only people in states that have the dimension can move along it
        if len(states) < len(self.graph.people_states):
            in_states = mod[states[0]]
            for state in states[1:]:
                in_states = in_states | mod[state]
            keep &= in_states.raw[uids]
        return strat, uids[keep], level[keep]

    def schedule_transfer(self, name, uids, dt, ti):
        """ Draw the time before a transfer for people who can move by it from the end of step `ti`; a modifier on it speeds it up, as it was when drawn """
        if len(uids):
            with np.errstate(divide='ignore'): # Stopped by a modifier: never
                dur = self.draw(f'transfer_{name}', uids)/self.mult('transition', name, uids)
            self.mod[f'ti_move_{name}'][uids] = ti + self.whole_steps(uids, dur, dt) # As for a dwell time
        return

    def step_transfers(self, dt):
        """ Movement between the levels of a stratification: the agent's property changes, and their disease state does not """
        sim = self.sim
        mod = self.mod
        self.new_round()
        for name,tf in self.graph.transfers.items():
            strat, uids, level = self.transfer_uids(name, tf)
            arr = sim.people[strat.attr]
            self.events[name] = ss.uids() # The people who moved this step, as an event (e.g. a delivery triggered by a symptom kept as a stratification)
            timed = name in self.tf_dists
            if timed: # A time drawn for each person, as for a dwell time
                due = mod[f'ti_move_{name}']
                held = np.zeros(len(due.raw), dtype=bool)
                held[uids] = True
                due.raw[~np.isnan(due.raw) & ~held] = np.nan # Those who can no longer move (they left the level or the states) forget their time
                self.schedule_transfer(name, uids[np.isnan(due.raw[uids])], dt, self.ti) # Those who can move now, who arrived in this step (or the one before, after it)
            if len(uids):
                if timed:
                    now = due.raw[uids] <= self.ti
                else:
                    hazard = (self.eval_for(tf.rate, uids) if tf.rate is not None else 1/self.eval_for(tf.dur, uids))*self.mult('transition', tf.name, uids)
                    now = self.draw(f'transfer_{name}', uids) < -np.expm1(-np.maximum(hazard, 0)*dt)
                moving = uids[now]
                self.events[name] = moving
                self.flowcounts[name] += self.n_of(moving)
                self.flow_groups[name] = np.bincount(self.groups(moving), weights=self.weights(moving), minlength=sim.groups.n_groups)
                if tf.increment is not None:
                    arr[moving] = strat.add(arr[moving], tf.increment)
                elif tf.p is None:
                    arr[moving] = strat.value(strat.index(tf.dst))
                elif len(moving): # One of several levels, chosen as they move
                    ps = self.branch_shares([f'{name}.{dst}' for dst in tf.dsts], [None if p is None else self.eval_for(p, moving) for p in tf.p], [None]*len(tf.p))
                    edges = np.cumsum([np.broadcast_to(p, len(moving)) for p in ps[:-1]], axis=0)
                    which = (self.draw(f'transfer_to_{name}', moving) >= edges).sum(axis=0)
                    arr[moving] = np.array([strat.value(strat.index(dst)) for dst in tf.dsts])[which]
                sim.groups.groups_changed(moving)
                if timed: # A move up a counter that can go on: the next time is drawn from now
                    due.raw[moving] = due.nan
                    if tf.increment is not None:
                        new = strat.shifted(tf.increment)[level[now]]
                        self.new_round() # They can have been drawn for already in this one
                        self.schedule_transfer(name, moving[strat.shifted(tf.increment)[new] != new], dt, self.ti)
        rates = sim.mixing.migration
        if rates is not None and mod is sim.declared[0]: # People relocating between places; they are the same people for every module, so it is done once
            uids = sim.people.auids
            arr = sim.people[sim.groups.strats[sim.mixing.space].attr]
            places = self.place_raw()[uids]
            p = -np.expm1(-rates.sum(axis=1)*dt)[places]
            rvs = self.draw('migrate', uids)
            moving = rvs < p
            share = np.cumsum(rates, axis=1)/np.maximum(rates.sum(axis=1), 1e-300)[:, None] # The same draw chooses where they go
            where = (rvs[moving]/p[moving] >= share[places[moving]].T).sum(axis=0)
            arr[uids[moving]] = np.minimum(where, rates.shape[0] - 1)
            sim.groups.groups_changed(uids[moving])
        return

    def step_counts(self, dt):
        """
        States that are not people (a reservoir, mosquitoes): advanced as an ODE over the step by RK4, with the people as they were at its start

        A count that is filled at a steady rate and decays settles where it would in
        continuous time (as for SimInf's environmental pressure), and a chain of counts
        (mosquitoes that are infected, then infectious) passes people along within a step.
        As the ODE does, the step is split into enough substeps that no count turns over
        by more than a quarter in one, so a long step (a month) does not make it grow
        or die out. Modifiers of the transitions act on them, as on any other.
        """
        graph = self.graph
        trs = [tr for tr in graph.transitions.values() if not tr.people]
        if not trs:
            return
        with self.evaluating(by_group=True): # Rates are worked out for each group, not for each agent
            plan = [] # For each transition, what holds for the whole step: its rate as a function of the counts, its modifier, where an inflow arrives, and where each source's flow goes
            for tr in trs:
                name = tr.name
                mult = self.mult_raw('transition', name)*(1.0 if self.adjust[name] is None else self.adjust[name])
                if np.ndim(mult) and np.shape(mult)[-1] != self.sim.groups.n_groups:
                    raise ss.CapabilityError(f'a modifier of "{self.mod.name}.{name}" has a condition on people, which a transition of counts ({tr.arrow}) cannot take; give it no where=, or a condition on time')
                if tr.dur is not None and not tr.inflow:
                    hazard = 1/self.dwell_mean(name)
                    rate = lambda hazard=hazard: hazard
                else:
                    rate = self.count_rate(tr.rate)
                srcs = [None] if tr.inflow else [src.name for src in tr.src]
                dsts = {src:[(dst.name, self.dim_map(src, dst.name), f'{name}.{dst.name}' if len(tr.dst) > 1 else None) for dst in tr.dst] for src in srcs}
                plan.append((tr, rate, mult, self.inflow_split(tr) if tr.inflow else None, dsts))
            start = {name:np.asarray(val, dtype=float) for name,val in self.counts.items()}

            def deriv(counts):
                """ The rate of change of each count, and the rate of each transition, with the counts given """
                self.counts = counts
                change = dict.fromkeys(counts, 0.0)
                flows = {}
                for tr,rate,mult,split,dsts in plan:
                    if tr.inflow:
                        amounts = {None: rate()*mult*split}
                    else:
                        hazard = rate()*mult
                        amounts = {src:counts[src]*hazard for src in dsts}
                        self._fastest = max(self._fastest, float(np.max(hazard)))
                    probs = self.branch_probs(tr) # A split's probabilities can use the counts
                    total = 0.0
                    for src,amount in amounts.items():
                        if src is not None:
                            change[src] = change[src] - amount
                        for (dst,dim_map,key),p in zip(dsts[src], probs):
                            out = amount*p
                            change[dst] = change[dst] + (out if dim_map is None else out @ dim_map)
                            if key is not None: # The flow to each destination
                                flows[key] = flows.get(key, 0.0) + out
                        total = total + amount
                    flows[tr.name] = total
                return change, flows

            def ahead(k, h):
                return {name:np.maximum(start[name] + h*k[name], 0) for name in start}

            self._fastest = 0.0 # The fastest rate out of a count, as the ODE works out its substeps
            k1, f1 = deriv(start)
            n_sub = max(1, int(np.ceil(self._fastest*dt/0.25)))
            h = dt/n_sub
            total = dict.fromkeys(f1, 0.0)
            for sub in range(n_sub):
                if sub:
                    k1, f1 = deriv(start)
                k2, f2 = deriv(ahead(k1, h/2))
                k3, f3 = deriv(ahead(k2, h/2))
                k4, f4 = deriv(ahead(k3, h))
                start = {name:np.maximum(start[name] + h/6*(k1[name] + 2*k2[name] + 2*k3[name] + k4[name]), 0) for name in start}
                for name in total:
                    total[name] = total[name] + h/6*(f1[name] + 2*f2[name] + 2*f3[name] + f4[name])
            self.counts = start
            grouped = self.sim.groups.n_groups > 1
            for name,val in total.items():
                self.flowcounts[name] += np.sum(val)
                if grouped:
                    self.flow_groups[name] = self.flow_groups.get(name, 0) + val
        return

    def count_rate(self, expr):
        """
        A rate of counts, as a function of them to call at each derivative of `step_counts()`

        The people are as they were at the start of the step, so only the parts of the rate
        that use a count of states that are not people (e.g. `Iv.n`) change within it; the
        rest is worked out once, rather than at every call.
        """
        E = ss.expressions
        known = self.count_exprs.get(id(expr))
        if known is None or known[0] is not expr: # Whether it uses the counts, worked out once
            def changes(leaf):
                if isinstance(leaf, E.Count):
                    return any(not state.people for state in leaf.states)
                if isinstance(leaf, ss.transitions.State):
                    return not leaf.people
                return isinstance(leaf, E.par) and (isinstance(leaf.raw, E.Expr) or (callable(leaf.raw) and not isinstance(leaf.raw, ss.parameters.TimeValue))) # A function of the sim can give an expression of counts
            known = self.count_exprs[id(expr)] = (expr, any(changes(leaf) for leaf in expr.leaves))
        if not known[1]:
            val = self.eval(expr)
            return lambda: val
        if isinstance(expr, E.BinOp):
            a, b = self.count_rate(expr.a), self.count_rate(expr.b)
            op = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':E.divide, '**':operator.pow}[expr.op]
            return lambda: op(a(), b())
        if isinstance(expr, E.Func) and expr.name != 'expit':
            args = [self.count_rate(arg) for arg in expr.args]
            func = getattr(np, expr.name)
            return lambda: 1.0*func(*[arg() for arg in args])
        return lambda: self.eval(expr)

    def step(self):
        """ Contact-mediated transitions """
        mod = self.mod
        self._moved = []
        self.new_round()
        self.start_phase(2)
        self.cache.clear() # The interventions have run since the step started
        for tr in self.graph.transmissions:
            self.events.pop(tr.name, None)
            forced = self.forced.pop(tr.name, None) # Infected by code (disease.infect()) since the last step's: news now, with the infections of this step
            if forced is not None:
                self.events[tr.name] = forced
                self._news = True
        pairs = [(tr, src) for tr in self.graph.transmissions for src in tr.src + (self.immune_states if not getattr(tr, 'generated', False) else [])] # With protection due to a state or memory: people who have had one strain can be infected by another
        together = self.transmit_together(pairs) if len(pairs) > 1 else None
        self._step_pairs = [] # Who infected whom by each of them, for sources()
        for i,(tr,src) in enumerate(pairs):
            uids = self.transmit(tr, src) if together is None else together[i]
            self._step_pairs += self._pairs
            if mod.infection_log is not None:
                self.log_infections(uids)
            self.move(tr, src, uids)
        self._infecting = True # See enter()
        try:
            self.finish_moves(mod.ti) # Infections are dated at the step's index, its end
        finally:
            self._infecting = False
        self.step_per(mod.ti)
        self.write_derived(self.sim.people.auids)
        self.cache.clear() # Who is in each state has changed (including by the same-step cascade), so values read after the step (e.g. dis.current_value('rel_sus')) are worked out again
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the end of the step, after every module's transmissions; the people are the same for every module, so it is done once
            self.step_register(mod.ti)
        self.record_stocks() # Deaths later in the step are taken out of them: see die()
        return

    def shift_states(self, uids, states, to_states, ti):
        """ Move people from each of some states to the matching one of others, in every module that has them (a movement register's to_states) """
        for mod in self.sim.declared:
            graph = mod.graph
            for frm,to in zip(states, to_states):
                if frm in graph.states and to in graph.states and graph.states[frm].people:
                    inds = uids[mod[frm].raw[uids]]
                    engine = mod.engine
                    if engine.on_event and frm in engine.scheduled:
                        mod.left_exit[inds] = mod[f'ti_exit_{frm}'].raw[inds]
                    mod[frm][inds] = False
                    engine.clear_exit(frm, inds)
                    engine.clear_exit(to, inds)
                    engine.fire(None, inds, engine.state_index[frm], engine.state_index[to])
                    engine.enter(graph.states[to], inds, ti, srcs=engine.state_index[frm])
        return

    def step_register(self, ti):
        """
        Apply the events of a movement register that happen at the start of a step, in order: see `ss.geo.Mobility`

        The people in each place are put in a random order once, and each event that moves or
        removes people takes the first of them who are still there and in its states. People
        who are added are made at the end, so the other events of the step do not choose them.
        """
        sim = self.sim
        people = sim.people
        mixing = sim.mixing
        ev = mixing.events()
        a, b = np.searchsorted(ev.ti, [ti, ti + 1])
        if a == b:
            return
        self.new_round()
        d = mixing.space
        place = people[sim.groups.strats[d].attr]
        uids = people.auids
        levels = sim.groups.level_raw(d)[uids]
        order = np.lexsort((self.draw('register', uids), levels))
        ordered = uids[order]
        bounds = np.searchsorted(levels[order], np.arange(sim.groups.strats[d].n + 1))
        moved_in = sc.ddict(list) # People who moved into each place in this step, who can be chosen after those who were there
        removed = np.zeros(people.uid.len_tot, dtype=bool)
        masks = dict() # Who is in each set of states
        scale = self.scale
        carry = getattr(self, '_carry', 0.0) # The fraction of an agent left over from rounding
        short = 0.0
        enter = [] # The places and states of the people to add, and how many
        for e in range(a, b):
            src, dst, states = ev.src[e], ev.dst[e], ev.states[e]
            want = ev.n[e]/scale + carry
            k = int(np.floor(want + 1e-9))
            carry = want - k
            if src < 0:
                enter.append((dst, states, k))
                continue
            cand = ordered[bounds[src]:bounds[src+1]]
            if moved_in[src]:
                cand = np.concatenate([cand, moved_in[src]])
            ok = (place.raw[cand] == src) & ~removed[cand]
            if states is not None:
                if states not in masks:
                    masks[states] = np.zeros(people.uid.len_tot, dtype=bool)
                    for mod in sim.declared:
                        for name in states:
                            if name in mod.graph.states and mod.graph.states[name].people:
                                masks[states] |= mod[name].raw
                ok &= masks[states][cand]
            chosen = ss.uids(cand[ok][:k])
            short += k - len(chosen)
            if ev.to_states[e] is not None: # Moved to other states, e.g. from S1 to S2 as they age
                self.shift_states(chosen, states, ev.to_states[e], ti)
                masks.clear()
            if dst >= 0:
                place[chosen] = dst
                moved_in[dst].extend(chosen.tolist())
            else:
                removed[chosen] = True
                people.request_death(chosen)
        self._carry = carry
        self.add_people(enter, ti, 'register')
        sim.groups.groups_changed() # People were moved and added
        self.check_short(short*scale)
        return

    def add_people(self, enter, ti, key, newborn=True):
        """
        Add agents to places, in given states

        Args:
            enter (list): for each place (0, if there are no places), the states to put people in (None for each module's first state) and the number of agents
            ti (int): the time they enter their states
            key (str): what adds them, which names the distributions their levels and states are drawn from
            newborn (bool): whether they are born, so are in the youngest age group; if not, their ages are drawn from the population's age distribution

        Returns the UIDs of the new agents.
        """
        sim = self.sim
        people = sim.people
        total = sum(k for dst,states,k in enter)
        if not total:
            return ss.uids()
        new = people.grow(total, placed=True)
        dsts = np.concatenate([np.full(k, dst) for dst,states,k in enter])
        d = sim.mixing.space
        for i,strat in enumerate(sim.groups.strats): # The place given, and the other levels as people are born into them
            if i == d:
                people[strat.attr][new] = strat.value(dsts)
            elif strat.kind == 'age':
                if strat.edges is None:
                    people[strat.attr][new] = 0
                elif newborn: # Otherwise their ages, drawn as the agents were made, give their groups
                    people.age[new] = strat.edges[0]
            elif strat.female is None: # A sex dimension is people.female, as drawn when the agents were made: see ss.People.grow()
                rvs = self.draw(f'{key}_{strat.name}', new)
                people[strat.attr][new] = strat.value(np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1))
        if newborn and not any(strat.kind == 'age' for strat in sim.groups.strats): # Born at 0, before they enter their states (whose dwell times can read their age)
            people.age[new] = 0
        rvs = self.draw(f'{key}_state', new)
        sim.groups.groups_changed(new) # Before they enter their states, so they are counted in their groups: see wrote()
        for mod in sim.declared:
            engine = mod.engine
            engine.draw_traits(new)
            placed = [s.name for s in mod.graph.people_states if d is None or engine.applies[s.name][d]]
            if not placed:
                continue
            choice = np.empty(total, dtype=object)
            start = 0
            for dst,states,k in enter: # The same states for each entry's people, so they are chosen for all of them at once
                own = np.array([name for name in (states or []) if name in placed] or placed[:1], dtype=object)
                choice[start:start+k] = own[np.minimum((rvs[start:start+k]*len(own)).astype(int), len(own) - 1)] # Shared equally between the states named
                start += k
            for name in dict.fromkeys(choice): # In a fixed order, since the draws are made in it
                engine.enter(mod.graph.states[name], new[choice == name], ti)
        return new

    def susceptibility(self, tr, src):
        """ The effective susceptibility of each agent (by UID) to a transmission from a state: `rel_sus` for those in the state, with the state's own, the modifiers on it and on the transmission, and the adjust= of the stratifications """
        out = times(self.current_value('rel_sus', self.mod[src.name].raw, src.rel_sus), self.mult_raw('transition', tr.name))
        return out if self.adjust[tr.name] is None else out*self.adjust[tr.name][self.sim.groups.group_raw()]

    #%% Branching processes

    def init_branching(self, fracs):
        """
        As a branching process (`ss.Sim(branching=True)`): the uninfected are counts by group, which infection takes no one from, and
        agents are made only for the infected, here for the people who start infected (rounded to whole people, at least one where any are given)

        The counts are of the people the population and the initial values give in each group, as in a compartmental run; the
        uninfected side's own flows (vaccination, waning, births and deaths) move them over each step (see `step_held()`).
        """
        sim = self.sim
        groups = sim.groups
        ng = groups.n_groups
        if len(sim.declared) > 1 or len(sim.networks) or self.strain is not None:
            what = 'several declared modules' if len(sim.declared) > 1 else 'a network' if len(sim.networks) else 'strains'
            raise ss.CapabilityError(f'"{self.mod.name}" runs as a branching process of agents, which cannot yet have {what}; run with a compartmental solver (e.g. solver="gillespie"), or without it')
        self.uninfected = set(self.graph.uninfected)
        self.branch_rng = self.make_rng(stream=3) # For the draws that are not per agent: the new cases of each group, and their infectors
        self.infector = np.zeros(0, dtype=ss.dtypes.int) # Who infected each agent, by UID (-1 for a seed, or for an infection from no one), and the generation of each
        self.generation = np.zeros(0, dtype=ss.dtypes.int)
        self.infected_ti = np.zeros(0, dtype=ss.dtypes.int)
        size = self.total_pop*groups.group_split
        self.held = {}
        for name,frac in fracs.items():
            n = size*np.broadcast_to(frac, ng)
            if name in self.uninfected:
                self.held[name] = n.astype(float)
                continue
            k = Compartmental.round_preserving(n).astype(int)
            if n.sum() > 0 and not k.sum(): # A seed of less than one person is one
                k[np.argmax(n)] = 1
            if k.sum():
                uids = self.make_in_groups(np.repeat(np.arange(ng), k))
                self.mod[name][uids] = True
                self.stamp(name, uids, self.seed_ti(name))
                self.add_infectors(uids, np.full(len(uids), -1))
                if self.mod.infection_log is not None and any(state.name == name for tr in self.graph.transmissions for state in tr.dst):
                    self.log_infections(uids, seeds=True)
        self.held_trs = [tr for tr in self.graph.transitions.values() if tr.people and not tr.acquisition and (all(src.name in self.uninfected for src in tr.src) if tr.src else tr.inflow and all(dst.name in self.uninfected for dst in tr.dst))] # The uninfected side's own flows
        back = [tr.name for tr in self.graph.transitions.values() if tr.src and any(src.name not in self.uninfected for src in tr.src) and any(dst.name in self.uninfected for dst in tr.dst or [])]
        if back:
            ss.note(f'"{self.mod.name}" runs as a branching process, so people who go back to the uninfected ({sc.strjoin(back)}) leave it: they are not counted there, nor infected again', sim=sim)
        return

    def held_n(self):
        """ In a branching process: the people a rate divides by, the uninfected, in each group """
        return sum(np.broadcast_to(held, self.sim.groups.n_groups) for held in self.held.values()) if self.held else np.zeros(self.sim.groups.n_groups)

    def add_infectors(self, uids, srcs):
        """ In a branching process: who infected some new agents (-1 for none), and so their generation, and when """
        n = self.sim.people.uid.len_tot
        if len(self.infector) < n:
            grow = n - len(self.infector)
            self.infector = np.append(self.infector, np.full(grow, -1, dtype=ss.dtypes.int))
            self.generation = np.append(self.generation, np.zeros(grow, dtype=ss.dtypes.int))
            self.infected_ti = np.append(self.infected_ti, np.zeros(grow, dtype=ss.dtypes.int))
        srcs = np.asarray(srcs, dtype=ss.dtypes.int)
        self.infected_ti[uids] = self.mod.ti if self._started else 0
        self.infector[uids] = srcs
        self.generation[uids] = np.where(srcs >= 0, self.generation[np.maximum(srcs, 0)] + 1, 0)
        return

    def infectious_window(self):
        """
        In a branching process: a time (in the sim's unit) by which nearly everyone infected has left the infected side's states that
        have exits, and so has infected everyone they will: five times the mean stay in those states, one after another, from their exits
        """
        total = 0.0
        for state in self.graph.people_states:
            if state.name in self.uninfected:
                continue
            exits = self.graph.exits[state.name]
            rates = []
            for tr in list(exits.timed) + list(exits.hazards) + list(exits.branches):
                try:
                    if tr.name in self.dwells:
                        mean = float(np.max(self.dwell_mean(tr.name)))
                    else:
                        with self.evaluating(by_group=True):
                            mean = 1/float(np.max(self.eval(tr.rate)))
                except Exception: # Not every exit has a mean (a rate that is a function of the agents): it is left out
                    continue
                if np.isfinite(mean) and mean > 0:
                    rates.append(1/mean)
            if rates:
                total += 1/sum(rates)
        return 5*total

    def make_in_groups(self, gs):
        """ Make new agents in given groups: at each one's levels of every stratification, with ages drawn from the population's within an age group """
        sim = self.sim
        people = sim.people
        groups = sim.groups
        new = people.grow(len(gs), placed=True)
        coords = groups.group_coords[gs]
        for d,strat in enumerate(groups.strats):
            levels = coords[:, d]
            if strat.kind == 'age':
                if strat.edges is not None:
                    people.age[new] = self.ages_in(strat, levels)
            elif strat.kind in ['strain', 'disease']:
                raise ss.CapabilityError(f'a branching process of agents cannot be stratified by "{strat.name}" yet; run with a compartmental solver')
            elif strat.female is not None: # A sex dimension is people.female
                people.sex[new] = strat.value(levels)
                groups.set_sex(new, from_level=True)
            else:
                people[strat.attr][new] = strat.value(levels)
        groups.groups_changed(new)
        for mod in sim.declared:
            mod.engine.draw_traits(new)
        return new

    def ages_in(self, strat, levels):
        """ Ages drawn from the population's age distribution within given levels of an age stratification """
        cdf = self.sim.mixing.age_cdf(strat)
        top = max(110, strat.edges[-1] + 10)
        xs = np.linspace(0, top, 4*int(top) + 1)
        F = cdf(xs)
        edges = np.append(strat.edges, top)
        lo, hi = cdf(edges[levels]), cdf(edges[levels + 1])
        return np.interp(lo + self.branch_rng.random(len(levels))*(hi - lo), F, xs)

    def held_rel_sus(self):
        """ The mean susceptibility of the uninfected: of a trait that varies between people, its mean """
        a = next((a for a in getattr(self, 'traits', []) if a.name == 'rel_sus'), None)
        return 1.0 if a is None else ss.transitions.dwell_stats(a.value, self.convert)[0]

    def transmit_held(self, tr, src):
        """
        In a branching process: the new cases of a transmission from a state of the uninfected, who are counts (see `init_branching()`)

        The hazard on one of them in each group is worked out from the infectious agents, as it is for an agent, and the
        number infected in each group is a Poisson draw of the expected number. Each is given an infector, drawn from the
        infectious agents in proportion to what each adds to the hazard in their group (or none, for a part of the rate that
        comes from no one), which is the same in distribution as drawing each infectious agent's offspring. The new cases are
        made as agents in the state, in their groups, so they are moved out of it as anyone infected is.
        """
        sim = self.sim
        mod = self.mod
        groups = sim.groups
        ng = groups.n_groups
        if not self.same_in_group(tr):
            raise ss.CapabilityError(f'"{mod.name}.{tr.name}" has a rate that differs between the people of a group, which a branching process of agents cannot give the uninfected, who are counts by group; stratify by what it depends on, or run with a compartmental solver')
        mixing = sim.mixing
        n = self.held_n()
        if mixing.space is None:
            place_of = np.zeros(ng, dtype=int)
            agent_place = None
            mat = np.full((1, 1), 1/n.sum() if n.sum() > 0 else 0.0) # The exposure of anyone to one infectious person
        else:
            place_of = np.asarray(mixing.places)
            agent_place = self.place_raw()
            n_places = groups.strats[mixing.space].n
            sizes = np.bincount(place_of, weights=n, minlength=n_places)
            mat = np.column_stack([mixing.by_place(np.arange(n_places), np.eye(n_places)[q], sizes, n_places) for q in range(n_places)]) # In place p, from one infectious person in place q
        with self.evaluating(by_group=True):
            const = np.zeros(ng) if tr.const is None else np.broadcast_to(np.asarray(self.eval(tr.const), dtype=float), ng)
            coefs = {state:np.broadcast_to(np.asarray(self.eval(coef), dtype=float), ng) for state,coef in tr.terms.items()}
        sources, weights = [], [] # The infectious agents, and what each adds to the hazard in each group
        for state,coef in coefs.items():
            inf = self.members(state)
            if not len(inf):
                continue
            rt = mod.rel_trans[inf]*self.mult('rel_trans', None, inf)*state.rel_trans
            pl = np.zeros(len(inf), dtype=int) if agent_place is None else agent_place[inf]
            weights.append(coef[:, None]*np.asarray(rt, dtype=float)[None, :]*mat[place_of[:, None], pl[None, :]])
            sources.append(inf)
        W = np.concatenate(weights, axis=1) if weights else np.zeros((ng, 0))
        sources = np.concatenate(sources) if sources else np.zeros(0, dtype=ss.dtypes.int)
        hazard = const + W.sum(axis=1)
        adjust = (1.0 if self.adjust[tr.name] is None else self.adjust[tr.name])*self.mult_raw('transition', tr.name)
        sus = src.rel_sus*self.held_rel_sus()*float(self.mult_raw('rel_sus'))*adjust
        expected = np.broadcast_to(self.held[src.name], ng)*-np.expm1(-np.maximum(hazard*sus, 0)*self.dt)
        k = self.branch_rng.poisson(np.maximum(expected, 0))
        if not k.sum():
            return ss.uids()
        gs = np.repeat(np.arange(ng), k)
        srcs = np.full(len(gs), -1, dtype=ss.dtypes.int)
        start = 0
        for g in np.flatnonzero(k):
            cum = np.cumsum(np.append(W[g], const[g]))
            pick = np.searchsorted(cum, self.branch_rng.random(k[g])*cum[-1], side='right')
            srcs[start:start+k[g]] = np.where(pick < len(sources), sources[np.minimum(pick, max(len(sources) - 1, 0))] if len(sources) else -1, -1)
            start += k[g]
        new = self.make_in_groups(gs)
        self.put(src.name, new, True)
        self.add_infectors(new, srcs)
        known = srcs >= 0
        self._pairs = [(ss.uids(srcs[known]), new[known], np.full(known.sum(), ss.dtypes.int_nan, dtype=ss.dtypes.int))]
        return new

    def step_held(self, dt):
        """ In a branching process: the uninfected side's own flows over a step (vaccination, waning, births and deaths), as expected values, one after another """
        ng = self.sim.groups.n_groups
        with self.evaluating(by_group=True):
            for tr in self.held_trs:
                mult = self.mult_raw('transition', tr.name)*(1.0 if self.adjust[tr.name] is None else self.adjust[tr.name])
                if np.ndim(mult) and np.shape(mult)[-1] != ng:
                    raise ss.CapabilityError(f'a modifier of "{self.mod.name}.{tr.name}" has a condition on people, which the uninfected of a branching process, who are counts by group, cannot take; give it no where=, or run with a compartmental solver')
                if tr.inflow:
                    inflow = np.broadcast_to(np.asarray(self.eval(tr.rate), dtype=float)*mult*self.inflow_split(tr), ng)*dt
                    if getattr(tr, 'per_capita', False):
                        inflow = inflow*self.held_n()
                    sources = [None]
                else:
                    hazard = 1/self.dwell_mean(tr.name) if tr.dur is not None else np.asarray(self.eval(tr.rate), dtype=float)
                    sources = [src.name for src in tr.src]
                total = np.zeros(ng)
                for name in sources:
                    if name is None:
                        amount = inflow
                    else:
                        amount = np.broadcast_to(self.held[name], ng)*-np.expm1(-np.maximum(hazard*mult, 0)*dt)
                        self.held[name] = np.broadcast_to(self.held[name], ng) - amount
                    for dst,p in zip(tr.dst or [], self.branch_probs(tr)):
                        if dst.name in self.held:
                            self.held[dst.name] = np.broadcast_to(self.held[dst.name], ng) + amount*p
                    total = total + amount
                self.flowcounts[tr.name] += float(total.sum())
        return

    def transmit(self, tr, src):
        """ Work out who in a state is infected this step """
        self._route_memo = {} # See route_beta()
        try:
            return self.transmit_from(tr, src)
        finally:
            self._route_memo = None

    def transmit_from(self, tr, src):
        """ As `transmit()` """
        mod = self.mod
        sim = self.sim
        dt = self.dt
        self._pairs = [] # Who infected whom over the networks, for the infection log
        if self.held is not None and src.name in self.held:
            return self.transmit_held(tr, src)
        routes = list(sim.networks.items())
        strains = self.strain_levels()
        if not routes and strains == [None] and self.same_in_group(tr): # The usual case, in one pass over the agents
            return self.transmit_groups(tr, src)
        if routes and strains == [None] and not any(np.any(np.asarray(self.route_beta(coef, nkey, route), dtype=float) > 0) for nkey,route in routes for coef in tr.terms.values()): # No route carries it (e.g. a mother-to-child transmission with a beta of 0): nobody is infected, and nothing is drawn
            return ss.uids()
        sus = self.members(src)
        if not len(sus):
            return ss.uids()
        protects = self.strain is not None and (src.name in self.protect or bool(self.sim.modifiers.get((mod.name, 'strain_sus', None)))) # Protected in part against some strains (see Engine.init_immunity(), and products by strain)
        carried = self.carriage is not None and self.applies[src.name][self.strain] and src.name in (self.carriage.carried or [src.name]) # People infected with strains already, who can carry another as well
        superinfects = carried and self.superinfects(tr, src) # ... and stay in the same state, so are only changed if they do not carry it already
        if not routes: # Homogeneous mixing: the same hazard for everyone, scaled by their susceptibility
            adjust = self.adjusted(tr.name, sus)
            hazards = []
            for strain in strains:
                with self.evaluating(strain=strain):
                    protect = self.protection(src, sus, strain) if protects else 1.0 # The strains they carry, or remember, protect them in part
                    hazards.append(self.strain_rate(tr, self.eval_for(tr.rate, sus), strain)*mod.rel_sus[sus]*src.rel_sus*adjust*self.mult('rel_sus', None, sus)*protect)
            return self.draw_infections(tr, sus, hazards, dt, carried, superinfects)

        # Otherwise, transmit over each route. Mixing pools give each person a hazard, and networks a chance for each contact
        sus_raw = self.susceptibility(tr, src)
        rel_sus = mod.rel_sus.asnew(sus_raw, copy=False)
        new_cases = []
        pairs = [] # Who infected whom, where that is known, so the strain can be passed on
        pools = any(isinstance(route, (ss.MixingPool, ss.MixingPools)) for nkey,route in routes)
        pool_hazard = [np.zeros(len(sus_raw)) for strain in strains] if pools else None
        any_pools = False
        by_strain = 1.0 # The infectiousness of each agent's strain
        if self.strain is not None:
            strain_raw = self.strain_raw()
            if self.strain_adjust[tr.name] is not None:
                by_strain = self.strain_adjust[tr.name][strain_raw]
        for i,(nkey,route) in enumerate(routes):
            if isinstance(route, ss.Network) and route.has_acts: # A probability per act, in each direction
                if len(route):
                    self.transmit_acts(tr, i, nkey, route, rel_sus, by_strain, new_cases, pairs)
                continue
            everyone = ss.uids(np.arange(len(sus_raw)))
            with self.evaluating(memo={} if len(tr.terms) > 1 else None): # The factor the source states share (e.g. a susceptibility of each person) is worked out once
                coefs = {state: self.route_beta(coef, nkey, route, everyone) for state,coef in tr.terms.items()}
            coefs, target = self.by_target(coefs, route) # For each source state
            route_sus = rel_sus if target is None else rel_sus.asnew(rel_sus.raw*target, copy=False)
            ref = max([np.max(c) for c in coefs.values()] + [0])
            if ref <= 0:
                continue
            mult = self.route_mult(tr.name, route) # From the modifiers of its contacts and of the transmission on it
            weights = None
            for state,c in coefs.items():
                share = (c/ref)*state.rel_trans
                w = mod[state.name].raw if np.ndim(share) == 0 and share == 1 else mod[state.name].raw*share # The usual case, one source state, needs no product
                weights = w if weights is None else weights + w
            rel_trans = mod.rel_trans.asnew(times(self.current_value('rel_trans', weights), by_strain), copy=False)
            beta = -np.expm1(-ref*dt)
            if isinstance(route, ss.Network):
                for pair in self.over_network(i, route, rel_trans.raw, route_sus.raw, [beta, beta], tr.name):
                    new_cases.append(pair[1])
                    pairs.append(pair)
            elif isinstance(route, (ss.MixingPool, ss.MixingPools)) and route.name in sim.mixing.pool_terms: # A contact matrix within each place: see ss.solvers.Mixing.init_pools()
                any_pools = True
                groups = sim.groups.group_raw()
                auids = sim.people.auids
                n = np.bincount(groups[auids], minlength=sim.groups.n_groups).astype(float)
                for k,strain in enumerate(strains):
                    share = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                    infectious = np.bincount(groups[auids], weights=(rel_trans.raw*share)[auids], minlength=sim.groups.n_groups)
                    expo = sim.mixing.term_exposure(sim.mixing.pool_terms[route.name], infectious, n)
                    pool_hazard[k] += ref*expo[groups]*route_sus.raw*mult(None, slice(None))
            elif isinstance(route, ss.MixingPools) and route.edges is not None and not route.pools: # A matrix with its own age bands: from the counts in each band
                any_pools = True
                for k,strain in enumerate(strains):
                    share = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                    auids, expo = route.band_exposure(rel_trans.raw*share)
                    pool_hazard[k][auids] += ref*expo*route_sus.raw[auids]*mult(None, auids)
            elif isinstance(route, (ss.MixingPool, ss.MixingPools)): # Has no direction
                any_pools = True
                for pool in (route.pools if isinstance(route, ss.MixingPools) else [route]):
                    if len(pool.src_uids) and len(pool.dst_uids):
                        dst = pool.dst_uids
                        for k,strain in enumerate(strains):
                            share = 1.0 if strain is None else self.strain_weights[strain_raw[pool.src_uids], strain]
                            trans = np.mean(rel_trans.raw[pool.src_uids]*share) # The infectiousness-weighted fraction of the source group that is infectious
                            pool_hazard[k][dst] += pool.pars.beta*ref*trans*pool.eff_contacts.raw[dst]*route_sus.raw[dst]*mult(None, dst)
            else: # Any other route works out its own transmission
                new_cases.append(ss.uids(route.compute_transmission(route_sus, rel_trans, beta, disease=mod)))
        if self.strain is not None and pairs: # Each new case takes the strain of whoever infected them, one of those who reached them (see Transmission)
            trgs, srcs, nets = self.pick_sources(pairs, tr.name)
            strain = strain_raw[srcs]
            if self.carriage is not None: # Someone who carries several strains passes on one of them
                weights = np.cumsum(self.strain_weights[strain], axis=1)
                rvs = self.draw(f'pass_{tr.name}', trgs)
                strain = np.minimum((weights <= rvs[:, None]).sum(axis=1), weights.shape[1] - 1)
            ok = np.ones(len(trgs), dtype=bool)
            if superinfects: # Infected with a strain they carry already: no change
                ok &= self.can_take(strain_raw[trgs], strain)
            if protects: # Protected in part against the strain that reached them (HP14)
                ok &= self.unprotected(tr.name, trgs, self.protection(src, trgs, strain))
            if not ok.all():
                drop = trgs[~ok]
                trgs, srcs, nets, strain = trgs[ok], srcs[ok], nets[ok], strain[ok]
                new_cases = [cases[~np.isin(cases, drop)] for cases in new_cases]
            pairs = [(srcs, trgs, nets)] # Who infected whom, for the infection log
            self.set_strain(trgs, strain, carried)
        self._pairs = pairs
        if any_pools:
            new_cases.append(self.draw_infections(tr, sus, [hazard[sus] for hazard in pool_hazard], dt, carried, superinfects))
        if not new_cases:
            return ss.uids()
        return ss.uids.concatenate(new_cases).unique()

    def transmit_together(self, pairs):
        """
        Who is infected by each transmission (and source state) in `pairs`, in one pass over each network rather than one for each (TS5),
        when that gives the same result: the transmissions take people from different states, which none of them transmits from, and have
        the same source states with coefficients in proportion (as tbsim's reinfections are, at a relative risk), with no strains and nothing
        that differs by transmission on a route. Each edge's chance is then the transmission's, and its random number is the same in every
        pass, so who is infected is as if each were worked out alone. None if they cannot be worked out together.
        """
        mod = self.mod
        sim = self.sim
        routes = list(sim.networks.items())
        if self.strain is not None or mod.infection_log is not None or not routes:
            return None
        srcs = [src.name for tr,src in pairs]
        sources = {state.name for tr,src in pairs for state in tr.terms}
        if len(set(srcs)) < len(srcs) or sources & set(srcs) or any(len(tr.terms) != len(pairs[0][0].terms) for tr,src in pairs):
            return None
        for nkey,route in routes:
            if not isinstance(route, ss.Network) or route.has_acts or route.custom_beta or any(sim.modifiers.get((mod.name, 'route', f'{tr.name}@{route.name}')) for tr,src in pairs):
                return None
        sus = [] # Each pair's susceptibility, by UID, as transmit() works it out
        for tr,src in pairs:
            sus.append(self.susceptibility(tr, src))
        everyone = ss.uids(np.arange(len(sus[0])))
        dt = self.dt
        new = [[] for pair in pairs]
        for nkey,route in routes:
            coefs = []
            for tr,src in pairs:
                c, target = self.by_target({state.name: self.route_beta(coef, nkey, route, everyone) for state,coef in tr.terms.items()}, route)
                if target is not None or any(np.ndim(v) for v in c.values()) or set(c) != set(coefs[0][0] if coefs else c):
                    return None
                coefs.append((c, max(list(c.values()) + [0])))
            first, ref = coefs[0]
            for c,r in coefs[1:]: # In proportion to the first, so the infectious weigh the same in each
                if (r > 0) != (ref > 0) or (ref > 0 and not all(abs(c[k]/r - first[k]/ref) <= 1e-12*abs(first[k]/ref) for k in c)):
                    return None
            betas = [-np.expm1(-r*dt) for c,r in coefs]
            top = max(betas)
            if top <= 0:
                continue
            weights = mod.rel_trans.raw*0
            for state in pairs[0][0].terms:
                weights = weights + mod[state.name].raw*(first[state.name]/ref)*state.rel_trans
            rel_trans = self.current_value('rel_trans', weights)
            route_sus = sum(s*(b/top) for s,b in zip(sus, betas)) # Each transmission's chance, as a share of the largest
            for _, trgs, _ in self.over_network(0, route, rel_trans, route_sus, [top, top], None):
                for i,(tr,src) in enumerate(pairs):
                    new[i].append(trgs[mod[src.name].raw[trgs]])
        self._pairs = []
        return [ss.uids.concatenate(cases).unique() if cases else ss.uids() for cases in new]

    def sources(self, uids):
        """
        Who infected each of some agents infected over a network in this step, and over which route: for a module's own code, e.g.
        to pass on a value of the person who infected them (`dis.engine.sources(dis.infection.uids)`), after the transmissions

        Returns:
            sources (array): the UID of whoever infected each agent, or NaN where it is not known (e.g. homogeneous mixing)
            networks (array): the index of the route each was infected over, or NaN
        """
        sources = np.full(len(uids), np.nan)
        networks = np.full(len(uids), np.nan)
        pairs = self._step_pairs # Of every transmission so far in this step
        if len(uids) and pairs:
            trgs, srcs, nets = self.first_infections(pairs)
            if len(trgs):
                inds = np.minimum(np.searchsorted(trgs, uids), len(trgs) - 1)
                found = trgs[inds] == uids
                sources[found] = srcs[inds[found]]
                networks[found] = nets[inds[found]]
        return sources, networks

    def log_infections(self, uids, seeds=False):
        """ Record new infections for an `ss.infection_log` analyzer: with who infected them and the network, where known; seeds have a source of -1 """
        if not len(uids):
            return
        if seeds:
            sources, networks = np.full(len(uids), -1.0), np.full(len(uids), np.nan)
        else:
            sources, networks = self.sources(uids)
        self.mod.infection_log.add_entries(uids, sources, self.mod.t.tvec[self.mod.ti], network=networks) # At the step's time point, as ti_I is (seeds at the start)
        return

    def same_in_group(self, tr):
        """ Whether the rate of a transmission is the same for everyone in a group, apart from their own susceptibility """
        if tr.name not in self.by_groups:
            leaves = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par)]
            funcs = [raw for raw in leaves if callable(raw) and not isinstance(raw, (ss.Dist, ss.parameters.TimeValue))]
            spread = any(id(leaf) in self.spread for leaf in tr.rate.leaves if isinstance(leaf, ss.par)) # As does a distribution across people
            mods = [m for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and leaf.name and not leaf.literal for m in self.sim.modifiers.get((self.mod.name, 'par', leaf.name), [])]
            per_agent = any(m.where is not None or isinstance(m.amount, ss.by) or (callable(m.amount) and not isinstance(m.amount, ss.parameters.TimeValue)) for m in mods) # As does a modifier of a parameter for some people
            props = any(isinstance(leaf, (ss.expressions.Prop, ss.expressions.attr)) or (isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by) and leaf.raw.on_people(self.sim)) for leaf in tr.rate.leaves) # As does a property or attribute of people
            self.by_groups[tr.name] = not spread and not per_agent and not props and not any(ss.utils.n_args(func) >= 2 for func in funcs) and not ss.expressions.clocks_in(tr.rate) # A function of the agents, or a clock, can give each a rate of their own
        return self.by_groups[tr.name] and np.ndim(self.mult_raw('transition', tr.name)) == 0 and np.ndim(self.mult_raw('rel_sus')) == 0

    def transmit_groups(self, tr, src):
        """
        Work out who is infected under homogeneous mixing, when the rate is the same for everyone in a group

        The rate is worked out once for each group, and one compiled pass over the agents
        gives the probability for everyone who is susceptible and exposed. The draw is the
        one the general case makes, so the agents infected are the same.
        """
        sim = self.sim
        mod = self.mod
        ng = sim.groups.n_groups
        with self.evaluating(by_group=True):
            rate = np.broadcast_to(np.asarray(self.eval(tr.rate), dtype=float), ng)
        adjust = (1.0 if self.adjust[tr.name] is None else self.adjust[tr.name])*self.mult_raw('transition', tr.name)
        key = self.crn_key(f'mix_{tr.name}', mod[src.name].uids if check_draws else None)
        if key is not None: # The usual case: the draws are made in the same pass, so only the agents infected are listed
            uids = ss.solvers.backends.passes.infect(mod[src.name].raw, sim.groups.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw, float(src.rel_sus), np.ascontiguousarray(np.broadcast_to(adjust, ng), dtype=float),
                                     float(self.mult_raw('rel_sus')), self.dt, sim.people.slot_raw(), *key)
            return ss.uids(uids)
        uids, probs = ss.solvers.backends.passes.infection_probs(mod[src.name].raw, sim.people.auids.view(np.ndarray), sim.groups.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw,
                                                 float(src.rel_sus), np.ascontiguousarray(np.broadcast_to(adjust, ng), dtype=float), float(self.mult_raw('rel_sus')), self.dt)
        uids = ss.uids(uids)
        return uids[self.draw(f'mix_{tr.name}', uids) < probs]

    def draw_infections(self, tr, sus, hazards, dt, carried=False, superinfects=False):
        """ Draw who is infected from a hazard for each strain (or one hazard, if there are no strains), and give each new case their strain """
        hazards = [np.broadcast_to(np.maximum(hazard, 0), len(sus)) for hazard in hazards]
        if superinfects: # Only those who do not carry a strain already can be infected with it
            levels = self.strain_raw()[sus]
            hazards = [hazard*self.can_take(levels, strain) for hazard,strain in zip(hazards, self.strain_levels())]
        total = np.sum(hazards, axis=0)
        p = -np.expm1(-total*dt)
        rvs = self.draw(f'mix_{tr.name}', sus)
        infected = rvs < p
        if self.strain is not None and infected.any(): # The same draw chooses the strain, in proportion to each one's hazard
            share = np.cumsum(hazards, axis=0)[:, infected]/total[infected]
            strain = (rvs[infected]/p[infected] >= share).sum(axis=0)
            self.set_strain(sus[infected], np.array(self.strain_levels())[np.minimum(strain, len(hazards) - 1)], carried)
        return sus[infected]

    def set_strain(self, uids, strain, carried=False):
        """ Give new cases their strain; if they carry strains already (`carried`), they carry this one as well """
        sim = self.sim
        uids = ss.uids(uids)
        if carried:
            strain = self.carriage.adds[self.strain_raw()[uids], strain]
        sim.people[sim.groups.strats[self.strain].attr][uids] = sim.groups.strats[self.strain].value(strain)
        sim.groups.groups_changed(uids)
        return

    def protection(self, state, uids, strain):
        """ The share of the force of infection of a strain (one, or one for each agent) that reaches some agents in a state: by the strains they carry there, and by the products they hold """
        out = super().protection(state, self.strain_raw()[uids], strain) if state.name in self.protect else 1.0
        if self.sim.modifiers.get((self.mod.name, 'strain_sus', None)) and len(uids):
            if np.ndim(strain): # One for each agent
                effects = np.array([np.broadcast_to(self.strain_effect(j), len(self.mod.rel_sus.raw)) for j in range(int(np.max(strain)) + 1)])
                out = out*effects[strain, uids]
            else:
                out = out*np.broadcast_to(self.strain_effect(strain), len(self.mod.rel_sus.raw))[uids]
        return out

    def route_beta(self, coef, nkey, route, uids=None):
        """
        The coefficient of a source state in a transmission's rate on one route: a rate per contact, or, on a network with acts, the hazard per act
        in each of its directions (see `ss.peract`); with `uids`, a value that differs between agents is for those agents. Worked out once for
        each in a transmission (see transmit()), since the source states often share it (e.g. every infectious state of an STI)
        """
        memo = self._route_memo
        key = (id(coef), nkey, None if uids is None else len(uids))
        if memo is not None and key in memo:
            val, self._per_act = memo[key]
            return val
        self._per_act = False
        with self.evaluating(route=(nkey, route)):
            val = self.eval(coef) if uids is None else self.eval_for(coef, uids)
        acts = getattr(route, 'has_acts', False)
        if acts and not self._per_act: # A beta that is an expression of parameters (beta=b*rel), worked out as a probability per act
            val = self.act_hazards(val, route)
        if memo is not None:
            memo[key] = (val, self._per_act)
        return val

    def by_target(self, coefs, route):
        """
        The coefficients of a transmission's source states on a route that is not a network with acts, when some have a value for each agent (e.g.
        `ss.by(sex=...)*I` over `ss.contacts(matrix, dims='sex')`): a value for whoever is infected, times a weight for each source state. Returns the weights,
        and the value for each agent (None if no coefficient differs between agents)
        """
        arrays = [np.asarray(c, dtype=float) for c in coefs.values() if np.ndim(c)]
        if not arrays:
            return coefs, None
        shape = np.broadcast_shapes(*[a.shape for a in arrays])
        target = np.max([np.broadcast_to(np.asarray(c, dtype=float), shape) for c in coefs.values()], axis=0) # Above 0 wherever any coefficient is, so none is lost where another is 0
        if target.ndim == 1 and len(target) == len(self.mod.rel_sus.raw): # By UID: the slots of no agent (room for births to come) hold no value (NaN), and count for nothing
            active = np.zeros(len(target), dtype=bool)
            active[self.sim.people.auids] = True
            target = np.where(active, target, 0.0)
        top = target.max()
        on = target > 0
        weights = {}
        for state,c in coefs.items():
            ratio = np.broadcast_to(np.asarray(c, dtype=float), shape)[on]/target[on]
            if len(ratio) and not ((ratio == ratio[0]) | (np.abs(ratio - ratio[0]) <= 1e-8 + 1e-5*np.abs(ratio[0]))).all(): # As np.allclose(), in fewer passes
                raise ss.CapabilityError(f'"{self.mod.name}" has source states whose values on "{route.name}" differ between people other than in proportion, which agents cannot run yet; write the rate as one value by person times the source states (e.g. ss.by(...)*(A + 0.5*B)), or run with solver="ode"')
            weights[state] = float(ratio[0])*top if len(ratio) else 0.0
        return weights, (target/top if top > 0 else target)

    def transmit_acts(self, tr, i, nkey, route, rel_sus, by_strain, new_cases, pairs):
        """
        Transmission over a network with acts: each edge from an infectious `u` to a susceptible `v` transmits in a step with
        probability 1 − exp(−H), where H = (edge weight) × (acts in the step) × (hazard per act in the edge's direction, for
        `u`'s state) × rel_trans[u] × rel_sus[v] × (the modifiers on this route); so `rel_trans`, `rel_sus`, source weights,
        and modifiers all scale the hazard of each act. A network written as code that overrides `net_beta()` keeps
        Starsim 3's contract instead (see `ss.Network.custom_beta`): it is given the probability per act of each edge, and what
        it returns is the probability for the partnership in the step, which `rel_trans`, `rel_sus` and the modifiers then multiply.
        """
        mod = self.mod
        sim = self.sim
        hazards = {state: np.asarray(self.route_beta(coef, nkey, route), dtype=float) for state,coef in tr.terms.items()} # Directions (by groups) for each source state
        if not any(np.any(h > 0) for h in hazards.values()):
            return
        trans = times(self.current_value('rel_trans'), by_strain)
        n_slots = len(mod.rel_sus.raw)
        groups = sim.groups.group_raw() if any(h.ndim > 1 and h.shape[1] != n_slots for h in hazards.values()) else None
        mult = self.route_mult(tr.name, route)
        shared = {} # The source states that share a hazard (e.g. every infectious state of an STI: see route_beta()), with their weights summed once, so each edge takes one
        for state,h in hazards.items():
            w = mod[state.name].raw*state.rel_trans
            if id(h) in shared:
                shared[id(h)][1] = shared[id(h)][1] + w
            else:
                shared[id(h)] = [h, w]
        shared = list(shared.values())

        def act_hazard(src, trg):
            """ The hazard per act of each edge, for the state its infectious end is in """
            dirs = route.edge_directions(src, trg)
            act = np.zeros(len(src))
            for h,w in shared:
                if h.ndim == 1:
                    hs = h[dirs]
                elif h.shape[1] == n_slots: # A value for each agent (e.g. by a property of people, ss.by(sex=...)) is that of the person they would infect
                    hs = h[dirs, trg]
                else: # A value for each group is for the group of the people they would infect
                    hs = h[dirs, groups[trg]]
                act += w[src]*hs
            return act

        source = functools.reduce(operator.add, [w for h,w in shared]) # Whether each agent is infectious, with the rel_trans of their state
        if route.custom_beta: # Starsim 3's contract, whose net_beta() is for every edge
            edge_beta = lambda d, inds, src, trg: route.edge_net_beta(inds, disease_beta=-np.expm1(-act_hazard(*route.ends(d))), disease=mod)*mult(inds, trg)
            out = route.transmit(np.where(source > 0, trans, 0), rel_sus.raw, edge_beta, mod.trans_rng) # Only the edges from someone infectious: from anyone else the hazard per act is 0
        else:
            weight = lambda inds: route.edges.beta[inds]*route.edges.acts[inds]
            prob = lambda d, inds, src, trg: -np.expm1(-weight(inds)*act_hazard(src, trg)*trans[src]*rel_sus.raw[trg]*mult(inds, trg))
            out = route.transmit(trans*source, rel_sus.raw, lambda d, inds, src, trg: weight(inds), mod.trans_rng, prob=prob)
        for srcs, trgs in out:
            new_cases.append(trgs)
            pairs.append((srcs, trgs, i))
        return
