"""
The compartmental engine: a declared model run as groups of people (`ode`, `binomial`, `gillespie`, and tau-leaping)
"""
import importlib
import numpy as np
import numba as nb
import scipy.sparse as sps
import sciris as sc
import starsim as ss
from .engine import Bunch, CountState, Engine, StockCtx, add_rows, at_ends, by_level, no_tables

__all__ = ['Compartmental', 'synced_event']

max_raced = 400 # The most compartments for the race between non-exponential dwell times out of one state for the same people (the product of their Erlang stages, per group): see Compartmental.build_layout()


def synced_event(tr, sim, home):
    """
    For a transition triggered by an event (`ss.on(other.tr)`, `ss.on(other.tr.to(B))`): the transition of another
    declared module it is, as Bunch(mod, tr, to), which a compartmental run realizes as a synchronized flow (see
    `Compartmental.init_syncs()`); None for anything else (an event of its own module, entering or leaving a state, or
    an event a module records in code)
    """
    trig = tr.trigger
    if trig is None or not trig.event or not isinstance(trig.x, ss.conditions.Event):
        return None
    kind, obj, dst = trig.x.args
    if kind not in ['tr', 'to']:
        return None
    name = ss.conditions.modname(obj.module)
    declared = [m.name for m in getattr(sim, 'declared', [])]
    if name in [None, home] or name not in declared:
        return None
    if getattr(obj, 'name', None) not in sim.get_declared(name).graph.transitions: # An event a module records in code
        return None
    return Bunch(mod=name, tr=obj.name, to=getattr(dst, 'name', dst))


def choose_kernels(owner, runs, seconds):
    """
    The kernels for a run of a program (or `runs` runs), compiled or as plain Python, chosen once for a run (see `ss.kernels.choose()`), for an
    engine or a crossing (`owner`, which keeps the choice as `compiled`); `seconds()` is about how long the run would take as plain Python
    """
    if runs == 1 and owner.compiled is not None:
        return ss.kernels if owner.compiled else ss.kernels.uncompiled()
    if owner.program is None: # Before the program is made (the events of a register at the start), as the kernels are now
        return ss.kernels.choose()
    kn = ss.kernels.choose(runs*seconds())
    if runs == 1:
        owner.compiled = kn is ss.kernels
    return kn


def step_program(owner, kn, rng, y, tables, row, engine, check_short):
    """
    One step of a program taken from Python (see `ss.kernels.one_step()`), for an engine or a crossing (`owner`, which has the program and the
    register), with the step and substeps of `engine` (for a crossing, its first); then the events of a register dated at the end of the step (with
    `check_short()` on what they could not take), and under Gillespie, the counts by place settled after them, as in SimInf. Returns the compartments and the flows.
    """
    sim, p, dt = engine.sim, owner.program, engine.dt
    solver = sim.integrator if sim.solver == 'ode' else sim.solver
    y, flows = kn.one_step(rng, y, p, dt, ss.kernels.solvers[solver], engine.substeps(), tables, row)
    if sim.mixing.register is not None: # The events dated at the end of the step
        check_short(kn.apply_events(rng, y, owner.register, engine.mod.ti, engine.integer))
    if solver == 'gillespie' and p.local and p.has_det: # By place, the counts are settled after the events
        kn.settle_places(y, p, dt, flows)
    return y, flows


def run_many_parts(owner, rngs, ys, args, per_step, n_cols, outputs, state_of, rows, grp_state, grp_flow):
    """
    Many runs of a program that differ only in their seed (`rngs`), in calls to the kernels that take all of them (see `ss.kernels.run_many()`), for an
    engine or a crossing (`owner`); a part of the run at a time, so the tables of values that vary in time (by group, at every step) are not all held at
    once. `args` are those of `run_args()`, and `outputs` the arrays the kernels fill: the stocks, flows and tallies, and those by group
    """
    y0, program, dt, npts, solver, n_sub = args
    stocks, flows, stocks_g, flows_g, tally, tally_g = outputs
    length = max(int(50e6//(sum(len(table) for table in [program.fixed, program.mult, program.consts, program.rel_trans])*per_step*n_cols*8)), 1) # Steps per part
    start = 0
    kn = owner.kernels(runs=len(rngs))
    while True:
        end = min(start + length, npts - 1)
        kn.run_many(rngs, ys, program, dt, start, end + 1, solver, n_sub, *owner.tables(start, end), owner.register, owner.shifts, stocks, flows, stocks_g, flows_g,
                    tally, tally_g, state_of, np.zeros(0, dtype=int) if rows is None else rows, grp_state, grp_flow, per_step)
        if end >= npts - 1:
            break
        start = end
    return


class Compartmental(ss.utils.NoDists, Engine): # Never searched for distributions: it draws from its own generator, and the distributions in the model are found through its graph
    """
    Runs a graph as compartments: deterministically (`ode`) or stochastically
    (`binomial`, `gillespie`, or `tau`, the exact solver's fallback for long runs)

    The states are expanded into compartments (one per branch and sub-stage) and the
    transitions into a flat list of flows, each with a source compartment, a rate,
    and the compartments it feeds. Every solver steps the same list of flows.
    """
    max_binomial = 100 # The most substeps the chain-binomial solver chooses in a step (see substeps())
    max_bytes = 500e6 # The most memory that run_many() may use for the values of each group, over all its runs; above it, they are kept every few steps (see ss.solvers.group_rows())

    def __init__(self, module, sim):
        super().__init__(module, sim)
        self.n_groups = sim.groups.n_groups
        self.last_flows = dict() # The flow along each transition in the last step, by group
        self.forced = dict() # People moved along a transition by an intervention (see force()) this step, by group, until the step is recorded
        self.program = None # The flows as arrays, for the kernels: see init_program()
        self.whole = None # Why the kernels cannot take the whole run in one call, as far as it is fixed when the sim is initialized (False if nothing stops them): see kernel_reason()
        self.given = None # The output of a run taken elsewhere, for run_all() to use: see ss.gradients
        self.jax_run = None # The run in JAX, if one has been made: see ss.gradients
        self.cross = next((c for c in sim.crossings if module.name in c.names), None)
        self._n_sub = None # The chain binomial's substeps, once chosen: see substeps()
        return

    def define(self):
        """ Replace the declared states on the module with their run-time form """
        self.init_dims()
        for state in self.graph.states.values():
            self.mod.setattribute(state.name, CountState(self, state))
        return

    def init_pre(self):
        for name,tr in self.graph.transitions.items():
            if tr.dies and tr.stays and self.integer:
                ss.note(f'"{name}" of "{self.mod.name}" splits between ss.null and ss.stay: with solver="{self.sim.solver}", the count of its deaths is drawn apart from the move, so agrees with the number who died on average, not in each run; run with solver="abm" or "ode" for counts that agree in each run', sim=self.sim)
        self.init_dwells()
        self.init_immunity()
        self.build_layout()
        self.init_syncs()
        self.init_adjust()
        self.init_movers()
        self.init_results()
        self.note_steps()
        return

    def aging_rates(self, strat):
        """
        The rate per year at which people age out of each age group (but the last): the density of people in the year below
        its upper edge, over the number in the group

        With a given split (ages uniform within each group), this is one over its width; with a named population, or
        the world's as a guess, it is from that pyramid, as agents age (so a group followed by a smaller one does not empty too fast).
        """
        cdf = self.sim.mixing.age_cdf(strat)
        lo, hi = strat.edges[:-1], strat.edges[1:]
        h = np.minimum(1.0, hi - lo) # A year below the edge, or the whole group if it is narrower
        size = cdf(hi) - cdf(lo)
        return np.divide((cdf(hi) - cdf(hi - h))/h, size, out=1/(hi - lo), where=size>0)

    def init_movers(self):
        """ Movement between groups: aging, and the declared transfers """
        sim = self.sim
        graph = self.graph
        coords = sim.groups.group_coords
        mults = np.cumprod((sim.groups.group_shape + (1,))[::-1])[::-1][1:] # How far apart adjacent levels of each dimension are
        self.movers = []
        shared = self.shared
        for d,strat in enumerate(sim.groups.strats):
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            if strat.kind == 'age' and strat.edges is not None and comps and sim.pars.use_aging is not False and shared:
                if sim.unit is None:
                    ss.note('nobody ages, since time is in timesteps and age is in years; give the sim a unit to turn aging on', sim=sim)
                    continue
                rates = self.aging_rates(strat)/ss.years(1).to(sim.unit_base).value
                src = np.flatnonzero(coords[:, d] < strat.n - 1)
                matrix = self.mover_matrix(src, src + int(mults[d]), rates[coords[src, d]])
                self.movers.append(sc.objdict(name=None, comps=comps, matrix=matrix, transfer=None))
        for name,tf in graph.transfers.items():
            d = sim.groups.dim_index(tf.dim)
            strat = sim.groups.strats[d]
            ng = self.n_groups
            if tf.increment is not None: # A counter: each level to the one k up (or down)
                self.note_counter(strat, f'the transfer "{name}" of "{self.mod.name}"')
                src = np.arange(ng)
                dsts = [strat.shifted(tf.increment)[coords[:, d]]]
            else:
                dsts = [strat.index(dst) for dst in tf.dsts]
                srcs = [strat.index(tf.src)] if tf.src is not None else [i for i in range(strat.n) if i != dsts[0]]
                src = np.flatnonzero(np.isin(coords[:, d], srcs))
            states = tf.states or [s.name for s in graph.people_states]
            comps = [ind for state in states if self.applies[state][d] for ind in self.state_comps[state]]
            s = next((i for i,strat in enumerate(sim.groups.strats) if getattr(strat, 'stages_of', None) == (self.mod.name, name)), None)
            k = 1 if s is None else sim.groups.strats[s].n
            shift = 0
            if s is not None: # Erlang stages (see ss.dimensions.transfer_stages()): up a stage at k over the mean, and from the last to the level it moves to, at its first stage
                src = src[sim.groups.group_inside[src, s]] if sim.groups.group_fold is not None else src
                up = coords[src, s] < k - 1
                self.movers.append(sc.objdict(name=None, comps=comps, matrix=self.mover_matrix(src[up], src[up] + int(mults[s]), float(k)), transfer=tf, dest=None))
                src = src[~up]
                shift = (k - 1)*int(mults[s])
            moved = [src + (np.broadcast_to(dst, (ng,))[src] - coords[src, d])*int(mults[d]) - shift for dst in dsts]
            matrix = self.mover_matrix(np.tile(src, len(dsts)), np.concatenate(moved), np.repeat(np.arange(1.0, len(dsts) + 1), len(src))) # Tagged by the level each entry moves people to
            dest = matrix.data.astype(int) - 1 if tf.p is not None else None # With several, the share of each is applied with the rate: see mover_rates()
            matrix.data[:] = float(k)
            self.movers.append(sc.objdict(name=name, comps=comps, matrix=matrix, transfer=tf, dest=dest))
        if sim.mixing.migration is not None and graph.people_states and shared: # People relocating between places
            d = sim.mixing.space
            rates = sim.mixing.migration
            src, dst = np.nonzero(rates[coords[:, d]]) # Each group, and each place its people move to
            matrix = self.mover_matrix(src, src + (dst - coords[src, d])*int(mults[d]), rates[coords[src, d], dst])
            self.movers.append(sc.objdict(name=None, comps=self.people_comps, matrix=matrix, transfer=None, places=(src, dst, int(mults[d])))) # Where each entry goes, for a restart with new rates: see restart_movement()
        for move in (sim.groups.dim_moves if shared else []): # Movement that belongs to the sim, not to this module: e.g. the protection of a product waning
            d = sim.groups.dim_index(move.dim)
            src = np.flatnonzero(coords[:, d] == move.src)
            matrix = self.mover_matrix(src, src + (move.dst - move.src)*int(mults[d]), 1/self.convert(move.dur))
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            self.movers.append(sc.objdict(name=None, comps=comps, matrix=matrix, transfer=None))
        for mover in self.movers: # The people due to leave a conveyor leave it over the step (in the ODE), so are not moved between groups within it
            mover.comps = [c for c in mover.comps if c not in self.leaving]
        return

    def mover_matrix(self, src, dst, rates):
        """
        The rates of a mover from group to group, as a sparse matrix (each group's people move along one dimension, so to only a few groups)

        Moving into a group with no level of a dimension is moving to its first level, so people outside a dimension's `where=` do not move along it.
        """
        ng = self.n_groups
        fold = self.sim.groups.group_fold
        if fold is not None:
            dst = fold[dst]
        rates = np.broadcast_to(rates, src.shape).astype(float)
        keep = (src != dst) & (rates != 0)
        matrix = sps.csr_array((rates[keep], (src[keep], dst[keep])), shape=(ng, ng)) # Summed where two entries fall in one place, and in order
        matrix.sum_duplicates()
        return matrix

    @property
    def shared(self):
        """ Whether this engine moves people in ways that belong to the sim rather than to the module (aging, relocating, the movement register): in a crossing, only one does (see `ss.solvers.Crossing.lead`), since they all see the same people """
        return self.cross is None or self.cross.lead == self.mod.name

    def init_register(self):
        """ The movement register as arrays for the kernels (with no events, if there is none): see `ss.kernels.Register` """
        sim = self.sim
        mixing = sim.mixing
        ng = self.n_groups
        npts = self.mod.t.npts
        ints = lambda vals: np.array(vals, dtype=np.int64)
        ends = lambda lists: ints(np.cumsum([0] + [len(x) for x in lists]))
        if mixing.register is None or not self.graph.people_states:
            self.register = ss.kernels.Register(starts=ints([0]), src=ints([]), dst=ints([]), n=np.zeros(0), sel=ints([]), sel_starts=ints([0]), sel_comps=ints([]), sel_to=ints([]),
                                                ent_starts=ints([0]), ent_comps=ints([]), ent_w=np.zeros((0, ng)), pg_starts=ints([0]), pg_groups=ints([]), step=0)
            return self.register
        ev = mixing.events()
        d = mixing.space
        sets = self.register_sets()
        index = {(s.states, s.to_states):i for i,s in enumerate(sets)}
        chosen = [[c for name in s.chosen for c in self.state_comps[name]] for s in sets]
        def target(shift, name, i, comp): # The compartment people chosen from a compartment of a state go to: the same, or the matching one of the state they are moved to
            if name not in shift:
                return comp
            to = self.state_comps[shift[name]]
            return to[min(i, len(to) - 1)]
        moved_to = [[target(s.shift, name, i, c) for name in s.chosen for i,c in enumerate(self.state_comps[name])] for s in sets]
        added = []
        for s in sets:
            for name in s.added:
                tracks = [track for track in self.tracks[name] if track.first is not None]
                if len(self.tracks[name]) > 1:
                    errormsg = f'The movement register adds people to "{name}", which splits on entry into {len(self.tracks[name])} branches; adding people to such a state is not supported yet'
                    raise NotImplementedError(errormsg)
                added.append((tracks[0].first, self.inflow_local[name]))
        pg_starts, pg_groups = by_level(sim.groups.group_coords[:, d], sim.groups.strats[d].n)
        self.register = ss.kernels.Register(
            starts     = ints(np.searchsorted(ev.ti, np.arange(npts + 1))),
            src        = ev.src,
            dst        = ev.dst,
            n          = ev.n,
            sel        = ints([index[key] for key in ev.sets])[ev.set_of],
            sel_starts = ends(chosen),
            sel_comps  = ints([c for comps in chosen for c in comps]),
            sel_to     = ints([c for comps in moved_to for c in comps]),
            ent_starts = ends([s.added for s in sets]),
            ent_comps  = ints([comp for comp,w in added]),
            ent_w      = np.array([w for comp,w in added], dtype=float).reshape(len(added), ng),
            pg_starts  = ints(pg_starts),
            pg_groups  = ints(pg_groups),
            step       = int(sim.groups.group_mults[d]),
        )
        return self.register

    def mover_rates(self):
        """ The current rate matrix of each mover, from group to group """
        out = []
        for mover in self.movers:
            tf = mover.transfer
            if tf is None:
                out.append(mover.matrix)
            else:
                rate = (self.eval(tf.rate) if tf.rate is not None else 1/self.eval(tf.dur))*self.mult_raw('transition', tf.name)
                matrix = mover.matrix.copy() # With the same entries, each times the rate of the group it is from
                rows = np.repeat(np.arange(self.n_groups), np.diff(matrix.indptr))
                matrix.data = matrix.data*(rate if np.ndim(rate) == 0 else np.asarray(rate)[rows])
                if mover.dest is not None: # And the share that moves to each level
                    ps = self.branch_shares([f'{tf.name}.{dst}' for dst in tf.dsts], [None if p is None else self.eval(p) for p in tf.p], [None]*len(tf.p))
                    matrix.data = matrix.data*np.array([np.broadcast_to(p, (self.n_groups,)) for p in ps])[mover.dest, rows]
                out.append(matrix)
        return out

    def build_layout(self):
        """ Expand the states into compartments and the transitions into flows """
        graph = self.graph
        self.comps = [] # One per (state, track, stage)
        self.state_comps = sc.objdict({name:[] for name in graph.states})
        self.tracks = sc.objdict() # For each state: the tracks an arrival is split between
        self.flows = []
        self.init_triggers()
        self.tr_index = {name:i for i,name in enumerate(list(graph.transitions) + graph.destinations())} # After the transitions, the flow to each destination of a transition with several ('cin.X')
        self._target_key = None # So that update_targets() works out where the new flows go

        def add_comp(state, track, stage):
            self.comps.append(Bunch(state=state, track=track, stage=stage))
            self.state_comps[state.name].append(len(self.comps) - 1)
            return len(self.comps) - 1

        def add_flow(tr, src, kind, named=True, to=None, stages=None):
            for strain in (self.strain_levels() if kind == 'contact' else [None]): # A transmission is one flow for each strain it can come from
                self.flows.append(Bunch(tr=tr, src=src, kind=kind, named=named and not tr.stays, to=to, stages=stages, index=self.tr_index[tr.name], strain=strain, record=None, routes=False,
                                        state=None if src is None else self.comps[src].state, rel_sus=1.0 if src is None else self.comps[src].state.rel_sus))
            return

        def add_exits(ind, state, exits, competing, stays, slot=None, stayed=False):
            """ The exits that act on a compartment: in a slot of a conveyor (see ss.kernels.convey()), or past it (slot=None), given how long people there have stayed (and not one they took ss.stay out of) """
            for other in competing:
                if stayed and other.stays:
                    continue
                first, stop = stays.get(other.name, (0, None))
                if (stop is None and slot is None) or (slot is not None and first <= slot and (stop is None or slot < stop)):
                    add_flow(other, ind, 'hazard')
            for other in exits.contact:
                add_flow(other, ind, 'contact')
            if any(state is s for s in self.immune_states): # Infected again, by another strain, as far as the strains they carry do not protect them
                for other in [tr for tr in self.graph.transmissions if not getattr(tr, 'generated', False)]:
                    add_flow(other, ind, 'contact')
            return

        def add_raced(state, j, track, exits, competing, stays):
            """
            Several non-exponential dwell times out of a state for the same people (e.g. clearance against seeking care, ST61): each
            is a clock of Erlang stages, and they run at once, so a compartment for each combination of their stages (k1 × k2 for
            two). A stage of one clock moves people along its own axis, and its last stage is its exit, whatever stage the others
            are in; this is the exact form of the race between the Erlang dwells that agents draw
            """
            raced = track.raced
            ks = [self.dwells[tr.name].k for tr in raced]
            if np.prod(ks) > max_raced:
                raise ss.CapabilityError(f'"{raced[0].name}" and "{raced[1].name}" are non-exponential dwell times out of "{state.name}" for the same people: their race takes a compartment for each combination of their Erlang stages ({" × ".join(str(k) for k in ks)} = {np.prod(ks)}), more than {max_raced}.\n'
                                         f'     Give fewer stages (e.g. ss.Transition(..., stages=4)), or run with solver="abm".')
            combos = list(np.ndindex(*ks))
            inds = {combo:add_comp(state, j, None) for combo in combos}
            for combo in combos:
                ind = inds[combo]
                for e,(tr,k) in enumerate(zip(raced, ks)):
                    last = combo[e] == k - 1
                    to = None if last else inds[combo[:e] + (combo[e] + 1,) + combo[e+1:]]
                    add_flow(tr, ind, 'stage', named=last, to=to, stages=k)
                    self.flows[-1].raced = True # Not a chain of stages for the chain binomial (see self.chains)
                add_exits(ind, state, exits, competing, stays)
            kgs = [getattr(self.dwells[tr.name], 'kg', None) for tr in raced]
            if any(kg is not None for kg in kgs): # Ragged: a group with fewer stages of a clock enters further along its axis, so it passes through its own number
                start = np.ravel_multi_index([k - (np.full(self.n_groups, k) if kg is None else kg) for k,kg in zip(ks, kgs)], ks)
                track.starts = ([inds[combo] for combo in combos], start)
            return inds[combos[0]]

        self.conveyors = [] # The slots of each fixed delay and minimum stay, and the flow out of its last: see ss.kernels.convey()
        for name,state in graph.states.items():
            exits = graph.exits[name]
            tracks = []
            competing = list(exits.hazards) # Transitions that act on every compartment of the state
            staged = [tr for tr in exits.timed if self.dwells[tr.name].staged]
            competing += [tr for tr in exits.timed if not self.dwells[tr.name].staged]
            branches = exits.branches + sc.tolist(exits.remainder)
            stays = self.stays(name, exits, competing, staged, branches) # The slots of a conveyor that each transition with after= or before= acts on
            if branches: # Split on entry: one track per branch
                if staged:
                    self.unsupported(staged[0], f'a non-exponential dwell time out of "{name}" alongside branches with p=')
                for tr in branches:
                    p = tr.p[0] if tr.p is not None and not tr.split else None # A split between several destinations takes what the branches leave over
                    tracks.append(Bunch(tr=tr, p=p, immediate=tr.rate is None and (tr.dur is None or tr.immediate))) # With neither (or dur=0), nobody waits in the branch
            else:
                apart = len(staged) > 1 # Several, each for other people (where=, e.g. by sex): a track for each, which each group enters by the one it has (see apart_shares())
                if apart and (stays or any(self.dwells[tr.name].fixed for tr in staged)):
                    self.unsupported(staged[1], f'two non-exponential dwell times out of "{name}" when one is a fixed delay or the state has after= or before=')
                if apart and any(tr.where is None for tr in staged): # For the same people: they race, as one track of stages by each (see add_raced())
                    tracks.append(Bunch(tr=None, p=None, immediate=False, raced=staged))
                else:
                    tracks += [Bunch(tr=tr, p=None, immediate=False, apart=apart) for tr in (staged or [None])]
            if exits.held: # People who took ss.stay, with no exit of their own (p=0, entered only from the split)
                tracks.append(Bunch(tr=None, p=ss.transitions.par(0.0, literal=True), immediate=False, stay=True))
            for tr in exits.triggered: # Passed straight through in the groups where the trigger holds: see update_targets()
                if tr.trigger.outcome: # Moved when the outcome happens: see ss.interventions.trigger_moves()
                    continue
                if tr.trigger.event: # Moved by the other module's flow (see init_syncs()), or at the end of the step (see step_triggers())
                    continue
                tracks.append(Bunch(tr=tr, p=None, immediate=True, trigger=True))

            for j,track in enumerate(tracks):
                tr = track.tr
                track.first = None
                if track.immediate:
                    continue
                if getattr(track, 'raced', None):
                    track.first = add_raced(state, j, track, exits, competing, stays)
                    continue
                dwell = self.dwells.get(tr.name) if tr is not None else None # None for a branch whose rate changes with time: a hazard out of its own compartment
                fixed = dwell is not None and dwell.fixed # A fixed delay: a conveyor, out of which people leave by it
                n_slots = self.delay_steps(tr) if fixed else max([n for pair in stays.values() for n in pair if n is not None], default=0) # Or a minimum stay: a conveyor, then the rest of the state
                slots = [add_comp(state, j, None) for i in range(n_slots + 1)] if n_slots else []
                k = 0 if fixed else dwell.k if dwell is not None else 1
                inds = [add_comp(state, j, i) for i in range(k)]
                track.first = (slots + inds)[0]
                kg = getattr(dwell, 'kg', None)
                track.starts = None if (kg is None or slots) else (inds, k - kg) # Ragged: a group with fewer stages enters further along, so it passes through its own number
                acting = slots if (fixed and self.sim.solver == 'ode') else slots[:-1] # The last slot is left by the people due to leave in the next step: in the ODE, over the step, with the other exits acting on them until they do (see ss.kernels.release_rates()); the stochastic solvers release them at a point in the step (see ss.kernels.convey()), which already keeps the delay's exits exact
                for i,ind in enumerate(acting):
                    add_exits(ind, state, exits, competing, stays, slot=i)
                if slots:
                    add_flow(tr if fixed else self.graph.transitions[next(iter(stays))], slots[-1], 'conveyor', named=fixed, to=None if fixed else inds[0])
                    self.conveyors.append(Bunch(slots=slots, flow=self.flows[-1]))
                for i,ind in enumerate(inds):
                    if tr is not None and dwell is None:
                        add_flow(tr, ind, 'hazard')
                    elif tr is not None:
                        last = i == k-1
                        add_flow(tr, ind, 'stage', named=last, to=None if last else inds[i+1], stages=k)
                    add_exits(ind, state, exits, competing, stays, stayed=getattr(track, 'stay', False))
            self.tracks[name] = tracks

        for tr in graph.inflows:
            add_flow(tr, None, 'inflow')
        if self.sim.solver != 'ode': # The stochastic solvers draw in the order of the flows: by the names of their states and transitions, so the draws do not depend on the order these are declared in (the ODE keeps the order given, so its sums are as they were)
            rank = lambda c: (1, '', 0) if c is None else (0, self.comps[c].state.name, self.state_comps[self.comps[c].state.name].index(c))
            self.flows.sort(key=lambda flow: (*rank(flow.src), flow.tr.name, -1 if flow.strain is None else flow.strain))

        for flow in self.flows: # The expression each flow's rate is worked out from, if it is not a dwell time, and whether that can change within a step
            tr = flow.tr
            flow.expr = None if (flow.kind in ['stage', 'conveyor'] or (flow.kind != 'contact' and tr.dur is not None)) else tr.rate
            if flow.strain and tr.const is not None: # A part of the rate with no source of infection has no strain, so is counted once, with the first
                flow.expr = flow.expr - tr.const
            flow.dynamic = flow.expr is not None and any(isinstance(leaf, (ss.State, ss.transitions.Count)) for leaf in flow.expr.leaves)

        self.n_comps = len(self.comps)
        self.n_trans = len(self.tr_index)
        self.people_comps = [i for i,comp in enumerate(self.comps) if comp.state.people]
        self.y = np.zeros((self.n_comps, self.n_groups))
        self.leaving = {cv.slots[-1] for cv in self.conveyors} # The last slot of each conveyor, whose people leave it over the next step
        self.by_src = sc.ddict(list) # The flows out of each compartment, for the stochastic solvers (which release the last slot of a conveyor by itself)
        for flow in self.flows:
            if flow.kind != 'conveyor':
                self.by_src[flow.src].append(flow)
        stage_flows = {flow.src:flow for flow in self.flows if flow.kind == 'stage' and flow.strain is None and not getattr(flow, 'raced', False)}
        self.chains = dict() # For the chain-binomial solver: for each compartment of a dwell time in stages, the number of stages left, its flow to the next, and the flow out of the last (see ss.kernels.draw_chain())
        for c,flow in stage_flows.items():
            d = flow.stages - self.comps[c].stage
            if d > 1 and self.comps[c].state.people:
                self.chains[c] = (d, flow, stage_flows[c + d - 1])
        return

    def note_counter(self, strat, what):
        """ In a compartmental run, a note when a counter's levels are not one apart, so people within a wider level are taken to be at its lower edge when it goes up """
        if strat.edges is not None and np.any(np.diff(strat.edges) != 1):
            ss.note(f'{what} adds to "{strat.name}", whose levels ({sc.strjoin(strat.labels)}) are not one apart, so a compartmental run takes the people in each level to be at its lower edge; agents add to each person\'s own value', sim=self.sim)
        return

    def varying_dwell(self, name, tr, raw, dw, to_unit):
        """
        A dwell time whose distribution has parameters with a value for each group (`ss.by`)

        Each group gets the Erlang stages that match its own mean and standard deviation (ragged stages); a function as well has no
        compartmental form. (Agents look the parameters up for each agent, and make one draw: see `Agents.varying_dwell()`.)
        """
        trans = ss.transitions
        dw.family = type(raw).__name__
        dw.staged = True
        funcs = [key for key,val in trans.varying_pars(raw).items() if not isinstance(val, ss.by)]
        if funcs:
            raise ss.CapabilityError(f'"{name}" cannot run with solver="{self.sim.solver}": its {dw.family} dwell time has a function as its parameter "{funcs[0]}". Run with solver="abm".')
        dims = [strat.name for strat in self.sim.groups.strats]
        for val in trans.varying_pars(raw).values():
            for dim in val.dims:
                if dim not in dims:
                    raise ss.errors.MissingDimension(f'the dwell time of "{name}" is given by "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}={["female", "male"] if dim in ["sex", "female"] else [...]}).')
        dw.varying = True
        grids = {key:val.per_group(self.sim, to_unit) for key,val in trans.varying_pars(raw).items()}
        plain = {k:v for k,v in trans.numeric_pars(raw, to_unit, per_agent=lambda val: None).items() if k not in grids}
        stats = [trans.dwell_stats(type(raw)(**{**plain, **{k:float(np.broadcast_to(g, (self.n_groups,))[i]) for k,g in grids.items()}}), lambda v: v) for i in range(self.n_groups)]
        dw.mean, dw.std = [np.array(x) for x in zip(*stats)]
        if tr.stages is not None:
            dw.kg = np.full(self.n_groups, tr.stages)
        elif trans.exact_stages(raw) is not None: # A gamma with the same whole-number shape in every group
            dw.kg = np.full(self.n_groups, trans.exact_stages(raw))
        else:
            dw.kg = np.array([trans.erlang_stages(m, s) for m,s in zip(dw.mean, dw.std)])
            dw.exact = False
            ss.note(f'{name} has a {dw.family} dwell time that differs by group, approximated by {sc.strjoin(sorted(set(dw.kg.tolist())), sep="|")} Erlang stages (each group keeps its mean)', sim=self.sim)
            cv = np.max(np.divide(dw.std, dw.mean, out=np.zeros(len(dw.mean)), where=np.asarray(dw.mean) > 0))
            if cv > ss.solvers.choose.max_cv: # solver='auto' runs it as agents (FP53, FP83)
                ss.warn(f'{name} has a {dw.family} dwell time whose coefficient of variation is up to {cv:.3g}, so one exponential stage keeps only its mean, and its long tail is lost; run with solver="abm" to keep it')
        dw.k = int(dw.kg.max())
        if np.all(dw.kg == dw.k):
            dw.kg = None # Not ragged
        return

    def note_stages(self, name, dw):
        """ In a compartmental run, a dwell time that is not exponential or Erlang is approximated by Erlang stages, which is said in one line """
        sim = self.sim
        approx = sim.pars.approx
        unit = f' {sim.unit_name}s'
        std = dw.mean/np.sqrt(dw.k)
        if f'{std:.3g}' == f'{dw.std:.3g}':
            kept = f'mean {dw.mean:n}{unit} and std {std:.3g}{unit}, as given'
        else:
            kept = f'mean {dw.mean:n}{unit} kept; std {std:.3g}{unit}, against {dw.std:.3g}{unit} given'
        what = f'{dw.k} Erlang stage{"s" if dw.k > 1 else ""} ({kept})'
        if approx is False:
            raise ss.CapabilityError(f'"{name}" cannot run with solver="{sim.solver}" and approx=False.\n'
                                     f'     It has a {dw.family} dwell time, which would be approximated by {what}.\n'
                                     f'     Run with solver="abm", give stages= on the transition, or allow the approximation.')
        else:
            fixed = ' (nearly fixed, not exponential)' if dw.std < dw.mean/2 else '' # e.g. ss.SIR(dur_inf=ss.days(7)), whose std is 1 day
            ss.note(f'{name} has a {dw.family} dwell time{fixed}, approximated by {what}', ss.errors.StagesUnderAuto, sim=sim, verbose=0 if approx else None)
            if dw.std > ss.solvers.choose.max_cv*dw.mean: # solver='auto' runs it as agents (FP53)
                ss.warn(f'{name} has a {dw.family} dwell time whose coefficient of variation is {dw.std/dw.mean:.3g}, so one exponential stage keeps only its mean, and its long tail is lost; run with solver="abm" to keep it')
        return

    def unsupported(self, tr, what):
        raise ss.CapabilityError(f'"{tr.name}" cannot run with solver="{self.sim.solver}": {what} has no compartmental form yet.\n'
                                 f'     Run with solver="abm".')

    def whole_steps(self, tr, what, dur, lowest, nearest=False):
        """ A time as a number of whole steps (up, or to the nearest with `nearest=True`; at least `lowest`), with a note if it is not one """
        step = getattr(getattr(self.mod, 't', None), 'dt', None) or getattr(getattr(self.sim, 'pars', None), 'dt', None) # The sim's, if the module has no timeline yet (see init_crossings())
        if not isinstance(step, ss.dur): # Laid out before the module is in a sim, for its equations (see ss.transitions.Graph.flow_terms()): one slot
            return max(lowest, 1)
        x = float(self.convert(dur))/step.to(self.sim.unit_base).value
        n = max(lowest, int(np.ceil(x - (0.5 if nearest else 0) - 1e-9)))
        if abs(x - round(x)) > 1e-6:
            unit = self.sim.unit_name
            dt = float(self.convert(dur))/x
            ss.note(f'{tr.name} has {what} of {x*dt:n} {unit}s, which is {n} step{"s" if n != 1 else ""} ({n*dt:n} {unit}s) here (agents round each dwell to whole steps at random, keeping its mean); a step that divides it makes it exact', ss.errors.LargeStep, sim=self.sim)
        return n

    def delay_steps(self, tr):
        """ The length of a fixed delay (`ss.constant`), plus any minimum stay before it, in whole steps: agents leave at the start of the step nearest its end """
        dur = self.dwells[tr.name].mean + (float(self.convert(tr.after)) if getattr(tr, 'after', None) is not None else 0.0)
        return self.whole_steps(tr, 'a fixed delay', dur, lowest=1, nearest=True)

    def stays(self, name, exits, competing, staged, branches):
        """
        For each transition out of a state with a minimum stay (`after=`) or a limit (`before=`): the first slot of a conveyor it acts on, and the slot it stops at (None if it does not)

        A person in slot j has been in the state j whole steps, and agents count a stay as whole steps
        too (see `Agents.stayed()`). A dwell time with a minimum stay starts after it, so its stages come
        after the conveyor; a fixed delay with one is a longer conveyor (see `delay_steps()`).
        """
        stays = sc.objdict()
        for tr in competing + staged + exits.contact:
            after, before = getattr(tr, 'after', None), getattr(tr, 'before', None)
            if after is None and before is None:
                continue
            if branches:
                self.unsupported(tr, f'a minimum stay (after=) or a limit (before=) out of "{name}", which splits on entry (p=),')
            elif tr in exits.contact:
                self.unsupported(tr, 'a minimum stay (after=) or a limit (before=) on a transmission')
            elif tr in staged and self.dwells[tr.name].fixed:
                continue
            first = 0 if after is None else self.whole_steps(tr, 'a minimum stay', after, lowest=0)
            stays[tr.name] = (first, None if before is None else self.whole_steps(tr, 'a limit (before=)', before, lowest=0))
        for tr in staged:
            if tr.name in stays and stays[tr.name][0] != max(n for pair in stays.values() for n in pair if n is not None):
                self.unsupported(tr, f'a dwell time that starts after a minimum stay, out of "{name}" with a longer stay or limit,')
        return stays

    def init_post(self):
        """ Set the initial conditions """
        fracs, counts = self.init_fractions()
        self.update_targets()
        amounts = sc.objdict({name:frac*self.total_pop*self.state_split[name] for name,frac in fracs.items()})
        first = self.graph.first.name if self.graph.people_states else None
        if first in amounts and len(amounts) > 1: # The first state has the rest of the people in each of its groups (which, for a dimension it does not have, take in every level)
            others = np.broadcast_to(sum(val for name,val in amounts.items() if name != first), (self.n_groups,))
            amounts[first] = np.maximum(self.total_pop*self.state_split[first] - np.bincount(self.collapse[first], weights=others, minlength=self.n_groups), 0)
        infected = {state.name for tr in self.graph.acquisitions for state in tr.dst} # Where an infection puts people
        for name,amount in amounts.items():
            self.arrive(name, amount, self.y, before=name in infected)
        for name,count in counts.items():
            self.arrive(name, count*self.state_split[name], self.y, before=name in infected)
        burnin = self.mod.burnin_plan(self) # People part way through at the start (ss.Pregnancy's burnin=)
        if burnin is not None:
            self.backfill_burnin(burnin)
        if self.integer: # People are whole; a count that is not people (a reservoir) need not be
            self.y[self.people_comps] = self.round_preserving(self.y[self.people_comps])
        if any(not tr.static for tr in self.triggers) and self.cross is None: # Those a trigger on time or counts holds for move on before the start, not counted as flows
            self.sweep_triggers()
            for name in self.flowcounts:
                self.flowcounts[name] = 0.0
            self.forced = dict()
        if self.sim.mixing.register is not None and self.cross is None: # The events dated at the start (in a crossing, they happen to everyone once the diseases have their people: see ss.solvers.Crossing.init())
            self.check_short(self.kernels().apply_events(self.make_rng(stream=2), self.y, self.init_register(), 0, self.integer))
        self.record_stocks() # The results at the first time point are the start
        self.n_start = sum(self.y[self.state_comps[s.name]].sum() for s in self.graph.infected if s.people) # The number infected at the start: see note_chance()
        self.mod.setattribute('dists', None) # Compartments draw from the engine's own generator, so the module's distributions are never sampled and need not be advanced each step
        return

    def backfill_burnin(self, sp):
        """
        A module's people who start part way through (`ss.Pregnancy(burnin=True)`: see its `burnin_plan()`): each group's
        share of the first state (`sp.share()`) is moved to the target, spread evenly over its compartments (stages, or the
        slots of a fixed time), as if they entered uniformly over the steps before the start (the module's share takes off those
        who would have left on the way)
        """
        first, target = self.state_comps[self.graph.first.name], self.state_comps[sp.target[0]]
        moved = self.y[first]*np.asarray(sp.share(), dtype=float)
        self.y[first] -= moved
        self.y[target] += moved.sum(axis=0)/len(target)
        ss.note(sp.note, sim=self.sim)
        return

    def restart(self):
        """ Go back to the start of the run, with the current values of the parameters and the seed: see `ss.Sim.restart()` """
        self.rng = self.make_rng()
        self.kernel_rng = self.make_rng(stream=1)
        self.compiled = None
        self.cache.clear()
        self.y = np.zeros((self.n_comps, self.n_groups))
        self.init_post()
        for info in self.outcomes.values():
            info.update(now=0.0, n=0.0, total=0.0)
            info.pending[:] = 0
            if info.stay is not None:
                info.leaving[:] = 0
        return

    def restart_movement(self):
        """
        Put the movement between places, worked out again from new values of its parameters (see `ss.geo.Mobility.fitted()`), into the program, as a restart with them needs (see `ss.Sim.restart()`); False if the new values move people between places the old ones did not, so the sim must be built again

        The program keeps its entries, which the new values fill (one that becomes 0 stays as an
        entry of 0), so a compiled run and the gradients' check that the layout is the same both hold.
        """
        mixing = self.sim.mixing
        def onto(new, rows, cols):
            """ The entries of a new sparse matrix at the old one's, or None if it has others """
            vals = np.asarray(new[rows, cols], dtype=float).ravel() if len(rows) else np.zeros(0)
            return vals if np.count_nonzero(vals) == np.count_nonzero(new.data) else None
        p = self.program
        if p is not None and mixing.share is not None:
            vals = onto(mixing.share, p.sh_pair[0], p.sh_pair[1])
            if vals is None:
                return False
            p.sh_val[:] = vals
        for mover in self.movers:
            if mover.get('places') is not None:
                src, dst, step = mover.places
                coords = self.sim.groups.group_coords[:, mixing.space]
                new = self.mover_matrix(src, src + (dst - coords[src])*step, mixing.migration[coords[src], dst])
                old = mover.matrix
                vals = onto(new, np.repeat(np.arange(old.shape[0]), np.diff(old.indptr)), old.indices)
                if vals is None:
                    return False
                mover.matrix = old.copy()
                mover.matrix.data = vals
        return True

    @staticmethod
    def at_home(share):
        """ Whether a time share between places (a sparse array) keeps everyone at home, as `np.allclose(share, np.eye(P))` would say, without forming either """
        coo = share.tocoo()
        diag = coo.row == coo.col
        home = np.zeros(share.shape[0])
        home[coo.row[diag]] = coo.data[diag]
        return bool(np.all(np.abs(coo.data[~diag]) <= 1e-8) and np.all(np.abs(home - 1) <= 1e-8 + 1e-5))

    @staticmethod
    def round_preserving(y):
        """ Round to whole numbers, keeping the total """
        out = np.floor(y)
        extra = int(round(y.sum() - out.sum()))
        if extra > 0:
            order = np.argsort((out - y).ravel(), kind='stable')[:extra] # Largest remainders first
            out.ravel()[order] += 1
        return out

    @property
    def integer(self):
        return self.sim.solver in ['gillespie', 'tau', 'binomial']

    #%% The context used to evaluate expressions

    def count(self, states, total=True):
        out = 0
        for state in states:
            out = out + self.y[self.state_comps[state.name]].sum(axis=0)
        return out.sum() if total else out

    def set_count(self, state, value):
        """ Set the number in a state, keeping its split between groups and sub-compartments """
        inds = self.state_comps[state.name]
        current = self.y[inds]
        total = current.sum()
        if total > 0:
            self.y[inds] = current*(value/total)
        else:
            self.y[inds[0]] = value*self.state_split[state.name]
        self.changed()
        return

    def changed(self):
        """ After something in Python has changed the people (e.g. an intervention): in a crossing, the program of the diseases together and the other diseases take the change """
        if self.cross is not None:
            self.cross.put(self)
        return

    def n_people(self, total=True):
        out = self.y[self.people_comps].sum(axis=0)
        return out.sum() if total else out

    def local_count(self, states):
        mixing = self.sim.mixing
        if mixing.space is None:
            return self.count(states)
        return np.bincount(mixing.places, weights=self.count(states, total=False))[mixing.places]

    def reduce(self, value, state=None, what=None):
        """ A modified value is a scalar or one value per group; if it differs by state, the row for the state it is used for """
        if np.ndim(value) == 2:
            if value.shape[0] == 1 or np.all(value == value[0]):
                return value[0]
            elif state is None or not state.people:
                raise ss.CapabilityError(f'a modifier of "{what}" has a condition on the state people are in, which cannot run with solver="{self.sim.solver}" for a quantity that is not tied to one state.\n'
                                         f'     Run with solver="abm".')
            return value[self.selector.states.index(state.name)]
        return value

    def state_counts(self):
        """ The number of people in each state (rows) and group (columns) """
        return np.array([self.y[self.state_comps[name]].sum(axis=0) for name in self.selector.states]).reshape(self.selector.shape)

    def take(self, y, frac):
        """ Some of the people in a compartment: a fraction, or a draw where the solver keeps whole numbers """
        frac = np.clip(frac, 0, 1)
        return self.rng.binomial(y.astype(int), frac).astype(float) if self.integer else y*frac

    def targets(self, changes):
        """ The group that each group's people move to when their level in some dimensions is changed (`{dimension index: level}`) """
        sim = self.sim
        out = np.arange(self.n_groups)
        for d,level in changes.items():
            coords = sim.groups.group_coords[:, d]
            level = sim.groups.strats[d].target(level)[coords] # A level, a counter's ('+', amount), or a clock's ('=', time)
            out = out + (level - coords)*int(sim.groups.group_mults[d])
        return sim.groups.folded(out)

    def shift(self, frac, changes):
        """
        Move a fraction of the people in each state and group to another group, without changing their state

        Args:
            frac (arr): the fraction to move, states by groups
            changes (dict): the level to move them to, for each dimension that changes
        """
        target = self.targets(changes)
        moving = target != np.arange(self.n_groups)
        moved = np.zeros(self.n_groups)
        for s,name in enumerate(self.selector.states):
            for comp in self.state_comps[name]:
                n = self.take(self.y[comp], frac[s])*moving
                self.y[comp] -= n
                np.add.at(self.y[comp], target, n)
                moved += n
        self.changed()
        return moved

    def force(self, frac, tr, dst, srcs=None, y=None):
        """ Move a fraction of the people in each state a transition leaves from (or in `srcs`), and each group, to a state, counted in the transition's flow if there is one (as when a treatment cures); a synchronized flow moves other modules' coordinates too; with `y`, of those people (by compartment and group) rather than the engine's """
        syn = self.syncs_of(tr, dst.name) if tr is not None else []
        total, recorded = self.move_states(frac, tr.src if srcs is None else srcs, dst, syn=syn, y=y, move=self.moves(tr) if tr is not None else None) # With its change of level (ST30)
        if tr is None:
            return total
        self.flowcounts[tr.name] += recorded.sum()
        self.forced[tr.name] = self.forced.get(tr.name, 0) + recorded
        if syn:
            self.pass_syncs({sy.row:recorded*sy.mask for sy in syn})
        return recorded

    def move_states(self, frac, srcs, dst, syn=None, y=None, move=None, given=None):
        """ Move a fraction of the people in each of some states, and each group, to a state: along a transition (see `force()`), or along none, as a one-off move (`ss.vaccinate(move=S >> V)`); with `syn` (synchronized flows), also returns the moves in this module's own groups; with `y`, of those people rather than the engine's; with `given`, the number to move from each compartment instead (see `move_some()`) """
        own = y is None
        y = self.y if own else y
        total = np.zeros(self.n_groups)
        recorded = np.zeros(self.n_groups)
        entry = self.entry[dst.name][0]
        for state in srcs:
            if not state.people:
                continue
            s = self.selector.states.index(state.name)
            base = self.dim_map(state.name, dst.name, move=move)
            dim_map = self.synced(base, syn) if syn else base
            for comp in self.state_comps[state.name]:
                n = self.take(y[comp], frac[s]) if given is None else given[comp]
                y[comp] -= n
                total += n if dim_map is None else n @ dim_map
                recorded += n if base is None else n @ base # In this module's own groups, before the others' coordinates move
        if self.integer and len(entry) > 1: # Whole people, split at random between the branches of the destination
            left, rest = total.copy(), 1.0
            for ind,w in entry[:-1]:
                k = self.rng.binomial(left.astype(int), np.clip(w/rest, 0, 1))
                y[ind] += k
                left, rest = left - k, rest - w
            y[entry[-1][0]] += left
        else:
            self.arrive(dst.name, total, y)
        if own:
            self.changed()
        return (total, recorded) if syn is not None else total

    def move_some(self, srcs, dst, p=None, n=None):
        """ A one-off move from code (`ss.move()`): a share `p` of the people in some states (or `n` of them, exactly where the solver keeps whole people) to a state, or of a count, to and from outside the model too (`dst` or `srcs` empty); returns the number moved """
        comps = [comp for state in srcs for comp in self.state_comps[state.name]]
        have = self.y[comps]
        total = have.sum()
        if n is not None:
            p = min(n/total, 1.0) if total > 0 else 0.0
        if not (srcs[0] if srcs else dst).people: # A count: an exact share, as its flows are
            moved = have*p if srcs else n*self.state_split[dst.name]
            for comp,amount in zip(comps, moved):
                self.y[comp] -= amount
            if dst is not None:
                self.arrive(dst.name, moved.sum(axis=0) if srcs else moved, self.y)
            self.changed()
            return float(moved.sum())
        given = None
        if n is not None and self.integer: # Exactly n whole people, chosen at random from those there
            draw = self.rng.multivariate_hypergeometric(np.round(have).astype(np.int64).ravel(), int(min(round(n), round(total))))
            given = dict(zip(comps, draw.reshape(have.shape).astype(float)))
        frac = np.full(self.selector.shape, float(p)) if np.ndim(p) == 0 else np.broadcast_to(p, self.selector.shape)
        return float(self.move_states(frac, srcs, dst, given=given).sum())

    def move_share(self, tr, frac, srcs):
        """ Move a share of each state and group (within the sources `srcs`) along a triggered transition: with p=, that share of them; with several destinations, split between them """
        ps = self.branch_probs(tr) if tr.split else [self.selector.per_person(tr.p[0]) if tr.p else 1.0]
        N = self.state_counts()
        for dst,p in zip(tr.dst, ps): # Each as a share of those still in the source
            cur = self.state_counts()
            self.force(np.clip(np.divide(frac*p*N, cur, out=np.zeros_like(cur), where=cur>0), 0, 1), tr, dst, srcs=srcs)
        return

    def step_triggers(self):
        """
        The transitions moved by an event of this module (`on=` one of its transitions), at the end of each step: the share of each
        source state and group the step's event happened to moves then, as a delivery's people do (triggers on conditions are
        passed straight through instead: see `update_targets()` and `sweep_triggers()`)
        """
        for tr in self.event_triggers:
            srcs = [s for s in tr.src if s.people]
            frac = np.asarray(self.selector.select(tr.trigger.x), dtype=float)*self.selector.in_states([s.name for s in srcs])
            if tr.where is not None and tr.where is not tr.trigger.x:
                frac = frac*self.selector.select(tr.where)
            if frac.any():
                self.move_share(tr, frac, srcs)
        return

    def event_frac(self, name, to=None):
        """ The people who went through a transition in the last step (or one of its branches, `to`), as a fraction of the state they are now in """
        tr = self.graph.transitions[name]
        out = self.selector.nobody()
        flow = self.last_flows.get(name)
        if flow is not None:
            for dst,p in zip(tr.dst, self.branch_probs(tr)):
                if dst.people and to in [None, dst.name]:
                    n = self.count([dst], total=False)
                    out[self.selector.states.index(dst.name)] = np.clip(np.divide(flow*p, n, out=np.zeros(self.n_groups), where=n>0), 0, 1)
        return out

    def stock(self, state):
        if self.n_groups > 1:
            self.stock_groups[state.name] = self.count([state], total=False)
        return self.count([state])

    def exposure(self, state):
        """ Exposure to people in a state through the sim's mixing: the chance that a contact is with someone in that state """
        infectious = self.count([state], total=False)*state.rel_trans*self.mult_raw('rel_trans', state=state)
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            infectious = infectious*self.strain_weights[self.sim.groups.group_coords[:, self.strain], self._strain]
        mixing = self.sim.mixing
        mults = [self.contact_mult(term.name) for term in mixing.terms] if self.net_mods else None
        if self._routes is not None: # A beta for each route (for each source state, under the partner-change model)
            routes = self._routes[state.name] if isinstance(self._routes, dict) else self._routes
            mults = [beta*mult for beta,mult in zip(routes, mults or [1.0]*len(routes))]
        if getattr(self, '_rtr', None) is not None: # Modifiers of the transmission on some routes
            mults = [r*mult for r,mult in zip(self.route_mults(*self._rtr), mults or [1.0]*len(mixing.terms))]
        return mixing.exposure(infectious, self.n_people(total=False), mults)

    def prop(self, cond):
        """ A property of people or a condition, in a rate (see `ss.expressions.Prop`): its value for each group (for a condition, the share of the group it holds for; on states, of the people in the state the flow leaves) """
        out = np.asarray(cond.eval(self.selector), dtype=float)
        if out.ndim == 2 and out.shape[0] > 1: # By state
            src = getattr(self, '_src', None)
            if src is None or src.name not in self.selector.states:
                raise ss.CapabilityError(f'the condition {cond} is on states, which in a compartmental run is known only for the people leaving a state by a rate; run with solver="abm"')
            out = out[self.selector.states.index(src.name)]
        return out[0] if out.ndim == 2 else out

    #%% Flows

    def call_uids(self, func, p=None):
        """ A function that takes uids, which a compartmental run does not have (see `Agents.call_uids()`) """
        name = getattr(func, '__name__', 'function')
        raise ss.CapabilityError(f'the function "{name}" cannot run with solver="{self.sim.solver}".\n'
                                 f'     It takes uids, which exist only per agent. Run with solver="abm".')

    def clock(self, clock):
        """ A clock, which a compartmental run does not keep (see `Agents.clock()`) """
        raise ss.CapabilityError(f'{clock} cannot run with solver="{self.sim.solver}" yet: a clock is kept per agent, and its compartmental form (levels of elapsed time) is not built yet.\n'
                                 f'     Run with solver="abm".')

    def mother(self, expr):
        """ `ss.mother(x)`, which needs agents (see `Agents.mother()`) """
        if expr.born:
            raise ss.CapabilityError(f'{expr} cannot run with solver="{self.sim.solver}": it reads each agent\'s mother, and compartments do not know who anyone\'s mother is.\n'
                                     f'     Run with solver="abm", or leave the solver as "auto".')
        raise ss.CapabilityError(f'{expr} cannot run with solver="{self.sim.solver}": the mother of an unborn person is kept per agent, and compartments have no fetuses.\n'
                                 f'     Run with solver="abm", or leave the solver as "auto".')

    def stay_comp(self, name):
        """ The compartment of a state for people who stay in it with no exit of their own """
        return next(track.first for track in self.tracks[name] if getattr(track, 'stay', False))

    #%% Synchronized flows: a transition of another module triggered by one of this module's

    def init_syncs(self):
        """
        Synchronized flows (proposal 03 §2.3): a transition of another module triggered by a transition of this one
        (`ss.on(a.tr)`, or a branch, `ss.on(a.tr.to(X))`). The two modules are tracked jointly (`ss.solvers.Crossing`),
        so this module's flow also carries the other's coordinate from the source of its transition to its destination,
        in the same flow: exact. Each has a row of its own after the transitions, which the flow credits, and which the
        other module records as its transition's flow (see `pass_syncs()`)
        """
        sim = self.sim
        self.syncs = []
        self._sync_maps = {}
        moved = {id(flow.tr) for flow in self.flows}
        for mod in sim.declared:
            for otr in (mod.graph.transitions.values() if mod is not self.mod else []):
                ev = synced_event(otr, sim, mod.name)
                if ev is None or ev.mod != self.mod.name:
                    continue
                tr = self.graph.transitions[ev.tr]
                where = f'"{mod.name}.{otr.name}" (triggered by {otr.trigger.x.label}) cannot run with solver="{sim.solver}"'
                if self.cross is None or mod.name not in self.cross.names:
                    raise ss.CapabilityError(f'{where}: "{mod.name}" and "{self.mod.name}" are not tracked jointly. Run with solver="abm".')
                if otr.where is not None:
                    raise ss.CapabilityError(f'{where} with where=: a synchronized flow moves everyone in its source. Put the condition in the source states, or run with solver="abm".')
                if id(tr) not in moved:
                    raise ss.CapabilityError(f'{where}: "{self.mod.name}.{tr.name}" moves people with no wait (passed through on entry), so it has no flow of its own to synchronize with. Run with solver="abm".')
                if not otr.dst or any(not st.people for st in otr.src + otr.dst):
                    raise ss.CapabilityError(f'{where}: a synchronized flow moves people between states of people (not to ss.null or a count). Run with solver="abm".')
                self.syncs.append(Bunch(tr=tr, to=ev.to, mod=mod, otr=otr, row=self.n_trans, S=None, mask=None, rec=None))
                self.n_trans += 1
                if self.integer and len(tr.dst) > 1: # Otherwise it is exact
                    ss.note(f'{mod.name}.{otr.name} is triggered by {otr.trigger.x.label}, a synchronized flow, which moves people in {mod.name} as {self.mod.name}.{tr.name} moves them; with solver="{sim.solver}" its count is drawn apart from the move, so agrees with it on average', sim=sim)
        return

    def syncs_of(self, tr, dst):
        """ The synchronized flows of a transition (to a destination), with their maps worked out """
        return [self.sync_maps(sy) for sy in getattr(self, 'syncs', []) if sy.tr is tr and sy.to in [None, dst]]

    def sync_maps(self, sy):
        """
        For a synchronized flow: the map between groups that moves the other module's coordinate (its dimension in the
        crossing) from the levels of its transition's source to those of its destination (entry, split by its branches),
        the share of each group that it moves, and where the flow is recorded in the other module's groups
        """
        if sy.S is not None:
            return sy
        sim = self.sim
        groups = sim.groups
        other = sy.mod.engine
        if not hasattr(other, 'entry'):
            other.update_targets()
        d, a = groups.dim_index(sy.mod.name), groups.dim_index(self.mod.name)
        strat = groups.strats[d]
        coords, mults = groups.group_coords, groups.group_mults
        ng = self.n_groups
        level = lambda ind: other.people_comps.index(ind)
        def scalar(w, what):
            w = np.broadcast_to(np.asarray(w, dtype=float), (ng,))
            if np.ptp(w) > 1e-12:
                raise ss.CapabilityError(f'"{sy.mod.name}.{sy.otr.name}" is triggered by {sy.otr.trigger.x.label}, and {what} differs between groups, which a synchronized flow cannot carry yet. Run with solver="abm".')
            return float(w[0])
        dst = sc.ddict(float) # Level of the other's dimension -> share, for people leaving the source
        ps = [scalar(p, 'its branch probability') for p in other.branch_probs(sy.otr)]
        for state,p in zip(sy.otr.dst, ps):
            comps, credits = other.entry[state.name]
            if credits:
                raise ss.CapabilityError(f'"{sy.mod.name}.{sy.otr.name}" is triggered by {sy.otr.trigger.x.label}, and its destination "{state.name}" is passed straight through, which a synchronized flow cannot carry yet. Run with solver="abm".')
            for ind,w in comps:
                dst[level(ind)] += p*scalar(w, 'where its destination is entered')
        stay = 1 - sum(ps)
        S = np.zeros((ng, ng))
        mask = np.zeros(ng)
        inside = np.zeros(ng, dtype=bool) # The groups whose people are in the other's source
        src = {l:(state.name, l) for state in sy.otr.src for l in strat.state_levels[state.name]}
        for g in range(ng):
            b = coords[g, d]
            if b not in src:
                S[g, g] = 1.0
                continue
            mask[g] = 1 - stay
            inside[g] = True
            for l,w in dst.items():
                S[g, g + (l - b)*int(mults[d])] += w
            if stay > 1e-12: # ss.stay: to the source state's compartment with no exit of its own
                held = [t for t in other.tracks[src[b][0]] if getattr(t, 'stay', False)]
                l = level(held[0].first) if held else b
                S[g, g + (l - b)*int(mults[d])] += stay
        own = sc.ddict(float) # This module's level after the move, where the other's flow is recorded
        for state,p in zip(sy.tr.dst, self.branch_probs(sy.tr)):
            if sy.to in [None, state.name]:
                for ind,w in self.entry[state.name][0]:
                    if ind in self.people_comps:
                        own[self.people_comps.index(ind)] += float(np.mean(p))*float(np.mean(w))
        total = sum(own.values()) or 1.0
        rec = [(g, groups.folded(g - coords[g, d]*int(mults[d]) + (l - coords[g, a])*int(mults[a])), w/total) for g in np.flatnonzero(mask) for l,w in own.items()] # From each of this module's groups to the other's
        sy.S, sy.mask, sy.inside = groups.fold_matrix(S), mask, inside
        sy.rec = tuple(np.array(v) for v in zip(*rec)) if rec else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
        return sy

    def synced(self, dim_map, syn):
        """ The map of a flow along a transition that other modules' transitions are triggered by: its own, then moving their coordinates """
        if not syn:
            return dim_map
        key = (id(dim_map),) + tuple(sy.row for sy in syn)
        if key not in self._sync_maps:
            if len(syn) > 1 and not np.any(np.sum([sy.inside for sy in syn], axis=0) > 1): # From different sources (e.g. starting from N, restarting from L): each moves the people in its own source as they were, not those another has just moved
                eye = self.sim.groups.fold_matrix(np.eye(self.n_groups))
                both = eye + sum(sy.S - eye for sy in syn)
                out = both if dim_map is None else dim_map @ both
            else:
                out = dim_map
                for sy in syn:
                    out = sy.S if out is None else out @ sy.S
            self._sync_maps[key] = (dim_map, out) # The map is kept with it, so its id is not reused
        return self._sync_maps[key][1]

    @staticmethod
    def sync_rec(sy, val):
        """ A synchronized flow by this module's groups (the last axis), as the other module's """
        gi, gj, w = sy.rec
        out = np.zeros(val.shape)
        np.add.at(out.T, gj, (val[..., gi]*w).T)
        return out

    def pass_syncs(self, flows):
        """ After a step, or a move by an intervention: give each module whose transition is triggered by one of this one's the people its flow moved, as its transition's flow """
        for sy in self.syncs:
            if isinstance(flows, dict) and sy.row not in flows:
                continue
            val = np.asarray(flows[sy.row], dtype=float)
            if not val.any():
                continue
            val = self.sync_rec(sy, val)
            other = sy.mod.engine
            name = sy.otr.name
            other.flowcounts[name] += val.sum()
            if getattr(other, 'stepped', None) == other.mod.ti: # It has taken its step already: add to what it recorded
                other.last_flows[name] = other.last_flows.get(name, 0) + val
                if other.n_groups > 1:
                    other.flow_groups[name] = other.flow_groups.get(name, 0) + val
            else:
                other.forced[name] = other.forced.get(name, 0) + val
        return

    def init_triggers(self):
        """ Triggers (`ss.on()`): on states and strata, the compartments where they hold vanish; on time or counts, they are applied at each step """
        sim = self.sim
        self.triggers = [tr for tr in self.graph.transitions.values() if tr.trigger is not None and not tr.trigger.event and not tr.trigger.outcome]
        self.event_triggers = [tr for tr in self.graph.transitions.values() if tr.trigger is not None and tr.trigger.event and synced_event(tr, sim, self.mod.name) is None] # An event of this module: moved at the end of the step, see step_triggers()
        for tr in self.triggers:
            tr.static = ss.conditions.is_static(tr.trigger.x, sim) and ss.conditions.is_static(tr.where, sim)
            if tr.static:
                ss.note(f'{tr.name} is triggered by {tr.trigger}: arrivals where it holds go straight to {sc.strjoin(d.name for d in tr.dst)}; people who come to meet it by aging, a transfer, or another disease are moved at the next step', sim=sim)
        return

    def trigger_frac(self, tr):
        """ Where a trigger holds, as the fraction of each state (rows) and group (columns) it selects """
        sel = self.selector
        out = np.asarray(sel.select(tr.trigger.x), dtype=float)
        if tr.where is not None:
            out = out*np.asarray(sel.select(tr.where), dtype=float)
        return np.broadcast_to(out, (len(sel.states), self.n_groups))

    def trigger_mask(self, tr, name):
        """ The groups of a state where a trigger on states and strata holds (1) or not (0); 0 for one on time or counts, which is applied at each step instead: see sweep_triggers() """
        if not tr.static:
            return np.zeros(self.n_groups)
        return np.clip(self.trigger_frac(tr)[self.selector.states.index(name)], 0, 1)

    def sweep_triggers(self):
        """ At each step: move the people a trigger holds for who are still in its source (who arrived by aging, a transfer, or another disease's move, or for whom a condition on time or counts has come to hold) """
        for tr in self.triggers:
            frac = np.array(self.trigger_frac(tr))
            rows = [self.selector.states.index(s.name) for s in tr.src if s.people]
            keep = np.zeros(frac.shape[0], dtype=bool)
            keep[rows] = True
            frac = np.where(keep[:, None], frac, 0.0)
            if not np.any(frac > 0):
                continue
            rest = 1.0
            for dst,p in zip(tr.dst, self.branch_probs(tr)):
                share = np.divide(p, rest, out=np.zeros(np.broadcast(p, rest).shape), where=np.asarray(rest) > 0)
                self.force(frac*share, tr, dst)
                frac = frac*(1 - share)
                rest = rest - p
        return

    def update_targets(self):
        """ Work out where each flow's arrivals go; done once per step, since branch probabilities can change (and only worked out again if they have) """
        probs = sc.objdict()
        for name,tracks in self.tracks.items():
            plain = [tr for tr in tracks if not getattr(tr, 'trigger', False)]
            apart = [track for track in plain if getattr(track, 'apart', False)]
            ps = self.apart_shares(name, apart) if apart else [None if tr.p is None else self.eval(tr.p) for tr in plain]
            mults = self.where_mults([tr.tr for tr in plain], name, [self.branch_mult(tr.tr) for tr in plain]) # With each branch's where=: closed in the groups it does not hold in
            probs[name] = self.branch_shares([tr.tr for tr in plain], ps, mults)
            if len(plain) < len(tracks): # Triggers on states and strata: in the groups where one holds, arrivals pass straight through (the compartments vanish)
                left = np.ones(self.n_groups)
                masks = []
                for track in tracks[len(plain):]:
                    mask = self.trigger_mask(track.tr, name)*left
                    masks.append(mask)
                    left = left - mask
                probs[name] = [p*left for p in probs[name]] + masks
        key = [np.asarray(p, dtype=float) for shares in probs.values() for p in shares] + [np.asarray(p, dtype=float) for tr in self.graph.transitions.values() if tr.dst is not None and tr.split for p in self.branch_probs(tr)]
        last = getattr(self, '_target_key', None)
        if last is not None and len(last) == len(key) and all(np.array_equal(a, b) for a,b in zip(last, key)):
            return
        self._target_key = key
        self.entry = sc.objdict() # For each state: [(compartment, weight)] and [(transition index, weight, parent)], where a credit's parent is the one it is nested in (its index in the list), or -1

        def enter(name, weight, comps, credits, path=(), parent=-1):
            if name in path: # The same-step cascade, by construction: see ss.solvers.Agents.junctions
                cycle = ' -> '.join(path[path.index(name):] + (name,))
                raise ss.errors.TriggerLoop(f'moves with no wait out of the states of "{self.mod.name}" form a loop: {cycle}.\n'
                                     f'     Give one of them a rate or a duration, or change the branches so they do not lead back.')
            for track,p in zip(self.tracks[name], probs[name]):
                if track.immediate: # Nobody waits in this branch: pass straight through to its destination
                    dps = self.branch_probs(track.tr)
                    dps = self.branch_probs(track.tr)
                    die = self.die_share(track.tr, dps) # Those who took ss.null leave the model, as by an outflow
                    credits.append((self.tr_index[track.tr.name], weight*p*(sum(dps) + die if track.tr.stays else 1), parent))
                    me = len(credits) - 1
                    for dst,dp in zip(track.tr.dst, dps):
                        sub = me
                        if track.tr.ends:
                            credits.append((self.tr_index[f'{track.tr.name}.{dst.name}'], weight*p*dp, me))
                            sub = len(credits) - 1
                        enter(dst.name, weight*p*dp, comps, credits, path + (name,), sub)
                    if track.tr.dies and track.tr.ends and track.tr.stays: # Otherwise, the flow less the flows to its states (see count_deaths())
                        credits.append((self.tr_index[f'{track.tr.name}.null'], weight*p*die, me))
                    if track.tr.stays: # Those who took ss.stay
                        comps.append((self.stay_comp(name), weight*p*(1 - sum(dps) - die)))
                elif getattr(track, 'starts', None) is not None:
                    inds, start = track.starts
                    for j in np.unique(start):
                        comps.append((inds[j], weight*p*(start == j)))
                else:
                    comps.append((track.first, weight*p))
            return

        for name in self.tracks:
            comps, credits = [], []
            enter(name, 1.0, comps, credits)
            self.entry[name] = (comps, credits)

        for flow in self.flows:
            if flow.to is not None: # A move to the next sub-stage
                flow.targets, flow.credits, flow.recs = [(flow.to, 1.0, None)], [], [-1]
            else:
                flow.targets, flow.credits, flow.recs = [], [], []
                src = None if flow.src is None else flow.state.name
                ps = self.branch_probs(flow.tr)
                for dst,p in zip(flow.tr.dst, ps):
                    comps, credits = self.entry[dst.name]
                    syn = self.syncs_of(flow.tr, dst.name)
                    for ind,w in comps:
                        dim_map = self.synced(self.dim_map(src, self.comps[ind].state.name, flow.strain, move=self.moves(flow.tr)), syn) # Moving other modules' coordinates too (a synchronized flow)
                        if dim_map is not None and np.ndim(w): # A branch share decided on entry, by the groups people land in, which can have a dimension the source lacks (TL16: it was read at the first level)
                            landed = dim_map*w
                            total = landed.sum(axis=1)
                            flow.targets.append((ind, p*total, np.divide(landed, total[:, None], out=np.zeros_like(landed), where=total[:, None] > 0)))
                            self._landed = True # The map changes with the shares, so the program is packed again whenever they do
                        else:
                            flow.targets.append((ind, w*p, dim_map)) # (compartment, share, where in it)
                    flow.recs += [self.tr_index[f'{flow.tr.name}.{dst.name}'] if (flow.named or flow.tr.stays) and flow.tr.ends else -1]*len(comps) # Also counted in the flow to this destination
                    off = len(flow.credits) # Credits are (transition index, weight, parent): the credit it is nested in, -1 for none, or -2 if it overlaps the others (drawn on its own)
                    flow.credits += [(ind, w*p, par + off if par >= 0 else -1) for ind,w,par in credits]
                    flow.credits += [(sy.row, p*sy.mask, -2) for sy in syn] # Moved in another module too
                die = self.die_share(flow.tr, ps) # Those who took ss.null go to no compartment: they leave the model, as by an outflow
                if flow.tr.dies and flow.tr.ends and flow.tr.stays: # Otherwise, the flow less the flows to its states (see count_deaths())
                    flow.credits.append((self.tr_index[f'{flow.tr.name}.null'], die, -2))
                if flow.tr.stays and src is not None: # Those who took ss.stay go to the state's compartment with no exit of its own, and are not in the flow
                    flow.targets.append((self.stay_comp(src), 1 - sum(ps) - die, None))
                    flow.recs.append(-1) # One for each target, so those of the flows after it line up
                    if flow.kind != 'stage' or flow.to is None:
                        flow.credits.append((flow.index, sum(ps) + die, -2))
                if flow.strain is not None:
                    flow.record = self.dim_map(src, src, flow.strain, record=True)
                elif self.enters(flow.tr, flow.state): # Recorded at the level people arrive at, e.g. E -> I where only I has a clinical status
                    flow.record = self.dim_map(src, flow.tr.dst[0].name)
        if self.program:
            weights = [item[1] for flow in self.flows for item in flow.targets + flow.credits]
            if getattr(self, '_landed', False) or len(weights) != len(self._weights) or not all(np.array_equal(a, b) for a,b in zip(weights, self._weights)):
                self.pack_targets()
        return

    def apart_shares(self, name, tracks):
        """ For several non-exponential dwell times out of a state, each for other people (where=): the share of each group that enters each one's track, 1 for the one it has (the first, if none) """
        ng = self.n_groups
        firsts = [next(flow for flow in self.flows if flow.tr is track.tr and flow.src == track.first) for track in tracks]
        active = np.array([np.broadcast_to(np.asarray(self.flow_mult(flow)) > 0, (ng,)) for flow in firsts], dtype=float)
        both = active.sum(axis=0) > 1
        if both.any():
            trs = [track.tr.name for track,a in zip(tracks, active[:, both]) if a.any()]
            raise ss.CapabilityError(f'"{trs[0]}" and "{trs[1]}" are non-exponential dwell times out of "{name}" for the same people, which have no compartmental form yet; give each a where= for other people (e.g. ss.ppl.female and ss.ppl.male), or run with solver="abm"')
        active[0] += active.sum(axis=0) == 0 # Those with none of them stay in the first, where nothing moves them
        return list(active)

    def add_per_births(self):
        """ After a step: the children of those who went through a transition with births driven by its flow (`ss.birth(per=)`), `n` for each, in the youngest groups of their parents' place """
        for tr in self.graph.inflows:
            if getattr(tr, 'per', None) is None or ss.conditions.modname(tr.per.module) not in [None, self.mod.name]: # Another module's children, placed here by the crossing (see ss.solvers.init_crossings())
                continue
            flow = np.broadcast_to(self.last_flows.get(tr.per.name, 0.0), (self.n_groups,))
            if not flow.any():
                continue
            n = tr.n_born.raw if isinstance(tr.n_born, ss.par) else tr.n_born
            if isinstance(n, ss.Dist):
                n = ss.transitions.dwell_stats(n, self.convert)[0] # The mean number of children
            born = float(n)*flow @ self.birth_map(tr)
            if self.integer: # Whole children
                born = self.rng.poisson(born).astype(float)
            self.arrive(tr.dst[0].name, born, self.y)
            if self.cross is not None: # Into the joint states, where the other diseases' children start by their dimensions' splits
                self.cross.put(self)
            for engine,other in self.births_of(tr): # Counted as this module's births, and as the children other modules place (ss.birth(per=) of this transition)
                engine.flowcounts[other.name] += np.sum(born)
                engine.last_flows[other.name] = engine.last_flows.get(other.name, 0.0) + born
                if engine.n_groups > 1:
                    engine.flow_groups[other.name] = engine.last_flows[other.name]
        return

    def births_of(self, tr):
        """ The engine and inflow of each declared module with an `ss.birth(per=)` of the same transition as `tr` (this module's own first) """
        per = (ss.conditions.modname(tr.per.module) or self.mod.name, tr.per.name)
        out = []
        for mod in self.sim.declared:
            for other in mod.graph.inflows:
                if getattr(other, 'per', None) is not None and (ss.conditions.modname(other.per.module) or mod.name, other.per.name) == per:
                    out.append((mod.engine, other))
        return sorted(out, key=lambda pair: pair[0] is not self)

    def birth_map(self, tr):
        """ Where the children of the people in each group are born (rows to columns): the youngest group of each dimension, by the split of the others, in their parents' place """
        key = ('birth', tr.name)
        if key not in self.maps:
            groups = self.sim.groups
            coords = groups.group_coords
            out = np.ones((groups.n_groups, groups.n_groups))
            for d,strat in enumerate(groups.strats):
                if strat.kind == 'space':
                    out *= coords[:, d][:, None] == coords[:, d][None, :]
                else:
                    out *= np.asarray(self.inflow_factors[tr.dst[0].name][d])[None, :]
            self.maps[key] = groups.fold_matrix(out)
        return self.maps[key]

    def arrive(self, name, amount, y, before=False):
        """ Add an amount to a state, split between its tracks; with before=True, as if they arrived a step before (people infectious at the start, who have been so for the first step, as agents date them), so one slot along a conveyor """
        along = {conv.slots[0]:conv.slots[1] for conv in self.conveyors if len(conv.slots) > 1} if before else {}
        for ind,w in self.entry[name][0]:
            y[along.get(ind, ind)] += amount*w
        return

    def flow_base(self, flow):
        """ The rate of a flow before its multipliers: the only part that can depend on the number in each compartment """
        if flow.kind == 'inflow' and getattr(flow.tr, 'per', None) is not None: # Births driven by a flow: added after each step (see add_per_births())
            return 0.0
        elif flow.kind == 'conveyor': # The last slot of a conveyor has no rate: its people leave in the step after they reach it, as its inflow in the program (see ss.kernels.convey())
            return 1.0
        elif flow.expr is None:
            kg = self.dwells[flow.tr.name].kg if flow.kind == 'stage' else None # Ragged stages: each group's own number
            return ((flow.stages if kg is None else kg) if flow.kind == 'stage' else 1)/self.dwell_mean(flow.tr.name)
        routes = flow.routes if flow.routes is not False else self.route_betas(flow)
        rtr = (flow.tr, flow.state) if flow.kind == 'contact' and self.route_mods(flow.tr) else None
        with self.evaluating(strain=flow.strain, routes=routes, rtr=rtr, src=flow.state): # The state, for a condition on states in the rate (see prop())
            return self.eval(flow.expr)

    def route_betas(self, flow):
        """
        For a transmission with a beta for each route (beta=dict(home=..., school=...)), or with a route that has acts: the beta
        of each term of the mixing, which exposure() applies (and the beta itself is then 1); None otherwise
        """
        flow.routes = None
        if flow.kind != 'contact':
            return None
        nets = {ss.utils.standardize_netkey(name):net for name,net in self.sim.all_networks().items()}
        terms = self.sim.mixing.terms
        lasting = any('partnership' in term for term in terms) # Under the partner-change model, by source state
        acts = lasting or any(getattr(net, 'has_acts', False) for net in nets.values())
        betas = [leaf for leaf in flow.expr.leaves if isinstance(leaf, ss.par) and (isinstance(leaf.raw, (dict, ss.peract)) or (acts and leaf.slot == 'rate'))]
        if len({id(b) for b in betas if isinstance(b.raw, dict)}) > 1:
            raise ss.CapabilityError(f'"{flow.tr.name}" has a rate for each route with its own sources (e.g. dict(random=b*(A + 0.5*B), home=b2*(A + B))), which only agents can run yet: run with solver="abm", or give one beta for each route times the same sources, e.g. dict(random=b, home=b2)*(A + 0.5*B).')
        if betas:
            raw = betas[0].raw
            def route(term, state=None): # The keys of a dict are checked when the sim is initialized
                key = ss.utils.standardize_netkey(term.name)
                val = ss.utils.route_beta(raw, term.name) if isinstance(raw, dict) else raw
                net = nets.get(key)
                if 'partnership' in term:
                    return self.partnership_chance(val, net, term, flow, state)
                return float(self.plain_value(val))
            flow.routes = {state.name:[route(term, state) for term in terms] for state in flow.tr.terms} if lasting else [route(term) for term in terms]
            self.routed.add(id(betas[0]))
        return flow.routes

    def partnership_chance(self, val, net, term, flow, state):
        """
        Under the partner-change model (see `ss.solvers.Mixing.partnerships()`), the chance that a partnership with
        someone in a source state, formed now, transmits to the people in each group before it stops being able to:
        `H/(H + k)` with `k = 1/D̄ + c`, for partnerships that end at a constant rate, or `H/(H + c)·(1 − E[exp(−(H + c)·T)])`
        over their duration `T` otherwise; `H` is the hazard per unit time in the partnership (acts per unit time times the
        hazard per act in the direction from the partner's side, or the beta itself on a network without acts), and `c`
        the rate at which the partner leaves the source state or the person leaves the state they are infected from
        """
        pt = term.partnership
        if getattr(net, 'has_acts', False):
            h = np.asarray(self.act_hazards(val, net), dtype=float)*pt.acts
        else:
            h = np.atleast_1d(np.asarray(self.plain_value(val), dtype=float))
        c = self.exit_rate(state) + self.exit_rate(flow.state)
        side = pt.group_side
        names = list(net.sides or {})
        dirs = net.directions
        out = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            if side[g] < 0:
                continue
            d = f'{names[term.src_side]}2{names[side[g]]}' if names else None
            if d is not None and d not in dirs: # No partnership joins these sides
                continue
            k = dirs.index(d) if d is not None else 0
            H = h[k] if h.ndim == 1 else h[k][g]
            if pt.durs is None:
                out[g] = H/(H + c + 1/pt.dur)
            else:
                out[g] = H/(H + c)*(1 - np.mean(np.exp(-(H + c)*pt.durs))) if H + c > 0 else 0.0
        self.partnership_info[(flow.tr.name, state.name, net.name, term.src_side)] = out
        return out

    def exit_rate(self, state):
        """ The rate at which people leave a state other than by infection (one over their mean time in it), as a mean over groups """
        exits = self.graph.exits[state.name]
        mean = lambda v: float(np.mean(v))
        rate = sum(1/mean(self.dwell_mean(tr.name)) for tr in exits.timed if tr.name in self.dwells)
        rate += sum(mean(self.flow_base(flow)*self.flow_mult(flow)) for flow in self.flows if flow.state is state and any(flow.tr is tr for tr in exits.hazards))
        branches = [tr for tr in exits.branches + [exits.remainder] if tr is not None and tr.name in self.dwells]
        if branches: # Split on entry: one over the mean of their times
            rate += 1/mean([mean(self.dwell_mean(tr.name)) for tr in branches])
        return rate

    def branch_where(self, tr):
        """ Whether a transition is a branch out of each of its sources whose where= is on states and strata, so closes it at entry (see `branch_mult()`) rather than changing its rate """
        return tr.where is not None and ss.conditions.is_static(tr.where, self.sim) and all(tr in self.graph.exits[s.name].branches + sc.tolist(self.graph.exits[s.name].remainder) for s in tr.src if s.people)

    @staticmethod
    def trait_mean(mod, a, sim):
        """ Whether an attribute (`ss.attr`) has a compartmental form: a trait drawn from a distribution, with no modifier, which a compartmental run takes at its mean (see `attr()`) """
        return a.kind == 'trait' and isinstance(a.value, ss.Dist) and not sim.modifiers.get((mod.name, 'attr', a.name))

    def attr(self, a):
        """ An attribute in a compartmental run: a trait is the mean of its distribution, as a parameter that is a distribution is (see `trait_mean()`) """
        mod = a.module if isinstance(a.module, ss.Module) else self.mod
        if not self.trait_mean(mod, a, self.sim):
            raise ss.CapabilityError(f'the attribute "{a.name}" is kept per agent, so cannot run with solver="{self.sim.solver}"; run with solver="abm"')
        return ss.transitions.dwell_stats(a.value, self.convert)[0]

    def flow_mult(self, flow):
        """ The multipliers on a flow's rate, which hold for the whole step """
        tr = flow.tr
        mult = 1.0
        if flow.kind == 'conveyor': # When people leave a conveyor is fixed
            mods = [m for m in self.sim.modifiers.get((self.mod.name, 'transition', tr.name), []) if not (m.name == f'where_{self.mod.name}_{tr.name}' and self.branch_where(tr))] # A branch's own where= closes it at entry instead: see branch_mult()
            if flow.named and mods:
                raise ss.CapabilityError(f'"{tr.name}" has a fixed delay and a modifier, which cannot run with solver="{self.sim.solver}": a compartmental run keeps the delay as a conveyor of whole steps, so cannot shorten or lengthen it.\n'
                                         f'     Run with solver="abm".')
            return mult
        elif flow.kind == 'contact':
            mult = flow.rel_sus*self.mult_raw('rel_sus', state=flow.state)
            if flow.strain is not None and self.strain_adjust[tr.name] is not None:
                mult = mult*self.strain_adjust[tr.name][flow.strain]
            if flow.state.name in self.protect: # Protected in part by the strains they carry
                mult = mult*self.protection(flow.state, self.sim.groups.group_coords[:, self.strain], flow.strain)
            if self.memory and flow.strain is not None: # And by those they remember
                mult = mult*self.memory_mult(flow.state, flow.strain)
            if flow.strain is not None and self.sim.modifiers.get((self.mod.name, 'strain_sus', None)): # By a product against acquiring the strain
                mult = mult*self.strain_effect(flow.strain, state=flow.state)
            if flow.strain is not None and self.carriage is not None and self.applies[flow.state.name][self.strain] and self.superinfects(tr, flow.state): # Infected with another strain while staying in the same state: only if they do not carry it already
                mult = mult*self.can_take(self.sim.groups.group_coords[:, self.strain], flow.strain)
        elif tr.acquisition and flow.src is not None: # An acquisition that is not contact-mediated, e.g. by a bite rate: the susceptibility of its source too
            mult = flow.rel_sus*self.mult_raw('rel_sus', state=flow.state)
        if self.adjust[tr.name] is not None:
            mult = mult*self.adjust[tr.name]
        return mult*self.mult_raw('transition', tr.name, state=flow.state) # An inflow's too (MA36)

    def rates(self):
        """ The rate of every flow, per unit time and per member of its source compartment (or in total, for an inflow) """
        return [self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows]

    #%% The program run by the kernels

    def init_program(self):
        """
        Flatten the flows into arrays for the kernels: see `ss.kernels`

        A value that is a function of the sim is called once a step (see `refresh_program()`),
        and held within it, unless it gives an expression of the model's states (`I.n`), which
        is flattened in its place (and again whenever the expression changes). A beta for each
        route, and a modifier of a network's contacts for some people only (`where=`), make the
        contacts through each term differ by exposure and group (see `set_contacts()`).
        """
        sim = self.sim
        mixing = sim.mixing
        kn = ss.kernels
        ng, nf = self.n_groups, len(self.flows)
        pars, sets, exposures = dict(), dict(), dict() # What the operations act on, in order
        ops, args, starts = [], [], [0]
        self.foreign_pars = getattr(self, 'foreign_pars', dict()) # Counts of other diseases' states, as values worked out each step

        def add(items, key, val=None):
            """ The position of an item, adding it if it is new """
            items.setdefault(key, val)
            return list(items).index(key)

        def check(p):
            raw = getattr(p, 'raw', None) # None for a property of people (ss.ppl.age), which does not change
            varies = isinstance(raw, ss.transitions.TimeValue) or (isinstance(raw, ss.by) and raw.varies)
            if varies and not any(p is par for par in self.time_pars):
                self.time_pars.append(p)

        def flatten(expr, strain, routes=None, rtr=None):
            """ Add the operations for an expression, and return the size of stack they need (`rtr`: the transmission and its source state, if modifiers act on it on some routes) """
            if isinstance(expr, ss.transitions.BinOp):
                a = flatten(expr.a, strain, routes, rtr)
                b = flatten(expr.b, strain, routes, rtr)
                ops.append(kn.opcodes[expr.op])
                args.append(0)
                return max(a, b + 1)
            elif isinstance(expr, ss.transitions.Func) and expr.name == 'expit': # As 1/(1 + exp(-x)), which the kernels have
                return flatten(1/(1 + np.exp(-expr.args[0])), strain, routes, rtr)
            elif isinstance(expr, ss.transitions.Func):
                depths = [flatten(arg, strain, routes, rtr) + i for i,arg in enumerate(expr.args)]
                ops.append(kn.opcodes[expr.name])
                args.append(0)
                return max(depths)
            elif isinstance(expr, ss.par):
                if id(expr) in self.func_exprs: # A function that gives an expression
                    return flatten(self.func_exprs[id(expr)][1], strain, routes, rtr)
                check(expr)
                ops.append(kn.CONST)
                args.append(add(pars, id(expr), expr))
            elif isinstance(expr, ss.transitions.Count) and self.split_states(expr.states)[1]: # Another disease's states, counted by it once a step, as a function of the sim is
                if id(expr) not in self.foreign_pars:
                    self.foreign_pars[id(expr)] = ss.par((lambda e: lambda sim: self.count_expr(e))(expr), name=f'{ss.transitions.format_expr(expr)} (counted by its disease)', literal=True)
                return flatten(self.foreign_pars[id(expr)], strain, routes, rtr)
            elif isinstance(expr, ss.transitions.Count):
                ops.append(kn.COUNT if expr.kind == 'n' else kn.TOTAL if expr.kind == 'total' else kn.FRAC)
                args.append(add(sets, tuple(ind for state in expr.states for ind in self.state_comps[state.name])))
            elif isinstance(expr, ss.expressions.Prop): # A value for each group, which does not change (a condition on states: for the state the flow leaves)
                key = (id(expr), getattr(getattr(self, '_src', None), 'name', None))
                if key not in self.foreign_pars:
                    self.foreign_pars[key] = ss.par(self.prop(expr.cond), name=str(expr.cond), literal=True)
                return flatten(self.foreign_pars[key], strain, routes, rtr)
            elif isinstance(expr, ss.expressions.attr): # A trait, at the mean of its distribution (see attr())
                if id(expr) not in self.foreign_pars:
                    self.foreign_pars[id(expr)] = ss.par(self.attr(expr), name=f'{expr.name} (its mean)', literal=True)
                return flatten(self.foreign_pars[id(expr)], strain, routes, rtr)
            elif isinstance(expr, ss.State):
                routes = routes[expr.name] if isinstance(routes, dict) else routes # Under the partner-change model, for each source state
                ops.append(kn.EXPOSURE)
                args.append(add(exposures, (expr.name, strain, None if routes is None else tuple(tuple(np.atleast_1d(r).tolist()) for r in routes), None if rtr is None else (rtr[0].name, rtr[1].name)), (expr, routes, rtr)))
            else:
                errormsg = f'"{self.mod.name}" has a rate with a part the compartmental solvers cannot work out: {expr!r}'
                raise TypeError(errormsg)
            return 1

        self.time_pars = [] # The parameters whose value changes over time
        self.series = dict() # Their values at each step: see tables()
        self.func_exprs = dict() # For a function of the sim that gives an expression of the model's states (I.n), by the id of its parameter: the parameter, the expression, and its key (see expr_key())
        for flow in self.flows:
            for leaf in ([] if flow.expr is None else flow.expr.leaves):
                if isinstance(leaf, ss.par) and callable(leaf.raw) and not isinstance(leaf.raw, (ss.Dist, ss.transitions.TimeValue)):
                    if id(leaf) not in self.func_exprs:
                        out = self.call(leaf.raw, leaf)
                        if isinstance(out, ss.transitions.Expr):
                            self.func_exprs[id(leaf)] = (leaf, out, self.expr_key(out))
                    if id(leaf) in self.func_exprs: # Worked out within the step, as the expression is
                        flow.dynamic = True
        depth = 1
        for flow in self.flows:
            if flow.dynamic:
                routes = (flow.routes if flow.routes is not False else self.route_betas(flow)) if flow.kind == 'contact' else None
                rtr = (flow.tr, flow.state) if flow.kind == 'contact' and self.route_mods(flow.tr) else None
                with self.evaluating(src=flow.state):
                    depth = max(depth, flatten(flow.expr, flow.strain, routes, rtr))
            else:
                for leaf in ([] if flow.expr is None else flow.expr.leaves) + sc.tolist(self.dwells.get(flow.tr.name, Bunch(par=None)).par):
                    check(leaf)
            starts.append(len(ops))
        for cv in self.conveyors: # After the flows' operations, each conveyor's (its flow, then its slots): see ss.kernels.convey()
            ops += [kn.CONVEY]*(len(cv.slots) + 1)
            args += [next(i for i,flow in enumerate(self.flows) if flow is cv.flow)] + cv.slots
            starts.append(len(ops))

        place_of = np.array(mixing.places) if mixing.space is not None else np.zeros(ng, dtype=int)
        pl_starts, pl_groups = by_level(place_of, place_of.max() + 1)
        self._local_mixing = all(term.kind == 'homogeneous' for term in mixing.terms) and (mixing.space is None or mixing.time_share is None or self.at_home(mixing.time_share))

        # Exposure: each group is in one cell of each mixing term, and the cells are mixed by the time share between places and the term's matrix between levels (see Mixing.term_exposure())
        nt = len(mixing.terms)
        sizes = np.ones(nt, dtype=int)
        block = np.ones(nt, dtype=int)
        levels = np.zeros((nt, ng), dtype=int)
        use = np.zeros((3, nt), dtype=bool) # Whether the share is used before prevalence is worked out, the matrix after, and then the share
        for t,term in enumerate(mixing.terms):
            if term.kind == 'homogeneous' and mixing.space is not None: # People mix where they are
                levels[t] = mixing.places
                sizes[t] = sim.groups.strats[mixing.space].n
                use[:, t] = [mixing.share is not None and mixing.present, False, mixing.share is not None]
            elif term.kind != 'homogeneous':
                levels[t] = term.cells if 'cells' in term else sim.groups.group_coords[:, term.dim]
                block[t] = len(term.matrix)
                sizes[t] = term.get('size', block[t])
                use[:, t] = [bool(term.get('pre')), True, bool(term.get('post'))]
        mat = np.zeros((nt, block.max(), block.max()))
        for t,term in enumerate(mixing.terms):
            if use[1, t]:
                mat[t, :block[t], :block[t]] = term.matrix
        share = mixing.share if mixing.share is not None else sps.csr_array((1, 1))

        self.prog_pars = list(pars.values())
        self.prog_exposures = []
        self.prog_routes = [] # The beta of each term of the mixing, for an exposure through a transmission with a beta for each route (or None)
        self.prog_rtrs = [] # The transmission and its source state, for an exposure through a transmission that modifiers act on, on some routes (or None)
        for (name,strain,key,rkey),(state,routes,rtr) in exposures.items():
            add(sets, tuple(self.state_comps[name]))
            masked = strain is not None and self.applies[name][self.strain] # Only the people who carry this strain
            self.prog_exposures.append((state, self.strain_weights[sim.groups.group_coords[:, self.strain], strain] if masked else 1.0))
            self.prog_routes.append(routes)
            self.prog_rtrs.append(rtr)
        by_group = any(routes is not None for routes in self.prog_routes) or any(getattr(m, 'where', None) is not None for m in self.net_mods) or any(self.prog_rtrs) # Contacts by exposure and group: see set_contacts()
        setlist = list(sets)
        order = {id(flow):i for i,flow in enumerate(self.flows)}
        flat = lambda lists: np.array([i for item in lists for i in item], dtype=int)
        ends = lambda lists: np.cumsum([0] + [len(item) for item in lists])

        self.program = kn.Program(
            n_trans    = self.n_trans,
            src        = np.array([-1 if flow.src is None or flow.kind == 'conveyor' else flow.src for flow in self.flows], dtype=int), # The types are given for a disease with no flows
            named      = np.array([flow.named for flow in self.flows], dtype=bool),
            index      = np.array([flow.index for flow in self.flows], dtype=int),
            record     = np.full(nf, -1),
            inflow     = np.array([self.inflow_split(flow.tr) if flow.src is None else np.zeros(ng) for flow in self.flows], dtype=float).reshape(nf, ng), # For the flow out of a conveyor, set by the ODE each step (see ss.kernels.convey())
            fixed      = np.zeros((nf, ng)),
            mult       = np.ones((nf, ng)),
            depth      = depth,
            starts     = np.array(starts, dtype=int),
            ops        = kn.check_ops(ops),
            args       = np.array(args, dtype=int),
            consts     = np.zeros((len(pars), ng)),
            set_starts = ends(setlist),
            set_comps  = flat(setlist),
            people     = np.array(self.people_comps, dtype=int),
            has_det    = any(self.det_flow(flow) for flow in self.flows),
            det        = np.array([self.det_flow(flow) for flow in self.flows], dtype=bool),
            places     = np.array(mixing.places) if mixing.space is not None else np.zeros(ng, dtype=int),
            n_places   = sim.groups.strats[mixing.space].n if mixing.space is not None else 1,
            exp_set    = np.array([setlist.index(tuple(self.state_comps[name])) for name,strain,key,rkey in exposures], dtype=int),
            rel_trans  = np.ones((len(exposures), ng)),
            levels     = levels,
            sizes      = sizes,
            block      = block,
            contacts   = np.array([term.get('contacts', 1.0) for term in mixing.terms], dtype=float).reshape(1, nt, 1)*np.ones((len(exposures), 1, ng) if by_group and exposures else (1, 1, 1)),
            use        = use,
            mat        = mat,
            sh_pair    = np.array([np.repeat(np.arange(share.shape[0]), np.diff(share.indptr)), share.indices], dtype=int).reshape(2, -1),
            sh_val     = share.data.astype(float),
            tgt_starts = np.zeros(nf + 1, dtype=int),
            tgt_comp   = np.zeros(0, dtype=int),
            tgt_w      = np.zeros((0, ng)),
            tgt_map    = np.zeros(0, dtype=int),
            tgt_rec    = np.zeros(0, dtype=int),
            crd_starts = np.zeros(nf + 1, dtype=int),
            crd_index  = np.zeros(0, dtype=int),
            crd_w      = np.zeros((0, ng)),
            crd_parent = np.zeros(0, dtype=int),
            maps       = np.zeros((0, ng, ng)),
            n_movers   = len(self.movers),
            mov_starts = np.array([ends([mover.comps for mover in self.movers]), ends([mover.matrix.data for mover in self.movers])], dtype=int),
            mov_comps  = flat([mover.comps for mover in self.movers]),
            mov_pair   = np.array([flat([np.repeat(np.arange(ng), np.diff(mover.matrix.indptr)) for mover in self.movers]), flat([mover.matrix.indices for mover in self.movers])], dtype=int).reshape(2, -1),
            mov_rate   = np.zeros(sum(mover.matrix.nnz for mover in self.movers)),
            src_starts = ends(list(self.by_src.values())),
            src_flows  = flat([[order[id(flow)] for flow in flows] for flows in self.by_src.values()]),
            src_chain  = np.array([self.chains[src][0] if src in self.chains else 0 for src in self.by_src], dtype=int),
            src_stage  = np.array([order[id(self.chains[src][1])] if src in self.chains else -1 for src in self.by_src], dtype=int),
            src_last   = np.array([order[id(self.chains[src][2])] if src in self.chains else -1 for src in self.by_src], dtype=int),
            local      = False, # Set in pack_targets()
            pl_starts  = pl_starts,
            pl_groups  = pl_groups,
        )
        self.set_contacts()
        self.pack_targets()
        self.kernel_rng = self.make_rng(stream=1) # As a whole run starts with, so stepping gives the same run
        return

    def set_contacts(self):
        """ The contacts through each term of the mixing in the program, times the modifiers of the network's contacts and of a transmission on a route; by exposure and group, with the beta of each route, if they differ by any of these """
        p = self.program
        terms = self.sim.mixing.terms
        by_group = any(routes is not None for routes in self.prog_routes) or any(self.prog_rtrs) # As in init_program(), also with one exposure in one group
        if not self.prog_exposures or (p.contacts.shape[0] == p.contacts.shape[2] == 1 and not self.net_mods and not by_group): # Not used, or as built
            return
        mults = [self.contact_mult(term.name) for term in terms] if self.net_mods else [1.0]*len(terms)
        for e in range(p.contacts.shape[0]):
            routes = self.prog_routes[e] if p.contacts.shape[0] > 1 or p.contacts.shape[2] > 1 or by_group else None
            rmults = self.route_mults(*self.prog_rtrs[e]) if self.prog_rtrs[e] is not None else [1.0]*len(terms)
            for t,term in enumerate(terms):
                p.contacts[e, t] = term.get('contacts', 1.0)*mults[t]*(1.0 if routes is None else routes[t])*rmults[t]
        return

    def route_mods(self, tr):
        """ Whether modifiers act on a transmission on some of its routes (`ss.modify(..., route=)`) """
        return any(self.sim.modifiers.get((self.mod.name, 'route', f'{tr.name}@{term.name}')) for term in self.sim.mixing.terms)

    def route_mults(self, tr, state):
        """ The multiplier on the contacts through each term of the mixing, from the modifiers of a transmission on that route: `1 − share·(1 − multiply)`, a number or one for each group (of the people who would be infected) """
        out = []
        for term in self.sim.mixing.terms:
            mult = 1.0
            for m in self.sim.modifiers.get((self.mod.name, 'route', f'{tr.name}@{term.name}')) or []:
                if m.active:
                    mult = mult*self.reduce(m.apply(1.0, self), state=state, what=f'{tr.name} on {term.name}')
            out.append(mult)
        return out

    @staticmethod
    def expr_key(out):
        """ What a function's expression is made of, to tell when it changes: its text, and the exact value of each number in it (None if it is not an expression) """
        if not isinstance(out, ss.transitions.Expr):
            return None
        return (ss.transitions.format_expr(out), tuple(repr(leaf.raw) for leaf in out.leaves if isinstance(leaf, ss.par)))

    def prog_value(self, par):
        """ The value of a parameter in the program: a beta for each route is 1 here (with any modifiers), since it is applied to the exposure through each (see `set_contacts()`) """
        return self.modified(par, 1.0) if (isinstance(par.raw, dict) or id(par) in self.routed) else self.value(par)

    @staticmethod
    def det_flow(flow):
        """ Whether a flow is of a state that is not people, so is never drawn as whole events """
        return not (flow.state.people if flow.src is not None else flow.tr.dst[0].people)

    def pack_targets(self):
        """ Put where each flow goes into the program; done again whenever a branch probability changes """
        ng = self.n_groups
        maps = dict() # Each map that is used, once

        def which(dim_map):
            if dim_map is None:
                return -1
            maps.setdefault(id(dim_map), dim_map)
            return list(maps).index(id(dim_map))

        self._weights = [item[1] for flow in self.flows for item in flow.targets + flow.credits] # To tell when they change
        emptied = lambda flow: [(flow.src, -1.0, None)] if flow.kind == 'conveyor' else [] # The flow out of a conveyor empties its last slot (see ss.kernels.convey())
        targets = [(ind, w, which(dim_map)) for flow in self.flows for ind,w,dim_map in flow.targets + emptied(flow)]
        credits = [(ind, w, par + start if par >= 0 else par) for flow,start in zip(self.flows, np.cumsum([0] + [len(flow.credits) for flow in self.flows])) for ind,w,par in flow.credits]
        weights = lambda items: np.array([np.broadcast_to(item[1], ng) for item in items], dtype=float).reshape(len(items), ng)
        self.program = self.program._replace(
            record     = np.array([which(flow.record) for flow in self.flows], dtype=int),
            tgt_starts = np.cumsum([0] + [len(flow.targets) + len(emptied(flow)) for flow in self.flows]),
            tgt_comp   = np.array([item[0] for item in targets], dtype=int),
            tgt_w      = weights(targets),
            tgt_map    = np.array([item[2] for item in targets], dtype=int),
            tgt_rec    = np.array([rec for flow in self.flows for rec in flow.recs + [-1]*len(emptied(flow))], dtype=int),
            crd_starts = np.cumsum([0] + [len(flow.credits) for flow in self.flows]),
            crd_index  = np.array([item[0] for item in credits], dtype=int),
            crd_w      = weights(credits),
            crd_parent = np.array([item[2] for item in credits], dtype=int),
            maps       = np.array(list(maps.values()), dtype=float).reshape(len(maps), ng, ng),
        )
        local = self._local_mixing and not maps and self.program.contacts.shape[0] == self.program.contacts.shape[2] == 1 and all(flow.record is None for flow in self.flows) # Places do not affect each other within a step: see ss.kernels.gillespie_places()
        self.program = self.program._replace(local=local)
        return

    def refresh_program(self):
        """ Set the numbers in the program that hold for this step: the rates that do not depend on the compartments, and the values the others use """
        if self.func_exprs and any(self.expr_key(self.call(par.raw, par)) != key for par,out,key in self.func_exprs.values()): # A function now gives another expression
            rng = self.kernel_rng
            self.init_program()
            self.kernel_rng = rng
        p = self.program
        for i,flow in enumerate(self.flows):
            if flow.dynamic:
                p.mult[i] = self.flow_mult(flow)
            else:
                p.fixed[i] = self.flow_base(flow)*self.flow_mult(flow)
        for i,par in enumerate(self.prog_pars):
            p.consts[i] = self.prog_value(par)
        for i,(state,mask) in enumerate(self.prog_exposures):
            p.rel_trans[i] = state.rel_trans*self.mult_raw('rel_trans', state=state)*mask
        if self.net_mods or any(self.prog_rtrs):
            self.set_contacts()
        for i,matrix in enumerate(self._mover_rates):
            p.mov_rate[p.mov_starts[1, i]:p.mov_starts[1, i+1]] = matrix.data
        return p

    @property
    def smooth(self):
        """ Whether any parameter varies smoothly in time (not as a step function), so is worked out at each stage of an ODE step """
        if getattr(self, '_smooth', None) is None:
            raws = [p.raw for p in self.graph.all_pars()]
            self._smooth = any(isinstance(raw, (ss.transitions.TimeValue, ss.by)) and not self.held(raw) for raw in raws)
        return self._smooth

    def step_tables(self, dt, smooth=True):
        """ For a step in the kernels: the program at the start, middle, and end of the step if a value varies smoothly in time, and the row to start at (or none, and -1, if the program holds for the whole step) """
        if not (smooth and self.smooth):
            self.refresh_program()
            return no_tables, -1
        p = self.program
        rows = []
        for frac in [0, 0.5, 1]:
            self.at_time(frac*dt)
            self.refresh_program()
            rows.append([p.fixed.copy(), p.mult.copy(), p.consts.copy(), p.rel_trans.copy()])
        self.at_time(None)
        return tuple(np.array([r[k] for r in rows]) for k in range(4)), 0

    def at_time(self, t):
        """
        Take values that vary smoothly in time at a time t (in the sim's unit) into the step, which runs from time point `ti-1` to `ti`: the year
        worked out as `tables()` does, so that a stepped run takes the same values as a whole one; with t=None, at its start, as for the rest
        """
        if self.smooth:
            years, ti = self.mod.t.yearvec, self.mod.t.ti
            end = years[ti]
            start = years[ti-1] if ti else end
            frac = None if t is None else t/self.dt
            self.year = None if frac is None else (start + end)/2 if frac == 0.5 else end if frac == 1 else start + frac*(end - start)
            self.cache.clear()
        return

    def changes(self, y, new):
        """ The rate of change of each compartment from new infections (`new=True`: arrivals only), or from every other flow; see `next_generation()` """
        self.y = y
        self.cache.clear()
        dy = np.zeros_like(y)
        for flow,rate in zip(self.flows, self.rates()):
            if flow.new != new:
                continue
            if flow.kind == 'conveyor': # A conveyor, as stages whose stays add up to its length: half a step in its first slot and in its last, as in the ODE
                rate = 2/self.dt
            amount = rate*y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr)
            if flow.src is not None and not new:
                dy[flow.src] -= amount
            for ind,w,dim_map in flow.targets:
                dy[ind] += amount*w if dim_map is None else (amount*w) @ dim_map
        if not new:
            for mover,matrix in zip(self.movers, self.mover_rates()):
                ym = y[mover.comps]
                dy[mover.comps] += ym @ matrix - ym*matrix.sum(axis=1)
            for cv in self.conveyors:
                for i,(a,b) in enumerate(zip(cv.slots[:-1], cv.slots[1:])):
                    moved = y[a]*(2 if i == 0 else 1)/self.dt
                    dy[a] -= moved
                    dy[b] += moved
        return dy

    def next_generation(self, start=False):
        """
        The next-generation matrix at the disease-free state, `F V⁻¹`, over the infected compartments of every group

        The infected compartments are those of the states people are infected into or
        infect from, of any count (such as a reservoir, or infectious mosquitoes) that the
        rate of an infection uses, and of the counts that infected people's numbers move
        a count into (mosquitoes biting them). A new infection is a flow from any other
        compartment into one of these. At the disease-free state, everyone is in the first
        state (so R0 is in a population with no immunity; with `start=True`, only the infected
        are moved there, for the reproduction number at the start), each group keeping its size, the infected counts are empty, and the other counts
        that are filled by an inflow (mosquitoes not infected) are at their equilibrium. The matrices are worked out by perturbing each infected
        compartment in turn. Returns None if nothing moves people into an infected state.
        See `ss.Disease.R0()`.
        """
        graph = self.graph
        states = {id(s) for s in graph.infected}
        found = None
        while found != states: # Until nothing more is found, since each count found can lead to another
            found = set(states)
            for flow in self.flows: # A count that infection depends on, such as a water reservoir, and the states that infection leads to
                if flow.src is not None and id(flow.state) not in states and flow.expr is not None:
                    single = [leaf.states[0] for leaf in flow.expr.leaves if isinstance(leaf, ss.transitions.Count) and len(leaf.states) == 1]
                    counts = {id(s) for s in single if not s.people}
                    by_infected = not flow.state.people and any(id(s) in states for s in single) # A count moved by the number infected, e.g. mosquitoes biting them
                    if counts or by_infected:
                        states |= counts | {id(dst) for dst in flow.tr.dst}
        infected = [i for i,comp in enumerate(self.comps) if id(comp.state) in states]
        self.update_targets()
        for flow in self.flows:
            flow.new = flow.src is not None and flow.src not in infected and any(ind in infected for ind,w,dim_map in flow.targets)
        if not any(flow.new for flow in self.flows): # Not an infection
            return None

        first = self.state_comps[graph.first.name][0]
        people = [i for i,comp in enumerate(self.comps) if comp.state.people and i != first and (not start or i in infected)] # At the start, only the infected; otherwise everyone, so immunity at the start is left out
        y_now = self.y # Working out the rates sets the compartments, so they are put back afterwards
        y0 = self.y.copy()
        y0[first] += y0[people].sum(axis=0) # Each group keeps its size
        y0[people] = 0
        y0[infected] = 0
        counts = [i for i,comp in enumerate(self.comps) if not comp.state.people and i not in infected]
        if counts: # Their equilibrium, from the rates at the disease-free state
            self.y = y0
            self.cache.clear()
            gain, loss = np.zeros_like(y0), np.zeros_like(y0)
            for flow,rate in zip(self.flows, self.rates()):
                if flow.src is not None:
                    loss[flow.src] += rate
                elif flow.kind == 'inflow':
                    for ind,w,dim_map in flow.targets:
                        amount = rate*self.inflow_split(flow.tr)*w
                        gain[ind] += amount if dim_map is None else amount @ dim_map
            for c in counts:
                settles = (gain[c] > 0) & (loss[c] > 0)
                y0[c] = np.where(settles, np.divide(gain[c], loss[c], out=np.zeros_like(gain[c]), where=settles), y0[c])
        cells = [(i, g) for i in infected for g in range(self.n_groups)]
        eps = 1e-6*max(1.0, y0.sum())
        F = np.zeros((len(cells), len(cells)))
        V = np.zeros((len(cells), len(cells)))
        for b,(i,g) in enumerate(cells): # Central differences, which are exact for rates that are linear or frequency-dependent in the infected
            up, down = y0.copy(), y0.copy()
            up[i, g] += eps
            down[i, g] -= eps
            new, other = [(self.changes(up, new) - self.changes(down, new))/(2*eps) for new in [True, False]]
            F[:, b] = [new[c] for c in cells]
            V[:, b] = [-other[c] for c in cells]
        self.y = y_now
        self.cache.clear()
        return F @ np.linalg.inv(V)

    #%% A whole run in the kernels

    def kernel_only(self):
        """ Whether the whole run can be taken in one call to the kernels: see `kernel_reason()` """
        return self.kernel_reason() is None

    def kernel_reason(self):
        """ Why the whole run cannot be taken in one call to the kernels (something in Python needs to see the model between one step and the next), in a few words, or None if it can """
        sim = self.sim
        mod = self.mod
        graph = self.graph
        if any(getattr(tr, 'per', None) is not None for tr in graph.inflows): # Added in Python after each step
            return f'"{next(tr.name for tr in graph.inflows if getattr(tr, "per", None) is not None)}" is a birth driven by a flow (ss.birth(per=)), added at each step'
        if self.syncs: # Recorded in the other module after each step
            return f'"{self.syncs[0].mod.name}.{self.syncs[0].otr.name}" is a synchronized flow of "{self.mod.name}.{self.syncs[0].tr.name}"'
        if self.triggers: # Applied in Python at each step, for those who come to meet them other than by a transition
            return f'"{self.triggers[0].name}" is triggered by {self.triggers[0].trigger}, which is applied at each step'
        if self.event_triggers: # Moved in Python at the end of each step
            return f'"{self.event_triggers[0].name}" is triggered by {self.event_triggers[0].trigger}, which is applied at each step'
        stepped = [m for mods in sim.modifiers.values() for m in mods if not getattr(m, 'tabulated', False)]
        if stepped:
            return f'the modifier "{stepped[0].name}" is applied each step (it has where=, when=, or a function as its amount, or changes a value that varies by person)'
        if self.net_mods: # The contacts are set once for a whole run
            return 'a modifier changes the contacts of a network'
        if any(key[:2] == (mod.name, 'route') for key,mods in sim.modifiers.items() if mods): # Likewise
            return f'a modifier acts on a transmission of "{mod.name}" on one route'
        if any(key[:2] == (mod.name, 'branch') for key,mods in sim.modifiers.items() if mods): # Where arrivals go is set once for a whole run
            return f'a modifier changes a branch probability of "{mod.name}"'
        read = [leaf.name for tr in graph.transitions.values() for p in sc.tolist(tr.p) if p is not None for leaf in p.leaves if isinstance(leaf, ss.par) and leaf.name and not leaf.literal]
        changed = [name for name in read if sim.modifiers.get((mod.name, 'par', name))]
        if changed: # Likewise, including ss.change(), which is otherwise tabulated
            return f'a modifier changes "{changed[0]}", which a branch probability of "{mod.name}" reads'
        if self.whole is None: # The rest cannot change, so is checked once
            self.whole = False # No reason
            own = [name for name in ['start_step', 'step_state', 'step', 'step_die', 'update_results', 'finish_step'] if getattr(type(mod), name) is not getattr(ss.Disease, name)]
            exprs = [p for tr in graph.transitions.values() for p in sc.tolist(tr.p) if p is not None] # The values that are read once for the whole run
            exprs += [w for info in self.outcomes.values() for key,w in info.sources if isinstance(w, ss.transitions.Expr)]
            funcs = [p for p in graph.all_pars() if callable(p.raw) and not isinstance(p.raw, (ss.Dist, ss.transitions.TimeValue))] # Called each step
            funcs += [leaf for expr in exprs for leaf in expr.leaves if not (isinstance(leaf, ss.par) and (isinstance(leaf.raw, ss.Dist) or not callable(leaf.raw)))]
            transfers = [name for name,tf in graph.transfers.items() if not self.constant_transfer(tf)]
            if self.program is None:
                self.init_program()
            if own:
                self.whole = f'"{mod.name}" defines its own {own[0]}()'
            elif graph.funcs:
                self.whole = f'"{graph.funcs.keys()[0]}" of "{mod.name}" is called each step'
            elif funcs:
                name = getattr(funcs[0], 'name', None) or getattr(getattr(funcs[0], 'raw', None), '__name__', 'a value')
                self.whole = f'"{name}" of "{mod.name}" is a function of the sim, so is called each step'
            elif transfers:
                tf = graph.transfers[transfers[0]]
                leaves = [leaf for expr in [tf.rate, tf.dur] + sc.tolist(tf.p) if expr is not None for leaf in expr.leaves]
                how = 'is changed by a modifier' if sim.modifiers.get((mod.name, 'transition', tf.name)) else 'differs between groups' if any(isinstance(getattr(leaf, 'raw', None), ss.by) for leaf in leaves) else 'changes in time'
                self.whole = f'the rate of the transfer "{transfers[0]}" of "{mod.name}" {how}'
            elif self.triggers or self.event_triggers:
                trig = (self.triggers + self.event_triggers)[0]
                self.whole = f'"{trig.name}" of "{mod.name}" is triggered by {trig.trigger}, so moves people between steps'
            elif mod.t.npts != sim.t.npts:
                self.whole = f'"{mod.name}" has a timestep of its own'
        return self.whole or None

    @property
    def transfer_index(self):
        """ The position of each transfer among the flows of a whole run (after the transitions, with the other movers) """
        return {mover.name:self.n_trans + i for i,mover in enumerate(self.movers) if mover.transfer is not None and mover.name is not None} # Not the moves between the stages of a transfer

    def constant_transfer(self, tf):
        """ Whether a transfer's rate holds for the whole run, so its mover can be set once before it (see `run_args()`): a rate or duration of plain values, with no modifiers """
        if self.sim.modifiers.get((self.mod.name, 'transition', tf.name)):
            return False
        exprs = [expr for expr in [tf.rate, tf.dur] + sc.tolist(tf.p) if expr is not None]
        return all(isinstance(leaf, ss.par) and not callable(leaf.raw) and not isinstance(leaf.raw, ss.by) for expr in exprs for leaf in expr.leaves)

    def tables(self, start=0, stop=None):
        """
        The parts of the program that can change over time, at each step (or as a single entry, if nothing varies)

        A value that varies in time is worked out for every step the first time it is
        needed, and kept for as long as the parameter holds the same object, so a sim
        that is run again does not work it out again.

        Args:
            start (int): the first step to give the tables for (for a run taken a part at a time: see `run_parts()`)
            stop (int): the step after the last (default: to the end)
        """
        sim = self.sim
        mod = self.mod
        p = self.program
        npts = mod.t.npts
        stop = npts if stop is None else stop
        self.cache.clear()
        per_step = self.per_step()
        if per_step == 3: # The year at the start, middle, and end of each step
            years = np.array(mod.t.yearvec[:npts], dtype=float)
            ends = np.append(years[1:], years[-1] + (years[-1] - years[-2] if npts > 1 else 0))
            times = np.ravel(np.column_stack([years, (years + ends)/2, ends]))
        for par in self.time_pars:
            raw = par.raw
            kept = self.series.get(id(par))
            inner = tuple(p.raw for p in (raw.pars if isinstance(raw, ss.transitions.TimeValue) else [])) # The values of any parameters inside it, which can change between runs
            if kept is None or kept[0] is not raw or kept[2] != per_step or kept[3] != inner:
                vals = []
                if per_step == 3:
                    years = times[np.arange(len(times)) // 3 * 3] if self.held(raw) else times # A held value is the one at the start of the step
                    vals = raw.series(years, sim, self.convert) # Every year at once (for an ss.by, years by groups)
                else:
                    whens = []
                    for ti in range(npts): # The values for the step from ti, which is step ti+1 (see ss.Loop)
                        sim.t.ti = mod.t.ti = min(ti + 1, npts - 1)
                        if isinstance(raw, ss.by): # Not by_group(), which holds a value for the step
                            whens.append(sim.t.now('year'))
                        else:
                            vals.append(self.convert(raw(sim)))
                    if isinstance(raw, ss.by):
                        vals = raw.series(whens, sim, self.convert)
                kept = self.series[id(par)] = (raw, np.array(vals, dtype=float).reshape(len(vals), -1), per_step, inner)
            self.cache[id(par)] = kept[1] # Steps by groups, where a value is otherwise one number or a number for each group
        sim.t.ti = mod.t.ti = 0
        rows = npts*per_step if self.time_pars else 1

        grouped = [] # Whether any value of each table differs by group (otherwise ss.gradients.capture() keeps it once for all of them)

        def table(vals):
            """ The values by group, at each step if any of them varies in time (a series over the steps), and otherwise once """
            steps = any(np.ndim(val) == 2 and len(val) > 1 for val in vals)
            grouped.append(any(np.shape(val)[-1] > 1 for val in vals if np.ndim(val))) # A series is steps by groups
            vals = [val[start*per_step:stop*per_step] if np.ndim(val) == 2 and len(val) > 1 else val for val in vals] # The steps asked for
            shape = ((stop - start)*per_step if steps else 1, self.n_groups)
            return np.array([np.broadcast_to(val, shape) for val in vals], dtype=float).reshape(len(vals), *shape).swapaxes(0, 1)

        fixed     = table([0.0 if flow.dynamic else self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows])
        mult      = table([self.flow_mult(flow) if flow.dynamic else 1.0 for flow in self.flows])
        consts    = table([self.prog_value(par) for par in self.prog_pars])
        rel_trans = table([state.rel_trans*self.mult_raw('rel_trans', state=state)*mask for state,mask in self.prog_exposures])
        series = {id(par):self.cache[id(par)] for par in self.time_pars}
        if any(key[0] == mod.name and not getattr(m, 'constant', False) for key,mods in sim.modifiers.items() for m in mods): # Modifiers that depend only on time (see kernel_only()): work out every part of the program at each step (and stage), with them applied
            rows = []
            for ti in range(start, stop):
                for frac in ([0, 0.5, 1] if per_step == 3 else [None]):
                    sim.t.ti = mod.t.ti = min(ti + 1, npts - 1) # The step from ti, which is step ti+1 (see ss.Loop)
                    self.at_time(None if frac is None else frac*self.dt)
                    self.cache.clear()
                    rows.append([[0.0 if flow.dynamic else self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows],
                                 [self.flow_mult(flow) if flow.dynamic else 1.0 for flow in self.flows],
                                 [self.prog_value(par) for par in self.prog_pars],
                                 [state.rel_trans*self.mult_raw('rel_trans', state=state)*mask for state,mask in self.prog_exposures]])
            sim.t.ti = mod.t.ti = 0
            self.at_time(None)
            self.cache.clear()
            shape = (self.n_groups,)
            fixed, mult, consts, rel_trans = [np.array([[np.broadcast_to(val, shape) for val in row[k]] for row in rows], dtype=float).reshape(len(rows), len(rows[0][k]), self.n_groups) for k in range(4)]
            grouped = [True]*4
        self.tables_by_group = grouped
        for par in self.time_pars: # For the results, a value is a series over the steps (and over the groups, for an ss.by)
            vals = series[id(par)][::per_step]
            self.cache[id(par)] = vals[:, 0] if vals.shape[1] == 1 else vals
        return [np.ascontiguousarray(table) for table in [fixed, mult, consts, rel_trans]]

    def per_step(self):
        """ The rows of the tables for each step: 3 (the start, middle, and end) for a run with a value that varies smoothly in time (except by Euler's method, which holds it for the step), and otherwise 1 """
        sim = self.sim
        if (sim.solver != 'ode' or sim.integrator == 'rk4') and any(not self.held(par.raw) for par in self.time_pars):
            return 3
        return 1

    @staticmethod
    def held(raw):
        """ Whether a value that varies in time is a step function (so is held for each step, as `ss.change()` would) """
        items = raw.items if isinstance(raw, ss.by) else [raw]
        return all(not isinstance(v, ss.transitions.TimeValue) or (isinstance(v, ss.timeseries) and v.mode == 'constant') for v in items)

    def run_args(self, tables=True):
        """ What the kernels need for a whole run, apart from the seed (and the tables, unless `tables=False`) """
        sim = self.sim
        self.update_targets()
        self._mover_rates = self.mover_rates()
        for i,matrix in enumerate(self._mover_rates):
            self.program.mov_rate[self.program.mov_starts[1, i]:self.program.mov_starts[1, i+1]] = matrix.data
        solver = ss.kernels.solvers[sim.integrator if sim.solver == 'ode' else sim.solver]
        if self.register is None:
            self.init_register()
        self.init_shifts()
        return [self.y, self.program, self.dt, self.mod.t.npts, solver, self.substeps()], self.tables() if tables else None

    def kernels(self, runs=1):
        """ The kernels to take the run (or `runs` runs) with, compiled or as plain Python, chosen once for a run: see `ss.kernels.choose()` """
        return choose_kernels(self, runs, self.plain_seconds)

    def plain_seconds(self):
        """ About how long the run would take as plain Python, from the size of its program and its rates at the start (see `ss.kernels.plain_seconds()`) """
        p = self.program
        solver = ss.kernels.solvers[self.sim.integrator if self.sim.solver == 'ode' else self.sim.solver]
        n_sub = ss.kernels.substeps(np.concatenate(self.start_rates() + [p.mov_rate]), self.dt, self.substeps(), solver)
        return ss.kernels.plain_seconds(p, self.y, self.mod.t.npts, solver, n_sub)

    def start_rates(self):
        """ The rates out of compartments at the start that do not depend on them, which a stiff ODE takes many substeps for, each as an array """
        return [np.ravel(self.flow_base(flow)*self.flow_mult(flow)) for flow in self.flows if flow.src is not None and not flow.dynamic and flow.kind != 'conveyor']

    def init_shifts(self):
        """ The deliveries that the kernels take in a whole run (see `ss.Delivery.tabulated`), as arrays: see `ss.kernels.Shifts` """
        sim = self.sim
        nc, ng, npts = len(self.comps), self.n_groups, self.mod.t.npts
        self.deliveries = [mod for mod in sim.interventions() if isinstance(mod, ss.Delivery) and mod.dis is self.mod and mod.tabulated]
        K = len(self.deliveries)
        due = np.zeros((K, npts), dtype=np.bool_)
        sel, cov = np.zeros((K, nc, ng)), np.zeros((K, nc, ng))
        to, rest = np.zeros((K, ng), dtype=np.int64), np.zeros((K, ng), dtype=np.int64)
        land = np.zeros((K, nc, ng))
        for k,mod in enumerate(self.deliveries):
            due[k], sel_k, cov_k, to[k], rest[k], land[k] = mod.compartments.shifts()
            for s,name in enumerate(self.selector.states):
                for comp in self.state_comps[name]:
                    sel[k, comp], cov[k, comp] = sel_k[s], cov_k[s]
        self.shifts = ss.kernels.Shifts(due=due, sel=sel, cov=cov, to=to, rest=rest, land=land)
        return self.shifts

    def run_all(self, use_jax=False):
        """ Take every step of the run in one call to the kernels (or to JAX: see `ss.gradients`), and fill in the results from what comes back """
        mod = self.mod
        graph = self.graph
        res = mod.results
        tally = None
        if self.given is not None: # The run has been taken already, by ss.gradients
            out, self.given = self.given, None
        elif self.cross is not None: # Together with the other diseases in the crossing, by one program
            out = self.cross.whole_run(self, use_jax=use_jax)
        elif use_jax and self.sim.solver in ['binomial', 'tau']: # The chain binomial or tau-leaping in JAX: see ss.perflib._chain
            *out, tally = importlib.import_module('starsim.perflib._chain').run(self)
            for k,mod in enumerate(self.deliveries):
                mod.compartments.store_shifts(tally[:, k])
        elif use_jax:
            out = ss.gradients.run(self)
        elif self.rows is not None: # The results by group are kept every few steps, so the run is taken a part at a time, and only those are kept
            self.store_parts(self.run_parts())
            return
        else:
            args, tables = self.run_args()
            tally = np.zeros((self.mod.t.npts, len(self.shifts.to), 2, self.n_groups))
            out = self.kernels().run(self.make_rng(stream=1), *args, *tables, self.register, self.shifts, tally, self.per_step(), 0)
            for k,mod in enumerate(self.deliveries):
                mod.compartments.store_shifts(tally[:, k])
        self.y, stocks, flows, short = out
        flows = self.count_deaths(flows)
        if tally is None and getattr(self, 'deliveries', None): # The ODE taken in JAX: the deliveries' results, from the compartments at each time point they happened at
            self.store_replayed(stocks)
        if self.rows is not None:
            self.check_short(short)
            self.store_parts([(0, stocks, flows)])
            return
        self.check_short(short)
        self.sim.t.ti = mod.t.ti = mod.t.npts - 1 # At the last time point, as after a run step by step
        store = lambda key, vals: self.store_run(key, vals.sum(axis=1), vals) # The total, and the values by group
        n = sc.objdict({name:store(name, stocks[:, self.state_comps[name]].sum(axis=1)) for name in graph.states})
        new = sc.objdict({name:store(name, flows[:, ind]) for name,ind in self.tr_index.items()})
        for name,ind in self.transfer_index.items():
            store(name, flows[:, ind])
        grouped = self.outcomes and self.n_groups > 1 # Outcomes are then worked out by group, and summed: a weight can be one per group
        derived = self.derived(n, new, outcomes=not grouped)
        stratified = [name for name in self.ratio_names() if getattr(res[name], 'strata', None) is not None]
        if stratified: # Sums and ratios of states by group
            ctx = StockCtx(self, sc.objdict({name:stocks[:, self.state_comps[name]].sum(axis=1) for name in graph.states}), sc.objdict({name:flows[:, ind] for name,ind in self.tr_index.items()}))
            for name in stratified:
                if name in self.ratios:
                    res[name].strata[:], res[name].denom[:] = [ss.transitions.evaluate(part, ctx) for part in self.ratios[name]]
                else:
                    res[name].strata[:] = ss.transitions.evaluate(graph.exprs[name], ctx)
        if grouped: # Outcomes by group: the same, from the flows of each group
            grouped = self.derived_outcomes(sc.objdict({name:flows[:, ind].T for name,ind in self.tr_index.items()}))
            for key,vals in grouped.items():
                derived[key] = vals.T
        for key,vals in derived.items():
            if np.ndim(vals) == 2:
                store(key, vals)
            else:
                res[key][:] = vals
        self.cache.clear()
        return

    def ratio_names(self):
        """ The results given as expressions, and the standard prevalence if it is kept by group as a ratio (see `init_results()`) """
        return list(self.graph.exprs) + [name for name in getattr(self, 'ratios', {}) if name not in self.graph.exprs]

    def store_run(self, key, total, by_group=None):
        """ Store a result of a whole run: its total at each time point, and its values by group if it keeps them; returns the totals """
        res = self.mod.results[key]
        res[:] = total
        if getattr(res, 'strata', None) is not None:
            res.strata[:] = by_group
        return res.values

    def group_needs(self):
        """ The states and flows (by position) that are needed by group: those of the results that keep their values by group, and what they are worked out from """
        graph = self.graph
        res = self.mod.results
        has = lambda key: getattr(res.get(key), 'strata', None) is not None
        exprs = any(has(name) for name in graph.exprs) # A sum or ratio by group can use any state or flow
        sources = {key for name,info in self.outcomes.items() if has(name) or has(f'n_{name}') for key,_ in info.sources} # The flows that outcomes kept by group come from
        kept_s = [i for i,name in enumerate(graph.states) if exprs or has(name)]
        kept_f = [ind for name,ind in (self.tr_index | self.transfer_index).items() if exprs or has(name) or name in sources]
        return kept_s, kept_f

    def run_parts(self, part_bytes=50e6):
        """
        Take a whole run a part at a time, in calls to the kernels that each take as many steps as fit in `part_bytes`

        Yields the time point each part starts at, and its compartments and flows at each time point (as `ss.kernels.run()`
        returns them, without the first, which the part before has, after the first part). The deliveries' results are
        stored as it goes.
        """
        args, _ = self.run_args(tables=False)
        y, program, dt, npts, solver, n_sub = args
        rng = self.make_rng(stream=1)
        every = int(self.group_ti[1]) # The steps in each row of the results by group
        per_point = (self.n_comps + program.n_trans + program.n_movers + 3*len(program.consts))*self.n_groups*8 # The compartments and flows, and the tables of values that vary in time
        length = max(int(part_bytes//per_point)//every, 1)*every # Steps per part: whole rows, so each part ends at the end of one
        short = 0.0
        start = 0
        kn = self.kernels()
        while start < npts - 1:
            end = min(start + length, npts - 1)
            tally = np.zeros((end + 1 - start, len(self.shifts.to), 2, self.n_groups))
            tables = self.tables(start, end) # Only for these steps, since by group at every step they can be large
            y, stocks, flows, more = kn.run(rng, y, program, dt, end + 1, solver, n_sub, *tables, self.register, self.shifts, tally, self.per_step(), start)
            short += more
            last = end == npts - 1
            for k,mod in enumerate(self.deliveries):
                mod.compartments.store_shifts(tally[:, k] if start == 0 else tally[1:, k], start=0 if start == 0 else start + 1, done=last) # The first row is the end of the part before
            yield (start, stocks, flows) if start == 0 else (start + 1, stocks[1:], flows[1:])
            start = end
        self.y = y
        self.check_short(short)
        return

    def store_parts(self, parts):
        """
        The results of a whole run whose results by group are kept every few steps (see `ss.solvers.group_rows()`), from its compartments and flows at each time point, taken a part at a time

        Args:
            parts (iterable): the time point each part starts at, and the compartments and flows at each of its time points (see `run_parts()`)
        """
        mod = self.mod
        graph = self.graph
        res = mod.results
        npts, ng = mod.t.npts, self.n_groups
        rows, ends = self.rows, self.group_ti
        states = list(graph.states)
        n_flows = self.n_trans + len(self.movers)
        kept_s, kept_f = self.group_needs()
        totals = sc.objdict(stocks=np.zeros((len(states), npts)), flows=np.zeros((n_flows, npts)))
        groups = sc.objdict(stocks=np.zeros((len(ends), len(kept_s), ng)), flows=np.zeros((len(ends), len(kept_f), ng))) # Only those needed
        pos_s, pos_f = {i:j for j,i in enumerate(kept_s)}, {i:j for j,i in enumerate(kept_f)}
        grouped = self.outcomes and ng > 1 # Outcomes are then worked out by group, from the flows they come from by group at each step
        sources = sorted({key for info in self.outcomes.values() for key,weight in info.sources if key in self.tr_index}) if grouped else []
        whole = grouped and not any(isinstance(weight, ss.transitions.Expr) for info in self.outcomes.values() for key,weight in info.sources) # For the whole run at once (see derived_outcomes()), unless a weight is an expression, which can change in time: then a step at a time, as a stepped run does
        src = sc.objdict({key:np.zeros((npts, ng)) for key in sources}) if whole else None # The flows the outcomes come from, by step and group
        if grouped:
            for name in self.outcomes:
                for key in [name, f'n_{name}']:
                    if getattr(res.get(key), 'strata', None) is not None:
                        res[key].strata[:] = 0
        for start,stocks,flows in parts:
            flows = self.count_deaths(flows)
            tis = start + np.arange(len(stocks))
            by_state = np.stack([stocks[:, self.state_comps[name]].sum(axis=1) for name in states], axis=1) # Time points by states by groups
            totals.stocks[:, tis] = by_state.sum(axis=-1).T
            totals.flows[:, tis] = flows.sum(axis=-1).T
            at_ends(groups.stocks, rows, ends, tis, by_state[:, kept_s])
            add_rows(groups.flows, rows, tis, flows[:, kept_f])
            if whole:
                for key in sources:
                    src[key][tis] = flows[:, self.tr_index[key]]
            elif grouped:
                for j,ti in enumerate(tis):
                    mod.t.ti = ti
                    self.cache.clear() # Values held for a step (see Engine.call() and by_group()) are for another step
                    self.flow_groups = sc.objdict({key:flows[j, self.tr_index[key]] for key in sources})
                    self.step_outcomes()
        if whole:
            tis = np.arange(npts)
            for key,vals in self.derived_outcomes(sc.objdict({key:vals.T for key,vals in src.items()})).items():
                vals = np.ascontiguousarray(vals.T) # Steps by groups
                res[key][:] = vals.sum(axis=1)
                if getattr(res[key], 'strata', None) is not None:
                    if key in self.outcomes: # A flow, added up over the steps of each row
                        add_rows(res[key].strata, rows, tis, vals)
                    else: # The number at the end of each row
                        at_ends(res[key].strata, rows, ends, tis, vals)
        self.flow_groups = sc.objdict()
        self.sim.t.ti = mod.t.ti = npts - 1
        store = self.store_run
        stock_g = lambda i: groups.stocks[:, pos_s[i]] if i in pos_s else None # By group, if needed
        flow_g = lambda ind: groups.flows[:, pos_f[ind]] if ind in pos_f else None
        n = sc.objdict({name:store(name, totals.stocks[i], stock_g(i)) for i,name in enumerate(states)})
        new = sc.objdict({name:store(name, totals.flows[ind], flow_g(ind)) for name,ind in self.tr_index.items()})
        for name,ind in self.transfer_index.items():
            store(name, totals.flows[ind], flow_g(ind))
        derived = self.derived(n, new, outcomes=not grouped)
        stratified = [name for name in self.ratio_names() if getattr(res[name], 'strata', None) is not None]
        if stratified: # Sums and ratios of states by group: from the states at the end of each row, and the flows over it
            ctx = StockCtx(self, sc.objdict({name:stock_g(i) for i,name in enumerate(states)}), sc.objdict({name:flow_g(ind) for name,ind in self.tr_index.items()}))
            for name in stratified:
                if name in self.ratios:
                    res[name].strata[:], res[name].denom[:] = [ss.transitions.evaluate(part, ctx) for part in self.ratios[name]]
                else:
                    res[name].strata[:] = ss.transitions.evaluate(graph.exprs[name], ctx)
        for key,vals in derived.items():
            res[key][:] = vals
        self.cache.clear()
        return

    def store_replayed(self, stocks):
        """ The results of the deliveries of a run taken in JAX, worked out again from the compartments at each time point they happened at (exact, since the run is an ODE) """
        sh = self.shifts
        tally = np.zeros((self.mod.t.npts, len(sh.to), 2, self.n_groups))
        for ti in np.flatnonzero(sh.due.any(axis=0)):
            if ti < len(stocks) - 1: # No step follows the last time point
                self.kernels().apply_shifts(self.rng, np.array(stocks[ti]), sh, ti, False, tally[ti+1]) # Expected numbers, so nothing is drawn; at the time point of the step from ti, its end
        for k,mod in enumerate(self.deliveries):
            mod.compartments.store_shifts(tally[:, k])
        return

    def run_many(self, seeds):
        """
        Take many runs that differ only in their seed, in calls to the kernels that take all of them, a part of the run at a time

        Args:
            seeds (list): the seed of each run, as would be given to the sim as `rand_seed`

        Returns each result that is a single series, as runs by steps, and (as `out.groups`) the
        values by group of the results that keep them, as runs by rows by groups: a row for each
        time point, or for every few steps if that would take more than `max_bytes` of memory (with
        the time point of each row as `out.group_ti`; see `ss.solvers.group_rows()`).
        """
        graph = self.graph
        res = self.mod.results
        self.restart()
        args, _ = self.run_args(tables=False)
        y0, npts = args[0], args[3]
        rngs = nb.typed.List([self.make_rng(stream=1, seed=seed) for seed in seeds]) # The random numbers that each run would have
        ng = self.n_groups
        states = list(graph.states)
        state_of = np.array([states.index(comp.state.name) for comp in self.comps])
        n_flows = self.program.n_trans + self.program.n_movers

        kept_s, kept_f = self.group_needs() # The states and flows to keep by group
        rows = group_ti = None
        if ng > 1 and (kept_s or kept_f or self.deliveries):
            row_bytes = len(seeds)*(len(kept_s) + len(kept_f) + 2*len(self.deliveries))*ng*8
            rows, group_ti = ss.solvers.group_rows(self.sim, npts, row_bytes, self.max_bytes, name=self.mod.name)
            rows = np.arange(npts) if rows is None else rows
        grp_state, grp_flow = np.full(len(states), -1), np.full(n_flows, -1)
        grp_state[kept_s] = np.arange(len(kept_s))
        grp_flow[kept_f] = np.arange(len(kept_f))
        n_rows = 0 if rows is None else rows[-1] + 1
        n_runs, nd = len(seeds), len(self.shifts.to)
        stocks, flows = np.zeros((n_runs, npts, len(states), 1)), np.zeros((n_runs, npts, n_flows, 1))
        stocks_g, flows_g = np.zeros((n_runs, n_rows, len(kept_s), ng)), np.zeros((n_runs, n_rows, len(kept_f), ng))
        tally, tally_g = np.zeros((n_runs, npts, nd, 2, 1)), np.zeros((n_runs, n_rows, nd, 2, ng))
        ys = np.repeat(y0[None], n_runs, axis=0) # The compartments of each run
        if self.sim.solver in ['binomial', 'tau'] and ss.solvers.choose_engine(self.sim)[0]: # In JAX, as a single run would be (see ss.solvers.choose_engine())
            stocks, flows, stocks_g, flows_g, tally, tally_g = importlib.import_module('starsim.perflib._chain').run_many(self, seeds, state_of, kept_s, kept_f, rows)
        else:
            run_many_parts(self, rngs, ys, args, self.per_step(), ng, (stocks, flows, stocks_g, flows_g, tally, tally_g), state_of, rows, grp_state, grp_flow)
        out = self.collect(stocks, flows, grouped=None if rows is None else (stocks_g, flows_g, kept_s, kept_f, group_ti), seeds=seeds)
        out.deliveries = sc.objdict({mod.name:mod.compartments.many_shifts(tally[:, :, k], None if rows is None else tally_g[:, :, k], group_ti) for k,mod in enumerate(self.deliveries)})
        return out

    def collect(self, stocks, flows, expected=False, grouped=None, seeds=None):
        """
        The results of many runs, from the number in each state and the flow along each transition and mover

        Args:
            stocks (array): runs by steps by states by groups (or by one group, the total)
            flows (array): runs by steps by transitions and movers by groups (or one)
            expected (bool): whether outcomes of whole events are their expected number, rather than drawn (as a particle filter needs)
            grouped (tuple): from `run_many()`: some states and flows by group instead, as runs by rows by states (or flows) by groups, which states and flows they are, and the time point of each row (or None for every one)
            seeds (list): the seed of each run, for the draws of the outcomes (see `derived_outcomes()`)

        Returns the results as `run_many()` describes them.
        """
        graph = self.graph
        res = self.mod.results
        states = list(graph.states)
        by_group = grouped is None and stocks.shape[-1] == self.n_groups and self.sim.groups.strats != [] # Whether the values of each group were kept
        groups = sc.objdict()
        denoms = sc.objdict() # The denominators by group of the ratios kept by group
        group_ti = None
        if by_group or grouped is not None:
            if by_group:
                stocks_g, flows_g, kept_s, kept_f = stocks, flows, range(len(states)), range(flows.shape[2])
            else:
                stocks_g, flows_g, kept_s, kept_f, group_ti = grouped
            pos = {ind:j for j,ind in enumerate(kept_f)}
            for j,i in enumerate(kept_s):
                groups[states[i]] = stocks_g[:, :, j, :]
            for name,ind in (self.tr_index | self.transfer_index).items():
                if ind in pos:
                    groups[name] = flows_g[:, :, pos[ind], :]
            stratified = [name for name in self.ratio_names() if getattr(res.get(name), 'strata', None) is not None]
            if stratified and all(name in groups for name in states): # Sums and ratios of states by group, as in run_all()
                ctx = StockCtx(self, sc.objdict({name:groups[name] for name in states}), sc.objdict({name:groups[name] for name in self.tr_index if name in groups}))
                for name in stratified:
                    if name in self.ratios:
                        groups[name], denoms[name] = [ss.transitions.evaluate(part, ctx) for part in self.ratios[name]]
                    else:
                        groups[name] = ss.transitions.evaluate(graph.exprs[name], ctx)
        outcomes = None # With the flows by group at every step: the outcomes by group, and their totals as the sums of them
        if (by_group or grouped is not None) and group_ti is None and self.outcomes:
            new_g = {name:np.moveaxis(flows_g[:, :, pos[ind], :], -1, 1) for name,ind in self.tr_index.items() if ind in pos} # Runs by groups by steps
            if all(key in new_g or key in self.outcomes for info in self.outcomes.values() for key,_ in info.sources):
                outs_g = self.derived_outcomes(new_g, expected=expected, seeds=seeds)
                for key,vals in outs_g.items():
                    groups[key] = np.moveaxis(vals, 1, -1)
                outcomes = sc.objdict({key:vals.sum(axis=1) for key,vals in outs_g.items()})
        if groups:
            for key in list(groups.keys()):
                if getattr(res.get(key), 'strata', None) is None:
                    del groups[key]
        stocks, flows = stocks.sum(axis=-1), flows.sum(axis=-1)
        n = sc.objdict({name:stocks[:, :, i] for i,name in enumerate(states)})
        new = sc.objdict({name:flows[:, :, ind] for name,ind in self.tr_index.items()})
        out = sc.objdict({name:vals for name,vals in n.items()})
        out.update(new)
        out.update({name:flows[:, :, ind] for name,ind in self.transfer_index.items()})
        out.update(self.derived(n, new, outcomes=outcomes is None, expected=expected, seeds=seeds))
        if outcomes is not None:
            out.update(outcomes)
        out.groups = groups
        out.denoms = denoms
        out.group_ti = group_ti
        self.cache.clear()
        return out

    def derived(self, n, new, outcomes=True, expected=False, seeds=None):
        """
        The results that are worked out from the number in each state and the flow along each transition

        Args:
            n (dict): the number in each state at each step: a series, or an array whose last dimension is the steps
            new (dict): the same for the flow along each transition
            outcomes (bool): whether to work out the outcomes
            expected (bool): see `derived_outcomes()`
            seeds (list): see `derived_outcomes()`

        Returns a dict of the outcomes and the results given as expressions.
        This is what `update_results()` does on each step, for every step at once.
        """
        graph = self.graph
        out = self.derived_outcomes(new, expected=expected, seeds=seeds) if outcomes else sc.objdict()
        if graph.exprs:
            ctx = StockCtx(self, n, new)
            for name,expr in graph.exprs.items():
                out[name] = ss.transitions.evaluate(expr, ctx)
        return out

    def derived_outcomes(self, new, expected=False, names=None, seeds=None):
        """ What `step_outcomes()` does on each step, for every step at once (with `expected`, outcomes of whole events are their expected number, rather than drawn); only the outcomes in `names`, if given (with those they are worked out from); for many runs (the first dimension of the flows), each drawn from the random numbers of its seed in `seeds` """
        npts = self.mod.t.npts
        draw = self.integer and not expected
        if draw and seeds is not None: # Each run drawn as the run on its own draws, rather than all from one generator
            saved, runs = self.rng, []
            for r,seed in enumerate(seeds):
                self.rng = self.make_rng(seed=seed)
                runs.append(self.derived_outcomes({k:v[r:r+1] for k,v in new.items()}, names=names))
            self.rng = saved
            return sc.objdict({k:np.concatenate([run[k] for run in runs]) for k in runs[0]})
        out = sc.objdict()
        todo = sc.objdict({name:info for name,info in self.outcomes.items() if names is None or name in names})
        if draw and todo: # A step at a time, each outcome in turn, as step_outcomes() draws them, so a run drawn whole is the run taken a step at a time
            return self.drawn_outcomes(todo, new)
        for name,info in todo.items():
            amount = 0.0
            for key,weight in info.sources:
                flow = new[key] if key in new else out[key]
                if isinstance(weight, ss.transitions.Expr):
                    weight = self.eval(weight)
                    if np.ndim(weight) == 1 and np.ndim(flow) >= 2 and np.shape(flow)[-2] == len(weight): # A weight for each group, with the flows by group
                        weight = weight[:, None]
                amount = amount + flow*weight
            kernel = info.kernel
            now = amount*kernel[0] if len(kernel) == 1 else np.apply_along_axis(np.convolve, -1, amount, kernel)[..., :npts]
            if info.stay is not None: # Each arrival leaves after its stay
                leaving = np.apply_along_axis(np.convolve, -1, now, info.stay)[..., :npts]
                out[f'n_{name}'] = np.cumsum(now - leaving, axis=-1)
            elif info.p_leave is not None:
                current = np.zeros(now.shape[:-1])
                out[f'n_{name}'] = np.zeros(now.shape)
                for ti in range(now.shape[-1]): # Every step, or every one of the steps given (a particle filter gives those a data point depends on)
                    current = current + now[..., ti] - current*info.p_leave
                    out[f'n_{name}'][..., ti] = current
            out[name] = now
        return out

    def drawn_outcomes(self, todo, new):
        """ The outcomes of `derived_outcomes()` when they are drawn: a step at a time, by `Engine.report_outcome()`, with the runs and groups (all but the last dimension of the flows) side by side """
        shape = np.shape(next(iter(new.values())))
        n_steps = shape[-1]
        flat = lambda vals: np.broadcast_to(vals, shape).reshape(-1, n_steps) # Runs and groups side by side, by steps
        flows = {key:flat(np.asarray(vals, dtype=float)) for key,vals in new.items()}
        width = int(np.prod(shape[:-1]))
        out = sc.objdict()
        bufs = sc.objdict()
        weights = sc.objdict()
        for name,info in todo.items():
            bufs[name] = sc.objdict(pending=np.zeros((len(info.kernel) + 1, width)), leaving=None if info.stay is None else np.zeros((len(info.stay) + 1, width)), n=np.zeros(width))
            weights[name] = []
            for key,weight in info.sources:
                weight = self.eval(weight) if isinstance(weight, ss.transitions.Expr) else weight
                if np.ndim(weight) == 1 and len(shape) >= 2 and shape[-2] == len(weight): # A weight for each group, with the flows by group
                    weight = flat(weight[:, None])[:, 0]
                weights[name].append(weight)
            out[name] = np.zeros((width, n_steps))
            if info.stay is not None or info.p_leave is not None:
                out[f'n_{name}'] = np.zeros((width, n_steps))
        for ti in range(n_steps):
            for name,info in todo.items():
                amount = 0.0
                for (key,_),weight in zip(info.sources, weights[name]):
                    amount = amount + self.outcome_share(flows[key][:, ti] if key in flows else out[key][:, ti], weight, True)
                out[name][:, ti] = self.report_outcome(info, bufs[name], np.asarray(amount, dtype=float), ti, True)
                if f'n_{name}' in out:
                    out[f'n_{name}'][:, ti] = bufs[name].n
        return sc.objdict({key:vals.reshape(shape) for key,vals in out.items()})

    #%% Stepping

    def substeps(self):
        """
        The number of substeps to take in each timestep (0 to let the solver choose)

        The chain-binomial solver takes enough substeps that no more than about a tenth
        of the people in a compartment leave it in one, as far as the rates are known at
        the start (at most `max_binomial`): each substep is off the ODE by an amount of
        order rate × dt (ST55: an acute stage of 1.7 months at monthly steps took everyone
        infected from 9% to 51%). It moves people through as many stages of a dwell time as
        they pass in a substep (see `ss.kernels.draw_chain()`), so it is the dwell's mean
        that counts, not its stages; a fixed delay is a conveyor of whole steps. `substeps=1`
        takes whole steps, as most chain-binomial models (and frameworks' discrete-time
        models) do.
        """
        n_sub = self.sim.pars.substeps
        if n_sub is None and self.sim.solver == 'binomial':
            if self._n_sub is None:
                self._n_sub = self.binomial_substeps()
            n_sub = self._n_sub
        return n_sub or 0

    def binomial_substeps(self):
        """ The substeps the chain-binomial solver takes in each step, from the fastest total rate out of a compartment of people at the start: see `substeps()` """
        totals = {} # The rates out of each compartment, summed: e.g. recovery and death from I
        for flow in self.flows:
            if flow.src is not None and not flow.dynamic and flow.kind != 'conveyor' and flow.state.people:
                rate = np.ravel(self.flow_base(flow)*self.flow_mult(flow))/(flow.stages if flow.kind == 'stage' else 1) # A dwell in stages, by its mean
                rate = np.where(np.isfinite(rate), rate, 0.0)
                totals[flow.src] = totals.get(flow.src, 0.0) + rate
        fastest = max((np.max(rate, initial=0.0) for rate in totals.values()), default=0.0)
        n_sub = max(int(np.ceil(fastest*self.dt/0.1 - 1e-9)), 1)
        if n_sub > self.max_binomial:
            n_sub = self.max_binomial
            unit = self.sim.unit_name
            ss.note(f'"{self.mod.name}" has a rate of {fastest:.3g} per {unit} out of a state, so the chain-binomial solver would take {int(np.ceil(fastest*self.dt/0.1))} substeps a step to follow it; it takes {n_sub}, with rate × dt = {fastest*self.dt/n_sub:.2g} in each. A smaller dt, substeps=, or solver="tau" follows it more closely', ss.errors.LargeStep, sim=self.sim)
        return n_sub

    def step_state(self):
        return

    def step(self):
        """ Advance the compartments by one timestep """
        if self.triggers: # Those who came to meet a trigger since the last step
            self.cache.clear()
            self.update_targets()
            self.sweep_triggers()
        if self.cross is not None: # Together with the other diseases in the crossing, once each has worked out its rates
            return self.cross.step(self)
        self.step_rates()
        sim = self.sim
        dt = self.dt
        solver = sim.integrator if sim.solver == 'ode' else sim.solver
        tables, row = self.step_tables(dt, smooth=solver != 'euler') # Euler holds values that vary in time for the step
        self.y, flows = step_program(self, self.kernels(), self.kernel_rng, self.y, tables, row, self, self.check_short)
        self.record_flows(flows)
        return

    def step_rates(self):
        """ The rates of the step, worked out again since the interventions have run since it started, and the program, the first time """
        self.cache.clear()
        self.update_targets()
        self._mover_rates = self.mover_rates()
        if self.program is None:
            self.init_program()
        return

    def count_deaths(self, flows):
        """ For a split between states and ss.null (with no ss.stay): those who died are its flow less those who went to a state, so the two agree in each run; flows are by flow along the axis before the last (the groups) """
        splits = [tr for tr in self.graph.transitions.values() if tr.dies and tr.ends and not tr.stays]
        if splits:
            flows = np.array(flows) # A copy, which can be written to
            for tr in splits:
                flows[..., self.tr_index[f'{tr.name}.null'], :] = flows[..., self.tr_index[tr.name], :] - sum(flows[..., self.tr_index[f'{tr.name}.{dst.name}'], :] for dst in tr.dst)
        return flows

    def record_flows(self, flows):
        """ After a step: the flow along each transition and mover over it, by group, and the people at its end """
        flows = self.count_deaths(flows)
        for name,ind in self.tr_index.items():
            self.flowcounts[name] += flows[ind].sum() # Moves by an intervention are counted already
            flow = flows[ind] + self.forced.pop(name, 0) # As the transition's own, for on= and the results by group
            self.last_flows[name] = flow
            if self.n_groups > 1:
                self.flow_groups[name] = flow
        if self.triggers or self.event_triggers: # At the end of the step, after its flows: those its events happened to, and those a trigger now holds for who are still in its source (e.g. moved by a transfer), as agents move them in the step
            self.step_triggers()
            self.cache.clear() # The flows have moved people since the rates were worked out
            self.sweep_triggers()
            for name in list(self.forced):
                self.last_flows[name] = self.last_flows[name] + self.forced.pop(name)
                if self.n_groups > 1:
                    self.flow_groups[name] = self.last_flows[name]
        self.add_per_births()
        for i,mover in enumerate(self.movers):
            if mover.name is not None:
                self.flowcounts[mover.name] += flows[self.n_trans + i].sum()
                self.flow_groups[mover.name] = flows[self.n_trans + i]
        self.record_stocks()
        self.stepped = self.mod.ti
        if self.syncs and self.cross is None:
            self.pass_syncs(flows)
        return
