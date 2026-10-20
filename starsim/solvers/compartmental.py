"""
The compartmental engine: a declared model run as groups of people (`ode`, `binomial`, `gillespie`, and tau-leaping)
"""
import numpy as np
import numba as nb
import scipy.sparse as sps
import sciris as sc
import starsim as ss
from .engine import Bunch, CountState, Engine, StockCtx, add_rows, at_ends, by_level, no_tables

__all__ = ['Compartmental', 'synced_event']


def at_dwell(expr, value):
    """ An expression with the time in a state (`P.dwell`) replaced by a value: that of the people in one slot of a conveyor by time in state """
    E = ss.transitions
    if isinstance(expr, E.Dwell):
        return value
    elif isinstance(expr, E.BinOp):
        return E.BinOp(expr.op, at_dwell(expr.a, value), at_dwell(expr.b, value))
    elif isinstance(expr, E.Func):
        return E.Func(expr.name, *[at_dwell(arg, value) for arg in expr.args])
    return expr


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
    other = sim.get_declared(name)
    if getattr(obj, 'name', None) not in other.graph.transitions:
        return None
    return Bunch(mod=name, tr=obj.name, to=getattr(dst, 'name', dst))


class Compartmental(Engine):
    """
    Runs a graph as compartments: deterministically (`ode`) or stochastically
    (`binomial`, `gillespie`, or `tau`, the exact solver's fallback for long runs)

    The states are expanded into compartments (one per branch and sub-stage) and the
    transitions into a flat list of flows, each with a source compartment, a rate,
    and the compartments it feeds. Every solver steps the same list of flows.
    """
    max_bytes = 500e6 # The most memory that run_many() may use for the values of each group, over all its runs; above it, they are kept every few steps (see ss.Sim.group_rows())

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
        return

    def define(self):
        """ Replace the declared states on the module with their run-time form """
        self.init_dims()
        for state in self.graph.states.values():
            self.mod.setattribute(state.name, CountState(self, state))
        return

    def init_pre(self):
        for name,tr in self.graph.transitions.items():
            if getattr(tr, 'per', None) is not None:
                raise ss.CapabilityError(f'"{name}" of "{self.mod.name}" is a birth driven by the flow of "{tr.per.name}" (ss.birth(per=)), which runs as agents for now, so it cannot run with solver="{self.sim.solver}".\n'
                                         f'     Run with solver="abm", or leave the solver as "auto".')
            if getattr(tr, 'offspring', None) is not None:
                raise ss.CapabilityError(f'"{name}" of "{self.mod.name}" has offspring (offspring=), which are added for each person who goes through it, so it cannot run with solver="{self.sim.solver}".\n'
                                         f'     Run with solver="abm", or leave the solver as "auto".')
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
        home = sim.mixing.commuting
        if home is not None and graph.people_states and shared: # Commuters leaving home for each place, and coming back
            d = sim.mixing.space
            leave, back = home.mobility.commute_rates(home, sim.unit_base)
            h, now = coords[:, home.dim], coords[:, d]
            out, to = np.nonzero(leave[h]*(now == h)[:, None]) # Each group at home, and each place its people go to
            away = np.flatnonzero(now != h) # Each group away, whose people go home
            src, dst = np.concatenate([out, away]), np.concatenate([to, h[away]])
            rates = np.concatenate([leave[h[out], to], np.full(len(away), back)])
            matrix = self.mover_matrix(src, src + (dst - now[src])*int(mults[d]), rates)
            self.movers.append(sc.objdict(name=None, comps=self.people_comps, matrix=matrix, transfer=None))
        for move in (sim.groups.dim_moves if shared else []): # Movement that belongs to the sim, not to this module: e.g. the protection of a product waning
            d = sim.groups.dim_index(move.dim)
            src = np.flatnonzero(coords[:, d] == move.src)
            matrix = self.mover_matrix(src, src + (move.dst - move.src)*int(mults[d]), 1/self.convert(move.dur))
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            self.movers.append(sc.objdict(name=None, comps=comps, matrix=matrix, transfer=None))
        for mover in self.movers: # The people due to leave a conveyor leave it over the step (in the ODE), so are not moved between groups within it
            mover.comps = [c for c in mover.comps if c not in self.leaving]
        self.clocks = [] # Clocks, whose people move up a level (a step) at the start of each step: where each group's people go, and the compartments that have it
        for d,strat in enumerate(sim.groups.strats):
            if strat.kind == 'clock':
                if self.cross is not None:
                    raise ss.CapabilityError(f'the clock "{strat.name}" cannot be used where the states of several diseases are tracked jointly (solver="{sim.solver}"); run with solver="abm"')
                level = coords[:, d]
                to = sim.groups.folded(np.where(level < strat.n - 2, np.arange(self.n_groups) + int(mults[d]), np.arange(self.n_groups))) # The open-ended level and none stay
                comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
                self.clocks.append((to, comps))
        return

    def step_clocks(self):
        """ Move everyone on a clock up a step, for the step that follows (see `init_movers()`) """
        for to,comps in self.clocks:
            for c in comps:
                self.y[c] = np.bincount(to, weights=self.y[c], minlength=self.n_groups)
        if self.clocks:
            self.changed()
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

        def add_flow(tr, src, kind, named=True, to=None, stages=None, at=None):
            for strain in (self.strain_levels() if kind == 'contact' else [None]): # A transmission is one flow for each strain it can come from
                self.flows.append(Bunch(tr=tr, src=src, kind=kind, named=named and not tr.stays, to=to, stages=stages, at=at, index=self.tr_index[tr.name], strain=strain, record=None, routes=False,
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
            if any(state is s for s in self.immune_states): # Infected again, by another strain, as far as their last strain does not protect them
                for other in self.graph.transmissions:
                    add_flow(other, ind, 'contact')
            return

        def add_slots(state, slotted, exits, competing):
            """ A state whose exits read the time in it (P.dwell): a conveyor of whole steps (see ss.kernels.convey()), each slot with the rates and destinations of its people's time in the state, then one compartment for the rest """
            n = self.n_slots[state.name] = self.dwell_slots(state, slotted)
            slots = [add_comp(state, 0, None) for i in range(n + 1)]
            rest = add_comp(state, 0, 0)
            acting = slots if self.sim.solver == 'ode' else slots[:-1] # As for a fixed delay
            for i,ind in list(enumerate(acting)) + [(n + 1, rest)]:
                for tr in slotted:
                    if i or tr.dur is None: # Nobody leaves by a dwell time in the step they arrive in, as for agents (see slot_rate())
                        add_flow(tr, ind, 'hazard', at=i)
                add_exits(ind, state, exits, competing, {})
            add_flow(slotted[0], slots[-1], 'conveyor', named=False, to=rest)
            self.conveyors.append(Bunch(slots=slots, flow=self.flows[-1]))
            return slots[0]

        self.slotted = graph.dwell_trs # The transitions that read the time in their source state: see add_slots()
        self.n_slots = dict()
        self._quantiles = dict()
        self.conveyors = [] # The slots of each fixed delay and minimum stay, and the flow out of its last: see ss.kernels.convey()
        for name,state in graph.states.items():
            exits = graph.exits[name]
            tracks = []
            competing = list(exits.hazards) # Transitions that act on every compartment of the state
            staged = [tr for tr in exits.timed if self.dwells[tr.name].staged]
            competing += [tr for tr in exits.timed if not self.dwells[tr.name].staged]
            branches = exits.branches + sc.tolist(exits.remainder)
            slotted = [tr for tr in exits.timed + exits.hazards if tr.name in self.slotted]
            competing = [tr for tr in competing if tr.name not in self.slotted]
            staged = [tr for tr in staged if tr.name not in self.slotted]
            stays = self.stays(name, exits, competing, staged, branches) # The slots of a conveyor that each transition with after= or before= acts on
            if slotted:
                if branches or staged or stays:
                    other = 'branches with p=' if branches else 'a non-exponential dwell time' if staged else 'a minimum stay (after=) or a limit (before=)'
                    self.unsupported(slotted[0], f'an exit that reads {name}.dwell, with {other} out of "{name}",')
                self.tracks[name] = [Bunch(tr=None, p=None, immediate=False, first=add_slots(state, slotted, exits, competing))]
                continue
            if branches: # Split on entry: one track per branch
                if staged:
                    self.unsupported(staged[0], f'a non-exponential dwell time out of "{name}" alongside branches with p=')
                for tr in branches:
                    p = tr.p[0] if tr.p is not None and not tr.split else None # A split between several destinations takes what the branches leave over
                    tracks.append(Bunch(tr=tr, p=p, immediate=tr.rate is None and (tr.dur is None or tr.immediate))) # With neither (or dur=0), nobody waits in the branch
            else:
                apart = len(staged) > 1 # Several, each for other people (where=, e.g. by sex): a track for each, which each group enters by the one it has (see apart_shares())
                if apart and (stays or any(tr.where is None or self.dwells[tr.name].fixed for tr in staged)):
                    self.unsupported(staged[1], f'two non-exponential dwell times out of "{name}" (unless each is for other people, by where=)')
                tracks += [Bunch(tr=tr, p=None, immediate=False, apart=apart) for tr in (staged or [None])]
            if exits.held: # People who stay, with no exit of their own: the rest of the branches (p=None), or those who took ss.stay (p=0, entered only from the split)
                tracks.append(Bunch(tr=None, p=None if exits.stay else ss.transitions.par(0.0, literal=True), immediate=False, stay=True))
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
            flow.row = None
            if flow.at is not None and flow.expr is not None: # A rate that reads the time in the state, at the time of the people in this slot
                flow.expr = at_dwell(flow.expr, ss.par(self.slot_times(flow)[1], literal=True))
            if flow.strain and tr.const is not None: # A part of the rate with no source of infection has no strain, so is counted once, with the first
                flow.expr = flow.expr - tr.const
            flow.dynamic = flow.expr is not None and any(isinstance(leaf, (ss.State, ss.transitions.Count)) for leaf in flow.expr.leaves)

        self.slot_ps = dict() # For each transition with several destinations that reads the time in its state: its flows out of each slot, and its p= with the time of each slot's people
        for name in self.slotted:
            tr = graph.transitions[name]
            flows = [flow for flow in self.flows if flow.tr is tr and flow.at is not None]
            if len(tr.dst) > 1:
                for row,flow in enumerate(flows):
                    flow.row = row
                times = ss.par(np.array([self.slot_times(flow)[0] for flow in flows]).reshape(-1, 1), literal=True)
                self.slot_ps[name] = (len(flows), [None if p is None else at_dwell(p, times) for p in tr.p])

        self.n_comps = len(self.comps)
        self.n_trans = len(self.tr_index)
        self.people_comps = [i for i,comp in enumerate(self.comps) if comp.state.people]
        self.y = np.zeros((self.n_comps, self.n_groups))
        self.leaving = {cv.slots[-1] for cv in self.conveyors} # The last slot of each conveyor, whose people leave it over the next step
        self.by_src = sc.ddict(list) # The flows out of each compartment, for the stochastic solvers (which release the last slot of a conveyor by itself)
        for flow in self.flows:
            if flow.kind != 'conveyor':
                self.by_src[flow.src].append(flow)
        stage_flows = {flow.src:flow for flow in self.flows if flow.kind == 'stage' and flow.strain is None}
        self.chains = dict() # For the chain-binomial solver: for each compartment of a dwell time in stages, the number of stages left, its flow to the next, and the flow out of the last (see ss.kernels.draw_chain())
        for c,flow in stage_flows.items():
            d = flow.stages - self.comps[c].stage
            if d > 1 and self.comps[c].state.people:
                self.chains[c] = (d, flow, stage_flows[c + d - 1])
        return

    def unsupported(self, tr, what):
        raise ss.CapabilityError(f'"{tr.name}" cannot run with solver="{self.sim.solver}": {what} has no compartmental form yet.\n'
                                 f'     Run with solver="abm".')

    def whole_steps(self, tr, what, dur, lowest, nearest=False):
        """ A time as a number of whole steps, as agents count it (up, or to the nearest with `nearest=True`; at least `lowest`), with a note if it is not one """
        step = getattr(getattr(self.mod, 't', None), 'dt', None) or getattr(getattr(self.sim, 'pars', None), 'dt', None) # The sim's, if the module has no timeline yet (see init_crossings())
        if not isinstance(step, ss.dur): # Laid out before the module is in a sim, for its equations (see ss.transitions.Graph.flow_terms()): one slot
            return max(lowest, 1)
        x = float(self.convert(dur))/step.to(self.sim.unit_base).value
        n = max(lowest, int(np.ceil(x - (0.5 if nearest else 0) - 1e-9)))
        if abs(x - round(x)) > 1e-6:
            unit = self.sim.unit_name
            dt = float(self.convert(dur))/x
            ss.note(f'{tr.name} has {what} of {x*dt:n} {unit}s, which is {n} step{"s" if n != 1 else ""} ({n*dt:n} {unit}s), as agents take it; a step that divides it makes it exact', ss.LargeStep, sim=self.sim)
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

    def dwell_slots(self, state, slotted):
        """ How many whole steps the conveyor by time in a state follows people for (see `build_layout()`): until all but 0.1% have left by its longest dwell time, and no longer than the run """
        if not isinstance(getattr(getattr(self.mod, 't', None), 'dt', None), ss.dur): # Laid out before the module is in a sim, for its equations: one slot
            return 1
        dt = self.dt
        n = self.mod.t.npts
        timed = [tr for tr in slotted if tr.dur is not None]
        if timed and not any(tr.rate is not None for tr in slotted):
            ends = [np.max(self.dwell_mean(tr.name))*np.log(1000) if self.dwells[tr.name].dist is None else self.quantiles(tr)[int(0.999*len(self.quantiles(tr)))] for tr in timed]
            n = min(n, int(np.ceil(max(ends)/dt)))
        n = max(n, 1)
        total = (n + 2)*self.n_groups
        if total > 10_000:
            ss.note(f'{state.name} is followed by its time in the state, as {n + 2} compartments a group ({total:,} in all; every {dt:n} {self.sim.unit_name}s to {n*dt:n}), since {sc.strjoin(tr.name for tr in slotted)} read{"s" if len(slotted) == 1 else ""} {state.name}.dwell', ss.LargeStep, sim=self.sim)
        return n

    def quantiles(self, tr):
        """ The quantiles of a transition's dwell time, in the sim's unit, from which its distribution is worked out under a conveyor by time in state (see `slot_rate()`) """
        if tr.name not in self._quantiles:
            dist = sc.dcp(self.dwells[tr.name].dist)
            dist.init(force=True)
            self._quantiles[tr.name] = np.sort(np.asarray(dist.ppf((np.arange(20_000) + 0.5)/20_000), dtype=float))
        return self._quantiles[tr.name]

    def slot_times(self, flow):
        """ The time in the state of the people in one slot of a conveyor by time in state, as agents count it: for its branch probabilities, whole steps when they leave; for a rate, to the middle of the step """
        n = self.n_slots[flow.state.name]
        dt = self.dt if isinstance(getattr(getattr(self.mod, 't', None), 'dt', None), ss.dur) else 1.0 # Or laid out for its equations
        i = flow.at
        if i > n: # The rest: the mean time in the state of those still in it
            timed = [tr for tr in self.graph.exits[flow.state.name].timed if tr.name in self.slotted]
            leave = (n + 0.5)*dt + (self.time_left(timed[0], (n + 0.5)*dt, 1.0) if timed else 0)
            return leave, leave
        return i*dt, (i + 0.5)*dt

    def time_left(self, tr, T, m):
        """ The mean time left in a transition's dwell time, sped up by m, for people who have been there a time T (an array, by group, if m is) """
        dw = self.dwells[tr.name]
        if dw.dist is None:
            return np.asarray(self.dwell_mean(tr.name))/m
        q = self.quantiles(tr)
        x = np.asarray(T*m, dtype=float)
        i = np.searchsorted(q, x, side='right')
        sums = np.concatenate([np.cumsum(q[::-1])[::-1], [0]])
        count = len(q) - i
        return np.where(count > 0, (sums[i]/np.maximum(count, 1) - x)/m, self.dt) # Past every quantile: a step

    def slot_rate(self, flow):
        """
        The rate out of one slot of a conveyor by time in state by a dwell time: from the share of people who leave in the slot's step, given that they have not left before; worked out for every slot at once, once a step

        A person who will leave after a dwell d leaves from the slot whose step d is nearest to, and
        at least one step after arriving, as agents leave at the start of the step nearest the end of
        their dwell, and no sooner than the next step. The flow's
        multipliers (a modifier, adjust=) speed the distribution up, which is exact while they are
        constant; the rate is returned over them, since `rates()` applies them.
        """
        tr = flow.tr
        key = ('slots', tr.name, flow.state.name)
        if key not in self.cache:
            dt = self.dt
            mult = np.asarray(self.flow_mult(flow), dtype=float)
            m = np.maximum(mult, 1e-12)
            n = self.n_slots[flow.state.name]
            if self.dwells[tr.name].dist is None:
                F = lambda t: -np.expm1(-t/self.dwell_mean(tr.name))
            else:
                q = self.quantiles(tr)
                F = lambda t: np.interp(t, q, (np.arange(len(q)) + 0.5)/len(q), left=0, right=1)
            ends = np.where(np.arange(n + 1) > 0, np.arange(n + 1) + 0.5, 0)*dt # Each slot takes the dwells from the end of the one before to its own: none in the first, which people arrive in (see above)
            cdf = F(m*np.concatenate([[0], ends])[:, None])
            share = np.clip(np.divide(cdf[1:] - cdf[:-1], 1 - cdf[:-1], out=np.ones(cdf[1:].shape), where=cdf[:-1] < 1), 0, 1 - 1e-12)
            rest = 1/np.maximum(self.time_left(tr, (n + 0.5)*dt, m), 1e-12) # The rest: at the rate that gives the mean time left
            rates = np.concatenate([-np.log1p(-share)/dt, np.broadcast_to(rest, (1, np.size(cdf[0])))])
            self.cache[key] = np.where(mult > 0, rates/m, 0.0)
        return self.cache[key][flow.at]

    def init_post(self):
        """ Set the initial conditions """
        fracs, counts = self.init_fractions()
        if any(sp.get('share') is not None for sp in self.spreads): # Phase B: a module's own burn-in (ss.Pregnancy's burnin=) as compartments
            ss.note(f'{self.mod.name}: burnin= has no compartmental form yet, so nobody starts part way through (as agents they do)', sim=self.sim)
            self.spreads = [sp for sp in self.spreads if sp.get('share') is None]
        if self.spreads and self.cross is not None:
            raise ss.CapabilityError(f'ss.spread() in "{self.mod.name}", which is tracked jointly with {sc.strjoin(n for n in self.cross.names if n != self.mod.name)}, cannot run with solver="{self.sim.solver}" yet. Run with solver="abm".')
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
        if self.spreads: # People who start part way through their natural history
            self.backfill_spreads()
        if self.integer: # People are whole; a count that is not people (a reservoir) need not be
            self.y[self.people_comps] = self.round_preserving(self.y[self.people_comps])
        if any(not tr.static for tr in self.triggers) and self.cross is None: # Those a trigger on time or counts holds for move on before the start, not counted as flows
            self.sweep_triggers()
            for name in self.flowcounts:
                self.flowcounts[name] = 0.0
            self.forced = dict()
        if self.sim.mixing.register is not None and self.cross is None: # The events dated at the start (in a crossing, they happen to everyone once the diseases have their people: see ss.solvers.Crossing.init())
            self.check_short(ss.kernels.apply_events(self.make_rng(stream=2), self.y, self.init_register(), 0, self.integer))
        self.record_stocks() # The results at the first time point are the start
        self.n_start = sum(self.y[self.state_comps[s.name]].sum() for s in self.graph.infected if s.people) # The number infected at the start: see note_chance()
        self.mod.setattribute('dists', None) # Compartments draw from the engine's own generator, so the module's distributions are never sampled and need not be advanced each step
        return

    def backfill_spreads(self):
        """
        People who start part way through their natural history (`ss.spread`): the model with nobody to infect and no
        inflows, fed a constant inflow over the window (or until the people still there settle, with no `over=`), and
        stepped to the start in the kernels, so they spread over the stages and slots of its dwell times; scaled to the
        share given in the target, and taken from the first state. The values are those at the start (as are the ages)
        """
        mod = self.mod
        sim = self.sim
        saved = self.y.copy()
        first = self.graph.first.name
        out = np.zeros_like(self.y)
        self._mover_rates = [np.zeros_like(np.asarray(r, dtype=float)) for r in self.mover_rates()] # Nobody moves between places (or ages) before the start
        if self.program is None:
            self.init_program()
        p = self.program
        inflows = [i for i,flow in enumerate(self.flows) if flow.src is None]
        kept = (p.fixed.copy(), p.mult.copy())
        self.refresh_program()
        p.fixed[inflows] = 0 # No births or immigration before the start
        p.mult[inflows] = 0
        solver = ss.kernels.solvers[sim.integrator if sim.solver == 'ode' else 'rk4'] # The expected numbers, whatever the solver
        rng = self.make_rng(stream=3)
        try:
            for sp in self.spreads:
                window = sp.spread.window(self.convert)
                lo, hi = (0, 2000) if window is None else [int(round(v/self.dt)) for v in window]
                hi = max(hi, lo)
                if sp.tr is not None: # Where a new infection goes
                    entry = list(zip(sp.tr.dst, [np.asarray(x, dtype=float) for x in self.branch_probs(sp.tr)]))
                else:
                    entry = [(sp.entry, 1.0)]
                self.y = np.zeros_like(saved)
                targets = [c for name in sp.target for c in self.state_comps[name]]
                last = None
                for k in range(-hi - 1, 0):
                    if -k - 1 >= lo: # Entered in this step
                        for dst,share in entry:
                            self.arrive(dst.name, share*self.state_split[first], self.y)
                    self.y, _ = ss.kernels.one_step(rng, self.y, p, self.dt, solver, self.substeps(), no_tables, -1)
                    if window is None: # Until those still there settle
                        now = self.y[targets].sum(axis=0)
                        if last is not None and np.allclose(now, last, rtol=1e-7, atol=0):
                            break
                        last = now
                x = sp.spread.value
                want = (x if x <= 1 else x/self.total_pop)*self.total_pop*self.state_split[first]
                have = self.y[targets].sum(axis=0)
                scale = np.divide(want, have, out=np.zeros_like(have), where=have > 0)
                out[targets] += self.y[targets]*scale
                what = 'infected' if sp.tr is not None else f'in {sp.entry.name}'
                when = '[' + ', '.join(f'{v*self.dt:n}' for v in [lo, hi]) + ']' + (f' {sim.pars.unit}s' if sim.pars.get('unit') else ' steps') if window is not None else 'long enough to be stationary'
                ss.note(f'{mod.name}: {x:n} {what}, back-dated over {when} (natural history only; values and ages as at the start)', sim=sim)
        finally:
            p.fixed[:], p.mult[:] = kept
            self.cache.clear()
        firsts = self.state_comps[first] # Those back-dated are taken from the first state
        total = saved[firsts].sum(axis=0)
        taken = out.sum(axis=0)
        keep = np.divide(np.maximum(total - taken, 0), total, out=np.zeros_like(total), where=total > 0)
        saved[firsts] *= keep
        self.y = saved + out
        return

    def restart(self):
        """ Go back to the start of the run, with the current values of the parameters and the seed: see `ss.Sim.restart()` """
        self.rng = self.make_rng()
        self.kernel_rng = self.make_rng(stream=1)
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
        Put the movement between places, worked out again from new values of its parameters (see `ss.Mobility.fitted()`), into the program, as a restart with them needs (see `ss.Sim.restart()`)

        The program keeps its entries, which the new values fill (one that becomes 0 stays as an
        entry of 0), so a compiled run and the gradients' check that the layout is the same both hold.
        """
        mixing = self.sim.mixing
        if mixing.commuting is not None:
            errormsg = 'This sim cannot be restarted with new values of the movement between places, since with back= they set where people are at the start, which is laid out when the sim is built; use sim.new(pars=...) instead'
            raise ValueError(errormsg)
        def onto(new, rows, cols, what):
            """ The entries of a new sparse matrix at the old one's, if it has no others """
            vals = np.asarray(new[rows, cols], dtype=float).ravel() if len(rows) else np.zeros(0)
            if np.count_nonzero(vals) != np.count_nonzero(new.data):
                errormsg = f'This sim cannot be restarted with these values of the {what}, since they move people between places that the values it was built with do not; use sim.new(pars=...) instead'
                raise ValueError(errormsg)
            return vals
        p = self.program
        if p is not None and mixing.share is not None:
            p.sh_val[:] = onto(mixing.share, p.sh_pair[0], p.sh_pair[1], 'movement between places')
        for mover in self.movers:
            if mover.get('places') is not None:
                src, dst, step = mover.places
                coords = self.sim.groups.group_coords[:, mixing.space]
                new = self.mover_matrix(src, src + (dst - coords[src])*step, mixing.migration[coords[src], dst])
                old = mover.matrix
                mover.matrix = old.copy()
                mover.matrix.data = onto(new, np.repeat(np.arange(old.shape[0]), np.diff(old.indptr)), old.indices, 'rates of migration')
        return

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

    def force(self, frac, tr, dst, srcs=None):
        """ Move a fraction of the people in each state a transition leaves from (or in `srcs`), and each group, to a state, counted in the transition's flow if there is one (as when a treatment cures); a synchronized flow moves other modules' coordinates too """
        syn = self.syncs_of(tr, dst.name) if tr is not None else []
        total, recorded = self.move_states(frac, tr.src if srcs is None else srcs, dst, syn=syn)
        if tr is None:
            return total
        self.flowcounts[tr.name] += recorded.sum()
        self.forced[tr.name] = self.forced.get(tr.name, 0) + recorded
        if syn:
            self.pass_syncs({sy.row:recorded*sy.mask for sy in syn})
        return recorded

    def move_states(self, frac, srcs, dst, syn=None):
        """ Move a fraction of the people in each of some states, and each group, to a state: along a transition (see `force()`), or along none, as a one-off move (`ss.vaccinate(move='S -> V')`); with `syn` (synchronized flows), also returns the moves in this module's own groups """
        total = np.zeros(self.n_groups)
        recorded = np.zeros(self.n_groups)
        entry = self.entry[dst.name][0]
        for state in srcs:
            if not state.people:
                continue
            s = self.selector.states.index(state.name)
            base = self.dim_map(state.name, dst.name)
            dim_map = self.synced(base, syn) if syn else base
            for comp in self.state_comps[state.name]:
                n = self.take(self.y[comp], frac[s])
                self.y[comp] -= n
                total += n if dim_map is None else n @ dim_map
                recorded += n if base is None else n @ base # In this module's own groups, before the others' coordinates move
        if self.integer and len(entry) > 1: # Whole people, split at random between the branches of the destination
            left, rest = total.copy(), 1.0
            for ind,w in entry[:-1]:
                k = self.rng.binomial(left.astype(int), np.clip(w/rest, 0, 1))
                self.y[ind] += k
                left, rest = left - k, rest - w
            self.y[entry[-1][0]] += left
        else:
            self.arrive(dst.name, total, self.y)
        self.changed()
        return (total, recorded) if syn is not None else total

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
        if self._routes is not None: # A beta for each route (for each source state, under the partnership approximation)
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
                ss.note(f'{mod.name}.{otr.name} is triggered by {otr.trigger.x.label}: a synchronized flow, which moves people in {mod.name} as {self.mod.name}.{tr.name} moves them (exact)'
                        + (f'; with solver="{sim.solver}" its count is drawn apart from the move, so agrees with it on average' if self.integer and len(tr.dst) > 1 else ''), sim=sim)
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
        src = {l:(state.name, l) for state in sy.otr.src for l in strat.state_levels[state.name]}
        for g in range(ng):
            b = coords[g, d]
            if b not in src:
                S[g, g] = 1.0
                continue
            mask[g] = 1 - stay
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
        sy.S, sy.mask = groups.fold_matrix(S), mask
        sy.rec = tuple(np.array(v) for v in zip(*rec)) if rec else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
        return sy

    def synced(self, dim_map, syn):
        """ The map of a flow along a transition that other modules' transitions are triggered by: its own, then moving their coordinates """
        if not syn:
            return dim_map
        key = (id(dim_map),) + tuple(sy.row for sy in syn)
        if key not in self._sync_maps:
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
            if not tr.static:
                ss.note(f'{tr.name} is triggered by {tr.trigger}, which is not only on states and strata; applied at each step', sim=sim)
            else:
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
        key = [np.asarray(p, dtype=float) for shares in probs.values() for p in shares] + [np.asarray(p, dtype=float) for tr in self.graph.transitions.values() if tr.dst is not None and tr.split and tr.name not in self.slotted for p in self.branch_probs(tr)]
        slot_probs = {name:[np.broadcast_to(p, (n, self.n_groups)) for p in self.branch_probs(self.graph.transitions[name], ps)] for name,(n,ps) in self.slot_ps.items()} # For every slot at once
        key += [p for probs in slot_probs.values() for p in probs]
        last = getattr(self, '_target_key', None)
        if last is not None and len(last) == len(key) and all(np.array_equal(a, b) for a,b in zip(last, key)):
            return
        self._target_key = key
        self.entry = sc.objdict() # For each state: [(compartment, weight)] and [(transition index, weight)]

        def enter(name, weight, comps, credits, path=()):
            if name in path: # The same-step cascade, by construction: see ss.solvers.Agents.junctions
                cycle = ' -> '.join(path[path.index(name):] + (name,))
                raise ss.TriggerLoop(f'moves with no wait out of the states of "{self.mod.name}" form a loop: {cycle}.\n'
                                     f'     Give one of them a rate or a duration, or change the branches so they do not lead back.')
            for track,p in zip(self.tracks[name], probs[name]):
                if track.immediate: # Nobody waits in this branch: pass straight through to its destination
                    dps = self.branch_probs(track.tr)
                    credits.append((self.tr_index[track.tr.name], weight*p*(sum(dps) if track.tr.stays else 1)))
                    for dst,dp in zip(track.tr.dst, dps):
                        if len(track.tr.dst) > 1:
                            credits.append((self.tr_index[f'{track.tr.name}.{dst.name}'], weight*p*dp))
                        enter(dst.name, weight*p*dp, comps, credits, path + (name,))
                    if track.tr.stays: # Those who took ss.stay
                        comps.append((self.stay_comp(name), weight*p*(1 - sum(dps))))
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
                ps = self.branch_probs(flow.tr) if flow.row is None else [p[flow.row] for p in slot_probs[flow.tr.name]]
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
                    flow.recs += [self.tr_index[f'{flow.tr.name}.{dst.name}'] if flow.named and len(flow.tr.dst) > 1 else -1]*len(comps) # Also counted in the flow to this destination
                    flow.credits += [(ind, w*p) for ind,w in credits]
                    flow.credits += [(sy.row, p*sy.mask) for sy in syn] # Moved in another module too
                if flow.tr.stays and src is not None: # Those who took ss.stay go to the state's compartment with no exit of its own, and are not in the flow
                    flow.targets.append((self.stay_comp(src), 1 - sum(ps), None))
                    if flow.kind != 'stage' or flow.to is None:
                        flow.credits.append((flow.index, sum(ps)))
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

    def arrive(self, name, amount, y, before=False):
        """ Add an amount to a state, split between its tracks; with before=True, as if they arrived a step before (people infectious at the start, who have been so for the first step, as agents date them), so one slot along a conveyor """
        along = {conv.slots[0]:conv.slots[1] for conv in self.conveyors if len(conv.slots) > 1} if before else {}
        for ind,w in self.entry[name][0]:
            y[along.get(ind, ind)] += amount*w
        return

    def flow_base(self, flow):
        """ The rate of a flow before its multipliers: the only part that can depend on the number in each compartment """
        if flow.kind == 'conveyor': # The last slot of a conveyor has no rate: its people leave in the step after they reach it, as its inflow in the program (see ss.kernels.convey())
            return 1.0
        elif flow.expr is None and flow.at is not None:
            return self.slot_rate(flow)
        elif flow.expr is None:
            kg = self.dwells[flow.tr.name].kg if flow.kind == 'stage' else None # Ragged stages: each group's own number
            return ((flow.stages if kg is None else kg) if flow.kind == 'stage' else 1)/self.dwell_mean(flow.tr.name)
        self._strain = flow.strain
        self._routes = flow.routes if flow.routes is not False else self.route_betas(flow)
        self._rtr = (flow.tr, flow.state) if flow.kind == 'contact' and self.route_mods(flow.tr) else None
        self._src = flow.state # For a condition on states in the rate (see prop())
        rate = self.eval(flow.expr)
        self._strain = self._routes = self._src = self._rtr = None
        return rate

    def route_betas(self, flow):
        """
        For a transmission with a beta for each route (beta=dict(home=..., school=...)), or with a route that has acts: the beta
        of each term of the mixing, which exposure() applies (and the beta itself is then 1); None otherwise. On one-off contacts
        with acts (`ss.PartnerNet(between=...)`), it is the chance that a contact transmits, for the people in each group (see
        `contact_chance()`)
        """
        flow.routes = None
        if flow.kind != 'contact':
            return None
        nets = {ss.standardize_netkey(name):net for name,net in self.sim.all_networks().items()}
        terms = self.sim.mixing.terms
        lasting = any('partnership' in term for term in terms) # Under the partnership approximation, by source state
        acts = lasting or any(getattr(net, 'has_acts', False) for net in nets.values())
        betas = [leaf for leaf in flow.expr.leaves if isinstance(leaf, ss.par) and (isinstance(leaf.raw, (dict, ss.peract)) or (acts and leaf.slot == 'rate'))]
        if betas:
            raw = betas[0].raw
            vals = {ss.standardize_netkey(k):v for k,v in raw.items()} if isinstance(raw, dict) else None # The keys are checked when the sim is initialized
            def route(term, state=None):
                key = ss.standardize_netkey(term.name)
                val = vals[key] if vals is not None else raw
                net = nets.get(key)
                if 'partnership' in term:
                    return self.partnership_chance(val, net, term, flow, state)
                return self.contact_chance(val, net, term) if getattr(net, 'has_acts', False) else float(self.plain_value(val))
            flow.routes = {state.name:[route(term, state) for term in terms] for state in flow.tr.terms} if lasting else [route(term) for term in terms]
            self.routed.add(id(betas[0]))
        return flow.routes

    def partnership_chance(self, val, net, term, flow, state):
        """
        Under the partnership approximation (see `ss.solvers.Mixing.partnerships()`), the chance that a partnership with
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

    def contact_chance(self, val, net, term):
        """ The chance that a one-off contact with acts transmits, for the people in each group: 1 − E[exp(−K·h)], with K the acts in a contact and h the hazard per act in the direction from the other group to theirs """
        h = np.asarray(self.act_hazards(val, net), dtype=float)
        acts = net.pars.acts
        draws = np.asarray(sc.dcp(acts).init(force=True).rvs(10_000), dtype=float) if isinstance(acts, ss.Dist) else np.array([net.acts_per_contact()])
        level = self.sim.groups.group_coords[:, term.dim]
        a, b = term.sides
        out = np.zeros(self.n_groups)
        for k,dst in enumerate([b, a]): # The directions, a2b then b2a, as in net.directions: for the people in the group infected
            hk = h[k] if h.ndim == 1 else h[k][level == dst]
            out[level == dst] = 1 - np.mean(np.exp(-np.multiply.outer(draws, np.atleast_1d(hk))), axis=0)
        return out

    def branch_where(self, tr):
        """ Whether a transition is a branch out of each of its sources whose where= is on states and strata, so closes it at entry (see `branch_mult()`) rather than changing its rate """
        return tr.where is not None and ss.conditions.is_static(tr.where, self.sim) and all(tr in self.graph.exits[s.name].branches + sc.tolist(self.graph.exits[s.name].remainder) for s in tr.src if s.people)

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
            if self.immunity is not None and any(flow.state is s for s in self.immune_states):
                mult = mult*self.protection(self.sim.groups.group_coords[:, self.strain], flow.strain)
            if flow.strain is not None and self.carriage is not None and self.applies[flow.state.name][self.strain] and self.superinfects(tr, flow.state): # Infected with another strain while staying in the same state: only if they do not carry it already
                mult = mult*self.can_take(self.sim.groups.group_coords[:, self.strain], flow.strain)
        elif tr.acquisition and flow.src is not None: # An acquisition that is not contact-mediated, e.g. by a bite rate: the susceptibility of its source too
            mult = flow.rel_sus*self.mult_raw('rel_sus', state=flow.state)
        if self.adjust[tr.name] is not None:
            mult = mult*self.adjust[tr.name]
        if flow.src is not None:
            mult = mult*self.mult_raw('transition', tr.name, state=flow.state)
        return mult

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
            elif isinstance(expr, ss.State):
                routes = routes[expr.name] if isinstance(routes, dict) else routes # Under the partnership approximation, for each source state
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
                self._src = flow.state
                depth = max(depth, flatten(flow.expr, flow.strain, routes, rtr))
                self._src = None
            else:
                for leaf in ([] if flow.expr is None else flow.expr.leaves) + sc.tolist(self.dwells.get(flow.tr.name, Bunch(par=None)).par):
                    check(leaf)
            starts.append(len(ops))
        for cv in self.conveyors: # After the flows' operations, each conveyor's (its flow, then its slots) and each update's (its value in the groups the count is in, then the compartment it sets): see ss.kernels.convey() and update_values()
            ops += [kn.CONVEY]*(len(cv.slots) + 1)
            args += [next(i for i,flow in enumerate(self.flows) if flow is cv.flow)] + cv.slots
            starts.append(len(ops))
        for upd in self.graph.updates.values():
            name = upd.state.name
            depth = max(depth, flatten(upd.value*ss.par((self.collapse[name] == np.arange(ng)).astype(float), literal=True), None))
            ops.append(kn.ASSIGN)
            args.append(self.state_comps[name][0])
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
            ops        = np.array(ops, dtype=int),
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
        credits = [(ind, w) for flow in self.flows for ind,w in flow.credits]
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
            maps       = np.array(list(maps.values()), dtype=float).reshape(len(maps), ng, ng),
        )
        local = self._local_mixing and not maps and self.program.contacts.shape[0] == self.program.contacts.shape[2] == 1 and all(flow.record is None for flow in self.flows) and not self.graph.updates # Places do not affect each other within a step (an update can read every place): see ss.kernels.gillespie_places()
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
        self.at_time(0)
        return tuple(np.array([r[k] for r in rows]) for k in range(4)), 0

    def at_time(self, t):
        """ Take values that vary in time at a time t (in the sim's unit) into the step: the year worked out as `tables()` does, so that a stepped run takes the same values as a whole one """
        if self.smooth:
            years, ti = self.mod.t.yearvec, self.mod.t.ti
            start = years[ti]
            end = years[ti+1] if ti + 1 < len(years) else start + (start - years[ti-1] if ti else 0)
            frac = t/self.dt
            self.year = None if frac == 0 else (start + end)/2 if frac == 0.5 else end if frac == 1 else start + frac*(end - start)
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

    def next_generation(self):
        """
        The next-generation matrix at the disease-free state, `F V⁻¹`, over the infected compartments of every group

        The infected compartments are those of the states people are infected into or
        infect from, of any count (such as a reservoir, or infectious mosquitoes) that the
        rate of an infection uses, and of the counts that infected people's numbers move
        a count into (mosquitoes biting them). A new infection is a flow from any other
        compartment into one of these. At the disease-free state, everyone infected is
        moved to the first state, the infected counts are empty, and the other counts
        that are filled by an inflow (mosquitoes not infected) are at their equilibrium. The matrices are worked out by perturbing each infected
        compartment in turn. Returns None if nothing moves people into an infected state.
        See `ss.Disease.R0()`.
        """
        graph = self.graph
        states = {id(s) for s in graph.infected}
        updated = {id(upd.state) for upd in graph.updates.values()} # A count set by an update has no flows, so is held at its value
        found = None
        while found != states: # Until nothing more is found, since each count found can lead to another
            found = set(states)
            for flow in self.flows: # A count that infection depends on, such as a water reservoir, and the states that infection leads to
                if flow.src is not None and id(flow.state) not in states and flow.expr is not None:
                    single = [leaf.states[0] for leaf in flow.expr.leaves if isinstance(leaf, ss.transitions.Count) and len(leaf.states) == 1 and id(leaf.states[0]) not in updated]
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
        people = [i for i in infected if self.comps[i].state.people]
        y_now = self.y # Working out the rates sets the compartments, so they are put back afterwards
        y0 = self.y.copy()
        y0[first] += y0[people].sum(axis=0)
        y0[infected] = 0
        counts = [i for i,comp in enumerate(self.comps) if not comp.state.people and i not in infected and id(comp.state) not in updated]
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
            funcs += [leaf for expr in exprs for leaf in expr.leaves if not (isinstance(leaf, ss.transitions.Dwell) or (isinstance(leaf, ss.par) and (isinstance(leaf.raw, ss.Dist) or not callable(leaf.raw))))] # The time in a state is each slot's own (see slot_times())
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
                self.whole = f'the transfer "{transfers[0]}" of "{mod.name}" has a rate that changes'
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
                    if isinstance(raw, ss.by):
                        vals = [raw.per_group(sim, self.convert, year=year) for year in years]
                    else:
                        vals = raw.series(years, sim, self.convert)
                else:
                    for ti in range(npts):
                        sim.t.ti = mod.t.ti = ti
                        vals.append(raw.per_group(sim, self.convert) if isinstance(raw, ss.by) else self.convert(raw(sim))) # Not by_group(), which holds a value for the step
                kept = self.series[id(par)] = (raw, np.array(vals, dtype=float).reshape(len(vals), -1), per_step, inner)
            self.cache[id(par)] = kept[1] # Steps by groups, where a value is otherwise one number or a number for each group
        sim.t.ti = mod.t.ti = 0
        rows = npts*per_step if self.time_pars else 1

        def table(vals):
            """ The values by group, at each step if any of them varies in time (a series over the steps), and otherwise once """
            steps = any(np.ndim(val) == 2 and len(val) > 1 for val in vals)
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
                for frac in ([0, 0.5, 1] if per_step == 3 else [0]):
                    sim.t.ti = mod.t.ti = ti
                    self.at_time(frac*self.dt)
                    self.cache.clear()
                    rows.append([[0.0 if flow.dynamic else self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows],
                                 [self.flow_mult(flow) if flow.dynamic else 1.0 for flow in self.flows],
                                 [self.prog_value(par) for par in self.prog_pars],
                                 [state.rel_trans*self.mult_raw('rel_trans', state=state)*mask for state,mask in self.prog_exposures]])
            sim.t.ti = mod.t.ti = 0
            self.at_time(0)
            self.cache.clear()
            shape = (self.n_groups,)
            fixed, mult, consts, rel_trans = [np.array([[np.broadcast_to(val, shape) for val in row[k]] for row in rows], dtype=float).reshape(len(rows), len(rows[0][k]), self.n_groups) for k in range(4)]
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

    def init_shifts(self):
        """ The deliveries that the kernels take in a whole run (see `ss.Delivery.tabulated`), as arrays: see `ss.kernels.Shifts` """
        sim = self.sim
        nc, ng, npts = len(self.comps), self.n_groups, self.mod.t.npts
        self.deliveries = [mod for mod in sim.interventions() if isinstance(mod, ss.Delivery) and mod.dis is self.mod and mod.tabulated]
        C = len(self.clocks) # Clocks first, so that a delivery that sets one does so after it has moved up
        K = C + len(self.deliveries)
        due = np.zeros((K, npts), dtype=np.bool_)
        sel, cov = np.zeros((K, nc, ng)), np.zeros((K, nc, ng))
        to, rest = np.zeros((K, ng), dtype=np.int64), np.zeros((K, ng), dtype=np.int64)
        for k,(clock_to,comps) in enumerate(self.clocks):
            due[k, 1:] = True
            sel[k, comps], cov[k, comps] = 1.0, 1.0
            to[k], rest[k] = clock_to, np.arange(ng)
        for k,mod in enumerate(self.deliveries, start=C):
            due[k], sel_k, cov_k, to[k], rest[k] = mod.shifts()
            for s,name in enumerate(self.selector.states):
                for comp in self.state_comps[name]:
                    sel[k, comp], cov[k, comp] = sel_k[s], cov_k[s]
        self.shifts = ss.kernels.Shifts(due=due, sel=sel, cov=cov, to=to, rest=rest)
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
        elif use_jax:
            out = ss.gradients.run(self)
        elif self.rows is not None: # The results by group are kept every few steps, so the run is taken a part at a time, and only those are kept
            self.store_parts(self.run_parts())
            return
        else:
            args, tables = self.run_args()
            tally = np.zeros((self.mod.t.npts, len(self.shifts.to), 2, self.n_groups))
            out = ss.kernels.run(self.make_rng(stream=1), *args, *tables, self.register, self.shifts, tally, self.per_step())
            for k,mod in enumerate(self.deliveries, start=len(self.clocks)):
                mod.store_shifts(tally[:, k])
        self.y, stocks, flows, short = out
        if (use_jax or tally is None) and getattr(self, 'deliveries', None): # A run taken in JAX: the deliveries' results, from the compartments at each time point they happened at
            self.store_replayed(stocks)
        if self.rows is not None:
            self.check_short(short)
            self.store_parts([(0, stocks, flows)])
            return
        self.check_short(short)
        self.sim.t.ti = mod.t.ti = mod.t.npts - 1 # At the last time point, as after a run step by step
        store = lambda key, vals: self.store_run(key, vals.sum(axis=1), vals) # The total, and the values by group
        n = sc.objdict({name:store(f'n_{name}', stocks[:, self.state_comps[name]].sum(axis=1)) for name in graph.states})
        new = sc.objdict({name:store(name, flows[:, ind]) for name,ind in self.tr_index.items()})
        for name,ind in self.transfer_index.items():
            store(name, flows[:, ind])
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            store('new_infections', sum(flows[:, self.tr_index[tr.name]] for tr in graph.transmissions))
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
        kept_s = [i for i,name in enumerate(graph.states) if exprs or has(f'n_{name}')]
        kept_f = [ind for name,ind in (self.tr_index | self.transfer_index).items() if exprs or has(name) or name in sources or (has('new_infections') and name in graph.transmissions)]
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
        while start < npts - 1:
            end = min(start + length, npts - 1)
            tally = np.zeros((end + 1 - start, len(self.shifts.to), 2, self.n_groups))
            tables = self.tables(start, end) # Only for these steps, since by group at every step they can be large
            y, stocks, flows, more = ss.kernels.run(rng, y, program, dt, end + 1, solver, n_sub, *tables, self.register, self.shifts, tally, self.per_step(), start)
            short += more
            last = end == npts - 1
            for k,mod in enumerate(self.deliveries, start=len(self.clocks)):
                mod.store_shifts(tally[:, k] if last else tally[:-1, k], start=start, done=last) # The last row is the start of the next part's first step
            yield (start, stocks, flows) if start == 0 else (start + 1, stocks[1:], flows[1:])
            start = end
        self.y = y
        self.check_short(short)
        return

    def store_parts(self, parts):
        """
        The results of a whole run whose results by group are kept every few steps (see `ss.Sim.group_rows()`), from its compartments and flows at each time point, taken a part at a time

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
        totals = sc.objdict(stocks=np.zeros((len(states), npts)), flows=np.zeros((n_flows, npts)), new_infections=np.zeros(npts))
        groups = sc.objdict(stocks=np.zeros((len(ends), len(kept_s), ng)), flows=np.zeros((len(ends), len(kept_f), ng))) # Only those needed
        pos_s, pos_f = {i:j for j,i in enumerate(kept_s)}, {i:j for j,i in enumerate(kept_f)}
        grouped = self.outcomes and ng > 1 # Outcomes are then worked out by group, a step at a time as a stepped run does, so the flows by group at each step need not be kept
        sources = sorted({key for info in self.outcomes.values() for key,weight in info.sources if key in self.tr_index}) if grouped else []
        if grouped:
            for name in self.outcomes:
                for key in [name, f'n_{name}']:
                    if getattr(res.get(key), 'strata', None) is not None:
                        res[key].strata[:] = 0
        for start,stocks,flows in parts:
            tis = start + np.arange(len(stocks))
            by_state = np.stack([stocks[:, self.state_comps[name]].sum(axis=1) for name in states], axis=1) # Time points by states by groups
            totals.stocks[:, tis] = by_state.sum(axis=-1).T
            totals.flows[:, tis] = flows.sum(axis=-1).T
            if 'new_infections' in res and 'new_infections' not in graph.transitions:
                totals.new_infections[tis] = sum(flows[:, self.tr_index[tr.name]] for tr in graph.transmissions).sum(axis=1)
            at_ends(groups.stocks, rows, ends, tis, by_state[:, kept_s])
            add_rows(groups.flows, rows, tis, flows[:, kept_f])
            if grouped:
                for j,ti in enumerate(tis):
                    mod.t.ti = ti
                    self.cache.clear() # Values held for a step (see Engine.call() and by_group()) are for another step
                    self.flow_groups = sc.objdict({key:flows[j, self.tr_index[key]] for key in sources})
                    self.step_outcomes()
        self.flow_groups = sc.objdict()
        self.sim.t.ti = mod.t.ti = npts - 1
        store = self.store_run
        stock_g = lambda i: groups.stocks[:, pos_s[i]] if i in pos_s else None # By group, if needed
        flow_g = lambda ind: groups.flows[:, pos_f[ind]] if ind in pos_f else None
        n = sc.objdict({name:store(f'n_{name}', totals.stocks[i], stock_g(i)) for i,name in enumerate(states)})
        new = sc.objdict({name:store(name, totals.flows[ind], flow_g(ind)) for name,ind in self.tr_index.items()})
        for name,ind in self.transfer_index.items():
            store(name, totals.flows[ind], flow_g(ind))
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            store('new_infections', totals.new_infections, sum(flow_g(self.tr_index[tr.name]) for tr in graph.transmissions) if getattr(res.new_infections, 'strata', None) is not None else None)
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
                ss.kernels.apply_shifts(self.rng, np.array(stocks[ti]), sh, ti, False, tally[ti]) # Expected numbers, so nothing is drawn
        for k,mod in enumerate(self.deliveries, start=len(self.clocks)):
            mod.store_shifts(tally[:, k])
        return

    def run_many(self, seeds):
        """
        Take many runs that differ only in their seed, in calls to the kernels that take all of them, a part of the run at a time

        Args:
            seeds (list): the seed of each run, as would be given to the sim as `rand_seed`

        Returns each result that is a single series, as runs by steps, and (as `out.groups`) the
        values by group of the results that keep them, as runs by rows by groups: a row for each
        time point, or for every few steps if that would take more than `max_bytes` of memory (with
        the time point of each row as `out.group_ti`; see `ss.Sim.group_rows()`).
        """
        graph = self.graph
        res = self.mod.results
        self.restart()
        args, _ = self.run_args(tables=False)
        y0, program, dt, npts, solver, n_sub = args
        rngs = nb.typed.List([self.make_rng(stream=1, seed=seed) for seed in seeds]) # The random numbers that each run would have
        ng = self.n_groups
        states = list(graph.states)
        state_of = np.array([states.index(comp.state.name) for comp in self.comps])
        n_flows = self.program.n_trans + self.program.n_movers

        kept_s, kept_f = self.group_needs() # The states and flows to keep by group
        rows = group_ti = None
        if ng > 1 and (kept_s or kept_f or self.deliveries):
            row_bytes = len(seeds)*(len(kept_s) + len(kept_f) + 2*len(self.deliveries))*ng*8
            rows, group_ti = self.sim.group_rows(npts, row_bytes, self.max_bytes, name=self.mod.name)
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
        per_step = self.per_step()
        length = max(int(50e6//(sum(len(table) for table in [program.fixed, program.mult, program.consts, program.rel_trans])*per_step*ng*8)), 1) # Steps per part, so the tables of values that vary in time (by group, at every step) are not all held at once
        start = 0
        while True:
            end = min(start + length, npts - 1)
            ss.kernels.run_many(rngs, ys, program, dt, start, end + 1, solver, n_sub, *self.tables(start, end), self.register, self.shifts, stocks, flows, stocks_g, flows_g,
                                tally, tally_g, state_of, np.zeros(0, dtype=int) if rows is None else rows, grp_state, grp_flow, per_step)
            if end >= npts - 1:
                break
            start = end
        out = self.collect(stocks, flows, grouped=None if rows is None else (stocks_g, flows_g, kept_s, kept_f, group_ti), seeds=seeds)
        out.deliveries = sc.objdict({mod.name:mod.many_shifts(tally[:, :, k], None if rows is None else tally_g[:, :, k], group_ti) for k,mod in enumerate(self.deliveries, start=len(self.clocks))})
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
                groups[f'n_{states[i]}'] = stocks_g[:, :, j, :]
            for name,ind in (self.tr_index | self.transfer_index).items():
                if ind in pos:
                    groups[name] = flows_g[:, :, pos[ind], :]
            if 'new_infections' in res and 'new_infections' not in graph.transitions and all(tr.name in groups for tr in graph.transmissions):
                groups.new_infections = sum(groups[tr.name] for tr in graph.transmissions)
            stratified = [name for name in self.ratio_names() if getattr(res.get(name), 'strata', None) is not None]
            if stratified and all(f'n_{name}' in groups for name in states): # Sums and ratios of states by group, as in run_all()
                ctx = StockCtx(self, sc.objdict({name:groups[f'n_{name}'] for name in states}), sc.objdict({name:groups[name] for name in self.tr_index if name in groups}))
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
        out = sc.objdict({f'n_{name}':vals for name,vals in n.items()})
        out.update(new)
        out.update({name:flows[:, :, ind] for name,ind in self.transfer_index.items()})
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            out.new_infections = sum(new[tr.name] for tr in graph.transmissions)
        out.update(self.derived(n, new, outcomes=outcomes is None, expected=expected, seeds=seeds))
        if outcomes is not None:
            out.update(outcomes)
        if 'cum_infections' in res and 'new_infections' in out:
            out.cum_infections = np.cumsum(out.new_infections, axis=-1)
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

        Returns a dict of the outcomes, the prevalence, and the results given as expressions.
        This is what `update_results()` does on each step, for every step at once.
        """
        graph = self.graph
        res = self.mod.results
        out = self.derived_outcomes(new, expected=expected, seeds=seeds) if outcomes else sc.objdict()
        if self.standard and 'prevalence' in res and 'prevalence' not in graph.exprs:
            people = sum(n[s.name] for s in graph.people_states)
            infected = sum(n[s.name] for s in graph.infected)
            out.prevalence = np.divide(infected, people, out=np.zeros(np.shape(people)), where=people>0)
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
        for name,info in self.outcomes.items():
            if names is not None and name not in names:
                continue
            amount = 0.0
            for key,weight in info.sources:
                flow = new[key] if key in new else out[key]
                if isinstance(weight, ss.transitions.Expr):
                    weight = self.eval(weight)
                    if np.ndim(weight) == 1 and np.ndim(flow) >= 2 and np.shape(flow)[-2] == len(weight): # A weight for each group, with the flows by group
                        weight = weight[:, None]
                if draw and np.all((0 <= weight) & (weight <= 1)): # A fraction of whole events is a draw
                    amount = amount + self.rng.binomial(np.round(flow).astype(int), weight)
                else:
                    amount = amount + flow*weight
            kernel = info.kernel
            if len(kernel) == 1:
                now = amount*kernel[0]
            elif draw: # Each event is reported after a delay drawn for it
                now = np.zeros(amount.shape[:-1] + (npts + len(kernel),))
                for ind in zip(*np.nonzero(amount)):
                    now[ind[:-1]][ind[-1]:ind[-1]+len(kernel)] += self.rng.multinomial(int(round(amount[ind])), kernel/kernel.sum())
                now = now[..., :npts]
            else:
                now = np.apply_along_axis(np.convolve, -1, amount, kernel)[..., :npts]
            if info.stay is not None: # Each arrival leaves after a stay drawn for it
                if draw:
                    leaving = np.zeros(now.shape[:-1] + (npts + len(info.stay),))
                    for ind in zip(*np.nonzero(now)):
                        leaving[ind[:-1]][ind[-1]:ind[-1]+len(info.stay)] += self.rng.multinomial(int(round(now[ind])), info.stay)
                    leaving = leaving[..., :npts]
                else:
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

    #%% Stepping

    def substeps(self):
        """
        The number of substeps to take in each timestep (0 to let the solver choose)

        The chain-binomial solver takes whole steps, as most chain-binomial models do: it
        moves people through as many stages of a dwell time as they pass in a step (see
        `ss.kernels.draw_chain()`).
        """
        n_sub = self.sim.pars.substeps
        if n_sub is None and self.sim.solver == 'binomial':
            n_sub = 1
        return n_sub or 0

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
        self.cache.clear() # The interventions have run since the step started
        self.update_targets()
        self._mover_rates = self.mover_rates()
        if self.program is None:
            self.init_program()
        sim = self.sim
        dt = self.dt
        solver = sim.integrator if sim.solver == 'ode' else sim.solver
        tables, row = self.step_tables(dt, smooth=solver != 'euler') # Euler holds values that vary in time for the step
        self.y, flows = ss.kernels.one_step(self.kernel_rng, self.y, self.program, dt, ss.kernels.solvers[solver], self.substeps(), tables, row)
        if self.sim.mixing.register is not None: # The events dated at the end of the step
            self.check_short(ss.kernels.apply_events(self.kernel_rng, self.y, self.register, self.mod.ti + 1, self.integer))
        if solver == 'gillespie' and self.program.local and self.program.has_det: # By place, the counts are settled after the events, as in SimInf
            ss.kernels.settle_places(self.y, self.program, dt, flows)
        self.record_flows(flows)
        self.step_clocks() # At the start of the next step, before its deliveries
        return

    def record_flows(self, flows):
        """ After a step: the flow along each transition and mover over it, by group, and the people at its end """
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
        for i,mover in enumerate(self.movers):
            if mover.name is not None:
                self.flowcounts[mover.name] += flows[self.n_trans + i].sum()
                self.flow_groups[mover.name] = flows[self.n_trans + i]
        self.record_stocks()
        self.stepped = self.mod.ti
        if self.syncs and self.cross is None:
            self.pass_syncs(flows)
        return
