"""
Diseases whose states are tracked jointly in a compartmental run
"""
import numpy as np
import numba as nb
import sciris as sc
import starsim as ss
from .engine import by_level, no_tables
from .compartmental import Compartmental

__all__ = ['Crossing']


class Crossing:
    """
    Diseases whose states are tracked jointly in a compartmental run, since a condition in one refers to a state of another (e.g. `ss.modify(hiv.rel_trans, multiply=3, where=syph.I)`)

    Each disease in the crossing is a dimension of the sim, named after it, whose levels are
    its compartments and which applies to the people of the others. Each keeps its own engine,
    which works out its rates and its results, and sees the people as it has them: by its own
    compartments, and by groups that include the compartments of the others (with its own
    dimension at its first level).

    They are run by one program: its compartments are the product of the diseases' compartments
    of people (and, for a compartment that is not people, such as a reservoir, one for each
    combination of the others'), and its groups are those of the sim's other dimensions. Each flow
    of a disease is a flow of the program for each combination of the others' compartments, so
    every rate is worked out from the people as they are at the start of the step (not each disease
    in turn), and a whole run, many runs, a particle filter, and a run in JAX take a crossing as they
    take one disease. The solvers that draw whole events in steps draw each disease's in turn, from
    those rates, so each disease's moves (e.g. through the stages of a dwell time) are drawn as they
    would be for it alone. The program's compartments (`y`) are the one copy of the people: each engine
    takes its view of them again after each step, and after anything in Python changes them (`put()`).

    Args:
        sim (`ss.Sim`): the sim
        names (list): the names of the diseases
    """
    def __init__(self, sim, names):
        self.sim = sim
        self.names = names
        self.engines = []
        self.layout = None # Where each engine's compartments and flows are in the program: see init_layout()
        self.program = None # The program of the diseases together: see build()
        self.built = None # What it was built from
        self.y = None # Its compartments, by group
        self.pending = dict() # The tables of each engine that has worked out its rates for the current step: see step()
        self.outs = None # The output of a whole run, for each engine that has not taken it yet: see whole_run()
        self.jax_run = None # The run in JAX, if one has been made: see ss.gradients
        self._lead = None
        return

    def __repr__(self):
        return f'Crossing({" × ".join(self.names)})'

    @property
    def lead(self):
        """ The disease whose engine moves people in ways that belong to the sim (aging, relocating, a movement register): the first, or the first of those with the most of the states that a movement register names """
        if self._lead is None:
            self._lead = self.names[0]
            if self.sim.mixing.register is not None:
                ev = self.sim.mixing.events()
                named = {name for key in ev.sets for states in key if states is not None for name in states}
                counts = [len(named & set(self.sim.get_declared(name).graph.states)) for name in self.names]
                self._lead = self.names[int(np.argmax(counts))]
        return self._lead

    def reset(self):
        """ Forget any step half taken, and any whole run not yet handed out, before the run starts again """
        self.pending.clear()
        self.outs = None
        return

    def shrink(self):
        """ Drop the arrays the run worked in, for saving """
        ss.shrink(self, ['sim', 'program', 'y', 'layout', 'pieces', 'register', 'shifts', 'jax_run', 'outs'])
        return

    #%% The people

    def joint(self, engine):
        """ The people in an engine as one array over the dimensions of the sim, with the engine's own compartments as the levels of its dimension """
        d = self.sim.groups.dim_index(engine.mod.name)
        y = engine.y[engine.people_comps].reshape((-1,) + self.sim.groups.group_shape)
        return np.moveaxis(np.take(y, 0, axis=d+1), 0, d)

    def set_joint(self, joint, engine):
        """ Give an engine the people in a joint array """
        sim = self.sim
        d = sim.groups.dim_index(engine.mod.name)
        n = len(engine.people_comps)
        y = np.zeros((n,) + sim.groups.group_shape)
        y[(slice(None),)*(d+1) + (0,)] = np.moveaxis(joint, d, 0)
        engine.y[engine.people_comps] = y.reshape(n, -1)
        return

    def init(self):
        """ Start the diseases independent of each other: the share of people in each compartment of one is the same in every compartment of the others """
        sim = self.sim
        self.engines = [sim.get_declared(name).engine for name in self.names]
        self.check()
        if self.layout is None:
            self.init_layout()
        dims = [sim.groups.dim_index(name) for name in self.names]
        joint = 1.0
        for engine,d in zip(self.engines, dims):
            own = [d] + [dd for dd,owner in self.private.items() if owner is engine]
            others = [dd for dd in dims if dd != d] + [dd for dd,owner in self.private.items() if owner is not engine]
            marginal = self.joint(engine).sum(axis=tuple(others), keepdims=True)
            total = marginal.sum(axis=tuple(own), keepdims=True)
            joint = joint*np.divide(marginal, total, out=np.zeros(marginal.shape), where=total>0)
        joint = joint*total # The number of people in each group is the same for every disease
        if self.engines[0].integer:
            joint = Compartmental.round_preserving(joint)
        self.set_joint(joint, self.engines[0])
        self.y = np.zeros((self.n_rows, self.n_cols))
        self.scatter(self.engines[0]) # The people, from the first, and the other compartments from each
        for engine in self.engines[1:]:
            self.scatter(engine, people=False)
        if sim.mixing.register is not None: # The events dated at the start, for everyone
            lead = sim.get_declared(self.lead).engine
            self.register = self.join_register(lead.init_register())
            lead.check_short(ss.kernels.apply_events(lead.make_rng(stream=2), self.y, self.register, 0, lead.integer))
        for engine in self.engines:
            engine.y = self.gather(engine)
            engine.record_stocks()
        self.rng = self.engines[0].make_rng(stream=1) # As a whole run starts with, so stepping gives the same run
        self.reset()
        return

    def check(self):
        """
        Each dimension other than the diseases' own has to apply to every state of people in the crossing, or to
        some states of one disease only (e.g. a strain), but not to its first state, where people born to the others start;
        and the diseases step together, so they have to have the same timestep
        """
        sim = self.sim
        self.private = dict() # The dimensions of some states of one disease only, and its engine
        for d,strat in enumerate(sim.groups.strats):
            if strat.kind == 'disease':
                continue
            flags = {f'{e.mod.name}.{s.name}':bool(e.applies[s.name][d]) for e in self.engines for s in e.graph.people_states}
            if len(set(flags.values())) < 2:
                continue
            owners = [e for e in self.engines if any(e.applies[s.name][d] for s in e.graph.people_states)]
            first = owners[0].graph.first.name
            if len(owners) > 1 or owners[0].applies[first][d]:
                some = [name for name,flag in flags.items() if flag]
                raise ss.CapabilityError(f'"{strat.name}" applies to some states of {sc.strjoin(self.names)} ({sc.strjoin(some)}) but not to others, so their states cannot be tracked jointly with solver="{sim.solver}" yet. '
                                         f'A dimension of some states of one disease only works if it is not its first state.\n'
                                         f'     Run with solver="abm".')
            self.private[d] = owners[0]
        steps = {(e.mod.t.npts, e.dt) for e in self.engines}
        if len(steps) > 1:
            raise ss.CapabilityError(f'{sc.strjoin(self.names)} have timesteps of their own, so their states cannot be tracked jointly with solver="{sim.solver}", since they step together.\n'
                                     f'     Give them the same timestep, or run with solver="abm".')
        return

    def init_layout(self):
        """
        Where each engine's compartments and flows, in each of its groups, are in the program's

        A compartment of people of an engine, in a group (with the engine's own dimension at its first level), is
        the program's compartment for it and the others' compartments that are the levels of their dimensions in
        that group, in the program's group for the levels of the other dimensions. For each engine, `idx` gives
        where each of its compartments, in each of its groups, is in the program's compartments by groups
        (flattened), or -1 for the groups it keeps empty; `fidx` does the same for its transitions and movers.
        """
        sim = self.sim
        coords = sim.groups.group_coords
        dims = [sim.groups.dim_index(name) for name in self.names]
        sizes = [sim.groups.strats[d].n for d in dims]
        rest = [d for d in range(len(sim.groups.strats)) if d not in dims] # The dimensions that are the program's groups
        shape = tuple(sim.groups.strats[d].n for d in rest)
        self.rest = rest
        self.n_cols = R = int(np.prod(shape))
        r_of = np.ravel_multi_index(coords[:, rest].T, shape) if rest else np.zeros(len(coords), dtype=int) # The program's group of each group
        self.n_people = int(np.prod(sizes))
        n_rows = self.n_people
        n_trans = n_movers = 0
        self.layout = []
        for k,engine in enumerate(self.engines):
            if len(engine.people_comps) != sizes[k]:
                errormsg = f'The dimension "{self.names[k]}" has {sizes[k]} levels, but the disease has {len(engine.people_comps)} compartments of people'
                raise RuntimeError(errormsg)
            others = [j for j in range(len(dims)) if j != k]
            osizes = [sizes[j] for j in others]
            O = int(np.prod(osizes)) # The combinations of the others' compartments
            valid = coords[:, dims[k]] == 0
            o_of = np.ravel_multi_index(coords[:, [dims[j] for j in others]].T, osizes)
            cols = np.zeros((O, R), dtype=np.int64) # The engine's group for each combination and each of the program's groups
            cols[o_of[valid], r_of[valid]] = np.flatnonzero(valid)
            rowmap = np.zeros((engine.n_comps, O), dtype=np.int64) # The program's compartment for each of the engine's, for each combination
            levels = np.zeros((len(dims), O), dtype=np.int64)
            levels[others] = np.unravel_index(np.arange(O), osizes)
            for i,c in enumerate(engine.people_comps):
                levels[k] = i
                rowmap[c] = np.ravel_multi_index(levels, sizes)
            counts = [c for c in range(engine.n_comps) if c not in engine.people_comps] # Compartments that are not people: a copy for each combination
            for i,c in enumerate(counts):
                rowmap[c] = n_rows + i*O + np.arange(O)
            n_rows += len(counts)*O
            idx = np.full((engine.n_comps, sim.groups.n_groups), -1, dtype=np.int64)
            idx[:, valid] = rowmap[:, o_of[valid]]*R + r_of[valid]
            self.layout.append(sc.objdict(O=O, valid=valid, slice_of=np.where(valid, o_of, -1), r_of=r_of, cols=cols, rowmap=rowmap, idx=idx, counts=counts, toff=n_trans, moff=n_movers))
            n_trans += engine.n_trans*O
            n_movers += len(engine.movers)*O
        self.n_rows = n_rows
        self.n_trans = n_trans
        self.n_movers = n_movers
        self.pieces = sc.objdict(y=[])
        for k,(engine,L) in enumerate(zip(self.engines, self.layout)):
            ivalid = np.flatnonzero(L.valid)
            o, r = L.slice_of[ivalid], r_of[ivalid]
            nt = engine.n_trans
            L.fidx = np.full((nt + len(engine.movers), sim.groups.n_groups), -1, dtype=np.int64)
            L.fmember = np.zeros((nt + len(engine.movers), n_trans + n_movers)) # Which of the program's transitions and movers are each of the engine's
            for i in range(nt + len(engine.movers)):
                first = L.toff + i*L.O if i < nt else n_trans + L.moff + (i - nt)*L.O
                L.fidx[i, ivalid] = (first + o)*R + r
                L.fmember[i, first:first+L.O] = 1
            states = list(engine.graph.states)
            L.comp_member = np.array([[comp.state.name == name for comp in engine.comps] for name in states], dtype=float) # Which compartments each state is
            L.member = np.zeros((len(states), n_rows)) # Which of the program's compartments each state is
            for c,comp in enumerate(engine.comps):
                L.member[states.index(comp.state.name), L.rowmap[c]] = 1
            mine = np.zeros(L.idx.shape, dtype=bool) # The program's compartments come from the first engine for the people, and from each for its others
            mine[slice(None) if k == 0 else L.counts] = True
            E = np.flatnonzero(mine & (L.idx >= 0))
            self.pieces.y.append((L.idx.ravel()[E], E))
        return

    def gather(self, engine):
        """ An engine's view of the people (and its other compartments), from the program's compartments """
        L = self.layout[self.engines.index(engine)]
        return np.where(L.idx >= 0, self.y.ravel()[np.maximum(L.idx, 0)], 0.0)

    def scatter(self, engine, people=True):
        """ Put an engine's compartments into the program's: all of them (each engine has every person), or only those that are not people """
        L = self.layout[self.engines.index(engine)]
        rows = slice(None) if people else L.counts
        idx = L.idx[rows]
        keep = idx >= 0
        self.y.ravel()[idx[keep]] = engine.y[rows][keep]
        return

    def put(self, engine):
        """ After something in Python has changed the people of one engine (e.g. an intervention moved some): give the change to the program and to the other engines """
        self.scatter(engine)
        for other in self.engines:
            if other is not engine:
                other.y = self.gather(other)
        return

    #%% The program

    def build(self, programs=None):
        """
        The program of the diseases together (see the class docstring), from the program of each

        The layout is worked out here, and again only if an engine's program is made again (e.g. when a branch
        probability changes); the numbers, which can change from step to step or run to run, are taken from the
        engines' by `numbers()`, from where each is in an engine's program (`self.pieces`).
        """
        kn = ss.kernels
        progs = programs or [engine.program for engine in self.engines]
        R = self.n_cols
        p0 = progs[0]
        L0 = self.layout[0]
        full = any(p.contacts.shape[0] > 1 or p.contacts.shape[2] > 1 for p in progs) or any(engine.net_mods for engine in self.engines) # Contacts by exposure and group
        src, named, det, index, record, starts, ops, args = [], [], [], [], [], [0], [], []
        tgt_comp, tgt_map, tgt_starts, crd_index, crd_starts = [], [], [0], [], [0]
        tgt_rec = []
        mov_comps, mov_pair, mov_cstarts, mov_estarts = [], [[], []], [0], [0]
        sets, exposures, maps = dict(), dict(), dict()
        groups, chain = [], [] # The flows out of each compartment, for the stochastic solvers: for each disease in turn, each disease's ended by an empty set (see ss.kernels.draw())
        conveyors = [] # The flow out of the last slot of each conveyor, then its slots
        for key in ['flows', 'consts', 'rel_trans', 'contacts', 'tgt', 'crd', 'maps', 'movers']:
            self.pieces[key] = []
        arr = lambda items: np.array(items, dtype=np.int64)
        pairs = lambda I, C: (arr(I), np.array(C, dtype=np.int64).reshape(len(I), R))

        positions = {} # The position of each key in its dict (sets, exposures, maps), kept so that finding it is not a search of the dict, which made building a crossing of three diseases take hours
        def add(items, key, val=None):
            pos = positions.setdefault(id(items), {})
            if key not in pos:
                items.setdefault(key, val)
                pos[key] = len(pos)
            return pos[key]

        def which_map(k, m, o):
            if m < 0:
                return -1
            M = progs[k].maps[m][self.layout[k].cols[o]]
            if not np.allclose(M[:, self.layout[k].cols[o]].sum(axis=1), M.sum(axis=1)):
                raise ss.CapabilityError(f'a transition of "{self.names[k]}" moves people between the compartments of another disease it is tracked jointly with, which cannot run with solver="{self.sim.solver}".\n'
                                         f'     Run with solver="abm".')
            return add(maps, (k, int(m), o))

        n_flows = n_consts = 0
        for k,(p,L) in enumerate(zip(progs, self.layout)):
            O = L.O
            split = p.contacts.shape[2] > 1 # Contacts that differ by the group of the people exposed, so an exposure is worked out for each combination of theirs
            F, FC, TI, TC, CI, CC, XI, XC, KI, KC, MI = [], [], [], [], [], [], [], [], [], [], []

            def exposure(e, o_src, o_sus):
                key = (k, int(e), o_src, o_sus if split else None)
                if key not in exposures:
                    comps = p.set_comps[p.set_starts[p.exp_set[e]]:p.set_starts[p.exp_set[e]+1]]
                    add(exposures, key, add(sets, tuple(sorted(L.rowmap[comps, o_src]))))
                    XI.append(e)
                    XC.append(L.cols[o_src])
                    KI.append(e if p.contacts.shape[0] > 1 else 0)
                    KC.append(L.cols[o_sus] if split else np.zeros(R, dtype=np.int64))
                return positions[id(exposures)][key]

            def exposure_sum(e, o_sus):
                """ The exposure from the infectious in every combination of the others' compartments, added up: an exposure whose set is minus the number of those before it that it adds up (see `ss.kernels.Program`), so each flow takes it once """
                key = (k, int(e), 'sum', o_sus if split else None)
                if key not in exposures:
                    for j in range(O):
                        exposure(e, j, o_sus)
                    add(exposures, key, -O)
                    XI.append(e)
                    XC.append(L.cols[0])
                    KI.append(e if p.contacts.shape[0] > 1 else 0)
                    KC.append(L.cols[o_sus] if split else np.zeros(R, dtype=np.int64))
                return positions[id(exposures)][key]

            for f in range(len(p.src)):
                for o in range(O):
                    s = p.src[f]
                    src.append(L.rowmap[s, o] if s >= 0 else -1)
                    named.append(p.named[f])
                    det.append(p.det[f])
                    index.append(L.toff + p.index[f]*O + o)
                    record.append(which_map(k, p.record[f], o))
                    F.append(f)
                    FC.append(L.cols[o])
                    for i in range(p.starts[f], p.starts[f+1]):
                        op, arg = p.ops[i], p.args[i]
                        if op == kn.CONST:
                            ops.append(op)
                            args.append(n_consts + arg*O + o)
                        elif op in [kn.COUNT, kn.FRAC, kn.TOTAL]: # The number in some compartments, whatever the others' compartments
                            comps = p.set_comps[p.set_starts[arg]:p.set_starts[arg+1]]
                            ops.append(op)
                            args.append(add(sets, tuple(sorted(L.rowmap[comps].ravel()))))
                        elif op == kn.EXPOSURE: # The exposure from the infectious in each combination of the others' compartments, added up
                            ops.append(op)
                            args.append(exposure_sum(arg, o))
                        else:
                            ops.append(op)
                            args.append(arg)
                    starts.append(len(ops))
                    for t in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                        tgt_comp.append(L.rowmap[p.tgt_comp[t], o])
                        tgt_map.append(which_map(k, p.tgt_map[t], o))
                        tgt_rec.append(L.toff + p.tgt_rec[t]*O + o if p.tgt_rec[t] >= 0 else -1)
                        TI.append(t)
                        TC.append(L.cols[o])
                    tgt_starts.append(len(tgt_comp))
                    for t in range(p.crd_starts[f], p.crd_starts[f+1]):
                        crd_index.append(L.toff + p.crd_index[t]*O + o)
                        CI.append(t)
                        CC.append(L.cols[o])
                    crd_starts.append(len(crd_index))
            for m in range(p.n_movers):
                comps = p.mov_comps[p.mov_starts[0, m]:p.mov_starts[0, m+1]]
                ents = np.arange(p.mov_starts[1, m], p.mov_starts[1, m+1])
                g, h = p.mov_pair[0, ents], p.mov_pair[1, ents]
                for o in range(O):
                    keep = L.slice_of[g] == o
                    if np.any(L.slice_of[h[keep]] != o):
                        raise ss.CapabilityError(f'a movement between groups of "{self.names[k]}" moves people between the compartments of another disease it is tracked jointly with, which cannot run with solver="{self.sim.solver}".\n'
                                                 f'     Run with solver="abm".')
                    mov_comps += list(L.rowmap[comps, o])
                    mov_cstarts.append(len(mov_comps))
                    mov_pair[0] += list(L.r_of[g[keep]])
                    mov_pair[1] += list(L.r_of[h[keep]])
                    mov_estarts.append(len(mov_pair[0]))
                    MI += list(ents[keep])
            for s in range(len(p.src_starts) - 1): # The engine's flows out of each compartment, for each combination of the others' compartments, with the stages that the chain-binomial solver passes several of in a step
                flows = p.src_flows[p.src_starts[s]:p.src_starts[s+1]]
                c = p.src[flows[0]]
                d = p.src_chain[s]
                for o in range(O):
                    groups.append(n_flows + flows*O + o)
                    chain.append((d, n_flows + p.src_stage[s]*O + o, n_flows + p.src_last[s]*O + o) if d > 1 else (0, -1, -1))
            if groups and len(groups[-1]): # Followed by the next disease's (an empty set is never last, so the kernels can read the next one's first flow)
                groups.append([])
                chain.append((0, -1, -1))
            for i in range(len(p.starts) - len(p.src) - 1): # Each conveyor, for each combination of the others' compartments (see ss.kernels.convey())
                a, b = p.starts[len(p.src)+i], p.starts[len(p.src)+i+1]
                if p.ops[a] != kn.CONVEY:
                    raise ss.CapabilityError(f'"{self.names[k]}" sets a count each step (ss.update), which cannot run with solver="{self.sim.solver}" in diseases tracked jointly yet.\n'
                                             f'     Run with solver="abm".')
                for o in range(O):
                    conveyors.append([n_flows + p.args[a]*O + o] + list(L.rowmap[p.args[a+1:b], o]))
            self.pieces.flows.append(pairs(F, FC))
            self.pieces.consts.append(pairs(np.repeat(np.arange(len(p.consts)), O), [L.cols[o] for i in range(len(p.consts)) for o in range(O)]))
            self.pieces.rel_trans.append(pairs(XI, XC))
            self.pieces.contacts.append(pairs(KI, KC))
            self.pieces.tgt.append(pairs(TI, TC))
            self.pieces.crd.append(pairs(CI, CC))
            self.pieces.maps.append(pairs([key[1] for key in maps if key[0] == k], [self.layout[k].cols[key[2]] for key in maps if key[0] == k]))
            self.pieces.movers.append(arr(MI))
            n_flows += len(p.src)*O
            n_consts += len(p.consts)*O

        # Mixing, which is the same for every engine, and is by the groups of the other dimensions only
        places = p0.places[L0.cols[0]]
        levels = p0.levels[:, L0.cols[0]]
        for p,L in zip(progs, self.layout):
            for o in range(L.O):
                if not (np.array_equal(p.places[L.cols[o]], places) and np.array_equal(p.levels[:, L.cols[o]], levels)):
                    raise ss.CapabilityError(f'people mix by the compartments of {sc.strjoin(self.names)}, so their states cannot be tracked jointly with solver="{self.sim.solver}".\n'
                                             f'     Run with solver="abm".')
        pl_starts, pl_groups = by_level(places, p0.n_places)

        if groups and not len(groups[-1]):
            groups.pop()
            chain.pop()
        for conv in conveyors: # After the flows' operations
            ops += [kn.CONVEY]*len(conv)
            args += conv
            starts.append(len(ops))
        src = arr(src)
        ends = lambda lists: np.cumsum([0] + [len(item) for item in lists])
        flat = lambda lists: arr([i for item in lists for i in item])
        setlist = list(sets)
        nf, nx, nt = len(src), len(exposures), len(p0.sizes)
        self.full = full
        self.program = kn.Program(
            n_trans    = self.n_trans,
            src        = src,
            named      = np.array(named, dtype=bool),
            index      = arr(index),
            record     = arr(record),
            inflow     = np.zeros((nf, R)),
            fixed      = np.zeros((nf, R)),
            mult       = np.ones((nf, R)),
            depth      = max(p.depth for p in progs) + 1,
            starts     = arr(starts),
            ops        = arr(ops),
            args       = arr(args),
            consts     = np.zeros((n_consts, R)),
            set_starts = ends(setlist),
            set_comps  = flat(setlist),
            people     = np.arange(self.n_people),
            has_det    = bool(any(p.has_det for p in progs)),
            det        = np.array(det, dtype=bool),
            places     = places,
            n_places   = p0.n_places,
            exp_set    = arr(list(exposures.values())),
            rel_trans  = np.ones((nx, R)),
            levels     = levels,
            sizes      = p0.sizes,
            block      = p0.block,
            contacts   = np.ones((nx, nt, R)) if full else p0.contacts.copy(),
            use        = p0.use,
            mat        = p0.mat,
            sh_pair    = p0.sh_pair,
            sh_val     = p0.sh_val,
            tgt_starts = arr(tgt_starts),
            tgt_comp   = arr(tgt_comp),
            tgt_w      = np.zeros((len(tgt_comp), R)),
            tgt_map    = arr(tgt_map),
            tgt_rec    = arr(tgt_rec),
            crd_starts = arr(crd_starts),
            crd_index  = arr(crd_index),
            crd_w      = np.zeros((len(crd_index), R)),
            maps       = np.zeros((len(maps), R, R)),
            n_movers   = self.n_movers,
            mov_starts = arr([mov_cstarts, mov_estarts]),
            mov_comps  = arr(mov_comps),
            mov_pair   = arr(mov_pair).reshape(2, -1),
            mov_rate   = np.zeros(len(mov_pair[0])),
            src_starts = ends(groups),
            src_flows  = flat(groups),
            src_chain  = arr([c[0] for c in chain]),
            src_stage  = arr([c[1] for c in chain]),
            src_last   = arr([c[2] for c in chain]),
            local      = all(p.local for p in progs) and not full,
            pl_starts  = pl_starts,
            pl_groups  = pl_groups,
        )
        J = self.numbers([{key:getattr(p, key) for key in ['inflow', 'tgt_w', 'crd_w', 'maps']} for p in progs])
        self.program = self.program._replace(**J)
        self.built = [id(p) for p in progs]
        self.register = self.join_register(self.sim.get_declared(self.lead).engine.register)
        self.shifts = self.no_shifts()
        return self.program

    def join_register(self, reg):
        """ The movement register of the lead engine (see `lead`) for the program: see `ss.kernels.Register` """
        ints = lambda vals: np.array(vals, dtype=np.int64)
        L = self.layout[self.names.index(self.lead)]
        R = self.n_cols
        if reg is None or not len(reg.src):
            return ss.kernels.Register(starts=ints([0]), src=ints([]), dst=ints([]), n=np.zeros(0), sel=ints([]), sel_starts=ints([0]), sel_comps=ints([]), sel_to=ints([]),
                                       ent_starts=ints([0]), ent_comps=ints([]), ent_w=np.zeros((0, R)), pg_starts=ints([0]), pg_groups=ints([]), step=0)
        sel_comps, sel_to, sel_starts = [], [], [0]
        ent_comps, ent_w, ent_starts = [], [], [0]
        for s in range(len(reg.sel_starts) - 1):
            for i in range(reg.sel_starts[s], reg.sel_starts[s+1]):
                sel_comps += list(L.rowmap[reg.sel_comps[i]])
                sel_to += list(L.rowmap[reg.sel_to[i]])
            sel_starts.append(len(sel_comps))
            for i in range(reg.ent_starts[s], reg.ent_starts[s+1]):
                found = [o for o in range(L.O) if np.any(reg.ent_w[i, L.cols[o]])]
                if len(found) > 1:
                    raise ss.CapabilityError(f'the movement register adds people to more than one compartment of {sc.strjoin(n for n in self.names if n != self.lead)}, which cannot run with solver="{self.sim.solver}" yet.\n'
                                             f'     Run with solver="abm".')
                o = found[0] if found else 0
                ent_comps.append(L.rowmap[reg.ent_comps[i], o])
                ent_w.append(reg.ent_w[i, L.cols[o]])
            ent_starts.append(len(ent_comps))
        d = self.sim.mixing.space
        pg_starts, pg_groups = by_level(np.asarray(self.sim.mixing.places)[L.cols[0]], self.sim.groups.strats[d].n)
        step = int(np.prod([self.sim.groups.strats[dd].n for dd in self.rest if dd > d])) # How far apart adjacent places are in the program's groups
        return reg._replace(sel_starts=ints(sel_starts), sel_comps=ints(sel_comps), sel_to=ints(sel_to), ent_starts=ints(ent_starts), ent_comps=ints(ent_comps),
                            ent_w=np.array(ent_w, dtype=float).reshape(len(ent_comps), R), pg_starts=ints(pg_starts), pg_groups=ints(pg_groups), step=step)

    def no_shifts(self):
        """ The deliveries that the kernels take: none, since deliveries are taken in Python when several diseases are declared (see `ss.Delivery.tabulated`) """
        npts = self.engines[0].mod.t.npts
        if any(getattr(engine, 'deliveries', None) for engine in self.engines):
            errormsg = 'The deliveries of diseases tracked jointly are not taken by the kernels'
            raise RuntimeError(errormsg)
        return ss.kernels.Shifts(due=np.zeros((0, npts), dtype=np.bool_), sel=np.zeros((0, self.n_rows, self.n_cols)), cov=np.zeros((0, self.n_rows, self.n_cols)),
                                 to=np.zeros((0, self.n_cols), dtype=np.int64), rest=np.zeros((0, self.n_cols), dtype=np.int64))

    def numbers(self, Xs, xp=np, reps=None, keys=None):
        """
        The numbers in the program, from those in each engine's: each a dict of the arrays of its program (by the
        names of `ss.kernels.Program`), its tables (`fixed`, `mult`, `consts`, `rel_trans`; see `ss.solvers.Compartmental.tables()`),
        and its compartments (`y`), as `ss.gradients.capture()` gives them

        Every number is one of an engine's, so this is a gather, in NumPy or in JAX (where it is differentiated through).

        Args:
            Xs (list): the numbers of each engine
            xp (module): NumPy, or JAX's NumPy
            reps (list): for each engine, the rows of the program's tables for each of its own (3 if the program's have the start, middle, and end of each step, and the engine's one row a step)
            keys (list): the numbers to give (default: those in the first engine's)
        """
        reps = reps or [1]*len(Xs)
        keys = keys or list(Xs[0].keys())
        cat = lambda parts, axis=0: xp.concatenate(parts, axis=axis)
        out = {}
        for key in keys:
            if key in ['fixed', 'mult', 'consts', 'rel_trans']: # Tables, with a row for each step (or one for all of them)
                vals = [xp.repeat(X[key], r, axis=0) if r > 1 and len(X[key]) > 1 else X[key] for X,r in zip(Xs, reps)]
                T = max(len(v) for v in vals)
                parts = [v[:, I[:, None], C] for v,(I,C) in zip(vals, self.pieces[dict(fixed='flows', mult='flows').get(key, key)])]
                out[key] = cat([xp.broadcast_to(part, (T,) + part.shape[1:]) for part in parts], axis=1)
            elif key in ['inflow', 'tgt_w', 'crd_w']:
                out[key] = cat([X[key][I[:, None], C] for X,(I,C) in zip(Xs, self.pieces[dict(inflow='flows', tgt_w='tgt', crd_w='crd')[key]])])
            elif key == 'maps':
                out[key] = cat([X[key][I[:, None, None], C[:, :, None], C[:, None, :]] for X,(I,C) in zip(Xs, self.pieces.maps)])
            elif key == 'mov_rate':
                out[key] = cat([X[key][I] for X,I in zip(Xs, self.pieces.movers)])
            elif key == 'contacts': # By exposure, term, and group, or once for every exposure and group
                if self.full:
                    out[key] = cat([xp.swapaxes(xp.swapaxes(X[key], 1, 2)[I[:, None], C], 1, 2) for X,(I,C) in zip(Xs, self.pieces.contacts)])
                else:
                    out[key] = Xs[0][key]
            elif key in ['mat', 'sh_val']: # The mixing, the same for every engine
                out[key] = Xs[0][key]
            elif key == 'y':
                y = xp.zeros(self.n_rows*self.n_cols)
                for X,(J,E) in zip(Xs, self.pieces.y):
                    vals = X[key].reshape(-1)[E]
                    if xp is np:
                        y[J] = vals
                    else:
                        y = y.at[J].set(vals)
                out[key] = y.reshape(self.n_rows, self.n_cols)
            elif key.startswith('sh_'): # No deliveries: see no_shifts()
                R = self.n_cols
                out[key] = xp.zeros(dict(sh_due=(0, Xs[0]['sh_due'].shape[1]), sh_sel=(0, self.n_rows, R), sh_cov=(0, self.n_rows, R), sh_to=(0, R, R), sh_rest=(0, R, R))[key])
        return out

    def split(self, k, y, stocks, flows, xp=np):
        """ Engine k's compartments at the end, at each time point, and its flows over each step, from the program's (as `ss.kernels.run()` returns them) """
        L = self.layout[k]
        idx, fidx = np.maximum(L.idx, 0), np.maximum(L.fidx, 0)
        yk = xp.where(L.idx >= 0, y.reshape(-1)[idx], 0.0)
        sk = xp.where(L.idx >= 0, stocks.reshape(stocks.shape[0], -1)[:, idx], 0.0)
        fk = xp.where(L.fidx >= 0, flows.reshape(flows.shape[0], -1)[:, fidx], 0.0)
        return yk, sk, fk

    def engine_stocks(self, k, stocks, by_group=True):
        """ Engine k's number in each state from the number in each of the program's compartments: arrays whose last two dimensions are compartments by groups (or by one, the total) to states by the engine's groups (or one) """
        L = self.layout[k]
        if by_group:
            per_comp = np.where(L.idx >= 0, stocks.reshape(stocks.shape[:-2] + (-1,))[..., np.maximum(L.idx, 0)], 0.0)
            return np.einsum('...cg,sc->...sg', per_comp, L.comp_member)
        return np.einsum('...rg,sr->...sg', stocks, L.member)

    def engine_flows(self, k, flows, by_group=True):
        """ Engine k's flow along each transition and mover from the program's, as `engine_stocks()` does """
        L = self.layout[k]
        if by_group:
            return np.where(L.fidx >= 0, flows.reshape(flows.shape[:-2] + (-1,))[..., np.maximum(L.fidx, 0)], 0.0)
        return np.einsum('...fg,ef->...eg', flows, L.fmember)

    #%% Running it

    def per_step(self):
        """ The rows of the program's tables for each step, and how many of them each row of each engine's is (see `numbers()`) """
        per = [engine.per_step() for engine in self.engines]
        return max(per), [max(per)//p for p in per]

    def engine_numbers(self, args, tables=None):
        """ The numbers of an engine's run, from what `ss.solvers.Compartmental.run_args()` gives: see `numbers()` """
        y, program = args[0], args[1]
        X = {key:getattr(program, key) for key in ['inflow', 'contacts', 'tgt_w', 'crd_w', 'maps', 'mov_rate']}
        X['y'] = y
        if tables is not None:
            X.update(zip(['fixed', 'mult', 'consts', 'rel_trans'], tables))
        return X

    def run_args(self, tables=True):
        """ What the kernels need for a whole run of the program, as `ss.solvers.Compartmental.run_args()` gives it for one engine, and the rows of its tables for each step """
        setups = [engine.run_args(tables=tables) for engine in self.engines]
        progs = [args[1] for args,_ in setups]
        if self.program is None or self.built != [id(p) for p in progs]:
            self.build(progs)
        per_step, reps = self.per_step()
        J = self.numbers([self.engine_numbers(args, tabs) for args,tabs in setups], reps=reps)
        program = self.program._replace(**{key:J[key] for key in ['inflow', 'contacts', 'tgt_w', 'crd_w', 'maps', 'mov_rate']})
        args = [J['y']] + [program] + setups[0][0][2:]
        return args, ([np.ascontiguousarray(J[key]) for key in ['fixed', 'mult', 'consts', 'rel_trans']] if tables else None), per_step

    def tables(self, start=0, stop=None):
        """ The program's tables for some steps (see `ss.solvers.Compartmental.tables()`) """
        per_step, reps = self.per_step()
        J = self.numbers([dict(zip(['fixed', 'mult', 'consts', 'rel_trans'], engine.tables(start, stop))) for engine in self.engines], reps=reps)
        return [np.ascontiguousarray(J[key]) for key in ['fixed', 'mult', 'consts', 'rel_trans']]

    def run(self):
        """ A whole run of the diseases together, in one call to the kernels: returns, for each engine, what `ss.kernels.run()` does """
        args, tables, per_step = self.run_args()
        tally = np.zeros((args[3], 0, 2, self.n_cols))
        y, stocks, flows, short = ss.kernels.run(self.engines[0].make_rng(stream=1), *args, *tables, self.register, self.shifts, tally, per_step)
        self.y = y
        return [self.split(k, y, stocks, flows) + (short if k == 0 else 0.0,) for k in range(len(self.engines))]

    def whole_run(self, engine, use_jax=False):
        """ The output of a whole run for one engine: the first engine to ask takes the run for all of them """
        if not self.outs:
            outs = ss.gradients.run_crossing(self) if use_jax else self.run()
            self.outs = dict(zip(self.names, outs))
        return self.outs.pop(engine.mod.name)

    def step(self, engine):
        """
        An engine's part of a step taken in Python: it works out its rates (as `ss.solvers.Compartmental.step()` does), and once
        every disease in the crossing has, they take the step together, and each records its flows and its people
        """
        sim = self.sim
        solver = sim.integrator if sim.solver == 'ode' else sim.solver
        engine.cache.clear() # The interventions have run since the step started
        engine.update_targets()
        engine._mover_rates = engine.mover_rates()
        if engine.program is None:
            engine.init_program()
        ti = engine.mod.ti
        stale = [t for t,tables in self.pending.values() if t != ti]
        if stale:
            errormsg = f'The diseases {sc.strjoin(self.names)} are tracked jointly, so they step together, but {sc.strjoin(n for n in self.names if n not in self.pending)} did not step at time point {stale[0]} (does it define its own step() that does not call the one it replaces?)'
            raise RuntimeError(errormsg)
        self.pending[engine.mod.name] = (ti, engine.step_tables(engine.dt, smooth=solver != 'euler')) # Euler holds values that vary in time for the step
        if len(self.pending) < len(self.engines):
            return
        steps = [self.pending.pop(name)[1] for name in self.names]
        progs = [e.program for e in self.engines]
        if self.program is None or self.built != [id(p) for p in progs]:
            self.build(progs)
        p = self.program
        row = 0 if any(r >= 0 for tables,r in steps) else -1
        Xs = []
        for q,(tables,r) in zip(progs, steps): # The values at the start, middle, and end of the step, if any vary smoothly in time, and otherwise as they are now
            X = dict(zip(['fixed', 'mult', 'consts', 'rel_trans'], tables if r >= 0 else [q.fixed[None], q.mult[None], q.consts[None], q.rel_trans[None]]))
            X.update(contacts=q.contacts, mov_rate=q.mov_rate)
            Xs.append(X)
        J = self.numbers(Xs)
        p.contacts[:] = J['contacts']
        p.mov_rate[:] = J['mov_rate']
        if row < 0:
            p.fixed[:], p.mult[:], p.consts[:], p.rel_trans[:] = [J[key][0] for key in ['fixed', 'mult', 'consts', 'rel_trans']]
            tables = no_tables
        else:
            tables = tuple(np.ascontiguousarray(J[key]) for key in ['fixed', 'mult', 'consts', 'rel_trans'])
        first = self.engines[0]
        self.y, flows = ss.kernels.one_step(self.rng, self.y, p, first.dt, ss.kernels.solvers[solver], first.substeps(), tables, row)
        if sim.mixing.register is not None: # The events dated at the end of the step
            sim.get_declared(self.lead).engine.check_short(ss.kernels.apply_events(self.rng, self.y, self.register, first.mod.ti + 1, first.integer))
        if solver == 'gillespie' and p.local and p.has_det: # By place, the counts are settled after the events, as in SimInf
            ss.kernels.settle_places(self.y, p, first.dt, flows)
        for k,e in enumerate(self.engines):
            L = self.layout[k]
            e.y = self.gather(e)
            e.record_flows(np.where(L.fidx >= 0, flows.ravel()[np.maximum(L.fidx, 0)], 0.0))
        return

    def run_many(self, seeds):
        """ Many runs that differ only in their seed, as `ss.solvers.Compartmental.run_many()` takes them for one engine: returns, for each engine, its name and its results """
        engines = self.engines
        for engine in engines:
            engine.restart()
        self.init()
        args, _, per_step = self.run_args(tables=False)
        y0, program, dt, npts, solver, n_sub = args
        rngs = nb.typed.List([engines[0].make_rng(stream=1, seed=seed) for seed in seeds]) # The random numbers that each run would have
        R, nc, nflows, n_runs = self.n_cols, self.n_rows, self.n_trans + self.n_movers, len(seeds)
        grouped = self.sim.groups.n_groups > 1 and any(s or f for s,f in [engine.group_needs() for engine in engines])
        rows = group_ti = None
        if grouped: # Every compartment and flow of the program by group, for the results that each engine keeps by group
            rows, group_ti = self.sim.group_rows(npts, n_runs*(nc + nflows)*R*8, engines[0].max_bytes, name=' × '.join(self.names))
            rows = np.arange(npts) if rows is None else rows
        n_rows = 0 if rows is None else rows[-1] + 1
        stocks, flows = np.zeros((n_runs, npts, nc, 1)), np.zeros((n_runs, npts, nflows, 1))
        stocks_g, flows_g = np.zeros((n_runs, n_rows, nc if grouped else 0, R)), np.zeros((n_runs, n_rows, nflows if grouped else 0, R))
        tally, tally_g = np.zeros((n_runs, npts, 0, 2, 1)), np.zeros((n_runs, n_rows, 0, 2, R))
        grp_state, grp_flow = (np.arange(nc), np.arange(nflows)) if grouped else (np.full(nc, -1), np.full(nflows, -1))
        ys = np.repeat(y0[None], n_runs, axis=0)
        length = max(int(50e6//((len(program.fixed) + len(program.mult) + len(program.consts) + len(program.rel_trans))*per_step*R*8)), 1) # Steps per part, as in ss.solvers.Compartmental.run_many()
        start = 0
        while True:
            end = min(start + length, npts - 1)
            ss.kernels.run_many(rngs, ys, program, dt, start, end + 1, solver, n_sub, *self.tables(start, end), self.register, self.shifts, stocks, flows, stocks_g, flows_g,
                                tally, tally_g, np.arange(nc), np.zeros(0, dtype=int) if rows is None else rows, grp_state, grp_flow, per_step)
            if end >= npts - 1:
                break
            start = end
        outs = []
        for k,engine in enumerate(engines):
            states = range(len(engine.graph.states))
            by_group = (self.engine_stocks(k, stocks_g), self.engine_flows(k, flows_g), states, range(engine.n_trans + len(engine.movers)), group_ti) if grouped else None
            out = engine.collect(self.engine_stocks(k, stocks, by_group=False), self.engine_flows(k, flows, by_group=False), grouped=by_group, seeds=seeds)
            out.deliveries = sc.objdict()
            outs.append((engine.mod.name, out))
        return outs
