"""
The rounds of moves of a declared disease as agents, compiled: which diseases can take them, the tables they take, and the
Python around the kernel (`ss.perflib.rounds`), which records the moves as the agent engine does
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import Bunch

__all__ = ['Lowered']


class Lowered:
    """
    The rounds of moves of a disease as agents, lowered to tables for the compiled kernel (`ss.perflib.rounds`)

    Made at the disease's first step, if every scheduled exit, branch, split, and trigger it has is one the kernel takes (see `reason()`):
    dwell times that are constant, exponential, or lognormal; triggers on the disease's own transitions, after a delay; and values that are
    numbers, values in time, or values by a property of people (`ss.by(sex=...)`) or by the sim's groups, with `where=` by a property of people;
    in a stratified sim too, and with modifiers of transmission only (of `rel_sus` or `rel_trans`, or of a parameter no table reads). A value
    that differs between people is a table by class of agent (e.g. one row for women and one for men, or one for each group), worked out by the
    Python engine for one agent of each class, so the kernel uses the numbers the Python engine would. The kernel writes the engine's own arrays (who is in each
    state, the times of entering, the scheduled exits), and `replay()` records the moves as the Python engine would (events, batches, flows).
    """
    def __init__(self, engine):
        self.engine = engine
        graph = engine.graph
        people = graph.people_states
        index = engine.state_index
        trs = list(graph.transitions.values())
        t_index = {tr.name:i for i,tr in enumerate(trs)}
        self.trs = trs
        self.t_index = t_index
        self.triggers = list(engine.triggers)
        self.names = [s.name for s in people]
        streams = self.streams = {} # The engine's distributions the kernel draws from, by key
        stream = lambda key: streams.setdefault(key, len(streams))
        S, T, K = len(people), len(trs), len(self.triggers)

        # The states and their scheduled exits
        mod = engine.mod
        self.kept = np.array([not isinstance(mod[f'ti_{s.name}'], ss.arrays.Unkept) for s in people]) # Whether the time of entering is kept
        self.sched = np.array([s.name in engine.scheduled for s in people])
        ex = [(s, i, tr) for s in people for i,tr in enumerate(engine.exits_of[s.name])]
        self.ex_off = np.cumsum([0] + [len(engine.exits_of[s.name]) for s in people]).astype(np.int64)
        self.ex_tr = np.array([t_index[tr.name] for s,i,tr in ex], dtype=np.int64)
        self.n_br = np.array([len(engine.branches_of(s.name)) for s in people], dtype=np.int64)
        self.ex_after = np.array([engine.after_of(tr) for s,i,tr in ex], dtype=float)
        self.ex_loop = np.concatenate([engine.loops_of[s.name] for s in people]).astype(bool) if ex else np.zeros(0, dtype=bool)
        self.ex_resc = np.array([i in engine.rescaled_of(s.name) for s,i,tr in ex], dtype=bool)
        self.has_mult = np.array([bool(engine.rescaled_of(s.name)) and f'mult_{s.name}' in mod.__dict__ for s in people]) # The states that keep how fast each exit ran when scheduled (see rescale_exits())
        self.rescale = [s.name for s in people if any(engine.varies(engine.dwells[engine.exits_of[s.name][i].name]) for i in engine.rescaled_of(s.name))] # Those whose dwell times can change after they are drawn (a value in time), whose exits are rescaled at each step (see rescale_exits())

        # The transitions with several destinations, and the triggers
        split = [tr for tr in trs if tr.split]
        width = max([len(tr.dst) for tr in split] + [1])
        self.tr_dst0 = np.array([index[tr.dst[0].name] if tr.dst and not tr.split and tr.dst[0].people else -1 for tr in trs], dtype=np.int64)
        self.sp_n = np.array([len(tr.dst) if tr.split else 0 for tr in trs], dtype=np.int64)
        self.sp_dst = np.full((T, width), -1, dtype=np.int64)
        for tr in split:
            self.sp_dst[t_index[tr.name], :len(tr.dst)] = [index[d.name] for d in tr.dst]
        self.sp_stays = np.array([bool(tr.stays) for tr in trs])
        self.tr_trigger = np.array([tr.trigger is not None for tr in trs])
        self.tg_tr = np.array([t_index[tr.name] for tr in self.triggers], dtype=np.int64)
        self.tg_event = np.array([t_index[tr.trigger.x.spec()['tr']] for tr in self.triggers], dtype=np.int64)
        self.tg_src = np.array([[any(src.name == s.name for src in tr.src) for s in people] for tr in self.triggers], dtype=bool).reshape(K, S)
        self.tg_delay = np.array([tr.name in engine.delays for tr in self.triggers])

        # The streams of draws
        self.br_stream = np.array([stream(f'branch_{s.name}') for s in people], dtype=np.int64)
        self.round_stream = np.array([stream(f'round_{s.name}') for s in people], dtype=np.int64)
        self.dw_stream = np.array([stream(f'dwell_{tr.name}') if tr.name in engine.dwells else 0 for tr in trs], dtype=np.int64)
        self.dst_stream = np.array([stream(f'dst_{tr.name}') for tr in trs], dtype=np.int64)
        self.after_stream = np.array([stream(f'after_{tr.name}') if tr.name in engine.delays else 0 for tr in self.triggers], dtype=np.int64)
        self.round_after_stream = np.array([stream(f'round_after_{tr.name}') if tr.name in engine.delays else 0 for tr in self.triggers], dtype=np.int64)
        self.seeds = np.array([engine.dists[key].seed for key in streams], dtype=np.int64)

        # The tables by class, filled as classes of agents appear (see fill())
        self.factors = self.class_factors()
        self.n_classes = C = int(np.prod([n for n,_,_ in self.factors])) if self.factors else 1
        self.static_cls = all((dim in ['sex', 'female']) if dim is not None else all(leaf.args == ('female',) for leaf in self.people_leaves(obj)) for _,obj,dim in self.factors) # Classes by sex do not change
        self.cls = None # The class of each agent, kept while it cannot change: see classes()
        self.cls_used = None
        self.have = np.zeros(C, dtype=bool) # The classes the tables have rows for
        self.br_off = np.cumsum([0] + [C*(n + 1) for n in self.n_br[:-1]]).astype(np.int64)
        self.br_edges = np.zeros(int(sum(C*(n + 1) for n in self.n_br))) # Where each branch starts, as a cumulative probability, by state and class
        self.adj = np.ones((T, C)) # The multiplier on each transition's rate (where= of a timed exit), which divides its dwell times
        self.dw_kind = np.full(T, -1, dtype=np.int64)
        self.dw_pars = np.zeros((T, C, 3))
        self.sp_edges = np.zeros((T, C, width + 1))
        self.dl_kind = np.zeros(K, dtype=np.int64)
        self.dl_pars = np.zeros((K, C, 3))
        self.tg_where = np.ones((K, C), dtype=bool)
        self.dynamic = any(self.table_parts(engine, v)[1] for v in self.table_values(engine)) # Whether a value can change in time, so the tables are worked out at each step
        self.ti = None # The step the tables were worked out for
        self.checked = None # The number of agents whose classes the tables were checked for: see fill()
        self.arrs = None # The arrays the kernel reads and writes, and as it takes them: see run()
        self.tuples = None
        return

    @classmethod
    def reason(cls, engine):
        """ Why the moves of a disease cannot run compiled, or None if they can """
        sim = engine.sim
        mod = engine.mod
        graph = engine.graph
        rounds = ss.perflib.rounds
        if not ss.options.crn:
            return 'without common random numbers'
        if len(graph.people_states) > rounds.MAX_STATES or len(engine.triggers) > rounds.MAX_TRIGGERS:
            return f'with more than {rounds.MAX_STATES} states or {rounds.MAX_TRIGGERS} triggers'
        if sim.people.scale.weighted:
            return 'with agents of different weights'
        if engine.strain is not None or engine.memory or engine.sets or engine.on_event or engine.derived or engine.lifetables or engine.late or engine.entered_dims or engine.also:
            return 'with strains, clocks of sets, attributes set at events, derived attributes, life tables, competing rates on entry, levels taken on entry, or others an event happens to'
        if any(engine.competing.values()):
            return 'with competing rates'
        if engine.subscribers or mod.__dict__.get('remembered'):
            return 'with modules that react to its events, or events it remembers'
        if not all(cls.unseen_mod(engine, key, m) for key,m in cls.modifiers_of(engine, keys=True)) or engine.branch_adjust or any(engine.adjust[tr.name] is not None for tr in graph.transitions.values() if not tr.contact):
            return 'with modifiers of what it moves (other than those of where=), or adjust= of them'
        per = [(ss.conditions.modname(tr.per.module) or m.name) for m in sim.declared for tr in m.graph.inflows if getattr(tr, 'per', None) is not None]
        if mod.name in per or any(tr.inflow and getattr(tr, 'per', None) is not None for tr in graph.transitions.values()):
            return 'with births driven by its flows, or flows that drive births'
        for tr in graph.transitions.values():
            if engine.moves(tr) or any(engine.enters(tr, src) for src in tr.src): # Including an infection, whose arrivals the kernel sends (see run())
                return f'"{tr.name}" changes a level (move= or increment=), or takes people into a dimension'
            if tr.contact or tr.inflow:
                continue
            if tr.outflow or tr.dies or engine.moves(tr) or getattr(tr, 'before', None) is not None or any(not d.people for d in tr.dst):
                return f'"{tr.name}" ends in a death, a change of level, or a count, or has a limit (before=)'
        for tr in engine.triggers:
            event = tr.trigger.x
            spec = event.spec() if tr.trigger.event else None
            if not tr.trigger.event or ss.conditions.modname(event.module) not in [None, mod.name] or isinstance(event, ss.conditions.child) or spec != dict(tr=spec.get('tr')) or spec['tr'] not in graph.transitions:
                return f'"{tr.name}" is triggered by something other than a transition of "{mod.name}"'
            if tr.where is not None and not cls.people_cond(tr.where):
                return f'"{tr.name}" has a where= that is not a property of people'
        for name in engine.scheduled:
            for tr in engine.branches_of(name):
                if getattr(tr, 'where', None) is not None and not cls.people_cond(tr.where):
                    return f'"{tr.name}" has a where= that is not a property of people'
        for name,dw in engine.dwells.items():
            if cls.dwell_kind(dw) is None or getattr(dw, 'mods', None) or dw.scale is not None:
                return f'the dwell time of "{name}" is of a kind the compiled rounds do not draw'
        for name,delay in engine.delays.items():
            if isinstance(ss.transitions.by_dist(getattr(delay, 'raw', None)), ss.Dist) and cls.dist_kind(engine.dists[f'after_{name}']) is None:
                return f'the delay of "{name}" is of a kind the compiled rounds do not draw'
        for value in cls.table_values(engine):
            if cls.table_parts(engine, value) is None:
                return f'{value} is not a number, a value in time, or a value by a property of people'
        widths = [{mod[f'ti_{s.name}'].raw.dtype for s in graph.people_states if not isinstance(mod[f'ti_{s.name}'], ss.arrays.Unkept)},
                  {mod[f'ti_exit_{name}'].raw.dtype for name in engine.scheduled}, {mod[f'ti_after_{name}'].raw.dtype for name in engine.delays}]
        if not engine.scheduled or any(len(w) > 1 for w in widths) or any(engine.dists[key].hash_dtype is not None for key in engine.dists.keys() if key.startswith(('branch_', 'round_', 'dwell_', 'dst_', 'after_'))):
            return 'with no scheduled exits, step numbers of different widths, or draws of another precision'
        return None

    #%% What the tables are made of

    @staticmethod
    def people_cond(cond):
        """ Whether a condition is a property of people (e.g. `ss.ppl.female`), or made of them (`~`, `&`, `|`), which is the same for each agent all step """
        if not isinstance(cond, ss.conditions.Cond):
            return False
        if cond.op == 'people':
            return True
        return cond.op in ['~', '&', '|'] and all(Lowered.people_cond(arg) for arg in cond.args)

    @staticmethod
    def people_leaves(cond):
        """ The properties of people a condition of them is made of (see `people_cond()`) """
        return [cond] if cond.op == 'people' else [leaf for arg in cond.args for leaf in Lowered.people_leaves(arg)]

    @staticmethod
    def modifiers_of(engine, keys=False):
        """ The modifiers of a disease's parameters and transitions (with what each modifies, as (module, kind, name), if `keys`) """
        return [((key, m) if keys else m) for key,mods in engine.sim.modifiers.items() if key[0] == engine.mod.name for m in mods]

    @staticmethod
    def unseen_mod(engine, key, m):
        """ Whether a modifier leaves the kernel's moves alone: one that a where= stands for (a table), or one of transmission only (rel_trans, a route, a contact-mediated transition, or a parameter that no table reads; rel_sus too, unless an acquisition is scheduled) """
        _, kind, name = key
        graph = engine.graph
        if Lowered.where_mod(engine, m) or kind in ['rel_trans', 'route']:
            return True
        if kind == 'rel_sus':
            return not any(tr.acquisition and not tr.contact for tr in graph.transitions.values())
        if kind == 'transition':
            return name in graph.transitions and graph.transitions[name].contact
        if kind == 'par':
            return name not in Lowered.par_names(engine)
        return False

    @staticmethod
    def par_names(engine):
        """ The names of the parameters the tables read (see `table_values()`) """
        E = ss.expressions
        out = set()
        def add(value):
            for leaf in (value.leaves if isinstance(value, E.Expr) else [value]):
                if isinstance(leaf, E.par):
                    out.add(leaf.name)
                    raws = list(leaf.raw.pars.values()) if isinstance(leaf.raw, ss.Dist) else [leaf.raw]
                    for raw in raws:
                        if isinstance(raw, E.Expr):
                            add(raw)
        for value in Lowered.table_values(engine):
            add(value)
        return out

    @staticmethod
    def where_mod(engine, m):
        """ Whether a modifier is one that a transition's where= stands for (see `ss.solvers.choose.init_where()`), for people by their properties: a multiplier by class, the same all run """
        return m.name.startswith(f'where_{engine.mod.name}_') and m.op == 'multiply' and sc.isnumber(m.amount) and Lowered.people_cond(m.where)

    @staticmethod
    def table_values(engine):
        """ The values the kernel takes as tables by class: the shares of the branches and splits, and the parameters of the dwell times and delays """
        out = []
        for name in engine.scheduled:
            for tr in engine.branches_of(name):
                if tr.p is not None and not tr.split:
                    out.append(tr.p[0])
        for tr in engine.graph.transitions.values():
            if tr.split:
                out += [p for p in list(tr.p) + [tr.p_stay, tr.p_die] if p is not None]
        out += [dw.par for dw in engine.dwells.values()]
        out += list(engine.delays.values())
        return out

    @staticmethod
    def table_parts(engine, value):
        """ For a value the kernel takes as a table by class: the values by a property of people (`ss.by`) it reads, and whether it can change in time; None if it is not one """
        E = ss.expressions
        bys = []
        dynamic = False
        for leaf in (value.leaves if isinstance(value, E.Expr) else [value]):
            if not isinstance(leaf, E.par):
                if sc.isnumber(leaf):
                    continue
                return None
            if id(leaf) in engine.spread:
                return None
            raws = list(leaf.raw.pars.values()) if isinstance(leaf.raw, ss.Dist) else [leaf.raw]
            for raw in raws:
                if raw is None or sc.isnumber(raw) or isinstance(raw, (ss.TimePar, str, np.ndarray)):
                    continue
                elif isinstance(raw, ss.transitions.TimeValue):
                    dynamic = True
                elif type(raw) is ss.by and engine.derived_strat(raw) is None and 'year' not in raw.dims and raw.table is None and (raw.on_people(engine.sim) or set(raw.dims) <= {strat.name for strat in engine.sim.groups.strats}):
                    bys.append(raw) # By a property of people, or by the sim's groups (a factor of the class: see class_factors())
                elif isinstance(raw, E.Expr):
                    inner = Lowered.table_parts(engine, raw)
                    if inner is None:
                        return None
                    bys += inner[0]
                    dynamic = dynamic or inner[1]
                else:
                    return None
        return bys, dynamic

    @staticmethod
    def dist_kind(dist):
        """ The kind of a distribution the kernel draws (see `ss.perflib.rounds`), or None """
        rounds = ss.perflib.rounds
        return {ss.constant:rounds.CONSTANT, ss.lognorm_ex:rounds.LOGNORMAL, ss.lognorm_im:rounds.LOGNORMAL, ss.expon:rounds.EXPON}.get(type(dist))

    @staticmethod
    def dwell_kind(dw):
        """ The kind of a dwell time the kernel draws, or None: a distribution of one of the kinds of `dist_kind()`, or a duration or rate (exponential) """
        if dw.dist is None:
            return ss.perflib.rounds.EXPON if dw.k == 1 and dw.family == 'exponential' else None
        return Lowered.dist_kind(dw.dist)

    def class_factors(self):
        """ What the class of an agent is made of: each property of people that a value by class reads (`ss.by`) or a where= is, as (number of levels, the ss.by or condition, the dimension) """
        engine = self.engine
        out = {}
        for value in self.table_values(engine):
            for by in self.table_parts(engine, value)[0]:
                if not by.on_people(engine.sim): # By the sim's groups
                    out.setdefault(('group',), (engine.sim.groups.n_groups, None, 'group'))
                    continue
                for dim in by.dims:
                    n, _ = by.positions(engine.sim, dim)
                    out.setdefault(('by', dim, by.keys_id(dim)), (n + 1, by, dim))
        wheres = [tr.where for tr in self.triggers if tr.where is not None] + [tr.where for name in engine.scheduled for tr in engine.branches_of(name) if getattr(tr, 'where', None) is not None]
        wheres += [m.where for m in self.modifiers_of(engine) if self.where_mod(engine, m)] # The others leave the moves alone (see unseen_mod())
        for cond in wheres:
            out.setdefault(('where', repr(cond)), (2, cond, None))
        return list(out.values())

    def classes(self):
        """ The class of each agent (by UID); kept from the last call while the factors are properties that do not change (sex) and no agents have been made """
        engine = self.engine
        n = len(engine.mod.rel_sus.raw)
        used = engine.sim.people.uid.len_used
        if self.cls is not None and len(self.cls) == n and self.cls_used == used:
            return self.cls
        cls = np.zeros(n, dtype=np.int64)
        for levels,obj,dim in self.factors:
            level = engine.sim.groups.group_raw() if dim == 'group' else obj.positions(engine.sim, dim)[1] if dim is not None else engine.selector.select(obj) # The group, a position along a dimension of an ss.by, or whether a where= holds
            cls = cls*levels + np.broadcast_to(np.asarray(level, dtype=np.int64), n)
        if self.static_cls:
            self.cls, self.cls_used = cls, used
        return cls

    def dist_pars(self, kind, dist, uids):
        """ The parameters of a distribution as the kernel draws it (see `ss.perflib.rounds.dwell()`), for some agents, as the distribution works them out for them """
        rounds = ss.perflib.rounds
        dist.process_size(uids)
        dist.process_pars()
        kw = dist.dist.kwds if dist.dist is not None else {}
        if kind == rounds.CONSTANT:
            cols = [dist._pars.v, 0.0, 0.0]
        elif kind == rounds.LOGNORMAL:
            cols = [kw['s'], kw['scale'], kw['loc']]
        else:
            cols = [kw.get('scale', 1.0), kw.get('loc', 0.0), 0.0]
        dist._n = dist._uids = dist._slots = None
        return np.stack([np.broadcast_to(np.asarray(c, dtype=float), len(uids)) for c in cols], axis=1)

    def fill(self, cls):
        """ The rows of the tables for the classes of the agents alive now that have none (or all of them, at each step, if a value can change in time), worked out by the Python engine for one agent of each """
        engine = self.engine
        people = engine.sim.people
        used = people.uid.len_used
        if not self.dynamic and self.static_cls and self.checked == used: # No agents since the last check, classes that do not change (by sex), and values that do not change in time
            return
        self.checked = used
        alive = people.auids
        present = np.zeros(self.n_classes, dtype=bool)
        present[cls[alive]] = True
        need = present if self.dynamic and self.ti != engine.ti else present & ~self.have
        if not need.any():
            return
        wanted = np.flatnonzero(need)
        first = np.full(self.n_classes, -1, dtype=np.int64)
        first[cls[alive][::-1]] = alive[::-1] # The first agent alive of each class
        reps = ss.uids(first[wanted])
        for i,s in enumerate(engine.graph.people_states):
            nb = self.n_br[i]
            if nb:
                edges = engine.branch_edges(s, reps) # nb + 1 rows
                for j,c in enumerate(wanted):
                    self.br_edges[self.br_off[i] + c*(nb + 1):self.br_off[i] + (c + 1)*(nb + 1)] = edges[:, j]
        for t,tr in enumerate(self.trs):
            if tr.name in engine.dwells:
                self.adj[t, wanted] = np.broadcast_to(np.asarray(engine.adjusted(tr.name, reps), dtype=float), len(reps))
                dw = engine.dwells[tr.name]
                self.dw_kind[t] = self.dwell_kind(dw)
                if dw.dist is None: # A unit exponential, times the mean
                    with engine.evaluating(uids=reps, for_uids=True):
                        mean = engine.dwell_mean(tr.name)
                    mean = np.asarray(mean if np.ndim(mean) == 0 or len(mean) == len(reps) else np.asarray(mean)[reps], dtype=float)
                    self.dw_pars[t, wanted] = 0.0
                    self.dw_pars[t, wanted, 0] = np.broadcast_to(mean, len(reps))
                else:
                    self.dw_pars[t, wanted] = self.dist_pars(self.dw_kind[t], engine.dists[f'dwell_{tr.name}'], reps)
            if tr.split:
                edges = engine.dest_edges(tr, reps, np.full(len(reps), engine.state_index[tr.src[0].name]))
                self.sp_edges[t, wanted, :edges.shape[0]] = edges.T
        for k,tr in enumerate(self.triggers):
            if tr.name in engine.delays:
                if isinstance(ss.transitions.by_dist(getattr(engine.delays[tr.name], 'raw', None)), ss.Dist):
                    dist = engine.dists[f'after_{tr.name}']
                    self.dl_kind[k] = self.dist_kind(dist)
                    self.dl_pars[k, wanted] = self.dist_pars(self.dl_kind[k], dist, reps)
                else: # An exponential with this mean
                    self.dl_kind[k] = ss.perflib.rounds.EXPON
                    self.dl_pars[k, wanted] = 0.0
                    self.dl_pars[k, wanted, 0] = np.broadcast_to(np.asarray(engine.eval_for(engine.delays[tr.name], reps), dtype=float), len(reps))
            if tr.where is not None:
                self.tg_where[k, wanted] = np.asarray(engine.selector.select(tr.where), dtype=bool)[reps]
        self.have |= need
        self.ti = engine.ti
        return

    #%% Running the rounds

    def take(self):
        """ The engine's arrays as the kernel takes them, in tuples of a few lengths (so Numba compiles the kernel a few times at most), kept while the population's arrays are the same """
        engine = self.engine
        mod = engine.mod
        people = engine.graph.people_states
        if self.arrs is None: # The arrays, found once (their values, .raw, are new when the population grows)
            pick = lambda key, use: mod[key] if use else None
            self.arrs = Bunch(members=[mod[s.name] for s in people], ti_in=[pick(f'ti_{s.name}', self.kept[i]) for i,s in enumerate(people)],
                              shifts=[pick(f'shift_{s.name}', self.kept[i]) for i,s in enumerate(people)], ti_exit=[pick(f'ti_exit_{s.name}', self.sched[i]) for i,s in enumerate(people)],
                              exit_idx=[pick(f'exit_{s.name}', self.sched[i]) for i,s in enumerate(people)], mults=[pick(f'mult_{s.name}', self.has_mult[i]) for i,s in enumerate(people)],
                              after=[pick(f'ti_after_{tr.name}', tr.name in engine.delays) for tr in self.triggers])
            first = lambda arrs, other: next((a for a in arrs if a is not None), other) # One of each kind (all of one width), or a stand-in
            self.arrs.xref = mod[f'ti_exit_{engine.scheduled[0]}']
            self.arrs.ref = first(self.arrs.ti_in, self.arrs.xref)
            self.arrs.aref = first(self.arrs.after, self.arrs.ref)
        arrs = self.arrs
        key = (len(mod.rel_sus.raw), id(arrs.members[0].raw), id(arrs.ref.raw), id(arrs.xref.raw))
        if self.tuples is None or self.tuples[0] != key:
            fills = Bunch(b=np.zeros(1, dtype=bool), i8=np.zeros(1, dtype=np.int8), f=np.zeros(1, dtype=ss.dtypes.float), ti=np.zeros(1, dtype=arrs.ref.raw.dtype),
                          x=np.zeros(1, dtype=arrs.xref.raw.dtype), a=np.zeros(1, dtype=arrs.aref.raw.dtype))
            MS = next(n for n in [8, 16, ss.perflib.rounds.MAX_STATES] if n >= len(people))
            MK = next(n for n in [4, 8, ss.perflib.rounds.MAX_TRIGGERS] if n >= len(self.triggers))
            pad = lambda arrs, fill, n: tuple(fill if a is None else a.raw for a in arrs) + (fill,)*(n - len(arrs))
            tuples = (pad(arrs.members, fills.b, MS), pad(arrs.ti_in, fills.ti, MS), pad(arrs.shifts, fills.i8, MS), pad(arrs.ti_exit, fills.x, MS),
                      pad(arrs.exit_idx, fills.i8, MS), pad(arrs.after, fills.a, MK), pad(arrs.mults, fills.f, MS))
            sentinels = tuple(int(v) for a in [arrs.ref, arrs.xref, arrs.aref] for v in [a.nan, a.lowest, a.never]) # Not set, the lowest step, and after the run, as stored
            self.tuples = (key, tuples, sentinels)
        return self.tuples[1], self.tuples[2]

    def work(self, N):
        """ The kernel's work arrays by UID, kept between calls and made again only when the population grows: who a trigger took, by the index of the round's draws (see `ss.perflib.rounds.run()`); each one's exit as they left; the last round each was put back (-1); and the agents of each delayed trigger """
        if getattr(self, '_work', None) is None or len(self._work[0]) != N:
            n_trig = len(self.triggers)
            self._work = (np.full(N, -1, dtype=np.int64), np.empty(N, dtype=np.int64), np.full(N, -1, dtype=np.int64), np.empty((n_trig, N if self.tg_delay.any() else 0), dtype=np.int64))
        return self._work

    def run(self, phase, ti, members=None):
        """
        The rounds of moves of a phase of a step, compiled (see `ss.perflib.rounds.run()`): phase 1, from the scheduled exits due at the start
        of the step (in place of `Agents.leave()` and `Agents.finish_moves()`), with `members` the agents in each state then; phase 2, from the
        moves of the infections (in place of `finish_moves()`); then the moves are recorded as the Python engine records them (see `replay()`)
        """
        engine = self.engine
        mod = engine.mod
        people = engine.graph.people_states
        S = len(people)
        cls = self.classes()
        self.fill(cls)
        (members_t, ti_in, shifts, ti_exit, exit_idx, after_due, mults), sentinels = self.take()
        slots = engine.sim.people.slot_raw()
        use_slots = slots is not None
        slots = np.asarray(slots, dtype=np.int64) if use_slots else np.zeros(1, dtype=np.int64)
        empty = np.zeros(0, dtype=np.int64)
        if phase == 1: # The agents in each state at the start of the step
            pro = [np.asarray(members[s.name], dtype=np.int64) if self.sched[i] and s.name in members else empty for i,s in enumerate(people)]
            pro_off = np.cumsum([0] + [len(p) for p in pro]).astype(np.int64)
            pro_uids = np.concatenate(pro)
            q_uid = q_tr = q_src = empty
        else: # The moves of the infections
            pro_off = np.zeros(S + 1, dtype=np.int64)
            pro_uids = empty
            moved, engine._moved = engine._moved, []
            q_uid = np.concatenate([np.asarray(u, dtype=np.int64) for tr,src,u in moved]) if moved else empty
            q_tr = np.concatenate([np.full(len(u), self.t_index[tr.name]) for tr,src,u in moved]) if moved else empty
            q_src = np.concatenate([np.full(len(u), engine.state_index[src.name]) for tr,src,u in moved]) if moved else empty
        ev0 = [np.unique(np.asarray(engine.new_events(tr), dtype=np.int64)) for tr in self.triggers] # What the triggers have not seen yet (see Agents.triggered())
        ev0_off = np.cumsum([0] + [len(e) for e in ev0]).astype(np.int64)
        ev0 = np.concatenate(ev0) if ev0 else empty
        step, rnd = engine.round
        loop_ti = mod.t.ti
        if step != loop_ti:
            rnd = -1 # As new_round() would start this step's
        base = ss.solvers.agents.ROUNDS + loop_ti*ss.solvers.agents.ROUND_MAX # The index of the draws, less the round (see Agents.draw())
        max_depth = (S + 1)*(2 if phase == 1 else 1) # As finish_moves()
        shift, scale = ss.perflib.draws.crn_bits(ss.dtypes.float)
        out = ss.perflib.rounds.run(phase, float(ti), float(engine.ti), int(engine.index_of(ti) - ti), base, rnd, max_depth, float(engine.dt), S, len(self.trs),
                                    q_uid, q_tr, q_src, len(q_uid), pro_off, pro_uids,
                                    members_t, ti_in, shifts, self.kept, ti_exit, exit_idx, self.sched, after_due, mults, self.has_mult, cls, slots, use_slots,
                                    *sentinels,
                                    self.ex_off, self.ex_tr, self.ex_after, self.ex_loop, self.ex_resc, self.n_br, self.br_edges, self.br_off, self.dw_kind, self.dw_pars, self.adj,
                                    self.tr_dst0, self.sp_n, self.sp_dst, self.sp_edges, self.sp_stays, self.tr_trigger,
                                    self.tg_tr, self.tg_event, self.tg_src, self.tg_delay, self.dl_kind, self.dl_pars, self.tg_where, ev0_off, ev0,
                                    self.seeds, self.br_stream, self.round_stream, self.dw_stream, self.dst_stream, self.after_stream, self.round_after_stream, shift, scale,
                                    *self.work(len(mod.rel_sus.raw)))
        *summary, rnd, err = out
        engine.round = (loop_ti, int(rnd))
        if err:
            raise ss.errors.TriggerLoop(f'moves with no wait out of the states of "{mod.name}" did not settle within a step.\n     Give one of them a rate or a duration, or change the branches so they do not lead back; or use a smaller dt.')
        self.replay(*summary)
        for tr in self.triggers: # They have seen everything recorded up to their last look, in the last round
            engine._seen[tr.name] = engine.n_batches
        if engine._news:
            engine._news = False
            engine.notify(ti)
        return

    def collapsed(self):
        """ The group each group is counted in while in each state (by state and group: see `Agents.groups()`), made once """
        if getattr(self, '_collapsed', None) is None:
            self._collapsed = np.stack([np.asarray(self.engine.collapse[name], dtype=np.int64) for name in self.names])
        return self._collapsed

    def replay(self, flow, ev_uid, ev_off, st_uid, st_off, dlive, b_start, b_end, ar_uid, ar_tr, ar_src, ar_dst, mv_uid, mv_src):
        """
        Record the moves as the Python engine records them (as `Agents.move()`, `stay()` and `record()`; see `ss.perflib.rounds.summarize()`):
        the flow and the events of each transition, without those put back; the number in each state (by group, if there are groups); and the batches of arrivals
        """
        engine = self.engine
        names = self.names
        for t in np.flatnonzero(flow | np.diff(ev_off) | np.diff(st_off)).tolist():
            tr = self.trs[t]
            engine.flowcounts[tr.name] += int(flow[t])
            uids = ss.uids(ev_uid[ev_off[t]:ev_off[t+1]])
            events = engine.events.get(tr.name)
            if events is not None and st_off[t+1] > st_off[t]: # Those put back are no longer its event (see Agents.stay())
                events = events[~np.isin(events, st_uid[st_off[t]:st_off[t+1]])]
            engine.events[tr.name] = uids if events is None else ss.uids.concatenate([events, uids])
        ng = engine.sim.groups.n_groups
        groups = engine.sim.groups.group_raw() if ng > 1 else None
        live = engine.live
        if live is not None and ng > 1: # By group: each arrival (or one put back) in the state they are now in, less each move out (see Agents.wrote())
            cell = lambda states, uids: np.bincount(states*ng + groups[uids], minlength=len(names)*ng)
            delta = (cell(np.where(ar_dst == ss.perflib.rounds.STAY, ar_src, ar_dst), ar_uid) - cell(mv_src, mv_uid)).reshape(len(names), ng)
            for i in np.flatnonzero(delta.any(axis=1)).tolist():
                live[names[i]] = live[names[i]] + delta[i]
        elif live is not None:
            for i in np.flatnonzero(dlive).tolist():
                live[names[i]] = live[names[i]] + int(dlive[i])
        if len(b_start):
            ti = engine.ti
            key = (ti, engine._phase[1] if engine._phase[0] == ti else -1)
            for i,j in zip(b_start.tolist(), b_end.tolist()):
                tr = self.trs[ar_tr[i]]
                d = int(ar_dst[i])
                engine.n_batches += 1
                engine.batches.append(Bunch(tr=tr.name, uids=ss.uids(ar_uid[i:j]), src=ar_src[i:j], dst=d, key=key, also=False, n=engine.n_batches))
                if tr.ends:
                    engine.flowcounts[f'{tr.name}.{names[d]}'] += j - i
                if ng > 1: # By the group each left, as Agents.move() and destinations() record them (an infection's flow is recorded by move() already)
                    uids = ar_uid[i:j]
                    left = self.collapsed()[ar_src[i:j], groups[uids]] # The group each is counted in, in the state they left (see Agents.groups())
                    if not tr.contact:
                        engine.add_flow_groups(tr.name, left, uids)
                    if tr.ends:
                        engine.add_flow_groups(f'{tr.name}.{names[d]}', left, uids)
            engine._news = True
        return
