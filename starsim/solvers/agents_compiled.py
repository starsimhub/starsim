"""
Not for the user: the agent engine with its moves compiled (`ss.solvers.Compiled`): a declared graph laid out once as tables, whose
moves of a step are one call (see `ss.solvers.backends.moves.step_moves()`)

Used with: agents; declared modules
"""
import types
import numpy as np
import sciris as sc
import starsim as ss
from .agents import Agents
from .agents_moving import ROUNDS, ROUND_MAX
from .backends import moves as kernel

__all__ = ['Compiled', 'compiled_reasons']

MAX_HOPS = 64 # The most moves of one agent in one call: more is a loop of moves with no wait


def compiled_reasons(engine):
    """ What of a graph the compiled moves don't take yet (empty if they take all of it), each of which the agent engine does """
    graph = engine.graph
    sim = engine.sim
    out = []
    if not ss.options.crn: # The agent engine then draws from each distribution in turn (dist.rvs()), as its other compiled passes do (see Agents.crn_key())
        out.append('no common random numbers (ss.options.crn=False)')
    if sim.pars.get('even_draws'):
        out.append('evenly spaced draws (even_draws=True)')
    if sim.groups.n_groups > 1:
        out.append('groups')
    if engine.held is not None or sim.pars.branching:
        out.append('a branching process')
    if engine.strain is not None:
        out.append('strains')
    if graph.count_states:
        out.append('states that are not people')
    if graph.inflows or graph.transfers:
        out.append('births or transfers')
    for key in ['attrs', 'sets', 'derived', 'derived_trs', 'also', 'on_event']:
        if getattr(engine, key, None):
            out.append(key)
    if any(engine.sets_of.values()):
        out.append('sets with clocks')
    for name, ex in graph.exits.items():
        for tr in engine.competing.get(name, []):
            if getattr(tr, 'min_stay', None) is not None:
                out.append(f'{tr.name}: a competing rate with a minimum stay')
    if not engine.scheduled and not any(tr.trigger is not None for tr in graph.transitions.values() if not tr.contact):
        out.append('nothing scheduled (every exit is a competing rate, which the agent engine draws in a compiled pass already)')
    for tr in graph.transitions.values():
        if tr.migration:
            out.append(f'{tr.name}: migration')
        if engine.moves(tr):
            out.append(f'{tr.name}: a change of level')
        if getattr(tr, 'birth_on', None) is not None:
            out.append(f'{tr.name}: births by a flow')
        if any(not getattr(d, 'people', False) or d.name not in engine.state_index for d in (tr.dest or [])):
            out.append(f'{tr.name}: a destination that is not a state of people')
        if tr.contact:
            continue
        if tr.trigger is not None:
            if trigger_kind(engine, tr) is None:
                out.append(f'{tr.name}: a trigger other than a time after an event of this module, an event of another module, or a condition on people')
        elif tr.name in engine.dwells:
            dw = engine.dwells[tr.name]
            if getattr(dw, 'mods', None) or getattr(dw, 'scale', None) is not None:
                out.append(f'{tr.name}: a dwell time with modifiers')
    return out


def trigger_kind(engine, tr):
    """
    How the call takes a trigger (`ss.on()`): 'after', a time after an event of this module (scheduled at the event); 'event', at once on an event
    of another module (whom it has happened to, found before each call); 'cond', at once while a condition on people holds (due for those
    in its source at the start of the step, and for anyone who enters it); None for one it does not take yet
    """
    x = tr.trigger.x
    delayed = tr.name in engine.delays
    if tr.trigger.event:
        spec = x.spec()
        if set(spec) != {'tr'}:
            return None
        own = ss.conditions.modname(x.module) in [None, engine.mod.name]
        if own and delayed and spec['tr'] in engine.graph.transitions:
            return 'after'
        if not own and not delayed:
            return 'event'
        return None
    if delayed or getattr(x, 'op', None) == 'event' or ss.conditions.module_refs(x):
        return None
    return 'cond'


def calls_back(engine, tr):
    """ Whether a transition's dwell time is drawn from a distribution with a parameter that is a function, which the call cannot draw, since the function may do more than give values (e.g. record when people entered): the engine draws it, after the call (see `Compiled.after_call()`) """
    dw = engine.dwells.get(tr.name)
    if dw is None or dw.dist is None:
        return None
    dist = engine.dists.get(f'dwell_{tr.name}')
    return any(callable(v) for v in getattr(dist, '_pars', {}).values())


class Compiled(Agents):
    """
    The agent engine, with the moves of a step compiled: those that are not by contact are one call over the agents, and those after the
    step's infections another, each taking every agent whose time has come through its exits, splits, and moves after events, as the
    same-step cascade does (see `ss.solvers.backends.moves`). Everything else is the agent engine's. A graph with what the call doesn't
    take yet (see `compiled_reasons()`) runs on the agent engine itself. The call makes the draws the agent engine's moves would make, in
    the same rounds (see `Moving.new_round()`), so the two give the same numbers.
    """

    def init_post(self):
        self.kplan = None # While the agent engine places people at the start
        super().init_post()
        self.uncompiled = compiled_reasons(self) # What the compiled moves don't take, if anything: then the agent engine's own moves run
        if not self.uncompiled:
            try:
                self.lay_out()
            except ss.CapabilityError as e:
                self.uncompiled = [str(e)]
                self.kplan = None
        return

    #%% The tables

    def lay_out(self):
        """ The graph as the tables of the call: states, exits, transitions, moves after events, and distributions """
        graph = self.graph
        index = self.state_index
        states = graph.people_states
        ns = len(states)
        trs = list(graph.transitions.values())
        tid = {tr.name: i for i, tr in enumerate(trs)}
        nt = len(trs)
        ints = lambda x: np.array(x, dtype=np.int64)
        ptr = lambda lists: ints(np.cumsum([0] + [len(x) for x in lists]))
        order, position, _ = self.sweep()
        k = types.SimpleNamespace()
        k.trs, k.tid, k.names = trs, tid, [s.name for s in states]
        k.pos = ints([position.get(tr.name, nt) for tr in trs])
        rated = {tr.name for name in k.names for tr in self.competing.get(name, [])}
        k.kind = ints([2 if tr.contact else 1 if tr.trigger is not None else 3 if tr.name in rated else 0 for tr in trs])
        k.split = np.array([bool(tr._split) for tr in trs])
        k.sched = np.array([s.name in self.scheduled for s in states])
        self.kdefer = [s.name for s in states if any(calls_back(self, tr) for tr in self.exits_of.get(s.name, []))] # Scheduled by the engine, after the call (see after_call())
        k.defer = np.array([nm in self.kdefer for nm in k.names])
        k.sched_k = k.sched & ~k.defer # The states whose exits the call schedules
        self.krescaled = [s.name for s in states if s.name not in self.kdefer and self.exits_of.get(s.name) and self.rescaled_of(s.name) and f'mult_{s.name}' in self.mod.__dict__] # What is left of a dwell is rescaled when its speed changes (see rescale_exits())

        # Exits of each state, in the engine's order (branches, then timed exits)
        exits = [self.exits_of[s.name] for s in states]
        k.ex_ptr, k.ex_tr = ptr(exits), ints([tid[tr.name] for e in exits for tr in e])
        k.nbr = ints([len(self.branches_of(s.name)) for s in states])
        k.loops = np.concatenate([self.loops_of[s.name] for s in states] + [np.zeros(0, bool)]).astype(bool)

        # Destinations
        dsts = [[index[d.name] for d in (tr.dest or [])] + [-1]*bool(tr._outflow or tr.may_die) for tr in trs] # -1 for a death
        k.dst_ptr, k.dst_state = ptr(dsts), ints([d for x in dsts for d in x])
        k.src_mask = np.zeros((nt, ns), dtype=bool)
        for i, tr in enumerate(trs):
            for s in tr.source or []:
                if s.people and s.name in index:
                    k.src_mask[i, index[s.name]] = True

        # Distributions: each transition's dwell time, and each trigger's delay
        self.kdists = [] # (key, dist) of each
        self.kspecs = [] # (family, how its parameters are found) of each
        k.dw = np.full(nt, -1, np.int64)
        k.aft = np.array([self.after_of(tr) for tr in trs], dtype=float)
        scheduled = {tr.name for e in self.exits_of.values() for tr in e} # Exits the call schedules; a competing rate is drawn by the engine's own pass (see leave()), so its dwell time is not read
        k.has_dw = np.array([tr.name in self.dwells and tr.name in scheduled and not tr.contact and tr.trigger is None and not calls_back(self, tr) for tr in trs])
        rescaled = {tr.name for nm in self.krescaled for i in self.rescaled_of(nm) for tr in [self.exits_of[nm][i]]}
        self.kmeans = {} # The transitions whose dwell times are exponential with a mean worked out each step (no distribution)
        k.adj = np.full(nt, -1, np.int64) # The multiplier on each dwell's rate (e.g. 0 outside its who=), worked out each step
        for i, tr in enumerate(trs):
            if k.has_dw[i]:
                k.adj[i] = self.kadd_mean(('adjust', tr.name, tr.name in rescaled))
        for i, tr in enumerate(trs):
            if k.has_dw[i]:
                dw = self.dwells[tr.name]
                if dw.dist is not None:
                    k.dw[i] = self.kadd_dist(f'dwell_{tr.name}', self.dists[f'dwell_{tr.name}'])
                else:
                    unit = self.dists[f'dwell_{tr.name}']
                    if not (type(unit) is ss.expon and unit.pars.mean == 1):
                        raise ss.CapabilityError(f'the dwell time of {tr.name} has stages (an Erlang distribution), which the compiled moves do not draw yet')
                    k.dw[i] = self.kadd_mean(('dwell', tr.name))
        k.comp_state = np.array([bool(self.competing.get(nm)) for nm in k.names]) # States with competing rates, which the agent engine draws (see step_state())
        k.has_late = np.array([nm in self.late for nm in k.names])
        k.halved = np.array([nm in self.halved for nm in k.names])
        self.kevents = [tr for tr in trs if tr.trigger is not None and not tr.contact and trigger_kind(self, tr) == 'event'] # Taken as arrivals, found before each call (see event_arrivals())
        trigs = [tr for tr in trs if tr.trigger is not None and not tr.contact and trigger_kind(self, tr) != 'event']
        k.tr_trig = ints([tid[tr.name] for tr in trigs])
        k.trig_dist = np.full(len(trigs), -1, np.int64)
        k.trig_fixed = np.ones(len(trigs), dtype=bool)
        self.ktrig_means = {}
        for g, tr in enumerate(trigs):
            if tr.name not in self.delays: # A condition: at once
                continue
            raw = getattr(self.delays[tr.name], 'raw', None)
            if isinstance(ss.transitions.by_dist(raw), ss.Dist):
                k.trig_dist[g] = self.kadd_dist(f'after_{tr.name}', self.dists[f'after_{tr.name}'])
            else:
                k.trig_dist[g] = self.kadd_mean(('after', tr.name))
        after = [[] for _ in trs]
        for g, tr in enumerate(trigs):
            if tr.trigger.event:
                after[tid[tr.trigger.x.spec()['tr']]].append(g)
        k.ae_ptr, k.ae_trig = ptr(after), ints([g for x in after for g in x])
        k.trig_where = np.full(len(trigs), -1, np.int64)
        k.trig_cond = np.full(len(trigs), -1, np.int64) # The column of the condition of a trigger on one
        k.lazy = np.zeros(nt, dtype=bool) # The splits whose shares are worked out on demand, for the agents that take them (see lay_out_values())
        self.kwhere = []
        for g, tr in enumerate(trigs):
            if tr.where is not None:
                k.trig_where[g] = len(self.kwhere)
                self.kwhere.append(tr.where)
            if not tr.trigger.event:
                k.trig_cond[g] = len(self.kwhere)
                self.kwhere.append(tr.trigger.x)
        k.trigs = trigs
        k.cond_trigs = [(g, [index[s.name] for s in tr.source if s.people and s.name in index]) for g, tr in enumerate(trigs) if not tr.trigger.event]

        # The columns of the table of shares: each state's branches, and each split's destinations
        self.kedges = [] # (key, function of uids giving the cumulative shares, one row per edge)
        k.br_col = np.full(ns, -1, np.int64)
        k.dst_col = np.full(nt, -1, np.int64)
        ncol = 0
        for i, s in enumerate(states):
            if k.nbr[i]:
                k.br_col[i] = ncol
                self.kedges.append((('branch', s.name), lambda u, s=s: self.branch_edges(s, u)))
                ncol += k.nbr[i] + 1
        for i, tr in enumerate(trs):
            if tr._split:
                k.dst_col[i] = ncol
                self.kedges.append((('dest', tr.name), lambda u, tr=tr: self.dest_edges(tr, u, np.full(len(u), -1))))
                ncol += len(dsts[i]) + 1
        k.ncol = ncol
        k.dst_width = [len(x) + 1 for x in dsts] # The columns of a split's shares: one per destination (a death included), and its end
        k.fam = ints([f for f, _ in self.kspecs])
        k.flowdst_w = max([len(x) for x in dsts] + [1])
        k.nt, k.ns, k.ntrig = nt, ns, len(trigs)
        k.one = np.ones(1)
        k.dt = float(self.dt)
        k.dst_keys = [[f'{tr.name}.{d.name}' for d in (tr.dest or [])] + [f'{tr.name}.dead']*bool(tr.may_die) for tr in trs]
        k.scan_sched = k.sched_k & ~k.comp_state # Exits of a state with competing rates too are taken by leave()
        k.stat1 = (k.sched_k, k.ex_ptr, k.ex_tr, k.nbr, k.br_col, k.loops, k.pos, k.kind, k.split, k.dst_ptr, k.dst_state, k.dst_col, k.dw, k.adj, k.aft, k.has_dw, k.src_mask, k.ae_ptr, k.ae_trig, k.tr_trig)
        k.stat2 = (k.trig_dist, k.trig_fixed, np.zeros(1), k.fam) # (trig_mean is not used: a delay given as a number is an exponential distribution, see kadd_mean())

        # The streams the call draws from, each as the agent engine draws it (see Moving.draw()): each distribution's, and each branch's and split's uniforms
        def stream(key):
            if key not in k.streams:
                k.streams.append(key)
            return k.streams.index(key)
        k.streams = []
        k.dstream = ints([stream(spec[1] if spec[0] == 'dist' else f'{spec[1]}_{spec[2]}') if spec[0] == 'dist' or spec[1] != 'adjust' else -1 for _, spec in self.kspecs] + [-1])
        k.bstream = ints([stream(f'branch_{s.name}') if k.nbr[i] else -1 for i, s in enumerate(states)])
        k.tstream = ints([stream(f'dst_{tr.name}') if tr._split else -1 for tr in trs])
        _, _, back = self.sweep()
        k.back = np.array([tr.name in back for tr in trs]) # A move by it means another pass, for a trigger before it that waits for its event (see finish_moves())
        k.againx = np.array([[any(position[e.name] <= position[tr.name] for e in self.exits_of.get(s.name, [])) for s in states] for tr in trs]).reshape(nt, ns) # An arrival by it in a state with a scheduled exit before it in the sweep means another pass
        self.kplan = k
        self.ktables = None
        self.kcrn = {} # By the module whose rounds these are: see crn_tables()
        self.kseen = self.n_batches
        return

    def kadd_dist(self, key, dist):
        """ A distribution the call draws from: its family and how to find its parameters (see refresh()) """
        if dist.unit is not None or dist.round or dist._use_ppf is False:
            raise ss.CapabilityError(f'the distribution of {key} has a unit or is rounded')
        if any(callable(v) for v in dist._pars.values()):
            raise ss.CapabilityError(f'a parameter of {key} is a function, which the compiled moves cannot call as agents enter')
        if isinstance(dist, ss.constant):
            fam = kernel.FAMILIES['constant']
        elif type(dist) is ss.uniform: # Starsim's own (low, high), drawn as SciPy's uniform (loc, scale)
            fam = kernel.FAMILIES['uniform']
        else:
            name = getattr(getattr(dist, 'dist', None), 'dist', None)
            name = getattr(name, 'name', None)
            if name not in kernel.FAMILIES:
                raise ss.CapabilityError(f'the distribution of {key} ({type(dist).__name__}) is not one the compiled moves draw yet')
            fam = kernel.FAMILIES[name]
        self.kdists.append((key, dist))
        self.kspecs = getattr(self, 'kspecs', []) + [(fam, ('dist', key, dist))]
        return len(self.kspecs) - 1

    def kadd_mean(self, what):
        """ An exponential whose mean is worked out at each step (a dwell time from a rate, or a delay given as a number) """
        self.kspecs = getattr(self, 'kspecs', []) + [(kernel.FAMILIES['expon'], ('mean',) + what)]
        return len(self.kspecs) - 1

    def dist_pars(self, key, dist):
        """ The parameters of a distribution, as the call takes them (SciPy's, see `ss.solvers.backends.moves.draw()`), each a number or a function giving one value per agent (by UID) """
        if isinstance(dist, ss.constant):
            names = ['value']
        elif type(dist) is ss.uniform:
            names = ['low', 'high']
        else:
            names = dict(expon=['loc', 'scale'], lognorm=['s', 'scale', 'loc'], norm=['loc', 'scale'], uniform=['loc', 'scale'], weibull_min=['c', 'scale', 'loc'])[dist.dist.dist.name]
        defaults = dict(loc=0.0, scale=1.0)
        held = self.agent_pars(key, dist)
        if held is not None and type(dist) is ss.uniform:
            raise ss.CapabilityError(f'the parameters of {key} differ by sex, which the compiled moves do not take yet for ss.uniform')
        if held is not None: # By sex: a table kept for the run (see agent_pars())
            pick = lambda nm: (lambda: self.agent_pars(key, dist)[2 if held[2] is not None else 1].get(nm, defaults.get(nm, 0.0)))
            return [('table', pick(nm)) for nm in names]
        dist.process_size(ss.uids([0]))
        dist.process_pars()
        src = dict(dist._pars) if getattr(dist, 'dist', None) is None else dict(dist.dist.kwds)
        out = []
        for nm in names:
            v = np.asarray(src.get(nm, defaults.get(nm, 0.0)), dtype=float)
            if v.size != 1:
                raise ss.CapabilityError(f'the parameters of {key} differ between people in a way the compiled moves do not take yet')
            out.append(('scalar', float(v.reshape(-1)[0])))
        if type(dist) is ss.uniform: # As loc and scale
            low, high = out[0][1], out[1][1]
            out = [('scalar', low), ('scalar', high - low)]
        return out

    def everyone(self, n):
        return ss.uids(np.arange(n))

    def value_source(self, func, fixed):
        """ A parameter worked out by a function of the agents: a number, if it is one and fixed; else a table kept for the run (fixed), or worked out at each step """
        v = np.asarray(func(ss.uids([0])), dtype=float)
        if fixed and v.size == 1 and np.ndim(func(self.everyone(len(self.sim.people.uid.raw)))) == 0:
            return ('scalar', float(v.reshape(-1)[0]))
        full = lambda u: np.broadcast_to(np.asarray(func(u), dtype=float), len(u))
        if fixed:
            key = ('kval', id(func))
            return ('table', lambda: self.lifelong_table(key, full))
        return ('vary', full) # For the agents the call takes (see vary())

    def lay_out_values(self):
        """ Where each value the call reads comes from: each distribution's parameters (see `kspecs`), and the shares of each branch and split """
        self.kpar_src = []
        for d, (fam, spec) in enumerate(self.kspecs):
            if spec[0] == 'dist':
                srcs = self.dist_pars(spec[1], spec[2])
                dist = spec[2]
                if type(dist) is ss.uniform and dist.ppf(np.full(1, 0.5, dtype=dist.hash_dtype or ss.dtypes.float)).dtype == np.float32: # Parameters that are plain numbers, which NumPy takes in single precision with the uniforms
                    self.kplan.fam[d] = kernel.FAMILIES['uniform32']
            elif spec[1] == 'adjust':
                name, changes = spec[2], spec[3]
                srcs = [self.value_source(lambda u, name=name: self.adjusted(name, u), fixed=not changes)] # Lifelong, unless the exit is rescaled when it changes (see rescale_exits())
            elif spec[1] == 'dwell':
                name = spec[2]
                def mean(u, name=name):
                    with self.evaluating(uids=u, for_uids=True):
                        return self.dwell_mean(name)
                srcs = [('scalar', 0.0), self.value_source(mean, fixed=not self.varies(self.dwells[name]))] # As the engine takes it (see varies())
            else: # A delay given as a number: an exponential with that mean
                name = spec[2]
                expr = self.delays[name]
                srcs = [('scalar', 0.0), self.value_source(lambda u, expr=expr: self.eval_for(expr, u), fixed=self.is_fixed(expr))]
            self.kpar_src.append(srcs)
        self.kedge_src = []
        everyone = lambda: self.everyone(len(self.sim.people.uid.raw))
        for key, func in self.kedges:
            func(ss.uids([0])) # So the engine knows whether they are fixed
            if self.__dict__.get('_fixed', {}).get(key, False):
                for_new = (lambda new, s=key[1]: self.branch_edges_for(self.graph.states[s], new).T) if key[0] == 'branch' else (lambda new, t=key[1]: self.dest_edges_for(self.graph.transitions[t], new, np.full(len(new), -1)).T)
                self.kedge_src.append(('table', lambda key=key, for_new=for_new: self.lifelong_table(key, for_new)))
            else:
                self.kedge_src.append(('vary', lambda u, func=func: np.asarray(func(u), dtype=float).T)) # For the agents the call takes (see vary())
        k = self.kplan
        k.lazy[:] = False
        self.klazy = {} # For each split worked out on demand: its first column, its width, and how to work it out
        c = 0
        for (key, _), (kind, get) in zip(self.kedges, self.kedge_src):
            width = len(self.branches_of(key[1])) + 1 if key[0] == 'branch' else k.dst_width[k.tid[key[1]]]
            i = k.tid.get(key[1]) if key[0] == 'dest' else None
            if kind == 'vary' and i is not None and not any(isinstance(x, ss.expressions.mother) for p in (k.trs[i].p or []) if p is not None for x in ss.expressions.nodes(p)):
                # Only for those who reach it, as they reach it; not a branch (drawn as people enter a state, so worked out for everyone in the call), nor shares that read other
                # agents (e.g. a fetus's mother, who may move in the same call), which are as they were at the start of the call
                k.lazy[i] = True
                self.klazy[i] = (c, width, get)
            c += width
        return

    def refresh(self, n):
        """ The tables of values by agent: rows for agents added since are filled in, and the values that change, worked out again """
        k = self.kplan
        t = self.ktables
        if t is None:
            self.lay_out_values()
            ncol = sum(1 for srcs in self.kpar_src for kind, _ in srcs if kind != 'scalar')
            pcol = np.zeros(3*len(self.kspecs), np.int64)
            scal = []
            c = 0
            for d, srcs in enumerate(self.kpar_src):
                for j, (kind, v) in enumerate(srcs):
                    if kind == 'scalar':
                        scal.append(v)
                        pcol[3*d + j] = -len(scal)
                    else:
                        pcol[3*d + j] = c
                        c += 1
            t = self.ktables = types.SimpleNamespace(n=0, used=0, wused=0, cap=0, PA=None, E=None, PS=np.array(scal + [0.0]), pcol=pcol, ncol=max(ncol, 1), varies=True, WH=None)
        if n > t.cap: # Room for more agents
            cap = max(n, int(1.5*t.cap))
            PA, E = np.zeros((cap, t.ncol)), np.zeros((cap, max(k.ncol, 1)))
            if t.PA is not None:
                PA[:t.n] = t.PA[:t.n]
                E[:t.n] = t.E[:t.n]
            t.PA, t.E, t.cap = PA, E, cap
        used = self.sim.people.uid.len_used
        grown = n > t.n # Room for more agents: the rows of the tables for them (see fill_new() for those made since)
        if t.varies is True: # Once: whether anything is worked out at each step
            t.varies = any(kind == 'vary' for srcs in self.kpar_src for kind, _ in srcs) or any(kind == 'vary' for kind, _ in self.kedge_src)
        fixed_where = not self.kwhere or all(self.is_fixed(w) for w in self.kwhere)
        if not grown and fixed_where and t.WH is not None and used <= t.wused:
            return t
        if grown:
            self.fill_rows(t, min(t.n, t.used), n)
            t.n, t.used = n, used
        if self.kwhere and (grown or t.WH is None or not fixed_where or used > t.wused): # Read by the scan (for triggers on a condition), so worked out for those made since at once
            t.WH = tuple(np.ascontiguousarray(np.broadcast_to(np.asarray(self.selector.select(w), dtype=bool), n)) for w in self.kwhere)
        elif not self.kwhere:
            t.WH = (np.zeros(1, bool),)
        t.wused = used
        return t

    def fill_rows(self, t, lo, hi):
        """ The rows of the tables kept for the run (see lifelong_table()), for agents `lo` to `hi` (by UID) """
        c = 0
        for d, srcs in enumerate(self.kpar_src):
            for j, (kind, get) in enumerate(srcs):
                if kind == 'scalar':
                    continue
                if kind == 'table':
                    vals = np.asarray(get(), dtype=float)
                    t.PA[lo:hi, c] = vals[lo:hi] if vals.ndim else vals
                c += 1
        c = 0
        for (key, _), (kind, get) in zip(self.kedges, self.kedge_src):
            width = len(self.branches_of(key[1])) + 1 if key[0] == 'branch' else self.kplan.dst_width[self.kplan.tid[key[1]]]
            if kind == 'table':
                vals = np.asarray(get(), dtype=float)
                t.E[lo:hi, c:c + width] = vals[lo:hi]
            c += width
        return

    def fill_new(self, agents):
        """
        Before a call, the rows of the tables kept for the run for agents made since they were filled (the arrays have room for agents not yet made, whose values, e.g.
        their sex, are not known until they are): only if the call takes one of them, since it reads the tables only for the agents it takes, and few are taken in the step they are made
        """
        t = self.ktables
        used = self.sim.people.uid.len_used
        if used > t.used and len(agents) and agents.max() >= t.used:
            self.fill_rows(t, t.used, used)
            t.used = used
        return

    def vary(self, agents):
        """ The values that change from step to step (e.g. shares by age), worked out for the agents the call takes, who are the only ones it reads them for """
        t = self.ktables
        if not t.varies or not len(agents):
            return
        uids = ss.uids(agents)
        c = 0
        for srcs in self.kpar_src:
            for kind, get in srcs:
                if kind == 'scalar':
                    continue
                if kind == 'vary':
                    t.PA[agents, c] = get(uids)
                c += 1
        c = 0
        for (key, _), (kind, get) in zip(self.kedges, self.kedge_src):
            width = len(self.branches_of(key[1])) + 1 if key[0] == 'branch' else self.kplan.dst_width[self.kplan.tid[key[1]]]
            if kind == 'vary':
                if key[0] == 'dest' and self.kplan.lazy[self.kplan.tid[key[1]]]:
                    t.E[agents, c:c + width] = np.nan # Worked out if they reach it (see run_moves())
                else:
                    t.E[agents, c:c + width] = get(uids)
            c += width
        return

    #%% The steps

    def arrays(self, n):
        """ The module's arrays the call writes, as tuples (a stand-in for a state that has none), kept until the arrays grow """
        held = self.__dict__.get('_karrays')
        if held is not None and held[0] == n and held[1] is self.mod[self.kplan.names[0]].raw and self.kplan.LATE is not None:
            return held[2]
        out = self.make_arrays(n)
        self._karrays = (n, self.mod[self.kplan.names[0]].raw, out)
        return out

    def make_arrays(self, n):
        mod = self.mod
        k = self.kplan
        names = k.names
        S = tuple(mod[nm].raw for nm in names)
        sti, ssh, kept = [], [], []
        for nm in names:
            arr = getattr(mod, f'ti_{nm}', None)
            ok = arr is not None and not isinstance(arr, ss.arrays.Unkept)
            kept.append(ok)
            sti.append(arr.raw if ok else None)
            ssh.append(arr.shift.raw if ok else None)
        def fdum(xs, dtype): # The same dtype for every one, as the call takes a tuple of arrays of one type
            dtype = next((x.dtype for x in xs if x is not None), np.dtype(dtype))
            return tuple(x if x is not None else np.zeros(1, dtype) for x in xs)
        lefts = {self.state_index[nm]: None for nm in names}
        for i, arrname in self.lefts.items():
            lefts[i] = mod[arrname].raw
        XT = [mod[f'ti_exit_{nm}'].raw if nm in self.scheduled else None for nm in names]
        XW = [mod[f'exit_{nm}'].raw if nm in self.scheduled else None for nm in names]
        xw_dtype = next((x.dtype for x in XW if x is not None), np.int8)
        TT = tuple(mod[f'ti_after_{tr.name}'].raw if tr.name in self.delays else np.full(n, np.nan, ss.dtypes.float) for tr in k.trigs) or (np.zeros(1, ss.dtypes.float),)
        k.LATE = fdum([mod[f'late_{nm}'].raw if nm in self.late else None for nm in names], ss.dtypes.float)
        sh_dtype = next((x.dtype for x in ssh if x is not None), np.int8)
        f = ss.dtypes.float
        return (S, fdum(sti, f), fdum(ssh, sh_dtype), np.array(kept), fdum([lefts[i] for i in range(len(names))], f),
                np.array([lefts[i] is not None for i in range(len(names))]), fdum(XT, f), fdum(XW, xw_dtype), TT)

    def crn_tables(self):
        """ For the call's draws (see `ss.solvers.backends.moves.rand()`): the seed of each stream in each round, by the transition the round is for (the first column for none), as `round_seed()` gives it, and the width of its uniforms """
        k = self.kplan
        m = max(len(k.streams), 1)
        RS = np.zeros((m, k.nt + 1), np.int64)
        ush, usc, z32 = np.zeros(m, np.uint64), np.zeros(m), np.zeros(m, dtype=bool)
        for r, key in enumerate(k.streams):
            dist = self.dists[key]
            seed = dist.seed
            RS[r] = [self.round_seed(seed, None)] + [self.round_seed(seed, tr.name) for tr in k.trs]
            dtype = np.dtype(dist.hash_dtype or ss.dtypes.float)
            ush[r], usc[r] = ss.solvers.backends.draws.crn_bits(dtype)
            z32[r] = dtype.itemsize <= 4 # SciPy's ndtri of these is single precision (see ss.normal.ppf())
        self.kcrn[self._source] = (RS, ush, usc, z32)
        return self.kcrn[self._source]

    def start_cascade(self):
        """ A same-step cascade begins, as `finish_moves()` begins one: its first pass is the next round, and the calls it takes (see `run_moves()`) count its passes """
        self.new_round()
        self._kbase = self.round[1]
        again = self.__dict__.get('_kagain')
        if again is None:
            again = self._kagain = np.zeros(MAX_HOPS + 2, dtype=bool)
        again[:] = False
        self.end_cascade()
        return

    def end_cascade(self):
        """ The round the cascade leaves the engine in, as `finish_moves()` leaves it: that of its last pass, for the last transition of the sweep """
        again = self._kagain
        npass = 1 if not again[0] else int(np.argmin(again)) + 1 if not again.all() else len(again)
        self.round = (self.step_key(), self._kbase + npass - 1, self.sweep()[0][-1].name)
        self._drawn = {}
        return

    def pass_of(self, tr, after, p):
        """ The pass of the cascade in which agents who arrived somewhere by transition `after` (by index, in pass `p`) are taken on by `tr`, as `finish_moves()` comes to it: in the same pass if it is later in the sweep, else in the next (which there is, since they wait to be sent) """
        k = self.kplan
        later = k.pos[tr] > k.pos[after]
        if not np.all(later):
            self._kagain[np.unique(np.asarray(p)[~np.asarray(later)])] = True
        return np.where(later, p, np.asarray(p) + 1)

    def in_round(self, p, tr):
        """ Draw as the agent engine would in pass `p` of the cascade, in the round of transition `tr` (by index) """
        self.round = (self.step_key(), self._kbase + int(p), self.kplan.trs[int(tr)].name)
        self._drawn = {}
        return

    def run_moves(self, scan, agents=None, arr_tr=None, arr_src=None, arr_kind=None, arr_p=None, arr_q=None, ti_due=None, ti_enter=None, mode=2, arrivals=None, cont=False):
        """ One call of the compiled moves (`cont`: in the cascade the last call began; else a new one), then their bookkeeping: counts, flows, events """
        k = self.kplan
        people = self.sim.people
        n = len(self.mod.rel_sus.raw)
        if not cont:
            self.start_cascade()
        if self.__dict__.get('_kmark') is None or len(self._kmark) != n: # Made as needed (and again after shrink())
            self._kmark, self._kidx = np.zeros(n, dtype=bool), np.empty(n, np.int64)
            self._karrays = None
        if self.__dict__.get('_kev') is None:
            self._kev = tuple(np.empty(4096, np.int64) for _ in range(5))
        t = self.refresh(n)
        S, STI, SSH, kept, LEFT, has_left, XT, XW, TT = self.arrays(n)
        alive = people.alive.raw
        ti = self.ti
        if ti_due is None:
            ti_due = ti
        if scan: # Those with a move due, as a list to look at
            for g, srcs in k.cond_trigs: # Due now for those in its source for whom the condition holds (and for anyone who enters it, in the call)
                inside = S[srcs[0]].copy() if len(srcs) == 1 else np.logical_or.reduce([S[i] for i in srcs])
                inside &= t.WH[k.trig_cond[g]]
                TT[g][:] = np.nan
                TT[g][inside] = -np.inf
            agents = kernel.due_agents(S, XT, k.scan_sched, TT, k.ntrig, alive, float(ti_due), float(ti), self._kmark, self._kidx)
            arr_tr = arr_src = self._kzeros(len(agents))
            arr_kind = arr_tr + 1
            if arrivals is not None: # Those leaving by competing rates first, then those with a move due
                agents = np.concatenate([arrivals[0], agents])
                arr_tr = np.concatenate([arrivals[1], arr_tr])
                arr_src = np.concatenate([arrivals[2], arr_src])
                arr_kind = np.concatenate([np.zeros(len(arrivals[0]), np.int64), arr_kind])
            if not len(agents):
                return
        if arr_p is None: # In the first pass, with no move yet
            arr_p = self._kzeros(len(agents))
        if arr_q is None:
            arr_q = self._kzeros(len(agents), -1)
        self.fill_new(agents)
        self.vary(agents)
        weighted = people.scale.weighted
        w = people.scale.raw if weighted else k.one.astype(people.scale.raw.dtype)
        if ti_enter is None:
            ti_enter = ti
        shift = self.index_of(ti_enter) - ti_enter
        slots = people.slot_raw()
        RS, ush, usc, z32 = self.kcrn.get(self._source) or self.crn_tables()
        crn = (np.empty(0, np.int64) if slots is None else slots, ROUNDS + self.step_key()*ROUND_MAX + self._kbase, RS, ush, usc, z32, k.dstream, k.bstream, k.tstream, k.back, k.againx, self._kagain)
        flow, flowdst, dlive = np.zeros(k.nt), np.zeros((k.nt, k.flowdst_w)), np.zeros(k.ns, np.int64)
        logs = []
        batch = (agents, arr_tr, arr_src, arr_kind, arr_p, arr_q)
        while batch is not None:
            agents, arr_tr, arr_src, arr_kind, arr_p, arr_q = batch
            total = len(agents)
            pend = tuple(np.empty(max(total, 1), np.int64) for _ in range(6))
            n_pend = np.zeros(1, np.int64)
            start = 0
            while True:
                ev = self._kev
                start, n_ev = kernel.step_moves(agents, arr_tr, arr_src, arr_kind, arr_p, arr_q, False, start, float(ti_due), float(ti), float(ti_enter), float(shift), alive,
                    S, STI, SSH, kept, LEFT, has_left, XT, XW, *k.stat1, TT, k.trig_where, t.WH, *k.stat2, t.pcol, t.PA, t.PS, t.E, people.phase(), k.dt, w, weighted, *crn,
                    *ev, 0, flow, flowdst, dlive, MAX_HOPS, mode, k.comp_state, k.LATE, k.has_late, k.halved, k.trig_cond, k.lazy, *pend, n_pend)
                logs.append(tuple(x[:n_ev].copy() for x in ev)) # Copies, since the buffer is used again by the next call, and the events and deaths booked keep these arrays
                if start >= total:
                    break
                self._kev = tuple(np.empty(2*len(ev[0]), np.int64) for _ in range(5)) # The log was full: more room, for the rest
            batch = self.fill_pending(pend, int(n_pend[0]))
        arrived = self.book(logs, flow, flowdst, dlive, ti_enter, ti_due)
        due, self._kdue = self.__dict__.get('_kdue'), None
        if due: # Scheduled after the call, and due now: taken by another, in the passes they come in
            cat = lambda i: np.concatenate([x[i] for x in due])
            more = self.run_moves(scan=False, agents=cat(0), arr_tr=cat(1), arr_src=cat(2), arr_kind=np.zeros(len(cat(0)), np.int64), arr_p=cat(3), arr_q=cat(4), ti_due=ti_due, ti_enter=ti_enter, mode=mode, cont=True)
            if more is not None:
                arrived = more if arrived is None else tuple(np.concatenate([x, y]) for x, y in zip(arrived, more))
        self.end_cascade()
        return arrived

    def fill_pending(self, pend, m):
        """ The agents a call stopped at a split whose shares were not worked out: their shares, worked out now, and them to take on from there, as the next batch (None if there are none) """
        if not m:
            return None
        pa, pt, ps, pk, pp, pq = [x[:m].copy() for x in pend]
        E = self.ktables.E
        for i in np.unique(pt).tolist():
            c, width, get = self.klazy[i]
            uids = pa[pt == i]
            E[uids, c:c + width] = get(ss.uids(uids))
        return pa, pt, ps, pk, pp, pq # Kind 0 takes the transition again as an arrival, from the state they are in; kind 1 looks again; each where it was in the cascade

    def _kzeros(self, m, value=0):
        """ An array of `m` zeros (or of -1s), kept for the next call, which the call only reads """
        key = '_kz' if value == 0 else '_kneg'
        z = self.__dict__.get(key)
        if z is None or len(z) < m:
            z = np.full(max(m, 2*len(z) if z is not None else 1024), value, np.int64)
            self.__dict__[key] = z
        return z[:m]

    def book(self, logs, flow, flowdst, dlive, ti_enter=None, ti_due=None):
        """ After a call: the counts of the states, the flows, and the moves as events (`dis.<transition>`, and the events triggers read); then what the engine does for those who entered a state (see after_call()) """
        k = self.kplan
        live = self.live
        if live is not None:
            for i in dlive.nonzero()[0].tolist():
                nm = k.names[i]
                live[nm] = live[nm] + dlive[i]
        fc = self.flowcounts
        for i in flow.nonzero()[0].tolist():
            tr = k.trs[i]
            fc[tr.name] += flow[i]
            if tr._ends:
                for j, key in enumerate(k.dst_keys[i]):
                    if flowdst[i, j]:
                        fc[key] += flowdst[i, j]
        ev_a, ev_t, ev_s, ev_d, ev_p = [np.concatenate(x) for x in zip(*logs)] if len(logs) > 1 else logs[0]
        n = len(ev_a)
        if not n:
            self.kseen = self.n_batches
            return None
        code = ev_t*(k.ns + 1) + ev_d
        if n == 1 or code.min() == code.max(): # One kind of move
            groups = [(0, n)]
        else:
            order = code.argsort(kind='stable')
            ev_a, ev_t, ev_s, ev_d, ev_p, code = ev_a[order], ev_t[order], ev_s[order], ev_d[order], ev_p[order], code[order]
            cuts = ((code[1:] != code[:-1]).nonzero()[0] + 1).tolist()
            groups = list(zip([0] + cuts, cuts + [n]))
        events = self.events
        for lo, hi in groups:
            tr = k.trs[int(ev_t[lo])]
            uids = ss.uids(ev_a[lo:hi])
            name = tr.name
            if ev_d[lo] < 0: # Deaths, requested as the agent engine does (see send()), and counted once each has one cause (see uncount_deaths())
                srcs = ev_s[lo:hi]
                self.record(tr, uids, srcs, -1)
                self.sim.people.request_death(uids, cause=self.mod)
                states = np.unique(srcs).tolist()
                self._died.append((tr, [self.graph.people_states[i] for i in states], [uids[srcs == i] for i in states]))
                events[name] = uids if name not in events else ss.uids.concatenate([events[name], uids])
                continue
            events[name] = uids if name not in events else ss.uids.concatenate([events[name], uids])
            self.record(tr, uids, ev_s[lo:hi], int(ev_d[lo]))
        self._news = True
        self.kseen = self.n_batches
        if self.kdefer or self.krescaled:
            self.after_call(ev_a, ev_t, ev_s, ev_d, ev_p, self.ti if ti_enter is None else ti_enter, self.ti if ti_due is None else ti_due)
        entered = k.comp_state[np.maximum(ev_d, 0)] & (ev_d >= 0)
        return ev_a[entered], ev_d[entered], ev_t[entered], ev_p[entered] # Who arrived in a state with competing rates, by which transition, in which pass (see step_state())

    def after_call(self, ev_a, ev_t, ev_s, ev_d, ev_p, ti_enter, ti_due):
        """
        For those who left or entered a state in a call: in a state the engine schedules (`kdefer`, whose dwell time is a function), the exit
        cleared, or scheduled as they enter, with the draws of the round they entered in (those due now are kept for another call, in `_kdue`,
        in the pass the cascade takes them in); in one whose exits are rescaled (`krescaled`), how fast each exit ran when it was drawn, as
        `schedule()` records it
        """
        mod = self.mod
        k = self.kplan
        index = self.state_index
        due = []
        for nm in self.kdefer:
            i = index[nm]
            left = ev_a[ev_s == i]
            if len(left):
                left = np.unique(left)
                self.clear_exit(nm, ss.uids(left[~mod[nm].raw[left]]))
            into = ev_d == i
            if not into.any():
                continue
            rounds = ev_p[into]*(k.nt + 1) + ev_t[into] # By the round they entered in
            for r in np.unique(rounds).tolist():
                p, tr = divmod(r, k.nt + 1)
                new = np.unique(ev_a[into][rounds == r])
                new = ss.uids(new[mod[nm].raw[new]])
                self.clear_exit(nm, new)
                self.in_round(p, tr)
                self.schedule(self.graph.states[nm], new, entered=ti_enter)
                now = new[mod[f'ti_exit_{nm}'].raw[new] <= ti_due] # Due by the call's time (the step's end, for those infected)
                if len(now):
                    which = mod[f'exit_{nm}'].raw[now]
                    now, which = now[which >= 0], which[which >= 0]
                    tids = np.array([k.tid[e.name] for e in self.exits_of[nm]], np.int64)[which.astype(np.int64)]
                    due.append((np.asarray(now, np.int64), tids, np.full(len(now), i, np.int64), self.pass_of(tids, tr, p).astype(np.int64), np.full(len(now), -1, np.int64)))
        for nm in self.krescaled:
            new = ev_a[ev_d == index[nm]]
            if not len(new):
                continue
            new = np.unique(new)
            new = ss.uids(new[mod[nm].raw[new]])
            which = mod[f'exit_{nm}'].raw[new]
            mults = np.ones(len(new))
            for i in self.rescaled_of(nm):
                chosen = which == i
                if chosen.any():
                    mults[chosen] = self.speed(self.exits_of[nm][i], new[chosen])
            mod[f'mult_{nm}'][new] = mults
        self._kdue = due or None
        return

    def event_arrivals(self):
        """ The people another module's event has happened to since the call last looked (see `new_events()`), who are in the source of a trigger on it: (agents, transitions, sources), for the call to take now """
        out = []
        if not self.kevents or not self._started:
            return out
        k = self.kplan
        mod = self.mod
        for tr in self.kevents:
            uids = self.new_events(tr)
            if not len(uids):
                continue
            if tr.where is not None:
                uids = uids[np.asarray(self.selector.select(tr.where), dtype=bool)[uids]]
            for src in tr.source:
                if src.people:
                    inside = np.asarray(uids[mod[src.name].raw[uids]], np.int64)
                    if len(inside):
                        out.append((inside, np.full(len(inside), k.tid[tr.name], np.int64), np.full(len(inside), self.state_index[src.name], np.int64)))
        return out

    def cond_arrivals(self):
        """ Those in the source of a trigger on a condition for whom it holds, as (agents, transitions, sources): outside a step's own call, which finds them itself (see run_moves()), e.g. for those placed at the start """
        out = []
        if not self._started:
            return out
        k = self.kplan
        mod = self.mod
        for g, srcs in k.cond_trigs:
            tr = k.trigs[g]
            holds = np.asarray(self.selector.select(tr.trigger.x), dtype=bool)
            if tr.where is not None:
                holds = holds & np.asarray(self.selector.select(tr.where), dtype=bool)
            for i in srcs:
                uids = np.flatnonzero(mod[k.names[i]].raw & holds)
                if len(uids):
                    out.append((uids.astype(np.int64), np.full(len(uids), k.tid[tr.name], np.int64), np.full(len(uids), i, np.int64)))
        return out

    def start_deferred(self):
        """ At the start of a step, in a state the engine schedules that has no competing rates (whose own pass schedules it, see leave()): those not yet scheduled, scheduled, and those due, as (agents, transitions, sources) """
        out = []
        k = self.kplan
        mod = self.mod
        for nm in self.kdefer:
            if self.competing.get(nm):
                continue
            uids = ss.uids(np.flatnonzero(mod[nm].raw))
            if not len(uids):
                continue
            unset = np.isnan(mod[f'ti_exit_{nm}'].raw[uids])
            if unset.any():
                self.schedule(self.graph.states[nm], uids[unset])
            now = uids[mod[f'ti_exit_{nm}'].raw[uids] <= self.ti]
            which = mod[f'exit_{nm}'].raw[now]
            now, which = now[which >= 0], which[which >= 0]
            if len(now):
                tids = np.array([k.tid[tr.name] for tr in self.exits_of[nm]], np.int64)
                out.append((np.asarray(now, np.int64), tids[which.astype(np.int64)], np.full(len(now), self.state_index[nm], np.int64)))
        return out

    def step_state(self):
        if self.kplan is None:
            return super().step_state()
        self._moved = []
        self.new_round()
        self.start_phase(0)
        self.place_newcomers()
        self.renew_events()
        k = self.kplan
        comp = [nm for nm, c in zip(k.names, k.comp_state) if c]
        members = self.start_counts(comp)
        missed = {nm: self.schedule(self.graph.states[nm], members[nm]) if (self.unscheduled is None or nm in self.unscheduled) and self.exits_of[nm] else None for nm in comp} # States with both: see Agents.step_state()
        self.unscheduled = set() if self.live is not None else None
        self._collect = [] # The competing rates, by the agent engine's own pass (see leave()), whose moves the call takes
        try:
            pending = []
            for nm in comp:
                if len(members[nm]):
                    self.leave(self.graph.states[nm], members[nm], missed[nm], pending)
            self.draw_pending(pending)
        finally:
            collected, self._collect = self._collect, None
        self._drawn_ti = self.ti
        for nm in self.krescaled: # What is left of the dwell times whose speed has changed, rescaled (those with competing rates too, by leave())
            if not self.competing.get(nm):
                uids = ss.uids(np.flatnonzero(self.mod[nm].raw))
                if len(uids):
                    self.rescale_exits(self.graph.states[nm], uids)
        parts = [(np.asarray(u, np.int64), np.full(len(u), k.tid[tr.name], np.int64), np.full(len(u), self.state_index[src.name], np.int64)) for tr, src, u in collected]
        parts += self.start_deferred() + self.event_arrivals()
        arrivals = tuple(np.concatenate(x) for x in zip(*parts)) if parts else None
        arrived = self.run_moves(scan=True, mode=0, arrivals=arrivals)
        n = len(self.mod.rel_sus.raw)
        row = None
        while arrived is not None and len(arrived[0]) and not self.wholestep: # Those who arrived in a state with competing rates have half a step of them now, once a step each, as the agent engine's arrive() gives them, in the round of their arrival
            if row is None:
                row = np.zeros(n, dtype=bool)
            ag, st, tr, ps = arrived
            rounds = (ps*(k.nt + 1) + tr)*k.ns + st
            parts = []
            for r in np.unique(rounds).tolist():
                rnd, i = divmod(r, k.ns)
                p, t = divmod(rnd, k.nt + 1)
                nm = k.names[i]
                uids = np.unique(ag[rounds == r])
                uids = uids[self.mod[nm].raw[uids] & ~row[uids]]
                row[uids] = True
                if not len(uids):
                    continue
                self._collect = []
                try:
                    self.in_round(p, t)
                    self.draw_competing(self.graph.states[nm], ss.uids(uids), 0.5*self.dt, key=f'entry_{nm}')
                finally:
                    collected, self._collect = self._collect, None
                for c, src, u in collected: # Taken by the cascade at the turn of the rate they left by
                    tid = k.tid[c.name]
                    parts.append((np.asarray(u, np.int64), np.full(len(u), tid, np.int64), np.full(len(u), self.state_index[src.name], np.int64), np.full(len(u), self.pass_of(tid, t, p), np.int64)))
            if not parts:
                break
            cat = lambda j: np.concatenate([x[j] for x in parts])
            arrived = self.run_moves(scan=False, agents=cat(0), arr_tr=cat(1), arr_src=cat(2), arr_kind=np.zeros(len(cat(0)), np.int64), arr_p=cat(3), mode=0, cont=True)
        self.step_transfers(self.dt) # None, but its round, as the agent engine takes it
        if self._news:
            self._news = False
            self.notify(self.ti)
        return

    def shrink(self):
        """ Forget the tables and buffers of the call, which hold a value for each agent (they are made again if the sim runs on), and the functions that make them """
        super().shrink()
        for key in ['_kmark', '_kidx', '_kev', '_karrays', '_kz', '_kneg', 'kedges', 'kedge_src', 'kpar_src', 'lifelong']:
            self.__dict__.pop(key, None)
        if self.__dict__.get('kplan') is not None:
            self.kplan.LATE = None
            self.ktables = None
        return

    def move(self, tr, src, uids):
        """ While the competing rates are drawn at the start of a step (see step_state()), those who leave are collected for the call, not moved """
        collect = self.__dict__.get('_collect')
        if collect is not None:
            if len(uids):
                collect.append((tr, src, uids))
            return
        return super().move(tr, src, uids)

    def finish_moves(self, ti):
        """ Moves made outside the call (by the agent engine, e.g. at the start, or by code): undone and taken again by it, with the moves after any events recorded since """
        if getattr(self, 'kplan', None) is None:
            return super().finish_moves(ti)
        moved, self._moved = self._moved, []
        agents, trs, srcs, kinds = [], [], [], []
        k = self.kplan
        for tr, src, uids in moved: # Back where they were, to take the move in the call
            if not len(uids):
                continue
            self.put(src.name, uids, True)
            self.flowcounts[tr.name] -= self.n_of(uids)
            ev = self.events.get(tr.name)
            if ev is not None:
                self.events[tr.name] = ev[~np.isin(ev, uids)]
            agents.append(np.asarray(uids)); trs.append(np.full(len(uids), k.tid[tr.name])); srcs.append(np.full(len(uids), self.state_index[src.name])); kinds.append(np.zeros(len(uids), np.int64))
        for a_, t_, s_ in self.event_arrivals() + self.cond_arrivals(): # Another module's events (e.g. a delivery, as it notifies this one), and the conditions that hold
            agents.append(a_); trs.append(t_); srcs.append(s_); kinds.append(np.zeros(len(a_), np.int64))
        for b in self.batches: # Events recorded since the last call (e.g. by an intervention's move): the moves a time after them
            if b.n > self.kseen and b.tr is not None and b.tr in k.tid and not b.also:
                i = k.tid[b.tr]
                if k.ae_ptr[i] < k.ae_ptr[i+1] and len(b.uids):
                    agents.append(np.asarray(b.uids)); trs.append(np.full(len(b.uids), i)); srcs.append(np.full(len(b.uids), -1)); kinds.append(np.full(len(b.uids), 2))
        if agents:
            self.run_moves(scan=False, agents=np.concatenate(agents).astype(np.int64), arr_tr=np.concatenate(trs).astype(np.int64), arr_src=np.concatenate(srcs).astype(np.int64),
                           arr_kind=np.concatenate(kinds).astype(np.int64), ti_due=ti, ti_enter=ti)
        else: # Its rounds all the same
            self.start_cascade()
        if self._news:
            self._news = False
            self.notify(ti)
        return

    def step(self):
        """ Contact-mediated transitions, as the agent engine takes them; those infected are taken to where they go by the call """
        if self.kplan is None:
            return super().step()
        mod = self.mod
        self._moved = []
        self.new_round()
        self.start_phase(2)
        self.cache.clear()
        for tr in self.graph.transmissions:
            self.events.pop(tr.name, None)
            forced = self.forced.pop(tr.name, None)
            if forced is not None:
                self.events[tr.name] = forced
                self._news = True
        pairs = [(tr, src) for tr in self.graph.transmissions for src in tr.source + (self.immune_states if not getattr(tr, 'generated', False) else [])]
        custom = self.custom_transmissions(pairs) if mod.custom_transmit else None # By the disease's own transmit()
        together = self.transmit_together(pairs) if len(pairs) > 1 and custom is None else None
        self._step_pairs = []
        k = self.kplan
        agents, trs, srcs = [], [], []
        for i, (tr, src) in enumerate(pairs):
            if custom is not None:
                uids, self._pairs = custom[i]
            else:
                uids = self.transmit(tr, src) if together is None else together[i]
            self._step_pairs += self._pairs
            if mod.infection_log is not None:
                self.log_infections(uids)
            if len(uids):
                agents.append(np.asarray(uids)); trs.append(np.full(len(uids), k.tid[tr.name])); srcs.append(np.full(len(uids), self.state_index[src.name]))
        for a_, t_, s_ in self.event_arrivals():
            agents.append(a_); trs.append(t_); srcs.append(s_)
        self._infecting = True
        try:
            if agents:
                agents = np.concatenate(agents).astype(np.int64)
                self.run_moves(scan=False, agents=agents, arr_tr=np.concatenate(trs).astype(np.int64), arr_src=np.concatenate(srcs).astype(np.int64),
                               arr_kind=np.zeros(len(agents), np.int64), ti_due=mod.ti + 1, ti_enter=mod.ti + 1, mode=1) # Infections are dated at the step's end, the next time point
            else:
                self.start_cascade()
            if self._news:
                self._news = False
                self.notify(mod.ti + 1)
        finally:
            self._infecting = False
        self.write_derived(self.sim.people.auids)
        self.cache.clear()
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]:
            self.step_register(mod.ti + 1)
        self.count_stocks()
        return
