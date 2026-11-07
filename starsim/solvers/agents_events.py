"""
Not for the user: agents taken event by event in continuous time within each step (`solver='abm-events'`; see `ss.solvers.Events`)

Used with: agents; declared modules
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import times
from .agents import Agents
from .backends import events as kernel

__all__ = ['Events']


class Events(Agents):
    """
    Runs a graph as agents, event by event in continuous time within each step (`solver='abm-events'`)

    It exists so that a model whose events are fast against a step (an epidemic on a network) needs neither small steps nor a pass over
    every edge in each of them. The engine is called once a step, as the agent engine is, and takes its agents through the step in one
    compiled pass (`ss.solvers.backends.events.step_events()`): each agent's exit at its exact time (a dwell time drawn on entry, or the
    competing rates out of its state), and the contacts of each infectious agent over the networks and mixing pools as a clock of its
    own, each thinned to its real hazard. Everything else (the networks, modifiers, rates, other modules) is held as it was at the start
    of the step, so the run is exact for values constant within each step; what the events change is read by the other modules at the
    next step. Triggers (`ss.on`) are taken after the pass, at the end of the step, by the agent engine's own sweep (see `step()`).
    Everything about agents that is not how this module moves them within a step is the agent engine's.
    """

    def init_pre(self):
        super().init_pre()
        self.refuse()
        self.scheduled = [] # The pass draws every exit (see step()), so the agent engine schedules none: someone it moves (by an intervention, or a trigger) has their exit drawn by the next step's pass
        return

    def refuse(self):
        """ What the first version does not take: a `CapabilityError` that says what, and why """
        mod = self.mod
        graph = self.graph
        sim = self.sim
        def no(what, why='which the events do not take yet'):
            raise ss.CapabilityError(f'"{mod.name}" cannot run with solver="abm-events": {what}, {why}. Run it with solver="abm" (a module can have its own, e.g. ss.Disease(..., solver="abm")).')
        if sim.pars.branching:
            no('the sim is a branching process (branching=True)')
        if self.strain is not None:
            no('it has strains')
        groups, mixing = sim.groups, sim.mixing
        if mixing.share is not None or mixing.migration is not None or mixing.register is not None:
            no('people move between places (by commuting, visiting, a force between places, migration, or a movement register)', 'which changes who meets whom, or the place people are in, in ways the events do not take yet: they take places whose people stay in them')
        if groups.group_fold is not None:
            no('a stratification applies to only some people (where=)', 'whose counts the agent engine takes again from the agents at each step, which the events do not do yet')
        for d,strat in enumerate(groups.strats):
            if strat.kind == 'disease' or any(not self.applies[s.name][d] for s in graph.people_states):
                no(f'the dimension "{strat.name}" is not one that every state of people has', 'so people take a level of it as they enter a state, which the events do not do yet')
        if graph.count_states:
            no(f'{sc.strjoin(s.name for s in graph.count_states)} {"is a state" if len(graph.count_states) == 1 else "are states"} that are not people')
        if self.attrs or self.sets or self.derived:
            no('it has attributes or clocks')
        if self.lifetables:
            no('it has a life table')
        if graph.inflows or graph.transfers:
            no('people enter it by a transition (births) or move by a transfer')
        if len(graph.transmissions) > 1:
            no(f'it has {len(graph.transmissions)} transmissions', 'and the events take one yet')
        for name,tr in graph.transitions.items():
            if tr.migration:
                no(f'"{name}" takes people out of the model by migration')
            if tr.stays:
                no(f'"{name}" has a destination that keeps people where they are (ss.stay)')
            if getattr(tr, 'min_stay', None) is not None:
                no(f'"{name}" has a minimum stay (min_stay=)')
            if any(not getattr(d, 'people', False) or d.name not in self.state_index for d in (tr.dest or [])):
                no(f'"{name}" goes to something that is not a state of people of the module')
            if self.moves(tr):
                no(f'"{name}" changes the level of a dimension (move= or increment=)')
            if self.rescaled_of(tr.source[0].name if tr.source else '') or getattr(self.dwells.get(name), 'mods', None):
                no(f'the dwell time of "{name}" has modifiers, which agents rescale when they change')
        for nkey,route in sim.networks.items():
            if isinstance(route, ss.Network):
                if type(route).net_beta is not ss.Network.net_beta:
                    no(f'"{nkey}" works out its own chance of transmission (net_beta())')
            elif self.is_matrix(route): # A contact matrix, taken as cells (see contact_cells())
                pass
            elif isinstance(route, ss.networks.MixingPools) or not isinstance(route, ss.MixingPool):
                no(f'"{nkey}" is a route ({type(route).__name__}) other than a network, an ss.MixingPool, or a contact matrix (by age bands, over the model\'s groups, or within each place)', 'such as mixing pools of its own, which the events do not take yet')
            for tr in graph.transmissions:
                if sim.modifiers.get((route.name, 'par', 'n_contacts')) or sim.modifiers.get((mod.name, 'route', f'{tr.name}@{route.name}')):
                    no(f'"{tr.name}" has a modifier on the route "{nkey}"')
        if graph.transmissions and not len(sim.networks):
            no(f'"{graph.transmissions[0].name}" is by homogeneous mixing with no route', 'whose rate reads counts that change within the step; give it a route, e.g. networks=ss.MixingPool(n_contacts=...)')
        return

    def init_post(self):
        super().init_post()
        graph = self.graph
        index = self.state_index
        self.trs = list(graph.transitions.values())
        tid = {tr.name:i for i,tr in enumerate(self.trs)}
        self.tid = tid
        ns = len(graph.people_states)
        ints = lambda x: np.array(x, dtype=np.int64)
        ptr = lambda lists: ints(np.cumsum([0] + [len(x) for x in lists]))

        # Each state's exits: competing rates, branches on entry, and timed exits
        names = [s.name for s in graph.people_states]
        haz = [self.competing[name] for name in names]
        self.haz = [tr for trs in haz for tr in trs]
        self.haz_ptr, self.haz_tr = ptr(haz), ints([tid[tr.name] for tr in self.haz])
        auids = self.sim.people.auids
        def same(edges, what):
            """ Shares that are the same for everyone, as the events take them yet: the cumulative shares (one row per branch, and its end) of the first agent """
            edges = np.asarray(edges, dtype=float).reshape(len(edges), -1)
            if edges.shape[1] > 1 and np.ptp(edges, axis=1).max() > 1e-12:
                raise ss.CapabilityError(f'"{self.mod.name}" cannot run with solver="abm-events": the shares of {what} differ between people, which the events do not take yet. Run it with solver="abm".')
            return edges[:, 0]
        brs, cums = [], []
        for state in graph.people_states:
            branches = self.branches_of(state.name)
            if branches:
                cums += list(same(self.branch_edges(state, auids), f'the branches out of {state.name}')[1:])
            brs.append(branches)
        self.br_ptr, self.br_tr, self.br_cum = ptr(brs), ints([tid[tr.name] for b in brs for tr in b]), np.array(cums, dtype=float)
        tms = [graph.exits[name].timed for name in names]
        self.tm_ptr, self.tm_tr = ptr(tms), ints([tid[tr.name] for t in tms for tr in t])

        # Each transition's dwell time: none, an exponential or Erlang with a fixed mean (drawn in the pass), or drawn for each agent at each step
        nt = len(self.trs)
        self.dw_kind, self.dw_k, self.dw_mean, self.dw_row = np.zeros(nt, np.int64), np.ones(nt, np.int64), np.zeros(nt), np.full(nt, -1, np.int64)
        self.drawn = [] # The transitions whose dwell times are drawn for each agent, each step
        for i,tr in enumerate(self.trs):
            dw = self.dwells.get(tr.name)
            if dw is None or tr._immediate or tr.contact or dw.is_rate:
                continue
            mean = self.dwell_mean(tr.name) if dw.dist is None and dw.scale is None else None
            adjust = self.adjusted(tr.name, ss.uids([0]))
            if mean is not None and np.ndim(mean) == 0 and np.ndim(adjust) == 0 and adjust == 1.0 and self.unmodified('transition', tr.name):
                self.dw_kind[i], self.dw_k[i], self.dw_mean[i] = 1, dw.k, float(mean)
            else:
                self.dw_kind[i], self.dw_row[i] = 2, len(self.drawn)
                self.drawn.append(tr)

        # Each transition's destinations, with their shares; -1 for a death
        dsts, cums = [], []
        for tr in self.trs:
            dst = [index[d.name] for d in (tr.dest or [])] + [-1]*bool(tr._outflow or tr._to_dead)
            if tr._split:
                frm = np.full(len(auids), index[tr.source[0].name])
                edges = same(self.dest_edges(tr, auids, frm), f'the destinations of "{tr.name}"')
                cum = list(edges[1:len(dst) + 1])
            else:
                cum = list((np.arange(len(dst)) + 1)/max(len(dst), 1))
            dsts.append(dst)
            cums += cum
        self.dst_ptr, self.dst_state, self.dst_cum = ptr(dsts), ints([d for x in dsts for d in x]), np.array(cums, dtype=float)
        self.inf_tr = tid[graph.transmissions[0].name] if graph.transmissions else -1

        # Each agent's state, version, and scheduled exit, as the pass left them
        self.kstate = np.full(0, -1, np.int64)
        self.ver = np.zeros(0, np.int64)
        self.t_exit = np.zeros(0)
        self.exit_tr = np.zeros(0, np.int64)
        return

    def step_state(self):
        """ At the start of the step: the newborn, and the moves interventions made in the last step, which are news now; the declared transitions are all taken in step() """
        self._moved = []
        self.new_round()
        self.start_phase(0)
        self.place_newcomers()
        self.renew_events()
        return

    def resize(self, n):
        """ Room for agents made since the last step """
        m = len(self.kstate)
        if m < n:
            self.kstate = np.concatenate([self.kstate, np.full(n - m, -1, np.int64)])
            self.ver = np.concatenate([self.ver, np.zeros(n - m, np.int64)])
            self.t_exit = np.concatenate([self.t_exit, np.full(n - m, np.inf)])
            self.exit_tr = np.concatenate([self.exit_tr, np.full(n - m, -1, np.int64)])
        return

    def act_weights(self, tr, nkey, route, hazards):
        """
        On a network with acts: the acts of each edge in the step, taken as spread evenly through it, so each edge's hazard over the step is
        held for it, as `transmit_acts()` has it (the edge's weight, times its acts, times the hazard per act in its direction for the source's
        state). Returns each source state's share of the largest hazard per act, and the hazard rate of each edge from p1 to p2 and from p2 to p1
        """
        hazards = {state: np.asarray(h, dtype=float) for state,h in hazards.items()} # By direction
        if any(h.ndim > 1 for h in hazards.values()):
            raise ss.CapabilityError(f'"{self.mod.name}" cannot run with solver="abm-events": its chance per act of "{tr.name}" on "{nkey}" differs between people, which the events do not take yet. Run it with solver="abm".')
        top = np.max(list(hazards.values()), axis=0)
        on = top > 0
        shares = {}
        for state,h in hazards.items():
            ratio = h[on]/top[on]
            if len(ratio) and np.ptp(ratio) > 1e-12:
                raise ss.CapabilityError(f'"{self.mod.name}" cannot run with solver="abm-events": the chances per act of the source states of "{tr.name}" on "{nkey}" are not in proportion in every direction, which the events do not take yet. Run it with solver="abm".')
            shares[state] = float(ratio[0]) if len(ratio) else 0.0
        edges = route.edges
        p1, p2 = np.asarray(edges.p1), np.asarray(edges.p2)
        rate = np.asarray(edges.beta, dtype=float)*np.asarray(edges.acts, dtype=float)/self.dt # The acts of the step, spread through it
        w1 = rate*top[route.edge_directions(p1, p2)]
        w2 = w1 if route.one_way or not route.sides else rate*top[route.edge_directions(p2, p1)]
        return shares, w1, w2

    def is_matrix(self, route):
        """ Whether a route mixes by a contact matrix between cells everyone in which has the same contacts: within each place or over the model's dimensions (`sim.mixing.pool_terms`), or by age bands of its own (`ss.contacts(ages=...)`) """
        return route.name in self.sim.mixing.pool_terms or (isinstance(route, ss.networks.MixingPools) and not route.pools and route.edges is not None)

    def contact_cells(self, route, n):
        """
        A contact matrix as cells (see `is_matrix()`): the cell of each agent (a level of the matrix, in their place if there are places; -1 for
        none), the matrix between the levels (rows: the level reached), and the number of cells, as the agent engine's exposure has them
        (see `ss.solvers.Mixing.term_exposure()` and `ss.networks.MixingPools.band_exposure()`)
        """
        sim = self.sim
        mixing = sim.mixing
        auids = sim.people.auids
        cell = np.full(n, -1, np.int64)
        term = mixing.pool_terms.get(route.name)
        if term is None: # By age bands of its own, from the first (younger people have no contacts)
            cell[auids] = sim.people.age_bins(route.edges)[auids] - 1
            mat = float(route.pars.beta)*np.asarray(route.pars.n_contacts, dtype=float).T # Indexed [source, destination]
            return cell, mat, len(mat)
        if term.kind == 'homogeneous': # Everyone with everyone in their own place
            of_group, mat, size = mixing.places, np.array([[float(term.contacts)]]), sim.groups.strats[mixing.space].n
        else:
            of_group = term.cells if 'cells' in term else sim.groups.group_coords[:, term.dim]
            mat = np.asarray(term.matrix, dtype=float)
            size = term.get('size', len(mat))
        cell[auids] = np.asarray(of_group)[sim.groups.group_raw()[auids]]
        return cell, mat, size

    def routes(self, n, everyone, sus_max):
        """ The routes of the transmission for the pass: each network's neighbors, each pool's members, and each contact matrix's cells, with the coefficients and bounds that its contacts are drawn and thinned at """
        sim = self.sim
        mod = self.mod
        tr = self.graph.transmissions[0] if self.graph.transmissions else None
        ns = len(self.graph.people_states)
        out = sc.objdict(kind=[], x_off=[], r_n=[], deg=[], scale=[], bound=[], share=[], lists=[], w1=[], w2=[], beta_off=[], members=[], mem_eff=[], mem_off=[], mem_n=[], ctc=[],
                         a_cell=[], c_ptr=[], c_first=[], c_row=[], c_cum=[])
        n_x = n_beta = n_mem = n_cell = n_row = n_ac = 0
        blks = {} # The number of cells a contact on each contact matrix can reach
        self.route_nets = [] # The index in sim.networks of each route of the pass, for who infected whom
        self._route_memo = {}
        try:
            for i,(nkey,route) in enumerate(sim.networks.items() if tr is not None else []):
                is_net = isinstance(route, ss.Network)
                if not is_net and mod not in route.diseases:
                    continue
                acts = is_net and route.has_acts
                with self.evaluating(memo={} if len(tr.terms) > 1 else None):
                    coefs = {state: self.route_beta(coef, nkey, route, None if acts else everyone) for state,coef in tr.terms.items()} # On a network with acts, by direction, as transmit_acts() has them
                if acts:
                    coefs, w1, w2 = self.act_weights(tr, nkey, route, coefs)
                if any(np.ndim(c) for c in coefs.values()):
                    raise ss.CapabilityError(f'"{mod.name}" cannot run with solver="abm-events": the coefficient of "{tr.name}" on "{nkey}" differs between people, which the events do not take yet. Run it with solver="abm".')
                ref = max([float(c) for c in coefs.values()] + [0.0])
                if ref <= 0:
                    continue
                elif not np.isfinite(ref):
                    raise ss.CapabilityError(f'"{mod.name}" cannot run with solver="abm-events": the coefficient of "{tr.name}" on "{nkey}" is a probability of 1, an infinite rate, which has no time to the next event. Run it with solver="abm", where it is a certain transmission.')
                share = np.zeros(ns)
                for state,c in coefs.items():
                    share[self.state_index[state.name]] = float(c)/ref*float(state.rel_trans)
                ctc = np.zeros(n) # Each agent's contacts on a pool or contact matrix (0 if not a source there)
                if is_net:
                    if not acts:
                        w1 = w2 = np.asarray(route.edges.beta, dtype=float)
                    index = route.edge_index(now=True) # The edges of each agent, kept while the edges are the same
                    if index is None: # No edges
                        index = [np.zeros(1, np.int64)] + [np.zeros(0, np.int64)]*2 + [np.zeros(1, np.int64)] + [np.zeros(0, np.int64)]*2
                    bound = sus_max*(max(float(w1.max()), float(w2.max()) if w2 is not w1 else 0.0) if len(w1) else 0.0) # A beta raised in place above the largest is caught by the pass (see step())
                    scale = ref
                    out.lists.append(index); out.w1.append(w1); out.w2.append(w2)
                    s1, s2 = index[0], index[3]
                    out.x_off.append(n_x); out.r_n.append(len(s1) - 1); out.deg.append(np.append(np.diff(s1) + (np.diff(s2) if not route.one_way else 0), 0)) # Each agent's neighbors, once a step, with the same length as s1 so x_off is the place in both
                    out.beta_off.append(n_beta); out.mem_off.append(0); out.mem_n.append(0)
                    n_x += len(index[0]); n_beta += len(w1)
                elif self.is_matrix(route): # A contact to a cell drawn by its weight, then to one of its members
                    cell, mat, size = self.contact_cells(route, n)
                    blk = len(mat)
                    inside = np.flatnonzero(cell >= 0)
                    count = np.bincount(cell[inside], minlength=size)
                    order = inside[np.argsort(cell[inside], kind='stable')] # The members of each cell, in turn
                    nb = count.reshape(-1, blk).astype(float) # By the cells a contact can reach: each place's (or one set), and the level in it
                    cum = np.cumsum(mat.T[None, :, :]*nb[:, None, :], axis=2) # [set, level of the source, level reached]: the contacts with each level, by its number
                    with np.errstate(invalid='ignore', divide='ignore'):
                        rate = np.where(nb > 0, cum[:, :, -1]/nb, 0.0).ravel() # As the hazard of a group in a compartmental run: the matrix times the share infectious of each level, over their number
                    out.a_cell.append(np.where(cell >= 0, cell + n_cell, -1)); out.c_ptr.append(n_mem + np.append(0, np.cumsum(count))[:-1])
                    ctc[inside] = rate[cell[inside]]
                    out.c_first.append(n_cell + np.arange(size)//blk*blk); out.c_row.append(n_row + np.arange(size)*blk); out.c_cum.append(cum.ravel())
                    out.members.append(order); out.mem_eff.append(np.ones(len(order))); out.mem_off.append(n_mem); out.mem_n.append(len(order)); out.x_off.append(n_ac); out.r_n.append(0); out.beta_off.append(0)
                    n_mem += len(order); n_cell += size; n_row += size*blk; n_ac += n
                    bound = sus_max
                    scale = ref
                    blks[len(out.kind)] = blk
                else:
                    dst, srcs = np.asarray(route.target_uids, dtype=np.int64), np.asarray(route.source_uids, dtype=np.int64)
                    if not len(dst) or not len(srcs):
                        continue
                    eff = np.asarray(route.eff_contacts.raw, dtype=float)[:n]
                    bound = sus_max*eff[dst].max()
                    scale = float(route.pars.beta)*ref/len(srcs)
                    ctc[srcs] = len(dst)
                    out.members.append(dst); out.mem_eff.append(eff[dst]); out.mem_off.append(n_mem); out.mem_n.append(len(dst)); out.x_off.append(0); out.r_n.append(0); out.beta_off.append(0)
                    n_mem += len(dst)
                out.kind.append(0 if is_net else 2 if self.is_matrix(route) else 1)
                self.route_nets.append(i)
                out.scale.append(scale)
                out.bound.append(bound)
                out.share.append(share)
                out.ctc.append(ctc)
        finally:
            self._route_memo = None
        cat = lambda arrs, dtype: np.concatenate(arrs).astype(dtype, copy=False) if len(arrs) > 1 else np.asarray(arrs[0], dtype=dtype) if arrs else np.zeros(1, dtype=dtype)
        nr = len(out.kind)
        index = [cat([ix[k] for ix in out.lists], np.int64) for k in range(6)] # s1, e1, o1, s2, e2, o2 of every network, end to end
        w1 = cat(out.w1, float)
        w2 = w1 if all(a is b for a,b in zip(out.w1, out.w2)) else cat(out.w2, float) # The same weight both ways, unless a network has acts
        return (np.array(out.kind, np.int64), np.array(out.x_off, np.int64), np.array(out.r_n, np.int64), np.array(out.scale, float), np.array(out.bound, float),
                np.array(out.share, float).reshape(nr, ns), *index, cat(out.deg, np.int64), w1, w2,
                np.array(out.beta_off, np.int64), cat(out.members, np.int64), cat(out.mem_eff, float), np.array(out.mem_off, np.int64), np.array(out.mem_n, np.int64),
                np.array(out.ctc, dtype=float).reshape(nr, n),
                np.array([blks.get(r, 0) for r in range(nr)], np.int64), cat(out.a_cell, np.int64), np.append(cat(out.c_ptr, np.int64)[:n_cell], n_mem).astype(np.int64),
                cat(out.c_first, np.int64), cat(out.c_row, np.int64), cat(out.c_cum, float))

    def step(self):
        """ The declared transitions of the step, event by event, then their moves written to the agents' states, flows and events """
        mod = self.mod
        sim = self.sim
        people = sim.people
        self._moved = []
        self.new_round()
        self.start_phase(2)
        self.cache.clear() # The interventions have run since the step started
        for tr in self.graph.transmissions:
            self.events.pop(tr.name, None)
            forced = self.forced.pop(tr.name, None) # Infected by code (disease.infect()) since the last step's: news now
            if forced is not None:
                self.events[tr.name] = forced
                self._news = True
        n = len(mod.rel_sus.raw)
        self.resize(n)
        alive = people.alive.raw[:n]
        names = [s.name for s in self.graph.people_states]
        cur = np.full(n, -1, np.int64)
        for k,name in enumerate(names):
            cur[mod[name].raw[:n]] = k
        cur[~alive] = -1
        fresh = cur != self.kstate # Put in a state since the last step (by an intervention, by code, at the start, or born), or gone
        self.t_exit[fresh] = np.inf
        self.exit_tr[fresh] = -1
        fresh &= cur >= 0
        state = cur.copy()

        # The values held for the step
        auids = people.auids
        everyone = ss.uids(np.arange(n))
        rate_rows, haz_rate, haz_row = [], np.zeros(len(self.haz)), np.full(len(self.haz), -1, np.int64)
        for j,tr in enumerate(self.haz):
            with self.evaluating(memo={}):
                rate = self.eval_for(tr.rate, auids)*self.adjusted(tr.name, auids)
            if np.ndim(rate) == 0:
                haz_rate[j] = max(float(rate), 0.0)
            else:
                row = np.zeros(n)
                row[auids] = np.maximum(rate, 0)
                haz_row[j] = len(rate_rows)
                rate_rows.append(row)
        rate_rows = np.array(rate_rows).reshape(len(rate_rows), n)
        pool = np.zeros((len(self.drawn), n))
        for i,tr in enumerate(self.drawn): # One for each agent, for an entry within the step
            self.new_round(tr.name)
            pool[i, auids] = self.sample_dwell(tr, auids)
        sus_state, sus_agent, w_agent, sus_max = np.zeros(len(names)), np.zeros(n), np.zeros(n), 0.0
        if self.graph.transmissions:
            tr = self.graph.transmissions[0]
            for src in tr.source:
                sus_state[self.state_index[src.name]] = float(src.rel_sus)
            sus_agent = np.broadcast_to(np.asarray(times(self.current_value('rel_sus'), self.mult_raw('transition', tr.name)), dtype=float), n)[:n].copy()
            sus_agent[~alive] = 0.0
            sus_max = (sus_agent.max() if n else 0.0)*sus_state.max()
            w_agent = np.broadcast_to(np.asarray(self.current_value('rel_trans'), dtype=float), n)[:n].copy()
        routes = self.routes(n, everyone, sus_max)

        # The pass
        dt = self.dt
        t0 = self.ti*dt
        seed = np.random.SeedSequence([int(self.dists.init.seed), int(self.step_key())]).generate_state(1, np.uint64)[0] # The step's key: by the module's seed and the step, so each agent's draws in it are its own (see backends.events.rand())
        e_a, e_tr, e_src, e_dst, e_t, e_by, e_rt, over = kernel.step_events(t0, t0 + dt, seed, state, self.ver, self.t_exit, self.exit_tr, fresh,
            self.haz_ptr, self.haz_tr, haz_rate, haz_row, rate_rows, self.br_ptr, self.br_tr, self.br_cum, self.tm_ptr, self.tm_tr,
            self.dw_kind, self.dw_k, self.dw_mean, self.dw_row, pool, self.dst_ptr, self.dst_state, self.dst_cum, self.inf_tr,
            sus_state, sus_agent, w_agent, *routes)
        self.kstate = state
        if over:
            raise RuntimeError(f'"{mod.name}" took a contact whose chance of infecting was above the bound it was drawn at: an edge\'s beta was raised in place (net.edges.beta[...] = ...); give the network new edge arrays instead, e.g. net.edges.beta = new_beta')

        # Written to the agents: who is in each state, the flows, and the events
        moved = np.unique(e_a)
        before, after = cur[moved], state[moved]
        for k,name in enumerate(names):
            left = moved[(before == k) & (after != k)]
            entered = moved[(before != k) & (after == k)]
            if len(left):
                self.put(name, ss.uids(left), False)
            if len(entered):
                self.put(name, ss.uids(entered), True)
        infected = e_by >= 0 # Who infected whom, and over which route, as the agent engine keeps them (see sources())
        nets = np.asarray(self.route_nets, dtype=ss.dtypes.int)[e_rt[infected]] if len(self.route_nets) else np.zeros(0, dtype=ss.dtypes.int)
        self._step_pairs = [(ss.uids(e_by[infected]), ss.uids(e_a[infected]), nets)]
        if mod.infection_log is not None:
            self.log_infections(ss.uids(e_a[infected]), sources=e_by[infected], networks=nets) # Each infection with its own source, including someone infected twice in the step
        ng = sim.groups.n_groups
        if len(e_a):
            order = np.lexsort((e_dst, e_tr))
            e_a, e_tr, e_src, e_dst = e_a[order], e_tr[order], e_src[order], e_dst[order]
            keys = e_tr*(len(names) + 1) + e_dst + 1 # One group for each transition and destination (-1, a death, included)
            cuts = np.flatnonzero(np.diff(keys)) + 1
            for lo,hi in zip(np.r_[0, cuts], np.r_[cuts, len(keys)]):
                tr = self.trs[e_tr[lo]]
                dst = self.graph.people_states[e_dst[lo]] if e_dst[lo] >= 0 else None
                uids = ss.uids(e_a[lo:hi])
                self.flowcounts[tr.name] += self.n_of(uids)
                key = f'{tr.name}.{"dead" if dst is None else dst.name}'
                if key in self.flowcounts:
                    self.flowcounts[key] += self.n_of(uids)
                if ng > 1: # By group, as the agent engine counts them (every state has every dimension: see refuse())
                    groups = self.groups(uids)
                    self.add_flow_groups(tr.name, groups, uids)
                    if key in self.flowcounts:
                        self.add_flow_groups(key, groups, uids)
                self.events[tr.name] = uids if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], uids])
                self.record(tr, uids, e_src[lo:hi], e_dst[lo])
                if dst is not None:
                    self.stamp(dst.name, uids, mod.ti + 1) # Dated at the end of the step it happened in, as an infection is
                else: # Died: at the end of the step, as by the agent engine's own outflows (see send())
                    people.request_death(uids, cause=mod)
                    srcs = sorted(set(e_src[lo:hi]))
                    self._died.append((tr, [self.graph.people_states[k] for k in srcs], [uids[e_src[lo:hi] == k] for k in srcs]))
        if self.triggers: # Who ss.on() moves, by the events of the step: at its end, as the agent engine's sweep takes them (see finish_moves())
            self._moved = []
            self.cache.clear()
            self.finish_moves(mod.ti + 1)
        if self._news:
            self._news = False
            self.notify(mod.ti + 1)
        self.cache.clear()
        self.count_stocks()
        return
