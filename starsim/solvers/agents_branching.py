"""
Not for the user: the agent engine, continued: branching processes of agents: agents only for the infected, and transmission between them (see `ss.solvers.Agents`)

Used with: agents; declared modules
"""
import functools
import operator
import numpy as np
import sciris as sc
import starsim as ss
from .engine import times
from .compartmental import Compartmental
from .agents_moving import check_draws

__all__ = ['Branching']

class Branching:
    """
    The part of the agent engine for branching processes of agents: agents only for the infected, and transmission between them

    It exists so that the uninfected kept as counts (`init_branching()`) can be read apart from the rest of the agent engine (`Agents`).
    """
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
            if isinstance(route, (ss.MixingPool, ss.MixingPools)) and mod not in route.diseases: # Not one of the diseases it is for
                continue
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
            elif isinstance(route, ss.MixingPools) and not route.pools: # A matrix of numbers: from the share of each group or band that is infectious
                any_pools = True
                for k,strain in enumerate(strains):
                    share = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                    auids, expo = route.exposure(rel_trans.raw*share)
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
            if not isinstance(route, ss.Network) or route.has_acts or any(sim.modifiers.get((mod.name, 'route', f'{tr.name}@{route.name}')) for tr,src in pairs):
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
        and modifiers all scale the hazard of each act.
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
        weight = lambda inds: route.edges.beta[inds]*route.edges.acts[inds]
        prob = lambda d, inds, src, trg: -np.expm1(-weight(inds)*act_hazard(src, trg)*trans[src]*rel_sus.raw[trg]*mult(inds, trg))
        out = route.transmit(trans*source, rel_sus.raw, lambda d, inds, src, trg: weight(inds), mod.trans_rng, prob=prob)
        for srcs, trgs in out:
            new_cases.append(trgs)
            pairs.append((srcs, trgs, i))
        return
