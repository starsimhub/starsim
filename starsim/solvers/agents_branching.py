"""
Not for the user: the agent engine, continued: branching processes of agents: agents only for the infected, and transmission between them (see `ss.solvers.Agents`)

Used with: agents; declared modules
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import times
from .compartmental import Compartmental

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
                if self.mod.infection_log is not None and any(state.name == name for tr in self.graph.transmissions for state in tr.dest):
                    self.log_infections(uids, seeds=True)
        self.held_trs = [tr for tr in self.graph.transitions.values() if tr.people and not tr.acquisition and (all(src.name in self.uninfected for src in tr.source) if tr.source else tr._inflow and all(dst.name in self.uninfected for dst in tr.dest))] # The uninfected side's own flows
        back = [tr.name for tr in self.graph.transitions.values() if tr.source and any(src.name not in self.uninfected for src in tr.source) and any(dst.name in self.uninfected for dst in tr.dest or [])]
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
        self.infected_ti[uids] = self.mod.ti + 1 if self._started else 0 # At the end of the step, the next time point
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
                            rate = float(np.max(self.eval(tr.rate)))
                        mean = 1/rate if rate > 0 else np.inf
                except (ValueError, TypeError): # Not every exit has a mean: a rate that is a function of the agents (which raises a ValueError, e.g. ss.CapabilityError, or a TypeError, by group) is left out
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
        cdf = self.sim.contact_rates.age_cdf(strat)
        top = max(110, strat.edges[-1] + 10)
        xs = np.linspace(0, top, 4*int(top) + 1)
        F = cdf(xs)
        edges = np.append(strat.edges, top)
        lo, hi = cdf(edges[levels]), cdf(edges[levels + 1])
        return np.interp(lo + self.branch_rng.random(len(levels))*(hi - lo), F, xs)

    def held_rel_sus(self):
        """ The mean susceptibility of the uninfected: of a trait that varies between people, its mean """
        a = next((a for a in self.traits if a.name == 'rel_sus'), None)
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
        mixing = sim.contact_rates
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
                    raise ss.CapabilityError(f'a modifier of "{self.mod.name}.{tr.name}" has a condition on people, which the uninfected of a branching process, who are counts by group, cannot take; give it no who=, or run with a compartmental solver')
                if tr._inflow:
                    inflow = np.broadcast_to(np.asarray(self.eval(tr.rate), dtype=float)*mult*self.inflow_split(tr), ng)*dt
                    if getattr(tr, 'per_capita', False):
                        inflow = inflow*self.held_n()
                    sources = [None]
                else:
                    hazard = 1/self.dwell_mean(tr.name) if tr.dur is not None else np.asarray(self.eval(tr.rate), dtype=float)
                    sources = [src.name for src in tr.source]
                total = np.zeros(ng)
                for name in sources:
                    if name is None:
                        amount = inflow
                    else:
                        amount = np.broadcast_to(self.held[name], ng)*-np.expm1(-np.maximum(hazard*mult, 0)*dt)
                        self.held[name] = np.broadcast_to(self.held[name], ng) - amount
                    for dst,p in zip(tr.dest or [], self.branch_probs(tr)):
                        if dst.name in self.held:
                            self.held[dst.name] = np.broadcast_to(self.held[dst.name], ng) + amount*p
                    total = total + amount
                self.flowcounts[tr.name] += float(total.sum())
        return
