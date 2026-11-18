"""
Not for the user: the agent engine, continued: transmission (see `ss.solvers.Agents`): over networks and mixing pools, with strains,
and by homogeneous mixing; `Transmission` is also the transmission that diseases written as code use, so that both give the same numbers

Used with: agents; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import Effects, times
from .agents_moving import ROUNDS, ROUND_MAX, step_key, check_draws

__all__ = ['Transmission', 'Transmitting']


class Transmission(Effects):
    """
    Transmission as agents, the same for a declared disease (`Agents`, for its transmissions) and one written as code (`ss.Disease.transmit(rel_trans=..., rel_sus=...)`,
    through its engine, `ss.solvers.Undeclared`): who can transmit and be infected, with the modifiers and products on `rel_trans` and `rel_sus`
    applied (`current_value()`); who is infected over a network (`over_network()`); and who infected whom (`first_infections()`)

    Where the two differ, the caller says how:

    - **Hazard or probability.** `over_network()` takes the chance of an edge in the step, and scales its hazard by every other factor (the edge's
      weight and acts, `rel_trans`, `rel_sus`, and the modifiers). A declared disease gives 1 − exp(−β·dt) from its rate, or, on a network with acts,
      works out a hazard per act (`Agents.transmit_acts()`); one written as code gives its beta as the probability in the step, converting a rate the
      same way (`ss.Rate.to_prob()`).
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
        Who is infected over a network in the step: an edge from `u` to `v` transmits with probability 1 − exp(−H), where H is the hazard of
        the network's chance `net_beta(beta)` (with the edge's weight, and acts if it has them), times `rel_trans[u]`, `rel_sus[v]`, and the
        modifiers on the route for `v`: every factor scales the hazard, so a contact infects at most once in the step, wherever an effect is
        put (see `ss.Network.transmit()`), as for a declared disease on a network with acts (see `transmit_acts()`)

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
        def edge_hazard(d, inds, src, trg): # For the edges that might transmit
            beta = betas[d]
            if isinstance(beta, np.ndarray): # A probability per act for each direction of the network
                beta = beta[route.edge_directions(src, trg)]
            elif not beta: # No random numbers are drawn
                return None
            if isinstance(beta, ss.Rate): # A disease written as code can give its beta as a rate
                beta = beta.to_prob(self.mod.t.dt)
            hazard = route.net_hazard(disease_beta=beta, inds=inds, disease=self.mod)
            m = mult(inds, trg)
            if np.ndim(m) == 0 and m == 1: # The usual case, no modifiers on the route
                return hazard
            with np.errstate(invalid='ignore'): # An infinite hazard times a modifier of 0 is none (NaN never transmits)
                return hazard*m
        return [(srcs, trgs, i) for srcs,trgs in route.transmit(rel_trans, rel_sus, edge_hazard, self.mod.trans_rng)]

    @staticmethod
    def first_infections(pairs):
        """
        Each agent infected, once, with who infected them and over which route: the first of `pairs` to reach them. Each pair is the sources, the
        targets, and the index of the route (one, or one for each), and any more values (e.g. a strain); returns the targets (sorted), and for
        each, the source (`ss.dtypes.int_nan` where not known), the route, and the more values, of the first pair to reach them
        """
        if not pairs:
            return ss.uids(), ss.uids(), np.empty(0, dtype=ss.dtypes.int)
        trgs, srcs, *rest = Transmission.stack_pairs(pairs)
        trgs, first = trgs.unique(return_index=True)
        return trgs, srcs[first], *[col[first] for col in rest]

    @staticmethod
    def stack_pairs(pairs):
        """ Pairs of who infected whom (each the sources, the targets, and the index of the route, one or one for each, and any more values), as one column each: the targets, the sources, and the rest """
        trgs = ss.uids.concatenate([pair[1] for pair in pairs])
        srcs = ss.uids.concatenate([pair[0] for pair in pairs])
        rest = [np.concatenate([np.broadcast_to(np.asarray(pair[c], dtype=ss.dtypes.int), len(pair[1])) for pair in pairs]) for c in range(2, len(pairs[0]))]
        return trgs, srcs, *rest

    def step_draw(self, key, uids):
        """ Uniforms for some agents from one of the transmission's distributions (`self.dists[key]`), keyed by the step alone, so the same agents get the same numbers, declared or written as code; without common random numbers, its next draws """
        dist = self.dists[key]
        if not len(uids):
            return np.empty(0)
        elif not ss.options.crn:
            return dist.rvs(uids)
        slots = self.sim.people.slot_raw()
        return ss.dists.hash_uniforms(dist.seed, ROUNDS + step_key(self.mod)*ROUND_MAX, uids if slots is None else slots[uids], dtype=dist.hash_dtype, even=dist.even)

    def pick_sources(self, pairs, name):
        """
        With strains, each agent reached in the step, once, with who infected them and the rest of that pair's values: one of those who reached
        them, each as likely (the first to happen is as likely to be any of them), drawn from `pick_<name>`. Each pair is the sources, the
        targets, the index of the route (one, or one for each), and any more values (e.g. a strain), as for `first_infections()`
        """
        trgs, *cols = self.stack_pairs(pairs)
        order = np.argsort(trgs, kind='stable')
        trgs, first, counts = np.unique(trgs[order], return_index=True, return_counts=True)
        if np.any(counts > 1): # Reached by several in the step
            first = first + np.minimum((self.step_draw(f'pick_{name}', ss.uids(trgs))*counts).astype(int), counts - 1)
        return ss.uids(trgs), *[col[order][first] for col in cols]

    def unprotected(self, name, uids, kept):
        """ Which of some agents reached by a strain are infected by it: each with the chance `kept`, the fraction of the force of infection of the strain that reaches them (1 − their protection against it), drawn from `protect_<name>` """
        return self.step_draw(f'protect_{name}', uids) < kept


    # Transmission written as code: what ss.Disease.transmit() and the methods it calls do, so that each can be overridden on its own

    def over_routes(self, rel_trans, rel_sus, betamap):
        """ The infections over each of the sim's routes in turn (`code_route()`), each a tuple of targets, sources, and network IDs: see `ss.Disease.transmit()` """
        return [self.code_route(i, route, betamap[ss.networks.standardize_netkey(nkey)], rel_trans, rel_sus) for i,(nkey,route) in enumerate(self.sim.networks.items())]

    def strain_cases(self, rel_trans, rel_sus, betamap):
        """
        The infections of a disease written as code with strains (`ss.AgentDisease(strains=...)`), as for a declared disease's: every strain in
        one pass, at the infectiousness of the strain each carries (`rel_trans` of `ss.Strains`); each new case is infected by one of those who
        reached them, at random (`pick_sources()`), whose strain they take, unless protected against it (`dis.strain_sus()`, `unprotected()`)

        Returns:
            The new cases, and for each the source, the route, and the index of the strain
        """
        dis = self.mod
        strains = dis.strains
        held = dis.strain.raw
        rel = strains.values['rel_trans'] if 'rel_trans' in strains.values else np.ones(len(strains))
        trans = np.asarray(rel_trans.raw if isinstance(rel_trans, ss.Arr) else rel_trans)*np.where(held >= 0, rel[np.maximum(held, 0)], 0.0) # Nobody without a strain transmits
        out = []
        for i,(nkey,route) in enumerate(self.sim.networks.items()):
            betas = betamap[ss.networks.standardize_netkey(nkey)]
            if isinstance(route, ss.Network): # On a network, each case has a source, whose strain they take
                trgs, srcs, nets = self.code_route(i, route, betas, trans, rel_sus)
                out.append((srcs, trgs, nets, held[srcs]))
            else: # Another route gives no sources, so each strain is transmitted over it by those who carry it
                for k in range(len(strains)):
                    trgs, srcs, nets = self.code_route(i, route, betas, trans*(held == k), rel_sus)
                    out.append((srcs, trgs, nets, k))
        out = [pair for pair in out if len(pair[1])]
        if not out:
            return ss.uids(), ss.uids(), np.empty(0, dtype=ss.dtypes.int), np.empty(0, dtype=ss.dtypes.int)
        new, sources, routes, strain = self.pick_sources(out, 'infection')
        kept = np.ones(len(new))
        for k in np.unique(strain):
            kept[strain == k] = np.broadcast_to(dis.strain_sus(k), trans.shape)[new[strain == k]]
        ok = self.unprotected('infection', new, kept)
        return new[ok], sources[ok], routes[ok], strain[ok]

    def code_route(self, i, route, betas, rel_trans, rel_sus):
        """
        The transmission of a disease written as code along a single network (or other route): see `ss.Disease.transmit()`

        On a network, each edge transmits with probability 1 − (1 − `net_beta()`)^(`rel_trans·rel_sus`·modifiers), the modifiers being those
        of the network's contacts and of `rel_sus` on the route (`ss.modify(..., network=)`) (see `ss.Network.transmit()`), as for a declared disease
        run as agents, by the same code (`over_network()`), with beta as the probability in the step. The modifiers of `rel_trans` and `rel_sus` (including products) are
        applied here.
        """
        dis = self.mod
        raw = lambda x: x.raw if isinstance(x, ss.Arr) else np.asarray(x) # An array of the disease's, or of values by UID
        rel_trans = times(raw(rel_trans), self.mult_raw('rel_trans'))
        rel_sus = times(raw(rel_sus), self.mult_raw('rel_sus'))

        # Main use case: networks
        if isinstance(route, ss.Network):
            pairs = self.over_network(i, route, rel_trans, rel_sus, betas, 'rel_sus')

        # Handle everything else: mixing pools, environmental transmission, etc.
        elif isinstance(route, ss.networks.Route):
            beta = betas[0].to_prob(dis.t.dt) if isinstance(betas[0], ss.Rate) else betas[0] # Mixing pools are unidirectional, so only the first beta is used
            rel_sus = times(rel_sus, self.route_mult('rel_sus', route)(None, slice(None)))
            as_arr = self.sim.people.age.asnew # Values by UID, as an array of the people
            target_uids = route.compute_transmission(as_arr(rel_sus, copy=False), as_arr(rel_trans, copy=False), beta, disease=dis)
            pairs = [(np.full(len(target_uids), dtype=ss.dtypes.int, fill_value=ss.dtypes.int_nan), target_uids, i)]
        else:
            errormsg = f'Cannot compute transmission via route {type(route)}; please subclass ss.networks.Route and define a compute_transmission() method'
            raise TypeError(errormsg)

        new_cases = ss.uids.concatenate([pair[1] for pair in pairs])
        sources = ss.uids.concatenate([pair[0] for pair in pairs])
        networks = np.full(len(new_cases), dtype=ss.dtypes.int, fill_value=i)
        return new_cases, sources, networks

    def code_betas(self, beta=None):
        """ The beta of a disease written as code (`beta`, by default its parameter `beta`) for each of the sim's networks (by name), as a pair (p1 to p2, p2 to p1), checked against them (see `ss.Disease.transmit()`) """
        dis = self.mod
        sim = self.sim
        β = dis.pars.beta if beta is None else beta
        unit = getattr(sim, 'unit', None) # None if time is counted in steps, when a bare number is a probability per step

        def scalar_beta(β):
            return isinstance(β, ss.Rate) or sc.isnumber(β)

        std = ss.networks.standardize_netkey
        nets = {std(k):net for k,net in sim.networks.items()}
        def pair_for(nkey, b):
            """
            A network's beta as a pair [p1 to p2, p2 to p1]: a probability per act as it resolves on the network (an array of a value for each of its directions,
            see `code_route()`); a single beta in both directions, except on a maternal network, from mother to child only; or a pair as given. A bare number
            is a rate in the sim's unit, as in a declared disease (on a network with acts, a probability per act)
            """
            net = nets.get(nkey)
            if isinstance(b, ss.peract):
                vals = b.resolve(net) if net is not None else b.values
                return vals*2 if len(vals) == 1 else [np.asarray(vals, dtype=float)]*2
            in_unit = lambda x: ss.per(x, unit) if sc.isnumber(x) and x and unit and not getattr(net, 'has_acts', False) else x
            if scalar_beta(b):
                b = in_unit(b)
                return [b, b*0] if getattr(net, 'one_way', False) else [b, b]
            return [in_unit(x) for x in b]

        if scalar_beta(β) or isinstance(β, ss.peract): # The same for every network
            items = [(k, β) for k in sim.networks.keys()]
        elif isinstance(β, dict): # By network; a default is the beta of every network it does not name, and the names that are not networks are not used
            if 'default' in β:
                unknown = [k for k in β if k != 'default' and std(k) not in nets]
                if unknown:
                    ss.note(f'the beta of "{dis.name}" names {sc.strjoin(unknown)}, which {"is not a network" if len(unknown) == 1 else "are not networks"} of the sim, so {"it is" if len(unknown) == 1 else "they are"} not used; the networks not named take the default', sim=sim)
                β = {k:ss.networks.route_beta(β, k) for k in sim.networks.keys()}
            items = β.items()
        else:
            errormsg = f'Invalid type {type(β)} for beta'
            raise TypeError(errormsg)
        betamap = {std(k):pair_for(std(k), b) for k,b in items}

        # Check that it matches the network
        netkeys = [std(k) for k in sim.networks.keys()]
        if set(betamap.keys()) != set(netkeys):
            missing = sorted(set(netkeys) - set(betamap.keys()))
            extra = sorted(set(betamap.keys()) - set(netkeys))
            errormsg = f'Network keys ({netkeys}) and beta keys ({list(betamap.keys())}) do not match for disease "{dis.name}"'
            if missing: errormsg += f'; missing beta for network(s) {missing}'
            if extra:   errormsg += f'; no network(s) found matching {extra}'
            raise ValueError(errormsg)

        return betamap


class Transmitting:
    """
    The part of the agent engine that works out who is infected in a step by each transmission: over each route (networks with or
    without acts, contact matrices, mixing pools), with strains, or by homogeneous mixing (see `transmit()`)

    It exists so that transmission can be read apart from the rest of the agent engine (`Agents`).
    """

    def transmit(self, tr, src):
        """ Work out who in a state is infected this step """
        self._new_strains = [] # The strain each new case was infected with, for the infection log (see set_strain())
        with self.evaluating(route_memo={}): # See route_beta()
            return self.transmit_from(tr, src)

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
        on_route = None
        if routes and strains == [None]: # Only the routes that carry it (e.g. not a mother-to-child transmission on the sexual network); if none do, nobody is infected, and nothing is drawn
            on_route = [self.carries(tr, nkey, route) for nkey,route in routes]
            if not any(on_route):
                return ss.uids()
        if not self.mod[src.name].raw.any(): # Nobody to infect
            return ss.uids()
        protects = self.strain is not None and (src.name in self.protect or bool(self.sim.modifiers.get((mod.name, 'strain_sus', None)))) # Protected in part against some strains (see Engine.init_immunity(), and products by strain)
        carried = self.carriage is not None and self.applies[src.name][self.strain] and src.name in (self.carriage.carried or [src.name]) # People infected with strains already, who can carry another as well
        superinfects = carried and self.superinfects(tr, src) # ... and stay in the same state, so are only changed if they do not carry it already
        if not routes:
            return self.transmit_mixing(tr, src, strains, protects, carried, superinfects)

        # Otherwise, transmit over each route. Mixing pools give each person a hazard, and networks a chance for each contact
        sus_raw = self.susceptibility(tr, src)
        rel_sus = mod.rel_sus.asnew(sus_raw, copy=False)
        new_cases = []
        pairs = [] # Who infected whom, where that is known, so the strain can be passed on
        pools = any(isinstance(route, ss.networks.PoolRoute) for nkey,route in routes)
        pool_hazard = [np.zeros(len(sus_raw)) for strain in strains] if pools else None
        any_pools = False
        by_strain = 1.0 # The infectiousness of each agent's strain
        strain_raw = None
        if self.strain is not None:
            strain_raw = self.strain_raw()
            if self.strain_adjust[tr.name] is not None:
                by_strain = self.strain_adjust[tr.name][strain_raw]
        for i,(nkey,route) in enumerate(routes):
            if on_route is not None and not on_route[i]: # Nothing on it to transmit (see carries())
                continue
            if isinstance(route, ss.networks.PoolRoute) and mod not in route.diseases: # Not one of the diseases it is for
                continue
            if isinstance(route, ss.Network) and route.has_acts: # A probability per act, in each direction
                if len(route):
                    prs = self.transmit_acts(tr, i, nkey, route, rel_sus, by_strain)
                    pairs += prs
                    new_cases += [pr[1] for pr in prs]
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
            scale = ref if np.isfinite(ref) else 1/dt # A probability of 1 is an infinite rate (ss.probperweek(1)): a certain transmission, beside which the others have their own chance, from their hazard in the step
            weights = None
            for state,c in coefs.items():
                frac = np.minimum(c/scale, np.finfo(float).max)*state.rel_trans # Each source state's hazard, relative to the scale: the beta below is that of the scale, and over a network every factor scales the hazard (see over_network())
                w = mod[state.name].raw if np.ndim(frac) == 0 and frac == 1 else mod[state.name].raw*frac # The usual case, one source state, needs no product
                weights = w if weights is None else weights + w
            rel_trans = times(self.current_value('rel_trans', weights), by_strain) # By UID
            beta = -np.expm1(-scale*dt)
            if isinstance(route, ss.Network):
                for pair in self.over_network(i, route, rel_trans, route_sus.raw, [beta, beta], tr.name):
                    new_cases.append(pair[1])
                    pairs.append(pair)
            elif isinstance(route, ss.networks.PoolRoute): # A hazard for each person
                any_pools = True
                self.add_pool_hazard(route, pool_hazard, strains, strain_raw, ref, rel_trans, route_sus, mult)
            else: # Any other route works out its own transmission
                new_cases.append(ss.uids(route.compute_transmission(route_sus, mod.rel_trans.asnew(rel_trans, copy=False), beta, disease=mod)))
        if self.strain is not None and pairs: # Each new case takes the strain of whoever infected them, one of those who reached them (see Transmission)
            pairs, new_cases = self.assign_strains(tr, src, pairs, new_cases, strain_raw, carried, superinfects, protects)
        self._pairs = pairs
        if any_pools:
            sus = self.members(src)
            new_cases.append(self.draw_infections(tr, sus, [hazard[sus] for hazard in pool_hazard], dt, carried, superinfects))
        if not new_cases:
            return ss.uids()
        return ss.uids.concatenate(new_cases).unique()

    def transmit_mixing(self, tr, src, strains, protects, carried, superinfects):
        """ Transmission with homogeneous mixing (no routes): the same hazard for everyone, scaled by their susceptibility """
        mod = self.mod
        sus = self.members(src)
        adjust = self.adjusted(tr.name, sus)
        hazards = []
        for strain in strains:
            with self.evaluating(strain=strain):
                protect = self.protection(src, sus, strain) if protects else 1.0 # The strains they carry, or remember, protect them in part
                hazards.append(self.strain_rate(tr, self.eval_for(tr.rate, sus), strain)*mod.rel_sus[sus]*src.rel_sus*adjust*self.mult('rel_sus', None, sus)*protect)
        return self.draw_infections(tr, sus, hazards, self.dt, carried, superinfects)

    def add_pool_hazard(self, route, pool_hazard, strains, strain_raw, ref, rel_trans, route_sus, mult):
        """ Over a mixing pool (or pools), each person's hazard of infection, for each strain, added to `pool_hazard` """
        sim = self.sim
        if route.name in sim.contact_rates.pool_terms: # A contact matrix within each place: see ss.solvers.ContactRates.init_pools()
            groups = sim.groups.group_raw()
            auids = sim.people.auids
            n = np.bincount(groups[auids], minlength=sim.groups.n_groups).astype(float)
            for k,strain in enumerate(strains):
                frac = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                infectious = np.bincount(groups[auids], weights=(rel_trans*frac)[auids], minlength=sim.groups.n_groups)
                expo = sim.contact_rates.term_exposure(sim.contact_rates.pool_terms[route.name], infectious, n)
                pool_hazard[k] += ref*expo[groups]*route_sus.raw*mult(None, slice(None))
        elif isinstance(route, ss.networks.MixingPools) and not route.pools: # A matrix of numbers: from the fraction of each group or band that is infectious
            for k,strain in enumerate(strains):
                frac = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                auids, expo = route.exposure(rel_trans*frac)
                pool_hazard[k][auids] += ref*expo*route_sus.raw[auids]*mult(None, auids)
        else: # Has no direction
            for pool in (route.pools if isinstance(route, ss.networks.MixingPools) else [route]):
                if len(pool.source_uids) and len(pool.target_uids):
                    dst = pool.target_uids
                    for k,strain in enumerate(strains):
                        frac = 1.0 if strain is None else self.strain_weights[strain_raw[pool.source_uids], strain]
                        trans = np.mean(rel_trans[pool.source_uids]*frac) # The infectiousness-weighted fraction of the source group that is infectious
                        pool_hazard[k][dst] += ref*trans*pool.eff_contacts.raw[dst]*route_sus.raw[dst]*mult(None, dst)
        return

    def assign_strains(self, tr, src, pairs, new_cases, strain_raw, carried, superinfects, protects):
        """ Each new case over a network takes the strain of whoever infected them, one of those who reached them (see Transmission), unless that changes nothing or they are protected; returns who infected whom, and the new cases left """
        trgs, srcs, nets = self.pick_sources(pairs, tr.name)
        strain = strain_raw[srcs]
        if self.carriage is not None: # Someone who carries several strains passes on one of them
            weights = np.cumsum(self.strain_weights[strain], axis=1)
            rvs = self.draw(f'pass_{tr.name}', trgs)
            strain = np.minimum((weights <= rvs[:, None]).sum(axis=1), weights.shape[1] - 1)
        ok = np.ones(len(trgs), dtype=bool)
        if superinfects: # Infected with a strain they carry already: no change
            ok &= self.can_take(strain_raw[trgs], strain)
        if protects: # Protected in part against the strain that reached them
            ok &= self.unprotected(tr.name, trgs, self.protection(src, trgs, strain))
        if not ok.all():
            drop = trgs[~ok]
            trgs, srcs, nets, strain = trgs[ok], srcs[ok], nets[ok], strain[ok]
            new_cases = [cases[~np.isin(cases, drop)] for cases in new_cases]
        pairs = [(srcs, trgs, nets)] # Who infected whom, for the infection log
        self.set_strain(trgs, strain, carried)
        return pairs, new_cases

    def transmit_together(self, pairs):
        """
        Who is infected by each transmission (and source state) in `pairs`, in one pass over each network rather than one for each,
        when that gives the same result: the transmissions take people from different states, which none of them transmits from, and have
        the same source states with coefficients in proportion (as tbsim's reinfections are, at a relative risk), with no strains and nothing
        that differs by transmission on a route. Each edge's hazard is then the transmission's (its rate relative to the largest scales the
        susceptibility, as every factor scales the hazard: see `over_network()`), and its random number is the same in every pass, so who is
        infected is as if each were worked out alone. None if they cannot be worked out together.
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
            if not np.isfinite(ref): # A certain transmission (see transmit_from())
                return None
            for c,r in coefs[1:]: # In proportion to the first, so the infectious weigh the same in each
                if (r > 0) != (ref > 0) or (ref > 0 and not all(abs(c[k]/r - first[k]/ref) <= 1e-12*abs(first[k]/ref) for k in c)):
                    return None
            rates = [r for c,r in coefs]
            top = max(rates)
            if top <= 0:
                continue
            weights = mod.rel_trans.raw*0
            for state in pairs[0][0].terms:
                weights = weights + mod[state.name].raw*(first[state.name]/ref)*state.rel_trans
            rel_trans = self.current_value('rel_trans', weights)
            route_sus = sum(s*(r/top) for s,r in zip(sus, rates)) # Each transmission's rate, as a fraction of the largest
            beta = -np.expm1(-top*dt)
            for _, trgs, _ in self.over_network(0, route, rel_trans, route_sus, [beta, beta], None):
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

    def log_infections(self, uids, seeds=False, sources=None, networks=None):
        """ Record new infections for an `ss.infection_log` analyzer: with who infected them and the network (given, or where known from this step's transmission); seeds have a source of -1 """
        if not len(uids):
            return
        if seeds:
            sources, networks = np.full(len(uids), -1.0), np.full(len(uids), np.nan)
        elif sources is None:
            sources, networks = self.sources(uids)
        extra = {}
        if self.strain is not None: # The strain transmitted (for seeds, and any infection whose strain was not chosen in transmission, the strains they carry)
            labels = np.array(self.sim.groups.strats[self.strain].labels, dtype=object)
            strain = labels[self.strain_raw()[uids]]
            for new, given in ([] if seeds else self._new_strains or []): # In order, so a later one (as set_strain() did) wins
                order = np.argsort(new)
                inds = np.minimum(np.searchsorted(new[order], uids), len(new) - 1)
                found = new[order][inds] == uids
                strain[found] = labels[given[order][inds[found]]]
            extra['strain'] = strain
        self.mod.infection_log.add_entries(uids, sources, self.mod.t.tvec[self.mod.ti], network=networks, **extra) # At the step's time point, as ti_I is (seeds at the start)
        return

    def same_in_group(self, tr):
        """ Whether the rate of a transmission is the same for everyone in a group, apart from their own susceptibility """
        if tr.name not in self.by_groups:
            leaves = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par)]
            funcs = [raw for raw in leaves if callable(raw) and not isinstance(raw, (ss.Dist, ss.parameters.TimeValue))]
            spread = any(id(leaf) in self.spread for leaf in tr.rate.leaves if isinstance(leaf, ss.par)) # As does a distribution across people
            mods = [m for leaf in tr.rate.leaves if isinstance(leaf, ss.par) for m in self.par_mods(leaf)]
            per_agent = any(m._who_now is not None or isinstance(m.amount, ss.by) or (callable(m.amount) and not isinstance(m.amount, ss.parameters.TimeValue)) for m in mods) # As does a modifier of a parameter for some people
            props = any(isinstance(leaf, (ss.expressions.Prop, ss.expressions.attr)) or (isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by) and leaf.raw.on_people(self.sim)) for leaf in tr.rate.leaves) # As does a property or attribute of people
            self.by_groups[tr.name] = not spread and not per_agent and not props and not any(ss.utils.takes_uids(func) for func in funcs) and not ss.expressions.clocks_in(tr.rate) # A function of the agents, or a clock, can give each a rate of their own
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
            uids = ss.solvers.numba.passes.infect(mod[src.name].raw, sim.groups.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw, float(src.rel_sus), np.ascontiguousarray(np.broadcast_to(adjust, ng), dtype=float),
                                     float(self.mult_raw('rel_sus')), self.dt, sim.people.slot_raw(), *key)
            return ss.uids(uids)
        uids, probs = ss.solvers.numba.passes.infection_probs(mod[src.name].raw, sim.people.auids.view(np.ndarray), sim.groups.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw,
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
            frac = np.cumsum(hazards, axis=0)[:, infected]/total[infected]
            strain = (rvs[infected]/p[infected] >= frac).sum(axis=0)
            self.set_strain(sus[infected], np.array(self.strain_levels())[np.minimum(strain, len(hazards) - 1)], carried)
        return sus[infected]

    def set_strain(self, uids, strain, carried=False):
        """ Give new cases their strain; if they carry strains already (`carried`), they carry this one as well """
        sim = self.sim
        uids = ss.uids(uids)
        if self._new_strains is not None and self.mod.infection_log is not None:
            self._new_strains.append((uids, np.broadcast_to(strain, len(uids))))
        if carried:
            strain = self.carriage.adds[self.strain_raw()[uids], strain]
        sim.people[sim.groups.strats[self.strain].attr][uids] = sim.groups.strats[self.strain].value(strain)
        sim.groups.groups_changed(uids)
        return

    def protection(self, state, uids, strain):
        """ The fraction of the force of infection of a strain (one, or one for each agent) that reaches some agents in a state: by the strains they carry there, and by the products they hold """
        out = super().protection(state, self.strain_raw()[uids], strain) if state.name in self.protect else 1.0
        if self.sim.modifiers.get((self.mod.name, 'strain_sus', None)) and len(uids):
            if np.ndim(strain): # One for each agent
                effects = np.array([np.broadcast_to(self.strain_effect(j), self.n_slots) for j in range(int(np.max(strain)) + 1)])
                out = out*effects[strain, uids]
            else:
                out = out*np.broadcast_to(self.strain_effect(strain), self.n_slots)[uids]
        return out

    def carries(self, tr, nkey, route):
        """ Whether a route carries a transmission: the coefficient of one of its source states is above 0 on it; kept for the run if they are all fixed (see route_beta()) """
        kept = self._route_kept
        key = ('carries', tr.name, nkey)
        if key in kept:
            return kept[key]
        out = any(np.any(np.asarray(self.route_beta(coef, nkey, route), dtype=float) > 0) for coef in tr.terms.values())
        if all(kept.get(id(coef)) for coef in tr.terms.values()):
            kept[key] = out
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
        kept = self._route_kept
        if key in kept: # Fixed for the run
            val, self._per_act = kept[key]
            return val
        self._per_act = False
        with self.evaluating(route=(nkey, route)):
            val = self.eval(coef) if uids is None else self.eval_for(coef, uids)
        acts = getattr(route, 'has_acts', False)
        if acts and not self._per_act: # A beta that is an expression of parameters (beta=b*rel), worked out as a probability per act
            val = self.act_hazards(val, route)
        if memo is not None:
            memo[key] = (val, self._per_act)
        fixed = kept.get(id(coef))
        if fixed is None:
            fixed = kept[id(coef)] = self.strain is None and self.is_route_fixed(coef)
        if fixed:
            if isinstance(val, np.ndarray):
                val.flags.writeable = False # Shared by every step from now on
            kept[key] = (val, self._per_act)
        return val

    def by_target(self, coefs, route):
        """
        The coefficients of a transmission's source states on a route that is not a network with acts, when some have a value for each agent (e.g.
        `ss.by(sex=...)*I` over `ss.contacts(sex=matrix)`): a value for whoever is infected, times a weight for each source state. Returns the weights,
        and the value for each agent (None if no coefficient differs between agents)
        """
        arrays = [np.asarray(c, dtype=float) for c in coefs.values() if np.ndim(c)]
        if not arrays:
            return coefs, None
        shape = np.broadcast_shapes(*[a.shape for a in arrays])
        target = np.max([np.broadcast_to(np.asarray(c, dtype=float), shape) for c in coefs.values()], axis=0) # Above 0 wherever any coefficient is, so none is lost where another is 0
        if target.ndim == 1 and len(target) == self.n_slots: # By UID: the slots of no agent (room for births to come) hold no value (NaN), and count for nothing
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

    def transmit_acts(self, tr, i, nkey, route, rel_sus, by_strain):
        """
        Transmission over a network with acts: each edge from an infectious `u` to a susceptible `v` transmits in a step with
        probability 1 − exp(−H), where H = (edge weight) × (acts in the step) × (hazard per act in the edge's direction, for
        `u`'s state) × rel_trans[u] × rel_sus[v] × (the modifiers on this route); so `rel_trans`, `rel_sus`, source weights,
        and modifiers all scale the hazard of each act. Returns who infected whom, as (sources, targets, `i`, the route's index) for each direction
        """
        mod = self.mod
        sim = self.sim
        hazards = {state: np.asarray(self.route_beta(coef, nkey, route), dtype=float) for state,coef in tr.terms.items()} # Directions (by groups) for each source state
        if not any(np.any(h > 0) for h in hazards.values()):
            return []
        trans = times(self.current_value('rel_trans'), by_strain)
        n_slots = self.n_slots
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

        source = sum(w for h,w in shared) # Whether each agent is infectious, with the rel_trans of their state
        weight = lambda inds: route.edges.weight[inds]*route.edges.acts[inds]
        prob = lambda d, inds, src, trg: -np.expm1(-weight(inds)*act_hazard(src, trg)*trans[src]*rel_sus.raw[trg]*mult(inds, trg))
        out = route.transmit(trans*source, rel_sus.raw, lambda d, inds, src, trg: weight(inds), mod.trans_rng, prob=prob)
        return [(srcs, trgs, i) for srcs,trgs in out]
