"""
Who meets whom in a compartmental run: the contact rates between groups that the networks become
"""
import numpy as np
import scipy.sparse as sps
import sciris as sc
import starsim as ss
from .engine import convert

__all__ = ['Mixing']


class Mixing:
    """
    Who meets whom in a compartmental run

    With no networks, mixing is homogeneous with one contact per unit time, so
    exposure to a state is the fraction of the population in it. A mixing pool
    sets the contact rate; mixing pools between age groups give a contact matrix;
    and an individual-level network is approximated by its mean number of contacts.

    Args:
        sim (`ss.Sim`): the sim
    """
    def __init__(self, sim):
        self.sim = sim
        self.terms = [sc.objdict(kind='homogeneous', contacts=1.0, name=None)]
        self.pool_terms = sc.objdict() # As agents: the contact matrices of mixing pools, within each place (see init_pools())
        self.dependence = 'frequency'
        self.init_space()
        return

    def init_space(self):
        """ Find the spatial dimension, if there is one, and the movement between its levels """
        sim = self.sim
        spaces = [d for d,strat in enumerate(sim.groups.strats) if strat.kind == 'space']
        if len(spaces) > 1:
            errormsg = f'The model has more than one spatial dimension ({sc.strjoin(sim.groups.strats[d].name for d in spaces)}); a person is in one place at a time, so combine them into one'
            raise ValueError(errormsg)
        self.space = spaces[0] if spaces else None # The position of the spatial dimension
        self.time_share = None # For commuting and visiting: the share of their time that people from each place spend in each place; for a moving force of infection, the share of each place's force that each place receives (a sparse array)
        self.present = False # Whether the time share also changes who is present in each place (for commuting)
        self.migration = None # For migration: the rate of moving from each place to each place
        self.register = None # For a movement register: the mobility that holds it; its events are worked out once the sim has a timeline
        self._events = None
        if spaces:
            strat = sim.groups.strats[self.space]
            self.places = sim.groups.group_coords[:, self.space] # The place of each group
            mobility = strat.mobility
            if mobility is not None and mobility.guessed:
                ss.note(f'the movement between the levels of "{strat.name}": {mobility.guessed}', sim=sim)
            if mobility is not None and mobility.kind in ['commute', 'visit']:
                self.time_share = mobility.time_share(strat)
                self.present = mobility.kind == 'commute' # Visitors are exposed to the residents of a place, and do not expose anyone there
            elif mobility is not None and mobility.kind == 'force':
                self.time_share = mobility.coupling(strat)
            elif mobility is not None and mobility.data is not None:
                self.register = mobility
            elif mobility is not None:
                self.migration = mobility.matrix(strat)
                if sps.issparse(self.migration):
                    self.migration = self.migration.toarray()
        self.share = self.time_share # Sparse, since most pairs of places share no one
        return

    def spread(self, v, block=1, by_columns=False):
        """ The product of the time share between places with values by place, or by place and level (`block` levels in each place): `v @ share` if `by_columns`, else `share @ v` """
        v = np.reshape(v, (-1, block))
        out = self.share.T @ v if by_columns else self.share @ v
        return out.ravel()

    def events(self):
        """ The events of the movement register, on the sim's timeline: see `ss.Mobility.events()` """
        if self._events is None:
            self._events = self.register.events(self.sim.groups.strats[self.space], self.sim)
        return self._events

    def by_place(self, places, infectious, n, n_places):
        """
        The chance that a contact is infectious, for the residents of each place

        People mix where they are: at home, or, if they commute, in each place in
        proportion to the time they spend there.

        Args:
            places (array): the place of each entry of the other arrays
            infectious (array): the (infectiousness-weighted) number in the infectious state
            n (array): the number of people
            n_places (int): the number of places
        """
        num = np.bincount(places, weights=infectious, minlength=n_places)
        den = np.bincount(places, weights=n, minlength=n_places)
        if self.share is not None and self.present: # Who is present in each place
            num, den = self.spread(num, by_columns=True), self.spread(den, by_columns=True)
        prev = np.divide(num, den, out=np.zeros(n_places), where=den>0)
        return prev if self.share is None else self.spread(prev)

    @staticmethod
    def mean(val, unit):
        """ The mean of a number of contacts, given as a number or a distribution; one given as a rate (e.g. `ss.perday(10)`) is per unit of the sim's time, as a bare number is """
        if isinstance(val, ss.Dist):
            pars = val.pars
            for key in ['v', 'lam', 'mean', 'loc']:
                if key in pars and (sc.isnumber(pars[key]) or isinstance(pars[key], ss.Rate)):
                    return float(convert(pars[key], unit))
            return ss.transitions.dwell_stats(val, lambda x: convert(x, unit))[0]
        return float(convert(val, unit))

    def init(self, approx=None):
        """ Turn the sim's networks into contact rates between groups """
        sim = self.sim
        nets = list(sim.pars.networks.values())
        if not nets:
            return
        self.terms = []
        for net in nets:
            beta = net.pars.get('beta', 1.0) if isinstance(net, (ss.MixingPool, ss.MixingPools)) else 1.0
            if isinstance(net, ss.MixingPools) and (getattr(net, 'dims', None) or getattr(net, 'full', None) is not None):
                self.terms.append(self.dim_matrices(net, beta))
            elif isinstance(net, ss.MixingPools):
                self.terms.append(self.age_matrix(net, beta))
            elif isinstance(net, ss.MixingPool):
                if net.pars.src is not None or net.pars.dst is not None:
                    raise ss.CapabilityError(f'cannot run the mixing pool "{net.name}" with solver="{sim.solver}": only a pool of everyone, or ss.MixingPools between age groups, has a compartmental form so far. Run with solver="abm".')
                self.terms.append(sc.objdict(kind='homogeneous', contacts=self.mean(net.pars.n_contacts, sim.unit_base)*beta, name=net.name))
            elif isinstance(net, ss.PartnerNet) and net.one_off: # Exact: a contact matrix between its two groups
                self.terms.append(self.two_groups(net))
            elif isinstance(net, ss.Network):
                kind = type(net).__name__
                contacts, source = self.net_contacts(net)
                if contacts is None:
                    raise ss.CapabilityError(f'cannot run with solver="{sim.solver}": {kind} has no compartmental approximation yet{source}. Run with solver="abm".')
                what = f'{kind} approximated by homogeneous mixing with {contacts:n} contacts per {sim.unit_name} ({source})'
                if approx is False:
                    raise ss.CapabilityError(f'cannot run with solver="{sim.solver}" and approx=False: {what[0].lower() + what[1:]}. Run with solver="abm".')
                else:
                    ss.note(what, sim=sim, verbose=0 if approx else None)
                self.terms.append(sc.objdict(kind='homogeneous', contacts=contacts, name=net.name))
        if self.space is not None and any(term.kind == 'matrix' for term in self.terms):
            self.init_space() # A contact matrix can add an age stratification, which changes the groups
            for term in self.terms:
                if term.kind == 'matrix':
                    self.within_places(term)
        return

    def net_contacts(self, net):
        """
        The mean number of contacts of a network that has a compartmental approximation (homogeneous mixing with that many contacts), and where it comes from

        A random network has its mean number of contacts, and a static network the mean degree of its graph. A network of
        partnerships has none yet: its contacts last, so their number per unit time is not its number of partners.
        """
        n_contacts = net.pars.get('n_contacts')
        if isinstance(net, ss.RandomNet) and n_contacts is not None:
            return self.mean(n_contacts, self.sim.unit_base), 'its mean number of contacts'
        elif isinstance(net, ss.StaticNet):
            graph = net.nx_graph
            if graph is not None and not callable(graph):
                return 2*graph.number_of_edges()/max(graph.number_of_nodes(), 1), 'the mean degree of its graph'
            elif graph is None or getattr(graph, '__name__', None) in ['fast_gnp_random_graph', 'gnp_random_graph', 'erdos_renyi_graph']: # A random graph: a number of contacts, or a chance of each pair
                p = net.pars.get('p')
                return (self.mean(n_contacts, self.sim.unit_base), 'its mean number of contacts') if p is None else (p*self.sim.pars.total_pop, 'its chance of each pair, times the population')
            return None, f' (it is made by {getattr(graph, "__name__", "a function")}, whose mean degree is not known before it runs)'
        elif isinstance(net, ss.PartnerNet):
            return None, ' (its partnerships last, so the number of partners is not a number of contacts per unit time)'
        return None, ''

    def two_groups(self, net):
        """
        One-off contacts between two groups (`ss.PartnerNet(between=...)` with no `dur=`) as a contact matrix over the
        stratification the two groups are levels of: one group's contacts per unit time with the other, and the other's
        with it, so that each contact has one of each (from the split of the stratification)
        """
        sim = self.sim
        sides = list(net.sides.items())
        strat = next((s for s in sim.groups.strats if all(isinstance(c, str) and c in s.labels for k,c in sides)), None)
        if strat is None:
            raise ss.CapabilityError(f'"{net.name}" is between {sc.strjoin(k for k,c in sides)}, which with solver="{sim.solver}" must be levels of one stratification, e.g. sim.stratify(role=["none", "{sides[0][0]}", "{sides[1][0]}"]), and between=("{sides[0][0]}", "{sides[1][0]}"). Or run with solver="abm".')
        d = [s.name for s in sim.groups.strats].index(strat.name)
        self.check_ragged(d, net)
        net.init_one_off(sim)
        split = np.asarray(strat.split, dtype=float) if getattr(strat, 'split', None) is not None and np.ndim(strat.split) == 1 else np.full(strat.n, 1/strat.n)
        a, b = [strat.labels.index(c) for k,c in sides]
        seeker, other = (a, b) if net.seeker == 0 else (b, a)
        matrix = np.zeros((strat.n, strat.n))
        matrix[seeker, other] = net.contact_rate
        matrix[other, seeker] = net.contact_rate*split[seeker]/split[other] if split[other] > 0 else 0.0
        return sc.objdict(kind='matrix', dim=d, matrix=matrix, name=net.name, sides=(a, b))

    def within_places(self, term):
        """
        Make a contact matrix act within each place

        The cells of the matrix become pairs of a place and a level, and the movement
        between places applies to each level: people mix by the matrix with the people
        present where they are. A matrix that is given over the places themselves already says who
        meets whom across them, so it is used as it is.

        The mixing is the product of the time share between places and the matrix between levels
        (`np.kron(share, matrix)`), but is kept as the two factors, which is much smaller and faster
        with many places: see `term_exposure()`.
        """
        if self.space in term.get('dims', []):
            return
        n_places = self.sim.groups.strats[self.space].n
        levels = term.cells if 'cells' in term else self.sim.groups.group_coords[:, term.dim]
        size = term.matrix.shape[0]
        term.cells = self.places*size + levels
        term.size = n_places*size
        term.pre = self.share is not None and self.present # Who is present in each place
        term.post = self.share is not None # Mixing between places
        return

    def age_matrix(self, net, beta):
        """ Aggregate the contact matrix of age-based mixing pools to the model's age groups """
        sim = self.sim
        pars = net.pars
        groups = list(pars.src.values())
        if not all(isinstance(g, ss.AgeGroup) for g in groups + list(pars.dst.values())):
            raise ss.CapabilityError(f'cannot run "{net.name}" with solver="{sim.solver}": only mixing pools between age groups have a compartmental form so far. Run with solver="abm".')
        top = np.inf # The last group is open: everyone older than its lower edge
        bands = np.array([[g.low, top if g.high is None else g.high] for g in groups], dtype=float)
        dst_bands = np.array([[g.low, top if g.high is None else g.high] for g in pars.dst.values()], dtype=float)
        if not any(strat.kind == 'age' for strat in sim.groups.strats): # Use the matrix's own age groups
            edges = sorted(bands[:, 0])
            sim.pars.stratify = sc.mergelists(sim.pars.stratify, dict(age=edges))
            sim.groups.init()
            ss.note(f'stratified by the age groups of "{net.name}" ({sc.strjoin(sim.groups.strats[-1].labels)}), since no age groups were given', sim=sim)
        d = [strat.kind for strat in sim.groups.strats].index('age')
        self.check_ragged(d, net)
        strat = sim.groups.strats[d]
        levels = np.column_stack([strat.edges, np.append(strat.edges[1:], top)])
        cdf = self.age_cdf(strat)

        def overlap(a, b):
            """ The share of the population in the overlap between each pair of age ranges """
            return np.clip(cdf(np.minimum(a[:, None, 1], b[None, :, 1])) - cdf(np.maximum(a[:, None, 0], b[None, :, 0])), 0, None)

        def share(a, b):
            """ The share of the people in each range of `a` who are in each range of `b` """
            size = (cdf(a[:, 1]) - cdf(a[:, 0]))[:, None]
            return np.divide(overlap(a, b), size, out=np.zeros((len(a), len(b))), where=size>0)

        to_dst = share(levels, dst_bands) # The share of each model group that is in each band
        from_src = share(bands, levels) # The share of each band that is in each model group
        contacts = np.array(pars.n_contacts, dtype=object)
        contacts = np.vectorize(lambda n: self.mean(n, sim.unit_base))(contacts).astype(float) # [source band, destination band]
        matrix = to_dst @ contacts.T @ from_src # Contacts per unit time of one person in model group i with people in model group j
        if len(bands) != len(levels) or not np.allclose(bands, levels):
            ss.note(f'contact matrix "{net.name}" aggregated from {len(bands)} age groups to the model\'s {strat.n} ({sc.strjoin(strat.labels)})', ss.MatrixAggregated, sim=sim)
        return sc.objdict(kind='matrix', dim=d, matrix=matrix*beta, name=net.name)

    def age_cdf(self, strat):
        """ The share of people younger than each age: from a named population (or, for a guessed age split, the world's), or from the age split, with ages uniform within each 5-year group or age group """
        pyramid = self.sim.pyramid(strat)
        if pyramid is not None:
            ages, counts = pyramid
            ends = np.append(ages, ages[-1] + 5)
            below = np.append(0, np.cumsum(counts))/np.sum(counts)
        else:
            ends = np.append(strat.edges, max(100, strat.edges[-1] + 10)) # As in ss.transitions.Strat: the last group ends at 100
            below = np.append(0, np.cumsum(strat.split))
        return lambda x: np.interp(x, ends, below)

    def check_ragged(self, d, net):
        """ Contacts by a dimension need everyone who transmits or is infected to have a level of it """
        sim = self.sim
        strat = sim.groups.strats[d]
        if not sim.groups.group_inside[:, d].all():
            where = ', '.join(f'{k}={sc.strjoin(sc.tolist(v))}' for k,v in strat.where.items())
            raise ss.PartialStratificationMixing(f'the contacts "{net.name}" are given by "{strat.name}", but only people where {where} have a level of it, so there is no level for the others to mix as.\n'
                                                 f'     Fix: leave out where=, or use contacts that do not depend on "{strat.name}".')
        for mod in sim.declared:
            own = [s for s in mod.graph.strats if s.name == strat.name and s.states is not None]
            if not own:
                continue
            mixing = [s.name for tr in mod.graph.transmissions for s in tr.src] + [s.name for s in mod.graph.infectious]
            missing = [name for name in dict.fromkeys(mixing) if name not in own[0].states]
            if missing:
                raise ss.PartialStratificationMixing(f'the contacts "{net.name}" are given by "{strat.name}", but "{mod.name}" is stratified by it in {sc.strjoin(own[0].states)} only, so there is no level of it for the people in {sc.strjoin(missing)} to mix as.\n'
                                                     f'     Fix: leave out states= to stratify every state, or use contacts that do not depend on "{strat.name}".')
        return

    def check_reciprocal(self, net, strat, mat):
        """ Say if a contact matrix is not reciprocal for the dimension's split: the contacts of the people in level i with level j should equal those of j with i, N_i·C_ij = N_j·C_ji """
        split = getattr(strat, 'split', None)
        if split is None or np.ndim(split) != 1 or len(split) != len(mat):
            return
        flows = np.asarray(split, dtype=float)[:, None]*mat # Contacts from level i to level j, per person in the population
        scale = np.maximum(np.maximum(flows, flows.T), 1e-300)
        gap = np.abs(flows - flows.T)/scale
        if np.max(gap[scale > 1e-12*flows.max()], initial=0) > 0.05:
            i, j = np.unravel_index(np.argmax(np.where(scale > 1e-12*flows.max(), gap, 0)), gap.shape)
            ss.note(f'the contact matrix of "{net.name}" for "{strat.name}" is not reciprocal for its split: {strat.labels[i]} have {mat[i, j]:.3g} contacts each with {strat.labels[j]} ({flows[i, j]:.3g} per person of the population), '
                    f'but {strat.labels[j]} have {mat[j, i]:.3g} each with {strat.labels[i]} ({flows[j, i]:.3g}); it is used as given, with rows as the contacts of a person in each level', sim=self.sim)
        return

    def dim_matrices(self, net, beta):
        """ A contact matrix for each of one or more of the model's dimensions: the contacts between two groups are the product of the entries; or one matrix over several dimensions together (`ss.contacts(matrix, dims=[...])`) """
        sim = self.sim
        full = getattr(net, 'full', None)
        if full is not None:
            net.init_full(sim) # Checks the dimensions and the shape
            names = [strat.name for strat in sim.groups.strats]
            cells = np.zeros(sim.groups.n_groups, dtype=int)
            for dim in full.dims:
                d = names.index(dim)
                self.check_ragged(d, net)
                cells = cells*sim.groups.strats[d].n + sim.groups.group_coords[:, d]
            return sc.objdict(kind='matrix', cells=cells, matrix=full.matrix*beta, name=net.name, dim=names.index(full.dims[0]), dims=[names.index(dim) for dim in full.dims])
        matrix = np.ones((1, 1))
        cells = np.zeros(sim.groups.n_groups, dtype=int) # Which row of the combined matrix each group is in
        for dim,mat in net.dims.items():
            names = [strat.name for strat in sim.groups.strats]
            if dim not in names:
                raise ss.MissingDimension(f'the contacts "{net.name}" are given by "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
            strat = sim.groups.strats[names.index(dim)]
            if len(mat) != strat.n:
                raise ss.ShapeError(f'the contact matrix of "{net.name}" for "{dim}" has {len(mat)} rows, but "{dim}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
            self.check_ragged(names.index(dim), net)
            self.check_reciprocal(net, strat, np.asarray(mat, dtype=float))
            cells = cells*strat.n + sim.groups.group_coords[:, names.index(dim)]
            matrix = np.kron(matrix, mat)
        return sc.objdict(kind='matrix', cells=cells, matrix=matrix*beta, name=net.name, dim=names.index(list(net.dims)[0]), dims=[names.index(dim) for dim in net.dims])

    def exposure(self, infectious, n, mults=None):
        """
        Exposure of each group to an infectious state: contacts per unit time, times the chance that a contact is infectious

        Args:
            infectious (array): the (infectiousness-weighted) number in the state, in each group
            n (array): the number of people in each group
            mults (list): a multiplier on the contacts of each term (a number, or one for each group), from the modifiers of its network
        """
        out = 0.0
        for t,term in enumerate(self.terms):
            mult = 1.0 if mults is None else mults[t]
            out = out + mult*self.term_exposure(term, infectious, n)
        return out

    def term_exposure(self, term, infectious, n):
        """ Exposure of each group through one term of the mixing: see `exposure()` """
        if term.kind == 'homogeneous' and self.space is not None: # People mix within their own place
            n_places = self.sim.groups.strats[self.space].n
            return term.contacts*self.by_place(self.places, infectious, n, n_places)[self.places]
        elif term.kind == 'homogeneous':
            total = n.sum()
            return term.contacts*infectious.sum()/total if total else 0.0
        levels = term.cells if 'cells' in term else self.sim.groups.group_coords[:, term.dim]
        block = term.matrix.shape[0]
        size = term.get('size', block)
        num = np.bincount(levels, weights=infectious, minlength=size)
        den = np.bincount(levels, weights=n, minlength=size)
        if term.get('pre'): # Who is present in each place
            num, den = self.spread(num, block, by_columns=True), self.spread(den, block, by_columns=True)
        prev = np.divide(num, den, out=np.zeros(size), where=den>0)
        prev = term.matrix @ prev if size == block else (prev.reshape(-1, block) @ term.matrix.T).ravel() # Mixing between the levels (in each place)
        if term.get('post'): # ... and between places
            prev = self.spread(prev, block)
        return prev[levels]

    def init_pools(self):
        """
        As agents, in a model with places: make the contact matrices of mixing pools act within each place

        A mixing pool's hazard is worked out from the share of each group that is infectious,
        so the contact matrix is used as in a compartmental run, from the agents' counts in
        each group (see `ss.solvers.Agents.infect()`), with any movement between places.
        """
        sim = self.sim
        self.pool_terms = sc.objdict()
        if self.space is None:
            return
        pools = [net for net in sim.pars.networks.values() if isinstance(net, ss.MixingPools) or (isinstance(net, ss.MixingPool) and net.pars.src is None and net.pars.dst is None)]
        for net in pools:
            beta = net.pars.get('beta', 1.0)
            if isinstance(net, ss.MixingPool): # A pool of everyone: within each place
                self.pool_terms[net.name] = sc.objdict(kind='homogeneous', contacts=self.mean(net.pars.n_contacts, sim.unit_base)*beta, name=net.name)
            else:
                self.pool_terms[net.name] = self.dim_matrices(net, beta) if (getattr(net, 'dims', None) or getattr(net, 'full', None) is not None) else self.age_matrix(net, beta)
        if any(term.kind == 'matrix' for term in self.pool_terms.values()):
            self.init_space() # A contact matrix can add an age stratification, which changes the groups
            for term in self.pool_terms.values():
                if term.kind == 'matrix':
                    self.within_places(term)
        return
