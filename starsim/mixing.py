"""
Mixing pools and contact matrices: mixing between groups of people, without edges

A mixing pool (`ss.MixingPool`) spreads a disease from one group of people to another (or among everyone) by their numbers of contacts, and
mixing pools (`ss.networks.MixingPools`, usually made by `ss.contacts()` from a contact matrix) do so between several groups. They are routes,
as networks are (see `ss.networks`), and the pools are also in `ss.networks` (e.g. `ss.networks.MixingPools`).

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import sciris as sc
import pandas as pd
import starsim as ss
from .networks import Route, people_mask

_ = None


class PoolRoute(Route):
    """ What a mixing pool (`ss.MixingPool`) and mixing pools (`ss.networks.MixingPools`) share: groups of people as source and target, without edges; not used directly """
    def __init__(self):
        super().__init__()
        self.diseases = None
        self.source_uids = None # The people in the source and target groups, from step()
        self.target_uids = None
        self.p_acquire = ss.bernoulli(p=0) # Placeholder value: see acquire()
        return

    def init_post(self):
        """ Find the diseases that transmit over it: every communicable disease (one whose beta on it is 0 transmits nothing there) """
        super().init_post()
        self.diseases = [d for d in self.sim.diseases.values() if transmits(d)]
        return

    def remove_uids(self, uids):
        """ Remove people who leave (e.g. die) from the groups given as UIDs

        Args:
            uids (array): the agents to remove
        """
        for key in ['source', 'target']:
            groups = self.pars[key]
            if isinstance(groups, ss.uids):
                self.pars[key] = groups.remove(uids)
            elif isinstance(groups, dict): # Mixing pools: a dict of groups
                for k,group in groups.items():
                    if isinstance(group, ss.uids):
                        groups[k] = group.remove(uids)
        return

    def acquire(self, uids, exposure, disease_beta):
        """ The people among `uids` infected in the step, for each one's exposure (contacts × infectiousness × susceptibility): the hazard of the
        step, from the disease's beta, is scaled by the exposure before being turned back into a probability, as for a declared disease (see ss.solvers.Transmission) """
        self.p_acquire.set(p=1 - (1 - disease_beta)**exposure) # I.e. 1 - exp(-hazard*dt), with the hazard -log(1 - disease_beta)/dt
        return self.p_acquire.filter(uids)

    def shrink(self):
        """ Remove the people in each group, to save memory (e.g. before saving the sim) """
        super().shrink()
        shrunk = ss.utils.shrink()
        self.diseases = shrunk
        self.source_uids = shrunk
        self.target_uids = shrunk
        return


class MixingPools(PoolRoute):
    """
    Mixing between groups of people, by a matrix of contacts between them: a mixing pool (`ss.MixingPool`) for each pair of groups, without edges

    Usually made by `ss.contacts()`, from a contact matrix. With no edges, mixing pools suit mixing
    in which who meets whom matters only by group (e.g. age group), not person by person.

    Every communicable disease transmits over them; a disease's beta on them is the disease's, by route (0 for one that does not spread here).

    Note: CRN safe as agents.

    Args:
        source (dict): the infectious groups, by name: each a condition over people (e.g. `ss.person.age < 15`), UIDs, or a function of the sim
        target (dict): the susceptible groups; as above
        n_contacts (array/`ss.Dist`): `n_contacts[i,j]`, the number of contacts that one person in target group `i` has with people in source group `j` (rows are the people who have the contacts, as in a contact survey and `ss.contacts()`). A matrix of numbers is the same for everyone in each group. One distribution whose parameters are matrices (e.g. `ss.poisson(M)`, or `ss.nbinom(mean=M, n=2)` for more spread) is drawn for each person when they enter a target group, and kept while they are in it. A matrix of distributions (one for each entry) is also drawn for each person, but is slower (a mixing pool for each pair of groups)
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        import starsim as ss
        mps = ss.networks.MixingPools(
            source = {'0-15': ss.person.age < 15, '15+': ss.person.age >= 15},
            target = {'0-15': ss.person.age < 15, '15+': ss.person.age >= 15},
            n_contacts = [[2.4, 0.91], [0.49, 0.16]],  # Illustrative contact matrix: row i, the contacts of a person in target group i
        )
        sim = ss.Sim(diseases='sis', networks=mps).run()
        sim.plot()

        # Contacts that differ between people: a Poisson draw for each person, with the matrix as its mean
        mps = ss.networks.MixingPools(source=mps.pars.source, target=mps.pars.target, n_contacts=ss.poisson([[2.4, 0.91], [0.49, 0.16]]))
        ```

    Transmission to a susceptible person `s` is `1 − (1 − p)^(rel_sus_s · Σ_j c_sj · trans_j)`, for the disease's chance `p` over the step, `s`'s contacts `c_sj`
    with source group `j`, and that group's mean infectiousness `trans_j`. This is exact for a force of infection proportional to the contacts and the
    mean infectiousness. For discrete contacts with random members of a group, it is a mean-field approximation: it uses the group's mean
    infectiousness rather than that of the people met, which slightly overstates infection (by a relative `p(1 − f)/2` when a fraction `f` of the
    group is infectious). It is exact when everyone in the group is equally infectious, and close when `p` is small.
    """
    def __init__(self, pars=None, source=_, target=_, n_contacts=_, **kwargs):
        super().__init__()
        self.define_pars(
            source = None,
            target = None,
            n_contacts = None,
        )
        self.update_pars(pars, **kwargs)
        self.validate_pars()
        self.contact_dist = None # With one distribution for the contacts: a copy, drawn one source group at a time (see draw_contacts())
        self.contact_scale = 1.0 # What its draws are multiplied by, to be per unit of the sim's time (see to_unit())
        if isinstance(self.pars.n_contacts, ss.Dist) and not isinstance(self.pars.n_contacts, ss.constant):
            self.contact_dist = sc.dcp(self.pars.n_contacts)
            self.define_states(
                ss.IntArr('contact_group', default=-1, label='Target group the contacts were drawn for'),
                *[ss.FloatArr(f'contacts_{j}', default=0.0, label=f'Contacts with source group {j}') for j in range(len(self.pars.source))],
            )
        self.pools = [] # A pool for each pair of groups, only if contacts differ between people and are a matrix of distributions: see plain_contacts()
        self.pool_groups = [] # The (target, source) groups of each pool, by position
        self.matrix = None # Otherwise, the number of contacts between groups, [target, source]
        self.edges = None # The lower edges of the matrix's own age bands (from ss.contacts()), whose exposure agents work out by band: see band_exposure()
        self.dims = None # The matrices over the model's own dimensions (from ss.contacts())
        self.age_edges = None # The lower edges of the age matrix's own groups, if given with ages= (see ss.solvers.ContactRates.dim_matrices())
        self.full = None # One matrix over several of the model's dimensions together (from ss.contacts())
        self.bare = None # A matrix given by position alone (ss.contacts(matrix)), until it is matched to the model's stratification with as many levels: see bare_contacts()
        self.unit = None # The unit of time the contacts are counted over, if not the sim's (e.g. 'day' for ss.contacts('Kenya')): see to_unit()
        self.ref = None # Where a matrix of a place came from (e.g. the study), from ss.contacts()
        return

    def to_unit(self, sim):
        """ Put the contacts in the sim's unit, once: a matrix counted over a unit of its own (`ss.contacts('Kenya')`, contacts a day) is scaled to contacts per unit of the sim's time """
        if self.unit is not None:
            factor = ss.solvers.convert(getattr(ss, f'per{self.unit.removesuffix("s")}')(1.0), sim.unit_base)
            if isinstance(self.pars.n_contacts, ss.Dist): # Each person's contacts are drawn in the matrix's unit, then scaled: see draw_contacts()
                self.contact_scale = factor
            else:
                self.pars.n_contacts = self.pars.n_contacts*factor
            if self.dims is not None:
                self.dims = {dim:mat*factor for dim,mat in self.dims.items()}
            if self.full is not None:
                self.full.matrix = self.full.matrix*factor
            self.unit = None
        return

    pools = () # Made in __init__()

    def __len__(self):
        """ The number of pools """
        return len(self.pools)

    def validate_pars(self):
        """ Check that source and target are dicts of groups, and n_contacts has a row for each target group (who has the contacts) and a column for each source group """
        p = self.pars
        if not isinstance(p.source, dict):
            errormsg = f'source= of ss.networks.MixingPools is a dict of groups, not {type(p.source)}'
            raise TypeError(errormsg)
        if not isinstance(p.target, dict):
            errormsg = f'target= of ss.networks.MixingPools is a dict of groups, not {type(p.target)}'
            raise TypeError(errormsg)
        p.source = sc.obj(p.source)
        p.target = sc.obj(p.target)
        if p.n_contacts is None:
            p.n_contacts = np.ones((len(p.target), len(p.source)))
        expected = (len(p.target), len(p.source))
        if isinstance(p.n_contacts, ss.Dist): # One distribution: its parameters are numbers or matrices
            try:
                np.broadcast_shapes(expected, *[np.shape(v) for v in p.n_contacts.pars.values() if sc.isarray(v)])
            except ValueError as e:
                errormsg = f'The parameters of the distribution of contacts (ss.{type(p.n_contacts).__name__}()) are a number, or a matrix with a row for each target group (who has the contacts) and a column for each source group, so shape {expected}'
                raise ValueError(errormsg) from e
            return
        p.n_contacts = np.array(p.n_contacts)
        actual = p.n_contacts.shape
        if actual != expected:
            errormsg = f'The mixing matrix has a row for each target group (who has the contacts) and a column for each source group, so shape {expected}, not {actual}.'
            raise ValueError(errormsg)
        return

    def check_full(self, sim):
        """ For one matrix over several stratifications together (`ss.contacts(matrix, dims=[...])`): check its shape against the levels of each """
        full = self.full
        names = [strat.name for strat in sim.groups.strats]
        for dim in full.dims:
            if dim not in names:
                raise sc.KeyNotFoundError(f'the contacts "{self.name}" are given by "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
        strats = [sim.groups.strats[names.index(dim)] for dim in full.dims]
        counts = tuple(strat.n for strat in strats)
        if (full.levels is not None and tuple(full.levels) != counts) or len(full.matrix) != np.prod(counts):
            levels = '; '.join(f'"{strat.name}" has {strat.n} ({sc.strjoin(strat.labels)})' for strat in strats)
            shape = full.matrix.shape if full.levels is None else tuple(full.levels)*2
            raise ValueError(f'the contact matrix of "{self.name}" has shape {shape}, but {levels}, so it needs shape {counts*2} (or {(int(np.prod(counts)),)*2}, the first stratification slowest).')
        return

    def plain_contacts(self):
        """ The contacts as an array of numbers, if each is the same for everyone in its group; else None """
        p = self.pars
        if isinstance(p.n_contacts, ss.Dist):
            return np.broadcast_to(p.n_contacts.pars.value, (len(p.target), len(p.source))).astype(float) if isinstance(p.n_contacts, ss.constant) else None
        values = [n.pars.value if isinstance(n, ss.constant) else n for n in p.n_contacts.flat]
        if all(sc.isnumber(v) for v in values):
            return np.array(values, dtype=float).reshape(p.n_contacts.shape)
        return None

    def init_pre(self, sim):
        """ Create a mixing pool for each pair of groups, unless the contacts between groups can be used directly

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        self.to_unit(sim)
        p = self.pars
        time_args = {k:p.get(k) for k in ss.Timeline.time_args} # get() allows None
        self.pools = []
        self.pool_groups = []
        if self.edges is not None or self.dims is not None or self.full is not None or self.name in sim.contact_rates.pool_terms: # Used from the counts in each group or band, not with a pool for each pair: see ss.solvers.ContactRates.init_pools()
            return
        self.matrix = self.plain_contacts()
        if self.matrix is not None or self.contact_dist is not None: # Everyone in a group has the same contacts, or one distribution: one pass over the groups does the work of a pool for each pair
            return
        for i,dk,target in p.target.enumitems():
            for j,sk,source in p.source.enumitems():
                n_contacts = p.n_contacts[i,j] # A number is the same for everyone in the group, as with no pools (a distribution elsewhere in the matrix does not change that)
                name = f'{self.name}:{sk}->{dk}' # Named after the whole, so that two of them (e.g. home and school contacts) can be in one sim
                mp = MixingPool(name=name, n_contacts=n_contacts, source=source, target=target, **time_args)
                mp.init_pre(sim)
                self.pools.append(mp)
                self.pool_groups.append((i, j))
        return

    def agent_reasons(self, sim):
        """ A number of contacts that differs between people is approximated by its mean in a compartmental run """
        nc = self.pars.n_contacts
        dists = [nc] if isinstance(nc, ss.Dist) else [n for n in nc.flat if isinstance(n, ss.Dist)]
        if any(not isinstance(d, ss.constant) for d in dists):
            return [ss.modules.Preference(f'the mixing pools "{self.name}" have numbers of contacts that differ between people')]
        return []

    def mean_matrix(self, unit):
        """ The mean number of contacts, [target, source], per unit of the sim's time (`unit`), as a compartmental run uses them """
        nc = self.pars.n_contacts
        if isinstance(nc, ss.Dist): # One distribution of matrices, as a matrix of distributions, one for each entry
            shape = (len(self.pars.target), len(self.pars.source))
            pars = {k:np.broadcast_to(v, shape) if sc.isarray(v) else v for k,v in nc.pars.items()}
            options = dict(unit=nc.unit) if nc.unit is not None else {}
            nc = [[type(nc)(**{k:v[i,j] if sc.isarray(v) else v for k,v in pars.items()}, **options) for j in range(shape[1])] for i in range(shape[0])]
        return self.contact_scale*np.vectorize(lambda n: ss.solvers.ContactRates.mean(n, unit), otypes=[float])(np.array(nc, dtype=object))

    def init_post(self):
        """ Find the diseases that use them, and initialize each mixing pool """
        super().init_post()
        for mp in self.pools:
            mp.init_post()
        return

    def band_exposure(self, weights):
        """
        As agents, for a matrix with its own age bands (`ss.contacts('Kenya')`), with no pool for each pair of bands: the
        UIDs of the people alive, and the exposure of each, from the infectiousness-weighted fraction of each band (`weights`,
        one for each agent) and the contacts between bands, as those pools would give, since everyone in a band has the
        same contacts. People younger than the first band (e.g. not yet born) have none
        """
        ppl = self.sim.people
        auids = ppl.auids
        band = ppl.age_bins(self.edges)[auids] # 0 if younger than the first
        nb = len(self.edges) + 1
        n = np.bincount(band, minlength=nb)
        infectious = np.bincount(band, weights=weights[auids], minlength=nb)
        n[0] = infectious[0] = 0
        prev = np.divide(infectious[1:], n[1:], out=np.zeros(nb-1), where=n[1:] > 0)
        rates = np.append(0.0, self.pars.n_contacts @ prev) # n_contacts is indexed [target, source]
        return auids, rates[band]

    def group_exposure(self, weights):
        """
        As agents, for a matrix of numbers: the UIDs of the people alive, and the exposure of each, as the pools for each pair of
        groups would give, from the infectiousness-weighted fraction of each source group (`weights`, one for each agent)
        """
        frac = np.array([weights[uids].mean() if len(uids) else 0.0 for uids in self.source_uids])
        auids = self.sim.people.auids
        if self.contact_dist is not None: # Each person's own contacts with each source group
            expo = sum(self.state_dict[f'contacts_{j}'].raw[auids]*s for j,s in enumerate(frac))
            return auids, expo
        expo = np.zeros(len(weights))
        for uids,value in zip(self.target_uids, self.matrix @ frac): # A person in several target groups has the exposure of each
            expo[uids] += value
        return auids, expo[auids]

    def exposure(self, weights):
        """ As agents, with no pool for each pair of groups: see `band_exposure()` and `group_exposure()` """
        return self.band_exposure(weights) if self.edges is not None else self.group_exposure(weights)

    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease):
        """
        Calculate transmission, for a disease written as code: through each pool, or for a matrix, from the fraction of each group or band
        that is infectious, as a declared disease's is (see `ss.solvers.Agents.transmit()`), since everyone in a group has the same contacts

        Args:
            rel_sus (Arr): Relative susceptibility
            rel_trans (Arr): Relative infectiousness
            disease_beta (float): The beta value for the disease, as the probability in the step
            disease (Disease): The disease
        """
        if disease_beta == 0 or disease not in self.diseases:
            return ss.uids()
        if self.pools:
            return ss.uids.concatenate([ss.uids(mp.compute_transmission(rel_sus, rel_trans, disease_beta, disease)) for mp in self.pools])
        sim = self.sim
        if self.name in sim.contact_rates.pool_terms:
            auids = sim.people.auids
            groups = sim.groups.group_raw()[auids]
            n, infectious = [np.bincount(groups, weights=w, minlength=sim.groups.n_groups) for w in [None, rel_trans.raw[auids]]]
            expo = sim.contact_rates.term_exposure(sim.contact_rates.pool_terms[self.name], infectious, n.astype(float))[groups]
        else:
            auids, expo = self.exposure(rel_trans.raw)
        return self.acquire(auids, expo*rel_sus.raw[auids], disease_beta)

    def remove_uids(self, uids):
        """ Remove people who leave from each mixing pool, and from the groups given as UIDs

        Args:
            uids (array): the agents to remove
        """
        for mp in self.pools:
            mp.remove_uids(uids)
        super().remove_uids(uids)
        return

    def step(self):
        """ Find the people in each group, once per step, and give them to each mixing pool (rather than each pool finding its own) """
        if not (self.pools or self.matrix is not None or self.contact_dist is not None): # From the counts in each group or band (ss.contacts()): see init_pre()
            return
        self.source_uids = [pool_uids(self.sim, group) for group in self.pars.source.values()]
        known = {id(group):uids for group,uids in zip(self.pars.source.values(), self.source_uids)} # A target group that is a source group (as for a matrix by age) is not found again
        self.target_uids = [known[id(group)] if id(group) in known else pool_uids(self.sim, group) for group in self.pars.target.values()]
        for mp,(i,j) in zip(self.pools, self.pool_groups):
            mp.target_uids = self.target_uids[i]
            mp.source_uids = self.source_uids[j]
        if self.contact_dist is not None:
            self.draw_contacts()
        return

    def draw_contacts(self):
        """ With one distribution for the contacts: draw them for each person who has entered a target group since the last step, from its row of the matrix (none outside every target group) """
        group = np.full(len(self.contact_group.raw), -1)
        for i,uids in enumerate(self.target_uids):
            if (group[uids] >= 0).any():
                errormsg = f'Someone is in two target groups of "{self.name}", whose contacts are drawn from one distribution for each person, so the target groups cannot overlap'
                raise ValueError(errormsg)
            group[uids] = i
        auids = self.sim.people.auids
        moved = auids[group[auids] != self.contact_group.raw[auids]]
        if not len(moved):
            return
        self.contact_group[moved] = group[moved]
        into = moved[group[moved] >= 0]
        rows = group[into]
        shape = (len(self.pars.target), len(self.pars.source))
        pars = {k:np.broadcast_to(v, shape) for k,v in self.pars.n_contacts.pars.items() if sc.isarray(v)} # The parameters that differ by entry
        for j in range(shape[1]):
            contacts = self.state_dict[f'contacts_{j}']
            contacts[moved] = 0.0
            if len(into):
                self.contact_dist.set(**{k:v[rows, j] for k,v in pars.items()}) # A value for each person: their target group's entry
                draws = self.contact_dist.rvs(into)
                check_drawn(draws, self.name, self.pars.n_contacts)
                contacts[into] = draws*self.contact_scale
        return

    def shrink(self):
        """ Remove the people in each group, to save memory (e.g. before saving the sim) """
        super().shrink()
        for mp in self.pools:
            mp.shrink()
        return


def age_band(group):
    """
    The ages a group of a mixing pool is for, as (low, high), if it is a condition on age alone: e.g. (5, 15) for
    `(ss.person.age >= 5) & (ss.person.age < 15)`, and (65, inf) for `ss.person.age >= 65`; else None. A compartmental run
    reads these to regroup the contacts to the model's age groups (see `ss.solvers.ContactRates.age_matrix()`)

    Args:
        group (any): the group
    """
    Cond = ss.conditions.Cond
    if not isinstance(group, Cond):
        return None
    elif group.op == '&':
        a, b = age_band(group.args[0]), age_band(group.args[1])
        return None if a is None or b is None else (max(a[0], b[0]), min(a[1], b[1]))
    elif group.op in ['>=', '>', '<', '<=']:
        age, value = group.args
        if isinstance(age, Cond) and age.op == 'people' and age.args == ('age',) and sc.isnumber(value):
            return (value, np.inf) if group.op in ['>=', '>'] else (0, value)
    return None


def transmits(disease):
    """ Whether a disease is communicable: declared with a contact-mediated transition, or written as code with a beta """
    if disease.graph is not None:
        return len(disease.graph.transmissions) > 0
    return 'beta' in disease.pars


def pool_uids(sim, group):
    """ The people in a group of a mixing pool

    Args:
        sim (Sim): the sim
        group (Cond/func/uids): the people, as a condition over people (e.g. `ss.person.age < 15`), a function of the sim, or UIDs (default: all active agents)
    """
    if group is None:
        return sim.people.auids
    elif isinstance(group, (ss.conditions.Cond, str)):
        auids = sim.people.auids
        return auids[people_mask(sim, group)[auids]]
    elif callable(group):
        out = group(sim)
        return out.uids if isinstance(out, ss.BoolArr) else out # A boolean array (e.g. sim.people.female) is over the agents alive, so is converted to their UIDs
    elif isinstance(group, ss.uids):
        return group
    errormsg = f'A group of a mixing pool is a condition over people (e.g. ss.person.age < 5), a function of the sim, or UIDs, not {type(group)}'
    raise TypeError(errormsg)


class MixingPool(PoolRoute):
    """
    Mixing between two groups of people (or everyone), without edges: used in place of a network

    Each person in the target group is exposed to the average infectiousness of the source group,
    times their number of contacts. With no edges, it suits mixing in which who meets whom does
    not matter person by person. For mixing between several groups by a contact matrix, see
    `ss.contacts()`.

    Every communicable disease transmits over it; a disease's beta on it is the disease's, by route (0 for one that does not spread here).

    Note: CRN safe as agents.

    Args:
        source (Cond/uids/func): the infectious side: a condition over people (e.g. `ss.person.age < 15`), UIDs, or a function of the sim; None for everyone alive
        target (Cond/uids/func): the susceptible side; as above
        n_contacts (float/`ss.Dist`): the number of contacts of each target agent per unit of the sim's time (or a rate, e.g. `ss.perday(10)`): a number is the same for everyone, and a distribution is drawn for each agent (e.g. `ss.poisson(4)`)

    Examples:
        ```python
        import starsim as ss

        # Set the parameters
        mp_pars = dict(
            source = ss.person.male, # Only men are infectious
            target = None, # Everyone is susceptible
            n_contacts = ss.poisson(mean=4),
        )

        # Seed 5% of the male population
        def p_init(sim, uids):
            return 0.05*sim.people.male[uids]

        # Create and run the sim
        sis = ss.SIS(init_prev=p_init)
        mp = ss.MixingPool(**mp_pars)
        sim = ss.Sim(diseases=sis, networks=mp)
        sim.run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, source=_, target=_, n_contacts=_, **kwargs):
        super().__init__()
        self.define_pars(
            source = None,
            target = None, # Same as source
            n_contacts = ss.constant(10),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.FloatArr('eff_contacts', default=self.pars.n_contacts, label='Effective number of contacts')
        )
        return

    def agent_reasons(self, sim):
        """ A pool of some people has no compartmental form yet, and a number of contacts that differs between people is approximated """
        n_contacts = self.pars.n_contacts
        if self.pars.source is not None or self.pars.target is not None:
            return [f'the mixing pool "{self.name}" is between some people (source= or target=), which only agents have']
        elif isinstance(n_contacts, ss.Dist) and not isinstance(n_contacts, ss.constant):
            return [ss.modules.Preference(f'the mixing pool "{self.name}" has a number of contacts that differs between people ({type(n_contacts).__name__})')]
        return []

    def shrink(self):
        """ Remove the people in each group, and its own copy of the timeline (one for each of what can be many pools), to save memory """
        super().shrink()
        self.setattribute('t', ss.utils.shrink())
        return

    def __len__(self):
        """ The number of people in the target group, if given as UIDs (else 0) """
        target = self.pars.get('target') # None before the parameters are defined
        return len(target) if isinstance(target, ss.uids) else 0

    def init_pre(self, sim):
        """ A number of contacts given as a rate (e.g. `ss.perday(10)`) is a number per unit of the sim's time, as a bare number is

        Args:
            sim (Sim): the sim being initialized
        """
        n = self.pars.n_contacts
        rate = n.pars.value if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            self.pars.n_contacts = ss.constant(ss.solvers.convert(rate, sim.unit_base))
            self.eff_contacts.default = self.pars.n_contacts
        super().init_pre(sim)
        return

    def init_post(self):
        """ Find the diseases that transmit over it, and check its contacts """
        super().init_post()
        if isinstance(self.pars.n_contacts, ss.Dist):
            check_drawn(self.eff_contacts.values, self.name, self.pars.n_contacts) # Over the people at the start, not each new one, which would cost a check of each pool on every step
        return

    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease):
        """
        The people infected over the pool in a step

        This is called by transmission (`ss.Disease.transmit()`, declared, or as code from `ss.AgentDisease`) together with network transmission.

        Args:
            rel_sus (float): Relative susceptibility
            rel_trans (float): Relative infectiousness
            disease_beta (float): The beta value for the disease, as the probability in the step
            disease (Disease): The disease

        Returns:
            UIDs of agents who acquired the disease at this step
        """
        if disease_beta == 0 or disease not in self.diseases or len(self.source_uids) == 0 or len(self.target_uids) == 0:
            return ss.uids()

        trans = np.mean(rel_trans[self.source_uids])
        acq = self.eff_contacts[self.target_uids] * rel_sus[self.target_uids]
        return self.acquire(self.target_uids, trans*acq, disease_beta)

    def step(self):
        """ Update the UIDs of the source and target """
        self.source_uids = pool_uids(self.sim, self.pars.source)
        self.target_uids = pool_uids(self.sim, self.pars.target)
        return


def check_drawn(draws, name, dist):
    """ Raise an error for a negative number of contacts drawn from a distribution, which would be a negative chance of infection """
    if len(draws) and draws.min() < 0:
        errormsg = f'The contacts of "{name}" are drawn from ss.{type(dist).__name__}(), which gave {np.sum(draws < 0):,} negative numbers of contacts (e.g. {draws.min():n}); use a distribution of values that are not negative, e.g. ss.lognormal(), ss.gamma(), or ss.nbinom()'
        raise ValueError(errormsg)
    return


def check_contacts(matrix, dim):
    """ The entries of a contact matrix are numbers of contacts, so are finite and not negative """
    bad = ~np.isfinite(matrix) | (matrix < 0)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        errormsg = f'The contact matrix of ss.contacts() for "{dim}" has {bad.sum()} entries that are negative or not finite, e.g. {matrix[i, j]} in row {i}, column {j}; each is a number of contacts per unit time'
        raise ValueError(errormsg)
    return


def contacts(matrix=None, ages=None, dims=None, name=None, study=None, layer='all', unit=None, **by_dim):
    """
    Mixing between groups from a contact matrix; returns `ss.networks.MixingPools`

    A matrix says what it is over. Given by position, it is over:

    - a place's age groups, by its name (`ss.contacts('Kenya')`), from `starsim_data`, by 5-year age group;
    - its own age groups, given by their lower edges (`ss.contacts(matrix, ages=[0, 15, 65])`), which need not be the model's (it is
      regrouped to them); a table labeled by age group (as `ss.data.contacts()` gives) says its own;
    - several of the model's stratifications together (`ss.contacts(matrix, dims=['sex', 'risk'])`), for mixing that is not a product,
      e.g. men mix with women, and by risk differently for each sex;
    - with none of these, the model's one stratification with as many levels (`ss.contacts(matrix)` for a model stratified into 3 age
      groups and a 3×3 matrix), found when the sim is built; with none or several, it is an error.

    Given by keyword, it is over the model's stratification of that name: its age groups (`ss.contacts(age=matrix)`), or another
    (`ss.contacts(risk=matrix)`). With several, the mixing is their product (`ss.contacts(age=A, area=B)`): the contacts between two
    groups are the product of the entries for each. A matrix by age with its own age groups combines with them the same way
    (`ss.contacts(A, ages=[0, 15, 65], area=B)`). For everyone mixing with everyone, use `ss.MixingPool(n_contacts=10)`.

    What the matrix multiplies is said by the rate of the infection, with I_j and N_j the number infectious and the number of people in group j, and N everyone:

    - `beta*I`: the fraction of each group who are infectious, β·Σ_j C_ij·I_j/N_j, with C the contacts of a person and β per contact (as for contact surveys);
    - `beta*I.n`: their number, β·Σ_j C_ij·I_j, density-dependent (as in Lloyd & Jansen's metapopulation);
    - `beta*I.frac` (or `beta*I.n/N`): their number over everyone, β·Σ_j C_ij·I_j/N (as in Keeling & Rohani's matrices, and WAIFW matrices).
    - a weighted count, `beta*(ss.by(w)*I).n` or `beta*I.n_infectious`: their number, each counted by its weight, β·Σ_j C_ij·w_j·I_j (for `n_infectious`, w_j is each group's infectiousness).

    So a textbook's matrix goes in as the textbook gives it, with no rescaling. See the user guide's page on population data for more.

    Args:
        matrix (array/str/df/`ss.Dist`): `matrix[i,j]`, the number of contacts per unit time that one person in group `i` has with people in group `j` (as in a contact survey), with `ages=` or `dims=`; or a place, by name or code ('Kenya', 'KE'), or a US state ('United States:Massachusetts'); or a table by age group, as `ss.data.contacts()` gives (with labels such as '0-4' and '75+', or an index named "age"); or a distribution whose parameters are matrices by age (with `ages=`) or tables by age group, for contacts that differ between people (e.g. `ss.contacts(ss.poisson(ss.data.contacts('Kenya')))`: see `ss.networks.MixingPools`)
        ages (list/array): the lower edge of each age group of the matrix, e.g. `[0, 15, 65]` for 0–14, 15–64, and 65+
        dims (list): for one matrix over several of the model's stratifications together: their names, and `matrix[i, k, j, l]` is the number of contacts per unit time
            that one person at levels `(i, k)` has with people at levels `(j, l)`, or the same as a square matrix over the combinations of levels, the first stratification slowest
        name (str): the name of the mixing (default: 'contacts', or the layer, e.g. 'home')
        study (str): for a place, the study: 'prem' (the default: Prem 2021, bundled, for 177 countries, with Prem 2017 for 10 more, e.g. Japan and Australia), 'prem2021' (177), 'prem2017' (152), or 'mistry2021' (36, by single year of age); for a US state, 'mistry2021_us' (the default; by single year of age); all but Prem 2021 are downloaded on first use
        layer (str/list): for a place, the setting: 'all' (the default), 'home', 'school', 'work', or 'community'; or a list of them, for a list of layers, each named by its setting
        unit (str): the unit of time the contacts are counted over, e.g. 'day' for contacts a day; by default 'day' for a place's matrix (as the studies give them), and otherwise the sim's unit, as for any bare number
        by_dim (dict): a matrix over each of the model's stratifications, by name, e.g. `age=[[10, 3], [3, 5]]` or `risk=[[2, 1], [4, 8]]`; with several, the contacts are the product of their entries

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.05*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        mixing = ss.contacts([[10, 3], [3, 5]], ages=[0, 18])
        sim = ss.Sim(sir, networks=mixing, unit='day', stop=100).run()

        sir.stratify(risk=['low', 'high'])
        mixing = ss.contacts(risk=[[2, 1], [1, 8]])

        home, school = ss.contacts('Kenya', layer=['home', 'school']) # Contacts a day, by 5-year age group
        sim = ss.Sim(sir, networks=[home, school], stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100)

        # A product: by age, then 90% of contacts within one's own area
        mixing = ss.contacts([[10, 3], [3, 5]], ages=[0, 18], area=[[0.9, 0.1], [0.1, 0.9]])

        # Keeling & Rohani's Program 3.1: a matrix of rates per year on fractions of everyone, so the rate is I.frac
        S, I = ss.states('S I')
        sis = ss.Disease(S >> I @ I.frac, I >> S @ ss.peryear(1), init_prev=0.001).stratify(risk={'high': 0.2, 'low': 0.8})
        sim = ss.Sim(sis, networks=ss.contacts(risk=[[10, 0.1], [0.1, 1]]), unit='year', stop=15).run()

        # Men mix with women, and the high-risk with each other more among men than among women: a matrix over sex and risk together
        matrix = np.zeros((2, 2, 2, 2)) # [sex, risk, sex, risk], with sex ['female', 'male'] and risk ['low', 'high']
        matrix[0, :, 1, :] = [[0.8, 0.2], [0.5, 1.5]] # Women's contacts with men
        matrix[1, :, 0, :] = [[0.8, 0.5], [0.2, 2.0]] # Men's contacts with women
        mixing = ss.contacts(matrix, dims=['sex', 'risk'])

        # Contacts that differ between people: lognormal, around a country's mean contacts a day
        C = ss.data.contacts('Kenya')
        mixing = ss.contacts(ss.lognormal(mean=C, std=C*0.5))
        ```
    """
    if 'source' in by_dim: # Otherwise read as a matrix over a stratification called "source"
        errormsg = f"ss.contacts() takes no source= (the infectious side of a mixing pool): the study a country's matrix is from is study=, e.g. ss.contacts('Japan', study={by_dim['source']!r})"
        raise TypeError(errormsg)
    if sc.isnumber(matrix): # Homogeneous mixing is a mixing pool of everyone
        errormsg = f'ss.contacts() takes a contact matrix; for everyone mixing with everyone, with {matrix} contacts per person per unit time, use ss.MixingPool(n_contacts={matrix})'
        raise TypeError(errormsg)
    if 'age' in by_dim and ages is not None: # Over the model's own age groups, or its own: not both
        errormsg = 'ss.contacts(age=matrix) is a matrix over the model\'s own age groups, so it takes no ages=; for a matrix with its own age groups, give it by position: ss.contacts(matrix, ages=[0, 15, 65]), and any others by name, e.g. ss.contacts(matrix, ages=[0, 15, 65], area=[[...]])'
        raise TypeError(errormsg)
    if isinstance(matrix, ss.Dist): # Contacts that differ between people, e.g. ss.contacts(ss.poisson(C), ages=[0, 15])
        return dist_contacts(matrix, ages=ages, name=name or 'contacts', unit=unit)
    if isinstance(layer, (list, tuple)): # Several settings, e.g. ss.contacts('Kenya', layer=['home', 'school', 'work', 'community'])
        return [contacts(matrix, ages=ages, dims=dims, study=study, layer=lay, unit=unit, **by_dim) for lay in layer]
    name = name or (layer if layer != 'all' else 'contacts')
    if isinstance(matrix, dict): # A matrix for each stratification is given by keyword
        errormsg = f'ss.contacts() takes a matrix for each stratification by name, e.g. ss.contacts({", ".join(f"{k}=..." for k in matrix)}), not as a dict'
        raise TypeError(errormsg)
    if isinstance(dims, str): # One stratification is a keyword
        errormsg = f'a matrix over one of the model\'s stratifications is given by its name: ss.contacts({dims}=matrix), not ss.contacts(matrix, dims={dims!r}); dims= is for one matrix over several together, e.g. dims=["sex", "risk"]'
        raise TypeError(errormsg)
    if dims is not None and (by_dim or isinstance(matrix, str) or ages is not None):
        errormsg = f'ss.contacts(matrix, dims={dims!r}) is one matrix over several of the model\'s stratifications together, so it takes no ages= (a matrix with its own age groups is ss.contacts(matrix, ages=[...])), no matrix by name, and a matrix, not a place'
        raise ValueError(errormsg)
    ref = None
    if isinstance(matrix, str): # A country's matrix, from starsim_data, as a table by age group
        matrix = ss.data.contacts(matrix, study=study, layer=layer)
    if isinstance(matrix, pd.DataFrame): # A table by age group (e.g. ss.data.contacts('Kenya')), with the lower edge of each from its labels
        if ages is None and dims is None and ss.data.is_age_index(matrix.index):
            ages, ref = ss.data.age_edges(matrix.index), matrix.attrs.get('ref')
        matrix = matrix.values
    if dims is not None: # One matrix over several of the model's stratifications together
        over = sc.tolist(dims)
        matrix = np.array(matrix, dtype=float)
        k = len(over)
        if matrix.ndim == 2*k and matrix.shape[:k] == matrix.shape[k:]:
            n = int(np.prod(matrix.shape[:k]))
            levels = matrix.shape[:k]
            matrix = matrix.reshape(n, n)
        elif matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            levels = None # Known when the sim is built
        else:
            errormsg = f'ss.contacts(matrix, dims={over}) needs a matrix of shape (levels of each, then levels of each again), e.g. (2, 3, 2, 3) for 2 and 3 levels, or square over the combinations of levels, e.g. (6, 6); got shape {matrix.shape}'
            raise ValueError(errormsg)
        check_contacts(matrix, sc.strjoin(over, sep=' and '))
        return matrix_pools(name, unit, full=sc.obj(dims=over, matrix=matrix, levels=levels))
    if matrix is not None and ages is None and not by_dim: # A matrix by position alone is over the model's one stratification with as many levels, found when the sim is built (see bare_contacts())
        matrix = np.array(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            errormsg = f'ss.contacts() needs a square contact matrix, with a row and a column for each group; got shape {matrix.shape}'
            raise ValueError(errormsg)
        check_contacts(matrix, 'the groups')
        return matrix_pools(name, unit, bare=matrix)
    if matrix is not None and ages is None: # With matrices by name too, a matrix by position says what it is over
        errormsg = 'ss.contacts() needs to know what a matrix given by position is over, when others are given by name: its own age groups, by their lower edges (ss.contacts(matrix, ages=[0, 15, 65], ...)), '
        errormsg += 'or one of the model\'s stratifications, by its name (e.g. ss.contacts(age=matrix, ...))'
        raise TypeError(errormsg)
    if matrix is None and ages is not None:
        errormsg = f'ss.contacts(ages={list(ages)}) gives the age groups of a matrix given by position, but there is none, e.g. ss.contacts(matrix, ages=[0, 15, 65])'
        raise ValueError(errormsg)
    if matrix is None and not by_dim:
        errormsg = 'ss.contacts() needs a matrix: a place (ss.contacts("Kenya")), a matrix with its own age groups (ss.contacts(matrix, ages=[...])), or one over the model\'s stratifications (ss.contacts(age=matrix), ss.contacts(risk=matrix))'
        raise ValueError(errormsg)

    # A matrix by age with its own age groups (checked here), alone or in a product with matrices over the model's stratifications
    if matrix is not None:
        matrix = np.array(matrix, dtype=float)
        if len(ages) != matrix.shape[0] or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            errormsg = f'ss.contacts() needs a square matrix and the lower edge of each of its age groups, e.g. ss.contacts(matrix, ages=[0, 15, 65]); got a matrix of shape {matrix.shape} and ages={list(ages)}'
            raise ValueError(errormsg)
        if not by_dim:
            check_contacts(matrix, 'age')
            groups = age_groups(ages)
            out = MixingPools(source=groups, target=dict(groups), n_contacts=matrix, name=name) # ss.networks.MixingPools is indexed as a contact survey, [target, source]
            out.ref = ref # Where a named matrix came from
            out.edges = np.asarray(ages, dtype=float)
            out.unit = unit or ('day' if ref is not None else None) # A country's matrix is contacts a day, whatever the sim's unit
            return out
        check_ages(ages)
        by_dim = dict(age=matrix) | by_dim # Regrouped to the model's age groups when the sim is built

    # A matrix for each of the model's own stratifications
    mats = {dim:np.array(mat, dtype=float) for dim,mat in by_dim.items()}
    for dim,mat in mats.items():
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            errormsg = f'The contact matrix of ss.contacts() for "{dim}" needs to be square, with a row and a column for each level of "{dim}"; got shape {mat.shape}'
            raise ValueError(errormsg)
        check_contacts(mat, dim)
    age_edges = np.asarray(ages) if matrix is not None else None # The lower edges of the age matrix's own groups, if given with ages= (see ss.solvers.ContactRates.dim_matrices())
    return matrix_pools(name, unit, dims=mats, age_edges=age_edges)


def matrix_pools(name, unit, full=None, bare=None, dims=None, age_edges=None):
    """
    Mixing pools with no groups of their own, whose matrix a run uses directly (see `ss.solvers.ContactRates.init_pools()`): one over several of the model's
    stratifications together (`full`), one given by position alone (`bare`), or one for each stratification by name (`dims`, with `age_edges` for an age matrix with its own groups)
    """
    out = MixingPools(source={}, target={}, n_contacts=np.zeros((0, 0)), name=name)
    out.full = full
    out.bare = bare
    out.dims = dims
    out.age_edges = age_edges
    out.unit = unit
    return out


def bare_contacts(net, strats):
    """ Match a matrix given by position alone (`ss.contacts(matrix)`) to the model's one stratification with as many levels, e.g. its 3 age groups for a 3×3 matrix, as if given by name (`ss.contacts(age=matrix)`) """
    n = len(net.bare)
    found = [strat for strat in strats if strat.n == n]
    if len(found) != 1:
        which = f'none of its stratifications has {n} levels' if not found else f'{sc.strjoin([strat.name for strat in found])} each have {n} levels'
        have = f' ({sc.strjoin([f"{strat.name}: {strat.n}" for strat in strats])})' if strats and not found else ''
        errormsg = f'The {n}×{n} matrix of ss.contacts() "{net.name}" is over the model\'s stratification with {n} levels, but {which}{have}. Say which it is over by name '
        errormsg += '(e.g. ss.contacts(age=matrix)), give its own age groups (ss.contacts(matrix, ages=[0, 15, 65])), or give several stratifications together (ss.contacts(matrix, dims=["sex", "risk"]))'
        raise ValueError(errormsg)
    net.dims = {found[0].name: net.bare}
    net.bare = None
    return


def check_ages(ages):
    """ The age groups of a contact matrix are given by their lower edges, which need to increase """
    if np.any(np.diff(np.array(ages, dtype=float)) <= 0):
        errormsg = f'The age groups of ss.contacts() are given by their lower edges, which need to increase, e.g. ages=[0, 15, 65]; got ages={list(ages)}'
        raise ValueError(errormsg)
    return


def age_groups(ages):
    """ The groups of a contact matrix by age, from the lower edge of each, e.g. {'0-15': ..., '15+': ...} for [0, 15] """
    check_ages(ages)
    out = {}
    for lo,hi in zip(ages, list(ages[1:]) + [None]):
        if hi is None:
            out[f'{lo}+'] = ss.person.age >= lo
        else:
            out[f'{lo}-{hi}'] = (ss.person.age >= lo) & (ss.person.age < hi)
    return out


def dist_contacts(dist, ages, name, unit):
    """
    `ss.contacts()` for a distribution whose parameters are matrices by age, e.g. `ss.contacts(ss.poisson(C), ages=[0, 15])`, or tables by age
    group, as `ss.data.contacts()` gives them (whose age groups and unit, contacts a day, are then read from the table, as for the table itself).
    Each person draws their own contacts with each age group: see `ss.networks.MixingPools`
    """
    dist = sc.dcp(dist)
    tables = {k:v for k,v in dist.pars.items() if isinstance(v, pd.DataFrame)}
    ref = None
    if tables:
        first = next(iter(tables.values()))
        if any(not table.index.equals(first.index) for table in tables.values()):
            errormsg = f'The parameters of ss.{type(dist).__name__}() are tables by different age groups; give them by the same ones'
            raise ValueError(errormsg)
        if ages is None and ss.data.is_age_index(first.index):
            ages, ref = ss.data.age_edges(first.index), first.attrs.get('ref')
        dist.pars.update({k:table.values for k,table in tables.items()})
    if ages is None:
        errormsg = f'ss.contacts() with a distribution (ss.{type(dist).__name__}()) needs the lower edge of each of its age groups, e.g. ss.contacts(ss.poisson(C), ages=[0, 15, 65]), or its parameters as tables by age group, as ss.data.contacts() gives'
        raise ValueError(errormsg)
    groups = age_groups(ages)
    out = MixingPools(source=groups, target=dict(groups), n_contacts=dist, name=name)
    out.ref = ref
    out.unit = unit or ('day' if ref is not None else None) # A country's matrix is contacts a day, whatever the sim's unit
    return out
