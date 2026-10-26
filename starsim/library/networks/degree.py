"""
Partnerships with a target number of partners at a time, as in EpiModel's networks fitted to target statistics

`DegreeNet` has lasting partnerships between anyone, with a target distribution of the number of
partners people have at a time (a mean, a share with two or more, a maximum), partnerships within a
level of a property of people (EpiModel's nodematch), and a number of partners for each level (nodefactor)
(`ss.SexualNet` takes the first two, matching on one property). `ss.interop.load(..., format='epimodel')`
makes a `DegreeNet` from a network fitted by EpiModel's `netest()`.
"""
import itertools
import numpy as np
import sciris as sc
import starsim as ss

_ = None
ss_float = ss.dtypes.float


class DegreeNet(ss.networks.DegreeTargets, ss.DynamicNetwork):
    """
    Partnerships that form and dissolve, with a target number of partners at a time

    The number of partners people have at any moment (`n_contacts`) is a target
    for the whole population, not a fixed number for each person: partnerships end
    at random, and people look for new ones at a rate that depends on how many
    they already have, chosen so that the distribution of the number of partners
    stays at the target. Everyone moves through it over time, as in EpiModel's
    networks fitted to target statistics. The network starts at the target, and
    partnerships ended by a partner leaving (dying) are allowed for, as EpiModel's
    d.rate correction does, so they last as long as given on average. Partnerships
    are not only sexual: people who share needles have partners too. For sexual
    partnerships, between men and women or between men, see `ss.SexualNet`.

    Note: not CRN safe when people are born or die.

    Args:
        n_contacts (float/`ss.Dist`/`ss.by`): the number of partners a person has at a time: a distribution over whole numbers (e.g. `ss.poisson(0.7)`), or a number, which is the mean of a Poisson; or a number for each level of a property of people (`ss.by(risk=[0.4, 1.6])`), as EpiModel's nodefactor
        dur (float/`ss.dur`/`ss.Dist`): how long partnerships last: a mean (they end at a constant rate), or a distribution; a bare number is in the sim's unit
        acts (rate/`ss.Dist`): if given, the number of acts per unit time in each partnership: a rate such as `ss.freqperweek(2)`, or a distribution of rates drawn once for each partnership; a disease's beta is then per act (see `ss.peract`)
        where (Cond/str/func): if given, who takes part: a condition over people (e.g. `ss.ppl.pwid`), as for `ss.modify()`; nobody else is in the network
        concurrency (float): if given, the share of people with two or more partners at once; the distribution of `n_contacts` is adjusted to give it, keeping its mean
        max_contacts (int): if given, nobody has more partners than this at once
        match (dict): if given, the share of partnerships between people at the same level of a property of people (a stratification, or an attribute such as `'female'`), e.g. `dict(risk=0.8)`, as EpiModel's nodematch; or of each of several, e.g. `dict(risk=0.8, region=0.6)`

    The realized network is recorded in the results (`mean_degree`, `concurrency`, and
    `match`, or with several properties `match_<name>` for each), and `net.check()` compares it with the targets.

    Examples:
        ```python
        import starsim.library as ssl

        # EpiModel's introductory network: 175 partnerships among 500 people, 110 people with two or more partners, nobody with four or more, and partnerships lasting 50 steps
        net = ssl.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)

        # Most partnerships within a risk group, and over half within a region
        net = ssl.DegreeNet(n_contacts=0.7, dur=50, match=dict(risk=0.8, region=0.6))

        # And four times as many partners in the high-risk group (with sim.stratify(risk=['low', 'high']))
        net = ssl.DegreeNet(n_contacts=ss.by(risk=[0.4, 1.6]), dur=50, match=dict(risk=0.8))

        # Needle sharing among people who inject drugs (a boolean property of people), with 20 injections a month in each partnership
        needles = ssl.DegreeNet(name='needles', where=ss.ppl.pwid, n_contacts=2, dur=ss.months(6), acts=ss.freqpermonth(20))
        ```
    """
    def __init__(self, pars=None, n_contacts=_, dur=_, concurrency=_, max_contacts=_, match=_, acts=_, where=_, **kwargs):
        super().__init__()
        self.define_pars(
            n_contacts = 1.0,
            dur = 10,
            concurrency = None,
            max_contacts = None,
            match = None,
            acts = None,
            where = None,
        )
        self.update_pars(pars, **kwargs)
        if self.pars.acts is not None:
            self.meta.acts = ss_float # Acts in each partnership in a step: its acts per unit time times the step
        self.rng_degree = ss.random() # The number of partners each person starts with
        self.rng_order  = ss.random() # The order in which people are paired
        self.rng_seek   = ss.random() # Whether each person looks for a partner on this step
        self.rng_end    = ss.random() # When each partnership ends, if it ends at a constant rate
        self.rng_stubs  = ss.random() # For shuffling the partners people start with
        self.rng_match  = ss.random() # Whether each person looks for a partner at their own level, with match=
        for key in list(self.pars.match or {})[1:]: # And of each other property, with several
            setattr(self, f'rng_match_{key}', ss.random())
        self.match_extra = np.zeros(len(self.pars.match or {})) # For each property, how many more look within their level than by the target alone: see match_pairs()
        self.match_results = {key:'match' if len(self.pars.match) == 1 else f'match_{key}' for key in self.pars.match or {}} # The result of the share at the same level of each property
        self.departures = 0.0 # The share of partnerships ended each step by a partner leaving (dying), estimated as the run goes
        self.lost = 0 # Partnerships ended that way on this step
        self.waiting = None # Those who looked for a partner on the last step and were left over
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        p = self.pars
        if isinstance(p.dur, ss.Dist):
            self.mean_dur = None
        else:
            dur = ss.solvers.convert(p.dur, sim.unit_base) if isinstance(p.dur, ss.TimePar) else p.dur
            self.mean_dur = dur/self.t.dt.to(sim.unit_base).value # In steps
            if self.mean_dur < 1:
                errormsg = f'Partnerships of "{self.name}" last {self.mean_dur:n} steps on average; they need to last at least one'
                raise ValueError(errormsg)
        self.init_seeking(self.mean_dur if self.mean_dur is not None else float(np.mean(sc.dcp(p.dur).init(force=True).rvs(100_000))))
        ss.networks.check_acts(self, p.acts)
        if isinstance(p.acts, ss.TimePar):
            self.acts_per_step = ss.solvers.convert(p.acts, sim.unit_base)*self.t.dt.to(sim.unit_base).value
        return

    def init_seeking(self, mean_dur):
        """ The target distribution of the number of partners (of each level, with `ss.by`), and the probability of looking for a partner with each number """
        p = self.pars
        self.by_level = None # With n_contacts=ss.by(...): the property of people whose levels have degrees of their own, and the distribution and seeking probabilities of each
        if isinstance(p.n_contacts, ss.by):
            if len(p.n_contacts.dims) != 1 or p.n_contacts.dim is None:
                errormsg = f'n_contacts= of "{self.name}" takes a number for each level of one property of people, e.g. ss.by(risk=[0.4, 1.6]), not {p.n_contacts}'
                raise ValueError(errormsg)
            self.by_level = p.n_contacts.dim
            self.dists_degree = [self.degree_dist(v, p.concurrency, p.max_contacts) for v in sc.tolist(p.n_contacts.values)]
        else:
            self.dist_degree = self.degree_dist(p.n_contacts, p.concurrency, p.max_contacts)
        if p.match is not None and (not isinstance(p.match, dict) or not p.match or not all(sc.isnumber(v) and 0 <= v <= 1 for v in p.match.values())):
            errormsg = f'match= of "{self.name}" takes properties of people and the share of partnerships at the same level of each, e.g. dict(risk=0.8), not {p.match}'
            raise ValueError(errormsg)
        if p.match is not None and self.by_level is None: # Who partners whom, but not how many: a common slip when porting a model with more partners at some levels (EpiModel's degree by risk)
            key = list(p.match.keys())[0]
            ss.note(f'"{self.name}" matches partners by {key}, but everyone has the same number of partners whatever their {key}; for more at some levels, give n_contacts=ss.by({key}=[...])', sim=self.sim)
        if self.by_level is not None: # Seeking probabilities for each level
            size = max(len(d) for d in self.dists_degree)
            self.dists_degree = [np.pad(d, (0, size - len(d))) for d in self.dists_degree]
            self.p_seeks = []
            for d in self.dists_degree:
                self.dist_degree = d
                self.init_seek(mean_dur)
                self.p_seeks.append(self.p_seek)
            self.p_seeks = np.array(self.p_seeks)
            self.dist_degree = np.mean(self.dists_degree, axis=0) # Replaced by the population's mix of levels at init_post()
        self.init_seek(mean_dur)
        return

    def taking_part(self):
        """ The people who can have partners: alive, and with where=, those it gives """
        uids = self.sim.people.alive.uids
        if self.pars.where is not None:
            uids = uids[ss.networks.people_mask(self.sim, self.pars.where)[uids]]
        return uids

    def seek_prob(self, uids):
        """ The probability that each person looks for a partner on this step, given how many they have """
        degree = np.minimum(self.degrees(uids), len(self.p_seek) - 1)
        if self.by_level is not None:
            return self.p_seeks[np.asarray(self.levels(self.by_level))[uids].astype(int), degree]
        return self.p_seek[degree]

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('new_partnerships', label='Partnerships formed', auto_plot=False))
        self.define_degree_results()
        for key,name in self.match_results.items():
            self.define_results(ss.Result(name, label=f'Share of partnerships at the same level{"" if name == "match" else f" of {key}"}', auto_plot=False))
        return

    def match_pairs(self, people, rvs=None):
        """
        Pair people who are looking for partners, who are in a random order: in turn, or, with `match=`, some within
        their own level, so that the share of partnerships at the same level is the target

        With several properties, each person looks within their own level of each with its own chance, as with one,
        and is paired with someone who looks within the same levels of the same properties. This gives each target
        when the properties are independent of each other; `net.check()` says how close it is otherwise.

        Args:
            people (uids): the people, once for each partner they look for
            rvs (array): a random number for each, for each property (properties by people), which says whether they look within their level (by default, drawn for each person)

        Returns the two lists of partners, and anyone left over.
        """
        if self.pars.match is None: # In turn
            n = len(people)//2
            return people[:n], people[n:2*n], people[2*n:]
        if rvs is None:
            rvs = [self.rng_match.rvs(people)] + [getattr(self, f'rng_match_{key}').rvs(people) for key in list(self.pars.match)[1:]]
        codes, sizes, levels = [], [], [] # The level each person looks within for each property (0 for none), how many there can be, and each person's level
        for (key,share),u,extra in zip(self.pars.match.items(), np.reshape(rvs, (len(self.pars.match), -1)), self.match_extra):
            level = np.unique(self.levels(key)[people], return_inverse=True)[1]
            counts = np.bincount(level)
            by_chance = np.sum((counts/max(len(people), 1))**2) # The share at the same level if they were paired at random
            within = np.clip((share - by_chance)/(1 - by_chance) + extra, 0, 1) if by_chance < 1 else 1.0
            codes.append(np.where(u < within, level + 1, 0))
            levels.append(level)
            sizes.append(len(counts) + 1)
        p1, p2 = [], []
        left = np.arange(len(people))
        K = len(codes)
        for keep in [ks for n in range(K, 0, -1) for ks in itertools.combinations(range(K), n)]: # Within the levels of every property each looks within; anyone left over, within those of fewer, and so on
            cell = np.zeros(len(left), dtype=np.int64)
            for k in keep:
                cell = cell*sizes[k] + codes[k][left]
            rest = [left[cell == 0]]
            for c in np.unique(cell[cell > 0]):
                group = left[cell == c]
                n = len(group)//2
                p1.append(group[:n])
                p2.append(group[n:2*n])
                rest.append(group[2*n:])
            left = np.concatenate(rest)
        n = len(left)//2 # The rest at random
        p1, p2 = np.concatenate(p1 + [left[:n]]), np.concatenate(p2 + [left[n:2*n]])
        if len(p1): # People left over without a partner at their level make the share at the same level fall short, so more look within it, smoothed over about 20 steps (as for departures: see end_times())
            for k,(share,level) in enumerate(zip(self.pars.match.values(), levels)):
                self.match_extra[k] = np.clip(self.match_extra[k] + 0.05*(share - np.mean(level[p1] == level[p2])), -1, 1)
        return people[p1], people[p2], people[left[2*n:]]

    def end_times(self, p1):
        """ How many steps each new partnership lasts """
        if self.mean_dur is None:
            return self.pars.dur.rvs(p1)
        u = self.rng_end.rvs(p1)
        p_end = max(1/self.mean_dur - self.departures, 0.1/self.mean_dur) # Partnerships also end when a partner leaves, so they are dissolved less often, to last as long as given on average (as EpiModel's d.rate correction does)
        return np.maximum(np.ceil(np.log1p(-u)/np.log1p(-p_end)), 1) # Geometric, so partnerships end at a constant rate

    def pair(self, p1, p2):
        """ Add partnerships between two lists of people, leaving out pairs who are already partners """
        if len(self):
            existing = set(zip(np.minimum(self.edges.p1, self.edges.p2).tolist(), np.maximum(self.edges.p1, self.edges.p2).tolist()))
            new = zip(np.minimum(p1, p2).tolist(), np.maximum(p1, p2).tolist())
            keep = np.array([pair not in existing for pair in new], dtype=bool)
            p1, p2 = p1[keep], p2[keep]
        keep = p1 != p2
        p1, p2 = p1[keep], p2[keep]
        acts = {} if self.pars.acts is None else dict(acts=self.pars.acts.rvs(p1).astype(ss_float) if isinstance(self.pars.acts, ss.Dist) else np.full(len(p1), self.acts_per_step, dtype=ss_float))
        self.append(p1=p1, p2=p2, beta=np.ones(len(p1), dtype=ss_float), dur=self.end_times(p1), **acts)
        return

    def init_post(self):
        """ Start at the target: give each person a number of partners, and pair them at random (with match=, some within their level) """
        ss.DynamicNetwork.init_post(self, add_pairs=False)
        uids = self.taking_part()
        if self.by_level is not None: # Each level from its own distribution
            level = np.asarray(self.levels(self.by_level))[uids].astype(int)
            cdfs = np.cumsum(self.dists_degree, axis=1)
            u = self.rng_degree.rvs(uids)
            degrees = np.array([np.searchsorted(cdfs[lev], ui) for lev,ui in zip(level, u)], dtype=int)
            degrees = np.minimum(degrees, cdfs.shape[1] - 1)
            shares = np.bincount(level, minlength=len(self.dists_degree))/max(len(level), 1)
            self.dist_degree = shares @ self.dists_degree # The population's target, for the results and check()
        else:
            degrees = self.start_degrees(uids, self.rng_degree)
        stubs = np.repeat(uids, degrees)
        ss.networks.fisher_yates_shuffle(stubs, self.rng_stubs.rng.random(len(stubs)))
        self.rng_stubs.jump() # Reset the RNG manually, as in RandomNet.shuffle_edges()
        rvs = None
        if self.pars.match is not None: # By position, since each person appears once for each partner
            rvs = self.rng_stubs.rng.random((len(self.pars.match), len(stubs)))
            self.rng_stubs.jump()
        p1, p2, left = self.match_pairs(stubs.view(ss.uids), rvs)
        self.pair(p1, p2)
        return

    def remove_uids(self, uids):
        """ Remove the partnerships of people who leave, counting them (see `end_times()`) """
        n = len(self)
        super().remove_uids(uids)
        self.lost += n - len(self)
        return

    def add_pairs(self):
        """ People look for partners at a rate that depends on how many they have, and those looking are paired at random """
        if len(self) or self.lost: # The share of partnerships ended by a partner leaving, smoothed over about 20 steps
            self.departures += 0.05*(self.lost/max(len(self) + self.lost, 1) - self.departures)
            self.lost = 0
        uids = self.taking_part()
        prob = self.seek_prob(uids)
        seek = self.rng_seek.rvs(uids) < prob
        if self.waiting is not None and len(self.waiting): # Those left over last step look again
            seek |= np.isin(uids, self.waiting) & (prob > 0)
        seekers = uids[seek]
        seekers = seekers[np.argsort(self.rng_order.rvs(seekers), kind='stable')]
        p1, p2, left = self.match_pairs(seekers)
        self.waiting = left[-1:]
        n = len(self)
        if len(p1):
            self.pair(p1, p2)
        self.formed = len(self) - n
        return

    def update_results(self):
        super().update_results()
        self.results.new_partnerships[self.ti] = getattr(self, 'formed', 0)
        self.update_degree_results(self.taking_part())
        if len(self):
            for key,name in self.match_results.items():
                level = self.levels(key)
                self.results[name][self.ti] = np.mean(level[self.edges.p1] == level[self.edges.p2])
        return


