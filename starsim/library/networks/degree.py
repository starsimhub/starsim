"""
Partnerships with a target number of partners at a time, as in EpiModel's networks fitted to target statistics

`DegreeNet` is `ss.PartnerNet` with a target distribution of the number of partners people
have at a time (a mean, a share with two or more, a maximum), partnerships within a level of a
property of people (EpiModel's nodematch), and a number of partners for each level (nodefactor);
`DegreeMFNet` is `ss.MFNet` with the same targets. `ss.interop.load(..., format='epimodel')`
makes a `DegreeNet` from a network fitted by EpiModel's `netest()`.
"""
import itertools
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import sciris as sc
import starsim as ss

_ = None


class DegreeTargets:
    """
    Targets for the number of partners people have at a time, for `DegreeNet` and `DegreeMFNet`

    The target is a distribution of the number of partners (`degree_dist()`). Partnerships
    end at random, and people look for new ones with a probability that depends on how many
    they already have (`p_seek`), chosen so that the target is the stable state.
    """
    @staticmethod
    def degree_dist(n_contacts, concurrency=None, max_contacts=None):
        """ The target share of people with 0, 1, 2, ... partners at a time """
        if isinstance(n_contacts, ss.by):
            errormsg = f'n_contacts={n_contacts} gives each level its own number of partners, which ssl.DegreeNet takes; here give one number or distribution'
            raise ValueError(errormsg)
        if sc.isnumber(n_contacts):
            n_contacts = ss.poisson(lam=n_contacts)
        pars = {k:v for k,v in n_contacts.pars.items()}
        top = max_contacts if max_contacts is not None else None
        if isinstance(n_contacts, ss.poisson):
            mean = pars['lam']
            k = np.arange(sc.ifelse(top, int(mean + 10*np.sqrt(mean) + 10)) + 1)
            base = sps.poisson.pmf(k, mean)
        elif isinstance(n_contacts, ss.constant):
            mean = int(pars['v'])
            k = np.arange(sc.ifelse(top, mean) + 1)
            base = (k == mean).astype(float)
        else: # Any other distribution, by sampling a copy of it
            draws = np.round(sc.dcp(n_contacts).init(force=True).rvs(100_000)).astype(int)
            mean = draws.mean()
            k = np.arange(sc.ifelse(top, draws.max()) + 1)
            base = np.bincount(np.clip(draws, 0, k[-1]), minlength=len(k)).astype(float)
        if concurrency is None and max_contacts is None:
            return base/base.sum()

        def tilt(x): # Reweight the distribution exponentially in the number of partners and in having two or more
            w = base*np.exp(x[0]*k + (x[1] if len(x) > 1 else 0)*(k >= 2))
            return w/w.sum()
        if concurrency is None:
            x = [spo.brentq(lambda a: tilt([a]) @ k - mean, -50, 50)]
        else:
            x = spo.fsolve(lambda x: [tilt(x) @ k - mean, tilt(x)[2:].sum() - concurrency], [0, 0])
        out = tilt(x)
        if abs(out @ k - mean) > 1e-6 or (concurrency is not None and abs(out[2:].sum() - concurrency) > 1e-6):
            errormsg = f'No distribution of the number of partners has a mean of {mean:n}{f", {concurrency:n} with two or more," if concurrency is not None else ""} and a maximum of {max_contacts}'
            raise ValueError(errormsg)
        return out

    def init_seek(self, mean_dur):
        """ The probability that someone with each number of partners looks for another on a step, given how many steps partnerships last on average """
        k = np.arange(len(self.dist_degree))
        p_end = 1/mean_dur
        ratio = np.divide(self.dist_degree[1:], self.dist_degree[:-1], out=np.zeros(len(k) - 1), where=self.dist_degree[:-1] > 0)
        self.p_seek = np.minimum(np.append((k[:-1] + 1)*p_end*ratio, 0), 1) # Chosen so the target is the stationary distribution: those with k partners who gain one balance those with k+1 who lose one
        return

    def start_degrees(self, uids, rng):
        """ A number of partners for each person, from the target, to start with """
        cdf = np.cumsum(self.dist_degree)
        return np.minimum(np.searchsorted(cdf, rng.rvs(uids)), len(cdf) - 1)

    def seeking(self, uids, rng):
        """ The people who look for a partner on this step """
        degree = np.minimum(self.degrees(uids), len(self.p_seek) - 1)
        return uids[rng.rvs(uids) < self.p_seek[degree]]

    def degrees(self, uids):
        """ The number of partners each person has """
        size = int(uids.max()) + 1 if len(uids) else 0
        counts = np.bincount(self.edges.p1, minlength=size)[:size] + np.bincount(self.edges.p2, minlength=size)[:size]
        return counts[uids]

    def people_seeking(self, people):
        """ The people the targets are for: see the subclasses """
        return people.alive.uids

    def define_degree_results(self):
        self.define_results(
            ss.Result('mean_degree', label='Mean number of partners', scale=False, auto_plot=False), # Per person, so not scaled by the population
            ss.Result('concurrency', label='Share with two or more partners', scale=False, auto_plot=False),
        )
        return

    def update_degree_results(self, uids):
        degree = self.degrees(uids)
        self.results.mean_degree[self.ti] = degree.mean() if len(degree) else 0
        self.results.concurrency[self.ti] = (degree >= 2).mean() if len(degree) else 0
        return

    def check(self, n_runs=3, output=False):
        """
        Run the sim again, and compare the network it makes with the network's targets

        Reports, for each target (the mean number of partners, the share with two or
        more, the maximum, and with `match=`, the share of partnerships at the same
        level), the target, the mean over the second half of each run, and the spread
        across runs; a target that is missed by more than the spread is an `ss.TargetStatsNotRecovered` warning.

        Args:
            n_runs (int): the number of runs
            output (bool): return the table instead of printing it
        """
        if not hasattr(self.sim, 'inputs'):
            errormsg = f'net.check() reruns the sim, which can only be rebuilt from the inputs it was made with; the network\'s results (mean_degree, concurrency) show how close it is to the targets'
            raise ValueError(errormsg)
        rows = []
        for i in range(n_runs):
            run = self.sim.new(rand_seed=i + 1, verbose=0).run()
            net = run.networks[self.name]
            res = net.results
            half = len(res.mean_degree)//2
            row = dict(mean_degree=res.mean_degree[half:].mean(), concurrency=res.concurrency[half:].mean(), max_contacts=net.degrees(net.people_seeking(run.people)).max())
            if 'new_partnerships' in res and getattr(net, 'mean_dur', None) is not None and res.new_partnerships[half:].mean() > 0: # Little's law: partnerships = rate formed × how long they last
                n_alive = len(run.people.alive.uids)
                row['dur'] = res.mean_degree[half:].mean()*n_alive/2/res.new_partnerships[half:].mean()
            if getattr(net, 'by_level', None) is not None:
                uids = run.people.alive.uids
                level = np.asarray(net.levels(net.by_level))[uids].astype(int)
                degree = net.degrees(uids)
                for lev in range(len(net.dists_degree)):
                    row[f'mean_degree[{lev}]'] = degree[level == lev].mean() if np.any(level == lev) else np.nan
            for key,name in getattr(self, 'match_results', {}).items():
                row[name] = res[name][half:].mean()
            rows.append(row)
        df = sc.dataframe(rows)
        k = np.arange(len(self.dist_degree))
        targets = dict(mean_degree=self.dist_degree @ k, concurrency=self.dist_degree[2:].sum(), max_contacts=self.pars.get('max_contacts'), dur=getattr(self, 'mean_dur', None))
        if getattr(self, 'by_level', None) is not None:
            targets.update({f'mean_degree[{lev}]':d @ np.arange(len(d)) for lev,d in enumerate(self.dists_degree)})
        for key,name in getattr(self, 'match_results', {}).items():
            targets[name] = self.pars.match[key]
        table = sc.dataframe(dict(target=[targets[c] for c in df.columns], realized=df.mean().values, spread=df.std().values), index=df.columns)
        for name,row in table.iterrows():
            if row.target is None:
                continue
            missed = row.realized > row.target if name == 'max_contacts' else abs(row.realized - row.target) > max(2*row.spread, 0.02*abs(row.target))
            if missed:
                ss.warn(f'the network "{self.name}" has {name} {row.realized:.3g}, against a target of {row.target:.3g}', ss.TargetStatsNotRecovered)
        if output:
            return table
        print(table)
        return



class DegreeNet(DegreeTargets, ss.PartnerNet):
    """
    Partnerships that form and dissolve, with a target number of partners at a time

    The number of partners people have at any moment (`n_contacts`) is a target
    for the whole population, not a fixed number for each person: partnerships end
    at random, and people look for new ones at a rate that depends on how many
    they already have, chosen so that the distribution of the number of partners
    stays at the target. Everyone moves through it over time, as in EpiModel's
    networks fitted to target statistics. The network starts at the target, and
    partnerships ended by a partner leaving (dying) are allowed for, as EpiModel's
    d.rate correction does, so they last as long as given on average. It takes the
    arguments of `ss.PartnerNet` (except one-off contacts), and:

    Args:
        n_contacts (float/`ss.Dist`/`ss.by`/dict): the number of partners a person has at a time: a distribution over whole numbers (e.g. `ss.poisson(0.7)`), or a number, which is the mean of a Poisson; or a number for each level of a property of people (`ss.by(risk=[0.4, 1.6])`), as EpiModel's nodefactor; or with `between=`, one for each group
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
        ```
    """
    def __init__(self, pars=None, n_contacts=_, dur=_, concurrency=_, max_contacts=_, match=_, acts=_, sides=_, between=_, where=_, beta=_, **kwargs):
        super().__init__()
        self.define_pars(
            concurrency = None,
            max_contacts = None,
            match = None,
        )
        self.update_pars(pars, **kwargs)
        self.set_sides()
        self.rng_match  = ss.random() # Whether each person looks for a partner at their own level, with match=
        for key in list(self.pars.match or {})[1:]: # And of each other property, with several
            setattr(self, f'rng_match_{key}', ss.random())
        self.match_extra = np.zeros(len(self.pars.match or {})) # For each property, how many more look within their level than by the target alone: see match_pairs()
        self.match_results = {key:'match' if len(self.pars.match) == 1 else f'match_{key}' for key in self.pars.match or {}} # The result of the share at the same level of each property
        self.departures = 0.0 # The share of partnerships ended each step by a partner leaving (dying), estimated as the run goes
        self.lost = 0 # Partnerships ended that way on this step
        return

    def init_seeking(self, mean_dur):
        """ The target distribution of the number of partners (of each level, with `ss.by` or `between=`), and the probability of looking for a partner with each number """
        p = self.pars
        self.by_level = None # With n_contacts=ss.by(...): the property of people whose levels have degrees of their own, and the distribution and seeking probabilities of each
        if p.between is not None: # Each group has its own number of partners, as levels do with ss.by()
            given = p.n_contacts if isinstance(p.n_contacts, dict) else {k:p.n_contacts for k in self.sides}
            if set(given) != set(self.sides):
                errormsg = f'n_contacts= of "{self.name}" is a number for each of its groups ({sc.strjoin(self.sides)}), not {p.n_contacts}'
                raise ValueError(errormsg)
            self.by_level = '_side'
            self.dists_degree = [self.degree_dist(given[k], p.concurrency, p.max_contacts) for k in self.sides]
        elif isinstance(p.n_contacts, ss.by):
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

    def seek_prob(self, uids):
        """ The probability that each person looks for a partner on this step, given how many they have """
        degree = np.minimum(self.degrees(uids), len(self.p_seek) - 1)
        if self.by_level is not None:
            return self.p_seeks[np.asarray(self.levels(self.by_level))[uids].astype(int), degree]
        return self.p_seek[degree]

    def init_results(self):
        super().init_results()
        self.define_degree_results()
        for key,name in self.match_results.items():
            self.define_results(ss.Result(name, label=f'Share of partnerships at the same level{"" if name == "match" else f" of {key}"}', auto_plot=False))
        return

    def levels(self, name=None):
        """ The level of every person in a property of people (by default, the one match= names), as an array indexed by UID """
        sim = self.sim
        name = list(self.pars.match.keys())[0] if name is None else name
        if name == '_side': # The group of each person, with between=
            return self.side_raw()
        names = [strat.name for strat in sim.groups.strats]
        if name in names: # A stratification
            return sim.groups.level_raw(names.index(name))
        return np.asarray(sim.people[name].raw)

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
        if self.pars.match is None:
            return super().match_pairs(people)
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
        return super().add_pairs()

    def update_results(self):
        super().update_results()
        self.update_degree_results(self.taking_part())
        if len(self):
            for key,name in self.match_results.items():
                level = self.levels(key)
                self.results[name][self.ti] = np.mean(level[self.edges.p1] == level[self.edges.p2])
        return


class DegreeMFNet(DegreeTargets, ss.MFNet):
    """
    Partnerships between men and women (`ss.MFNet`), with a target number of partners at a time

    Partnerships end when their duration is up, and people look for new ones at a rate
    that depends on how many they have, so the distribution of the number of partners
    (among people who take part, after their debut) stays at the target. In an open
    population, where partners die and newcomers start with none, people look more (or
    less) often by a factor that keeps the mean number of partners at the target; with
    partnerships much longer than people's time in the network, it takes a few decades to
    settle. The network starts at the target, and its results and `net.check()` say how
    close it is. It takes the arguments of `ss.MFNet`, and:

    Args:
        n_contacts (float/`ss.Dist`): the number of partners people have at a time: a distribution over whole numbers, or a number, which is the mean of a Poisson
        concurrency (float): the share of people with two or more partners at once
        max_contacts (int): the most partners anyone has at once

    Examples:
        ```python
        import starsim.library as ssl
        net = ssl.DegreeMFNet(n_contacts=ss.poisson(0.7), dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)), concurrency=0.1)
        ```
    """
    def __init__(self, pars=None, n_contacts=_, concurrency=_, max_contacts=_, **kwargs):
        super().__init__()
        self.define_pars(
            n_contacts = None,
            concurrency = None,
            max_contacts = None,
        )
        self.update_pars(pars, **kwargs)
        self.rng_degree = ss.random() # The number of partners each person starts with
        self.rng_seek   = ss.random() # Whether each person looks for a partner on this step
        self.rng_order  = ss.random() # The order in which people are paired
        self.waiting = ss.uids() # People of one sex who looked for a partner and were left over, who look again
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        p = self.pars
        if p.n_contacts is None:
            errormsg = f'"{self.name}" needs n_contacts=, the target number of partners at a time (concurrency= and max_contacts= are targets as well)'
            raise ValueError(errormsg)
        self.dist_degree = self.degree_dist(p.n_contacts, p.concurrency, p.max_contacts)
        mean = ss.transitions.dwell_stats(p.dur, lambda v: ss.solvers.convert(v, sim.unit_base))[0]
        self.mean_steps = mean/self.t.dt.to(sim.unit_base).value
        self.init_seek(self.mean_steps)
        self.boost = 1.0 # How much more (or less) often people look than in a closed population, adjusted as the run goes: see add_pairs()
        return

    def init_results(self):
        super().init_results()
        self.define_degree_results()
        return

    def people_seeking(self, people):
        """ The people who take part, after their debut """
        return self.active(people).uids

    def init_post(self):
        """ Start at the target: give each person who takes part a number of partners, and pair men with women at random """
        self.set_network_states()
        ss.SexualNetwork.init_post(self, add_pairs=False)
        people = self.sim.people
        uids = self.people_seeking(people)
        stubs = np.repeat(uids, self.start_degrees(uids, self.rng_degree))
        rng = self.rng_order.rng # By position, since each person appears once for each partner
        p1, p2 = self.pair_sexes(stubs, rng.random(len(stubs)))
        n = len(p1)
        dur = self.pars.dur.rvs(3*n) # What is left of partnerships under way: one picked in proportion to how long it lasts, and a uniform share of it
        dur = dur[rng.choice(len(dur), size=n, p=dur/dur.sum())]*rng.random(n) if n else dur[:0]
        self.rng_order.jump()
        self.add(p1, p2, dur)
        return

    def pair_sexes(self, people, order):
        """ Pair the men and the women among some people, in the order of a random number for each, leaving out pairs who are already partners; anyone left over waits """
        people = people[np.argsort(order, kind='stable')]
        female = self.sim.people.female.raw[people]
        men, women = people[~female], people[female]
        n = min(len(men), len(women))
        p1, p2 = men[:n], women[:n]
        self.waiting = ss.uids.concatenate([men[n:], women[n:]]).unique() # Otherwise the surplus of one sex would be lost on every step
        if len(self) and n:
            existing = set(zip(self.edges.p1.tolist(), self.edges.p2.tolist()))
            keep = np.array([pair not in existing for pair in zip(p1.tolist(), p2.tolist())], dtype=bool)
            p1, p2 = p1[keep], p2[keep]
        return p1, p2

    def add(self, p1, p2, dur=None):
        """ Add partnerships, with their durations (drawn if not given) and acts """
        if len(p1):
            dur = self.pars.dur.rvs(len(p1)) if dur is None else dur
            self.append(p1=p1, p2=p2, beta=np.ones(len(p1)), dur=dur, acts=self.draw_acts(len(p1)))
        return len(p1)

    def add_pairs(self):
        """ Those looking for a partner on this step, given how many they have, are paired, men with women """
        active = self.people_seeking(self.sim.people)
        seekers = self.seeking(active, self.rng_seek)
        waiting = self.waiting[np.isin(self.waiting, active)]
        waiting = waiting[self.p_seek[np.minimum(self.degrees(waiting), len(self.p_seek) - 1)] > 0] # Unless they have as many partners as anyone can now
        seekers = ss.uids.concatenate([seekers, waiting]).unique()
        p1, p2 = self.pair_sexes(seekers, self.rng_order.rvs(seekers))
        n = self.add(p1, p2)
        if len(active): # In an open population, partners leave and newcomers start with none, so people look more (or less) often, by as much as keeps the mean number of partners at the target (a factor adjusted by a fifth of the gap each step, on a log scale)
            target = self.dist_degree @ np.arange(len(self.dist_degree))
            mean = self.degrees(active).mean()
            gap = target/max(mean, 1e-3*target)
            self.boost = np.clip(self.boost*gap**0.5, 0.01, 1000.0)
            self.init_seek(self.mean_steps/(self.boost*gap)) # The gap itself too, which damps the swings of the slow adjustment
        return n

    def update_results(self):
        super().update_results()
        self.update_degree_results(self.people_seeking(self.sim.people))
        return
