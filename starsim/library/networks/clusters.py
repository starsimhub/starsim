"""
Networks of clusters (e.g. households), and Covasim's hybrid population of households, schools, workplaces, and community

These exist for diseases that spread in fixed groups, where the contacts of one person are also contacts of each other. They have no compartmental form, since their edges are between individuals.

Used with: agents; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int
_ = None # Sentinel for "not provided" default arguments


class ClusterNet(ss.Network):
    """
    Fully connected clusters of agents, e.g. households, schools, or workplaces

    Each agent (optionally, only those within an age range) is assigned to one
    cluster, with cluster sizes drawn from `cluster_size`, and every pair of agents
    within a cluster is connected. Agents are assigned to clusters in a random order
    (from a CRN-safe random number per agent), so clusters are not correlated with
    other networks. Based on `make_microstructured_contacts()` from Covasim v3: as
    there, clusters of size 0 are skipped, and the last cluster is truncated to fit.

    Note: CRN safe, unless `dynamic=True` and people are born or die.

    Args:
        cluster_size (int/`ss.Dist`): the size of each cluster; a number is the mean of a Poisson distribution
        age_range (list): if supplied, only agents with `age_range[0] <= age < age_range[1]` are included
        beta (float): the beta value for each edge
        dynamic (bool): if True, the clusters are recreated on each timestep; if False (default), they are created once and never change
        pars (dict): the same parameters, as a dict

    Example:
        ```python
        households = ssl.ClusterNet(name='households', cluster_size=ss.poisson(3))
        sim = ss.Sim(diseases='sis', networks=households)
        sim.run()
        ```
    """
    def __init__(self, pars=None, cluster_size=_, age_range=_, beta=_, dynamic=_, **kwargs):
        super().__init__()
        self.define_pars(
            cluster_size = ss.poisson(lam=4),
            age_range = None,
            beta = 1.0,
            dynamic = False,
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('cluster', label='Cluster ID'),
        )
        self.rng_order = ss.random() # Order in which agents are assigned to clusters
        self.rng_order.hash_dtype = np.float64 # Avoid ties, as for ss.RandomNet(method='safe')
        self.n_clusters = 0
        return

    def step(self):
        """ If dynamic, recreate the clusters; otherwise, do nothing """
        if self.pars.dynamic:
            self.remove_edges(np.ones(len(self), dtype=bool))
            self.cluster.set_nan(self.cluster.auids)
            self.n_clusters = 0
            self.add_pairs()
        return

    def add_pairs(self):
        """ Assign agents to clusters, and connect every pair of agents within each cluster """
        uids = self.eligible(self.pars.age_range).uids
        uids = uids[np.argsort(self.rng_order.rvs(uids), kind='stable')] # Shuffle; use a stable sort since the default order of ties depends on the CPU
        n = len(uids)
        if not n:
            return

        # Draw cluster sizes until there are enough to cover everyone
        sizes = np.empty(0, dtype=ss_int)
        while sizes.sum() < n:
            new = np.round(self.pars.cluster_size.rvs(n)).astype(ss_int) # n clusters is enough unless some have size 0
            if not sizes.size and not np.any(new > 0): # No clusters at all, e.g. a size of 0; as for an ss.RandomNet with no contacts, the network is empty
                return
            sizes = np.concatenate([sizes, new[new > 0]])
        n_clusters = np.searchsorted(np.cumsum(sizes), n) + 1
        sizes = sizes[:n_clusters]
        sizes[-1] -= sizes.sum() - n # Truncate the last cluster
        self.cluster[uids] = self.n_clusters + np.repeat(np.arange(n_clusters), sizes)
        self.n_clusters += n_clusters

        # Connect every pair (i<j) within each cluster, looping over cluster sizes rather than clusters
        starts = np.cumsum(sizes) - sizes
        p1 = [np.empty(0, dtype=ss_int)]
        p2 = [np.empty(0, dtype=ss_int)]
        for size in np.unique(sizes[sizes > 1]):
            these = starts[sizes == size][:, None]
            i, j = np.triu_indices(size, k=1)
            p1.append((these + i).ravel())
            p2.append((these + j).ravel())
        p1 = uids[np.concatenate(p1)]
        p2 = uids[np.concatenate(p2)]
        beta = np.full(len(p1), self.pars.beta, dtype=ss_float)
        self.append(p1=p1, p2=p2, beta=beta)
        return


class HybridNet(ss.Network):
    """
    Covasim v3's "hybrid" population: households, schools, workplaces, and community

    When added to a sim, this expands into four separate networks: households (`h`),
    schools (`s`), and workplaces (`w`), each an `ssl.ClusterNet`, and community (`c`),
    an `ss.RandomNet` with Poisson-distributed contacts. Schools and workplaces are
    restricted by age. As in Covasim v3, the networks are static; unlike it, schools and
    workplaces are clusters by default, so that a person's contacts also know each other
    (use `clustered=False` for Covasim v3's random networks). With the default Poisson
    sizes, clusters give the same mean number of contacts as a random network.
    Each network can then be modified separately, e.g. `sim.networks.s`.

    The default `beta` values are Covasim v3's per-layer relative transmissibility,
    and are applied to the edges of each network; for per-network transmissibility
    on the disease instead, set them to 1 and use e.g. `ss.SIS(beta=dict(h=0.3, s=0.06, w=0.06, c=0.03))`.

    Note: CRN safe, unless `dynamic=True` and people are born or die.

    Args:
        household_size (int/`ss.Dist`): the size of each household; a number is the mean of a Poisson distribution
        contacts (dict): the mean number of contacts per agent in schools, workplaces, and community (numbers or `ss.Dist`s)
        beta (dict): the beta value for each edge in each network
        school_ages (list): the age range of agents in schools
        work_ages (list): the age range of agents in workplaces
        dynamic (bool): whether the networks are recreated on each timestep
        method (str): how the targets of edges in random networks are chosen (see `ss.RandomNet`); 'uniform' as in Covasim v3
        clustered (bool): if True (default), schools and workplaces are `ssl.ClusterNet`s, sized to give `contacts` for Poisson and constant distributions (other distributions are used directly as cluster sizes, which gives a different mean); if False, they are `ss.RandomNet`s, as in Covasim v3
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        sim = ss.Sim(diseases='sis', networks=ssl.HybridNet())
        sim.run()
        print(sim.networks.keys()) # ['h', 's', 'w', 'c']

        # Larger households and smaller schools; contacts and beta can be partial
        hybrid = ssl.HybridNet(household_size=ss.poisson(3), contacts=dict(s=10))
        ```
    """
    def __init__(self, pars=None, household_size=_, contacts=_, beta=_, school_ages=_, work_ages=_, dynamic=_, method=_, clustered=_, **kwargs):
        super().__init__()
        self.define_pars(
            household_size = ss.poisson(lam=2.0), # Covasim v3 default (contacts['h'])
            contacts = dict(s=20, w=16, c=20),
            beta = dict(h=3.0, s=0.6, w=0.6, c=0.3),
            school_ages = [6, 22],
            work_ages = [22, 65],
            dynamic = False,
            method = 'fast',
            clustered = True,
        )
        defaults = sc.dcp(sc.obj(contacts=self.pars.contacts, beta=self.pars.beta))
        self.update_pars(pars, **kwargs)
        for key,default in defaults.items(): # Allow partial dicts, e.g. contacts=dict(s=10)
            self.pars[key] = sc.mergedicts(default, self.pars[key])
        return

    def expand(self):
        """ Create the four networks; called automatically when added to a sim """
        p = self.pars
        n = {k: ss.poisson(lam=v) if sc.isnumber(v) else v for k,v in p.contacts.items()}
        kw = dict(dynamic=p.dynamic, dt=self.t.dt, start=self.t.start, stop=self.t.stop) # Pass the timeline to each network # TODO: make this more general for one module creating another
        rkw = dict(method=p.method, **kw)

        def group(name, label, ages):
            """ Make a school or workplace network """
            if p.clustered:
                size = n[name]
                if isinstance(size, ss.constant): # A cluster of N people gives each N-1 contacts (Poisson sizes already give the right mean)
                    size = ss.constant(size.pars.v + 1)
                return ClusterNet(name=name, label=label, cluster_size=size, age_range=ages, beta=p.beta[name], **kw)
            return ss.RandomNet(name=name, label=label, n_contacts=n[name], age_range=ages, beta=p.beta[name], **rkw)

        networks = [
            ClusterNet(name='h', label='Households', cluster_size=p.household_size, beta=p.beta['h'], **kw),
            group('s', 'Schools', p.school_ages),
            group('w', 'Workplaces', p.work_ages),
            ss.RandomNet(name='c', label='Community', n_contacts=n['c'], beta=p.beta['c'], **rkw),
        ]
        return networks

    def step(self):
        """ Not used, since the networks returned by expand() are used instead """
        errormsg = 'ssl.HybridNet() should be added to a sim, which replaces it with the networks from expand()'
        raise RuntimeError(errormsg)
