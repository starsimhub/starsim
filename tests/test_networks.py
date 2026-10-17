"""
Test networks
"""

# %% Imports and settings
import pytest
import warnings
import sciris as sc
import numpy as np
import starsim as ss
import starsim.library as ssl
import scipy.stats as sps
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100
medium = 1000

# %% Define the tests

@sc.timer()
def test_manual():
    sc.heading('Testing manual networks')

    # Make completely abstract layers
    n_edges = 10_000
    n_agents = medium
    p1 = np.random.randint(n_agents, size=n_edges)
    p2 = np.random.randint(n_agents, size=n_edges)
    beta = np.ones(n_edges)
    nw1 = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')

    # Create a maternal network
    sim = ss.Sim(n_agents=n_agents)
    sim.init()
    nw2 = ss.PrenatalNet()
    nw2.init_pre(sim)
    nw2.add_pairs(mother_uids=[1, 2, 3], unborn_uids=[100, 101, 102])

    # Tidy
    o = sc.objdict(nw1=nw1, nw2=nw2)
    return o


@sc.timer()
def test_random():
    sc.heading('Testing random networks')

    # Manual creation
    nw1 = ss.RandomNet()
    ss.Sim(n_agents=small, networks=nw1, copy_inputs=False).init() # This initializes the network

    # Automatic creation as part of sim
    s2 = ss.Sim(n_agents=small, networks='random').init()
    nw2 = s2.networks[0]

    # Increase the number of contacts
    nwdict = dict(type='random', n_contacts=20)
    s3 = ss.Sim(n_agents=small, networks=nwdict).init()
    nw3 = s3.networks[0]

    # Checks
    assert np.array_equal(nw1.p2, nw2.p2), 'Implicit and explicit creation should give the same network'
    assert len(nw3) == len(nw2)*2, 'Doubling n_contacts should produce twice as many contacts'

    # Tidy
    o = sc.objdict(nw1=nw1, nw2=nw2, nw3=nw3)
    return o


@sc.timer()
def test_randomfast():
    sc.heading('Testing the (default, fast) RandomNet vs the exact RandomExactNet')
    n_contacts = 10

    s1 = ss.Sim(n_agents=medium, networks=ss.RandomExactNet(n_contacts=n_contacts), rand_seed=1).init()
    s2 = ss.Sim(n_agents=medium, networks=ss.RandomNet(n_contacts=n_contacts), rand_seed=1).init()
    nw1 = s1.networks[0]
    nw2 = s2.networks[0]

    # Same number of edges as RandomExactNet, and the mean degree matches n_contacts
    assert len(nw1) == len(nw2), 'RandomNet should produce the same number of edges as RandomExactNet'
    deg = np.bincount(np.concatenate([np.asarray(nw2.p1), np.asarray(nw2.p2)]), minlength=medium)
    assert np.isclose(deg.mean(), n_contacts, rtol=0.05), f'Mean degree should be ~{n_contacts}, not {deg.mean()}'

    # The fast version has variable (Poisson-like) degree, unlike RandomExactNet's fixed degree
    deg1 = np.bincount(np.concatenate([np.asarray(nw1.p1), np.asarray(nw1.p2)]), minlength=medium)
    assert deg1.std() == 0, 'RandomExactNet should have fixed degree'
    assert deg.std() > 1, 'RandomNet should have variable degree'

    # Reproducible given the same seed
    s3 = ss.Sim(n_agents=medium, networks=ss.RandomNet(n_contacts=n_contacts), rand_seed=1).init()
    assert np.array_equal(nw2.p2, s3.networks[0].p2), 'RandomNet should be reproducible for a fixed seed'

    # With uniform targets, the variance in degree is lower: about n_contacts/2 from the sources plus n_contacts/2 from the targets (as in Covasim v3)
    n = 10_000
    s4 = ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=ss.poisson(20), uniform_targets=True), rand_seed=1).init()
    nw4 = s4.networks[0]
    deg4 = np.bincount(np.concatenate([np.asarray(nw4.p1), np.asarray(nw4.p2)]), minlength=n)
    assert np.isclose(deg4.mean(), 20, rtol=0.05), f'Mean degree should be ~20, not {deg4.mean()}'
    assert np.isclose(deg4.var(), 15, rtol=0.1), f'Degree variance with uniform targets should be ~15, not {deg4.var()}'

    o = sc.objdict(nw1=nw1, nw2=nw2, nw4=nw4)
    return o


@sc.timer()
def test_randomsafe():
    sc.heading('Testing the RandomSafe network')

    def auid_similarity(s1, s2):
        """ Compute similarity between UID lists """
        auids1 = s1.people.auids
        auids2 = s2.people.auids
        similarity = sc.similarity(auids1, auids2)
        return similarity

    # Set up a sim with some small number of births and deaths
    pars = sc.objdict(n_agents=small, dur=5, verbose=False)
    births = ss.Births(birth_rate=ss.freqperyear(5))
    simil = sc.objdict()

    # Confirm that non-safe networks diverge immediately
    s1 = ss.Sim(pars, networks='random', label='random-nobirths').run()
    s2 = ss.Sim(pars, networks='random', label='random-births', demographics=births).run()

    n1 = s1.networks[0].to_edgelist()
    n2 = s2.networks[0].to_edgelist()
    simil.uid = auid_similarity(s1, s2)
    simil.rand = sc.similarity(n1, n2)
    print(f'Similarity in UIDs after {pars.dur} timesteps with/without births:\n{simil.uid:%}')
    print(f'Similarity for random networks after {pars.dur} timesteps:\n{simil.rand:%}\n')
    assert simil.rand < 0.5, 'Random networks were more similar than expected'

    # Confirm that safe networks don't
    s3 = ss.Sim(pars, networks='randomsafe', label='safe-nobirths').run()
    s4 = ss.Sim(pars, networks='randomsafe', label='safe-births', demographics=births).run()

    n3 = s3.networks[0].to_edgelist()
    n4 = s4.networks[0].to_edgelist()
    simil.uid2 = auid_similarity(s3, s4)
    simil.safe = sc.similarity(n3, n4)
    print(f'Similarity in UIDs after {pars.dur} timesteps with/without births:\n{simil.uid2:%}')
    print(f'Similarity for random-safe networks after {pars.dur} timesteps:\n{simil.safe:%}\n')
    assert simil.safe > 0.5, 'RandomSafe networks were less similar than expected'

    # With many agents, some random numbers are tied; check that these edges don't all go to agent 0
    n = 20_000
    s5 = ss.Sim(n_agents=n, networks=ss.RandomSafeNet(n_edges=10), rand_seed=1).init()
    nw5 = s5.networks[0]
    deg5 = np.bincount(np.concatenate([np.asarray(nw5.p1), np.asarray(nw5.p2)]), minlength=n)
    assert deg5.max() < 40, f'No agent should have many more than 20 contacts, but agent {deg5.argmax()} has {deg5.max()}'

    o = sc.objdict(s1=s1, s2=s2, s3=s3, s4=s4, similarity=simil)
    return o


@sc.timer()
def test_random_stratified():
    """ Test that RandomNet reproduces the mixing structure of a risk-stratified network (issue #1399) """
    sc.heading('Testing RandomNet with risk stratification')
    n = 20_000
    fracs = [0.75, 0.20, 0.05]
    contacts = np.array([0.25, 0.93, 8.55]) # Realized degree per risk group
    rng = np.random.default_rng(1)
    group = rng.choice([0, 1, 2], size=n, p=fracs)
    n_contacts = ss.constant(v=lambda self, sim, uids: contacts[group[np.asarray(uids)]])

    def measure(net):
        sim = ss.Sim(n_agents=n, networks=net, rand_seed=1, verbose=0).init()
        nw = sim.networks[0]
        p1, p2 = np.asarray(nw.p1), np.asarray(nw.p2)
        deg = np.bincount(np.concatenate([p1, p2]), minlength=n)
        degrees = np.array([deg[group == g].mean() for g in range(3)])
        core = np.mean((group[p1] == 2) & (group[p2] == 2)) # Fraction of edges within the high-risk core
        return degrees, core

    exact_deg, exact_core = measure(ss.RandomExactNet(n_contacts=n_contacts))
    fast_deg, fast_core = measure(ss.RandomNet(n_contacts=n_contacts))
    print(f'Requested degrees:      {contacts}')
    print(f'RandomExactNet degrees: {exact_deg}, core edge fraction: {exact_core:n}')
    print(f'RandomNet degrees:      {fast_deg}, core edge fraction: {fast_core:n}')

    # The realized degree per risk group must match what was requested, not just on average
    assert np.allclose(fast_deg, contacts, rtol=0.1), f'RandomNet degrees {fast_deg} do not match requested {contacts}'
    assert np.allclose(fast_deg, exact_deg, rtol=0.1), f'RandomNet degrees {fast_deg} do not match RandomExactNet {exact_deg}'

    # Most edges should involve the high-risk core on both ends, as with RandomExactNet
    assert np.isclose(fast_core, exact_core, rtol=0.1), f'RandomNet core clustering {fast_core:n} does not match RandomExactNet {exact_core:n}'

    o = sc.objdict(exact_deg=exact_deg, fast_deg=fast_deg, exact_core=exact_core, fast_core=fast_core)
    return o


@sc.timer()
def test_network_uids():
    """ Test that all networks form edges between valid UIDs, not positional indices (issue #1418) """
    sc.heading('Testing that network edges reference valid UIDs')
    netclasses = [ss.RandomNet, ss.RandomExactNet, ss.RandomSafeNet]
    out = sc.objdict()
    for netclass in netclasses:
        name = netclass.__name__
        # A sim with births and deaths, so the UIDs of living agents are not contiguous
        sim = ss.Sim(n_agents=medium, networks=netclass(), diseases='sis', demographics=True, start=2000, stop=2005, rand_seed=1, verbose=0).run()
        nw = sim.networks[0]
        alive = set(sim.people.auids.tolist())
        refs = set(np.concatenate([np.asarray(nw.p1), np.asarray(nw.p2)]).tolist())
        invalid = sorted(refs - alive)
        print(f'{name}: {len(nw)} edges, {len(invalid)} references to non-living agents')
        assert not len(invalid), f'{name} formed edges with {len(invalid)} non-living agents: {invalid[:10]}'
        assert isinstance(nw.p1, ss.uids), f'{name}.p1 should be ss.uids, not {type(nw.p1)}'
        assert isinstance(nw.p2, ss.uids), f'{name}.p2 should be ss.uids, not {type(nw.p2)}'
        out[name] = sim
    return out



@sc.timer()
def test_erdosrenyi():
    sc.heading('Testing Erdos-Renyi network')

    def test_ER(n, p, nw, alpha=0.01):
        """
        Because each edge exists i.i.d. with probability p, the degree
        distribution of an Erdos-Renyi network should be Binomally distributed.
        Here, we test if the observed degree distribution, f_obs, matches the
        expected distribution, f_exp, which comes from the binomial probabiltiy
        mass function from the scipy stats library.

        Args:
            n (int): number of agents
            p (float): edge probability
            nw (Network): The network to test
            alpha (float): The test significance level

        Returns:
            Chi-Squared p-value and asserts if the statistical test fails,
            indicating that the observed degree distribution is unlikely to come
            from the the correct binomial.
        """
        p12 = np.concatenate([nw.edges['p1'], nw.edges['p2']]) # p1 and p2 are the UIDs of agents, used here to determine degree
        upper = sps.binom.ppf(n=n-1, p=p, q=0.999) # Figure out the 99.9th percentile expected upper bound on the degree
        bins = np.arange(upper+1) # Create bins
        counts = np.histogram(p12, bins=np.arange(n+1))[0] # Count how many times each agent is connected
        f_obs = np.histogram(counts, bins=bins)[0] # Form the degree distribution
        pp = sps.binom.pmf(bins[:-1], n=n, p=p) # Computed the theoretical probability distribution
        f_exp = f_obs.sum()*pp / pp.sum() # Scale
        if f_obs.sum() and f_exp.sum(): # Skip if zero
            p_value = sps.chisquare(f_obs, f_exp).pvalue # Compute the X2 p-value
            assert not p_value < alpha
        else:
            p_value = 1.0
        return p_value

    # Manual creation
    p = 0.1
    nw1 = ssl.networks.ErdosRenyiNet(p=p)
    ss.Sim(n_agents=small, networks=nw1, copy_inputs=False).init() # This initializes the network
    test_ER(small, p, nw1)

    # Automatic creation as part of sim
    ss.register_modules(ssl.networks)
    s2 = ss.Sim(n_agents=small, networks='erdosrenyi').init()
    nw2 = s2.networks[0]

    # Larger example with higher p
    p=0.6
    nwdict = dict(type='erdosrenyi', p=p)
    s3 = ss.Sim(n_agents=medium, networks=nwdict).init()
    nw3 = s3.networks[0]
    test_ER(medium, p, nw3)

    # Checks
    assert np.array_equal(nw1.p2, nw2.p2), 'Implicit and explicit creation should give the same network'

    # Tidy
    o = sc.objdict(nw1=nw1, nw2=nw2, nw3=nw3)
    return o


@sc.timer()
def test_disk():
    sc.heading('Testing Disk network')

    # Visualize the path of agents
    nw1 = ssl.networks.DiskNet()
    s1 = ss.Sim(n_agents=5, dur=ss.days(50), networks=nw1, copy_inputs=False).init() # This initializes the network

    if sc.options.interactive:
        fig, ax = plt.subplots()
        vdt = nw1.pars.v * s1.t.dt

        colors = sc.vectocolor(np.linspace(0, 1, s1.pars.n_agents))
        ax.scatter(nw1.x, nw1.y, s=50, c=colors)
        for i in range(s1.t.npts):
            ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k-', lw=1)
            ax.quiver(nw1.x, nw1.y, vdt * np.cos(nw1.theta), vdt * np.sin(nw1.theta), color=colors)
            ax.set_aspect('equal', adjustable='box') #ax.set_xlim([0,1]); ax.set_ylim([0,1])
            s1.run_one_step()

    # Simulate SIR on a DiskNet
    nw2 = ssl.networks.DiskNet(r=0.15, v=ss.freq(0.05, unit=ss.year))
    s2 = ss.Sim(n_agents=small, networks=nw2, diseases='sir').init() # This initializes the network
    s2.run()

    if sc.options.interactive:
        s2.plot()
        plt.show()

    return s1, s2


@sc.timer()
def test_static():
    sc.heading('Testing static networks')

    # Create with p
    p = 0.2
    n = 100
    nc = p*n
    nd1 = dict(type='static', p=p)
    nw1 = ss.Sim(n_agents=n, networks=nd1).init().networks[0]

    # Create with n_contacts
    nd2 = dict(type='static', n_contacts=nc)
    nw2 = ss.Sim(n_agents=n, networks=nd2).init().networks[0]

    # Check
    assert len(nw1) == len(nw2), 'Networks should be the same length'
    target = n*n*p/2
    assert target/2 < len(nw1) < target*2, f'Network should be approximately length {target}'

    # Tidy
    o = sc.objdict(nw1=nw1, nw2=nw2)
    return o


@sc.timer()
def test_null():
    sc.heading('Testing NullNet...')
    people = ss.People(n_agents=small)
    network = ssl.networks.NullNet()
    sir = ss.SIR(dur_inf=10, beta=0.1)
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    return sim


def make_dhs_data(n=100, include_sexes=False):
    """ Helper to construct synthetic household data """
    age_strings = []
    sex_strings = []
    for i in range(n):
        household_size = np.random.randint(1, 6)
        ages = np.random.randint(0, 80, household_size)
        age_strings.append(sc.strjoin(ages))
        if include_sexes:
            sexes = np.random.choice([1, 2], household_size)  # DHS convention: 1=male, 2=female
            sex_strings.append(sc.strjoin(sexes))
    if include_sexes:
        return sc.dataframe(hh_id=np.arange(n), ages=age_strings, sexes=sex_strings)
    return sc.dataframe(hh_id=np.arange(n), ages=age_strings)


@sc.timer()
def test_household():
    sc.heading('Testing HouseholdNet...')
    dhs_data = make_dhs_data(n=small)

    # Test ssl.networks.HouseholdNet (static)
    household = ssl.networks.HouseholdNet(dhs_data=dhs_data, dynamic=False)
    sim = ss.Sim(n_agents=small, diseases='sis', networks=household)
    sim.run()

    # Check that household_ids are assigned
    assert np.all(np.isfinite(household.household_ids[:]))

    # Test with sexes provided
    dhs_data_sex = make_dhs_data(include_sexes=True)
    household2 = ssl.networks.HouseholdNet(dhs_data=dhs_data_sex, dynamic=False)
    sim2 = ss.Sim(n_agents=small, diseases='sis', networks=household2)
    sim2.run()

    return sim


@sc.timer()
def test_dynamic_household():
    sc.heading('Testing dynamic HouseholdNet...')
    dhs_data = make_dhs_data(n=medium)

    household = ssl.networks.HouseholdNet(dhs_data=dhs_data, dynamic=True)
    preg = ss.Pregnancy()
    sim = ss.Sim(n_agents=medium, diseases='sis', networks=household, demographics=preg, copy_inputs=False)
    sim.run()

    # All delivered agents should have household IDs; undelivered embryos should not.
    # Agents born in the sim with age >= dt have been through at least one full step
    # after delivery, so add_births() must have assigned them a household ID.
    # Agents with age < dt at the end may be undelivered embryos whose age crossed
    # zero only due to the end-of-step aging in finish_step.
    ppl = sim.people
    hids = household.household_ids[:]
    born_in_sim = ppl.parent.notnan
    delivered = ~born_in_sim | (ppl.age >= sim.t.dt_year)
    assert np.all(np.isfinite(hids[delivered.uids])), 'Some delivered agents are missing household IDs'

    return sim


@sc.timer()
def test_household_coarse_pregnancy_dt():
    """ Regression test: a Pregnancy module running on a coarser dt than the sim must not corrupt
    HouseholdNet move-out timing.

    Move-out is evaluated once per pregnancy, so the resulting number of households should be
    ~invariant to the pregnancy module's dt. Previously, HouseholdNet compared
    pregnancy.ti_delivery (an index in the pregnancy module's own timeline) against sim.ti; when
    pregnancy ran on a coarser dt (e.g. ss.Pregnancy(dt=ss.months(3))), pregnant non-heads became
    re-eligible far too often, greatly over-fragmenting households. """
    sc.heading('Testing HouseholdNet with a coarse-dt Pregnancy module...')
    dhs_data = make_dhs_data(n=medium)

    def run(preg_dt):
        household = ssl.networks.HouseholdNet(dhs_data=dhs_data, dynamic=True, prob_move_out=0.5)
        preg = ss.Pregnancy(dt=preg_dt)
        sim = ss.Sim(n_agents=medium, diseases='sis', networks=household, demographics=preg,
                     start='2000-01-01', stop='2020-12-31', dt=ss.months(1), rand_seed=1, copy_inputs=False)
        sim.run()
        return household.n_households

    n_month   = run(ss.months(1))
    n_quarter = run(ss.months(3))
    rel = abs(n_quarter - n_month) / n_month
    assert rel < 0.15, f'Coarse-dt pregnancy over-fragments households: monthly={n_month}, quarterly={n_quarter} (rel diff {rel:.1%})'

    return n_month, n_quarter


@sc.timer()
def test_other():
    sc.heading('Other network tests...')

    print('Testing MSM network')
    msm = ss.MSMNet(participation=0.3)
    sim = ss.Sim(diseases=dict(type='sis', beta=0.5), networks=msm, copy_inputs=False)
    sim.run()

    print('Testing MF network relative participation')
    mf = ss.Sim(n_agents=1000, networks=ss.MFNet(rel_part_rates=0.5), verbose=0).init().networks.mfnet
    assert 0.35 < mf.participant.mean() < 0.55 # Default participation is 0.9, so expect ~0.45

    print('Testing other network methods')
    msm.validate()
    inds1 = msm.get_inds([0])
    contacts = msm.find_contacts(inds1['p1'])
    assert contacts[0] == inds1['p2']
    inds2 = msm.pop_inds([0])
    assert inds1 == inds2

    return msm


@sc.timer()
def test_edge_ops():
    sc.heading('Testing network edge operations')
    df = sc.dataframe(p1=[0, 1, 2, 3], p2=[1, 2, 3, 4])
    net = ss.Network(name='mynet').from_df(df) # Create from a dataframe
    assert net.name == 'mynet' and len(net) == 4
    assert np.array_equal(net['p1'], df.p1) # Access edge columns like a dict

    # Find contacts (both directions, including the agents themselves if they're each other's contacts), as an array or a set
    net2 = ss.Network(p1=[1,2,3,4], p2=[2,3,1,4])
    assert np.array_equal(net2.find_contacts([1,3]), [1,2,3])
    assert ss.find_contacts(net2.p1, net2.p2, [1,3]) == {1,2,3}
    assert net.beta.sum() == 4 # Beta defaults to 1

    net.from_df(dict(p1=[0, 1], p2=[1, 2], beta=[1, 2])) # Replace the edges with a dict of arrays
    assert net['beta'][1] == 2

    net.remove_edges(net.p1 == 0) # Remove edges by mask
    assert len(net) == 1 and net.p1[0] == 1
    net.remove_edges([0]) # ...or by index
    assert len(net) == 0
    ss.RandomNet().validate() # A network with no edges yet is valid
    return net


@sc.timer()
def test_dynamic():
    sc.heading('Testing static (dynamic=False) networks')
    kw = dict(n_agents=small, diseases='sis', verbose=0)
    sim = ss.Sim(networks=ss.RandomNet(dynamic=False), **kw).init()
    p1 = sim.networks[0].p1.copy()
    sim.run()
    assert np.array_equal(p1, sim.networks[0].p1) # Static edges don't change

    sim = ss.Sim(networks=ss.RandomNet(dynamic=False), demographics=True, stop=2010, **kw).run()
    assert sim.networks[0].members.max() < small # Agents born during the sim are not added
    return sim


@sc.timer()
def test_cluster():
    sc.heading('Testing cluster networks')
    net = ss.ClusterNet(cluster_size=ss.poisson(5), age_range=[18, 65])
    sim = ss.Sim(n_agents=medium, networks=net, copy_inputs=False).init()
    ages = sim.people.age
    clusters = net.cluster[net.p1]
    assert np.array_equal(clusters, net.cluster[net.p2]) # Edges are only within clusters
    assert np.all(ages[net.members] >= 18) and np.all(ages[net.members] < 65) # Only within the age range
    sizes = np.bincount(net.cluster.notnanvals.astype(int))
    assert len(net) == (sizes*(sizes-1)/2).sum() # Clusters are fully connected
    return net


@sc.timer()
def test_hybrid():
    sc.heading('Testing hybrid networks')
    sis = ss.SIS(beta=dict(h=0.1, s=0.05, w=0.05, c=0.02)) # Per-network beta
    sim = ss.Sim(n_agents=medium, diseases=sis, networks=ss.HybridNet(contacts=dict(c=10)), verbose=0)
    sim.run()
    nets = sim.networks
    assert list(nets.keys()) == ['h', 's', 'w', 'c'] # Expanded into four networks
    assert np.all(sim.people.age[nets.s.members] < 22) # Schools are age-restricted
    assert nets.h.edges.beta[0] == 3.0 # Default v3 per-network beta
    assert nets.c.pars.n_contacts.pars.lam == 10 # Partial update of contacts

    # As in Covasim v3, choose the targets of random edges uniformly
    sim2 = ss.Sim(n_agents=medium, diseases='sis', networks=ss.HybridNet(uniform_targets=True), verbose=0).init()
    assert all(sim2.networks[k].pars.uniform_targets for k in ['s', 'w', 'c'])
    return sim


@sc.timer()
def test_partners():
    sc.heading('Testing partnerships with a target number of partners at a time')
    net = ss.PartnerNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50) # EpiModel's introductory network
    S, I = ss.states(2)
    sis = ss.Disease(S >> I @ ss.prob(0.64), I >> S @ ss.prob(0.05), init=dict(I=10), name='sis')
    sim = ss.Sim(sis, networks=net, n_agents=medium, stop=400, verbose=0).run()
    net = sim.networks[0]
    res = net.results
    k = np.arange(4)
    assert np.isclose(net.dist_degree @ k, 0.7) and np.isclose(net.dist_degree[2:].sum(), 0.22) # The targets
    assert np.isclose(res.mean_degree[200:].mean(), 0.7, rtol=0.05) # ... are what the network has
    assert np.isclose(res.concurrency[200:].mean(), 0.22, rtol=0.1)
    assert net.degrees(sim.people.alive.uids).max() <= 3
    assert sim.results.sis.n_I[-1] > 100 # Infection spreads through the partnerships
    table = net.check(n_runs=2, output=True)
    assert list(table.index) == ['mean_degree', 'concurrency', 'max_contacts']

    # Targets that cannot be met together are an error, and a compartmental run uses the mean number of partners
    with pytest.raises(ValueError):
        ss.PartnerNet.degree_dist(0.7, concurrency=0.9, max_contacts=3)
    ode = ss.Sim(sis, networks=ss.PartnerNet(n_contacts=0.7, dur=50), total_pop=medium, solver='ode', stop=100, verbose=0).run()
    assert ode.results.sis.n_I[-1] > 10

    # With births and deaths, partnerships also end when a partner dies, which is allowed for so the mean holds
    net = ss.PartnerNet(n_contacts=0.5, dur=50, name='main')
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.0), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    demog = [ss.Births(birth_rate=ss.peryear(50)), ss.Deaths(death_rate=ss.peryear(50))]
    vital = ss.Sim(sir, networks=net, demographics=demog, n_agents=2000, dt=ss.weeks(1), start=2000, stop=2030, verbose=0).run()
    assert abs(vital.networks.main.results.mean_degree[-500:].mean() - 0.5) < 0.03

    # Several networks of the same kind take their names from their variables, or from the keys of a dict
    main, casual = ss.PartnerNet(n_contacts=0.5, dur=50), ss.PartnerNet(n_contacts=0.3, dur=5)
    two = ss.Disease(infection=ss.transmission('S -> I', beta=dict(main=0.1, casual=0.05)), recovery=ss.progression('I -> R', dur=10), init_prev=0.05, name='two')
    assert list(ss.Sim(two, networks=[main, casual], n_agents=500, stop=5, verbose=0).init().networks.keys()) == ['main', 'casual']
    keyed = ss.Sim(two, networks=dict(main=ss.PartnerNet(dur=50), casual=ss.PartnerNet(dur=5)), n_agents=500, stop=5, verbose=0).init()
    assert list(keyed.networks.keys()) == ['main', 'casual']
    return sim


def test_partner_targets():
    sc.heading('Testing other targets: partnerships within a level, and targets on ss.MFNet')

    # Most partnerships between people at the same level of risk
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.0), name='sir')
    sir.stratify(risk=['low', 'high'], split=[0.8, 0.2])
    for share in [None, 0.9]:
        net = ss.PartnerNet(n_contacts=0.7, dur=50, match=None if share is None else dict(risk=share), name='net')
        sim = ss.Sim(sir, networks=net, n_agents=5000, stop=300, verbose=0).run()
        net = sim.networks.net
        level = sim.level_raw(sim.dim_index('risk'))
        within = np.mean(level[net.edges.p1] == level[net.edges.p2])
        assert np.isclose(net.results.mean_degree[150:].mean(), 0.7, rtol=0.05)
        assert np.isclose(within, 0.68, atol=0.03) if share is None else np.isclose(within, share, atol=0.02) # By chance, 0.8² + 0.2²
    assert 'match' in net.check(n_runs=2, output=True).index

    # A target number of partners at a time between men and women, with concurrency
    duration = ss.lognorm_ex(mean=ss.years(2), std=ss.years(1))
    mf = ss.MFNet(n_contacts=ss.poisson(0.7), concurrency=0.1, duration=duration, name='mf')
    sim = ss.Sim(diseases=ss.SIS(), networks=mf, n_agents=5000, start=2000, stop=2030, dt=ss.months(1), verbose=0).run()
    res = sim.networks.mf.results
    assert np.isclose(res.mean_degree[60:].mean(), 0.7, rtol=0.03) and np.isclose(res.concurrency[60:].mean(), 0.1, rtol=0.1)
    with pytest.raises(ValueError, match='n_contacts'):
        ss.Sim(diseases=ss.SIS(), networks=ss.MFNet(concurrency=0.1), n_agents=500, verbose=0).init()
    return sim


@sc.timer()
def test_bare_units():
    sc.heading('Testing the warning for a bare number in another unit than the sim')
    sim = lambda net: ss.Sim(n_agents=200, diseases=ss.SIS(beta=ss.perday(0.05), dur_inf=ss.days(10)), networks=net, dt=ss.days(1), stop=ss.days(10), verbose=0)
    with pytest.warns(RuntimeWarning, match='read in years'): # The default duration is in years, but the sim is in days
        sim(ss.MFNet(duration=5)).init()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim(ss.MFNet(duration=ss.days(5))).init() # A unit says which is meant
        ss.Sim(n_agents=200, diseases='sis', networks=ss.MFNet(duration=5), verbose=0).init() # The same unit as the sim
    return


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Run tests
    man  = test_manual()
    rand = test_random()
    fast = test_randomfast()
    safe = test_randomsafe()
    strat = test_random_stratified()
    nuids = test_network_uids()
    stat = test_static()
    erdo = test_erdosrenyi()
    disk = test_disk()
    null = test_null()
    hh   = test_household()
    ehh  = test_dynamic_household()
    ehh2 = test_household_coarse_pregnancy_dt()
    oth  = test_other()
    ops  = test_edge_ops()
    dyn  = test_dynamic()
    clus = test_cluster()
    hyb  = test_hybrid()
    part = test_partners()
    targ = test_partner_targets()
    bare = test_bare_units()

    T.toc()