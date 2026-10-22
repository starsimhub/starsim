"""
Test networks
"""

# %% Imports and settings
import pytest
import warnings
import sciris as sc
import numpy as np
import networkx as nx
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
    nw1 = ss.Sim(n_agents=small, networks=ss.RandomNet()).init().networks[0] # The sim's copy, initialized

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
    sc.heading("Testing the (default, fast) RandomNet vs method='exact'")
    n_contacts = 10

    s1 = ss.Sim(n_agents=medium, networks=ss.RandomNet(n_contacts=n_contacts, method='exact'), rand_seed=1).init()
    s2 = ss.Sim(n_agents=medium, networks=ss.RandomNet(n_contacts=n_contacts), rand_seed=1).init()
    nw1 = s1.networks[0]
    nw2 = s2.networks[0]

    # Same number of edges as method='exact', and the mean degree matches n_contacts
    assert len(nw1) == len(nw2), "RandomNet should produce the same number of edges as method='exact'"
    deg = np.bincount(np.concatenate([np.asarray(nw2.p1), np.asarray(nw2.p2)]), minlength=medium)
    assert np.isclose(deg.mean(), n_contacts, rtol=0.05), f'Mean degree should be ~{n_contacts}, not {deg.mean()}'

    # The fast version has variable (Poisson-like) degree, unlike method='exact''s fixed degree
    deg1 = np.bincount(np.concatenate([np.asarray(nw1.p1), np.asarray(nw1.p2)]), minlength=medium)
    assert deg1.std() == 0, "method='exact' should have fixed degree"
    assert deg.std() > 1, 'RandomNet should have variable degree'

    # Reproducible given the same seed
    s3 = ss.Sim(n_agents=medium, networks=ss.RandomNet(n_contacts=n_contacts), rand_seed=1).init()
    assert np.array_equal(nw2.p2, s3.networks[0].p2), 'RandomNet should be reproducible for a fixed seed'

    # With uniform targets, the variance in degree is lower: about n_contacts/2 from the sources plus n_contacts/2 from the targets (as in Covasim v3)
    n = 10_000
    s4 = ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=ss.poisson(20), method='uniform'), rand_seed=1).init()
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
    births = ss.Births(birth_rate=ss.time.freq(5, 'year'))
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
    s3 = ss.Sim(pars, networks=ssl.RandomSafeNet(), label='safe-nobirths').run()
    s4 = ss.Sim(pars, networks=ssl.RandomSafeNet(), label='safe-births', demographics=births).run()

    n3 = s3.networks[0].to_edgelist()
    n4 = s4.networks[0].to_edgelist()
    simil.uid2 = auid_similarity(s3, s4)
    simil.safe = sc.similarity(n3, n4)
    print(f'Similarity in UIDs after {pars.dur} timesteps with/without births:\n{simil.uid2:%}')
    print(f'Similarity for random-safe networks after {pars.dur} timesteps:\n{simil.safe:%}\n')
    assert simil.safe > 0.5, 'RandomSafe networks were less similar than expected'

    # With many agents, some random numbers are tied; check that these edges don't all go to agent 0
    n = 20_000
    s5 = ss.Sim(n_agents=n, networks=ssl.RandomSafeNet(n_edges=10), rand_seed=1).init()
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

    exact_deg, exact_core = measure(ss.RandomNet(n_contacts=n_contacts, method='exact'))
    fast_deg, fast_core = measure(ss.RandomNet(n_contacts=n_contacts))
    print(f'Requested degrees:      {contacts}')
    print(f'Exact degrees: {exact_deg}, core edge fraction: {exact_core:n}')
    print(f'RandomNet degrees:      {fast_deg}, core edge fraction: {fast_core:n}')

    # The realized degree per risk group must match what was requested, not just on average
    assert np.allclose(fast_deg, contacts, rtol=0.1), f'RandomNet degrees {fast_deg} do not match requested {contacts}'
    assert np.allclose(fast_deg, exact_deg, rtol=0.1), f'RandomNet degrees {fast_deg} do not match the exact method {exact_deg}'

    # Most edges should involve the high-risk core on both ends, as with method='exact'
    assert np.isclose(fast_core, exact_core, rtol=0.1), f'RandomNet core clustering {fast_core:n} does not match the exact method {exact_core:n}'

    o = sc.objdict(exact_deg=exact_deg, fast_deg=fast_deg, exact_core=exact_core, fast_core=fast_core)
    return o


@sc.timer()
def test_network_uids():
    """ Test that all networks form edges between valid UIDs, not positional indices (issue #1418) """
    sc.heading('Testing that network edges reference valid UIDs')
    netclasses = [ss.RandomNet, lambda: ss.RandomNet(method='exact'), ssl.RandomSafeNet]
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
    nw1 = ss.Sim(n_agents=small, networks=ssl.networks.ErdosRenyiNet(p=p)).init().networks[0] # The sim's copy, initialized
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
    s1 = ss.Sim(n_agents=5, dur=ss.days(50), networks=ssl.networks.DiskNet()).init()
    nw1 = s1.networks[0]

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
    nw2 = ssl.networks.DiskNet(r=0.15, v=ss.time.freq(0.05, unit=ss.years(1)))
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
    nw1 = ss.Sim(n_agents=n, networks=ssl.StaticNet(p=p)).init().networks[0]

    # Create with n_contacts
    nw2 = ss.Sim(n_agents=n, networks=ssl.StaticNet(n_contacts=nc)).init().networks[0]

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
    sim = ss.Sim(n_agents=medium, diseases='sis', networks=household, demographics=preg)
    sim.run()
    household, preg = sim.networks[0], sim.demographics[0] # The sim's copies, which ran

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
    pregnancy.ti_exit_P (an index in the pregnancy module's own timeline) against sim.ti; when
    pregnancy ran on a coarser dt (e.g. ss.Pregnancy(dt=ss.months(3))), pregnant non-heads became
    re-eligible far too often, greatly over-fragmenting households. """
    sc.heading('Testing HouseholdNet with a coarse-dt Pregnancy module...')
    dhs_data = make_dhs_data(n=medium)

    def run(preg_dt):
        household = ssl.networks.HouseholdNet(dhs_data=dhs_data, dynamic=True, prob_move_out=0.5)
        preg = ss.Pregnancy(dt=preg_dt)
        sim = ss.Sim(n_agents=medium, diseases='sis', networks=household, demographics=preg,
                     start='2000-01-01', stop='2020-12-31', dt=ss.months(1), rand_seed=1)
        sim.run()
        return sim.networks[0].n_households

    n_month   = run(ss.months(1))
    n_quarter = run(ss.months(3))
    rel = abs(n_quarter - n_month) / n_month
    assert rel < 0.15, f'Coarse-dt pregnancy over-fragments households: monthly={n_month}, quarterly={n_quarter} (rel diff {rel:.1%})'

    return n_month, n_quarter


@sc.timer()
def test_other():
    sc.heading('Other network tests...')

    print('Testing a sexual network between men')
    msm = ss.SexualNet(between='mm', participation=0.3)
    sim = ss.Sim(diseases=dict(type='sis', beta=0.5), networks=msm)
    sim.run()
    msm = sim.networks[0] # The sim's copy, which ran
    assert len(msm) and sim.people.male.raw[msm.edges.p1].all() and sim.people.male.raw[msm.edges.p2].all()

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
    assert np.array_equal(net.edges.p1, df.p1)

    # Find contacts (both directions, including the agents themselves if they're each other's contacts), as an array or a set
    net2 = ss.Network(p1=[1,2,3,4], p2=[2,3,1,4])
    assert np.array_equal(net2.find_contacts([1,3]), [1,2,3])
    assert ss.utils.find_contacts(net2.p1, net2.p2, [1,3]) == {1,2,3}
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
    net = ssl.ClusterNet(cluster_size=ss.poisson(5), age_range=[18, 65])
    sim = ss.Sim(n_agents=medium, networks=net).init()
    net = sim.networks[0] # The sim's copy, initialized
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
    sim = ss.Sim(n_agents=medium, diseases=sis, networks=ssl.HybridNet(contacts=dict(c=10)), verbose=0)
    sim.run()
    nets = sim.networks
    assert list(nets.keys()) == ['h', 's', 'w', 'c'] # Expanded into four networks
    assert np.all(sim.people.age[nets.s.members] < 22) # Schools are age-restricted
    assert nets.h.edges.beta[0] == 3.0 # Default v3 per-network beta
    assert nets.c.pars.n_contacts.pars.lam == 10 # Partial update of contacts

    # As in Covasim v3, choose the targets of random edges uniformly
    sim2 = ss.Sim(n_agents=medium, diseases='sis', networks=ssl.HybridNet(method='uniform'), verbose=0).init()
    assert all(sim2.networks[k].pars.method == 'uniform' for k in ['s', 'w', 'c'])
    return sim


@sc.timer()
def test_partners():
    sc.heading('Testing lasting partnerships, and a target number of partners at a time')

    # Partnerships that form at a constant rate: a Poisson number of partners at a time, with the mean given
    S, I = ss.states(2)
    sis = ss.Disease(S >> I @ ss.prob(0.64), I >> S @ ss.prob(0.05), init_n=dict(I=10), name='sis')
    sim = ss.Sim(sis, networks=ssl.DegreeNet(n_contacts=0.7, dur=50), n_agents=2000, stop=300, verbose=0).run()
    e = sim.networks[0].edges
    degree = np.bincount(np.concatenate([e.p1, e.p2]), minlength=2000)
    assert np.isclose(degree.mean(), 0.7, rtol=0.1) and np.isclose(degree.var(), degree.mean(), rtol=0.15)
    assert sim.results.sis.n_I[-1] > 100 # Infection spreads through the partnerships

    # The targets of EpiModel's introductory network, from the library
    net = ssl.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)
    sim = ss.Sim(sis, networks=net, n_agents=medium, stop=400, verbose=0).run()
    net = sim.networks[0]
    res = net.results
    k = np.arange(4)
    assert np.isclose(net.dist_degree @ k, 0.7) and np.isclose(net.dist_degree[2:].sum(), 0.22) # The targets
    assert np.isclose(res.mean_degree[200:].mean(), 0.7, rtol=0.1) # ... are what the network has (partnerships last 50 steps, so 200 steps of 1,000 agents vary by about 5% between seeds)
    assert np.isclose(res.concurrency[200:].mean(), 0.22, rtol=0.1)
    assert net.degrees(sim.people.alive.uids).max() <= 3
    assert sim.results.sis.n_I[-1] > 100 # Infection spreads through the partnerships
    table = net.check(n_runs=2, output=True)
    assert list(table.index) == ['mean_degree', 'concurrency', 'max_contacts', 'dur']
    assert np.isclose(table.realized.dur, 50, rtol=0.15) # How long partnerships last, from how many there are and how many form

    # Acts in each partnership (acts=, per unit time): beta is then per act, so two acts a day at 0.4 transmit as one partnership at 1 - 0.6**2 a day
    def prev(acts, beta):
        S, I = ss.states(2)
        sis2 = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=I >> S @ ss.prob(0.05), init_n=dict(I=10), name='sis')
        return ss.Sim(sis2, networks=ssl.DegreeNet(n_contacts=0.7, dur=50, acts=acts), n_agents=medium, stop=200, rand_seed=1, verbose=0).run().results.sis.n_I.values
    assert np.allclose(prev(ss.freqperday(2), ss.peract(0.4)), prev(None, ss.prob(1 - 0.6**2)))
    with pytest.raises(ValueError, match='per unit time'): # A bare number would be acts in a step, which depends on the step
        prev(2, ss.peract(0.4))

    # A number of partners for each level of a property, as EpiModel's nodefactor
    byrisk = ssl.DegreeNet(n_contacts=ss.by(risk=[0.4, 1.6]), dur=50)
    sim2 = ss.Sim(sis, networks=byrisk, n_agents=medium, stop=200, verbose=0).stratify(risk=['low', 'high'], split=[0.8, 0.2]).run()
    level = np.asarray(sim2.networks[0].levels('risk'))[sim2.people.alive.uids]
    degree = sim2.networks[0].degrees(sim2.people.alive.uids)
    assert np.isclose(degree[level == 0].mean(), 0.4, rtol=0.25) and np.isclose(degree[level == 1].mean(), 1.6, rtol=0.25)

    # Targets that cannot be met together are an error; partnerships are approximate under an ODE (see test_partnership_approx()), and a static network is approximated by the mean degree of its graph
    with pytest.raises(ValueError):
        ssl.DegreeNet.degree_dist(0.7, concurrency=0.9, max_contacts=3)
    with pytest.raises(ss.CapabilityError, match='partner-change model'):
        ss.Sim(sis, networks=ssl.DegreeNet(n_contacts=0.7, dur=50), total_pop=medium, solver='ode', stop=100, approx=False, verbose=0).run()
    ring = ss.Sim(sis, networks=ssl.StaticNet(graph=nx.cycle_graph(1000)), total_pop=1000, solver='ode', stop=10, verbose=0).init()
    assert ring.mixing.terms[0].contacts == 2 # Not n_contacts=10, whatever the graph

    # With births and deaths, partnerships also end when a partner dies, which is allowed for so the mean holds
    net = ssl.DegreeNet(n_contacts=0.5, dur=50, name='main')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    demog = [ss.Births(birth_rate=ss.peryear(50)), ss.Deaths(death_rate=ss.peryear(50))]
    vital = ss.Sim(sir, networks=net, demographics=demog, n_agents=2000, dt=ss.weeks(1), start=2000, stop=2030, verbose=0).run()
    assert abs(vital.networks.main.results.mean_degree[-500:].mean() - 0.5) < 0.03

    # Several networks of the same kind take their names from the keys of a dict
    S, I, R = ss.states('S I R')
    two = ss.Disease(infection=ss.Transition(S, I, rate=dict(main=0.1, casual=0.05) * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.05, name='two')
    keyed = ss.Sim(two, networks=dict(main=ss.SexualNet(dur=50), casual=ss.SexualNet(dur=5)), n_agents=500, stop=5, verbose=0).init()
    assert list(keyed.networks.keys()) == ['main', 'casual']
    return sim


def test_partner_targets():
    sc.heading('Testing other targets: partnerships within a level, and targets between men and women')

    # Most partnerships between people at the same level of risk
    S, I = ss.states('S I')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), name='sir')
    sir.stratify(risk=['low', 'high'], split=[0.8, 0.2])
    for share in [None, 0.9]:
        net = ssl.DegreeNet(n_contacts=0.7, dur=50, match=None if share is None else dict(risk=share), name='net')
        sim = ss.Sim(sir, networks=net, n_agents=5000, stop=300, verbose=0).run()
        net = sim.networks.net
        level = sim.groups.level_raw(sim.groups.dim_index('risk'))
        within = np.mean(level[net.edges.p1] == level[net.edges.p2])
        assert np.isclose(net.results.mean_degree[150:].mean(), 0.7, rtol=0.05)
        assert np.isclose(within, 0.68, atol=0.03) if share is None else np.isclose(within, share, atol=0.02) # By chance, 0.8² + 0.2²
    assert 'match' in net.check(n_runs=2, output=True).index

    # Within the levels of several properties at once, each with its own target (EpiModel's nodematch on each)
    sir.stratify(region=['a', 'b', 'c'], split=[0.5, 0.3, 0.2])
    net = ssl.DegreeNet(n_contacts=0.7, dur=50, match=dict(risk=0.9, region=0.7), name='net')
    res = ss.Sim(sir, networks=net, n_agents=5000, stop=300, verbose=0).run().networks.net.results
    assert np.isclose(res.match_risk[150:].mean(), 0.9, atol=0.02) and np.isclose(res.match_region[150:].mean(), 0.7, atol=0.02) # By chance, 0.68 and 0.38

    # A target number of partners at a time between men and women, with concurrency
    duration = ss.lognorm_ex(mean=ss.years(2), std=ss.years(1))
    mf = ss.SexualNet(n_contacts=ss.poisson(0.7), concurrency=0.1, dur=duration, name='mf')
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=mf, n_agents=5000, start=2000, stop=2030, dt=ss.months(1), verbose=0).run()
    res = sim.networks.mf.results
    assert np.isclose(res.mean_degree[60:].mean(), 0.7, rtol=0.03) and np.isclose(res.concurrency[60:].mean(), 0.1, rtol=0.1)
    with pytest.raises(ValueError, match='n_contacts'):
        ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=ss.SexualNet(concurrency=0.1), n_agents=500, verbose=0).init()

    # Exactly one partner each: people with none always look (before, nobody did, and almost nobody had a partner)
    mf = ss.SexualNet(n_contacts=ss.constant(1), participation=1.0, debut=ss.constant(0), dur=ss.months(6), name='mf')
    sim = ss.Sim(networks=mf, n_agents=2000, start=2000, stop=2005, dt=ss.months(1), verbose=0).run()
    assert sim.networks.mf.results.mean_degree[-1] > 0.95

    # In an open population with long partnerships, newcomers join and the target holds (HP9 to HP11: before, 0.65 against 0.87, and nobody born in the run took part)
    mf = ss.SexualNet(n_contacts=0.87, dur=ss.lognorm_ex(mean=ss.years(20), std=ss.years(5)), participation=ss.bernoulli(p=lambda self, sim, uids: np.where(sim.people.age[uids] < 10, 0, 0.9)), name='mf')
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=mf, demographics=[ss.Births(birth_rate=35), ss.Deaths(death_rate=15)], n_agents=5000, start=2000, stop=2040, verbose=0).run()
    ppl, mf = sim.people, sim.networks.mf
    born = (ppl.alive & (ppl.age >= 18) & (ppl.age < 40)).uids # All born during the run, when the probability at birth was 0
    assert np.isclose(mf.participant[born].mean(), 0.9, atol=0.04)
    assert np.isclose(mf.results.mean_degree[-10:].mean(), 0.87, rtol=0.05)
    return sim


@sc.timer()
def test_bare_units():
    sc.heading('Testing the warning for a bare number in another unit than the sim')
    sim = lambda net: ss.Sim(n_agents=200, diseases=ss.SIS(beta=ss.peract(0.05), dur_inf=ss.days(10)), networks=net, dt=ss.days(1), stop=ss.days(10), verbose=0)
    with pytest.warns(RuntimeWarning, match='read in years'): # The default duration is in years, but the sim is in days
        sim(ss.SexualNet(dur=5)).init()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim(ss.SexualNet(dur=ss.days(5))).init() # A unit says which is meant
        ss.Sim(n_agents=200, diseases=ss.SIS(beta=ss.peract(0.01)), networks=ss.SexualNet(dur=5), verbose=0).init() # The same unit as the sim
    return


def chance(beta, acts=ss.constant(ss.freqperyear(1)), dt=ss.years(1), steps=1, connectors=None, net=None, n=40_000, f2m=False, seed=1):
    """ The chance that a partnership with an infectious partner transmits over some steps, measured over many lasting partnerships (from men to women, or with f2m=True, women to men) """
    S, I = ss.states('S I')
    si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.5, name='si')
    net = net or ss.SexualNet
    net = net(dur=ss.constant(ss.years(100)), participation=ss.bernoulli(p=1.0), debut=ss.constant(0), acts=acts)
    sim = ss.Sim(si, networks=net, connectors=connectors, n_agents=n, start=2000, stop=2000 + steps*dt.years, dt=dt, rand_seed=seed, verbose=0).init()
    e = sim.networks[0].edges
    female = sim.people.female.raw
    men, women = np.where(female[e.p1], e.p2, e.p1), np.where(female[e.p1], e.p1, e.p2)
    src, trg = (women, men) if f2m else (men, women)
    before = sim.diseases.si.I.raw.copy()
    sim.run()
    at_risk = before[src] & ~before[trg]
    return sim.diseases.si.I.raw[trg[at_risk]].mean()


@sc.timer()
def test_per_act():
    sc.heading('Testing a probability per act on a network with acts, which keeps its meaning at any step')

    # A partnership with 80 acts a year transmits with 1 - 0.99**80 = 0.552 a year at per-act 0.01, whatever the step (ST2: 0.531 monthly and 0.465 weekly with whole acts per step, which a rate no longer gives)
    for dt in [ss.years(1/12), ss.weeks(1)]:
        p = chance(ss.peract(m2f=0.01, f2m=0), acts=ss.constant(ss.freqperyear(80)), dt=dt, steps=round(1/dt.years))
        assert np.isclose(p, 1 - 0.99**80, atol=0.025), f'{dt}: {p:.3f}'

    # And an epidemic over 10 years is the same at monthly and weekly steps (it was 38% against 12% with ss.probpermonth(0.01), and 37% against 29% with -log(1 - p) as a rate per month)
    def prev(dt, beta=ss.peract(0.01)):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.05, name='si')
        sims = [ss.Sim(si, networks=ss.SexualNet(dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1))), n_agents=5000, start=2000, stop=2010, dt=dt, rand_seed=seed, verbose=0) for seed in [1, 2]]
        return np.mean([sim.run().results.si.n_I[-1]/5000 for sim in sims])
    monthly, weekly = prev(ss.years(1/12)), prev(ss.weeks(1))
    assert np.isclose(monthly, weekly, atol=0.03), f'{monthly:.3f} against {weekly:.3f}'
    assert prev(ss.years(1/12), 0.01) == monthly # A bare number on a network with acts is per act too (as in Starsim 3)

    # Each direction has its own probability per act (ST3): from women to men at 0.15 per act, 2, 4, and 8 acts in a step give Starsim 3's 0.278, 0.478, and 0.728 (v4 had 0.232, 0.346, and 0.429)
    for k,target in zip([2, 4, 8], [0.278, 0.478, 0.728]):
        p = chance(ss.peract(m2f=0.3, f2m=0.15), acts=ss.constant(ss.freqperyear(k)), f2m=True)
        assert np.isclose(p, target, atol=0.02), f'{k} acts: {p:.3f} against {target}'

    # The directions are named, so the order of a network's partners does not matter (HP3: [b, 0] infected only women on ss.SexualNet, and only men on HPVsim's network, which puts women in p1)
    class WomenFirst(ss.SexualNet):
        def append(self, edges=None, **kwargs):
            kwargs['p1'], kwargs['p2'] = kwargs['p2'], kwargs['p1']
            return super().append(edges, **kwargs)
    for net in [ss.SexualNet, WomenFirst]:
        assert chance(ss.peract(m2f=0.5, f2m=0), net=net, n=4000) > 0.3
        assert chance(ss.peract(m2f=0.5, f2m=0), net=net, n=4000, f2m=True) == 0
    assert ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=WomenFirst(), n_agents=200, verbose=0).init().networks[0].edges.p1[:10].tolist() # Women in p1, and it runs

    # rel_trans, rel_sus, and modifiers multiply the hazard per act (D5): beta 0.03 with rel_trans 10 is 1 - 0.97**10 = 0.263 per act
    tenfold = lambda b: [ss.modify('si.rel_trans', multiply=10)] if b is None else [ss.modify('si.rel_trans', multiply=2), ss.modify('si.pars.beta', multiply=5)]
    p10 = chance(ss.peract(m2f=0.03, f2m=0), connectors=tenfold(None))
    assert np.isclose(p10, 1 - 0.97**10, atol=0.015), p10
    assert p10 == chance(ss.peract(m2f=0.03, f2m=0), connectors=tenfold(2)) # rel_trans 2 and beta times 5 is the same as rel_trans 10

    # A network written as code with Starsim 3's net_beta() keeps Starsim 3's numbers: the chance for the partnership in the step, which rel_sus then scales (0.44 at p=0.3, 6 acts, rel_sus=0.5, against 0.66 per act)
    class V3Net(ss.SexualNet):
        def net_beta(self, disease_beta=None, inds=None, disease=None): # As Starsim 3 calls it: for every edge (STIsim's takes uids= instead of inds=)
            return self.edges.beta*(1 - (1 - disease_beta)**self.edges.acts)
    half = lambda: [ss.modify('si.rel_sus', multiply=0.5)]
    core = chance(ss.peract(0.3), acts=ss.constant(ss.freqperyear(6)), connectors=half())
    coded = chance(ss.peract(0.3), acts=ss.constant(ss.freqperyear(6)), connectors=half(), net=V3Net)
    assert np.isclose(core, 1 - 0.7**3, atol=0.015) and np.isclose(coded, 0.5*(1 - 0.7**6), atol=0.015), (core, coded)
    return


class Couples(ss.SexualNet):
    """ Every man and woman in a couple, for life, with a fixed number of acts per step (acts=) """
    def __init__(self, acts=2, **kwargs):
        super().__init__(**kwargs)
        self.n_acts = acts
    def add_pairs(self):
        self.participant[self.sim.people.auids] = True
        m, f = [self.available(self.sim.people, sex) for sex in ['male', 'female']]
        n = min(len(m), len(f))
        self.append(p1=m[:n], p2=f[:n], beta=np.ones(n), dur=np.full(n, 10.0), acts=np.full(n, float(self.n_acts)))
        return


@sc.timer()
def test_beta_readings():
    sc.heading('Testing how a beta is read on networks with and without acts, and their directions')
    kw = dict(n_agents=500, start=2000, stop=2002, verbose=0)
    def sis(beta):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.1, name='sis')
    mf = ss.SexualNet()
    assert ss.SexualNet().directions == ['m2f', 'f2m'] and ss.PrenatalNet().directions == ['mother2child']
    assert ss.RandomNet().directions == []
    assert Couples().directions == ['m2f', 'f2m'] and ss.SexualNet(between='mm').directions == [] # A sexual network of its own is between men and women unless it says otherwise; men only: one beta
    sim = ss.Sim(sis(ss.peract(m2f=0.2, f2m=0.1)), networks=Couples(name='couples', acts=2), **kw).run()
    assert sim.results.sis.infection.sum() > 0
    with pytest.raises(ss.errors.RouteMismatch, match="between='mf'"):
        ss.Sim(sis(ss.peract(m2f=0.01, f2m=0.01)), networks=ss.SexualNet(between='mm'), **kw).init()
    assert ss.peract({('m', 'f'): 0.3, ('f', 'm'): 0.15}) == ss.peract(m2f=0.3, f2m=0.15) # A direction as a pair of sides, or by name
    with pytest.raises(ss.errors.WrongDimension, match=r'ss.peract\(0.01\)'): # A rate per unit time times acts per unit time is not a hazard (D2)
        ss.Sim(sis(ss.permonth(0.01)), networks=ss.SexualNet(), **kw).init()
    sim = ss.Sim(diseases='sis', networks=['sexual', 'random'], n_agents=500, verbose=0).init() # But the default of ss.SIS is a probability per act there, as in Starsim 3
    assert sim.diseases.sis.pars.beta.raw['sexualnet'] == ss.peract(0.05) and any('as in Starsim 3' in note for note in sim.notes)
    with pytest.raises(ss.errors.RouteMismatch, match=r'ss.peract\(m2f=0.3, f2m=0.15\)'): # A pair has no direction names (D3, N7)
        ss.Sim(sis(dict(sexualnet=[0.3, 0.15])), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ss.errors.RouteMismatch, match='no acts'): # No act to be per
        ss.Sim(sis(ss.peract(0.01)), networks='random', **kw).init()
    with pytest.raises(ss.errors.RouteMismatch, match='f2m'): # Every direction needs a value ...
        ss.Sim(sis(ss.peract(m2f=0.01)), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ss.errors.RouteMismatch, match='m2f and f2m'): # ... and one it does not have raises, listing those it has
        ss.Sim(sis(ss.peract(m2f=0.01, f2m=0.01, m2m=0.02)), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ss.errors.RouteMismatch, match='no sides'):
        ss.Sim(sis(ss.peract(m2f=0.01, f2m=0.01)), networks=ssl.DegreeNet(name='needles', dur=ss.years(1), acts=ss.freqpermonth(20)), **kw).init()
    with pytest.raises(ss.errors.OutOfBounds):
        ss.peract(1.5)
    with pytest.raises(ValueError, match='step'): # A whole number of acts per step, drawn once, depends on the step (D4)
        ss.Sim(sis(0.01), networks=ss.SexualNet(acts=ss.poisson(lam=ss.freqperyear(80))), **kw).init()
    with pytest.raises(ValueError, match='frequency'): # A hazard is not a number of acts (as agents, it would be the chance of an act in a step)
        ss.Sim(sis(0.01), networks=ss.SexualNet(acts=ss.peryear(80)), **kw).init()

    # Mother-to-child networks transmit one way, so one value is enough (a pair was [b, 0])
    S, I = ss.states('S I')
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=ss.peract(m2f=0.01, f2m=0.005), maternalnet=ss.peryear(0.5)) * I), init_prev=0.1, name='hiv')
    sim = ss.Sim(hiv, networks=[ss.SexualNet(), ss.MaternalNet()], demographics=ss.Pregnancy(), **kw).run()
    assert sim.results.hiv.n_I[-1] > 0
    return


@sc.timer()
def test_route_modifiers():
    sc.heading('Testing modifiers of a transmission on one route: condoms, needle programs')

    # A condom of efficacy 0.9 in half the acts multiplies the hazard by 1 - 0.5*0.9 (D6): p=0.1, 10 acts gives 1 - exp(-10*0.1054*0.55) = 0.440 (STIsim's split of the acts gives 0.438)
    condoms = lambda **kw: [ss.modify('si.infection', route='sexualnet', **kw)]
    p = chance(ss.peract(m2f=0.1, f2m=0), acts=ss.constant(ss.freqperyear(10)), connectors=condoms(efficacy=0.9, share=0.5), n=100_000)
    assert np.isclose(p, 1 - np.exp(10*np.log(0.9)*0.55), atol=0.01), p
    assert p == chance(ss.peract(m2f=0.1, f2m=0), acts=ss.constant(ss.freqperyear(10)), connectors=condoms(multiply=0.1, share=0.5), n=100_000) # efficacy= is multiply=1 - efficacy
    assert chance(ss.peract(m2f=0.1, f2m=0), connectors=condoms(multiply=0, share=1)) == 0 # Nothing on that route
    with pytest.raises(ValueError, match='efficacy'):
        ss.modify('si.infection', efficacy=0.9, multiply=0.1)

    # ... and leaves the other routes alone
    def run(mods=None, beta=ss.peryear(0.5)):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=ss.peract(0.05), randomnet=beta) * I), init_prev=0.05, name='si')
        return ss.Sim(si, networks=[ss.SexualNet(), ss.RandomNet(n_contacts=2)], connectors=mods, n_agents=2000, start=2000, stop=2005, verbose=0).run().results.si.n_I[-1]
    assert run(condoms(multiply=0, share=1)) > run(condoms(multiply=0, share=1), beta=0) > 0 # Infections only over the random network
    with pytest.raises(ss.errors.RouteMismatch, match='not a route'):
        run(condoms(multiply=0, share=1) + [ss.modify('si.infection', multiply=0.5, route='needles')])
    with pytest.raises(ValueError, match='transmission'):
        run([ss.modify('si.rel_sus', multiply=0.5, route='sexualnet')])

    # Under an ODE, over a mixing pool, it is a multiplier on that route's hazard, by group with a share by group, as agents
    def pool(solver, mods, seed=1):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=dict(pool=ss.perday(0.03), home=ss.perday(0.02)) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
        nets = [ss.MixingPool(n_contacts=4, name='pool'), ss.MixingPool(n_contacts=2, name='home')]
        sim = ss.Sim(sir, networks=nets, interventions=mods, stratify=dict(risk=['low', 'high'], split=[0.7, 0.3]), total_pop=10_000, n_agents=10_000, solver=solver, unit='day', stop=100, rand_seed=seed, verbose=0)
        return sim.run().results.sir.n_R.values
    share = lambda: [ss.modify('sir.infection', multiply=0.2, share=ss.by(risk=[0.3, 0.9]), route='pool')]
    ode = pool('ode', share())
    assert np.allclose(ode, pool('ode', [ss.modify('sir.infection', multiply=ss.by(risk=[1 - 0.3*0.8, 1 - 0.9*0.8]), route='pool')])) # The same as its multiplier, 1 - share*(1 - multiply)
    ratio = lambda solver, seeds: np.mean([pool(solver, share(), seed)[-1] for seed in seeds])/np.mean([pool(solver, None, seed)[-1] for seed in seeds])
    ode_ratio, abm_ratio = ratio('ode', [1]), ratio('abm', range(12)) # 12 seeds, since near its threshold one run's epidemic varies a lot (seeds 0-3 alone gave 0.22 here); each solver's own epidemic (agents have about 8% fewer infections here either way), cut by the same share: 0.26
    assert ode_ratio < 0.5 and np.isclose(abm_ratio, ode_ratio, rtol=0.1), (abm_ratio, ode_ratio)
    return


@sc.timer()
def test_needles():
    sc.heading('Testing a needle-sharing network beside a sexual one (injecting drug use)')
    def run(mf_beta, needles_beta=ss.peract(0.006), nsp=False):
        needles = ssl.DegreeNet(name='needles', where=ss.ppl.pwid, n_contacts=2, dur=ss.months(6), acts=ss.freqpermonth(20)) # Among people who inject only; no sides, so one value
        S, I = ss.states('S I')
        hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=mf_beta, needles=needles_beta) * I), init_prev=0.05, name='hiv')
        intv = ss.modify(hiv.infection, multiply=0.2, share=0.5, route='needles', start=2005) if nsp else None # Needle and syringe programs: half of injections with a clean needle, 80% less risk
        sim = ss.Sim(hiv, networks=[ss.SexualNet(), needles], interventions=intv, n_agents=5000, start=2000, stop=2010, dt=ss.months(1), verbose=0)
        sim.stratify(pwid=[False, True], split=[0.9, 0.1]).init()
        sim.before = sim.diseases.hiv.I.raw.copy()
        return sim.run()
    sim = run(0)
    pwid = sim.people.pwid.raw.astype(bool)
    net = sim.networks.needles
    assert pwid[net.edges.p1].all() and pwid[net.edges.p2].all() and np.isclose(2*len(net)/pwid[sim.people.alive.uids].sum(), 2, rtol=0.1) # Only people who inject, with 2 partners each on average
    new = sim.diseases.hiv.I.raw & ~sim.before
    assert not np.any(new & ~pwid) and np.any(new & pwid) # Nobody outside them is infected with beta 0 on sexualnet
    low = ss.peract(0.0005)
    assert run(ss.peract(m2f=0.001, f2m=0.0005), low, nsp=True).results.hiv.infection.cum[-1] < run(ss.peract(m2f=0.001, f2m=0.0005), low).results.hiv.infection.cum[-1]
    return


@sc.timer()
def test_partnership_approx():
    sc.heading('Testing the partner-change model: lasting partnerships under the ODE, binomial, and Gillespie solvers')
    def sis(beta, dur=None):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), **(dict(recovery=ss.Transition(I, S, dur=dur)) if dur else {}), init_prev=0.05, name='d') # SI with no dur
    def run(solver, disease, net, dt=ss.months(1), stratify=None, seed=1, n=5000, **kw):
        sim = ss.Sim(disease, networks=net, n_agents=n, total_pop=n, start=2000, stop=2020, dt=dt, solver=solver, rand_seed=seed, verbose=0, **kw)
        return sim.stratify(**(stratify or dict(sex=['female', 'male'])))

    # With partnerships that end at a constant rate and no way out of the infectious state, the chance over a partnership is H/(H + 1/D̄)
    sim = run('ode', sis(ss.peract(0.05)), ssl.DegreeNet(name='pairs', n_contacts=1, dur=ss.years(2), acts=ss.freqperyear(10)), stratify=dict(risk=['low', 'high'])).init()
    H = 10*-np.log(0.95)
    assert sim.diseases.d.describe(output=True).count('partner-change model (approx)') == 1
    assert np.isclose(sim.diseases.d.engine.partnership_info[('infection', 'I', 'pairs', 0)][0], H/(H + 1/2)), 'H/(H + 1/D)'

    # Prevalence at 10 and 20 years under the ODE against agents (4 runs here, 1 for gono; 20 runs of 10,000 agents gave the ratios in the comments): it is close
    # for risk groups with short partnerships, too high for long monogamous partnerships with a long infection, and too low with concurrency
    mf = lambda **kw: ss.SexualNet(participation=ss.bernoulli(p=1.0), debut=ss.constant(0), **kw)
    cases = dict(
        risk = ([0.9, 1.1], [0.9, 1.1], dict(disease=sis(ss.peract(0.002)), stratify=dict(risk=['low', 'medium', 'high'], split=[0.8, 0.15, 0.05]), # 0.999, 0.995
                 net=ssl.DegreeNet(n_contacts=ss.by(risk=[0.5, 1.5, 4]), match=dict(risk=0.7), dur=ss.months(6), acts=ss.freqperyear(80)))),
        gono = ([0.08, 0.4], [0.06, 0.4], dict(disease=sis(ss.peract(m2f=0.05, f2m=0.03), ss.months(4)), dt=ss.weeks(1), # 0.19, 0.13: concurrency (Poisson partners), which the approximation leaves out
                 net=ss.SexualNet(participation=ss.bernoulli(p=1.0), debut=ss.constant(0), n_contacts=1, dur=ss.lognorm_ex(mean=ss.weeks(4), std=ss.weeks(2)), acts=ss.freqperweek(2)))),
        hiv  = ([1.0, 1.25], [1.1, 1.4], dict(disease=sis(ss.peract(m2f=0.004, f2m=0.002)), # 1.105, 1.247: infection kept within pairs, which the approximation spreads
                 net=mf(dur=ss.lognorm_ex(mean=ss.years(5), std=ss.years(3)), acts=ss.freqperyear(80)))),
    )
    for name,(at10,at20,kw) in cases.items():
        prev = lambda solver, seed=1: run(solver, seed=seed, **sc.dcp(kw)).run().results.d.n_I.values[[120, -1] if name != 'gono' else [522, -1]]/5000
        ratio = prev('ode')/np.mean([prev('abm', seed) for seed in range(1 if name == 'gono' else 4)], axis=0) # One run is enough for the weekly gono, whose ratios are far from its bounds (0.18-0.26 and 0.11-0.13 over 8 seeds)
        assert at10[0] < ratio[0] < at10[1] and at20[0] < ratio[1] < at20[1], f'{name}: {ratio}'

    # The other compartmental solvers use the same force of infection; approx=False raises, sides need a dimension, and 'auto' keeps agents
    hiv = cases['hiv'][2]
    ode = run('ode', **sc.dcp(hiv)).run().results.d.n_I[-1]
    for solver in ['binomial', 'gillespie']: # Stochastic, in 5,000 people
        assert np.isclose(run(solver, **sc.dcp(hiv)).run().results.d.n_I[-1], ode, rtol=0.1), solver
    with pytest.raises(ss.CapabilityError, match='abm'):
        run('ode', approx=False, **sc.dcp(hiv)).init()
    with pytest.raises(ss.CapabilityError, match=r'stratify\(sex='):
        ss.Sim(sis(0.01), networks=ss.SexualNet(), solver='ode', verbose=0).init()
    assert run('auto', **sc.dcp(hiv)).init().solver == 'abm'
    return


@sc.timer()
def test_peract_spec():
    sc.heading('Testing how a probability per act is described, written in the model spec, and written as code')
    def make(beta):
        S, P, EL = ss.states('S P EL')
        syph = ss.Disease(S, P, EL, infection=ss.Transition(S, P, rate=beta * (P + 0.55*EL)),
                          latent=ss.Transition(P, EL, dur=ss.years(1)), clear=ss.Transition(EL, S, dur=ss.years(2)), init=dict(P=0.05), name='syph')
        condoms = ss.modify(syph.infection, efficacy=0.5, share=ss.timeseries({2000: 0.05, 2020: 0.3}), route='sexualnet', name='condoms')
        return ss.Sim(syph, networks=ss.SexualNet(), interventions=condoms, n_agents=500, start=2000, stop=2002, verbose=0)
    sim = make(ss.peract(m2f=0.3, f2m=0.15)).init()
    text = sim.diseases.syph.describe(output=True)
    for line in ['source P + 0.55 EL', 'beta on sexualnet: per act, m2f 0.3, f2m 0.15', "sexualnet's sides m and f", '"condoms"']:
        assert line in text, line
    for beta, value in [(ss.peract(m2f=0.3, f2m=0.15), dict(type='peract', m2f=0.3, f2m=0.15)), (0.01, dict(type='peract', value=0.01))]:
        sim = make(beta)
        spec = sim.to_spec()['modules'][0]
        assert next(p['value'] for p in spec['pars'] if p['name'] == 'beta') == value
        assert ('transitions.infection.beta' in spec['guessed']) == sc.isnumber(beta) # A bare number on a network with acts is read as per act, which the spec says
        with pytest.warns(ss.errors.LossyExport): # The network is not written
            code = sim.to_code()
        assert "efficacy=0.5, share=ss.timeseries" in code and "route=['sexualnet']" in code
        ns = {}
        exec(code.replace('sim = ss.Sim(', 'sim = None and ss.Sim('), ns) # The disease, without the sim the code builds
        assert ns['syph'].pars.beta.raw == (beta if isinstance(beta, ss.peract) else ss.peract(beta))
    return


@sc.timer()
def test_beta_default():
    sc.heading('Testing a beta for each route with a default for the routes it does not name')
    kw = dict(n_agents=1000, stop=2, verbose=0)
    m2f = ss.par(0.06, name='m2f')
    def make(**betas):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=betas * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.1, name='d')

    # A library disease names routes the sim may not have: those are dropped with a note, and the routes it does not name take the default
    sim = ss.Sim(make(sexualnet=ss.peract(m2f=m2f, f2m=0.5*m2f), msm=ss.peract(0.1), default=0), networks=[ss.SexualNet(), ss.MaternalNet()], **kw).run()
    assert sim.results.d.n_I[-1] > 0
    assert any('msm' in note and 'default' in note for note in sim.notes), sim.notes
    assert 'beta on maternalnet' in sim.diseases.d.describe(output=True)

    # Without a default, a name that is not a route, or a route with no value, still raises, since it is likely a typo
    with pytest.raises(ss.errors.RouteMismatch, match='no route is called msm'):
        ss.Sim(make(sexualnet=ss.peract(0.01), maternalnet=0, msm=ss.peract(0.1)), networks=[ss.SexualNet(), ss.MaternalNet()], **kw).init()
    with pytest.raises(ss.errors.RouteMismatch, match='maternalnet has no value'):
        ss.Sim(make(sexualnet=ss.peract(0.01)), networks=[ss.SexualNet(), ss.MaternalNet()], **kw).init()

    # The default is the same as giving each route its value, as agents and as an ODE, which agree; it can be a typed rate or a parameter
    def pools(solver, **betas):
        nets = [ss.MixingPool(n_contacts=4, name='pool'), ss.MixingPool(n_contacts=2, name='home')]
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=betas * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
        sim = ss.Sim(sir, networks=nets, total_pop=10_000, n_agents=10_000, solver=solver, unit='day', stop=100, rand_seed=1, verbose=0)
        return sim.run().results.sir.n_R.values
    res = {}
    for solver in ['abm', 'ode']:
        res[solver] = pools(solver, pool=ss.perday(0.03), home=ss.perday(0.02))
        assert np.array_equal(res[solver], pools(solver, pool=ss.perday(0.03), school=ss.perday(0.1), default=ss.perday(0.02))), solver
        assert np.array_equal(res[solver], pools(solver, pool=ss.perday(0.03), default=ss.par(ss.perday(0.02), name='other'))), solver
    assert np.isclose(res['abm'][-1], res['ode'][-1], rtol=0.15), (res['abm'][-1], res['ode'][-1])

    # The default is kept in the model spec, and written as code
    sim = ss.Sim(make(sexualnet=ss.peract(0.01), msm=ss.peract(0.1), default=0), networks=[ss.SexualNet(), ss.MaternalNet()], **kw)
    beta = next(p['value'] for p in sim.to_spec()['modules'][0]['pars'] if p['name'] == 'beta')
    assert beta['default'] == 0 and 'msm' in beta # Bare, since it is per act on a route with acts
    with pytest.warns(ss.errors.LossyExport):
        assert "'default': 0.0" in sim.to_code()

    # A disease written as code reads the default too
    sim = ss.Sim(diseases=ssl.AgentSIS(beta=dict(random=0.05, msm=0.1, default=0)), networks=['random', ss.SexualNet()], **kw).run()
    assert sim.results.agentsis.prevalence[-1] > 0 and any('msm' in note for note in sim.notes)
    assert sim.diseases.agentsis.validate_beta()['sexual'] == [0, 0]
    return


def test_per_act_beta():
    """ A bare beta on a network with acts is a probability per act, so it does not depend on dt or the unit """
    sc.heading('Testing per-act beta on a sexual network')
    def run(dt, unit='year', beta=0.003):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.05, name='si')
        mf = ss.SexualNet(acts=ss.constant(ss.freqperyear(96)))
        sim = ss.Sim(si, networks=mf, n_agents=3000, start='2000-01-01', stop='2005-01-01', unit=unit, dt=dt, verbose=0, rand_seed=1).run()
        return sim.results.si.infection.sum()
    weekly, monthly, quarterly = [run(dt) for dt in [ss.weeks(1), ss.months(1), ss.months(3)]]
    assert monthly == run(ss.months(1), unit='month'), 'Infections should not depend on the unit'
    assert max(weekly, monthly, quarterly) < 1.3*min(weekly, monthly, quarterly), f'Infections should not depend on dt: {weekly}, {monthly}, {quarterly}'
    return


def test_condoms():
    """ A sexual network's edges carry the share of acts with a condom, which a modifier of the transmission on that route reads """
    sc.heading('Testing condoms on a sexual network')
    class CondomNet(ss.SexualNet):
        def add_pairs(self):
            super().add_pairs()
            self.edges.condoms[:] = self.pars.share
            return
    def make(share=0.0, eff=0.8):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=0.02 * I), init_prev=0.05, name='si')
        net = CondomNet(name='mf', acts=ss.constant(ss.freqperyear(50)))
        net.define_pars(share=share)
        condoms = ss.modify(si.infection, efficacy=eff, share=lambda net: net.edges.condoms, route='mf', name='condoms')
        return ss.Sim(si, networks=net, interventions=condoms, n_agents=2000, start=2000, stop=2005, verbose=0, rand_seed=1)
    sim = make(share=0.5).init()
    mf, si = sim.networks.mf, sim.diseases.si
    assert np.all(mf.edges.condoms == 0.5)
    assert si.value('beta') == si.engine.value(si.pars.beta), 'The current value of a declared parameter is a number'
    none, half, full = [make(share=s).run().results.si.infection.sum() for s in [0.0, 0.5, 1.0]]
    assert none > half > full, f'Condoms should reduce infections: {none}, {half}, {full}'
    assert make(share=1.0, eff=0.0).run().results.si.infection.sum() == none, 'With no efficacy, condoms do nothing'
    plain = ss.Sim(diseases=ss.SIS(beta=dict(mf=0.05)), networks=ss.SexualNet(name='mf'), n_agents=500, verbose=0).run() # append() without condoms
    assert len(plain.networks.mf.edges.condoms) == len(plain.networks.mf)
    return


def test_coded_declared_transmission():
    """ A disease written as code and a declared one, run as agents, infect the same people over the same contacts with the same seed """
    sc.heading('Testing that coded and declared diseases transmit the same way')
    class SI(ss.Infection):
        def __init__(self, beta, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(beta=beta)
        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            self.infected[uids] = True
            return

    def pairs(coded, beta, net, mods=None):
        S, I = ss.states('S I')
        si = SI(beta=beta, name='si') if coded else ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0, name='si')
        sim = ss.Sim(diseases=si, networks=net, connectors=mods, analyzers=ss.infection_log(), n_agents=2000, start=2000, stop=2003, verbose=0, rand_seed=3).init()
        seeds = ss.uids(np.arange(0, 2000, 10)) # The same people infected at the start
        si = sim.diseases.si
        if coded:
            si.set_prognoses(seeds, sources=-1)
        else:
            si.S[seeds] = False
            si.I[seeds] = True
        df = sim.run().analyzers[0].to_df()
        df = df[df.source >= 0].sort_values('target')
        return list(zip(df.source, df.target, df.t))

    # Over a random network, with and without modifiers of its contacts and of the transmission on it
    random = lambda: ss.RandomNet(n_contacts=4)
    mods = lambda target: [ss.modify(target, multiply=0.5, route='randomnet'), ss.modify('randomnet.pars.n_contacts', multiply=0.6)]
    plain = pairs(True, ss.peryear(0.3), random())
    assert len(plain) > 100 and plain == pairs(False, ss.peryear(0.3), random())
    modified = pairs(True, ss.peryear(0.3), random(), mods('si.rel_sus')) # A disease written as code has no transmission to name, so a modifier on a route is of its rel_sus
    assert 0 < len(modified) < 0.6*len(plain) and modified == pairs(False, ss.peryear(0.3), random(), mods('si.infection'))
    # Over a sexual network with acts (in which the same couples form whatever the disease, and whatever people's ages, which differ when a disease is declared), with a probability per act in each direction
    beta = ss.peract(m2f=0.1, f2m=0.05)
    couples = lambda: Couples(name='mf', debut=ss.constant(0), acts=4)
    directed = pairs(True, beta, couples())
    assert len(directed) > 50 and directed == pairs(False, beta, couples())
    assert pairs(True, beta, couples(), [ss.modify('si.rel_sus', multiply=0, route='mf')]) == [] # No transmission on that route
    with pytest.raises(ValueError, match='rel_sus'):
        pairs(True, beta, Couples(name='mf', acts=4), [ss.modify('si.rel_trans', multiply=0, route='mf')])
    return


@sc.timer()
def test_between():
    sc.heading('Testing ss.SexualNet between two groups, with targets by group and risk matching, and subclasses that pair their own')
    kw = dict(n_agents=5000, start=2000, stop=2015, dt=ss.months(1), verbose=0)
    everyone = dict(debut=ss.constant(0), participation=ss.bernoulli(1.0))

    # Sex workers and clients (ST22), each with their own number of partners at a time
    sw = ss.SexualNet(name='sexwork', between=('fsw', 'client'), n_contacts=dict(fsw=4, client=0.8), dur=ss.months(6), **everyone)
    assert sw.directions == ['fsw2client', 'client2fsw']
    sim = ss.Sim(diseases=ss.SIS(beta=dict(sexwork=ss.peract(fsw2client=0.02, client2fsw=0.01))), networks=sw, **kw)
    sim.stratify(work=['none', 'fsw', 'client'], split=[0.88, 0.02, 0.1]).run()
    net = sim.networks.sexwork
    level = sim.groups.level_raw([s.name for s in sim.groups.strats].index('work'))
    uids = sim.people.alive.uids
    degree = net.degrees(uids)
    assert (level[net.edges.p1] == 1).all() and (level[net.edges.p2] == 2).all() # Each partnership is a sex worker and a client
    assert np.isclose(degree[level[uids] == 1].mean(), 4, rtol=0.15) and np.isclose(degree[level[uids] == 2].mean(), 0.8, rtol=0.15), (degree[level[uids] == 1].mean(), degree[level[uids] == 2].mean())
    with pytest.raises(ValueError, match='each group of between'):
        ss.Sim(diseases='sis', networks=ss.SexualNet(n_contacts=dict(male=1, fsw=2), dur=ss.years(1)), n_agents=200, verbose=0).init()

    # Risk matching between men and women (ST52)
    net = ss.SexualNet(n_contacts=0.8, dur=ss.years(2), match=dict(risk=0.8), **everyone)
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=net, **kw).stratify(risk=['low', 'high'], split=[0.7, 0.3]).run()
    res = sim.networks[0].results
    assert np.isclose(res.match[-24:].mean(), 0.8, atol=0.05) and np.isclose(res.mean_degree[-1], 0.8, rtol=0.1), (res.match[-24:].mean(), res.mean_degree[-1])

    # A subclass that forms its own partnerships gets none from the base, and only between= of its parameters (HP45); between men, its error suggests between='mm' (ST45)
    class OwnMSM(ss.SexualNet):
        def step(self):
            self.end_pairs()
            if not len(self):
                men = self.sim.people.male.uids
                n = len(men)//2
                self.append(p1=men[:n], p2=men[n:2*n], beta=np.ones(n), dur=np.full(n, 100.0), acts=np.ones(n))
    assert list(OwnMSM().pars.keys()) == ['between']
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=OwnMSM(name='msm'), n_agents=500, verbose=0).init()
    assert len(sim.networks.msm) == 0
    with pytest.raises(ss.errors.RouteMismatch, match="between='mm'"):
        ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=OwnMSM(name='msm'), n_agents=500, verbose=0).run()
    assert len(ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=OwnMSM(name='msm', between='mm'), n_agents=500, stop=2002, verbose=0).run().networks.msm)
    return


@sc.timer()
def test_transmit_candidates():
    sc.heading('Testing that the edges that might transmit are found exactly, by a byte for each agent for many edges, or directly for a few')
    rng = np.random.default_rng(1)
    n = 1000
    rel_trans = np.where(rng.random(n) < 0.2, rng.random(n), 0).astype(np.float32)
    rel_sus = np.where(rng.random(n) < 0.7, 1.0, 0.0)
    rel_sus[:5] = np.nan # A candidate, ruled out later by the chance itself
    for n_edges in [50, 5000]: # Under an eighth of the agents, read directly
        p1, p2 = rng.integers(n, size=(2, n_edges))
        found = ss.perflib._passes.candidates(p1, p2, rel_trans, rel_sus)
        for (src, trg), inds in zip([(p1, p2), (p2, p1)], found):
            assert np.array_equal(inds, np.flatnonzero((rel_trans[src] != 0) & (rel_sus[trg] != 0))), f'Wrong candidates with {n_edges} edges'
    return


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Run tests
    man  = test_manual()
    rand = test_random()
    betw = test_between()
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
    acts = test_per_act_beta()
    cond = test_condoms()
    same = test_coded_declared_transmission()
    peract = test_per_act()
    read = test_beta_readings()
    route = test_route_modifiers()
    needles = test_needles()
    pa = test_partnership_approx()
    spec = test_peract_spec()
    default = test_beta_default()
    cand = test_transmit_candidates()

    T.toc()