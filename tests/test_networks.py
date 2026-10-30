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
    o = sc.obj(nw1=nw1, nw2=nw2)
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
    s4 = ss.Sim(n_agents=small, diseases='sir', networks=ss.RandomNet(n_contacts=0)).run() # No contacts, no edges (it was a ZeroDivisionError)
    assert len(s4.networks[0]) == 0 and s4.results.sir.infection.sum() == 0

    # Tidy
    o = sc.obj(nw1=nw1, nw2=nw2, nw3=nw3)
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

    # Self-loops are made by default, and dropped with remove_self_loops=True
    loops = []
    for remove in [False, True]:
        e = ss.Sim(n_agents=50, networks=ss.RandomNet(method='exact', remove_self_loops=remove), rand_seed=1).init().networks[0].edges
        loops.append((e.p1 == e.p2).sum())
    assert loops[0] > 0 and loops[1] == 0, f'remove_self_loops=True should drop the edges joining an agent to itself, not leave {loops[1]} of {loops[0]}'

    o = sc.obj(nw1=nw1, nw2=nw2, nw4=nw4)
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
    pars = sc.obj(n_agents=small, dur=5, verbose=False)
    births = ss.Births(birth_rate=ss.time.freq(5, 'year'))
    simil = sc.obj()

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
    s3 = ss.Sim(pars, networks=ss.RandomNet(method='safe'), label='safe-nobirths').run()
    s4 = ss.Sim(pars, networks=ss.RandomNet(method='safe'), label='safe-births', demographics=births).run()

    n3 = s3.networks[0].to_edgelist()
    n4 = s4.networks[0].to_edgelist()
    simil.uid2 = auid_similarity(s3, s4)
    simil.safe = sc.similarity(n3, n4)
    print(f'Similarity in UIDs after {pars.dur} timesteps with/without births:\n{simil.uid2:%}')
    print(f'Similarity for random-safe networks after {pars.dur} timesteps:\n{simil.safe:%}\n')
    assert simil.safe > 0.5, 'RandomSafe networks were less similar than expected'

    # With many agents, some random numbers are tied; check that these edges don't all go to agent 0
    n = 20_000
    s5 = ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=20, method='safe'), rand_seed=1).init()
    nw5 = s5.networks[0]
    deg5 = np.bincount(np.concatenate([np.asarray(nw5.p1), np.asarray(nw5.p2)]), minlength=n)
    assert deg5.max() < 40, f'No agent should have many more than 20 contacts, but agent {deg5.argmax()} has {deg5.max()}'

    o = sc.obj(s1=s1, s2=s2, s3=s3, s4=s4, similarity=simil)
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

    o = sc.obj(exact_deg=exact_deg, fast_deg=fast_deg, exact_core=exact_core, fast_core=fast_core)
    return o


@sc.timer()
def test_network_uids():
    """ Test that all networks form edges between valid UIDs, not positional indices (issue #1418) """
    sc.heading('Testing that network edges reference valid UIDs')
    netclasses = [ss.RandomNet, lambda: ss.RandomNet(method='exact'), lambda: ss.RandomNet(method='safe')]
    out = sc.obj()
    for netclass in netclasses:
        name = netclass.__name__
        # A sim with births and deaths, so the UIDs of living agents are not contiguous
        sim = ss.Sim(n_agents=medium, networks=netclass(), diseases='sis', demographics=True, start=2000, stop=2005, rand_seed=1, verbose=0).run()
        nw = sim.networks[0]
        alive = set(sim.people.auids.tolist())
        refs = set(np.concatenate([np.asarray(nw.p1), np.asarray(nw.p2)]).tolist())
        invalid = ss.uids(sorted(refs - alive))
        if nw.one_step: # Edges that last one step keep those who died in the last step, until the next step would end them
            invalid = invalid[sim.people.ti_dead.raw[invalid] != sim.t.ti]
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
    o = sc.obj(nw1=nw1, nw2=nw2, nw3=nw3)
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
    assert len(s2.networks[0]) > 0 and s2.results.sir.infection.cum[-1] > 0, 'Expected edges, and infections through them'

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
    o = sc.obj(nw1=nw1, nw2=nw2)
    return o


@sc.timer()
def test_null():
    sc.heading('Testing NullNet...')
    people = ss.People(n_agents=small)
    network = ssl.networks.NullNet()
    sir = ss.SIR(dur_inf=10, beta=0.1)
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    net = sim.networks.nullnet
    assert len(net) == small and np.all(net.edges.beta == 0) and (net.edges.p1 == net.edges.p2).all(), 'Expected one zero-weight self-edge per agent'
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

    # With the default household data, households have more than one member
    sim3 = ss.Sim(n_agents=medium, diseases='sis', networks=ssl.HouseholdNet(dhs_data='default'), demographics=ss.Pregnancy(), start=2000, stop=2010, verbose=0).run()
    hh_ids = sim3.networks.householdnet.household_ids
    sizes = np.bincount(hh_ids[hh_ids.notnan].astype(int)) # Unborn agents have no household
    sizes = sizes[sizes > 0]
    assert len(sizes) > 1 and sizes.mean() > 1 and len(sim3.networks.householdnet) > 0

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
    assert ss.networks.find_contacts(net2.p1, net2.p2, [1,3]) == {1,2,3}
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
    assert isinstance(nets.s, ssl.ClusterNet) and isinstance(nets.w, ssl.ClusterNet) # Clusters by default
    constant = ss.Sim(n_agents=small, networks=ssl.HybridNet(contacts=dict(s=ss.constant(20))), verbose=0).init().networks.s
    assert constant.pars.cluster_size.pars.v == 21 # A cluster of 21 gives 20 contacts each
    closed = ss.Sim(n_agents=small, diseases='sis', networks=ssl.HybridNet(contacts=dict(s=0)), verbose=0).run()
    assert len(closed.networks.s) == 0 and len(closed.networks.w) > 0 # No contacts is an empty network

    # As in Covasim v3, use random schools and workplaces, and choose the targets of edges uniformly
    sim2 = ss.Sim(n_agents=medium, diseases='sis', networks=ssl.HybridNet(clustered=False, method='uniform'), verbose=0).init()
    assert all(sim2.networks[k].pars.method == 'uniform' for k in ['s', 'w', 'c'])
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
    with pytest.warns(RuntimeWarning, match='set only the mean') as record: # The same unit as the sim: only that the std of the default was kept (see Pars._update_dist())
        ss.Sim(n_agents=200, diseases=ss.SIS(beta=ss.peract(0.01)), networks=ss.SexualNet(dur=5), verbose=0).init()
    assert not any('read in' in str(w.message) for w in record)
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
    with pytest.raises(ValueError, match="between='mf'"):
        ss.Sim(sis(ss.peract(m2f=0.01, f2m=0.01)), networks=ss.SexualNet(between='mm'), **kw).init()
    assert ss.peract({('m', 'f'): 0.3, ('f', 'm'): 0.15}) == ss.peract(m2f=0.3, f2m=0.15) # A direction as a pair of sides, or by name
    with pytest.raises(TypeError, match=r'ss.peract\(0.01\)'): # A rate per unit time times acts per unit time is not a hazard
        ss.Sim(sis(ss.permonth(0.01)), networks=ss.SexualNet(), **kw).init()
    sim = ss.Sim(diseases='sis', networks=['sexual', 'random'], n_agents=500, verbose=0).init() # But the default of ss.SIS is a probability per act there, as in Starsim 3
    assert sim.diseases.sis.pars.beta.raw['sexualnet'] == ss.peract(0.05) and any('which has acts' in note for note in sim.notes)
    with pytest.raises(ValueError, match=r'ss.peract\(m2f=0.3, f2m=0.15\)'): # A pair has no direction names
        ss.Sim(sis(dict(sexualnet=[0.3, 0.15])), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ValueError, match='no acts'): # No act to be per
        ss.Sim(sis(ss.peract(0.01)), networks='random', **kw).init()
    with pytest.raises(ValueError, match='f2m'): # Every direction needs a value ...
        ss.Sim(sis(ss.peract(m2f=0.01)), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ValueError, match='m2f and f2m'): # ... and one it does not have raises, listing those it has
        ss.Sim(sis(ss.peract(m2f=0.01, f2m=0.01, m2m=0.02)), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ValueError, match='no sides'):
        ss.Sim(sis(ss.peract(m2f=0.01, f2m=0.01)), networks=ss.DegreeNet(name='needles', dur=ss.years(1), acts=ss.freqpermonth(20)), **kw).init()
    with pytest.raises(ValueError, match='A probability per act must be between 0 and 1'):
        ss.peract(1.5)
    with pytest.raises(ValueError, match='step'): # A whole number of acts per step, drawn once, depends on the step
        ss.Sim(sis(0.01), networks=ss.SexualNet(acts=ss.poisson(lam=ss.freqperyear(80))), **kw).init()
    with pytest.raises(ValueError, match='frequency'): # A hazard is not a number of acts (as agents, it would be the chance of an act in a step)
        ss.Sim(sis(0.01), networks=ss.SexualNet(acts=ss.peryear(80)), **kw).init()

    # Mother-to-child networks transmit one way, so one value is enough (a pair was [b, 0])
    S, I = ss.states('S I')
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=ss.peract(m2f=0.01, f2m=0.005), maternalnet=ss.peryear(0.5)) * I), init_prev=0.1, name='hiv')
    sim = ss.Sim(hiv, networks=[ss.SexualNet(), ss.MaternalNet()], demographics=ss.Pregnancy(), **kw).run()
    assert sim.results.hiv.I[-1] > 0
    return


@sc.timer()
def test_peract_describe():
    sc.heading('Testing how a probability per act is described')
    def make(beta):
        S, P, EL = ss.states('S P EL')
        syph = ss.Disease(S, P, EL, infection=ss.Transition(S, P, rate=beta * (P + 0.55*EL)),
                          latent=ss.Transition(P, EL, dur=ss.years(1)), clear=ss.Transition(EL, S, dur=ss.years(2)), init=dict(P=0.05), name='syph')
        condoms = ss.modify(syph.infection, efficacy=0.5, share=ss.timeseries({2000: 0.05, 2020: 0.3}), route='sexualnet', name='condoms')
        return ss.Sim(syph, networks=ss.SexualNet(), interventions=condoms, n_agents=500, start=2000, stop=2002, verbose=0)
    sim = make(ss.peract(m2f=0.3, f2m=0.15)).init()
    text = sim.diseases.syph.describe(output=True)
    for line in ['from P + 0.55 EL', 'beta on sexualnet: per act, m2f 0.3, f2m 0.15', "sexualnet's sides m and f", '"condoms"']:
        assert line in text, line
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
    assert sim.results.d.I[-1] > 0
    assert any('msm' in note and 'default' in note for note in sim.notes), sim.notes
    assert 'beta on maternalnet' in sim.diseases.d.describe(output=True)

    # Without a default, a name that is not a route, or a route with no value, still raises, since it is likely a typo
    with pytest.raises(ValueError, match='no route is called msm'):
        ss.Sim(make(sexualnet=ss.peract(0.01), maternalnet=0, msm=ss.peract(0.1)), networks=[ss.SexualNet(), ss.MaternalNet()], **kw).init()
    with pytest.raises(ValueError, match='maternalnet has no value'):
        ss.Sim(make(sexualnet=ss.peract(0.01)), networks=[ss.SexualNet(), ss.MaternalNet()], **kw).init()

    # The default is the same as giving each route its value, as agents and as an ODE, which agree; it can be a typed rate or a parameter
    def pools(solver, **betas):
        nets = [ss.MixingPool(n_contacts=4, name='pool'), ss.MixingPool(n_contacts=2, name='home')]
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=betas * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
        sim = ss.Sim(sir, networks=nets, total_pop=10_000, n_agents=10_000, solver=solver, unit='day', stop=100, rand_seed=1, verbose=0)
        return sim.run().results.sir.R.values
    res = {}
    for solver in ['abm', 'ode']:
        res[solver] = pools(solver, pool=ss.perday(0.03), home=ss.perday(0.02))
        assert np.array_equal(res[solver], pools(solver, pool=ss.perday(0.03), school=ss.perday(0.1), default=ss.perday(0.02))), solver
        assert np.array_equal(res[solver], pools(solver, pool=ss.perday(0.03), default=ss.par(ss.perday(0.02), name='other'))), solver
    assert np.isclose(res['abm'][-1], res['ode'][-1], rtol=0.15), (res['abm'][-1], res['ode'][-1])

    # With no networks (homogeneous mixing), the default is the beta, and with none, that is an error
    sim = ss.Sim(make(sexualnet=ss.peract(0.01), default=ss.peryear(2)), **kw).run()
    assert sim.results.d.I[-1] > 0
    with pytest.raises(ValueError, match='homogeneous mixing'):
        ss.Sim(make(sexualnet=ss.peract(0.01)), **kw).run()

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
    assert 1*si.pars.beta == si.engine.value(si.pars.beta), 'The current value of a declared parameter is a number'
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
def test_beta_by_person():
    sc.heading('Testing a beta per act that differs between the people who would be infected')
    def make(beta):
        S, I = ss.states('S I')
        return ss.Disease(infection=S >> I @ beta * I, recovery=I >> S @ ss.years(1), init_prev=0.2, name='d')
    def male_infections(beta):
        sim = ss.Sim(make(beta), networks=ss.SexualNet(), n_agents=2000, stop=3, rand_seed=1, verbose=0).run()
        d = sim.diseases.d
        return np.count_nonzero(d.I & sim.people.male & (d.ti_I > 0)) # Infected during the run, not seeded

    # By the sex of the susceptible, as by direction: men are infected only by f2m, so with f2m = 0 never
    m2f = ss.par(0.2, name='m2f')
    assert male_infections(m2f * ss.by(sex=dict(female=1.0, male=0.0))) == 0
    assert male_infections(ss.peract(m2f=0.2, f2m=0.1)) == male_infections(m2f * ss.by(sex=dict(female=1.0, male=0.5)))
    return


def test_infection_sources():
    sc.heading('Testing who infected whom, for a module\'s own code (engine.sources())')
    found = []
    def check(sim):
        d = sim.diseases.d
        new = d.infection.uids | d.mtct.uids
        sources, nets = d.engine.sources(new)
        found.append((len(new), np.count_nonzero(~np.isnan(sources))))
    S, I = ss.states('S I')
    d = ss.Disease(infection=S >> I @ dict(sexualnet=ss.peract(0.2), default=0) * I, mtct=S >> I @ dict(maternalnet=ss.peryear(2), default=0) * I, recovery=I >> S @ ss.years(1), init_prev=0.2, name='d')
    sim = ss.Sim(d, networks=[ss.SexualNet(), ss.MaternalNet()], demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), n_agents=2000, stop=3, analyzers=check, rand_seed=1, verbose=0).run()
    n_new, n_found = np.sum(found, axis=0)
    assert n_new > 0 and n_found == n_new # Each infection over a network has its source, whichever transmission it was by
    return sim


def test_transmit_candidates():
    sc.heading('Testing that the edges that might transmit are found exactly, by a byte for each agent for many edges, or directly for a few')
    rng = np.random.default_rng(1)
    n = 1000
    rel_trans = np.where(rng.random(n) < 0.2, rng.random(n), 0).astype(np.float32)
    rel_sus = np.where(rng.random(n) < 0.7, 1.0, 0.0)
    rel_sus[:5] = np.nan # A candidate, ruled out later by the chance itself
    for n_edges in [50, 5000]: # Under an eighth of the agents, read directly
        p1, p2 = rng.integers(n, size=(2, n_edges))
        found = ss.solvers.backends.passes.candidates(p1, p2, rel_trans, rel_sus)
        for (src, trg), inds in zip([(p1, p2), (p2, p1)], found):
            assert np.array_equal(inds, np.flatnonzero((rel_trans[src] != 0) & (rel_sus[trg] != 0))), f'Wrong candidates with {n_edges} edges'
    return


@sc.timer()
def test_shared_pass():
    sc.heading('Testing that transmissions from different states over one network, worked out in one pass, infect whom each would alone')
    S, I, R = ss.states('S I R')
    foi = ss.peryear(3)*I
    sir = ss.Disease(infection=S >> I @ foi, reinfection=R >> I @ 0.4*foi, recovery=I >> R @ ss.peryear(2), init=dict(I=0.05, R=0.3), name='sir')
    sim = ss.Sim(diseases=sir, networks=ss.RandomNet(n_contacts=5), n_agents=medium, start=2000, stop=2002, dt=ss.years(0.1), verbose=0)
    sim.run(until='2001-01-01')
    eng = sim.diseases.sir.engine
    pairs = [(tr, src) for tr in eng.graph.transmissions for src in tr.src]
    rng = sim.diseases.sir.trans_rng
    ind = rng.dists[0].ind
    together = eng.transmit_together(pairs)
    assert together is not None and sum(map(len, together)) > 0, 'Expected the two transmissions to be worked out together, and to infect someone'
    for (tr,src),uids in zip(pairs, together):
        rng.jump(to=ind, force=True) # The draws the pass had, for each transmission alone
        assert np.array_equal(uids, eng.transmit(tr, src)), f'Expected {tr.name} to infect the same people in the shared pass as alone'
    return together


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
    oth  = test_other()
    ops  = test_edge_ops()
    dyn  = test_dynamic()
    clus = test_cluster()
    hyb  = test_hybrid()
    bare = test_bare_units()
    acts = test_per_act_beta()
    cond = test_condoms()
    same = test_coded_declared_transmission()
    read = test_beta_readings()
    peract = test_peract_describe()
    default = test_beta_default()
    person = test_beta_by_person()
    sources = test_infection_sources()
    cand = test_transmit_candidates()
    shared = test_shared_pass()

    T.toc()
