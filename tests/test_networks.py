"""
Test networks
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import starsim_examples as sse
import scipy.stats as sps
import matplotlib.pyplot as plt

sc.options(interactive=False) # Assume not running interactively

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
    nw2 = ss.MaternalNet()
    nw2.init_pre(sim)
    nw2.add_pairs(mother_inds=[1, 2, 3], unborn_inds=[100, 101, 102], dur=[1, 1, 1])

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

    o = sc.objdict(s1=s1, s2=s2, s3=s3, s4=s4, similarity=simil)
    return o



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
    nw1 = sse.ErdosRenyiNet(p=p)
    ss.Sim(n_agents=small, networks=nw1, copy_inputs=False).init() # This initializes the network
    test_ER(small, p, nw1)

    # Automatic creation as part of sim
    ss.register_modules(sse)
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
    nw1 = sse.DiskNet()
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
    nw2 = sse.DiskNet(r=0.15, v=ss.freq(0.05, unit=ss.year))
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
    network = sse.NullNet()
    sir = ss.SIR(dur_inf=10, beta=0.1)
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    return sim


@sc.timer()
def test_dhs():
    sc.heading('Testing DHS networks...')
    # Construct DHS data
    n = 100
    age_strings = []
    for i in range(n):
        household_size = np.random.randint(1,6)
        ages = np.random.randint(0, 80, household_size)
        age_strings.append(sc.strjoin(ages))
    dhs_data = sc.dataframe(hh_id=np.arange(n), ages=age_strings)

    # Create the network and run
    household_dhs = sse.HouseholdDHSNet(dhs_data=dhs_data)
    sim = ss.Sim(n_agents=small, diseases='sis', networks=household_dhs)
    sim.run()
    return sim


@sc.timer()
def test_other():
    sc.heading('Other network tests...')

    print('Testing MSM network')
    msm = ss.MSMNet(participation=0.3)
    sim = ss.Sim(diseases=dict(type='sis', beta=0.5), networks=msm, copy_inputs=False)
    sim.run()

    print('Testing other network methods')
    msm.validate()
    inds1 = msm.get_inds([0])
    contacts = msm.find_contacts(inds1['p1'])
    assert contacts[0] == inds1['p2']
    inds2 = msm.pop_inds([0])
    assert inds1 == inds2

    return msm



# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Run tests
    man  = test_manual()
    rand = test_random()
    safe = test_randomsafe()
    stat = test_static()
    erdo = test_erdosrenyi()
    disk = test_disk()
    null = test_null()
    dhs  = test_dhs()
    oth  = test_other()

    T.toc()