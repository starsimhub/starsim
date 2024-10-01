"""
Test networks
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import scipy.stats as sps

sc.options(interactive=False) # Assume not running interactively

small = 100
medium = 1000

# %% Define the tests

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
        p_value = sps.chisquare(f_obs, f_exp).pvalue # Compute the X2 p-value
        assert not p_value < alpha
        return p_value

    # Manual creation
    p = 0.1
    nw1 = ss.ErdosRenyiNet(p=p)
    ss.Sim(n_agents=small, networks=nw1, copy_inputs=False).init() # This initializes the network
    test_ER(small, p, nw1)

    # Automatic creation as part of sim
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


def test_disk():
    sc.heading('Testing Disk network')

    # Visualize the path of agents
    nw1 = ss.DiskNet()
    s1 = ss.Sim(n_agents=5, dur=50, networks=nw1, copy_inputs=False).init() # This initializes the network

    if sc.options.interactive:
        # Visualize motion:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fig, ax = plt.subplots()
        vdt = nw1.pars.v * s1.pars.dt

        cmap = mpl.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, s1.pars.n_agents))
        ax.scatter(nw1.x, nw1.y, s=50, c=colors)
        for i in range(s1.pars.dur):
            ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k-', lw=1)
            ax.quiver(nw1.x, nw1.y, vdt * np.cos(nw1.theta), vdt * np.sin(nw1.theta), color=colors)
            ax.set_aspect('equal', adjustable='box') #ax.set_xlim([0,1]); ax.set_ylim([0,1])
            s1.run_one_step()

    # Simulate SIR on a DiskNet
    nw2 = ss.DiskNet(r=0.15, v=0.05)
    s2 = ss.Sim(n_agents=small, networks=nw2, diseases='sir').init() # This initializes the network
    s2.run()

    if sc.options.interactive:
        s2.plot()
        plt.show()

    return s1, s2


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


def test_null():
    sc.heading('Testing NullNet...')
    people = ss.People(n_agents=small)
    network = ss.NullNet()
    sir = ss.SIR(pars=dict(dur_inf=10, beta=0.1))
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    return sim


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
    stat = test_static()
    erdo = test_erdosrenyi()
    disk = test_disk()
    null = test_null()
    oth  = test_other()

    T.toc()