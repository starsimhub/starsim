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
    sim.initialize()
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
    ss.Sim(n_agents=small, networks=nw1, copy_inputs=False).initialize() # This initializes the network
    
    # Automatic creation as part of sim
    s2 = ss.Sim(n_agents=small, networks='random').initialize()
    nw2 = s2.networks[0]
    
    # Increase the number of contacts
    nwdict = dict(type='random', n_contacts=20)
    s3 = ss.Sim(n_agents=small, networks=nwdict).initialize()
    nw3 = s3.networks[0]
    
    # Checks
    assert np.array_equal(nw1.p2, nw2.p2), 'Implicit and explicit creation should give the same network'
    assert len(nw3) == len(nw2)*2, 'Doubling n_contacts should produce twice as many contacts'
    
    # Tidy
    o = sc.objdict(nw1=nw1, nw2=nw2, nw3=nw3)
    return o

def test_erdosrenyi():
    sc.heading('Testing Erdos-Renyi network')

    def test_ER(n, p, nw):
        p12 = np.concatenate([nw.edges['p1'], nw.edges['p2']])
        upper = sps.binom.ppf(n=n, p=p, q=0.999)
        bins = np.arange(upper+1)
        counts = np.histogram(p12, bins=np.arange(n+1))[0]
        f_obs = np.histogram(counts, bins=bins)[0]
        pp = sps.binom.pmf(bins[:-1], n=n, p=p)
        f_exp = f_obs.sum()*pp / pp.sum()
        assert not sps.chisquare(f_obs, f_exp).pvalue < 0.01
        return

    # Manual creation
    p = 0.1
    nw1 = ss.ErdosRenyiNet(p=p)
    s1 = ss.Sim(n_agents=small, networks=nw1, copy_inputs=False).initialize() # This initializes the network
    test_ER(small, p, nw1)

    # Automatic creation as part of sim
    s2 = ss.Sim(n_agents=small, networks='erdosrenyi').initialize()
    nw2 = s2.networks[0]
    
    # Larger example with higher p
    p=0.6
    nwdict = dict(type='erdosrenyi', p=p)
    s3 = ss.Sim(n_agents=medium, networks=nwdict).initialize()
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
    s1 = ss.Sim(n_agents=5, n_years=50, networks=nw1, copy_inputs=False).initialize() # This initializes the network

    if sc.options.interactive:
        # Visualize motion:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fig, ax = plt.subplots()
        vdt = nw1.pars.v * s1.pars.dt

        cmap = mpl.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, s1.pars.n_agents))
        ax.scatter(nw1.x, nw1.y, s=50, c=colors)
        for i in range(s1.pars.n_years):
            ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k-', lw=1)
            ax.quiver(nw1.x, nw1.y, vdt * np.cos(nw1.theta), vdt * np.sin(nw1.theta), color=colors)
            ax.set_aspect('equal', adjustable='box') #ax.set_xlim([0,1]); ax.set_ylim([0,1])
            s1.step()

    # Simulate SIR on a DiskNet
    nw2 = ss.DiskNet(r=0.15, v=0.05)
    s2 = ss.Sim(n_agents=small, networks=nw2, diseases=[ss.SIR()], copy_inputs=False).initialize() # This initializes the network
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
    nw1 = ss.Sim(n_agents=n, networks=nd1).initialize().networks[0]
    
    # Create with n_contacts
    nd2 = dict(type='static', n_contacts=nc)
    nw2 = ss.Sim(n_agents=n, networks=nd2).initialize().networks[0]
    
    # Check
    assert len(nw1) == len(nw2), 'Networks should be the same length'
    target = n*n*p/2
    assert target/2 < len(nw1) < target*2, f'Network should be approximately length {target}'
    
    # Tidy
    o = sc.objdict(nw1=nw1, nw2=nw2)
    return o


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    man = test_manual()
    rnd = test_random()
    sta = test_static()
    er = test_erdosrenyi()
    d = test_disk()

    sc.toc(T)
    print('Done.')