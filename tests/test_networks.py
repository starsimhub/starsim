"""
Test networks
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss

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

    sc.toc(T)
    print('Done.')