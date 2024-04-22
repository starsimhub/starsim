"""
Test objects from base.py
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import matplotlib.pyplot as pl

sc.options(interactive=False) # Assume not running interactively

# %% Define the tests

def test_people():
    sc.heading('Testing people object')

    # Base people contains only the states defined in base.base_states
    ppl = ss.People(100)  # BasePeople
    del ppl

    # Possible to initialize people with extra states, e.g. a geolocation
    def geo_func(n):
        locs = [1,2,3]
        return np.random.choice(locs, n)
    extra_states = [
        ss.IntArr('geolocation', default=geo_func),
    ]
    ppl = ss.People(100, extra_states=extra_states)

    # Possible to add a module to people outside a sim (not typical workflow)
    ppl.add_module(ss.HIV())

    return ppl


def test_networks():
    sc.heading('Testing networks')

    # Make completely abstract layers
    n_edges = 10_000
    n_people = 1000
    p1 = np.random.randint(n_people, size=n_edges)
    p2 = np.random.randint(n_people, size=n_edges)
    beta = np.ones(n_edges)
    nw1 = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
    nw2 = ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand')  # Alternate method

    sim = ss.Sim()
    sim.initialize()
    
    nw3 = ss.MaternalNet()
    nw3.initialize(sim)
    nw3.add_pairs(mother_inds=[1, 2, 3], unborn_inds=[100, 101, 102], dur=[1, 1, 1])

    return nw1, nw2, nw3


def test_microsim(do_plot=False):
    sc.heading('Test small HIV simulation')

    # Make HIV module
    hiv = ss.HIV()
    # Set beta. The first entry represents transmission risk from infected p1 -> susceptible p2
    # Need to be careful to get the ordering right. The set-up here assumes that in the simple
    # sexual  network, p1 is male and p2 is female. In the maternal network, p1=mothers, p2=babies.
    hiv.pars['beta'] = {'mf': [0.15, 0.10], 'maternal': [0.2, 0]}

    sim = ss.Sim(
        people=ss.People(100),
        networks=[ss.MFNet(), ss.MaternalNet()],
        demographics=ss.Pregnancy(),
        diseases=hiv
    )
    sim.initialize()
    sim.run()

    if do_plot:
        pl.figure()
        pl.plot(sim.tivec, sim.results.hiv.n_infected)
        pl.title('HIV number of infections')

    return sim


def test_ppl_construction():
    sc.heading('Test making people and providing them to a sim')

    def init_debut(module, sim, uids):
        # Test setting the mean debut age by sex, 16 for men and 21 for women.
        loc = np.full(len(uids), 16)
        loc[sim.people.female[uids]] = 21
        return loc

    mf_pars = {
        'debut': ss.normal(loc=init_debut, scale=2),  # Age of debut can vary by using callable parameter values
    }
    sim_pars = {'networks': [ss.MFNet(mf_pars)], 'n_agents': 100}
    gon_pars = {'beta': {'mf': [0.08, 0.04]}, 'p_death': 0.2}
    gon = ss.Gonorrhea(pars=gon_pars)

    sim = ss.Sim(pars=sim_pars, diseases=[gon])
    sim.initialize()
    sim.run()
    pl.figure()
    pl.plot(sim.tivec, sim.results.gonorrhea.n_infected)
    pl.title('Number of gonorrhea infections')

    return sim


def test_arrs():
    sc.heading('Testing Arr objects')
    o = sc.objdict()
    
    # Create a sim with only births
    pars = dict(n_agents=1000, diseases='sis', networks='random')
    p1 = sc.mergedicts(pars, birth_rate=10)
    p2 = sc.mergedicts(pars, death_rate=10)
    s1 = ss.Sim(pars=p1).run()
    s2 = ss.Sim(pars=p2).run()
    
    # Tests
    assert len(s1.people.auids) > len(s2.people.auids), 'Should have more people with births'
    assert np.array_equal(s1.people.age, s1.people.age.raw[s1.people.auids]), 'Different ways of indexing age should match'
    assert np.array_equal(s1.people.alive.uids, s1.people.auids), 'Active/alive agents should match'
    assert np.all(s2.people.alive), 'All agents should be alive when indexed like this'
    assert not np.all(s2.people.alive.raw), 'Some agents should not be alive when indexed like this'
    assert np.array_equal(~s1.people.female, s1.people.male), 'Definition of men does not match'
    assert isinstance(s1.people.age < 5, ss.BoolArr), 'Performing logical operations should return a BoolArr'
    
    o.s1 = s1
    o.s2 = s2   
    
    return o



# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    ppl = test_people()
    nw1, nw2, nw3 = test_networks()
    sim1 = test_microsim(do_plot)
    sim2 = test_ppl_construction()
    sims = test_arrs()

    sc.toc(T)
    pl.show()
    print('Done.')