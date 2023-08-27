"""
Test objects from base.py
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import stisim as ss
import matplotlib.pyplot as plt


# %% Define the tests

def test_people():
    sc.heading('Testing people object')

    # Base people contains only the states defined in base.base_states
    ppl = ss.People(100)  # BasePeople
    del ppl

    # Possible to initialize people with extra states, e.g. a geolocation
    extra_states = ss.ndict(
        ss.State('geolocation', int, lambda n: np.random.choice([1,2,3],n)),
    )
    ppl = ss.People(100, extra_states=extra_states)

    # Possible to add a module to people outside a sim (not typical workflow)
    ppl.add_module(ss.HIV())

    return ppl


def test_networks():

    # Make completely abstract layers
    n_edges = 10_000
    n_people = 1000
    p1 = np.random.randint(n_people, size=n_edges)
    p2 = np.random.randint(n_people, size=n_edges)
    beta = np.ones(n_edges)
    nw1 = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
    nw2 = ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand')  # Alternate method

    # Make people, then make a dynamic sexual layer and update it
    ppl = ss.People(100)  # BasePeople
    ppl.initialize()  # This seems to be necessary, although not completely clear why...
    nw3 = ss.hpv_network()
    nw3.initialize(ppl)
    nw3.update(ppl, ti=1, dt=1)  # Update by providing a timestep & current time index

    nw4 = ss.maternal()
    nw4.initialize(ppl)
    nw4.add_pairs(mother_inds=[1, 2, 3], unborn_inds=[100, 101, 102], dur=[1, 1, 1])

    return nw1, nw2, nw3, nw4


def test_microsim():
    sc.heading('Test making people and providing them to a sim')

    networks = [ss.simple_sexual(), ss.maternal()]
    ppl = ss.People(100, networks=networks)

    # Make HIV module
    hiv = ss.HIV()
    # Set beta. The first entry represents transmission risk from infected p1 -> susceptible p2
    # Need to be careful to get the ordering right. The set-up here assumes that in the simple
    # sexual  network, p1 is male and p2 is female. In the maternal network, p1=mothers, p2=babies.
    hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}

    sim = ss.Sim(people=ppl, modules=[hiv, ss.Pregnancy()])
    sim.initialize()
    sim.run()

    plt.figure()
    plt.plot(sim.tivec, sim.results.hiv.n_infected)
    plt.title('HIV number of infections')

    return sim

def test_ppl_construction():

    sim_pars = {'networks': [ss.simple_sexual()], 'n_agents': 100}
    gon_pars = {'beta': {'simple_sexual': [0.08, 0.04]}}
    gon = ss.Gonorrhea(pars=gon_pars)

    sim = ss.Sim(pars=sim_pars, modules=[gon])
    sim.initialize()
    sim.run()
    plt.figure()
    plt.plot(sim.tivec, sim.results.gonorrhea.n_infected)
    plt.title('Number of gonorrhea infections')

    return sim



# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    ppl = test_people()
    nw1, nw2, nw3, nw4 = test_networks()
    sim1 = test_microsim()
    sim2 = test_ppl_construction()

    sc.toc(T)
    print('Done.')
