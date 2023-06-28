"""
Test objects from base.py
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import stisim as ss
from collections import defaultdict
import pytest
import matplotlib.pyplot as plt


# %% Define the tests

def test_parsobj():
    sc.heading('Testing parameters object')

    pars1 = {'a': 1, 'b': 2}
    parsobj = ss.ParsObj(pars1)

    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']

    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        ss.ParsObj(pars2)

    return parsobj


def test_people():
    sc.heading('Testing base people object')

    # Base people contains only the states defined in base.base_states
    ppl = ss.BasePeople(100)  # BasePeople
    del ppl

    # Possible to initialize people with extra states, e.g. a geolocation
    extra_states = ss.named_dict(
        ss.StochState('geolocation', int, distdict=dict(dist='choice', par1=[1, 2, 3])),
    )
    ppl = ss.People(100, states=extra_states)

    # Possible to add a module to people outside a sim (not typical workflow)
    ppl.add_module(ss.HIV())

    return ppl


def test_networks():

    # Make completely abstract layers
    n = 10_000
    n_people = 1000
    p1 = np.random.randint(n_people, size=n)
    p2 = np.random.randint(n_people, size=n)
    beta = np.ones(n)
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
    sc.heading('Testing basic sim')

    ppl = ss.People(100)
    ppl.networks['random'] = ss.simple_sexual()
    sim = ss.Sim(people=ppl, modules=[ss.HIV(), ss.Pregnancy()])
    sim.initialize()
    sim.run()
    plt.figure()
    plt.plot(sim.tvec, sim.results.hiv.n_infected)
    plt.title('HIV number of infections')
    plt.show()
    return sim

def test_ppl_construction():

    sim = ss.Sim(networks=[ss.simple_sexual()], modules=[ss.HIV(), ss.Pregnancy()])
    sim.initialize()

    return sim



# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    # parsobj = test_parsobj()
    # ppl = test_people()
    # nw1, nw2, nw3, nw4 = test_networks()
    sim1 = test_microsim()
    # sim = test_ppl_construction()

    sc.toc(T)
    print('Done.')
