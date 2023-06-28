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

    # Possible to initialize people with extra states
    extra_states = ss.named_dict(
        ss.State('debut', float),
    )
    ppl = ss.People(100, states=extra_states)  # BasePeople
    ppl.add_module(ss.HIV())

    return ppl


def test_networkss():

    # Make completely abstract layers
    n = 10_000
    n_people = 1000
    p1 = np.random.randint(n_people, size=n)
    p2 = np.random.randint(n_people, size=n)
    beta = np.ones(n)
    nw1 = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
    nw2 = ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand')  # Alternate method

    # Make people with some extra states, then make a dynamic sexual layer and update it
    states = ss.named_dict(
        ss.State('debut', float),
        ss.State('active', bool, True),
    )
    ppl = ss.People(100, states=states)  # BasePeople
    nw3 = ss.DynamicSexualNetwork()
    nw3.initialize(ppl)  # Initialize with ti=0
    nw3.update(ppl, ti=1, dt=1)  # Update with a timestep of 1 and

    nw4 = ss.Maternal()
    nw4.initialize(ppl)
    nw4.add_pairs(mother_inds=[1, 2, 3], unborn_inds=[100, 101, 102], dur=[1, 1, 1])

    return nw1, nw2, nw3, nw4


def test_microsim():
    sc.heading('Testing basic sim')

    ppl = ss.People(100)
    ppl.contacts['random'] = ss.simple_sexual()

    pars = defaultdict(dict)
    pars['hiv']['beta'] = {'random': 0.3}

    sim = ss.Sim(people=ppl, modules=[ss.HIV, ss.Pregnancy], pars=pars)
    sim.run()
    plt.figure()
    plt.plot(sim.tvec, sim.results.hiv.n_infected)
    plt.title('HIV number of infections')
    plt.show()
    return sim

def test_sim_construction():

    # Make people first and feed them in
    ppl = ss.People(100)

    # Get the sim to make people
    sim = ss.Sim()
    sim.initialize()  # Makes the people

    return



# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    # parsobj = test_parsobj()
    # ppl = test_people()
    # nw1, nw2, nw3, nw4 = test_networks()
    # sim = test_microsim()
    ppl = ss.People(100)
    ppl.contacts['random'] = ss.simple_sexual()

    # pars = defaultdict(dict)
    # pars['hiv']['beta'] = {'random': 0.3}

    sim = ss.Sim(people=ppl, modules=[ss.HIV, ss.Pregnancy])
    sim.initialize()
    # sim.run()

    sc.toc(T)
    print('Done.')
