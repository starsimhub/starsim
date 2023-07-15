"""
Test objects from parameters.py and default_parameters.py
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import stisim as ss
from stisim import utils as ssu
import matplotlib.pyplot as plt


# %% Define the tests



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


def test_microsim():
    sc.heading('Test making people and providing them to a sim')

    ppl = ss.People(100)
    ppl.networks = ssu.named_dict(ss.simple_sexual(), ss.maternal())

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
    plt.show()

    return sim



# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    ppl = test_people()
    sim1 = test_microsim()

    sc.toc(T)
    print('Done.')
