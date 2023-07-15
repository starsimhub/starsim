"""
Test objects from parameters.py and default_parameters.py
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import stisim
from stisim import utils as ssu
import matplotlib.pyplot as plt
import pytest

# %% Define the tests

class TestParsObj:
    @pytest.fixture
    def pars_obj(self):
        return stisim.ParsObj({'key1': 1, 'key2': 2})

    def test_getitem_existing_key(self, pars_obj):
        assert pars_obj['key1'] == 1

    def test_getitem_nonexistent_key(self, pars_obj):
        with pytest.raises(sc.KeyNotFoundError):
            value = pars_obj['nonexistent']

    def test_setitem_existing_key(self, pars_obj):
        pars_obj['key1'] = 10
        assert pars_obj['key1'] == 10

    def test_setitem_nonexistent_key(self, pars_obj):
        with pytest.raises(sc.KeyNotFoundError):
            pars_obj['nonexistent'] = 100

    def test_update_pars_valid_dict(self, pars_obj):
        pars_obj.update_pars({'key3': 3})
        assert pars_obj['key3'] == 3

    def test_update_pars_invalid_dict(self, pars_obj):
        with pytest.raises(TypeError):
            pars_obj.update_pars('invalid_dict')

    def test_update_pars_existing_key(self, pars_obj):
        with pytest.raises(sc.KeyNotFoundError):
            pars_obj.update_pars({'key1': 100}, create=False)

    def test_update_pars_nonexistent_key(self, pars_obj):
        with pytest.raises(sc.KeyNotFoundError):
            pars_obj.update_pars({'nonexistent': 200}, create=False)


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



# # %% Run as a script
# if __name__ == '__main__':
#     # Start timing
#     T = sc.tic()
#
#     # Run tests
#     ppl = test_people()
#     sim1 = test_microsim()
#
#     sc.toc(T)
#     print('Done.')
