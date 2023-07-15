"""
Test objects from parameters.py and default_parameters.py
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import stisim as ss
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



def test_microsim():
    sc.heading('Test sim with default parameters')
    # NOTE: NOT FUNCTIONAL YET
    pars = ss.make_default_pars()
    parset = ss.ParameterSet(pars)
    sim = ss.Sim(pars=parset)
    sim.initialize()
    sim.run()

    return sim



# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    sim1 = test_microsim()

    sc.toc(T)
    print('Done.')
