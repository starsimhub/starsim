"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import starsim.fit
import numpy as np
import pandas as pd
import sciris as sc
import pytest


#%% Define the tests


@sc.timer()
def test_strain_data():
    sc.heading('Testing data by strain for strains with courses of their own: long data with a strain column is each strain\'s own result')
    def genotype(name):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=S >> I @ ss.peryear(3)*I, clearance=I >> R @ ss.years(1), name=name)
    def make(data):
        hpv = ss.Disease(name='hpv', init_prev=0.2, strains=ss.Strains([genotype('hpv16'), genotype('hpv18')], name='genotype'))
        return ss.Sim(hpv, n_agents=500, start=2000, stop=2010, data=data, verbose=0)
    data = pd.DataFrame(dict(year=[2003, 2006]*2, genotype=['hpv16']*2 + ['hpv18']*2, clearance=[100, 50, 90, 40]))
    sim = make(data).run()
    assert sim.data.columns.tolist() == ['hpv16.clearance', 'hpv18.clearance']
    assert np.isfinite(ss.fit.mismatch(sim))
    with pytest.raises(ss.errors.UnknownDataColumn): # One spelling: the strain's own result
        make(pd.DataFrame({'year':[2003], 'hpv.clearance[genotype=hpv16]':[100]})).init()
    return sim


#%% Run as a script
if __name__ == '__main__':
    test_strain_data()
