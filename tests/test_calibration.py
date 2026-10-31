"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import starsim.fit
import numpy as np
import pandas as pd
import sciris as sc
import scipy.optimize as spo
import pytest
from unittest import mock


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
    with pytest.raises(KeyError, match="name each strain's own result"): # One spelling: the strain's own result
        make(pd.DataFrame({'year':[2003], 'hpv.clearance[genotype=hpv16]':[100]})).init()
    return sim


@sc.timer()
def test_best_trial():
    sc.heading('Testing that a search of a deterministic sim that ends short of its best point gives that point, and the methods that were removed')
    beta = ss.par(0.3, bounds=[0.1, 1])
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ beta*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
    pars = dict(total_pop=10_000, unit='day', stop=60, verbose=0)
    data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_data('I')
    search = spo.minimize
    def stalled(fun, x0, **kwargs): # As SciPy's CG did after a failed line search: back at the start, having tried better
        res = search(fun, x0, **kwargs)
        res.x, res.fun = x0, fun(x0)
        return res
    with mock.patch.object(spo, 'minimize', stalled):
        fit = ss.Sim(sir, data=data, **pars).fit(method='nelder-mead', scan=0, verbose=0)
    assert np.isclose(fit.pars.beta, 0.5, rtol=0.01) and fit.mismatch <= fit.trials.objective.min()*1.001
    with pytest.raises(ValueError, match='Unknown fitting method'):
        ss.Sim(sir, data=data, **pars).fit(method='cg', verbose=0)
    return fit


#%% Run as a script
if __name__ == '__main__':
    test_strain_data()
    test_best_trial()
