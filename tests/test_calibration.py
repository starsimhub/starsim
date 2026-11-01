"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import starsim.fit
import numpy as np
import sciris as sc
import scipy.optimize as spo
import pytest
from unittest import mock


#%% Define the tests


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
    test_best_trial()
