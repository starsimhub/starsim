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
    sc.heading('Testing that a search of a deterministic sim that ends short of its best point gives that point, and that an unknown method is refused')
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

    # By default, a likelihood that JAX does not cover (a function) is fitted without gradients, rather than raising
    dur = ss.par(ss.days(8), bounds=[3, 15])
    k = ss.par(0.5, bounds=[0.1, 1])
    infection = S >> I @ beta*I
    sir = ss.Disease(infection, I >> R @ dur, ss.outcome(k*infection, name='seen'), init=dict(I=0.01), name='sir')
    data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_data('seen')
    normal = lambda observed, model: -0.5*((observed - model)/(0.1*model + 1))**2
    fit = ss.Sim(sir, data=data, **pars).fit(likelihood=dict(seen=normal), verbose=0)
    assert fit.method == 'nelder-mead' and np.isclose(fit.pars.beta, 0.5, rtol=0.05)
    return fit


#%% Run as a script
if __name__ == '__main__':
    test_best_trial()
