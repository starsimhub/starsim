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
    data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_synthetic_data('I')
    search = spo.minimize
    def stalled(fun, x0, **kwargs): # As SciPy's CG did after a failed line search: back at the start, having tried better
        res = search(fun, x0, **kwargs)
        res.x, res.fun = x0, fun(x0)
        return res
    with mock.patch.object(spo, 'minimize', stalled):
        fit = ss.Sim(sir, data=data, **pars).fit(method=ss.fit.method('nelder-mead', scan=0), verbose=0)
    assert np.isclose(fit.pars.beta, 0.5, rtol=0.01) and fit.objective <= fit.trials.objective.min()*1.001
    with pytest.raises(ValueError, match='Unknown fitting method'):
        ss.Sim(sir, data=data, **pars).fit(method='cg', verbose=0)

    # By default, a likelihood that JAX does not cover (a function) is fitted without gradients, rather than raising
    dur = ss.par(ss.days(8), bounds=[3, 15])
    k = ss.par(0.5, bounds=[0.1, 1])
    infection = S >> I @ beta*I
    sir = ss.Disease(infection, I >> R @ dur, ss.outcome(k*infection, name='seen'), init=dict(I=0.01), name='sir')
    data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_synthetic_data('seen')
    normal = lambda observed, model: -0.5*((observed - model)/(0.1*model + 1))**2
    fit = ss.Sim(sir, data=data, **pars).fit(likelihood=dict(seen=normal), verbose=0)
    assert fit.method == 'nelder-mead' and np.isclose(fit.pars.beta, 0.5, rtol=0.05)
    return fit


@sc.timer()
def test_fit_keywords():
    sc.heading('Testing a method with its settings, likelihoods for every series or none, and sim.optimize()')
    beta = ss.par(0.3, bounds=[0.1, 1])
    S, I, R = ss.states(3)
    infection = S >> I @ beta*I
    sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
    pars = dict(total_pop=10_000, unit='day', stop=40, verbose=0)
    data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_synthetic_data(['cases', 'I'], resample='week', rand_seed=1)
    sim = ss.Sim(sir, data=data, **pars)
    fit = sim.fit(likelihood=ss.poisson(), method=ss.fit.method('nelder-mead', scan=2)) # One likelihood for every series
    assert fit.measure == 'negative log-likelihood' and np.isclose(fit.pars.beta, 0.5, rtol=0.05) and fit.objective < fit.objective_before
    with pytest.raises(ValueError, match='but I does not'): # Some series with a likelihood and some without (it silently minimized the loss)
        sim.fit(likelihood=dict(cases=ss.poisson()))
    with pytest.raises(ValueError, match='loss= is the measure of a fit without likelihoods'):
        sim.fit(likelihood=ss.poisson(), loss='rmse')
    with pytest.raises(TypeError, match='ss.fit.method'): # A setting of a method is not a keyword of sim.fit()
        sim.fit(**dict(method='optuna', n_trials=5)) # As a dict, so the rewrite script leaves the old spelling
    with pytest.raises(TypeError, match='takes draws=, warmup=, chains=, scan=, require_jax=, not particles='):
        ss.fit.method('nuts', particles=100)

    # Spending allocated to reach a goal
    S, I, R = ss.states(3)
    a, b = ss.par(50.0, name='spend_a'), ss.par(50.0, name='spend_b') # Program a cuts transmission twice as much as b
    cov = lambda spend: np.minimum(spend/0.01/(S + I + R).total, 1)
    sir = ss.Disease(S >> I @ 0.3*(1 - 0.5*cov(a) - 0.25*cov(b))*I.frac, I >> R @ ss.days(10), init_n=dict(I=10), name='sir')
    sim = ss.Sim(sir, total_pop=10_000, unit='day', stop=100, verbose=0)
    best = sim.optimize(ss.minimize('sir.R'), spend=['spend_a', 'spend_b'])
    assert best.value < best.value_before and best.alloc['spend_a'] > best.alloc['spend_b'] # The program that does more gets more
    with pytest.raises(TypeError, match='goal=ss.minimize'):
        sim.optimize(minimize='sir.R', spend=['spend_a', 'spend_b'])
    return fit


#%% Run as a script
if __name__ == '__main__':
    test_best_trial()
    test_fit_keywords()
