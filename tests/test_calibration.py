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

do_plot = True


#%% Define the tests

@sc.timer()
def test_particle_mcmc():
    sc.heading('Testing particle MCMC')
    def make(beta=0.5, data=None):
        beta = ss.par(beta, bounds=[0.1, 1], name='beta')
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        cases = ss.outcome(0.5 * infection, delay=ss.days(3), likelihood=ss.poisson())
        sir = ss.Disease(infection, I >> R @ ss.days(6), cases, init=dict(I=0.002), name='sir')
        return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-04-01', solver='binomial', total_pop=10_000, data=data, verbose=0, rand_seed=3)

    truth = make().run().results.sir
    weekly = np.add.reduceat(truth.cases.values[1:], np.arange(0, len(truth.cases) - 1, 7))[:-1] # A flow is compared with its total since the last data point
    data = pd.DataFrame(dict(cases=np.random.default_rng(1).poisson(weekly)), index=truth.timevec[7::7][:len(weekly)])

    # The particle filter's estimate of the likelihood is highest at the value that made the data
    sim = make(data=data).init()
    filt = ss.fit.ParticleFilter(sim)
    assert filt.series[0].spans[-1][0] > 70 # The last weekly point is worked out from its own week, not from the start
    est = {}
    for beta in [0.4, 0.5, 0.6]:
        sim.restart(pars=dict(beta=beta))
        est[beta] = filt(n_particles=100, seed=1)
    assert est[0.5] > max(est[0.4], est[0.6])

    # Particle MCMC, for a stochastic compartmental sim whose data has a likelihood (asked for by name: the default is a best fit)
    assert ss.fit.default_method(sim, True)[0] == 'optuna'
    fit = make(beta=0.3, data=data).calibrate(method='pmcmc', draws=150, warmup=150, chains=2, particles=50, parallel=False, rand_seed=1)
    fit.summarize()
    table = fit.table.set_index('par')
    assert fit.method == 'pmcmc'
    assert len(fit.samples) == 300 and len(fit.diagnostics.acceptance) == 2
    assert table.lower.beta < 0.5 < table.upper.beta # The posterior covers the value that made the data
    with pytest.raises(ss.NoPosterior): # A posterior needs a likelihood
        make(data=data).calibrate(method='pmcmc', gof='nae')

    # A best fit scores each point by the mean of several runs, with fresh seeds at each point under Optuna, and reports the score of the best values over the first seeds
    best = make(beta=0.3, data=data).calibrate(method='optuna', trials=8, rand_seed=1)
    assert len(best.scores) == 4 # The default for a stochastic compartmental sim
    assert ss.fit.Problem(make(data=data).new(solver='abm', n_agents=2000)).n_runs == 1 # But one for agents, whose runs are slow (Optuna's search learns from nearby points)
    assert np.isclose(best.mismatch, np.mean(best.scores))
    assert best.trials.objective.nunique() == len(best.trials) # Fresh seeds, so no two trials share their chance
    return fit


@sc.timer()
def test_fitted_mobility():
    sc.heading('Testing a fitted share of time away from home: restarted, run over values, differentiated, and fitted by IF2')
    def make(away=0.1, data=None, solver='binomial'):
        S, I, R = ss.states(3)
        infection = S >> I @ 0.4 * I
        cases = ss.outcome(infection, likelihood=ss.poisson())
        sir = ss.Disease(infection, I >> R @ ss.days(6), cases, init=dict(I=ss.by(patch=[20, 0])), name='sir')
        sir.stratify(patch=ss.geo(names=['a', 'b']), mobility=ss.Mobility(matrix=[[0, 1], [1, 0]], away=ss.par(away, bounds=[0.001, 0.3], name='away')))
        return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-04-01', solver=solver, total_pop=20_000, data=data, verbose=0, rand_seed=3)

    # A restart with a new value works out the movement again, as a new sim does
    sim = make(solver='ode').init()
    n_I = lambda away: sim.restart(pars=dict(away=away)).run().results.sir.n_I.values.copy()
    assert np.allclose(n_I(0.01), make(away=0.01, solver='ode').run().results.sir.n_I.values)
    msim = ss.parallel(make(solver='ode'), pars=dict(away=[0.01, 0.1]))
    assert np.allclose(msim.results.sir.n_I.select(away=0.01).values.ravel(), n_I(0.01))

    # Its gradient is that of the runs
    h = 1e-5
    grad = sim.restart(pars=dict(away=0.1)).gradient(results='n_I', pars='away').away.sir.n_I
    assert np.allclose(grad, (n_I(0.1 + h) - n_I(0.1 - h))/(2*h), rtol=1e-3, atol=1e-2)

    # IF2 gives each particle its own movement, and finds the value that made the data
    truth = make(away=0.01, solver='ode').run().results.sir
    weekly = np.add.reduceat(truth.cases.values[1:], np.arange(0, len(truth.cases) - 1, 7))[:-1]
    data = pd.DataFrame(dict(cases=np.random.default_rng(1).poisson(weekly)), index=truth.timevec[7::7][:len(weekly)])
    fit = make(data=data).calibrate(method='if2', particles=100, iterations=10, chains=2, rand_seed=1, verbose=False)
    assert 0.005 < fit.table.set_index('par').best.away < 0.02, fit.table
    return fit


#%% Run as a script
if __name__ == '__main__':
    do_plot = True
    test_particle_mcmc()
    test_fitted_mobility()
