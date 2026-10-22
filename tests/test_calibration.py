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
    fit = make(beta=0.3, data=data).calibrate(method='pmcmc', draws=100, warmup=100, chains=2, particles=30, parallel=False, rand_seed=1)
    fit.summarize()
    table = fit.table.set_index('par')
    assert fit.method == 'pmcmc'
    assert len(fit.samples) == 200 and len(fit.diagnostics.acceptance) == 2
    assert table.lower.beta < 0.5 < table.upper.beta # The posterior covers the value that made the data
    with pytest.raises(ss.errors.NoPosterior): # A posterior needs a likelihood
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
        sir = ss.Disease(infection, I >> R @ ss.days(6), cases, init_n=dict(I=ss.by(patch=[20, 0])), name='sir')
        sir.stratify(patch=ss.geo.location(names=['a', 'b']), mobility=ss.geo.Mobility(matrix=[[0, 1], [1, 0]], away=ss.par(away, bounds=[0.001, 0.3], name='away')))
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


@sc.timer()
def test_fit_inputs(tmp_path):
    sc.heading('Testing data by calendar year, results on their own timeline, settings of code, and workers that resume')

    # Data by calendar year (an integer year column) is the year each row names, not 1 January of it; one parameter of a deterministic sim takes one search
    def make(beta=0.3, data=None):
        beta = ss.par(beta, bounds=[0.05, 1], name='beta')
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ beta*I, I >> R @ ss.years(2), ss.Transition(R, S, dur=ss.years(3)), init_prev=0.01, name='sir')
        return ss.Sim(sir, start=2000, stop=2020, dt=ss.years(0.05), data=data, verbose=0)
    res = make().run().results.sir
    flows, stocks = res.S_to_I.resample('year'), res.n_I.resample('year', kind='ratio') # A flow summed over each year, a stock averaged over it
    df = pd.DataFrame(dict(year=flows.timevec, S_to_I=flows.values, n_I=stocks.values)).iloc[1:-1]
    fit = make(beta=0.5, data=df).calibrate(verbose=0)
    assert np.isclose(fit.pars.beta, 0.3, rtol=1e-3) # Before, 0.279, from data compared a year off
    assert len(fit.trials) < 60 # Before, 3 searches (131 evaluations)

    # A module with a timestep of its own, a setting of an intervention written as code, a bounded parameter passed through a package's Pars, and a column that a loss of its own does not read
    class cure(ss.Intervention):
        def __init__(self, p=0.1, **kwargs):
            super().__init__(**kwargs)
            self.p = p # A setting kept as an attribute, as Covasim's are
            return
        def step(self):
            sis = self.sim.diseases.sis
            cured = sis.I.uids[np.random.default_rng(self.ti).random(sis.I.count()) < self.p]
            sis.I[cured] = False
            sis.S[cured] = True
            return
    defaults = sc.dcp(ss.RandomNet().pars) # As a package builds a module's parameters
    defaults.update(n_contacts=ss.par(4, bounds=[2, 10]))
    net = ss.RandomNet()
    net.pars.update(defaults)
    data = pd.DataFrame(dict(year=np.arange(2002, 2010), sis_prevalence=0.1, notes=1))
    sim = ss.Sim(diseases=ss.SIS(dt=ss.years(1)), networks=net, interventions=cure(p=ss.par(0.1, bounds=[0.01, 0.5])), n_agents=300, start=2000, stop=2010, data=data, verbose=0)
    with pytest.raises(ss.errors.UnknownDataColumn): # Compared with the results, the column must match one
        sim.calibrate()
    loss = lambda sim: abs(sim.results.sis.prevalence[-1] - 0.1)
    path = tmp_path/'fit.log'
    fit = sim.calibrate(gof=loss, trials=6, n_workers=2, storage=path, verbose=0)
    assert list(fit.pars.keys()) == ['n_contacts', 'p'] and len(fit.trials) == 6
    assert np.isfinite(ss.fit.mismatch(fit.sim, fit.sim.data[['sis_prevalence']])) # Before, a broadcast error
    again = sim.calibrate(gof=loss, trials=8, storage=path, verbose=0) # Resumed: two more trials
    assert len(again.trials) == 8 and np.allclose(again.trials.objective[:6], fit.trials.objective)
    return fit

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
    test_particle_mcmc()
    test_fitted_mobility()
    test_strain_data()
