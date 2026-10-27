"""
Test calibration
"""

#%% Imports and settings
import io
import contextlib
import unittest.mock
import starsim as ss
import starsim.fit
import starsim.library as ssl
import numpy as np
import pandas as pd
import scipy.special as sps
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

    # A likelihood can be a function of the data and the model's values, as for one the core lacks; here the Poisson's own, which gives the same
    poisson = lambda observed, model: observed*np.log(np.maximum(model, 1e-9)) - np.maximum(model, 1e-9) - sps.gammaln(observed + 1)
    filt = ss.fit.ParticleFilter(sim, likelihood=dict(cases=poisson))
    assert np.isclose(filt(n_particles=100, seed=1), est[0.6]) # The sim was last restarted at 0.6
    ran = make(data=data).run()
    assert np.isclose(ss.fit.loglik(ran, likelihood=dict(cases=poisson)), ss.fit.loglik(ran))

    # Particle MCMC, for a stochastic compartmental sim whose data has a likelihood (asked for by name: the default is a best fit)
    assert ss.fit.default_method(sim, True)[0] == 'optuna'
    fit = make(beta=0.3, data=data).fit(method='pmcmc', draws=100, warmup=100, chains=2, particles=30, parallel=False, rand_seed=1)
    fit.summarize()
    with unittest.mock.patch('sciris.isjupyter', return_value=True), contextlib.redirect_stdout(io.StringIO()) as out: # In a notebook, the table is returned, which shows itself, and not printed too
        assert fit.summarize() is fit.table and 'beta' not in out.getvalue()
    fig = fit.plot() # Weekly data: the model's cases and the band over the same weeks, as the fit compares them (they were daily, seven times too low)
    lines = fig.axes[0].lines
    assert all(len(line.get_xdata()) == len(data) for line in lines) and 0.5 < np.nanmax(lines[-1].get_ydata())/data.cases.max() < 2
    table = fit.table.set_index('par')
    assert fit.method == 'pmcmc'
    assert len(fit.samples) == 200 and len(fit.diagnostics.acceptance) == 2
    assert table.lower.beta < 0.5 < table.upper.beta # The posterior covers the value that made the data
    with pytest.raises(ValueError, match='not every data series has a likelihood'): # A posterior needs a likelihood
        make(data=data).fit(method='pmcmc', gof='nae')

    # A best fit scores each point by the mean of several runs, with fresh seeds at each point under Optuna, and reports the score of the best values over the first seeds
    best = make(beta=0.3, data=data).fit(method='optuna', n_trials=8, rand_seed=1)
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
        sir.stratify(patch=ss.geo.geography(names=['a', 'b']), mobility=ss.geo.Mobility(matrix=[[0, 1], [1, 0]], away=ss.par(away, bounds=[0.001, 0.3], name='away')))
        return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-04-01', solver=solver, total_pop=20_000, data=data, verbose=0, rand_seed=3)

    # A restart with a new value works out the movement again, as a new sim does
    sim = make(solver='ode').init()
    n_I = lambda away: sim.restart(pars=dict(away=away)).run().results.sir.I.values.copy()
    assert np.allclose(n_I(0.01), make(away=0.01, solver='ode').run().results.sir.I.values)
    msim = ss.parallel(make(solver='ode'), pars=dict(away=[0.01, 0.1]))
    assert np.allclose(msim.results.sir.I.select(away=0.01).values.ravel(), n_I(0.01))

    # Its gradient is that of the runs
    h = 1e-5
    grad = sim.restart(pars=dict(away=0.1)).gradient(results='I', pars='away').away.sir.I
    assert np.allclose(grad, (n_I(0.1 + h) - n_I(0.1 - h))/(2*h), rtol=1e-3, atol=1e-2)

    # IF2 gives each particle its own movement, and finds the value that made the data
    truth = make(away=0.01, solver='ode').run().results.sir
    weekly = np.add.reduceat(truth.cases.values[1:], np.arange(0, len(truth.cases) - 1, 7))[:-1]
    data = pd.DataFrame(dict(cases=np.random.default_rng(1).poisson(weekly)), index=truth.timevec[7::7][:len(weekly)])
    fit = make(data=data).fit(method='if2', particles=100, iterations=10, chains=2, rand_seed=1, verbose=False)
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
    flows, stocks = res.S_to_I.resample('year'), res.I.resample('year', summarize_by='mean') # A flow summed over each year, a stock averaged over it
    df = pd.DataFrame(dict(year=flows.timevec, S_to_I=flows.values, I=stocks.values)).iloc[1:-1]
    fit = make(beta=0.5, data=df).fit(verbose=0)
    assert np.isclose(fit.pars.beta, 0.3, rtol=1e-3) # Before, 0.279, from data compared a year off
    assert len(fit.trials) < 60 # Before, 3 searches (131 evaluations)

    # A module with a timestep of its own, a setting of an intervention written as code, a bounded parameter passed through a package's Pars, and a column that a loss of its own does not read
    class cure(ss.Intervention):
        def __init__(self, p=0.1, **kwargs):
            super().__init__(**kwargs)
            self.p = p # A setting kept as an attribute, as Covasim's are
            self.p_retest = p # Held by two attributes, as Covasim's symp_quar_prob copies symp_prob: one parameter
            return
        def step(self):
            sis = self.sim.diseases.sis
            cured = sis.I.uids[np.random.default_rng(self.ti).random(sis.I.count()) < self.p]
            sis.I[cured] = False
            sis.S[cured] = True
            return
    defaults = sc.dcp(ss.RandomNet().pars) # As a package builds a module's parameters
    defaults.update(n_contacts=ss.par(4, bounds=[2, 10]))
    net = ss.RandomNet(pars=defaults)
    data = pd.DataFrame(dict(year=np.arange(2002, 2010), sis_prevalence=0.1, notes=1))
    sim = ss.Sim(diseases=ss.SIS(dt=ss.years(1)), networks=net, interventions=cure(p=ss.par(0.1, bounds=[0.01, 0.5])), n_agents=300, start=2000, stop=2010, data=data, verbose=0)
    sim.pars.diseases.pars.init_prev = 0.2 # Changed after the sim was built: the fit's runs use it too
    with pytest.raises(KeyError, match='the data column "notes" matches no result'): # Compared with the results, the column must match one
        sim.fit()
    loss = lambda sim: abs(sim.results.sis.prevalence[-1] - 0.1)
    path = tmp_path/'fit.log'
    fit = sim.fit(gof=loss, n_trials=6, n_workers=2, storage=path, verbose=0)
    assert list(fit.pars.keys()) == ['n_contacts', 'p'] and len(fit.trials) == 6
    assert fit.sim.diseases.sis.pars.init_prev.value == 0.2 and fit.sim.interventions.cure.p == fit.sim.interventions.cure.p_retest
    assert np.isfinite(ss.fit.mismatch(fit.sim, fit.sim.data[['sis_prevalence']])) # Before, a broadcast error
    again = sim.fit(gof=loss, n_trials=8, storage=path, verbose=0) # Resumed: two more trials
    assert len(again.trials) == 8 and np.allclose(again.trials.objective[:6], fit.trials.objective)
    return fit


def test_new_subclass_init():
    """ A sim subclass whose init() builds modules from its inputs (as STIsim's does): sim.new() makes it again from the inputs, not with those modules added """
    class MySim(ss.Sim):
        def init(self, **kwargs):
            if not self.pars.diseases:
                self.pars.diseases = [ss.SIS()] # Built by init(), as STIsim builds its diseases from their names
            return super().init(**kwargs)
    sim = MySim(n_agents=200, dur=10, verbose=0)
    sim.pars.analyzers = [ss.infection_log()] # Added after the sim was made, so kept
    sim.init()
    new = sim.new().init()
    assert list(new.diseases.keys()) == ['sis'] and list(new.analyzers.keys()) == ['infection_log']
    return new


@sc.timer()
def test_fit_what_is_not_a_parameter():
    sc.heading('Testing fitting a value set in code, values in a dict parameter, a modifier\'s amount, and data from an analyzer added after the sim was made')

    class clearance(ss.Intervention): # A module written as code with a dict parameter, as HPVsim's cin_fn
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(cure=dict(p=0.05, start=0))
            self.update_pars(**kwargs)
            return
        def step(self):
            sis = self.sim.diseases.sis
            p = self.pars.cure['p'] # Its code sees the value
            cured = sis.I.uids[np.random.default_rng(self.ti).random(sis.I.count()) < p]
            sis.I[cured] = False
            sis.S[cured] = True
            return

    class counter(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('infected', scale=True), ss.Result('by_age', columns=['young', 'old'], shape=(self.t.npts, 2), scale=True))
            return
        def step(self):
            sis, age = self.sim.diseases.sis, self.sim.people.age
            self.results.infected[self.ti] = sis.I.count()
            self.results.by_age[self.ti] = [(sis.I & (age < 30)).count(), (sis.I & (age >= 30)).count()]
            return

    def set_trans(sim, rel=ss.par(1.0, bounds=[0.5, 2])): # A value set in code, as Covasim's custom_fn: a keyword argument of a function given as a module
        sim.diseases.sis.rel_trans[:] = rel
        return

    data = pd.DataFrame({'year':np.arange(2002, 2010), 'counter.infected':60, 'counter.by_age.young':20}) # An analyzer's results, one of them a column
    cures = [clearance(name=name, cure=dict(p=ss.par(0.05, bounds=[0.01, 0.3]), start=0)) for name in ['c1', 'c2']]
    sim = ss.Sim(diseases=ss.SIS(), networks='random', interventions=cures + [set_trans], n_agents=300, start=2000, stop=2010, data=data, verbose=0)
    sim.pars.analyzers += counter() # Added after the sim was made: kept by sim.new(), so the fit's runs have it
    fit = sim.fit(method='optuna', n_trials=6, n_workers=2, verbose=0)
    assert set(fit.pars.keys()) == {'c1.cure.p', 'c2.cure.p', 'rel'} # Named with its module where the name alone is ambiguous
    assert fit.sim.interventions.c1.pars.cure['p'] == fit.pars['c1.cure.p'] and fit.sim.interventions.set_trans.pars.rel == fit.pars.rel
    assert len(fit.study.trials) == 6 # Optuna's study, kept in memory although its workers shared a file
    with pytest.raises(TypeError, match='n_trials='): # One name for the number of trials
        sim.fit(method='optuna', trials=6, verbose=0)
    problem = ss.fit.Problem(sim, reseed=False, verbose=0)
    assert problem.choose('optuna') == 'optuna' and not problem.fresh # The same seeds at every point, even under Optuna

    # A modifier's amount given as ss.par is among its parameters, so it is set by name and fitted whatever it changes, here a module written as code
    def make():
        hiv = ssl.HIV(beta=0.05, init_prev=0.1)
        return ss.Sim(diseases=hiv, networks=ss.SexualNet(), connectors=ss.modify(hiv.rel_sus, multiply=ss.par(1.0, bounds=[0, 2])), n_agents=1000, start=2000, stop=2010, verbose=0)
    assert list(ss.fit.bounded_pars(make().init())[0].keys()) == ['multiply']
    assert make().run().results.hiv.new_infections.values.sum() > 0
    assert make().new(pars=dict(multiply=0)).run().results.hiv.new_infections.values.sum() == 0
    return fit


@sc.timer()
def test_fit_arrays():
    sc.heading('Testing fitting an array, with an element fixed, and a penalty on the values')
    target = np.array([0.2, 0.5])
    def knots(sim, curve=ss.par(np.array([0.3, 0.3, 0.9]), bounds=[np.array([0, 0, 0.9]), np.array([1, 1, 0.9])])): # The last element fixed: its bounds are equal
        return
    sim = ss.Sim(n_agents=100, start=2000, stop=2002, interventions=knots, verbose=0)
    gof = lambda sim: np.sum((sim.interventions.knots.pars.curve[:2] - target)**2)
    fit = sim.fit(gof=gof, method='nelder-mead', verbose=0)
    assert list(fit.table.par) == ['curve[0]', 'curve[1]']
    assert np.allclose(fit.pars.curve, [0.2, 0.5, 0.9], atol=1e-3)
    smooth = sim.fit(gof=gof, penalty=lambda pars: 10*np.diff(pars['curve'][:2])[0]**2, method='nelder-mead', verbose=0)
    assert np.isclose(np.diff(smooth.pars.curve[:2])[0], 0.3/21, atol=2e-3) # The minimum of (a - 0.2)^2 + (b - 0.5)^2 + 10(b - a)^2
    assert np.isclose(smooth.mismatch, 2*(0.15 - 0.15/21)**2 + 10*(0.3/21)**2, atol=1e-4) # The penalty is part of what is minimized
    return fit


@sc.timer()
def test_fit_fresh_seeds():
    sc.heading('Testing that a fit with fresh seeds at each point returns a value among its best trials')
    def make(init_prev=ss.par(0.005, bounds=[0.001, 0.1]), data=None, rand_seed=1): # No transmission, so the number infected is the infections at the start (as Typhoidsim's test)
        sis = ss.SIS(beta=0, dur_inf=ss.years(100), init_prev=init_prev)
        return ss.Sim(diseases=sis, n_agents=2000, start=2000, stop=2002, dt=ss.months(1), data=data, rand_seed=rand_seed, verbose=0)
    data = make(init_prev=0.02, rand_seed=2).run().to_data('sis.I')
    fit = make(data=data).fit(method='optuna', n_trials=30, n_runs=4, verbose=0)
    top = fit.trials.nsmallest(5, 'objective').init_prev
    assert top.min() <= fit.pars.init_prev <= top.max(), f'{fit.pars.init_prev:n} is outside the best trials ({top.min():n} to {top.max():n})' # Before, a dense region of worse trials could pull it away
    assert np.isclose(fit.pars.init_prev, 0.02, rtol=0.25)
    return fit


@sc.timer()
def test_fit_sim_setting():
    sc.heading('Testing fitting a sim setting given as ss.par (n_agents), which builds a new sim for each value')
    S, I = ss.states('S I')
    make = lambda n_agents, data=None: ss.Sim(ss.Disease(S >> I @ 0.2*I, I >> S @ ss.days(20), init_prev=0.1, name='sis'), networks='random', n_agents=n_agents, start=0, stop=60, unit='day', solver='abm', data=data, verbose=0)
    data = make(2000).run().to_data('sis.I') # With no total_pop, the counts are of agents, so they tell the number apart
    fit = make(ss.par(1000, bounds=[500, 4000]), data=data).fit(verbose=0)
    assert np.isclose(fit.pars.n_agents, 2000, rtol=0.1), fit.pars
    assert len(fit.sim.new(pars=fit.pars).init().people) == int(fit.pars.n_agents) # pars= takes the setting by name
    with pytest.raises(sc.KeyNotFoundError, match='setting of the sim'):
        make(2000).new(pars=dict(n_agents=3000)).init() # Not given as ss.par, so not a value by name
    return fit


#%% Run as a script
if __name__ == '__main__':
    test_particle_mcmc()
    test_fitted_mobility()
    test_fit_what_is_not_a_parameter()
    test_fit_arrays()
    test_fit_fresh_seeds()
    test_fit_sim_setting()
