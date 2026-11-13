"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import starsim.internals.fitting
import numpy as np
import sciris as sc
import scipy.optimize as spo
import scipy.special as spsp
import scipy.stats as sps
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
        fit = ss.Sim(sir, data=data, **pars).fit(method=ss.fitting.method('nelder-mead', scan=0), verbose=0)
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
    fit = sim.fit(likelihood=ss.poisson(), method=ss.fitting.method('nelder-mead', scan=2)) # One likelihood for every series
    assert fit.measure == 'negative log-likelihood' and np.isclose(fit.pars.beta, 0.5, rtol=0.05) and fit.objective < fit.objective_before
    with pytest.raises(ValueError, match='but I does not'): # Some series with a likelihood and some without (it silently minimized the loss)
        sim.fit(likelihood=dict(cases=ss.poisson()))
    with pytest.raises(ValueError, match='loss= is the measure of a fit without likelihoods'):
        sim.fit(likelihood=ss.poisson(), loss='rmse')
    declared = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases', likelihood=ss.poisson()), init=dict(I=0.01), name='sir')
    dsim = ss.Sim(declared, data=data.drop(columns='I'), **pars)
    quick = ss.fitting.method('nelder-mead', scan=2, maxiter=20)
    assert dsim.fit(method=quick).measure == 'negative log-likelihood' # The declared likelihood
    assert dsim.fit(loss='rmse', method=quick).measure == 'loss' == dsim.fit(likelihood=False, method=quick).measure # loss= in its place (with a note), or none
    with pytest.raises(ValueError, match='likelihood=False'): # Some series with a likelihood and some without
        ss.Sim(declared, data=data, **pars).fit(method=quick)
    with pytest.raises(TypeError, match='ss.fitting.method'): # A setting of a method is not a keyword of sim.fit()
        sim.fit(**dict(method='optuna', n_trials=5)) # As a dict, so the rewrite script leaves the old spelling
    with pytest.raises(TypeError, match='takes draws=, warmup=, chains=, scan=, ess=, rhat=, max_draws=, batch=, require_jax=, not particles='):
        ss.fitting.method('nuts', particles=100)

    # Spending allocated to reach a goal
    S, I, R = ss.states(3)
    a, b = ss.par(50.0, name='spend_a'), ss.par(50.0, name='spend_b') # Program a cuts transmission twice as much as b
    cov = lambda spend: np.minimum(spend/0.01/(S + I + R).total, 1)
    sir = ss.Disease(S >> I @ 0.3*(1 - 0.5*cov(a) - 0.25*cov(b))*I.frac, I >> R @ ss.days(10), init_n=dict(I=10), name='sir')
    sim = ss.Sim(sir, total_pop=10_000, unit='day', stop=100, verbose=0)
    best = sim.optimize(ss.fitting.minimize('sir.R'), spend=['spend_a', 'spend_b'])
    assert best.value < best.value_before and best.alloc['spend_a'] > best.alloc['spend_b'] # The program that does more gets more
    with pytest.raises(TypeError, match='goal=ss.fitting.minimize'):
        sim.optimize(minimize='sir.R', spend=['spend_a', 'spend_b'])
    return fit


@sc.timer()
def test_posterior_choices():
    sc.heading('Testing prior draws truncated to the bounds, ABC-SMC (refused for a deterministic sim, and with no claim of convergence), and the default fit of an ODE without JAX')
    # Prior draws: a normal prior truncated to the bounds, as the samplers use it, not clipped to them (which put 31% of the draws at the bound)
    beta = ss.par(0.3, bounds=[0.1, 0.5], prior=ss.normal(0.45, 0.1), name='beta')
    S, I, R = ss.states(3)
    sim = ss.Sim(ss.Disease(S >> I @ beta*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir'), unit='day', stop=30, verbose=0)
    draws = np.array(ss.fitting.prior_draws(sim, 2000, rand_seed=1)['beta'])
    truncated = sps.truncnorm((0.1 - 0.45)/0.1, (0.5 - 0.45)/0.1, loc=0.45, scale=0.1)
    assert 0.1 < draws.min() and draws.max() < 0.5 and abs(draws.mean() - truncated.mean()) < 3*truncated.std()/np.sqrt(len(draws))

    # ABC-SMC: refused for a deterministic sim, whose width it would set by the number of generations; for a stochastic one, it reports its tolerance rather than convergence
    beta = ss.par(0.3, bounds=[0.1, 1])
    infection = S >> I @ beta*I
    sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
    pars = dict(total_pop=2000, unit='day', stop=40, verbose=0)
    data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_synthetic_data('cases', resample='week', rand_seed=1)
    abc = ss.fitting.method('abc-smc', samples=20, generations=2)
    with pytest.raises(ss.CapabilityError, match='method="pmcmc"'):
        ss.Sim(sir, data=data, **pars).fit(method=abc, likelihood=ss.poisson())
    with pytest.raises(ss.CapabilityError, match='needs a likelihood'):
        ss.Sim(sir, data=data, **pars).fit(method=abc)
    fit = ss.Sim(sir, data=data, solver='binomial', **pars).fit(method=abc)
    diag = fit.diagnostics
    assert fit.converged is None and 'approximate posterior' in fit.why and np.isfinite(diag.tolerance) and 1 <= diag.generations <= 2 and 0 < diag.acceptance[0] <= 1

    # An ODE that JAX can't run (here, for a module written as code): L-BFGS-B with SciPy's finite differences for three parameters, and particle MCMC for a posterior
    dur, k = ss.par(ss.days(8), bounds=[3, 15]), ss.par(0.5, bounds=[0.1, 1])
    sir = ss.Disease(infection, I >> R @ dur, ss.outcome(k*infection, name='cases', likelihood=ss.poisson()), init=dict(I=0.01), name='sir')
    pars = dict(total_pop=10_000, unit='day', stop=60, verbose=0, interventions=lambda sim: None)
    data = ss.Sim(sir, pars=dict(beta=0.5, k=0.3), **pars).run().to_synthetic_data('cases', resample='week', rand_seed=1)
    sim = ss.Sim(sir, data=data, **pars)
    method, why = ss.fitting.default_method(sim.init(), use_lik=True, n_pars=3)
    assert method == 'l-bfgs-b' and 'finite differences' in why and 'method="pmcmc"' in why
    assert ss.fitting.default_method(sim, use_lik=True, n_pars=2)[0] == 'nelder-mead'
    assert ss.fitting.Problem(sim, verbose=0).choose() == 'l-bfgs-b' # Not refused for the gradients' "Python in the loop"
    fit = ss.Sim(sir, data=data, **pars).fit(method=ss.fitting.method('l-bfgs-b', scan=0, maxiter=5), verbose=0) # A few steps, by SciPy's finite differences
    assert fit.objective < fit.objective_before
    return fit


@sc.timer()
def test_pmcmc():
    sc.heading('Testing particle MCMC: on an ODE (the exact likelihood), chains from different starts, and projections that continue the runs kept with the draws')
    beta = ss.par(0.3, bounds=[0.1, 1])
    S, I, R = ss.states(3)
    infection = S >> I @ beta*I
    sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases', likelihood=ss.poisson()), init=dict(I=0.001), name='sir')
    pars = dict(total_pop=2000, unit='day', stop=80, verbose=0)
    data = ss.Sim(sir, pars=dict(beta=0.4), **pars).run().to_synthetic_data('cases', resample='week', rand_seed=1)
    sim = ss.Sim(sir, data=data, **pars)

    # On an ODE, the same random walk on the exact likelihood, with chains from different points of the scan
    grid = np.linspace(0.36, 0.44, 101)
    runner = sim.new().init()
    loglik = np.array([ss.fitting.loglik(runner.restart(pars=dict(beta=b)).run()) for b in grid])
    cdf = np.cumsum(np.exp(loglik - loglik.max()))
    lo, hi = np.interp([0.025, 0.975], cdf/cdf[-1], grid) # The exact 95% interval
    with mock.patch.object(ss.fitting, 'pmmh', wraps=ss.fitting.pmmh) as called:
        fit = sim.fit(method=ss.fitting.method('pmcmc', draws=300, warmup=300, chains=3, parallel=False))
    z0 = called.call_args.args[1]
    assert len(np.unique(z0.round(6), axis=0)) == 3 # Each chain from its own start
    t = fit.table
    assert abs(t.lower[0] - lo) < 0.003 and abs(t.upper[0] - hi) < 0.003, (t.lower[0], t.upper[0], lo, hi)
    assert fit.trajectories is None # An ODE's run at each draw is its trajectory

    # With draws='auto' (the default), the chains are taken further in batches until they agree, with enough effective draws, and no further
    auto = sim.fit(method=ss.fitting.method('pmcmc', warmup=300, chains=3, parallel=False, batch=100), verbose=False)
    n = len(auto.samples)//3
    assert n % 100 == 0 and n < 20_000, n
    assert auto.table.rhat.max() <= 1.01 and auto.table.ess.min() >= 400, auto.table
    draws = np.stack([auto.samples[auto.samples.chain == c][['beta']].to_numpy() for c in range(3)])
    rhat, ess = ss.fitting.diagnose(draws[:, :n - 100]) # One batch fewer was not yet enough
    assert not (np.all(rhat <= 1.01) and np.all(ess >= 400)), (rhat, ess)
    with pytest.raises(ValueError, match='its likelihood is exact'):
        sim.fit(method=ss.fitting.method('pmcmc', particles=100))

    # A stochastic sim: a run drawn from the filter with each draw, which a projection continues
    sim = ss.Sim(sir, data=data.iloc[:6], solver='binomial', **pars) # Data to day 42
    fit = sim.fit(method=ss.fitting.method('pmcmc', draws=40, warmup=40, chains=2, particles=50, parallel=False))
    assert len(fit.trajectories) == len(fit.samples)
    path = fit.trajectories[-1].sir
    assert path.ti == 42 and path.stocks.sir.shape == (42, 3)
    weekly = 0.5*path.flows.sir[:, 0].reshape(6, 7).sum(axis=1) # The cases expected in the run, which follows the data
    assert np.all(np.abs(weekly - data.cases.values[:6]) < 5*np.sqrt(data.cases.values[:6]) + 5)
    proj = fit.project(n_runs=5, rand_seed=1, stop=100)
    rows = fit.samples.sample(n=5, random_state=1).index # The draws projected
    for k,i in enumerate(rows):
        assert np.array_equal(proj.results.sir.I.values[k, :42], fit.trajectories[i].sir.stocks.sir[:, 1]) # Up to the last data point, the run kept
    assert len(set(proj.results.sir.I.values[:, 41])) > 1 and proj.results.sir.I.values.shape == (5, 101)
    proj = fit.project(n_runs=5, rand_seed=1, stop=100, restart=True) # Each draw from the start instead
    assert proj.results.sir.I.values.shape == (5, 101) and proj.sims[0].results_ready
    vaccination = ss.vaccinate(efficacy=0.9, start=50, capacity=ss.perday(20)) # An intervention the kernels don't take, so the runs can't be continued
    with pytest.warns(RuntimeWarning, match='from the start'):
        proj = fit.project(dict(vx=vaccination), n_runs=3, rand_seed=1, stop=100)
    assert proj.sims[0].results_ready
    with pytest.raises(ValueError, match='restart=False'):
        fit.project(dict(vx=vaccination), n_runs=3, rand_seed=1, stop=100, restart=False)
    with pytest.warns(RuntimeWarning, match='stop before it'): # Ending before the runs kept do: from the start
        fit.project(n_runs=3, rand_seed=1, stop=30)
    abc = sim.fit(method=ss.fitting.method('abc-smc', samples=20, generations=2))
    assert all(path.sir.ti == 42 and len(path.sir.stocks.sir) == 42 for path in abc.trajectories) # Kept to the last data point, as particle MCMC's, so scenarios differ after the data
    vsim = ss.Sim(sir, data=data.iloc[:6], solver='binomial', interventions=ss.vaccinate(ss.Vaccine(efficacy=1.0), when=[10], coverage=0.5), **pars) # A fit with a product's effect, and a delivery before the last data point
    vfit = vsim.fit(method=ss.fitting.method('pmcmc', draws=10, warmup=10, chains=2, particles=20, parallel=False))
    with pytest.warns(RuntimeWarning, match='due before'): # A continued run would leave the delivery out
        vfit.project(n_runs=2, stop=100)
    return fit


@sc.timer()
def test_stochastic_scoring():
    sc.heading('Testing how a stochastic sim is scored: by the particle filter in Optuna\'s search, by the mean of runs (labeled so) otherwise, and by the mean likelihood of runs in compare_models()')
    def make(solver='binomial', data=None):
        beta = ss.par(0.3, bounds=[0.1, 1], name='beta')
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ beta*I, I >> R @ ss.days(6), init_n=dict(I=5), name='sir')
        return ss.Sim(sir, data=data, total_pop=300, unit='day', stop=56, solver=solver, rand_seed=1, verbose=0)
    data = make().new(pars=dict(beta=0.35), rand_seed=7).run().to_synthetic_data('infection', resample='week')
    lik = dict(infection=ss.nbinom(n=5))

    # Optuna scores each point by the filter's estimate of the log-likelihood, with 100 trials for one parameter
    fit = make(data=data).fit(likelihood=lik)
    assert fit.method == 'optuna' and fit.measure == 'negative log-likelihood (estimated by a particle filter of 200 particles)'
    assert len(fit.scores) == 1 and fit.objective == fit.scores[0] and len(fit.study.trials) == 100
    runs = make(data=data).fit(likelihood=lik, method=ss.fitting.method('optuna', n_trials=5, n_runs=2)) # With n_runs=, by the mean of the runs' log-likelihoods, which is not the likelihood
    assert runs.measure == 'negative mean log-likelihood of 2 runs' and len(runs.scores) == 2
    assert make(data=data).fit(method=ss.fitting.method('optuna', n_trials=5)).measure == 'mean loss of 4 runs' # No likelihood, so no filter
    with pytest.raises(ValueError, match='not both'):
        make(data=data).fit(likelihood=lik, method=ss.fitting.method('optuna', particles=50, n_runs=2))
    with pytest.raises(ValueError, match='particle filter, which cannot run this sim'):
        make(data=data).fit(method=ss.fitting.method('optuna', particles=50))
    assert ss.fitting.method('optuna', particles=50).scoring == dict(particles=50) and ss.fitting.method('pmcmc', particles=50).options == dict(particles=50)
    filt = lambda sim: ss.fitting.ParticleFilter(sim, likelihood=lik)(n_particles=50, rand_seed=1)
    assert filt(make(data=data).init()) == filt(make(data=data).run()) # A sim already run is filtered from its start, not from where it ended

    # compare_models() scores a stochastic model by the log of its mean likelihood over runs at the best-fit values
    sims = dict(ode=make('ode', data=data), binomial=make(data=data))
    comp = ss.fitting.compare_models(sims, holdout=0.3, likelihood=lik, n_runs=10, method=ss.fitting.method('optuna', n_trials=10))
    train, test = comp.splits[0]
    best = comp.fits.binomial.sim
    lls = [ss.fitting.loglik(best.new(rand_seed=1 + i).run() if i else best, data=test, likelihood=lik) for i in range(10)]
    assert np.isclose(comp.table.prediction_error.binomial, -(spsp.logsumexp(lls) - np.log(10))) and 'over 10 runs' in comp.measure
    assert np.isclose(comp.table.prediction_error.ode, -ss.fitting.loglik(comp.fits.ode.sim, data=test, likelihood=lik)) # A deterministic one is run once
    return fit


#%% Run as a script
if __name__ == '__main__':
    test_best_trial()
    test_fit_keywords()
    test_posterior_choices()
    test_pmcmc()
    test_stochastic_scoring()
