"""
Contract tests for inference: the likelihoods that the fitting methods use match the
likelihood of a run, and each method recovers known values from simulated data
"""
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.internals.fitting
import corpus

start, stop = '2020-02-23', '2020-05-31' # As corpus.seasonal()
weeks = pd.date_range('2020-03-01', periods=13, freq='W') # The data of the seasonal models: weekly totals
outcome_weeks = pd.date_range('2020-01-08', '2020-05-27', freq='W') # The data of corpus.outcomes()


# %% Helper functions

def make_seasonal(b=0.5, init=100, data=None, **kwargs):
    """ corpus.seasonal(), with the number infectious at the start, so that a large population can start with enough of them for a stochastic run to be close to the ODE """
    dis = corpus.make_seasonal(b, dates=pd.date_range(start, stop), init=init)
    return corpus.make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


def make_detection(d=0.2, init=100, data=None, **kwargs):
    """ The summer2 pattern: case detection as a fitted value times a time series (a scale-up), in a rate that is not a transmission """
    dates = pd.date_range(start, stop)
    scaleup = ss.timeseries(dict(zip(dates.strftime('%Y-%m-%d'), np.linspace(0.2, 1.0, len(dates)))))
    d = ss.par(d, bounds=[0.05, 1])
    S, I, T, R = ss.states('S I T R')
    dis = ss.Disease(infection=S >> I @ 0.5*I, recovery=I >> R @ 0.25, detection=I >> T @ d*scaleup, treatment=T >> R @ 0.5, init_n=dict(I=init), name='sir')
    return corpus.make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


def make_inside(x, data=None, **kwargs):
    """ Fitted values inside values that vary in time: the mean and amplitude of a seasonal curve (which the rate is not linear in), and the values of a time series """
    b, a, g = [ss.par(v, bounds=[0.05, 1]) for v in x]
    S, I, R = ss.states('S I R')
    infection = S >> I @ ss.seasonal(mean=b, amplitude=a, peak=14/365)*I
    recovery = I >> R @ ss.timeseries({start:g, stop:0.5*g + 0.1}, interpolation='linear')
    dis = ss.Disease(infection, recovery, init_n=dict(I=100), name='sir')
    return corpus.make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


# Each model with data: how to make it (with the fitted value first), the data series, its dates, and the fitted value that makes the data
models = sc.obj(
    outcomes = sc.obj(build=lambda x, **kw: corpus.outcomes(beta=x, **kw), col='cases',     dates=outcome_weeks, truth=0.3),
    seasonal = sc.obj(build=lambda x, **kw: make_seasonal(x, **kw),        col='infection', dates=weeks,         truth=0.5),
    detect   = sc.obj(build=lambda x, **kw: make_detection(x, **kw),       col='detection', dates=weeks,         truth=0.2),
    inside   = sc.obj(build=lambda x, **kw: make_inside(x, **kw),          col='infection', dates=weeks,         truth=np.array([0.5, 0.3, 0.25])),
)


def simulate(model, x=None, seed=1, col=None, **kwargs):
    """ Data from a run of a model at known values: Poisson draws around the run's total over each week (as the likelihood compares them) """
    x = sc.ifelse(x, model.truth)
    col = sc.ifelse(col, model.col)
    probe = model.build(x, data=pd.DataFrame({col:0.0}, index=model.dates), **kwargs).run()
    expected = ss.results.compare(probe)[col][1]
    return pd.DataFrame({col:np.random.default_rng(seed).poisson(expected)}, index=model.dates)


def fitted_pars(sim):
    """ The parameters that a fit would fit: those with bounds """
    return {name:p for mod in sim.declared for name,p in mod.graph.pars.items() if p.bounds is not None}


def jax_loglik(sim, points, likelihood=None):
    """ The log-likelihood at each point from the log-density in JAX (less the priors), and from a run of the sim """
    jax = ss.gradients.jax()
    sim.init()
    fitted = fitted_pars(sim)
    names = list(fitted)
    x0 = np.array([p.raw for p in fitted.values()], dtype=float)
    set_pars = lambda x: sim.restart(pars=dict(zip(names, x)))
    density = ss.gradients.Density(sim, fitted, x0, set_pars, likelihood=likelihood)
    out = []
    for x in points:
        x = np.atleast_1d(np.array(x, dtype=float))
        prior = sum(ss.fitting.logprior(p, val) for p,val in zip(fitted.values(), x))
        with jax.enable_x64(True):
            from_jax = float(density(jax.numpy.asarray(x))) - prior
        from_run = ss.fitting.loglik(set_pars(x).run(), likelihood=likelihood)
        out.append((float(x[0]), from_jax, from_run))
    return out


# %% The likelihoods match the run

@pytest.mark.parametrize('key', ['outcomes', 'seasonal', 'detect', 'poisson_weekly', 'inside', 'substeps'])
def test_jax_loglik(key):
    """ The log-density in JAX equals the sim's log-likelihood, also where a step needs more substeps than at the initial values """
    if key == 'substeps':
        sim = corpus.outcomes(beta=0.06, likelihood=ss.poisson(), data=simulate(models.outcomes, 0.35)) # Built where a step needs 1 substep, evaluated where it needs several
        likelihood = None
        points = [1.0]
    elif key == 'poisson_weekly':
        sim = corpus.poisson_weekly() # Its data, and the likelihood declared on its outcome, are its own
        likelihood = None
        points = [0.25, 0.3, 0.35]
    else:
        model = models[key]
        sim = model.build(model.truth, data=simulate(model))
        likelihood = {model.col:ss.poisson()}
        points = [0.8*model.truth, model.truth, 1.2*model.truth] # Around the values that made the data, where a fit spends its time
    for x,from_jax,from_run in jax_loglik(sim, points, likelihood=likelihood):
        assert np.isfinite(from_run) and np.isclose(from_jax, from_run, rtol=1e-8, atol=1e-8), f'{key} at {x}: JAX {from_jax} against the sim\'s {from_run}' # The same sums in double precision
    return


@pytest.mark.parametrize('key', ['outcomes', 'detect'])
def test_particle_filter(key):
    """ The particle filter against direct runs and the ODE, at a large population """
    model = models[key]
    pop = 1e8 # Large, so that a stochastic run is close to the ODE
    init = None if key == 'outcomes' else dict(init=pop*1e-4) # Enough infectious at the start that chance early on does not shift the epidemic (corpus.outcomes() starts with 1%)
    kw = sc.mergedicts(dict(total_pop=pop), init)
    lik = {model.col:ss.nbinom(n=20)} # Wider than a Poisson, so that the noise in the data, not the chance in a run, dominates, and the likelihood has an ODE limit
    data = simulate(model, **kw)
    sim = model.build(model.truth, data=data, solver='tau', **kw).init()
    filt = ss.fitting.ParticleFilter(sim, likelihood=lik)
    with ss.options.context(threads=1):
        one = filt(n_particles=20, rand_seed=0)
    assert filt(n_particles=20, rand_seed=0) == one # Each particle has its own random numbers, so the estimate does not depend on the threads
    name = list(fitted_pars(sim))[0]
    values = [0.8*model.truth, model.truth, 1.2*model.truth]
    n_seeds = 3
    pf, direct = {}, {}
    for x in values:
        sim.restart(pars={name:x})
        pf[x] = np.array([filt(n_particles=50, rand_seed=seed) for seed in range(n_seeds)])
        direct[x] = np.array([ss.fitting.loglik(model.build(x, data=data, solver='tau', rand_seed=seed, **kw).run(), likelihood=lik) for seed in range(n_seeds)])

    x = model.truth # At the truth: the filter, direct runs, and the ODE agree
    ode = ss.fitting.loglik(model.build(x, data=data, **kw).run(), likelihood=lik)
    se = np.sqrt(pf[x].var(ddof=1)/n_seeds + direct[x].var(ddof=1)/n_seeds) # The Monte Carlo error of the difference
    assert abs(pf[x].mean() - direct[x].mean()) < 3*se + 0.5 # 0.5 allows for the gap between the mean log-likelihood of runs and the log of their mean likelihood (measured at 0.1 at most)
    assert abs(direct[x].mean() - ode) < 2.0 # At this population, runs are at the ODE limit, up to tau-leaping's own step (measured at 1.1 at most)
    best_pf = max(values, key=lambda x: pf[x].mean())
    best_direct = max(values, key=lambda x: direct[x].mean())
    assert best_pf == best_direct == model.truth # The filter peaks where direct runs do, at the values that made the data
    return pf, direct


# %% Simulation-based calibration

def sbc_fit(method, truth=0.35, seed=1):
    """ Simulate data from corpus.outcomes() at a known beta, and fit beta (starting from 0.25) by a method """
    model = models.outcomes
    stochastic = dict(total_pop=1e4, dt=ss.days(2), rand_seed=seed + 1) # A small population and two-day steps, so quick
    if method == 'if2': # Maximum likelihood of a stochastic compartmental sim, with data on cases (an outcome, with a delay)
        kw = dict(solver='binomial', **stochastic)
        sim = corpus.outcomes(beta=0.25, data=simulate(model, truth, seed=seed, **kw), **kw)
        return sim.fit(method=ss.fitting.method(method, particles=200, iterations=30, chains=2), likelihood=dict(cases=ss.poisson()), rand_seed=seed, verbose=False)
    elif method in ['pmcmc', 'abc-smc']: # Steps taken whole, and data on infections; ABC-SMC gives no posterior for a deterministic sim
        kw = dict(solver=ss.solvers.binomial(substeps=1), **stochastic)
        sim = corpus.outcomes(beta=0.25, data=simulate(model, truth, seed=seed, col='infection', **kw), **kw)
        if method == 'abc-smc':
            return sim.fit(method=ss.fitting.method(method, samples=60, generations=5), rand_seed=seed, verbose=False)
        return sim.fit(method=ss.fitting.method(method, draws=100, warmup=100, chains=2, particles=20, parallel=False), likelihood=dict(infection=ss.poisson()), rand_seed=seed, verbose=False) # Not in parallel, since forking after JAX is loaded can deadlock
    sim = corpus.outcomes(beta=0.25, likelihood=ss.poisson(), data=simulate(model, truth, seed=seed)) # Cases, with the likelihood declared on the outcome
    kwargs = dict(nuts=dict(draws=100, warmup=100, chains=2), optuna=dict(n_trials=40)).get(method, {})
    return sim.fit(method=ss.fitting.method(method, **kwargs) if kwargs else method, rand_seed=seed, verbose=False)


# How close an optimizer must come: the likelihood of 20 weeks of 10,000 or more cases is sharp (its 95% interval is about 0.2% of beta wide); the tolerances
# allow for the stopping rule of each method: Optuna samples rather than converges, and IF2 fits a stochastic sim (of 10,000 people), whose best value is further from the truth
rtols = {'nelder-mead':0.02, 'l-bfgs-b':0.02, 'optuna':0.05, 'if2':0.05}

@pytest.mark.parametrize('method', ['nelder-mead', 'l-bfgs-b', 'optuna', 'nuts', 'pmcmc', 'abc-smc', 'if2'])
def test_sbc(method):
    """ Each method recovers a known value from simulated data, and says whether it converged """
    truth = 0.35
    fit = sbc_fit(method, truth=truth, seed=1) # Two short chains of particle MCMC, or of IF2, pass both checks at some seeds only
    table = fit.table.set_index('par')
    assert fit.method == method and fit.why
    if fit.samples is None:
        assert abs(table.best.beta - truth) < rtols[method]*truth and fit.converged # The optimizer recovers the value that made the data
    else:
        assert table.lower.beta < truth < table.upper.beta # The posterior covers the value that made the data
        assert method == 'abc-smc' or table.rhat.beta < (1.3 if method == 'pmcmc' else 1.1) # The chains agree (two chains of 100 draws of particle MCMC are short and sticky, so their R-hat is noisy)
        assert (fit.converged is None) == (method == 'abc-smc') # Chains this short are not converged by the rule of an R-hat of at most 1.01, and say so; ABC-SMC makes no claim
    if method == 'if2': # And a rate that varies in time (the summer2 pattern), whose values each particle's rates are worked out from, a data point at a time
        model = models.detect
        kw = dict(solver='binomial', total_pop=1e5)
        detect = model.build(0.7*model.truth, data=simulate(model, **kw), **kw).fit(method=ss.fitting.method('if2', particles=100, iterations=10, chains=2), likelihood={model.col:ss.nbinom(n=20)}, verbose=False)
        assert np.isclose(detect.table.best[0], model.truth, rtol=0.1)
    return fit


# %% Fitting what users fit

def test_fitted_where_used():
    """ Parameters fitted where users put them: a modifier's amount, and a typed value; projections of the fit """
    def make(effect, dur):
        beta = ss.par(0.3, name='beta')
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=dur), init_n=dict(I=10), name='sir')
        masks = ss.modify(sir.pars.beta, multiply=effect, when=ss.schedule(start=30)) # A parameter used only by an intervention
        return ss.Sim(sir, interventions=masks, unit='day', stop=60, verbose=0)
    truth = make(0.4, ss.days(12)).run().results.sir.infection
    df = pd.DataFrame(dict(time=np.arange(0, 56, 5), infection=[np.asarray(truth)[t:t+5].sum() for t in range(0, 56, 5)])).set_index('time') # Each over the five days from its time
    sim = make(ss.par(0.6, bounds=[0.1, 1], name='effect'), ss.par(ss.days(8), bounds=[ss.days(3), ss.weeks(4)], name='dur'))
    sim.data = ss.SimData(df)
    fit = sim.fit(verbose=0)
    assert np.isclose(fit.pars.effect, 0.4, rtol=1e-3) and np.isclose(fit.pars.dur.value, 12, rtol=1e-3) and isinstance(fit.pars.dur, ss.days)
    assert np.shape(fit.project(dict(base=None)).results.sir.I.values)[0] == 1 # By default, one run: the runs of a deterministic best fit would all be the same
    return fit


def test_likelihood_profile():
    """ The profile likelihood of a parameter, with the others refitted """
    def make(solver='ode'):
        beta, gamma = ss.par(ss.perday(0.3), bounds=[0.05, 1.0], name='beta'), ss.par(ss.perday(0.15), bounds=[0.01, 0.5], name='gamma')
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        cases = ss.outcome(0.6 * infection, likelihood=ss.poisson())
        return ss.Sim(ss.Disease(infection, I >> R @ gamma, cases, init_n=dict(I=10), name='sir'), total_pop=10_000, unit='day', stop=84, solver=solver, verbose=0)
    truth = make().new(pars=dict(beta=ss.perday(0.4))).run().results.sir.cases.values
    rng = np.random.default_rng(1)
    days = np.arange(0, len(truth) - 6, 7) # Whole weeks, each from its day
    sim = make()
    sim.data = ss.SimData(pd.DataFrame(dict(time=days, cases=[rng.poisson(truth[t:t+7].sum()) for t in days])))
    prof = ss.fitting.likelihood_profile(sim, 'beta', values=np.linspace(0.37, 0.43, 7))
    low, high = prof.interval
    assert low < 0.4 < high and 0.37 < low and high < 0.43, f'Expected a 95% interval around 0.4 within the values, not {prof.interval}'
    assert {'beta', 'gamma', 'nll', 'within'} <= set(prof.table.columns) and prof.table.gamma.is_monotonic_increasing # Refitted: a faster recovery makes up for a higher beta
    alone = ss.fitting.likelihood_profile(sim, 'beta', pars=['beta']) # With gamma fixed, on the default grid: narrower
    assert alone.interval[0] < 0.4 < alone.interval[1] and alone.interval[1] - alone.interval[0] < high - low
    return prof


if __name__ == '__main__':
    T = sc.timer()
    for key in models:
        test_jax_loglik(key)
    for method in ['nelder-mead', 'nuts', 'if2']:
        test_sbc(method)
    T.toc()
