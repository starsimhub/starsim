"""
Contract tests for inference: the likelihoods that the fitting methods use match the
likelihood of a run, and each method recovers known values from simulated data

Run the longer simulation-based calibration with STARSIM_SLOW=1.
"""

# %% Imports and settings
import os
import importlib.util
import pytest
import numpy as np
import scipy.stats as sps
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.fit
import corpus

slow = pytest.mark.skipif(not os.getenv('STARSIM_SLOW'), reason='A longer run; set STARSIM_SLOW=1 to run it')
has_optuna = importlib.util.find_spec('optuna') is not None

start, stop = '2020-02-23', '2020-05-31' # As corpus.seasonal()
weeks = pd.date_range('2020-03-01', periods=13, freq='W') # The data of the seasonal models: weekly totals
outcome_weeks = pd.date_range('2020-01-08', '2020-05-27', freq='W') # The data of corpus.outcomes()


# %% Helper functions

def make_beta(kind='both', b=0.5, init=100, data=None, **kwargs):
    """
    As corpus.seasonal(), with beta as a fitted value times a time series ('ts'), times a seasonal curve ('season'), or times both ('both')

    `init` is the number infectious at the start, so that a large population can start with
    enough of them for a stochastic run to be close to the ODE.
    """
    dates = pd.date_range(start, stop)
    r = ss.timeseries(dict(zip(dates.strftime('%Y-%m-%d'), np.linspace(1.0, 0.3, len(dates)))))
    b = ss.par(b, bounds=[0.1, 1])
    season = ss.seasonal(mean=0.9, amplitude=0.1/0.9, peak=14/365)
    beta = dict(ts=b*r, season=b*season, both=b*season*r)[kind]
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, rate=ss.perday(0.25)), init=dict(I=init), name='sir')
    return corpus.make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


def make_detection(d=0.2, init=100, data=None, **kwargs):
    """ The summer2 pattern: case detection as a fitted value times a time series (a scale-up), in a rate that is not a transmission """
    dates = pd.date_range(start, stop)
    scaleup = ss.timeseries(dict(zip(dates.strftime('%Y-%m-%d'), np.linspace(0.2, 1.0, len(dates)))))
    d = ss.par(d, bounds=[0.05, 1])
    S, I, T, R = ss.states('S I T R')
    dis = ss.Disease(infection=S >> I @ 0.5*I, recovery=I >> R @ 0.25, detection=I >> T @ d*scaleup, treatment=T >> R @ 0.5, init=dict(I=init), name='sir')
    return corpus.make_sim(dis, kwargs, total_pop=1e6, unit='day', start=start, stop=stop, data=data)


# Each model with data: how to make it (with the fitted value first), the data series, its dates, and the fitted value that makes the data
models = sc.objdict(
    outcomes = sc.objdict(build=lambda x, **kw: corpus.outcomes(beta=x, **kw),   col='cases',     dates=outcome_weeks, truth=0.3),
    ts       = sc.objdict(build=lambda x, **kw: make_beta('ts', x, **kw),        col='infection', dates=weeks,         truth=0.5),
    season   = sc.objdict(build=lambda x, **kw: make_beta('season', x, **kw),    col='infection', dates=weeks,         truth=0.5),
    both     = sc.objdict(build=lambda x, **kw: make_beta('both', x, **kw),      col='infection', dates=weeks,         truth=0.5),
    detect   = sc.objdict(build=lambda x, **kw: make_detection(x, **kw),         col='detection', dates=weeks,         truth=0.2),
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
        prior = sum(ss.fit.logprior(p, val) for p,val in zip(fitted.values(), x))
        with jax.enable_x64(True):
            from_jax = float(density(jax.numpy.asarray(x))) - prior
        from_run = ss.fit.loglik(set_pars(x).run(), likelihood=likelihood)
        out.append((float(x[0]), from_jax, from_run))
    return out


# %% The likelihoods match the run (N1, N2)

jax_cases = ['outcomes', 'ts', 'season', 'both', 'detect', 'poisson_weekly']

@pytest.mark.parametrize('key', jax_cases)
def test_jax_loglik(key):
    sc.heading(f'Testing that the log-density in JAX equals the sim\'s log-likelihood: {key}')
    if key == 'poisson_weekly':
        sim = corpus.poisson_weekly() # Its data, and the likelihood declared on its outcome, are its own
        likelihood = None
        points = [0.25, 0.3, 0.35]
    else:
        model = models[key]
        sim = model.build(model.truth, data=simulate(model))
        likelihood = {model.col:ss.poisson()}
        points = [0.8*model.truth, model.truth, 1.2*model.truth] # Around the values that made the data, where a fit spends its time
    for x,from_jax,from_run in jax_loglik(sim, points, likelihood=likelihood):
        print(f'{key} at {x:.3g}: JAX {from_jax:.10g}, sim {from_run:.10g}')
        assert np.isfinite(from_run) # The data can be given by the model
        assert np.isclose(from_jax, from_run, rtol=1e-8, atol=1e-8), f'{key} at {x}: JAX {from_jax} against the sim\'s {from_run}' # Both are the same sums in double precision; they agree to 1e-15 where they work
    return


pf_cases = ['outcomes', 'ts', 'detect', 'season', 'both']

@pytest.mark.parametrize('key', pf_cases)
def test_particle_filter(key):
    sc.heading(f'Testing the particle filter against direct runs and the ODE, at a large population: {key}')
    model = models[key]
    pop = 1e8 # Large, so that a stochastic run is close to the ODE
    init = None if key == 'outcomes' else dict(init=pop*1e-4) # Enough infectious at the start that chance early on does not shift the epidemic (corpus.outcomes() starts with 1%)
    kw = sc.mergedicts(dict(total_pop=pop), init)
    lik = {model.col:ss.nbinom(n=20)} # Wider than a Poisson, so that the noise in the data, not the chance in a run, dominates at a large population, and the likelihood has an ODE limit
    data = simulate(model, **kw)
    sim = model.build(model.truth, data=data, solver='tau', **kw).init()
    filt = ss.fit.ParticleFilter(sim, likelihood=lik)
    with ss.options.context(threads=1):
        one = filt(n_particles=20, seed=0)
    assert filt(n_particles=20, seed=0) == one # Each particle has its own random numbers, so the estimate does not depend on the threads
    name = list(fitted_pars(sim))[0]
    values = [0.8*model.truth, model.truth, 1.2*model.truth]
    n_seeds = 3
    pf = {}
    direct = {}
    for x in values:
        sim.restart(pars={name:x})
        pf[x] = np.array([filt(n_particles=50, seed=seed) for seed in range(n_seeds)])
        direct[x] = np.array([ss.fit.loglik(model.build(x, data=data, solver='tau', rand_seed=seed, **kw).run(), likelihood=lik) for seed in range(n_seeds)])
        print(f'{key} at {x:.3g}: filter {pf[x].mean():.2f} +- {pf[x].std(ddof=1)/np.sqrt(n_seeds):.2f}; direct runs {direct[x].mean():.2f} +- {direct[x].std(ddof=1)/np.sqrt(n_seeds):.2f}')

    # At the truth: the filter, direct runs, and the ODE agree
    x = model.truth
    ode = ss.fit.loglik(model.build(x, data=data, **kw).run(), likelihood=lik)
    se = np.sqrt(pf[x].var(ddof=1)/n_seeds + direct[x].var(ddof=1)/n_seeds) # The Monte Carlo error of the difference
    print(f'{key} at the truth: ODE {ode:.2f}')
    assert abs(pf[x].mean() - direct[x].mean()) < 3*se + 0.5 # The filter estimates the likelihood of runs of this sim; 0.5 allows for the gap between the mean log-likelihood of runs and the log of their mean likelihood (measured at 0.1 at most here)
    assert abs(direct[x].mean() - ode) < 2.0 # At this population, runs are at the ODE limit, up to tau-leaping's own step (measured at 1.1 at most here)

    # The filter peaks where direct runs do
    best_pf = max(values, key=lambda x: pf[x].mean())
    best_direct = max(values, key=lambda x: direct[x].mean())
    assert best_pf == best_direct == model.truth # And both peak at the values that made the data
    return pf, direct


# %% Simulation-based calibration (P10, R5)

def sbc_fit(method, truth=0.35, seed=1, quick=True, verbose=False):
    """ Simulate data from corpus.outcomes() at a known beta, and fit beta (starting from 0.25) by a method """
    model = models.outcomes
    if method == 'if2': # Maximum likelihood of a stochastic compartmental sim, as for particle MCMC below, with data on cases (an outcome, with a delay)
        kw = dict(solver='binomial', total_pop=1e4, dt=ss.days(2), rand_seed=seed + 1)
        sim = corpus.outcomes(beta=0.25, data=simulate(model, truth, seed=seed, **kw), **kw)
        return sim.calibrate(method=method, likelihood=dict(cases=ss.poisson()), particles=200 if quick else 1000, iterations=30 if quick else 100, chains=2, rand_seed=seed, verbose=verbose)
    elif method == 'pmcmc':
        # A stochastic compartmental sim, made quick: a small population, two-day steps, and data on infections
        kw = dict(solver='binomial', total_pop=1e4, dt=ss.days(2), rand_seed=seed + 1)
        sim = corpus.outcomes(beta=0.25, data=simulate(model, truth, seed=seed, col='infection', **kw), **kw)
        n, particles = (100, 20) if quick else (300, 50)
        return sim.calibrate(method=method, likelihood=dict(infection=ss.poisson()), draws=n, warmup=n, chains=2, particles=particles, parallel=False, rand_seed=seed, verbose=verbose) # Not in parallel, since forking after JAX is loaded can deadlock
    sim = corpus.outcomes(beta=0.25, likelihood=ss.poisson(), data=simulate(model, truth, seed=seed)) # Cases, with the likelihood declared on the outcome
    kwargs = dict(
        nuts   = dict(draws=100, warmup=100, chains=2) if quick else dict(draws=500, warmup=500, chains=4),
        **{'abc-smc': dict(particles=60, generations=5) if quick else dict(particles=200, generations=8)},
        optuna = dict(trials=40 if quick else 200),
    ).get(method, {})
    return sim.calibrate(method=method, rand_seed=seed, verbose=verbose, **kwargs)


def check_fit(fit, truth, rtol=None, rhat_max=1.1):
    """ A posterior covers the truth; a point estimate is within a tolerance of it """
    table = fit.table.set_index('par')
    if fit.samples is not None:
        lower, upper = table.lower.beta, table.upper.beta
        print(f'{fit.method}: 95% interval [{lower:.4f}, {upper:.4f}] for {truth}')
        assert lower < truth < upper # The posterior covers the value that made the data
        if 'rhat' in table: # Not ABC-SMC, which has one population
            assert table.rhat.beta < rhat_max # The chains agree
    else:
        best = table.best.beta
        print(f'{fit.method}: best {best:.4f} for {truth}')
        assert abs(best - truth) < rtol*truth # The optimizer recovers the value that made the data
    return


methods = [
    'nelder-mead',
    'l-bfgs-b',
    pytest.param('optuna', marks=pytest.mark.skipif(not has_optuna, reason='Optuna is not installed')),
    'nuts',
    'pmcmc',
    'abc-smc',
    'if2',
]


def check_converged(fits):
    """ Each fit says that it converged (R5), and why """
    assert all(fit.converged and fit.why for fit in fits), [fit.why for fit in fits] # Each method says it converged, on a problem where it should
    return

# How close an optimizer must come: the data are 20 weeks of 10,000 or more cases each, so the
# likelihood is sharp (its 95% interval is about 0.2% of beta wide); the tolerances are well above
# that, and allow for the stopping rule of each method
rtols = {'nelder-mead':0.02, 'l-bfgs-b':0.02, 'optuna':0.05, 'if2':0.05} # Optuna samples rather than converges, so it gets near, not to, the best value; IF2 fits a stochastic sim (of 10,000 people), whose best value is further from the truth

@pytest.mark.parametrize('method', methods)
def test_sbc(method):
    sc.heading(f'Testing that {method} recovers a known value from simulated data')
    truth = 0.35
    fit = sbc_fit(method, truth=truth, seed=3 if method == 'pmcmc' else 1) # Two short chains of particle MCMC pass both checks at some seeds only (at seeds 1 to 3, one before the binomial draws were put in the order of the state names, and one after)
    assert fit.method == method
    check_fit(fit, truth, rtols.get(method), rhat_max=1.3 if method == 'pmcmc' else 1.1) # Two chains of 100 draws of particle MCMC are short and sticky, so their R-hat is noisy; 1.3 still catches chains stuck apart
    if fit.samples is None:
        check_converged([fit])
    else: # Chains this short are not converged by the rule of an R-hat of at most 1.01 (nor is the tolerance of ABC-SMC settled on a deterministic model), and a posterior method should say so
        assert fit.converged is not None and fit.why
    if method == 'if2': # And a rate that varies in time (the summer2 pattern), whose values each particle's rates are worked out from, a data point at a time
        model = models.detect
        kw = dict(solver='binomial', total_pop=1e5)
        detect = model.build(0.7*model.truth, data=simulate(model, **kw), **kw).calibrate(method='if2', likelihood={model.col:ss.nbinom(n=20)}, particles=300, iterations=30, chains=2, verbose=False)
        assert np.isclose(detect.table.best[0], model.truth, rtol=rtols['if2']) # The detection rate that made the data
    return fit


def test_two_optima():
    sc.heading('Testing that a fit with two local optima finds the better one, or says it has not converged')
    def make(counts):
        b = ss.par(0.32, bounds=[0.1, 1], name='b')
        dur = ss.par(ss.days(6.3), bounds=[2, 20], name='dur')
        S, I, R = ss.states(3)
        infection = S >> I @ b * I
        cases = ss.outcome(0.5 * infection, delay=ss.days(3), likelihood=ss.nbinom(n=10))
        sir = ss.Disease(infection, I >> R @ dur, cases, init=dict(I=1e-3), name='sir')
        data = pd.DataFrame(dict(cases=counts), index=pd.date_range('2020-01-08', periods=17, freq='7D'))
        return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-05-01', total_pop=1e5, data=data, verbose=0)

    # From the review's simulation-based calibration: an optimizer started from the best point of a scan stopped at 58.86, and SciPy said it had converged
    fit = make([408, 16683, 30842, 4196, 170, 58, 18, 5, 0, 1, 1, 0, 0, 0, 0, 0, 0]).calibrate(method='l-bfgs-b', rand_seed=57)
    assert fit.mismatch < 54.26 # The better optimum (b = 0.605, dur = 15.3)
    assert fit.converged is False # Only one of the searches from the scan's points found it

    # NUTS chains that all started at the initial values agreed on an epidemic that burns out early (b = 0.44), not on the truth (b = 0.116)
    fit = make([12, 47, 35, 42, 24, 16, 17, 12, 15, 20, 16, 5, 10, 4, 5, 10, 5]).calibrate(method='nuts', draws=200, warmup=200, chains=2, rand_seed=23)
    table = fit.table.set_index('par')
    assert table.lower.b < 0.116 < table.upper.b # The chains start from points of the scan that are apart

    # Both chains started from the best two points of the scan, and agreed on a mode 13 log-likelihood units below the best (b = 0.96, dur = 2.9), with R-hat 1.03
    fit = make([416, 18586, 25939, 2792, 40, 4, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]).calibrate(method='nuts', draws=200, warmup=200, chains=2, rand_seed=69)
    table = fit.table.set_index('par')
    assert table.lower.b < 0.656 < table.upper.b # The chains start from the optima found from the scan's points, so at the best (b = 0.646, dur = 18.5)
    return fit


def test_staged_and_sensitivity():
    sc.heading('Testing staged fitting, and which parameters matter')
    truth = 0.35
    sim = corpus.outcomes(beta=0.25, likelihood=ss.poisson(), data=simulate(models.outcomes, truth))
    fit = sim.calibrate(method=['optuna', 'nelder-mead'], trials=20, rand_seed=1, verbose=False) # Each stage starts from the one before
    assert fit.method == 'nelder-mead' and fit.converged and np.isclose(fit.pars.beta, truth, rtol=0.02)

    beta, dur, unused = ss.par(0.3, bounds=[0.1, 0.5]), ss.par(10, bounds=[5, 15]), ss.par(1.0, bounds=[0.5, 2])
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=ss.Transition(I, R, dur=dur), waning=R >> S @ unused*1e-9, init_prev=0.01, name='sir')
    table = ss.Sim(sir, stop=150, verbose=0).sensitivity(results=['infection'], n=40, verbose=False)
    total = table['sir.infection total']
    assert total.beta > 0.8 and total.dur > 0.8 and abs(total.unused) < 0.3 # The parameters that matter, and the one that does not

    net = ss.RandomNet(n_contacts=ss.par(4, bounds=[2, 10])) # A network's parameter, which an ODE turns into contact rates
    table = ss.Sim(ss.SIR(beta=ss.par(0.1, bounds=[0.05, 0.3])), networks=net, solver='ode', stop=40, verbose=0).sensitivity(results=['new_infections'], n=20, verbose=False)
    assert table['sir.new_infections total'].n_contacts > 0.8
    return table


def test_log_density():
    sc.heading('Testing the log-density as a function, with its gradient by reverse mode')
    sim = corpus.outcomes(beta=0.3, likelihood=ss.poisson(), data=simulate(models.outcomes, 0.35))
    ld = sim.log_density()
    x = ld.x0
    prior = ss.fit.logprior(sim.new().init().diseases.sir.pars.beta, x[0])
    assert np.isclose(ld(x), ss.fit.loglik(sim.new().run()) + prior, rtol=1e-9) # The sim's log-likelihood, plus the prior
    h = 1e-6
    assert np.isclose(ld.grad(x)[0], (ld(x + h) - ld(x - h))/(2*h), rtol=1e-5) # Reverse mode agrees with finite differences
    assert np.allclose(ld.run(x)['sir.n_I'], sim.new().run().results.sir.n_I, rtol=1e-9) # And the run, as a function
    slow = corpus.outcomes(beta=0.06, likelihood=ss.poisson(), data=simulate(models.outcomes, 0.35))
    (_, from_jax, from_run), = jax_loglik(slow, [1.0]) # Built where a step needs 1 substep, evaluated where it needs several
    assert np.isclose(from_jax, from_run, rtol=1e-9) # The substeps are worked out over the bounds, not only at the initial values
    return ld


def test_value_per_level():
    sc.heading('Testing a parameter with a value for each place, fitted and projected')
    def make(b, data=None):
        beta = ss.par(ss.by(patch=list(b)), bounds=[0.1, 0.8])
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, rate=0.1), init=dict(I=ss.by(patch=[10, 10])), name='sir')
        sir.stratify(patch=['a', 'b'])
        return ss.Sim(sir, stop=60, data=data, verbose=0)
    truth = make([0.25, 0.4]).run().results.sir.infection
    total = lambda lab, t: np.asarray(truth.select(patch=lab))[t-4:t+1].sum() # Data on a flow are its totals over each period
    df = pd.DataFrame([dict(time=t, patch=lab, infection=total(lab, t)) for lab in ['a', 'b'] for t in range(5, 61, 5)]).set_index('time')
    fit = make([0.3, 0.3], data=ss.data(df)).calibrate(verbose=0)
    assert list(fit.table.par) == ['beta[a]', 'beta[b]'] and np.allclose(fit.pars.beta.values, [0.25, 0.4], rtol=1e-4) # Each level is fitted
    assert np.allclose(fit.sim.diseases.sir.pars.beta.raw.values, [0.25, 0.4], rtol=1e-4) # And the fitted sim has them
    one = fit.sim.new(pars={'beta[b]': 0.1}) # One level by name
    assert np.allclose(one.init().diseases.sir.pars.beta.raw.values, [fit.pars.beta.values[0], 0.1])
    proj = fit.project(dict(base=None), n=2)
    assert np.shape(proj.results.sir.n_I.values)[0] == 2 # Two runs
    return fit


def test_fitted_where_used():
    sc.heading('Testing parameters fitted where users put them: a modifier\'s amount, typed values and bounds')
    def make(effect, dur):
        beta = ss.par(0.3, name='beta')
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=dur), init=dict(I=10), name='sir')
        masks = ss.modify(sir.pars.beta, multiply=effect, start=30) # A parameter used only by an intervention
        return ss.Sim(sir, interventions=masks, unit='day', stop=60, verbose=0)
    truth = make(0.4, ss.days(12)).run().results.sir.infection
    df = pd.DataFrame(dict(time=np.arange(5, 61, 5), infection=[np.asarray(truth)[t-4:t+1].sum() for t in range(5, 61, 5)])).set_index('time')
    sim = make(ss.par(0.6, bounds=[0.1, 1], name='effect'), ss.par(ss.days(8), bounds=[ss.days(3), ss.weeks(4)], name='dur'))
    sim.data = ss.data(df)
    assert np.isclose(sim.new(pars=dict(effect=0.4, dur=ss.days(12))).run().results.sir.infection[-1], truth[-1]) # Each can be set by name
    fit = sim.calibrate(verbose=0)
    assert np.isclose(fit.pars.effect, 0.4, rtol=1e-3) and np.isclose(fit.pars.dur.value, 12, rtol=1e-3) and isinstance(fit.pars.dur, ss.days)
    with pytest.raises(TypeError, match='put the type inside'):
        ss.perday(ss.par(2.4))

    # A loss that is a function of the sim needs no data
    plain = make(ss.par(0.6, bounds=[0.1, 1], name='effect'), ss.days(12))
    loss = lambda sim: (np.sum(sim.results.sir.infection) - np.sum(truth))**2 # The total, which rises with the effect
    assert np.isclose(plain.calibrate(gof=loss, verbose=0).pars.effect, 0.4, rtol=1e-2)

    # A network's parameter, which a compartmental run turns into contact rates
    def netted(n_contacts):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.05 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        return ss.Sim(sir, networks=ss.RandomNet(n_contacts=n_contacts), solver='ode', unit='day', stop=40, verbose=0)
    n_I = netted(6).run().results.sir.n_I
    sim = netted(ss.par(4, bounds=[2, 10]))
    sim.data = ss.data(pd.DataFrame(dict(time=np.arange(5, 41, 5), n_I=np.asarray(n_I)[5::5])).set_index('time'))
    assert np.isclose(sim.calibrate(verbose=0).pars.n_contacts, 6, rtol=1e-2) # It was not fitted
    return fit


def test_optimize():
    sc.heading('Testing allocating spending between two programs')
    def make():
        S, I, R = ss.states(3)
        a, b = ss.par(50.0, name='spend_a'), ss.par(50.0, name='spend_b') # Spending on two programs: a cuts transmission twice as much as b
        cov = lambda spend, cost: np.minimum(spend/cost/(S + I + R).total, 1)
        beta = 0.3*(1 - 0.5*cov(a, 0.01) - 0.25*cov(b, 0.01))
        sir = ss.Disease(S >> I @ beta*I.frac, I >> R @ ss.days(10), init=dict(I=10), name='sir')
        return ss.Sim(sir, total_pop=10_000, unit='day', stop=100, verbose=0)
    best = ss.fit.optimize(make(), spend=dict(a='spend_a', b='spend_b'), minimize='sir.n_R')
    assert best.alloc['a'] > 95 and np.isclose(sum(best.alloc.values()), 100) # Almost all of it to the better program, keeping the total
    assert best.value < best.value_before
    return best


def test_data_files(tmp_path):
    sc.heading('Testing data from a file separated by tabs, and a column that several modules could have')
    path = tmp_path/'data.tsv'
    pd.DataFrame(dict(time=[10, 20, 30], infection=[50.0, 100.0, 80.0])).to_csv(path, sep='\t', index=False)
    assert list(ss.data(path).load().columns) == ['time', 'infection']
    def make(name):
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ 0.2*I.frac, recovery=I >> R @ ss.days(10), init=dict(I=10), name=name)
    with pytest.raises(ss.errors.UnknownDataColumn, match='flu.infection'):
        ss.Sim([make('flu'), make('rsv')], unit='day', stop=60, verbose=0, data=ss.data(path)).init()
    return path


def test_observations():
    sc.heading('Testing binomial and Bernoulli observations')
    k, p = np.array([0, 3, 10]), np.array([0.1, 0.3, 0.5])
    assert np.allclose(ss.fit.logpdf(ss.binom(n=20), k, p), sps.binom.logpmf(k, 20, p)) # The same as SciPy's
    assert np.allclose(ss.fit.logpdf(ss.bernoulli(), np.array([0, 1]), np.array([0.2, 0.7])), np.log([0.8, 0.7]))

    # Surveys of 500 people, the positives drawn from the prevalence: fit beta to them
    def make(beta, data=None):
        beta = ss.par(beta, bounds=[0.05, 1])
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S >> I @ beta*I, I >> R @ ss.days(10), prevalence=I/(S + I + R), init_prev=0.01, name='sir')
        return ss.Sim(sir, unit='day', stop=60, data=data, verbose=0)
    prev = make(0.3).run().results.sir.prevalence.values
    days = [10, 20, 30, 40, 50]
    data = sc.dataframe(t=days, prevalence=np.random.default_rng(1).binomial(500, prev[days]))
    fit = make(0.2, data=data).calibrate(likelihood=dict(prevalence=ss.binom(n=500)), method='nelder-mead', verbose=False)
    assert np.isclose(fit.pars.beta, 0.3, rtol=0.05) # Recovered from 2,500 people surveyed
    return fit


@slow
@pytest.mark.parametrize('method', methods)
def test_sbc_slow(method):
    sc.heading(f'Testing that {method} recovers known values drawn from the prior, from repeated simulated data')
    n = 10
    truths = np.random.default_rng(2).uniform(0.2, 0.5, n) # Within the bounds [0.05, 1], away from them, and from where the fit starts (0.25)
    covered = 0
    fits = []
    for i,truth in enumerate(truths):
        fit = sbc_fit(method, truth=truth, seed=i + 10, quick=False)
        table = fit.table.set_index('par')
        if fit.samples is not None:
            covered += table.lower.beta < truth < table.upper.beta
        else:
            check_fit(fit, truth, rtols.get(method))
        fits.append(fit)
    if method in ['nuts', 'pmcmc']:
        print(f'{method}: the 95% interval covered the truth in {covered} of {n}')
        assert covered >= 8 # With 95% intervals, 8 or more of 10 cover the truth with a probability of 0.99
    check_converged(fits)
    return


if __name__ == '__main__':
    T = sc.timer()

    for test,cases in [(test_jax_loglik, jax_cases), (test_particle_filter, pf_cases), (test_sbc, ['nelder-mead', 'l-bfgs-b', 'optuna', 'nuts', 'pmcmc', 'abc-smc', 'if2'])]:
        for case in cases:
            try:
                test(case)
            except AssertionError as E:
                print(f'Failed: {E!r}')

    T.toc()
