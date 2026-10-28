"""
Test the ODE run in JAX: that it matches the kernels, its gradients, gradient-based fitting, and many runs with different values
"""

# %% Imports and settings
import warnings
import importlib
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.fit
from test_kernels import models, largest_diff

kw = dict(verbose=0)


def make_sir(beta=0.4, dur=8, init=0.01, data=None, **kwargs):
    """ An SIR model whose infection rate has a function in it, and with reported cases and the prevalence """
    beta = ss.par(beta, bounds=[0.1, 1], name='beta')
    dur  = ss.par(ss.days(dur), bounds=[2, 20], name='dur')
    init = ss.par(init, bounds=[1e-4, 0.1], name='init')
    S, I, R = ss.states(3)
    infection = S >> I @ beta * np.exp(-I.frac) * I # Fewer contacts when many are ill
    recovery  = I >> R @ dur
    cases     = ss.outcome(0.5 * infection, delay=ss.days(3))
    sir = ss.Disease(infection, recovery, cases, prevalence=I/(S + I + R), init=dict(I=init), name='sir')
    return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-05-01', data=data, **kw, **kwargs)


def run_jax(sim):
    """ Take a whole run in JAX, as ss.parallel() does for many runs """
    sim.init()
    sim.jax = True
    return sim.run()


@sc.timer()
def test_jax_run():
    sc.heading('Testing that a run in JAX matches the kernels')
    for key,(make,pars) in models.items():
        sims = [ss.Sim(make(), **sc.mergedicts(pars, kw)).run(), run_jax(ss.Sim(make(), **sc.mergedicts(pars, kw)))]
        diff = largest_diff(*sims)
        stepped = [note for note in sims[1].notes if 'JAX cannot' in note]
        print(f'{key:8s} {diff:0.1e} {"(run by the kernels: " + stepped[0] + ")" if stepped else ""}')
        assert diff < 1e-9, f'{key}: the run in JAX differs from the kernels by {diff}'
        assert not stepped # Including the TB model, whose transfer has a constant rate
    return sims


def compare(make, n=30):
    """ Final states (and deliveries) of n runs by the kernels and by JAX: the z of each difference in means, and whether every JAX run has the same people """
    out = {}
    for jax in [False, True]:
        rows = []
        for seed in range(n):
            sim = make(seed)
            sim.jax = jax
            sim.run()
            assert not [note for note in sim.notes if 'JAX cannot' in note]
            rows.append([mod.results[state][-1] for mod in sim.declared for state in mod.graph.states] + [mod.results.delivered.values.sum() for mod in sim.interventions() if 'delivered' in mod.results])
        out[jax] = np.array(rows)
    a, b = out[True], out[False]
    z = (a.mean(0) - b.mean(0))/np.sqrt((a.var(0) + b.var(0))/n + 1e-12)
    print(f'JAX {a.mean(0).round()}, kernels {b.mean(0).round()}, z {z.round(2)}')
    return z


@sc.timer()
def test_jax_binomial():
    sc.heading('Testing the chain binomial and tau-leaping in JAX against the kernels, in distribution')
    make, pars = models.seir # Seasonal transmission, a dwell time in stages, and a branch
    z = compare(lambda seed: ss.Sim(make(), **sc.mergedicts(pars, kw), solver='binomial', rand_seed=seed))
    assert np.all(np.abs(z) < 4), f'The chain binomial in JAX differs from the kernels: z = {z}'

    # Deliveries, with substeps; tau-leaping with a movement register; two diseases tracked jointly
    def deliveries(seed):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir').stratify(age=[0, 15, 65])
        offer = ss.vaccinate(ss.Vaccine(efficacy=0.9), where=sir.S, coverage=ss.by(age=[0.6, 0.3, 0.0]), start=10)
        return ss.Sim(sir, interventions=offer, solver='binomial', substeps=2, unit='day', stop=60, total_pop=20_000, rand_seed=seed, **kw)
    def register(seed):
        moves = sc.dataframe(date=[5, 10, 20], n=[1000, 500, 300], states=[None, 'S', None], **{'from':['a', None, 'b'], 'to':['b', 'c', None]})
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_n=dict(I=ss.by(place=[20, 0, 0])), name='sir')
        sir.stratify(place=['a', 'b', 'c'], kind='space', split=[0.5, 0.3, 0.2], mobility=ss.geo.Mobility(data=moves))
        return ss.Sim(sir, solver='tau', total_pop=10_000, unit='day', stop=40, rand_seed=seed, **kw)
    def crossing(seed):
        S, I = ss.states('S I')
        a = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), cure=ss.Transition(I, S, dur=ss.days(10)), init_prev=0.05, name='a')
        S, I, R = ss.states('S I R')
        b = ss.Disease(infection=ss.Transition(S, I, rate=0.25 * I), recovery=ss.Transition(I, R, dur=ss.lognorm_ex(8, 3)), init_prev=0.01, name='b')
        return ss.Sim(diseases=[a, b], connectors=ss.modify(b.rel_sus, multiply=2.0, where=a.I), solver='binomial', unit='day', stop=60, total_pop=10_000, rand_seed=seed, **kw)
    for make in [deliveries, register, crossing]:
        z = compare(make, n=20)
        assert np.all(np.abs(z) < 4), f'{make.__name__}: JAX differs from the kernels: z = {z}'
    return z


@sc.timer()
def test_jax_engine():
    sc.heading('Testing the choice of JAX or the kernels, and the guard on the FFI handler')
    chain = importlib.import_module('starsim.perflib.chain')
    assert chain.check() == chain.api_version() # The handler reports the XLA FFI API version of the installed JAX
    make, pars = models.towns
    sim = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='binomial').run()
    assert sim.engine_reason.startswith('kernels: a short run') # Shorter than JAX takes to start
    after = ss.solvers.choose.jax_after
    try:
        ss.solvers.choose.jax_after = 0 # As if it were a long run
        sim = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='binomial', rand_seed=3).run()
        assert sim.engine_reason.startswith('JAX: a long run')
        msim = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='binomial').run(n_runs=4) # Taken together in JAX, each as it would be alone
        assert np.array_equal(msim.results.sir.I.values[2], sim.results.sir.I.values) # Seeds 1 to 4
        ode = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='ode').run() # The ODE too, which agrees with the kernels
        assert ode.engine_reason.startswith('JAX: a long run')
        ref = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='ode')
        ref.jax = False
        assert np.allclose(ode.results.sir.I.values, ref.run().results.sir.I.values, rtol=1e-9, atol=0)
    finally:
        ss.solvers.choose.jax_after = after
    reported = chain.reported_version
    try: # A handler that reports another version is not used: the run falls back to the kernels, with a note
        chain.reported_version = lambda handler: (0, 0)
        sim = ss.Sim(make(), **sc.mergedicts(pars, kw), solver='binomial')
        sim.jax = True
        sim.run()
        assert any('JAX cannot' in note and 'API version' in note for note in sim.notes)
    finally:
        chain.reported_version = reported
    return sim


@sc.timer()
def test_gradient():
    sc.heading('Testing gradients of results against finite differences')
    sim = make_sir()
    grad = sim.gradient(results=['I', 'cases'], pars=['beta', 'dur'])
    assert not sim.initialized # The sim itself is not changed
    for name,step in [('beta', 1e-6), ('dur', 1e-5)]:
        x0 = 0.4 if name == 'beta' else 8
        runs = [make_sir(**{name:x0 + sign*step}).run().results.sir for sign in [1, -1]]
        for key in ['I', 'cases']:
            fd = (runs[0][key].values - runs[1][key].values)/(2*step)
            diff = np.abs(grad[name].sir[key] - fd).max()/np.abs(fd).max()
            print(f'd {key}/d {name}: largest {np.abs(fd).max():0.4g}, differs from finite differences by {diff:0.1e}')
            assert diff < 1e-5
    assert np.all(grad.beta.sir.I[1:20] > 0) # A higher beta means more infectious people early on (after the start, which it does not change)
    with pytest.raises(ValueError): # Gradients are for the ODE only
        sim.new(solver='gillespie').gradient()
    return grad


@sc.timer()
def test_fit():
    sc.heading('Testing that calibration uses the gradients')
    truth = make_sir(beta=0.5, dur=6, init=0.002).run().results.sir
    data = pd.DataFrame(dict(prevalence=truth.prevalence.values[::7]), index=truth.timevec[::7])
    fit = make_sir(data=data).fit(gof='rmse', verbose=True)
    print(fit.table)
    assert fit.method == 'l-bfgs-b'
    assert np.allclose(fit.table.best, [0.5, 6, 0.002], rtol=1e-2)
    assert ss.fit.default_method(fit.sim, False, n_pars=2)[0] == 'nelder-mead' # Faster than gradients for one or two parameters
    assert ss.fit.default_method(fit.sim, True)[0] == 'l-bfgs-b' # A best fit, with a likelihood too: a posterior is asked for by name
    return fit


@sc.timer()
def test_identifiability():
    sc.heading('Testing the notes on parameters that the data cannot tell apart')
    def make(kind='product', data=None):
        beta = ss.par(0.1, bounds=[0.01, 1], name='beta')
        c = ss.par(4, bounds=[1, 10], name='c')
        S, I, R = ss.states(3)
        rate = beta * c * I if kind == 'product' else (beta + c/40) * I if kind == 'sum' else beta * I
        sir = ss.Disease(S >> I @ rate, I >> R @ ss.par(ss.days(8), bounds=[2, 20], name='dur'), init=dict(I=0.01), name='sir')
        return ss.Sim(sir, unit='day', stop=100, data=data, **kw)
    truth = make().run().results.sir
    data = pd.DataFrame(dict(I=truth.I.values[::7]), index=truth.timevec[::7])
    notes = {kind:make(kind, data=data).fit(pars=['beta', 'dur'] + (['c'] if kind != 'plain' else [])).sim.notes for kind in ['product', 'sum', 'plain']}
    assert any('only multiplied together' in note for note in notes['product']) # From the model
    assert any('almost the same way' in note for note in notes['sum']) # From the gradients at the best fit
    assert not notes['plain']
    return notes


@sc.timer()
def test_posterior():
    sc.heading('Testing a posterior from NUTS')
    truth = make_sir(beta=0.5, dur=6).run().results.sir
    weekly = np.add.reduceat(truth.cases.values[1:], np.arange(0, len(truth.cases) - 1, 7))[:-1] # A flow is compared with its total since the last data point
    data = pd.DataFrame(dict(cases=np.random.default_rng(1).poisson(weekly)), index=truth.timevec[7::7][:len(weekly)])
    fit = make_sir(dur=6, data=data).fit(pars=['beta'], likelihood=dict(cases=ss.poisson()), method='nuts', draws=100, warmup=100, chains=2, rand_seed=1, verbose=True)
    fit.summarize()
    table = fit.table.set_index('par')
    assert fit.method == 'nuts'
    assert len(fit.samples) == 200 and list(fit.samples.columns) == ['chain', 'draw', 'beta']
    assert table.lower.beta < 0.5 < table.upper.beta # The posterior covers the value that made the data
    assert np.all(table.rhat < 1.1)
    assert np.isnan(ss.fit.diagnose(np.random.default_rng(1).random((2, 3, 1)))[1]).all() # Chains too short for an effective sample size
    chains = np.zeros((4, 4000)) # Four AR(1) chains with a lag-1 correlation of 0.5, whose effective sample size is a third of the draws
    noise = np.random.default_rng(1).standard_normal(chains.shape)
    for t in range(1, chains.shape[1]):
        chains[:, t] = 0.5*chains[:, t-1] + noise[:, t]
    rhat, ess = ss.fit.diagnose(chains[:, :, None])
    assert rhat[0] < 1.01 and np.isclose(ess[0], chains.size/3, rtol=0.05)
    assert ss.fit.diagnose(chains[:, :, None] + np.array([0, 0, 0, 1.0])[:, None, None])[0][0] > 1.05 # One chain apart from the others
    assert fit.diagnostics.band['cases'].shape == (2, len(data)) # Over the weeks of the data, as the fit compares them
    with pytest.raises(ValueError, match='not every data series has a likelihood'): # A posterior needs a likelihood
        make_sir(data=data).fit(pars=['beta'], method='nuts')
    return fit


def make_coinfection(beta=0.15, data=None, dt=0.25, **kwargs):
    """ Two diseases that modify each other, so whose states are tracked jointly (see ss.solvers.Crossing), one with a transfer at a constant rate """
    beta = ss.par(beta, bounds=[0.05, 0.5], name='beta')
    S, I = ss.states('S I')
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), death=ss.death(I, rate=0.05), init_prev=0.05, name='hiv')
    hiv.stratify(art=[False, True], states=['I'], adjust=dict(death=[1.0, 0.2]), switch=ss.transfer(art=True, rate=ss.peryear(0.1)))
    S, I = ss.states('S I')
    syph = ss.Disease(infection=ss.Transition(S, I, rate=1.0 * I), cure=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.05, name='syph')
    coinfection = [
        ss.modify(hiv.rel_trans, multiply=3.0, where=syph.I, name='hiv_trans'),
        ss.modify(syph.rel_sus, multiply=2.0, where=hiv.I, name='syph_sus'),
    ]
    return ss.Sim(diseases=[hiv, syph], connectors=coinfection, start=2000, stop=2030, dt=ss.years(dt), data=data, **(dict(solver='ode', total_pop=20_000) | kw | kwargs))


@sc.timer()
def test_crossing():
    sc.heading('Testing diseases tracked jointly: a whole run, many runs, in JAX, restarted, differentiated, filtered, and fitted')
    sim = make_coinfection()
    sim.init()
    assert sim.kernel_only() # Modifiers on states, and a transfer at a constant rate, are worked out before the run
    sim.run()
    stepped = make_coinfection().run(until=2030) # Step by step
    jax = run_jax(make_coinfection())
    for other in [stepped, jax]:
        for key in ['hiv', 'syph']:
            assert np.allclose(other.results[key].I, sim.results[key].I, rtol=1e-9)
    assert sim.results.hiv.switch[1:].sum() > 0 # The transfer's flow is recorded

    # Each flow of three diseases tracked jointly takes the exposure summed over the others' compartments as one number, worked out once (adding it up in each flow took 513 operations here, and three HPV genotypes 383 s instead of 9 s)
    S, I, R = ss.states('S I R')
    ds = [ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=5), waning=ss.Transition(R, S, dur=20), init_prev=0.01, name=f'd{i}') for i in range(3)]
    mods = [ss.modify(ds[i].rel_sus, multiply=0.5, where=ds[(i+1) % 3].R, name=f'm{i}') for i in range(3)]
    three = ss.Sim(diseases=ds, connectors=mods, unit='day', stop=100, total_pop=10_000, solver='ode', **kw).run()
    assert len(three.crossings[0].program.ops) == 81 and np.isclose(three.results.d0.I[50], 656.392, rtol=1e-6) # One operation for each of the 81 flows; the same run

    # The diseases are one model, so every flow is worked out from the people at the start of the step (each disease in turn, before, was 1% off a fine step)
    fine = make_coinfection(dt=0.025).run()
    assert np.allclose(fine.results.syph.I[::10], sim.results.syph.I, rtol=1e-5)

    # Many runs at once take the diseases together too (each was run on its own before, with 23% fewer HIV infections at the end)
    msim = make_coinfection().run(n_runs=2)
    assert np.allclose(msim.results.hiv.I.values[1], sim.results.hiv.I, rtol=1e-12)

    # Restarted with other values, as a fresh sim
    restarted = make_coinfection().init().restart(pars={'hiv.beta':0.2}).run()
    fresh = make_coinfection(beta=0.2).run().results
    assert np.allclose(restarted.results.hiv.I, fresh.hiv.I, rtol=1e-12)

    # The gradient of each disease's results, through the other
    grad = make_coinfection().gradient(results=['I'], pars=['hiv.beta'])
    step = 1e-6
    runs = [make_coinfection(beta=0.15 + sign*step).run().results for sign in [1, -1]]
    for key in ['hiv', 'syph']:
        fd = (runs[0][key].I.values - runs[1][key].I.values)/(2*step)
        assert np.abs(grad['hiv.beta'][key].I - fd).max() < 1e-5*np.abs(fd).max()

    # The particle filter takes the diseases together, under a stochastic solver, as runs of their program
    large = make_coinfection(beta=0.2, total_pop=1e7).run().results # Large, so that runs differ little
    data = pd.DataFrame({'hiv.I':large.hiv.I.values, 'syph.I':large.syph.I.values}, index=large.hiv.timevec)[::8]
    lik = {'hiv.I':ss.nbinom(n=20), 'syph.I':ss.nbinom(n=20)}
    pf = np.mean([ss.fit.ParticleFilter(make_coinfection(beta=0.2, data=data, solver='binomial', total_pop=1e7).init(), likelihood=lik)(n_particles=20, seed=seed) for seed in range(3)])
    direct = np.mean([ss.fit.loglik(make_coinfection(beta=0.2, data=data, solver='binomial', total_pop=1e7, rand_seed=seed).run(), likelihood=lik) for seed in range(3)])
    print(f'Particle filter {pf:.2f}, direct runs {direct:.2f}')
    assert abs(pf - direct) < 1 # The filter estimates the likelihood of runs of the sim

    # NUTS, with the log-density in JAX
    data = pd.DataFrame({'hiv.I':fresh.hiv.I.values, 'syph.I':fresh.syph.I.values}, index=fresh.hiv.timevec)[::8]
    fit = make_coinfection(data=data).fit(likelihood={'hiv.I':ss.normal(scale=50), 'syph.I':ss.normal(scale=50)}, method='nuts', draws=50, warmup=50, chains=2, rand_seed=1, verbose=True)
    table = fit.table.set_index('par')
    assert fit.method == 'nuts'
    assert table.lower['hiv.beta'] < 0.2 < table.upper['hiv.beta']
    return sim


@sc.timer()
def test_posterior_data():
    sc.heading('Testing a posterior from data by stratum, and from the number in an outcome')
    def make(beta=0.4, data=None):
        beta = ss.par(beta, bounds=[0.1, 1], name='beta')
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        hosp = ss.outcome(0.05 * infection, delay=ss.days(3), duration=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3)))
        sir = ss.Disease(infection, I >> R @ ss.days(8), hosp, init=dict(I=0.01), name='sir')
        sir.stratify(age=[0, 15, 65])
        return ss.Sim(sir, unit='day', stop=100, total_pop=1e5, data=data, **kw)

    # Data in long form: a column says which age group each row is for
    truth = make(beta=0.5).run().results.sir
    rows = [dict(t=t, age=age, I=n, n_hosp=h) for age in ['0-14', '15-64', '65+'] for t,n,h in zip(truth.timevec[::10], truth.I.select(age=age).values[::10], truth.n_hosp.select(age=age).values[::10])]
    sim = make(data=pd.DataFrame(rows)).init()
    assert 'I[age=15-64]' in sim.data.columns
    with pytest.raises(KeyError, match='which is not a level of "age"'): # Not a level of age
        make(data=pd.DataFrame([dict(t=10, age='0-4', I=1)])).init()

    # Fitted with the log-density in JAX: it is checked against runs of the sim, and sampled again with Python in the loop if they differ
    with sc.capture() as txt, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = make(data=pd.DataFrame(rows)).fit(likelihood=dict(I=ss.normal(scale=50), n_hosp=ss.normal(scale=5)), method='nuts', draws=50, warmup=50, chains=2, rand_seed=1, verbose=True)
    assert 'Python in the loop' not in txt and not any('differs' in str(w.message) for w in caught)
    table = fit.table.set_index('par')
    assert table.lower.beta < 0.5 < table.upper.beta
    return fit


@sc.timer()
def test_parallel_pars():
    sc.heading('Testing many runs with different values, in one call')
    betas = [0.3, 0.4, 0.5]
    singles = [make_sir(beta=beta).run().results.sir.I.values for beta in betas]
    for n in [1, ss.run.jax_runs//3 + 1]: # Enough runs are taken in one call to JAX
        msim = ss.parallel(make_sir(), pars=dict(beta=betas*n))
        n_I = msim.results.sir.I.values[:3]
        assert (msim.sims[0].diseases.sir.engine.jax_run is not None) == (n > 1)
        assert msim.run_pars.beta.tolist() == betas*n
        assert np.allclose(n_I, singles, rtol=1e-9), f'{3*n} runs differ from single runs'
    msim = ss.parallel(make_sir(), pars=[dict(beta=0.3), dict(beta=0.5, dur=ss.days(4))], n_runs=2) # A list of sets, each run twice
    assert len(msim.results.sir.I.values) == 4
    return msim


@sc.timer()
def test_several_values():
    sc.heading('Testing gradients with respect to a value for each level and a time series, in forward and reverse mode')
    def by_place(b, data=None):
        beta = ss.par(ss.by(patch=list(b)), bounds=[0.1, 0.8], name='beta')
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ 0.1, init_n=dict(I=ss.by(patch=[10, 10, 10])), name='sir')
        sir.stratify(patch=['a', 'b', 'c'])
        return ss.Sim(sir, unit='day', start=0, stop=60, data=data, **kw)

    # A gradient for each level, of the results and of a function of the sim, as finite differences give them
    b = np.array([0.3, 0.4, 0.5])
    grad = by_place(b).gradient('I', pars='beta')
    cost = lambda sim: float(np.sum(sim.results.sir.infection))
    slopes = by_place(b).gradient(cost)
    assert list(grad.keys()) == list(slopes.keys()) == ['beta[a]', 'beta[b]', 'beta[c]']
    h = 1e-6
    for k,label in enumerate(['a', 'b', 'c']):
        sims = [by_place(b + sign*h*np.eye(3)[k]).run() for sign in [1, -1]]
        fd = (sims[0].results.sir.I.values - sims[1].results.sir.I.values)/(2*h)
        assert np.abs(grad[f'beta[{label}]'].sir.I - fd).max() < 1e-5*np.abs(fd).max()
        assert np.isclose(slopes[f'beta[{label}]'], (cost(sims[0]) - cost(sims[1]))/(2*h), rtol=1e-5)

    # The log-density and a fit by gradients, whose gradient is the same in either mode, checkpointed or not
    truth = by_place([0.25, 0.4, 0.35]).run().results.sir.infection
    df = pd.DataFrame([dict(time=t, patch=lab, infection=np.asarray(truth.select(patch=lab))[t-4:t+1].sum()) for lab in ['a', 'b', 'c'] for t in range(5, 61, 5)]).set_index('time')
    sim = by_place([0.3]*3, data=ss.simdata(df))
    ld = sim.log_density(likelihood=dict(infection=ss.poisson()))
    assert ld.names == ['beta[a]', 'beta[b]', 'beta[c]']
    fd = [(ld(ld.x0 + 1e-6*e) - ld(ld.x0 - 1e-6*e))/2e-6 for e in np.eye(3)]
    assert np.allclose(ld.grad(ld.x0), fd, rtol=1e-5)
    x = ld.x0*1.1
    out = {}
    for name,rmin,ckpt in [('forward', np.inf, np.inf), ('reverse', 0, np.inf), ('checkpointed', 0, 0)]:
        reverse_min, checkpoint_bytes = ss.fit.reverse_min, ss.gradients.checkpoint_bytes
        ss.fit.reverse_min, ss.gradients.checkpoint_bytes = rmin, ckpt
        try:
            problem = ss.fit.Problem(sim, likelihood=dict(infection=ss.poisson()), verbose=0)
            out[name] = problem.with_gradient(x)
            if name == 'reverse': # With more substeps than at the start: taken again with more
                out['more'] = problem.with_gradient(x*2)
        finally:
            ss.fit.reverse_min, ss.gradients.checkpoint_bytes = reverse_min, checkpoint_bytes
    for name in ['reverse', 'checkpointed']:
        assert np.isclose(out[name][0], out['forward'][0], rtol=1e-10) and np.allclose(out[name][1], out['forward'][1], rtol=1e-8), name
    assert np.isclose(out['more'][0], problem.objective(x*2), rtol=1e-8)
    fit = sim.fit(likelihood=dict(infection=ss.poisson()), method='l-bfgs-b', verbose=0)
    assert np.allclose(fit.pars.beta.values, [0.25, 0.4, 0.35], rtol=1e-3) # Fitted by gradients, which a value for each level used not to have

    # A duration for each level, whose rate (one over it) is not linear in it, is worked out exactly, as a single value is
    S, I, R = ss.states('S I R')
    dur = ss.par(ss.by(patch=[ss.days(8), ss.days(10), ss.days(12)]), bounds=[ss.days(4), ss.days(20)], name='dur')
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ dur, init_n=dict(I=ss.by(patch=[10, 10, 10])), name='sir')
    sir.stratify(patch=['a', 'b', 'c'])
    ld = ss.Sim(sir, unit='day', start=0, stop=60, data=ss.simdata(df), **kw).log_density(likelihood=dict(infection=ss.poisson()))
    x = ld.x0*1.3 # Away from where the numbers in the run were worked out
    fd = [(ld(x + 1e-6*e) - ld(x - 1e-6*e))/2e-6 for e in np.eye(3)]
    assert ld.names == ['dur[a]', 'dur[b]', 'dur[c]'] and np.allclose(ld.grad(x), fd, rtol=1e-5)
    run = ss.Sim(sir, unit='day', start=0, stop=60, data=ss.simdata(df), pars={'dur': ss.by(patch=[ss.days(v) for v in x])}, **kw).run()
    assert np.isclose(ld(x), ss.fit.loglik(run, likelihood=dict(infection=ss.poisson())) + sum(ss.fit.logprior(dur, v) for v in x), rtol=1e-9) # As a run of the sim gives it

    # A time series, fitted value by value
    days = [0, 20, 40, 60]
    def in_time(vals, data=None):
        beta = ss.par(ss.timeseries(days, list(vals), 'linear'), bounds=[0.05, 1], name='beta')
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ 0.1, init_n=dict(I=10), name='sir')
        return ss.Sim(sir, unit='day', start=0, stop=80, data=data, **kw)
    true = [0.4, 0.2, 0.3, 0.15]
    truth = in_time(true).run().results.sir.infection
    df = pd.DataFrame(dict(infection=[np.asarray(truth)[t-4:t+1].sum() for t in range(5, 81, 5)]), index=range(5, 81, 5))
    fit = in_time([0.3]*4, data=ss.simdata(df)).fit(likelihood=dict(infection=ss.poisson()), verbose=0)
    assert fit.method == 'l-bfgs-b' and list(fit.table.par) == ['beta[0]', 'beta[20]', 'beta[40]', 'beta[60]']
    assert isinstance(fit.pars.beta, ss.timeseries) and np.allclose(fit.pars.beta.values, true, rtol=1e-3)

    # A likelihood that is a function, which the log-likelihood in JAX does not cover: an error, unless Python in the loop is asked for
    poisson = lambda observed, model: observed*np.log(model) - model
    sim = in_time([0.3]*4, data=ss.simdata(df))
    with pytest.raises(ss.CapabilityError, match='nopython=False'):
        sim.fit(likelihood=dict(infection=poisson), method='l-bfgs-b', verbose=0)
    problem = ss.fit.Problem(sim, likelihood=dict(infection=poisson), nopython=False, verbose=0)
    value, grad = problem.with_gradient(problem.x0)
    assert np.isfinite(value) and np.all(np.isfinite(grad))
    return fit


if __name__ == '__main__':
    do_plot = True
    T = sc.timer()

    sims = test_jax_run()
    binom = test_jax_binomial()
    engine = test_jax_engine()
    grad = test_gradient()
    fit = test_fit()
    ident = test_identifiability()
    post = test_posterior()
    cross = test_crossing()
    post_data = test_posterior_data()
    msim = test_parallel_pars()
    several = test_several_values()

    T.toc()
