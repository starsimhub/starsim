"""
Test the ODE run in JAX: that it matches the kernels, its gradients, gradient-based fitting, and many runs with different values
"""

# %% Imports and settings
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.internals.fitting
from corpus import kernel_models as models, largest_diff

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
    with ss.options.context(backend='jax'):
        return sim.run()


@sc.timer()
def test_jax_run():
    sc.heading('Testing that a run in JAX matches the kernels')
    for key in ['seir', 'commute', 'malaria', 'tb', 'delay', 'met']:
        sims = [models[key]().run(), run_jax(models[key]())]
        diff = largest_diff(*sims)
        stepped = [note for note in sims[1].notes if 'JAX cannot' in note]
        assert diff < 1e-9, f'{key}: the run in JAX differs from the kernels by {diff}'
        assert not stepped, stepped # Including the TB model, whose transfer has a constant rate
    return


@sc.timer()
def test_gradient():
    sc.heading('Testing gradients of results against finite differences')
    sim = make_sir()
    grad = sim.gradient(results=['I', 'cases'], pars=['beta', 'dur'])
    assert not sim.initialized # The sim itself is not changed
    assert np.array_equal(sim.gradient('I', pars='sir.beta').beta.sir.I, grad.beta.sir.I) # A parameter named by its module too (it was an IndexError)
    for name,step in [('beta', 1e-6), ('dur', 1e-5)]:
        x0 = 0.4 if name == 'beta' else 8
        runs = [make_sir(**{name:x0 + sign*step}).run().results.sir for sign in [1, -1]]
        for key in ['I', 'cases']:
            fd = (runs[0][key].values - runs[1][key].values)/(2*step)
            diff = np.abs(grad[name].sir[key] - fd).max()/np.abs(fd).max()
            assert diff < 1e-5, f'd {key}/d {name} differs from finite differences by {diff:0.1e}'
    assert np.all(grad.beta.sir.I[1:20] > 0) # A higher beta means more infectious people early on (after the start, which it does not change)
    with pytest.raises(ValueError): # Gradients are for the ODE only
        sim.new(solver='gillespie').gradient()
    return grad


@sc.timer()
def test_fit():
    sc.heading('Testing that calibration uses the gradients')
    truth = make_sir(beta=0.5, dur=6, init=0.002).run().results.sir
    data = pd.DataFrame(dict(prevalence=truth.prevalence.values[::7]), index=truth.timevec[::7])
    fit = make_sir(data=data).fit(loss='rmse', verbose=False)
    assert fit.method == 'l-bfgs-b'
    assert np.allclose(fit.table.best, [0.5, 6, 0.002], rtol=1e-2)
    assert ss.fitting.default_method(fit.sim, False, n_pars=2)[0] == 'nelder-mead' # Faster than gradients for one or two parameters
    assert ss.fitting.default_method(fit.sim, True)[0] == 'l-bfgs-b' # A best fit, with a likelihood too: a posterior is asked for by name
    return fit


def make_coinfection(beta=0.15, data=None, dt=0.25, **kwargs):
    """ Two diseases that modify each other, so whose states are tracked jointly (see ss.solvers.JointStates), one with a transfer at a constant rate """
    beta = ss.par(beta, bounds=[0.05, 0.5], name='beta')
    S, I = ss.states('S I')
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), death=ss.death(I, rate=0.05*ss.by(art=[1.0, 0.2])), init_prev=0.05, name='hiv')
    hiv.stratify(art=ss.levels([False, True], states=['I']), switch=ss.transfer(dict(art=True), rate=ss.peryear(0.1)))
    S, I = ss.states('S I')
    syph = ss.Disease(infection=ss.Transition(S, I, rate=1.0 * I), cure=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.05, name='syph')
    coinfection = [
        ss.modify(hiv.rel_trans, multiply=3.0, who=syph.I, name='hiv_trans'),
        ss.modify(syph.rel_sus, multiply=2.0, who=hiv.I, name='syph_sus'),
    ]
    return ss.Sim(diseases=[hiv, syph], connectors=coinfection, start=2000, stop=2030, dt=ss.years(dt), data=data, **(dict(solver='ode', total_pop=20_000) | kw | kwargs))


@sc.timer()
def test_joint_states():
    sc.heading('Testing diseases tracked jointly: a whole run, many runs, in JAX, and differentiated')
    sim = make_coinfection()
    sim.init()
    assert sim._compiled_reason() is None # Modifiers on states, and a transfer at a constant rate, are worked out before the run
    sim.run()
    stepped = make_coinfection().run(until=2030) # Step by step
    jax = run_jax(make_coinfection())
    for other in [stepped, jax]:
        for key in ['hiv', 'syph']:
            assert np.allclose(other.results[key].I, sim.results[key].I, rtol=1e-9)
    assert sim.results.hiv.switch[1:].sum() > 0 # The transfer's flow is recorded

    # The diseases are one model, so every flow is worked out from the people at the start of the step
    fine = make_coinfection(dt=0.025).run()
    assert np.allclose(fine.results.syph.I[::10], sim.results.syph.I, rtol=1e-5)

    # Many runs at once take the diseases together too
    msim = ss.parallel(make_coinfection(), n_runs=2)
    assert np.allclose(msim.results.hiv.I.values[1], sim.results.hiv.I, rtol=1e-12)

    # The gradient of each disease's results, through the other
    grad = make_coinfection().gradient(results=['hiv.I', 'syph.I'], pars=['hiv.beta'])
    step = 1e-6
    runs = [make_coinfection(beta=0.15 + sign*step).run().results for sign in [1, -1]]
    for key in ['hiv', 'syph']:
        fd = (runs[0][key].I.values - runs[1][key].I.values)/(2*step)
        assert np.abs(grad['hiv.beta'][key].I - fd).max() < 1e-5*np.abs(fd).max()

    return sim


@sc.timer()
def test_parallel_pars():
    sc.heading('Testing many runs with different values, in one call')
    betas = [0.3, 0.4, 0.5]
    singles = [make_sir(beta=beta).run().results.sir.I.values for beta in betas]
    for jax in [False, True]: # With ss.options.backend = 'jax', the runs are taken in one call to JAX
        sim = make_sir()
        with ss.options.context(backend='jax' if jax else ss.options.backend):
            msim = ss.parallel(sim, pars=dict(beta=betas*2))
        n_I = msim.results.sir.I.values[:3]
        assert (msim.sims[0].diseases.sir.engine.jax_run is not None) == bool(jax)
        assert msim.run_pars.beta.tolist() == betas*2
        assert np.allclose(n_I, singles, rtol=1e-9), f'runs (jax={jax}) differ from single runs'
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
    df = pd.DataFrame([dict(time=t, patch=lab, infection=np.asarray(truth.select(patch=lab))[t:t+5].sum()) for lab in ['a', 'b', 'c'] for t in range(0, 56, 5)]).set_index('time') # Each over the five days from its time
    sim = by_place([0.3]*3, data=ss.SimData(df))
    ld = sim.log_density(likelihood=dict(infection=ss.poisson()))
    assert ld.names == ['beta[a]', 'beta[b]', 'beta[c]']
    fd = [(ld(ld.x0 + 1e-6*e) - ld(ld.x0 - 1e-6*e))/2e-6 for e in np.eye(3)]
    assert np.allclose(ld.grad(ld.x0), fd, rtol=1e-5)
    x = ld.x0*1.1
    out = {}
    for name,rmin,ckpt in [('forward', np.inf, np.inf), ('reverse', 0, np.inf), ('checkpointed', 0, 0)]:
        reverse_min, checkpoint_bytes = ss.fitting.reverse_min, ss.gradients.checkpoint_bytes
        ss.fitting.reverse_min, ss.gradients.checkpoint_bytes = rmin, ckpt
        try:
            problem = ss.fitting.Problem(sim, likelihood=dict(infection=ss.poisson()), verbose=0)
            out[name] = problem.with_gradient(x)
            if name == 'reverse': # With more substeps than at the start: taken again with more
                out['more'] = problem.with_gradient(x*2)
        finally:
            ss.fitting.reverse_min, ss.gradients.checkpoint_bytes = reverse_min, checkpoint_bytes
    for name in ['reverse', 'checkpointed']:
        assert np.isclose(out[name][0], out['forward'][0], rtol=1e-10) and np.allclose(out[name][1], out['forward'][1], rtol=1e-8), name
    assert np.isclose(out['more'][0], problem.objective(x*2), rtol=1e-8)
    fit = sim.fit(likelihood=dict(infection=ss.poisson()), method='l-bfgs-b', verbose=0)
    assert np.allclose(fit.pars.beta.values, [0.25, 0.4, 0.35], rtol=1e-3) # Fitted by gradients, which a value for each level used not to have

    # A time series, fitted value by value
    days = [0, 20, 40, 60]
    def in_time(vals, data=None):
        beta = ss.par(ss.timeseries(days, list(vals), 'linear'), bounds=[0.05, 1], name='beta')
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ 0.1, init_n=dict(I=10), name='sir')
        return ss.Sim(sir, unit='day', start=0, stop=80, data=data, **kw)
    true = [0.4, 0.2, 0.3, 0.15]
    truth = in_time(true).run().results.sir.infection
    df = pd.DataFrame(dict(infection=[np.asarray(truth)[t:t+5].sum() for t in range(0, 76, 5)]), index=range(0, 76, 5))
    fit = in_time([0.3]*4, data=ss.SimData(df)).fit(likelihood=dict(infection=ss.poisson()), verbose=0)
    assert fit.method == 'l-bfgs-b' and list(fit.table.par) == ['beta[0]', 'beta[20]', 'beta[40]', 'beta[60]']
    assert isinstance(fit.pars.beta, ss.timeseries) and np.allclose(fit.pars.beta.values, true, rtol=1e-3)

    return fit


if __name__ == '__main__':
    T = sc.timer()
    test_jax_run()
    test_gradient()
    test_fit()
    test_joint_states()
    test_parallel_pars()
    test_several_values()
    T.toc()
