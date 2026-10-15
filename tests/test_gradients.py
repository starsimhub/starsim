"""
Test the ODE run in JAX: that it matches the kernels, its gradients, gradient-based fitting, and many runs with different values
"""

# %% Imports and settings
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from test_kernels import models, largest_diff

pytest.importorskip('jax')
kw = dict(verbose=0)


def make_sir(beta=0.4, dur=8, init=0.01, data=None, **kwargs):
    """ An SIR model whose infection rate has a function in it, and with reported cases """
    beta = ss.par(beta, bounds=[0.1, 1], name='beta')
    dur  = ss.par(ss.days(dur), bounds=[2, 20], name='dur')
    init = ss.par(init, bounds=[1e-4, 0.1], name='init')
    S, I, R = ss.states(3)
    infection = S >> I @ beta * np.exp(-I.frac) * I # Fewer contacts when many are ill
    recovery  = I >> R @ dur
    cases     = ss.outcome(0.5 * infection, delay=ss.days(3))
    sir = ss.Disease(infection, recovery, cases, init=dict(I=init), name='sir')
    return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-05-01', data=data, **kw, **kwargs)


@sc.timer()
def test_jax_run():
    sc.heading('Testing that a run in JAX matches the kernels')
    for key,(make,pars) in models.items():
        sims = [ss.Sim(make(), **sc.mergedicts(pars, kw, integrator=integrator)).run() for integrator in ['rk4', 'jax']]
        diff = largest_diff(*sims)
        stepped = [note for note in sims[1].notes if 'integrator="jax" cannot' in note]
        print(f'{key:8s} {diff:0.1e} {"(run by the kernels: " + stepped[0] + ")" if stepped else ""}')
        assert diff < 1e-9, f'{key}: the run in JAX differs from the kernels by {diff}'
        assert bool(stepped) == (key == 'tb') # The TB model has a transfer, so is stepped by the kernels
    return sims


@sc.timer()
def test_gradient():
    sc.heading('Testing gradients of results against finite differences')
    sim = make_sir()
    grad = sim.gradient(results=['n_I', 'cases'], pars=['beta', 'dur'])
    assert not sim.initialized # The sim itself is not changed
    for name,step in [('beta', 1e-6), ('dur', 1e-5)]:
        x0 = 0.4 if name == 'beta' else 8
        runs = [make_sir(**{name:x0 + sign*step}).run().results.sir for sign in [1, -1]]
        for key in ['n_I', 'cases']:
            fd = (runs[0][key].values - runs[1][key].values)/(2*step)
            diff = np.abs(grad[name].sir[key] - fd).max()/np.abs(fd).max()
            print(f'd {key}/d {name}: largest {np.abs(fd).max():0.4g}, differs from finite differences by {diff:0.1e}')
            assert diff < 1e-5
    assert np.all(grad.beta.sir.n_I[1:20] > 0) # A higher beta means more infectious people early on (after the start, which it does not change)
    with pytest.raises(ValueError): # Gradients are for the ODE only
        sim.new(solver='tau').gradient()
    return grad


@sc.timer()
def test_fit():
    sc.heading('Testing that calibration uses the gradients')
    truth = make_sir(beta=0.5, dur=6, init=0.002).run().results.sir
    data = pd.DataFrame(dict(prevalence=truth.prevalence.values[::7]), index=truth.timevec[::7])
    fit = make_sir(data=data).calibrate(gof='rmse', verbose=True)
    print(fit.table)
    assert fit.method == 'l-bfgs-b'
    assert np.allclose(fit.table.best, [0.5, 6, 0.002], rtol=1e-2)
    return fit


@sc.timer()
def test_posterior():
    sc.heading('Testing a posterior from NUTS')
    pytest.importorskip('blackjax')
    truth = make_sir(beta=0.5, dur=6).run().results.sir
    weekly = np.add.reduceat(truth.cases.values[1:], np.arange(0, len(truth.cases) - 1, 7))[:-1] # A flow is compared with its total since the last data point
    data = pd.DataFrame(dict(cases=np.random.default_rng(1).poisson(weekly)), index=truth.timevec[7::7][:len(weekly)])
    fit = make_sir(dur=6, data=data).calibrate(pars=['beta'], likelihood=dict(cases=ss.poisson()), draws=100, warmup=100, chains=2, rand_seed=1, verbose=True) # NUTS, the default when every series has a likelihood
    fit.summarize()
    table = fit.table.set_index('par')
    assert fit.method == 'nuts'
    assert len(fit.samples) == 200 and list(fit.samples.columns) == ['chain', 'draw', 'beta']
    assert table.lower.beta < 0.5 < table.upper.beta # The posterior covers the value that made the data
    assert np.all(table.rhat < 1.1)
    assert fit.band['cases'].shape == (2, len(truth.cases))
    with pytest.raises(ss.CodedError, match='E1303'): # A posterior needs a likelihood
        make_sir(data=data).calibrate(pars=['beta'], method='nuts')
    return fit


@sc.timer()
def test_parallel_pars():
    sc.heading('Testing many runs with different values, in one call')
    betas = [0.3, 0.4, 0.5]
    singles = [make_sir(beta=beta).run().results.sir.n_I.values for beta in betas]
    for integrator in ['rk4', 'jax']:
        msim = ss.parallel(make_sir(integrator=integrator), pars=dict(beta=betas))
        n_I = msim.results.sir.n_I.values
        assert (msim.sims[0].diseases.sir.engine.jax_run is not None) == (integrator == 'jax') # The runs were taken in one call to JAX
        assert msim.run_pars.beta.tolist() == betas
        assert np.allclose(n_I, singles, rtol=1e-9), f'Runs with integrator={integrator} differ from single runs'
    msim = ss.parallel(make_sir(), pars=[dict(beta=0.3), dict(beta=0.5, dur=ss.days(4))], n_runs=2) # A list of sets, each run twice
    assert len(msim.results.sir.n_I.values) == 4
    return msim


if __name__ == '__main__':
    do_plot = True
    T = sc.timer()

    sims = test_jax_run()
    grad = test_gradient()
    fit = test_fit()
    post = test_posterior()
    msim = test_parallel_pars()

    T.toc()
