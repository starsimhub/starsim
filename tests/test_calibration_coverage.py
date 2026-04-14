"""
Test calibration components and helpers for coverage improvement.
"""

import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss
import matplotlib.pyplot as plt
import pytest
import scipy.stats as sps
from starsim.calibration import (
    linear_interp, step_containing, linear_accum,
    BetaBinomial, Binomial, DirichletMultinomial, GammaPoisson, Normal,
)

n_agents = 500
do_plot = False
sc.options(interactive=False)

def make_sim():
    """ Create a default SIR sim for calibration tests """
    return ss.Sim(n_agents=n_agents, start=ss.date('2020-01-01'), stop=ss.date('2020-02-12'),
        dt=ss.days(1), diseases=ss.SIR(beta=ss.peryear(0.075), init_prev=ss.bernoulli(0.02)),
        networks=ss.RandomNet(n_contacts=ss.poisson(4)), verbose=0)

def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base sim with calib_pars """
    for k, pars in calib_pars.items():
        if k == 'rand_seed': sim.pars.rand_seed = pars
        elif k == 'beta': sim.pars.diseases.pars.beta = pars['value']
    return sim

dates = [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']]
exp_prev = pd.DataFrame({'n': [200, 197, 195], 'x': [30, 35, 10]}, index=pd.Index(dates, name='t'))
ext_prev = lambda sim: pd.DataFrame({'n': sim.results.n_alive, 'x': sim.results.sir.n_infected}, index=pd.Index(sim.results.timevec, name='t'))


@sc.timer()
def test_conform_helpers(do_plot=do_plot):
    """ Test linear_interp, step_containing, and linear_accum """
    sc.heading('Testing conform helpers...')
    actual = pd.DataFrame({'x': [0.0, 10.0, 20.0, 30.0]}, index=pd.Index([0.0, 1.0, 2.0, 3.0], name='t'))
    exp = pd.DataFrame({'x': [0, 0]}, index=pd.Index([0.5, 1.5], name='t'))
    assert np.isclose(linear_interp(exp, actual)['x'].iloc[0], 5.0, rtol=0.01), 'linear_interp failed'
    step_containing(exp, actual)  # Just check it runs

    exp_inc = pd.DataFrame({'x': [0, 0]}, index=pd.MultiIndex.from_arrays([[0.0, 2.0], [2.0, 4.0]], names=['t', 't1']))
    actual_inc = pd.DataFrame({'x': [10.0]*5}, index=pd.Index(range(5), name='t', dtype=float))
    assert np.isclose(linear_accum(exp_inc, actual_inc)['x'].iloc[0], 20.0, rtol=0.01), 'linear_accum failed'
    return True


@sc.timer()
def test_validate_conform(do_plot=do_plot):
    """ Test _validate_conform accepts valid and rejects invalid inputs """
    sc.heading('Testing validate_conform...')
    for c in ['prevalent', 'incident', 'step_containing', 'none']:
        BetaBinomial(name='t', expected=exp_prev, extract_fn=ext_prev, conform=c)
    BetaBinomial(name='t', expected=exp_prev, extract_fn=ext_prev, conform=lambda e, a: a)
    with pytest.raises(ValueError):
        BetaBinomial(name='t', expected=exp_prev, extract_fn=ext_prev, conform='bad')
    with pytest.raises(Exception):
        BetaBinomial(name='t', expected=exp_prev, extract_fn=ext_prev, conform=123)
    return True


@sc.timer()
def test_nll_subclasses(do_plot=do_plot):
    """ Test compute_nll for BetaBinomial, Binomial, Normal, GammaPoisson, DirichletMultinomial """
    sc.heading('Testing NLL subclasses...')
    act = pd.DataFrame({'n': [500]*3, 'x': [28, 33, 12], 't': dates, 'rand_seed': [0]*3}).set_index('rand_seed')

    # BetaBinomial
    bb = BetaBinomial(name='bb', expected=exp_prev, extract_fn=ext_prev, conform='prevalent')
    nll = bb.compute_nll(exp_prev, act)
    assert np.isclose(nll[0], -sps.betabinom.logpmf(k=30, n=200, a=29, b=473), rtol=1e-10), 'BetaBinomial NLL mismatch'

    # Binomial
    bn = Binomial(name='bn', expected=exp_prev, extract_fn=ext_prev, conform='prevalent')
    act_bn = pd.DataFrame({'n': [500]*3, 'x': [75, 80, 30], 't': dates, 'rand_seed': [0]*3}).set_index('rand_seed')
    nll_bn = bn.compute_nll(exp_prev, act_bn)
    assert np.isclose(nll_bn[0], -sps.binom.logpmf(k=30, n=200, p=75/500), rtol=1e-10), 'Binomial NLL mismatch'
    assert np.allclose(Binomial.get_p(pd.DataFrame({'p': [0.1], 'x': [5], 'n': [50]})), [0.1]), 'get_p failed'

    # Normal (fixed, ML, array sigma2, compute_var)
    exp_n = pd.DataFrame({'x': [0.13, 0.16, 0.06]}, index=pd.Index(dates, name='t'))
    act_n = pd.DataFrame({'x': [0.12, 0.18, 0.05], 't': dates, 'rand_seed': [0]*3}).set_index('rand_seed')
    nf = Normal(name='nf', expected=exp_n, extract_fn=lambda s: None, conform='prevalent', sigma2=0.05)
    assert np.isclose(nf.compute_nll(exp_n, act_n)[0], -sps.norm.logpdf(0.13, 0.12, np.sqrt(0.05)), rtol=1e-10), 'Normal fixed NLL mismatch'
    assert np.all(np.isfinite(Normal(name='nm', expected=exp_n, extract_fn=lambda s: None, conform='prevalent').compute_nll(exp_n, act_n))), 'Normal ML NLL failed'
    assert np.all(np.isfinite(Normal(name='na', expected=exp_n, extract_fn=lambda s: None, conform='prevalent', sigma2=np.array([.01,.05,.1])).compute_nll(exp_n, act_n))), 'Normal array NLL failed'
    assert np.isclose(nf.compute_var(pd.Series([1.0, 2.0, 3.0]), 2.0), 2.0/3.0, rtol=1e-10), 'compute_var mismatch'

    # GammaPoisson
    exp_gp = pd.DataFrame({'n': [100, 27, 54], 'x': [740, 325, 200],
        't': [ss.date(d) for d in ['2020-01-07', '2020-01-14', '2020-01-27']],
        't1': [ss.date(d) for d in ['2020-01-08', '2020-01-15', '2020-01-29']]}).set_index(['t', 't1'])
    gp = GammaPoisson(name='gp', expected=exp_gp, extract_fn=lambda s: None, conform='incident')
    act_gp = pd.DataFrame({'n': [110, 30, 60], 'x': [700, 300, 180],
        't': exp_gp.index.get_level_values('t'), 't1': exp_gp.index.get_level_values('t1'), 'rand_seed': [0]*3}).set_index('rand_seed')
    assert np.isclose(gp.compute_nll(exp_gp, act_gp)[0], -sps.nbinom.logpmf(k=740, n=701, p=111/211), rtol=1e-10), 'GammaPoisson NLL mismatch'
    with pytest.raises(AssertionError):  # Int validation
        GammaPoisson(name='f', expected=pd.DataFrame({'n': [100], 'x': [10.5], 't': [dates[0]], 't1': [dates[1]]}).set_index(['t', 't1']), extract_fn=lambda s: None, conform='incident')

    # DirichletMultinomial
    dm_d = [ss.date('2020-01-07'), ss.date('2020-01-21')]
    exp_dm = pd.DataFrame({'x_0': [40, 60], 'x_1': [40, 60], 'x_2': [40, 60]}, index=pd.Index(dm_d, name='t'))
    dm = DirichletMultinomial(name='dm', expected=exp_dm, extract_fn=lambda s: None, conform='none')
    act_dm = pd.DataFrame({'x_0': [35, 55], 'x_1': [45, 65], 'x_2': [38, 58], 't': dm_d, 'rand_seed': [0]*2}).set_index('rand_seed')
    assert np.isclose(dm.compute_nll(exp_dm, act_dm)[0], -sps.dirichlet_multinomial.logpmf(x=np.array([40,40,40]), n=120, alpha=np.array([36.,46.,39.])), rtol=1e-10), 'DM NLL mismatch'
    return True


@sc.timer()
def test_component_eval(do_plot=do_plot):
    """ Test eval with single sim, MultiSim, include_fn, weight, and repr """
    sc.heading('Testing component eval...')
    sim = make_sim(); sim.run()
    ms = ss.MultiSim(make_sim(), n_runs=2, parallel=False); ms.run()

    comp = BetaBinomial(name='My Comp', expected=exp_prev, extract_fn=ext_prev, conform='prevalent')
    assert np.isfinite(comp.eval(sim)) and comp.actual is not None, 'Single sim eval failed'
    assert np.isfinite(comp.eval(ms)) and len(comp.actual.index.unique()) == 2, 'MultiSim eval failed'

    comp_excl = BetaBinomial(name='x', expected=exp_prev, extract_fn=ext_prev, conform='prevalent', include_fn=lambda s: False)
    assert comp_excl.eval(ms) == np.inf, 'Expected inf when all excluded'

    nll1 = comp.eval(sim)
    comp_w2 = BetaBinomial(name='w2', expected=exp_prev, extract_fn=ext_prev, conform='prevalent', weight=2.0)
    assert np.isclose(comp_w2.eval(sim), 2.0 * nll1, rtol=1e-10), 'Weight scaling failed'
    assert BetaBinomial(name='w0', expected=exp_prev, extract_fn=ext_prev, conform='prevalent', weight=0).eval(sim) == 0, 'Zero weight should return 0'
    assert 'My Comp' in repr(comp), 'Expected name in repr'
    return nll1


@sc.timer()
def test_calibration_orchestration(do_plot=do_plot):
    """ Test Calibration: calibrate, check_fit, to_df, to_json, component-based """
    sc.heading('Testing calibration orchestration...')
    cpars = dict(beta=dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True))
    eval_fn = lambda sim, expected: sum((s.results.sir.prevalence[np.searchsorted(s.results.timevec, expected[0], side='left')] - expected[1])**2
        for s in (sim.sims if isinstance(sim, ss.MultiSim) else [sim]))

    calib = ss.Calibration(calib_pars=cpars, sim=make_sim(), build_fn=build_sim, reseed=True,
        eval_fn=eval_fn, eval_kw=dict(expected=(ss.date('2020-01-12'), 0.13)),
        total_trials=10, die=True, debug=True, verbose=False)
    calib.calibrate()
    assert calib.calibrated and 'beta' in calib.best_pars, 'Calibration failed'
    assert isinstance(calib.check_fit(do_plot=False), (bool, np.bool_)), 'check_fit should return bool'
    assert len(calib.to_df(top_k=3)) <= 3, 'to_df failed'
    assert 'pars' in calib.to_json()[0], 'to_json failed'

    # Component-based
    comp = ss.BetaBinomial(name='bb', conform='step_containing', expected=exp_prev, extract_fn=ext_prev)
    calib2 = ss.Calibration(calib_pars=cpars, sim=make_sim(), build_fn=build_sim, reseed=True,
        components=[comp], total_trials=10, die=True, debug=True, verbose=False)
    calib2.calibrate()
    assert calib2.calibrated, 'Component calibration failed'
    return calib


@sc.timer()
def test_calibration_db_options(do_plot=do_plot):
    """ Test keep_db and continue_db options """
    sc.heading('Testing calibration DB options...')
    import tempfile, os
    db_path = os.path.join(tempfile.mkdtemp(), 'test.db')
    kw = dict(calib_pars=dict(beta=dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True)),
        sim=make_sim(), build_fn=build_sim, reseed=True, eval_fn=lambda sim, **kw: np.random.rand(),
        total_trials=5, die=True, debug=True, verbose=False, db_name=db_path)
    ss.Calibration(**kw, keep_db=True, continue_db=False).calibrate()
    assert os.path.exists(db_path), 'DB should exist with keep_db=True'
    c2 = ss.Calibration(**kw, keep_db=False, continue_db=True)
    c2.calibrate()
    assert c2.calibrated, 'continue_db calibration failed'
    if os.path.exists(db_path): os.remove(db_path)
    return c2


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    test_conform_helpers(do_plot=do_plot)
    test_validate_conform(do_plot=do_plot)
    test_nll_subclasses(do_plot=do_plot)
    test_component_eval(do_plot=do_plot)
    test_calibration_orchestration(do_plot=do_plot)
    test_calibration_db_options(do_plot=do_plot)
    T.toc()
    if do_plot: plt.show()
