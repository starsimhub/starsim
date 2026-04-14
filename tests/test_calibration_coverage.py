"""
Test calibration components and helper functions for coverage improvement.

Tests CalibComponent subclasses (BetaBinomial, Binomial, DirichletMultinomial,
GammaPoisson, Normal), their compute() and plot methods, conform helpers, and
the Calibration orchestration class.
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
    CalibComponent, BetaBinomial, Binomial,
    DirichletMultinomial, GammaPoisson, Normal,
)

n_agents = 500
do_plot = False
sc.options(interactive=False)


#%% Helper functions

def make_sim():
    """ Create a default SIR sim for calibration tests """
    sir = ss.SIR(
        beta = ss.peryear(0.075),
        init_prev = ss.bernoulli(0.02),
    )
    random = ss.RandomNet(n_contacts=ss.poisson(4))
    sim = ss.Sim(
        n_agents = n_agents,
        start = ss.date('2020-01-01'),
        stop = ss.date('2020-02-12'),
        dt = ss.days(1),
        diseases = sir,
        networks = random,
        verbose = 0,
    )
    return sim


def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """
    sir = sim.pars.diseases
    for k, pars in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue
        v = pars['value']
        if k == 'beta':
            sir.pars.beta = v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')
    return sim


def make_expected_prevalent():
    """ Create expected data for prevalent-type components """
    dates = [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']]
    expected = pd.DataFrame({
        'n': [200, 197, 195],
        'x': [30, 35, 10],
    }, index=pd.Index(dates, name='t'))
    return expected


def make_expected_incident():
    """ Create expected data for incident-type components """
    expected = pd.DataFrame({
        'n': [100, 27, 54],
        'x': [740, 325, 200],
        't': [ss.date(d) for d in ['2020-01-07', '2020-01-14', '2020-01-27']],
        't1': [ss.date(d) for d in ['2020-01-08', '2020-01-15', '2020-01-29']],
    }).set_index(['t', 't1'])
    return expected


def make_extract_fn_prevalent():
    """ Create extract function for prevalent data """
    def extract(sim):
        return pd.DataFrame({
            'n': sim.results.n_alive,
            'x': sim.results.sir.n_infected,
        }, index=pd.Index(sim.results.timevec, name='t'))
    return extract


def make_extract_fn_incident():
    """ Create extract function for incident data """
    def extract(sim):
        return pd.DataFrame({
            'x': sim.results.sir.new_infections,
            'n': sim.results.n_alive * sim.t.dt_year,
        }, index=pd.Index(sim.results.timevec, name='t'))
    return extract


#%% Tests for helper functions (conform)

@sc.timer()
def test_linear_interp(do_plot=do_plot):
    """ Test that linear_interp interpolates prevalent data correctly """
    sc.heading('Testing linear_interp...')

    # Create actual data at integer timepoints
    actual = pd.DataFrame({
        'x': [0.0, 10.0, 20.0, 30.0],
    }, index=pd.Index([0.0, 1.0, 2.0, 3.0], name='t'))

    # Expected data at midpoints
    expected = pd.DataFrame({
        'x': [999, 999],  # Values don't matter, only index used
    }, index=pd.Index([0.5, 1.5], name='t'))

    result = linear_interp(expected, actual)

    rtol = 0.01  # Exact interpolation, tight tolerance
    assert np.isclose(result['x'].iloc[0], 5.0, rtol=rtol), \
        f'Expected interpolated value ~5.0 at t=0.5, got {result["x"].iloc[0]}'
    assert np.isclose(result['x'].iloc[1], 15.0, rtol=rtol), \
        f'Expected interpolated value ~15.0 at t=1.5, got {result["x"].iloc[1]}'
    return result


@sc.timer()
def test_step_containing(do_plot=do_plot):
    """ Test that step_containing finds the correct step for prevalent data """
    sc.heading('Testing step_containing...')

    actual = pd.DataFrame({
        'x': [100.0, 200.0, 300.0, 400.0],
    }, index=pd.Index([0.0, 1.0, 2.0, 3.0], name='t'))

    expected = pd.DataFrame({
        'x': [999, 999],
    }, index=pd.Index([0.5, 2.5], name='t'))

    result = step_containing(expected, actual)

    # searchsorted with 'left' should find the step at or after the query
    assert result['x'].iloc[0] == 100.0 or result['x'].iloc[0] == 200.0, \
        f'Expected step_containing to find step near t=0.5, got {result["x"].iloc[0]}'
    return result


@sc.timer()
def test_linear_accum(do_plot=do_plot):
    """ Test that linear_accum accumulates incident data correctly """
    sc.heading('Testing linear_accum...')

    # Create actual data: constant flow of 10 per step
    actual = pd.DataFrame({
        'x': [10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=pd.Index([0.0, 1.0, 2.0, 3.0, 4.0], name='t'))

    # Expected: accumulation between t=0..2 and t=2..4
    expected = pd.DataFrame({
        'x': [999, 999],
    }, index=pd.MultiIndex.from_arrays(
        [[0.0, 2.0], [2.0, 4.0]], names=['t', 't1']
    ))

    result = linear_accum(expected, actual)

    rtol = 0.01  # Exact cumsum differencing
    assert np.isclose(result['x'].iloc[0], 20.0, rtol=rtol), \
        f'Expected accumulated value ~20.0 for t=0..2, got {result["x"].iloc[0]}'
    assert np.isclose(result['x'].iloc[1], 20.0, rtol=rtol), \
        f'Expected accumulated value ~20.0 for t=2..4, got {result["x"].iloc[1]}'
    return result


#%% Tests for validate_conform

@sc.timer()
def test_validate_conform(do_plot=do_plot):
    """ Test that _validate_conform accepts valid inputs and rejects invalid ones """
    sc.heading('Testing validate_conform...')

    expected = make_expected_prevalent()
    extract_fn = make_extract_fn_prevalent()

    # Valid string conforms
    for conform_str in ['prevalent', 'incident', 'step_containing', 'none']:
        comp = BetaBinomial(
            name='test', expected=expected,
            extract_fn=extract_fn, conform=conform_str,
        )
        assert comp is not None, f'Expected valid conform "{conform_str}" to be accepted'

    # Valid callable conform
    comp = BetaBinomial(
        name='test', expected=expected,
        extract_fn=extract_fn, conform=lambda e, a: a,
    )
    assert comp is not None, 'Expected callable conform to be accepted'

    # Invalid string conform
    with pytest.raises(ValueError):
        BetaBinomial(
            name='test', expected=expected,
            extract_fn=extract_fn, conform='invalid_conform',
        )

    # Invalid type conform
    with pytest.raises(Exception):
        BetaBinomial(
            name='test', expected=expected,
            extract_fn=extract_fn, conform=123,
        )

    return comp


#%% Tests for CalibComponent subclasses: compute_nll

@sc.timer()
def test_betabinomial_nll(do_plot=do_plot):
    """ Test BetaBinomial compute_nll returns valid negative log-likelihoods """
    sc.heading('Testing BetaBinomial NLL...')

    expected = make_expected_prevalent()
    comp = BetaBinomial(
        name='test_bb', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
    )

    # Create actual data mimicking simulation output
    actual = pd.DataFrame({
        'n': [500, 500, 500],
        'x': [28, 33, 12],
        't': expected.index,
        'rand_seed': [0, 0, 0],
    }).set_index('rand_seed')

    nll = comp.compute_nll(expected, actual)

    assert len(nll) == 3, f'Expected 3 NLL values, got {len(nll)}'
    assert np.all(np.isfinite(nll)), f'Expected all finite NLL values, got {nll}'
    assert np.all(nll >= 0), f'Expected all non-negative NLL values (negative log-likelihood), got {nll}'

    # Verify against scipy directly for the first row
    e_n, e_x = 200, 30
    a_n, a_x = 500, 28
    expected_nll = -sps.betabinom.logpmf(k=e_x, n=e_n, a=a_x+1, b=a_n-a_x+1)
    rtol = 1e-10  # Numerical precision
    assert np.isclose(nll[0], expected_nll, rtol=rtol), \
        f'BetaBinomial NLL mismatch: got {nll[0]}, expected {expected_nll}'

    return nll


@sc.timer()
def test_binomial_nll(do_plot=do_plot):
    """ Test Binomial compute_nll with both explicit p and computed p """
    sc.heading('Testing Binomial NLL...')

    expected = make_expected_prevalent()
    comp = Binomial(
        name='test_binom', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
    )

    # Actual data with n and x (p computed as x/n)
    actual = pd.DataFrame({
        'n': [500, 500, 500],
        'x': [75, 80, 30],
        't': expected.index,
        'rand_seed': [0, 0, 0],
    }).set_index('rand_seed')

    nll = comp.compute_nll(expected, actual)
    assert len(nll) == 3, f'Expected 3 NLL values, got {len(nll)}'
    assert np.all(np.isfinite(nll)), f'Expected all finite NLL values, got {nll}'

    # Verify first row: p = 75/500 = 0.15
    e_n, e_x = 200, 30
    p = 75 / 500
    expected_nll = -sps.binom.logpmf(k=e_x, n=e_n, p=p)
    rtol = 1e-10
    assert np.isclose(nll[0], expected_nll, rtol=rtol), \
        f'Binomial NLL mismatch: got {nll[0]}, expected {expected_nll}'

    return nll


@sc.timer()
def test_binomial_get_p(do_plot=do_plot):
    """ Test Binomial.get_p with explicit p column and computed p """
    sc.heading('Testing Binomial.get_p...')

    # With explicit p
    df_with_p = pd.DataFrame({'p': [0.1, 0.2], 'x': [5, 10], 'n': [100, 100]})
    p = Binomial.get_p(df_with_p)
    assert np.allclose(p, [0.1, 0.2]), f'Expected explicit p=[0.1, 0.2], got {p}'

    # Without p, computed as x/n
    df_without_p = pd.DataFrame({'x': [5, 10], 'n': [100, 100]})
    p = Binomial.get_p(df_without_p)
    rtol = 1e-10
    assert np.isclose(p.iloc[0], 0.05, rtol=rtol), f'Expected p=0.05, got {p.iloc[0]}'
    assert np.isclose(p.iloc[1], 0.10, rtol=rtol), f'Expected p=0.10, got {p.iloc[1]}'

    return p


@sc.timer()
def test_normal_nll(do_plot=do_plot):
    """ Test Normal compute_nll with user-provided and ML-estimated variance """
    sc.heading('Testing Normal NLL...')

    dates = [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']]
    expected = pd.DataFrame({
        'x': [0.13, 0.16, 0.06],
    }, index=pd.Index(dates, name='t'))

    # With user-provided scalar sigma2
    comp_fixed = Normal(
        name='test_normal_fixed', expected=expected,
        extract_fn=lambda sim: None, conform='prevalent',
        sigma2=0.05,
    )

    actual = pd.DataFrame({
        'x': [0.12, 0.18, 0.05],
        't': dates,
        'rand_seed': [0, 0, 0],
    }).set_index('rand_seed')

    nll_fixed = comp_fixed.compute_nll(expected, actual)
    assert len(nll_fixed) == 3, f'Expected 3 NLL values, got {len(nll_fixed)}'
    assert np.all(np.isfinite(nll_fixed)), f'Expected all finite NLL values, got {nll_fixed}'

    # Verify first row against scipy
    expected_nll = -sps.norm.logpdf(x=0.13, loc=0.12, scale=np.sqrt(0.05))
    rtol = 1e-10
    assert np.isclose(nll_fixed[0], expected_nll, rtol=rtol), \
        f'Normal NLL (fixed sigma2) mismatch: got {nll_fixed[0]}, expected {expected_nll}'

    # With ML-estimated variance (sigma2=None)
    comp_ml = Normal(
        name='test_normal_ml', expected=expected,
        extract_fn=lambda sim: None, conform='prevalent',
    )
    nll_ml = comp_ml.compute_nll(expected, actual)
    assert len(nll_ml) == 3, f'Expected 3 NLL values from ML estimation, got {len(nll_ml)}'
    assert np.all(np.isfinite(nll_ml)), f'Expected all finite NLL values from ML estimation'

    return nll_fixed, nll_ml


@sc.timer()
def test_normal_compute_var(do_plot=do_plot):
    """ Test Normal.compute_var computes ML variance correctly """
    sc.heading('Testing Normal.compute_var...')

    comp = Normal(
        name='test', expected=pd.DataFrame({'x': [1.0]}, index=pd.Index([0.0], name='t')),
        extract_fn=lambda sim: None, conform='prevalent',
    )

    # Known case: expected=[1, 2, 3], actual=2 -> diffs=[-1, 0, 1], SSE=2, N=3, var=2/3
    expected_x = pd.Series([1.0, 2.0, 3.0])
    actual_x = 2.0
    var = comp.compute_var(expected_x, actual_x)
    rtol = 1e-10
    assert np.isclose(var, 2.0/3.0, rtol=rtol), \
        f'Expected variance=2/3, got {var}'

    # Scalar case: single value
    var_single = comp.compute_var(5.0, 3.0)
    assert np.isclose(var_single, 4.0, rtol=rtol), \
        f'Expected variance=4.0 for (5-3)^2/1, got {var_single}'

    return var


@sc.timer()
def test_normal_array_sigma2(do_plot=do_plot):
    """ Test Normal with array-valued sigma2 (per-timepoint variance) """
    sc.heading('Testing Normal with array sigma2...')

    dates = [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']]
    expected = pd.DataFrame({
        'x': [0.13, 0.16, 0.06],
    }, index=pd.Index(dates, name='t'))

    sigma2_arr = np.array([0.01, 0.05, 0.10])
    comp = Normal(
        name='test_normal_arr', expected=expected,
        extract_fn=lambda sim: None, conform='prevalent',
        sigma2=sigma2_arr,
    )

    actual = pd.DataFrame({
        'x': [0.12, 0.18, 0.05],
        't': dates,
        'rand_seed': [0, 0, 0],
    }).set_index('rand_seed')

    nll = comp.compute_nll(expected, actual)
    assert len(nll) == 3, f'Expected 3 NLL values, got {len(nll)}'
    assert np.all(np.isfinite(nll)), f'Expected all finite NLL values, got {nll}'
    return nll


@sc.timer()
def test_gammapoisson_nll(do_plot=do_plot):
    """ Test GammaPoisson compute_nll returns valid negative log-likelihoods """
    sc.heading('Testing GammaPoisson NLL...')

    expected = make_expected_incident()
    comp = GammaPoisson(
        name='test_gp', expected=expected,
        extract_fn=make_extract_fn_incident(), conform='incident',
    )

    # Create actual incident data with matching multi-index
    actual = pd.DataFrame({
        'n': [110, 30, 60],
        'x': [700, 300, 180],
        't': expected.index.get_level_values('t'),
        't1': expected.index.get_level_values('t1'),
        'rand_seed': [0, 0, 0],
    }).set_index('rand_seed')

    nll = comp.compute_nll(expected, actual)
    assert len(nll) == 3, f'Expected 3 NLL values, got {len(nll)}'
    assert np.all(np.isfinite(nll)), f'Expected all finite NLL values, got {nll}'
    assert np.all(nll >= 0), f'Expected all non-negative NLL values, got {nll}'

    # Verify first row
    e_n, e_x = 100, 740
    a_n, a_x = 110, 700
    T = e_n
    beta = 1 + a_n
    expected_nll = -sps.nbinom.logpmf(k=e_x, n=1+a_x, p=beta/(beta+T))
    rtol = 1e-10
    assert np.isclose(nll[0], expected_nll, rtol=rtol), \
        f'GammaPoisson NLL mismatch: got {nll[0]}, expected {expected_nll}'

    return nll


@sc.timer()
def test_gammapoisson_int_validation(do_plot=do_plot):
    """ Test that GammaPoisson requires integer n and x columns """
    sc.heading('Testing GammaPoisson integer validation...')

    # Float x should fail
    expected_float = pd.DataFrame({
        'n': [100],
        'x': [10.5],  # Not integer
        't': [ss.date('2020-01-07')],
        't1': [ss.date('2020-01-08')],
    }).set_index(['t', 't1'])

    with pytest.raises(AssertionError):
        GammaPoisson(
            name='test_gp_fail', expected=expected_float,
            extract_fn=make_extract_fn_incident(), conform='incident',
        )

    return True


@sc.timer()
def test_dirichletmultinomial_nll(do_plot=do_plot):
    """ Test DirichletMultinomial compute_nll returns valid NLLs """
    sc.heading('Testing DirichletMultinomial NLL...')

    dates = [ss.date('2020-01-07'), ss.date('2020-01-21')]
    expected = pd.DataFrame({
        'x_0': [40, 60],
        'x_1': [40, 60],
        'x_2': [40, 60],
    }, index=pd.Index(dates, name='t'))

    comp = DirichletMultinomial(
        name='test_dm', expected=expected,
        extract_fn=lambda sim: None, conform='none',
    )

    actual = pd.DataFrame({
        'x_0': [35, 55],
        'x_1': [45, 65],
        'x_2': [38, 58],
        't': [dates[0], dates[1]],
        'rand_seed': [0, 0],
    }).set_index('rand_seed')

    nll = comp.compute_nll(expected, actual)
    assert len(nll) == 2, f'Expected 2 NLL values, got {len(nll)}'
    assert np.all(np.isfinite(nll)), f'Expected all finite NLL values, got {nll}'
    assert np.all(nll >= 0), f'Expected all non-negative NLL values, got {nll}'

    # Verify first row
    e_x = np.array([40, 40, 40])
    a_x = np.array([35.0, 45.0, 38.0])
    n = e_x.sum()
    expected_nll = -sps.dirichlet_multinomial.logpmf(x=e_x, n=n, alpha=a_x+1)
    rtol = 1e-10
    assert np.isclose(nll[0], expected_nll, rtol=rtol), \
        f'DirichletMultinomial NLL mismatch: got {nll[0]}, expected {expected_nll}'

    return nll


#%% Tests for CalibComponent eval with a real sim

@sc.timer()
def test_component_eval_single_sim(do_plot=do_plot):
    """ Test CalibComponent.eval with a single sim extracts and conforms data """
    sc.heading('Testing component eval with single sim...')

    sim = make_sim()
    sim.run()

    expected = make_expected_prevalent()
    comp = BetaBinomial(
        name='test_eval', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
    )

    nll = comp.eval(sim)
    assert np.isfinite(nll), f'Expected finite NLL from eval, got {nll}'
    assert comp.actual is not None, 'Expected actual data to be stored after eval'
    assert comp.nll is not None, 'Expected nll to be stored after eval'

    return nll


@sc.timer()
def test_component_eval_multisim(do_plot=do_plot):
    """ Test CalibComponent.eval with a MultiSim combines seeds correctly """
    sc.heading('Testing component eval with MultiSim...')

    sim = make_sim()
    ms = ss.MultiSim(sim, n_runs=3, parallel=False)
    ms.run()

    expected = make_expected_prevalent()
    comp = BetaBinomial(
        name='test_eval_ms', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
    )

    nll = comp.eval(ms)
    assert np.isfinite(nll), f'Expected finite NLL from MultiSim eval, got {nll}'

    seeds = comp.actual.index.unique()
    assert len(seeds) == 3, f'Expected 3 unique seeds from 3-sim MultiSim, got {len(seeds)}'

    return nll


@sc.timer()
def test_component_include_fn(do_plot=do_plot):
    """ Test CalibComponent include_fn filters out sims """
    sc.heading('Testing component include_fn...')

    sim = make_sim()
    ms = ss.MultiSim(sim, n_runs=3, parallel=False)
    ms.run()

    expected = make_expected_prevalent()
    # include_fn that rejects all sims
    comp = BetaBinomial(
        name='test_include', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
        include_fn=lambda s: False,
    )

    nll = comp.eval(ms)
    assert nll == np.inf, f'Expected inf NLL when all sims excluded, got {nll}'
    assert comp.actual is None, 'Expected actual=None when all sims excluded'

    return nll


@sc.timer()
def test_component_weight(do_plot=do_plot):
    """ Test that component weight scales the NLL correctly """
    sc.heading('Testing component weight...')

    sim = make_sim()
    sim.run()

    expected = make_expected_prevalent()

    comp_w1 = BetaBinomial(
        name='w1', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
        weight=1.0,
    )
    comp_w2 = BetaBinomial(
        name='w2', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
        weight=2.0,
    )

    nll1 = comp_w1.eval(sim)
    nll2 = comp_w2.eval(sim)

    rtol = 1e-10  # Exact multiplication
    assert np.isclose(nll2, 2.0 * nll1, rtol=rtol), \
        f'Expected weight=2 NLL to be 2x weight=1 NLL, got {nll2} vs {nll1}'

    return nll1, nll2


@sc.timer()
def test_component_zero_weight(do_plot=do_plot):
    """ Test that weight=0 returns 0 regardless of NLL """
    sc.heading('Testing component zero weight...')

    sim = make_sim()
    sim.run()

    expected = make_expected_prevalent()
    comp = BetaBinomial(
        name='w0', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
        weight=0,
    )

    nll = comp.eval(sim)
    assert nll == 0, f'Expected zero NLL with weight=0, got {nll}'
    return nll


@sc.timer()
def test_component_repr(do_plot=do_plot):
    """ Test CalibComponent __repr__ returns expected string """
    sc.heading('Testing component repr...')

    expected = make_expected_prevalent()
    comp = BetaBinomial(
        name='My Component', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
    )

    r = repr(comp)
    assert 'My Component' in r, f'Expected component name in repr, got "{r}"'
    return r


#%% Tests for plotting (call to check for exceptions, not visual quality)

@sc.timer()
def test_betabinomial_plot(do_plot=do_plot):
    """ Test BetaBinomial plot and plot_facet_bootstrap run without error """
    sc.heading('Testing BetaBinomial plotting...')

    sim = make_sim()
    ms = ss.MultiSim(sim, n_runs=2, parallel=False)
    ms.run()

    expected = make_expected_prevalent()
    comp = BetaBinomial(
        name='BB Plot Test', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
    )
    comp.eval(ms)

    if do_plot:
        fig = comp.plot(bootstrap=False)
        fig2 = comp.plot(bootstrap=True)
        return fig, fig2

    return comp


@sc.timer()
def test_normal_plot(do_plot=do_plot):
    """ Test Normal plot runs without error """
    sc.heading('Testing Normal plotting...')

    sim = make_sim()
    sim.run()

    dates = [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']]
    expected = pd.DataFrame({
        'x': [0.13, 0.16, 0.06],
    }, index=pd.Index(dates, name='t'))

    comp = Normal(
        name='Normal Plot Test', expected=expected,
        extract_fn=lambda sim: pd.DataFrame({
            'x': sim.results.sir.prevalence,
        }, index=pd.Index(sim.results.timevec, name='t')),
        conform='prevalent',
        sigma2=0.05,
    )
    comp.eval(sim)

    if do_plot:
        fig = comp.plot(bootstrap=False)
        return fig

    return comp


#%% Tests for the Calibration orchestration class

@sc.timer()
def test_calibration_basic(do_plot=do_plot):
    """ Test basic calibration with a single parameter and custom eval_fn """
    sc.heading('Testing basic calibration...')

    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    sim = make_sim()

    def eval_fn(sim, expected):
        date, p = expected
        if not isinstance(sim, ss.MultiSim):
            sim = ss.MultiSim(sims=[sim])
        ret = 0
        for s in sim.sims:
            ind = np.searchsorted(s.results.timevec, date, side='left')
            prev = s.results.sir.prevalence[ind]
            ret += (prev - p) ** 2
        return ret

    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        reseed = True,
        eval_fn = eval_fn,
        eval_kw = dict(expected=(ss.date('2020-01-12'), 0.13)),
        total_trials = 10,
        die = True,
        debug = True,
        verbose = False,
    )

    calib.calibrate()

    assert calib.calibrated, 'Expected calibration to be marked as calibrated'
    assert calib.best_pars is not None, 'Expected best_pars to be populated'
    assert 'beta' in calib.best_pars, 'Expected "beta" in best_pars'

    return calib


@sc.timer()
def test_calibration_to_df(do_plot=do_plot):
    """ Test Calibration.to_df returns sorted DataFrame """
    sc.heading('Testing calibration to_df...')

    calib = test_calibration_basic(do_plot=False)

    df = calib.to_df()
    assert isinstance(df, pd.DataFrame), 'Expected DataFrame from to_df'
    assert len(df) > 0, 'Expected non-empty DataFrame'
    assert 'value' in df.columns, 'Expected "value" column in DataFrame'

    # Test top_k
    df_top = calib.to_df(top_k=3)
    assert len(df_top) <= 3, f'Expected at most 3 rows with top_k=3, got {len(df_top)}'

    return df


@sc.timer()
def test_calibration_to_json(do_plot=do_plot):
    """ Test Calibration.to_json returns valid JSON structure """
    sc.heading('Testing calibration to_json...')

    calib = test_calibration_basic(do_plot=False)

    json_data = calib.to_json()
    assert isinstance(json_data, list), 'Expected list from to_json'
    assert len(json_data) > 0, 'Expected non-empty JSON'
    assert 'mismatch' in json_data[0], 'Expected "mismatch" key in JSON entries'
    assert 'pars' in json_data[0], 'Expected "pars" key in JSON entries'

    return json_data


@sc.timer()
def test_calibration_check_fit(do_plot=do_plot):
    """ Test that check_fit runs and returns a boolean """
    sc.heading('Testing calibration check_fit...')

    calib = test_calibration_basic(do_plot=False)

    result = calib.check_fit(do_plot=False)
    assert isinstance(result, (bool, np.bool_)), f'Expected bool from check_fit, got {type(result)}'

    return result


@sc.timer()
def test_calibration_with_component(do_plot=do_plot):
    """ Test calibration using a Normal component instead of custom eval_fn """
    sc.heading('Testing calibration with Normal component...')

    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    sim = make_sim()

    dates = [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']]
    prevalence = ss.Normal(
        name='Disease prevalence',
        conform='prevalent',
        expected=pd.DataFrame({
            'x': [0.13, 0.16, 0.06],
        }, index=pd.Index(dates, name='t')),
        extract_fn=lambda sim: pd.DataFrame({
            'x': sim.results.sir.prevalence,
        }, index=pd.Index(sim.results.timevec, name='t')),
        sigma2=0.05,
    )

    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        reseed = True,
        components = [prevalence],
        total_trials = 10,
        die = True,
        debug = True,
        verbose = False,
    )

    calib.calibrate()
    assert calib.calibrated, 'Expected calibration with component to complete'
    assert calib.best_pars is not None, 'Expected best_pars from component calibration'

    return calib


@sc.timer()
def test_calibration_with_betabinomial(do_plot=do_plot):
    """ Test calibration using BetaBinomial component (from tutorial pattern) """
    sc.heading('Testing calibration with BetaBinomial component...')

    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    sim = make_sim()

    prevalence_component = ss.BetaBinomial(
        name='SIR Prevalence',
        conform='step_containing',
        expected=pd.DataFrame({
            'n': [200, 197, 195],
            'x': [30, 35, 10],
        }, index=pd.Index([ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']], name='t')),
        extract_fn=lambda sim: pd.DataFrame({
            'x': sim.results.sir.n_infected,
            'n': sim.results.n_alive,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    calib = ss.Calibration(
        sim = sim,
        calib_pars = calib_pars,
        build_fn = build_sim,
        reseed = True,
        components = [prevalence_component],
        total_trials = 10,
        die = True,
        debug = True,
        verbose = False,
    )

    calib.calibrate()
    assert calib.calibrated, 'Expected BetaBinomial calibration to complete'

    return calib


@sc.timer()
def test_calibration_prune_fn(do_plot=do_plot):
    """ Test calibration with a prune_fn that prunes some trials """
    sc.heading('Testing calibration prune_fn...')

    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    sim = make_sim()

    # Prune trials where beta > 0.2
    def prune(pars):
        return pars['beta']['value'] > 0.2

    def eval_fn(sim, **kwargs):
        return np.random.rand()

    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        reseed = True,
        eval_fn = eval_fn,
        prune_fn = prune,
        total_trials = 10,
        die = False,  # Don't die on pruned trials
        debug = True,
        verbose = False,
    )

    calib.calibrate()
    assert calib.calibrated, 'Expected calibration with prune_fn to complete'

    return calib


@sc.timer()
def test_calibration_continue_db(do_plot=do_plot):
    """ Test calibration with keep_db and continue_db options """
    sc.heading('Testing calibration continue_db...')
    import tempfile
    import os

    db_path = os.path.join(tempfile.mkdtemp(), 'test_continue.db')

    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    sim = make_sim()

    def eval_fn(sim, **kwargs):
        return np.random.rand()

    # First calibration: keep the database
    calib1 = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        reseed = True,
        eval_fn = eval_fn,
        total_trials = 5,
        die = True,
        debug = True,
        verbose = False,
        db_name = db_path,
        keep_db = True,
        continue_db = False,
    )
    calib1.calibrate()

    assert os.path.exists(db_path), 'Expected database file to exist with keep_db=True'

    # Second calibration: continue from existing database
    calib2 = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        reseed = True,
        eval_fn = eval_fn,
        total_trials = 5,
        die = True,
        debug = True,
        verbose = False,
        db_name = db_path,
        keep_db = False,
        continue_db = True,
    )
    calib2.calibrate()

    assert calib2.calibrated, 'Expected continuation calibration to complete'

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)

    return calib2


@sc.timer()
def test_eval_fit_multiple_components(do_plot=do_plot):
    """ Test _eval_fit sums NLLs from multiple components """
    sc.heading('Testing _eval_fit with multiple components...')

    sim = make_sim()
    sim.run()

    expected = make_expected_prevalent()

    comp1 = BetaBinomial(
        name='comp1', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
        weight=1.0,
    )
    comp2 = BetaBinomial(
        name='comp2', expected=expected,
        extract_fn=make_extract_fn_prevalent(), conform='prevalent',
        weight=1.0,
    )

    calib = ss.Calibration(
        calib_pars = dict(beta=dict(low=0.01, high=0.30, guess=0.15)),
        sim = sim,
        build_fn = build_sim,
        components = [comp1, comp2],
        total_trials = 1,
        debug = True,
        verbose = False,
    )

    nll1 = comp1.eval(sim)
    nll2 = comp2.eval(sim)
    total = calib._eval_fit(sim)

    rtol = 0.01  # Small tolerance for floating point summation
    assert np.isclose(total, nll1 + nll2, rtol=rtol), \
        f'Expected _eval_fit to sum component NLLs: {total} != {nll1} + {nll2}'

    return total


#%% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Helper function tests
    test_linear_interp(do_plot=do_plot)
    test_step_containing(do_plot=do_plot)
    test_linear_accum(do_plot=do_plot)

    # Conform validation
    test_validate_conform(do_plot=do_plot)

    # NLL tests
    test_betabinomial_nll(do_plot=do_plot)
    test_binomial_nll(do_plot=do_plot)
    test_binomial_get_p(do_plot=do_plot)
    test_normal_nll(do_plot=do_plot)
    test_normal_compute_var(do_plot=do_plot)
    test_normal_array_sigma2(do_plot=do_plot)
    test_gammapoisson_nll(do_plot=do_plot)
    test_gammapoisson_int_validation(do_plot=do_plot)
    test_dirichletmultinomial_nll(do_plot=do_plot)

    # Component eval tests
    test_component_eval_single_sim(do_plot=do_plot)
    test_component_eval_multisim(do_plot=do_plot)
    test_component_include_fn(do_plot=do_plot)
    test_component_weight(do_plot=do_plot)
    test_component_zero_weight(do_plot=do_plot)
    test_component_repr(do_plot=do_plot)

    # Plot tests
    test_betabinomial_plot(do_plot=do_plot)
    test_normal_plot(do_plot=do_plot)

    # Calibration orchestration tests
    calib = test_calibration_basic(do_plot=do_plot)
    test_calibration_to_df(do_plot=do_plot)
    test_calibration_to_json(do_plot=do_plot)
    test_calibration_check_fit(do_plot=do_plot)
    test_calibration_with_component(do_plot=do_plot)
    test_calibration_with_betabinomial(do_plot=do_plot)
    test_calibration_prune_fn(do_plot=do_plot)
    test_calibration_continue_db(do_plot=do_plot)
    test_eval_fit_multiple_components(do_plot=do_plot)

    T.toc()

    if do_plot:
        plt.show()
