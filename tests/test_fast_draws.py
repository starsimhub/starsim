"""
Test the exact draws in starsim/perflib/draws.py against their distributions
"""

# %% Imports and settings
import numpy as np
import numba as nb
import sciris as sc
import scipy.special as spec
import scipy.stats as sps
import starsim as ss

D = ss.perflib.draws
N = 200_000 # Draws for each distribution
rng = np.random.default_rng(1)


# %% Define the tests

@sc.timer()
def test_math():
    sc.heading('Testing the vectorizable exp, log, log1p, expm1, and the inverse normal CDF')
    x = np.concatenate([-rng.uniform(0, 708, 10_000), rng.uniform(0, 709, 1000), -10**rng.uniform(-15, 0, 1000)])
    y = 10**rng.uniform(-300, 300, 10_000)
    z = np.concatenate([rng.uniform(-0.999, 5, 10_000), 10**rng.uniform(-17, -1, 1000)*rng.choice([-1, 1], 1000)])
    u = np.concatenate([rng.random(10_000), 10**rng.uniform(-300, -1, 1000), 1 - 10**rng.uniform(-16, -1, 1000)])
    for f, a, ref in [(D.exp, x, np.exp), (D.log, y, np.log), (D.log1p, z, np.log1p), (D.expm1, x, np.expm1), (D.ndtri, u, spec.ndtri)]:
        out = np.array([f(v) for v in a[::50]])
        r = ref(a[::50])
        err = np.max(np.abs(out - r)/np.maximum(np.abs(r), 1e-300 if f is not D.ndtri else 1))
        print(f'{f.__name__}: max relative error {err:.1e}')
        assert err < 2e-15, f.__name__
    return


@sc.timer()
def test_uniforms():
    sc.heading('Testing the counter-based uniforms, and the draws run as plain Python')
    slots = rng.integers(0, 10**9, 1000)
    u = np.array([D.uniform(D.key(11, 4), s) for s in slots])
    assert np.array_equal(u, ss.distributions.hash_uniforms(11, 4, slots, dtype=np.float64)) # The same hash as ss.Dist's common random numbers
    assert np.array_equal(D.uniforms(D.key(11, 4), 5, np.empty(3)), [D.uniform(D.key(11, 4), i) for i in [5, 6, 7]])

    # The plain Python copy gives the same draws
    P = D.plain_copy()
    np.seterr(over="ignore") # The plain copy's random streams overflow, on purpose (as compiled code does silently)
    n = rng.integers(0, 200, 300).astype(float)
    p = 10**rng.uniform(-4, 0, 300)
    u = rng.random(300)
    assert np.array_equal(P.binomials(n, p, u, np.empty(300), np.empty((9, 300))), D.binomials(n, p, u, np.empty(300), np.empty((9, 300))))
    assert np.array_equal([P.poisson(l, v) for l, v in zip(p*50, u)], [D.poisson(l, v) for l, v in zip(p*50, u)])

    # A mean (of the side drawn) of 2^53 or more (from a bug upstream: it used to hang or overflow), or NaN, gives NaN, in a row as on its own
    n = np.array([np.inf, 1e20, np.nan, 50, 50])
    p = np.array([0.3, 0.7, 0.3, np.nan, 0.3])
    assert np.array_equal(D.binomials(n, p, u[:5], np.empty(5), np.empty((9, 5))), [D.binomial(*a) for a in zip(n, p, u[:5])], equal_nan=True)
    assert np.isnan(D.binomial(np.inf, 0.3, 0.5)) and D.binomial(50, 0.3, 0.5) >= 0
    lam = np.array([np.inf, 1e19, np.nan, 50])
    assert np.array_equal(D.poissons(lam, u[:4], np.empty(4), np.empty((6, 4))), [D.poisson(*a) for a in zip(lam, u[:4])], equal_nan=True)
    assert np.isnan(D.poisson(1e19, 0.5)) and D.poisson(50, 0.5) >= 0
    return


@sc.timer()
def test_dist_ppf():
    sc.heading('Testing that the compiled inverse CDFs of ss.Dist give what SciPy gives')
    u = rng.random(20_000)
    n = rng.integers(1, 200, 20_000)
    lam = 10**rng.uniform(-3, 2, 20_000) # Some above a mean of 30, which SciPy takes
    cases = [
        (ss.distributions.inverse('expon', u, 3.0), sps.expon.ppf(u, scale=3)),
        (ss.distributions.inverse('weibull', u, 1.7, 2.0), sps.weibull_min.ppf(u, 1.7, scale=2)),
        (ss.distributions.inverse('binomial', u, n, 0.3, big=n*0.3 > 30, fallback=lambda v, i: sps.binom.ppf(v, n[i], 0.3)), sps.binom.ppf(u, n, 0.3)),
        (ss.distributions.inverse('nbinom', u, 2.0, 0.3), sps.nbinom.ppf(u, 2, 0.3)),
        (ss.distributions.inverse('poisson', u, lam, big=lam > 30, fallback=lambda v, i: sps.poisson.ppf(v, lam[i])), sps.poisson.ppf(u, lam)),
    ]
    for ours, scipy in cases:
        assert np.allclose(ours, scipy, rtol=1e-12) # The same values (up to rounding at a step of a discrete CDF, which is about as likely as a uniform landing on it)
    return


if __name__ == '__main__':
    T = sc.timer()
    test_math()
    test_uniforms()
    test_dist_ppf()
    T.toc()
