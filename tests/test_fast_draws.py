"""
Test the exact draws in starsim/perflib/_draws.py against their distributions
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


def tv(x, pmf):
    """ The total variation distance between draws and a pmf, and about how big it is from sampling noise alone """
    k = np.arange(x.min(), x.max() + 1)
    emp = np.bincount((x - k[0]).astype(int), minlength=len(k))/len(x)
    p = pmf(k)
    return 0.5*np.abs(emp - p).sum() + 0.5*(1 - p.sum()), np.sqrt(len(k)/len(x))/2


@nb.jit(nopython=True)
def scalar_binomial(n, p, u, out):
    for i in range(n.size):
        out[i] = D.binomial(n[i], p[i], u[i])
    return out


@nb.jit(nopython=True)
def scalar_poisson(lam, u, out):
    for i in range(lam.size):
        out[i] = D.poisson(lam[i], u[i])
    return out


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
def test_binomial_poisson():
    sc.heading('Testing binomial and Poisson draws: exact, rows the same as scalar, monotone in the uniform')
    w = np.empty((9, N))
    for n, p in [(5, 0.001), (100, 0.003), (100, 0.05), (1000, 0.02), (40, 0.7), (12, 0.708), (20_000, 1e-3), (100_000, 0.2), (10, 1.0), (0, 0.3), (7, 0.0)]:
        nn, pp, u = np.full(N, float(n)), np.full(N, p), rng.random(N)
        x = D.binomials(nn, pp, u, np.empty(N), w)
        d, noise = tv(x, lambda k: sps.binom.pmf(k, n, p))
        print(f'Binomial n={n}, p={p}: TV {d:.4f} (noise ~{noise:.4f})')
        assert d < 3*noise + 1e-3, (n, p)
        assert np.array_equal(x, scalar_binomial(nn, pp, u, np.empty(N))), 'The row and scalar draws differ'
        if n*min(p, 1 - p) <= D.INV_MAX:
            order = np.argsort(u)
            assert np.all(np.diff(x[order]) >= 0), 'Not monotone in the uniform'
    for lam in [0.001, 0.4, 3.0, 9.6, 25.0, 100.0]:
        ll, u = np.full(N, lam), rng.random(N)
        x = D.poissons(ll, u, np.empty(N), w)
        d, noise = tv(x, lambda k: sps.poisson.pmf(k, lam))
        print(f'Poisson lam={lam}: TV {d:.4f} (noise ~{noise:.4f})')
        assert d < 3*noise + 1e-3, lam
        assert np.array_equal(x, scalar_poisson(ll, u, np.empty(N)))

    # Mixed rows, as a model's groups give them
    n = rng.integers(0, 50_000, 1000).astype(float)*(rng.random(1000) > 0.3)
    p = np.where(rng.random(1000) < 0.3, 1.0, 10**rng.uniform(-6, 0, 1000))
    u = rng.random(1000)
    assert np.array_equal(D.binomials(n, p, u, np.empty(1000), w), scalar_binomial(n, p, u, np.empty(1000)))

    # The steps of the rows compile to vector code (a Numba change could silently undo it)
    for step in [D._step_binomial, D._step_poisson]:
        fresh = nb.jit(nopython=True, error_model='numpy')(step.py_func) # Not from the cache, which keeps no assembly
        fresh(*([np.ones(64)]*(6 if step is D._step_poisson else 7)), 1.0, 1.0)
        asm = '\n'.join(fresh.inspect_asm().values())
        assert 'mulpd' in asm, f'{step.__name__} is not vectorized'
    return


@sc.timer()
def test_others():
    sc.heading('Testing the other distributions against their laws')
    cases = [
        ('expon', (3.0,), sps.expon(scale=3)),
        ('weibull', (1.7, 2.0), sps.weibull_min(1.7, scale=2)),
        ('weibull', (0.4, 1.0), sps.weibull_min(0.4)),
    ]
    for i, (f, pars, law) in enumerate(cases):
        k = D.key(3, i)
        x = D.many[f].from_key(*pars, k, np.empty(N))
        ks = sps.kstest(x, law.cdf).statistic*np.sqrt(N)
        print(f'{f}{pars}: KS statistic x sqrt(N) = {ks:.2f}')
        assert ks < 2.0, (f, pars) # A 0.1% chance by noise alone

    k = D.key(4, 0)
    for n, p in [(0.05, 0.01), (1.0, 0.3), (5.0, 0.5), (2.0, 0.02), (50.0, 0.4)]:
        x = D.many.nbinom.from_key(n, p, k, np.empty(N))
        d, noise = tv(x, lambda j: sps.nbinom.pmf(j, n, p))
        print(f'Negative binomial n={n}, p={p}: TV {d:.4f} (noise ~{noise:.4f})')
        assert d < 3*noise + 1e-3, (n, p)
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
    test_binomial_poisson()
    test_others()
    test_uniforms()
    test_dist_ppf()
    T.toc()
