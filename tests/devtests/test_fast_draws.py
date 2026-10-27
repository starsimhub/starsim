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


def tv(x, pmf):
    """ The total variation distance between draws and a pmf, and about how big it is from sampling noise alone """
    k = np.arange(x.min(), x.max() + 1)
    emp = np.bincount((x - k[0]).astype(int), minlength=len(k))/len(x)
    p = pmf(k)
    return 0.5*np.abs(emp - p).sum() + 0.5*(1 - p.sum()), np.sqrt(len(k)/len(x))/2


@nb.jit(nopython=True)
def uniforms(k, n):
    """ The first n uniforms of the stream with key k """
    out = np.empty(n)
    for i in range(n):
        out[i] = D.uniform(k, i)
    return out


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
        x = D.many[f](*pars, uniforms(k, N), np.empty(N))
        ks = sps.kstest(x, law.cdf).statistic*np.sqrt(N)
        print(f'{f}{pars}: KS statistic x sqrt(N) = {ks:.2f}')
        assert ks < 2.0, (f, pars) # A 0.1% chance by noise alone

    k = D.key(4, 0)
    for n, p in [(0.05, 0.01), (1.0, 0.3), (5.0, 0.5), (2.0, 0.02), (50.0, 0.4)]:
        x = D.many.nbinom(n, p, uniforms(k, N), np.empty(N))
        d, noise = tv(x, lambda j: sps.nbinom.pmf(j, n, p))
        print(f'Negative binomial n={n}, p={p}: TV {d:.4f} (noise ~{noise:.4f})')
        assert d < 3*noise + 1e-3, (n, p)
    return


if __name__ == '__main__':
    T = sc.timer()
    test_binomial_poisson()
    test_others()
    T.toc()
