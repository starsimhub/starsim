"""
Test ss.jit: functions compiled by Numba or traced by JAX, with the same results, and what falls back to Python
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss


def longest_run(x):
    """ A loop: the longest run of values above the mean """
    m = x.mean()
    best = cur = 0
    for v in x:
        cur = cur + 1 if v > m else 0
        best = max(best, cur)
    return best


def score(x):
    """ Elementwise, then summed """
    return np.sum(np.log1p(np.exp(-x**2)))


def draws(n, p, lam, rng):
    """ Array draws from a NumPy Generator, which Numba alone doesn't take """
    return rng.binomial(n, p) + rng.poisson(lam) + rng.binomial(3, 0.5)


# %% Define the tests

def test_function():
    sc.heading('Testing a function compiled by Numba, with the same results')
    x = np.random.default_rng(1).normal(size=10_000)
    assert ss.jit(longest_run)(x) == longest_run(x)
    assert np.isclose(ss.jit(score)(x), score(x), rtol=1e-13) # The same up to the order of the sum (NumPy's is pairwise)
    n = np.array([0, 10, 1000, 100_000]*5)
    p = np.linspace(0, 1, 20)
    lam = np.array([0, 0.5, 5, 20, 1000.]*4) # Poisson means of 10 or more are drawn as NumPy does, not as Numba does
    out = ss.jit(draws)(n, p, lam, np.random.default_rng(2))
    assert np.array_equal(out, draws(n, p, lam, np.random.default_rng(2)))

    @ss.jit
    def uncompilable(x):
        return sc.findinds(x > 0) # Numba can't call Sciris
    assert np.array_equal(uncompilable(x), sc.findinds(x > 0)) # Runs as Python, with the same result
    assert uncompilable.failed and uncompilable.said
    return


if __name__ == '__main__':
    T = sc.timer()
    test_function()
    T.toc()
