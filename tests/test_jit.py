"""
Test ss.jit: functions compiled by Numba or traced by JAX, with the same results, and what falls back to Python
"""

# %% Imports and settings
import pytest
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


# %% Define the tests

def test_jax():
    sc.heading('Testing a function traced by JAX, and its gradient')
    import jax
    jnp = jax.numpy
    x = np.random.default_rng(1).normal(size=100)
    with jax.enable_x64(True):
        js = ss.jit(score)
        xj = jnp.asarray(x)
        assert np.isclose(float(js(xj)), score(x), rtol=1e-12)
        grad = jax.grad(js)(xj)
        expected = -2*x*np.exp(-x**2)/(1 + np.exp(-x**2))
        assert np.allclose(grad, expected, rtol=1e-10)
        jl = ss.jit(longest_run)
        assert jl(xj) == longest_run(x) # JAX can't trace a loop over values, so it is run as NumPy (with a note)
        with pytest.raises(TypeError):
            jax.grad(ss.jit(lambda x: x[0] if x[0] > 0 else x[1]))(xj) # Under a trace there is no going back to NumPy, so it says why
    return


if __name__ == '__main__':
    T = sc.timer()
    test_jax()
    T.toc()
