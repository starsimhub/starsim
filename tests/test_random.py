"""
Test the RNG object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss
from starsim.random import RNG
import pytest

@pytest.fixture
def rng(request, slots=5, base_seed=1, name='Test', **kwargs):
    return make_rng(slots, base_seed, name, **kwargs)

def make_rng(slots=5, base_seed=1, name='Test', **kwargs):
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=base_seed)
    rng = ss.RNG(name, **kwargs)
    rng.initialize(rng_container, slots=slots)
    return rng


# %% Define the tests
def test_random(rng, n=5):
    """ Simple random draw """
    sc.heading('test_random: Testing simple random draw from a RNG object')
    draws = rng.random(n)
    print(f'Sampled {n} random draws', draws)
    assert len(draws) == n
    return draws


def test_SingleRNG_using_np_singleton(rng, n=5):
    """ Make sure the SingleRNG is using the numpy.random singleton """
    if ss.options.multirng:
        pytest.skip('Skipping multirng mode')
    sc.heading('test_singleton_rng: Testing RNG object')

    np.random.seed(0)
    draws_np = np.random.rand(n)
    np.random.seed(0)
    draws_SingleRNG = rng.random(n)
    assert np.array_equal(draws_np, draws_SingleRNG)
    return draws_np, draws_SingleRNG


def test_reset(rng, n=5):
    """ Sample, reset, sample """
    sc.heading('test_reset: Testing sample, reset, sample')

    draws1 = rng.random(n)
    print(f'Random sample of size {n} returned {draws1}')

    print('Reset')
    rng.reset()

    draws2 = rng.random(n)
    print(f'After reset, random sample of size {n} returned {draws2}')

    if isinstance(rng, ss.MultiRNG):
        assert np.array_equal(draws1, draws2)
    else:
        assert not np.array_equal(draws1, draws2)
    return draws1, draws2


def test_step(rng, n=5):
    """ Sample, step, sample """
    sc.heading('test_step: Testing sample, step, sample')

    draws1 = rng.random(n)
    print(f'Random sample of size {n} returned {draws1}')

    print('Reset')
    rng.step(1)

    draws2 = rng.random(n)
    print(f'After reset, random sample of size {n} returned {draws2}')

    assert not np.array_equal(draws1, draws2)
    return draws1, draws2


def test_seed(n=5):
    """ Changing seeds """
    sc.heading('test_seed: Testing sample with seeds 0 and 1')

    # NOTE: SingleRNG is using the numpy random singleton, and thus ignores base_seed here

    rng0 = make_rng(n, base_seed=0)
    draws0 = rng0.uniform(size=n)
    print(f'Random sample of size {n} for rng0 with base_seed 0 returned {draws0}')

    rng1 = make_rng(n, base_seed=1)
    draws1 = rng1.uniform(size=n)
    print(f'Random sample of size {n} for rng1 with base_seed 1 returned {draws1}')

    assert not np.array_equal(draws0, draws1)
    return draws0, draws1


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    n=5

    for multirng in [True, False]:
        ss.options(multirng=multirng)
        sc.heading('Testing with multirng set to', multirng)

        # Run tests - some will only pass if multirng is True
        test_random(n)
        test_reset(n)
        test_step(n)
        test_seed(n)

    sc.toc(T)
    print('Done.')
