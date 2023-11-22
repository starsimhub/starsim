"""
Test the RNG object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import NotResetException, RNG
import pytest

@pytest.fixture(params=['single','multi'])
def rng(request, slots=5, base_seed=1, name='Test', **kwargs):
    return make_rng(request.param, slots, base_seed, name, **kwargs)

def make_rng(rng_type, slots=5, base_seed=1, name='Test', **kwargs):
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=base_seed)
    if rng_type == "multi":
        rng = ss.MultiRNG(name, **kwargs)
    else:
        rng = ss.SingleRNG(name, **kwargs)
    rng.initialize(rng_container, slots=slots)

    return rng


# %% Define the tests

def test_sample(rng, n=5):
    """ Simple non-RNG safe sample """
    sc.heading('test_sample: Testing RNG object')
    uids = np.arange(0,n,2) # every other to make it interesting

    draws = rng.random(uids)
    print('Sampled random draws', draws, 'for uids', uids)

    assert len(draws) == 3
    return draws


def test_neg_binomial(rng, n=5):
    """ Negative Binomial """
    sc.heading('test_neg_binomial: Testing negative binomial')

    nb = ss.neg_binomial(mean=80, dispersion=40)
    draws = rng.sample(nb, n)
    print(f'Sampling n={n} negative binomial draws via ss.neg_binomial returned {draws}.')

    print('Step')
    rng.step(1) # Prepare to call again

    # Now try calling with UIDs instead of n
    uids = np.arange(0,n,2) # every other to make it interesting
    draws_u = rng.sample(nb, uids)
    print(f'Now sampling for uids {uids} returned {draws_u}.')

    assert len(draws) == n and len(draws_u) == len(uids)
    return draws, draws_u


def test_reset(rng, n=5):
    """ Sample, reset, sample """
    sc.heading('test_reset: Testing sample, reset, sample')

    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'Random sample for uids {uids} returned {draws1}')

    print('Reset')
    rng.reset()

    draws2 = rng.random(uids)
    print(f'After reset, random sample for uids {uids} returned {draws2}')

    if ss.options.multirng:
        assert np.array_equal(draws1, draws2)
    return draws1, draws2


def test_step(rng, n=5):
    """ Sample, step, sample """
    sc.heading('test_step: Testing sample, step, sample')

    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'Random sample for uids {uids} returned {draws1}')

    print('Reset')
    rng.step(1)

    draws2 = rng.random(uids)
    print(f'After reset, random sample for uids {uids} returned {draws2}')

    if ss.options.multirng:
        assert not np.array_equal(draws1, draws2)
    return draws1, draws2

@pytest.mark.parametrize("rng_type", ['single','multi'])
def test_seed(rng_type, n=5):
    """ Changing seeds """
    sc.heading('test_seed: Testing sample with seeds 0 and 1')
    uids = np.arange(0,n,2) # every other to make it interesting
    dist = ss.uniform()

    rng0 = make_rng(rng_type, n, base_seed=0)
    draws0 = rng0.sample(dist, uids)
    print(f'Random sample for uids {uids} for rng0 with seed {rng0.seed} returned {draws0}')

    rng1 = make_rng(rng_type, n, base_seed=1)
    draws1 = rng1.sample(dist, uids)
    print(f'Random sample for uids {uids} for rng1 with seed {rng1.seed} returned {draws1}')

    assert not np.array_equal(draws0, draws1)
    return draws0, draws1


def test_repeat(rng, n=5):
    """ Sample, sample - should raise and exception"""
    sc.heading('test_repeat: Testing sample, sample - should raise an exception')
    uids = np.arange(0,n,2) # every other to make it interesting
    dist = ss.uniform()

    draws1 = rng.sample(dist, uids)
    print(f'Random sample for uids {uids} returned {draws1}')

    print('Attempting to sample again without resetting, should raise and exception.')
    with pytest.raises(NotResetException):
        rng.sample(dist, uids)
    return rng


def test_boolmask(rng, n=5):
    """ Simple sample with a boolean mask"""
    sc.heading('test_boolmask: Testing RNG object with a boolean mask')
    uids = np.arange(0,n,2) # every other to make it interesting
    mask = np.full(n, False)
    mask[uids] = True
    dist = ss.uniform()

    draws_bool = rng.sample(dist, mask)
    print(f'Random sample for boolean mask {mask} returned {draws_bool}')

    print('Reset')
    rng.reset()

    draws_uids = rng.sample(dist, uids)
    print(f'After reset, random sample for uids {uids} returned {draws_uids}')

    if isinstance(rng, ss.MultiRNG):
        assert np.array_equal(draws_bool, draws_uids)

    return draws_bool, draws_uids


def test_empty(rng):
    """ Simple sample with a boolean mask"""
    sc.heading('test_empty: Testing empty draw')
    uids = np.array([]) # EMPTY
    draws = rng.random(uids)
    print(f'Random sample for (empty) uids {uids} returned {draws}')

    assert len(draws) == 0
    return draws


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    n=5

    for multirng in [True, False]:
        ss.options(multirng=multirng)
        sc.heading('Testing with multirng set to', multirng)

        # Run tests - some will only pass if multirng is True
        test_sample(n)
        test_neg_binomial(n)
        test_reset(n)
        test_step(n)
        test_seed(n)
        test_boolmask(n)
        test_empty()

        if multirng:
            test_repeat(n)

    sc.toc(T)
    print('Done.')