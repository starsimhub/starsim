"""
Test the RNG object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import NotResetException, RNG
import pytest


def make_rng(slots, base_seed=1, name='Test'):
    """ Create and initialize a random number generator within a container """
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=base_seed)
    rng = RNG(name)
    rng.initialize(rng_container, slots=slots)

    return rng


# %% Define the tests

def test_sample(n=5):
    """ Simple sample """
    sc.heading('test_sample: Testing RNG object')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws = rng.random(uids)
    print('Sampled random draws', draws, 'for uids', uids)

    assert len(draws) == len(uids)
    return draws


def test_neg_binomial(n=5):
    """ Negative Binomial """
    sc.heading('test_neg_binomial: Testing negative binomial')
    rng = make_rng(n)

    # Can call directly:
    #rng.negative_binomial(n=40, p=1/3, size=5)

    # Or through a Distribution
    nb = ss.neg_binomial(mean=80, dispersion=40)
    draws = rng.sample(nb, n)
    print(f'Sampling n={n} negative binomial draws via ss.neg_binomial returned {draws}.')

    print('Step')
    rng.step(1) # Prepare to call again

    # Now try calling with UIDs instead of n
    uids = np.arange(0,n,2) # every other to make it interesting
    draws_u = rng.sample(nb, n)
    print(f'Now sampling for uids {uids} returned {draws_u}.')

    assert len(draws) == n and len(draws_u) == len(uids)
    return draws, draws_u


def test_reset(n=5):
    """ Sample, reset, sample """
    sc.heading('test_reset: Testing sample, reset, sample')
    rng = make_rng(n)
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


def test_step(n=5):
    """ Sample, step, sample """
    sc.heading('test_step: Testing sample, step, sample')
    rng = make_rng(n)
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


def test_seed(n=5):
    """ Changing seeds """
    sc.heading('test_seed: Testing sample with seeds 0 and 1')
    uids = np.arange(0,n,2) # every other to make it interesting

    rng0 = make_rng(n, base_seed=0)
    draws0 = rng0.random(uids)
    print(f'Random sample for uids {uids} for rng0 with seed {rng0.seed} returned {draws0}')

    rng1 = make_rng(n, base_seed=1)
    draws1 = rng1.random(uids)
    print(f'Random sample for uids {uids} for rng1 with seed {rng1.seed} returned {draws1}')

    assert not np.array_equal(draws0, draws1)
    return draws0, draws1


def test_repeat(n=5):
    """ Sample, sample - should raise and exception"""
    sc.heading('test_repeat: Testing sample, sample - should raise an exception')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'Random sample for uids {uids} returned {draws1}')

    print('Attempting to sample again without resetting, should raise and exception.')
    with pytest.raises(NotResetException):
        rng.random(uids)
    return rng


def test_boolmask(n=5):
    """ Simple sample with a boolean mask"""
    sc.heading('test_boolmask: Testing RNG object with a boolean mask')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting
    mask = np.full(n, False)
    mask[uids] = True

    draws_bool = rng.random(mask)
    print(f'Random sample for boolean mask {mask} returned {draws_bool}')

    print('Reset')
    rng.reset()

    draws_uids = rng.random(uids)
    print(f'After reset, random sample for uids {uids} returned {draws_uids}')

    if ss.options.multirng:
        assert np.array_equal(draws_bool, draws_uids)
    return draws_bool, draws_uids


def test_empty():
    """ Simple sample with a boolean mask"""
    sc.heading('test_empty: Testing empty draw')
    rng = make_rng(5) # Number of slots is arbitrary
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