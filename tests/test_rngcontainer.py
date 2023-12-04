"""
Test the RNGContainer object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import NotInitializedException, SeedRepeatException, RepeatNameException
import pytest


# %% Define the tests

@pytest.fixture
def rng_container():
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)
    return rng_container

@pytest.fixture
def rngs(request):
    return [ss.RNG('rng0'), ss.RNG('rng1')]

@pytest.fixture
def dists(request):
    return [ss.uniform(rng='rng0'), ss.uniform(rng='rng1')]

def test_rng(rng_container, rngs, n=5):
    """ Simple sample from rng """
    sc.heading('test_container: Testing RNGContainer object')

    rng0, rng1 = rngs
    rng0.initialize(rng_container, slots=n)

    draws = rng0.random(n)
    print(f'Created seed and sampled: {draws}')

    assert len(draws) == n
    return draws

def test_dists(rng_container, dists, n=5):
    """ Simple sample from distribution """
    sc.heading('test_container: Testing RNGContainer object')

    d0 = dists[0]
    d0.rng.initialize(rng_container, slots=n)

    uids = np.arange(0,n,2) # every other to make it interesting
    draws = d0.sample(uids)
    print(f'Created seed and sampled: {draws}')

    assert len(draws) == len(uids)
    return draws


def test_seed(rng_container, rngs, n=5):
    """ Test assignment of seeds """
    sc.heading('test_seed: Testing assignment of seeds')

    rng0, rng1 = rngs
    rng0.initialize(rng_container, slots=n)
    rng1.initialize(rng_container, slots=n)

    print(f'Random generators rng0 and rng1 were assigned seeds {rng0.seed} and {rng1.seed}, respectively')

    if ss.options.multirng:
        assert rng1.seed != rng0.seed
    else:
        assert rng1.seed == rng0.seed
    return rng0, rng1


def test_reset(rng_container, dists, n=5):
    """ Sample, reset, sample """
    sc.heading('test_reset: Testing sample, reset, sample')

    d0 = dists[0]
    d0.rng.initialize(rng_container, slots=n)

    uids = np.arange(0,n,2) # every other to make it interesting
    s_before = d0.sample(uids)

    rng_container.reset() # Return to step 0

    s_after = d0.sample(uids)

    print(f'Initial sample', s_before)
    print('Reset')
    print(f'After reset sample', s_after)

    if ss.options.multirng:
        assert np.array_equal(s_after, s_before)
    else:
        assert not np.array_equal(s_after, s_before)
    return s_before, s_after


def test_step(rng_container, dists, n=5):
    """ Sample, step, sample """
    sc.heading('test_step: Testing sample, step, sample')


    d0 = dists[0]
    d0.rng.initialize(rng_container, slots=n)

    uids = np.arange(0,n,2) # every other to make it interesting
    s_before = d0.sample(uids)

    rng_container.step(10) # 10 steps

    s_after = d0.sample(uids)

    print(f'Initial sample', s_before)
    print('Step')
    print(f'Subsequent sample', s_after)

    assert not np.array_equal(s_after, s_before)
    return s_before, s_after


def test_initialize(rngs, n=5):
    """ Sample without initializing, should raise exception """
    sc.heading('test_initialize: Testing without initializing, should raise exception.')
    rng_container = ss.RNGContainer()
    #rng_container.initialize(base_seed=3) # Do not initialize

    with pytest.raises(NotInitializedException):
        rngs[0].initialize(rng_container, slots=n)
    return rngs[0]


def test_seedrepeat(rng_container, n=5):
    """ Two random number generators with the same seed, should raise exception """
    sc.heading('test_seedrepeat: Testing two random number generators with the same seed, should raise exception.')

    rng0 = ss.MultiRNG('rng0', seed_offset=0)
    rng0.initialize(rng_container, slots=n)

    with pytest.raises(SeedRepeatException):
        rng1 = ss.MultiRNG('rng1', seed_offset=0)
        rng1.initialize(rng_container, slots=n)
    return rng0, rng1


def test_samplingorder(rng_container, dists, n=5):
    """ Ensure sampling from one RNG doesn't affect another """
    sc.heading('test_samplingorder: Testing from multiple random number generators to test if sampling order matters')

    uids = np.arange(0,n,2) # every other to make it interesting

    d0, d1 = dists
    d0.rng.initialize(rng_container, slots=n)
    d1.rng.initialize(rng_container, slots=n)

    s_before = d0.sample(uids)
    _ = d1.sample(uids)

    rng_container.reset()

    _ = d1.sample(uids)
    s_after = d0.sample(uids)

    print(f'When sampling rng0 before rng1:', s_before)
    print('Reset')
    print(f'When sampling rng0 after rng1:', s_after)

    if ss.options.multirng:
        assert np.array_equal(s_before, s_after)
    else:
        assert not np.array_equal(s_before, s_after)

    return s_before, s_after


def test_repeatname(rng_container, n=5):
    """ Test two random number generators with the same name """
    sc.heading('test_repeatname: Testing if two random number generators with the same name are allowed')

    rng0 = ss.RNG('test')
    rng0.initialize(rng_container, slots=n)

    rng1 = ss.RNG('test')
    with pytest.raises(RepeatNameException):
        rng1.initialize(rng_container, slots=n)
    return rng0, rng1


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    test_rng()
    test_dists()
    test_seed()
    test_reset()
    test_step()
    test_initialize()
    test_seedrepeat()
    test_samplingorder()
    test_repeatname()

    sc.toc(T)
    print('Done.')