"""
Test the RNGContainer object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import NotInitializedException, SeedRepeatException, RepeatNameException


# %% Define the tests

def test_container(n=5):
    """ Simple sample """
    sc.heading('test_container: Testing RNGContainer object')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)

    rng = ss.MultiRNG('rng1')
    rng.initialize(rng_container, slots=n)

    uids = np.arange(0,n,2) # every other to make it interesting
    draws = rng.random(uids)
    print(f'Created seed and sampled: {draws}')

    assert len(draws) == len(uids)
    return draws


def test_seed(n=5):
    """ Test assignment of seeds """
    sc.heading('test_seed: Testing assignment of seeds')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)

    rng0 = ss.MultiRNG('rng0')
    rng0.initialize(rng_container, slots=n)

    rng1 = ss.MultiRNG('rng1')
    rng1.initialize(rng_container, slots=n)
    print(f'Random generators rng0 and rng1 were assigned seeds {rng0.seed} and {rng1.seed}, respectively')

    assert rng1.seed != rng0.seed
    return rng0, rng1


def test_reset(n=5):
    """ Sample, reset, sample """
    sc.heading('test_reset: Testing sample, reset, sample')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)

    rng = ss.MultiRNG('rng0')
    rng.initialize(rng_container, slots=n)

    uids = np.arange(0,n,2) # every other to make it interesting
    s_before = rng.random(uids)

    rng_container.reset() # Return to step 0

    s_after = rng.random(uids)

    print(f'Initial sample', s_before)
    print('Reset')
    print(f'After reset sample', s_after)

    assert np.array_equal(s_after, s_before)
    return s_before, s_after


def test_step(n=5):
    """ Sample, step, sample """
    sc.heading('test_step: Testing sample, step, sample')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)

    rng = ss.MultiRNG('rng0')
    rng.initialize(rng_container, slots=n)

    uids = np.arange(0,n,2) # every other to make it interesting
    s_before = rng.random(uids)

    rng_container.step(10) # 10 steps

    s_after = rng.random(uids)

    print(f'Initial sample', s_before)
    print('Step')
    print(f'Subsequent sample', s_after)

    assert not np.array_equal(s_after, s_before)
    return s_before, s_after


def test_initialize(n=5):
    """ Sample without initializing, should raise exception """
    sc.heading('test_initialize: Testing without initializing, should raise exception.')
    rng_container = ss.RNGContainer()
    #rng_container.initialize(base_seed=3) # Do not initialize

    rng = ss.MultiRNG('rng0')

    try:
        rng.initialize(rng_container, slots=n)
        assert False # Should not get here!
    except NotInitializedException as e:
        print(f'YAY! Got exception: {e}')
    return rng


def test_seedrepeat(n=5):
    """ Two random number generators with the same seed, should raise exception """
    sc.heading('test_seedrepeat: Testing two random number generators with the same seed, should raise exception.')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)

    rng = ss.MultiRNG('rng0', seed_offset=0)
    rng.initialize(rng_container, slots=n)

    try:
        rng1 = ss.MultiRNG('rng1', seed_offset=0)
        rng1.initialize(rng_container, slots=n)
        assert False # Should not get here!
    except SeedRepeatException as e:
        print(f'YAY! Got exception: {e}')
    return rng, rng1


def test_samplingorder(n=5):
    """ Ensure sampling from one RNG doesn't affect another """
    sc.heading('test_samplingorder: Testing from multiple random number generators to test if sampling order matters')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=10)

    uids = np.arange(0,n,2) # every other to make it interesting

    rng0 = ss.MultiRNG('rng0')
    rng0.initialize(rng_container, slots=n)

    rng1 = ss.MultiRNG('rng1')
    rng1.initialize(rng_container, slots=n)

    s_before = rng0.random(uids)
    _ = rng1.random(uids)

    rng_container.reset()

    _ = rng1.random(uids)
    s_after = rng0.random(uids)

    print(f'When sampling rng0 before rng1:', s_before)
    print('Reset')
    print(f'When sampling rng0 after rng1:', s_after)

    assert np.array_equal(s_before, s_after)
    return s_before, s_after


def test_repeatname(n=5):
    """ Test two random number generators with the same name """
    sc.heading('test_repeatname: Testing if two random number generators with the same name are allowed')
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=17)

    rng0 = ss.MultiRNG('test')
    rng0.initialize(rng_container, slots=n)

    rng1 = ss.MultiRNG('test')
    try:
        rng1.initialize(rng_container, slots=n)
        assert False # Should not get here!
    except RepeatNameException as e:
        print(f'YAY! Got exception: {e}')
    return rng0, rng1


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    test_container()
    test_seed()
    test_reset()
    test_step()
    test_initialize()
    test_seedrepeat()
    test_samplingorder()
    test_repeatname()

    sc.toc(T)
    print('Done.')