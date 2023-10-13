"""
Test the Streams object from streams.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.streams import NotInitializedException, SeedRepeatException, RepeatNameException


# %% Define the tests

def test_streams(n=5):
    """ Simple sample """
    sc.heading('Testing streams object')
    streams = ss.Streams()
    streams.initialize(base_seed=10)

    rng = ss.MultiStream('stream1')
    rng.initialize(streams)

    uids = np.arange(0,n,2) # every other to make it interesting
    draws = rng.random(uids)
    print(f'\nCREATED SEED AND SAMPLED: {draws}')

    return len(draws) == len(uids)


def test_seed(n=5):
    """ Sample, reset, sample """
    sc.heading('Testing streams reset')
    streams = ss.Streams()
    streams.initialize(base_seed=10)

    rng0 = ss.MultiStream('stream0')
    rng0.initialize(streams)

    rng1 = ss.MultiStream('stream1')
    rng1.initialize(streams)

    return rng1.seed != rng0.seed


def test_reset(n=5):
    """ Sample, step, sample """
    sc.heading('Testing sample, step, sample')
    streams = ss.Streams()
    streams.initialize(base_seed=10)

    rng = ss.MultiStream('stream0')
    rng.initialize(streams)

    uids = np.arange(0,n,2) # every other to make it interesting
    s_before = rng.random(uids)

    streams.reset() # Return to step 0

    s_after = rng.random(uids)

    return np.all(s_after == s_before)


def test_step(n=5):
    """ Sample, step, sample """
    sc.heading('Testing sample, step, sample')
    streams = ss.Streams()
    streams.initialize(base_seed=10)

    rng = ss.MultiStream('stream0')
    rng.initialize(streams)

    uids = np.arange(0,n,2) # every other to make it interesting
    s_before = rng.random(uids)

    streams.step(10) # 10 steps

    s_after = rng.random(uids)

    return np.all(s_after != s_before)


def test_initialize(n=5):
    """ Sample without initializing, should raise exception """
    sc.heading('Testing without initializing, should raise exception.')
    streams = ss.Streams()
    #streams.initialize(base_seed=3)

    rng = ss.MultiStream('stream0')

    try:
        rng.initialize(streams)
        return False # Should not get here!
    except NotInitializedException as e:
        print(f'YAY! Got exception: {e}')
    return True


def test_seedrepeat(n=5):
    """ Two streams with the same seed, should raise exception """
    sc.heading('Testing two streams with the same seed, should raise exception.')
    streams = ss.Streams()
    streams.initialize(base_seed=10)

    rng = ss.MultiStream('stream0', seed_offset=0)
    rng.initialize(streams)

    try:
        rng1 = ss.MultiStream('stream1', seed_offset=0)
        rng1.initialize(streams)
        return False # Should not get here!
    except SeedRepeatException as e:
        print(f'YAY! Got exception: {e}')
    return True


def test_samplingorder(n=5):
    """ Ensure sampling from one stream doesn't affect another """
    sc.heading('Testing from multiple streams to test if sampling order matters')
    streams = ss.Streams()
    streams.initialize(base_seed=10)

    uids = np.arange(0,n,2) # every other to make it interesting

    rng0 = ss.MultiStream('stream0')
    rng0.initialize(streams)

    rng1 = ss.MultiStream('stream1')
    rng1.initialize(streams)

    s_before = rng0.random(uids)
    _ = rng1.random(uids)

    streams.reset()

    _ = rng1.random(uids)
    s_after = rng0.random(uids)

    return np.all(s_before == s_after)


def test_repeatname(n=5):
    """ Test two streams with the same name """
    sc.heading('Testing if two streams with the same name are allowed')
    streams = ss.Streams()
    streams.initialize(base_seed=17)

    rng0 = ss.MultiStream('test')
    rng0.initialize(streams)

    rng1 = ss.MultiStream('test')
    try:
        rng1.initialize(streams)
        return False # Should not get here!
    except RepeatNameException as e:
        print(f'YAY! Got exception: {e}')
    return True


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    assert test_streams()
    assert test_seed()
    assert test_reset()
    assert test_step()
    assert test_initialize()
    assert test_seedrepeat()
    assert test_samplingorder()
    assert test_repeatname()

    sc.toc(T)
    print('Done.')