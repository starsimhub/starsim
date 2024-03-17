"""
Test the RNGs object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss
import scipy.stats as sps
import pytest

n = 5

def make_dist(name='test', **kwargs):
    """ Make a default Dist for testing """
    dist = ss.random(name=name, **kwargs).initialize()
    return dist

def make_dists(**kwargs):
    """ Make a Dists object with two distributions in it """
    distlist = [make_dist(), make_dist()]
    dists = ss.Dists(distlist)
    dists.initialize()
    return dists


# %% Define the tests

def test_seed():
    """ Test assignment of seeds """
    sc.heading('Testing assignment of seeds')
    
    # Create and initialize two distributions
    dists = make_dists()
    dist0, dist1 = dists.dists.values()

    print(f'Dists dist0 and dist1 were assigned seeds {dist0.seed} and {dist1.seed}, respectively')
    assert dist0.seed != dist1.seed
    return dist0, dist1


def test_reset(n=n):
    """ Sample, reset, sample """
    sc.heading('Testing sample, reset, sample')
    dists = make_dists()
    distlist = dists.dists.values()

    # Reset via the container, but only 
    before = sc.autolist()
    after = sc.autolist()
    for dist in distlist:
        before += list(dist(n))
    dists.reset() # Return to step 0
    for dist in distlist:
        after += list(dist(n))

    print(f'Initial sample:\n{before}')
    print(f'After reset sample:\n{after}')
    assert np.array_equal(before, after)
    
    return before, after


def test_jump(n=n):
    """ Sample, jump, sample """
    sc.heading('Testing sample, jump, sample')
    dists = make_dists()
    
    distlist = dists.dists.values()

    # Jump via the contianer
    before = sc.autolist()
    after = sc.autolist()
    for dist in distlist:
        before += list(dist(n))
    dists.jump(to=10) # Jump to 10th step
    for dist in distlist:
        after += list(dist(n))

    print(f'Initial sample:\n{before}')
    print(f'After jump sample:\n{after}')
    assert not np.array_equal(before, after)
    
    return before, after


def test_samplingorder(rng_container, dists, n=5):
    """ Ensure sampling from one RNG doesn't affect another """
    sc.heading('test_samplingorder: Testing from multiple random number generators to test if sampling order matters')

    uids = np.arange(0,n,2) # every other to make it interesting

    d0, d1 = dists
    if ss.options.multirng:
        d0.rng.initialize(rng_container, slots=n)
        d1.rng.initialize(rng_container, slots=n)

    s_before = d0.rvs(uids)
    _ = d1.rvs(uids)

    rng_container.reset()

    _ = d1.rvs(uids)
    s_after = d0.rvs(uids)

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

    T = sc.timer()

    o1 = test_seed()
    o2 = test_reset(n)
    o3 = test_jump(n)

    T.toc()
