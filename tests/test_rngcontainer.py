"""
Test the RNGs object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss

n = 5 # Default number of samples

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


def test_order(n=n):
    """ Ensure sampling from one RNG doesn't affect another """
    sc.heading('Testing from multiple random number generators to test if sampling order matters')
    dists = make_dists()
    d0, d1 = dists.dists.values()

    # Sample d0, d1
    before = d0(n)
    _ = d1(n)
    
    dists.reset()
    
    # Sample d1, d0
    _ = d1(n)
    after = d0(n)

    print(f'When sampling rng0 before rng1: {before}')
    print(f'When sampling rng0 after rng1: {after}')
    assert np.array_equal(before, after)

    return before, after


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    o1 = test_seed()
    o2 = test_reset(n)
    o3 = test_jump(n)
    o4 = test_order(n)

    T.toc()
