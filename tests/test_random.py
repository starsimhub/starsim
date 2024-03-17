"""
Test the RNG object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss

n = 5 # Number of samples to draw

def make_dist(seed=1, name='test', **kwargs):
    """ Make a default distribution for testing """
    dist = ss.Dist(dist='random', name=name, seed=seed, **kwargs)
    dist.initialize()
    return dist


# %% Define the tests
def test_random(n=n):
    """ Simple random draw """
    sc.heading('Testing simple random draw from a RNG object')
    
    dist = make_dist()
    draws = dist(n)
    print(f'Sampled {n} random draws', draws)
    assert len(draws) == n
    return draws


def test_reset(n=n):
    """ Sample, reset, sample """
    sc.heading('Testing sample, reset, sample')
    
    # Make dist and draw twice
    dist = make_dist()
    draws1 = dist(n)
    dist.reset()
    draws2 = dist(n)
    
    # Print and test results
    print(f'Random sample of size {n} returned:\n{draws1}')
    print(f'After reset, random sample of size {n} returned:\n{draws2}')
    assert np.array_equal(draws1, draws2)

    return draws1, draws2


def test_jump(n=n):
    """ Sample, jump, sample """
    sc.heading('Testing sample, jump, sample')
    
    dist = make_dist()
    draws1 = dist(n)
    dist.jump()
    draws2 = dist(n)
    
    print(f'Random sample of size {n} returned:\n{draws1}')
    print(f'After jump, random sample of size {n} returned:\n{draws2}')
    assert not np.array_equal(draws1, draws2)
    
    return draws1, draws2


def test_seed(n=n):
    """ Changing seeds """
    sc.heading('Testing sample with seeds 0 and 1')

    dist0 = make_dist(seed=0)
    dist1 = make_dist(seed=1)
    draws0 = dist0(n)
    draws1 = dist1(n)
    
    print(f'Random sample of size {n} for dist0 with seed=0 returned:\n{draws0}')
    print(f'Random sample of size {n} for dist1 with seed=1 returned:\n{draws1}')
    assert not np.array_equal(draws0, draws1)
    
    return draws0, draws1


def test_urvs(n=n):
    """ Simple sample from distribution by UID """
    sc.heading('Testing UID sample')
    
    dist = make_dist()
    uids = np.arange(0, n, 2) # every other to make it interesting
    draws = dist.urvs(uids)
    print(f'Created seed and sampled: {draws}')
    assert len(draws) == len(uids)
    
    # Draws without UIDs should match the first element only
    dist2 = make_dist()
    draws2 = dist2.rvs(len(uids))
    assert draws[0] == draws2[0]
    assert not np.array_equal(draws, draws2)
    
    return draws


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    o1 = test_random(n)
    o2 = test_reset(n)
    o3 = test_jump(n)
    o4 = test_seed(n)
    o5 = test_urvs(n)

    T.toc()
