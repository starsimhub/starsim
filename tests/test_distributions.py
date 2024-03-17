"""
Test SciPy distributions from distributions.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss
import scipy.stats as sps

n = 5


# %% Define the tests

def test_basic():
    """ Basic scipy.stats test """
    sc.heading('Test basic scipy.stats usage')
    spsdist = sps.norm(loc=1, scale=1) # Make a distribution
    d = ss.Dist(dist=spsdist).initialize() # Convert it to Starsim
    sample = d.rvs(1)  # Draw a sample

    # Draw some samples
    d.reset()
    m = 10
    samples = d.rvs(m) # Draw several samples
    
    # Draw UID samples
    d.reset()
    uids = np.array([0,3,4])
    samples_uid = d.urvs(uids=uids)
    
    # Print and test
    for s in [sample, samples, samples_uid]:
        print(s)
    assert sample == samples[0] == samples_uid[0], 'Samples should match after reset'
    assert len(samples) == m, 'Incorrect number of samples'
    assert len(samples_uid) == len(uids), 'Incorrect number of samples'
    
    return d


def test_uniform_scalar(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform_scalar: Testing uniform with scalar parameters')

    rng = ss.RNG('Uniform')
    rng.initialize(container=None, slots=n)
    dist = sps.uniform(loc=1, scale=4)
    dist.random_state = rng
    d = ss.ScipyDistribution(dist)

    uids = np.array([1,3])
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return d


def test_uniform_callable(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform_callable: Testing uniform with callable parameters')

    sim = ss.Sim().initialize()

    loc = lambda self, sim, uids: sim.people.age[uids] # Low
    scale = 1 # Width, could also be a lambda
    dist = sps.uniform(loc=loc, scale=scale)

    d = ss.ScipyDistribution(dist, 'Uniform')
    d.initialize(sim, context=None)

    uids = np.array([1,3])
    draws = d.rvs(uids)
    print(sim.people.age[uids])
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_uniform_array(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform: Testing uniform with a array parameters')

    rng = ss.RNG('Uniform')
    rng.initialize(container=None, slots=n)

    uids = np.array([1, 3])
    loc = np.array([1, 100]) # Low
    scale = np.array([2, 25]) # Width

    dist = sps.uniform(loc=loc, scale=scale)
    dist.random_state = rng

    d = ss.ScipyDistribution(dist)
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_repeat_slot():
    """ Test behavior of repeated slots """
    sc.heading('test_repeat_slot: Test behavior of repeated slots')

    rng = ss.RNG('Uniform')
    slots = ss.UIDArray(values=np.array([4,2,3,2,2,3]), uid=np.arange(6))
    n = len(slots)
    rng.initialize(container=None, slots=slots)

    uids = np.arange(n)
    loc = np.arange(n) # Low
    scale = 1 # Width

    dist = sps.uniform(loc=loc, scale=scale)
    dist.random_state = rng

    d = ss.ScipyDistribution(dist)
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)

    unique_slots = np.unique(slots)
    for s in unique_slots:
        inds = np.where(slots==s)[0]
        frac, integ = np.modf(draws[inds])
        assert np.allclose(integ, loc[inds]) # Integral part should match the loc
        if ss.options.multirng:
            # Same random numbers, so should be same fractional part
            assert np.allclose(frac, frac[0])
    return draws



# %% Run as a script
if __name__ == '__main__':
    
    T = sc.timer()

    o1 = test_basic()
    # test_uniform_scalar(n)
    # test_uniform_scalar_str(n)
    # test_uniform_callable(n)
    # test_uniform_array(n)

    T.toc()
