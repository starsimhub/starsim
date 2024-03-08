"""
Test Distributions from distributions.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss
from starsim.random import RNG
import scipy.stats as sps
from starsim.distributions import ScipyDistribution
from starsim.states import UIDArray
import pytest
import matplotlib.pyplot as plt


do_plot = True
sc.options(interactive=False) # Assume not running interactively


@pytest.fixture(params=[5, 50])
def n(request):
    yield request.param


# %% Define the tests
def test_basic(n):
    dist = sps.norm(loc=1, scale=1) # Make a distribution
    if ss.options.multirng:
        rng = ss.MultiRNG('Uniform')
        rng.initialize(container=None, slots=n)
        dist.random_state = rng
    d = ScipyDistribution(dist)

    sample = d.rvs(1)  # Draw a sample

    # Reset before the next call
    if ss.options.multirng: d.random_state.reset()
    samples = d.rvs(10) # Draw several samples

    if ss.options.multirng: d.random_state.reset()
    samples_uid = d.rvs(size=np.array([1,3,4])) # Draw three samples

    return sample, samples, samples_uid


def test_uniform_scalar(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform_scalar: Testing uniform with scalar parameters')

    rng = ss.MultiRNG('Uniform')
    rng.initialize(container=None, slots=n)
    dist = sps.uniform(loc=1, scale=4)
    dist.random_state = rng
    d = ScipyDistribution(dist)

    uids = np.array([1,3])
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_uniform_scalar_str(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform_scalar_str: Testing uniform with scalar parameters')

    dist = sps.uniform(loc=1, scale=4)
    d = ScipyDistribution(dist, 'Uniform') # String here!
    if ss.options.multirng:
        d.rng.initialize(container=None, slots=n) # Only really needed for testing as initializing the distribution will do something similar.

    uids = np.array([1,3])
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_uniform_callable(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform_callable: Testing uniform with callable parameters')

    sim = ss.Sim().initialize()

    loc = lambda self, sim, uids: sim.people.age[uids] # Low
    scale = 1 # Width, could also be a lambda
    dist = sps.uniform(loc=loc, scale=scale)

    d = ScipyDistribution(dist, 'Uniform')
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

    rng = ss.MultiRNG('Uniform')
    rng.initialize(container=None, slots=n)

    uids = np.array([1, 3])
    loc = np.array([1, 100]) # Low
    scale = np.array([2, 25]) # Width

    dist = sps.uniform(loc=loc, scale=scale)
    dist.random_state = rng

    d = ScipyDistribution(dist)
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_repeat_slot():
    """ Test behavior of repeated slots """
    sc.heading('test_repeat_slot: Test behavior of repeated slots')

    rng = ss.MultiRNG('Uniform')
    slots = UIDArray(values=np.array([4,2,3,2,2,3]), uid=np.arange(6))
    n = len(slots)
    rng.initialize(container=None, slots=slots)

    uids = np.arange(n)
    loc = np.arange(n) # Low
    scale = 1 # Width

    dist = sps.uniform(loc=loc, scale=scale)
    dist.random_state = rng

    d = ScipyDistribution(dist)
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
    sc.options(interactive=do_plot)

    n = 5
    nTrials = 3

    times = {}
    for multirng in [True, False]:
        ss.options(multirng=multirng)
        key = f'MultiRNG: {multirng}'
        times[key] = []
        sc.heading('Testing with multirng set to', multirng)

        for trial in range(nTrials):
            T = sc.tic()

            # Run tests - some will only pass if multirng is True
            test_basic(n)
            test_uniform_scalar(n)
            test_uniform_scalar_str(n)
            test_uniform_callable(n)
            test_uniform_array(n)

            times[key].append(sc.toc(T, doprint=False, output=True))

    print('Times:', times)

    plt.show()
    print('Done.')
