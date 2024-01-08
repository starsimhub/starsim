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
import pytest
import matplotlib.pyplot as plt

@pytest.fixture(params=[5, 50])
def n(request):
    yield request.param


# %% Define the tests
def test_basic():
    dist = sps.norm(loc=1, scale=1) # Make a distribution
    d = ScipyDistribution(dist)

    sample = d.rvs(1)  # Draw a sample
    samples = d.rvs(10) # Draw several samples

    if ss.options.multirng:
        print('Attempting to sample for specific UIDs, but have not provided a MultiRNG so an Exception should be raised.')
        with pytest.raises(Exception):
            d.rvs(np.array([1,3,8])) # Draw samples for specific uids
    else:
        d.rvs(np.array([1,3,8])) # Draw three samples

    mu = 5 # mean (mu) of the underlying normal
    sigma = 1 # stdev (sigma) of the underlying normal
    s = sigma 
    loc = 0
    scale = np.exp(mu)
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(sps.lognorm.ppf(0.01, s=s, loc=loc, scale=scale), sps.lognorm.ppf(0.99, s=s, loc=loc, scale=scale), 100)
    ax.plot(x, sps.lognorm.pdf(x, s=s, loc=loc, scale=scale), 'r-', lw=5, alpha=0.6, label='lognorm pdf')

    mean, var, skew, kurt = sps.lognorm.stats(s, loc=loc, scale=scale, moments='mvsk')
    print('mean', mean, 'var', var, 'skew', skew, 'kurt', kurt)
    print('calc mean', np.exp(mu + sigma**2/2), 'calc var', (np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2)) # Check against math

    temp_poisson_samples = sps.poisson(mu=2).rvs(10)  # Sample from a temporary distribution
    return sample, samples, temp_poisson_samples

def test_uniform_scalar(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform: Testing uniform with scalar parameters')

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
    sc.heading('test_uniform: Testing uniform with scalar parameters')

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
    sc.heading('test_uniform: Testing uniform with callable parameters')

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
    slots = np.array([4,2,3,2,2,3])
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
    # Start timing
    ###T = sc.tic()

    n = 5
    nTrials = 3

    for multirng in [True, False]:
        ss.options(multirng=multirng)
        sc.heading('Testing with multirng set to', multirng)

        times = []
        for trial in range(nTrials):
            T = sc.tic()

            # Run tests - some will only pass if multirng is True
            test_basic()
            test_uniform_scalar(n)
            test_uniform_scalar_str(n)
            test_uniform_callable(n)
            test_uniform_array(n)

            times.append(sc.toc(T, doprint=False, output=True))

    print(times)
    ###sc.toc(T)

    plt.show()
    print('Done.')