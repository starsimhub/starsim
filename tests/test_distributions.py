"""
Test Distributions from distributions.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import RNG
import scipy.stats as sps
from stisim.distributions import ScipyDistribution
import pytest

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
        d.rvs(np.array([1,3,8])) # Draw samples for specific uids

    '''
    ss.State('foo', float, fill_value=dist)  # Use distribution as the fill value for a state
    #disease.pars['immunity'] = dist  # Store the distribution as a parameter
    #disease.pars['immunity'].sample(5)  # Draw some samples from the parameter
    '''
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

@pytest.mark.skip(reason="Random number generators not yet implemented using strings")
def test_uniform_scalar_str(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform: Testing uniform with scalar parameters')

    #rng = ss.MultiRNG('Uniform')
    #rng.initialize(container=None, slots=n)
    d = ss.uniform(low=1, high=5, rng='Uniform')
    d.rng.initialize(container=None, slots=n) # Only really needed for testing as initializing the distribution will do something similar.

    uids = np.array([1,3])
    draws = d.sample(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_uniform_callable(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform: Testing uniform with callable parameters')

    sim = ss.Sim().initialize()

    rng = ss.MultiRNG('Uniform')
    rng.initialize(container=None, slots=n)

    loc = lambda sim, uids: sim.people.age[uids] # Low
    scale = 1 # Width, could also be a lambda
    dist = sps.uniform(loc=loc, scale=scale)
    dist.random_state = rng

    d = ScipyDistribution(dist)
    d.gen.dist.initialize(sim) # Handled automatically in the model code

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
    print('Done.')