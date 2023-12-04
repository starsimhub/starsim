"""
Test Distributions from distributions.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import RNG


# %% Define the tests
def test_basic():
    dist = ss.normal(1,1) # Make a distribution
    dist()  # Draw a sample
    dist(10) # Draw several samples
    dist.sample(10) # Same as above
    ss.State('foo', float, fill_value=dist)  # Use distribution as the fill value for a state
    #disease.pars['immunity'] = dist  # Store the distribution as a parameter
    #disease.pars['immunity'].sample(5)  # Draw some samples from the parameter
    ss.poisson(rate=1).sample(10)  # Sample from a temporary distribution

def test_uniform_scalar(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform: Testing uniform with scalar parameters')

    rng = ss.MultiRNG('Uniform')
    rng.initialize(container=None, slots=n)
    d = ss.uniform(low=1, high=5, rng=rng)

    uids = np.array([1,3])
    draws = d.sample(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws

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

    low = lambda sim, uids: sim.people.age[uids]
    high = lambda sim, uids: sim.people.age[uids] + 1
    d = ss.uniform(low=low, high=high, rng='Uniform')
    d.initialize(sim)

    uids = np.array([1,3])
    draws = d.sample(uids)
    print(sim.people.age)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_uniform_array(n):
    """ Create a uniform distribution """
    sc.heading('test_uniform: Testing uniform with a array parameters')

    rng = ss.MultiRNG('Uniform')
    rng.initialize(container=None, slots=n)

    uids = np.array([1, 3])
    low = np.array([4, 5])
    high = np.array([5, 6])
    d = ss.uniform(low=low, high=high, rng=rng)

    draws = d.sample(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_gamma_scalar(n):
    """ Create a gamma distribution """
    sc.heading('test_gamma: Testing gamma with scalar parameters')

    rng = ss.MultiRNG('gamma')
    rng.initialize(container=None, slots=n)
    d = ss.gamma(shape=1, scale=2, rng=rng)

    uids = np.array([1,3])
    draws = d.sample(uids)
    print(f'Gamma sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_gamma_callable(n):
    """ Create a gamma distribution """
    sc.heading('test_gamma: Testing gamma with callable parameters')

    sim = ss.Sim().initialize()

    rng = ss.MultiRNG('gamma')
    rng.initialize(container=sim.rng_container, slots=sim.people.slot)

    shape = lambda sim, uids: sim.people.age[uids]
    scale = lambda sim, uids: sim.people.age[uids] + 1
    d = ss.gamma(shape=shape, scale=scale, rng=rng)
    d.initialize(sim)

    uids = np.array([1,3])
    draws = d.sample(uids)
    print(f'Gamma sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    return draws


def test_gamma_callable_bool(n):
    """ Create a gamma distribution """
    sc.heading('test_gamma: Testing gamma with callable parameters and boolean UIDs')

    sim = ss.Sim().initialize()

    rng = ss.MultiRNG('gamma')
    rng.initialize(container=sim.rng_container, slots=sim.people.slot)

    shape = lambda sim, uids: sim.people.age[uids]
    scale = lambda sim, uids: sim.people.age[uids] + 1
    d = ss.gamma(shape=shape, scale=scale, rng=rng)
    d.initialize(sim)

    uids = sim.people.age < 15
    draws = d.sample(uids)
    print(f'Gamma sample for uids {uids} returned {draws}')

    assert len(draws) == sum(uids)
    return draws


def test_gamma_array(n):
    """ Create a gamma distribution """
    sc.heading('test_gamma: Testing gamma with a array parameters')

    rng = ss.MultiRNG('gamma')
    rng.initialize(container=None, slots=5)

    uids = np.array([1, 3])
    shape = np.array([4, 5])
    scale = np.array([5, 6])
    d = ss.gamma(shape=shape, scale=scale, rng=rng)

    draws = d.sample(uids)
    print(f'Gamma sample for uids {uids} returned {draws}')

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
            #test_basic()
            test_uniform_scalar(n)
            test_uniform_scalar_str(n)
            test_uniform_callable(n)
            test_uniform_array(n)
            test_gamma_scalar(n)
            test_gamma_callable(n)
            test_gamma_array(n)
            test_gamma_callable_bool(n)

            times.append(sc.toc(T, doprint=False, output=True))

    print(times)
    ###sc.toc(T)
    print('Done.')