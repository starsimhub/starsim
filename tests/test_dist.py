"""
Test the Dist object from distributions.py
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as pl
import pytest

n = 1_000_000 # For testing statistical properties and performance of distributions
m = 5 # For just testing that drawing works
sc.options(interactive=False)


def plot_rvs(rvs, times=None, nrows=None):
    fig = pl.figure(figsize=(12,12))
    nrows, ncols = sc.getrowscols(len(rvs), nrows=nrows)
    for i,name,r in rvs.enumitems():
        pl.subplot(nrows, ncols, i+1)
        pl.hist(r.astype(float))
        title = times[name] if times else name
        pl.title(title)
        sc.commaticks()
    sc.figlayout()
    return fig


# %% Define the tests
def test_dist(m=m):
    """ Test the Dist class """
    sc.heading('Testing the basic Dist call')
    dist = ss.Dist(distname='random', name='test')
    dist.initialize()
    rvs = dist(m)
    print(rvs)
    assert 0 < rvs.min() < 1, 'Values should be between 0 and 1'
    return rvs


def test_custom_dists(n=n, do_plot=False):
    """ Test all custom dists """
    sc.heading('Testing all custom distributions')
    
    o = sc.objdict()
    dists = sc.objdict()
    rvs = sc.objdict()
    times = sc.objdict()
    for name in ss.dist_list:
        func = getattr(ss, name)
        dist = func(name='test')
        dist.initialize()
        dists[name] = dist
        sc.tic()
        rvs[name] = dist.rvs(n)
        times[name] = sc.toc(name, unit='ms', output='message')
        print(f'{name:10s}: mean = {rvs[name].mean():n}')
    
    if do_plot:
        plot_rvs(rvs, times=times, nrows=5)
    
    o.dists = dists
    o.rvs = rvs
    o.times = times
    return o
        

def test_dists(n=n, do_plot=False):
    """ Test the Dists container """
    sc.heading('Testing Dists container')
    testvals = np.zeros((2,2))

    # Create the objects twice
    for i in range(2):
        
        # Create a complex object containing various distributions
        obj = sc.prettyobj()
        obj.a = sc.objdict()
        obj.a.mylist = [ss.random(), ss.Dist(distname='uniform', low=2, high=3)]
        obj.b = dict(d3=ss.weibull(c=2), d4=ss.delta(v=0.3))
        dists = ss.Dists(obj)
        
        # Call each distribution twice
        for j in range(2):
            rvs = sc.objdict()
            for key,dist in dists.dists.items():
                rvs[str(dist)] = dist(n)
                dist.jump() # Reset
            testvals[i,j] = rvs[0][283] # Pick one of the random values and store it
    
    # Check that results are as expected
    print(testvals)
    assert np.all(testvals[0,:] == testvals[1,:]), 'Newly initialized objects should match'
    assert np.all(testvals[:,0] != testvals[:,1]), 'After jumping, values should be different'
            
    if do_plot:
        plot_rvs(rvs)
    
    o = sc.objdict()
    o.dists = dists
    o.rvs = rvs
    return dists


def test_scipy(m=m):
    """ Test that SciPy distributions also work """
    sc.heading('Testing SciPy distributions')
    
    # Make SciPy distributions in two different ways
    dist1 = ss.Dist(dist=sps.expon, name='scipy', scale=2).initialize() # Version 1: callable
    dist2 = ss.Dist(dist=sps.expon(scale=2), name='scipy').initialize() # Version 2: frozen
    rvs1 = dist1(m)
    rvs2 = dist2(m)
    
    # Check that they match
    print(rvs1)
    assert np.array_equal(rvs1, rvs2), 'Arrays should match'
    
    return dist1, dist2


def test_exceptions(m=m):
    """ Check that exceptions are being appropriately raised """
    sc.heading('Testing exceptions and strict')
    
    # Create a strict distribution
    dist = ss.random(strict=True, auto=False)
    with pytest.raises(ss.distributions.DistNotInitializedError):
        dist(m) # Check that we can't call an uninitialized
    
    # Initialize and check we can't call repeatedly
    dist.initialize()
    rvs = dist(m)
    with pytest.raises(ss.distributions.DistNotReadyError):
        dist(m) # Check that we can't call an already-used distribution
    
    # Check that we can with a non-strict Dist
    dist2 = ss.random(strict=False)
    rvs2 = sc.autolist()
    for i in range(2):
        rvs2 += dist2(m) # We should be able to call multiple times with no problem
    
    print(rvs)
    print(rvs2)
    assert np.array_equal(rvs, rvs2[0]), 'Separate dists should match'
    assert not np.array_equal(rvs2[0], rvs2[1]), 'Multiple calls to the same dist should not match'
    
    return dist, dist2
    

def test_reset(m=m):
    """ Check that reset works as expected """
    sc.heading('Testing reset')
    
    # Create and draw two sets of random numbers
    dist = ss.random(seed=533).initialize()
    r1 = dist.rvs(m)
    r2 = dist.rvs(m)
    assert all(r1 != r2)
    
    # Reset to the most recent state
    dist.reset(-1)
    r3 = dist.rvs(m)
    assert all(r3 == r2)
    
    # Reset to the initial state
    dist.reset(0)
    r4 = dist.rvs(m)
    assert all(r4 == r1)
    
    for r in [r1, r2, r3, r4]:
        print(r)
    
    return dist


def test_callable(n=n):
    """ Test callable parameters """
    sc.heading('Testing a uniform distribution with callable parameters')
    
    # Define a fake people object
    sim = sc.prettyobj()
    sim.n = 10
    sim.people = sc.prettyobj()
    sim.people.uid = np.arange(sim.n)
    sim.people.slot = np.arange(sim.n)
    sim.people.age = np.random.uniform(0, 90, size=sim.n)

    # Define a parameter as a function
    def custom_loc(module, sim, uids):
        out = sim.people.age[uids]
        return out
    
    scale = 1
    d = ss.normal(loc=custom_loc, scale=scale).initialize(sim=sim)

    uids = np.array([1, 3, 7, 9])
    draws = d.rvs(uids)
    print(f'Input ages were: {sim.people.age[uids]}')
    print(f'Output samples were: {draws}')

    meandiff = np.abs(sim.people.age[uids] - draws).mean()
    assert meandiff < scale*3
    return d


def test_array(n=n):
    """ Test array parameters """
    sc.heading('Testing uniform with a array parameters')

    uids = np.array([1, 3])
    low  = np.array([1, 100]) # Low
    high = np.array([3, 125]) # High

    d = ss.uniform(low=low, high=high).initialize(slots=np.arange(uids.max()+1))
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    for i in range(len(uids)):
        assert low[i] < draws[i] < low[i] + high[i], 'Invalid value'
    return draws


def test_repeat_slot():
    """ Test behavior of repeated slots """
    sc.heading('Test behavior of repeated slots')

    # Initialize parameters
    slots = np.array([4,2,3,2,2,3])
    n = len(slots)
    uids = np.arange(n)
    low = np.arange(n)
    high = low + 1

    # Draw values
    d = ss.uniform(low=low, high=high).initialize(slots=slots)
    draws = d.rvs(uids)
    
    # Print and test
    print(f'Uniform sample for slots {slots} returned {draws}')
    assert len(draws) == len(slots)

    unique_slots = np.unique(slots)
    for s in unique_slots:
        inds = np.where(slots==s)[0]
        frac, integ = np.modf(draws[inds])
        assert np.allclose(integ, low[inds]), 'Integral part should match the low parameter'
        assert np.allclose(frac, frac[0]), 'Same random numbers, so should be same fractional part'
    return draws



# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    
    T = sc.timer()
    
    o1 = test_dist()
    o2 = test_custom_dists(do_plot=do_plot)
    o3 = test_dists(do_plot=do_plot)
    o4 = test_scipy()
    o5 = test_exceptions()
    o6 = test_reset()
    o7 = test_callable()
    o8 = test_array()
    o9 = test_repeat_slot()
    
    T.toc()
