"""
Test Dist and Dists
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import starsim as ss
import pylab as pl
import pytest

n = 1_000_000 # For testing statistical properties and performance of distributions
m = 5 # For just testing that drawing works
sc.options(interactive=False)


def plot_rvs(rvs, times=None, nrows=None):
    fig = pl.figure(figsize=(12,12))
    nrows, ncols = sc.getrowscols(len(rvs), nrows=nrows)
    for i,name,r in rvs.enumitems():
        pl.subplot(nrows, ncols, i+1)
        pl.hist(r)
        title = times[name] if times else name
        pl.title(title)
        sc.commaticks()
    sc.figlayout()
    return fig


# %% Define the tests
def test_dist(m=m):
    """ Test the Dist class """
    sc.heading('Testing the basic Dist call')
    dist = ss.Dist('random', 'test')
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
        obj.a.mylist = [ss.random(), ss.Dist('uniform', low=2, high=3)]
        obj.b = dict(d3=ss.weibull(a=2), d4=ss.delta(v=0.3))
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

@pytest.mark.skip
def test_exceptions(m=m):
    """ Check that exceptions are being appropriately raised """
    sc.heading('Testing exceptions and strict')
    
    # Create a strict distribution
    dist = ss.random()
    with pytest.raises(ss.dists.DistNotInitializedError):
        dist(m) # Check that we can't call an uninitialized
    
    # Initialize and check we can't call repeatedly
    dist.initialize()
    rvs = dist(m)
    with pytest.raises(ss.dists.DistNotReadyError):
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
    

# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    
    T = sc.timer()
    
    o1 = test_dist()
    o2 = test_custom_dists(do_plot=do_plot)
    o3 = test_dists(do_plot=do_plot)
    o4 = test_scipy()
    # o5 = test_exceptions() # TODO: re-enable once strict=True
    
    T.toc()
