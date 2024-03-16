"""
Test Dist and Dists
"""

import numpy as np
import sciris as sc
import starsim as ss
import pylab as pl
import pytest

n = 1_000_000
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
def test_dist(n=n):
    """ Test the Dist class """
    dist = ss.Dist('random', 'test')
    dist.initialize()
    rvs = dist(n)
    assert 0 < rvs.min() < 1
    return rvs


def test_custom_dists(n=n, do_plot=False):
    """ Test all custom dists """
    o = sc.objdict()
    dists = sc.objdict()
    rvs = sc.objdict()
    times = sc.objdict()
    for name in ss.dists.dist_list:
        func = getattr(ss, name)
        dist = func(name='test')
        dist.initialize()
        dists[name] = dist
        sc.tic()
        rvs[name] = dist.rvs(n)
        times[name] = sc.toc(name, unit='ms', output='message')
    
    if do_plot:
        plot_rvs(rvs, times=times, nrows=5)
    
    o.dists = dists
    o.rvs = rvs
    o.times = times
    return o
        

def test_dists(n=n, do_plot=False):
    """ Test the Dists container """
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
                with pytest.raises(ss.dists.DistNotReady):
                    dist(n) # Check that we can't call an already-used distribution
                dist.jump() # Reset
            testvals[i,j] = rvs[0][283] # Pick one of the random values and store it
    
    # Check that results are as expected
    assert np.all(testvals[0,:] == testvals[1,:]) # Newly initialized objects should match
    assert np.all(testvals[:,0] != testvals[:,1]) # After jumping values should be different
            
    if do_plot:
        plot_rvs(rvs)
    
    o = sc.objdict()
    o.dists = dists
    o.rvs = rvs
    return dists


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    
    T = sc.timer()
    
    o1 = test_dist()
    o2 = test_custom_dists(do_plot=do_plot)
    o3 = test_dists(do_plot=do_plot)
    
    T.toc()
