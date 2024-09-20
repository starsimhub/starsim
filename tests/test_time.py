"""
Test different time units and timesteps
"""

import numpy as np
import sciris as sc
import starsim as ss

small = 100
medium = 1000
sc.options(interactive=False)


# %% Define the tests
def test_ratio():
    sc.heading('Test behavior of time_ratio()')
    
    assert ss.time_ratio() == 1.0
    assert ss.time_ratio(dt1=1.0, dt2=0.1) == 10.0
    assert ss.time_ratio(dt1=0.5, dt2=5) == 0.1
    assert ss.time_ratio(dt1=0.5, dt2=5) == 0.1
    
    assert ss.time_ratio(unit1='year', unit2='day') == 365
    assert ss.time_ratio(unit1='day', unit2='year') == 1/365
    
    return


def test_classes():
    sc.heading('Test behavior of dur() and rate()')
    
    # Test duration dt
    d1 = ss.dur(2)
    d2 = ss.dur(3)
    d3 = ss.dur(2, parent_dt=0.1)
    d4 = ss.dur(3, parent_dt=0.2)
    d5 = ss.dur(2, self_dt=10)
    d6 = ss.dur(3, self_dt=5)
    for d in [d1,d2,d3,d4,d5,d6]: d.initialize()
    
    assert d1 + d2 == 2+3
    assert d3 + d4 == 2/0.1 + 3/0.2
    assert d3 * 2 == 2/0.1*2
    assert d3 / 2 == 2/0.1/2
    assert d5 + d6 == 2*10 + 3*5
    
    # Test rate dt
    r1 = ss.rate(2)
    r2 = ss.rate(3)
    r3 = ss.rate(2, parent_dt=0.1)
    r4 = ss.rate(3, parent_dt=0.2)
    r5 = ss.rate(2, self_dt=10)
    r6 = ss.rate(3, self_dt=5)
    for r in [r1,r2,r3,r4,r5,r6]: r.initialize()
    
    assert r1 + r2 == 2 + 3
    assert r3 + r4 == 2*0.1 + 3*0.2
    assert r3 * 2 == 2*0.1*2
    assert r3 / 2 == 2*0.1/2
    assert r5 + r6 == 2/10 + 3/5
    
    # Test duration units
    d7 = ss.dur(2, unit='year').initialize(parent_unit='day')
    d8 = ss.dur(3, unit='day').initialize(parent_unit='day')
    assert d7 + d8 == 2*365+3
    
    # Test rate units
    rval = 0.7
    r7 = ss.rate(rval, unit='week', self_dt=1).initialize(parent_unit='day')
    r8 = ss.rate(rval, self_dt=1).initialize(parent_dt=0.1)
    assert np.isclose(r7.x, rval/7) # A limitation of this approach, not exact!
    assert np.isclose(r8.x, rval/10)
    
    # Test time_prob
    tpval = 0.1
    tp0 = ss.time_prob(tpval).initialize(parent_dt=1.0)
    tp1 = ss.time_prob(tpval).initialize(parent_dt=0.5)
    tp2 = ss.time_prob(tpval).initialize(parent_dt=2)
    assert tp0.x == tpval
    assert np.isclose(tp1.x, tpval/2, rtol=0.1)
    assert np.isclose(tp2.x, tpval*2, rtol=0.1)
    assert tp1.x > tpval/2
    assert tp2.x < tpval*2
    
    return d3, d8, r3, r8, tp1
    

def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')
    
    pars = dict(
        diseases = 'sis',
        networks = 'random',
        n_agents = medium,
    )
    
    sims = sc.objdict()
    sims.y = ss.Sim(pars, unit='year', label='Year', start=2000, end=2002, dt=1/365)
    sims.d = ss.Sim(pars, unit='day', label='Day', start='2000-01-01', end='2002-01-01', dt=1)
    
    for sim in sims.values():
        sim.run()
        if do_plot:
            sim.plot()
    
    # Uncomment this test once it might potentially pass lol
    rtol = 0.01
    vals = [sim.summary.sis_cum_infections for sim in [sims.y, sims.d]]
    # assert np.isclose(*vals, rtol=rtol), f'Values for cum_infections do not match ({vals})'
        
    return sims


def test_multi_timestep(do_plot=False):
    sc.heading('Test behavior of different modules having different timesteps')
    
    pars = dict(
        diseases = ss.SIS(unit='day', dt=1.0),
        demographics = ss.Pregnancy(unit='year', dt=0.25),
        networks = ss.RandomNet(unit='week'),
        n_agents = medium,
    )
    
    sim = ss.Sim(pars, unit='day', dt=2, start='2000-01-01', end='2002-01-01')
    sim.run()
    
    if do_plot:
        sim.plot()
        
    return sim


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    
    T = sc.timer()
    
    # o1 = test_ratio()
    # o2 = test_classes()
    # o3 = test_units(do_plot)
    o4 = test_multi_timestep(do_plot)
    
    T.toc()
