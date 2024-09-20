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
    
    d1 = ss.dur(2)
    d2 = ss.dur(3)
    d3 = ss.dur(2, parent_dt=0.1)
    d4 = ss.dur(3, parent_dt=0.2)
    d5 = ss.dur(2, self_dt=10)
    d6 = ss.dur(3, self_dt=5)
    for d in [d1,d2,d3,d4,d5,d6]: d.initialize()
    
    assert d1 + d2 == 5
    assert d3 + d4 == 35
    assert d3 * 2 == 40
    assert d3 / 2 == 10
    assert d5 + d6 == 35
    
    r1 = ss.rate(2)
    r2 = ss.rate(3)
    r3 = ss.rate(2, parent_dt=0.1)
    r4 = ss.rate(3, parent_dt=0.2)
    r5 = ss.rate(2, self_dt=10)
    r6 = ss.rate(3, self_dt=5)
    for r in [r1,r2,r3,r4,r5,r6]: r.initialize()
    
    assert r1 + r2 == 5
    assert r3 + r4 == 0.8
    assert r3 * 2 == 0.4
    assert r3 / 2 == 0.1
    assert r5 + r6 == 0.8
    
    d5 = ss.dur(2, unit='year').initialize(parent_unit='day')
    d6 = ss.dur(3, unit='day').initialize(parent_unit='day')
    assert d5 + d6 == 2*365+3
    
    r5 = ss.rate(0.7, unit='week', self_dt=1).initialize(parent_unit='day')
    r6 = ss.rate(0.7, self_dt=1).initialize(parent_dt=0.1)
    assert np.isclose(r5.x, 0.1) # A limitation of this approach, not exact!
    assert np.isclose(r6.x, 0.07)
    
    return d3, d4, r3, r4
    

def test_units():
    sc.heading('Test behavior of year vs day units')
    
    pars = dict(
        diseases = 'sis',
        networks = 'random',
        n_agents = small,
    )
    
    sims = sc.objdict()
    sims.y = ss.Sim(pars, unit='year', label='Year', start=2000, end=2002, dt=1/365)
    sims.d = ss.Sim(pars, unit='day', label='Day', start='2000-01-01', end='2002-01-01', dt=1)
    
    for sim in sims.values():
        sim.run()
        
    rtol = 0.01
    # assert np.isclose(sims.y.summary.cum_infections, sims.d.summary.cum_infections, rtol=rtol)
        
    return sims



# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    
    T = sc.timer()
    
    o1 = test_ratio()
    o2 = test_classes()
    # o3 = test_units()
    
    T.toc()
