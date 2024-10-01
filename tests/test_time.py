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
    for d in [d1,d2,d3,d4]: d.init()
    
    assert d1 + d2 == 2+3
    assert d3 + d4 == 2/0.1 + 3/0.2
    assert d3 * 2 == 2/0.1*2
    assert d3 / 2 == 2/0.1/2
    
    # Test rate dt
    r1 = ss.rate(2)
    r2 = ss.rate(3)
    r3 = ss.rate(2, parent_dt=0.1)
    r4 = ss.rate(3, parent_dt=0.2)
    for r in [r1,r2,r3,r4]: r.init()
    
    assert r1 + r2 == 2 + 3
    assert r3 + r4 == 2*0.1 + 3*0.2
    assert r3 * 2 == 2*0.1*2
    assert r3 / 2 == 2*0.1/2
    
    # Test duration units
    d5 = ss.dur(2, unit='year').init(parent_unit='day')
    d6 = ss.dur(3, unit='day').init(parent_unit='day')
    assert d5 + d6 == 2*365+3
    
    # Test rate units
    rval = 0.7
    r5 = ss.rate(rval, unit='week').init(parent_unit='day')
    r6 = ss.rate(rval).init(parent_dt=0.1)
    assert np.isclose(r5.values, rval/7) # A limitation of this approach, not exact!
    assert np.isclose(r6.values, rval/10)
    
    # Test time_prob
    tpval = 0.1
    tp0 = ss.time_prob(tpval).init(parent_dt=1.0)
    tp1 = ss.time_prob(tpval).init(parent_dt=0.5)
    tp2 = ss.time_prob(tpval).init(parent_dt=2)
    assert np.isclose(tp0.values, tpval)
    assert np.isclose(tp1.values, tpval/2, rtol=0.1)
    assert np.isclose(tp2.values, tpval*2, rtol=0.1)
    assert tp1.values > tpval/2
    assert tp2.values < tpval*2
    
    return d3, d4, r3, r4, tp1
    

def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')

    sis = ss.SIS(
        beta = ss.beta(0.05, 'day'),
        init_prev = ss.bernoulli(p=0.1),
        dur_inf = ss.lognorm_ex(mean=ss.dur(10, 'day')),
        waning = ss.rate(0.05, 'day'),
        imm_boost = 1.0,
    )

    rnet = ss.RandomNet(
        n_contacts = 10,
        dur = 0, # Note; network edge durations are required to have the same unit as the network
    )

    pars = dict(
        diseases = sis,
        networks = rnet,
        n_agents = small,
    )
    
    sims = sc.objdict()
    sims.y = ss.Sim(pars, unit='year', label='Year', start=2000, stop=2002, dt=1/365)
    sims.d = ss.Sim(pars, unit='day', label='Day', start='2000-01-01', stop='2002-01-01', dt=1)
    
    for sim in sims.values():
        sim.run()
        if do_plot:
            sim.plot()
    
    # Check that results match to within stochastic uncertainty
    rtol = 0.05
    vals = [sim.summary.sis_cum_infections for sim in [sims.y, sims.d]]
    assert np.isclose(*vals, rtol=rtol), f'Values for cum_infections do not match ({vals})'
        
    return sims


def test_multi_timestep(do_plot=False):
    sc.heading('Test behavior of different modules having different timesteps')
    
    pars = dict(
        diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1, beta=ss.beta(0.01)),
        demographics = ss.Births(unit='year', dt=0.25),
        networks = ss.RandomNet(unit='week'),
        n_agents = small,
    )
    
    sim = ss.Sim(pars, unit='day', dt=2, start='2000-01-01', stop='2002-01-01')
    sim.run()
    
    if do_plot:
        sim.plot()
        
    return sim


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    
    T = sc.timer()
    
    o1 = test_ratio()
    o2 = test_classes()
    o3 = test_units(do_plot)
    o4 = test_multi_timestep(do_plot)
    
    T.toc()
