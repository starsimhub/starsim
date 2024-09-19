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
def test_classes():
    sc.heading('Test behavior of dur() and rate()')
    
    d1 = ss.dur(2)
    d2 = ss.dur(3)
    d3 = ss.dur(2, dt=0.1)
    d4 = ss.dur(3, dt=0.2)
    
    assert d1 + d2 == 5
    assert d3 + d4 == 35
    assert d3 * 2 == 40
    assert d3 / 2 == 10
    
    r1 = ss.rate(2)
    r2 = ss.rate(3)
    r3 = ss.rate(2, dt=0.1)
    r4 = ss.rate(3, dt=0.2)
    
    assert r1 + r2 == 5
    assert r3 + r4 == 0.8
    assert r3 * 2 == 0.4
    assert r3 / 2 == 0.1
    
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
    
    o1 = test_classes()
    # o2 = test_units()
    
    
    T.toc()
