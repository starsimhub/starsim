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
    
    o1 = test_units()
    
    T.toc()
