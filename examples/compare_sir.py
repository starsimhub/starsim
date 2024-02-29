"""
Compare Starsim and SimpleABM in a simple, static SIR model.

NB: because the models are parameterized differently, they are not expected to give
identical results.
"""

import simpleabm as sa
import starsim as ss
import sciris as sc

sasim = sa.Sim()

sssim = ss.Sim(
    pars=dict(
        n_agents=1e3, 
        start=2000, 
        end=2020, 
        dt=0.2, 
        verbose=False,
        diseases=dict(
            type='sir', 
            beta=dict(static=0.1),
            dur_inf=10,
        ),
        networks='static'
    )
)
sssim.initialize()

with sc.timer('SimpleABM'):
    sasim.run()

with sc.timer('Starsim'):
    sssim.run()
    
sasim.plot()
sssim.diseases.sir.plot()