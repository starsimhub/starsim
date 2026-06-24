"""
An SIR model on the engine.

Swapping the disease module is all it takes -- ssr.SIR gives susceptible ->
infected -> recovered (permanent immunity) instead of SIS's S <-> I. The network
and the rest of the API are unchanged.
"""
import numpy as np
import starsim as ss
import starsim.rust as ssr

sim = ss.Sim(
    diseases = ssr.SIR(beta=0.05, init_prev=0.01),
    networks = ssr.RandomNet(n_contacts=10),
    n_agents = 50_000,
    dur      = 80,
    verbose  = 0,
)
sim.run()

res = sim.results.sir
print(f'susceptible: {res.n_susceptible[0]:.0f} -> {res.n_susceptible[-1]:.0f}')
print(f'infected:    peak={res.n_infected.max():.0f}')
print(f'recovered:   {res.n_recovered[0]:.0f} -> {res.n_recovered[-1]:.0f} '
      f'(classic SIR burn-through)')
