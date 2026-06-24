"""
A multi-module model: disease + network + vital dynamics.

The engine supports a dynamic population: births append agents (who enter
susceptible) and deaths remove them. Everything composes the same way -- just
pass ssr demographics modules alongside the disease and network.
"""
import numpy as np
import starsim as ss
import starsim.rust as ssr

sim = ss.Sim(
    diseases     = ssr.SIS(beta=0.05, init_prev=0.01),
    networks     = ssr.RandomNet(n_contacts=10),
    demographics = [ssr.Births(birth_rate=30), ssr.Deaths(death_rate=10)],
    n_agents     = 10_000,
    dur          = 50,
    verbose      = 0,
)
sim.run()  # whole loop (including births/deaths) runs in Rust

res = sim.results.sis
# Living population each step ~ susceptible + infected (recovered fold back to S in SIS)
n_alive = np.asarray(res.n_susceptible) + np.asarray(res.n_infected)
print(f'population: start={n_alive[0]:.0f} -> end={n_alive[-1]:.0f} '
      f'(net growth from births>deaths)')
print(f'infected:   peak={res.n_infected.max():.0f}, end={res.n_infected[-1]:.0f}')
