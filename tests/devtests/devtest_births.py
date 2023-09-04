"""
Test births and deaths
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt


ppl = ss.People(10000)
births = ss.births({'birth_rates': 0.02})
deaths = ss.background_deaths({'death_rates': 0.015})

sim = ss.Sim(people=ppl, modules=[births, deaths])
sim.initialize()
sim.run()

fig, ax = plt.subplots(2, 1)
ax[0].plot(sim.tivec, sim.results.births)
ax[0].plot(sim.tivec, sim.results.deaths)
ax[1].plot(sim.tivec, sim.results.n_alive)

ax[0].title('Births and deaths')
ax[1].title('Population size')

