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
ax[0].plot(sim.tivec, sim.results.births.new)
ax[0].plot(sim.tivec, sim.results.background_deaths.new)
ax[1].plot(sim.tivec, sim.results[0])

ax[0].set_title('Births and deaths')
ax[1].set_title('Population size')

plt.show()