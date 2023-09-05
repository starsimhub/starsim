"""
Test births and deaths
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import pandas as pd

ppl = ss.People(10000)

# Parameters
simple_birth = {'birth_rates': 20}
simple_death = {'death_rates': 0.015}

realistic_birth = {'birth_rates': pd.read_csv('../test_data/nigeria_births.csv')}
realistic_death = {'death_rates': pd.read_csv('../test_data/nigeria_deaths.csv')}

births = ss.births(realistic_birth)
deaths = ss.background_deaths(realistic_death)

sim = ss.Sim(people=ppl, modules=[births, deaths])
sim.initialize()
sim.run()

fig, ax = plt.subplots(2, 1)
ax[0].plot(sim.tivec, sim.results.births.new, label='Births')
ax[0].plot(sim.tivec, sim.results.background_deaths.new, label='Deaths')
ax[1].plot(sim.tivec, sim.results.n_alive)

ax[0].set_title('Births and deaths')
ax[1].set_title('Population size')
ax[0].legend()

plt.show()