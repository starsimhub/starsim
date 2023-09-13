"""
Test births and deaths
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import pandas as pd


ppl = ss.People(10000)#, age_data=pd.read_csv('../test_data/nigeria_age.csv'))

# Parameters
simple_birth = {'birth_rates': 20}
simple_death = {'death_rates': 0.015}

realistic_birth = {'birth_rates': pd.read_csv(ss.root/'tests/test_data/nigeria_births.csv')}
realistic_death = {'death_rates': pd.read_csv(ss.root/'tests/test_data/nigeria_deaths.csv')}

series_death = {'death_rates': pd.Series(
            index=[0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            data=[0.0046355, 0.000776, 0.0014232, 0.0016693, 0.0021449, 0.0028822, 0.0039143, 0.0053676, 0.0082756, 0.01, 0.02, 0.03, 0.04, 0.06, 0.11, 0.15, 0.21, 0.30],
        )}

births = ss.births(realistic_birth)
deaths = ss.background_deaths(series_death)
gon = ss.Gonorrhea({'p_death': 0.5, 'initial': 1000})
sim = ss.Sim(people=ppl, modules=[births, deaths, gon], networks=ss.simple_sexual(), remove_dead=True, n_years=100)
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