# Trialing different ways of including births

import numpy as np
import sciris as sc
from collections import defaultdict
import matplotlib.pyplot as plt
import stisim as ss

# Make some people with no contact networks


# Option 1: default, simplest option assumes a static population
def test_static():
    people = ss.People(100)
    sim = ss.Sim(people, label='static')
    sim.run()
    return sim


# Option 2: next level up in terms of complexity - new agents added ex nihilo via a birth rate
def test_births():
    people = ss.People(100)
    pars = defaultdict()
    pars['birth_rate'] = 0.03  # Could also make this time varying
    pars['death_rate'] = 0.02  # Ditto
    sim = ss.Sim(people, pars=pars, label='births')
    sim.run()
    return sim


# Option 3: new agents added via pregnancies
def test_pregnancy():
    people = ss.People(100)
    people.contacts['maternal'] = ss.Maternal()
    pars = defaultdict()
    pars['death_rate'] = 0.02
    sim = ss.Sim(people, [ss.Pregnancy], pars=pars, label='pregnancies')  # Question, how can we change the birth rate?
    sim.run()
    plt.figure()
    plt.plot(sim.tvec, sim.results.pregnancy.births)
    plt.title('Births')
    plt.show()
    return sim


# RUn and compare pop sizes
sim0 = test_static()
sim1 = test_births()
sim2 = test_pregnancy()

result_summary = f'# agents by population demography method: \n'
for sim in [sim0, sim1, sim2]:
    result_summary += f'{sim.label.rjust(11)}: {sim.people.n} \n'
print(result_summary)

print('Done.')
