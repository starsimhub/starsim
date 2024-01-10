"""
Test demographics
"""

# %% Imports and settings
import numpy as np
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps


def test_death_data():
    """ Show that death data can be added in multiple formats """

    # Death rate input types
    realistic_death = pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')
    series_death = pd.Series(
        index=[0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        data=[0.0046355, 0.000776, 0.0014232, 0.0016693, 0.0021449, 0.0028822, 0.0039143, 0.0053676, 0.0082756, 0.01,
              0.02, 0.03, 0.04, 0.06, 0.11, 0.15, 0.21, 0.30],
    )

    # Run through combinations
    ppl = ss.People(1000)
    bdm1 = ss.background_deaths(data=0.02)
    sim1 = ss.Sim(people=ppl, demographics=bdm1)
    sim1.run()

    ppl = ss.People(1000)
    bdm2 = ss.background_deaths(data=realistic_death)
    sim2 = ss.Sim(people=ppl, demographics=bdm2)
    sim2.run()

    ppl = ss.People(1000)
    bdm3 = ss.background_deaths(data=series_death)
    sim3 = ss.Sim(people=ppl, demographics=bdm3)
    sim3.run()

    ppl = ss.People(1000)
    bdm4 = ss.background_deaths(pars={'death_prob': sps.bernoulli(p=0.02)})
    sim4 = ss.Sim(people=ppl, demographics=bdm4)
    sim4.run()

    fig, ax = plt.subplots(2, 1)
    labels = ['Constant deaths 1', 'Realistic deaths', 'Series deaths', 'Constant deaths 2']
    for sn,sim in enumerate([sim1, sim2, sim3, sim4]):
        ax[0].plot(sim.tivec, sim.results.background_deaths.new, label=labels[sn])
        ax[1].plot(sim.tivec, sim.results.n_alive)

    ax[0].set_title('Births and deaths')
    ax[1].set_title('Population size')
    ax[0].legend()
    fig.tight_layout()
    plt.show()

    return



if __name__ == '__main__':

    test_death_data()

