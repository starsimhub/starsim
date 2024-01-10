"""
Test demographics
"""

# %% Imports and settings
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
    bdm1 = ss.background_deaths(pars={'death_prob': 0.015})
    sim1 = ss.Sim(people=ppl, demographics=bdm1, label='Constant deaths from bernoulli (0.01)')
    sim1.run()

    ppl = ss.People(1000)
    death_prob = ss.standardize_data(data=series_death, metadata=ss.omerge({
            'data_cols': {'year': 'Time', 'sex': 'Sex', 'age': 'AgeGrpStart', 'value': 'mx'},
            'sex_keys': {'f': 'Female', 'm': 'Male'},
        }))
    bdm2 = ss.background_deaths(pars={'death_prob': death_prob})
    sim2 = ss.Sim(people=ppl, demographics=bdm2, label='Series deaths')
    sim2.run()

    ppl = ss.People(1000)
    bdm3 = ss.background_deaths(pars={'death_prob': realistic_death})
    sim3 = ss.Sim(people=ppl, demographics=bdm3, label='Realistic deaths')
    sim3.run()

    fig, ax = plt.subplots(2, 1)
    for sim in [sim1, sim2, sim3,]:
        ax[0].plot(sim.tivec, sim.results.background_deaths.new, label=sim.label)
        ax[1].plot(sim.tivec, sim.results.n_alive)

    ax[0].set_title('New background deaths')
    ax[1].set_title('Population size')
    ax[1].set_xlabel('Time step')
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Count')
    ax[0].legend()
    fig.tight_layout()
    plt.show()

    return

def test_birth_data():
    """ Show that birth data can be added in multiple formats """

    return

if __name__ == '__main__':
    # test_death_data()
    test_birth_data()