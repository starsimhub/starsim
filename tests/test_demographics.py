"""
Test demographics (background deaths) by showing that death data can be added in multiple formats.
"""

# %% Imports and settings
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps


def test_fixed_death_rate():
    """ Simple fixed death rate for all agents """
    ppl = ss.People(1000)
    bdm1 = ss.background_deaths(pars={'death_rate': 0.015})
    sim1 = ss.Sim(people=ppl, demographics=bdm1, label='Constant death rate')
    sim1.run()
    return sim1


def test_series_death_rate():
    """ Death rate from a pandas series """
    series_death = pd.Series(
        index=[0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        data=[0.0046355, 0.000776, 0.0014232, 0.0016693, 0.0021449, 0.0028822, 0.0039143, 0.0053676, 0.0082756, 0.01,
              0.02, 0.03, 0.04, 0.06, 0.11, 0.15, 0.21, 0.30],
    )
    ppl = ss.People(1000)
    death_rate = ss.standardize_data(data=series_death, metadata=ss.omerge({
            'data_cols': {'year': 'Time', 'sex': 'Sex', 'age': 'AgeGrpStart', 'value': 'mx'},
            'sex_keys': {'f': 'Female', 'm': 'Male'},
        }))
    bdm2 = ss.background_deaths(pars={'death_rate': death_rate})
    sim2 = ss.Sim(people=ppl, demographics=bdm2, label='Using age-specific data from a pandas series')
    sim2.run()
    return sim2


def test_file_death_rate():
    """ Realistic death rates in dataframe format, read from a csv file """
    realistic_death = pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')
    ppl = ss.People(1000)
    bdm3 = ss.background_deaths(pars={'death_rate': realistic_death})
    sim3 = ss.Sim(people=ppl, demographics=bdm3, label='Realistic death rates read from a CSV file')
    sim3.run()
    return sim3

def test_birth_data():
    """ Show that birth data can be added in multiple formats """

    # Parameters
    realistic_birth = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')

    ppl = ss.People(1000)
    births = ss.births(pars={'birth_rates':realistic_birth})
    sim1 = ss.Sim(people=ppl, demographics=births)
    sim1.run()

    ppl = ss.People(1000)
    births = ss.births(pars={'birth_rates': 36, 'units': 1/1000})
    sim2 = ss.Sim(people=ppl, demographics=births)
    sim2.run()

    fig, ax = plt.subplots(2, 1)
    for sim in [sim1, sim2,]:
        ax[0].plot(sim.tivec, sim.results.births.new, label=sim.label)
        ax[1].plot(sim.tivec, sim.results.n_alive)

    ax[0].set_title('New births')
    ax[1].set_title('Population size')
    ax[1].set_xlabel('Time step')
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Count')
    ax[0].legend()
    fig.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    # sim1 = test_fixed_death_rate()
    # sim2 = test_series_death_rate()
    # sim3 = test_file_death_rate()

    test_birth_data()


    if do_plot:
        fig, ax = plt.subplots(2, 1)
        for sim in [sim1, sim2, sim3]:
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

