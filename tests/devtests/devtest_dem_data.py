"""
Test different formats for demographic data
"""

# %% Imports and settings
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

do_plot = True


def test_fixed_death_rate():
    """ Simple fixed death rate for all agents """
    ppl = ss.People(1000)
    bdm1 = ss.Deaths(pars={'death_rate': 0.015})
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
        'sex_keys': {'Female':'f', 'Male':'m'},
    }))
    bdm2 = ss.Deaths(pars={'death_rate': death_rate})
    sim2 = ss.Sim(people=ppl, demographics=bdm2, label='Using age-specific data from a pandas series')
    sim2.run()
    return sim2


def test_file_death_rate():
    """ Realistic death rates in dataframe format, read from a csv file """
    realistic_death = pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')
    ppl = ss.People(1000)
    bdm3 = ss.Deaths(pars={'death_rate': realistic_death})
    sim3 = ss.Sim(people=ppl, demographics=bdm3, label='Realistic death rates read from a CSV file')
    sim3.run()
    return sim3


def test_file_birth_data():
    """ Test births using CSV data """
    ppl = ss.People(1000)
    realistic_birth = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')
    births = ss.Births(pars={'birth_rate': realistic_birth})
    sim1 = ss.Sim(people=ppl, demographics=births, label='UN birth rates read from a CSV file')
    sim1.run()
    return sim1


def test_crude_birth_data():
    """ Test births using a crude rate """
    ppl = ss.People(1000)
    births = ss.Births(pars={'birth_rate': 36, 'units': 1 / 1000})
    sim2 = ss.Sim(people=ppl, demographics=births, label='Overall crude birth rate')
    sim2.run()
    return sim2


def test_fertility_data():
    """ Testing fertility data can be added in multiple formats """
    fertility_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')
    pregnancy = ss.Pregnancy(pars={'fertility_rate': fertility_rates})
    ppl = ss.People(1000)
    sim = ss.Sim(people=ppl, demographics=pregnancy, label='UN fertility rates read from a CSV file')
    sim.run()

    assert np.array_equal(sim.results.pregnancy.pregnancies, sim.results.pregnancy.births)

    return sim


if __name__ == '__main__':

    # Deaths
    sim_death1 = test_fixed_death_rate()
    sim_death2 = test_series_death_rate()
    sim_death3 = test_file_death_rate()

    # Test births
    sim_birth1 = test_file_birth_data()
    sim_birth2 = test_crude_birth_data()

    # Test fertility
    sim_fert = test_fertility_data()


    if do_plot:
        # Plot deaths
        fig, ax = plt.subplots(2, 1)
        for sim in [sim_death1, sim_death2, sim_death3]:
            ax[0].plot(sim.tivec, sim.results.deaths.new, label=sim.label)
            ax[1].plot(sim.tivec, sim.results.n_alive)

        ax[0].set_title('New background deaths')
        ax[1].set_title('Population size')
        ax[1].set_xlabel('Time step')
        ax[0].set_ylabel('Count')
        ax[1].set_ylabel('Count')
        ax[0].legend()
        fig.tight_layout()
        plt.show()

        # Plot births
        fig, ax = plt.subplots(2, 1)
        for sim in [sim_birth1, sim_birth2,]:
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

        # Plot fert
        sim = sim_fert
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(sim.yearvec, sim.results.pregnancy.pregnancies, label='Pregnancies')
        ax[0].plot(sim.yearvec, sim.results.pregnancy.births, ':', label='Births')
        ax[1].plot(sim.yearvec, sim.results.n_alive, label='Population')
        ax[0].set_title('Pregnancies and births')
        ax[1].set_title('Population size')
        ax[1].set_xlabel('Year')
        ax[0].set_ylabel('Count')
        ax[1].set_ylabel('Count')
        ax[0].legend()
        fig.tight_layout()

        plt.show()
