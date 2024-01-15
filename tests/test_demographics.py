"""
Test demographics (background deaths) by showing that death data can be added in multiple formats.
"""

# %% Imports and settings
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
import sciris as sc

do_plot=True

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

def test_file_birth_data():
    """ Test births using CSV data """
    ppl = ss.People(1000)
    realistic_birth = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')
    births = ss.births(pars={'birth_rate':realistic_birth})
    sim1 = ss.Sim(people=ppl, demographics=births, label='UN birth rates read from a CSV file')
    sim1.run()
    return sim1

def test_crude_birth_data():
    """ Test births using a crude rate """
    ppl = ss.People(1000)
    births = ss.births(pars={'birth_rate': 36, 'units': 1/1000})
    sim2 = ss.Sim(people=ppl, demographics=births, label='Overall crude birth rate')
    sim2.run()
    return sim2

def test_fertility_data():
    """ Testing fertility data can be added in multiple formats """
    fertility_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')
    pregnancy = ss.Pregnancy(pars={'fertility_rate':fertility_rates})
    ppl = ss.People(1000)
    sim = ss.Sim(people=ppl, demographics=pregnancy, label='UN fertility rates read from a CSV file')
    sim.run()

    assert np.array_equal(sim.results.pregnancy.pregnancies, sim.results.pregnancy.births)

    return sim


def test_nigeria(which='births', plot_init=False):
    """
    Make a Nigeria sim with demographic modules
    Switch between which='births' or 'pregnancy' to determine which demographic module to use
    """

    # Make demographic modules
    demographics = sc.autolist()

    if which == 'births':
        birth_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')
        births = ss.births(pars={'birth_rate':birth_rates})
        demographics += births
    elif which == 'pregnancy':
        fertility_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')
        pregnancy = ss.Pregnancy(pars={'fertility_rate': fertility_rates, 'rel_fertility': 1})  # 4/3
        demographics += pregnancy


    death_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')
    death = ss.background_deaths(pars={'death_rate': death_rates})
    demographics += death

    # Make people
    n_agents = 10_000
    nga_pop_1995 = 106819805
    age_data = pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv')
    ppl = ss.People(n_agents, age_data=age_data)
    dt = 1

    sim = ss.Sim(
        dt=dt,
        total_pop=nga_pop_1995,
        start=1995,
        n_years=35,
        people=ppl,
        demographics=demographics,
    )

    if plot_init:
        sim.initialize()
        # Plot histograms of the age distributions - simulated vs data
        bins = np.arange(0, 101, 1)
        init_scale = nga_pop_1995 / n_agents
        counts, bins = np.histogram(sim.people.age, bins)
        plt.bar(bins[:-1], counts * init_scale, alpha=0.5, label='Simulated')
        plt.bar(bins, age_data.value.values * 1000, alpha=0.5, color='r', label='Data')
        plt.legend(loc='upper right')
        plt.show()

    sim.run()

    nigeria_popsize = pd.read_csv(ss.root / 'tests/test_data/nigeria_popsize.csv')
    data = nigeria_popsize[(nigeria_popsize.year >= 1995) & (nigeria_popsize.year <= 2030)]

    nigeria_cbr = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')
    cbr_data = nigeria_cbr[(nigeria_cbr.Year >= 1995) & (nigeria_cbr.Year <= 2030)]

    nigeria_cmr = pd.read_csv(ss.root / 'tests/test_data/nigeria_cmr.csv')
    cmr_data = nigeria_cmr[(nigeria_cmr.Year >= 1995) & (nigeria_cmr.Year <= 2030)]

    # Check
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].scatter(data.year, data.n_alive, alpha=0.5)
    ax[0].plot(sim.yearvec, sim.results.n_alive, color='k')
    ax[0].set_title('Population')

    ax[1].plot(sim.yearvec, sim.results.background_deaths.cmr, label='Simulated CMR')
    ax[1].scatter(cmr_data.Year, cmr_data.CMR, label='Data CMR')
    ax[1].set_title('CMR')
    ax[1].legend()

    if which == 'births':
        ax[2].plot(sim.yearvec, sim.results.births.cbr, label='Simulated CBR')
    elif which == 'pregnancy':
        ax[2].plot(sim.yearvec, sim.results.pregnancy.cbr, label='Simulated CBR')
    ax[2].scatter(cbr_data.Year, cbr_data.CBR, label='Data CBR')
    ax[2].set_title('CBR')
    ax[2].legend()

    if which == 'pregnancy':
        ax[3].plot(sim.yearvec, sim.results.pregnancy.pregnancies / dt, label='Pregnancies')
        ax[3].plot(sim.yearvec, sim.results.pregnancy.births / dt, label='Births')
        ax[3].set_title('Pregnancies and births')
        ax[3].legend()

    fig.tight_layout

    plt.show()

    return sim


if __name__ == '__main__':

    # Test Nigeria demographic consistency
    sim = test_nigeria(which='pregnancy', plot_init=True)

    # # Deaths
    # sim_death1 = test_fixed_death_rate()
    # sim_death2 = test_series_death_rate()
    # sim_death3 = test_file_death_rate()
    #
    # # Test births
    # sim_birth1 = test_file_birth_data()
    # sim_birth2 = test_crude_birth_data()
    #
    # # Test fertility
    # sim_fert = test_fertility_data()
    #
    #
    # if do_plot:
    #     # Plot deaths
    #     fig, ax = plt.subplots(2, 1)
    #     for sim in [sim_death1, sim_death2, sim_death3]:
    #         ax[0].plot(sim.tivec, sim.results.background_deaths.new, label=sim.label)
    #         ax[1].plot(sim.tivec, sim.results.n_alive)
    #
    #     ax[0].set_title('New background deaths')
    #     ax[1].set_title('Population size')
    #     ax[1].set_xlabel('Time step')
    #     ax[0].set_ylabel('Count')
    #     ax[1].set_ylabel('Count')
    #     ax[0].legend()
    #     fig.tight_layout()
    #
    #     # Plot births
    #     fig, ax = plt.subplots(2, 1)
    #     for sim in [sim_birth1, sim_birth2,]:
    #         ax[0].plot(sim.tivec, sim.results.births.new, label=sim.label)
    #         ax[1].plot(sim.tivec, sim.results.n_alive)
    #
    #     ax[0].set_title('New births')
    #     ax[1].set_title('Population size')
    #     ax[1].set_xlabel('Time step')
    #     ax[0].set_ylabel('Count')
    #     ax[1].set_ylabel('Count')
    #     ax[0].legend()
    #     fig.tight_layout()
    #
    #     # Plot fert
    #     sim = sim_fert
    #     fig, ax = plt.subplots(2, 1)
    #     ax[0].plot(sim.yearvec, sim.results.pregnancy.pregnancies, label='Pregnancies')
    #     ax[0].plot(sim.yearvec, sim.results.pregnancy.births, ':', label='Births')
    #     ax[1].plot(sim.yearvec, sim.results.n_alive, label='Population')
    #     ax[0].set_title('Pregnancies and births')
    #     ax[1].set_title('Population size')
    #     ax[1].set_xlabel('Year')
    #     ax[0].set_ylabel('Count')
    #     ax[1].set_ylabel('Count')
    #     ax[0].legend()
    #     fig.tight_layout()
    #
    #     plt.show()
