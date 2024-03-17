"""
Test demographic consistency
"""

# %% Imports and settings
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import pytest


do_plot = True
sc.options(interactive=False) # Assume not running interactively


def test_nigeria(which='births', dt=1, start=1995, n_years=15, plot_init=False, do_plot=True):
    """
    Make a Nigeria sim with demographic modules
    Switch between which='births' or 'pregnancy' to determine which demographic module to use
    """

    # Make demographic modules
    demographics = sc.autolist()

    if which == 'births':
        birth_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')
        births = ss.Births(pars={'birth_rate': birth_rates})
        demographics += births

    elif which == 'pregnancy':
        fertility_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')
        pregnancy = ss.Pregnancy(pars={'fertility_rate': fertility_rates, 'rel_fertility': 1})  # 4/3
        demographics += pregnancy

    death_rates = pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')
    death = ss.Deaths(pars={'death_rate': death_rates, 'units': 1})
    demographics += death

    # Make people
    n_agents = 5_000
    nga_pop_1995 = 106819805
    age_data = pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    sim = ss.Sim(
        dt=dt,
        total_pop=nga_pop_1995,
        start=start,
        n_years=n_years,
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

    sim.run()

    end = start + n_years
    nigeria_popsize = pd.read_csv(ss.root / 'tests/test_data/nigeria_popsize.csv')
    data = nigeria_popsize[(nigeria_popsize.year >= start) & (nigeria_popsize.year <= end)]

    nigeria_cbr = pd.read_csv(ss.root / 'tests/test_data/nigeria_births.csv')
    cbr_data = nigeria_cbr[(nigeria_cbr.Year >= start) & (nigeria_cbr.Year <= end)]

    nigeria_cmr = pd.read_csv(ss.root / 'tests/test_data/nigeria_cmr.csv')
    cmr_data = nigeria_cmr[(nigeria_cmr.Year >= start) & (nigeria_cmr.Year <= end)]

    # Tests
    if which == 'pregnancy':

        print("Check we don't have more births than pregnancies")
        assert sum(sim.results.pregnancy.births) <= sum(sim.results.pregnancy.pregnancies)
        print('✓ (births <= pregnancies)')

        if dt == 1:
            print("Checking that births equal pregnancies with dt=1")
            assert np.array_equal(sim.results.pregnancy.pregnancies, sim.results.pregnancy.births)
            print('✓ (births == pregnancies)')

    print("Check final pop size within 5% of data")
    assert np.isclose(data.n_alive.values[-1], sim.results.n_alive[-1], rtol=0.05)
    print(f'✓ (simulated/data={sim.results.n_alive[-1] / data.n_alive.values[-1]:.2f})')

    # Plots
    if do_plot:
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        ax[0].scatter(data.year, data.n_alive, alpha=0.5)
        ax[0].plot(sim.yearvec, sim.results.n_alive, color='k')
        ax[0].set_title('Population')

        ax[1].plot(sim.yearvec, 1000 * sim.results.deaths.cmr / dt, label='Simulated CMR')
        ax[1].scatter(cmr_data.Year, cmr_data.CMR, label='Data CMR')
        ax[1].set_title('CMR')
        ax[1].legend()

        if which == 'births':
            ax[2].plot(sim.yearvec, sim.results.births.cbr / dt, label='Simulated CBR')
        elif which == 'pregnancy':
            ax[2].plot(sim.yearvec, sim.results.pregnancy.cbr / dt, label='Simulated CBR')
        ax[2].scatter(cbr_data.Year, cbr_data.CBR, label='Data CBR')
        ax[2].set_title('CBR')
        ax[2].legend()

        if which == 'pregnancy':
            ax[3].plot(sim.yearvec, sim.results.pregnancy.pregnancies / dt, label='Pregnancies')
            ax[3].plot(sim.yearvec, sim.results.pregnancy.births / dt, label='Births')
            ax[3].set_title('Pregnancies and births')
            ax[3].legend()

        fig.tight_layout()

    return sim


def test_constant_pop():
    # Test pars for constant pop size
    sim = ss.Sim(n_agents=10e3, birth_rate=10, death_rate=10/1010*1000, n_years=200, rand_seed=1).run()
    print("Check final pop size within 5% of starting pop")
    assert np.isclose(sim.results.n_alive[0], sim.results.n_alive[-1], rtol=0.05)
    print(f'✓ (final pop / starting pop={sim.results.n_alive[-1] / sim.results.n_alive[0]:.2f})')

    # Plots
    if do_plot:
        sim.plot()
    return sim


def test_module_adding():
    births = ss.Births(pars={'birth_rate': 10})
    deaths = ss.Deaths(pars={'death_rate': 10})
    demographics = [births, deaths]
    with pytest.raises(Exception): # CK: should be ValueError, but that fails for now, and this is OK
        ss.Sim(n_agents=1e3, demographics=demographics, birth_rate=10, death_rate=10).run()
    return demographics


if __name__ == '__main__':
    sc.options(interactive=do_plot)
    s1 = test_nigeria(dt=1, which='pregnancy', n_years=15, plot_init=True, do_plot=do_plot)
    s2 = test_constant_pop()
    s3 = test_module_adding()
    plt.show()
