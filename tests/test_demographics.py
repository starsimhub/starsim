"""
Test demographic consistency
"""
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import pytest

sc.options(interactive=False) # Assume not running interactively
datadir = ss.root / 'tests/test_data'


def test_nigeria(which='births', dt=1, start=1995, dur=15, do_plot=False):
    """
    Make a Nigeria sim with demographic modules
    Switch between which='births' or 'pregnancy' to determine which demographic module to use
    """
    sc.heading('Testing Nigeria demographics')

    # Make demographic modules
    demographics = sc.autolist()

    if which == 'births':
        birth_rates = pd.read_csv(datadir/'nigeria_births.csv')
        births = ss.Births(pars={'birth_rate': birth_rates})
        demographics += births

    elif which == 'pregnancy':
        fertility_rates = pd.read_csv(datadir/'nigeria_asfr.csv')
        pregnancy = ss.Pregnancy(pars={'fertility_rate': fertility_rates, 'rel_fertility': 1})  # 4/3
        demographics += pregnancy

    death_rates = pd.read_csv(datadir/'nigeria_deaths.csv')
    death = ss.Deaths(pars={'death_rate': death_rates, 'rate_units': 1})
    demographics += death

    # Make people
    n_agents = 5_000
    nga_pop_1995 = 106819805
    age_data = pd.read_csv(datadir/'nigeria_age.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    sim = ss.Sim(
        dt=dt,
        total_pop=nga_pop_1995,
        start=start,
        dur=dur,
        people=ppl,
        demographics=demographics,
    )

    if do_plot:
        sim.init()
        # Plot histograms of the age distributions - simulated vs data
        bins = np.arange(0, 101, 1)
        init_scale = nga_pop_1995 / n_agents
        counts, bins = np.histogram(sim.people.age, bins)
        plt.bar(bins[:-1], counts * init_scale, alpha=0.5, label='Simulated')
        plt.bar(bins, age_data.value.values * 1000, alpha=0.5, color='r', label='Data')
        plt.legend(loc='upper right')

    sim.run()

    stop = start + dur
    nigeria_popsize = pd.read_csv(datadir/'nigeria_popsize.csv')
    data = nigeria_popsize[(nigeria_popsize.year >= start) & (nigeria_popsize.year <= stop)]

    nigeria_cbr = pd.read_csv(datadir/'nigeria_births.csv')
    cbr_data = nigeria_cbr[(nigeria_cbr.Year >= start) & (nigeria_cbr.Year <= stop)]

    nigeria_cmr = pd.read_csv(datadir/'nigeria_cmr.csv')
    cmr_data = nigeria_cmr[(nigeria_cmr.Year >= start) & (nigeria_cmr.Year <= stop)]

    # Tests
    if which == 'pregnancy':

        print("Check we don't have more births than pregnancies")
        assert sum(sim.results.pregnancy.births) <= sum(sim.results.pregnancy.pregnancies)
        print('✓ (births <= pregnancies)')

        if dt == 1:
            print("Checking that births equal pregnancies with dt=1")
            assert np.array_equal(sim.results.pregnancy.pregnancies, sim.results.pregnancy.births)
            print('✓ (births == pregnancies)')

    rtol = 0.05
    print(f'Check final pop size within {rtol*100:n}% of data')
    assert np.isclose(data.n_alive.values[-1], sim.results.n_alive[-1], rtol=rtol), f'Final population size not within {rtol*100:n}% of data'
    print(f'✓ (simulated/data={sim.results.n_alive[-1] / data.n_alive.values[-1]:.2f})')

    # Plots
    if do_plot:
        tvec = sim.timevec
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        ax[0].scatter(data.year, data.n_alive, alpha=0.5)
        ax[0].plot(tvec, sim.results.n_alive, color='k')
        ax[0].set_title('Population')

        ax[1].plot(tvec, 1000 * sim.results.deaths.cmr, label='Simulated CMR')
        ax[1].scatter(cmr_data.Year, cmr_data.CMR, label='Data CMR')
        ax[1].set_title('CMR')
        ax[1].legend()

        if which == 'births':
            ax[2].plot(tvec, sim.results.births.cbr, label='Simulated CBR')
        elif which == 'pregnancy':
            ax[2].plot(tvec, sim.results.pregnancy.cbr, label='Simulated CBR')
        ax[2].scatter(cbr_data.Year, cbr_data.CBR, label='Data CBR')
        ax[2].set_title('CBR')
        ax[2].legend()

        if which == 'pregnancy':
            ax[3].plot(tvec, sim.results.pregnancy.pregnancies, label='Pregnancies')
            ax[3].plot(tvec, sim.results.pregnancy.births, label='Births')
            ax[3].set_title('Pregnancies and births')
            ax[3].legend()

        fig.tight_layout()

    return sim


def test_constant_pop(do_plot=False):
    """ Test pars for constant pop size """
    sc.heading('Testing constant population size')
    sim = ss.Sim(n_agents=10e3, birth_rate=10, death_rate=10/1010*1000, dur=200, rand_seed=1).run()
    print("Check final pop size within 5% of starting pop")
    assert np.isclose(sim.results.n_alive[0], sim.results.n_alive[-1], rtol=0.05)
    print(f'✓ (final pop / starting pop={sim.results.n_alive[-1] / sim.results.n_alive[0]:.2f})')

    # Plots
    if do_plot:
        sim.plot()
    return sim


def test_module_adding():
    """ Test that modules can't be added twice """
    sc.heading('Testing module duplication')
    births = ss.Births(pars={'birth_rate': 10})
    deaths = ss.Deaths(pars={'death_rate': 10})
    demographics = [births, deaths]
    with pytest.raises(ValueError):
        ss.Sim(n_agents=1e3, demographics=demographics, birth_rate=10, death_rate=10).run()
    return demographics


def test_aging():
    """ Test that aging is configured properly """
    sc.heading('Testing aging')
    n_agents = int(1e3)

    # With no demograhpics, people shouldn't age
    s1 = ss.Sim(n_agents=n_agents).init()
    orig_ages = s1.people.age.raw.copy()
    orig_age = orig_ages.mean()
    s1.run()
    end_age = s1.people.age.mean()
    assert orig_age == end_age, f'By default there should be no aging, but {orig_age} != {end_age}'

    # We should be able to manually turn on aging
    s2 = ss.Sim(n_agents=n_agents, use_aging=True).run()
    age2 = s2.people.age.mean()
    assert orig_age < age2, f'With aging, start age {orig_age} should be less than end age {age2}'

    # Aging should turn on automatically if we add demographics
    s3 = ss.Sim(n_agents=n_agents, demographics=True).run()
    agent = s3.people.auids[0] # Find first alive agent
    orig_agent_age = orig_ages[agent]
    age3 = s3.people.age[ss.uids(agent)]
    assert orig_agent_age < age3, f'With demographics, original agent age {orig_agent_age} should be less than end age {age3}'

    # ...but can be turned off manually
    s4 = ss.Sim(n_agents=n_agents, demographics=True, use_aging=False).run()
    agent = s4.people.auids[0] # Find first alive agent
    orig_agent_age = orig_ages[agent]
    age4 = s4.people.age[ss.uids(agent)]
    assert orig_agent_age == age4, f'With aging turned off, original agent age {orig_agent_age} should match end age {age4}'

    return s3 # Most interesting sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    s1 = test_nigeria(do_plot=do_plot)
    s2 = test_nigeria(do_plot=do_plot, dt=1/12, which='pregnancy')
    s3 = test_constant_pop(do_plot=do_plot)
    s4 = test_module_adding()
    s5 = test_aging()
    plt.show()
