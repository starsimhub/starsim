import starsim as ss
import sciris as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

n_agents = 1_000
sc.options(interactive=False) # Assume not running interactively
datadir = ss.root / 'tests/test_data'

def run(which='births', dt=1, start=1995, dur=15, do_plot=False):
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
    demographics = [ss.Births(country_code='MEX')]
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


def run02():
    """ Test all different ways of creating a sim """


    # Check different ways of specifying a sim
    s1 = ss.Sim(n_agents=n_agents, diseases='sir', networks='random').run() # Supply strings directly
    s2 = ss.Sim(pars=dict(n_agents=n_agents, diseases='sir', networks='random')).run() # Supply as parameters
    s3 = ss.Sim(n_agents=n_agents, diseases=ss.SIR(), networks=ss.RandomNet()).run() # Supply as objects
    ss.check_sims_match(s1, s2, s3), 'Sims should match'

    # Check different ways of setting a distribution
    kw = dict(n_agents=n_agents, networks='random')
    d1 = ss.lognorm_ex(10) # Create a distribution with an argument
    d2 = ss.lognorm_ex(mean=10, std=2) # Create a distribution with kwargs
    d3 = ss.normal(loc=10) # Create a different type of distribution

    # Check specifying dist with a scalar
    s4 = ss.Sim(diseases=dict(type='sir', dur_inf=10), **kw).run() # Supply values as a scalar
    s5 = ss.Sim(diseases=dict(type='sir', dur_inf=d1), **kw).run() # Supply as a distribution
    ss.check_sims_match(s4, s5), 'Sims should match'

    # Check specifying dist with a list and dict
    s6 = ss.Sim(diseases=dict(type='sir', dur_inf=[10,2]), **kw).run() # Supply values as a list
    s7 = ss.Sim(diseases=dict(type='sir', dur_inf=dict(mean=10, std=2)), **kw).run() # Supply values as a dict
    s8 = ss.Sim(diseases=dict(type='sir', dur_inf=d2), **kw).run() # Supply as a distribution
    ss.check_sims_match(s6, s7, s8), 'Sims should match'

    # Check changing dist type
    s9  = ss.Sim(diseases=dict(type='sir', dur_inf=dict(type='normal', loc=10)), **kw).run() # Supply values as a dict
    s10 = ss.Sim(diseases=dict(type='sir', dur_inf=d3), **kw).run() # Supply values as a distribution
    ss.check_sims_match(s9, s10), 'Sims should match'



    return s1

if __name__ == '__main__':
    run()