"""
Test demographics
"""

# %% Imports and settings
import numpy as np
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt


def test_nigeria():
    """ Make a sim with Nigeria demographic data """

    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    pregnancy = ss.Pregnancy(fertility_rates)
    death_rates = dict(
        death_rates=pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv'),
    )
    death = ss.background_deaths(death_rates)

    # Make people
    ppl = ss.People(10000, age_data=pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv'))

    sim = ss.Sim(
        dt=1/12,
        total_pop=93963392,
        start=1990,
        n_years=40,
        people=ppl,
        demographics=[
            pregnancy,
            death
        ],
    )

    sim.run()

    nigeria_popsize = pd.read_csv(ss.root / 'tests/test_data/nigeria_popsize.csv')
    data = nigeria_popsize[(nigeria_popsize.year >= 1990) & (nigeria_popsize.year <= 2030)]

    # Check
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].plot(sim.yearvec, sim.results.n_alive)
    ax[0].scatter(data.year, data.n_alive)
    ax[0].set_title('Population')

    ax[1].plot(sim.yearvec, sim.results.new_deaths)
    ax[1].set_title('Deaths')

    ax[2].plot(sim.yearvec, sim.results.pregnancy.pregnancies, label='Pregnancies')
    ax[2].plot(sim.yearvec, sim.results.pregnancy.births, label='Births')
    ax[2].set_title('Pregnancies and births')
    ax[2].legend()

    fig.tight_layout

    plt.show()

    return sim


if __name__ == '__main__':

    sim = test_nigeria()

