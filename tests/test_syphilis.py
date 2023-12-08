"""
Run syphilis
"""

# %% Imports and settings
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
 
def test_syph():

    # Make syphilis module
    syph = ss.Syphilis()
    syph.pars['beta'] = {'mf': [0.5, 0.35], 'maternal': [0.99, 0]}
    syph.pars['init_prev'] = 0.005

    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    death_rates = {'death_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')}
    pregnancy = ss.Pregnancy(fertility_rates)
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ppl = ss.People(100000)
    mf = ss.mf(
        pars=dict(dur=ss.lognormal(2, 5))
    )
    maternal = ss.maternal()
    ppl.networks = ss.ndict(mf, maternal)
    sim = ss.Sim(dt=1/12, start=1950, n_years=70, people=ppl, diseases=syph, demographics=[pregnancy, death])
    sim.run()

    plt.figure()
    plt.plot(sim.yearvec, syph.results.new_infections)
    plt.title('Syphilis infections')
    plt.show()

    return sim


if __name__ == '__main__':
    sim = test_syph()

