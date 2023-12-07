"""
Run RSV
"""

# %% Imports and settings
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
 
def test_rsv():

    # Make rsv module
    rsv = ss.RSV()
    rsv.pars['beta'] = {'randomnetwork': 0.5, 'maternal': 0}
    rsv.pars['init_prev'] = 0.05

    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    death_rates = {'death_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')}
    pregnancy = ss.Pregnancy(fertility_rates)
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ppl = ss.People(10000)
    RandomNetwork = ss.RandomNetwork(n_contacts=ss.poisson(10))
    maternal = ss.maternal()
    ppl.networks = ss.ndict(RandomNetwork, maternal)
    sim = ss.Sim(dt=1/12, people=ppl, diseases=[rsv], demographics=[pregnancy, death])
    sim.run()

    plt.figure()
    plt.plot(sim.yearvec, rsv.results.n_infected)
    plt.title('RSV infections')
    plt.show()

    return sim


if __name__ == '__main__':
    sim = test_rsv()

