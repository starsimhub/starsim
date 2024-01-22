"""
Run syphilis
"""

# %% Imports and settings
import numpy as np
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps


def make_syph_sim():
    """ Make a sim with syphilis - used by several subsequent tests """
    syph = ss.Syphilis()
    syph.pars['beta'] = {'mf': [0.95, 0.75], 'maternal': [0.99, 0]}
    syph.pars['seed_infections'] = sps.bernoulli(p=0.1)

    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')}
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(5000, age_data=pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv'))
    mf = ss.mf(
        pars=dict(duration_dist=ss.lognorm(mean=0.1, stdev=0.5))
    )
    maternal = ss.maternal()
    ppl.networks = ss.ndict(mf, maternal)

    sim_kwargs = dict(
        dt=1/12,
        total_pop=93963392,
        start=1990,
        n_years=40,
        people=ppl,
        diseases=syph,
        demographics=[pregnancy, death],
    )

    return sim_kwargs


def test_syph():

    sim_kwargs = make_syph_sim()

    class check_states(ss.Analyzer):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.okay = True
            return

        def update_results(self, sim):
            return self.apply(sim)

        def apply(self, sim):
            """
            Checks states that should be mutually exlusive and collectively exhaustive
            """
            sppl = sim.people.syphilis

            # Infection states: people must be exactly one of these
            s1 = (sppl.susceptible | sppl.exposed | sppl.primary | sppl.secondary | sppl.latent_temp | sppl.latent_long | sppl.tertiary | sppl.congenital).all()
            if not s1:
                raise ValueError('States should be collectively exhaustive but are not.')
            s2 = ~(sppl.susceptible & sppl.exposed & sppl.primary & sppl.secondary & sppl.latent_temp & sppl.latent_long & sppl.tertiary & sppl.congenital).any()
            if not s2:
                raise ValueError('States should be mutually exclusive but are not.')

            checkall = np.array([s1, s2])
            if not checkall.all():
                self.okay = False

            return

    sim = ss.Sim(analyzers=[check_states], **sim_kwargs)
    sim.run()

    # Check plots
    burnin = 10
    pi = int(burnin/sim.dt)

    plt.figure()
    plt.stackplot(
        sim.yearvec[pi:],
        sim.results.syphilis.n_susceptible[pi:],
        sim.results.syphilis.n_congenital[pi:],
        sim.results.syphilis.n_exposed[pi:],
        sim.results.syphilis.n_primary[pi:],
        sim.results.syphilis.n_secondary[pi:],
        (sim.results.syphilis.n_latent_temp[pi:]+sim.results.syphilis.n_latent_long[pi:]),
        sim.results.syphilis.n_tertiary[pi:],
    )
    plt.legend(['Susceptible', 'Congenital', 'Exposed', 'Primary', 'Secondary', 'Latent', 'Tertiary'], loc='lower right')
    plt.show()

    return sim


if __name__ == '__main__':

    sim = test_syph()
