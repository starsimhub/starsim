"""
Run syphilis
"""

# %% Imports and settings
import numpy as np
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt


def make_syph_sim():
    """ Make a sim with syphilis - used by several subsequent tests """
    syph = ss.Syphilis()
    syph.pars['beta'] = {'mf': [0.95, 0.75], 'maternal': [0.99, 0]}
    syph.pars['init_prev'] = 0.05

    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    pregnancy = ss.Pregnancy(fertility_rates)
    death_rates = dict(
        death_rates=pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv'),
        rel_death=0.4,
    )
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ppl = ss.People(10000)
    mf = ss.mf(
        pars=dict(dur=ss.lognormal(1, 5))
    )
    maternal = ss.maternal()
    ppl.networks = ss.ndict(mf, maternal)

    sim_kwargs = dict(
        dt=1/12,
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

            # Infection states: people must be exactly one of susceptible/infectious/inactive
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

    # Check
    plt.figure()
    plt.plot(sim.yearvec, sim.results.syphilis.new_infections)
    plt.title('Syphilis infections')
    plt.show()

    plt.figure()
    n_alive = sim.results.n_alive
    plt.stackplot(
        sim.yearvec,
        sim.results.syphilis.n_susceptible/n_alive,
        sim.results.syphilis.n_congenital/n_alive,
        sim.results.syphilis.n_exposed/n_alive,
        sim.results.syphilis.n_primary/n_alive,
        sim.results.syphilis.n_secondary/n_alive,
        (sim.results.syphilis.n_latent_temp+sim.results.syphilis.n_latent_long)/n_alive,
        sim.results.syphilis.n_tertiary/n_alive,
    )
    plt.legend(['Susceptible', 'Congenital', 'Exposed', 'Primary', 'Secondary', 'Latent', 'Tertiary'], loc='lower right')
    plt.show()

    return sim


def test_syph_intvs():

    sim_kwargs = make_syph_sim()

    # Interventions
    # screen_eligible = lambda sim: sim.demographics.pregnancy.pregnant
    screen_eligible = lambda sim: sim.people.networks.mf.active(sim.people)
    syph_screening = ss.syph_screening(
        product='rpr',
        prob=0.99,
        eligibility=screen_eligible,
        start_year=2020,
        label='syph_screening',
    )

    treat_eligible = lambda sim: sim.get_intervention('syph_screening').outcomes['positive']
    bpg = ss.syph_treatment(
        prob=0.9,
        product='bpg',
        eligibility=treat_eligible,
        label='bpg'
    )

    sim = ss.Sim(interventions=[syph_screening, bpg], **sim_kwargs)
    sim.run()

    return sim



if __name__ == '__main__':

    # sim0 = test_syph()
    sim1 = test_syph_intvs()
