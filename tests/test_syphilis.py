"""
Run syphilis
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps


def make_syph_sim():
    """ Make a sim with syphilis - used by several subsequent tests """
    syph = ss.Syphilis()
    syph.pars['beta'] = {'mf': [0.95, 0.75], 'maternal': [0.99, 0]}
    syph.pars['seed_infections'] = sps.bernoulli(p=0.1)

    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    pregnancy = ss.Pregnancy(fertility_rates)
    death_rates = dict(
        death_rates=pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv'),
    )
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(500, age_data=pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv')) # CK: temporary small pop size
    mf = ss.mf(
        pars=dict(duration_dist=ss.lognorm(mean=0.1, stdev=0.5))
    )
    maternal = ss.maternal()
    ppl.networks = ss.ndict(mf, maternal)

    sim_kwargs = dict(
        dt=1/2, # CK: temporary long dt
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
    burnin = 0
    pi = int(burnin/sim.dt)
    plt.figure()
    plt.plot(sim.yearvec[pi:], sim.results.syphilis.new_infections[pi:])
    plt.title('Syphilis infections')
    plt.show()

    plt.figure()
    n_alive = sim.results.n_alive[pi:]
    plt.stackplot(
        sim.yearvec[pi:],
        sim.results.syphilis.n_susceptible[pi:]/n_alive,
        sim.results.syphilis.n_congenital[pi:]/n_alive,
        sim.results.syphilis.n_exposed[pi:]/n_alive,
        sim.results.syphilis.n_primary[pi:]/n_alive,
        sim.results.syphilis.n_secondary[pi:]/n_alive,
        (sim.results.syphilis.n_latent_temp[pi:]+sim.results.syphilis.n_latent_long[pi:])/n_alive,
        sim.results.syphilis.n_tertiary[pi:]/n_alive,
    )
    plt.legend(['Susceptible', 'Congenital', 'Exposed', 'Primary', 'Secondary', 'Latent', 'Tertiary'], loc='lower right')
    plt.show()

    return sim


def test_syph_intvs():

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

    sim_kwargs0 = make_syph_sim()
    sim_base = ss.Sim(**sim_kwargs0)
    sim_base.run()

    sim_kwargs1 = make_syph_sim()
    sim_intv = ss.Sim(interventions=[syph_screening, bpg], **sim_kwargs1)
    sim_intv.run()

    # Check plots
    burnin = 10
    pi = int(burnin/sim_base.dt)
    plt.figure()
    plt.plot(sim_base.yearvec[pi:], sim_base.results.syphilis.prevalence[pi:], label='Baseline')
    plt.plot(sim_base.yearvec[pi:], sim_intv.results.syphilis.prevalence[pi:], label='S&T')
    plt.ylim([0, 0.25])
    plt.axvline(x=2020, color='k', ls='--')
    plt.title('Syphilis prevalence')
    plt.legend()
    plt.show()

    return sim_base, sim_intv



if __name__ == '__main__':

    sim0 = test_syph()
    # sim_base, sim_intv = test_syph_intvs()
