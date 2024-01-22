"""
Run syphilis
"""

# %% Imports and settings
import numpy as np
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import sciris as sc


class Incident_Syphilis(ss.Syphilis):
    """ Same prognoses and parameters, but overwrite make_new_cases """
    def __init__(self, pars=None):
        super().__init__(pars)

        # Additional parameters
        default_pars = dict(
            incidence_dist=sps.bernoulli(p=1e-01),
        )
        self.pars = ss.omerge(default_pars, self.pars)

    def make_new_cases(self, sim):
        sus_uids = ss.true(self.susceptible)
        new_cases = self.pars.incidence_dist.filter(sus_uids)
        if len(new_cases):
            self._set_cases(sim, new_cases)
        return new_cases


def make_syph_sim(dt=1/12):
    """ Make a sim with incident syphilis """
    syph = Incident_Syphilis()
    syph.pars['seed_infections'] = sps.bernoulli(p=0.1)

    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')}
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(5000, age_data=pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv'))

    sim_kwargs = dict(
        dt=dt,
        total_pop=93963392,
        start=1990,
        n_years=40,
        people=ppl,
        diseases=syph,
        demographics=[pregnancy, death],
    )

    return sim_kwargs


def test_syph(dt=1/12):

    sim_kwargs = make_syph_sim(dt=dt)
    sim = ss.Sim(**sim_kwargs)
    sim.run()

    # Check plots
    burnin = 10
    pi = int(burnin/sim.dt)

    fig, ax = plt.subplots(3, 1)
    ax = ax.ravel()
    ax[0].stackplot(
        sim.yearvec[pi:],
        # sim.results.syphilis.n_susceptible[pi:],
        sim.results.incident_syphilis.n_congenital[pi:],
        sim.results.incident_syphilis.n_exposed[pi:],
        sim.results.incident_syphilis.n_primary[pi:],
        sim.results.incident_syphilis.n_secondary[pi:],
        (sim.results.incident_syphilis.n_latent_temp[pi:]+sim.results.incident_syphilis.n_latent_long[pi:]),
        sim.results.incident_syphilis.n_tertiary[pi:],
    )
    ax[0].legend(['Congenital', 'Exposed', 'Primary', 'Secondary', 'Latent', 'Tertiary'], loc='lower right')

    ax[1].plot(sim.yearvec[pi:], sim.results.incident_syphilis.prevalence[pi:])
    ax[1].set_title('Syphilis prevalence')

    ax[2].plot(sim.yearvec[pi:], sim.results.n_alive[pi:])
    ax[2].set_title('Population')

    fig.tight_layout()
    plt.show()

    return sim


def test_syph_intvs(dt=1/12, do_plot=False):

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

    sim_kwargs1 = make_syph_sim(dt=dt)
    sim_intv = ss.Sim(analyzers=[check_states], interventions=[syph_screening, bpg], **sim_kwargs1)
    sim_intv.run()

    # Check plots
    if do_plot:
        # Run baseline
        sim_kwargs0 = make_syph_sim(dt=dt)
        sim_base = ss.Sim(**sim_kwargs0)
        sim_base.run()

        burnin = 10
        pi = int(burnin/sim_base.dt)
        plt.figure()
        plt.plot(sim_base.yearvec[pi:], sim_base.results.syphilis.prevalence[pi:], label='Baseline')
        plt.plot(sim_intv.yearvec[pi:], sim_intv.results.syphilis.prevalence[pi:], label='S&T')
        plt.ylim([0, 0.1])
        plt.axvline(x=2020, color='k', ls='--')
        plt.title('Syphilis prevalence')
        plt.legend()
        plt.show()

        return sim_base, sim_intv

    else:
        return sim_intv


if __name__ == '__main__':

    sim = test_syph(dt=1)
    # sim_base, sim_intv = test_syph_intvs(dt=1/2, do_plot=True)

