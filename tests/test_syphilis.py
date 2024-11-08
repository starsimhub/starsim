"""
Run syphilis
"""

# %% Imports and settings
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

quick_run = True
datadir = ss.root / 'tests/test_data'


def make_syph_sim(dt=1, n_agents=500):
    """ Make a sim with syphilis - used by several subsequent tests """
    syph = ss.Syphilis()
    syph.pars.beta = dict(mf=[0.25, 0.15], maternal=[0.99, 0])
    syph.pars.init_prev = ss.bernoulli(p=0.1)

    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(datadir/'nigeria_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(datadir/'nigeria_deaths.csv'), 'rate_units': 1}
    death = ss.Deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(n_agents, age_data=datadir/'nigeria_age.csv')

    # Marital
    mf = ss.MFNet(duration=1/24)
    maternal = ss.MaternalNet()

    sim_kwargs = dict(
        dt=dt,
        total_pop=93963392,
        start=1990,
        dur=40,
        people=ppl,
        diseases=syph,
        networks=ss.ndict(mf, maternal),
        demographics=[pregnancy, death],
    )

    return sim_kwargs


class check_states(ss.Analyzer):
    """ Analyzer to check consistency of states """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'check_states'
        self.okay = True
        return

    def step(self):
        """
        Checks states that should be mutually exlusive and collectively exhaustive
        """
        sim = self.sim
        sppl = sim.diseases.syphilis

        # Infection states: people must be exactly one of these
        s1a = (sppl.susceptible | sppl.exposed | sppl.primary | sppl.secondary | sppl.latent_temp | sppl.latent_long | sppl.tertiary | sppl.congenital).all()
        s1b = (sppl.naive | sppl.sus_not_naive | sppl.exposed | sppl.primary | sppl.secondary | sppl.latent_temp | sppl.latent_long | sppl.tertiary | sppl.congenital).all()
        if not (s1a & s1b):
            raise ValueError('States should be collectively exhaustive but are not.')
        s2a = ~(sppl.susceptible & sppl.exposed & sppl.primary & sppl.secondary & sppl.latent_temp & sppl.latent_long & sppl.tertiary & sppl.congenital).any()
        s2b = ~(sppl.naive & sppl.sus_not_naive & sppl.exposed & sppl.primary & sppl.secondary & sppl.latent_temp & sppl.latent_long & sppl.tertiary & sppl.congenital).any()
        if not (s2a & s2b):
            raise ValueError('States should be mutually exclusive but are not.')
        s3 = ~(sppl.susceptible & sppl.infected).any()
        if not s3:
            raise ValueError('States S and I should be mutually exclusive but are not.')

        checkall = np.array([s1a, s1b, s2a, s2b, s3])
        if not checkall.all():
            self.okay = False
        return


def test_syph(dt=1, n_agents=500, do_plot=False):

    sim_kwargs = make_syph_sim(dt=dt, n_agents=n_agents)
    sim = ss.Sim(analyzers=check_states(), **sim_kwargs)
    sim.run()

    # Check plots
    burnin = 0
    pi = int(burnin/dt)

    if do_plot:
        tvec = sim.timevec[pi:]
        res = sim.results.syphilis
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        ax[0].stackplot(
            tvec,
            # res.n_susceptible[pi:],
            res.n_congenital[pi:],
            res.n_exposed[pi:],
            res.n_primary[pi:],
            res.n_secondary[pi:],
            (res.n_latent_temp[pi:]+res.n_latent_long[pi:]),
            res.n_tertiary[pi:],
        )
        ax[0].legend(['Congenital', 'Exposed', 'Primary', 'Secondary', 'Latent', 'Tertiary'], loc='lower right')

        ax[1].plot(tvec, res.prevalence[pi:])
        ax[1].set_title('Syphilis prevalence')

        ax[2].plot(tvec, sim.results.n_alive[pi:])
        ax[2].set_title('Population')

        ax[3].plot(tvec, res.new_infections[pi:])
        ax[3].set_title('New infections')

        fig.tight_layout()
        plt.show()

    return sim


def test_syph_intvs(dt=1, n_agents=500, do_plot=False):

    # Interventions
    # screen_eligible = lambda sim: sim.demographics.pregnancy.pregnant
    screen_eligible = lambda sim: sim.networks.mfnet.active(sim.people)
    syph_screening = ss.syph_screening(
        product='rpr',
        prob=0.99,
        eligibility=screen_eligible,
        start_year=2020,
        label='syph_screening',
    )

    treat_eligible = lambda sim: ss.uids(sim.interventions['syph_screening'].outcomes['positive'])
    bpg = ss.syph_treatment(
        prob=0.9,
        product='bpg',
        eligibility=treat_eligible,
        label='bpg'
    )

    sim_kwargs1 = make_syph_sim(dt=dt, n_agents=n_agents)
    sim_intv = ss.Sim(analyzers=[check_states()], interventions=[syph_screening, bpg], **sim_kwargs1)
    sim_intv.run()

    # Check plots
    if do_plot:
        # Run baseline
        sim_kwargs0 = make_syph_sim(dt=dt, n_agents=n_agents)
        sim_base = ss.Sim(**sim_kwargs0)
        sim_base.run()

        burnin = 10
        syph_b = sim_base.diseases.syphilis
        syph_i = sim_intv.diseases.syphilis
        pi = int(burnin/syph_b.t.dt)
        plt.figure()
        plt.plot(syph_b.timevec[pi:], syph_b.results.prevalence[pi:], label='Baseline')
        plt.plot(syph_i.timevec[pi:], syph_i.results.prevalence[pi:], label='S&T')
        plt.axvline(x=2020, color='k', ls='--')
        plt.title('Syphilis prevalence')
        plt.legend()
        plt.show()

        return sim_base, sim_intv

    else:
        return sim_intv


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    dt = [1/12, 1][quick_run]
    n_agents = [20e3, 500][quick_run]

    sim = test_syph(dt=dt, n_agents=n_agents, do_plot=do_plot)
    sim_base, sim_intv = test_syph_intvs(dt=dt, n_agents=n_agents, do_plot=True)

    T.toc()

