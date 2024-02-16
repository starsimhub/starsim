"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import numpy as np

n_agents = 2_000


def test_default():
    """ Create, run, and plot a sim with default settings """
    sim = ss.Sim(n_agents=n_agents).run()
    sim.plot()
    return sim


def make_sim_pars():
    pars = dict(
        n_agents=n_agents,
        birth_rate=20,
        death_rate=0.015,
        networks=dict(
            name='random',
            n_contacts=4  # sps.poisson(mu=4),
        ),
        diseases=dict(
            name='sir',
            dur_inf=10,
            beta=0.1,
        )
    )
    return pars


def test_simple():
    """ Create, run, and plot a sim by passing a parameters dictionary """
    pars = make_sim_pars()
    sim = ss.Sim(pars)
    sim.run()
    sim.plot()
    return sim


def test_simple_vax(do_plot=False):
    """ Create and run a sim with vaccination """
    ss.set_seed(1)
    pars = make_sim_pars()
    sim_base = ss.Sim(pars=pars)
    sim_base.run()

    ss.set_seed(1)
    pars = make_sim_pars()
    sim_intv = ss.Sim(pars=pars, interventions=ss.routine_vx(start_year=2015, prob=0.5, product=ss.sir_vaccine()))
    sim_intv.run()

    # Check plots
    if do_plot:
        import matplotlib.pyplot as plt
        pi = 0

        plt.figure()
        plt.plot(sim_base.yearvec[pi:], sim_base.results.sir.prevalence[pi:], label='Baseline')
        plt.plot(sim_intv.yearvec[pi:], sim_intv.results.sir.prevalence[pi:], label='Vax')
        plt.axvline(x=2015, color='k', ls='--')
        plt.title('Prevalence')
        plt.legend()
        plt.show()

    return sim_base, sim_intv


def test_components():
    """ Create, run, and plot a sim by assembling components """
    people = ss.People(n_agents=n_agents)
    network = ss.networks.random(pars=dict(n_contacts=4))
    sir = ss.SIR(pars=dict(dur_inf=10, beta=0.1))
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    sim.plot()
    return sim


def test_parallel():
    """ Test running two identical sims in parallel """
    pars = make_sim_pars()
    s1 = ss.Sim(pars)
    s2 = ss.Sim(pars)
    # s1, s2 = ss.parallel(s1, s2).sims
    # assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)
    return s1, s2


if __name__ == '__main__':
    # sim1 = test_default()
    # sim2 = test_simple()
    sim_b, sim_i = test_simple_vax(do_plot=True)
    # sim3 = test_components()
    # s1, s2 = test_parallel()
