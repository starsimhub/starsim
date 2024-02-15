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
        n_agents = n_agents,
        birth_rate = 20,
        death_rate = 0.015,
        networks = dict(
            type = 'randomnet',
            n_contacts = 4  # sps.poisson(mu=4),
        ),
        diseases = dict(
            type = 'sir',
            dur_inf = 10,
            beta = 0.1,
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


def test_simple_vax():
    """ Create and run a sim with vaccination """
    pars = make_sim_pars()
    sim = ss.Sim(pars, interventions=ss.routine_vx(prob=0.5, product=ss.sir_vaccine()))
    sim.run()
    sim.plot()
    return sim


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
    sim1 = test_default()
    sim2 = test_simple()
    sim = test_simple_vax()
    sim3 = test_components()
    s1, s2 = test_parallel()
