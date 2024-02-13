"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import scipy.stats as sps

n_agents = 2_000


def test_default():
    """ Create, run, and plot a sim with default settings """
    sim = ss.Sim(n_agents=n_agents).run()
    sim.plot()
    return sim

def test_simple():
    """ Create, run, and plot a sim with specified parameters """

    pars = dict(
        n_agents = n_agents,
        birth_rate=20,
        death_rate=0.015,
        networks = dict(
            name= 'random',
            n_contacts =4 # sps.poisson(mu=4),
        ),
        diseases = dict(
            name = 'sir',
            dur_inf = 10,
            beta = 0.1,
        )
    )

    sim = ss.Sim(pars)
    sim.run()
    sim.plot()
    return sim


def test_components():
    """ Create, run, and plot a sim by assembling components """
    people = ss.People(n_agents=n_agents)
    network = ss.networks.random(n_contacts=4)
    sir = ss.SIR(dur_inf=10, beta=0.1)
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    sim.plot()
    return sim


if __name__ == '__main__':

    sim1 = test_default()
    sim2 = test_simple()
    sim3 = test_components()


