"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss


def test_simple():

    pars = dict(
        n_agents = 10_000,
        networks = dict(
            name= 'random',
            n_contacts = 4,
        ),
        diseases = dict(
            name = 'sir',
            dur_inf = 10,
            beta=0.1
        )
    )

    sim = ss.Sim(pars)
    sim.run()
    # sim.plot_states()
    return sim


def test_simple_v2():

    pars = dict(
        n_agents = 10_000,
        networks = dict(
            name= 'random',
            n_contacts = 4,
        ),
        diseases = dict(
            name = 'sir',
            dur_inf = 10,
            beta=0.1
        )
    )

    sim = ss.Sim(pars)
    sim.run()
    # sim.plot_states()
    return  sim


if __name__ == '__main__':

    sim1 = test_simple()
    sim1 = test_simple_v2()


