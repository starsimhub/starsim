"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import scipy.stats as sps

def test_simple():

    pars = dict(
            n_agents = 10_000,
        birth_rate=20,
                death_rate=0.015,
        networks = dict(
            name= 'random',
            n_contacts =4 # sps.poisson(mu=4),
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

    sim = test_simple()


