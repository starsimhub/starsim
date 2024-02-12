"""
Test simple APIs
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps


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
    return  sim


if __name__ == '__main__':

    sim = test_simple()


