"""
Test simple APIs
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

def test_s0():

    pars = dict(
        n_agents = 10_000,
        networks =ss.mf(
        pars=dict(
            duration_dist=ss.lognorm(mean=1/24, stdev=0.5),
            acts=ss.lognorm(mean=80, stdev=30),
        )
    ),
        diseases = ss.SIR(
            dur_inf = 10,
        )
    )

    sim = ss.Sim(pars)
    sim.run()
    sim.plot_states()
    return  sim

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


