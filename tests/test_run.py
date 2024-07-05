"""
Test run
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def make_sim_pars():
    pars = sc.objdict(
        n_agents = n_agents,
        birth_rate = 20,
        death_rate = 15,
        networks = sc.objdict(
            type = 'randomnet',
            n_contacts = 4,
        ),
        diseases = sc.objdict(
            type = 'sir',
            dur_inf = 10,
            beta = 0.1,
        )
    )
    return pars


def test_parallel():
    """ Test running two identical sims in parallel """
    sc.heading('Testing parallel...')
    pars = make_sim_pars()

    # Check that two identical sims match
    msim = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars, label='Sim2')])
    msim.run(keep_people=True)
    s1, s2 = msim.sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    # Check that two non-identical sims don't match
    pars2 = sc.dcp(pars)
    pars2.diseases.beta *= 2
    sims = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars2, label='Sim2')])
    sims.run(keep_people=True)
    s1, s2 = sims.sims
    assert not np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


def test_multisim():
    """ Check MultiSim methods """
    sc.heading('Testing MultiSim')
    pars = make_sim_pars()
    msim = ss.MultiSim(ss.Sim(pars))
    msim.run(n_runs=4)
    
    msim.reset()
    msim.shrink()
    msim.reduce()
    msim.mean()
    msim.median()
    # msim.plot()
    return msim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    s1, s2 = test_parallel()
    msim = test_multisim()

    T.toc()
    
    if do_plot:
        plt.show()
