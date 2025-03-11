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
        demographics = True,
        networks = 'random',
        diseases = sc.objdict(type='sir', beta=0.1), # To allow for modification later
    )
    return pars


def test_parallel():
    """ Test running two identical sims in parallel """
    sc.heading('Testing parallel...')
    pars = make_sim_pars()

    # Check that two identical sims match
    msim = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars, label='Sim2')])
    msim.run(shrink=False)
    s1, s2 = msim.sims
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True), "Sims don't match and should"

    # Check that two non-identical sims don't match
    pars2 = sc.dcp(pars)
    pars2.diseases.beta *= 2
    sims = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars2, label='Sim2')])
    sims.run(shrink=False)
    s1, s2 = sims.sims
    assert not np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True), "Sims do match and shouldn't"

    return s1, s2


def test_multisim():
    """ Check MultiSim methods """
    sc.heading('Testing MultiSim')
    pars = make_sim_pars()
    msim = ss.MultiSim(ss.Sim(pars))
    msim.init_sims()
    msim.run(n_runs=4)

    # Plot individual sims
    msim.plot()

    # Reduce and plot mean
    msim.mean()
    msim.plot()

    # Export results
    res_df = msim.results.to_df(resample='2y')
    assert res_df.sir_n_susceptible_low.loc['2030-12-31'] == msim.results.sir_n_susceptible.low[29:31].mean()

    # Reduce and plot median
    msim.median()
    msim.plot()

    # Other methods
    msim.show()
    msim.summarize()
    msim.reset()

    return msim


def test_other():
    """ Check other run options """
    sc.heading('Testing other run options')
    # Check parallel
    pars = make_sim_pars()
    pars2 = sc.dcp(pars)
    pars2.diseases.beta *= 2
    s1 = ss.Sim(pars)
    s2 = ss.Sim(pars2)
    ss.parallel(s1, s2, inplace=False)
    assert not s1.initialized
    ss.parallel(s1, s2, inplace=True)
    assert not np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True), "Sims do match and shouldn't"

    # Check single run
    s3 = ss.Sim(pars)
    s4 = ss.Sim(pars)
    s3 = ss.single_run(s3)
    s4 = ss.single_run(s4, n_agents=2000)
    assert not np.allclose(s3.summary[:], s4.summary[:], rtol=0, atol=0, equal_nan=True), "Sims do match and shouldn't"

    return s3,s4


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    s1, s2 = test_parallel()
    msim = test_multisim()
    s3,s4 = test_other()

    T.toc()

    if do_plot:
        plt.show()
