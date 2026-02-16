"""
Test Starsim debugging features
"""
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100
medium = 1000

debug_pars = dict(
    n_agents = small,
    start = '2000-01-01',
    stop = '2010-01-01',
    diseases = 'sis',
    networks = 'random'
)

# %% Define the tests

@sc.timer()
def test_profile():
    sc.heading('Testing sim.profile()')

    # Based on advanced_profiling.ipynb
    sim = ss.Sim(**debug_pars)
    prof = sim.profile()
    return prof


@sc.timer()
def test_diagnostics_rvs():
    sc.heading('Testing sim diagnostics for random numbers')
    sim = ss.Sim(**debug_pars, diagnostics='rvs')
    sim.run()

    rvs = sim.diagnostics.rvs
    key = 'diseases_sis_trans_rng_dists_0' # Should be unique every time
    hashes = [entry[key].hash for entry in rvs.values()]
    assert len(hashes) == len(set(hashes))
    return rvs


@sc.timer()
def test_diagnostics_states():
    sc.heading('Testing sim diagnostics for states')
    sim = ss.Sim(**debug_pars, demographics=True)
    sim.set_diagnostics(states=True, detailed=True)
    sim.run()

    states = sim.diagnostics.states
    assert len(states[-1].alive) == len(sim.people)
    return states


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    prof = test_profile()
    rvs = test_diagnostics_rvs()
    states = test_diagnostics_states()

    sc.toc(T)
    plt.show()
    print('Done.')
