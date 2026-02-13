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

# %% Define the tests

@sc.timer()
def test_profile():
    sc.heading('Testing sim.profile()')

    # Based on advanced_profiling.ipynb
    pars = dict(
        n_agents = small,
        start = '2000-01-01',
        stop = '2010-01-01',
        diseases = 'sis',
        networks = 'random'
    )
    sim = ss.Sim(pars)
    prof = sim.profile()

    return prof


@sc.timer()
def test_debug_rvs():
    sc.heading('Testing sim.debug_rvs()')
    print('Feature not yet implemented')
    pass


@sc.timer()
def test_debug_states():
    sc.heading('Testing sim.debug_states()')
    print('Feature not yet implemented')
    pass


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    prof = test_profile()
    test_debug_rvs()
    test_debug_states()

    sc.toc(T)
    plt.show()
    print('Done.')
