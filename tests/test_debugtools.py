"""
Test Starsim debugging features
"""
import sciris as sc
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100
medium = 1000

# %% Define the tests

@sc.timer()
def test_profile():
    sc.heading('Testing sim.profile()')

    pass


@sc.timer()
def test_debug_rvs():
    sc.heading('Testing sim.debug_rvs()')

    pass


@sc.timer()
def test_debug_states():
    sc.heading('Testing sim.debug_states()')

    pass


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    test_profile()
    test_debug_rvs()
    test_debug_states()

    sc.toc(T)
    plt.show()
    print('Done.')
