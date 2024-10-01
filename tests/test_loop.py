"""
Test the Loop class
"""

# %% Imports and settings
import sciris as sc
import starsim as ss

sc.options(interactive=False) # Assume not running interactively

pars = sc.objdict(
    dur      = 20,
    n_agents = 1000,
    diseases = 'sis',
    networks = 'random',
)


def test_run_options():
    sc.heading('Testing run options...')
    s1 = ss.Sim(pars).init()
    s2 = ss.Sim(pars).init()

    # Automatic run
    s1.run()

    # Manual run
    while s2.loop.index < len(s2.loop):
        s2.run_one_step()
    s2.finalize()

    assert s1.summary == s2.summary, 'Sims do not match'

    return s2.loop


def test_loop_plotting():
    sc.heading('Testing loop plotting...')
    sim = ss.Sim(pars).run()
    sim.loop.plot()
    sim.loop.plot_cpu()
    return sim.loop


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Run tests
    l1 = test_run_options()
    l2 = test_loop_plotting()

    T.toc()