"""
Test the Dist object from distributions.py
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as plt
import pytest

n = 1_000_000 # For testing statistical properties and performance of distributions
m = 5 # For just testing that drawing works
dpy = 365.25 # Number of days per year
sc.options(interactive=False)


# %% Define the tests


@sc.timer()
def test_hist_plotting():
    """ Test that histogram plotting works """
    sc.heading('Test plotting of distribution histograms')

    # Based on the advanced distributions user guide
    sir = ss.SIR(
        init_prev = ss.bernoulli(p=0.15),
        dur_inf = ss.weibull(c=2, loc=1, scale=2)
    )
    sim = ss.Sim(n_agents=100, diseases=sir, dur=15, networks=ss.RandomNet())
    sim.run()

    # Plot the distribution
    rvs = sim.diseases.sir.pars.dur_inf.plot_hist()
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    # ss.options.warnings = 'error' # Turn warnings into exceptions for debugging

    T = sc.timer()

    o14 = test_hist_plotting()

    T.toc()
