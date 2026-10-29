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


def test_lognorm_im_unit():
    """ TY5: unit= on ss.lognorm_im() (whose parameters are of the log) scales its draws, as a dwell time """
    def mean_dwell(dur):
        class M(ss.Disease):
            S, I, R = ss.states('S I R')
            rec = ss.Transition(I, R, dur=dur)
        sim = ss.Sim(diseases=M(init=dict(I=1.0)), n_agents=5000, dur=ss.days(200), dt=ss.days(1), unit='day', verbose=0, rand_seed=1).run()
        I = sim.results.m.I.values
        return I.sum()/I[0]
    mu, sigma = np.log(3), 0.5
    a = mean_dwell(ss.lognorm_im(mean=mu, sigma=sigma, unit='week'))
    b = mean_dwell(ss.par(ss.lognorm_im(mean=mu, sigma=sigma), name='d')*ss.weeks(1))
    f = lambda self, sim, uids: np.full(len(uids), mu) # A function of the agents
    c = mean_dwell(ss.lognorm_im(mean=f, sigma=sigma, unit='week'))
    assert a == b == c
    assert abs(a - 7*np.exp(mu + sigma**2/2)) < 2 # In days, not weeks (or 7× too short)
    return a


def test_dist_default_warnings():
    """ MS8: a typed duration for a distribution default sets its mean, and says so; in a declared model, a single value replaces the distribution, and says so """
    defaults = dict(dur=ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)))
    with pytest.warns(RuntimeWarning, match='set only the mean'):
        p = ss.pars(defaults, dur=ss.days(100))
    assert isinstance(p.dur.raw, ss.lognorm_ex) and p.dur.raw.pars.mean == 100
    class M(ss.Disease):
        S, I = ss.states('S I')
        dur = ss.par(ss.lognorm_ex(ss.days(5), ss.days(5)))
        inf = S >> I @ ss.perday(0.1)*I
        rec = I >> S @ dur
    with pytest.warns(RuntimeWarning, match='replaced its default distribution'):
        m = M(dur=ss.days(10))
    assert isinstance(m.pars.dur.raw, ss.dur)
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    # ss.options.warnings = 'error' # Turn warnings into exceptions for debugging

    T = sc.timer()

    o14 = test_hist_plotting()

    test_lognorm_im_unit()
    test_dist_default_warnings()
    T.toc()
