"""
Test the distributions: units on a lognormal's log parameters, and the warnings about distribution defaults
"""

import numpy as np
import sciris as sc
import starsim as ss


def test_lognormal_log_unit():
    sc.heading('Testing that unit= on ss.lognormal(log_mean=, log_std=) (whose parameters are of the log) scales its draws, as a dwell time')
    def mean_dwell(dur):
        class M(ss.Disease):
            S, I, R = ss.states('S I R')
            rec = ss.Transition(I, R, dur=dur)
        sim = ss.Sim(diseases=M(init=dict(I=1.0)), n_agents=5000, dur=ss.days(200), dt=ss.days(1), unit='day', verbose=0, rand_seed=1).run()
        I = sim.results.m.I.values
        return I.sum()/I[0]
    mu, sigma = np.log(3), 0.5
    a = mean_dwell(ss.lognormal(log_mean=mu, log_std=sigma, unit='week'))
    b = mean_dwell(ss.par(ss.lognormal(log_mean=mu, log_std=sigma), name='d')*ss.weeks(1))
    f = lambda self, sim, uids: np.full(len(uids), mu) # A function of the agents
    c = mean_dwell(ss.lognormal(log_mean=f, log_std=sigma, unit='week'))
    assert a == b == c
    assert abs(a - 7*np.exp(mu + sigma**2/2)) < 2 # In days, not weeks (or 7× too short)
    return


if __name__ == '__main__':
    test_lognormal_log_unit()
