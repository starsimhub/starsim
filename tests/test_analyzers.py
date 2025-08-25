"""
Test analyzers
"""

import sciris as sc
import starsim as ss
sc.options(interactive=False) # Assume not running interactively


@sc.timer()
def test_infection_log(do_plot=False):
    """ Test infection log """
    sc.heading('Testing infection log')
    sim = ss.Sim(n_agents=1000, dt=0.2, dur=15, diseases='sir', networks='random', analyzers='infection_log')
    sim.run()
    log = sim.analyzers[0]
    assert len(log.logs[0]) > 900, 'Expect almost everyone to be infected'
    if do_plot:
        log.plot()
        log.animate()
    return log


@sc.timer()
def test_dynamics_by_age(do_plot=False):
    """ Test dynamics by age """
    sc.heading('Testing dynamics by age')
    by_age = ss.dynamics_by_age('sis.infected', age_bins=(0, 10, 30, 100))
    sim = ss.Sim(diseases='sis', networks='random', analyzers=by_age, copy_inputs=False)
    sim.run()

    # Tests
    h = by_age.hist
    assert h[0].sum() < h[10].sum() < h[30].sum(), 'Expected larger bins to have more infections'
    if do_plot:
        by_age.plot()
    return by_age


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    log    = test_infection_log(do_plot=do_plot)
    by_age = test_dynamics_by_age(do_plot=do_plot)

    T.toc()
