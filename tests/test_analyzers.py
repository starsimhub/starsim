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
    df = log.to_df()
    assert len(df) > 900, 'Expect almost everyone to be infected'
    assert set(df.network.dropna()) == {'randomnet'} # Network recorded for each transmission
    assert df.network.isna().sum() == df.source.lt(0).sum() # Seed infections have no network
    if do_plot:
        log.plot()
        log.animate()
    return log


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    log    = test_infection_log(do_plot=do_plot)

    T.toc()
