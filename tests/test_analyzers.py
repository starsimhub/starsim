"""
Test analyzers
"""

import numpy as np
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
    assert df.t.dtype == float # Times as numbers, which can be subtracted (they were durations)
    mixed = ss.Sim(ss.SIR(beta=0.3), n_agents=300, unit='day', stop=10, analyzers=ss.infection_log(), verbose=0).run().analyzers[0].to_df()
    assert 'network' not in mixed.columns # With no networks no route is known, so no column of blanks
    if do_plot:
        log.plot()
        log.animate()
    return log


@sc.timer()
def test_shared_log():
    """ Several infection logs share the disease's, each step's entries can be read as the sim runs, and the strain transmitted is logged """
    sc.heading('Testing a shared infection log')
    class steps(ss.Analyzer):
        def init_pre(self, sim):
            super().init_pre(sim)
            self.n = []
        def step(self):
            self.n.append(len(self.sim.diseases.d.infection_log.to_df(ti=self.ti)))

    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=S >> I @ ss.peryear(1)*I, recovery=I >> R @ ss.years(2), init=dict(I=0.05), strains=ss.Strains(['A', 'B']), name='d')
    sim = ss.Sim(dis, n_agents=500, dur=5, networks='random', analyzers=[ss.infection_log(name='a'), ss.infection_log(name='b'), steps()], verbose=0).run()
    a, b = sim.analyzers.a, sim.analyzers.b
    assert a.log is b.log is sim.diseases.d.infection_log # Shared, not taken by the first
    df = a.to_df()
    assert sim.analyzers.steps.n == np.bincount(np.searchsorted(sim.t.yearvec, df.t), minlength=sim.t.npts).tolist() # Each step's entries, as they were logged
    assert sim.analyzers.steps.n[1:] == sim.results.d.infection.values[1:].tolist() # After the seeds, the step's infections
    assert set(df.strain) == {'A', 'B'}
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    log    = test_infection_log(do_plot=do_plot)
    shared = test_shared_log()

    T.toc()
