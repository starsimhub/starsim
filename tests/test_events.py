"""
Test agents taken event by event in continuous time (solver='abm-events'); the agreement with the other solvers is in devtests/test_events.py
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

kw = dict(verbose=0)


def make_sirs(**kwargs):
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=S >> I @ 0.05*I, recovery=I >> R @ 1, waning=R >> S @ (1/14), init_prev=0.01, name='sirs', **kwargs)


def test_events():
    sc.heading('Agents by events: reproducible, people kept, flows that add up, and a module by events beside one by steps')
    make = lambda seed: ss.Sim(make_sirs(), networks=ssl.StaticNet(n_contacts=50), n_agents=2000, solver='abm-events', unit='day', stop=30, dt=1, rand_seed=seed, **kw).run()
    sim = make(1)
    res = sim.results.sirs
    assert sim.solver == 'abm' and 'events' in sim.solver_reason
    assert isinstance(sim.diseases.sirs.engine, ss.solvers.Events)
    assert np.all(res.S.values + res.I.values + res.R.values == 2000)
    assert np.array_equal(np.diff(res.I.values), (res.infection.values - res.recovery.values)[1:]) # Each step's flows are its events
    assert res.I.values.max() > 300 # Infection at 2.5 a day spreads well within a step of a day
    assert np.array_equal(res.I.values, make(1).results.sirs.I.values)
    assert not np.array_equal(res.I.values, make(2).results.sirs.I.values)

    S, I, R = ss.states('S I R')
    other = ss.Disease(infection=S >> I @ 0.05*I, recovery=I >> R @ 0.2, init_prev=0.01, name='other')
    mixed = ss.Sim([make_sirs(solver='abm-events'), other], networks=ssl.StaticNet(n_contacts=10), n_agents=1000, solver='abm', unit='day', stop=20, **kw).run()
    assert isinstance(mixed.diseases.sirs.engine, ss.solvers.Events) and type(mixed.diseases.other.engine) is ss.solvers.Agents
    assert mixed.results.other.R[-1] > 0 and mixed.results.sirs.R.values.max() > 0

    pools = lambda: [ss.MixingPool(n_contacts=ss.constant(10), name='big'), ss.MixingPool(n_contacts=ss.constant(1), name='small')] # A person in both has each pool's own contacts
    S, I, R = ss.states('S I R')
    both = ss.Sim(ss.Disease(S, I, R, infection=S >> I @ 0.02*I, recovery=I >> R @ ss.days(6), init_n=dict(I=30), name='d'), networks=pools(), n_agents=3000, solver='abm-events', unit='day', stop=60, **kw).run()
    assert both.results.d.R[-1] > 400 # About 740, as by steps; it was 36 when the pools shared the last one's contacts
    return


def test_events_refuse():
    sc.heading('What the events do not take yet is a CapabilityError that says what')
    with pytest.raises(ss.CapabilityError, match='infection log'):
        ss.Sim(make_sirs(), networks=ss.RandomNet(), n_agents=500, solver='abm-events', analyzers='infection_log', **kw).init()
    with pytest.raises(ss.CapabilityError, match='no route'):
        ss.Sim(make_sirs(), n_agents=500, solver='abm-events', **kw).init()
    with pytest.raises(ValueError, match='abm-events'):
        make_sirs(solver='events')
    with pytest.raises(ss.CapabilityError, match='for agents only'):
        ss.Sim(make_sirs(solver='abm-events'), total_pop=500, solver='ode', **kw).init()
    return


if __name__ == '__main__':
    T = sc.timer()
    test_events()
    test_events_refuse()
    T.toc()
