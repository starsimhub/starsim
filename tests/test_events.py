"""
Test agents taken event by event in continuous time (solver=ss.solvers.abm(step='events')); the agreement with the other solvers is in devtests/test_events.py
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
    make = lambda seed: ss.Sim(make_sirs(), networks=ssl.StaticNet(n_contacts=50), n_agents=2000, solver=ss.solvers.abm(step='events'), unit='day', stop=30, dt=1, rand_seed=seed, **kw).run()
    sim = make(1)
    res = sim.results.sirs
    assert isinstance(sim.solver, ss.solvers.abm) and 'events' in sim.solver_reason
    assert isinstance(sim.diseases.sirs.engine, ss.solvers.Events)
    assert np.all(res.S.values + res.I.values + res.R.values == 2000)
    assert np.array_equal(np.diff(res.I.values), (res.infection.values - res.recovery.values)[:-1]) # Each step's flows are its events
    assert res.I.values.max() > 300 # Infection at 2.5 a day spreads well within a step of a day
    assert np.array_equal(res.I.values, make(1).results.sirs.I.values)
    assert not np.array_equal(res.I.values, make(2).results.sirs.I.values)

    S, I, R = ss.states('S I R')
    other = ss.Disease(infection=S >> I @ 0.05*I, recovery=I >> R @ 0.2, init_prev=0.01, name='other')
    mixed = ss.Sim([make_sirs(solver=ss.solvers.abm(step='events')), other], networks=ssl.StaticNet(n_contacts=10), n_agents=1000, solver='abm', unit='day', stop=20, **kw).run()
    assert isinstance(mixed.diseases.sirs.engine, ss.solvers.Events) and isinstance(mixed.diseases.other.engine, ss.solvers.Agents) and not isinstance(mixed.diseases.other.engine, ss.solvers.Events)
    assert mixed.results.other.R[-1] > 0 and mixed.results.sirs.R.values.max() > 0

    pools = lambda: [ss.MixingPool(n_contacts=ss.constant(10), name='big'), ss.MixingPool(n_contacts=ss.constant(1), name='small')] # A person in both has each pool's own contacts
    S, I, R = ss.states('S I R')
    both = ss.Sim(ss.Disease(S, I, R, infection=S >> I @ 0.02*I, recovery=I >> R @ ss.days(6), init_n=dict(I=30), name='d'), networks=pools(), n_agents=3000, solver=ss.solvers.abm(step='events'), unit='day', stop=60, **kw).run()
    assert both.results.d.R[-1] > 400 # About 740, as by steps; it was 36 when the pools shared the last one's contacts
    return


def test_events_more():
    sc.heading('By events: the infection log, a network with acts, triggers at the end of the step, and infections by group')
    sim = ss.Sim(make_sirs(), networks=[ssl.StaticNet(n_contacts=10), ss.MixingPool(n_contacts=2)], n_agents=1000, solver=ss.solvers.abm(step='events'), unit='day', stop=20, analyzers='infection_log', **kw).run()
    df = sim.analyzers[0].to_df()
    inf = df[df.source != -1]
    assert len(inf) == sim.results.sirs.infection.values.sum() and set(inf.network) == {'staticnet', 'mixingpool'} # Each infection, with its source and route (in a pool too)

    S, I = ss.states('S I')
    sti = ss.Disease(infection=S >> I @ dict(sexualnet=ss.peract(m2f=0.1, f2m=0.0))*I, recovery=I >> S @ ss.peryear(0.2), init_prev=0.1, name='sti')
    sim = ss.Sim(sti, networks=ss.SexualNet(acts=ss.constant(ss.per(80, 'year'))), n_agents=1000, solver=ss.solvers.abm(step='events'), dt=1/12, start=2000, stop=2004, analyzers='infection_log', **kw).run()
    df = sim.analyzers[0].to_df()
    new = ss.uids(df.target[df.source != -1].to_numpy(dtype=int))
    assert len(new) > 20 and sim.people.female[new].all() # From men to women only

    S, E, I, Q, R = ss.states('S E I Q R')
    d = ss.Disease(infection=S >> E @ 0.3*I, onset=E >> I @ 0.5, recovery=I >> R @ 0.2, isolate=I >> Q @ ss.on(I.enter), release=Q >> R @ 0.2, init_n=dict(E=20), name='d')
    res = ss.Sim(d, networks=ssl.StaticNet(n_contacts=8), n_agents=1000, solver=ss.solvers.abm(step='events'), unit='day', stop=30, **kw).run().results.d
    assert res.onset.values.sum() > 100 and np.all(res.I.values == 0) # Isolated at the end of the step of their onset, unless they recovered first
    assert 0.8*res.onset.values.sum() < res.isolate.values.sum() <= res.onset.values.sum()

    S, I, R = ss.states('S I R')
    m = ss.Disease(S >> I @ 0.14*I, I >> R @ ss.days(8), init_n=dict(I=10), name='m')
    m.stratify(age={0: 0.3, 15: 0.6, 65: 0.1})
    res = ss.Sim(m, networks=ss.contacts(ages=[0, 15, 65], matrix=[[10, 6, 1], [3, 8, 1], [3, 6, 3]]), n_agents=2000, solver=ss.solvers.abm(step='events'), unit='day', stop=40, **kw).run().results.m
    by_age = [res.infection.select(age=a).sum() for a in ['0-14', '15-64', '65+']]
    assert np.isclose(sum(by_age), res.infection.values.sum()) and by_age[0] > by_age[2] > 0
    return


def test_events_refuse():
    sc.heading('What the events do not take yet is a CapabilityError that says what')
    m = make_sirs()
    m.stratify(patch=ss.levels(ss.geo.geography(names=['a', 'b'], populations=[600, 400], x=[0, 20], y=[0, 0]), mobility=ss.geo.gravity(away=0.01)))
    with pytest.raises(ss.CapabilityError, match='move between places'):
        ss.Sim(m, networks=ss.MixingPool(n_contacts=3), n_agents=500, solver=ss.solvers.abm(step='events'), **kw).init()
    with pytest.raises(ss.CapabilityError, match='no route'):
        ss.Sim(make_sirs(), n_agents=500, solver=ss.solvers.abm(step='events'), **kw).init()
    with pytest.raises(ValueError, match="step='events'"):
        make_sirs(**{'solver': 'abm-events'}) # The old name says what replaces it
    with pytest.raises(ss.CapabilityError, match='for agents only'):
        ss.Sim(make_sirs(solver=ss.solvers.abm(step='events')), total_pop=500, solver='ode', **kw).init()
    return


if __name__ == '__main__':
    T = sc.timer()
    test_events()
    test_events_more()
    test_events_refuse()
    T.toc()
