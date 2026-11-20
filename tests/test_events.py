"""
Test clocks (time in a state or a set of states), and agents taken event by event in continuous time (solver=ss.solvers.abm(step='events'));
the agreement of the events with the other solvers is in devtests/test_events.py
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss
import starsim.library as ssl
from corpus import BV

kw = dict(verbose=0)


@sc.timer()
def test_clocks():
    sc.heading('Testing clocks: X.time_in, (A | B).time_in, and X.progress')
    def make(rate=None, **kwargs):
        S, A, B, C = ss.states('S A B C')
        return ss.Disease(S, A, B, C,
            start = S >> A @ ss.perday(0.05),
            ab    = A >> B @ ss.days(4),
            bc    = ss.Transition(B, C, rate=rate) if rate is not None else B >> C @ ss.days(4),
            init  = dict(B=0.1), name='d', **kwargs), (A | B)
    seen = sc.obj(n=0)
    def check(sim):
        d = sim.diseases.d
        st = d.states # The declarations, in the sim; a clock of a set is made before the sim, so it is kept
        ti, dt = d.ti, d.t.dt.to('day').value
        u = d.B.uids
        placed = d.ti_B[u] == -1 # At the start, dated by the step before the first
        assert np.allclose(d.engine.eval_for(st.B.time_in, u), (ti - d.ti_B[u] - placed)*dt) # The steps since the step they entered in, by a move at its start
        both = d.engine.eval_for(seen.ab, u)
        came = u[~np.isnan(d.ti_A[u])] # From A; the others were placed in B, so entered the set then
        assert np.allclose(both[np.isin(u, came)], np.maximum(ti - d.ti_A[came], 0)*dt) # Moving from A to B did not reset it
        assert not np.isnan(both).any()
        prog = d.engine.eval_for(st.A.progress, d.A.uids)
        assert np.all((prog >= 0) & (prog <= 1))
        assert np.isnan(d.engine.eval_for(st.A.time_in, d.S.uids[:10])).all() # NaN outside the domain
        seen.n += len(came)
    d, ab = make(check=check)
    seen.ab = ab.time_in
    ss.Sim(d, n_agents=2000, unit='day', stop=40, **kw).run()
    assert seen.n > 100
    # In a rate, a modifier amount, and a condition
    S, A, B, C = ss.states('S A B C')
    late = ss.Transition(B, C, rate=ss.perday(1)*(B.time_in >= ss.days(10)), name='late') # Only after 10 days in B
    d = ss.Disease(S, A, B, C, start=S >> A @ ss.perday(0.05), ab=A >> B @ ss.days(4), late=late, init=dict(B=0.1), name='d')
    sim = ss.Sim(d, n_agents=2000, unit='day', stop=40, **kw).run()
    assert sim.results.d.late[:10].sum() == 0 and sim.results.d.late.sum() > 100 # A split that reads a clock is in test_branching.py

    # A clock of a state with no exit, read only by a modifier, is known when the sim is built
    S, I, R = ss.states(3)
    d = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='d')
    protect = ss.modify(d.rel_sus, multiply=0.5, who=R.time_in > 5) # R has no exit, so nothing else keeps its times
    assert isinstance(ss.Sim(d, n_agents=500, stop=30, connectors=protect, **kw).run().solver, ss.solvers.abm)

    S, A, B, C = ss.states('S A B C') # A clock read outside its domain
    with pytest.raises(ValueError):
        ss.Disease(S, A, B, C, ab=A >> B @ ss.days(4), sc_=ss.Transition(S, C, rate=ss.perday(1)*(B.time_in > 1)))
    return sim


@sc.timer()
def test_bv_recurrence():
    sc.heading('Testing BV recurrence hazard decays with time since treatment, P.time_in')
    def recur_rate(sim, uids): # The same hazard as a function of the time of entering
        bv = sim.diseases.bv
        months = (sim.ti - bv.ti_P[uids]) * sim.t.dt.months # The steps since the step of entering P
        return ss.permonth(0.06 * np.exp(-months / 3))
    class Old(BV):
        recur = ss.Transition(BV.P, BV.CST4, rate=recur_rate)

    kw2 = dict(n_agents=10_000, start=2000, stop=2005, dt=ss.months(1), unit='year', **kw)
    new = ss.Sim(BV(name='bv'), **kw2).run().results.bv
    old = ss.Sim(Old(name='bv'), **kw2).run().results.bv
    assert new.recur.sum() > 500
    assert np.array_equal(new.recur, old.recur) and np.array_equal(new.P, old.P) # The same hazards, so the same draws
    return new


@sc.timer()
def test_stamps():
    sc.heading('A time of entering reads as the step it happened in: ti_X.isclose(sim.ti) for a move at the start of the step and an infection at its end')
    S, E, I, R = ss.states('S E I R')
    d = ss.Disease(inf=S >> E @ ss.perday(0.4)*I, lat=E >> I @ ss.days(2), rec=I >> R @ ss.days(3), init=dict(E=0.01), name='d')
    def check(sim):
        d, ti = sim.diseases.d, sim.ti
        if ti > 0:
            assert d.ti_E.isclose(ti).count() == d.results.inf[ti] and d.ti_I.isclose(ti).count() == d.results.lat[ti] # Infected, then moved on, in this step
    sim = ss.Sim(d, networks=ss.RandomNet(), analyzers=check, n_agents=5000, unit='day', stop=30, **kw).run()
    assert sim.results.d.inf.sum() > 100 and sim.results.d.lat.sum() > 100
    return


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

    return


def test_events_more():
    sc.heading('By events: the infection log, and triggers at the end of the step')
    sim = ss.Sim(make_sirs(), networks=[ssl.StaticNet(n_contacts=10), ss.MixingPool(n_contacts=2)], n_agents=1000, solver=ss.solvers.abm(step='events'), unit='day', stop=20, analyzers='infection_log', **kw).run()
    df = sim.analyzers[0].to_df()
    inf = df[df.source != -1]
    assert len(inf) == sim.results.sirs.infection.values.sum() and set(inf.network) == {'staticnet', 'mixingpool'} # Each infection, with its source and route (in a pool too)

    S, E, I, Q, R = ss.states('S E I Q R')
    d = ss.Disease(infection=S >> E @ 0.3*I, onset=E >> I @ 0.5, recovery=I >> R @ 0.2, isolate=I >> Q @ ss.on(I.enter), release=Q >> R @ 0.2, init_n=dict(E=20), name='d')
    res = ss.Sim(d, networks=ssl.StaticNet(n_contacts=8), n_agents=1000, solver=ss.solvers.abm(step='events'), unit='day', stop=30, **kw).run().results.d
    assert res.onset.values.sum() > 100 and np.all(res.I.values == 0) # Isolated at the end of the step of their onset, unless they recovered first
    assert 0.8*res.onset.values.sum() < res.isolate.values.sum() <= res.onset.values.sum()

    # What the events do not take yet is a CapabilityError that says what
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make_sirs(solver=ss.solvers.abm(step='events')), total_pop=500, solver='ode', **kw).init()
    return


if __name__ == '__main__':
    T = sc.timer()
    test_clocks()
    test_bv_recurrence()
    test_stamps()
    test_events()
    test_events_more()
    T.toc()
