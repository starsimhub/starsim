"""
Test events (transitions, branches, entering and leaving states) and clocks (time in a state or a set of states)
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


@sc.timer()
def test_clocks():
    sc.heading('Testing clocks: X.time, (A | B).time, and X.progress')
    def make(rate=None, **kwargs):
        S, A, B, C = ss.states('S A B C')
        return ss.Disease(S, A, B, C,
            start = S >> A @ ss.perday(0.05),
            ab    = A >> B @ ss.days(4),
            bc    = ss.Transition(B, C, rate=rate) if rate is not None else B >> C @ ss.days(4),
            init  = dict(B=0.1), name='d', **kwargs), (A | B)
    seen = sc.objdict(n=0)
    def check(sim):
        d = sim.diseases.d
        st = d.states # The declarations, in the sim; a clock of a set is made before the sim, so it is kept
        ti, dt = d.ti, d.t.dt.to('day').value
        u = d.B.uids
        placed = d.ti_B[u] == 0 # At the start, which is a step before the first step's index
        assert np.allclose(d.engine.eval_for(st.B.time, u), (ti - d.ti_B[u] - placed)*dt) # The steps since the step they entered in, by a move at its start
        both = d.engine.eval_for(seen.ab, u)
        came = u[~np.isnan(d.ti_A[u])] # From A; the others were placed in B, so entered the set then
        assert np.allclose(both[np.isin(u, came)], np.maximum(ti - d.ti_A[came], 0)*dt) # Moving from A to B did not reset it
        assert not np.isnan(both).any()
        prog = d.engine.eval_for(st.A.progress, d.A.uids)
        assert np.all((prog >= 0) & (prog <= 1))
        assert np.isnan(d.engine.eval_for(st.A.time, d.S.uids[:10])).all() # NaN outside the domain
        seen.n += len(came)
    d, ab = make(check=check)
    seen.ab = ab.time
    ss.Sim(d, n_agents=2000, unit='day', stop=40, **kw).run()
    assert seen.n > 100

    # In a rate, a modifier amount, and a condition
    S, A, B, C = ss.states('S A B C')
    late = ss.Transition(B, C, rate=ss.perday(1)*(B.time >= ss.days(10)), name='late') # Only after 10 days in B
    d = ss.Disease(S, A, B, C, start=S >> A @ ss.perday(0.05), ab=A >> B @ ss.days(4), late=late, init=dict(B=0.1), name='d')
    sim = ss.Sim(d, n_agents=2000, unit='day', stop=40, **kw).run()
    assert sim.results.d.late[:10].sum() == 0 and sim.results.d.late.sum() > 100
    S, B, C, D = ss.states('S B C D') # A split when taken reads the dwell just completed (HPV: CIN from the time in P)
    long = 1.0*(B.time >= ss.days(5))
    d = ss.Disease(S, B, C, D, sb=S >> B @ ss.perday(0.05), split=ss.Transition(B, [C, D], p=[long, None], dur=ss.days(5)), name='d')
    with ss.options.context(keep_times=True): # The times of entering C and D are read below
        s = ss.Sim(d, n_agents=2000, unit='day', stop=60, **kw).run().diseases.d
    c, dd = s.C.uids, s.D.uids
    assert len(c) > 100 and len(dd) > 100
    assert np.all(s.ti_C[c] - s.ti_B[c] >= 5) and np.all(s.ti_D[dd] - s.ti_B[dd] < 5)
    d, ab = make()
    vx = ss.vaccinate(ss.Vaccine(efficacy=0), where=d.B & (d.B.time >= ss.days(2)), name='vx')
    halve = ss.modify(d.bc, multiply=0.5**(ab.time/ss.days(4)), where=ab, name='halve')
    sim = ss.Sim(d, interventions=vx, connectors=halve, n_agents=2000, unit='day', stop=40, **kw).run()
    assert sim.results.vx.delivered.sum() > 0 and sim.solver == 'abm'

    # Errors: outside the domain, and no scheduled dwell; as compartments, see test_conveyor.py
    S, A, B, C = ss.states('S A B C')
    with pytest.raises(ValueError, match='which has a value only for people in B, but moves people from S'):
        ss.Disease(S, A, B, C, ab=A >> B @ ss.days(4), sc_=ss.Transition(S, C, rate=ss.perday(1)*(B.time > 1)))
    d, ab = make()
    with pytest.raises(ValueError, match='but applies to everyone'):
        ss.Sim(d, connectors=ss.modify(d.bc, multiply=ab.time/ss.days(1)), n_agents=100, **kw).init()
    S, A, B, C = ss.states('S A B C')
    with pytest.raises(ValueError, match='so no dwell time is scheduled'):
        ss.Disease(S, A, B, C, ab=A >> B @ ss.perday(0.2), bc=ss.Transition(A, C, rate=ss.perday(1)*A.progress))
    d, ab = make()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(d, connectors=ss.modify(d.bc, multiply=ab.time/ss.days(1), where=ab), solver='ode', **kw).init()
    d, ab = make()
    vx = ss.vaccinate(ss.Vaccine(efficacy=0), where=d.B & (d.B.time >= ss.days(2)), name='vx')
    assert ss.Sim(d, interventions=vx, **kw).init().solver == 'abm' # A delivery reading a clock: the reason is printed
    return sim


class BV(ss.Disease):
    CST1, CST4, P = ss.states('CST1 CST4 P')
    first = 'CST1'
    recur0, recur_tau = ss.par(ss.permonth(0.06)), ss.par(ss.months(3))
    onset  = CST1 >> CST4 @ ss.permonth(0.05)
    tx     = CST4 >> P @ ss.months(2) # Treated after two months, on average: a dwell time, which ends in the state-change row
    recur  = ss.Transition(P, CST4, rate=recur0*np.exp(-P.time/recur_tau)) # Largest on entry
    settle = ss.Transition(P, CST1, dur=ss.months(6))


@sc.timer()
def test_bv_recurrence():
    sc.heading('Port: BV recurrence hazard decays with time since treatment, P.time')
    def recur_rate(sim, uids): # The workaround (bv_gud_connectors/port.py)
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
    sc.heading('A time of entering reads as the step it happened in: ti_X == sim.ti for a move at the start of the step and an infection at its end')
    S, E, I, R = ss.states('S E I R')
    d = ss.Disease(inf=S >> E @ ss.perday(0.4)*I, lat=E >> I @ ss.days(2), rec=I >> R @ ss.days(3), init=dict(E=0.01), name='d')
    def check(sim):
        d, ti = sim.diseases.d, sim.ti
        if ti > 0:
            assert (d.ti_E == ti).count() == d.results.inf[ti] and (d.ti_I == ti).count() == d.results.lat[ti] # Infected, then moved on, in this step
    sim = ss.Sim(d, networks=ss.RandomNet(), analyzers=check, n_agents=5000, unit='day', stop=30, **kw).run()
    assert sim.results.d.inf.sum() > 100 and sim.results.d.lat.sum() > 100
    return


if __name__ == '__main__':
    sc.options(interactive=False)
    test_clocks()
    test_bv_recurrence()
    test_stamps()
