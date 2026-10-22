"""
Test events (transitions, branches, entering and leaving states) and clocks (time in a state or a set of states)
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


def make_seir(**kwargs):
    S, E, I, R, D = ss.states('S E I R D')
    infected = E | I
    return ss.Disease(S, E, I, R, D,
        infection = S >> E @ 0.3*I,
        onset     = E >> I @ ss.days(3),
        outcome   = ss.Transition(I, [R, D], p=[0.9, None], dur=7),
        init_prev = 0.01, name='d', **kwargs), infected


@sc.timer()
def test_events():
    sc.heading('Testing events: transitions, branches, and entering and leaving states')
    checked = sc.objdict(n=0)
    def check(sim): # After the transmission, every event of the step is visible
        d = sim.diseases.d
        st = d.graph.states
        infected = st.E | st.I
        inf, onset = d.infection.uids, d.onset.uids
        assert np.array_equal(infected.enter.uids, inf) # Entering the set from outside; E -> I is within it
        assert np.array_equal(st.I.enter.uids, onset)
        assert np.array_equal(infected.exit.uids, d.outcome.uids)
        assert np.array_equal(np.sort(ss.uids.concatenate([d.outcome.to(st.R).uids, d.outcome.to(st.D).uids])), np.sort(d.outcome.uids))
        assert d.R[d.outcome.to(st.R).uids].all()
        checked.n += len(inf)
    d, infected = make_seir(check=check)
    sim = ss.Sim(d, n_agents=5000, unit='day', stop=60, **kw).run()
    assert checked.n > 1000

    # As conditions, and in on=: one delivery for each person who enters the set, and each who dies
    for on, flow in [(lambda d: (d.E | d.I).enter, 'infection'), (lambda d: d.outcome.to(d.D), None)]:
        d, infected = make_seir()
        event = on(d)
        vx = ss.vaccinate(ss.Vaccine(efficacy=0), on=event, name='vx')
        res = ss.Sim(d, interventions=vx, n_agents=5000, unit='day', stop=60, **kw).run().results
        target = res.d[flow][:-1].sum() if flow else res.d.n_D[-2] # Events in a step are acted on by the next, so not those of the last
        assert np.isclose(res.vx.delivered.sum(), target, rtol=0.02), f'{event}: {res.vx.delivered.sum()} delivered, {target} expected'

    # A branch is an event under a compartmental solver too; entering a state is not, yet
    d, infected = make_seir()
    ode = ss.Sim(d, interventions=ss.vaccinate(ss.Vaccine(efficacy=0), on=d.outcome.to(d.D), name='vx'), solver='ode', total_pop=5000, unit='day', stop=60, **kw).run().results
    assert np.isclose(ode.vx.delivered.sum(), 0.1*ode.d.outcome[:-1].sum(), rtol=0.01)
    d, infected = make_seir()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(d, interventions=ss.vaccinate(ss.Vaccine(efficacy=0), on=infected.enter), solver='ode', **kw).run()
    with pytest.raises(ValueError):
        d.outcome.to(d.E)
    return sim


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
        st = d.graph.states # The declarations, in the sim; a clock of a set is made before the sim, so it is kept
        ti, dt = d.ti - 1, d.t.dt.to('day').value # The start of the step, which clocks are worked out at
        u = d.B.uids
        assert np.allclose(d.engine.eval_for(st.B.time, u), (ti - d.ti_B[u])*dt)
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

    # Errors: outside the domain (E212), and no scheduled dwell (E213); as compartments, see test_conveyor.py
    S, A, B, C = ss.states('S A B C')
    with pytest.raises(ss.errors.ClockOutsideDomain):
        ss.Disease(S, A, B, C, ab=A >> B @ ss.days(4), sc_=ss.Transition(S, C, rate=ss.perday(1)*(B.time > 1)))
    d, ab = make()
    with pytest.raises(ss.errors.ClockOutsideDomain):
        ss.Sim(d, connectors=ss.modify(d.bc, multiply=ab.time/ss.days(1)), n_agents=100, **kw).init()
    S, A, B, C = ss.states('S A B C')
    with pytest.raises(ss.errors.NoScheduledDwell):
        ss.Disease(S, A, B, C, ab=A >> B @ ss.perday(0.2), bc=ss.Transition(A, C, rate=ss.perday(1)*A.progress))
    d, ab = make()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(d, connectors=ss.modify(d.bc, multiply=ab.time/ss.days(1), where=ab), solver='ode', **kw).init()
    d, ab = make()
    vx = ss.vaccinate(ss.Vaccine(efficacy=0), where=d.B & (d.B.time >= ss.days(2)), name='vx')
    assert ss.Sim(d, interventions=vx, **kw).init().solver == 'abm' # A delivery reading a clock: the reason is printed
    return sim


@sc.timer()
def test_syphilis_latent():
    sc.heading('Port: syphilis latent infectiousness decays with time since latent onset, (EL | LL).time')
    hl = 1.0 # Years
    def make(form, ll=True):
        S, E, EL, LL = ss.states('S E EL LL')
        syph = ss.Disease(S, E, EL, LL,
            infection = S >> E @ 0.8*(E + EL + LL),
            latency   = E >> EL @ ss.months(3),
            to_late   = EL >> LL @ ss.years(2),
            init_prev = 0.02, init=dict(EL=0.01, LL=0.01) if ll else dict(EL=0.01), name='syph')
        if form == 'old': # The workaround: relies on ti_EL being kept after people leave EL for LL, and is NaN for seeds in LL
            def latent_decay(sim):
                s = sim.diseases.syph
                return np.exp(-np.log(2)/hl*(sim.ti - 1 - s.ti_EL.floats().raw)*sim.t.dt.years) # Step numbers as floats, NaN if never in EL; from the start of the step
            amount = latent_decay
        else:
            amount = 0.5**((EL | LL).time/ss.years(hl))
        decay = ss.modify(syph.rel_trans, multiply=amount, where=EL | LL, name='latent_decay')
        return ss.Sim(syph, connectors=decay, n_agents=5000, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw)
    old, new = [make(form, ll=False).run().results.syph for form in ['old', 'new']] # With no seeds in LL, the workaround is right, and the same
    assert np.array_equal(old.infection, new.infection) and old.infection.sum() > 100
    sim = make('new').init()
    assert sim.diseases.syph.rel_trans.effective[sim.diseases.syph.LL.uids].min() == 1 # Seeds in LL start their latent clock at the start
    sim.run()
    s = sim.diseases.syph
    eff = s.rel_trans.effective[s.LL.uids]
    assert not np.isnan(eff).any() and eff.min() < 0.3
    late = s.LL.uids[s.ti_EL[s.LL.uids] > 0]
    latent = s.graph.states.EL | s.graph.states.LL
    assert np.allclose(s.engine.eval_for(latent.time, late), (s.ti - 1 - s.ti_EL[late])/12) # From latent onset to the start of the last step, as the workaround meant
    with pytest.raises(ss.errors.ClockOutsideDomain): # The workaround with seeds in LL is NaN for them
        make('old').run()
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
        months = (sim.ti - 1 - bv.ti_P[uids]) * sim.t.dt.months # From the start of the step
        return ss.permonth(0.06 * np.exp(-months / 3))
    class Old(BV):
        recur = ss.Transition(BV.P, BV.CST4, rate=recur_rate)

    kw2 = dict(n_agents=10_000, start=2000, stop=2005, dt=ss.months(1), unit='year', **kw)
    new = ss.Sim(BV(name='bv'), **kw2).run().results.bv
    old = ss.Sim(Old(name='bv'), **kw2).run().results.bv
    assert new.recur.sum() > 500
    assert np.array_equal(new.recur, old.recur) and np.array_equal(new.n_P, old.n_P) # The same hazards, so the same draws
    return new


@sc.timer()
def test_entry_hazard():
    sc.heading('As agents, people entering a state in the state-change row face half a step of its hazards then')
    kw2 = dict(n_agents=50_000, start=2000, stop=2005, unit='year', **kw)
    weekly, monthly = [ss.Sim(BV(name='bv'), dt=dt, **kw2).run().results.bv.recur.sum() for dt in [ss.weeks(1), ss.months(1)]]
    assert np.isclose(weekly, monthly, rtol=0.03), f'weekly {weekly:n}, monthly {monthly:n}' # Without the hazard on entry, monthly steps were about 20% lower
    return


if __name__ == '__main__':
    sc.options(interactive=False)
    test_events()
    test_clocks()
    test_syphilis_latent()
    test_bv_recurrence()
    test_entry_hazard()
