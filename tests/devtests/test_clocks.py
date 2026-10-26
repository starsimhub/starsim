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
        st = d.states
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
        target = res.d[flow][:-1].sum() if flow else res.d.D[-2] # Events in a step are acted on by the next, so not those of the last
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
                return np.exp(-np.log(2)/hl*(sim.ti - 1 - s.ti_EL.internal(slice(None)))*sim.t.dt.years) # The engine's own times (NaN if never in EL), from the start of the step, as the clock reads them; ti_EL itself reads as the step of entering, a step later for a move at a step's end
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
    latent = s.states.EL | s.states.LL
    assert np.allclose(s.engine.eval_for(latent.time, late), (s.ti - 1 - s.ti_EL.internal(late))/12) # From latent onset to the start of the last step, as the workaround meant
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
def test_entry_hazard():
    sc.heading('As agents, people entering a state in the state-change row face half a step of its hazards then')
    kw2 = dict(n_agents=50_000, start=2000, stop=2005, unit='year', **kw)
    weekly, monthly = [ss.Sim(BV(name='bv'), dt=dt, **kw2).run().results.bv.recur.sum() for dt in [ss.weeks(1), ss.months(1)]]
    assert np.isclose(weekly, monthly, rtol=0.03), f'weekly {weekly:n}, monthly {monthly:n}' # Without the hazard on entry, monthly steps were about 20% lower
    return


def test_keep_time():
    """ MG2: ss.State(keep_time=True) keeps the time of entering it for code to read; a clock as a class member is a helper, not a result """
    def make(keep):
        class C(ss.Disease):
            S = ss.State()
            I = ss.State(keep_time=keep)
            inf = S >> I @ ss.perday(0.05)
            duration = I.time # Not a result
        return C(init_prev=0)
    sim = ss.Sim(diseases=make(True), n_agents=200, dur=ss.days(20), dt=ss.days(1), **kw).run()
    c = sim.diseases.c
    assert 'duration' not in sim.results.c
    u = c.I.uids
    assert len(u) and np.all(c.ti - c.ti_I[u] >= 0)
    sim = ss.Sim(diseases=make(False), n_agents=200, dur=ss.days(20), dt=ss.days(1), **kw).run()
    with pytest.raises(ss.errors.TimeNotKept, match='keep_time'):
        sim.diseases.c.ti_I.values
    return


if __name__ == '__main__':
    sc.options(interactive=False)
    test_events()
    test_syphilis_latent()
    test_entry_hazard()
    test_keep_time()
