"""
Test triggered moves (ss.on), the rest staying (ss.stay), lazy handles (ss.person, ss.ref), and distributions by group
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


@sc.timer()
def test_lazy():
    sc.heading('Testing lazy conditions on people and ss.ref')
    def make(cond):
        S, I = ss.states('S I')
        d = ss.Disease(S, I, onset=ss.Transition(S, I, rate=ss.peryear(0.5), who=cond), name='d')
        return ss.Sim(d, n_agents=2000, start=2000, stop=2005, unit='year', **kw).run().diseases.d
    new = make(ss.person.female & (ss.person.age >= 15)) # Before there are any people
    assert new.I.count() > 100
    ppl = new.sim.people
    assert np.all(ppl.female[new.I.uids]) and np.all(ppl.age[new.I.uids] >= 15)

    # A state by path, for a module made before its siblings exist
    def pair(where):
        S, I = ss.states('S I')
        a = ss.Disease(sa=ss.Transition(S, I, rate=ss.peryear(0.3)), name='a')
        S, I = ss.states('S I')
        b = ss.Disease(sb=ss.Transition(S, I, rate=ss.peryear(1.0), who=where), name='b')
        return ss.Sim([a, b], n_agents=2000, start=2000, stop=2005, unit='year', **kw).run()
    sim = pair(ss.ref('a.I'))
    a, b = sim.diseases.a, sim.diseases.b
    assert np.all(a.I[b.I.uids]) and b.I.count() > 100
    return sim


def check_cancel(sim, names=('hpv16', 'hpv18')):
    """ An analyzer: nobody progresses in one genotype beside cancer of the other """
    d = sim.diseases
    for g,o in [names, names[::-1]]:
        assert not np.any(d[o].X.raw & (d[g].S.raw | d[g].P.raw | d[g].C.raw)), f'step {sim.ti}: {g} progresses beside {o} cancer'


@sc.timer()
def test_syphilis_stay():
    sc.heading('Testing syphilis tertiary, with the rest staying, and ss.stay')
    def make(dur, form='explicit'):
        LL, T, Sp = ss.states('LL T Sp')
        if form == 'implicit': # Was a fake stay_LL = ss.Transition(LL, T, p=0.65, rate=ss.peryear(0))
            tertiary = ss.Transition(LL, T, p=0.35, dur=dur)
        else:
            tertiary = ss.Transition(LL, {T: 0.35, ss.stay: None}, dur=dur)
        return ss.Disease(LL, T, Sp, tertiary=tertiary, cure=(LL + T) >> Sp @ 0, default_state='LL', name='syph') # Everyone starts in LL; cure has a rate of 0, moved only by a treatment, so is not the remainder
    with pytest.raises(ValueError): # The rest stay only if asked to, with ss.stay
        make(ss.years(2), 'implicit')
    for dur in [ss.normal(ss.years(2), ss.years(0.2)), ss.constant(ss.years(2))]: # The second is a fixed dwell (a conveyor) as compartments
        for solver in ['abm', 'ode']:
            size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
            sim = ss.Sim(make(dur), solver=solver, start=2000, stop=2006, unit='year', **size, **kw).run()
            res = sim.results.syph
            assert np.isclose(res.T[-1]/1e4, 0.35, atol=0.02), f'{solver}: {res.T[-1]/1e4:.3f} in T'
            assert np.isclose(res.tertiary.sum(), res.T[-1], rtol=1e-6) # Staying is not a flow
            if isinstance(dur, ss.constant):
                assert res.T[1] == 0 # Nobody before the fixed dwell ends
            elif solver == 'abm':
                s = sim.diseases.syph # Those who stayed: no new dwell time, and LL.time_in keeps running
                assert 'states.LL.stay' not in s.graph.guessed
                u = s.LL.uids
                assert np.all(s.ti_LL[u] == -1) and np.all(np.isinf(s.ti_exit_LL[u])) # Placed at the start, dated by the step before the first
    return


@sc.timer()
def test_ct_by_sex():
    sc.heading('Testing CT with sex-specific durations in one transition, Dists with ss.by')
    ms = [[1, 0.5], [6, 1]] # Months: [mean, std] for women, then men
    def make(form='pars'):
        E, P, A, Y, S = ss.states('E P A Y S')
        if form == 'pars': # A Dist whose parameters are values by group
            onset = ss.lognormal(mean=ss.by(sex=dict(female=ss.months(ms[0][0]), male=ss.months(ms[1][0]))), std=ss.by(sex=dict(female=ss.months(ms[0][1]), male=ss.months(ms[1][1]))))
        else: # A value by group that is a Dist: the same
            onset = ss.by(sex={sex:ss.lognormal(ss.months(m), ss.months(sd)) for sex,(m,sd) in zip(['female', 'male'], ms)})
        p_symp = ss.by(sex=dict(female=0.2, male=0.54))
        return ss.Disease(E, P, A, Y, S,
            symp    = ss.Transition(E, P, p=p_symp), # Was one transition per sex, with p=by_sex(...) and sync_sex()
            asymp   = ss.Transition(E, A, p=1 - p_symp),
            onset   = P >> Y @ onset,
            clear_a = A >> S @ ss.months(6),
            clear_y = Y >> S @ ss.months(3),
            default_state='E', name='ct') # Everyone starts in E
    kw2 = dict(start=2000, stop=2002, dt=ss.months(0.25), unit='year', **kw)
    strat = dict(sex=['female', 'male'])
    sims = sc.obj()
    for form in ['pars', 'dists']:
        sims[form] = ss.Sim(make(form), n_agents=10_000, stratify=strat, **kw2).run()
    ct, ppl = sims.pars.diseases.ct, sims.pars.people
    for f,(mean,_) in zip([True, False], ms):
        went = ppl.auids[~np.isnan(ct.ti_Y[ppl.auids])] # Everyone who went through P to Y
        u = went[ppl.female[went] == f]
        dwell = (ct.ti_Y[u] - ct.ti_P[u] - 1)*0.25 # P was entered at the start (index 0), and Y in the step the dwell ended before, at its start
        assert np.isclose(dwell.mean(), mean, rtol=0.15), f'female={f}: mean dwell {dwell.mean():.2f} months, not {mean}'
    n_P = sims.pars.results.ct.P[0]
    assert np.isclose(n_P, 10_000*(0.5*0.2 + 0.5*0.54), rtol=0.05)
    assert np.array_equal(sims.pars.results.ct.P, sims.dists.results.ct.P) # The two forms are one distribution

    # As compartments: each sex has its own Erlang stages (ragged), keeping its mean
    ode = ss.Sim(make(), solver='ode', total_pop=10_000, stratify=strat, **kw2).run()
    r, a = ode.results.ct, sims.pars.results.ct
    for sex in ['female', 'male']:
        assert np.allclose(r.P.select(sex=sex)[::4], a.P.select(sex=sex)[::4], atol=150), sex
    return sims


@sc.timer()
def test_with_conveyor():
    sc.heading('ss.on out of a fixed dwell (a conveyor), in joint states made by a trigger; a clock there needs agents')
    size = lambda solver: dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    def make(clock):
        S, I = ss.states('S I')
        a = ss.Disease(sa=ss.Transition(S, I, rate=ss.peryear(0.5)), name='a')
        E, I, Z = ss.states('E I Z')
        latent = E >> I @ (ss.peryear(1)*(E.time_in > ss.years(1))) if clock else ss.Transition(E, I, dur=ss.constant(ss.years(2)))
        return [a, ss.Disease(E, I, Z, latent=latent, flag=ss.Transition(E, Z, when=a.I), default_state='E', name='b')] # The trigger tracks a and b jointly
    out = [ss.Sim(make(False), solver=solver, start=2000, stop=2004, dt=ss.months(1), unit='year', **size(solver), **kw).run().results.b for solver in ['abm', 'ode']]
    for key in ['I', 'Z']:
        assert np.allclose(out[0][key], out[1][key], atol=300), f'{key}: {out[0][key][::12]} as agents, {out[1][key][::12]} as compartments'
    with pytest.raises(ss.CapabilityError):
        ss.Sim(make(True), solver='ode', start=2000, stop=2004, dt=ss.months(1), unit='year', **size('ode'), **kw).init()
    return


@sc.timer()
def test_after_event():
    sc.heading('Testing an exact dwell until an event of another module, and a time after it')
    A, B = ss.states('A B')
    def make(delay=None):
        a = ss.Disease(A, B, go=ss.Transition(A, B, dur=ss.uniform(ss.days(5), ss.days(15))), default_state='A', name='a')
        X, Y = ss.states('X Y')
        return [a, ss.Disease(X, Y, end=ss.Transition(X, Y, when=ss.on(a.go, delay=delay)), default_state='X', name='b')]
    for delay,lag in [(None, 0), (ss.constant(ss.days(5)), 5)]:
        res = ss.Sim(diseases=make(delay), n_agents=2000, unit='day', stop=30, verbose=0).run().results
        assert np.array_equal(res.b.end.values[lag:], res.a.go.values[:len(res.a.go) - lag]) # Each person leaves X when a.go happens to them, or exactly 5 days after
    return


@sc.timer()
def test_stay_keeps_course():
    sc.heading('Testing that the rest of a split on an event (ss.stay) keep their course, and that two times after an event in the same step both happen')
    common = dict(n_agents=2000, start=2000, stop=2004, dt=ss.months(1), **kw)

    # A share move to P a month after infection, and the rest stay in A, where they still clear 6 months after infection
    S, A, P = ss.states('S A P')
    infection = S >> A @ ss.peryear(1.0)
    clearance = A >> S @ ss.constant(ss.months(6))
    progress = ss.Transition(A, {P: 0.0, ss.stay: None}, when=ss.on(infection, delay=ss.constant(ss.months(1))))
    res = ss.Sim(ss.Disease(infection, clearance, progress, name='d'), **common).run().results.d
    assert np.array_equal(res.clearance.values[6:], res.infection.values[:-6]) # Everyone clears exactly 6 months after infection

    # A share seek care 3 months after symptoms and stay symptomatic; everyone clears 3 months after symptoms, in the same step or, if seeking care then, the next
    S, P, Y = ss.states('S P Y')
    infection = S >> P @ ss.peryear(1.0)
    symptoms = P >> Y @ ss.constant(ss.months(1))
    seek = ss.Transition(Y, {Y: 0.5, ss.stay: None}, when=ss.on(symptoms, delay=ss.constant(ss.months(3))))
    clearance = ss.Transition(Y, S, when=ss.on(symptoms, delay=ss.constant(ss.months(3))))
    sought = []
    count = lambda sim: sought.append(len(sim.diseases.d.seek.uids)) # Its event: those who sought care, not those who stayed
    res = ss.Sim(ss.Disease(infection, symptoms, seek, clearance, name='d'), analyzers=count, **common).run().results.d
    assert res.Y[-1] <= res.symptoms.values[-4:].sum() # Nobody is left symptomatic for good: only those whose symptoms began in the last 4 months
    assert sum(sought) == res.seek.sum() > 0
    return


def test_treated_event():
    sc.heading('A move by a treatment is the event of its transition, for ss.on() in another module, as agents and as compartments')
    out = {}
    for solver in ['abm', 'ode']:
        S, I = ss.states('S I')
        a = ss.Disease(inf=ss.Transition(S, I, rate=ss.peryear(0.5)), cure=ss.Transition(I, S, rate=0), name='a')
        S, F = ss.states('S F')
        b = ss.Disease(S, F, flag=S >> F @ ss.on(a.cure), unflag=F >> S @ ss.years(0.5), name='b')
        tx = ss.treat(ss.Treatment(cure={a.cure: 1.0}), who=a.I, rate=ss.peryear(1.0), name="tx")
        size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        out[solver] = ss.Sim([a, b], interventions=tx, solver=solver, start=2000, stop=2005, dt=ss.months(1), unit='year', **size, **kw).run().results
    r, ro = out['abm'], out['ode']
    assert r.b.flag.sum() > 1000 and r.b.flag.sum() <= r.a.cure.sum()
    assert np.isclose(r.b.flag.sum(), ro.b.flag.sum(), rtol=0.1)
    return


def make_hpv_cancel(order, names=('hpv16', 'hpv18')):
    states = {g:ss.states('S P C X Z') for g in names}
    cin = {g:ss.Transition(st[2], {st[3]: 0.4, st[0]: None}, dur=ss.years(3)) for g,st in states.items()} # To cancer, or cleared
    mods = []
    for g in order:
        o = next(n for n in names if n != g)
        S, P, C, X, Z = states[g]
        mods.append(ss.Disease(S, P, C, X, Z,
            infection = S >> P @ ss.peryear(0.3),
            precin    = P >> C @ ss.years(2),
            cin       = cin[g],
            canceled = ss.Transition([S, P, C], Z, when=cin[o].to(states[o][3])), # The other genotype's branch to cancer
            name = g))
    return mods


def test_hpv_cancel_event():
    sc.heading('HPV: canceling the other genotypes on the event of one reaching cancer, ss.on(other.cin.to(X)), as agents and as compartments')
    names = ['hpv16', 'hpv18']
    common = dict(start=2000, stop=2020, unit='year', **kw)
    abm = ss.Sim(make_hpv_cancel(names), analyzers=check_cancel, n_agents=20_000, **common).run()
    assert all(abm.results[g].canceled.sum() > 0 for g in names)
    ode = ss.Sim(make_hpv_cancel(names[::-1]), solver='ode', total_pop=20_000, **common).run() # In the other module order
    assert ode.joint_states
    for g in names:
        a, o = abm.results[g], ode.results[g]
        assert np.isclose(o.canceled.sum(), a.canceled.sum(), rtol=0.05), f'{g}: {o.canceled.sum():.0f} canceled as an ODE, {a.canceled.sum():.0f} as agents'
        other = ode.results[next(n for n in names if n != g)]
        assert np.isclose(o.Z[-1], other.X[-1], rtol=1e-6)
    return abm


@sc.timer()
def test_paths_and_memory():
    sc.heading('Testing that a path means whatever is at it, in who= and when=, and the memory of every event')
    def sir():
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ ss.perday(0.1)*I, recovery=ss.Transition(I, R, dur=ss.days(10)), waning=R >> S @ ss.perday(0.05), init_prev=0.05, name='sir')
    pars = dict(n_agents=1000, dur=ss.days(60), unit='day', solver='abm', **kw)

    # A path as a string is ss.ref() of it, and a path to a transition is its event
    a, b = [ss.Sim(sir(), interventions=ss.Event(when=when, name='ev'), **pars).run().results.ev.delivered for when in ['sir.infection', ss.ref('sir.infection')]]
    assert a.sum() > 0 and np.array_equal(a, b)
    def other(**kwargs):
        X, Y = ss.states('X Y')
        return ss.Disease(X, Y, go=ss.Transition(X, Y, **kwargs), default_state='X', name='other')
    res = ss.Sim([sir(), other(when=ss.ref('sir.infection'))], **pars).run().results
    assert res.other.go.sum() > 0 and np.all(res.other.go <= res.sir.infection) # Each person infected leaves X then (once, though they may be infected again): an event of another module
    res = ss.Sim([sir(), other(when='sir.I')], **pars).run().results
    assert res.other.go[0] > 0 # A state is a condition: those already in it move at once
    assert ss.Sim([sir(), other(rate=ss.perday(0.5), who='sir.I')], **pars).run().results.other.go.sum() > 0
    with pytest.raises(TypeError): # An event says when, not who
        ss.Sim([sir(), other(rate=ss.perday(0.5), who='sir.infection')], **pars).init()

    # Every event remembers: entering or leaving a state, and a branch of a transition, as a transition does
    def delivered(who):
        d = sir()
        return ss.Sim(d, interventions=ss.Event(d.R >> d.S, who=who(d), when=ss.days(40), name='ev'), **pars).run().results.ev.delivered.sum()
    assert delivered(lambda d: d.infection.n_times >= 2) == delivered(lambda d: d.I.enter.n_times >= 2) == delivered(lambda d: d.infection.to(d.I).n_times >= 2)
    assert delivered(lambda d: d.recovery.since < 3) == delivered(lambda d: d.I.exit.since < 3) > 0

    return


def test_code_every_solver():
    sc.heading('Code that counts and moves people means the same under every solver: .frac, conditions in code, people.n, a fraction of an agent moved, and ss.move(I >> R)')
    S, I, R = ss.states('S I R')
    out = {}
    for solver in ['ode', 'abm']:
        sir = ss.Disease(S >> I @ 0.0*I, I >> R @ ss.days(10), init_prev=0.2, name='sir')
        sim = ss.Sim(sir, solver=solver, total_pop=100_000, n_agents=10_000, stratify=dict(age=[0, 15, 65]), unit='day', stop=ss.days(10), rand_seed=1, verbose=0).init()
        sir, people = sim.diseases.sir, sim.people
        out[solver] = [sir.I.frac, (sir.I | sir.R).frac, (sir.I & (ss.person.age < 15)).n, people.n]
        moved = sum(ss.move(sir.S, sir.R, n=0.3) for _ in range(500)) # 0.03 agents each time as agents: rounded at random, not to 0
        assert np.isclose(moved, 150, rtol=0.4), moved
        I0 = sir.I.n
        assert np.isclose(ss.move(sir.I >> sir.R, p=0.5), I0/2, rtol=0.2) # A move written with the states, as ss.Event() takes
        with pytest.raises(ValueError, match='not both'):
            ss.move(sir.I >> sir.R, sir.S)
        if solver == 'ode':
            with pytest.raises(ss.CapabilityError):
                people.alive.sum() # Counts agents, which a compartmental run has none of
    assert np.allclose(out['ode'], out['abm'], rtol=0.15), out
    assert out['ode'][-1] == 100_000
    return


def test_move_outside():
    sc.heading('Code adds and removes people under every solver, and every disease has them')
    class Churn(ss.Intervention):
        def step(self):
            a = self.sim.diseases.a
            ss.move(a.I, ss.dead, p=0.02) # Some of those infectious with a die
            ss.move(ss.outside, a.I, n=5) # Some arrive infectious
            ss.move(ss.unborn, self.sim.diseases.b.S, n=3) # Some are born
    out = {}
    for solver in ['ode', 'binomial', 'abm']:
        S, I, R = ss.states('S I R')
        a = ss.Disease(S >> I @ ss.perday(0.3)*I.frac, I >> R @ ss.perday(0.1), init_n=dict(I=200), name='a')
        S, I, R = ss.states('S I R')
        b = ss.Disease(S >> I @ ss.perday(0.2)*I.frac, I >> R @ ss.perday(0.05), init_n=dict(I=200), name='b')
        sim = ss.Sim(diseases=[a, b], interventions=Churn(), solver=solver, total_pop=20_000, n_agents=20_000, unit='day', stop=40, rand_seed=1, verbose=0).run()
        r = sim.results
        na, nb = r.a.S + r.a.I + r.a.R, r.b.S + r.b.I + r.b.R
        assert np.allclose(na, nb), solver # The same people in both, though they are run apart
        out[solver] = na[-1]
    assert np.allclose(list(out.values()), out['ode'], rtol=0.02) and out['ode'] < 20_000, out
    return


def test_set_values():
    sc.heading('Setting the number in states by hand moves people as ss.move() does')
    def by_hand(sim):
        sir = sim.diseases.sir
        moved = sir.S.values*0.01
        sir.S.values -= moved
        sir.R.values += moved
    def by_move(sim):
        ss.move(sim.diseases.sir.S, sim.diseases.sir.R, p=0.01)
    out = []
    for fn in [by_hand, by_move]:
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_n=dict(I=10), name='sir')
        sir.stratify(age={0: 0.3, 15: 0.7})
        sim = ss.Sim(sir, interventions=fn, total_pop=10_000, stop=ss.days(60), verbose=0).run()
        out.append(sim.results.sir.R.values)
    assert np.allclose(out[0], out[1])
    return


@sc.timer()
def test_class_trigger_rebound():
    sc.heading('Testing that a trigger on several events in a class is of each instance\'s transitions')
    class D(ss.Disease):
        S, I, R, V = ss.states('S I R V')
        infection = S >> I @ 0.3*I
        recovery = I >> R @ ss.days(10)
        vax = ss.Transition(R, V, when=recovery | infection)
    d = D(name='d')
    event = ss.conditions.event_leaves(d.vax._trigger.x)[0]
    assert event.obj is d.recovery and event.module is d # Not the class's transition, which has no module

    # A transition not yet in a module has no rate per agent: the error says so
    S, I = ss.states(2)
    with pytest.raises(ss.CapabilityError):
        ss.Transition(S, I, rate=0.1).hazard([0])


if __name__ == '__main__':
    sc.options(interactive=False)
    test_lazy()
    test_syphilis_stay()
    test_ct_by_sex()
    test_with_conveyor()
    test_after_event()
    test_stay_keeps_course()
    test_treated_event()
    test_hpv_cancel_event()
    test_paths_and_memory()
    test_code_every_solver()
    test_move_outside()
    test_set_values()
