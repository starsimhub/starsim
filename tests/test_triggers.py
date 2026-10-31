"""
Test triggered moves (ss.on), the rest staying (ss.stay), lazy handles (ss.people, ss.state, ss.any), and distributions by group
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


@sc.timer()
def test_lazy():
    sc.heading('Testing lazy conditions on people, ss.state, and ss.any')
    def make(cond):
        S, I = ss.states('S I')
        d = ss.Disease(S, I, onset=ss.Transition(S, I, rate=ss.peryear(0.5), where=cond), name='d')
        return ss.Sim(d, n_agents=2000, start=2000, stop=2005, unit='year', **kw).run().diseases.d
    new = make(ss.ppl.female & (ss.ppl.age >= 15)) # Before there are any people
    assert new.I.count() > 100
    ppl = new.sim.people
    assert np.all(ppl.female[new.I.uids]) and np.all(ppl.age[new.I.uids] >= 15)

    # A state by path, for a module made before its siblings exist
    def pair(where):
        S, I = ss.states('S I')
        a = ss.Disease(sa=ss.Transition(S, I, rate=ss.peryear(0.3)), name='a')
        S, I = ss.states('S I')
        b = ss.Disease(sb=ss.Transition(S, I, rate=ss.peryear(1.0), where=where), name='b')
        return ss.Sim([a, b], n_agents=2000, start=2000, stop=2005, unit='year', **kw).run()
    sim = pair(ss.state('a.I'))
    a, b = sim.diseases.a, sim.diseases.b
    assert np.all(a.I[b.I.uids]) and b.I.count() > 100
    with pytest.raises(ValueError):
        ss.state('X')
    return sim


def check_cancel(sim, names=('hpv16', 'hpv18')):
    """ An analyzer: nobody progresses in one genotype beside cancer of the other """
    d = sim.diseases
    for g,o in [names, names[::-1]]:
        assert not np.any(d[o].X.raw & (d[g].S.raw | d[g].P.raw | d[g].C.raw)), f'step {sim.ti}: {g} progresses beside {o} cancer'


@sc.timer()
def test_hpv_cancel():
    sc.heading('Port: HPV, canceling the other genotypes when one reaches cancer, with ss.on(), as agents and as compartments')
    names = ['hpv16', 'hpv18']
    def make(g):
        S, P, C, X, Z = ss.states('S P C X Z')
        [other] = [ss.state(f'{o}.X') for o in names if o != g] # The sibling does not exist yet
        return ss.Disease(S, P, C, X, Z,
            infection = S >> P @ ss.peryear(0.3),
            precin    = P >> C @ ss.years(2),
            clear     = C >> S @ ss.years(4),
            cancer    = C >> X @ ss.years(4),
            canceled = (S + P + C) >> Z @ ss.on(other), # Was rate=ss.peryear(1e3), where=other_cancer
            name = g)
    sims = [ss.Sim([make(g) for g in order], analyzers=check_cancel, n_agents=5000, start=2000, stop=2020, unit='year', **kw).run() for order in [names, names[::-1]]]
    for sim in sims:
        assert sim.results.hpv16.canceled.sum() > 50 and sim.results.hpv18.canceled.sum() > 50

    # As compartments, tracked jointly; arrivals by the other's move are canceled at the next loop time
    ode = ss.Sim([make(g) for g in names], solver='ode', total_pop=5000, start=2000, stop=2020, unit='year', **kw).run()
    assert ode.crossings and np.isclose(ode.results.hpv16.canceled.sum(), sims[0].results.hpv16.canceled.sum(), rtol=0.25)
    return ode


@sc.timer()
def test_syphilis_stay():
    sc.heading('Port: syphilis tertiary, with the rest staying, and ss.stay')
    def make(form):
        LL, T, Sp = ss.states('LL T Sp')
        dur = ss.normal(ss.years(2), ss.years(0.2))
        if form == 'implicit': # Was a fake stay_LL = ss.Transition(LL, T, p=0.65, rate=ss.peryear(0))
            tertiary = ss.Transition(LL, T, p=0.35, dur=dur)
        else:
            tertiary = ss.Transition(LL, [T, ss.stay], p=[0.35, None], dur=dur)
        return ss.Disease(LL, T, Sp, tertiary=tertiary, cure=(LL + T) >> Sp, first='LL', name='syph') # Everyone starts in LL; cure is undriven, so not the remainder
    out = sc.obj()
    with pytest.raises(ValueError, match='ss.stay'): # The rest stay only if asked to, with ss.stay
        make('implicit')
    for solver in ['abm', 'ode']:
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        sim = ss.Sim(make('explicit'), solver=solver, start=2000, stop=2006, unit='year', **size, **kw).run()
        res = sim.results.syph
        assert np.isclose(res.T[-1]/1e4, 0.35, atol=0.02), f'{solver}: {res.T[-1]/1e4:.3f} in T'
        assert np.isclose(res.tertiary.sum(), res.T[-1], rtol=1e-6) # Staying is not a flow
        out[f'explicit_{solver}'] = sim
    assert 'states.LL.stay' not in out.explicit_abm.diseases.syph.graph.guessed
    s = out.explicit_abm.diseases.syph # Those who stayed: no new dwell time, and LL.time keeps running
    u = s.LL.uids
    assert np.all(s.ti_LL[u] == 0) and np.all(np.isinf(s.ti_exit_LL[u]))

    # The remainder of branches whose p= is not a constant
    def branches(p):
        I, H, R = ss.states('I H R')
        return ss.Disease(I, H, R, severe=ss.Transition(I, H, p=p, dur=ss.years(1)), mild=ss.Transition(I, R, dur=ss.years(1)), first='I', name='d') # Everyone starts in I
    for solver in ['abm', 'ode']:
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        ns = [ss.Sim(branches(p), solver=solver, start=2000, stop=2010, unit='year', **size, **kw).run().results.d.H[-1] for p in [0.3, lambda sim: 0.3]]
        assert np.isclose(ns[1]/1e4, 0.3, atol=0.02) and np.isclose(ns[0], ns[1], rtol=0.05), f'{solver}: {ns}'
    with pytest.raises(ValueError, match='has ss.stay as its only destination'):
        S, A = ss.states('S A')
        ss.Disease(S, A, x=ss.Transition(S, ss.stay, p=0.5))
    return out


@sc.timer()
def test_ct_by_sex():
    sc.heading('Port: CT with sex-specific durations in one transition, Dists with ss.by')
    ms = [[1, 0.5], [6, 1]] # Months: [mean, std] for women, then men
    def make(form='pars'):
        E, P, A, Y, S = ss.states('E P A Y S')
        if form == 'pars': # A Dist whose parameters are values by group
            onset = ss.lognorm_ex(mean=ss.by(sex=dict(female=ss.months(ms[0][0]), male=ss.months(ms[1][0]))), std=ss.by(sex=dict(female=ss.months(ms[0][1]), male=ss.months(ms[1][1]))))
        else: # A value by group that is a Dist: the same
            onset = ss.by(sex={sex:ss.lognorm_ex(ss.months(m), ss.months(sd)) for sex,(m,sd) in zip(['female', 'male'], ms)})
        p_symp = ss.by(sex=dict(female=0.2, male=0.54))
        return ss.Disease(E, P, A, Y, S,
            symp    = ss.Transition(E, P, p=p_symp), # Was one transition per sex, with p=by_sex(...) and sync_sex()
            asymp   = ss.Transition(E, A, p=1 - p_symp),
            onset   = P >> Y @ onset,
            clear_a = A >> S @ ss.months(6),
            clear_y = Y >> S @ ss.months(3),
            first='E', name='ct') # Everyone starts in E
    kw2 = dict(start=2000, stop=2002, dt=ss.months(0.25), unit='year', **kw)
    sims = sc.obj()
    for key,form,strat in [('nostrat', 'pars', None), ('pars', 'pars', dict(sex=['female', 'male'])), ('dists', 'dists', dict(sex=['female', 'male']))]:
        sims[key] = ss.Sim(make(form), n_agents=10_000, stratify=strat, **kw2).run() # As agents, sex needs no stratification
    ct, ppl = sims.nostrat.diseases.ct, sims.nostrat.people
    for f,(mean,_) in zip([True, False], ms):
        went = ppl.auids[~np.isnan(ct.ti_Y[ppl.auids])] # Everyone who went through P to Y
        u = went[ppl.female[went] == f]
        dwell = (ct.ti_Y[u] - ct.ti_P[u] - 1)*0.25 # P was entered at the start (index 0), and Y in the step the dwell ended before, at its start
        assert np.isclose(dwell.mean(), mean, rtol=0.15), f'female={f}: mean dwell {dwell.mean():.2f} months, not {mean}'
    n_P = sims.nostrat.results.ct.P[0]
    assert np.isclose(n_P, 10_000*(0.5*0.2 + 0.5*0.54), rtol=0.05)
    assert np.array_equal(sims.pars.results.ct.P, sims.dists.results.ct.P) # The two forms are one distribution

    # As compartments: each sex has its own Erlang stages (ragged), keeping its mean
    ode = ss.Sim(make(), solver='ode', total_pop=10_000, stratify=dict(sex=['female', 'male']), **kw2).run()
    r, a = ode.results.ct, sims.pars.results.ct
    for sex in ['female', 'male']:
        assert np.allclose(r.P.select(sex=sex)[::4], a.P.select(sex=sex)[::4], atol=150), sex
    assert any('4|36 Erlang stages' in note for note in ode.notes)
    with pytest.raises(ss.CapabilityError, match='stratify by it'): # As compartments, sex needs a stratification
        ss.Sim(make(), solver='ode', **kw2).init()

    # A Dist with a function as a parameter (FPsim: duration of use by age), and typed rates inside ss.by
    def callable_dist(solver):
        A, B = ss.states('A B')
        d = ss.Disease(A, B, ab=ss.Transition(A, B, dur=ss.lognorm_ex(mean=lambda self, sim, uids: np.full(len(uids), 0.5), std=0.1)), first='A', name='d') # Everyone starts in A
        size = dict(n_agents=1000) if solver == 'abm' else dict(total_pop=1000)
        return ss.Sim(d, solver=solver, **size, start=2000, stop=2002, dt=ss.months(1), unit='year', verbose=0).run()
    assert callable_dist('abm').results.d.ab.sum() == 1000
    with pytest.raises(ss.CapabilityError):
        callable_dist('ode')
    assert ss.permonth(ss.by(age=[1, 2])).values == [ss.permonth(1), ss.permonth(2)]
    tab = ss.by(ss.permonth(np.ones((2, 2))), dims=['age', 'parity'])
    assert tab.wrap is ss.permonth
    return sims


@sc.timer()
def test_with_conveyor():
    sc.heading('ss.stay and ss.on out of a fixed dwell (a conveyor), in a crossing made by a trigger; a clock there needs agents')
    size = lambda solver: dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    for solver in ['abm', 'ode']: # Those who stay at the end of a fixed dwell are not in the flow
        LL, T, Sp = ss.states('LL T Sp')
        d = ss.Disease(LL, T, Sp, tertiary=ss.Transition(LL, [T, ss.stay], p=[0.35, None], dur=ss.constant(ss.years(2))), cure=(LL + T) >> Sp, first='LL', name='syph') # Everyone (1 or more is people)
        res = ss.Sim(d, solver=solver, start=2000, stop=2004, unit='year', **size(solver), **kw).run().results.syph
        assert res.T[1] == 0 and np.isclose(res.T[-1]/1e4, 0.35, atol=0.02), f'{solver}: {res.T}'
        assert np.isclose(res.tertiary.sum(), res.T[-1], rtol=1e-6), f'{solver}: {res.tertiary.sum()} in the flow, {res.T[-1]} in T'

    def make(clock):
        S, I = ss.states('S I')
        a = ss.Disease(sa=ss.Transition(S, I, rate=ss.peryear(0.5)), name='a')
        E, I, Z = ss.states('E I Z')
        latent = E >> I @ (ss.peryear(1)*(E.time > ss.years(1))) if clock else ss.Transition(E, I, dur=ss.constant(ss.years(2)))
        return [a, ss.Disease(E, I, Z, latent=latent, flag=ss.Transition(E, Z, where=a.I), first='E', name='b')] # The trigger tracks a and b jointly
    out = [ss.Sim(make(False), solver=solver, start=2000, stop=2004, dt=ss.months(1), unit='year', **size(solver), **kw).run().results.b for solver in ['abm', 'ode']]
    for key in ['I', 'Z']:
        assert np.allclose(out[0][key], out[1][key], atol=300), f'{key}: {out[0][key][::12]} as agents, {out[1][key][::12]} as compartments'
    with pytest.raises(ss.CapabilityError, match='E.time is a clock'):
        ss.Sim(make(True), solver='ode', start=2000, stop=2004, dt=ss.months(1), unit='year', **size('ode'), **kw).init()
    return


@sc.timer()
def test_after_event():
    sc.heading('Testing an exact dwell until an event of another module, and a time after it')
    A, B = ss.states('A B')
    def make(**kw):
        a = ss.Disease(A, B, go=ss.Transition(A, B, dur=ss.uniform(ss.days(5), ss.days(15))), first='A', name='a')
        X, Y = ss.states('X Y')
        return [a, ss.Disease(X, Y, end=ss.Transition(X, Y, where=a.go, **kw), first='X', name='b')]
    for delay,lag in [(None, 0), (ss.constant(ss.days(5)), 5)]:
        res = ss.Sim(diseases=make(**({} if delay is None else dict(dur=delay))), n_agents=2000, unit='day', stop=30, verbose=0).run().results
        assert np.array_equal(res.b.end.values[lag:], res.a.go.values[:len(res.a.go) - lag]) # Each person leaves X when a.go happens to them, or exactly 5 days after
    with pytest.warns(RuntimeWarning, match='ss.constant'): # A function as dur= is the mean of an exponential
        ss.Sim(ss.Disease(A, B, go=ss.Transition(A, B, dur=lambda sim: 10.0), first='A', name='a'), n_agents=100, verbose=0).init()
    return


@sc.timer()
def test_stay_keeps_course():
    sc.heading('Testing that the rest of a split on an event (ss.stay) keep their course, and that two times after an event in the same step both happen')
    common = dict(n_agents=2000, start=2000, stop=2004, dt=ss.months(1), **kw)

    # A share move to P a month after infection, and the rest stay in A, where they still clear 6 months after infection
    S, A, P = ss.states('S A P')
    infection = S >> A @ ss.peryear(1.0)
    clearance = A >> S @ ss.constant(ss.months(6))
    progress = ss.Transition(A, [P, ss.stay], p=[0.0, None], where=infection, dur=ss.constant(ss.months(1)))
    res = ss.Sim(ss.Disease(infection, clearance, progress, name='d'), **common).run().results.d
    assert np.array_equal(res.clearance.values[6:], res.infection.values[:-6]) # Everyone clears exactly 6 months after infection

    # A share seek care 3 months after symptoms and stay symptomatic; everyone clears 3 months after symptoms, in the same step or, if seeking care then, the next
    S, P, Y = ss.states('S P Y')
    infection = S >> P @ ss.peryear(1.0)
    symptoms = P >> Y @ ss.constant(ss.months(1))
    seek = ss.Transition(Y, [Y, ss.stay], p=[0.5, None], where=symptoms, dur=ss.constant(ss.months(3)))
    clearance = ss.Transition(Y, S, where=symptoms, dur=ss.constant(ss.months(3)))
    sought = []
    count = lambda sim: sought.append(len(sim.diseases.d.seek.uids)) # Its event: those who sought care, not those who stayed
    res = ss.Sim(ss.Disease(infection, symptoms, seek, clearance, name='d'), analyzers=count, **common).run().results.d
    assert res.Y[-1] <= res.symptoms.values[-4:].sum() # Nobody is left symptomatic for good: only those whose symptoms began in the last 4 months
    assert sum(sought) == res.seek.sum() > 0
    return


def treated_event(solver):
    S, I = ss.states('S I')
    a = ss.Disease(inf=ss.Transition(S, I, rate=ss.peryear(0.5)), cure=ss.Transition(I, S), name='a')
    S, F = ss.states('S F')
    b = ss.Disease(S, F, flag=S >> F @ ss.on(a.cure), unflag=F >> S @ ss.years(0.5), name='b')
    tx = ss.treat(ss.Treatment(moves=a.cure), where=a.I, coverage=ss.peryear(1.0), name="tx")
    size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
    return ss.Sim([a, b], interventions=tx, solver=solver, start=2000, stop=2005, dt=ss.months(1), unit='year', **size, **kw).run().results


def test_treated_event():
    sc.heading('A move by a treatment is the event of its transition, for ss.on() in another module, as agents and as compartments')
    r, ro = treated_event('abm'), treated_event('ode')
    assert r.b.flag.sum() > 1000 and r.b.flag.sum() <= r.a.cure.sum()
    assert np.isclose(r.b.flag.sum(), ro.b.flag.sum(), rtol=0.1)
    return r


def make_hpv_cancel(order, names=('hpv16', 'hpv18')):
    states = {g:ss.states('S P C X Z') for g in names}
    cin = {g:ss.Transition(st[2], [st[3], st[0]], p=[0.4, None], dur=ss.years(3)) for g,st in states.items()} # To cancer, or cleared
    mods = []
    for g in order:
        o = next(n for n in names if n != g)
        S, P, C, X, Z = states[g]
        mods.append(ss.Disease(S, P, C, X, Z,
            infection = S >> P @ ss.peryear(0.3),
            precin    = P >> C @ ss.years(2),
            cin       = cin[g],
            canceled = ss.Transition([S, P, C], Z, where=cin[o].to(states[o][3])), # The other genotype's branch to cancer
            name = g))
    return mods


def test_hpv_cancel_event():
    sc.heading('HPV: canceling the other genotypes on the event of one reaching cancer, ss.on(other.cin.to(X)), as agents and as compartments')
    names = ['hpv16', 'hpv18']
    common = dict(start=2000, stop=2020, unit='year', **kw)
    abm = ss.Sim(make_hpv_cancel(names), analyzers=check_cancel, n_agents=20_000, **common).run()
    assert all(abm.results[g].canceled.sum() > 0 for g in names)
    odes = [ss.Sim(make_hpv_cancel(order), solver='ode', total_pop=20_000, **common).run() for order in [names, names[::-1]]]
    for ode in odes:
        for g in names:
            a, o = abm.results[g], ode.results[g]
            assert np.isclose(o.canceled.sum(), a.canceled.sum(), rtol=0.05), f'{g}: {o.canceled.sum():.0f} canceled as an ODE, {a.canceled.sum():.0f} as agents'
            other = ode.results[next(n for n in names if n != g)]
            assert np.isclose(o.Z[-1], other.X[-1], rtol=1e-6)
    return abm


if __name__ == '__main__':
    sc.options(interactive=False)
    test_lazy()
    test_hpv_cancel()
    test_syphilis_stay()
    test_ct_by_sex()
    test_with_conveyor()
    test_after_event()
    test_stay_keeps_course()
    test_treated_event()
    test_hpv_cancel_event()
