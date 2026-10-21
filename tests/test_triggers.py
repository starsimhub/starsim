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

    # A state by path, for a module made before its siblings exist, and a union of several
    def pair(where):
        a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.3)), name='a')
        b = ss.Disease(sb=ss.progression('S -> I', rate=ss.peryear(1.0), where=where), name='b')
        return ss.Sim([a, b], n_agents=2000, start=2000, stop=2005, unit='year', **kw).run()
    sim = pair(ss.any(ss.state(f'{g}.I') for g in ['a']))
    a, b = sim.diseases.a, sim.diseases.b
    assert np.all(a.I[b.I.uids]) and b.I.count() > 100
    with pytest.raises(ValueError):
        ss.state('X')
    with pytest.raises(ValueError):
        ss.any([])
    return sim


@sc.timer()
def test_bv_debut():
    sc.heading('Port: BV debut as soon as women turn 15, with ss.on()')
    adult_f = ss.ppl.female & (ss.ppl.age >= 15)
    def make():
        U, CST1, CST4 = ss.states('U CST1 CST4')
        return ss.Disease(U, CST1, CST4,
            debut = ss.progression('U -> CST1 | CST4', on=adult_f, p=[0.8, None]), # Was rate=ss.perday(1), where=adult_f
            onset = CST1 >> CST4 @ ss.peryear(0.2),
            clear = CST4 >> CST1 @ ss.months(6),
            name='bv')
    checked = sc.objdict(n=0)
    def check(sim): # After every module has stepped, nobody who qualifies is still waiting
        bv, ppl = sim.diseases.bv, sim.people
        u = bv.U.uids
        assert not np.any(ppl.female[u] & (ppl.age[u] >= 15)), f'step {sim.ti}: women 15+ still in U'
        checked.n += 1
    demog = lambda: [ss.Births(birth_rate=20), ss.Deaths(death_rate=10)]
    sim = ss.Sim(make(), analyzers=check, demographics=demog(), n_agents=5000, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw).run()
    res = sim.results.bv
    assert checked.n >= len(res.timevec) - 1 and res.debut[1:].sum() > 100
    assert res.n_U[0] < 0.7*5000 # Those 15+ at the start debuted before it
    assert 'as soon as it holds' in sim.diseases.bv.describe(output=True)
    assert ss.Sim(make(), stratify=dict(sex=['female', 'male']), n_agents=100, start=2000, stop=2001, unit='year', **kw).init().solver == 'abm' # As agents for now

    # A trigger has no rate to modify
    bv = make()
    with pytest.raises(ss.UnmodifiableTarget):
        ss.Sim(bv, connectors=ss.modify(bv.debut, multiply=2), n_agents=100, **kw).init()
    return sim


def test_bv_debut_ode():
    """ As compartments: the compartments of women 15+ in U are empty at the start, and aging into them is moved on at each step """
    adult_f = ss.ppl.female & (ss.ppl.age >= 15)
    U, CST1, CST4 = ss.states('U CST1 CST4')
    bv = ss.Disease(U, CST1, CST4, debut=ss.progression('U -> CST1 | CST4', on=adult_f, p=[0.8, None]), onset=CST1 >> CST4 @ ss.peryear(0.2), clear=CST4 >> CST1 @ ss.months(6), name='bv')
    strat = [dict(sex=['female', 'male']), dict(age=[0, 10, 15, 50, 100])]
    ode = ss.Sim(bv, solver='ode', total_pop=10_000, stratify=strat, use_aging=True, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw).run() # v4 ages people only with births or deaths, unless asked
    r = ode.results.bv
    waiting = r.n_U.select(sex='female', age='15-49')
    assert waiting[0] == 0 and waiting.max() < 10 and r.debut[2:].sum() > 50 # Only those who aged in during the step
    assert any('arrivals where it holds go straight' in note for note in ode.notes)
    return ode


@sc.timer()
def test_on_event():
    sc.heading('Testing ss.on() with an event: edge-triggered, in the same step, whatever the module order')
    def make(order):
        a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.5)), ia=ss.progression('I -> S', dur=ss.years(1)), name='a')
        S, F, G = ss.states('S F G')
        b = ss.Disease(S, F, G, flag=S >> F @ ss.on(a.sa), unflag=F >> S @ ss.years(0.2), name='b') # Flagged when infected with a; level semantics would flag them again on return
        mods = [a, b] if order else [b, a]
        return ss.Sim(mods, n_agents=3000, start=2000, stop=2005, dt=ss.months(1), unit='year', **kw).run()
    for order in [True, False]:
        sim = make(order)
        res = sim.results
        assert res.b.flag.sum() > 100
        assert res.b.flag.sum() <= res.a.sa.sum() # Only those infected (and in S of b at the time)
        assert np.isclose(res.b.flag[1:].sum(), res.a.sa[1:].sum(), rtol=0.25)

    # The events of a state by path, for modules made from strings: the same as the state's own
    def by_path(path):
        a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.5)), ia=ss.progression('I -> S', dur=ss.years(1)), name='a')
        S, F, G = ss.states('S F G')
        b = ss.Disease(S, F, G, flag=S >> F @ ss.on(ss.state('a.I').enter if path else a.I.enter), unflag=F >> S @ ss.years(0.2), name='b')
        on = [ss.state('a.I').exit] if path else [a.I.exit] # A transition by path (ss.state('a.sa')) is not known to be an event when the delivery is made, so is not an on=
        dx = [ss.screen(ss.Diagnostic(), on=x, name=f'dx{i}') for i,x in enumerate(on)]
        return ss.Sim([a, b], interventions=dx, n_agents=3000, start=2000, stop=2005, dt=ss.months(1), unit='year', **kw).run().results
    direct, path = by_path(False), by_path(True)
    assert path.b.flag.sum() > 100 and np.array_equal(path.b.flag.values, direct.b.flag.values)
    assert path.dx0.delivered.sum() > 100 and np.array_equal(path.dx0.delivered.values, direct.dx0.delivered.values)

    # An event of its own module, as compartments: those it happened to move at the end of the step
    S, F, G = ss.states('S F G')
    a = ss.Disease(S, F, G, sa=(sa := S >> F @ ss.peryear(0.5)), flag=F >> G @ ss.on(sa), name='a')
    res = ss.Sim(a, solver='ode', **kw).run().results.a
    assert res.flag.sum() > 0 and np.isclose(res.flag.sum(), res.sa.sum(), rtol=0.01) and res.n_F.max() < 1e-6*res.n_S[0]
    return sim


@sc.timer()
def test_join_events():
    sc.heading('Testing events joined with |: branches and whole transitions, in any order, before they are named')
    def make(join):
        S, P, C, X = ss.states('S P C X')
        precin = ss.Transition(P, [C, S], p=[0.2, None], dur=ss.years(1.5))
        cin = ss.Transition(C, [S, X], p=[0.5, None], dur=ss.years(2))
        clear = ss.Transition(X, S, dur=ss.years(1))
        on = join(precin, cin, clear, S)
        kinds = sorted(e.args[0] for e in ss.expressions.events_in(on))
        assert kinds == ['to', 'to', 'tr'], kinds
        nab = ss.attr(ss.uniform(0.5, 0.95), on=on, combine='max', init=0)
        return ss.Disease(S, P, C, X, infection=S >> P @ ss.peryear(0.4)*(P + C), precin=precin, cin=cin, clear=clear, nab=nab, init_prev=0.3, name='hpv')
    joins = [
        lambda a, b, tr, S: a.to(S) | b.to(S) | tr,
        lambda a, b, tr, S: tr | a.to(S) | b.to(S),
        lambda a, b, tr, S: tr | (a.to(S) | b.to(S)),
        lambda a, b, tr, S: a.to(S) | tr | b.to(S),
    ]
    for join in joins:
        sim = ss.Sim(make(join), networks='random', n_agents=2000, start=2000, stop=2010, unit='year', **kw).run()
        assert (sim.diseases.hpv.nab.values > 0).sum() > 50
    return sim


@sc.timer()
def test_hpv_cancel():
    sc.heading('Port: HPV, cancelling the other genotypes when one reaches cancer, with ss.on()')
    names = ['hpv16', 'hpv18']
    def make(g):
        S, P, C, X, Z = ss.states('S P C X Z')
        other = ss.any(ss.state(f'{o}.X') for o in names if o != g) # The siblings do not exist yet
        return ss.Disease(S, P, C, X, Z,
            infection = S >> P @ ss.peryear(0.3),
            precin    = P >> C @ ss.years(2),
            clear     = C >> S @ ss.years(4),
            cancer    = C >> X @ ss.years(4),
            cancelled = (S + P + C) >> Z @ ss.on(other), # Was rate=ss.peryear(1e3), where=other_cancer
            name = g)
    def check(sim):
        d = sim.diseases
        for g,o in [names, names[::-1]]:
            assert not np.any(d[o].X.raw & (d[g].S.raw | d[g].P.raw | d[g].C.raw)), f'step {sim.ti}: {g} progresses beside {o} cancer'
    for order in [names, names[::-1]]:
        sim = ss.Sim([make(g) for g in order], analyzers=check, n_agents=5000, start=2000, stop=2020, unit='year', **kw).run()
        res = sim.results
        assert res.hpv16.cancelled.sum() > 50 and res.hpv18.cancelled.sum() > 50
    return sim


def test_hpv_cancel_ode():
    """ As compartments, tracked jointly; arrivals by the other's move are cancelled at the next loop time """
    names = ['hpv16', 'hpv18']
    def make(g):
        S, P, C, X, Z = ss.states('S P C X Z')
        other = ss.any(ss.state(f'{o}.X') for o in names if o != g)
        return ss.Disease(S, P, C, X, Z, infection=S >> P @ ss.peryear(0.3), precin=P >> C @ ss.years(2), clear=C >> S @ ss.years(4),
                          cancer=C >> X @ ss.years(4), cancelled=(S + P + C) >> Z @ ss.on(other), name=g)
    sim = ss.Sim([make(g) for g in names], n_agents=5000, start=2000, stop=2020, unit='year', **kw).run()
    ode = ss.Sim([make(g) for g in names], solver='ode', total_pop=5000, start=2000, stop=2020, unit='year', **kw).run()
    assert ode.crossings and np.isclose(ode.results.hpv16.cancelled.sum(), sim.results.hpv16.cancelled.sum(), rtol=0.25)
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
        return ss.Disease(LL, T, Sp, tertiary=tertiary, cure=(LL + T) >> Sp, name='syph') # Everyone starts in LL; cure is undriven, so not the remainder
    out = sc.objdict()
    with pytest.raises(ss.BadBranchSplit, match='ss.stay'): # The rest stay only if asked to, with ss.stay
        make('implicit')
    for solver in ['abm', 'ode']:
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        sim = ss.Sim(make('explicit'), solver=solver, start=2000, stop=2006, unit='year', **size, **kw).run()
        res = sim.results.syph
        assert np.isclose(res.n_T[-1]/1e4, 0.35, atol=0.02), f'{solver}: {res.n_T[-1]/1e4:.3f} in T'
        assert np.isclose(res.tertiary.sum(), res.n_T[-1], rtol=1e-6) # Staying is not a flow
        out[f'explicit_{solver}'] = sim
    assert 'states.LL.stay' not in out.explicit_abm.diseases.syph.graph.guessed
    s = out.explicit_abm.diseases.syph # Those who stayed: no new dwell time, and LL.time keeps running
    u = s.LL.uids
    assert np.all(s.ti_LL[u] == 0) and np.all(np.isinf(s.ti_exit_LL[u]))

    # The remainder of branches whose p= is not a constant
    def branches(p):
        I, H, R = ss.states('I H R')
        return ss.Disease(I, H, R, severe=ss.Transition(I, H, p=p, dur=ss.years(1)), mild=ss.Transition(I, R, dur=ss.years(1)), name='d') # Everyone starts in I
    for solver in ['abm', 'ode']:
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        ns = [ss.Sim(branches(p), solver=solver, start=2000, stop=2010, unit='year', **size, **kw).run().results.d.n_H[-1] for p in [0.3, lambda sim: 0.3]]
        assert np.isclose(ns[1]/1e4, 0.3, atol=0.02) and np.isclose(ns[0], ns[1], rtol=0.05), f'{solver}: {ns}'
    with pytest.raises(ss.BadBranchSplit):
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
            name='ct') # Everyone starts in E
    kw2 = dict(start=2000, stop=2002, dt=ss.months(0.25), unit='year', **kw)
    sims = sc.objdict()
    for key,form,strat in [('nostrat', 'pars', None), ('pars', 'pars', dict(sex=['female', 'male'])), ('dists', 'dists', dict(sex=['female', 'male']))]:
        sims[key] = ss.Sim(make(form), n_agents=10_000, stratify=strat, **kw2).run() # As agents, sex needs no stratification
    ct, ppl = sims.nostrat.diseases.ct, sims.nostrat.people
    for f,(mean,_) in zip([True, False], ms):
        went = ppl.auids[~np.isnan(ct.ti_Y[ppl.auids])] # Everyone who went through P to Y
        u = went[ppl.female[went] == f]
        dwell = (ct.ti_Y[u] - ct.ti_P[u])*0.25
        assert np.isclose(dwell.mean(), mean, rtol=0.15), f'female={f}: mean dwell {dwell.mean():.2f} months, not {mean}'
    n_P = sims.nostrat.results.ct.n_P[0]
    assert np.isclose(n_P, 10_000*(0.5*0.2 + 0.5*0.54), rtol=0.05)
    assert np.array_equal(sims.pars.results.ct.n_P, sims.dists.results.ct.n_P) # The two forms are one distribution

    # As compartments: each sex has its own Erlang stages (ragged), keeping its mean
    ode = ss.Sim(make(), solver='ode', total_pop=10_000, stratify=dict(sex=['female', 'male']), **kw2).run()
    r, a = ode.results.ct, sims.pars.results.ct
    for sex in ['female', 'male']:
        assert np.allclose(r.n_P.select(sex=sex)[::4], a.n_P.select(sex=sex)[::4], atol=150), sex
    assert any('4|36 Erlang stages' in note for note in ode.notes)
    with pytest.raises(ss.CapabilityError, match='stratify by it'): # As compartments, sex needs a stratification
        ss.Sim(make(), solver='ode', **kw2).init()

    # A Dist with a function as a parameter (FPsim: duration of use by age), and typed rates inside ss.by
    def callable_dist(solver):
        A, B = ss.states('A B')
        d = ss.Disease(A, B, ab=ss.Transition(A, B, dur=ss.lognorm_ex(mean=lambda self, sim, uids: np.full(len(uids), 0.5), std=0.1)), name='d') # Everyone starts in A
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
def test_with_conveyor(solvers=('abm',)):
    sc.heading('ss.stay and ss.on out of a fixed dwell (a conveyor), in a crossing made by a trigger; a clock there needs agents')
    size = lambda solver: dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    for solver in solvers: # Those who stay at the end of a fixed dwell are not in the flow
        LL, T, Sp = ss.states('LL T Sp')
        d = ss.Disease(LL, T, Sp, tertiary=ss.Transition(LL, [T, ss.stay], p=[0.35, None], dur=ss.constant(ss.years(2))), cure=(LL + T) >> Sp, init=dict(LL=10_000), name='syph') # Everyone (1 or more is people)
        res = ss.Sim(d, solver=solver, start=2000, stop=2004, unit='year', **size(solver), **kw).run().results.syph
        assert res.n_T[1] == 0 and np.isclose(res.n_T[-1]/1e4, 0.35, atol=0.02), f'{solver}: {res.n_T}'
        assert np.isclose(res.tertiary.sum(), res.n_T[-1], rtol=1e-6), f'{solver}: {res.tertiary.sum()} in the flow, {res.n_T[-1]} in T'
    if 'ode' not in solvers:
        return

    def make(clock):
        a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.5)), name='a')
        E, I, Z = ss.states('E I Z')
        latent = E >> I @ (ss.peryear(1)*(E.time > ss.years(1))) if clock else ss.Transition(E, I, dur=ss.constant(ss.years(2)))
        return [a, ss.Disease(E, I, Z, latent=latent, flag=ss.Transition(E, Z, on=a.I), init=dict(E=10_000), name='b')] # The trigger tracks a and b jointly
    out = [ss.Sim(make(False), solver=solver, start=2000, stop=2004, dt=ss.months(1), unit='year', **size(solver), **kw).run().results.b for solver in ['abm', 'ode']]
    for key in ['n_I', 'n_Z']:
        assert np.allclose(out[0][key], out[1][key], atol=300), f'{key}: {out[0][key][::12]} as agents, {out[1][key][::12]} as compartments'
    with pytest.raises(ss.CapabilityError, match='E.time is a clock'):
        ss.Sim(make(True), solver='ode', start=2000, stop=2004, dt=ss.months(1), unit='year', **size('ode'), **kw).init()
    return


@sc.timer()
def test_after_event():
    sc.heading('Testing an exact dwell until an event of another module, and a time after it (ST18)')
    A, B = ss.states('A B')
    def make(**kw):
        a = ss.Disease(A, B, go=ss.progression('A -> B', dur=ss.uniform(ss.days(5), ss.days(15))), name='a')
        X, Y = ss.states('X Y')
        return [a, ss.Disease(X, Y, end=ss.Transition(X, Y, after=a.go, **kw), name='b')]
    for delay,lag in [(None, 0), (ss.constant(ss.days(5)), 5)]:
        res = ss.Sim(diseases=make(**({} if delay is None else dict(dur=delay))), n_agents=2000, unit='day', stop=30, verbose=0).run().results
        assert np.array_equal(res.b.end.values[lag:], res.a.go.values[:len(res.a.go) - lag]) # Each person leaves X when a.go happens to them, or exactly 5 days after
    with pytest.warns(RuntimeWarning, match='ss.constant'): # A function as dur= is the mean of an exponential
        ss.Sim(ss.Disease(A, B, go=ss.progression('A -> B', dur=lambda sim: 10.0), name='a'), n_agents=100, verbose=0).init()
    return


def test_with_conveyor_ode():
    return test_with_conveyor(solvers=('abm', 'ode'))


def make_on_event(order, solver='abm'):
    a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.5)), ia=ss.progression('I -> S', dur=ss.years(1)), name='a')
    S, F, G = ss.states('S F G')
    b = ss.Disease(S, F, G, flag=S >> F @ ss.on(a.sa), unflag=F >> S @ ss.years(0.2), name='b')
    mods = [a, b] if order else [b, a]
    size = dict(n_agents=3000) if solver == 'abm' else dict(total_pop=3000)
    return ss.Sim(mods, solver=solver, start=2000, stop=2005, dt=ss.months(1), unit='year', **size, **kw).run()


def treated_event(solver):
    a = ss.Disease(inf=ss.progression('S -> I', rate=ss.peryear(0.5)), cure=ss.progression('I -> S'), name='a')
    S, F = ss.states('S F')
    b = ss.Disease(S, F, flag=S >> F @ ss.on(a.cure), unflag=F >> S @ ss.years(0.5), name='b')
    tx = ss.treat(ss.Treatment(moves=a.cure), where=a.I, coverage=ss.peryear(1.0), name="tx")
    size = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
    return ss.Sim([a, b], interventions=tx, solver=solver, start=2000, stop=2005, dt=ss.months(1), unit='year', **size, **kw).run().results


def test_treated_event():
    sc.heading('A move by a treatment is the event of its transition, for ss.on() in another module')
    r = treated_event('abm')
    assert r.b.flag.sum() > 1000 and r.b.flag.sum() <= r.a.cure.sum()
    return r


def test_on_event_ode():
    for order in [True, False]:
        ode = make_on_event(order, solver='ode') # First, so that it fails fast until B3
        res = make_on_event(order).results
        r = ode.results
        assert np.isclose(r.b.flag.sum(), res.b.flag.sum(), rtol=0.05), f'{r.b.flag.sum():.0f} as an ODE, {res.b.flag.sum():.0f} as agents'
        assert np.isclose(r.b.flag.sum(), r.b.unflag.sum() + r.b.n_F[-1], rtol=1e-6) # The flow is the move
    ro, ra = treated_event('ode'), treated_event('abm')
    assert np.isclose(ra.b.flag.sum(), ro.b.flag.sum(), rtol=0.1)


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
            cancelled = ss.Transition([S, P, C], Z, on=cin[o].to(states[o][3])), # The other genotype's branch to cancer
            name = g))
    return mods


def test_hpv_cancel_event():
    sc.heading('HPV: cancelling the other genotypes on the event of one reaching cancer, ss.on(other.cin.to(X)), as agents')
    names = ['hpv16', 'hpv18']
    def check(sim):
        d = sim.diseases
        for g,o in [names, names[::-1]]:
            assert not np.any(d[o].X.raw & (d[g].S.raw | d[g].P.raw | d[g].C.raw)), f'step {sim.ti}: {g} progresses beside {o} cancer'
    abm = ss.Sim(make_hpv_cancel(names), analyzers=check, n_agents=20_000, start=2000, stop=2020, unit='year', **kw).run()
    assert all(abm.results[g].cancelled.sum() > 0 for g in names)
    return abm


def test_hpv_cancel_event_ode():
    names = ['hpv16', 'hpv18']
    common = dict(start=2000, stop=2020, unit='year', **kw)
    odes = [ss.Sim(make_hpv_cancel(order), solver='ode', total_pop=20_000, **common).run() for order in [names, names[::-1]]] # First, so that it fails fast until B3
    abm = ss.Sim(make_hpv_cancel(names), n_agents=20_000, **common).run()
    for ode in odes:
        for g in names:
            a, o = abm.results[g], ode.results[g]
            assert np.isclose(o.cancelled.sum(), a.cancelled.sum(), rtol=0.05), f'{g}: {o.cancelled.sum():.0f} cancelled as an ODE, {a.cancelled.sum():.0f} as agents'
            other = ode.results[next(n for n in names if n != g)]
            assert np.isclose(o.n_Z[-1], other.n_X[-1], rtol=1e-6)


if __name__ == '__main__':
    sc.options(interactive=False)
    test_lazy()
    test_bv_debut()
    test_on_event()
    test_join_events()
    test_hpv_cancel()
    test_syphilis_stay()
    test_ct_by_sex()
    test_with_conveyor()
    test_after_event()
