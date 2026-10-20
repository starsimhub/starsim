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


@pytest.mark.xfail(reason='Phase B, task B2: triggers (ss.on) as compartments', strict=True)
def test_bv_debut_ode():
    """ As compartments: the compartments of women 15+ in U are empty at the start, and aging into them is moved on at each step """
    adult_f = ss.ppl.female & (ss.ppl.age >= 15)
    U, CST1, CST4 = ss.states('U CST1 CST4')
    bv = ss.Disease(U, CST1, CST4, debut=ss.progression('U -> CST1 | CST4', on=adult_f, p=[0.8, None]), onset=CST1 >> CST4 @ ss.peryear(0.2), clear=CST4 >> CST1 @ ss.months(6), name='bv')
    strat = [dict(sex=['female', 'male']), dict(age=[0, 10, 15, 50, 100])]
    ode = ss.Sim(bv, solver='ode', total_pop=10_000, stratify=strat, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw).run()
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
    with pytest.raises(ss.CapabilityError): # Not yet as compartments
        a = ss.Disease(sa=ss.progression('S -> I', rate=ss.peryear(0.5)), name='a')
        b = ss.Disease(flag=ss.progression('S -> F', on=a.sa), name='b')
        ss.Sim([a, b], solver='ode', **kw).init()
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


@pytest.mark.xfail(reason='Phase B, task B2: triggers (ss.on) as compartments', strict=True)
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
    with pytest.raises(ss.BadBranchSplit, match='ss.stay'): # The rest stay only if asked to (pending a decision: ss.transitions.rest_stay)
        make('implicit')
    for form in ['implicit', 'explicit']:
        for solver in ['abm', 'ode']:
            size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
            ss.transitions.rest_stay = form == 'implicit'
            try:
                sim = ss.Sim(make(form), solver=solver, start=2000, stop=2006, unit='year', **size, **kw).run()
            finally:
                ss.transitions.rest_stay = False
            res = sim.results.syph
            assert np.isclose(res.n_T[-1]/1e4, 0.35, atol=0.02), f'{form} ({solver}): {res.n_T[-1]/1e4:.3f} in T'
            assert np.isclose(res.tertiary.sum(), res.n_T[-1], rtol=1e-6) # Staying is not a flow
            out[f'{form}_{solver}'] = sim
    desc = out.implicit_abm.diseases.syph.describe(output=True)
    assert 'the other 65% stay' in desc and '[guessed]' in desc
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
            onset = ss.by(sex=dict(zip(['female', 'male'], [ss.lognorm_ex(ss.months(m), ss.months(sd)) for m,sd in ms])))
        p_symp = ss.by(sex=dict(female=0.2, male=0.54))
        return ss.Disease(E, P, A, Y, S,
            latency_P = ss.Transition(E, P, p=p_symp), # Was one transition per sex, with p=by_sex(...) and sync_sex()
            latency_A = ss.Transition(E, A, p=1 - p_symp),
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
    with pytest.raises(ss.CapabilityError, match='stratify'): # By sex given by name, with no dimension of sex: to stratify, to run as compartments
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
    sc.heading('ss.stay and ss.on out of a fixed dwell (a conveyor), and a clock in a crossing made by a trigger')
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
    for clock in [False, True]:
        out = [ss.Sim(make(clock), solver=solver, start=2000, stop=2004, dt=ss.months(1), unit='year', **size(solver), **kw).run().results.b for solver in ['abm', 'ode']]
        for key in ['n_I', 'n_Z']:
            assert np.allclose(out[0][key], out[1][key], atol=300), f'clock={clock}, {key}: {out[0][key][::12]} as agents, {out[1][key][::12]} as compartments'
    return


@pytest.mark.xfail(reason='Phase B, tasks B1 and B2: the conveyor with ss.stay, clocks and triggers as compartments', strict=True)
def test_with_conveyor_ode():
    return test_with_conveyor(solvers=('abm', 'ode'))


if __name__ == '__main__':
    sc.options(interactive=False)
    test_lazy()
    test_bv_debut()
    test_on_event()
    test_hpv_cancel()
    test_syphilis_stay()
    test_ct_by_sex()
    test_with_conveyor()
