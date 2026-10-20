"""
Test the declared ss.Pregnancy: states, conception, outcomes by gestational age, births, events, and lactation
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(start=2000, stop=2010, dt=ss.months(1), verbose=0)
strat = [dict(sex=['female', 'male']), dict(age=[0, 15, 50], split=[0.3, 0.4, 0.3])]


@sc.timer()
def test_states(ode=False):
    sc.heading('Testing ss.Pregnancy: conception, abortion, and losses by gestational age, as agents and compartments')
    thr = ss.weeks(20)
    def make():
        return ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_abortion=0.1, p_loss=ss.bernoulli(0.02), loss_threshold=thr)

    ga = sc.objdict(miscarriage=[], stillbirth=[], seen=set())
    def check(sim): # Each loss is classified by the gestational age it happened at
        preg = sim.demographics.pregnancy
        for b in preg.engine.batches:
            key = (b.tr, b.key, tuple(b.uids))
            if b.tr in ['miscarriage', 'stillbirth'] and key not in ga.seen:
                ga.seen.add(key)
                uids = b.uids[~preg.P.raw[b.uids]] # Not those pregnant again in the same step, whose time of conception is the new one
                ga[b.tr] += list((b.key[0] - preg.ti_P[uids])*preg.t.dt.weeks) # At the step it happened
        assert not np.any(preg.P & ~sim.people.female), 'only women are pregnant'

    abm = ss.Sim(demographics=make(), analyzers=check, n_agents=20_000, stratify=strat, use_aging=False, **kw).run()
    ra = abm.results.pregnancy
    assert len(ga.miscarriage) > 100 and len(ga.stillbirth) > 100
    assert max(ga.miscarriage) < thr.weeks <= min(ga.stillbirth)
    assert np.isclose(ra.abortion.sum()/ra.conception.sum(), 0.1, atol=0.02) # At once, as a branch on entry

    # Conception is one transition, so a modifier of it is a modifier of fertility (e.g. contraception)
    half = ss.Sim(demographics=(preg := make()), connectors=ss.modify(preg.conception, multiply=0.5), n_agents=20_000, stratify=strat, use_aging=False, **kw).run()
    assert np.isclose(half.results.pregnancy.conception.sum(), 0.5*ra.conception.sum(), rtol=0.15)
    if not ode:
        return abm

    ode = ss.Sim(demographics=make(), solver='ode', total_pop=20_000, stratify=strat, use_aging=False, **kw).run()
    ro = ode.results.pregnancy
    for key in ['conception', 'abortion', 'delivery']:
        assert np.isclose(ra[key].sum(), ro[key].sum(), rtol=0.1), f'{key}: agents {ra[key].sum():.0f}, ODE {ro[key].sum():.0f}'
    loss = lambda r: r.miscarriage.sum() + r.stillbirth.sum()
    assert np.isclose(loss(ra), loss(ro), rtol=0.1)
    assert ro.stillbirth.sum() > 0.25*loss(ro) # The threshold is a level of the gestation conveyor
    assert any('conveyor' in note for note in ode.notes)
    return abm


@pytest.mark.xfail(reason='Phase B, task B1: gestational age (P.time) as levels of a conveyor in a compartmental run', strict=True)
def test_states_ode():
    return test_states(ode=True)


@sc.timer()
def test_births(ode=False):
    sc.heading('Testing ss.birth(per=): children from deliveries, linked to the mother, twins, and placed in other modules')
    def make():
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), embryos_per_pregnancy=ss.choice(a=[1, 2], p=[0.8, 0.2]))
        S, I = ss.states('S I')
        d = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.2), recovery=I >> S @ ss.years(1), born=ss.birth('-> I', per=preg.delivery), name='d')
        return dict(demographics=preg, diseases=d)
    abm = ss.Sim(**make(), n_agents=20_000, stratify=strat, **kw).run()
    ra = abm.results
    assert np.isclose(ra.pregnancy.birth.sum()/ra.pregnancy.delivery.sum(), 1.2, atol=0.05) # Twins
    assert ra.d.born.sum() == ra.pregnancy.birth.sum() # Placed in I, as the other module's ss.birth(per=) says
    if ode:
        ode = ss.Sim(**make(), solver='ode', total_pop=20_000, stratify=strat, **kw).run()
        ro = ode.results
        for key in ['delivery', 'birth']:
            assert np.isclose(ra.pregnancy[key].sum(), ro.pregnancy[key].sum(), rtol=0.1)
        assert np.isclose(ro.d.born.sum(), ro.pregnancy.birth.sum()) # An inflow of the delivery flow, into the youngest groups
        assert np.isclose(ro.pregnancy.n_NP[-1] + ro.pregnancy.n_P[-1], ro.d.n_S[-1] + ro.d.n_I[-1])
        assert ro.pregnancy.n_NP.select(age='0-14').values[-1] > ro.pregnancy.n_NP.select(age='0-14').values[0]
        return ode

    # As agents: each child has the mother as parent, twins share a delivery, and newborns enter I
    ppl = abm.people
    kids = ppl.parent.notnan.uids
    moms = ppl.parent[kids]
    assert len(kids) == ra.pregnancy.birth.sum() and np.all(ppl.female[moms])
    assert np.isclose(ppl.female[kids].mean(), 0.5, atol=0.03)
    newest = kids[ppl.age[kids] < 0.1] # Born in the last step
    assert len(newest) and abm.diseases.d.I[newest].mean() > 0.8

    with pytest.raises(ValueError):
        ss.birth('-> S', rate=ss.peryear(0.01), per=abm.diseases.d.infection)
    return abm


@pytest.mark.xfail(reason='Phase B, tasks B1 and B4: gestational age as a conveyor, and births driven by a flow, as compartments', strict=True)
def test_births_ode():
    return test_births(ode=True)


@sc.timer()
def test_burnin():
    sc.heading('Testing burnin=: women pregnant at the start, as agents')
    pars = dict(start=2000, stop=2003, dt=ss.months(1), verbose=0, stratify=strat, use_aging=False)
    make = lambda burnin=True: ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_abortion=0.1, burnin=burnin)
    abm = ss.Sim(demographics=make(), n_agents=20_000, **pars).run()
    none = ss.Sim(demographics=make(False), n_agents=20_000, **pars).run()
    assert none.results.pregnancy.delivery[:8].sum() == 0, 'Without burn-in, nobody delivers before the first pregnancies come to term'
    r = abm.results.pregnancy
    early, later = r.delivery[1:9].sum(), r.delivery[13:21].sum()
    assert np.isclose(early, later, rtol=0.2), f'{early} deliveries in the first 8 months, against {later} a year later'
    assert np.isclose(r.n_P[0], r.n_P[18], rtol=0.15), f'{r.n_P[0]} pregnant at the start, {r.n_P[18]} later'
    assert r.conception[0] == 0 and r.abortion[0] == 0, 'Burn-in events are not counted in the results'

    # The TFR and ASFR, as in Starsim 3: about the fertility rate over the ages women can conceive at (against compartments once P.time can run as them: Phase B1)
    tfr = abm.results.pregnancy.tfr[13:25].sum()
    assert 2.5 < tfr < 3.5, f'TFR {tfr}'
    asfr = abm.demographics.pregnancy.asfr
    assert asfr.shape == (11, len(abm.timevec)) and asfr[:3, 24].sum() == 0 and asfr[4, 24] > 50 # None under 15; per 1000 women, over a year

    # sim.new() keeps burnin= (it replays the sim's inputs)
    again = abm.new().run().results.pregnancy
    assert np.array_equal(again.n_P.values, r.n_P.values) and np.array_equal(again.delivery.values, r.delivery.values)
    return abm


@pytest.mark.xfail(strict=True, reason='Phase B: burnin= as compartments is deferred (a compartmental run notes it and starts with nobody pregnant)')
def test_burnin_ode():
    pars = dict(start=2000, stop=2003, dt=ss.months(1), verbose=0, stratify=strat, use_aging=False)
    make = lambda: ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_abortion=0.1)
    abm = ss.Sim(demographics=make(), n_agents=20_000, **pars).run()
    ode = ss.Sim(demographics=make(), solver='ode', total_pop=20_000, **pars).run()
    ra, ro = abm.results.pregnancy, ode.results.pregnancy
    assert np.isclose(ra.n_P[0], ro.n_P[0], rtol=0.05) and np.isclose(ra.delivery[1:9].sum(), ro.delivery[1:9].sum(), rtol=0.1)


@sc.timer()
def test_events():
    sc.heading('Testing preg.* events: ss.on() in another module, on= on deliveries, ss.child(), and ANC at a visit after conception')
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_loss=ss.bernoulli(0.01))
    anc_ga = preg.attr('anc_ga', ss.uniform(ss.months(1), ss.months(8))) # Per pregnancy: ss.attr(..., on=preg.conception)
    S, F, I = ss.states('S F I')
    d = ss.Disease(S, F, I,
        flag      = S >> F @ ss.on(preg.conception), # Edge-triggered, in the same step, whatever the module order
        unflag    = F >> S @ ss.on(preg.delivery),
        infection = S >> I @ ss.peryear(0.05),
        recovery  = I >> S @ ss.years(1),
        name='d')
    test = lambda name: ss.Diagnostic(sensitivity=dict(I=0.9), specificity=0.99, name=name)
    anc = ss.screen(test('t1'), where=preg.P & (preg.P.time >= anc_ga), coverage=0.85, disease='d', name='anc') # Once a pregnancy, at its visit
    first = ss.screen(test('t2'), on=preg.conception, coverage=0.5, disease='d', name='first')
    seen = sc.objdict(child=0, flagged=0, fresh=0, gap=[])
    def check(sim):
        p = sim.demographics.pregnancy
        moms = ss.child(p.birth).uids
        assert set(moms) == set(p.live_birth.uids)
        seen.child += len(moms)
        lost = p.loss.uids # Since the last live birth, through a later loss
        sb = p.engine.eval_for(p.since_birth, lost)
        seen.gap += list(sb[np.isfinite(sb)])
    sim = ss.Sim(demographics=preg, diseases=d, interventions=[anc, first], analyzers=check, n_agents=10_000, **kw).run()
    r = sim.results
    assert 0.8*r.pregnancy.conception.sum() < r.d.flag.sum() <= r.pregnancy.conception.sum()
    assert r.d.unflag.sum() <= r.pregnancy.delivery.sum() and r.d.unflag.sum() > 0.8*r.pregnancy.delivery.sum()
    assert seen.child == r.pregnancy.delivery.sum() # Every delivery is a live birth here
    assert np.isclose(r.first.delivered.sum(), 0.5*r.pregnancy.conception.sum(), rtol=0.1)
    reached = r.pregnancy.delivery.sum() # Nearly every pregnancy reaches its visit (losses are few)
    assert 0.7*reached < r.anc.delivered.sum() < r.pregnancy.conception.sum()
    assert len(seen.gap) and min(seen.gap) > 0 # A loss does not reset the time since the last live birth
    with pytest.raises(ss.CapabilityError): # Agents only
        ss.Sim(demographics=(p := ss.Pregnancy(fertility_rate=ss.freqperyear(100))), connectors=ss.modify(p.conception, multiply=0.5, where=p.since_birth < ss.years(1)),
               solver='ode', total_pop=1000, stratify=strat, **kw).run()
    return sim


@sc.timer()
def make_fetal():
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), maternal_rate=ss.peryear(0.05))
    S, I, C, Cs = ss.states('S I C Cs')
    d = ss.Disease(S, I, C, Cs,
        infection = S >> I @ ss.peryear(0.1),
        mtct      = ss.Transition(S, [C, Cs], rate=ss.peryear(3)*ss.mother(I), p=[0.6, None]), # In utero, from the mother's state
        fetal     = C >> ss.null @ ss.peryear(4), # A fetal death before delivery: a loss for the mother, by gestational age
        still     = Cs >> ss.null @ ss.on(preg.delivery), # At delivery, before birth: a stillbirth
        born      = Cs >> S @ ss.on(preg.birth),
        name='d')
    return preg, d


@sc.timer()
def test_fetus():
    sc.heading('Testing fetuses as agents (fetus="auto"), ss.mother(), and fetal removals as losses')
    make = make_fetal
    preg, d = make()
    sim = ss.Sim(demographics=preg, diseases=d, networks=ss.MaternalNet(), n_agents=10_000, **kw).run()
    p, ppl, r = sim.demographics.pregnancy, sim.people, sim.results
    assert p.fetus and 'ss.mother in d.mtct' in p.fetus_reason
    assert r.d.mtct.sum() > 200 and r.d.still.sum() > 50
    assert r.pregnancy.stillbirth.sum() >= r.d.still.sum() and r.pregnancy.miscarriage.sum() > 50
    assert r.pregnancy.birth.sum() == r.pregnancy.delivery.sum() - r.d.still.sum() # A stillborn fetus is not born
    fetal = (p.fetal & ppl.alive).uids
    assert np.all(p.P[ppl.parent[fetal]]) and np.all(ppl.age[fetal] < 0) # Every fetus has a pregnant mother, and is lost with her
    assert len(sim.networks.maternalnet) == len(fetal)
    born, unborn = (ppl.alive & (ppl.age >= 0)).sum(), (ppl.alive & (ppl.age < 0)).sum()
    assert np.isclose(r.d.n_S[-1] + r.d.n_I[-1] + r.d.n_C[-1] + r.d.n_Cs[-1], born + unborn, rtol=0.01) and np.isclose(r.n_alive[-1], born, rtol=0.01) # The unborn are in the disease's states (so they can be infected before birth), though not among the people (as in test_mtct)

    # Nothing reads fetuses: children are made at delivery
    plain = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), n_agents=2000, **kw).init()
    assert not plain.demographics.pregnancy.fetus
    preg, d = make()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(demographics=preg, diseases=d, solver='ode', total_pop=1000, stratify=strat, **kw).run()
    return sim


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='Phase B7: a fetal level of the pregnant compartments, whose death is a loss for the mother by gestational age, is deferred')
def test_fetus_ode():
    common = dict(networks=ss.MaternalNet(), stratify=strat, use_aging=False, **kw)
    abm = ss.Sim(**dict(zip(['demographics', 'diseases'], make_fetal())), n_agents=30_000, **common).run().results
    ode = ss.Sim(**dict(zip(['demographics', 'diseases'], make_fetal())), solver='ode', total_pop=30_000, **common).run().results
    loss = lambda r: r.pregnancy.miscarriage.sum() + r.pregnancy.stillbirth.sum()
    for key,a,o in [('mtct', abm.d.mtct.sum(), ode.d.mtct.sum()), ('fetal', abm.d.fetal.sum(), ode.d.fetal.sum()), ('still', abm.d.still.sum(), ode.d.still.sum()), ('loss', loss(abm), loss(ode)), ('birth', abm.pregnancy.birth.sum(), ode.pregnancy.birth.sum())]:
        assert np.isclose(o, a, rtol=0.1), f'{key}: {o:.0f} as an ODE, {a:.0f} as agents'


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='Phase B7: ss.Lactation as compartments (start and restart as synchronized flows of live births) is deferred')
def test_lactation_ode():
    def lactation(solver):
        size = dict(n_agents=30_000) if solver == 'abm' else dict(total_pop=30_000)
        return ss.Sim(demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(150)), ss.Lactation(p_breastfeed=0.8)], solver=solver, stratify=strat, use_aging=False, **size, **kw).run()
    a, o = lactation('abm').results.lactation, lactation('ode').results.lactation
    for key,tol in [('start', 0.05), ('wean', 0.05), ('n_L', 0.15), ('restart', 0.3)]:
        va, vo = (a[key][-1], o[key][-1]) if key.startswith('n_') else (a[key].sum(), o[key].sum())
        assert np.isclose(vo, va, rtol=tol), f'{key}: {vo:.0f} as an ODE, {va:.0f} as agents'


@sc.timer()
def test_lactation():
    sc.heading('Testing ss.Lactation: started by live births, ended by weaning or an infant death; BreastfeedingNet; LAM')
    def make(lam=False):
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(150))
        S, I = ss.states('S I')
        d = ss.Disease(S, I,
            infection = ss.transmission('S -> I', beta=dict(maternalnet=0.3, breastfeedingnet=0.5)),
            death     = ss.Transition(I, ss.null, rate=ss.peryear(2), where=ss.ppl.age < 1), # Infant deaths end breastfeeding
            init=dict(I=0.3), name='d')
        mods = []
        if lam:
            lact = ss.Lactation()
            mods.append(ss.modify(preg.conception, multiply=0.1, where=lact.L & (preg.since_birth < ss.months(6))))
            return dict(demographics=[preg, lact], diseases=d, connectors=mods, networks=[ss.MaternalNet(), ss.BreastfeedingNet()])
        return dict(demographics=preg, diseases=d, networks=[ss.MaternalNet(), ss.BreastfeedingNet()])
    checked = sc.objdict(n=0)
    def check(sim):
        lact, net = sim.demographics.lactation, sim.networks.breastfeedingnet
        assert np.all(lact.L[net.p1]), 'every breastfeeding pair has a lactating mother'
        checked.n += len(net)
    sim = ss.Sim(**make(), analyzers=check, n_agents=10_000, **kw).run()
    r, ppl, hiv = sim.results, sim.people, sim.diseases.d
    assert 'lactation' in sim.demographics # lactation='auto': there is a BreastfeedingNet
    assert np.isclose(r.lactation.start.sum() + r.lactation.restart.sum(), r.pregnancy.birth.sum(), rtol=0.05) # p_breastfeed=1
    assert r.lactation.wean.sum() > 500 and r.lactation.stop.sum() > 10 and checked.n > 0
    kids = ppl.parent.notnan.uids
    infected = kids[hiv.I[kids]]
    assert len(infected) > 50 # In utero and by breastfeeding
    lam = ss.Sim(**make(lam=True), n_agents=10_000, **kw).run()
    assert lam.results.pregnancy.conception.sum() < 0.99*r.pregnancy.conception.sum() # Few conceive within 6 months of a birth anyway
    return sim


def make_hiv_vertical(bf=True):
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(120))
    S, I = ss.states('S I')
    beta = dict(maternalnet=ss.peryear(0.2), breastfeedingnet=ss.peryear(0.3 if bf else 0))
    hiv = ss.Disease(S, I, infection=ss.transmission('S -> I', beta=beta), death=ss.Transition(I, ss.null, rate=ss.peryear(0.05)), init=dict(I=0.2), name='hiv')
    return dict(demographics=[preg, ss.Lactation(p_breastfeed=0.8)], diseases=hiv, networks=[ss.MaternalNet(), ss.BreastfeedingNet()])


@sc.timer()
def test_hiv_vertical():
    sc.heading('Port: HIV vertical transmission, in utero over MaternalNet and by breastfeeding (04 §5.3)')
    common = dict(stratify=strat, use_aging=False, **kw)
    inf = [ss.Sim(**make_hiv_vertical(bf), n_agents=100_000, **common).run().results.hiv.infection.sum() for bf in [False, True]]
    utero, post = inf[0], inf[1] - inf[0]
    assert utero > 200 and post > 200, f'{utero:.0f} in utero, {post:.0f} by breastfeeding'
    return inf


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='Phase B7: MaternalNet as a fetal level and BreastfeedingNet as mass action in compartments are deferred')
def test_hiv_vertical_ode():
    common = dict(stratify=strat, use_aging=False, **kw)
    inf = lambda solver, bf: ss.Sim(**make_hiv_vertical(bf), **(dict(n_agents=100_000) if solver == 'abm' else dict(solver='ode', total_pop=100_000)), **common).run().results.hiv.infection.sum()
    o = [inf('ode', bf) for bf in [False, True]]
    a = [inf('abm', bf) for bf in [False, True]]
    for what,va,vo in zip(['in utero', 'breastfeeding'], [a[0], a[1] - a[0]], [o[0], o[1] - o[0]]):
        assert np.isclose(vo, va, rtol=0.2), f'{what}: {vo:.0f} as an ODE, {va:.0f} as agents'


@sc.timer()
def test_legacy():
    sc.heading('Testing back-compatibility: ss.LegacyPregnancy fires the same events, and ss.Pregnancy has the Starsim 3 names')
    out = sc.objdict()
    for cls in [ss.Pregnancy, ss.LegacyPregnancy]:
        preg = cls(fertility_rate=ss.freqperyear(100), p_maternal_death=ss.bernoulli(0.01), burnin=False)
        S, I, C, F = ss.states('S I C F')
        d = ss.Disease(S, I, C, F,
            infection = S >> I @ ss.peryear(0.1),
            mtct      = S >> C @ ss.peryear(3)*ss.mother(I),
            flag      = S >> F @ ss.on(preg.conception),
            unflag    = F >> S @ ss.on(preg.delivery), name='d')
        sim = ss.Sim(demographics=preg, diseases=d, networks=ss.MaternalNet(), n_agents=5000, **kw).run()
        r, p = sim.results, sim.demographics.pregnancy
        out[cls.__name__] = r
        assert r.d.flag.sum() > 0.5*r.pregnancy.pregnancies.sum() and r.d.unflag.sum() > 0.5*r.d.flag.sum()
        assert r.d.mtct.sum() > 100 and r.pregnancy.maternal_deaths.sum() > 0
        assert np.all(p.pregnant[p.pregnant.uids]) and np.all(p.ti_delivery[p.pregnant.uids] >= p.ti - 1) # Within a step: ss.LegacyPregnancy keeps fractional due times
        assert np.nanmax(p.gestation[p.pregnant.uids]) < 45 and np.all(p.parity >= 0)
        tris = [p.tri1_uids, p.tri2_uids, p.tri3_uids] # Starsim 3 names read by STIsim (e.g. BV, PMTCT)
        assert all(len(t) for t in tris) and sum(len(t) for t in tris) == len(p.pregnant.uids)
        if cls is ss.Pregnancy:
            with pytest.warns(RuntimeWarning): # No preterm births yet: rel_ptb is kept for code that writes it
                p.set_ptb()
            assert np.all(p.rel_ptb[p.pregnant.uids] >= 1)
    a, b = out.Pregnancy.pregnancy, out.LegacyPregnancy.pregnancy
    for key in ['pregnancies', 'births']:
        assert np.isclose(a[key].sum(), b[key].sum(), rtol=0.15), key

    # A Starsim 3 disease that transmits over the maternal network, with the declared pregnancy
    sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), diseases=ss.SIS(beta=dict(maternalnet=1.0, random=0.05)),
                 networks=[ss.MaternalNet(), ss.RandomNet()], n_agents=3000, **kw).run()
    assert sim.demographics.pregnancy.fetus and sim.results.sis.new_infections.sum() > 0
    with pytest.warns(FutureWarning):
        ss.Pregnancy().add_delivery_callback(lambda m, n: None)
    return sim


def make_syphilis():
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(120))
    S, E, S2, EL, LL, Cnnd, Cstill, Ccs, Cnorm, CS = ss.states('S E S2 EL LL Cnnd Cstill Ccs Cnorm CS')
    active, late = E | S2, LL
    def by_stage(a, e, l): # v3 birth_outcomes, by the mother's stage at infection
        return a*ss.mother(active) + e*ss.mother(EL) + l*ss.mother(late)
    syph = ss.Disease(S, E, S2, EL, LL, Cnnd, Cstill, Ccs, Cnorm, CS,
        infection  = S >> E @ ss.peryear(0.02),
        secondary  = E >> S2 @ ss.months(2),
        latent     = S2 >> EL @ ss.months(4),
        to_late    = EL >> LL @ ss.years(2),
        mtct       = ss.Transition(S, [Cnnd, Cstill, Ccs, Cnorm], rate=ss.peryear(0.9)*ss.mother(S2 | EL | LL),
                                   p=[by_stage(.25, .05, 0), by_stage(.25, .10, .10), by_stage(.45, .40, .10), None]),
        stillbirth = Cstill >> ss.null @ ss.on(preg.delivery), # Fetal removal at delivery: a stillbirth for the mother
        nnd        = Cnnd >> ss.null @ ss.on(preg.birth), # Born, then dies: a neonatal death
        born_cs    = Ccs >> CS @ ss.on(preg.birth),
        born_ok    = Cnorm >> S @ ss.on(preg.birth),
        init=dict(S2=0.03, EL=0.05, LL=0.1), name='syph')
    return dict(demographics=preg, diseases=syph)


syph_kw = dict(stratify=strat, use_aging=False, **kw)
syph_keys = ['mtct', 'stillbirth', 'nnd', 'born_cs', 'born_ok']


@sc.timer()
def test_syphilis():
    sc.heading('Port: syphilis congenital outcomes by maternal stage, with ss.mother() and ss.on(preg.delivery/birth) (03 §4.2)')
    sim = ss.Sim(**make_syphilis(), n_agents=50_000, **syph_kw).run()
    r, p = sim.results, sim.demographics.pregnancy
    assert p.fetus and 'ss.mother' in p.fetus_reason
    total = r.syph.mtct.sum()
    out = {k:r.syph[k].sum() for k in syph_keys[1:]}
    assert total > 500 and all(v > 10 for k,v in out.items() if k != 'nnd') and out['nnd'] > 0, out # Neonatal deaths are rare
    assert sum(out.values()) <= total # The rest unborn at the end, or lost earlier
    assert r.pregnancy.stillbirth.sum() >= out['stillbirth'] # Also a stillbirth of the pregnancy, for ANC and contraception to see
    assert r.pregnancy.neonatal_deaths.sum() >= out['nnd']
    assert np.isclose(r.pregnancy.birth.sum(), r.pregnancy.delivery.sum() - out['stillbirth'], rtol=0.01)
    return sim


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='Phase B7: the unborn as a fetal level of the mother\'s pregnant compartments (ss.mother, routing at the delivery and the birth) is deferred')
def test_syphilis_ode():
    r = ss.Sim(**make_syphilis(), n_agents=50_000, **syph_kw).run().results
    ode = ss.Sim(**make_syphilis(), solver='ode', total_pop=50_000, **syph_kw).run()
    o, po = ode.results, ode.demographics.pregnancy
    assert 'fetal_syph (S, Cnnd, Cstill, Ccs, Cnorm)' in po.fetus_reason
    for key,tol in zip(syph_keys, [0.1, 0.25, 0.5, 0.2, 0.1]):
        assert np.isclose(o.syph[key].sum(), r.syph[key].sum(), rtol=tol), f'{key}: {o.syph[key].sum():.0f} as an ODE, {r.syph[key].sum():.0f} as agents'
    assert np.isclose(o.pregnancy.stillbirth.sum(), o.syph.stillbirth.sum()) and np.isclose(o.pregnancy.neonatal_deaths.sum(), o.syph.nnd.sum())
    assert np.isclose(o.pregnancy.birth.sum(), o.pregnancy.delivery.sum() - o.syph.stillbirth.sum())
    assert np.isclose(o.syph.n_CS[-1], o.syph.born_cs.sum())
    n = lambda res: sum(res[f'n_{s}'][-1] for s in ['S', 'E', 'S2', 'EL', 'LL', 'CS'])
    assert np.isclose(n(o.syph), o.pregnancy.n_NP[-1] + o.pregnancy.n_P[-1])


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='Phase B7: ss.on(preg.conception/delivery) in another module as synchronized flows in compartments is deferred')
def test_events_ode():
    def flags(solver):
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_loss=ss.bernoulli(0.01))
        S, F = ss.states('S F')
        d = ss.Disease(S, F, flag=S >> F @ ss.on(preg.conception), unflag=F >> S @ ss.on(preg.delivery), name='d')
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        return ss.Sim(demographics=preg, diseases=d, solver=solver, stratify=strat, use_aging=False, **size, **kw).run().results
    ra, ro = flags('abm'), flags('ode')
    for key in ['flag', 'unflag']:
        assert np.isclose(ro.d[key].sum(), ra.d[key].sum(), rtol=0.05), f'{key}: {ro.d[key].sum():.0f} as an ODE, {ra.d[key].sum():.0f} as agents'


def make_postpartum():
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(150), p_loss=ss.bernoulli(0.02))
    preg.stratify(parity=['0', '1', '2', '3+'], split=[1, 0, 0, 0])
    return preg


@sc.timer()
def test_postpartum():
    sc.heading('Port: FPsim postpartum, the time since the last birth surviving later losses, and parity as a counter moved on delivery')
    seen = sc.objdict(n=0)
    def check(sim): # Every loss after a birth leaves the time since it running
        p = sim.demographics.pregnancy
        lost = p.loss.uids
        sb = p.engine.eval_for(p.since_birth, lost)
        had = np.isfinite(sb)
        seen.n += had.sum()
        assert np.all(sb[had] > 0)
    abm = ss.Sim(demographics=make_postpartum(), analyzers=check, n_agents=20_000, stratify=strat, use_aging=False, **kw).run()
    assert seen.n > 20
    p, ppl = abm.demographics.pregnancy, abm.people
    women = (ppl.female & (ppl.age >= 15)).uids
    assert np.all(ppl.parity[women] == np.minimum(p.parity[women], 3)) # The dimension, capped, as the counter
    return abm


@pytest.mark.xfail(strict=True, raises=ss.CapabilityError, reason='Phase B1: clocks (P.time) as compartments are deferred, so this pregnancy cannot run with solver="ode"')
def test_postpartum_ode():
    by = lambda sim: sum(np.asarray(sim.results.pregnancy[k].select(sex='female').by('parity').values)[-1] for k in ['n_NP', 'n_P'])
    abm = ss.Sim(demographics=make_postpartum(), n_agents=20_000, stratify=strat, use_aging=False, **kw).run()
    ode = ss.Sim(demographics=make_postpartum(), solver='ode', total_pop=20_000, stratify=strat, use_aging=False, **kw).run()
    a, o = by(abm), by(ode)
    assert np.allclose(a[1:], o[1:], rtol=0.1), (a, o)


if __name__ == '__main__':
    T = sc.timer()
    abm = test_states()
    births = test_births()
    events = test_events()
    fetus = test_fetus()
    lact = test_lactation()
    hiv = test_hiv_vertical()
    legacy = test_legacy()
    syph = test_syphilis()
    pp = test_postpartum()
    T.toc()
