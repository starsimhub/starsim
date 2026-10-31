"""
Test the declared ss.Pregnancy: states, conception, outcomes by gestational age, births, events, and lactation
"""
import numpy as np
import sciris as sc
import os
import tempfile
import pytest
import starsim as ss

kw = dict(start=2000, stop=2010, dt=ss.months(1), verbose=0)
strat = [dict(sex=['female', 'male']), dict(age=[0, 15, 50], split=[0.3, 0.4, 0.3])]


@sc.timer()
def test_states():
    sc.heading('Testing ss.Pregnancy: conception, abortion, and losses by gestational age, as agents and in the small compartmental form')
    thr = ss.weeks(20)
    def make():
        return ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_abortion=0.1, loss_rate=ss.probpermonth(0.02), loss_threshold=thr)

    ga = sc.obj(miscarriage=[], stillbirth=[], seen=set())
    def check(sim): # Each loss is classified by the gestational age it happened at
        preg = sim.demographics.pregnancy
        for b in preg.engine.batches:
            key = (b.tr, b.key, tuple(b.uids))
            if b.tr in ['miscarriage', 'stillbirth'] and key not in ga.seen:
                ga.seen.add(key)
                step = b.key[0] + 1 # The index of the step it happened in (the batch is dated at the step's start)
                uids = b.uids[preg.ti_P[b.uids] < step] # Not those who conceived again in the same step, whose time of entering P is now that of the new pregnancy
                ga[b.tr] += list((step - preg.ti_P[uids])*preg.t.dt.weeks) # At the step it happened
        assert not np.any(preg.P & ~sim.people.female), 'only women are pregnant'

    abm = ss.Sim(demographics=make(), analyzers=check, n_agents=20_000, stratify=strat, use_aging=False, **kw).run()
    ra = abm.results.pregnancy
    assert len(ga.miscarriage) > 100 and len(ga.stillbirth) > 100
    assert max(ga.miscarriage) < thr.weeks <= min(ga.stillbirth)
    assert np.isclose(ra.abortion.sum()/ra.conception.sum(), 0.1, atol=0.02) # At once, as a branch on entry

    # Conception is one transition, so a modifier of it is a modifier of fertility (e.g. contraception)
    half = ss.Sim(demographics=(preg := make()), connectors=ss.modify(preg.conception, multiply=0.5), n_agents=20_000, stratify=strat, use_aging=False, **kw).run()
    assert np.isclose(half.results.pregnancy.conception.sum(), 0.5*ra.conception.sum(), rtol=0.15)

    # The small compartmental form
    ode = ss.Sim(demographics=make(), solver='ode', total_pop=20_000, stratify=strat, use_aging=False, **kw).run()
    ro = ode.results.pregnancy
    for key in ['conception', 'abortion', 'delivery']:
        assert np.isclose(ra[key].sum(), ro[key].sum(), rtol=0.1), f'{key}: agents {ra[key].sum():.0f}, ODE {ro[key].sum():.0f}'
    loss = lambda r: r.miscarriage.sum() + r.stillbirth.sum()
    assert np.isclose(loss(ra), loss(ro), rtol=0.1)
    assert np.isclose(ro.P[0], ra.P[0], rtol=0.1) # Women pregnant at the start (burnin=), spread evenly over gestation
    assert np.isclose(ro.miscarriage.sum()/loss(ro), thr.weeks/40.09, rtol=0.01) # The share of the mean gestation below the threshold, as nothing reads P.time
    assert any('small compartmental form' in str(note) for note in ode.notes)
    return abm


@sc.timer()
def test_births():
    sc.heading('Testing ss.birth(per=): children from deliveries, linked to the mother, twins, and placed in other modules')
    def make():
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), embryos_per_pregnancy=ss.choice(a=[1, 2], p=[0.8, 0.2]))
        S, I = ss.states('S I')
        d = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.2), recovery=I >> S @ ss.years(1), born=ss.birth(I, per=preg.delivery), name='d')
        return dict(demographics=preg, diseases=d)
    abm = ss.Sim(**make(), n_agents=20_000, stratify=strat, **kw).run()
    ra = abm.results
    assert np.isclose(ra.pregnancy.birth.sum()/ra.pregnancy.delivery.sum(), 1.2, atol=0.05) # Twins
    assert ra.d.born.sum() == ra.pregnancy.birth.sum() # Placed in I, as the other module's ss.birth(per=) says

    # As agents: each child has the mother as parent, twins share a delivery, and newborns enter I
    ppl = abm.people
    kids = ppl.parent.notnan.uids
    moms = ppl.parent[kids]
    assert len(kids) == ra.pregnancy.birth.sum() and np.all(ppl.female[moms])
    assert np.isclose(ppl.female[kids].mean(), 0.5, atol=0.03)
    newest = kids[ppl.age[kids] < 0.1] # Born in the last step
    assert len(newest) and abm.diseases.d.I[newest].mean() > 0.8

    with pytest.raises(ValueError):
        S = ss.State()
        ss.birth(S, rate=ss.peryear(0.01), per=abm.diseases.d.infection)

    # In the small compartmental form
    ode = ss.Sim(**make(), solver='ode', total_pop=20_000, stratify=strat, **kw).run()
    ro = ode.results
    for key in ['delivery', 'birth']:
        assert np.isclose(ra.pregnancy[key].sum(), ro.pregnancy[key].sum(), rtol=0.1)
    assert np.isclose(ro.d.born.sum(), ro.pregnancy.birth.sum()) # An inflow of the delivery flow, into the youngest groups
    assert np.isclose(ro.pregnancy.NP[-1] + ro.pregnancy.P[-1], ro.d.S[-1] + ro.d.I[-1])
    assert ro.pregnancy.NP.select(age='0-14').values[-1] > ro.pregnancy.NP.select(age='0-14').values[0]
    return abm


@sc.timer()
def test_compartmental():
    sc.heading('Testing the small compartmental form against agents at three step sizes, and its conservation of people')
    make = lambda: ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_abortion=0.1, loss_rate=ss.probpermonth(0.02), burnin=False)
    assert ss.Sim(demographics=make(), total_pop=20_000, **kw).init().solver == 'abm' # Only when a compartmental solver is named
    for dt in [ss.weeks(1), ss.months(1), ss.months(3)]:
        pars = dict(start=2000, stop=2005, dt=dt, verbose=0, stratify=strat, use_aging=False)
        ras = [ss.Sim(demographics=make(), n_agents=20_000, rand_seed=seed, **pars).run().results.pregnancy for seed in range(2)]
        ro = ss.Sim(demographics=make(), solver='ode', total_pop=20_000, **pars).run().results.pregnancy
        half = lambda r: r.P.values[len(r.P)//2:].mean() # The number pregnant, once settled
        for key,f in [('delivery', lambda r: r.delivery.sum()), ('birth', lambda r: r.birth.sum()), ('P', half)]:
            va = np.mean([f(r) for r in ras])
            assert np.isclose(f(ro), va, rtol=0.05), f'{key} at {dt}: {f(ro):.0f} in compartments, {va:.0f} as agents'

    # People are conserved: the start, plus births, less maternal deaths
    sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100), maternal_rate=ss.peryear(0.05), embryos_per_pregnancy=ss.choice(a=[1, 2], p=[0.8, 0.2])),
                 solver='binomial', total_pop=20_000, stratify=strat, **kw).run()
    r = sim.results.pregnancy
    assert r.antepartum.sum() > 0 and r.birth.sum() > 1.1*r.delivery.sum() # Twins
    assert np.isclose(r.NP[-1] + r.P[-1], r.NP[0] + r.P[0] + r.birth.sum() - r.antepartum.sum())
    return sim


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
    assert np.isclose(r.P[0], r.P[18], rtol=0.15), f'{r.P[0]} pregnant at the start, {r.P[18]} later'
    assert np.isclose(r.birth[1], r.delivery[1] - r.stillbirth[1], rtol=0.1), 'The first deliveries have their births'
    assert r.conception[0] == 0 and r.abortion[0] == 0, 'Burn-in events are not counted in the results'

    # The TFR and ASFR, as in Starsim 3: about the fertility rate over the ages women can conceive at (against compartments once P.time can run as them: Phase B1)
    tfr = abm.results.pregnancy.tfr[13:25].mean() # Per year at each step, whatever the step (with monthly steps, it was the TFR of the month)
    assert 2.5 < tfr < 3.5, f'TFR {tfr}'
    assert abm.results.pregnancy.tfr.kind == 'ratio' # So it is averaged over time, not summed
    asfr = abm.demographics.pregnancy.asfr
    assert asfr.shape == (11, len(abm.timevec)) and asfr[:3, 24].sum() == 0 and asfr[4, 24] > 50 # None under 15; per 1000 women, over a year

    # sim.new() keeps burnin= (it replays the sim's inputs)
    again = abm.new().run().results.pregnancy
    assert np.array_equal(again.P.values, r.P.values) and np.array_equal(again.delivery.values, r.delivery.values)
    return abm


@sc.timer()
def test_events():
    sc.heading('Testing preg.* events: ss.on() in another module, where= on deliveries, the mothers of births, and ANC at a visit after conception')
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), loss_rate=ss.probperweek(0.01))
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
    first = ss.screen(test('t2'), where=preg.conception, coverage=0.5, disease='d', name='first')
    seen = sc.obj(child=0, flagged=0, fresh=0, gap=[])
    def check(sim):
        p = sim.demographics.pregnancy
        moms = np.unique(sim.people.parent[p.birth.uids]) # The mothers of the children born
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
def test_mother_cond():
    sc.heading('Testing ss.mother(born=True): newborns of mothers who tested positive in the last year')
    for by_path in [False, True]: # With the event itself, or by its path, for a module made before the pregnancy module exists
        seen = mother_cond(by_path)
    with pytest.raises(ValueError, match=r"not a transition.*ss.state\('pregnancy.P'\).enter"): # A path in where= is a transition's event, so a state is an error
        ss.Sim(demographics=ss.Pregnancy(), diseases='sis', networks='random', interventions=ss.screen(ss.Diagnostic(), where='pregnancy.P', name='scr'), n_agents=100, **kw).init()
    def make(solver): # Nothing else here needs agents
        S, I = ss.states('S I')
        d = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.2), births=ss.birth(S, rate=ss.peryear(0.03)), name='d')
        nb = ss.screen(ss.Diagnostic(sensitivity=1.0), who=d.births & ss.mother(I, born=True), disease='d')
        return ss.Sim(diseases=d, interventions=nb, solver=solver, total_pop=1000, **kw)
    with pytest.raises(ss.CapabilityError, match='mother'): # Agents only: compartments do not know anyone's mother
        make('ode').run()
    assert make('auto').init().solver == 'abm'
    return seen


def mother_cond(by_path):
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100))
    S, I = ss.states('S I')
    d = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.2), name='d')
    screening = ss.screen(ss.Diagnostic(sensitivity=dict(I=1.0), specificity=1.0, name='dx'), where=preg.P, coverage=0.5, disease='d', name='screening')
    tested = ss.mother(screening.positive.since <= ss.years(1), born=True)
    trigger = dict(who=tested, where='pregnancy.birth') if by_path else dict(who=preg.birth & tested)
    newborn = ss.screen(ss.Diagnostic(sensitivity=1.0, name='nbdx'), **trigger, disease='d', name='newborn')
    seen = sc.obj(last={}, n=0)
    def check(sim): # Run after both screens, so it sees their events
        ppl = sim.people
        fired = lambda event: np.flatnonzero(ss.networks.people_mask(sim, ss.conditions.as_cond(event)))
        for uid in fired(screening.positive):
            seen.last[int(uid)] = sim.ti
        kids = ss.conditions.Event('tr', sim.demographics.pregnancy.birth).uids
        moms = ppl.parent.raw[kids]
        expected = {int(k) for k,m in zip(kids, moms) if not np.isnan(m) and sim.ti - seen.last.get(int(m), -np.inf) <= 12} # Within a year, with monthly steps
        reached = set(fired(newborn).tolist())
        assert reached == expected
        seen.n += len(reached)
    ss.Sim(demographics=preg, diseases=d, interventions=[screening, newborn, check], n_agents=10_000, **kw).run()
    assert seen.n > 50
    return seen


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
    assert np.isclose(r.d.S[-1] + r.d.I[-1] + r.d.C[-1] + r.d.Cs[-1], born + unborn, rtol=0.01) and np.isclose(r.alive[-1], born, rtol=0.01) # The unborn are in the disease's states (so they can be infected before birth), though not among the people (as in test_mtct)

    # Nothing reads fetuses: children are made at delivery
    plain = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), n_agents=2000, **kw).init()
    assert not plain.demographics.pregnancy.fetus

    # A mother who dies of another cause loses her pregnancy, whether or not fetuses are agents, and fetuses don't face background deaths
    losses = []
    for fetus in [False, True]:
        r = ss.Sim(demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(150), fetus=fetus), ss.Deaths(death_rate=ss.freqperyear(100))], n_agents=5000, rand_seed=1, **kw).run().results.pregnancy
        losses.append(r.miscarriage.sum() + r.stillbirth.sum())
    assert losses[0] > 50 and losses[0] == losses[1]
    return sim # Agents only: test_fetus_ode


@sc.timer()
def test_lactation():
    sc.heading('Testing breastfeeding (preg.lactating): started by live births, ended by weaning or an infant death; BreastfeedingNet; LAM')
    def make(lam=False):
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(150))
        S, I = ss.states('S I')
        d = ss.Disease(S, I,
            infection = ss.Transition(S, I, rate=dict(maternalnet=0.3, breastfeedingnet=0.5) * I),
            death     = ss.Transition(I, ss.null, rate=ss.peryear(2), where=ss.ppl.age < 1), # Infant deaths end breastfeeding
            init=dict(I=0.3), name='d')
        mods = [ss.modify(preg.conception, multiply=0.1, where=preg.lactating & (preg.since_birth < ss.months(6)))] if lam else []
        return dict(demographics=preg, diseases=d, connectors=mods, networks=[ss.MaternalNet(), ss.BreastfeedingNet()])
    checked = sc.obj(n=0, weaned=0, stopped=0, last=None)
    def check(sim):
        preg, net = sim.demographics.pregnancy, sim.networks.breastfeedingnet
        assert np.all(preg.lactating[net.p1]), 'every breastfeeding pair has a lactating mother'
        checked.n += len(net)
        now = preg.lactating.raw.copy()
        if checked.last is not None: # Who stopped in this step, among those alive: weaned (at the ti_weaned drawn), or early (an infant death, which sets it to that step)
            n = len(checked.last)
            ended = np.flatnonzero(checked.last & ~now[:n] & sim.people.alive.raw[:n])
            early = preg.ti_weaned.raw[ended] == np.round(preg.ti_weaned.raw[ended])
            checked.weaned += (~early).sum()
            checked.stopped += early.sum()
        checked.last = now
    sim = ss.Sim(**make(), analyzers=check, n_agents=10_000, **kw).run()
    r, ppl, hiv = sim.results, sim.people, sim.diseases.d
    preg = sim.demographics.pregnancy
    moms = preg.ti_live_birth.notnan.uids
    assert len(moms) > 500 and np.all(preg.ti_weaned.notnan[moms]) # p_breastfeed=1: every mother breastfed
    assert checked.weaned > 500 and checked.stopped > 10 and checked.n > 0
    kids = ppl.parent.notnan.uids
    infected = kids[hiv.I[kids]]
    assert len(infected) > 50 # In utero and by breastfeeding
    lam = ss.Sim(**make(lam=True), n_agents=10_000, **kw).run()
    assert lam.results.pregnancy.conception.sum() < 0.99*r.pregnancy.conception.sum() # Few conceive within 6 months of a birth anyway
    return sim


@sc.timer()
def test_lactation_subclass():
    sc.heading('Testing breastfeeding with a subclass of ss.Pregnancy that has another name')
    class FP(ss.Pregnancy):
        def __init__(self, **kwargs):
            kwargs.setdefault('name', 'fp')
            super().__init__(**kwargs)
    sim = ss.Sim(demographics=FP(fertility_rate=ss.freqperyear(150)), networks=[ss.MaternalNet(), ss.BreastfeedingNet()], n_agents=5000, **kw).run()
    r = sim.results
    assert r.fp.birth.sum() > 100 and len(sim.networks.breastfeedingnet) > 0 # Found by type rather than by the name "pregnancy"
    return sim


@sc.timer()
def test_code_pars():
    sc.heading('Testing parameters a subclass of a declared module defines and reads in its own code')
    class FP(ss.Pregnancy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(age_scale=0.5) # Read by conception_rate(), not by the graph
        def conception_rate(self, sim):
            return super().conception_rate(sim)*self.pars.age_scale
    base = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), n_agents=5000, **kw).run()
    half = ss.Sim(demographics=FP(fertility_rate=ss.freqperyear(100)), n_agents=5000, **kw).run()
    assert half.results.pregnancy.conception.sum() < 0.7*base.results.pregnancy.conception.sum()
    with pytest.raises(KeyError, match='has no parameter named age_scal'): # A misspelled parameter from the user is still an error
        FP(age_scal=0.3)
    preg = ss.Pregnancy()
    preg.pars['rel_fertilty'] = 2 # ... and so is one added later that nothing reads
    with pytest.raises(KeyError):
        ss.Sim(demographics=preg, n_agents=500, **kw).init()
    return half


def make_hiv_vertical(bf=True):
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(120), p_breastfeed=0.8)
    S, I = ss.states('S I')
    beta = dict(maternalnet=ss.peryear(0.2), breastfeedingnet=ss.peryear(0.3 if bf else 0))
    hiv = ss.Disease(S, I, infection=ss.Transition(S, I, rate=beta * I), death=ss.Transition(I, ss.null, rate=ss.peryear(0.05)), init=dict(I=0.2), name='hiv')
    return dict(demographics=preg, diseases=hiv, networks=[ss.MaternalNet(), ss.BreastfeedingNet()])


@sc.timer()
def test_hiv_vertical():
    sc.heading('Port: HIV vertical transmission, in utero over MaternalNet and by breastfeeding (04 §5.3)')
    common = dict(stratify=strat, use_aging=False, **kw)
    inf = [ss.Sim(**make_hiv_vertical(bf), n_agents=50_000, **common).run().results.hiv.infection.sum() for bf in [False, True]]
    utero, post = inf[0], inf[1] - inf[0]
    assert utero > 100 and post > 100, f'{utero:.0f} in utero, {post:.0f} by breastfeeding'
    return inf


@sc.timer()
def test_starsim3():
    sc.heading('Testing back-compatibility: ss.Pregnancy has the Starsim 3 names, results, and callbacks')
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_maternal_death=ss.bernoulli(0.01), burnin=False)
    S, I, C, F = ss.states('S I C F')
    d = ss.Disease(S, I, C, F,
        infection = S >> I @ ss.peryear(0.1),
        mtct      = S >> C @ ss.peryear(3)*ss.mother(I),
        flag      = S >> F @ ss.on(preg.conception),
        unflag    = F >> S @ ss.on(preg.delivery), name='d')
    seen = sc.obj(conceived=0, delivered=0)
    def count(sim):
        if sim.ti == 0: # Callbacks, for modules written as code
            p = sim.demographics.pregnancy
            p.add_callback(p.conception, lambda uids: seen.__setitem__('conceived', seen.conceived + len(uids)))
            p.add_callback(p.birth, lambda kids: seen.__setitem__('delivered', seen.delivered + len(kids)))
    sim = ss.Sim(demographics=preg, diseases=d, networks=ss.MaternalNet(), analyzers=count, n_agents=5000, **kw).run()
    r, p = sim.results, sim.demographics.pregnancy
    assert r.d.flag.sum() > 0.5*r.pregnancy.conception.sum() and r.d.unflag.sum() > 0.5*r.d.flag.sum()
    assert r.d.mtct.sum() > 100 and r.pregnancy.maternal_deaths.sum() > 0
    assert np.all(np.ceil(p.ti_exit_P[p.P.uids]) >= p.ti) and np.all(p.parity >= 0) # Each due at the end of this step or later
    assert np.max((p.ti - p.ti_P[p.P.uids])*p.t.dt.weeks) < 45 # Weeks of gestation
    assert np.all(p.rel_ptb[p.P.uids] > 0)
    assert seen.conceived > 0.8*r.pregnancy.conception.sum() and seen.delivered > 0.8*r.pregnancy.birth.sum()
    born = (sim.people.age > 0) & np.isfinite(p.gestation_at_birth)
    assert born.sum() > 100 and np.isclose(r.pregnancy.n_preterm.sum(), np.sum(p.gestation_at_birth[born.uids] < 37), rtol=0.2)

    # A Starsim 3 disease that transmits over the maternal network
    sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), diseases=ss.SIS(beta=dict(maternalnet=1.0, random=0.05)),
                 networks=[ss.MaternalNet(), ss.RandomNet()], n_agents=3000, **kw).run()
    assert sim.demographics.pregnancy.fetus and sim.results.sis.infection.sum() > 0
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


def test_events_ode():
    def flags(solver):
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), loss_rate=ss.probperweek(0.01))
        S, F = ss.states('S F')
        d = ss.Disease(S, F, flag=S >> F @ ss.on(preg.conception), unflag=F >> S @ ss.on(preg.delivery), name='d')
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        return ss.Sim(demographics=preg, diseases=d, solver=solver, stratify=strat, use_aging=False, **size, **kw).run().results
    ro = flags('ode')
    ra = flags('abm')
    for key in ['flag', 'unflag']:
        assert np.isclose(ro.d[key].sum(), ra.d[key].sum(), rtol=0.05), f'{key}: {ro.d[key].sum():.0f} as an ODE, {ra.d[key].sum():.0f} as agents'


def make_postpartum(**kwargs):
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(150), loss_rate=ss.probperweek(0.02), **kwargs)
    preg.stratify(parity=['0', '1', '2', '3+'], split=[1, 0, 0, 0])
    return preg


@sc.timer()
def test_postpartum():
    sc.heading('Port: FPsim postpartum, the time since the last birth surviving later losses, and parity as a counter moved on delivery')
    seen = sc.obj(n=0)
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


def test_postpartum_ode():
    by = lambda sim: sum(np.asarray(sim.results.pregnancy[k].select(sex='female').by('parity').values)[-1] for k in ['NP', 'P'])
    abm = ss.Sim(demographics=make_postpartum(burnin=False), n_agents=20_000, stratify=strat, use_aging=False, **kw).run() # Burn-in as compartments is B5's
    ode = ss.Sim(demographics=make_postpartum(burnin=False), solver='ode', total_pop=20_000, stratify=strat, use_aging=False, **kw).run()
    a, o = by(abm), by(ode)
    assert np.allclose(a[1:], o[1:], rtol=0.1), (a, o)


def test_save_load():
    """ A pregnancy model saved and loaded mid-run (fetuses, events, memory, the engine's state) ends as if it had not been """
    sc.heading('Testing save and load of a pregnancy model mid-run')
    sim = ss.Sim(**make_syphilis(), n_agents=5000, **syph_kw)
    sim.init()
    sim.run(until=12)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'preg.sim')
        sim.save(path)
        loaded = ss.load(path)
    loaded.run()
    sim.run()
    for key in ['mtct', 'stillbirth', 'born_cs']:
        assert np.array_equal(sim.results.syph[key].values, loaded.results.syph[key].values), key
    assert np.array_equal(sim.results.pregnancy.delivery.values, loaded.results.pregnancy.delivery.values)
    return loaded


def test_fertility_function():
    """ FP13: each of the three rates is a parameter, which can be a function of (sim, uids); conception_rate() is fertility's default """
    def fert(sim, uids):
        ppl = sim.people
        return ss.peryear(np.where(ppl.female[uids] & (ppl.age[uids] > 15) & (ppl.age[uids] < 45), 0.3, 0.0))
    out = {}
    for label,preg in [('default', ss.Pregnancy()), ('function', ss.Pregnancy(fertility=fert))]:
        sim = ss.Sim(demographics=preg, n_agents=2000, rand_seed=1, **(kw | dict(stop=2003))).run()
        out[label] = sim.results.pregnancy.conception.sum()
    assert out['function'] > 1.5*out['default'] # 0.3 a year, about three times the default's
    return out


@sc.timer()
def test_coded_reads():
    sc.heading('Testing what code reads of the pregnancy: the delivery before the step\'s conceptions (FP2), its memory as an array (FP9), ss.mother() of children (FP10), one rate as data (ST2), and a preterm redraw (ST5)')
    # A woman who delivers faces conception again in the step, after the delivery's records
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(300))
    lam = ss.modify(preg.conception, multiply=0, where=preg.since_birth < ss.months(3))
    sim = ss.Sim(n_agents=3000, demographics=preg, interventions=lam, start=2000, stop=2004, dt=ss.months(1), verbose=0, rand_seed=1)
    same = []
    sim.init()
    p = sim.demographics.pregnancy
    for ti in range(sim.t.npts - 1):
        sim.run_one_step()
        same.append((p.P & p.ti_P.eq(p.ti_delivered)).sum())
    assert sum(same) == 0 and p.results.birth.sum() > 300, same
    assert not any('delivery: people leave after' in str(note) for note in sim.notes) # A dwell of 32 to 42 weeks loses nobody to monthly steps (FP7)

    # Its memory and conditions as arrays, with the names declarations use
    uids = p.ti_live_birth.notnan.uids
    assert np.allclose(p.since_birth[uids], sim.ti - p.ti_live_birth[uids]) # In the sim's unit (months)
    assert np.array_equal((p.P & p.lactating)[uids], p.P[uids] & p.lactating[uids])
    assert len(p.since_birth.values) == len(sim.people) and np.all(np.isinf(p.since_birth[p.ti_live_birth.isnan.uids]))

    # A preterm risk for one redraw, not kept as her baseline; and a delivery due in the past is taken in the next step
    moms = p.P.uids
    base = p.rel_ptb_base[moms].copy()
    weeks = p.dur_gestation[moms].copy()
    p.preterm_risk(moms, rel=20)
    assert np.array_equal(p.rel_ptb_base[moms], base) and np.mean(p.dur_gestation[moms] < 37) > np.mean(weeks < 37) + 0.2
    p.set_delivery(moms[:5], sim.ti - 3)
    assert np.all(p.ti_exit_P[moms[:5]] == sim.ti + 1)

    # ss.mother() in the infant rate reads the children's mothers, with no fetuses as agents
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(300), infant_rate=ss.peryear(0.5)*ss.mother(ss.ppl.age > 30))
    sim = ss.Sim(n_agents=2000, demographics=preg, start=2000, stop=2003, dt=ss.months(1), verbose=0).run()
    assert not sim.demographics.pregnancy.fetus and sim.results.pregnancy.infant_mortality.sum() > 20

    # One fertility rate is a rate per woman-year, rescaled to the women not pregnant, as data are
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100))
    sim = ss.Sim(n_agents=5000, demographics=preg, start=2000, stop=2001, dt=ss.months(1), verbose=0).init()
    p, ppl = sim.demographics.pregnancy, sim.people
    fecund = ppl.female.raw & (ppl.age.raw >= 15) & (ppl.age.raw < 50) & ppl.alive.raw
    rate = np.asarray(p.conception_rate(sim).value)
    expected = 0.1*fecund.sum()/(fecund & ~p.P.raw).sum()
    assert np.allclose(rate[fecund], expected) and np.all(rate[~fecund] == 0)
    return sim


if __name__ == '__main__':
    T = sc.timer()
    abm = test_states()
    births = test_births()
    comp = test_compartmental()
    burnin = test_burnin()
    events = test_events()
    mom = test_mother_cond()
    fetus = test_fetus()
    lact = test_lactation()
    lact_sub = test_lactation_subclass()
    code = test_code_pars()
    hiv = test_hiv_vertical()
    starsim3 = test_starsim3()
    syph = test_syphilis()
    test_events_ode()
    pp = test_postpartum()
    test_postpartum_ode()
    saved = test_save_load()
    test_fertility_function()
    reads = test_coded_reads()
    T.toc()
