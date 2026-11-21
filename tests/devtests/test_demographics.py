"""
Test demographics: the declared ss.Pregnancy (states, conception, outcomes by gestational age, births, events, and lactation), and its risks
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(start=2000, stop=2010, dt=ss.months(1), verbose=0)
strat = dict(sex=['female', 'male'], age={0: 0.3, 15: 0.4, 50: 0.3})


@sc.timer()
def test_states():
    sc.heading('Testing ss.Pregnancy: abortion, and losses by gestational age')
    thr = ss.weeks(20)
    def make():
        return ss.Pregnancy(fertility_rate=ss.peryear(0.1), p_abortion=0.1, loss_rate=ss.probpermonth(0.02), loss_threshold=thr)

    ga = sc.obj(miscarriage=[], stillbirth=[], seen=set())
    def check(sim): # Each loss is classified by the gestational age it happened at
        preg = sim.demographics.pregnancy
        for b in preg.engine.batches:
            key = (b.tr, b.key, tuple(b.uids))
            if b.tr in ['miscarriage', 'stillbirth'] and key not in ga.seen:
                ga.seen.add(key)
                step = b.key[0] # The index of the step it happened in (the batch is dated at the step's start, which is its index)
                uids = b.uids[preg.ti_P[b.uids] < step] # Not those who conceived again in the same step, whose time of entering P is now that of the new pregnancy
                ga[b.tr] += list((step - preg.ti_P[uids])*preg.t.dt.weeks) # At the step it happened
        assert not np.any(preg.P & ~sim.people.female), 'only women are pregnant'

    abm = ss.Sim(demographics=make(), analyzers=check, n_agents=20_000, stratify=strat, aging=False, **kw).run()
    ra = abm.results.pregnancy
    assert len(ga.miscarriage) > 100 and len(ga.stillbirth) > 100
    assert max(ga.miscarriage) < thr.weeks <= min(ga.stillbirth)
    assert np.isclose(ra.abortion.sum()/ra.conception.sum(), 0.1, atol=0.02) # At once, as a branch on entry

    return abm


@sc.timer()
def test_compartmental():
    sc.heading('Testing the small compartmental form against agents, and its conservation of people')
    thr = ss.weeks(20)
    make = lambda: ss.Pregnancy(fertility_rate=ss.peryear(0.1), p_abortion=0.1, loss_rate=ss.probpermonth(0.02), loss_threshold=thr)
    assert isinstance(ss.Sim(demographics=make(), total_pop=20_000, **kw).init().solver, ss.solvers.abm) # Only when a compartmental solver is named
    ra = ss.Sim(demographics=make(), n_agents=20_000, stratify=strat, aging=False, **kw).run().results.pregnancy
    ode = ss.Sim(demographics=make(), solver='ode', total_pop=20_000, stratify=strat, aging=False, **kw).run()
    ro = ode.results.pregnancy
    loss = lambda r: r.miscarriage.sum() + r.stillbirth.sum()
    for key,f in [('conception', lambda r: r.conception.sum()), ('abortion', lambda r: r.abortion.sum()), ('delivery', lambda r: r.delivery.sum()), ('birth', lambda r: r.birth.sum()), ('loss', loss), ('P at the start', lambda r: r.P[0])]:
        assert np.isclose(f(ra), f(ro), rtol=0.1), f'{key}: agents {f(ra):.0f}, ODE {f(ro):.0f}'
    assert np.isclose(ro.miscarriage.sum()/loss(ro), thr.weeks/40.09, rtol=0.01) # The share of the mean gestation below the threshold, as nothing reads P.time_in
    assert any('small compartmental form' in str(note) for note in ode.notes)

    # People are conserved: the start, plus births, less maternal deaths
    sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.1), maternal_death_rate=ss.peryear(0.05), children=ss.choice(a=[1, 2], p=[0.8, 0.2])),
                 solver='binomial', total_pop=20_000, stratify=strat, **kw).run()
    r = sim.results.pregnancy
    assert r.antepartum.sum() > 0 and r.birth.sum() > 1.1*r.delivery.sum() # Twins
    assert np.isclose(r.NP[-1] + r.P[-1], r.NP[0] + r.P[0] + r.birth[:-1].sum() - r.antepartum[:-1].sum()) # The last row is the start of the last step, before its flows
    return sim


@sc.timer()
def test_fertility():
    sc.heading('Testing the TFR and ASFR, with women pregnant at the start (burnin=)')
    abm = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.1), p_abortion=0.1), n_agents=20_000, start=2000, stop=2003, dt=ss.months(1), verbose=0, stratify=strat, aging=False).run() # The TFR: about the fertility rate over the ages women can conceive at, as in Starsim 3
    tfr = abm.results.pregnancy.tfr[12:24].mean() # Per year at each step, whatever the step
    assert 2.5 < tfr < 3.5, f'TFR {tfr}'
    assert abm.results.pregnancy.tfr.kind == 'ratio' # So it is averaged over time, not summed
    asfr = abm.demographics.pregnancy.asfr
    assert asfr.shape == (11, len(abm.timevec)) and asfr[:3, 23].sum() == 0 and asfr[4, 23] > 50 # None under 15; per 1000 women, over a year

    return abm


@sc.timer()
def test_events():
    sc.heading('Testing preg.* events: ss.on() in another module, and the mothers of births')
    preg = ss.Pregnancy(fertility_rate=ss.peryear(0.1), loss_rate=ss.probperweek(0.01))
    S, F, I = ss.states('S F I')
    d = ss.Disease(S, F, I,
        flag      = S >> F @ ss.on(preg.conception), # Edge-triggered, in the same step, whatever the module order
        unflag    = F >> S @ ss.on(preg.delivery),
        infection = S >> I @ ss.peryear(0.05),
        recovery  = I >> S @ ss.years(1),
        name='d')
    seen = sc.obj(child=0, flagged=0, fresh=0, gap=[])
    def check(sim):
        p = sim.demographics.pregnancy
        moms = np.unique(sim.people.parent[p.birth.uids]) # The mothers of the children born
        assert set(moms) == set(p.live_birth.uids)
        seen.child += len(moms)
        lost = p.loss.uids # Since the last live birth, through a later loss
        sb = p.engine.eval_for(p.since_birth, lost)
        seen.gap += list(sb[np.isfinite(sb)])
    sim = ss.Sim(demographics=preg, diseases=d, analyzers=check, n_agents=10_000, **kw).run()
    r = sim.results
    assert 0.8*r.pregnancy.conception.sum() < r.d.flag.sum() <= r.pregnancy.conception.sum()
    assert r.d.unflag.sum() <= r.pregnancy.delivery.sum() and r.d.unflag.sum() > 0.8*r.pregnancy.delivery.sum()
    assert seen.child == r.pregnancy.delivery.sum() # Every delivery is a live birth here
    assert len(seen.gap) and min(seen.gap) > 0 # A loss does not reset the time since the last live birth
    with pytest.raises(ss.CapabilityError): # Agents only
        ss.Sim(demographics=(p := ss.Pregnancy(fertility_rate=ss.peryear(0.1))), connectors=ss.modify(p.conception, multiply=0.5, who=p.since_birth < ss.years(1)),
               solver='ode', total_pop=1000, stratify=strat, **kw).run()

    return sim


def make_fetal():
    preg = ss.Pregnancy(fertility_rate=ss.peryear(0.1), maternal_death_rate=ss.peryear(0.05))
    S, I, C, Cs = ss.states('S I C Cs')
    d = ss.Disease(S, I, C, Cs,
        infection = S >> I @ ss.peryear(0.1),
        mtct      = ss.Transition(S, {C: 0.6, Cs: None}, rate=ss.peryear(3)*ss.mother(I)), # In utero, from the mother's state
        fetal     = C >> ss.dead @ ss.peryear(4), # A fetal death before delivery: a loss for the mother, by gestational age
        still     = Cs >> ss.dead @ ss.on(preg.delivery), # At delivery, before birth: a stillbirth
        born      = Cs >> S @ ss.on(preg.birth),
        name='d')
    return preg, d


@sc.timer()
def test_fetus():
    sc.heading('Testing fetuses as agents (fetal_agents="auto"), ss.mother(), and fetal removals as losses')
    make = make_fetal
    preg, d = make()
    sim = ss.Sim(demographics=preg, diseases=d, networks=ss.PrenatalNet(), n_agents=10_000, **kw).run()
    p, ppl, r = sim.demographics.pregnancy, sim.people, sim.results
    assert p.fetal_agents and 'ss.mother in d.mtct' in p.fetus_reason
    assert r.d.mtct.sum() > 200 and r.d.still.sum() > 50
    assert r.pregnancy.stillbirth.sum() >= r.d.still.sum() and r.pregnancy.miscarriage.sum() > 50
    assert r.pregnancy.birth.sum() == r.pregnancy.delivery.sum() - r.d.still.sum() # A stillborn fetus is not born
    fetal = (p.fetal & ppl.alive).uids
    orphan = ~ppl.alive[ppl.parent[fetal]] # Her mother died in the last step: lost at the next, which the run does not take
    assert np.all(ppl.ti_dead[fetal[orphan]] == sim.ti) and np.all(p.P[ppl.parent[fetal[~orphan]]]) and np.all(ppl.age[fetal] < 0) # Every fetus has a pregnant mother, and is lost with her
    assert len(sim.networks.prenatalnet) == (~orphan).sum() # An orphan's edge went with her mother
    born, unborn = (ppl.alive & (ppl.age >= 0)).sum(), (ppl.alive & (ppl.age < 0)).sum()
    assert np.isclose(r.d.S[-1] + r.d.I[-1] + r.d.C[-1] + r.d.Cs[-1], born + unborn, rtol=0.01) and np.isclose(r.alive[-1], born, rtol=0.01) # The unborn are in the disease's states (so they can be infected before birth), though not among the people

    # A mother who dies of another cause loses her pregnancy, whether or not fetuses are agents, and fetuses don't face background deaths
    losses = []
    for fetus in [False, True]:
        r = ss.Sim(demographics=[ss.Pregnancy(fertility_rate=ss.peryear(0.15), fetal_agents=fetus), ss.Deaths(death_rate=ss.peryear(0.1))], n_agents=5000, rand_seed=1, **kw).run().results.pregnancy
        losses.append(r.miscarriage.sum() + r.stillbirth.sum())
    assert losses[0] > 50 and losses[0] == losses[1]

    return sim


@sc.timer()
def test_lactation():
    sc.heading('Testing breastfeeding (preg.lactating): started by live births, ended by weaning or an infant death; and BreastfeedingNet')
    def make():
        preg = ss.Pregnancy(fertility_rate=ss.peryear(0.15))
        S, I = ss.states('S I')
        d = ss.Disease(S, I,
            infection = ss.Transition(S, I, rate=dict(prenatalnet=0.3, breastfeedingnet=0.5) * I),
            death     = ss.Transition(I, ss.dead, rate=ss.peryear(2), who=ss.person.age < 1), # Infant deaths end breastfeeding
            init=dict(I=0.3), name='d')
        return dict(demographics=preg, diseases=d, networks=[ss.PrenatalNet(), ss.BreastfeedingNet()])
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
    return sim


@sc.timer()
def test_coded_reads():
    sc.heading('Testing what code reads of the pregnancy: the delivery before the step\'s conceptions, and that no delivery is lost to monthly steps')
    # A woman who delivers faces conception again in the step, after the delivery's records
    preg = ss.Pregnancy(fertility_rate=ss.peryear(0.3))
    lam = ss.modify(preg.conception, multiply=0, who=preg.since_birth < ss.months(3))
    sim = ss.Sim(n_agents=3000, demographics=preg, interventions=lam, start=2000, stop=2004, dt=ss.months(1), verbose=0, rand_seed=1)
    same = []
    sim.init()
    p = sim.demographics.pregnancy
    for ti in range(sim.t.npts - 1):
        sim.run_one_step()
        same.append((p.P & p.ti_P.eq(p.ti_delivered)).sum())
    assert sum(same) == 0 and p.results.birth.sum() > 300, same
    assert not any('delivery: people leave after' in str(note) for note in sim.notes) # A dwell of 32 to 42 weeks loses nobody to monthly steps

    return sim


@sc.timer()
def test_pregnancy_risks():
    """ A chance per week of a loss is per week whatever the time step, and the relative risk of preterm birth changes the share born preterm """
    sc.heading('Testing pregnancy risks...')
    def run(dt, p_loss=0, rr=1.0):
        preg = ss.Pregnancy(fertility_rate=ss.peryear(0.15), loss_rate=ss.probperweek(p_loss), preterm=ss.preterm(rel=ss.constant(1), by_age=[[18, 35, 1000], [rr, rr, rr]]), burnin=False)
        sim = ss.Sim(n_agents=10_000, start=2000, stop=2003, dt=dt, demographics=preg, rand_seed=1, verbose=0).run()
        p = sim.demographics.pregnancy
        ga = p.gestation_at_birth.raw[~np.isnan(p.gestation_at_birth.raw)]
        return np.mean(ga < 37), 1 - sim.results.pregnancy.birth.sum()/np.nansum(p.n_pregnancies.raw)

    weekly, monthly = [run(dt, p_loss=0.01)[1] for dt in [ss.weeks(1), ss.months(1)]]
    assert abs(weekly - monthly) < 0.04, f'The share lost should not depend on the time step: {weekly:.3f} weekly, {monthly:.3f} monthly'

    base, triple = [run(ss.weeks(1), rr=rr)[0] for rr in [1, 3]]
    p_pre = 0.001 + 0.002 + 0.005 + 0.012 + 0.026 # The default distribution's preterm durations, 32 to 36 weeks
    assert abs(base - p_pre) < 0.02 and abs(triple - 3*p_pre) < 0.03, f'Expected preterm shares of about {p_pre:.3f} and {3*p_pre:.3f}, not {base:.3f} and {triple:.3f}'
    return base, triple


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
