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
def test_events():
    sc.heading('Testing preg.* events: ss.on() in another module, on= on deliveries, ss.child(), and ANC at a visit after conception')
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_loss=ss.bernoulli(0.01))
    anc_ga = preg.attr('anc_ga', ss.uniform(ss.months(1), ss.months(8))) # Per pregnancy
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


if __name__ == '__main__':
    T = sc.timer()
    abm = test_states()
    births = test_births()
    events = test_events()
    T.toc()
