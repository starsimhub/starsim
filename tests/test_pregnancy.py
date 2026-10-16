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
                ga[b.tr] += list((b.key[0] - preg.ti_P[b.uids])*preg.t.dt.weeks) # At the step it happened
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


if __name__ == '__main__':
    T = sc.timer()
    abm = test_states()
    T.toc()
