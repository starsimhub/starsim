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


def test_burnin_ode():
    sc.heading('Testing burnin= in the small compartmental form: women pregnant at the start, as many as agents have, with a note')
    pars = dict(start=2000, stop=2003, dt=ss.months(1), verbose=0, stratify=strat, use_aging=False)
    make = lambda: ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_abortion=0.1)
    ode = ss.Sim(demographics=make(), solver='ode', total_pop=20_000, **pars).run()
    abm = ss.Sim(demographics=make(), n_agents=20_000, **pars).run()
    assert np.isclose(ode.results.pregnancy.P[0], abm.results.pregnancy.P[0], rtol=0.1) and any('burnin=True' in str(note) for note in ode.notes)
    assert ss.Sim(demographics=make(), total_pop=20_000, **pars).init().solver == 'abm'


@sc.timer()
def test_burnin_pregnancies():
    sc.heading('Testing burnin=: women pregnant at the start count their pregnancies, so deliveries plus losses are pregnancies')
    seen = sc.objdict(start=None, lost=None)
    def check(sim):
        p = sim.demographics.pregnancy
        if sim.ti == 0:
            preg = p.P.uids
            seen.start = (len(preg), p.n_pregnancies[preg], p.parity[preg])
            seen.lost = np.zeros(len(sim.people.uid.raw) + 100_000)
        for tr in ['abortion', 'miscarriage', 'stillbirth']: # Each woman's losses in the run
            uids = p.engine.fresh(f'test_{tr}', tr, unique=False)
            np.add.at(seen.lost, uids, 1)

    # Women pregnant at the start count that pregnancy
    ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), analyzers=check, n_agents=2000, verbose=0).init().run(until=0)
    n, npreg, parity = seen.start
    assert n > 10 and np.all(npreg == 1 + parity), f'Women pregnant at the start have {np.unique(npreg)} pregnancies'

    # With deliveries before the start (short pregnancies), counted in parity, and losses in the run
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(600), dur_pregnancy=ss.uniform(ss.weeks(10), ss.weeks(70)), p_abortion=0.1, p_loss=ss.bernoulli(0.01))
    preg.stratify(parity=['0', '1', '2', '3+'], split=[1, 0, 0, 0])
    sim = ss.Sim(demographics=preg, analyzers=check, n_agents=10_000, start=2000, stop=2004, dt=ss.months(1), verbose=0).run()
    n, npreg, parity = seen.start
    assert np.all(npreg == 1 + parity) and np.sum(parity > 0) > 10, 'A delivery before the start counts as a pregnancy too'
    p, ppl = sim.demographics.pregnancy, sim.people
    alive = ppl.alive.uids
    lost = seen.lost[alive]
    assert lost.sum() > 50
    assert np.array_equal(p.n_pregnancies[alive], p.parity[alive] + lost + p.P[alive]), 'For each woman, deliveries plus losses (plus a current pregnancy) are her pregnancies'
    return sim


@sc.timer()
def test_mmr():
    sc.heading('Testing the maternal mortality ratio, and infant mortality')
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100), p_maternal_death=ss.bernoulli(0.05), p_infertile=ss.bernoulli(0.2), infant_rate=ss.peryear(0.05))
    sim = ss.Sim(demographics=preg, n_agents=5000, start=2000, stop=2010, dt=ss.months(1), verbose=0).run()
    r = sim.results.pregnancy
    births, md = r.birth.values, r.maternal_deaths.values
    assert md.sum() > 30
    assert np.allclose(r.mmr.values, sc.safedivide(md, births, default=0)*100e3), 'MMR: maternal deaths per 100,000 births'
    assert r.mmr.summarize_by == 'mean'
    mmr = md.sum()/births.sum()*100e3
    assert 3500 < mmr < 6500, f'An MMR of {mmr:.0f} per 100,000 from a 5% chance of death at delivery'
    imr = r.infant_mortality.sum()/births.sum()
    assert 0.02 < imr < 0.06, f'{imr:.3f} of children died before the age of 1 at a rate of 0.05 per year (less for the last year)'

    # The reproduction: no KeyError
    sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), n_agents=500, start=2000, stop=2002).run()
    assert len(sim.results.pregnancy.mmr) == len(sim.timevec)
    return sim


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


def agents_only(make, match):
    """ A model that only agents run: solver='auto' chooses them, and a compartmental solver named explicitly stops before the run, saying why """
    assert ss.Sim(**make(), total_pop=1000, **kw).init().solver == 'abm'
    with pytest.raises(ss.CapabilityError, match=match):
        ss.Sim(**make(), solver='ode', total_pop=1000, **kw).init()
    return


def test_fetus_ode():
    sc.heading('Testing fetuses (ss.mother() and a maternal network): agents only')
    agents_only(lambda: dict(zip(['demographics', 'diseases'], make_fetal()), networks=ss.MaternalNet()), match='mother')
    agents_only(lambda: dict(zip(['demographics', 'diseases'], make_fetal())), match='mother') # ss.mother() alone
    S, I = ss.states('S I')
    agents_only(lambda: dict(demographics=ss.Pregnancy(), diseases=ss.Disease(infection=ss.Transition(S, I, rate=dict(maternalnet=0.3) * I)), networks=ss.MaternalNet()), match='MaternalNet')
    agents_only(lambda: dict(demographics=ss.Pregnancy(fetus=True)), match='fetus=True')


def test_lactation_ode():
    sc.heading('Testing ss.Lactation: agents only')
    agents_only(lambda: dict(demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(150)), ss.Lactation(p_breastfeed=0.8)]), match='Lactation')


def make_hiv_vertical(bf=True):
    preg = ss.Pregnancy(fertility_rate=ss.freqperyear(120))
    S, I = ss.states('S I')
    beta = dict(maternalnet=ss.peryear(0.2), breastfeedingnet=ss.peryear(0.3 if bf else 0))
    hiv = ss.Disease(S, I, infection=ss.Transition(S, I, rate=beta * I), death=ss.Transition(I, ss.null, rate=ss.peryear(0.05)), init=dict(I=0.2), name='hiv')
    return dict(demographics=[preg, ss.Lactation(p_breastfeed=0.8)], diseases=hiv, networks=[ss.MaternalNet(), ss.BreastfeedingNet()])


def test_hiv_vertical_ode():
    sc.heading('Testing HIV vertical transmission (MaternalNet, BreastfeedingNet, ss.Lactation): agents only')
    agents_only(make_hiv_vertical, match='Lactation')


def test_delivery_shift():
    """ set_delivery() reschedules deliveries, as ssl.mnch.FetalHealth does """
    sc.heading('Testing rescheduled deliveries')
    sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100), burnin=False), n_agents=3000, **kw)
    sim.init()
    sim.run(until=24)
    p = sim.demographics.pregnancy
    uids = p.P.uids
    due = np.ceil(p.ti_exit_P[uids])
    later = uids[due > p.ti + 3]
    p.set_delivery(later, p.ti + 1) # Due at the next step
    assert np.all(np.ceil(p.ti_exit_P[later]) == p.ti + 1)
    sim.run(until=sim.ti + 2)
    assert not np.any(p.P[later]) and np.all(p.ti_delivered[later] >= p.ti - 2), 'Every rescheduled delivery happened'
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


def test_syphilis_ode():
    sc.heading('Testing congenital syphilis (ss.mother()): agents only')
    agents_only(make_syphilis, match='mother')


def test_two_sims():
    """ Each sim's pregnancy drives its births by its own delivery: ss.birth(per=) kept the class's transition, so a second sim reached into the first sim's objects """
    sc.heading('Testing two pregnancy sims in a row')
    sims = []
    for i in range(2):
        sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.freqperyear(100)), n_agents=300, start=2000, stop=2002, verbose=0).run()
        trs = sim.demographics.pregnancy.graph.transitions
        assert trs.birth.per.tr is trs.delivery and trs.delivery is not ss.Pregnancy.delivery
        assert trs.delivery.dur.raw.sim is sim
        sims.append(sim)
    assert sims[0].results.pregnancy.birth.sum() > 0 and sims[1].results.pregnancy.birth.sum() > 0
    return sims


def test_path_triggers():
    """ A trigger on another module's transition by path (ss.state('pregnancy.delivery')) is its event, as ss.on(preg.delivery) is, so it reaches fetuses at their mother's delivery (it was a condition that never held for them) """
    sc.heading('Testing triggers on a transition by path')
    def make(by_path):
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(100))
        S, I, Cs = ss.states('S I Cs')
        delivery = ss.state('pregnancy.delivery') if by_path else preg.delivery
        d = ss.Disease(S, I, Cs, infection=S >> I @ ss.peryear(0.1), mtct=ss.Transition(S, Cs, rate=ss.peryear(3)*ss.mother(I)),
                       still=ss.Transition(Cs, ss.null, on=delivery), name='d')
        return ss.Sim(demographics=preg, diseases=d, networks=ss.MaternalNet(), n_agents=5000, start=2000, stop=2005, verbose=0).run()
    direct, path = make(False), make(True)
    assert path.results.d.still.sum() > 0, 'A fetus is removed at its mother\'s delivery'
    assert np.array_equal(direct.results.d.still.values, path.results.d.still.values)
    return path


if __name__ == '__main__':
    T = sc.timer()
    test_burnin_ode()
    npreg = test_burnin_pregnancies()
    mmr = test_mmr()
    test_fetus_ode()
    test_lactation_ode()
    test_hiv_vertical_ode()
    shift = test_delivery_shift()
    test_syphilis_ode()
    two = test_two_sims()
    paths = test_path_triggers()
    T.toc()
