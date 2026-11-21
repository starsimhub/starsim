"""
Test demographics: births, deaths, migration, and the declared ss.Pregnancy
"""

import warnings
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.library as ssl
import pytest

sc.options(interactive=False) # Assume not running interactively
datadir = ss.root / 'tests/test_data'


@sc.timer()
def test_nigeria():
    sc.heading('Testing Nigeria demographics: the population size from UN birth and death rates')
    births = ss.Births(birth_rate=pd.read_csv(datadir/'nigeria_births.csv'), rate_per=1000)
    deaths = ss.Deaths(death_rate=pd.read_csv(datadir/'nigeria_deaths.csv'), rate_per=1)
    ppl = ss.People(5_000, age_data=pd.read_csv(datadir/'nigeria_age.csv'))
    sim = ss.Sim(dt=1, total_pop=106_819_805, start=1995, dur=ss.years(15), people=ppl, demographics=[births, deaths], verbose=0)
    sim.run()
    popsize = pd.read_csv(datadir/'nigeria_popsize.csv')
    data = popsize[popsize.year == 2010]
    assert np.isclose(data.alive.values[-1], sim.results.alive[-1], rtol=0.05), 'Final population size not within 5% of data'
    return sim


@sc.timer()
def test_aging():
    sc.heading('Testing aging: off by default, on with demographics, and set either way')
    kw = dict(n_agents=1000, verbose=0)
    orig = ss.Sim(**kw).init().people.age.raw.copy()
    for pars, ages in [(dict(), False), (dict(aging=True), True), (dict(demographics=True), True), (dict(demographics=True, aging=False), False)]:
        sim = ss.Sim(**kw, **pars).run()
        agent = sim.people.auids[0] # The first alive agent
        assert (sim.people.age.raw[agent] > orig[agent]) == ages, f'{pars}: aging should be {ages}'
    return sim


def test_pregnancy():
    sc.heading('Testing ss.Pregnancy: durations, and gestational age at birth')
    demog = [ss.Pregnancy(fertility_rate=ss.peryear(0.02), burnin=True), ss.Deaths(death_rate=ss.peryear(0.0099))]
    sim = ss.Sim(n_agents=5000, demographics=demog, dur=ss.years(10), dt=ss.years(1/12), rand_seed=1, networks=ss.PrenatalNet(), verbose=0).run()

    # Everyone pregnant conceived within the length of a pregnancy, and has a delivery time
    preg = sim.demographics.pregnancy
    weeks = ss.years((sim.ti - preg.ti_P[preg.P.uids]) * sim.t.dt_year).weeks # Since conception
    assert np.all((weeks >= 0) & (weeks <= 45)), 'Some pregnancies are longer than 45 weeks'
    assert np.all(~np.isnan(preg.ti_exit_P[preg.P.uids])), 'Some pregnant women have no scheduled delivery'

    # Everyone born during the sim has their gestational age at birth, about 40 weeks on average
    born_during_sim = (sim.people.age < sim.t.dur.years) & (sim.people.age > sim.t.dt.years)
    assert not (sim.people.pregnancy.gestation_at_birth.isnan & born_during_sim).any()
    mean_gestation = np.nanmean(sim.people.pregnancy.gestation_at_birth[born_during_sim])
    assert np.isclose(mean_gestation, 40, atol=2), f'Mean gestational age at birth is {mean_gestation:.1f} weeks, expected ~40 weeks'

    return sim


@sc.timer()
def test_period_rates():
    """ Rates labeled by the start of their period hold until the next label, and 5-year fertility data cover the last group whole """
    sc.heading('Testing rates by period...')

    # Death rates for 2010-2014 and from 2015, labeled 2010 and 2015: 2013 and 2014 have the 2010 rate, not the nearest year's
    data = pd.DataFrame(dict(Time=[2010, 2010, 2015, 2015], Sex=['Female', 'Male']*2, AgeGrpStart=0, mx=[0.04, 0.04, 0.0, 0.0]))
    sim = ss.Sim(n_agents=10_000, demographics=ss.Deaths(death_rate=data, rate_per=1), start=2010, stop=2015, dt=ss.months(1), verbose=0).run()
    deaths = sim.results.death.values[:-1] # The step from 2015 has the 2015 rate, 0
    assert deaths.min() > 0, 'Every step before 2015 has the 2010 rate'
    expected = 10_000*(1 - np.exp(-0.04*5))
    assert abs(deaths.sum() - expected) < 4*np.sqrt(expected), f'Expected about {expected:.0f} deaths, not {deaths.sum()}'

    # Fertility by 5-year age group: the last group (45) is 45-49, not just 45
    data = pd.DataFrame(dict(Time=2010, AgeGrp=[15, 20, 25, 30, 35, 40, 45], ASFR=0.2))
    preg = ss.Sim(n_agents=100, demographics=ss.Pregnancy(fertility_rate=data, rate_per=1), start=2010, stop=2011, verbose=0).init().demographics.pregnancy
    frd = preg.fertility_rate_data
    assert list(frd.columns[-2:]) == [45, 50] and frd[45].iloc[0] == 0.2 and frd[50].iloc[0] == 0

    return sim


@sc.timer()
def test_rate_per():
    """ Rates of births, deaths, conceptions, and migration are per person unless rate_per= says otherwise, with an error for a rate that looks per 1,000; and they are read from the parameters at init """
    sc.heading('Testing rate_per, and changing the rates after a module is made...')
    kw = dict(n_agents=2000, start=2000, stop=2005, verbose=0)
    for rate, per, raises in [(0.02, None, False), (20, None, True), (20, 1000, False), (2, 1, False), (ss.perday(0.005), None, True)]:
        make = lambda: ss.Sim(demographics=ss.Births(birth_rate=rate, rate_per=per), **kw).init().demographics.births
        if raises: # A rate above 1 per person per year, without rate_per
            with pytest.raises(ValueError, match='rate_per=1000'):
                make()
        else:
            assert make().pars.rate_per == sc.ifelse(per, 1) # Never guessed from the size
    with pytest.raises(ValueError, match='rate_per=1000'):
        ss.death(rate=10)

    # A place's UN rates: per 1,000 (births, fertility, migration) or per person (deaths by age), which is known, so no warning, and another rate_per is an error
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim = ss.Sim(demographics=[ss.Births(birth_rate='Kenya'), ss.Deaths(death_rate='Kenya')], **kw).init()
        assert sim.demographics.births.pars.rate_per == 1000 and sim.demographics.deaths.pars.rate_per == 1 and ss.Pregnancy(fertility_rate='Kenya').pars.rate_per == 1000

    # The rates are read when the sim is initialized, so a change to the parameters after the module is made counts
    deaths = []
    for rate in [0.01, 0.2]:
        mod = ss.Deaths(death_rate=0.01)
        mod.pars.death_rate = ss.peryear(rate)
        deaths.append(ss.Sim(demographics=mod, **kw).run().results.deaths.death.sum())
    assert deaths[1] > 5*deaths[0], f'Deaths at 0.2 a year ({deaths[1]}) should be far more than at 0.01 ({deaths[0]})'
    return deaths


@sc.timer()
def test_lifetable_deaths():
    """ Life tables by year and sex in ss.Deaths, and a death rate by its own ages, or a function, under both solvers """
    sc.heading('Testing life tables by year and sex, and death rates by single year of age...')
    ages = np.arange(0, 105, 5)
    table = lambda mx: ss.lifetable(ages, -np.diff(np.append(np.exp(-mx*ages), 0))) # A constant hazard, to age 105
    tables = {2000: dict(female=table(0.01), male=table(0.03)), 2005: dict(female=table(0.02), male=table(0.06))}
    sim = ss.Sim(n_agents=20_000, demographics=ss.Deaths(lifetable=tables), start=2000, stop=2010, verbose=0).run()
    female, dead = sim.people.female.raw, ~sim.people.alive.raw
    for sel,rates in [(female, [0.01, 0.02]), (~female, [0.03, 0.06])]:
        expected = 1 - np.exp(-5*rates[0] - 6*rates[1]) # The 2000 table for the steps from 2000 to 2004, then the 2005 one for those from 2005 to 2010
        assert abs(dead[sel].mean() - expected) < 4*np.sqrt(expected/sel.sum()), f'{dead[sel].mean():.3f} died, not {expected:.3f}'

    # A death rate by single year of age on 5-year age groups (which only agents read), and a function of the sim per year
    mx = ss.by(dict(age=np.arange(100), value=[ss.peryear(0.001*np.exp(0.07*a)) for a in range(100)]))
    def flat(sim): return 0.02
    def make(rate, solver, stratify, **pop):
        S, I = ss.states(2)
        sim = ss.Sim(ss.Disease(S, I, inf=S >> I @ 0*I, death=ss.death(rate=rate), name='d'), solver=solver, start=2000, stop=2005, verbose=0, **pop)
        if stratify:
            sim.stratify(age=list(range(0, 100, 5)))
        return sim
    with pytest.raises(ss.CapabilityError):
        make(mx, 'ode', True, total_pop=50_000).init()
    assert isinstance(make(mx, 'auto', True, total_pop=50_000).init().solver, ss.solvers.abm)
    coarse = ss.by(dict(age=np.arange(0, 100, 5), value=[ss.peryear(np.mean([0.001*np.exp(0.07*a) for a in range(lo, lo + 5)])) for lo in range(0, 100, 5)])) # The same, given by the groups
    for rate,fine,stratify in [(coarse, mx, True), (ss.peryear(flat), ss.peryear(flat), False)]:
        n = [make(r, solver, stratify, **pop).run().results.d.death.sum() for r,solver,pop in [(fine, 'abm', dict(n_agents=50_000)), (rate, 'ode', dict(total_pop=50_000))]]
        assert np.isclose(n[0], n[1], rtol=0.06), f'{n[0]:n} deaths as agents, {n[1]:n} as compartments'
    assert np.isclose(n[1], 50_000*(1 - np.exp(-0.02*6)), rtol=1e-3) # Six steps, from 2000 to 2005
    return sim


@sc.timer()
def test_births_parents():
    """ The parents of ss.Births and ss.birth() are women of the ages given, and newborns are born where their parent is """
    sc.heading('Testing the parents of births...')
    S, I = ss.states(2)
    dis = ss.Disease(onset=S >> I @ ss.peryear(0.01), name='dis')
    for births, ages in [(ss.Births(birth_rate=ss.peryear(0.04)), (15, 50)), (ss.birth(rate=0.04, mothers=True, min_age=20, max_age=30), (20, 30))]:
        sim = ss.Sim(diseases=dis.copy(), n_agents=5000, start=2000, stop=2003, verbose=0, stratify=dict(district=ss.geo.geography(names=['a', 'b', 'c'], populations=[4000, 3000, 3000])),
                     demographics=[births, ss.Deaths(death_rate=ss.peryear(0.01))]).run()
        ppl = sim.people
        kids = ss.uids(np.flatnonzero(ppl.parent.raw >= 0))
        parents = ppl.parent.raw[kids]
        assert len(kids) > 300 and ppl.female.raw[parents].all()
        both = ppl.alive.raw[kids] & ppl.alive.raw[parents] # Whose ages have both gone up since
        age_at_birth = ppl.age.raw[parents[both]] - ppl.age.raw[kids[both]]
        assert age_at_birth.min() >= ages[0] - 0.1 and age_at_birth.max() < ages[1] + 0.1 # As old as the women who can be mothers (to within a step)
        assert np.array_equal(ppl.district.raw[kids], ppl.district.raw[parents]) # Born where the parent is

    # The same for ss.Pregnancy, including the pregnancies before the start, and a dimension called sex is the newborn's
    sim = ss.Sim(diseases=dis.copy(), n_agents=5000, start=2000, stop=2003, verbose=0, stratify=dict(district=ss.geo.geography(names=['a', 'b', 'c'], populations=[4000, 3000, 3000]), sex=['female', 'male']),
                 demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.15))).run()
    ppl = sim.people
    kids = ss.uids(np.flatnonzero(ppl.parent.raw >= 0))
    assert np.array_equal(ppl.district.raw[kids], ppl.district.raw[ppl.parent.raw[kids]])
    assert np.array_equal(ppl.sex.raw[kids] == 0, ppl.female.raw[kids])
    return sim


@sc.timer()
def test_unborn():
    """ The unborn of ss.Pregnancy are not counted among the people, are not seeded, and a single beta on a maternal network goes from mother to child """
    sc.heading('Testing the unborn...')
    S = ss.State() # The first state, by its name, where people start unless put elsewhere
    I = ss.State(init=0.5)
    hiv = ss.Disease(infection=S >> I @ ss.par(dict(prenatalnet=0.33, breastfeedingnet=0.12)) * I, name='hiv')
    sim = ss.Sim(hiv, demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.15)), networks=[ss.PrenatalNet(), ss.BreastfeedingNet()],
                 n_agents=5000, start=2010, stop=2011, dt=ss.months(1), verbose=0).init()
    ppl, hiv = sim.people, sim.diseases.hiv
    unborn = ppl.unborn_uids()
    assert len(unborn) > 50 and not hiv.I.raw[unborn].any()
    mothers = ss.uids(ppl.parent.raw[unborn])
    I0 = hiv.I.raw.copy()
    sim.run()
    assert not (hiv.I.raw[mothers] & ~I0[mothers]).any()
    r = sim.results
    born = ppl.alive.raw[ppl.auids] & (ppl.age.raw[ppl.auids] >= 0)
    assert r.alive[-1] < r.alive[-1] + r.pregnancy.unborn[-1] and r.pregnancy.unborn[-1] > 0
    assert np.isclose(ppl.n, born.sum())
    return sim


@sc.timer()
def test_cause_of_death():
    """ Someone whose death two modules request in one step dies once, of one cause, counted once """
    sc.heading('Testing causes of death...')
    S, I = ss.states(2)
    sis = ss.Disease(infection=S >> I @ ss.peryear(0)*I, death=ss.death(I, rate=ss.peryear(1.0)), init_n=dict(I=10_000), name='sis')
    sim = ss.Sim(diseases=sis, demographics=ss.Deaths(death_rate=ss.peryear(1)), n_agents=10_000, start=2010, stop=2011, dt=ss.months(1), verbose=0).run()
    r = sim.results
    by_disease, by_deaths, total = r.sis.death.sum(), r.deaths.death.sum(), r.death.sum()
    assert by_disease + by_deaths == total
    ppl = sim.people
    dead = ss.uids(np.flatnonzero(~ppl.alive.raw[:ppl.uid.len_used]))
    causes = np.array(ppl.causes)[ppl.cause_of_death.raw[dead]]
    assert (causes == 'sis').sum() == by_disease and (causes == 'deaths').sum() == by_deaths # One cause each, kept per person
    assert abs(by_disease - by_deaths) < 4*np.sqrt(total) # The same rate, so the same share of deaths, whichever module runs first
    return sim


@sc.timer()
def test_declared_newborns():
    """ Agents added by ss.Births and ss.Pregnancy (including the unborn) enter a declared disease, with a level of each dimension """
    sc.heading('Testing newborns in a declared disease')
    for dem in [ss.Births(birth_rate=0.04), ss.Pregnancy(fertility_rate=0.04)]:
        S, I = ss.states('S I')
        sis = ss.Disease(infection=ss.Transition(S, I, rate=2 * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.1, name='sis')
        sim = ss.Sim(sis, n_agents=2000, demographics=[dem, ss.Deaths(death_rate=0.01)], stratify=dict(risk={'lo': 0.2, 'hi': 0.8}), start=2000, stop=2020, verbose=0, rand_seed=1)
        sim.run()
        r = sim.results
        n_states = r.sis.S[-1] + r.sis.I[-1]
        alive = r.alive[-1] + (r.pregnancy.unborn[-1] if 'pregnancy' in r else 0) # The unborn are in the disease's states, but not counted among the people
        assert np.isclose(n_states, alive), f'{dem.name}: {n_states} in states but {alive} alive'
        new = sim.people.auids[sim.people.auids >= 2000]
        assert len(new) > 100
        frac = sim.people.risk[new].mean()
        assert 0.7 < frac < 0.9, f'{dem.name}: {frac:.2f} of newborns are high risk, not 0.8'
        assert sim.diseases.sis.I[new].any() # Newborns can be infected
    return


@sc.timer()
def test_birth_death_hooks():
    sc.heading('Testing on_birth() and on_death() in coded modules')

    class Watch(ss.Module):
        def __init__(self):
            super().__init__(name='watch')
            self.births = 0
            self.deaths = 0
        def on_birth(self, parents, children):
            assert len(parents) == len(children)
            self.births += len(children)
            preg = self.sim.demographics.get('pregnancy')
            if preg is not None: # In the step of the delivery: the mothers delivered in it
                assert np.isin(parents, (preg.ti_delivered == self.sim.ti).uids).all()
        def on_death(self, uids):
            self.deaths += len(uids)
        def step(self):
            pass

    sim = ss.Sim([Watch(), ss.Births(birth_rate=30, rate_per=1000), ss.Deaths(death_rate=20, rate_per=1000)], n_agents=1000, start=2000, stop=2010, verbose=0).run()
    assert sim.watch.births == sim.results.births.birth.sum() > 0
    assert sim.watch.deaths == sim.results.death.sum() > 0
    sim = ss.Sim(Watch(), demographics=[ss.birth(rate=0.03), ss.death(rate=0.02)], n_agents=1000, start=2000, stop=2010, verbose=0).run() # Declared births tell the modules too
    assert sim.watch.births == sim.results.birth.sum() > 0
    for fetus in [False, True]:
        sim = ss.Sim([Watch(), ss.Pregnancy(fertility_rate=0.04, fetal_agents=fetus)], n_agents=1000, start=2000, stop=2005, verbose=0).run()
        assert sim.watch.births == sim.results.pregnancy.birth.sum() > 0
    return sim


@sc.timer()
def test_rates_and_counts():
    sc.heading('Testing net migration of either sign, init_n= as a number of people, and both kinds of maternal death')
    for rate, per, growth in [(ss.peryear(0.01), None, 0.1), (-10, 1000, -0.1)]: # A rate per person, or a bare number per 1,000 per year
        sim = ss.Sim(demographics=[ss.migration(rate=rate, rate_per=per)], start=2000, stop=2010, solver='ode', verbose=0).run()
        assert np.isclose(sim.results.alive[-1], 10_000*np.exp(growth), rtol=1e-3)
    assert [tr.name for tr in ss.migration(rate=ss.timeseries({2000: 5, 2010: -5}), rate_per=1000)] == ['immigration', 'emigration'] # Both, as it changes sign
    for solver in ['abm', 'ode']:
        sim = ss.Sim(ss.SIR(init_n=5, beta=0), n_agents=1000, total_pop=1000, stop=2001, solver=solver, verbose=0).run()
        assert sim.results.sir.I[0] == 5
    both = ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.1), p_maternal_death=0.05, maternal_death_rate=ss.peryear(0.1)), n_agents=3000, start=2000, stop=2005, dt=ss.months(1), verbose=0).run().results.pregnancy
    assert both.antepartum.sum() > 0 and both.maternal_death.sum() > both.antepartum.sum() # While pregnant and at delivery, together
    return sim


@sc.timer()
def test_birth_forms():
    """ ss.birth()/ss.death() and the classes they stand for, ss.Births/ss.Deaths, give the same births and deaths, for every kind of rate """
    sc.heading('Testing that ss.birth() and ss.death() give what ss.Births and ss.Deaths give...')
    kw = dict(n_agents=20_000, start=2000, stop=2010, dt=ss.years(0.5), verbose=0)
    ts = lambda: ss.timeseries({2000: 0.04, 2010: 0.02}, interpolation='linear')
    pairs = { # A time series, and a place's rates, in the lower-case form and in the class
        'series':  ([ss.birth(rate=ts(), rel_birth=0.5, mothers=True), ss.death(rate=ts(), rel_death=0.5)], [ss.Births(birth_rate=ts(), rel_birth=0.5), ss.Deaths(death_rate=ts(), rel_death=0.5)]),
        'place':   ([ss.birth(rate='Kenya', mothers=True), ss.death(rate='Kenya')], [ss.Births(birth_rate='Kenya'), ss.Deaths(death_rate='Kenya')]),
    }
    for key, (lower, upper) in pairs.items():
        a, b = [ss.Sim(demographics=dem, **kw).run() for dem in [lower, upper]]
        births = [a.results.birth.sum(), b.results.births.birth.sum()]
        assert np.isclose(*births, rtol=0.06), f'{key}: {births}' # About 4,000 births each, so within 4 standard deviations
        assert np.isclose(a.results.death.sum(), b.results.death.sum(), rtol=0.08), key
        kids = a.people.parent.raw[a.people.auids] >= 0
        assert kids.sum() > 0.9*births[0] and a.people.female.raw[a.people.parent.raw[a.people.auids[kids]]].all(), key # Each newborn has a mother, as with ss.Births

    # Newborns face deaths by age in the step they are born, whatever the order of the modules or the form of births
    kw = dict(start=2000, stop=2010, dt=ss.years(1), verbose=0)
    final = {}
    for label, dem in [('births first', lambda: [ss.Births(birth_rate='Kenya'), ss.death(rate='Kenya')]), ('deaths first', lambda: [ss.death(rate='Kenya'), ss.Births(birth_rate='Kenya')]), ('declared', lambda: [ss.birth(rate='Kenya'), ss.death(rate='Kenya')])]:
        final[label] = np.mean([ss.Sim(people=ss.population('Kenya', year=2000, n_agents=5000), demographics=dem(), rand_seed=seed, **kw).run().results.alive[-1] for seed in [1, 2]])
    vals = np.array(list(final.values()))
    assert np.ptp(vals)/vals.mean() < 0.015, final
    return final


@sc.timer()
def test_sim_births_deaths():
    """ Births and deaths are read the same way whatever form gave them: the sim's birth and death """
    sc.heading('Testing the sim\'s births and deaths for every form...')
    kw = dict(start=2000, stop=2010, dt=ss.years(0.5), verbose=0)
    declared = lambda: [ss.birth(rate=0.03), ss.death(rate=0.01)]
    sims = sc.objdict(
        sir    = ss.Sim(diseases=ss.SIR(), networks='random', demographics=declared(), n_agents=2000, **kw),
        coded  = ss.Sim(demographics=[ss.Births(birth_rate=0.03), ss.Deaths(death_rate=0.01)], n_agents=2000, **kw),
        ode    = ss.Sim(diseases=ss.SIR(), demographics=declared(), total_pop=2000, solver='ode', **kw),
    )
    for key,sim in sims.items():
        res = sim.run().results
        assert res.birth.sum() > 0 and res.death.sum() > 0, key
        cbr, cmr = ss.solvers.crude_rates(res.birth.values, res.death.values, res.alive.values, ss.solvers.people_t(sim).dt_year) # Not results of the sim, only of the demographics modules
        assert 'cbr' not in res and np.isclose(cbr[:-1].mean(), 30, rtol=0.25) and np.isclose(cmr[:-1].mean(), 10, rtol=0.5), f'{key}: {cbr.mean():n}, {cmr.mean():n}' # Per 1,000 per year
    res = ss.Sim(diseases=ss.SIS(), networks='random', n_agents=500, verbose=0).run().results # With nothing that makes births or deaths, there are no such results, rather than zeros
    assert 'birth' not in res and 'death' not in res and 'emigration' not in res
    res = ss.Sim(diseases=ss.SIS(), networks='random', demographics=ss.death(rate=0), n_agents=500, verbose=0).run().results # A death rate of zero has deaths, all zero
    assert 'death' in res and res.death.sum() == 0 and 'birth' not in res
    assert np.isclose(sims.ode.results.alive[-1], 2000 + sims.ode.results.birth[:-1].sum() - sims.ode.results.death[:-1].sum())
    coded = sims.coded.results
    assert coded.birth.sum() == coded.births.birth.sum() and coded.death.sum() == coded.deaths.death.sum() and 'cbr' not in coded and coded.births.cbr.sum() > 0 and coded.deaths.cmr.sum() > 0 # The same as the modules' own, which have the crude rates
    return sims


# %% The declared ss.Pregnancy

@sc.timer()
def test_burnin_pregnancies():
    sc.heading('Testing burnin=: women pregnant at the start count their pregnancies, so deliveries plus losses are pregnancies')
    seen = sc.obj(start=None, lost=None)
    def snapshot(sim): # The women pregnant at the start, once the sim is initialized
        p = sim.demographics.pregnancy
        preg = p.P.uids
        seen.start = (len(preg), p.n_pregnancies[preg], p.parity[preg])
        seen.lost = np.zeros(len(sim.people.uid.raw) + 100_000)
        return sim
    def check(sim):
        p = sim.demographics.pregnancy
        for tr in ['abortion', 'miscarriage', 'stillbirth']: # Each woman's losses in the run
            uids = p.engine.fresh(f'test_{tr}', tr, unique=False)
            np.add.at(seen.lost, uids, 1)

    # Women pregnant at the start count that pregnancy
    snapshot(ss.Sim(demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.1)), analyzers=check, n_agents=2000, verbose=0).init())
    n, npreg, parity = seen.start
    assert n > 10 and np.all(npreg == 1 + parity), f'Women pregnant at the start have {np.unique(npreg)} pregnancies'

    # With deliveries before the start (short pregnancies), counted in parity, and losses in the run
    preg = ss.Pregnancy(fertility_rate=ss.peryear(0.6), dur_pregnancy=ss.uniform(ss.weeks(10), ss.weeks(70)), p_abortion=0.1, loss_rate=ss.probperweek(0.01))
    preg.stratify(parity={'0': 1, '1': 0, '2': 0, '3+': 0})
    sim = snapshot(ss.Sim(demographics=preg, analyzers=check, n_agents=10_000, start=2000, stop=2004, dt=ss.months(1), verbose=0).init()).run()
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
    preg = ss.Pregnancy(fertility_rate=ss.peryear(0.1), p_maternal_death=ss.bernoulli(0.05), p_infertile=ss.bernoulli(0.2), infant_death_rate=ss.peryear(0.05))
    sim = ss.Sim(demographics=preg, n_agents=5000, start=2000, stop=2010, dt=ss.months(1), verbose=0).run()
    r = sim.results.pregnancy
    births, md = r.birth.values, r.maternal_death.values
    assert md.sum() > 30
    assert np.allclose(r.mmr.values, sc.safedivide(md, births, default=0)*100e3), 'MMR: maternal deaths per 100,000 births'
    assert r.mmr.kind == 'ratio'
    mmr = md.sum()/births.sum()*100e3
    assert 3500 < mmr < 6500, f'An MMR of {mmr:.0f} per 100,000 from a 5% chance of death at delivery'
    imr = r.infant_mortality.sum()/births.sum()
    assert 0.02 < imr < 0.06, f'{imr:.3f} of children died before the age of 1 at a rate of 0.05 per year (less for the last year)'

    # The relative risk of preterm birth by age: each edge is the upper edge of its group, so 18 and 35 are in the next group, and edges need not be whole years
    preg = sim.demographics.pregnancy
    uids = ss.uids(np.arange(5))
    sim.people.age[uids] = [17.9, 18, 34.9, 35, 40.4]
    preg.rel_preterm_base[uids] = 1
    preg.pars.preterm.by_age = np.array([[18, 35, 40.5, 1000], [1.2, 1, 1.2, 1.5]])
    preg.preterm_risk(uids)
    assert np.allclose(preg.rel_preterm[uids], [1.2, 1, 1, 1.2, 1.2]), '1.2 below 18, 1 from 18 to 35, 1.2 from 35 to 40.5'
    return sim


def test_path_triggers():
    """ A trigger on another module's transition by path (ss.ref('pregnancy.delivery')) is its event, as ss.on(preg.delivery) is, so it reaches fetuses at their mother's delivery """
    sc.heading('Testing triggers on a transition by path')
    def make(by_path):
        preg = ss.Pregnancy(fertility_rate=ss.peryear(0.1))
        S, I, Cs = ss.states('S I Cs')
        delivery = ss.ref('pregnancy.delivery') if by_path else preg.delivery
        d = ss.Disease(S, I, Cs, infection=S >> I @ ss.peryear(0.1), mtct=ss.Transition(S, Cs, rate=ss.peryear(3)*ss.mother(I)),
                       still=ss.Transition(Cs, ss.dead, when=delivery), name='d')
        return ss.Sim(demographics=preg, diseases=d, networks=ss.PrenatalNet(), n_agents=5000, start=2000, stop=2005, verbose=0).run()
    direct, path = make(False), make(True)
    assert path.results.d.still.sum() > 0, 'A fetus is removed at its mother\'s delivery'
    assert np.array_equal(direct.results.d.still.values, path.results.d.still.values)
    return path


def test_rate_tables():
    sc.heading('Tables of rates by year: births and deaths hold each year until the next, fertility is interpolated to every year')
    births = ss.Births(birth_rate=pd.DataFrame(dict(Year=[2000, 2010], CBR=[0.02, 0.04])))
    rows, years, _ = births.birth_rows
    assert rows[ss.demographics.period_year(years, 2005)] == 0.02 and rows[ss.demographics.period_year(years, 2010)] == 0.04 # Held, as ss.timeseries holds
    preg = ss.Pregnancy(fertility_rate=pd.DataFrame(dict(Time=[2000, 2000, 2010, 2010], AgeGrp=[15, 20, 15, 20], ASFR=[0.1, 0.2, 0.3, 0.4])))
    rows, years, _ = ss.demographics.rate_rows(preg.standardize_fertility_data())
    assert np.allclose(rows[years[sc.findnearest(years, 2005.4)]][1:3], [0.2, 0.3]) # Interpolated (by age group from 15, after the one below it), and read at the nearest year

    # A chance of maternal death at delivery set to a number after the module is made works as one given when it is made
    preg = ss.Pregnancy(fertility_rate=ss.peryear(0.2), burnin=False)
    preg.pars.p_maternal_death = 0.5
    sim = ss.Sim(demographics=preg, n_agents=2000, start=2000, stop=2002, dt=ss.months(1), verbose=0).run()
    assert sim.results.pregnancy.maternal_death.sum() > 0
    return


def test_births_by_age():
    sc.heading('A birth rate by age and sex runs as agents as in compartments; mothers are chosen by their rates; pregnancies lost at the mother\'s death are counted')
    asfr = ss.by(dict(age=[0, 15, 50]*2, sex=['female']*3 + ['male']*3, value=[0, 0.15, 0, 0, 0, 0]))
    def make(solver):
        kw = dict(n_agents=10_000, rand_seed=1) if solver == 'abm' else dict(total_pop=10_000)
        sim = ss.Sim(demographics=[ss.birth(rate=ss.peryear(asfr), mothers=True), ss.death(rate=ss.peryear(0.01))], start=2000, stop=2010, dt=ss.months(1), solver=solver, verbose=0, **kw)
        if solver == 'ode':
            sim.stratify(age=np.arange(0, 100, 0.5), sex=['female', 'male'])
        return sim.run()
    abm, ode = make('abm'), make('ode')
    assert np.isclose(abm.results.alive[-1], ode.results.alive[-1], rtol=0.02)
    ppl = abm.people
    kids = ppl.auids[ppl.parent.raw[ppl.auids] != ppl.parent.nan]
    moms = ppl.parent.raw[kids]
    moms, kids = moms[ppl.alive.raw[moms]], kids[ppl.alive.raw[moms]]
    at_birth = ppl.age.raw[moms] - ppl.age.raw[kids]
    assert len(kids) and ppl.female.raw[moms].all() and at_birth.min() >= 15 - 1e-6 and at_birth.max() < 50

    # A woman who dies while pregnant loses the pregnancy, counted as a miscarriage or stillbirth (the deaths of everyone are not held by the pregnancy)
    sim = ss.Sim(diseases=ss.SIS(beta=0, init_prev=0), demographics=[ss.Pregnancy(fertility_rate=200, rate_per=1000), ss.death(rate=ss.peryear(0.2))],
                 n_agents=5000, start=2000, stop=2003, dt=ss.months(1), solver='abm', rand_seed=1, verbose=0).run()
    r = sim.results.pregnancy
    lost = r.P[0] + r.conception[:-1].sum() - r.delivery[:-1].sum() - r.P[-1]
    assert lost > 0 and np.isclose(r.miscarriage[:-1].sum() + r.stillbirth[:-1].sum(), lost, rtol=0.05)
    assert 'deaths' not in sim.pregnancy.graph.transitions

    # Ages on the grid of steps reach whole years on time, not a hair below
    sim = ss.Sim(demographics=[ss.Births(birth_rate=30, rate_per=1000)], n_agents=2000, start=2000, stop=2030, dt=ss.months(1), verbose=0, rand_seed=1).run()
    ppl = sim.people
    age = ppl.age.raw[ppl.auids[ppl.auids >= 2000]] # Those born in the run
    assert age.dtype == np.float64 and not np.any((np.ceil(age) - age > 0) & (np.ceil(age) - age < 1e-6))

    # Fertility rates are by the age at delivery, by default
    asfr = sc.dataframe(Time=2000, AgeGrp=[20, 25, 30], ASFR=[0, 300, 0])
    for fertility_age, lo, hi in [('delivery', 25, 30), ('conception', 25.6, 30.9)]:
        sim = ss.Sim(demographics=ss.Pregnancy(fertility_rate=asfr, rate_per=1000, fertility_age=fertility_age), n_agents=5000, start=2000, stop=2005, dt=ss.months(1), rand_seed=1, verbose=0).run()
        ppl = sim.people
        kids = ppl.auids[ppl.parent.raw[ppl.auids] != ppl.parent.nan]
        at_birth = ppl.age.raw[ppl.parent.raw[kids]] - ppl.age.raw[kids]
        assert len(kids) > 100 and lo - 0.25 <= np.percentile(at_birth, 1) and np.percentile(at_birth, 99) <= hi # The 1st percentile is 24.8 to 25.1 over seeds
    return


def test_sex_by_age():
    sc.heading('A named population is female by its fraction at each age, under both solvers, and children are born female by its fraction at birth, unless the birth says otherwise')
    pop = ss.data.population('India', year=1950)
    female, total = pop.female.values, (pop.female + pop.male).values
    old = female[14:].sum()/total[14:].sum() # Aged 70 and over, about half female (against 48.6% overall)
    demog = lambda p_female=None: [ss.birth(rate=ss.peryear(0.04), p_female=p_female), ss.death(rate=ss.peryear(0.01))]
    sim = ss.Sim(people=ss.population('India', year=1950, n_agents=50_000), demographics=demog(), start=1950, stop=1955, dt=ss.months(1), rand_seed=1, verbose=0).run()
    ppl = sim.people
    first, new = ppl.auids[ppl.auids < 50_000], ppl.auids[ppl.auids >= 50_000]
    assert abs(ppl.female.raw[first][ppl.age.raw[first] >= 75].mean() - old) < 0.02
    assert abs(ppl.female.raw[new].mean() - female[0]/total[0]) < 0.02 # About 10,000 births
    for p_female, born in [(None, female[0]/total[0]), (0.3, 0.3)]:
        sim = ss.Sim(people=ss.population('India', year=1950), demographics=demog(p_female), start=1950, stop=1951, dt=ss.months(1), solver='ode', verbose=0)
        sim.stratify(age=np.arange(0, 75, 5), sex=['female', 'male']).init()
        ds, grid = sim.groups.joint
        assert np.isclose(grid[-1, 0]/grid[-1].sum(), old, atol=1e-3)
        tr = next(tr for tr in sim.declared[0].graph.transitions.values() if tr.per_capita)
        split = sim.declared[0].engine.inflow_split(tr)
        assert np.isclose(split[split > 0][0], born)
    return


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
