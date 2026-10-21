"""
Test demographic consistency
"""

import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl
import pytest

sc.options(interactive=False) # Assume not running interactively
datadir = ss.root / 'tests/test_data'


@sc.timer()
def test_nigeria(which='births', dt=1, start=1995, dur=15, do_plot=False):
    """
    Make a Nigeria sim with demographic modules
    Switch between which='births' or 'pregnancy' to determine which demographic module to use
    """
    sc.heading('Testing Nigeria demographics')

    # Make demographic modules
    demographics = sc.autolist()

    if which == 'births':
        birth_rates = pd.read_csv(datadir/'nigeria_births.csv')
        births = ss.Births(birth_rate=birth_rates)
        demographics += births

    elif which == 'pregnancy':
        fertility_rates = pd.read_csv(datadir/'nigeria_asfr.csv')
        pregnancy = ss.Pregnancy(fertility_rate=fertility_rates, rel_fertility=1, burnin=False)  # 4/3
        demographics += pregnancy

    death_rates = pd.read_csv(datadir/'nigeria_deaths.csv')
    death = ss.Deaths(death_rate=death_rates, rate_units=1)
    demographics += death

    # Make people
    n_agents = 5_000
    nga_pop_1995 = 106819805
    age_data = pd.read_csv(datadir/'nigeria_age.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    sim = ss.Sim(
        dt=dt,
        total_pop=nga_pop_1995,
        start=start,
        dur=dur,
        people=ppl,
        demographics=demographics,
    )

    if do_plot:
        sim.init()
        # Plot histograms of the age distributions - simulated vs data
        bins = np.arange(0, 101, 1)
        init_scale = nga_pop_1995 / n_agents
        counts, bins = np.histogram(sim.people.age, bins)
        plt.bar(bins[:-1], counts * init_scale, alpha=0.5, label='Simulated')
        plt.bar(bins, age_data.value.values * 1000, alpha=0.5, color='r', label='Data')
        plt.legend(loc='upper right')

    sim.run()

    stop = start + dur
    nigeria_popsize = pd.read_csv(datadir/'nigeria_popsize.csv')
    data = nigeria_popsize[(nigeria_popsize.year >= start) & (nigeria_popsize.year <= stop)]

    nigeria_cbr = pd.read_csv(datadir/'nigeria_births.csv')
    cbr_data = nigeria_cbr[(nigeria_cbr.Year >= start) & (nigeria_cbr.Year <= stop)]

    nigeria_cmr = pd.read_csv(datadir/'nigeria_cmr.csv')
    cmr_data = nigeria_cmr[(nigeria_cmr.Year >= start) & (nigeria_cmr.Year <= stop)]

    # Tests
    if which == 'pregnancy':

        print("Check we don't have more births than pregnancies")
        assert sum(sim.results.pregnancy.birth) <= sum(sim.results.pregnancy.conception)
        print('✓ (births <= pregnancies)')

        if dt == 1:
            print("Checking that births equal pregnancies with dt=1")
            assert np.array_equal(sim.results.pregnancy.conception, sim.results.pregnancy.birth)
            print('✓ (births == pregnancies)')

    rtol = 0.05
    print(f'Check final pop size within {rtol*100:n}% of data')
    assert np.isclose(data.n_alive.values[-1], sim.results.n_alive[-1], rtol=rtol), f'Final population size not within {rtol*100:n}% of data'
    print(f'✓ (simulated/data={sim.results.n_alive[-1] / data.n_alive.values[-1]:.2f})')

    # Plots
    if do_plot:
        tvec = sim.timevec
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        ax[0].scatter(data.year, data.n_alive, alpha=0.5)
        ax[0].plot(tvec, sim.results.n_alive, color='k')
        ax[0].set_title('Population')

        ax[1].plot(tvec, 1000 * sim.results.deaths.cmr, label='Simulated CMR')
        ax[1].scatter(cmr_data.Year, cmr_data.CMR, label='Data CMR')
        ax[1].set_title('CMR')
        ax[1].legend()

        if which == 'births':
            ax[2].plot(tvec, sim.results.births.cbr, label='Simulated CBR')
        elif which == 'pregnancy':
            ax[2].plot(tvec, sim.results.pregnancy.cbr, label='Simulated CBR')
        ax[2].scatter(cbr_data.Year, cbr_data.CBR, label='Data CBR')
        ax[2].set_title('CBR')
        ax[2].legend()

        if which == 'pregnancy':
            ax[3].plot(tvec, sim.results.pregnancy.conception, label='Pregnancies')
            ax[3].plot(tvec, sim.results.pregnancy.birth, label='Births')
            ax[3].set_title('Pregnancies and births')
            ax[3].legend()

        fig.tight_layout()

    return sim


@sc.timer()
def test_constant_pop(do_plot=False):
    """ Test pars for constant pop size """
    sc.heading('Testing constant population size')
    sim = ss.Sim(n_agents=10e3, demographics=[ss.birth(rate=ss.peryear(0.01)), ss.death(rate=ss.peryear(0.01/1.01))], dur=ss.years(200), rand_seed=1).run()
    print("Check final pop size within 5% of starting pop")
    assert np.isclose(sim.results.n_alive[0], sim.results.n_alive[-1], rtol=0.05)
    print(f'✓ (final pop / starting pop={sim.results.n_alive[-1] / sim.results.n_alive[0]:.2f})')

    # Plots
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_module_adding():
    """ Test that modules can't be added twice """
    sc.heading('Testing module duplication')
    births = ss.Births(birth_rate=ss.time.freq(10, 'year'))
    deaths = ss.Deaths(death_rate=ss.time.freq(10, 'year'))
    demographics = [births, deaths]
    with pytest.raises(ValueError): # Starsim 3's birth_rate= is removed, rather than added to the modules given
        ss.Sim(n_agents=1e3, demographics=demographics, birth_rate=ss.time.freq(10, 'year')).run()
    sim = ss.Sim(n_agents=1e3, demographics=[ss.Births(), ss.Deaths()], people_results=False, verbose=0).run() # Without people's results (MA32)
    assert sim.results.deaths.new.sum() > 0
    return demographics


@sc.timer()
def test_aging():
    """ Test that aging is configured properly """
    sc.heading('Testing aging')
    n_agents = int(1e3)

    # With no demograhpics, people shouldn't age
    s1 = ss.Sim(n_agents=n_agents).init()
    orig_ages = s1.people.age.raw.copy()
    orig_age = orig_ages.mean()
    s1.run()
    end_age = s1.people.age.mean()
    assert orig_age == end_age, f'By default there should be no aging, but {orig_age} != {end_age}'

    # We should be able to manually turn on aging
    s2 = ss.Sim(n_agents=n_agents, use_aging=True).run()
    age2 = s2.people.age.mean()
    assert orig_age < age2, f'With aging, start age {orig_age} should be less than end age {age2}'

    # Aging should turn on automatically if we add demographics
    s3 = ss.Sim(n_agents=n_agents, demographics=True).run()
    agent = s3.people.auids[0] # Find first alive agent
    orig_agent_age = orig_ages[agent]
    age3 = s3.people.age[ss.uids(agent)]
    assert orig_agent_age < age3, f'With demographics, original agent age {orig_agent_age} should be less than end age {age3}'

    # ...but can be turned off manually
    s4 = ss.Sim(n_agents=n_agents, demographics=True, use_aging=False).run()
    agent = s4.people.auids[0] # Find first alive agent
    orig_agent_age = orig_ages[agent]
    age4 = s4.people.age[ss.uids(agent)]
    assert orig_agent_age == age4, f'With aging turned off, original agent age {orig_agent_age} should match end age {age4}'

    return s3 # Most interesting sim


def test_pregnancy():
    """ Test pregnancy module"""
    sc.heading('Testing pregnancy module')

    sim = ss.Sim(
        n_agents=20e3, # Large enough that the <18 mother cohort has a robust sample (~85 babies) for the gestational-age assertions below
        demographics=[
            ss.Pregnancy(fertility_rate=ss.time.freq(20, 'year'), burnin=True),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
        ],
        dur=ss.years(10),
        rand_seed=1,
        dt=ss.years(1/12),
        networks=ss.PrenatalNet(),
    )
    sim.run()

    # Tests
    # Check that everyone in their 2nd trimester conceived 3-6 months ago
    preg = sim.demographics.pregnancy
    weeks = ss.years((sim.ti - preg.ti_P[preg.P.uids]) * sim.t.dt_year).weeks # Since conception
    assert np.all((weeks >= 0) & (weeks <= 45)), 'Some pregnancies are longer than 45 weeks'

    # Check that every pregnant woman has a defined delivery time
    assert np.all(~np.isnan(preg.ti_exit_P[preg.P.uids])), 'Some pregnant women have no scheduled delivery'
    sc.printgreen('✓ Pregnancy duration tests passed')

    # Check that everyone born during the sim has gestational age at birth recorded
    born_during_sim = (sim.people.age < sim.pars.dur.years) & (sim.people.age > sim.t.dt.years)
    aa = (sim.people.pregnancy.gestation_at_birth.isnan & born_during_sim).uids
    assert len(aa) == 0, f'Some babies born during the sim are missing gestational age at birth: {len(aa)} found'

    # Check that the mean gestational age at birth is approximately 40 weeks
    mean_gestation = np.nanmean(sim.people.pregnancy.gestation_at_birth[born_during_sim])
    assert np.isclose(mean_gestation, 40, atol=2), f'Mean gestational age at birth is {mean_gestation:.1f} weeks, expected ~40 weeks'

    # Check that the mean gestational age of babies born to women <18 or >35 is shorter than otherwise
    mothers = sim.people.parent[born_during_sim]
    age_of_mother = sim.people.age[mothers]
    age_of_mother_at_birth = age_of_mother - sim.people.age[born_during_sim]
    key_cohorts = {
        '<18': age_of_mother_at_birth < 18,
        '>35': age_of_mother_at_birth > 35,
        '18-35': (age_of_mother_at_birth >= 18) & (age_of_mother_at_birth <= 35),
    }
    mean_results = {k:None for k in key_cohorts.keys()}
    ptb_results = {k:None for k in key_cohorts.keys()}
    for k, v in key_cohorts.items():
        these_mothers_idx = v.nonzero()[-1]
        babies_of_these_mothers = born_during_sim.uids[these_mothers_idx]
        mean_gestation = sim.people.pregnancy.gestation_at_birth[babies_of_these_mothers].mean()
        print(f'Mean gestational age at birth for mothers aged {k} at birth: {mean_gestation:.1f} weeks')
        mean_results[k] = mean_gestation
        n = len(babies_of_these_mothers)
        print(f'  {n} babies born to mothers aged {k} at birth')
        # Share of babies born preterm (<37 weeks)
        n_preterm = np.sum(sim.people.pregnancy.gestation_at_birth[babies_of_these_mothers] < 37)
        preterm_share = n_preterm / n if n > 0 else np.nan
        print(f'  Preterm births (<37 weeks): {n_preterm} ({preterm_share:.1%})')
        ptb_results[k] = preterm_share

    # Assertions
    assert mean_results['<18'] < mean_results['18-35'], '<18 mothers do not have lower gestational age at birth than 18-35 mothers'
    assert mean_results['>35'] < mean_results['18-35'], '>35 mothers do not have lower gestational age at birth than 18-35 mothers'
    rel = sim.demographics.pregnancy.rel_ptb # Each mother's relative risk of preterm birth at her last conception (the preterm shares themselves are too few here to compare: see test_pregnancy_risks)
    risk = {k:np.mean(rel[ss.uids(np.asarray(mothers)[np.asarray(v)])]) for k,v in key_cohorts.items()}
    assert risk['<18'] > 1.1*risk['18-35'] and risk['>35'] > 1.1*risk['18-35'], f'Mothers <18 and >35 should have a higher relative risk of preterm birth: {risk}'

    sc.printgreen('✓ Gestational age at birth by maternal age tests passed')

    return sim


def test_pregnancy_short_sim():
    """ Test that Pregnancy.finalize() works for sims shorter than 1 year """
    sc.heading('Testing pregnancy with short sim duration')

    sim = ss.Sim(
        n_agents=1e3,
        demographics=[
            ss.Pregnancy(fertility_rate=ss.time.freq(10, 'year'), burnin=False),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
        ],
        dur=ss.days(250),
        rand_seed=1,
        dt=ss.days(10),
        networks=ss.PrenatalNet(),
    )
    sim.run()  # Should not raise ValueError in finalize()

    # Check that a number or a non-choice distribution works for dur_pregnancy, including with fertility data
    asfr = pd.read_csv(datadir/'nigeria_asfr.csv')
    for dur_preg in [39, ss.normal(ss.weeks(39), ss.weeks(1))]:
        preg = ss.Sim(n_agents=500, demographics=ss.Pregnancy(fertility_rate=asfr, dur_pregnancy=dur_preg), dur=ss.years(1), dt=ss.months(1), verbose=0).run().demographics.pregnancy
        assert 0.7 < preg.mean_dur_pregnancy < 0.8, 'Mean pregnancy should be roughly 39 weeks' # 39 weeks = 0.75 years

    sc.printgreen('✓ Short sim pregnancy finalize test passed')
    return sim


@sc.timer()
def test_loss_classification():
    """
    Test that prenatal deaths are correctly classified as miscarriage (<20w)
    or stillbirth (>=20w) based on gestational age.
    """
    sc.heading('Testing loss classification...')

    # Use a high loss rate so both early and late losses occur
    sim = ss.Sim(
        n_agents=5_000,
        demographics=[
            ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), p_loss=ss.bernoulli(p=0.05), burnin=True),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
        ],
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
        networks=ss.PrenatalNet(),
    )
    sim.run()

    preg = sim.demographics.pregnancy
    mc = preg.results['miscarriage'].sum()
    sb = preg.results['stillbirth'].sum()
    births = preg.results['birth'].sum()
    print(f'  Miscarriages: {mc}, Stillbirths: {sb}, Births: {births}')

    assert mc > 0, 'Expected some miscarriages with p_loss=0.05'
    assert sb > 0, 'Expected some stillbirths with p_loss=0.05'
    assert mc > sb, 'Expected more miscarriages than stillbirths (pregnancies spend more time at early GA)'
    assert births > 0, 'Expected some births despite losses'

    sc.printgreen('✓ Loss classification tests passed')
    return sim


@sc.timer()
def test_background_loss():
    """
    Test that background pregnancy loss (p_loss) produces losses and that
    higher p_loss produces more losses.
    """
    sc.heading('Testing background loss...')

    results = dict()
    for label, p_loss in [('none', 0), ('low', 0.01), ('high', 0.05)]:
        sim = ss.Sim(
            n_agents=5_000,
            demographics=[
                ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), p_loss=ss.bernoulli(p=p_loss), burnin=True),
                ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
            ],
            dur=ss.years(5),
            dt=ss.years(1/12),
            rand_seed=1,
            networks=ss.PrenatalNet(),
        )
        sim.run()
        preg = sim.demographics.pregnancy
        total_losses = preg.results['miscarriage'].sum() + preg.results['stillbirth'].sum()
        births = preg.results['birth'].sum()
        results[label] = dict(losses=total_losses, births=births)
        print(f'  p_loss={p_loss}: losses={total_losses}, births={births}')

    assert results['low']['losses'] > results['none']['losses'], 'Expected more losses with p_loss=0.01 than p_loss=0'
    assert results['high']['losses'] > results['low']['losses'], 'Expected more losses with higher p_loss'
    assert results['high']['births'] < results['none']['births'], 'Expected fewer births with p_loss > 0'

    sc.printgreen('✓ Background loss tests passed')
    return sim


@sc.timer()
def test_nnd():
    """
    Test passive neonatal death detection using neonatal sepsis disease.
    Verify deaths of newborns within 28 days are classified as NNDs.
    """
    sc.heading('Testing neonatal death detection...')

    sim = ss.Sim(
        n_agents=5_000,
        demographics=[
            ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), burnin=True),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
        ],
        diseases=ssl.mnch.NeonatalSepsis(),
        dur=ss.years(3),
        dt=ss.days(7), # Weekly, since a sepsis death (about a week after birth) is only seen at a step, and at monthly steps a newborn is a month old by then
        rand_seed=1,
        networks=ss.PrenatalNet(),
    )
    sim.run()

    preg = sim.demographics.pregnancy
    neonatal_deaths = preg.results['neonatal_deaths'].sum()
    births = preg.results['birth'].sum()
    print(f'  Neonatal deaths: {neonatal_deaths}, Births: {births}')

    assert neonatal_deaths > 0, 'Expected neonatal deaths from neonatal sepsis'
    assert neonatal_deaths < births, 'Neonatal deaths should be fewer than total births'

    sc.printgreen('✓ Neonatal death detection tests passed')
    return sim


@sc.timer()
def test_count_based_births():
    """
    The non-CRN count-based birth draw (ss.options.crn=False) matches the per-agent
    Bernoulli scan distributionally, and both paths are reproducible for a fixed seed.
    """
    sc.heading('Testing count-based (non-CRN) births...')

    def cum_births(crn, seed):
        ss.options.crn = crn
        sim = ss.Sim(n_agents=3000, dur=15, dt=0.25, demographics=ss.Births(birth_rate=ss.peryear(30)),
                     rand_seed=seed, verbose=0).run()
        return float(sim.results.births.cumulative[-1])

    orig_crn = ss.options.crn
    try:
        nseeds = 12
        per_agent = np.array([cum_births(True, s) for s in range(nseeds)])
        count_based = np.array([cum_births(False, s) for s in range(nseeds)])

        # Distributional match: mean cumulative births agree within stochastic tolerance
        rel_diff = abs(per_agent.mean() - count_based.mean()) / per_agent.mean()
        assert rel_diff < 0.05, f'Count-based births differ from per-agent by {rel_diff*100:.1f}% (mean {count_based.mean():.0f} vs {per_agent.mean():.0f})'
        assert count_based.std() > 0, 'Count-based births should be stochastic'

        # Both paths are reproducible: identical seed -> identical result
        assert cum_births(True, 0) == cum_births(True, 0), 'CRN births not reproducible'
        assert cum_births(False, 0) == cum_births(False, 0), 'Count-based births not reproducible'
    finally:
        ss.options.crn = orig_crn

    print(f'  Per-agent mean: {per_agent.mean():.0f}, count-based mean: {count_based.mean():.0f} ({rel_diff*100:.2f}% difference)')

    # A bare rate above 1 is per 1,000 people, and below 1 per person, unless rate_units says
    births = [ss.Births(birth_rate=rate, rate_units=units).pars.rate_units for rate,units in [(30, None), (0.03, None), (0.5, 1000), (0.5, 1)]]
    assert births == [1e-3, 1, 1e-3, 1]
    with pytest.warns(RuntimeWarning): # Could be either
        ss.Deaths(death_rate=0.5)
    return per_agent, count_based


@sc.timer()
def test_birth_data_cols():
    """ Birth rate data can use either "Year" or "Time" (the UN label) for the year column """
    sc.heading('Testing birth data column handling...')

    years = [1990, 2000, 2010]
    vals = [45.0, 42.0, 38.0]
    expected = pd.Series(vals, index=pd.Index(years, name='year', dtype=float))

    def check(births, msg):
        """ The standardized birth rates match the input, regardless of column labels """
        actual = births.pars.birth_rate.loc[years] # Drop the -inf age entries etc.
        assert np.allclose(actual.values, expected.values), f'{msg}: expected {vals}, got {actual.values}'
        return

    # Both spellings of the year column work without a metadata override
    year_df = pd.DataFrame({'Year':years, 'CBR':vals})
    time_df = pd.DataFrame({'Time':years, 'CBR':vals})
    check(ss.Births(birth_rate=year_df), 'Year column')
    check(ss.Births(birth_rate=time_df), 'Time column')

    # An explicit override takes precedence over the synonyms: here "Year" holds junk,
    # so if the override were ignored, the values would not match
    both_df = pd.DataFrame({'Year':[0, 0, 0], 'Time':years, 'CBR':vals})
    check(ss.Births(birth_rate=both_df, metadata=dict(data_cols=dict(year='Time', value='CBR'))), 'Explicit override')

    # A genuinely missing column raises an informative error
    missing_df = pd.DataFrame({'Date':years, 'CBR':vals})
    with pytest.raises(ValueError) as excinfo:
        ss.Births(birth_rate=missing_df)
    errormsg = str(excinfo.value)
    for expect in ['Year', 'year', 'Date', 'CBR', 'data_cols']: # Names the column, lists those present, points at the override
        assert expect in errormsg, f'Expected "{expect}" in error message:\n{errormsg}'

    # The same applies to the value column
    with pytest.raises(ValueError) as excinfo:
        ss.Births(birth_rate=pd.DataFrame({'Year':years, 'Rate':vals}))
    assert 'CBR' in str(excinfo.value)

    # Deaths and pregnancy data likewise accept either spelling
    death_data = dict(Sex=['Female', 'Male'], AgeGrpStart=[0, 0], mx=[0.05, 0.06])
    time_deaths = ss.Deaths(death_rate=pd.DataFrame(dict(Time=[1990, 1990], **death_data))) # The default
    year_deaths = ss.Deaths(death_rate=pd.DataFrame(dict(Year=[1990, 1990], **death_data))) # Via the synonym
    assert time_deaths.death_rate_data.equals(year_deaths.death_rate_data)

    print('✓ (birth/death data column handling)')
    return


@sc.timer()
def test_period_rates():
    """ Rates labeled by the start of their period hold until the next label, and 5-year fertility data cover the last group whole """
    sc.heading('Testing rates by period...')

    # Death rates for 2010-2014 and from 2015, labeled 2010 and 2015: 2013 and 2014 have the 2010 rate, not the nearest year's (TL24)
    assert ss.demographics.period_year([2010, 2010, 2015, 2015], 2014.9) == 2010 and ss.demographics.period_year([2010, 2015], 2015) == 2015
    data = pd.DataFrame(dict(Time=[2010, 2010, 2015, 2015], Sex=['Female', 'Male']*2, AgeGrpStart=0, mx=[0.04, 0.04, 0.0, 0.0]))
    sim = ss.Sim(n_agents=10_000, demographics=ss.Deaths(death_rate=data, rate_units=1), start=2010, stop=2015, dt=ss.months(1), verbose=0).run()
    deaths = sim.results.new_deaths.values[1:]
    assert deaths.min() > 0, 'Every step before 2015 has the 2010 rate'
    expected = 10_000*(1 - np.exp(-0.04*5))
    assert abs(deaths.sum() - expected) < 4*np.sqrt(expected), f'Expected about {expected:.0f} deaths, not {deaths.sum()}'

    # Fertility by 5-year age group: the last group (45) is 45-49, not just 45 (TL25)
    data = pd.DataFrame(dict(Time=2010, AgeGrp=[15, 20, 25, 30, 35, 40, 45], ASFR=0.2))
    preg = ss.Sim(n_agents=100, demographics=ss.Pregnancy(fertility_rate=data, rate_units=1), start=2010, stop=2011, verbose=0).init().demographics.pregnancy
    frd = preg.fertility_rate_data
    assert list(frd.columns[-2:]) == [45, 50] and frd[45].iloc[0] == 0.2 and frd[50].iloc[0] == 0
    return sim


@sc.timer()
def test_lifetable_deaths():
    """ Life tables by year and sex in ss.Deaths (FP33), and a death rate by its own ages, or a function, under both solvers (MA20) """
    sc.heading('Testing life tables by year and sex, and death rates by single year of age...')
    ages = np.arange(0, 105, 5)
    table = lambda mx: ss.lifetable(ages, -np.diff(np.append(np.exp(-mx*ages), 0))) # A constant hazard, to age 105
    tables = {2000: dict(female=table(0.01), male=table(0.03)), 2005: dict(female=table(0.02), male=table(0.06))}
    sim = ss.Sim(n_agents=20_000, demographics=ss.Deaths(lifetable=tables), start=2000, stop=2010, verbose=0).run()
    female, dead = sim.people.female.raw, ~sim.people.alive.raw
    for sel,rates in [(female, [0.01, 0.02]), (~female, [0.03, 0.06])]:
        expected = 1 - np.exp(-5*rates[0] - 5*rates[1]) # The 2000 table to 2005, then the 2005 one
        assert abs(dead[sel].mean() - expected) < 4*np.sqrt(expected/sel.sum()), f'{dead[sel].mean():.3f} died, not {expected:.3f}'
    with pytest.raises(ValueError, match='female and male'):
        ss.Deaths(lifetable=dict(women=table(0.01), men=table(0.03)))

    # A death rate by single year of age on 5-year age groups (which only agents read), and a function of the sim per year
    mx = ss.by(dict(age=np.arange(100), value=[ss.peryear(0.001*np.exp(0.07*a)) for a in range(100)]))
    def flat(sim): return 0.02
    def make(rate, solver, stratify, **pop):
        S, I = ss.states(2)
        sim = ss.Sim(ss.Disease(S, I, inf=S >> I @ 0*I, death=ss.death(rate=rate), name='d'), solver=solver, start=2000, stop=2005, verbose=0, **pop)
        if stratify:
            sim.stratify(age=list(range(0, 100, 5)))
        return sim
    with pytest.raises(ss.CapabilityError, match='split'):
        make(mx, 'ode', True, total_pop=50_000).init()
    assert make(mx, 'auto', True, total_pop=50_000).init().solver == 'abm'
    coarse = ss.by(dict(age=np.arange(0, 100, 5), value=[ss.peryear(np.mean([0.001*np.exp(0.07*a) for a in range(lo, lo + 5)])) for lo in range(0, 100, 5)])) # The same, given by the groups
    for rate,fine,stratify in [(coarse, mx, True), (ss.peryear(flat), ss.peryear(flat), False)]:
        n = [make(r, solver, stratify, **pop).run().results.d.death.sum() for r,solver,pop in [(fine, 'abm', dict(n_agents=50_000)), (rate, 'ode', dict(total_pop=50_000))]]
        assert np.isclose(n[0], n[1], rtol=0.06), f'{n[0]:n} deaths as agents, {n[1]:n} as compartments'
    assert np.isclose(n[1], 50_000*(1 - np.exp(-0.02*5)), rtol=1e-3)
    return sim


@sc.timer()
def test_births_parents():
    """ ss.Births' parents are women of reproductive age, and newborns are born where their parent is """
    sc.heading('Testing the parents of births...')
    S, I = ss.states(2)
    dis = ss.Disease(onset=S >> I @ ss.peryear(0.01), name='dis')
    sim = ss.Sim(diseases=dis, n_agents=5000, start=2000, stop=2003, verbose=0, stratify=dict(district=ss.geo(names=['a', 'b', 'c']), split=[0.4, 0.3, 0.3]),
                 demographics=[ss.Births(birth_rate=ss.peryear(0.04)), ss.Deaths(death_rate=ss.peryear(0.01))]).run()
    ppl = sim.people
    kids = ss.uids(np.flatnonzero(ppl.parent.raw >= 0))
    parents = ppl.parent.raw[kids]
    assert len(kids) > 300 and ppl.female.raw[parents].all() # TL27: before, about half were men
    both = ppl.alive.raw[kids] & ppl.alive.raw[parents] # Whose ages have both gone up since
    age_at_birth = ppl.age.raw[parents[both]] - ppl.age.raw[kids[both]]
    assert age_at_birth.min() >= 15 - 0.1 and age_at_birth.max() < 50 + 0.1 # As old as the women who can be mothers (to within a step)
    assert np.array_equal(ppl.district.raw[kids], ppl.district.raw[parents]) # MA18: born where the parent is

    # The same for ss.Pregnancy, including the pregnancies before the start, and a dimension called sex is the newborn's
    sim = ss.Sim(diseases=dis.copy(), n_agents=5000, start=2000, stop=2003, verbose=0, stratify=[dict(district=ss.geo(names=['a', 'b', 'c']), split=[0.4, 0.3, 0.3]), dict(sex=['female', 'male'])],
                 demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.15), rate_units=1)).run()
    ppl = sim.people
    kids = ss.uids(np.flatnonzero(ppl.parent.raw >= 0))
    assert np.array_equal(ppl.district.raw[kids], ppl.district.raw[ppl.parent.raw[kids]])
    assert np.array_equal(ppl.sex.raw[kids] == 0, ppl.female.raw[kids])
    return sim


@sc.timer()
def test_unborn():
    """ The unborn of ss.Pregnancy are not counted among the people, are not seeded, and a single beta on a maternal network goes from mother to child """
    sc.heading('Testing the unborn...')
    I = ss.State(init=0.5)
    S = ss.State()
    hiv = ss.Disease(infection=S >> I @ ss.par(dict(prenatalnet=0.33, breastfeedingnet=0.12)) * I, name='hiv')
    sim = ss.Sim(hiv, demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.15), rate_units=1), networks=[ss.PrenatalNet(), ss.BreastfeedingNet()],
                 n_agents=5000, start=2010, stop=2011, dt=ss.months(1), verbose=0).init()
    ppl, hiv = sim.people, sim.diseases.hiv
    unborn = ppl.unborn_uids()
    assert len(unborn) > 50 and not hiv.I.raw[unborn].any() # TL26: before, half of them started infected
    mothers = ss.uids(ppl.parent.raw[unborn])
    I0 = hiv.I.raw.copy()
    sim.run()
    assert not (hiv.I.raw[mothers] & ~I0[mothers]).any() # TL31: before, mothers were infected by their children
    r = sim.results
    born = ppl.alive.raw[ppl.auids] & (ppl.age.raw[ppl.auids] >= 0)
    assert r.n_alive[-1] < r.n_alive[-1] + r.pregnancy.n_unborn[-1] and r.pregnancy.n_unborn[-1] > 0
    assert np.isclose(ppl.n_alive, born.sum())
    return sim


@sc.timer()
def test_pregnancy_risks():
    """ p_loss is per week whatever the time step, and the relative risk of preterm birth changes the share born preterm """
    sc.heading('Testing pregnancy risks...')
    def run(dt, p_loss=0, rr=1.0):
        preg = ss.Pregnancy(fertility_rate=ss.peryear(150), p_loss=p_loss, rr_ptb=ss.constant(1), rr_ptb_age=np.array([[18, 35, 1000], [rr, rr, rr]]), burnin=False)
        sim = ss.Sim(n_agents=10_000, start=2000, stop=2003, dt=dt, demographics=preg, rand_seed=1, verbose=0).run()
        p = sim.demographics.pregnancy
        ga = p.gestation_at_birth.raw[~np.isnan(p.gestation_at_birth.raw)]
        return np.mean(ga < 37), 1 - sim.results.pregnancy.birth.sum()/np.nansum(p.n_pregnancies.raw)

    # TL29: before, 0.01 a step lost 44% weekly and 26% monthly
    weekly, monthly = [run(dt, p_loss=0.01)[1] for dt in [ss.weeks(1), ss.months(1)]]
    assert abs(weekly - monthly) < 0.04, f'The share lost should not depend on the time step: {weekly:.3f} weekly, {monthly:.3f} monthly'

    # TL30: before, the preterm share was 0.046 whatever the relative risk
    base, triple = [run(ss.weeks(1), rr=rr)[0] for rr in [1, 3]]
    p_pre = 0.001 + 0.002 + 0.005 + 0.012 + 0.026 # The default distribution's preterm durations, 32 to 36 weeks
    assert abs(base - p_pre) < 0.02 and abs(triple - 3*p_pre) < 0.03, f'Expected preterm shares of about {p_pre:.3f} and {3*p_pre:.3f}, not {base:.3f} and {triple:.3f}'
    return base, triple


@sc.timer()
def test_cause_of_death():
    """ Someone whose death two modules request in one step dies once, of one cause, counted once """
    sc.heading('Testing causes of death...')
    S, I = ss.states(2)
    sis = ss.Disease(infection=S >> I @ ss.peryear(0)*I, death=ss.death('I ->', rate=ss.peryear(1.0)), init=dict(I=10_000), name='sis')
    sim = ss.Sim(diseases=sis, demographics=ss.Deaths(death_rate=ss.peryear(1.0), rate_units=1), n_agents=10_000, start=2010, stop=2011, dt=ss.months(1), verbose=0).run()
    r = sim.results
    by_disease, by_deaths, total = r.sis.death.sum(), r.deaths.new.sum(), r.new_deaths.sum()
    assert by_disease + by_deaths == total # Before, about 4.5% were counted twice
    ppl = sim.people
    dead = ss.uids(np.flatnonzero(~ppl.alive.raw[:ppl.uid.len_used]))
    causes = np.array(ppl.causes)[ppl.cause_of_death.raw[dead]]
    assert (causes == 'sis').sum() == by_disease and (causes == 'deaths').sum() == by_deaths # One cause each, kept per person
    assert abs(by_disease - by_deaths) < 4*np.sqrt(total) # The same rate, so the same share of deaths, whichever module runs first
    return sim


@sc.timer()
def test_split_by_age():
    """ A split over age and another dimension, with each age group's row summing to 1, keeps people's ages, and newborns take the youngest group's row """
    sc.heading('Testing a split by age and another dimension...')
    S, I = ss.states(2)
    dis = ss.Disease(onset=S >> I @ ss.peryear(0.01), name='dis')
    dis.stratify(age=[0, 1, 5])
    dis.stratify(wealth=['poor', 'rich'], split=ss.by([[0, 1], [1, 0], [1, 0]], dims=['age', 'wealth']))
    sim = ss.Sim(diseases=dis, n_agents=5000, start=2000, stop=2002, rand_seed=1, verbose=0, demographics=[ss.Births(birth_rate=ss.peryear(0.04)), ss.Deaths(death_rate=ss.peryear(0.01))]).run()
    n = lambda t, age: (sim.results.dis.n_S.select(age=age).by('wealth').values[t] + sim.results.dis.n_I.select(age=age).by('wealth').values[t])
    assert n(0, 0).sum() < 0.05*5000 # TL28: before, each age group was a third of the people
    assert n(0, 1)[1] == 0 and n(0, 2)[1] == 0 # Only those under 1 are rich...
    assert n(-1, 0)[0] == 0 and n(-1, 0)[1] > 0 # ...and newborns too
    return sim


@sc.timer()
def test_declared_newborns():
    """ Agents added by ss.Births and ss.Pregnancy (including the unborn) enter a declared disease, with a level of each dimension """
    sc.heading('Testing newborns in a declared disease')
    for dem in [ss.Births(birth_rate=40), ss.Pregnancy(fertility_rate=40)]:
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=2), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.1, name='sis')
        sim = ss.Sim(sis, n_agents=2000, demographics=[dem, ss.Deaths(death_rate=10)], stratify=dict(risk=['lo', 'hi'], split=[0.2, 0.8]), start=2000, stop=2020, verbose=0, rand_seed=1)
        sim.run()
        r = sim.results
        n_states = r.sis.n_S[-1] + r.sis.n_I[-1]
        alive = r.n_alive[-1] + (r.pregnancy.n_unborn[-1] if 'pregnancy' in r else 0) # The unborn are in the disease's states, but not counted among the people
        assert np.isclose(n_states, alive), f'{dem.name}: {n_states} in states but {alive} alive'
        new = sim.people.auids[sim.people.auids >= 2000]
        assert len(new) > 100
        share = sim.people.risk[new].mean()
        assert 0.7 < share < 0.9, f'{dem.name}: {share:.2f} of newborns are high risk, not 0.8'
        assert sim.diseases.sis.I[new].any() # Newborns can be infected
    print('✓ (newborns in declared states)')
    return


if __name__ == '__main__':
    T = sc.timer()

    s1 = test_nigeria()
    s2 = test_nigeria(dt=1/12, which='pregnancy')
    s3 = test_constant_pop()
    s4 = test_module_adding()
    s5 = test_aging()
    s6 = test_pregnancy()
    s7 = test_loss_classification()
    s8 = test_background_loss()
    s9 = test_nnd()
    s10 = test_count_based_births()
    s11 = test_birth_data_cols()
    s12 = test_period_rates()
    s12b = test_lifetable_deaths()
    s13 = test_births_parents()
    s14 = test_unborn()
    s15 = test_pregnancy_risks()
    s16 = test_cause_of_death()
    s17 = test_split_by_age()
    s12 = test_declared_newborns()

    T.toc()
