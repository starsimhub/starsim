"""
Test different time units and timesteps
"""

import numpy as np
import sciris as sc
import starsim as ss

small = 100
medium = 1000
sc.options(interactive=False)

# ss.options.warnings = 'error'


# %% Define the tests
@sc.timer()
def test_ratio():
    sc.heading('Test datedur time ratio calculation')

    assert ss.dur(1) / ss.dur(1) == 1
    assert ss.dur(1) / ss.dur(0.1) == 10
    assert ss.dur(0.5) / ss.dur(5) == 0.1

    assert ss.datedur(years=1) / ss.datedur(days=1) == 365
    assert np.isclose(ss.datedur(years=1) / ss.datedur(weeks=1) * 7, 365)
    assert ss.datedur(years=1) / ss.datedur(months=1) == 12

    return


@sc.timer()
def test_classes():
    sc.heading('Test behavior of dur() and rate()')

    # Test duration dt
    d1 = ss.dur(2)
    d2 = ss.dur(3)
    d3 = ss.dur(2/0.1)
    d4 = ss.dur(3/0.2)

    assert d1 + d2 == 2+3
    assert d3 + d4 == 2/0.1 + 3/0.2
    assert d3 * 2 == 2/0.1*2
    assert d3 / 2 == 2/0.1/2

    # Test rate dt
    r1 = ss.freq(2)
    r2 = ss.freq(3)
    r3 = ss.freq(2/0.1)
    r4 = ss.freq(3/0.2)

    assert r1 + r2 == ss.freq(5)
    assert r3 + r4 == ss.freq(20+15)
    assert r3 * 2 == ss.freq(4/0.1)
    assert r4 / 2 == ss.freq(1.5/0.2)

    # Test duration units
    d5 = ss.datedur(years=2)
    d6 = ss.datedur(days=5)
    assert (d5 + d6).years == 2 + 5/365
    assert (d5 + d6)/ss.datedur(days=1) == 365*2+5

    # Test rate units
    rval = 0.7
    r5 = ss.freq(rval, ss.week)
    assert np.isclose(r5*ss.days(1), rval/7) # These should be close, but not match exactly
    assert np.isclose(r5*ss.weeks(0.1), rval/10)
    assert r5*ss.days(1) == rval * ss.days(1) / ss.weeks(1) # These should match exactly
    assert r5*ss.weeks(0.1) == rval * ss.weeks(0.1) / ss.weeks(1)

    # Test prob
    tpval = 0.1
    tp0 = ss.probperyear(tpval)
    assert np.isclose(tp0*ss.years(1), tpval), 'Multiplication by the base denominator should not change the value'
    assert np.isclose(tp0*ss.dur(0.5), tpval/2, rtol=0.1) # These should be close, but not match exactly
    assert np.isclose(tp0*ss.dur(2), tpval*2, rtol=0.1)
    assert tp0*ss.dur(0.5) == 1 - np.exp(np.log(1-0.1) * ss.dur(0.5)/ss.dur(1)) # These should be close, but not match exactly
    assert tp0*ss.dur(2) == 1 - np.exp(np.log(1-0.1) * ss.dur(2)/ss.dur(1)) # These should be close, but not match exactly

    return d3, d4, r3, r4, tp0


@sc.timer()
def test_time_class():
    sc.heading('Test different instances of ss.Timeline')

    def sim(start, stop, dt, **kwargs):
        """ Generate a fake sim """
        sim = sc.prettyobj()
        sim.t = ss.Timeline(start=start, stop=stop, dt=dt, **kwargs)
        sim.pars = ss.SimPars()
        sim.t.init(None)
        return sim

    print('Testing dates vs. numeric')
    s1 = sim(start=2000, stop=2002, dt=0.1)
    t1 = ss.Timeline(start='2001-01-01', stop='2001-06-30', dt=ss.days(2))
    t1.init(sim=s1)
    # assert np.array_equal(s1.t.timevec, s1.t.yearvec)
    assert len(s1.t.timevec) == 21
    assert t1.npts == sc.daydiff('2001-01-01', '2001-06-30')//2 + 1
    assert isinstance(s1.t.start, ss.date)
    assert isinstance(t1.start, ss.date)
    assert s1.t.tvec[-1] == ss.date('2002-01-01')
    assert t1.tvec[-1] == ss.date('2001-06-30')

    print('Testing weeks vs. days')
    s2 = sim(start='2000-06-01', stop='2001-05-01', dt=ss.days(1))
    t2 = ss.Timeline(start='2000-06-01', stop='2001-05-01', dt=ss.weeks(1))
    t2.init(sim=s2)
    assert np.array_equal(s2.t.timevec, s2.t.datevec)
    assert isinstance(s2.t.start, ss.date)
    assert t2.npts*7 == s2.t.npts + 1

    print('Testing different units and dt')
    s3 = sim(start=2001, stop=2003, dt=ss.years(0.1))
    t3 = ss.Timeline(start='2001-01-01', stop='2003-01-01', dt=ss.days(2))
    t3.init(sim=s3)
    assert np.array_equal(s3.t.timevec, s3.t.datevec)
    assert s3.t.datevec[-1] == ss.date('2003-01-01')
    assert s3.t.npts == 21

    print('Testing durations 1')
    s4 = sim(start=0, stop=ss.years(10), dt=1.0)
    assert s4.t.tvec[0] == ss.years(0)
    assert s4.t.datevec[-1] == ss.datedur(years=10)
    assert len(s4.t) == 11

    print('Testing durations 2')
    s4 = sim(start=0, stop=ss.datedur(months=10), dt=ss.datedur(months=1))
    assert s4.t.datevec[0] == ss.dur(0)
    assert s4.t.datevec[-1] == ss.datedur(months=10)
    assert len(s4.t) == 11

    print('Testing numeric 1')
    s5 = sim(start=None, stop=30, dt=None)
    assert s5.t.datevec[0] == ss.dur(0)
    assert s5.t.datevec[-1] == ss.datedur(years=30)
    assert len(s5.t) == 31

    print('Testing numeric 2')
    s6 = sim(start=2, stop=None, dt=None)  # Will default to start=dur(2), dur=ss.dur(50), end=start+dur
    assert s6.t.tvec[0] == ss.dur(2)
    assert s6.t.tvec[-1] == ss.datedur(years=52).years
    assert len(s6.t) == 51

    return [s1, t1, s2, t2]


@sc.timer()
def test_callable_dists():
    sc.heading('Testing callable distributions')
    def loc(module, sim, uids):
        return np.arange(uids)
    module = ss.mock_module(dt=ss.day)
    d = ss.normal(loc, 2, unit='days', module=module, strict=False)
    d.init()
    d.rvs(10)
    return d


@sc.timer()
def test_syntax():
    """ Verify that a range of supported operations run without raising an error """
    sc.heading('Testing syntax')
    from starsim import date, dur, datedur, years, prob, per

    assert float(date(1500))==1500
    assert np.isclose(float(date(1500.1)), 1500.1) # Not exactly equal, but very close

    assert np.all((years(1)*np.arange(5)) == (np.arange(5)*years(1)))

    tv = ss.Timeline(start=2001, stop=2003, dt=ss.years(0.1)) # Mixing floats and durs

    assert np.isclose(datedur(weeks=1)/datedur(days=1), 7) # TODO: would be nice if this were exact, but maybe impossible

    assert np.isclose(float(datedur(weeks=1) - datedur(days=1)), 6/365)

    assert np.isclose((date(2050)-date(2020)).years, years(30).years, rtol=1/365) # Not exact due to leap years

    assert np.isclose((ss.freqperweek(1)+ss.freqperday(1)).value, ss.freqperweek(8).value) # CKTODO: would be nice if this were exact

    assert date('2020-01-01') + datedur(weeks=52) == date('2020-12-30') # Should give us 30th December 2020
    assert date('2020-01-01') + 52*datedur(weeks=1) == date('2020-12-30')# Should give us 30th December 2020
    assert date('2020-01-01') + 52*dur(1/52) == date('2021-01-01') # Should give us 1st Jan 2021
    assert date('2020-01-01') + datedur(years=1) == date('2021-01-01') # Should give us 1st Jan 2021

    # These should all work - confirm the sizes
    assert len(ss.Timeline(date('2020-01-01'), date('2020-06-01'), ss.days(1)).init()) == 153
    assert len(ss.Timeline(date('2020-01-01'), date('2020-06-01'), ss.months(1)).init()) == 6
    assert len(ss.Timeline(ss.days(0), ss.days(30), ss.days(1)).init()) == 31
    assert len(ss.Timeline(ss.days(0), ss.months(1), ss.days(30)).init()) == 2
    assert len(ss.Timeline(ss.days(0), ss.years(1), ss.weeks(1)).init()) == 53
    assert len(ss.Timeline(ss.days(0), ss.years(1), ss.months(1)).init()) == 13
    assert len(ss.Timeline(start=ss.years(0), stop=ss.years(1), dt=ss.years(1/12)).init()) == 13
    assert len(ss.Timeline(date('2020-01-01'), date('2030-06-01'), ss.days(1)).init()) == 3805
    assert len(ss.Timeline(date(2020), date(2030.5), ss.years(0.1)).init()) == 106

    # Operations on date vectors
    date.arange(2020,2030)+years(1) # add years to date array
    date.arange(2020,2030)+datedur(years=1) # add datedur to date array

    # Construction of various duration ranges and addition with durations and dates
    dur.arange(dur(0),dur(10),dur(1)) + years(1)
    dur.arange(dur(0),dur(10), datedur(years=1)) + years(1)
    dur.arange(dur(0), datedur(years=10), datedur(years=1)) + years(1)

    # TODO: datedur calculations no longer work due to using __array_ufunc__ for performance -- could consider enabling later
    # dur.arange(dur(years=0), datedur(years=10), datedur(years=1)) + years(1)
    # dur.arange(dur(0),dur(10),dur(1)) + datedur(years=1)
    # dur.arange(dur(0),dur(10), datedur(years=1)) + datedur(years=1)
    # dur.arange(dur(0), datedur(years=10), datedur(years=1)) + datedur(years=1)
    # dur.arange(dur(years=0), datedur(years=10), datedur(years=1)) + datedur(years=1)
    dur.arange(dur(0),dur(10),dur(1)) + date(2000)
    dur.arange(dur(0),dur(10), datedur(years=1)) + date(2000)
    dur.arange(dur(0), datedur(years=10), datedur(years=1)) + date(2000)
    dur.arange(datedur(years=0), datedur(years=10), datedur(years=1)) + date(2000)

    # Rates
    assert (1/years(1)) == ss.freqperyear(1)
    assert (2/years(1)) == ss.freqperyear(2)
    assert (4/years(1)) == ss.freqperyear(4)
    assert (4/datedur(1)) == ss.freqperyear(4)
    assert (ss.freqperday(5)*datedur(days=1)) == 5
    assert 2/ss.freq(0.25) == dur(8)
    assert 1/(2*ss.freq(0.25)) == dur(2)
    assert ss.freq(0.5)/ss.freq(1) == 0.5

    # Probabilities
    p = prob(0.1, datedur(years=1))
    f = lambda factor: 1 - np.exp(-(-np.log(1 - p.value))/factor)
    assert p*datedur(years=2) == f(0.5)
    assert p * dur(0.5) == f(2)
    assert p * datedur(months=1) == f(12)

    p = prob(0.1, dur(1))
    assert p*datedur(years=2) == f(0.5)
    assert p * dur(0.5 ) == f(2)
    assert p * datedur(months=1) == f(12)

    p = per(0.1, datedur(years=1))
    f = lambda factor: 1 - np.exp(-p.value/factor)
    assert p*datedur(years=2) == f(0.5)
    assert p * dur(0.5) == f(2)
    assert p * datedur(months=1) == f(12)

    p = per(0.1, dur(1))
    assert p*datedur(years=2) == f(0.5)
    assert p * dur(0.5) == f(2)
    assert p * datedur(months=1) == f(12)

    return tv


@sc.timer()
def test_multi_timestep(do_plot=False):
    sc.heading('Test behavior of different modules having different timesteps')

    pars = dict(
        diseases = ss.SIS(dt=ss.days(1), init_prev=0.1, beta=ss.peryear(0.01)),
        demographics = ss.Births(dt=0.25),
        networks = ss.RandomNet(dt=ss.weeks(1)),
        n_agents = small,
        verbose = 0,
    )

    sim = ss.Sim(pars, dt=ss.days(2), start='2000-01-01', stop='2002-01-01')
    sim.run()

    twoyears = 366*2
    quarters = 2/0.25
    assert len(sim) == twoyears//2
    assert len(sim.diseases.sis) == twoyears
    assert len(sim.demographics.births) == quarters+1
    assert sim.t.tvec[-1] == ss.date('2001-12-31') # Every second day, this is the last time point
    assert sim.diseases.sis.t.tvec[-1] == ss.date('2002-01-01') # Every day, we can match the last time point exactly
    assert sim.demographics.births.t.tvec[-1] == ss.date('2002-01-01') # Every quarter, we can match the last time point exactly

    if do_plot:
        sim.plot()

    return sim


@sc.timer()
def test_mixed_timesteps():
    sc.heading('Test behavior of different combinations of timesteps')

    siskw = dict(dur_inf=ss.datedur(days=50), beta=ss.perday(0.01), waning=ss.perday(0.005))
    kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', copy_inputs=False, verbose=0)

    print('Year-year')
    sis1 = ss.SIS(dt=1/365, **sc.dcp(siskw))
    sim1 = ss.Sim(dt=1/365, diseases=sis1, label='year-year', **kw)

    print('Day-day')
    sis2 = ss.SIS(dt=ss.days(1), **sc.dcp(siskw))
    sim2 = ss.Sim(dt=ss.days(1), diseases=sis2, label='day-day', **kw)

    print('Day-year')
    sis3 = ss.SIS(dt=ss.days(1), **sc.dcp(siskw))
    sim3 = ss.Sim(dt=1/365, diseases=sis3, label='day-year', **kw)

    print('Year-day')
    sis4 = ss.SIS(dt=1/365, **sc.dcp(siskw))
    sim4 = ss.Sim(dt=ss.days(1), diseases=sis4, label='year-day', **kw)

    msim = ss.parallel(sim1, sim2, sim3, sim4)

    # Check that all results are close
    threshold = 0.02
    summary = msim.summarize()
    for key,res in summary.items():
        if res.mean:
            ratio = res.std/res.mean
            assert ratio < threshold, f'Result {key} exceeds threshold: {ratio:n} > {threshold}'
            print(f'✓ Result {key} within threshold: {ratio:n} < {threshold}')
    return msim


@sc.timer()
def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')

    sis = ss.SIS(
        beta = ss.probperday(0.05),
        init_prev = ss.bernoulli(p=0.1),
        dur_inf = ss.lognorm_ex(mean=ss.datedur(days=10)),
        waning = ss.perday(0.05),
        imm_boost = 1.0,
    )

    rnet = ss.RandomNet(
        n_contacts = 10,
        dur = 0, # Note; network edge durations are required to have the same unit as the network
    )

    pars = dict(
        diseases = sis,
        networks = rnet,
        n_agents = small,
    )

    sims = sc.objdict()
    sims.y = ss.Sim(pars, label='Year', start=2000, stop=2002, dt=1/365, verbose=0)
    sims.d = ss.Sim(pars, label='Day', start='2000-01-01', stop='2002-01-01', dt=ss.days(1), verbose=0)

    for sim in sims.values():
        sim.run()
        if do_plot:
            sim.plot()

    # Check that results match to within stochastic uncertainty
    rtol = 0.05
    vals = [sim.summary.sis_cum_infections for sim in [sims.y, sims.d]]
    assert np.isclose(*vals, rtol=rtol), f'Values for cum_infections do not match ({vals})'

    return sims


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer('\nTotal time')

    o1 = test_ratio()
    o2 = test_classes()
    o3 = test_time_class()
    o4 = test_callable_dists()
    o5 = test_syntax()
    o6 = test_multi_timestep(do_plot)
    o7 = test_mixed_timesteps()
    o8 = test_units(do_plot)

    T.toc()
