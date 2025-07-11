"""
Test different time units and timesteps
"""

import numpy as np
import sciris as sc
import starsim as ss
from starsim.time import *

small = 100
medium = 1000
sc.options(interactive=False)


# %% Define the tests
def test_ratio():
    sc.heading('Test time ratio calculation')

    assert ss.Dur(1) / ss.Dur(1) == 1
    assert ss.Dur(1) / ss.Dur(0.1) == 10
    assert ss.Dur(0.5) / ss.Dur(5) == 0.1

    assert ss.Dur(years=1) / ss.Dur(days=1) == 365.25
    assert ss.Dur(years=1) / ss.Dur(weeks=1) * 7 == 365.25
    assert ss.Dur(years=1) / ss.Dur(months=1) == 12

    return


def test_classes():
    sc.heading('Test behavior of dur() and rate()')

    # Test duration dt
    d1 = ss.Dur(2)
    d2 = ss.Dur(3)
    d3 = ss.Dur(2/0.1)
    d4 = ss.Dur(3/0.2)

    assert d1 + d2 == 2+3
    assert d3 + d4 == 2/0.1 + 3/0.2
    assert d3 * 2 == 2/0.1*2
    assert d3 / 2 == 2/0.1/2

    # Test rate dt
    r1 = ss.Rate(2)
    r2 = ss.Rate(3)
    r3 = ss.Rate(2/0.1)
    r4 = ss.Rate(3/0.2)

    assert r1 + r2 == ss.Rate(5)
    assert r3 + r4 == ss.Rate(20+15)
    assert r3 * 2 == ss.Rate(4/0.1)
    assert r4 / 2 == ss.Rate(1.5/0.2)

    # Test duration units
    d5 = ss.Dur(years=2)
    d6 = ss.Dur(days=3)
    assert d5 + d6 == 2 + 3/365.25
    assert (d5 + d6)/ss.Dur(days=1) == 365.25*2+3

    # Test rate units
    rval = 0.7
    r5 = ss.Rate(rval, ss.Dur(weeks=1))
    assert np.isclose(r5*ss.days(1), rval/7) # These should be close, but not match exactly
    assert np.isclose(r5*ss.Dur(weeks=0.1), rval/10)
    assert r5*ss.days(1) == rval * ss.days(1) / ss.weeks(1) # These should match exactly
    assert r5*ss.Dur(weeks=0.1) == rval * ss.weeks(0.1) / ss.weeks(1)

    # Test timeprob
    tpval = 0.1
    tp0 = ss.timeprob(tpval)
    assert tp0*ss.Dur(1) == tpval, 'Multiplication by the base denominator should not change the value'
    assert np.isclose(tp0*ss.Dur(0.5), tpval/2, rtol=0.1) # These should be close, but not match exactly
    assert np.isclose(tp0*ss.Dur(2), tpval*2, rtol=0.1)
    assert tp0*ss.Dur(0.5) == 1 - np.exp(np.log(1-0.1) * ss.Dur(0.5)/ss.Dur(1)) # These should be close, but not match exactly
    assert tp0*ss.Dur(2) == 1 - np.exp(np.log(1-0.1) * ss.Dur(2)/ss.Dur(1)) # These should be close, but not match exactly

    return d3, d4, r3, r4, tp0


def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')

    sis = ss.SIS(
        beta = ss.timeprob(0.05, ss.days(1)),
        init_prev = ss.bernoulli(p=0.1),
        dur_inf = ss.lognorm_ex(mean=ss.Dur(days=10)),
        waning = ss.Rate(0.05, ss.days(1)),
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


def test_multi_timestep(do_plot=False):
    sc.heading('Test behavior of different modules having different timesteps')

    pars = dict(
        diseases = ss.SIS(dt=ss.days(1), init_prev=0.1, beta=ss.timeprob(0.01)),
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


def test_mixed_timesteps():
    sc.heading('Test behavior of different combinations of timesteps')

    siskw = dict(dur_inf=ss.Dur(days=50), beta=ss.timeprob(0.01, ss.days(1)), waning=ss.Rate(0.005, ss.days(1)))
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


def test_time_class():
    sc.heading('Test different instances of ss.Time')

    def sim(start, stop, dt, **kwargs):
        """ Generate a fake sim """
        sim = sc.prettyobj()
        sim.t = ss.Time(start=start, stop=stop, dt=dt, **kwargs)
        sim.pars = ss.SimPars()
        sim.t.init(None)
        return sim

    print('Testing dates vs. numeric')
    s1 = sim(start=2000, stop=2002, dt=0.1)
    t1 = ss.Time(start='2001-01-01', stop='2001-06-30', dt=ss.days(2))
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
    t2 = ss.Time(start='2000-06-01', stop='2001-05-01', dt=ss.weeks(1))
    t2.init(sim=s2)
    assert np.array_equal(s2.t.timevec, s2.t.datevec)
    assert isinstance(s2.t.start, ss.date)
    assert t2.npts*7 == s2.t.npts + 1

    print('Testing different units and dt')
    s3 = sim(start=2001, stop=2003, dt=ss.years(0.1))
    t3 = ss.Time(start='2001-01-01', stop='2003-01-01',dt=ss.days(2))
    t3.init(sim=s3)
    assert np.array_equal(s3.t.timevec, s3.t.datevec)
    assert s3.t.datevec[-1] == ss.date('2003-01-01')
    assert s3.t.npts == 21

    print('Testing durations 1')
    s4 = sim(start=0, stop=ss.Dur(10), dt=1.0)
    assert s4.t.datevec[0] == ss.Dur(0)
    assert s4.t.datevec[-1] == ss.Dur(years=10)
    assert len(s4.t) == 11

    print('Testing durations 2')
    s4 = sim(start=0, stop=ss.Dur(months=10), dt=ss.Dur(months=1))
    assert s4.t.datevec[0] == ss.Dur(0)
    assert s4.t.datevec[-1] == ss.Dur(months=10)
    assert len(s4.t) == 11

    print('Testing numeric 1')
    s5 = sim(start=None, stop=30, dt=None)
    assert s5.t.datevec[0] == ss.Dur(0)
    assert s5.t.datevec[-1] == ss.Dur(years=30)
    assert len(s5.t) == 31

    print('Testing numeric 2')
    s6 = sim(start=2, stop=None, dt=None)  # Will default to start=Dur(2), dur=Dur(50), end=start+dur
    assert s6.t.datevec[0] == ss.Dur(2)
    assert s6.t.datevec[-1] == ss.Dur(years=52)
    assert len(s6.t) == 51

    return [s1, t1, s2, t2]

def test_callable_dists():
    def loc(module, sim, uids):
        return np.array([ss.Dur(x) for x in range(uids)])
    module = sc.objdict(t=sc.objdict(dt=ss.Dur(days=1)))
    d = ss.normal(loc, ss.Dur(days=1), module=module, strict=False)
    d.init()
    d.rvs(10)

def test_pickling():
    import pickle
    x = ss.date('2020-01-01')
    s = pickle.dumps(x)
    pickle.loads(s)

def test_syntax():
    # Verify that a range of supported operations run without raising an error

    assert float(date(1500))==1500
    assert float(date(1500.1))==1500.1

    assert np.all((YearDur(1)*np.arange(5)) == (np.arange(5)*YearDur(1)))

    t = Time(start=2001, stop=2003, dt=ss.years(0.1)) # Mixing floats and durs

    assert Dur(weeks=1)/Dur(days=1) == 7

    assert np.isclose(float(DateDur(weeks=1) - DateDur(days=1)),6/365.25)

    assert date(2050) - date(2020) == YearDur(30)

    assert (perweek(1)+perday(1)) == perweek(8)

    assert date('2020-01-01') + Dur(weeks=52)   == date('2020-12-30') # Should give us 30th December 2020
    assert date('2020-01-01') + 52*Dur(weeks=1)  == date('2020-12-30')# Should give us 30th December 2020
    assert date('2020-01-01') + 52*Dur(1/52) == date('2021-01-01') # Should give us 1st Jan 2021
    assert date('2020-01-01') + Dur(years=1) == date('2021-01-01') # Should give us 1st Jan 2021

    # These should all work - confirm the sizes
    assert len(Time(date('2020-01-01'), date('2020-06-01'), Dur(days=1)).init()) == 153
    assert len(Time(date('2020-01-01'), date('2020-06-01'), Dur(months=1)).init()) == 6
    assert len(Time(Dur(days=0), Dur(days=30), Dur(days=1)).init()) == 31
    assert len(Time(Dur(days=0), Dur(months=1), Dur(days=30)).init()) == 2
    assert len(Time(Dur(days=0), Dur(years=1), Dur(weeks=1)).init()) == 53
    assert len(Time(Dur(days=0), Dur(years=1), Dur(months=1)) .init()) == 13
    assert len(Time(Dur(0), Dur(1), Dur(1/12)).init()) == 13
    assert len(Time(date('2020-01-01'), date('2030-06-01'), Dur(days=1)).init()) == 3805
    assert len(Time(date(2020), date(2030.5), Dur(0.1)).init()) == 106

    # Operations on date vectors
    date.arange(2020,2030)+YearDur(1) # add YearDur to date array
    date.arange(2020,2030)+DateDur(years=1) # add DateDur to date array

    # Construction of various duration ranges and addition with durations and dates
    Dur.arange(Dur(0),Dur(10),Dur(1)) + YearDur(1)
    Dur.arange(Dur(0),Dur(10),Dur(years=1)) + YearDur(1)
    Dur.arange(Dur(0),Dur(years=10),Dur(years=1)) + YearDur(1)
    Dur.arange(Dur(years=0),Dur(years=10),Dur(years=1)) + YearDur(1)
    Dur.arange(Dur(0),Dur(10),Dur(1)) + DateDur(years=1)
    Dur.arange(Dur(0),Dur(10),Dur(years=1)) + DateDur(years=1)
    Dur.arange(Dur(0),Dur(years=10),Dur(years=1)) + DateDur(years=1)
    Dur.arange(Dur(years=0),Dur(years=10),Dur(years=1)) + DateDur(years=1)
    Dur.arange(Dur(0),Dur(10),Dur(1)) + date(2000)
    Dur.arange(Dur(0),Dur(10),Dur(years=1)) + date(2000)
    Dur.arange(Dur(0),Dur(years=10),Dur(years=1)) + date(2000)
    Dur.arange(Dur(years=0),Dur(years=10),Dur(years=1)) + date(2000)

    # Rates
    assert (1/YearDur(1)) == ss.peryear(1)
    assert (2/YearDur(1)) == ss.peryear(2)
    assert (4/YearDur(1)) == ss.peryear(4)
    assert (4/DateDur(1)) == ss.peryear(4)
    assert (perday(5)*Dur(days=1)) == 5
    assert 2/Rate(0.25) == Dur(8)
    assert 1/(2*Rate(0.25)) == Dur(2)
    assert Rate(0.5)/Rate(1) == 0.5

    # Probabilities
    p = timeprob(0.1, Dur(years=1))
    f = lambda factor: 1 - np.exp(-(-np.log(1 - p.value))/factor)
    assert p*Dur(years=2) == f(0.5)
    assert p * Dur(0.5) == f(2)
    assert p * Dur(months=1) == f(12)

    p = timeprob(0.1, Dur(1))
    assert p*Dur(years=2) == f(0.5)
    assert p * Dur(0.5 ) == f(2)
    assert p * Dur(months=1) == f(12)

    p = rateprob(0.1, Dur(years=1))
    f = lambda factor: 1 - np.exp(-p.value/factor)
    assert p*Dur(years=2) == f(0.5)
    assert p * Dur(0.5) == f(2)
    assert p * Dur(months=1) == f(12)

    p = rateprob(0.1, Dur(1))
    assert p*Dur(years=2) == f(0.5)
    assert p * Dur(0.5) == f(2)
    assert p * Dur(months=1) == f(12)

# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    o1 = test_ratio()
    o2 = test_classes()
    o3 = test_units(do_plot)
    o4 = test_multi_timestep(do_plot)
    o5 = test_mixed_timesteps()
    o6 = test_time_class()
    test_callable_dists()
    test_syntax()
    T.toc()
