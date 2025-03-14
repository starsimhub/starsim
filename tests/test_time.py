"""
Test different time units and timesteps
"""

import numpy as np
import sciris as sc
import starsim as ss

small = 100
medium = 1000
sc.options(interactive=False)


# %% Define the tests
def test_ratio():
    sc.heading('Test time ratio calculation')

    assert ss.Dur(1)/ss.Dur(1) == 1
    assert ss.Dur(1)/ss.Dur(0.1) == 10
    assert ss.Dur(0.5)/ss.Dur(5) == 0.1

    assert ss.Dur(years=1)/ss.Dur(days=1) == 365.25
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
    assert np.isclose(r5*ss.days(1), rval/7) # A limitation of this approach, not exact!
    assert np.isclose(r5*ss.Dur(weeks=0.1), rval/10)

    # Test time_prob
    tpval = 0.1
    tp0 = ss.TimeProb(tpval)

    assert np.isclose(tp0*ss.Dur(1), tpval)
    assert np.isclose(tp0*ss.Dur(0.5), tpval/2, rtol=0.1)
    assert np.isclose(tp0*ss.Dur(2), tpval*2, rtol=0.1)
    assert tp0*ss.Dur(0.5) > tpval/2
    assert tp0*ss.Dur(2) < tpval*2

    return d3, d4, r3, r4, tp0


def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')

    sis = ss.SIS(
        beta = ss.TimeProb(0.05, ss.days(1)),
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
        diseases = ss.SIS(dt=ss.days(1), init_prev=0.1, beta=ss.TimeProb(0.01)),
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
    assert sim.t.tvec[-1] == ss.Date('2001-12-31') # Every second day, this is the last time point
    assert sim.diseases.sis.t.tvec[-1] == ss.Date('2002-01-01') # Every day, we can match the last time point exactly
    assert sim.demographics.births.t.tvec[-1] == ss.Date('2002-01-01') # Every quarter, we can match the last time point exactly

    if do_plot:
        sim.plot()

    return sim


def test_mixed_timesteps():
    sc.heading('Test behavior of different combinations of timesteps')

    siskw = dict(dur_inf=ss.Dur(days=50), beta=ss.TimeProb(0.01, ss.days(1)), waning=ss.Rate(0.005, ss.days(1)))
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

    def sim(start, stop, dt):
        """ Generate a fake sim """
        sim = sc.prettyobj()
        sim.t = ss.Time(start=start, stop=stop, dt=dt, sim=True)
        return sim

    print('Testing dates vs. numeric')
    s1 = sim(start=2000, stop=2002, dt=0.1)
    t1 = ss.Time(start='2001-01-01', stop='2001-06-30', dt=ss.days(2))
    t1.init(sim=s1)
    # assert np.array_equal(s1.t.timevec, s1.t.yearvec)
    assert len(s1.t.timevec) == 21
    assert t1.npts == sc.daydiff('2001-01-01', '2001-06-30')//2 + 1
    assert isinstance(s1.t.start, ss.Date)
    assert isinstance(t1.start, ss.Date)
    assert s1.t.tvec[-1] == ss.Date('2002-01-01')
    assert t1.tvec[-1] == ss.Date('2001-06-30')

    print('Testing weeks vs. days')
    s2 = sim(start='2000-06-01', stop='2001-05-01', dt=ss.days(1))
    t2 = ss.Time(start='2000-06-01', stop='2001-05-01', dt=ss.weeks(1))
    t2.init(sim=s2)
    assert np.array_equal(s2.t.timevec, s2.t.datevec)
    assert isinstance(s2.t.start, ss.Date)
    assert t2.npts*7 == s2.t.npts + 1

    print('Testing different units and dt')
    s3 = sim(start=2001, stop=2003, dt=ss.years(0.1))
    t3 = ss.Time(start='2001-01-01', stop='2003-01-01',dt=ss.days(2))
    t3.init(sim=s3)
    assert np.array_equal(s3.t.timevec, s3.t.datevec)
    assert s3.t.datevec[-1] == ss.Date('2003-01-01')
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

    return [s1, t1, s2, t2]


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    # o1 = test_ratio()
    # o2 = test_classes()
    # o3 = test_units(do_plot)
    # o4 = test_multi_timestep(do_plot)
    # o5 = test_mixed_timesteps()
    o6 = test_time_class()

    T.toc()
