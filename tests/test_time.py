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
    sc.heading('Test behavior of time_ratio()')

    assert ss.time_ratio() == 1.0
    assert ss.time_ratio(dt1=1.0, dt2=0.1) == 10.0
    assert ss.time_ratio(dt1=0.5, dt2=5) == 0.1
    assert ss.time_ratio(dt1=0.5, dt2=5) == 0.1

    assert ss.time_ratio(unit1='year', unit2='day') == 365.25
    assert ss.time_ratio(unit1='day', unit2='year') == 1/365.25

    return


def test_classes():
    sc.heading('Test behavior of dur() and rate()')

    # Test duration dt
    d1 = ss.dur(2)
    d2 = ss.dur(3)
    d3 = ss.dur(2, parent_dt=0.1)
    d4 = ss.dur(3, parent_dt=0.2)
    for d in [d1,d2,d3,d4]: d.init()

    assert d1 + d2 == 2+3
    assert d3 + d4 == 2/0.1 + 3/0.2
    assert d3 * 2 == 2/0.1*2
    assert d3 / 2 == 2/0.1/2

    # Test rate dt
    r1 = ss.rate(2)
    r2 = ss.rate(3)
    r3 = ss.rate(2, parent_dt=0.1)
    r4 = ss.rate(3, parent_dt=0.2)
    for r in [r1,r2,r3,r4]: r.init()

    assert r1 + r2 == 2 + 3
    assert r3 + r4 == 2*0.1 + 3*0.2
    assert r3 * 2 == 2*0.1*2
    assert r3 / 2 == 2*0.1/2

    # Test duration units
    d5 = ss.dur(2, unit='year').init(parent_unit='day')
    d6 = ss.dur(3, unit='day').init(parent_unit='day')
    assert d5 + d6 == 2*365.25+3

    # Test rate units
    rval = 0.7
    r5 = ss.rate(rval, unit='week').init(parent_unit='day')
    r6 = ss.rate(rval).init(parent_dt=0.1)
    assert np.isclose(r5.values, rval/7) # A limitation of this approach, not exact!
    assert np.isclose(r6.values, rval/10)

    # Test time_prob
    tpval = 0.1
    tp0 = ss.time_prob(tpval).init(parent_dt=1.0)
    tp1 = ss.time_prob(tpval).init(parent_dt=0.5)
    tp2 = ss.time_prob(tpval).init(parent_dt=2)
    assert np.isclose(tp0.values, tpval)
    assert np.isclose(tp1.values, tpval/2, rtol=0.1)
    assert np.isclose(tp2.values, tpval*2, rtol=0.1)
    assert tp1.values > tpval/2
    assert tp2.values < tpval*2

    return d3, d4, r3, r4, tp1


def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')

    sis = ss.SIS(
        beta = ss.beta(0.05, 'day'),
        init_prev = ss.bernoulli(p=0.1),
        dur_inf = ss.lognorm_ex(mean=ss.dur(10, 'day')),
        waning = ss.rate(0.05, 'day'),
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
    sims.y = ss.Sim(pars, unit='year', label='Year', start=2000, stop=2002, dt=1/365, verbose=0)
    sims.d = ss.Sim(pars, unit='day', label='Day', start='2000-01-01', stop='2002-01-01', dt=1, verbose=0)

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
        diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1, beta=ss.beta(0.01)),
        demographics = ss.Births(unit='year', dt=0.25),
        networks = ss.RandomNet(unit='week'),
        n_agents = small,
        verbose = 0,
    )

    sim = ss.Sim(pars, unit='day', dt=2, start='2000-01-01', stop='2002-01-01')
    sim.run()

    twoyears = 366*2
    quarters = 2/0.25
    assert len(sim) == twoyears//2
    assert len(sim.diseases.sis) == twoyears
    assert len(sim.demographics.births) == quarters

    if do_plot:
        sim.plot()

    return sim


def test_mixed_timesteps():
    sc.heading('Test behavior of different combinations of timesteps')

    siskw = dict(dur_inf=ss.dur(50, 'day'), beta=ss.beta(0.01, 'day'), waning=ss.rate(0.005, 'day'))
    kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', copy_inputs=False, verbose=0)

    print('Year-year')
    sis1 = ss.SIS(unit='year', dt=1/365, **sc.dcp(siskw))
    sim1 = ss.Sim(unit='year', dt=1/365, diseases=sis1, label='year-year', **kw)

    print('Day-day')
    sis2 = ss.SIS(unit='day', dt=1.0, **sc.dcp(siskw))
    sim2 = ss.Sim(unit='day', dt=1.0, diseases=sis2, label='day-day', **kw)

    print('Day-year')
    sis3 = ss.SIS(unit='day', dt=1.0, **sc.dcp(siskw))
    sim3 = ss.Sim(unit='year', dt=1/365, diseases=sis3, label='day-year', **kw)

    print('Year-day')
    sis4 = ss.SIS(unit='year', dt=1/365, **sc.dcp(siskw))
    sim4 = ss.Sim(unit='day', dt=1.0, diseases=sis4, label='year-day', **kw)

    msim = ss.parallel(sim1, sim2, sim3, sim4)

    # Check that all results are close
    threshold = 0.01
    summary = msim.summarize()
    for key,res in summary.items():
        if res.mean:
            ratio = res.std/res.mean
            assert ratio < threshold, f'Result {key} exceeds threshold: {ratio:n} > {threshold}'
            print(f'✓ Result {key} within threshold: {ratio:n} < {threshold}')
    return msim


def test_time_class():
    sc.heading('Test different instances of ss.Time')

    def sim(start, stop, dt, unit):
        """ Generate a fake sim """
        sim = sc.prettyobj()
        sim.t = ss.Time(start=start, stop=stop, dt=dt, unit=unit, sim=True)
        return sim

    print('Testing dates vs. numeric')
    s1 = sim(start=2000, stop=2002, dt=0.1, unit='year')
    t1 = ss.Time(start='2001-01-01', stop='2001-06-30', dt=2.0, unit='day')
    t1.init(sim=s1)
    assert np.array_equal(s1.t.timevec, s1.t.yearvec)
    assert len(s1.t.timevec) == 21
    assert t1.npts == sc.daydiff('2001-01-01', '2001-06-30')//2 + 1
    assert sc.isnumber(s1.t.start)
    assert isinstance(t1.start, ss.date)
    assert s1.t.datevec[-1] == ss.date('2002-01-01')
    assert t1.datevec[-1] == ss.date('2001-06-30')
    assert t1.abstvec[0] == 1.0

    print('Testing weeks vs. days')
    s2 = sim(start='2000-06-01', stop='2001-05-01', dt=1, unit='day')
    t2 = ss.Time(start='2000-06-01', stop='2001-05-01', dt=1, unit='week')
    t2.init(sim=s2)
    assert np.array_equal(s2.t.timevec, s2.t.datevec)
    assert isinstance(s2.t.start, ss.date)
    assert t2.npts*7 == s2.t.npts + 1
    assert np.array_equal(t2.abstvec, s2.t.abstvec[::7])

    print('Testing different units and dt')
    s3 = sim(start='2001-01-01', stop='2003-01-01', dt=0.1, unit='year')
    t3 = ss.Time(start='2001-01-01', stop='2003-01-01', dt=2.0, unit='day')
    t3.init(sim=s3)
    assert np.array_equal(s3.t.timevec, s3.t.datevec)
    assert s3.t.datevec[-1] == ss.date('2003-01-01')
    assert s3.t.npts == 21
    assert np.isclose(s3.t.abstvec.mean(), t3.abstvec.mean(), atol=1e-3)

    print('Testing unitless')
    s4 = sim(start=0, stop=10, dt=1.0, unit='unitless')
    t4 = ss.Time(start=2, stop=9, dt=0.1, unit='unitless')
    t4.init(sim=s4)
    assert np.array_equal(s4.t.timevec, s4.t.abstvec)
    assert s4.t.datevec[0] == ss.date(ss.time.default_start_year)
    assert len(s4.t) == 11
    assert len(t4) == (9-2)/0.1+1

    return [s1, t1, s2, t2]


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

    T.toc()
