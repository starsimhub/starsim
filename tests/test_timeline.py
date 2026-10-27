"""
Test the Timeline object
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl

small = 100
medium = 1000
sc.options(interactive=False)
# ss.options.warnings = 'error' # For additional debugging

@sc.timer()
def test_timeline_lengths():
    sc.heading('Test different instances of ss.Timeline')

    # These should all work
    t = ss.Timeline(start=2001, stop=2003, dt=ss.years(0.1)) # Mixing floats and durs
    assert len(ss.Timeline(ss.date('2020-01-01'), ss.date('2020-06-01'), ss.days(1))) == 153
    assert len(ss.Timeline(ss.date('2020-01-01'), ss.date('2020-06-01'), ss.months(1))) == 6
    assert len(ss.Timeline(ss.days(0), ss.days(30), ss.days(1))) == 31
    assert len(ss.Timeline(ss.days(0), ss.months(1), ss.days(30))) == 2
    assert len(ss.Timeline(ss.days(0), ss.years(1), ss.weeks(1))) == 53
    assert len(ss.Timeline(ss.days(0), ss.years(1), ss.months(1))) == 13
    assert len(ss.Timeline(start=ss.years(0), stop=ss.years(1), dt=ss.years(1/12))) == 13
    assert len(ss.Timeline(ss.date('2020-01-01'), ss.date('2030-06-01'), ss.days(1))) == 3805
    assert len(ss.Timeline(ss.date(2020), ss.date(2030.5), ss.years(0.1))) == 106

    # Sub-daily steps are exact, even in a leap year
    t2 = ss.Timeline(ss.date('2000-01-01'), ss.date('2000-01-10'), ss.days(0.25)).init()
    assert len(t2) == 37
    assert np.allclose(t2.tvec.days_since('2000-01-01'), np.arange(37)*0.25)
    with pytest.warns(RuntimeWarning, match='max_steps'): # More steps than ss.options.max_steps
        ss.Timeline(ss.years(0), ss.years(200), ss.days(1)).init()

    return t


@sc.timer()
def test_timeline_combinations():
    sc.heading('Test additional combinations of inputs for ss.Timeline')

    # Compare day and week dt vs. float-year and date representations 
    tl1 = ss.Timeline(start=2000, stop=2005, dt='day')
    tl2 = ss.Timeline(start='2000-01-01', stop='2005-01-01', dt='day')
    tl3 = ss.Timeline(start=2000, stop=2020, dt='week')
    tl4 = ss.Timeline(start='2000-01-01', stop='2020-01-01', dt='week')
    
    assert len(tl1) == len(sc.daterange('2000-01-01', '2005-01-01')) - 2 # Leap years not counted
    assert len(tl2) == len(sc.daterange('2000-01-01', '2005-01-01')) # Leap years are counted if using dates
    assert len(tl3) == np.floor(20*365/7) + 1 # 365/7 is Starsim's approximation of the number of weeks in a year, +1 for inclusive range
    assert len(tl4) == len(sc.daterange('2000-01-01', '2020-01-01', interval='week'))

    # Corner cases: mixed dur types, datedur dt, long ranges, leap years
    tl5 = ss.Timeline(start=ss.days(0), stop=ss.days(365), dt='week') # Days start/stop, week dt (type mismatch)
    tl6 = ss.Timeline(start=ss.years(2000), stop=ss.years(2010), dt=ss.datedur(months=1))  # datedur dt with dur start
    tl7 = ss.Timeline(start=2000, stop=2100, dt='week') # Long range amplifies float discrepancy
    tl8 = ss.Timeline(start=2000, stop=2001, dt='day') # Leap year (2000) with day dt

    assert len(tl5) == 53   # 365 days / 7 days per week + 1 inclusive
    assert len(tl6) == 121  # 10 years * 12 months + 1 inclusive
    assert len(tl7) == int(np.floor(100*365/7)) + 1
    assert len(tl8) == 366  # 2000 is a leap year: 366 days, but float-year path uses 365.25

    return tl1


@sc.timer()
def test_timeline():
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
    assert len(s1.t.timevec) == 21
    assert t1.npts == sc.daydiff('2001-01-01', '2001-06-30')//2 + 1
    assert isinstance(s1.t.start, ss.years)
    assert isinstance(t1.start, ss.date)
    assert s1.t.tvec[-1] == ss.years(2002)
    assert np.array_equal(s1.t.relvec[:3], [0, 0.1, 0.2]) # Numbers in the sim's units, not ss.datedur objects
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
    assert s4.t.datevec[0] == ss.months(0)
    assert s4.t.datevec[-1] == ss.datedur(months=10)
    assert len(s4.t) == 11

    print('Testing numeric 1')
    s5 = sim(start=None, stop=30, dt=None)
    assert s5.t.datevec[0] == ss.years(0)
    assert s5.t.datevec[-1] == ss.datedur(years=30)
    assert len(s5.t) == 31

    print('Testing numeric 2')
    s6 = sim(start=2, stop=None, dt=None)  # Will default to start=ss.years(2), dur=ss.years(50), end=start+dur
    assert s6.t.tvec[0] == ss.years(2)
    assert s6.t.tvec[-1] == ss.datedur(years=52).years
    assert len(s6.t) == 51

    print('Testing that a timeline is not a timestep index')
    with pytest.raises(TypeError):
        s1.t == 10 # Should be s1.ti == 10
    assert s1.t == s1.t and s1.t != t1 # Comparing timelines still uses identity

    return [s1, t1, s2, t2]


@sc.timer()
def test_timeline_syntax():
    sc.heading('Test all ss.Timeline input options')

    kw = sc.objdict()

    # Categories of tests
    keymap = dict(
        a = 'start',
        b = 'stop',
        c = 'dur',
        d = 'dt',
        e = 'start and stop',
        f = 'start and dur',
        g = 'stop and dur',
        h = 'multiple',
    )
    dd20 = ss.date(2010) - ss.date(1990) # This was ss.datedur(days=7305), but is now just ss.datedur(years=20), since we switched from pd.Timedelta to dateutil.relativedelta

    # Test start
    kw.a1 = [dict(start=None, stop=None, dur=None, dt=None)           , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.a2 = [dict(start=1990, stop=None, dur=None, dt=None)           , dict(start=ss.years(1990), stop=ss.years(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a3 = [dict(start=ss.years(1990), stop=None, dur=None, dt=None) , dict(start=ss.years(1990), stop=ss.years(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a4 = [dict(start='1990.1.1', stop=None, dur=None, dt=None)     , dict(start=ss.date(1990), stop=ss.date(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a5 = [dict(start=ss.date(1990), stop=None, dur=None, dt=None)  , dict(start=ss.date(1990), stop=ss.date(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a6 = [dict(start=ss.days(5), stop=None, dur=None, dt=None)     , dict(start=ss.days(5), stop=ss.days(55), dur=ss.days(50), dt=ss.days(1))]

    # Test stop
    kw.b1 = [dict(start=None, stop=2010, dur=None, dt=None)           , dict(start=ss.years(1960), stop=ss.years(2010), dur=ss.years(50), dt=ss.years(1))]
    kw.b2 = [dict(start=None, stop=1990, dur=None, dt=None)           , dict(start=ss.years(1940), stop=ss.years(1990), dur=ss.years(50), dt=ss.years(1))]
    kw.b3 = [dict(start=None, stop='1990.1.1', dur=None, dt=None)     , dict(start=ss.date(1940), stop=ss.date(1990), dur=ss.years(50), dt=ss.years(1))]
    kw.b4 = [dict(start=None, stop=ss.years(2010), dur=None, dt=None) , dict(start=ss.years(1960), stop=ss.years(2010), dur=ss.years(50), dt=ss.years(1))]
    kw.b5 = [dict(start=None, stop=ss.date(2050), dur=None, dt=None)  , dict(start=ss.date(2000), stop=ss.date(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.b6 = [dict(start=None, stop=ss.days(100), dur=None, dt=None)   , dict(start=ss.days(0), stop=ss.days(100), dur=ss.days(100), dt=ss.days(1))]

    # Test dur
    kw.c1 = [dict(start=None, stop=None, dur=10, dt=None)                    , dict(start=ss.years(2000), stop=ss.years(2010), dur=ss.years(10), dt=ss.years(1))]
    kw.c2 = [dict(start=None, stop=None, dur=ss.years(10), dt=None)          , dict(start=ss.years(2000), stop=ss.years(2010), dur=ss.years(10), dt=ss.years(1))]
    kw.c3 = [dict(start=None, stop=None, dur=ss.datedur(months=24), dt=None) , dict(start=ss.date(2000), stop=ss.date(2002), dur=ss.datedur(months=24), dt=ss.years(1))]
    kw.c4 = [dict(start=None, stop=None, dur=ss.days(50), dt=None)           , dict(start=ss.years(2000), stop=ss.years(2000)+ss.days(50), dur=ss.days(50), dt=ss.days(1))]
    kw.c5 = [dict(start=None, stop=None, dur='1990.1.1', dt=None)            , 'exception']

    # Test dt
    kw.d1 = [dict(start=None, stop=None, dur=None, dt=1)                    , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.d2 = [dict(start=None, stop=None, dur=None, dt=ss.years(1))          , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.d3 = [dict(start=None, stop=None, dur=None, dt=ss.days(1))           , dict(start=ss.years(2000), stop=ss.years(2000)+ss.days(50), dur=ss.days(50), dt=ss.days(1))]
    kw.d4 = [dict(start=None, stop=None, dur=None, dt='month')              , dict(start=ss.years(2000), stop=ss.years(2000)+ss.months(50), dur=ss.months(50), dt=ss.months(1))]
    kw.d5 = [dict(start=None, stop=None, dur=None, dt=ss.datedur(months=1)) , dict(start=ss.date(2000), stop=ss.date(2050), dur=ss.years(50), dt=ss.datedur(months=1))]

    # Test start and stop
    kw.e1 = [dict(start=1990, stop=2010, dur=None, dt=None)                , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.e2 = [dict(start=ss.years(1990), stop=2010, dur=None, dt=None)      , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.e3 = [dict(start=1990, stop=ss.years(2010), dur=None, dt=None)      , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.e4 = [dict(start=ss.date(1990), stop=2010, dur=None, dt=None)       , dict(start=ss.date(1990), stop=ss.date(2010), dur=dd20, dt=ss.years(1))]
    kw.e5 = [dict(start=1990, stop=ss.date(2010), dur=None, dt=None)       , dict(start=ss.date(1990), stop=ss.date(2010), dur=dd20, dt=ss.years(1))]
    kw.e6 = [dict(start=ss.years(0), stop=ss.days(365), dur=None, dt=None) , dict(start=ss.years(0), stop=ss.years(1), dur=ss.years(1), dt=ss.years(1))]

    # Test start and dur
    kw.f1 = [dict(start=1990, stop=None, dur=20, dt=None)                   , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.f2 = [dict(start=1990, stop=None, dur=ss.years(20), dt=None)         , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.f3 = [dict(start=1990, stop=None, dur=ss.datedur(years=20), dt=None) , dict(start=ss.date(1990), stop=ss.date(2010), dur=ss.datedur(years=20), dt=ss.years(1))]
    kw.f4 = [dict(start=1990, stop=None, dur='year', dt=None)               , 'exception']

    # Test stop and dur
    kw.g1 = [dict(start=None, stop=1990, dur=20, dt=None)                   , dict(start=ss.years(1970), stop=ss.years(1990), dur=ss.years(20), dt=ss.years(1))]
    kw.g2 = [dict(start=None, stop=ss.date(1990), dur=20, dt=None)          , dict(start=ss.date(1970), stop=ss.date(1990), dur=ss.years(20), dt=ss.years(1))]
    kw.g3 = [dict(start=None, stop=ss.date('1990-01-21'), dur=ss.days(20), dt=None) , dict(start=ss.date('1990.1.1'), stop=ss.date('1990.1.21'), dur=ss.days(20), dt=ss.days(1))]
    kw.g4 = [dict(start=1990, stop=2020, dur=20, dt=None)                   , 'exception']

    # Test multiple
    kw.h1 = [dict(start=ss.years(1990), stop=ss.date(2010), dur=None, dt='month') , dict(start=ss.date(1990), stop=ss.date(2010), dur=dd20, dt=ss.months(1))]
    kw.h2 = [dict(start=1990, stop=2010, dur=None, dt=ss.datedur(months=1))       , dict(start=ss.date(1990), stop=ss.years(2010), dur=dd20, dt=ss.datedur(months=1))]
    kw.h3 = [dict(start=1990, dur=40, dt='month')                                 , dict(start=ss.years(1990), stop=ss.years(1990)+ss.months(40), dur=ss.months(40), dt=ss.months(1))]


    mismatches = []
    for key in kw:
        category = keymap[key[0]]
        sc.printcyan(f'\nTesting {category}: {key}')
        args,expected = kw[key]
        if expected != 'exception':
            t = ss.Timeline(**args).init()
            actual = dict(start=t.start, stop=t.stop, dur=t.dur, dt=t.dt)
        else:
            try:
                t = ss.Timeline(**args).init()
                actual = 'passed'
            except:
                actual = expected

        print('Input:    ', args)
        print('Expected: ', expected)
        print('Actual:   ', actual)
        if expected == actual:
            sc.printgreen('✓ Success')
        else:
            sc.printred('× Mismatch :(')
            mismatches.append(sc.objdict(key=key, args=args, expected=expected, actual=actual))

    if len(mismatches):
        errormsg = f'{len(mismatches)} output(s) did not match expected value(s):\n'
        for m in mismatches:
            errormsg += f'\n  Key: {m.key}'
            errormsg += f'\n  Input:    {m.args}'
            errormsg += f'\n  Expected: {m.expected}'
            errormsg += f'\n  Actual:   {m.actual}\n'
        raise ValueError(errormsg)

    return kw


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
def test_step_count():
    """
    A disease with dt='month' should step exactly once per timestep, even when
    the sim's dt is ss.years(1/12) (type mismatch: years vs months).

    When the disease sets dt='month' via its pars (propagated to self.t via
    update_pars), and the sim uses dt=ss.years(1/12), the Timeline type check
    (years != months) prevents tvec sharing. Both build tvecs independently,
    introducing floating-point differences (~2e-13) that cause the loop planner
    to create duplicate plan entries — the disease steps twice on some timesteps.
    """
    sc.heading('Test no double-stepping')

    for sim_dt, label in [(ss.months(1), 'ss.months(1)'), (ss.years(1/12), 'ss.years(1/12)')]:
        sim = ss.Sim(
            diseases=ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.years(2), dt='month'),
            networks='random', n_agents=100, dur=ss.years(1), dt=sim_dt,
        )
        sim.init()
        plan = sim.loop.to_df()
        disease_steps = plan[plan.label.str.contains('sis.step_state')]
        counts_per_ti = disease_steps.groupby('ti').size()
        max_calls = counts_per_ti.max()
        assert max_calls == 1, f"With sim dt={label}: step_state scheduled {max_calls}x on a single timestep (should be 1)"
        print(f'  dt={label}: OK (max 1 step_state per timestep)')

    return sim


@sc.timer()
def test_coded_unit():
    sc.heading('Test that the unit of a sim with only coded modules is honored')

    class Step(ss.Module): # A module written as code, with no declared states or transitions
        def step(self):
            pass

    sim = ss.Sim(connectors=Step(), unit='day', dt=1, start=0, stop=30, verbose=0).init()
    assert sim.unit == 'day' and sim.t.dt == ss.days(1) and len(sim.t) == 31
    sim = ss.Sim(connectors=Step(), unit='week', stop=10, verbose=0).init()
    assert sim.t.dt == ss.weeks(1) and len(sim.t) == 11
    sim = ss.Sim(connectors=Step(), verbose=0).init() # Nothing has a unit: time is in timesteps
    assert sim.unit is None and sim.t.start == 0 and len(sim.t) == 51
    return sim


def test_sim_start_stop_dur():
    sc.heading('A sim given a stop and dur starts at stop - dur, and raises if start, stop, and dur disagree')
    kw = dict(diseases='sis', networks='random', n_agents=small, verbose=0)
    sim = ss.Sim(stop=2035, dur=10, **kw).init()
    assert sim.t.start == ss.years(2025) and sim.t.stop == ss.years(2035)
    sim = ss.Sim(stop=100, dur=30, unit='day', **kw).init() # Also in a unit other than years
    assert sim.t.start == ss.days(70)
    sim = ss.Sim(start=2000, stop=2010, dur=10, **kw).init() # Three that agree are fine
    with pytest.raises(ValueError, match='disagree'):
        ss.Sim(start=2000, stop=2035, dur=10, **kw).init()
    return sim


# Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer('\nTotal time')

    o1 = test_timeline_lengths()
    o2 = test_timeline_combinations()
    o3 = test_timeline()
    o4 = test_timeline_syntax()
    o5 = test_multi_timestep(do_plot)
    o8 = test_step_count()
    o9 = test_coded_unit()
    o10 = test_sim_start_stop_dur()

    T.toc()
