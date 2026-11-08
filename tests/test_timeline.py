"""
Test the Timeline object
"""
import warnings
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
    tl6 = ss.Timeline(start=ss.years(2000), dur=ss.years(10), dt=ss.months(1))  # Calendar months as the step, with a start that is a duration (a time from 0)
    tl7 = ss.Timeline(start=2000, stop=2100, dt='week') # Long range amplifies float discrepancy
    tl8 = ss.Timeline(start=2000, stop=2001, dt='day') # Leap year (2000) with day dt

    assert len(tl5) == 53   # 365 days / 7 days per week + 1 inclusive
    assert len(tl6) == 121  # 10 years * 12 months + 1 inclusive
    assert len(tl7) == int(np.floor(100*365/7)) + 1
    assert len(tl8) == 366  # Numeric years are 365 days, so 365 steps +1 inclusive, even though 2000 is a leap year

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
    assert np.array_equal(s1.t.relvec[:3], [0, 0.1, 0.2]) # Numbers in the sim's units, not ss.dates.datedur objects
    assert t1.tvec[-1] == ss.date('2001-06-30')

    print('Testing weeks vs. days')
    s2 = sim(start='2000-06-01', stop='2001-05-01', dt=ss.days(1))
    t2 = ss.Timeline(start='2000-06-01', stop='2001-05-01', dt=ss.weeks(1))
    t2.init(sim=s2)
    assert np.array_equal(s2.t.timevec, s2.t.datevec)
    assert isinstance(s2.t.start, ss.date)
    assert t2.npts == (s2.t.npts - 1)//7 + 1 # Whole weeks in the span, +1 inclusive

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
    assert s4.t.datevec[-1] == ss.years(10)
    assert len(s4.t) == 11

    print('Testing durations 2')
    s4 = sim(start=0, stop=ss.months(10), dt=ss.months(1))
    assert s4.t.tvec[0] == ss.months(0)
    assert s4.t.tvec[-1] == ss.months(10)
    assert len(s4.t) == 11

    print('Testing numeric 1')
    s5 = sim(start=None, stop=30, dt=None) # A bare stop before 1800, alone, is a time from 0
    assert s5.t.datevec[0] == ss.years(0)
    assert s5.t.datevec[-1] == ss.years(30)
    assert len(s5.t) == 31

    print('Testing numeric 2')
    s6 = sim(start=2, stop=None, dt=None)  # Will default to start=ss.years(2), a length of ss.years(50)
    assert s6.t.tvec[0] == ss.years(2)
    assert s6.t.tvec[-1] == ss.years(52).years
    assert len(s6.t) == 51

    print('Testing that a timeline is not a timestep index')
    with pytest.raises(TypeError):
        s1.t == 10 # Should be s1.ti == 10
    assert s1.t == s1.t and s1.t != t1 # Comparing timelines still uses identity

    return [s1, t1, s2, t2]


@sc.timer()
def test_timeline_syntax():
    sc.heading('Test all ss.Timeline input options')

    kw = sc.obj()

    # Categories of tests
    keymap = dict(
        a = 'start',
        b = 'stop',
        c = 'dur (the length of the run)',
        d = 'dt',
        e = 'start and stop',
        f = 'start and a stop that is a duration (a time, as the start is)',
        g = 'stop with no start (a calendar stop starts in 2000)',
        h = 'multiple',
    )
    dd20 = ss.date(2010) - ss.date(1990) # The time between two dates, in days: ss.days(7305)
    dd30 = ss.date(2030) - ss.date(2000)

    # Test start: a date or a bare year (from 1800 on) gives the timeline a calendar; any other number, or a duration, is a time from 0
    kw.a1 = [dict(start=None, stop=None, dt=None)           , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.a2 = [dict(start=1990, stop=None, dt=None)           , dict(start=ss.years(1990), stop=ss.years(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a3 = [dict(start=ss.years(1990), stop=None, dt=None) , dict(start=ss.years(1990), stop=ss.years(2040), dur=ss.years(50), dt=ss.years(1))] # 1,990 years from 0: no calendar
    kw.a4 = [dict(start='1990.1.1', stop=None, dt=None)     , dict(start=ss.date(1990), stop=ss.date(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a5 = [dict(start=ss.date(1990), stop=None, dt=None)  , dict(start=ss.date(1990), stop=ss.date(2040), dur=ss.years(50), dt=ss.years(1))]
    kw.a6 = [dict(start=ss.days(5), stop=None, dt=None)     , dict(start=ss.days(5), stop=ss.days(105), dur=ss.days(100), dt=ss.days(1))]

    # Test stop
    kw.b1 = [dict(start=None, stop=2010, dt=None)           , dict(start=ss.years(2000), stop=ss.years(2010), dur=ss.years(10), dt=ss.years(1))] # A calendar stop starts at the default start, 2000
    kw.b2 = [dict(start=None, stop=1990, dt=None)           , 'exception'] # Before the default start
    kw.b3 = [dict(start=None, stop='1990.1.1', dt=None)     , 'exception']
    kw.b4 = [dict(start=None, stop=ss.years(20), dt=None)   , dict(start=ss.years(0), stop=ss.years(20), dur=ss.years(20), dt=ss.years(1))] # A typed stop is a time, as a start is: 20 years from 0
    kw.b7 = [dict(start=None, stop=ss.years(2030), dt=None) , dict(start=ss.years(2000), stop=ss.years(2030), dur=ss.years(30), dt=ss.years(1))] # A typed calendar year starts at the default start, 2000
    kw.b5 = [dict(start=None, stop=ss.date(2030), dt=None)  , dict(start=ss.date(2000), stop=ss.date(2030), dur=dd30, dt=ss.years(1))]
    kw.b6 = [dict(start=None, stop=ss.days(100), dt=None)   , dict(start=ss.days(0), stop=ss.days(100), dur=ss.days(100), dt=ss.days(1))]

    # Test dur, the length of the run: the stop is the start plus it
    kw.c1 = [dict(start=None, stop=None, dur=10, dt=None)                , dict(start=ss.years(2000), stop=ss.years(2010), dur=ss.years(10), dt=ss.years(1))]
    kw.c2 = [dict(start=None, stop=None, dur=ss.years(10), dt=None)      , dict(start=ss.years(2000), stop=ss.years(2010), dur=ss.years(10), dt=ss.years(1))]
    kw.c3 = [dict(start=1990, stop=None, dur=ss.years(1990), dt=None)    , dict(start=ss.years(1990), stop=ss.years(3980), dur=ss.years(1990), dt=ss.years(1))] # Always the length, never a year
    kw.c4 = [dict(start=ss.date(1990), stop=None, dur=ss.years(20), dt=None), dict(start=ss.date(1990), stop=ss.date(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.c5 = [dict(start=None, stop=None, dur=ss.days(100), dt=None)      , dict(start=ss.days(0), stop=ss.days(100), dur=ss.days(100), dt=ss.days(1))]
    kw.c6 = [dict(start=2000, stop=2010, dur=10, dt=None)                , 'exception'] # Both the stop and the length
    kw.c7 = [dict(start=2000, stop=None, dur='2010-01-01', dt=None)      , 'exception'] # A date is a stop, not a length

    # Test dt: with no stop, the length is 50 of the unit, whatever the type of the step
    kw.d1 = [dict(start=None, stop=None, dt=1)                 , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.d2 = [dict(start=None, stop=None, dt=ss.years(1))       , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.years(1))]
    kw.d3 = [dict(start=None, stop=None, dt=ss.days(1))        , dict(start=ss.days(0), stop=ss.days(100), dur=ss.days(100), dt=ss.days(1))] # Alone, the unit is that of the first duration given
    kw.d4 = [dict(start=None, stop=None, dt='month')           , dict(start=ss.months(0), stop=ss.months(50), dur=ss.months(50), dt=ss.months(1))]
    kw.d5 = [dict(start=2000, stop=None, dt=ss.months(1))      , dict(start=ss.years(2000), stop=ss.years(2050), dur=ss.years(50), dt=ss.months(1))]

    # Test start and stop
    kw.e1 = [dict(start=1990, stop=2010, dt=None)                , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))]
    kw.e2 = [dict(start=ss.years(1990), stop=2010, dt=None)      , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))] # No calendar: 2010 is a time from 0, in years
    kw.e4 = [dict(start=ss.date(1990), stop=2010, dt=None)       , dict(start=ss.date(1990), stop=ss.date(2010), dur=dd20, dt=ss.years(1))]
    kw.e5 = [dict(start=1990, stop=ss.date(2010), dt=None)       , dict(start=ss.date(1990), stop=ss.date(2010), dur=dd20, dt=ss.years(1))]
    kw.e7 = [dict(start='2025-01-01', stop=50, dt=None)          , 'exception'] # The year 50: a length is dur=50
    kw.e8 = [dict(start=0, stop='2020-01-01', dt=None)           , 'exception'] # A date with no calendar

    # Test start and a stop that is a duration
    kw.f1 = [dict(start=1990, stop=ss.years(20), dt=None)        , 'exception'] # The time 20 from 0, in a sim with a calendar: the length is dur=ss.years(20)
    kw.f2 = [dict(start=1990, stop=ss.years(2010), dt=None)      , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.years(1))] # In calendar years, the year 2010
    kw.f3 = [dict(start=ss.date(1990), stop=ss.years(20), dt=None), 'exception']
    kw.f4 = [dict(start=ss.years(0), stop=ss.days(365), dt=None) , dict(start=ss.years(0), stop=ss.years(1), dur=ss.years(1), dt=ss.years(1))]

    # Test stop with no start
    kw.g1 = [dict(start=None, stop=2030, dt=None)                , dict(start=ss.years(2000), stop=ss.years(2030), dur=ss.years(30), dt=ss.years(1))]
    kw.g2 = [dict(start=None, stop=30, dt=None)                  , dict(start=ss.years(0), stop=ss.years(30), dur=ss.years(30), dt=ss.years(1))] # A bare stop before 1800 alone is a time from 0

    # Test multiple
    kw.h1 = [dict(start=ss.years(1990), stop=ss.date(2010), dt='month') , 'exception'] # A date with no calendar
    kw.h2 = [dict(start=1990, stop=2010, dt=ss.months(1))       , dict(start=ss.years(1990), stop=ss.years(2010), dur=ss.years(20), dt=ss.months(1))]
    kw.h3 = [dict(start=1990, dur=ss.months(40), dt='month')    , dict(start=ss.years(1990), stop=ss.years(1990)+ss.months(40), dur=ss.months(40), dt=ss.months(1))]
    kw.h4 = [dict(start='2020-01-01', stop='2020-03-01', dt=None), 'exception'] # Shorter than one step


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
            mismatches.append(sc.obj(key=key, args=args, expected=expected, actual=actual))

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
        demographics = ss.Births(dt=ss.years(0.25)),
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

    # Typed values in days and years: a step of a day is said, unless the run is written in days
    sis = lambda: ss.SIS(beta=ss.perday(0.002), dur_inf=ss.years(10))
    with pytest.warns(RuntimeWarning, match='different units'):
        ss.Sim(diseases=sis(), n_agents=small, stop=100, verbose=0).init()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim = ss.Sim(diseases=sis(), n_agents=small, start='2000', dur=ss.days(100), verbose=0).init()
    assert sim.t.dt == ss.days(1)
    return sim


def test_bare_times():
    """ How a bare number is read as a time is decided once, from the start (or with none, the stop, if the unit is years): a year with a calendar, else a time in the sim's unit from 0; a duration is a time from 0, as stop= too; dur= is the length """
    sc.heading('Bare numbers as times')
    kw = dict(diseases=ss.SIS(beta=ss.peryear(0.05), dur_inf=ss.years(10)), networks='random', n_agents=small, verbose=0) # A model in years
    with pytest.raises(ValueError, match='both given'): # The stop, or the length, not both
        ss.Sim(stop=2035, dur=10, **kw).init()
    sim = ss.Sim(start=2020, dur=10, **kw).init() # A bare length is in the sim's unit
    assert sim.t.stop == ss.years(2030) and sim.t.dur == ss.years(10)
    assert ss.Sim(start=2020, dur=10, unit='day', **kw).init().t.stop == ss.years(2030) # In years in a sim of calendar years, whatever the unit
    assert ss.Sim(dur=10, unit='day', **kw).init().t.stop == ss.days(10) # Else in the sim's unit
    assert sim.new(stop=2025).init().t.stop == ss.years(2025) # A new stop replaces the length,
    assert ss.Sim(start=2020, stop=2025, **kw).new(dur=3).init().t.stop == ss.years(2023) # and a new length the stop
    sim = ss.Sim(stop=2030, **kw).init() # A calendar stop alone starts in 2000
    assert sim.t.start == ss.years(2000) and sim.t.calendar == 'years'
    sim = ss.Sim(start=2020, stop=2021, unit='day', **kw).init() # A year whatever the unit
    assert sim.t.calendar == 'years' and len(sim.t) == 366
    sim = ss.Sim(stop=3650, unit='day', **kw).init() # With no start, the explicit unit wins: 3,650 days from 0 (it ran from 2000 to the year 3650)
    assert sim.t.calendar is None and sim.t.start == ss.days(0) and sim.t.stop == ss.days(3650)
    with pytest.raises(ValueError, match='dur=50'): # The year 50 after a dated start
        ss.Sim(start='2025-01-01', stop=50, unit='day', **kw).init()
    with pytest.raises(ValueError, match='shorter than one step'):
        ss.Sim(start='2020-01-01', stop='2020-03-01', **kw).init() # The yearly step of a model in years
    for stop in [1899, 2000]: # With no calendar, a typed stop is a time from 0, however large
        sim = ss.Sim(start=0, stop=ss.years(stop), **kw).init()
        assert sim.t.start == ss.years(0) and sim.t.stop == ss.years(stop)
    with pytest.raises(ValueError, match=r'dur=ss.years\(20\)'): # With a calendar, a typed stop is a time too: the length is dur=
        ss.Sim(start=1950, stop=ss.years(20), **kw).init()
    assert ss.Sim(start=1950, stop=ss.years(1970), **kw).init().t.stop == ss.years(1970) # A typed calendar year
    assert ss.Sim(start=1950, dur=ss.years(20), **kw).init().t.stop == ss.years(1970)
    t = ss.Timeline(start=2000, stop=2020, dt=1.0)
    t.update(dur=ss.years(3)) # Also when updated
    assert t.stop == ss.years(2003) and len(t.tvec) == 4
    t.update(stop=2010) # A new stop replaces the length
    assert t.stop == ss.years(2010) and len(t.tvec) == 11

    # Times of events and modifiers, on the same axis
    dated = ss.Sim(start='2000-01-01', dur=ss.years(40), dt=ss.years(1), **kw).init()
    years = ss.Sim(start=2000, stop=2040, **kw).init()
    for sim in [dated, years]:
        assert ss.connectors.to_ti(sim, 2030) == 30 # A bare number is a year with either calendar
        assert ss.connectors.to_ti(sim, sim.pars.start + ss.years(30)) == 30 # A time after the start
        with pytest.raises(ValueError, match='sim.pars.start'): # A duration is a time from 0, which needs a sim with no calendar
            ss.connectors.to_ti(sim, ss.years(30))
        with pytest.raises(ValueError, match='calendar year'):
            ss.connectors.to_ti(sim, 30)
    assert ss.connectors.to_ti(years, years.t.tvec[30]) == 30 # A time of the timeline itself, in calendar years
    nocal = ss.Sim(start=0, stop=40, **kw).init()
    assert nocal.t.calendar is None and ss.connectors.to_ti(nocal, 30) == ss.connectors.to_ti(nocal, ss.years(30)) == 30
    with pytest.raises(ValueError, match='no calendar'):
        ss.connectors.to_ti(nocal, '2020-01-01')
    return sim


@sc.timer()
def test_daily_dates():
    sc.heading('A daily step from a year as a number has whole days, all at once')
    t = ss.Sim(start=2020, stop=2022, dt=ss.days(1), verbose=0).init().t
    assert not t.tvec.subdaily
    assert t.timevec[59] == ss.date('2020-02-29') and t.timevec[-1] == ss.date('2022-01-01') # A leap year, and the end
    assert all(d == d.normalize() for d in t.timevec), 'Expected whole days'
    return


@sc.timer()
def test_schedule():
    sc.heading('Testing schedules (ss.schedule)')

    class Clinic(ss.Module):
        def __init__(self):
            super().__init__(name='clinic')
            self.poll = ss.schedule(every=ss.months(3), start='2010-04-01')
            self.campaigns = ss.schedule(['2010-01-01', '2011-06-15', '2030-01-01'])
            self.new_year = ss.schedule(lambda now: now.month == 1 and now.day == 1)
            self.weekly = ss.schedule(every=ss.weeks(1))
            self.log = sc.objdict(poll=[], campaigns=[], new_year=[], weekly=0)
        def step(self):
            for key in ['poll', 'campaigns', 'new_year']:
                if getattr(self, key).due():
                    self.log[key].append(str(self.now))
            self.log.weekly += self.weekly.due()

    sim = ss.Sim(modules=Clinic(), n_agents=10, dt=ss.days(1), start='2010-01-01', stop='2012-01-01', verbose=0).run()
    log = sim.clinic.log
    assert log.poll == ['2010-04-01', '2010-07-01', '2010-10-01', '2011-01-01', '2011-04-01', '2011-07-01', '2011-10-01', '2012-01-01'] # A step from each day, the last included
    assert log.campaigns == ['2010-01-01', '2011-06-15']
    assert log.new_year == ['2010-01-01', '2011-01-01', '2012-01-01']
    assert log.weekly == 105

    sim = ss.Sim(modules=Clinic(), n_agents=10, dt=ss.months(1), start='2010-01-01', stop='2011-01-01', verbose=0).run() # Several due in one step are counted
    assert sim.clinic.log.weekly == 53

    # A duration is a time, counted from 0, whether or not it is in a list (how often is every=); with a calendar, it is an error
    t = ss.Timeline(start=0, stop=ss.days(30), dt=ss.days(1))
    assert list(ss.schedule(ss.days(10)).init(t).steps) == list(ss.schedule([ss.days(10)]).init(t).steps) == [10]
    assert len(ss.schedule(every=ss.days(10)).init(t).steps) == 4
    with pytest.raises(ValueError, match='sim.pars.start'):
        ss.schedule(ss.days(10)).init(ss.Timeline(start='2020-01-01', stop='2020-02-01', dt=ss.days(1)))
    return sim


@sc.timer()
def test_when_dates():
    sc.heading('A function of the date and a schedule in when=: read on the calendar, and the same in ss.Event, ss.modify, and ss.schedule')
    def mk():
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=ss.months(2)), init_prev=0.01, name='sir')

    # A function of the date with a calendar of years is given the date of each step, as for one of dates
    sir = mk()
    sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=1.2, when=lambda now: now.month in [6, 7, 8], name='m'), start=2000, stop=2002, dt=ss.months(1), verbose=0).init()
    assert sim.interventions.m.when.mask.sum() == 6
    sir = mk()
    ss.Sim(sir, interventions=ss.Event(sir.S >> sir.R, p=0.1, when=lambda now: now.month == 1), start=2000, stop=2002, dt=ss.months(1), verbose=0).run()

    # With no calendar there is no date
    sir = mk()
    with pytest.raises(ValueError, match='needs a calendar'):
        ss.Sim(sir, interventions=ss.Event(sir.S >> sir.R, p=0.1, when=lambda now: now.month == 1), unit='day', stop=5, verbose=0).run()

    # Calendar months, and a bare interval in the sim's unit, are the same steps whichever reads the schedule
    for every in [ss.months(1), 7]:
        sir = mk()
        m = ss.modify(sir.pars.beta, multiply=1.2, when=ss.schedule(every=every, start='2000-01-31'), name='m')
        ev = ss.Event(sir.S >> sir.R, p=0.1, when=ss.schedule(every=every, start='2000-01-31'), name='e')
        sim = ss.Sim(sir, interventions=[m, ev], start='2000-01-01', stop='2000-06-01', unit='day', verbose=0).init()
        sched = ss.schedule(every=every, start='2000-01-31').init(sim)
        steps = list(sched.steps)
        assert list(np.flatnonzero(sim.interventions.m.when.mask)) == steps == sorted(sim.interventions.e.round_tis)
        if every == 7:
            assert np.all(np.diff(steps) == 7)
        else:
            assert [str(sim.t.tvec[k]) for k in steps[:3]] == ['2000-01-31', '2000-02-29', '2000-03-31'] # Not drifting from the 31st
    assert repr(ss.schedule(every=ss.months(1), start=2010)) == 'ss.schedule(every=ss.months(1), start=2010)'
    sir = mk()
    assert 'when=ss.schedule(start=2001, stop=2005)' in ss.modify(sir.pars.beta, multiply=0.5, when=ss.schedule(start=2001, stop=2005), name='m').describe()

    # A bare time long before a sim with a calendar is a year, so surely a mistake
    with pytest.warns(match='long before the sim starts'):
        sir = ss.SIR(beta=0.1)
        ss.Sim(diseases=sir, interventions=ss.modify(sir.pars.beta, multiply=ss.timeseries({0: 0.5, 91: 1.0})), start='2020-01-01', stop='2020-01-10', unit='day', verbose=0).run()
    return sim


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
    o11 = test_daily_dates()
    o12 = test_when_dates()

    T.toc()
