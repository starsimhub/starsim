"""
Test time: durations, rates, and dates, their arithmetic and units, and the timeline
"""
import operator
import warnings
import datetime as dt
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
import pytest


@sc.timer()
def test_dates():
    sc.heading('Testing dates: calendar arithmetic, and arrays of dates and durations')
    # Whole days, weeks, months, and years follow the calendar, even around a leap day
    d = ss.date('2020-03-05')
    assert d + ss.days(301) == ss.days(301) + d == ss.date('2020-12-31')
    assert d - ss.days(301) == ss.date('2019-05-09')
    assert d + ss.weeks(52) == ss.date('2021-03-04')
    assert d + ss.years(1) == ss.date('2021-03-05')
    assert ss.date('2020-01-01') + 52*ss.years(1/52) == ss.date('2021-01-01')
    dates = ss.date.arange('2020-01-31', '2020-05-31', ss.dates.datedur(months=1)) # The end of a month stays the end of a month
    assert [str(d) for d in dates] == ['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31']
    assert len(ss.date.arange('2020-03-01', '2020-01-01', ss.days(1))) == 0
    assert ss.date('2020-05') == ss.date('2020-05-01') and ss.date('2020') == ss.date(2020)
    monthly = ss.Sim(stop=2, dt=1/12, unit='year', verbose=0).init()
    assert ss.connectors.to_ti(monthly, ss.months(7)) == 7 # The step that starts at 7 months

    # Arrays of dates and durations
    a = ss.DateArray(['2020-01-01', '2021-01-01'])
    b = ss.DateArray([ss.years(1), ss.years(2)])
    d0, m1 = ss.date('2020-01-01'), ss.months(1)
    cases = [ # Each operation, and what it gives
        (a + m1, ss.DateArray([ss.date('2020-02-01'), ss.date('2021-02-01')])), # Whole months follow the calendar
        (m1 + b, ss.DateArray([ss.months(13), ss.months(25)])),
        (d0 + b, ss.DateArray([ss.date('2021-01-01'), ss.date('2022-01-01')])),
        (d0 - ss.years([1, 2]), ss.DateArray([ss.date('2019-01-01'), ss.date('2018-01-01')])),
        (a - d0, ss.DateArray([ss.days(0), ss.days(366)])), # The time between dates is in days
    ]
    for i,(out,expected) in enumerate(cases):
        assert np.array_equal(out, expected), f'Case {i}: {out} ≠ {expected}'
    with pytest.raises(TypeError): # Adding two dates is not defined
        a + d0
    assert np.array_equal(a.days_since('2020-01-01'), [0, 366])
    assert d0 == '2020-01-01' and d0 < '2020-02-01'
    assert ss.DateArray(np.array([ss.days(0), ss.days(0.5)], dtype=object)).subdaily
    return


@sc.timer()
def test_timepar_float():
    sc.heading('Test that timepars cannot be silently stripped of their units')
    for val in [ss.peryear(1), ss.prob(0.5, 'year'), ss.years(1), ss.dates.datedur(weeks=1)]: # float() would give the numerator alone
        with pytest.raises(TypeError):
            float(val)
    with pytest.raises(TypeError):
        np.exp(np.log(2)/ss.years(1))
    assert np.isfinite(ss.peryear(1)) # Unit-independent ufuncs are still allowed
    assert ss.days(365).value == 365.0 and ss.days(365).years == 1.0 and ss.days(365).to_dt(ss.days(1)) == 365.0

    # Comparing with a bare number is strict, since a number has no unit, except for zero, which is the same in every unit
    assert ss.years(1) == ss.days(365) and ss.days(365) != 365 and ss.years(1) != 1
    with pytest.raises(TypeError):
        ss.days(5) < 10
    assert ss.days(5) > 0 and ss.perday(0) == 0
    return


@sc.timer()
def test_conversion():
    sc.heading('Test resolving timepars against the timestep, linking them to a module, and converting probabilities')
    assert ss.years(100).to_dt(ss.months(3)) == 400
    assert ss.dates.datedur(years=1).to_dt(ss.months(1)) == 12
    assert np.isclose(ss.freq(120, 'year').to_dt(ss.months(1)), 10)
    with pytest.raises(ValueError): # Unlinked timepars raise rather than guessing
        ss.years(100).to_dt()

    # A probability multiplied by a duration, and to() another unit, goes via the rate, exactly
    p = ss.prob(0.1, 'year')
    assert p*ss.years(0.5) == -np.expm1(np.log1p(-0.1) * ss.years(0.5)/ss.years(1))
    assert p*ss.dates.datedur(months=1) == -np.expm1(np.log1p(-0.1)/12)
    p = ss.prob(0.5, 'year')
    expected = 1 - (1 - 0.5)**(1/12) # Not the linear 0.5/12
    assert np.isclose(p.to('month').value, expected) and np.isclose(p.to_prob(ss.months(1)), expected)
    assert ss.prob(0.5, 'day').to(ss.time.probperyear).value == 1 # Scaling up stays a probability
    assert np.isclose(ss.prob(ss.peryear(0.5), 'month').value, 1 - np.exp(-0.5/12)) # Converting to or from a prob uses the underlying rate
    assert np.isclose(ss.peryear(0.5).to(ss.permonth).value, 0.5/12) # ss.per is linear

    # Module durations and rates are linked to the module dt, and timepars derived by arithmetic stay linked
    sim = ss.Sim(n_agents=100, start=2000, stop=2002, dt=ss.months(1), diseases=ssl.AgentSIS(name='sis'), networks=ss.RandomNet(), verbose=0).init()
    sis = sim.diseases.sis
    assert sis.pars.beta.default_dur == sis.t.dt
    sis.pars.update(dur_test=ss.years(100), create=True) # Parameters are locked after initialization, so need create=True
    sis.link_timepars()
    assert (2*sis.pars.dur_test).to_dt() == 200*12
    assert np.isclose((sis.pars.beta*0.5).to_prob(), sis.pars.beta.to_prob(scale=0.5))
    assert (ss.peryear(1) + sis.pars.waning).default_dur == sis.t.dt # Linked from either side
    return


@sc.timer()
def test_arithmetic():
    """ The arithmetic of typed values, and the check of the slot a typed value lands in """
    sc.heading('Testing the arithmetic of typed values')
    ops = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv, 'to_events':lambda a, b: a.to_events(b)}
    d, r, f, p = ss.days, ss.perday, ss.freqperday, ss.time.probperday
    assert sum([d(1), d(2)]) == d(3) # sum() starts from 0
    rows = [ # Left, op, right, and the result: a value, or the code of the error
        (d(3), '+', ss.weeks(1), d(10)),  (d(10), '-', ss.weeks(1), d(3)), # Units are converted
        (d(3), '*', 2, d(6)),  (d(6), '/', d(3), 2),
        (r(0.1), '+', ss.peryear(36.5), r(0.2)), # Competing hazards add
        (r(0.1), '*', 2, r(0.2)),
        (r(0.1), '*', d(3), 1 - np.exp(-0.3)), # A probability, not the expected count
        (f(0.1), '*', d(3), 0.3), (r(0.1), 'to_events', d(3), 0.3), # An expected count
        (p(0.1), '+', p(0.1), p(0.19)),  (2, '*', p(0.1), p(0.19)),  (p(0.19), '-', p(0.1), p(0.1)), # Through the rates, consistently
        (p(0.1), '*', d(2), 0.19),
        (r(0.2), '/', r(0.1), 2),
        (1, '/', d(10), r(0.1)),  (2, '/', r(0.1), d(20)),
        (r(0.1), '+', p(0.1), TypeError),  (r(0.1), '+', 1, TypeError),  (d(3), '+', 1, TypeError),  (d(3), '+', r(0.1), TypeError),
        (r(0.1), '*', r(0.1), TypeError),  (r(0.1), '/', d(3), TypeError),  (d(3), '*', d(3), TypeError),
    ]
    for a, op, b, expected in rows:
        row = f'{a!r} {op} {b!r}'
        if isinstance(expected, type):
            with pytest.raises(expected):
                ops[op](a, b)
        else:
            out = ops[op](a, b)
            if isinstance(expected, ss.TimePar):
                assert type(out) is type(expected) and np.isclose(out.value, expected.value), f'{row} gave {out!r}, not {expected!r}'
            else:
                assert np.isclose(out, expected), f'{row} gave {out}, not {expected}'
    ts = ss.timeseries({'2020-01-01': 1.0, '2020-03-01': 0.5})
    assert isinstance(r(0.1) * ts, ss.expressions.Expr) # A rate times a time series is a rate that varies in time
    out = ss.peryear(ts) # A time series in a type is one of values of that type
    assert isinstance(out, ss.timeseries) and [type(v) for v in out.values] == [ss.peryear]*2

    # The slot is checked: a rate times a duration, or a duration, is not a rate, and a rate is not a dwell time
    S, I, R = ss.states('S I R')
    for slip in [lambda: I >> R @ (ss.peryear(0.1) * ss.days(3)), lambda: ss.Transition(I, R, rate=ss.days(10)), lambda: ss.Transition(I, R, dur=ss.prob(0.1, 'day'))]:
        with pytest.raises(TypeError):
            slip()
    sir = ss.Disease(S >> I @ 2.5 * I / ss.days(10), I >> R @ (1/ss.days(10)), name='sir') # A duration that divides is fine
    return sir


@sc.timer()
def test_bare_units():
    sc.heading('Testing the units of bare numbers beside typed ones')
    S, I, R = ss.states('S I R')

    # A bare dt beside calendar years is in years, with a warning, though the model's unit is months
    sis = ss.Disease(inf=S >> I @ ss.permonth(0.0)*I, rec=I >> S @ ss.months(6), init_n=dict(I=1000), name='d')
    with pytest.warns(RuntimeWarning):
        sim = ss.Sim(sis, start=2000, stop=2020, dt=1/12, n_agents=1000, verbose=0).init()
    assert sim.t.npts == 241

    # A bare beta is in the unit of the disease's own durations, beside births and deaths per year
    sir = ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), name='sir')
    sim = ss.Sim(sir, demographics=[ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.02))], start='2000', stop='2001', n_agents=500, verbose=0).init()
    assert sim.unit == 'day'

    # A rate of zero needs no unit, with no warning that its unit was guessed
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        zero = ss.Disease(S, I, R, infection=S >> I @ 0*I, recovery=I >> R @ ss.days(8), wane=R >> S @ 0, init_prev=0.1, name='zero')
        assert ss.Sim(zero, n_agents=200, stop=ss.days(10), verbose=0).run().results.zero.infection.sum() == 0

    # A bare number is a year in a sim with a calendar, dates or numeric years alike; a duration is that long after the start
    for start in [2020, '2020-01-01']:
        sim = ss.Sim(sir, start=start, stop=2021, n_agents=100, verbose=0).init()
        assert np.allclose(ss.timeseries({2000: 1, 2030.5: 2, '2040-01-01': 3}).years(sim), [2000, 2030.5, 2040])
        assert ss.timeseries({2000: 1, 2030: 2}).years(sim) is ss.timeseries({2000: 3, 2030: 4}).years(sim) # Bare times are worked out once for each sim, as a model can have thousands of time series
        with pytest.raises(ValueError, match='calendar year'): # The year 30, not 30 days after the start
            ss.connectors.to_ti(sim, 30)
        assert ss.connectors.to_ti(sim, ss.days(30)) == ss.connectors.to_ti(sim, '2020-01-31')

    # With no calendar, a duration as a time is a point counted from 0, as tvec is; another unit is converted
    sim = ss.Sim(sir, start=0, stop=ss.days(60), n_agents=100, verbose=0).init()
    assert ss.connectors.to_ti(sim, ss.days(30)) == ss.connectors.to_ti(sim, 30) == 30 and ss.connectors.to_ti(sim, ss.weeks(2)) == 14
    assert ss.Sim(sir, unit='day', stop=1095, n_agents=100, verbose=0).init().t.stop == ss.days(1095) # With no start, a bare stop in days is never a year

    # A time series of bare numbers as a crude birth rate is per year, as a bare number is
    alive = [ss.Sim(demographics=ss.birth(rate=rate), total_pop=1000, start='2000-01-01', stop='2001-01-01', verbose=0).run().results.alive[-1] for rate in [0.04, ss.timeseries({2000: 0.04, 2030: 0.02})]]
    assert np.isclose(alive[0], alive[1]) and alive[0] < 1100
    return


@sc.timer()
def test_timeline():
    sc.heading('Test the lengths of timelines, and the ss.Timeline input options: any two of the start, stop, and dur define the third')
    cases = [ # Start, stop, dt, and the number of points
        (ss.date('2020-01-01'), ss.date('2020-06-01'), ss.days(1), 153),
        (ss.days(0), ss.years(1), ss.weeks(1), 53),
        (ss.days(0), ss.years(1), ss.months(1), 13),
        (2000, 2005, 'day', len(sc.daterange('2000-01-01', '2005-01-01')) - 2), # Numeric years are 365 days, so leap days are not counted
        ('2000-01-01', '2005-01-01', 'day', len(sc.daterange('2000-01-01', '2005-01-01'))), # They are with dates
        (ss.date('2000-01-01'), ss.date('2000-01-10'), ss.days(0.25), 37), # Sub-daily steps are exact
    ]
    for start, stop, dt, n in cases:
        assert len(ss.Timeline(start, stop, dt)) == n, f'{start} to {stop} by {dt}: {len(ss.Timeline(start, stop, dt))} points, not {n}'

    dd30 = ss.date(2030) - ss.date(2000) # The time between two dates, in days
    y = ss.years
    cases = [ # The inputs, and the start, stop, dur, and dt they give (or None for an error)
        # A date or a bare year (from 1800 on) gives the timeline a calendar; any other number, or a duration, is a time from 0
        (dict(), (y(2000), y(2050), y(50), y(1))),
        (dict(start='1990.1.1'), (ss.date(1990), ss.date(2040), y(50), y(1))),
        (dict(start=ss.days(5)), (ss.days(5), ss.days(105), ss.days(100), ss.days(1))),
        (dict(stop=2010), (y(2000), y(2010), y(10), y(1))), # A calendar stop starts at the default start, 2000
        (dict(stop=y(20)), (y(0), y(20), y(20), y(1))), # A typed stop is a time, as a start is: 20 years from 0
        (dict(stop=ss.date(2030)), (ss.date(2000), ss.date(2030), dd30, y(1))),
        (dict(stop=30), (y(0), y(30), y(30), y(1))), # A bare stop before 1800 alone is a time from 0
        (dict(start=1990, dur=y(1990)), (y(1990), y(3980), y(1990), y(1))), # Always the length, never a year
        (dict(stop=2010, dur=10), (y(2000), y(2010), y(10), y(1))), # The start is the stop minus the length
        (dict(dt=ss.days(1)), (ss.days(0), ss.days(100), ss.days(100), ss.days(1))), # Alone, the unit is that of the first duration given
        (dict(start=2000, dt=ss.months(1)), (y(2000), y(2050), y(50), ss.months(1))),
        (dict(start=y(1990), stop=2010), (y(1990), y(2010), y(20), y(1))), # No calendar: 2010 is a time from 0, in years
        (dict(start=1990, stop=y(2010)), (y(1990), y(2010), y(20), y(1))), # In calendar years, the year 2010
        (dict(start=2000, stop=2020, dur=10), None), # All three, which disagree
        (dict(start=1990, stop=y(20)), None), # The time 20 from 0, in a sim with a calendar: the length is dur=ss.years(20)
        (dict(start='2020-01-01', stop='2020-03-01'), None), # Shorter than one step
    ]
    for args, expected in cases:
        if expected is None:
            with pytest.raises(ValueError):
                ss.Timeline(**args).init()
        else:
            t = ss.Timeline(**args).init()
            assert (t.start, t.stop, t.dur, t.dt) == expected, f'{args}: {(t.start, t.stop, t.dur, t.dt)}, not {expected}'
    return


@sc.timer()
def test_sim_times():
    sc.heading('Test the times of a sim: its unit, the timesteps of its modules, and durations as times')
    kw = dict(diseases=ss.SIS(beta=ss.peryear(0.05), dur_inf=ss.years(10)), networks='random', n_agents=100, verbose=0) # A model in years
    assert ss.Sim(start=2020, dur=10, unit='day', **kw).init().t.stop == ss.years(2030) # In years in a sim of calendar years, whatever the unit
    assert ss.Sim(dur=10, unit='day', **kw).init().t.stop == ss.days(10) # Else in the sim's unit
    assert ss.Sim(start=2020, stop=2025, **kw).new(dur=3).init().t.stop == ss.years(2023) # A new length replaces the stop
    t = ss.Sim(start=dt.date(2020, 1, 1), stop=np.datetime64('2021-01-01'), **kw).init().t # Python's and NumPy's dates are dates too (an R Date is a datetime.date)
    assert t.calendar == 'dates' and t.stop == ss.date('2021-01-01')

    # A sim with only modules written as code honors its unit
    class Step(ss.Module):
        def step(self):
            pass
    assert ss.Sim(connectors=Step(), unit='week', stop=10, verbose=0).init().t.dt == ss.weeks(1)

    # Modules with their own timesteps, each stepping once per its step
    sim = ss.Sim(diseases=ss.SIS(dt=ss.days(1), init_prev=0.1, beta=ss.peryear(0.01)), demographics=ss.Births(dt=ss.years(0.25)), networks=ss.RandomNet(dt=ss.weeks(1)),
                 n_agents=100, dt=ss.days(2), start='2000-01-01', stop='2002-01-01', verbose=0).run()
    assert len(sim) == 366 and len(sim.diseases.sis) == 366*2 and len(sim.demographics.births) == 9
    assert sim.t.tvec[-1] == ss.date('2001-12-31') and sim.diseases.sis.t.tvec[-1] == ss.date('2002-01-01')
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.years(2), dt='month'), networks='random', n_agents=100, dur=ss.years(1), dt=ss.years(1/12)).init()
    plan = sim.loop.to_df()
    assert plan[plan.label.str.contains('sis.step_state')].groupby('ti').size().max() == 1 # Months and twelfths of a year are the same steps

    # With a calendar, a duration is that long after the start: until=, a module's start=, and the keys of a periodic ss.timeseries
    assert ss.Sim(start=2000, stop=2030, **kw).run(until=ss.years(5)).t.ti == 5
    sim = ss.Sim(ss.Module(name='late', start=ss.days(200)), start='2020-01-15', stop='2021-01-01', dt=ss.days(1), **kw).init()
    assert sim.late.t.start == ss.date('2020-08-02')
    S, I, R = ss.states('S I R')
    ts = ss.timeseries({ss.years(k/12): ss.peryear(1 + k % 12) for k in range(13)}, interpolation='linear', period=ss.years(1))
    sim = ss.Sim(ss.Disease(S >> I @ ts*I, I >> R @ ss.days(10), init_prev=0.01), start='2020-03-01', stop='2022-01-01', dt=ss.days(7), n_agents=100, verbose=0).run()
    y0 = sim.t.yearvec[0]
    assert [ts.at(y, sim).value for y in [y0, y0 + 0.5, y0 + 1.5]] == [1, 7, 7] # From the sim's start, each year
    return


@sc.timer()
def test_schedule():
    sc.heading('Testing schedules (ss.schedule), and in when=: read on the calendar, and the same in ss.Event, ss.modify, and ss.schedule')
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

    sim = ss.Sim(Clinic(), n_agents=10, dt=ss.days(1), start='2010-01-01', stop='2012-01-01', verbose=0).run()
    log = sim.clinic.log
    assert log.poll == ['2010-04-01', '2010-07-01', '2010-10-01', '2011-01-01', '2011-04-01', '2011-07-01', '2011-10-01', '2012-01-01'] # A step from each day, the last included
    assert log.campaigns == ['2010-01-01', '2011-06-15']
    assert log.new_year == ['2010-01-01', '2011-01-01', '2012-01-01']
    assert log.weekly == 105
    sim = ss.Sim(Clinic(), n_agents=10, dt=ss.months(1), start='2010-01-01', stop='2011-01-01', verbose=0).run() # Several due in one step are counted
    assert sim.clinic.log.weekly == 53
    t = ss.Timeline(start=0, stop=ss.days(30), dt=ss.days(1)) # A duration is a time, that long after the start
    assert list(ss.schedule(at=ss.days(10)).init(t).steps) == [10]

    # A date given to a delivery in a sim in calendar years is the step from it
    sim = ss.Sim(ss.SIS(), interventions=ss.vaccinate(efficacy=0.8, when=['2001'], coverage=0.5, capacity=50), n_agents=500, start=2000, stop=2002, dt=ss.months(1), verbose=0).run()
    assert np.flatnonzero(sim.interventions[0].results.delivered.values).tolist() == [12]

    # Calendar months, and a bare interval in the sim's unit, are the same steps whichever reads the schedule
    for every in [ss.months(1), 7]:
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=ss.months(2)), init_prev=0.01, name='sir')
        m = ss.modify(sir.pars.beta, multiply=1.2, when=ss.schedule(every=every, start='2000-01-31'), name='m')
        ev = ss.Event(sir.S >> sir.R, p=0.1, when=ss.schedule(every=every, start='2000-01-31'), name='e')
        sim = ss.Sim(sir, interventions=[m, ev], start='2000-01-01', stop='2000-06-01', unit='day', verbose=0).init()
        steps = list(ss.schedule(every=every, start='2000-01-31').init(sim).steps)
        assert list(np.flatnonzero(sim.interventions.m._when_date.mask)) == steps == sorted(sim.interventions.e.round_tis)
        if every == 7:
            assert np.all(np.diff(steps) == 7)
        else:
            assert [str(sim.t.tvec[k]) for k in steps[:3]] == ['2000-01-31', '2000-02-29', '2000-03-31'] # Not drifting from the 31st
    return


@sc.timer()
def test_time_regressions():
    sc.heading('Testing fixed bugs in time, dates, and timelines')
    # A sim with no calendar has its times as numbers, not dates in the year 10 AD, and they can be added to
    sim = ss.Sim(diseases='sis', networks='random', n_agents=200, unit='year', start=10, stop=20, verbose=0).run()
    assert sim.t.calendar is None and np.array_equal(sim.timevec[:2], [10, 11])
    assert np.array_equal(sim.timevec + 1, np.arange(11, 22))

    # A failed addition leaves a date a date; an object array of dates minus a date is in days
    d = ss.date(2020)
    with pytest.raises(Exception):
        d + ss.dates.datedur(years=10_000)
    assert type(d) is ss.date
    diffs = np.array([ss.date(2020), ss.date(2021)], dtype=object) - ss.date(2019)
    assert [x.value for x in diffs] == [365, 731]
    assert abs(ss.dates.datedur(months=-1)) == ss.months(1)

    # Outside the run, now() gives each key's own kind of time
    t = ss.Timeline(start='2020-01-01', stop='2020-12-31', dt=ss.weeks(1))
    t.ti = -1
    assert t.now('year') == ss.date('2019-12-25').years and t.now() == ss.date('2019-12-25')

    # A module whose timeline is shifted from the sim's, with as many points, is not read as on the sim's times
    sim = ss.Sim(n_agents=100, start=2000, stop=2010, demographics=ss.Births(start=2002, stop=2012), verbose=0).init()
    births = sim.demographics.births
    assert births.match_time_inds() is not Ellipsis and births.match_time_inds()[0] == 2

    # Rates: a probability times an array of numbers; equality with something that is not a rate; a rate times ss.by(patch=fn)
    assert np.allclose((ss.probperyear(0.1)*np.array([1, 2])).value, [0.1, 0.19])
    assert ss.peryear(1) != ss.days(1) and ss.peryear(1) == ss.permonth(1/12)
    assert isinstance(ss.perday(1)*ss.by(patch=lambda sim: [1, 2]), ss.expressions.Expr)
    return


if __name__ == '__main__':
    T = sc.timer('\nTotal time')
    test_dates()
    test_timepar_float()
    test_conversion()
    test_arithmetic()
    test_bare_units()
    test_timeline()
    test_sim_times()
    test_schedule()
    test_time_regressions()
    T.toc()
