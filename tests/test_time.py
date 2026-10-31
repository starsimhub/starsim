"""
Test different time units and timesteps
"""
import operator
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.library as ssl
import pytest
import matplotlib.pyplot as plt

# ss.options.warnings = 'error' # For additional debugging


@sc.timer()
def test_ratio():
    sc.heading('Test dur/datedur time ratio calculation')

    assert ss.years(1) / ss.years(1) == 1
    assert ss.days(1) / ss.days(0.1) == 10
    assert ss.weeks(0.5) / ss.weeks(5) == 0.1

    assert ss.datedur(years=1) / ss.datedur(days=1) == 365
    assert np.isclose(ss.datedur(years=1) / ss.datedur(weeks=1) * 7, 365)
    assert ss.datedur(years=1) / ss.datedur(months=1) == 12

    return


@sc.timer()
def test_classes():
    sc.heading('Test behavior of dur() and rate()')

    # Test duration dt
    d1 = ss.years(2)
    d2 = ss.years(3)
    d3 = ss.years(2/0.1)
    d4 = ss.years(3/0.2)

    assert d1 + d2 == 2+3
    assert d3 + d4 == 2/0.1 + 3/0.2
    assert d3 * 2 == 2/0.1*2
    assert d3 / 2 == 2/0.1/2

    # Test rate dt
    r1 = ss.time.freq(2, 'year')
    r2 = ss.time.freq(3, 'year')
    r3 = ss.time.freq(2/0.1, 'year')
    r4 = ss.time.freq(3/0.2, 'year')

    assert r1 + r2 == ss.time.freq(5, 'year')
    assert r3 + r4 == ss.time.freq(20+15, 'year')
    assert r3 * 2 == ss.time.freq(4/0.1, 'year')
    assert r4 / 2 == ss.time.freq(1.5/0.2, 'year')

    # Test duration units
    d5 = ss.datedur(years=2)
    d6 = ss.datedur(days=5)
    assert (d5 + d6).years == 2 + 5/365
    assert (d5 + d6)/ss.datedur(days=1) == 365*2+5

    # Test rate units
    rval = 0.7
    r5 = ss.time.freq(rval, ss.weeks(1))
    assert np.isclose(r5*ss.days(1), rval/7) # These should be close, but not match exactly
    assert np.isclose(r5*ss.weeks(0.1), rval/10)
    assert r5*ss.days(1) == rval * ss.days(1) / ss.weeks(1) # These should match exactly
    assert r5*ss.weeks(0.1) == rval * ss.weeks(0.1) / ss.weeks(1)

    # Test prob
    tpval = 0.1
    tp0 = ss.prob(tpval, 'year')
    assert np.isclose(tp0*ss.years(1), tpval), 'Multiplication by the base denominator should not change the value'
    assert np.isclose(tp0*ss.years(0.5), tpval/2, rtol=0.1) # These should be close, but not match exactly
    assert np.isclose(tp0*ss.years(2), tpval*2, rtol=0.1)
    assert tp0*ss.years(0.5) == -np.expm1(np.log1p(-0.1) * ss.years(0.5)/ss.years(1)) # These should be close, but not match exactly
    assert tp0*ss.years(2) == -np.expm1(np.log1p(-0.1) * ss.years(2)/ss.years(1)) # These should be close, but not match exactly

    # The unit factors survive pickling, as parallel workers unpickle them
    import pickle
    assert pickle.loads(pickle.dumps(ss.time.factors)).years.days == 365

    return d3, d4, r3, r4, tp0



@sc.timer()
def test_callable_dists():
    sc.heading('Testing callable distributions')
    def loc(module, sim, uids):
        return np.arange(uids)
    module = ss.debugtools.mock_module(dt=ss.days(1))
    d = ss.normal(loc, 2, unit='days', module=module, strict=False)
    d.init()
    d.rvs(10)
    return d


@sc.timer()
def test_syntax():
    """ Verify that a range of supported operations run without raising an error """
    sc.heading('Testing full syntax')

    assert float(ss.date(1500))==1500
    assert np.isclose(float(ss.date(1500.1)), 1500.1) # Not exactly equal, but very close
    assert np.all((ss.years(1)*np.arange(5)) == (np.arange(5)*ss.years(1)))
    assert ss.datedur(weeks=1)/ss.datedur(days=1) == 7
    assert np.isclose((ss.datedur(weeks=1) - ss.datedur(days=1)).years, 6/365)
    assert np.isclose((ss.date(2050)-ss.date(2020)).years, ss.years(30).years, rtol=1/365) # Not exact due to leap years
    assert np.isclose((ss.time.freq(1, 'week')+ss.time.freq(1, 'day')).value, ss.time.freq(8, 'week').value) # CKTODO: would be nice if this were exact
    assert np.isclose((ss.peryear(1)+ss.permonth(1)).value, 13) # Rates add directly, not via probabilities

    assert ss.date('2020-01-01') + ss.datedur(weeks=52) == ss.date('2020-12-30') # Should give us 30th December 2020
    assert ss.date('2020-01-01') + 52*ss.datedur(weeks=1) == ss.date('2020-12-30')# Should give us 30th December 2020
    assert ss.date('2020-01-01') + 52*ss.years(1/52) == ss.date('2021-01-01') # Should give us 1st Jan 2021
    assert ss.date('2020-01-01') + ss.datedur(years=1) == ss.date('2021-01-01') # Should give us 1st Jan 2021
    assert ss.date('2000-01-01') + ss.days(200) == ss.days(200) + ss.date('2000-01-01') == ss.date('2000-07-19') # Days are added as days (was a day late after the leap day)
    monthly = ss.Sim(stop=2, dt=1/12, unit='year', verbose=0).init()
    assert ss.connectors.to_ti(monthly, ss.months(7)) == ss.connectors.to_ti(monthly, 7/12) == 8 # The step that starts at 7 months (was the next, from rounding)

    # Operations on date vectors
    ss.date.arange(2020,2030)+ss.years(1) # add years to date array
    ss.date.arange(2020,2030)+ss.datedur(years=1) # add datedur to date array

    # Construction of various duration ranges and addition with durations and dates
    ss.dur.arange(ss.years(0), ss.years(10),ss.years(1)) + ss.years(1)
    ss.dur.arange(ss.years(0), ss.years(10), ss.datedur(years=1)) + ss.years(1)
    ss.dur.arange(ss.years(0), ss.datedur(years=10), ss.datedur(years=1)) + ss.years(1)

    # Datedur calculations -- these are not all currently working
    # ss.dur.arange(ss.datedur(years=0), ss.datedur(years=10), ss.datedur(years=1)) + ss.years(1)
    # ss.dur.arange(ss.datedur(years=0), ss.datedur(years=10), ss.datedur(years=1)) + ss.datedur(years=1)
    # ss.dur.arange(ss.datedur(years=0), ss.datedur(years=10), ss.datedur(years=1)) + ss.date(2000)
    # ss.dur.arange(ss.years(0), ss.years(10), ss.years(1)) + ss.datedur(years=1)
    # ss.dur.arange(ss.years(0), ss.years(10), ss.datedur(years=1)) + ss.datedur(years=1)
    # ss.dur.arange(ss.years(0), ss.datedur(years=10), ss.datedur(years=1)) + ss.datedur(years=1)
    ss.dur.arange(ss.years(0), ss.years(10), ss.years(1)) + ss.date(2000)
    ss.dur.arange(ss.years(0), ss.years(10), ss.datedur(years=1)) + ss.date(2000)
    ss.dur.arange(ss.years(0), ss.datedur(years=10), ss.datedur(years=1)) + ss.date(2000)

    # Rates
    assert (1/ss.years(1)) == ss.peryear(1)
    assert (2/ss.years(1)) == ss.peryear(2)
    assert (4/ss.years(1)) == ss.peryear(4)
    assert (4/ss.datedur(1)) == ss.peryear(4)
    assert (ss.time.freq(5, 'day')*ss.datedur(days=1)) == 5
    assert 2/ss.time.freq(0.25, 'year') == ss.years(8)
    assert 1/(2*ss.time.freq(0.25, 'year')) == ss.years(2)
    assert ss.time.freq(0.5, 'year')/ss.time.freq(1, 'year') == 0.5

    # Probabilities
    p = ss.prob(0.1, ss.datedur(years=1))
    f = lambda factor: -np.expm1(np.log1p(-p.value)/factor)
    assert p*ss.datedur(years=2) == f(0.5)
    assert p * ss.years(0.5) == f(2)
    assert p * ss.datedur(months=1) == f(12)

    p = ss.prob(0.1, ss.years(1))
    assert p*ss.datedur(years=2) == f(0.5)
    assert p * ss.years(0.5 ) == f(2)
    assert p * ss.datedur(months=1) == f(12)

    p = ss.per(0.1, ss.datedur(years=1))
    f = lambda factor: -np.expm1(-p.value/factor)
    assert p*ss.datedur(years=2) == f(0.5)
    assert p * ss.years(0.5) == f(2)
    assert p * ss.datedur(months=1) == f(12)

    p = ss.per(0.1, ss.years(1))
    assert p*ss.datedur(years=2) == f(0.5)
    assert p * ss.years(0.5) == f(2)
    assert p * ss.datedur(months=1) == f(12)

    return p


@sc.timer()
def test_datearray_operations():
    sc.heading('Test DateArray add/sub operations')

    # A range of fixed steps is made at once, and is the same as stepping; one that ends before it starts is empty
    days = ss.date.arange('2020-01-01', '2020-03-01', ss.days(7))
    assert days[-1] == ss.date('2020-02-26') and np.all(np.diff([d.to_year() for d in days]) > 0)
    assert all(days[i] + ss.datedur(ss.days(7)) == days[i+1] for i in range(len(days) - 1))
    assert len(ss.date.arange('2020-03-01', '2020-01-01', ss.days(1))) == 0
    assert len(ss.date.arange('2020-03-01', '2020-01-01', ss.months(1))) == 0

    # A rate cannot be subtracted from a number
    with pytest.raises(TypeError):
        1 - ss.peryear(0.1)

    a = ss.DateArray([ss.date('2020-01-01'), ss.date('2021-01-01')])
    b = ss.DateArray([ss.years(1), ss.years(2)])
    c = ss.years([1, 2])

    d0, m1 = ss.date('2020-01-01'), ss.months(1)
    plus_month = ss.DateArray([ss.date('2020-01-31'), ss.date('2021-01-31')])
    months_plus = ss.DateArray([ss.months(13), ss.months(25)])
    dates_plus = ss.DateArray([ss.date('2021-01-01'), ss.date('2022-01-01')])
    dates_minus = ss.DateArray([ss.date('2019-01-01'), ss.date('2018-01-01')])
    cases = [ # Each operation, and what it gives
        (lambda: a + m1, plus_month),
        (lambda: m1 + a, plus_month),
        (lambda: m1 + b, months_plus),
        (lambda: b + m1, months_plus),
        (lambda: m1 + c, ss.months([13, 25])),
        (lambda: c + m1, ss.years([1 + 1/12, 2 + 1/12])),
        (lambda: b + d0, dates_plus),
        (lambda: d0 + b, dates_plus),
        (lambda: c + d0, dates_plus),
        (lambda: d0 + c, dates_plus),
        (lambda: a - m1, ss.DateArray([ss.date('2019-12-02'), ss.date('2020-12-02')])),
        (lambda: m1 - b, ss.DateArray([ss.months(-11), ss.months(-23)])),
        (lambda: b - m1, ss.DateArray([ss.months(11), ss.months(23)])),
        (lambda: m1 - c, ss.months([-11, -23])),
        (lambda: c - m1, ss.years([1 - 1/12, 2 - 1/12])),
        (lambda: d0 - a, ss.DateArray([ss.days(0), ss.days(-366)])), # The time between dates is in days
        (lambda: a - d0, ss.DateArray([ss.days(0), ss.days(366)])),
        (lambda: d0 - b, dates_minus),
        (lambda: d0 - c, dates_minus),
    ]
    for i,(func,expected) in enumerate(cases):
        assert np.array_equal(func(), expected), f'Case {i}: {func()} ≠ {expected}'
    for func in [lambda: a + d0, lambda: d0 + a, lambda: m1 - a, lambda: b - d0, lambda: c - d0]: # Not defined
        with pytest.raises(TypeError):
            func()

    # Exact elapsed days, rather than calendar differences
    assert np.array_equal(a.days_since('2020-01-01'), [0, 366])
    assert np.array_equal(ss.DateArray(['2020-01-01', '2021-01-01']), a) # Strings are read as dates
    with pytest.raises(TypeError):
        b.days_since('2020-01-01')

    # A date compares with a string as with the date it names, and a string that is not a date raises, for == as for < (HP79)
    assert d0 == '2020-01-01' and d0 != '2020-01-02' and d0 < '2020-02-01'
    for func in [lambda: d0 == 'sim', lambda: d0 != '2020-13-01', lambda: d0 < 'sim']:
        with pytest.raises(ValueError):
            func()

    return


@sc.timer()
def test_timepar_float():
    sc.heading('Test that timepars cannot be silently stripped of their units')

    # float() would return the numerator alone, so peryear(1), permonth(1) and perday(1)
    # would all be 1.0 despite differing by a factor of 365
    for rate in [ss.peryear(1), ss.permonth(1), ss.perday(1), ss.prob(0.5, 'year'), ss.time.freq(80, 'year')]:
        with pytest.raises(TypeError):
            float(rate)
        assert sc.isnumber(rate.value) # The escape hatch still works

    # Durations are ambiguous rather than incomplete: years(1), months(1) and days(1) would all be 1.0
    for d in [ss.years(1), ss.months(1), ss.days(365), ss.datedur(weeks=1)]:
        with pytest.raises(TypeError):
            float(d)
    assert ss.days(365).value == 365.0 # The raw value, in the duration's own units
    assert ss.days(365).years == 1.0 # The value in years
    assert ss.days(365).to_dt(ss.days(1)) == 365.0 # The value in timesteps

    # Single-input ufuncs took the same path, e.g. np.exp(np.log(2)/ss.years(1)) == np.exp(np.log(2)/ss.years(12))
    f1 = np.log(2)/ss.years(1)
    f12 = np.log(2)/ss.years(12)
    for f in [f1, f12]:
        for ufunc in [np.exp, np.log, np.sqrt]:
            with pytest.raises(TypeError):
                ufunc(f)
    assert not np.isclose(f1.to_dt(ss.months(1)), f12.to_dt(ss.months(1))) # The correct values do differ

    # Unit-independent ufuncs are still allowed
    assert np.isfinite(ss.peryear(1))
    assert np.isnan(ss.time.freq(np.nan, 'year'))
    assert np.sign(ss.peryear(2)) == 1

    # Dates are unaffected, since a date has an unambiguous float representation
    assert float(ss.date(1500)) == 1500

    # Durations must still be plottable: matplotlib converts them via ss.DateConverter rather than float()
    tvec = ss.Timeline(start=ss.days(0), stop=ss.days(2), dt=ss.days(1)).init().tvec
    for x in [tvec, list(tvec), pd.Series(tvec)]: # A DateArray, a plain list, and a pandas column
        fig, ax = plt.subplots()
        line, = ax.plot(x, [1, 2, 3])
        assert np.allclose(line.get_xdata(orig=False), [0, 1, 2]), 'Durations should plot as their own units, not years'
        plt.close(fig)

    return


@sc.timer()
def test_years_conversion():
    """ float() on a dur gives the magnitude, so code that needs years must ask for years """
    sc.heading('Test the years conversion and the sites that depend on it')

    # Dates, durations and datedurs all define .years
    assert ss.days(365).years == 1.0
    assert ss.months(6).years == 0.5
    assert ss.datedur(days=365).years == 1.0
    assert ss.date(2000.5).years == 2000.5

    # DateArray.subdaily must compare the spacing in years, not in the array's own units
    subdaily = ss.DateArray(np.array([ss.days(0), ss.days(0.5)], dtype=object))
    daily = ss.DateArray(np.array([ss.days(0), ss.days(1)], dtype=object))
    assert subdaily.subdaily, 'Half-day spacing should count as sub-daily'
    assert not daily.subdaily, 'One-day spacing should not count as sub-daily'
    floatvec = ss.DateArray(np.array([0.0, 0.5/365])) # A plain float array is already in years
    assert floatvec.subdaily, 'Half-day spacing should count as sub-daily for a float vector too'

    # Timeline.now('yearvec') outside the sim period must also be in years
    tl = ss.Timeline(start=ss.days(0), stop=ss.days(100), dt=ss.days(1))
    tl.init()
    tl.ti = 501 # Deliberately past the end, to hit the extrapolation branch: the step that starts at time point 500
    assert np.isclose(tl.now('yearvec'), 500/365), 'now("yearvec") should be in years, not days'

    return


@sc.timer()
def test_to_dt():
    sc.heading('Test resolving timepars against the timestep')

    # Durations resolve to a number of timesteps
    assert ss.years(100).to_dt(ss.months(1)) == 1200
    assert ss.years(100).to_dt(ss.months(3)) == 400
    assert ss.years(100).to_dt(ss.years(1)) == 100
    assert ss.months(100).to_dt(ss.months(1)) == 100
    assert ss.datedur(years=1).to_dt(ss.months(1)) == 12

    # Rates resolve to their per-timestep value
    assert np.isclose(ss.peryear(0.1).to_dt(ss.months(1)), ss.peryear(0.1).to_prob(ss.months(1)))
    assert np.isclose(ss.time.freq(120, 'year').to_dt(ss.months(1)), 10)

    # Unlinked timepars raise rather than guessing
    with pytest.raises(ValueError):
        ss.years(100).to_dt()

    # Inside a module, the timestep is filled in automatically
    module = ss.debugtools.mock_module(dt=ss.months(1))
    dur = ss.years(100).set_default_dur(module.dt)
    assert dur.to_dt() == 1200

    # Rates can convert to a duration unit as well as a rate class
    assert ss.peryear(1).to('month') == ss.peryear(1).to(ss.permonth)
    assert isinstance(ss.prob(0.5, 'year').to('month'), ss.time.probpermonth)
    assert isinstance(ss.time.freq(80, 'year').to(ss.months), ss.time.freqpermonth)

    return


@sc.timer()
def test_link_timepars():
    sc.heading('Test that module durations and rates are linked to the module dt')

    sim = ss.Sim(n_agents=100, start=2000, stop=2002, dt=ss.months(1), diseases=ssl.AgentSIS(name='sis'), # Written as code, whose parameters are linked to its timestep
                 networks=ss.RandomNet(), verbose=0)
    sim.init()
    sis = sim.diseases.sis
    assert sis.pars.beta.default_dur == sis.t.dt # Rates were already linked
    assert sis.pars.waning.default_dur == sis.t.dt

    # Durations are now linked too
    dur = ss.years(100)
    sis.pars.update(dur_test=dur, create=True) # Parameters are locked after initialization, so need create=True
    sis.link_timepars()
    assert sis.pars.dur_test.to_dt() == 100*12

    # Timepars derived by arithmetic stay linked
    shed = ss.time.freq(12, 'year').set_default_dur(sis.t.dt)
    assert (shed*10).to_events() == 10
    assert np.isclose((sis.pars.beta*0.5).to_prob(), sis.pars.beta.to_prob(scale=0.5))
    assert (ss.peryear(1) + sis.pars.waning).default_dur == sis.t.dt # Linked from either side
    assert (2*sis.pars.dur_test).to_dt() == 200*12
    with pytest.raises(ValueError): # An unlinked rate gives an informative error
        ss.time.freq(12, 'year').to_events()

    return


@sc.timer()
def test_prob_conversion():
    """ Probabilities are not linear in time, so converting one between units must go via the rate """
    sc.heading('Test converting probabilities between units')

    # to() used to rescale the value linearly, giving 0.5/12 rather than 1-(1-0.5)**(1/12)
    p = ss.prob(0.5, 'year')
    expected = 1 - (1 - 0.5)**(1/12)
    for target in [ss.time.probpermonth, ss.months, 'month']:
        assert np.isclose(p.to(target).value, expected)
    assert np.isclose(p.to_prob(ss.months(1)), expected) # to() and to_prob() now agree
    assert np.isclose(ss.prob(ss.prob(0.5, 'year'), 'month').value, expected) # As does direct construction

    # Scaling up used to raise, since 0.5*365 fails the prob range check
    assert ss.prob(0.5, 'day').to(ss.time.probperyear).value == 1
    assert ss.prob(ss.prob(0.5, 'day'), 'year').value == 1

    # Edge cases and round trips
    assert p.to(ss.time.probperyear).value == 0.5 # A no-op conversion is exact
    assert np.isclose(p.to(ss.time.probpermonth).to(ss.time.probperyear).value, 0.5)
    assert ss.prob(0, 'year').to(ss.time.probpermonth).value == 0
    assert ss.prob(1, 'year').to(ss.time.probpermonth).value == 1 # A certainty stays a certainty
    arr = ss.prob(np.array([0.1, 0.5]), 'year').to(ss.time.probpermonth).value
    assert np.allclose(arr, 1 - (1 - np.array([0.1, 0.5]))**(1/12))

    # Converting to or from a prob uses the underlying rate, not the raw value
    assert np.isclose(ss.prob(ss.peryear(0.5), 'month').value, 1 - np.exp(-0.5/12))
    assert np.isclose(ss.peryear(ss.prob(0.05, 'month')).value, -np.log(1-0.05)*12)

    # ss.per and ss.freq are genuinely linear, so they are unchanged
    assert np.isclose(ss.peryear(0.5).to(ss.permonth).value, 0.5/12)
    assert np.isclose(ss.time.freq(12, 'year').to(ss.time.freqpermonth).value, 1.0)
    assert np.isclose(ss.permonth(1).to(ss.peryear).value, 12)

    # A unitless prob has no time unit to convert against, so it raises rather than guessing
    with pytest.raises(TypeError):
        ss.prob(ss.prob(0.5, 'year'))

    return


@sc.timer()
def test_arithmetic():
    """ The arithmetic table in 02-types.md, row by row, and the check of the slot a typed value lands in """
    sc.heading('Testing the arithmetic of typed values')
    ops = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    d, r, f, p = ss.days, ss.perday, ss.time.freqperday, ss.time.probperday
    assert sum([d(1), d(2)]) == d(3) # sum() starts from 0
    rows = [ # Left, op, right, and the result: a value, or the code of the error
        (d(3), '+', ss.weeks(1), d(10)),  (d(10), '-', ss.weeks(1), d(3)), # Units are converted
        (d(3), '*', 2, d(6)),  (d(6), '/', 2, d(3)),
        (d(6), '/', d(3), 2),
        (r(0.1), '+', ss.peryear(36.5), r(0.2)),  (r(0.2), '-', ss.peryear(36.5), r(0.1)), # Competing hazards add
        (r(0.1), '*', 2, r(0.2)),  (r(0.2), '/', 2, r(0.1)),
        (r(0.1), '*', d(3), 1 - np.exp(-0.3)), # A probability, not the expected count
        (f(0.1), '*', d(3), 0.3), # An expected count
        (p(0.1), '+', p(0.1), p(0.19)),  (2, '*', p(0.1), p(0.19)),  (p(0.19), '-', p(0.1), p(0.1)), # Through the rates, consistently
        (p(0.1), '*', d(2), 0.19),
        (r(0.2), '/', r(0.1), 2),
        (1, '/', d(10), r(0.1)),  (2, '/', r(0.1), d(20)),  (2, '/', f(0.1), d(20)),
        (r(0.1), '+', p(0.1), TypeError),  (r(0.1), '-', f(0.1), TypeError),  (r(0.1), '+', 1, TypeError),  (1, '-', r(0.1), TypeError),  (d(3), '+', 1, TypeError),  (d(3), '-', 1, TypeError),  (1, '-', d(3), TypeError),  (d(3), '+', r(0.1), TypeError),  (1, '/', p(0.1), TypeError),
        (r(0.1), '*', r(0.1), TypeError),  (r(0.1), '/', d(3), TypeError),  (d(3), '*', d(3), TypeError),  (d(3), '/', r(0.1), TypeError),
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
    for out in [r(0.1) * ts, ts * r(0.1)]:
        assert isinstance(out, ss.expressions.Expr) # A rate times a time series is a rate that varies in time, in either order

    # The slot is checked: a rate times a duration, or a duration, is not a rate, and a rate is not a dwell time
    S, I, R = ss.states('S I R')
    for slip in [lambda: ss.Transition(I, R, rate=ss.peryear(0.1) * ss.days(3)), lambda: I >> R @ (ss.peryear(0.1) * ss.days(3)),
                 lambda: ss.Transition(I, R, rate=ss.days(10)), lambda: ss.Transition(I, R, dur=ss.prob(0.1, 'day')),
                 lambda: ss.Disease(S >> I @ ss.perday(0.3) * I * ss.days(3), I >> R @ ss.days(10), name='sir')]:
        with pytest.raises(TypeError):
            slip()
    sir = ss.Disease(S >> I @ 2.5 * I / ss.days(10), I >> R @ (1/ss.days(10)), name='sir') # A duration that divides is fine
    assert ss.Transition(I, R, dur=2/ss.perday(0.1)).dur.raw == ss.days(20)
    return sir


# Run as a script
@sc.timer()
def test_bare_units():
    sc.heading('Testing the units of bare numbers beside typed ones')
    def sis(rec):
        S, I = ss.states('S I')
        return ss.Disease(inf=S >> I @ ss.permonth(0.0)*I, rec=I >> S @ rec, init_n=dict(I=1000), name='d')

    # A bare dt beside calendar years is in years, with a warning, though the model's unit is months (before, 2,881 points instead of 241)
    with pytest.warns(RuntimeWarning, match='is read in years'):
        sim = ss.Sim(sis(ss.months(6)), start=2000, stop=2020, dt=1/12, n_agents=1000, verbose=0).init()
    assert sim.t.npts == 241

    # A choice of durations as a dwell, typed or with unit= (it crashed, and unit= was refused); seeds infectious by contact are counted to the end of the step they leave in, as infections by contact are
    for rec in [ss.choice(a=[ss.months(9), ss.months(10)]), ss.choice(a=[9, 10], unit='month')]:
        n_I = ss.Sim(sis(rec), start=2000, stop=2002, dt=ss.months(1), n_agents=1000, verbose=0).run().results.d.I
        assert n_I[9] == 1000 and 0 < n_I[10] < 1000 and n_I[11] == 0

    # The numbers a function gives a distribution are in its unit= (they were read as years, so nobody left)
    fn = lambda self, sim, uids: np.full(len(uids), 6.0)
    n_I = ss.Sim(sis(ss.lognorm_ex(mean=fn, std=1, unit='month')), start=2000, stop=2003, dt=ss.months(1), n_agents=1000, verbose=0).run().results.d.I
    assert n_I[24] == 0

    # A bare beta is in the unit of the disease's own durations, beside births and deaths per year (it raised IncompatibleTypes)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(S >> I @ 1.9*I, I >> R @ ss.days(8), name='sir')
    sim = ss.Sim(sir, demographics=[ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.02))], start='2000', stop='2001', n_agents=500, verbose=0).init()
    assert sim.unit == 'day'

    # A peak of a seasonal value outside the year is refused, and a time of year works (T6a: 142 was taken as January)
    with pytest.raises(ValueError):
        ss.seasonal(peak=142)
    assert np.isclose(ss.seasonal(peak=ss.days(142)).peak_frac, 142/365)

    # A cycle of another period (TY15f): two peaks a year, in a declared rate
    rains = ss.seasonal(mean=ss.perday(0.2), amplitude=0.5, peak='April', period=ss.years(0.5))
    assert np.allclose(rains.factor(np.array([3.5, 9.5])/12), 1.5) and np.allclose(rains.factor(np.array([0.5, 6.5])/12), 0.5)
    sis = ss.Disease(S >> I @ rains*I, I >> S @ ss.days(10), init_prev=0.1, name='sis')
    ss.Sim(sis, start='2000', stop='2002', verbose=0).run()

    # A bare number from 1900 on is a year in a sim with dates, as with numeric years; a smaller one is in the sim's unit after the start (2000 was read as 2000 days)
    for start in [2020, '2020-01-01']:
        sim = ss.Sim(sir, start=start, stop=2021, n_agents=100, verbose=0).init()
        assert np.allclose(ss.timeseries({2000: 1, 2030.5: 2, '2040-01-01': 3}).years(sim), [2000, 2030.5, 2040])
    assert ss.connectors.to_time(sim, 30) == ss.date('2020-01-31') # Days, the sim's unit

    # A time series of bare numbers as a crude birth rate is per year, as a bare number is (it was per day, and the population exploded)
    alive = [ss.Sim(demographics=ss.birth(rate=rate), total_pop=1000, start='2000-01-01', stop='2001-01-01', verbose=0).run().results.alive[-1] for rate in [0.04, ss.timeseries({2000: 0.04, 2030: 0.02})]]
    assert np.isclose(alive[0], alive[1]) and alive[0] < 1100
    return


@sc.timer()
def test_time_text():
    sc.heading('Testing times as a run prints them, and dates on a timeline in calendar years')
    import io, contextlib, warnings
    def make():
        S, I, R = ss.states('S I R')
        return ss.Disease(S >> I @ ss.perday(1.9)*I, I >> R @ ss.days(8), init_n=dict(I=10), name='measles')

    # Numbers as numbers in the timeline's own unit, dates as dates without a time of day (they were "days 2000 to 2002" and "on day 2000-01-08 00:27:36")
    sim = ss.Sim(make(), total_pop=1e5, start=2000, stop=2002, verbose=0).run()
    assert 'years 2000 to 2002' in sim.start_lines()[0] and 'on year 2000.02' in sim.end_line()
    sim = ss.Sim(make(), total_pop=1e5, start='2000', stop='2002', verbose=0).run()
    assert '2000-01-01 to 2002-01-01' in sim.start_lines()[0] and sim.end_line().endswith('on 2000-01-08')
    sim = ss.Sim(make(), total_pop=1e5, stop=ss.days(7300), dt=7, verbose=0).init() # A stop that is not a whole number of steps is said (it ended at day 7294 silently)
    assert 'days 0 to 7294 (the last whole step before 7300)' in sim.start_lines()[0]

    # With no unit, times are steps (they were "days 0 to 50"), and a module of its own with nothing per person has no population (it was "10,000 people, as an ODE")
    S, I, R = ss.states('S I R')
    sim = ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ 0.1, name='sir'), stop=50, verbose=0).run()
    assert '10,000 people, as an ODE, steps 0 to 50' in sim.start_lines()[0] and 'on step ' in sim.end_line()
    class Own(ss.Module):
        def step(self):
            pass
    assert 'own, as an ODE, steps 0 to 50' in ss.Sim(Own(), stop=50, verbose=0).init().start_lines()[0]
    assert '500 people, as an ODE' in ss.Sim(Own(), total_pop=500, stop=50, verbose=0).init().start_lines()[0]

    # A date given to a delivery in a sim in calendar years, without warnings (they were "Dates with years < 1 are not valid")
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim = ss.Sim(ss.SIS(), interventions=ss.vaccinate(efficacy=0.8, coverage=0.5, at=['2001'], capacity=50), n_agents=500, start=2000, stop=2002, dt=ss.months(1), verbose=0).run()
    assert np.flatnonzero(sim.interventions[0].results.delivered.values).tolist() == [13] # The step from 2001

    # The step count warning says where the unit came from (it said "from None" with unit='day')
    with pytest.warns(RuntimeWarning, match="unit='day'"):
        ss.Sim(make(), total_pop=1e5, stop=ss.years(30), unit='day', verbose=0).init()
    return


@sc.timer()
def test_calendar_months():
    sc.heading('Test calendar months: dates from the 31st, sims stepped by them, and what ss.datedur takes')
    from dateutil.relativedelta import relativedelta

    # Each date is a whole number of months from the start, so the end of a month stays the end of a month
    dates = ss.date.arange('2020-01-31', '2020-05-31', ss.datedur(months=1))
    assert [str(d) for d in dates] == ['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31']

    # A calendar duration as a plain one, in either spelling of the unit
    assert ss.datedur(months=6).to('year') == ss.years(0.5) and ss.datedur(months=6).years == 0.5

    # A declared disease stepped by calendar months runs as with ss.months(1)
    kw = dict(solver='ode', start='2020-01-01', stop='2022-01-01', verbose=0)
    a, b = [ss.Sim(ss.SIR(beta=ss.peryear(3)), dt=dt, **kw).run() for dt in [ss.datedur(months=1), ss.months(1)]]
    assert np.allclose(a.results.sir.I, b.results.sir.I)

    # Hours or an absolute day are not calendar units, so they are refused rather than dropped
    assert ss.datedur(relativedelta(months=2, days=3)) == ss.datedur(months=2, days=3)
    for bad in [relativedelta(months=2, hours=3), relativedelta(day=31)]:
        with pytest.raises(ValueError):
            ss.datedur(bad)
    return a


if __name__ == '__main__':
    T = sc.timer('\nTotal time')

    o1 = test_ratio()
    o2 = test_classes()
    o3 = test_callable_dists()
    o4 = test_syntax()
    o5 = test_datearray_operations()
    o6 = test_timepar_float()
    o7 = test_to_dt()
    o8 = test_link_timepars()
    o9 = test_years_conversion()
    o10 = test_prob_conversion()
    o11 = test_arithmetic()
    o12 = test_bare_units()
    o13 = test_calendar_months()

    T.toc()
