"""
Test different time units and timesteps
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
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
    r1 = ss.freqperyear(2)
    r2 = ss.freqperyear(3)
    r3 = ss.freqperyear(2/0.1)
    r4 = ss.freqperyear(3/0.2)

    assert r1 + r2 == ss.freqperyear(5)
    assert r3 + r4 == ss.freqperyear(20+15)
    assert r3 * 2 == ss.freqperyear(4/0.1)
    assert r4 / 2 == ss.freqperyear(1.5/0.2)

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
    assert np.isclose(tp0*ss.years(0.5), tpval/2, rtol=0.1) # These should be close, but not match exactly
    assert np.isclose(tp0*ss.years(2), tpval*2, rtol=0.1)
    assert tp0*ss.years(0.5) == 1 - np.exp(np.log(1-0.1) * ss.years(0.5)/ss.years(1)) # These should be close, but not match exactly
    assert tp0*ss.years(2) == 1 - np.exp(np.log(1-0.1) * ss.years(2)/ss.years(1)) # These should be close, but not match exactly

    return d3, d4, r3, r4, tp0



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
    sc.heading('Testing full syntax')

    assert float(ss.date(1500))==1500
    assert np.isclose(float(ss.date(1500.1)), 1500.1) # Not exactly equal, but very close
    assert np.all((ss.years(1)*np.arange(5)) == (np.arange(5)*ss.years(1)))
    assert np.isclose(ss.datedur(weeks=1)/ss.datedur(days=1), 7) # TODO: would be nice if this were exact, but maybe impossible
    assert np.isclose((ss.datedur(weeks=1) - ss.datedur(days=1)).years, 6/365)
    assert np.isclose((ss.date(2050)-ss.date(2020)).years, ss.years(30).years, rtol=1/365) # Not exact due to leap years
    assert np.isclose((ss.freqperweek(1)+ss.freqperday(1)).value, ss.freqperweek(8).value) # CKTODO: would be nice if this were exact

    assert ss.date('2020-01-01') + ss.datedur(weeks=52) == ss.date('2020-12-30') # Should give us 30th December 2020
    assert ss.date('2020-01-01') + 52*ss.datedur(weeks=1) == ss.date('2020-12-30')# Should give us 30th December 2020
    assert ss.date('2020-01-01') + 52*ss.years(1/52) == ss.date('2021-01-01') # Should give us 1st Jan 2021
    assert ss.date('2020-01-01') + ss.datedur(years=1) == ss.date('2021-01-01') # Should give us 1st Jan 2021

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
    assert (1/ss.years(1)) == ss.freqperyear(1)
    assert (2/ss.years(1)) == ss.freqperyear(2)
    assert (4/ss.years(1)) == ss.freqperyear(4)
    assert (4/ss.datedur(1)) == ss.freqperyear(4)
    assert (ss.freqperday(5)*ss.datedur(days=1)) == 5
    assert 2/ss.freqperyear(0.25) == ss.years(8)
    assert 1/(2*ss.freqperyear(0.25)) == ss.years(2)
    assert ss.freqperyear(0.5)/ss.freqperyear(1) == 0.5

    # Probabilities
    p = ss.prob(0.1, ss.datedur(years=1))
    f = lambda factor: 1 - np.exp(-(-np.log(1 - p.value))/factor)
    assert p*ss.datedur(years=2) == f(0.5)
    assert p * ss.years(0.5) == f(2)
    assert p * ss.datedur(months=1) == f(12)

    p = ss.prob(0.1, ss.years(1))
    assert p*ss.datedur(years=2) == f(0.5)
    assert p * ss.years(0.5 ) == f(2)
    assert p * ss.datedur(months=1) == f(12)

    p = ss.per(0.1, ss.datedur(years=1))
    f = lambda factor: 1 - np.exp(-p.value/factor)
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

    a = ss.DateArray([ss.date('2020-01-01'), ss.date('2021-01-01')])
    b = ss.DateArray([ss.years(1), ss.years(2)])
    c = ss.years([1, 2])

    expected_dates_plus_month = ss.DateArray([ss.date('2020-01-31'), ss.date('2021-01-31')])
    assert np.array_equal(a + ss.months(1), expected_dates_plus_month)
    assert np.array_equal(ss.months(1) + a, expected_dates_plus_month)

    expected_months_plus = ss.DateArray([ss.months(13), ss.months(25)])
    assert np.array_equal(ss.months(1) + b, expected_months_plus)
    assert np.array_equal(b + ss.months(1), expected_months_plus)

    expected_months_plus_float = ss.months([13, 25])
    assert np.array_equal(ss.months(1) + c, expected_months_plus_float)
    expected_years_plus = ss.years([1 + 1/12, 2 + 1/12])
    assert np.array_equal(c + ss.months(1), expected_years_plus)

    with pytest.raises(TypeError):
        _ = a + ss.date('2020-01-01')
    with pytest.raises(TypeError):
        _ = ss.date('2020-01-01') + a

    expected_dates_plus = ss.DateArray([ss.date('2021-01-01'), ss.date('2022-01-01')])
    assert np.array_equal(b + ss.date('2020-01-01'), expected_dates_plus)
    assert np.array_equal(ss.date('2020-01-01') + b, expected_dates_plus)
    assert np.array_equal(c + ss.date('2020-01-01'), expected_dates_plus)
    assert np.array_equal(ss.date('2020-01-01') + c, expected_dates_plus)

    expected_dates_minus_month = ss.DateArray([ss.date('2019-12-02'), ss.date('2020-12-02')])
    assert np.array_equal(a - ss.months(1), expected_dates_minus_month)

    with pytest.raises(TypeError):
        _ = ss.months(1) - a

    expected_months_minus = ss.DateArray([ss.months(-11), ss.months(-23)])
    assert np.array_equal(ss.months(1) - b, expected_months_minus)

    expected_months_sub = ss.DateArray([ss.months(11), ss.months(23)])
    assert np.array_equal(b - ss.months(1), expected_months_sub)
    expected_months_minus_float = ss.months([-11, -23])
    assert np.array_equal(ss.months(1) - c, expected_months_minus_float)
    expected_years_minus = ss.years([1 - 1/12, 2 - 1/12])
    assert np.array_equal(c - ss.months(1), expected_years_minus)

    expected_datedur_from_date = ss.DateArray([ss.datedur(years=0), ss.datedur(years=-1)])
    assert np.array_equal(ss.date('2020-01-01') - a, expected_datedur_from_date)

    expected_datedur_from_array = ss.DateArray([ss.datedur(years=0), ss.datedur(years=1)])
    assert np.array_equal(a - ss.date('2020-01-01'), expected_datedur_from_array)

    with pytest.raises(TypeError):
        _ = b - ss.date('2020-01-01')
    with pytest.raises(TypeError):
        _ = c - ss.date('2020-01-01')

    expected_dates_from_date = ss.DateArray([ss.date('2019-01-01'), ss.date('2018-01-01')])
    assert np.array_equal(ss.date('2020-01-01') - b, expected_dates_from_date)
    assert np.array_equal(ss.date('2020-01-01') - c, expected_dates_from_date)

    return


@sc.timer()
def test_timepar_float():
    sc.heading('Test that timepars cannot be silently stripped of their units')

    # float() would return the numerator alone, so peryear(1), permonth(1) and perday(1)
    # would all be 1.0 despite differing by a factor of 365
    for rate in [ss.peryear(1), ss.permonth(1), ss.perday(1), ss.probperyear(0.5), ss.freqperyear(80)]:
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
    assert np.isnan(ss.freqperyear(np.nan))
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
    tl.ti = 500 # Deliberately past the end, to hit the extrapolation branch
    assert np.isclose(tl.now('yearvec'), 500/365), 'now("yearvec") should be in years, not days'

    # ss.MFNet compares the timestep against agent ages, which are in years, so a
    # monthly timestep must be passed as 1/12 of a year rather than as the bare 1
    class SpyNet(ss.MFNet):
        def set_network_states(self, upper_age=None):
            self.spied_upper_age = upper_age
            return super().set_network_states(upper_age=upper_age)

    sim = ss.Sim(n_agents=200, start=2000, stop=2002, dt=ss.months(1), diseases=ss.SIS(),
                 networks=SpyNet(), verbose=0)
    sim.run()
    upper_age = sim.networks[0].spied_upper_age
    assert np.isclose(upper_age, 1/12), f'Expected 1/12 of a year, not {upper_age}'

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
    assert np.isclose(ss.freqperyear(120).to_dt(ss.months(1)), 10)

    # Unlinked timepars raise rather than guessing
    with pytest.raises(ValueError):
        ss.years(100).to_dt()

    # Inside a module, the timestep is filled in automatically
    module = ss.mock_module(dt=ss.months(1))
    dur = ss.years(100).set_default_dur(module.dt)
    assert dur.to_dt() == 1200

    # Rates can convert to a duration unit as well as a rate class
    assert ss.peryear(1).to('month') == ss.peryear(1).to(ss.permonth)
    assert isinstance(ss.probperyear(0.5).to('month'), ss.probpermonth)
    assert isinstance(ss.freqperyear(80).to(ss.months), ss.freqpermonth)

    return


@sc.timer()
def test_link_timepars():
    sc.heading('Test that module durations and rates are linked to the module dt')

    sim = ss.Sim(n_agents=100, start=2000, stop=2002, dt=ss.months(1), diseases=ss.SIS(),
                 networks=ss.RandomNet(), verbose=0)
    sim.init()
    sis = sim.diseases.sis
    assert sis.pars.beta.default_dur == sis.t.dt # Rates were already linked
    assert sis.pars.waning.default_dur == sis.t.dt

    # Durations are now linked too
    dur = ss.years(100)
    sis.pars.dur_test = dur
    sis.link_timepars()
    assert sis.pars.dur_test.to_dt() == 100*12

    return


@sc.timer()
def test_prob_conversion():
    """ Probabilities are not linear in time, so converting one between units must go via the rate """
    sc.heading('Test converting probabilities between units')

    # to() used to rescale the value linearly, giving 0.5/12 rather than 1-(1-0.5)**(1/12)
    p = ss.probperyear(0.5)
    expected = 1 - (1 - 0.5)**(1/12)
    for target in [ss.probpermonth, ss.months, 'month']:
        assert np.isclose(p.to(target).value, expected)
    assert np.isclose(p.to_prob(ss.months(1)), expected) # to() and to_prob() now agree
    assert np.isclose(ss.probpermonth(ss.probperyear(0.5)).value, expected) # As does direct construction

    # Scaling up used to raise, since 0.5*365 fails the prob range check
    assert ss.probperday(0.5).to(ss.probperyear).value == 1
    assert ss.probperyear(ss.probperday(0.5)).value == 1

    # Edge cases and round trips
    assert p.to(ss.probperyear).value == 0.5 # A no-op conversion is exact
    assert np.isclose(p.to(ss.probpermonth).to(ss.probperyear).value, 0.5)
    assert ss.probperyear(0).to(ss.probpermonth).value == 0
    assert ss.probperyear(1).to(ss.probpermonth).value == 1 # A certainty stays a certainty
    arr = ss.probperyear(np.array([0.1, 0.5])).to(ss.probpermonth).value
    assert np.allclose(arr, 1 - (1 - np.array([0.1, 0.5]))**(1/12))

    # Converting to or from a prob uses the underlying rate, not the raw value
    assert np.isclose(ss.probpermonth(ss.peryear(0.5)).value, 1 - np.exp(-0.5/12))
    assert np.isclose(ss.peryear(ss.probpermonth(0.05)).value, -np.log(1-0.05)*12)

    # ss.per and ss.freq are genuinely linear, so they are unchanged
    assert np.isclose(ss.peryear(0.5).to(ss.permonth).value, 0.5/12)
    assert np.isclose(ss.freqperyear(12).to(ss.freqpermonth).value, 1.0)
    assert np.isclose(ss.permonth(1).to(ss.peryear).value, 12)

    # A unitless prob has no time unit to convert against, so it raises rather than guessing
    with pytest.raises(TypeError):
        ss.prob(ss.probperyear(0.5))

    return


# Run as a script
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

    T.toc()
