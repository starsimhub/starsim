"""
Test different time units and timesteps
"""
import numpy as np
import sciris as sc
import starsim as ss

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
    assert np.isclose(float(ss.datedur(weeks=1) - ss.datedur(days=1)), 6/365)
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


# Run as a script
if __name__ == '__main__':
    T = sc.timer('\nTotal time')

    o1 = test_ratio()
    o2 = test_classes()
    o3 = test_callable_dists()
    o4 = test_syntax()

    T.toc()
