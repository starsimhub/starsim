"""
Test time.py for coverage improvement.
"""

import sciris as sc
import numpy as np
import starsim as ss
import pickle
import pytest
from starsim.time import approx_compare, normalize_unit, get_dur_class, get_rate_class

do_plot = False
sc.options(interactive=False)


@sc.timer()
def test_helpers(do_plot=do_plot):
    """ Test approx_compare, normalize_unit, get_dur_class, get_rate_class """
    sc.heading('Testing helpers...')
    assert approx_compare(1.0, '==', 1.0+1e-15) is True, 'Close values should be equal'
    assert approx_compare(1.0, '<', 2.0) is True, '1 < 2'
    assert approx_compare(2.0, '>', 1.0) is True, '2 > 1'
    with pytest.raises(ValueError):
        approx_compare(1.0, '!=', 2.0)  # pragma: no cover
    assert normalize_unit('year') == 'years', 'Should add s'
    assert normalize_unit(ss.years(1)) == 'years', 'Should extract base'
    with pytest.raises(ValueError):
        normalize_unit('seconds')  # pragma: no cover
    with pytest.raises(TypeError):
        normalize_unit(42)  # pragma: no cover
    assert get_dur_class('years') is ss.years, 'Should return ss.years'
    assert get_rate_class('peryear') is ss.peryear, 'Should return ss.peryear'
    return True


@sc.timer()
def test_date_repr_hash_json_pickle(do_plot=do_plot):
    """ Test date repr, str, hash, to_json/from_json, pickle, from_year, from_array """
    sc.heading('Testing date methods...')
    d = ss.date('2020-06-15')
    assert '<' in repr(d) and '2020' in repr(d), f'Bad repr: {repr(d)}'
    assert '<' not in str(d), 'str should not have brackets'
    assert {d: 1}[d] == 1, 'Date should be hashable'

    # JSON round-trip
    assert d == ss.date.from_json(d.to_json()), 'JSON round-trip failed'
    with pytest.raises(ValueError):
        ss.date.from_json({'bad': 'x'})  # pragma: no cover

    # Pickle
    assert pickle.loads(pickle.dumps(d)) == d, 'Pickle round-trip failed'

    # from_year branches
    assert ss.date.from_year(2020) == ss.date('2020-01-01'), 'Int year'
    assert ss.date.from_year(2020.5).year == 2020, 'Float year'
    assert ss.date.from_year(2020.5, day_round=False).year == 2020, 'No-round'
    assert isinstance(ss.date.from_year(0.5, allow_zero=True), ss.datedur), 'Year<1 -> datedur'
    with pytest.raises(ValueError):
        ss.date.from_year(0.5, allow_zero=False)  # pragma: no cover

    # from_array
    assert len(ss.date.from_array(np.array([2020.0, 2021.0]))) == 2, 'from_array failed'
    return d


@sc.timer()
def test_date_comparisons(do_plot=do_plot):
    """ Test date comparison operators with numbers and arrays """
    sc.heading('Testing date comparisons...')
    d = ss.date('2020-06-15')
    assert d < 2021 and d > 2019 and d <= 2021 and d >= 2019 and d != 2019, 'Scalar comparisons failed'
    arr = np.array([2019.0, 2020.0, 2021.0])
    assert (d < arr)[-1] == True, 'Array < failed'
    assert (d > arr)[0] == True, 'Array > failed'
    assert (d <= arr)[-1] == True, 'Array <= failed'
    assert (d >= arr)[0] == True, 'Array >= failed'
    assert (d == arr)[1] or not (d == arr)[0], 'Array == failed'  # Approximate
    assert (d != arr)[0] == True, 'Array != failed'
    return True


@sc.timer()
def test_dur_ops(do_plot=do_plot):
    """ Test dur repr, str, hash, neg, abs, comparisons, rtruediv """
    sc.heading('Testing dur ops...')
    assert 'years' in repr(ss.years(2)), 'repr missing "years"'
    assert str(ss.years(1)) == 'year', 'str(years(1)) should be "year"'
    assert str(ss.days(5)) == 'days(5)', 'str(days(5)) wrong'
    assert hash(ss.years(1)) == hash(ss.days(365)), 'Equal dur hashes should match'
    assert (-ss.years(2)).value == -2, 'Negation failed'
    assert abs(-ss.years(2)).value == 2, 'abs failed'
    assert ss.years(1) < ss.years(2) and ss.years(2) > ss.years(1), 'Dur comparison failed'
    assert ss.years(1) <= ss.years(1) and ss.years(1) >= ss.years(1), 'Dur le/ge failed'
    assert ss.years(1) != ss.years(2), 'Dur ne failed'
    assert isinstance(1 / ss.years(1), ss.freq), '1/dur should be freq'
    with pytest.raises(ZeroDivisionError):
        _ = 1 / ss.years(0)  # pragma: no cover
    assert (0 / ss.years(1)).value == 0, '0/dur should be freq(0)'
    return True


@sc.timer()
def test_datedur_methods(do_plot=do_plot):
    """ Test datedur repr, to_dict, to_dur, to_array, to_numpy, is_variable, scale, abs, truediv, compare, str """
    sc.heading('Testing datedur methods...')
    dd = ss.datedur(years=1, days=10)
    assert 'years=1' in repr(dd), f'Bad repr: {repr(dd)}'
    assert repr(ss.datedur(0)) == 'datedur(0)', 'Zero repr'
    assert dd.to_dict()['years'] == 1, 'to_dict failed'
    assert dd.to_array()[0] == 1, 'to_array failed'
    assert isinstance(dd.to_numpy(), float), 'to_numpy should be float'
    assert isinstance(dd.to_dur(), ss.dur), 'to_dur should return dur'
    assert ss.datedur(0).to_dur().value == 0, 'Zero to_dur'
    assert dd.is_variable, 'years datedur should be variable'
    assert not ss.datedur(days=5).is_variable, 'days-only should not be variable'
    assert isinstance(dd.scale(dd.value, 2), type(dd.value)), 'scale should return DateOffset'
    assert abs(ss.datedur(years=-1)) > 0, 'abs of negative should be positive'
    assert np.isclose(ss.datedur(weeks=1) / ss.datedur(days=1), 7.0, rtol=1e-10), '1w/1d should be 7'
    assert ss.datedur.round_duration(dict(weeks=2.5)) is not None, 'round_duration failed'
    assert 'year' in dd.str(), 'str() should mention year'
    assert ss.datedur(days=1).str() == 'day', 'Single day str'
    assert ss.datedur(years=1) < ss.datedur(years=2), 'datedur compare failed'
    assert ss.datedur(years=1) < 2.0, 'datedur vs number failed'
    return dd


@sc.timer()
def test_rate_methods(do_plot=do_plot):
    """ Test Rate repr, eq, div, rtruediv, neg, to_prob, to_events, set_default_dur """
    sc.heading('Testing Rate methods...')
    assert 'peryear' in repr(ss.peryear(0.5)), 'Bad peryear repr'
    assert repr(ss.per(0.5, ss.years(1))).startswith('per('), 'Base rate repr'
    assert ss.freqperyear(1) == ss.freqperyear(1), 'Rate equality failed'
    assert ss.freqperyear(2) / ss.freqperyear(1) == 2.0, 'Rate/rate ratio'
    assert (ss.peryear(1) / 2).value == 0.5, 'Rate/scalar'
    assert isinstance(2 / ss.freqperyear(0.5), ss.dur), 'scalar/rate should be dur'

    # to_prob edge cases
    r = ss.peryear(0)
    r.set_default_dur(ss.years(1))
    assert r.to_prob(ss.years(1)) == 0, 'Zero rate -> zero prob'

    # to_events
    assert ss.freqperyear(10).to_events(ss.years(2)) == 20, 'to_events failed'
    # to_events with numeric dur
    ev2 = ss.freqperyear(10).to_events(2.0)
    assert isinstance(ev2, ss.freq), 'to_events with numeric dur should return freq'

    # Unitless Rate eq
    r1 = ss.prob(0.5)
    r2 = ss.prob(0.5)
    assert r1 == r2, 'Unitless prob equality failed'
    return True


@sc.timer()
def test_datearray_properties(do_plot=do_plot):
    """ Test DateArray is_float, years, to_date, to_float, to_human, is_, pickle, unit """
    sc.heading('Testing DateArray properties...')
    darr = ss.DateArray([ss.date('2020-01-01'), ss.date('2021-01-01')])
    assert darr.is_date and not darr.is_float, 'Type checks failed'
    assert np.isclose(darr.years[0], 2020.0, atol=0.01), 'years property failed'
    assert isinstance(darr.to_float(), ss.DateArray), 'to_float should return DateArray'
    assert not isinstance(darr.to_float(to_numpy=True), ss.DateArray), 'to_numpy should be plain array'
    assert darr.to_date() is not None, 'to_date failed'
    assert darr.to_human() is not None, 'to_human failed'
    assert darr.is_('date'), 'is_("date") should be True'
    with pytest.raises(ValueError):
        darr.is_('invalid')  # pragma: no cover
    assert pickle.loads(pickle.dumps(darr)).shape == darr.shape, 'DateArray pickle failed'
    assert ss.DateArray([1.0, 2.0], unit=ss.years).is_dur, 'Explicit unit should work'
    import datetime as dt
    assert ss.DateArray([dt.date(2020, 1, 1)]).is_date, 'datetime conversion failed'
    return darr


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    test_helpers(do_plot=do_plot)
    test_date_repr_hash_json_pickle(do_plot=do_plot)
    test_date_comparisons(do_plot=do_plot)
    test_dur_ops(do_plot=do_plot)
    test_datedur_methods(do_plot=do_plot)
    test_rate_methods(do_plot=do_plot)
    test_datearray_properties(do_plot=do_plot)
    T.toc()
