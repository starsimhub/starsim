"""
Functions and classes for handling time
"""
import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss

__all__ = ['date', 'Dur', 'YearDur', 'DateDur', 'Rate', 'timeprob', 'rateprob',
           'Time', 'years', 'months', 'weeks', 'days', 'perday', 'perweek', 'permonth', 'peryear']

#%% Base classes

class date(pd.Timestamp):
    """
    Define a point in time, based on `pd.Timestamp`

    Args:
        date (int/float/str/datetime): Any type of date input (ints and floats will be interpreted as years)
        kwargs (dict): passed to pd.Timestamp()

    **Examples**:

        ss.date(2020) # Returns <2020-01-01>
        ss.date(year=2020) # Returns <2020-01-01>
        ss.date(year=2024.75) # Returns <2024-10-01>
        ss.date('2024-04-04') # Returns <2024-04-04>
        ss.date(year=2024, month=4, day=4) # Returns <2024-04-04>
    """

    def __new__(cls, *args, **kwargs):
        """ Check if a year was supplied, and preprocess it; complex due to pd.Timestamp implementation """
        single_year_arg = False
        if len(args) == 1:
            arg = args[0]
            if arg is None:
                return pd.Timestamp(None)
            elif isinstance(arg, pd.Timestamp):
                return cls._reset_class(sc.dcp(arg))
            elif sc.isnumber(arg):
                single_year_arg = True
        year_kwarg = len(args) == 0 and len(kwargs) == 1 and 'year' in kwargs
        if single_year_arg:
            return cls.from_year(args[0])
        if year_kwarg:
            return cls.from_year(kwargs['year'])

        # Otherwise, proceed as normal
        out = super(date, cls).__new__(cls, *args, **kwargs)
        out = cls._reset_class(out)
        return out

    @classmethod
    def _reset_class(cls, obj):
        """ Manually reset the class from pd.Timestamp to ss.date """
        obj.__class__ = date
        return obj

    def __reduce__(self):
        # This function is internally used when pickling (rather than getstate/setstate)
        # due to Pandas implementing C functions for this. We can wrap the Pandas unpickler
        # with a function that wraps calls _reset_class to support unpickling date
        # objects at the instance level
        unpickling_func, args = super().__reduce__()
        return (self.__class__._rebuild, (self.__class__, unpickling_func, args))

    @staticmethod
    def _rebuild(cls, unpickling_func, args):
        out = unpickling_func(*args)
        out = cls._reset_class(out)
        return out

    def __repr__(self, bracket=True):
        """ Show the date in brackets, e.g. <2024.04.04> """
        _ = ss.options.date_sep
        y = f'{self.year:04d}'
        m = f'{self.month:02d}'
        d = f'{self.day:02d}'
        string = y + _ + m + _ + d

        if self._time_repr != '00:00:00':
            string += ' ' + self._time_repr[:8]

        if bracket:
            string = '<' + string + '>'
        return string

    def __str__(self):
        """ Like repr, but just the date, e.g. 2024.04.04 """
        return self.__repr__(bracket=False)

    def replace(self, *args, **kwargs):
        """ Returns a new ss.date(); pd.Timestamp is immutable """
        out = super().replace(*args, **kwargs)
        out = self.__class__._reset_class(out)
        return out

    def disp(self, **kwargs):
        """ Show the full object """
        return sc.pr(self, **kwargs)

    @classmethod
    def from_year(cls, year):
        """
        Convert an int or float year to a date.

        **Examples**:

            ss.date.from_year(2020) # Returns <2020-01-01>
            ss.date.from_year(2024.75) # Returns <2024-10-01>
        """

        if isinstance(year, int):
            return cls(year=year, month=1, day=1)
        else:
            year_start = pd.Timestamp(year=int(year), month=1, day=1)
            year_end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
            return cls._reset_class(year_start + year % 1 * (year_end - year_start))

    def to_year(self):
        """
        Convert a date to a floating-point year

        **Examples**:

            ss.date('2020-01-01').to_year() # Returns 2020.0
            ss.date('2024-10-01').to_year() # Returns 2024.7486
        """
        year_start = pd.Timestamp(year=self.year,month=1,day=1).timestamp()
        year_end = pd.Timestamp(year=self.year+1,month=1,day=1).timestamp()
        return self.year + (self.timestamp()-year_start)/(year_end-year_start)

    def __float__(self):
        return self.to_year()

    @property
    def years(self):
        # This matches Dur.years
        return self.to_year()

    def to_pandas(self):
        """ Convert to a standard pd.Timestamp instance """
        return pd.Timestamp(self.to_numpy()) # Need to convert to NumPy first or it doesn't do anything

    def _timestamp_add(self, other):
        """ Uses pd.Timestamp's __add__ and avoids creating duplicate objects """
        orig_class = self.__class__
        self.__class__ = pd.Timestamp
        out = orig_class._reset_class(self + other)
        orig_class._reset_class(self)
        return out

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__add__)(other)

        if isinstance(other, DateDur):
            return self._timestamp_add(other.unit)
        elif isinstance(other, YearDur):
            return date(self.to_year() + other.unit)
        elif isinstance(other, pd.DateOffset):
            return self._timestamp_add(other)
        elif isinstance(other, pd.Timestamp):
            raise TypeError('Cannot add a date to another date')
        elif sc.isnumber(other):
            raise TypeError(f'Attempted to add a number ({other}) to a date, which is not supported. Only durations can be added to dates e.g., "ss.years({other})" or "ss.days({other})"')
        else:
            raise TypeError(f'Attempted to add an instance of {other.__class__.__name__} to a date, which is not supported. Only durations can be added to dates.')

    def __sub__(self, other):

        if isinstance(other, np.ndarray):
            return np.vectorize(self.__sub__)(other)

        if isinstance(other, DateDur):
            return date(self.to_pandas() - other.unit)
        elif isinstance(other, YearDur):
            return date(self.to_year() - other.unit)
        elif isinstance(other, pd.DateOffset):
            return date(self.to_pandas() - other)
        elif isinstance(other, date):
            return YearDur(self.years-other.years)
        else:
            raise TypeError('Unsupported type')
    #
    def __radd__(self, other): return self.__add__(other)
    # def __iadd__(self, other): return self.__add__(other) # I think pd.Timestamp is immutable so these shouldn't be implemented?
    def __rsub__(self, other): return self.__sub__(other) # TODO: check if this should be reversed
    # def __isub__(self, other): return self.__sub__(other)

    def __lt__(self, other):
        if sc.isnumber(other):
            return self.to_year() < other
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__lt__)(other)
        else:
            return super().__lt__(other)

    def __gt__(self, other):
        if sc.isnumber(other):
            return self.to_year() > other
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__gt__)(other)
        else:
            return super().__gt__(other)

    def __le__(self, other):
        if sc.isnumber(other):
            return self.to_year() <= other
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__le__)(other)
        else:
            return super().__le__(other)

    def __ge__(self, other):
        if sc.isnumber(other):
            return self.to_year() >= other
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__ge__)(other)
        else:
            return super().__ge__(other)

    def __eq__(self, other):
        if sc.isnumber(other):
            return self.to_year() == other
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__eq__)(other)
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        if sc.isnumber(other):
            return self.to_year() != other
        elif isinstance(other, np.ndarray):
            return np.vectorize(self.__ne__)(other)
        else:
            return super().__ne__(other)

    def __hash__(self):
        # As equality with float years is implemented, this allows interoperability with dicts that use float year keys
        return hash(self.to_year())

    # Convenience methods based on application usage

    @classmethod
    def from_array(cls, array):
        """
        Convert an array of float years into an array of date instances

        Args:
            array: An array of float years

        Returns:
            An array of date instances
        """
        return np.vectorize(cls)(array)

    @classmethod
    def arange(cls, low, high, step=1):
        """
        Construct an array of dates

        Functions similarly to np.arange, but returns date objects

        Example usage:

        >>> date.arange(2020, 2025)
            array([<2020.01.01>, <2021.01.01>, <2022.01.01>, <2023.01.01>,
                   <2024.01.01>], dtype=object)

        Args:
            low: Lower bound - can be a date or a numerical year
            high: Upper bound - can be a date or a numerical year
            step: Assumes 1 calendar year steps by default

        Returns:
            An array of date instances
        """

        if isinstance(step, ss.DateDur):
            if not isinstance(low, date):
                low = cls(low)
            if not isinstance(high, date):
                high = cls(high)

            tvec = []
            t = low
            while t <= high:
                tvec.append(t)
                t += step
            return np.array(tvec)
        else:
            low = low.years if isinstance(low, date) else low
            high = high.years if isinstance(high, date) else high
            return cls.from_array(np.arange(low, high, step))

    def to_json(self):
        """ Returns a JSON representation of the date """
        out = {'ss.date': str(self)}
        return out

    @staticmethod
    def from_json(json):
        """ Reconstruct a date from a JSON; reverse of `to_json()` """
        if not (isinstance(json, dict) and len(json) == 1 and list(json.keys())[0] == 'ss.date'):
            errormsg = f'Expecting a dict with a single key "ss.date", not {json}'
            raise ValueError(errormsg)
        return date(json['ss.date'])


class Dur():
    # Base class for durations/dts
    # Subclasses for date-durations and fixed-durations

    # Conversion ratios from date-based durations to fixed durations
    ratios = sc.objdict(
        years=1,
        months=12,
        weeks=365.25/12/7, # This calculation allows a year to consist of 365.25 days. Note that this means a
        days=7,
        hours=24,
        minutes=60,
        seconds=60,
        milliseconds=1000,
        microseconds=1000,
        nanoseconds=1000,
    )

    def __new__(cls, *args, **kwargs):
        # Return
        if cls is Dur:
            if args:
                if isinstance(args[0], (pd.DateOffset, DateDur)):
                    return super().__new__(DateDur)
                elif isinstance(args[0], YearDur):
                    return super().__new__(YearDur)
                else:
                    assert len(args) == 1, f'Dur must be instantiated with only 1 arg (which is in years), or keyword arguments. {len(args)} args were given.'
                    return super().__new__(YearDur)
            else:
                return super().__new__(DateDur)
        return super().__new__(cls)

    # NB. Durations are considered to be equal if their year-equivalent duration is equal
    # That would mean that Dur(years=1)==Dur(1) returns True - probably less confusing than having it return False?
    def __hash__(self):
        return hash(self.years)

    def __float__(self):
        return float(self.years)

    @property
    def years(self):
        # Return approximate conversion into years
        raise NotImplementedError

    @property
    def to_numpy(self):
        raise NotImplementedError

    @property
    def is_variable(self):
        """
        Returns True if the duration has variable length

        Some date-based durations like years and months correspond to a variable
        period of time. This attribute captures whether or not the quantity has
        variable length.
        """
        raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

    def __lt__(self, other):
        try:
            return self.years < other.years
        except:
            return self.years < other

    def __gt__(self, other):
        try:
            return self.years > other.years
        except:
            return self.years > other

    def __le__(self, other):
        try:
            return self.years <= other.years
        except:
            return self.years <= other

    def __ge__(self, other):
        try:
            return self.years >= other.years
        except:
            return self.years >= other

    def __eq__(self, other):
        try:
            return self.years == other.years
        except:
            return self.years == other

    def __ne__(self, other):
        try:
            return self.years != other.years
        except:
            return self.years != other

    def __truediv__(self, other):
        raise NotImplementedError('Dur subclasses are required to implement this method')

    def __rtruediv__(self, other):
        # If a Dur is divided by a Dur then we will call __truediv__
        # If a float is divided by a Dur, then we should return a rate
        # If a rate is divided by a Dur, then we will call Rate.__truediv__
        # We also need divide the duration by the numerator when calculating the rate
        if self.years == 0:
            raise ZeroDivisionError('Cannot divide by a duration of zero') # TODO: consider Rate(Dur(np.inf))
        elif sc.isnumber(other) and other == 0:
            return Rate(0)
        else:
            return Rate(other, self)

    def str(self):
        # This method should return a readable representation of the duration e.g., 'day', '2 days', 'year'
        # Not using __str__ because we still want print(Dur) to show the repr
        raise NotImplementedError('Dur subclasses are required to implement this method')

    @classmethod
    def arange(cls, low, high, step):
        """
        Construct an array of Dur instances

        For this function, the low, high, and step must ALL be specified, and they must
        all be Dur instances. Mixing Dur types (YearDur and DateDur) is permitted.

        Args:
            low: Starting point e.g., ss.Dur(0)
            high:
            step:

        Returns:
        """

        assert isinstance(low, Dur), 'Low input must be an ss.Dur'
        assert isinstance(high, Dur), 'High input must be an ss.Dur'
        assert isinstance(step, Dur), 'Step input must be an ss.Dur'

        tvec = []
        t = low
        while t <= high:
            tvec.append(t)
            t += step
        return np.array(tvec)


class YearDur(Dur):
    # Year based duration e.g., if requiring 52 weeks per year

    def __init__(self, years=0):
        """
        Construct a year-based fixed duration

        Supported inputs are
            - A float number of years (so can use values like 1/52 for 1 week)
            - A YearDur instance (as a copy-constructor)
            - A DateDur instance (to convert from date-based to fixed representation)

        Args:
            years: float, YearDur, DateDur
        """
        if isinstance(years, Dur):
            self.unit = years.years
        else:
            self.unit = years

    def to_numpy(self):
        return self.unit

    @property
    def years(self):
        return self.unit # Needs to return float so matplotlib can plot it correctly

    @property
    def is_variable(self):
        return False

    def __add__(self, other):
        if isinstance(other, Dur):
            return self.__class__(self.unit + other.years)
        elif isinstance(other, date):
            return date.from_year(other.to_year() + self.unit)
        else:
            return self.__class__(self.unit + other)

    def __sub__(self, other):
        if isinstance(other, Dur):
            return self.__class__(max(self.unit - other.years, 0))
        elif isinstance(other, date):
            return date.from_year(other.to_year() - self.unit)
        else:
            return self.__class__(max(self.unit - other, 0))

    def __mul__(self, other: float):
        if isinstance(other, np.ndarray):
            return other*self # This means np.arange(2)*ss.years(1) and ss.years(1)*np.arange(2) both give [ss.years(1), ss.years(2)] as the output
        elif isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, Dur):
            raise Exception('Cannot multiply a duration by a duration')
        elif isinstance(other, date):
            raise Exception('Cannot multiply a duration by a date')
        return self.__class__(self.unit*other)

    def __truediv__(self, other):
        if isinstance(other, Dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            raise Exception('Cannot divide a duration by a rate')
        else:
            return self.__class__(self.unit / other)

    def __repr__(self):
        return f'<YearDur: {self.unit} years>'

    def str(self):
        if self.unit == 1:
            return 'year'
        else:
            return f'{self.unit} years'

    def __abs__(self):
        return self.__class__(abs(self.unit))


class DateDur(Dur):
    # date based duration e.g., if requiring a week to be 7 calendar days later

    def __init__(self, *args, **kwargs):
        """
        Create a date-based duration

        Supported arguments are
            - Single positional argument
                - A float number of years
                - A pd.DateOffset
                - A Dur instance
            - Keyword arguments
                - Argument names that match time periods in `Dur.ratios` e.g., 'years', 'months'
                  supporting floating point values. If the value is not an integer, the values
                  will be cascaded to smaller units using `Dur.ratios`

        If no arguments are specified, a DateDur with zero duration will be produced

        Args:
            args:
            kwargs:
        """
        if args:
            assert not kwargs, f'DateDur must be instantiated with only 1 arg (which is in years), or keyword arguments.'
            assert len(args) == 1, f'DateDur must be instantiated with only 1 arg (which is in years), or keyword arguments. {len(args)} args were given.'
            if isinstance(args[0], pd.DateOffset):
                self.unit = self._round_duration(args[0])
            elif isinstance(args[0], DateDur):
                self.unit = args[0].unit # pd.DateOffset is immutable so this should be OK
            elif isinstance(args[0], YearDur):
                self.unit = self._round_duration({'years': args[0].years})
            elif sc.isnumber(args[0]):
                self.unit = self._round_duration({'years': args[0]})
            else:
                raise TypeError('Unsupported input')
        else:
            self.unit = self._round_duration(kwargs)

    @classmethod
    def _as_array(cls, dateoffset: pd.DateOffset) -> np.ndarray:
        """
        Return array representation of a pd.DateOffset

        In the array representation, each element corresponds to one of the time units
        in self.ratios e.g., a 1 year, 2 week duration would be [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]

        Args:
            dateoffset: A pd.DateOffset object

        Returns:
            A numpy array with as many elements as `DateDur.ratios`
        """
        return np.array([dateoffset.kwds.get(k, 0) for k in cls.ratios.keys()])

    @classmethod
    def _as_args(cls, x) -> dict:
        """
        Return a dictionary representation of a DateOffset or DateOffset array

        This function takes in either a pd.DateOffset, or the result of `_as_array`, and returns
        a dictionary with the same keys as `DateDur.ratios` and the values from the input. The
        output of this function can be passed to the `Dur()` constructor to create a new DateDur object.

        Args:
            x: A pd.DateOffset or an array with the same number of elements as `DateDur.ratios`

        Returns:
            Dictionary with keys from `DateDur.ratios` and values from the input
        """
        if isinstance(x, pd.DateOffset):
            x = cls._as_array(x)
        return {k: v for k, v in zip(cls.ratios.keys(), x)}

    @property
    def years(self) -> float:
        """
        Return approximate conversion into years

        This allows interoperability with YearDur objects (which would typically be expected to
        occur if module parameters have been entered with `DateDur` durations, but the simulation
        timestep is in `YearDur` units).

        The conversion is based on `DateDur.ratios` which defines the conversion from each time unit
        to the next

        Returns:
            A float representing the duration in years
        """
        denominator = 1
        years = 0
        for k, v in self.ratios.items():
            denominator *= v
            years += self.unit.kwds.get(k, 0)/denominator
        return years

    @property
    def is_variable(self):
        if self.unit.kwds.get('years',0) or self.unit.kwds.get('months',0):
            return True
        else:
            return False

    @classmethod
    def _round_duration(cls, vals) -> pd.DateOffset:
        """
        Round a dictionary of duration values by overflowing remainders

        The input can be
            - A numpy array of length `cls.ratios` containing values in key order
            - A pd.DateOffset instance
            - A dictionary with keys from `cls.ratios`

        The output will be a pd.DateOffset with integer values, where non-integer values
        have been handled by overflow using the ratios in `cls.ratios`. For example, 2.5 weeks
        would first become 2 weeks and 0.5*7 = 3.5 days, and then become 3 days + 0.5*24 = 12 hours.

        Negative values are supported - -1.5 weeks for example will become (-1w, -3d, -12h)

        Returns:
            A pd.DateOffset
        """
        if isinstance(vals, np.ndarray):
            d = sc.objdict({k:v for k,v in zip(cls.ratios, vals)})
        elif isinstance(vals, pd.DateOffset):
            d = sc.objdict.fromkeys(cls.ratios.keys(),0)
            d.update(vals.kwds)
        elif isinstance(vals, dict):
            d = sc.objdict.fromkeys(cls.ratios.keys(),0)
            d.update(vals)
        else:
            raise TypeError()

        if all([isinstance(val, int) for val in d.values()]):
            if isinstance(vals, pd.DateOffset): return vals  # pd.DateOffset is immutable so this should be OK
            return pd.DateOffset(**d)

        for i in range(len(cls.ratios)-1):
            remainder, div = np.modf(d[i])
            d[i] = int(div)
            d[i+1] += remainder * cls.ratios[i+1]
        d[-1] = int(d[-1])

        return pd.DateOffset(**d)


    def _scale(self, dateoffset: pd.DateOffset, scale: float) -> pd.DateOffset:
        """
        Scale a pd.DateOffset by a factor

        This function will automatically cascade remainders to finer units using `DateDur.ratios` so for
        example 2.5 weeks would first become 2 weeks and 0.5*7 = 3.5 days,
        and then become 3 days + 0.5*24 = 12 hours.

        Args:
            dateoffset: A pd.DateOffset instance
            scale: A float scaling factor (must be positive)

        Returns:
            A pd.DateOffset instance scaled by the requested amount
        """
        return self._round_duration(self._as_array(dateoffset) * scale)

    def to_numpy(self):
        # TODO: This is a quick fix to get plotting to work, but this should do something more sensible
        return self.years
        # return self.unit.to_numpy()

    def __pow__(self, other):
        raise Exception('Cannot multiply a duration by a duration')

    def __mul__(self, other: float):
        if isinstance(other, np.ndarray):
            return other*self
        elif isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, Dur):
            raise Exception('Cannot multiply a duration by a duration')
        return self.__class__(self._scale(self.unit, other))

    def __truediv__(self, other):
        if isinstance(other, DateDur):
            # We need to convert both dates into a common timebase to do the division
            # We *can* just do self.years/other.years, however this can cause numerical precision losses
            # yet at the same time is not necessary, because the common timebase depends on the durations.
            # For example, DateDur(weeks=1)/DateDur(days=1) should return 7, but we get 6.9999999 if we convert
            # both to years. Instead though, we can just convert days into weeks instead of days into years, and
            # then divide 1 week by (1/7) days.
            self_array = self._as_array(self.unit)
            other_array = other._as_array(other.unit)
            unit = np.argmax((self_array != 0) | (other_array != 0))
            denominator = 1
            a = 0
            b = 0
            for i, v in enumerate(self.ratios.values()):
                if i < unit:
                    continue
                if i > unit:
                    denominator *= v
                a += self_array[i] / denominator
                b += other_array[i] / denominator
            return a/b
        elif isinstance(other, Dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            raise Exception('Cannot divide a duration by a rate')
        return self.__class__(self._scale(self.unit, 1/other))

    def __repr__(self):
        if self.years == 0:
            return '<DateDur: 0>'
        else:
            labels = self.ratios.keys()
            vals = self._as_array(self.unit).astype(float)

            time_portion = vals[4:]
            time_portion[-4] += time_portion[-1]*1e-9 + time_portion[-2]*1e-6 + time_portion[-3]*1e-3 # Collapse it down to seconds
            time_portion[-4] = int(vals[-4]*100)/100
            time_str = ':'.join(f'{np.round(v,1):02g}' for v in time_portion[:3])

            return '<DateDur: ' +  ','.join([f'{k}={int(v)}' for k, v in zip(labels[:4], vals[:4]) if v!=0]) + (f', +{time_str}' if time_str != '00:00:00' else '') + '>'

    def str(self):
        # Friendly representation e.g., 'day', '1 year, 2 months, 1 day'
        vals = self._as_array(self.unit)

        # If we have only one nonzero value, return 'year', 'month', etc.
        if np.count_nonzero(vals == 1) == 1:
            for unit, val in zip(self.ratios, vals):
                if val == 1:
                    return unit[:-1]

        strs = [f'{v} {k[:-1] if abs(v) == 1 else k}' for k, v in zip(self.ratios, vals) if v != 0]
        return ', '.join(strs)


    def __add__(self, other):
        if isinstance(other, date):
            return other + self.unit
        elif isinstance(other, DateDur):
            return self.__class__(**self._as_args(self._as_array(self.unit) + self._as_array(other.unit)))
        elif isinstance(other, pd.DateOffset):
            return self.__class__(**self._as_args(self._as_array(self.unit) + self._as_array(other)))
        elif isinstance(other, YearDur):
            kwargs = {k: v for k, v in zip(self.ratios, self._as_array(self.unit))}
            kwargs['years'] += other.years
            return self.__class__(**kwargs)
        else:
            raise TypeError('For a DateDur instance, it is only possible to add or subtract dates, Dur objects, or pd.DateOffset objects')

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __abs__(self):
        # Cannot implement this because it's ambiguous how to resolve cases like
        # DateDur(months=-1,days=1) - this is a sensible DateDur interpreted as 'go back 1 month, then go forward 1 day'
        # but just taking the absolute value of all of the components wouldn't work because this would be on average 1 month + 1 day
        # whereas it should be 1 month - 1 day. This could probably be resolved? But is an edge case, unlikely to be needed
        # (whereas abs(YearDur) arises when comparing dates, which automatically return YearDur)
        raise NotImplementedError('The absolute value of a DateDur instance is undefined as components (e.g., months, days) may have different signs.')


class Rate():
    """
    Store a value per unit time e.g., 2 per day
    - self.value - the numerator (e.g., 2) - a scalar float
    - self.unit - the denominator (e.g., 1 day) - a Dur object
    """

    def __init__(self, value:float, unit:Dur=None):
        if unit is None:
            self.unit = YearDur(1)
        elif isinstance(unit, Dur):
            self.unit = unit
        else:
            self.unit = Dur(unit)

        assert sc.isnumber(value), 'Value must be a scalar number'
        self.value = value

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.value} per {self.unit.str()}>' # Use str to get the friendly representation

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return other*self
        elif isinstance(other, Dur):
            return self.value*(other/self.unit)
        else:
            return self.__class__(self.value*other, self.unit)

    def __neg__(self):
        return -1*self

    def __add__(self, other):
        if self.__class__ == other.__class__:
            return self.__class__(self.value+other*self.unit, self.unit)
        elif not isinstance(other, Rate):
            if sc.isnumber(other) or isinstance(other, np.ndarray):
                raise TypeError('Only rates can be added to rates. This error most commonly occurs if the rate needs to be multiplied by `self.t.dt` to get a number of events per timestep.')
            else:
                raise TypeError('Only rates can be added to rates')
        else:
            raise TypeError('Can only add rates of identical types (e.g., Rate+Rate, timeprob+timeprob)')

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if self.__class__ == other.__class__:
            return self.__class__(self.value-other*self.unit, self.unit)
        elif not isinstance(other, Rate):
            if sc.isnumber(other) or isinstance(other, np.ndarray):
                raise TypeError('Only rates can be added to rates. This error most commonly occurs if the rate needs to be multiplied by `self.t.dt` to get a number of events per timestep.')
            else:
                raise TypeError('Only rates can be added to rates')
        else:
            raise TypeError('Can only subtract rates of identical types (e.g., Rate+Rate, timeprob+timeprob)')

    def __eq__(self, other):
        return self.value == other.value/other.unit*self.unit

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        # This is for <rate>/<other>
        if isinstance(other, Rate):
            # Convert the other rate onto our dt, then divide the value
            return self.value/(other.value/other.unit*self.unit)
        elif isinstance(other, Dur):
            raise Exception('Cannot divide a rate by a duration')
        else:
            return self.__class__(self.value/other, self.unit)

    def __rtruediv__(self, other):
        # If a float is divided by a rate
        return other * (self.unit/self.value)

    def to_prob(self, dur, v=1):
        """
        Calculate a time-specific probability value

        This function is mainly useful for subclasses where the multiplication by a duration is non-linear
        (e.g., `timeprob`) and therefore it is important to apply the factor prior to multiplication by duration.
        This function avoids creating an intermediate array of rates, and is therefore much higher performance.

        e.g.

        >>> p = ss.timeprob(0.05)*self.cd4*self.t.dt

        and

        >>> p = ss.timeprob(0.05).scale(self.cd4,self.t.dt)

        are equivalent, except that the second one is (much) faster.

        Args:
            v: The factor to scale the rate by. This factor is applied before multiplication by the duration
            dur: An ss.Dur instance to scale the rate by (often this is `dt`)

        Returns:
            A numpy float array of values
        """
        if isinstance(dur, np.ndarray):
            factor = (dur/self.unit).astype(float)
        else:
            factor = dur/self.unit
        return (self.value*v)*factor


class timeprob(Rate):
    """
    `timeprob` represents the probability of an event occurring during a
    specified period of time.

    The class is designed to allow conversion of a probability from one
    duration to another through multiplication. However, the behavior of this
    conversion depends on the data type of the the object being multiplied.

    When multiplied by a duration (type ss.Dur), the underlying constant rate is
    calculated as
        `rate = -np.log(1 - self.value)`.
    Then, the probability over the new duration is
        `p = 1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the new duration to the original duration.

    For example,
    >>> p = ss.timeprob(0.8, ss.years(1))
    indicates a 80% chance of an event occurring in one year.

    >>> p*ss.years(1)
    When multiplied by the original denominator, 1 year in this case, the
    probability remains unchanged, 80%.

    >>> p * ss.years(2)
    Multiplying `p` by `ss.years(2)` does not simply double the
    probability to 160% (which is not possible), but rather returns a new
    probability of 96% representing the chance of the event occurring at least
    once over the new duration of two years.

    However, the behavior is different when a `timeprob` object is multiplied
    by a scalar or array. In this case, the probability is simply scaled. This scaling
    may result in a value greater than 1, which is not valid. For example,
    >>> p * 2
    raises an AssertionError because the resulting probability (160%) exceeds 100%.

    Use `rateprob` instead if `timeprob` if you would prefer to directly
    specify the instantaneous rate.
    """

    def __init__(self, value, unit=None):
        assert 0 <= value <= 1, 'Value must be between 0 and 1'
        return super().__init__(value, unit)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return other*self # This means np.arange(2)*ss.years(1) and ss.years(1)*np.arange(2) both give [ss.years(1), ss.years(2)] as the output
        elif isinstance(other, Dur):
            if self.value == 0:
                return 0
            elif self.value == 1:
                return 1
            else:
                factor = self.unit/other
                if factor == 1:
                    return self.value # Avoid expensive calculation and precision issues
                rate = -np.log(1 - self.value)
                return 1 - np.exp(-rate/factor)
        else:
            return self.__class__(self.value*other, self.unit)

    def to_prob(self, dur, v=1):
        if isinstance(dur, np.ndarray):
            factor = (dur/self.unit).astype(float)
        else:
            factor = dur/self.unit
        rate = -np.log(1-(self.value*v))
        return 1-np.exp(-rate*factor)

    def __truediv__(self, other): raise NotImplementedError()
    def __rtruediv__(self, other): raise NotImplementedError()


class rateprob(Rate):
    """
    A `rateprob` represents an instantaneous rate of an event occurring. Rates
    must be non-negative, but need not be less than 1.

    Through multiplication, rate can be modified or converted to a probability,
    depending on the data type of the object being multiplied.

    When a `rateprob` is multiplied by a scalar or array, the rate is simply
    scaled. Such multiplication occurs frequently in epidemiological models,
    where the base rate is multiplied by "rate ratio" or "relative rate" to
    represent agents experiencing higher (multiplier > 1) or lower (multiplier <
    1) event rates.

    Alternatively, when a `rateprob` is multiplied by a duration (type
    ss.Dur), a probability is calculated. The conversion from rate to
    probability on multiplication by a duration is
        `1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the multiplied duration to the original
    period (denominator).

    For example, consider
    >>> p = ss.rateprob(0.8, ss.years(1))
    When multiplied by a duration of 1 year, the calculated probability is
        `1 - np.exp(-0.8)`, which is approximately 55%.
    >>> p*ss.years(1)

    When multiplied by a scalar, the rate is simply scaled.
    >>> p*2

    The difference between `timeprob` and `rateprob` is subtle, but important. `rateprob` works directly
    with the instantaneous rate of an event occurring. In contrast, `timeprob` starts with a probability and a duration,
    and the underlying rate is calculated. On multiplication by a duration,
    * rateprob: rate -> probability
    * timeprob: probability -> rate -> probability

    The behavior of both classes is depending on the data type of the object being multiplied.
    """
    def __init__(self, value, unit=None):
        assert value >= 0, 'Value must be >= 0'
        return super().__init__(value, unit)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return other*self
        elif isinstance(other, Dur):
            if self.value == 0:
                return 0
            else:
                factor = self.unit/other
                return 1 - np.exp(-self.value/factor)
        else:
            return self.__class__(self.value*other, self.unit)

    def to_prob(self, dur, v=1):
        if isinstance(dur, np.ndarray):
            factor = (dur/self.unit).astype(float)
        else:
            factor = dur/self.unit
        rate = self.value*v
        return 1-np.exp(-rate*factor)

    def __truediv__(self, other):
        if isinstance(other, Dur):
            raise NotImplementedError()
        return super().__truediv__(other)  # Fixed the call to super().__truediv__

    def __rtruediv__(self, other): raise NotImplementedError()


#%% Simulation time vectors

class Time:
    """
    Handle time vectors for both simulations and modules.

    Each module can have its own time instance, in the case where the time vector
    is defined by absolute dates, these time vectors are by definition aligned. Otherwise
    they can be specified using Dur objects which express relative times (they can be added
    to a date to get an absolute time)

    Args:
        start : ss.date or ss.Dur
        stop : ss.date if start is an ss.date, or an ss.Dur if start is an ss.Dur
        dt (ss.Dur): Simulation step size
        pars (dict): if provided, populate parameter values from this dictionary
        parent (obj): if provided, populate missing parameter values from a 'parent" `Time` instance
        name (str): if provided, name the `Time` object
        init (bool): whether or not to immediately initialize the Time object
        sim (bool/Sim): if True, initializes as a sim-specific `Time` instance; if a Sim instance, initialize the absolute time vector

    The `Time` object, after initialization, has the following attributes:

    - `ti` (int): the current timestep
    - `npts` (int): the number of timesteps
    - `tvec` (array): time either as absolute `ss.date` instances, or relative `ss.Dur` instances
    - `yearvec` (array): time represented as floating-point years

    **Examples**:

        t1 = ss.Time(start=2000, stop=2020, dt=1.0)
        t2 = ss.Time(start='2021-01-01', stop='2021-04-04', dt=ss.days(2))
    """

    # Allowable time arguments
    time_args = ['start', 'stop', 'dt']
    default_dur = Dur(50)
    default_start = date(2000)
    default_dt = Dur(1)

    def __init__(self, start=None, stop=None, dt=None, dur=None, name=None):

        self.name = name
        self.start = start
        self.stop = stop
        self.dt = dt
        self.dur = dur
        self.ti = 0 # The time index, e.g. 0, 1, 2
        self._tvec    = None # The time vector for this instance in date or Dur format
        self._yearvec = None # Time vector as floating point years
        self.initialized = False # Call self.init(sim) to initialise the object

        return

    @property
    def tvec(self):
        if self._tvec is None:
            raise Exception('Time object has not yet been not initialized - call `init()` before using the object')
        else:
            return self._tvec

    @property
    def yearvec(self):
        if self._yearvec is None:
            raise Exception('Time object has not yet been not initialized - call `init()` before using the object')
        else:
            return self._yearvec

    def __repr__(self):
        if self.initialized:
            return f'<Time t={self.tvec[self.ti]}, ti={self.ti}, {self.start}-{self.stop} dt={self.dt}>'
        else:
            return '<Time (uninitialized)>'

    def disp(self):
        return sc.pr(self)

    @property
    def timevec(self):
        # Backwards-compatibility function for now - TODO: consider deprecating?
        return self.tvec

    @property
    def datevec(self):
        # Backwards-compatibility function for now - TODO: consider deprecating?
        return self.tvec

    @property
    def npts(self):
        try:
            return self.tvec.shape[0]
        except:
            return 0

    @property
    def dt_year(self):
        return self.dt.years

    def __bool__(self):
        """ Always truthy """
        return True

    def __len__(self):
        """ Length is the number of timepoints """
        return self.npts

    @property
    def ready(self):
        """ Check if all parameters are in place to be initialized """
        return not any([getattr(self, k) is None for k in self.time_args])

    @property
    def is_absolute(self):
        """
        Check whether the fundamental simulation unit is absolute

        A time vector is absolute if the start is a date rather than a Dur
        A relative time vector can be made absolute by adding a date to it
        """
        try:
            return isinstance(self.start, date)
        except:
            return False

    def update(self, pars=None, parent=None, reset=True, force=None, **kwargs):
        """ Reconcile different ways of supplying inputs """
        pars = sc.mergedicts(pars)
        stale = False

        for key in self.time_args:
            current_val = getattr(self, key, None)
            parent_val = getattr(parent, key, None)
            kw_val = kwargs.get(key)
            par_val = pars.get(key)

            # Special handling for dt: don't inherit dt if the units are different
            if key == 'dt':
                if isinstance(parent, Time):
                    if parent.unit != self.unit:
                        parent_val = 1.0

            if force is False: # Only update missing (None) values
                val = sc.ifelse(current_val, kw_val, par_val, parent_val)
            elif force is None: # Prioritize current value
                val = sc.ifelse(kw_val, par_val, current_val, parent_val)
            elif force is True: # Prioritize parent value
                val = sc.ifelse(kw_val, par_val, parent_val, current_val)
            else:
                errormsg = f'Invalid value {force} for force: must be False, None, or True'
                raise ValueError(errormsg)

            if val != current_val:
                setattr(self, key, val)
                stale = True

        if stale and reset and self.initialized:
            self.init()
        return

    def init(self, sim=None):
        """ Initialize all vectors """

        if sim is not None:
            self.dt = sc.ifelse(self.dt, sim.t.dt, sim.pars.dt)
            self.start = sc.ifelse(self.start, sim.t.start, sim.pars.start)
            self.stop = sc.ifelse(self.stop, sim.t.stop, sim.pars.stop)
            self.dur = sc.ifelse(self.dur, sim.t.dur, sim.pars.dur)

        if sc.isnumber(self.dur):
            self.dur = Dur(self.dur)
        assert self.dur is None or isinstance(self.dur, Dur), 'Time.dur must be a number, a Dur object or None'

        match (self.start, self.stop, self.dur):
            case (None, None, None):
                start = self.default_start
                dur = self.default_dur
                stop = start+dur

            case (start, None, None):
                if isinstance(start, Dur):
                    pass  # Already a Dur which is fine
                elif sc.isnumber(start) and start < 1900:
                    start = Dur(start)
                else:
                    start = date(start)
                dur = self.default_dur
                stop = start+dur

            case (None, stop, None):
                if isinstance(stop, Dur):
                    start = stop.__class__(0)
                elif sc.isnumber(stop) and stop < 1900:
                    stop = Dur(stop)
                    start = Dur(0)
                else:
                    stop = date(stop)
                    start = self.default_start
                dur = stop-start

            case (None, None, dur):
                start = self.default_start
                stop = start+dur

            case (start, None, dur):
                if isinstance(start, Dur):
                    pass
                elif sc.isnumber(start) and start < 1900:
                    start = Dur(start)
                else:
                    start = date(start)
                stop = start+dur

            case (None, stop, dur):
                if isinstance(stop, Dur):
                    pass
                elif sc.isnumber(stop) and stop < 1900:
                    stop = Dur(stop)
                else:
                    stop = date(stop)
                start = stop-dur

            case (start, stop, dur):
                # Note that this block will run if dur is None and if it is not None, which is fine because
                # we are ignoring dur in this case (if the user specifies start and stop, they'll be used)
                if sc.isstring(start):
                    start = date(start)
                if sc.isstring(stop):
                    stop = date(stop)

                if sc.isnumber(start) and sc.isnumber(stop):
                    if stop < 1900:
                        start = Dur(start)
                        stop = Dur(stop)
                    else:
                        start = date(start)
                        stop = date(stop)
                elif sc.isnumber(start):
                    start = stop.__class__(start)
                elif sc.isnumber(stop):
                    stop = start.__class__(stop)
                dur = stop-start
            case _:
                raise Exception('Failed to match start, stop, and dur') # This should not occur

        assert isinstance(start, (date, Dur)), 'Start must be a date or Dur'
        assert isinstance(stop, (date, Dur)), 'Stop must be a date or Dur'
        assert type(start) is type(stop), 'Start and stop must be the same type'
        assert start <= stop, 'Start must be before stop'

        self.start = start
        self.stop = stop
        self.dur = dur

        if self.dt is None:
            self.dt = self.default_dt

        if sc.isnumber(self.dt):
            self.dt = Dur(self.dt)

        if sim is not None:
            if self.is_absolute != sim.t.is_absolute:
                raise Exception('Cannot mix absolute/relative times across modules')

        if sim is not None and sim.t.initialized and \
                all([type(getattr(self, key)) == type(getattr(sim.t, key)) for key in ('start', 'stop', 'dt')]) and \
                all([getattr(self, key) == getattr(sim.t, key) for key in ('start', 'stop', 'dt')]):
            self._yearvec = sc.dcp(sim.t._yearvec)
            self._tvec    = sc.cp(sim.t._tvec)  # ss.Dates are immutable so only need shallow copy
            self.initialized = True
            return self

        # We need to populate both the tvec (using dates) and the yearvec (using years). However, we
        # need to decide which of these quantities to prioritise considering that the calendar dates
        # don't convert consistently into fractional years due to varying month/year lengths. We will
        # prioritise one or the other depending on what type of quantity the user has specified for dt
        if isinstance(self.dt, YearDur):
            # If dt has been specified as a YearDur then preference setting fractional years. So first
            # calculate the fractional years, and then convert them to the equivalent dates
            self._yearvec = np.round(self.start.years + np.arange(0, self.stop.years - self.start.years + self.dt.years, self.dt.years), 12)  # Subtracting off self.start.years in np.arange increases floating point precision for that part of the operation, reducing the impact of rounding
            if isinstance(self.stop, Dur):
                self._tvec = np.array([self.stop.__class__(x) for x in self._yearvec])
            else:
                self._tvec = np.array([date(x) for x in self._yearvec])
        else:
            # If dt has been specified as a DateDur then preference setting dates. So first
            # calculate the dates/durations, and then convert them to the equivalent fractional years
            if isinstance(self.stop, Dur):
                self._tvec = ss.Dur.arange(self.start, self.stop, self.dt)
            else:
                self._tvec = ss.date.arange(self.start, self.stop, self.dt)
            self._yearvec = np.array([x.years for x in self._tvec])

        self.initialized = True
        return self

    def now(self, key=None):
        """
        Get the current simulation time

        Args:
            which (str): which type of time to get: default (None), "year", "date", "tvec", or "str"

        **Examples**:

            t = ss.Time(start='2021-01-01', stop='2022-02-02', dt=1, unit='week')
            t.ti = 25
            t.now() # Returns <2021-06-25>
            t.now('date') # Returns <2021-06-25>
            t.now('year') # Returns 2021.479
            t.now('str') # Returns '2021-06-25'
        """
        if key in [None, 'none', 'time', 'str']:
            vec = self.tvec
        elif key == 'year':
            vec = self.yearvec
        else:
            errormsg = f'Invalid key "{key}": must be None, abs, date, or year'
            raise ValueError(errormsg)

        if 0 <= self.ti < len(vec):
            now = vec[self.ti]
        else:
            now = self.tvec[0] + self.dt*self.ti
            if key == 'year':
                now = float(now)

        if key == 'str':
            now = str(now)

        return now


#%% Convenience functions

def years(x: float) -> Dur:
    return Dur(x)

def months(x: float) -> Dur:
    return Dur(months=x)

def weeks(x: float) -> Dur:
    return Dur(weeks=x)

def days(x: float) -> Dur:
    return Dur(days=x)

def perday(v):
    """Shortcut to specify rate per calendar day"""
    return Rate(v, Dur(days=1))

def perweek(v):
    """Shortcut to specify rate per calendar week"""
    return Rate(v, Dur(weeks=1))

def permonth(v):
    """Shortcut to specify rate per calendar month"""
    return Rate(v, Dur(months=1))

def peryear(v):
    """Shortcut to specify rate per numeric year"""
    return Rate(v, YearDur(1))