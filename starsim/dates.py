"""
Dates: `ss.date`, an array of them (`ss.DateArray`), and calendar durations (`ss.datedur`)
"""
import numbers
import operator
import calendar
import datetime as dt
import dateutil.relativedelta as drd
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .time import dur, factors, get_dur_class, Rate, valid_bases, wrong_dimension

__all__ = ['DateArray', 'date', 'datedur']


class DateArray(np.ndarray):
    """ Lightweight wrapper for an array of dates """
    # TODO: Consider splitting out into DurArray which could have additional validation/functionality for mixed duration types
    def __new__(cls, arr=None, unit=None):
        arr = sc.toarray(arr)
        if isinstance(arr, np.ndarray): # Shortcut to typical use case, where the input is an array
            if unit is None:
                if not len(arr): # Nothing to tell the unit from, e.g. a range of dates that ends before it starts
                    unit = ss.date
                elif isinstance(arr[0], (ss.date, ss.dur)):
                    unit = type(arr[0]) # Get the unit from the input
                elif isinstance(arr[0], (str, np.datetime64, dt.date, dt.datetime, pd.Timestamp)):
                    arr = np.fromiter((ss.date(x) for x in arr), dtype=object)
                    unit = ss.date
                else:
                    try: # Else, if the input is datelike (>1), assume date
                        assert np.all(arr >= 1.0)
                        unit = ss.date
                    except: # Otherwise, assume years
                        unit = ss.years
            else:
                if not isinstance(unit, type):
                    raise TypeError('unit must be a type (e.g., `ss.months`), not an instance (e.g., `ss.months(1)` or `ss.month`.')
                if not issubclass(unit, (ss.date, ss.dur)):
                    raise TypeError(f'unit must be a subclass of ss.date or ss.dur, {unit} is not derived from one of these parent classes.')

            out = arr.view(cls)
            out._unit = unit
            return out
        else:
            errormsg = f'Argument must be an array, not {type(arr)}'
            raise TypeError(errormsg)

    def __array_finalize__(self, obj):
        # Ensure that the unit is propagated when slicing and copying with dcp
        if obj is not None:
            self._unit = getattr(obj, "_unit", ss.date)
        return

    def __reduce__(self):
        # Ensure that the unit is propagated when pickling - works in conjunction with __setstate__ below
        reduced = super().__reduce__()
        if len(reduced) >= 2:
            return *reduced[0:2], reduced[2] + (self.unit, ), *reduced[3:]
        else:
            return reduced

    def __setstate__(self, state):
        super().__setstate__(state[:-1])
        self._unit = state[-1]

    @property
    def unit(self):
        try:    return self._unit
        except: return ss.date

    def __add__(self, other):
        return NotImplemented # Delegate to the other class - typically Date__radd__ or Dur.__radd__

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return NotImplemented # Delegate to other class

    def __rsub__(self, other):
        return self.__sub__(other)

    @property
    def is_date(self):
        return issubclass(self.unit, ss.date)

    @property
    def is_dur(self):
        return issubclass(self.unit, ss.dur)

    @property
    def is_datedur(self):
        return issubclass(self.unit, ss.datedur)

    @property
    def is_float(self):
        """ Whether the array holds plain numbers (False if it is empty) """
        return len(self) > 0 and isinstance(self[0], numbers.Number)

    @property
    def subdaily(self):
        """ Check if the array has sub-daily timesteps """
        try:
            delta = self[1] - self[0]
            years = getattr(delta, 'years', delta) # A date or duration difference must be converted to years, since that is what ss.date.subdaily() expects; a plain float is already in years
            return ss.date.subdaily(years)
        except:
            return False

    @property
    def years(self):
        """ Represent the dates as floating point years """
        if isinstance(self.unit, type) and issubclass(self.unit, date) and len(self) > 1:
            ns = np.array([d.value for d in self], dtype=np.int64)
            if not np.any(ns % 1_000_000_000): # Whole seconds (as every timeline's are): all at once, with the same result as date.to_year() for each
                return date._years_of(ns)
        return np.array([self.unit(d).years for d in self]) # Convert to the stored unit's type, then convert to years

    def to_date(self, inplace=False, day_round=None, die=True):
        """ Convert to ss.date

        Args:
            inplace (bool): whether to modify in place
            day_round (bool): whether to round dates to the nearest day (otherwise, keep timestamp); if None, round if and only if the span of the first timestep is at least one day
            die (bool): if False, then fall back to float if conversion to date fails (e.g. year 0)
        """
        day_round = sc.ifelse(day_round, not(self.subdaily))
        if day_round and self.unit is ss.years and len(self) and not inplace: # All at once
            years = self.to_float(to_numpy=True)
            if np.all(years >= 1):
                return ss.date.from_array(years)
        try:
            vals = [ss.date(x, day_round=day_round, allow_zero=False) for x in self]
        except Exception as e:
            if die:
                raise e
            else:
                return self.to_float(inplace=inplace)

        if inplace:
            if self.dtype != object:
                errormsg = f'Cannot convert a DateArray of dtype {self.dtype} to dates in place, since dates require dtype object; use inplace=False instead'
                raise TypeError(errormsg)
            self[:] = vals
            return
        else:
            return ss.DateArray(vals, unit=ss.date)

    def to_float(self, inplace=False, to_numpy=False):
        """ Convert to a float, returning a new DateArray unless inplace=True """
        if not len(self):
            vals = []
        elif self.is_dur and not self.is_datedur:
            vals = [x.value for x in self] # Use the magnitude in the array's own units, e.g. days(100) -> 100.0
        elif self.is_date or self.is_datedur: # Neither has a magnitude of its own, so use years
            vals = [x.years for x in self]
        else:
            vals = [float(x) for x in self] # Already floats
        if inplace:
            self[:] = vals
        else:
            if to_numpy:
                return np.array(vals)
            else:
                return ss.DateArray(vals, unit=self.unit) # Keep the unit information

    def to_human(self):
        """
        Return the most human-friendly (i.e. plotting-friendly) version of the dates,
        i.e. ss.date if possible, float otherwise
        """
        if self.is_date or not self.is_float:
            return self.to_date(die=False)
        else:
            return sc.dcp(self)

    def to_numpy(self):
        return self.to_array()

    def to_array(self, *args, **kwargs):
        """ Force conversion to an array """
        return np.array(self, *args, **kwargs)

    def days_since(self, start):
        """
        Exact (elapsed) number of days between each date and start, as an array of floats

        Unlike `dates - start`, which gives calendar differences (e.g. `ss.datedur(months=1)`), this counts days exactly.

        **Example**:

            dates = ss.DateArray(['2000-01-01', '2000-02-01', '2000-03-01'])
            dates.days_since('2000-01-01') # Returns array([ 0., 31., 60.])
        """
        if not self.is_date:
            errormsg = f'days_since() requires an array of dates, not {self.unit}'
            raise TypeError(errormsg)
        return np.asarray((pd.DatetimeIndex(self) - ss.date(start))/pd.Timedelta(days=1))


class date(pd.Timestamp):
    """
    Define a point in time, based on `pd.Timestamp`

    Args:
        date (int/float/str/datetime): Any type of date input (ints and floats will be interpreted as years)
        allow_zero (bool): if True, allow a year 0 by creating a datedur instead; if False, raise an exception; if None, give a warning
        kwargs (dict): passed to pd.Timestamp()

    Examples:
        ```python
        ss.date(2020) # Returns <2020-01-01>
        ss.date(year=2020) # Returns <2020-01-01>
        ss.date(year=2024.75) # Returns <2024-10-01>
        ss.date('2024-04-04') # Returns <2024-04-04>
        ss.date(year=2024, month=4, day=4) # Returns <2024-04-04>
        ```
    """
    def __new__(cls, *args, day_round=True, allow_zero=None, **kwargs):
        """ Check if a year was supplied, and preprocess it; complex due to pd.Timestamp implementation """
        single_year_arg = False
        if len(args) == 1:
            arg = args[0]
            if arg is None:
                return pd.Timestamp(None)
            elif type(arg) is date: # Dates are immutable, so need no copy
                return arg
            elif isinstance(arg, pd.Timestamp):
                return cls._reset_class(sc.dcp(arg))
            elif sc.isnumber(arg): # e.g. 2020
                single_year_arg = True
            elif isinstance(arg, ss.dur): # e.g. ss.years(2020)
                arg = arg.years
                single_year_arg = True

        # Handle converting a float year to a date
        years = None
        year_kwarg = len(args) == 0 and len(kwargs) == 1 and 'year' in kwargs

        if single_year_arg:
            years = arg
        if year_kwarg:
            years = kwargs['year']

        if years is not None:
            return cls.from_year(years, day_round=day_round, allow_zero=allow_zero)

        # Otherwise, proceed as normal
        else:
            self = super(date, cls).__new__(cls, *args, **kwargs)
            self = cls._reset_class(self)
            return self

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
        """ Show the date in brackets, e.g. <2024-04-04> """
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
        """ Like repr, but just the date, e.g. 2024-04-04 """
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
    def from_year(cls, year, day_round=True, allow_zero=None):
        """
        Convert an int or float year to a date.

        Args:
            year (float): the year to round
            day_round (bool): whether to round to the nearest day
            allow_zero (bool): whether to allow year 0 (if so, return ss.datedur instead)

        Examples:
            ```python
            ss.date.from_year(2020) # Returns <2020-01-01>
            ss.date.from_year(2024.75) # Returns <2024-10-01>
            ```
        """
        if year < 1:
            warnmsg = f'Dates with years < 1 are not valid ({year = }); returning ss.datedur instead'
            if allow_zero is False:
                raise ValueError(warnmsg)
            elif allow_zero is None:
                ss.warn(warnmsg)
            return ss.datedur(years=year)
        elif isinstance(year, int):
            return cls(year=year, month=1, day=1)
        else:
            if day_round: # As sc.yeartodate(), but without converting to and from a datetime.date
                full = int(year)
                start = pd.Timestamp(year=full, month=1, day=1)
                days = int(np.round((year - full)*(366 if calendar.isleap(full) else 365)))
                timestamp = start + pd.Timedelta(days=days)
            else:
                year_start = pd.Timestamp(year=int(year), month=1, day=1)
                year_end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                timestamp = year_start + year % 1 * (year_end - year_start)
            return cls._reset_class(timestamp)

    def to_year(self):
        """ Convert a date to a floating-point year

        Examples:
            ```python
            ss.date('2020-01-01').to_year() # Returns 2020.0
            ss.date('2024-10-01').to_year() # Returns 2024.7486
            ```
        """
        year_start = pd.Timestamp(year=self.year,month=1,day=1).timestamp()
        year_end = pd.Timestamp(year=self.year+1,month=1,day=1).timestamp()
        return self.year + (self.timestamp()-year_start)/(year_end-year_start)

    @staticmethod
    def _years_of(ns):
        """ As `to_year()`, for many dates at once, given as nanoseconds since 1970 (in whole seconds, for which it is the same to the last bit) """
        def seconds(x): # As pd.Timestamp.timestamp(): rounded to microseconds, with halves away from zero
            v = x/1e9
            return np.copysign(np.floor(np.abs(v)*1e6 + 0.5), v)/1e6
        year = ns.astype('datetime64[ns]').astype('datetime64[Y]')
        start = seconds(year.astype('datetime64[ns]').astype(np.int64))
        end = seconds((year + np.timedelta64(1, 'Y')).astype('datetime64[ns]').astype(np.int64))
        return year.astype(np.int64) + 1970 + (seconds(ns) - start)/(end - start)

    def __float__(self):
        return self.to_year()

    @property
    def years(self):
        """ Return the date as a number of years """
        return self.to_year()

    @staticmethod
    def subdaily(years): # TODO: add to dur as well? Could define a DateTime class that both inherit from, with this and any other duplicate methods
        """ Check if a subdaily timestep is used

        A date has no concept of a timestep, but add this as a convienence method since
        this is required by other methods (e.g. `ss.date.arange()`).
        """
        days = years*factors.years.days
        return days < 1.0

    def round(self, to='d'):
        """ Round to a given interval (by default a day) """
        timestamp = super().round(to)
        return self._reset_class(timestamp)

    def to_pandas(self):
        """ Convert to a standard pd.Timestamp instance """
        return pd.Timestamp(self.to_numpy()) # Need to convert to NumPy first or it doesn't do anything

    def _timestamp_add(self, other):
        """ Uses pd.Timestamp's __add__ and avoids creating duplicate objects """
        orig_class = self.__class__  # pylint: disable=access-member-before-definition
        self.__class__ = pd.Timestamp
        out = orig_class._reset_class(self + other)
        orig_class._reset_class(self)
        return out

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__add__)(other)
        elif isinstance(other, ss.datedur):
            return self._timestamp_add(other.value)
        elif isinstance(other, ss.dur):
            if other.is_array:
                return ss.DateArray(np.vectorize(self.__add__)(other))
            elif other.base in ['days', 'weeks'] and np.isclose(other.days, round(other.days)): # A whole number of days is added as days, not as a share of a year (which is a day off after a leap day)
                return self._timestamp_add(pd.Timedelta(days=round(other.days)))
            else:
                return date(self.to_year() + other.years)
        elif isinstance(other, (pd.DateOffset, dt.timedelta)):
            return self._timestamp_add(other)
        elif isinstance(other, pd.Timestamp):
            raise TypeError('Cannot add a date to another date')
        elif sc.isnumber(other):
            errormsg = f'Attempted to add "{other}" to a date, which is not supported. Only durations can be added to dates e.g., "ss.years({other})" or "ss.days({other})"'
            raise TypeError(errormsg)
        else:
            errormsg = f'Attempted to add an instance of {type(other)} to a date, which is not supported. Only durations can be added to dates.'
            raise TypeError(errormsg)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__sub__)(other)
        elif isinstance(other, ss.datedur):
            return ss.date(self.to_pandas() - other.value)
        elif isinstance(other, ss.dur):
            if other.is_array:
                return ss.DateArray(np.vectorize(self.__sub__)(other))
            else:
                return ss.date(self.to_year() - other.years)
        elif isinstance(other, (pd.DateOffset, dt.timedelta)):
            return ss.date(self.to_pandas() - other)
        elif isinstance(other, (ss.date, pd.Timestamp, dt.date, dt.datetime)):
            if isinstance(other, pd.Timestamp):
                other = other.to_pydatetime()
            delta = drd.relativedelta(self.to_pydatetime(), other)
            return ss.datedur(delta)
        else:
            errormsg = f'Attempted to subtract "{other}" ({type(other)}) from a date, which is not supported. Only durations can be subtracted from dates e.g., "ss.years({other})" or "ss.days({other})"'
            raise TypeError(errormsg)

    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return other.__class__([o-self for o in other]) # TODO: is this right?
        elif isinstance(other, ss.dur):
            # As this is __rsub__, this corresponds to `other - self` hence this operation is not permitted
            raise TypeError("Cannot subtract a date from a duration")
        elif isinstance(other, (ss.date, dt.date, dt.datetime)): # As date - date, so the same interval gives the same calendar duration
            return ss.date(other) - self
        else:
            errormsg = f'Attempted to subtract "{self}" from "{other}" ({type(other)}), which is not supported. Only dates can be subtracted from dates e.g., datetime.date(2020, 1, 1) - ss.date(2010), which will return ss.datedur.'
            raise TypeError(errormsg)

    def _compare(self, other, op):
        """ Compare with a number as a year, with each element of an array, with a string as the date it names (e.g. `sim.now == '2020-04-01'`), and otherwise as a timestamp """
        if sc.isnumber(other):
            return op(self.to_year(), other)
        elif isinstance(other, np.ndarray):
            return np.vectorize(lambda x: self._compare(x, op))(other)
        elif isinstance(other, str): # A string that is not a date raises, for == as for <, rather than comparing unequal
            return op(self, date(other))
        else:
            return getattr(pd.Timestamp, f'__{op.__name__}__')(self, other)

    def __lt__(self, other): return self._compare(other, operator.lt)
    def __gt__(self, other): return self._compare(other, operator.gt)
    def __le__(self, other): return self._compare(other, operator.le)
    def __ge__(self, other): return self._compare(other, operator.ge)
    def __eq__(self, other): return self._compare(other, operator.eq)
    def __ne__(self, other): return self._compare(other, operator.ne)

    def __hash__(self):
        # As equality with float years is implemented, this allows interoperability with dicts that use float year keys
        return hash(self.to_year())

    # Convenience methods based on application usage

    @classmethod
    def from_array(cls, array, day_round=True, allow_zero=True, date_type=None):
        """
        Convert an array of float years into an array of date instances

        Args:
            array (array): An array of float years
            day_round (bool): Whether to round to the nearest day
            allow_zero (bool): if True, allow a year 0 by creating a datedur instead; if False, raise an exception; if None, give a warning
            date_type (type): Optionally convert to a class other than `ss.date` (e.g. `ss.datedur`)

        Returns:
            An array of date instances
        """
        if date_type is None:
            date_type = cls
        kwargs = {}
        if date_type is ss.date:
            years = np.asarray(array)
            if day_round and years.ndim == 1 and years.dtype.kind in 'if' and np.all(years >= 1): # All at once, as from_year() does for each
                full = years.astype(np.int64)
                leap = (full % 4 == 0) & ((full % 100 != 0) | (full % 400 == 0))
                days = np.round((years - full)*np.where(leap, 366, 365)).astype(np.int64)
                stamps = (full - 1970).astype('datetime64[Y]').astype('datetime64[D]') + days
                return DateArray([cls._reset_class(stamp) for stamp in pd.DatetimeIndex(stamps.astype('datetime64[us]'))])
            kwargs = dict(day_round=day_round, allow_zero=allow_zero) # These arguments are only valid for ss.date, not ss.datedur
        return DateArray(np.vectorize(date_type)(array, **kwargs))

    @classmethod
    def arange(cls, start, stop, step=1.0, inclusive=True, day_round=None, allow_zero=True):
        """
        Construct an array of dates

        Functions similarly to np.arange, but returns date objects

        Example usage:

        >>> date.arange(2020, 2025)
            array([<2020-01-01>, <2021-01-01>, <2022-01-01>, <2023-01-01>,
                   <2024-01-01>, <2025-01-01>], dtype=object)

        Args:
                    start (float/ss.date/ss.dur): Lower bound - can be a date or a numerical year
        stop (float/ss.date/ss.dur): Upper bound - can be a date or a numerical year
        step (float/ss.dur): Assumes 1 calendar year steps by default
            inclusive (bool): Whether to include "stop" in the output
            day_round (bool): Whether to round to the nearest day (by default, True if step > 1 day)
            allow_zero (bool): if True, allow a year 0 by creating a datedur instead; if False, raise an exception; if None, give a warning

        Returns:
            An array of date instances
        """
        # Handle the special (but common) case of start < 1
        if (sc.isnumber(start) or isinstance(start, ss.datedur)) and start < 1.0:
            date_type = ss.datedur
            kw = {}
        else:
            date_type = cls
            kw = dict(day_round=day_round, allow_zero=allow_zero)

        # Convert this first
        if isinstance(step, ss.dur) and not isinstance(step, ss.datedur):
            if step.value == int(step.value) or isinstance(step, (ss.days, ss.weeks)): # e.g. ss.days(2), or fixed-length units like ss.days(0.25), but not ss.years(0.1)
                step = ss.datedur(step) # TODO: check if this does exact rather than to-years-and-back unit conversion
            else:
                step = step.years # Don't try to convert e.g. ss.years(0.1) to a datedur, you get rounding errors
        elif isinstance(step, str):
            dur_class = get_dur_class(step)
            step = ss.datedur(dur_class(1)) # e.g. ss.datedur(ss.years(1.0))

        # For handling floating point issues
        atol = 0.5 / factors.years.days # It's close if it's within half a day
        def within_day(y1, y2):
            return np.isclose(y1, y2, rtol=0, atol=atol)

        # We're creating dates from datedurs, step exactly
        if isinstance(step, ss.datedur):
            if not isinstance(start, date):
                start = date_type(start, **kw)
            if not isinstance(stop, date):
                stop = date_type(stop, **kw)

            atol = min(atol, step.years/2) # For sub-daily steps, it's only close if it's within half a step
            compare = (lambda t: t < stop or within_day(t.years, stop.years)) if inclusive else (lambda t: t < stop)
            kwds = step.value.kwds
            if not kwds.get('years') and not kwds.get('months'): # A step of fixed length, so the dates can be made all at once
                delta = pd.Timedelta(**{k:v for k,v in kwds.items() if k in ['weeks', 'days', 'hours', 'minutes', 'seconds']})
                first = start.to_pandas()
                n = max(int((stop.to_pandas() - first)/delta) + 2, 0) # Up to two past the stop, which are then checked as below
                tvec = [cls._reset_class(stamp) for stamp in pd.date_range(first, periods=n, freq=delta)]
                while tvec and not compare(tvec[-1]):
                    tvec.pop()
                return DateArray(tvec)

            tvec = []
            t = start
            while compare(t):
                tvec.append(t)
                t += step
            return DateArray(tvec)

        # We're converting them from float years, do it approximately
        elif sc.isnumber(step):
            day_round = sc.ifelse(day_round, not(cls.subdaily(step)))
            start = start.years if isinstance(start, (date, ss.dur)) else start
            stop = stop.years if isinstance(stop, (date, ss.dur)) else stop
            n_steps = (stop - start) / step
            if not n_steps.is_integer():
                rounded_steps = round(n_steps)
                rounded_stop = start + rounded_steps * step
                if within_day(stop, rounded_stop):
                    stop = rounded_stop
            arr = sc.inclusiverange(start, stop, step) if inclusive else np.arange(start, stop, step)
            return cls.from_array(arr, day_round=day_round, allow_zero=allow_zero, date_type=date_type)
        else:
            errormsg = f'Cannot construct date range from {start = }, {stop = }, and {step = }. Expecting ss.date(), ss.dur(), or numbers as inputs.'
            raise TypeError(errormsg)

    def to_json(self):
        """ Returns a JSON representation of the date """
        out = {'ss.date': str(self)}
        return out


class datedur(dur):
    """ Date based duration e.g., if requiring a week to be 7 calendar days later """
    base = 'years' # datedur uses 'years' as the default unit for conversion
    timepar_type = 'dur'
    timepar_subtype = 'datedur'

    def __init__(self, *args, **kwargs):
        """
        Create a date-based duration

        Supported arguments are
            - Single positional argument
                - A float number of years
                - A pd.DateOffset
                - A dateutil.relativedelta
                - A dur instance
            - Keyword arguments
                - Argument names that match time periods in `ss.time.factors` e.g., 'years', 'months'
                  supporting floating point values. If the value is not an integer, the values
                  will be cascaded to smaller units using `ss.time.factors`

        If no arguments are specified, a datedur with zero duration will be produced

        Args:
            args:
            kwargs:
        """
        super().__init__()
        if args:
            assert not kwargs, 'datedur must be instantiated with only 1 arg (which is in years), or keyword arguments.'
            assert len(args) == 1, f'datedur must be instantiated with only 1 arg (which is in years), or keyword arguments. {len(args)} args were given.'
            arg = args[0]
            if isinstance(arg, pd.DateOffset):
                self.value = self.round_duration(arg)
            elif isinstance(arg, datedur):
                self.value = sc.dcp(arg.value) # pd.DateOffset is immutable so this should be OK
            elif isinstance(arg, dur):
                self.value = self.round_duration(years=arg.years)
            elif sc.isnumber(arg):
                self.value = self.round_duration(years=arg)
            elif isinstance(arg, pd.Timedelta):
                self.value = self.round_duration(days=arg.days)
            elif isinstance(arg, drd.relativedelta):
                self.value = pd.DateOffset(years=arg.years, months=arg.months, days=arg.days, hours=arg.hours, minutes=arg.minutes, seconds=arg.seconds)
            else:
                errormsg = f'Unsupported input {args}.\nExpecting number, ss.dur, ss.datedur, pd.DateOffset, or dateutil.relativedelta'
                raise TypeError(errormsg)
        else:
            if 'value' in kwargs:
                self.value = self.round_duration(years=kwargs.pop('value'))
            else:
                self.value = self.round_duration(kwargs)

    def __repr__(self):
        if self.years == 0:
            return 'datedur(0)'
        else:
            vals = self.to_array().astype(float)
            time_portion = vals[4:]
            time_str = ':'.join(f'{np.round(v,1):02g}' for v in time_portion[:3])
            return 'datedur(' +  ', '.join([f'{k}={int(v)}' for k, v in zip(self.factor_keys[:4], vals[:4]) if v!=0]) + (f', +{time_str}' if time_str != '00:00:00' else '') + ')' # Sorry

    @classmethod
    def _as_array(cls, dateoffset, *args, **kwargs):
        """
        Return array representation of a pd.DateOffset

        In the array representation, each element corresponds to one of the time units
        in `ss.time.factors` e.g., a 1 year, 2 week duration would be [1, 0, 2, 0, 0, 0, 0, 0]

        Args:
            dateoffset: A pd.DateOffset object

        Returns:
            A numpy array with as many elements as `ss.time.factors`
        """
        return np.array([dateoffset.kwds.get(k, 0) for k in cls.factor_keys], *args, **kwargs)

    @classmethod
    def _as_args(cls, x):
        """
        Return a dictionary representation of a DateOffset or DateOffset array

        This function takes in either a pd.DateOffset, or the result of `_as_array`, and returns
        a dictionary with the same keys as `ss.time.factors` and the values from the input. The
        output of this function can be passed to the `dur()` constructor to create a new datedur object.

        Args:
            x: A pd.DateOffset or an array with the same number of elements as `ss.time.factors`

        Returns:
            Dictionary with keys from `ss.time.factors` and values from the input
        """
        if isinstance(x, pd.DateOffset):
            x = cls._as_array(x)
        return {k: v for k, v in zip(cls.factor_keys, x)}

    def _compare_args(self, other):
        """ Need to handle numbers differently, since self.value is not a float """
        if isinstance(other, (ss.dur, ss.date)):
            return self.years, other.years, True
        elif sc.isnumber(other) or isinstance(other, np.ndarray):
            return self.years, other, True # self.years instead of self.value is the difference with ss.dur
        else:
            return None, None, False

    def to_array(self, *args, **kwargs):
        """ Convert to a Numpy array (NB, different than to_numpy() which converts to fractional years """
        return self._as_array(self.value, *args, **kwargs)

    def to_dict(self):
        """ Convert to a dictionary """
        return self._as_args(self.to_array())

    def to_dur(self):
        """ Convert to the smallest non-zero value, e.g. ss.datedur(years=1, days=10).to_dur() = ss.days(375) """
        unit = None
        for k in valid_bases:
            v = self.value.kwds.get(k, 0)
            if v != 0:
                unit = k
        if unit is not None:
            value = self.to(unit)
            dur_class = get_dur_class(unit)
            return dur_class(value)
        else:
            return ss.years(0)

    def replace(self, value):
        """
        Return an `ss.dur` with the supplied value; see `TimePar.replace()`

        Unlike other timepars, a `datedur` stores its value as a date offset rather than
        a number, so it can't simply be copied. Instead, the finest unit the `datedur`
        was specified with is used, e.g. `ss.datedur(days=10).replace(20)` is `ss.days(20)`.
        """
        return self.to_dur().replace(value)

    def to(self, unit):
        """
        Return approximate conversion into the provided quantity

        This allows interoperability with years objects (which would typically be expected to
        occur if module parameters have been entered with `datedur` durations, but the simulation
        timestep is in `years` units).

        The conversion is based on `ss.time.factors` which defines the conversion from each time unit
        to the next.

        Args:
            unit (str): the unit to convert to: years, months, weeks, or days

        Returns:
            A float representing the duration in that unit
        """
        return sum([self.value.kwds.get(k, 0)/v for k, v in ss.time.factors[unit].items()])

    @property
    def years(self):
        """ Shortcut to datedur.to('years') """
        return self.to('years')

    @property
    def months(self):
        """ Shortcut to datedur.to('months') """
        return self.to('months')

    @property
    def weeks(self):
        """ Shortcut to datedur.to('weeks') """
        return self.to('weeks')

    @property
    def days(self):
        """ Shortcut to datedur.to('days') """
        return self.to('days')

    @classmethod
    def round_duration(cls, vals=None, **kwargs):
        """
        Round a dictionary of duration values by overflowing remainders

        The input can be
            - A numpy array of length `ss.time.factors` containing values in key order
            - A pd.DateOffset instance
            - A dictionary with keys from `ss.time.factors`

        The output will be a pd.DateOffset with integer values, where non-integer values
        have been handled by overflow using the factors in `ss.time.factors`. For example, 2.5 weeks
        would first become 2 weeks and 0.5*7 = 3.5 days, and then become 3 days + 0.5*24 = 12 hours.

        Negative values are supported - -1.5 weeks for example will become (-1w, -3d, -12h)

        Returns:
            A pd.DateOffset
        """
        if isinstance(vals, np.ndarray): # Main use case: convert from an array
            d = sc.objdict({k:v for k,v in zip(cls.factor_keys, vals)})
        else:
            d = sc.objdict.fromkeys(cls.factor_keys, 0)
            if isinstance(vals, pd.DateOffset):
                d.update(vals.kwds)
            elif isinstance(vals, dict):
                d.update(vals)
            elif vals is not None:
                errormsg = f'Could not interpret values {vals}; expecting, array, pd.DateOffset, or dict'
                raise TypeError(errormsg)
            d = sc.odict(sc.mergedicts(d, kwargs))

        if all([isinstance(val, int) for val in d.values()]):
            if isinstance(vals, pd.DateOffset): return vals  # pd.DateOffset is immutable so this should be OK
            return pd.DateOffset(**d)

        for i in range(len(cls.factors)-1):
            remainder, div = np.modf(d[i])
            d[i] = int(div)
            d[i+1] += remainder * cls.factor_vals[i+1]/cls.factor_vals[i]
        d[-1] = round(d[-1])

        return pd.DateOffset(**d)

    def str(self):
        # Friendly representation e.g., 'day', '1 year, 2 months, 1 day'
        vals = self.to_array()

        # If we have only one nonzero value, return 'years', 'months', etc.
        if np.count_nonzero(vals == 1) == 1:
            for unit, val in zip(self.factor_keys, vals):
                if val == 1:
                    return unit[:-1]

        strs = [f'{v} {k[:-1] if abs(v) == 1 else k}' for k, v in zip(self.factor_keys, vals) if v != 0]
        return ', '.join(strs)

    def scale(self, dateoffset, scale):
        """
        Scale a pd.DateOffset by a factor

        This function will automatically cascade remainders to finer units using `ss.time.factors` so for
        example 2.5 weeks would first become 2 weeks and 0.5*7 = 3.5 days,
        and then become 3 days + 0.5*24 = 12 hours.

        Args:
            dateoffset: A pd.DateOffset instance
            scale: A float scaling factor (must be positive)

        Returns:
            A pd.DateOffset instance scaled by the requested amount
        """
        return self.round_duration(self._as_array(dateoffset) * scale)

    def __pow__(self, other):
        raise Exception('Cannot multiply a duration by a duration')

    def __mul__(self, other):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, dur):
            wrong_dimension(self, '*', other)
        return self._new(self.scale(self.value, other))

    def __truediv__(self, other):
        if isinstance(other, datedur):
            # We need to convert both dates into a common timebase to do the division
            # We *can* just do self.years/other.years, however this can cause numerical precision losses
            # yet at the same time is not necessary, because the common timebase depends on the durations.
            # For example, datedur(weeks=1)/datedur(days=1) should return 7, but we get 6.9999999 if we convert
            # both to years. Instead though, we can just convert days into weeks instead of days into years, and
            # then divide 1 week by (1/7) days.
            # return self.years/other.years
            self_array = self.to_array()
            other_array = other.to_array()
            unit = np.nonzero((self_array != 0) | (other_array != 0))[0].max() # Finest nonzero unit, e.g. days
            a = 0
            b = 0
            for i, v in enumerate(self.factor_vals):
                a += self_array[i] * self.factor_vals[unit] / v
                b += other_array[i] * self.factor_vals[unit] / v
            return a/b
        elif isinstance(other, dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            wrong_dimension(self, '/', other)
        return self._new(self.scale(self.value, 1/other))

    def __add__(self, other):
        if isinstance(other, date):
            return other + self.value
        elif isinstance(other, datedur):
            return self._new(**self._as_args(self.to_array() + self._as_array(other.value)), other=other)
        elif isinstance(other, pd.DateOffset):
            return self._new(**self._as_args(self.to_array() + self._as_array(other)))
        elif isinstance(other, dur):
            kwargs = {k: v for k, v in zip(self.factor_keys, self.to_array())}
            kwargs['years'] += other.years
            return self._new(**kwargs, other=other)
        elif isinstance(other, DateArray):
            return DateArray(np.vectorize(self.__add__)(other))
        else:
            errormsg = f'For datedur, it is only possible to add/subtract dates, dur objects, or pd.DateOffset objects, not {type(other)}'
            raise TypeError(errormsg)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __rsub__(self, other):
        return other.__add__(-1*self)

    def __abs__(self):
        # TODO: think about whether this is sufficiently accurate
        return abs(self.years)
