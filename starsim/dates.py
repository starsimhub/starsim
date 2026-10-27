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

        The same as `dates - start` (which gives `ss.days`), but as plain numbers, all at once.

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
        allow_zero (bool): if True, allow a year 0 by creating a duration (ss.years) instead; if False, raise an exception; if None, give a warning
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
        string = f'{self.year:04d}-{self.month:02d}-{self.day:02d}'

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
            allow_zero (bool): whether to allow year 0 (if so, return ss.years instead)

        Examples:
            ```python
            ss.date.from_year(2020) # Returns <2020-01-01>
            ss.date.from_year(2024.75) # Returns <2024-10-01>
            ```
        """
        if year < 1:
            warnmsg = f'Dates with years < 1 are not valid ({year = }); returning ss.years instead'
            if allow_zero is False:
                raise ValueError(warnmsg)
            elif allow_zero is None:
                ss.warn(warnmsg)
            return ss.years(year)
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
        elif isinstance(other, (ss.date, pd.Timestamp, dt.date, dt.datetime)): # The time between two dates, in days
            return ss.days((self.to_numpy() - pd.Timestamp(other).to_numpy())/np.timedelta64(1, 'D'))
        else:
            errormsg = f'Attempted to subtract "{other}" ({type(other)}) from a date, which is not supported. Only durations can be subtracted from dates e.g., "ss.years({other})" or "ss.days({other})"'
            raise TypeError(errormsg)

    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return other.__class__([o-self for o in other]) # TODO: is this right?
        elif isinstance(other, ss.dur):
            # As this is __rsub__, this corresponds to `other - self` hence this operation is not permitted
            raise TypeError("Cannot subtract a date from a duration")
        elif isinstance(other, (ss.date, dt.date, dt.datetime)): # As date - date
            return ss.date(other) - self
        else:
            errormsg = f'Attempted to subtract "{self}" from "{other}" ({type(other)}), which is not supported. Only dates can be subtracted from dates e.g., datetime.date(2020, 1, 1) - ss.date(2010), which will return ss.days.'
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
            allow_zero (bool): if True, allow a year 0 by creating a duration (ss.years) instead; if False, raise an exception; if None, give a warning
            date_type (type): Optionally convert to a class other than `ss.date` (e.g. `ss.years`)

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
            kwargs = dict(day_round=day_round, allow_zero=allow_zero) # These arguments are only valid for ss.date, not a duration
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
            allow_zero (bool): if True, allow a year 0 by creating a duration (ss.years) instead; if False, raise an exception; if None, give a warning

        Returns:
            An array of date instances
        """
        # Handle the special (but common) case of start < 1, which is a time since 0 rather than a date
        if (sc.isnumber(start) or isinstance(start, ss.datedur)) and start < 1.0:
            date_type = ss.years
            kw = {}
        else:
            date_type = cls
            kw = dict(day_round=day_round, allow_zero=allow_zero)

        # Convert this first: a step of days or weeks has a fixed length, one of whole months or years follows the calendar, and any other is in years
        if isinstance(step, str):
            step = get_dur_class(step)(1) # e.g. ss.years(1)
        if isinstance(step, (ss.days, ss.weeks)):
            step = pd.Timedelta(days=step.days)
        elif isinstance(step, ss.dur) and not isinstance(step, ss.datedur):
            step = ss.datedur(step) if float(step.value).is_integer() else step.years # Not e.g. ss.years(0.1) as a calendar duration

        # For handling floating point issues
        atol = 0.5 / factors.years.days # It's close if it's within half a day
        def within_day(y1, y2):
            return np.isclose(y1, y2, rtol=0, atol=atol)

        # We're creating dates from calendar steps or steps of fixed length, step exactly
        if isinstance(step, (ss.datedur, pd.Timedelta)):
            if not isinstance(start, date):
                start = date_type(start, **kw)
            if not isinstance(stop, date):
                stop = date_type(stop, **kw)

            if isinstance(step, ss.datedur) and not step.kwds.get('years') and not step.kwds.get('months'): # Calendar days and weeks have a fixed length
                step = pd.Timedelta(days=step.days)
            step_years = step/pd.Timedelta(days=factors.years.days) if isinstance(step, pd.Timedelta) else step.years
            atol = min(atol, step_years/2) # For sub-daily steps, it's only close if it's within half a step
            compare = (lambda t: t < stop or within_day(t.years, stop.years)) if inclusive else (lambda t: t < stop)
            if isinstance(step, pd.Timedelta) and isinstance(start, date): # A step of fixed length, so the dates can be made all at once
                delta = step
                first = start.to_pandas()
                n = max(int((stop.to_pandas() - first)/delta) + 2, 0) # Up to two past the stop, which are then checked as below
                tvec = [cls._reset_class(stamp) for stamp in pd.date_range(first, periods=n, freq=delta)]
                while tvec and not compare(tvec[-1]):
                    tvec.pop()
                return DateArray(tvec)

            if isinstance(step, pd.Timedelta): # A time since 0 rather than a date, stepped in days
                step = ss.days(step/pd.Timedelta(days=1))
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
    """
    A calendar duration, e.g. `ss.datedur(months=1)`: a thin wrapper of `pd.DateOffset`

    Added to a date, it follows the calendar, so a month from 31 January is the end of February and
    a year from 1 March 2020 is 1 March 2021; `dt=ss.datedur(months=1)` steps a sim by calendar months.
    It takes whole numbers of years, months, weeks, and days (as keywords), or one of: a `pd.DateOffset`,
    a `dateutil.relativedelta`, a number of years, or a duration in one of these units (e.g. `ss.days(7)`).
    As a number (`.years`, `.days`, or divided by another duration) it is approximate, as other durations
    are: a year is 365 days, and a month a twelfth of a year. The time between two dates (`date - date`)
    is `ss.days`, not a `datedur`.

    Examples:
        ```python
        ss.date('2020-01-31') + ss.datedur(months=1) # 2020-02-29
        ss.datedur(years=1)/ss.datedur(months=1) # 12
        ```
    """
    base = 'years' # As a number, in years
    timepar_type = 'dur'
    timepar_subtype = 'datedur'
    units = ['years', 'months', 'weeks', 'days'] # The calendar units it takes
    twelfths = dict(years=365*12, months=365, weeks=7*12, days=12) # Each unit in twelfths of a day, so that one calendar duration divided by another is exact, e.g. a year by a month is 12

    def __init__(self, value=None, **kwargs):
        super().__init__()
        if isinstance(value, datedur):
            value = value.value
        elif isinstance(value, drd.relativedelta):
            kwargs = dict(years=value.years, months=value.months, days=value.days)
        elif isinstance(value, dur):
            kwargs = {value.base: value.value} # e.g. ss.days(7) is 7 calendar days
        elif sc.isnumber(value):
            kwargs = dict(years=value)
        elif value is not None and not isinstance(value, pd.DateOffset):
            errormsg = f'ss.datedur() takes whole numbers of {sc.strjoin(self.units)} (e.g. ss.datedur(months=1)), a pd.DateOffset, a number of years, or a duration, not {type(value).__name__}'
            raise TypeError(errormsg)
        if isinstance(value, pd.DateOffset) and (set(value.kwds) - set(self.units) or value.n != 1):
            errormsg = f'ss.datedur() takes whole numbers of {sc.strjoin(self.units)}, not {value}'
            raise ValueError(errormsg)
        if not isinstance(value, pd.DateOffset):
            bad = {k:v for k,v in kwargs.items() if k not in self.units or not float(v).is_integer()}
            if bad:
                errormsg = f'ss.datedur() takes whole numbers of {sc.strjoin(self.units)}, not {sc.strjoin([f"{k}={v}" for k,v in bad.items()])}; for a duration that is not on the calendar, use e.g. ss.years(0.5) or ss.days(1.5)'
                raise ValueError(errormsg)
            value = pd.DateOffset(**({k:int(v) for k,v in kwargs.items()} or dict(days=0))) # pd.DateOffset() with nothing is a day
        self.value = value

    @property
    def kwds(self):
        """ The number of each calendar unit, e.g. `dict(months=1)` """
        return {k:v for k,v in self.value.kwds.items() if k in self.units}

    def __repr__(self):
        kwds = {k:v for k,v in self.kwds.items() if v}
        return f'datedur({", ".join(f"{k}={v}" for k,v in kwds.items())})' if kwds else 'datedur(0)'

    def __str__(self):
        """ e.g. 'month', or '1 year, 2 months' """
        kwds = {k:v for k,v in self.kwds.items() if v}
        if list(kwds.values()) == [1]:
            return next(iter(kwds)).removesuffix('s')
        return ', '.join(f'{v} {k.removesuffix("s") if abs(v) == 1 else k}' for k,v in kwds.items()) or '0 days'

    def _compare_args(self, other):
        """ Need to handle numbers differently, since self.value is not a float """
        if isinstance(other, (ss.dur, ss.date)):
            return self.years, other.years, True
        elif sc.isnumber(other) or isinstance(other, np.ndarray):
            return self.years, other, True # self.years instead of self.value is the difference with ss.dur
        else:
            return None, None, False

    def to_dur(self):
        """ As a plain duration, in the finest unit it has, e.g. ss.datedur(years=1, days=10).to_dur() = ss.days(375) """
        unit = next((k for k in reversed(self.units) if self.kwds.get(k)), None)
        return get_dur_class(unit)(self.to(unit)) if unit is not None else ss.years(0)

    def replace(self, value):
        """
        Return an `ss.dur` with the supplied value; see `TimePar.replace()`

        A `datedur` stores a date offset rather than a number, so the finest unit it was given in is
        used, e.g. `ss.datedur(days=10).replace(20)` is `ss.days(20)`.
        """
        return self.to_dur().replace(value)

    def to(self, unit):
        """ The approximate number of a unit ('years', 'months', 'weeks', or 'days'), e.g. ss.datedur(months=6).to('years') is 0.5 """
        return sum(self.kwds.get(k, 0)/v for k,v in factors[unit].items())

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

    def _combine(self, kwds):
        """ A calendar duration with these numbers of each unit added to its own """
        return {k:self.kwds.get(k, 0) + kwds.get(k, 0) for k in self.units if k in self.kwds or k in kwds}

    def __pow__(self, other):
        raise Exception('Cannot multiply a duration by a duration')

    def __mul__(self, other):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, dur):
            wrong_dimension(self, '*', other)
        kwds = {k:v*other for k,v in self.kwds.items()}
        if all(float(v).is_integer() for v in kwds.values()): # Still whole calendar units, e.g. 52*ss.datedur(weeks=1)
            return self._new(**kwds)
        return self.to_dur()*other # Otherwise a plain duration, e.g. ss.datedur(months=1)*0.5 is ss.months(0.5)

    def __truediv__(self, other):
        if isinstance(other, datedur): # Exactly, in twelfths of a day, e.g. a week by a day is 7
            a, b = [sum(x.kwds.get(k, 0)*v for k,v in self.twelfths.items()) for x in [self, other]]
            return a/b
        elif isinstance(other, dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            wrong_dimension(self, '/', other)
        return self*(1/other)

    def __add__(self, other):
        if isinstance(other, date):
            return other + self.value
        elif isinstance(other, datedur):
            return self._new(**self._combine(other.kwds), other=other)
        elif isinstance(other, pd.DateOffset):
            return self._new(**self._combine(datedur(other).kwds))
        elif isinstance(other, DateArray):
            return DateArray(np.vectorize(self.__add__)(other))
        elif isinstance(other, dur):
            if other.is_scalar and other.base in self.units and float(other.value).is_integer(): # Whole calendar units, e.g. ss.years(1)
                return self._new(**self._combine({other.base:other.value}), other=other)
            return ss.years(self.years) + other # Otherwise, as a plain duration
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
        return abs(self.years)
