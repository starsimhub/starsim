"""
Functions and classes for handling time

Hierarchy of TimePars:
TimePar  # All time parameters
├── dur  # All durations, units of *time*
│   ├── days  # Duration with units of days
│   ├── weeks
│   ├── months
│   ├── years
│   └── datedur  # Calendar durations
└── Rate  # All rates, units of *per* (e.g. per time or per event)
    ├── per  # Probability rates over time (e.g., death rate per year)
    │   ├── perday
    │   ├── perweek
    │   ├── permonth
    │   └── peryear
    ├── prob  # Unitless probability (e.g., probability of death per infection)
    │   ├── probperday
    │   ├── probperweek
    │   ├── probpermonth
    │   └── probperyear
    └── freq  # Number of events (e.g., number of acts per year)
        ├── freqperday
        ├── freqperweek
        ├── freqpermonth
        └── freqperyear
"""
import numbers
import datetime as dt
import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss
ss_float = ss.dtypes.float

# General classes; specific classes are listed below
__all__ = ['DateArray', 'date', 'TimePar', 'dur', 'datedur', 'Rate', 'prob', 'per', 'freq']

def approx_compare(a, op='==', b=None, **kwargs):
    """ Floating-point issues are common working with dates, so allow approximate matching

    Specifically, this replaces e.g. "a <= b" with "a < b or np.isclose(a,b)"

    Args:
        a (int/float): the first value to compare
        op (str): the operation: ==, <, or >
        b (int/float): the second value to compare
        **kwargs (dict): passed to `np.isclose()`
    """
    close = np.isclose(a, b, **kwargs)
    if close:       return True
    elif op == '<': return a < b
    elif op == '>': return a > b
    else:
        errormsg = f'Unsupported operation "{op}", should be "==", "<", or ">"'
        raise ValueError(errormsg)

#%% Define dates
class DateArray(np.ndarray):
    """ Lightweight wrapper for an array of dates """
    def __new__(cls, arr=None):
        arr = sc.toarray(arr)
        if isinstance(arr, np.ndarray): # Shortcut to typical use case, where the input is an array
            return arr.view(cls)
        else:
            errormsg = f'Argument must be an array, not {type(arr)}'
            raise TypeError(errormsg)

    def __add__(self, other):
        cls = self.__class__
        if self.is_date:
            return cls([np.vectorize()])
            np.vectorize(ss.date.__add__)(other)

    def __radd__(self, other):
        return self.__add__(other)

    def is_(self, which):
        """ Checks if the DateArray is comprised of ss.date objects """
        if isinstance(which, type(type)):
            types = which
        elif isinstance(which, str):
            mapping = dict(date=ss.date, dur=ss.dur, datedur=ss.datedur, float=numbers.Number, scalar=numbers.Number)
            if which in mapping:
                types = mapping[which]
            else:
                errormsg = f'"which" must be "date", "dur", "datedur", "float", or a type, not "{which}"'
                raise ValueError(errormsg)

        if not len(self):
            ss.warn('Checking the type of a length-zero DateArray always returns False')
            return False

        # Do the check
        out = isinstance(self[0], types)
        return out

    @property
    def is_date(self):
        return self.is_('date')

    @property
    def is_dur(self):
        return self.is_('dur')

    @property
    def is_datedur(self):
        return self.is_('datedur')

    @property
    def is_float(self):
        return self.is_('float')

    @property
    def subdaily(self):
        """ Check if the array has sub-daily timesteps """
        try:
            delta = float(self[1] - self[0]) # float should return years
            return ss.date.subdaily(delta)
        except:
            return False

    @property
    def years(self):
        """ Represent the dates as floating point years """
        return np.array([d.years for d in self])

    def to_date(self, inplace=False, day_round=None, die=True):
        """ Convert to ss.date

        Args:
            inplace (bool): whether to modify in place
            round (bool): whether to round dates to the nearest day (otherwise, keep timestamp); if None, round if and only if the span of the first timestep is at least one day
            die (bool): if False, then fall back to float if conversion to date fails (e.g. year 0)
        """
        day_round = sc.ifelse(day_round, not(self.subdaily))
        try:
            vals = [ss.date(x, day_round=day_round, allow_zero=False) for x in self]
        except Exception as e:
            if die:
                raise e
            else:
                return self.to_float(inplace=inplace)

        if inplace:
            self[:] = vals
        else:
            return ss.DateArray(vals)

    def to_float(self, inplace=False):
        """ Convert to a float, returning a new DateArray unless inplace=True """
        vals = [float(x) for x in self]
        if inplace:
            self[:] = vals
        else:
            return ss.DateArray(vals)

    def to_human(self):
        """
        Return the most human-friendly (i.e. plotting-friendly) version of the dates,
        i.e. ss.date if possible, float otherwise
        """
        return self.to_date(die=False)

    def to_numpy(self):
        return self.to_array()

    def to_array(self, *args, **kwargs):
        """ Force conversion to an array """
        return np.array(self, *args, **kwargs)


class date(pd.Timestamp):
    """
    Define a point in time, based on `pd.Timestamp`

    Args:
        date (int/float/str/datetime): Any type of date input (ints and floats will be interpreted as years)
        allow_zero (bool): if True, allow a year 0 by creating a datedur instead; if False, raise an exception; if None, give a warning
        kwargs (dict): passed to pd.Timestamp()

    **Examples**:

        ss.date(2020) # Returns <2020-01-01>
        ss.date(year=2020) # Returns <2020-01-01>
        ss.date(year=2024.75) # Returns <2024-10-01>
        ss.date('2024-04-04') # Returns <2024-04-04>
        ss.date(year=2024, month=4, day=4) # Returns <2024-04-04>
    """
    def __new__(cls, *args, day_round=True, allow_zero=None, **kwargs):
        """ Check if a year was supplied, and preprocess it; complex due to pd.Timestamp implementation """
        single_year_arg = False
        if len(args) == 1:
            arg = args[0]
            if arg is None:
                return pd.Timestamp(None)
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
            return cls.from_year(arg, day_round=day_round, allow_zero=allow_zero)

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
    def from_year(cls, year, day_round=True, allow_zero=None):
        """
        Convert an int or float year to a date.

        Args:
            year (float): the year to round
            day_round (bool): whether to round to the nearest day
            allow_zero (bool): whether to allow year 0 (if so, return ss.datedur instead)

        **Examples**:

            ss.date.from_year(2020) # Returns <2020-01-01>
            ss.date.from_year(2024.75) # Returns <2024-10-01>
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
            if day_round:
                date = sc.yeartodate(year)
                timestamp = pd.Timestamp(date).round('d')
            else:
                year_start = pd.Timestamp(year=int(year), month=1, day=1)
                year_end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                timestamp = year_start + year % 1 * (year_end - year_start)
            return cls._reset_class(timestamp)

    def to_year(self):
        """ Convert a date to a floating-point year

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
        """ Round to a given interval (by default a day """
        timestamp = self.round(to)
        self._reset_class(timestamp)
        return

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
        elif isinstance(other, ss.datedur):
            return self._timestamp_add(other.value)
        elif isinstance(other, ss.dur):
            return date(self.to_year() + other.years)
        elif isinstance(other, pd.DateOffset):
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
            return ss.date(self.to_year() - other.years)
        elif isinstance(other, pd.DateOffset):
            return ss.date(self.to_pandas() - other)
        elif isinstance(other, (ss.date, dt.date, dt.datetime)):
            if not isinstance(other, pd.Timestamp):
                other = pd.Timestamp(other)
            return ss.datedur(self.to_pandas() - other)
        else:
            errormsg = f'Attempted to subtract "{other}" ({type(other)}) from a date, which is not supported. Only durations can be subtracted from dates e.g., "ss.years({other})" or "ss.days({other})"'
            raise TypeError(errormsg)

    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return other.__class__([o-self for o in other]) # TODO: is this right?
        elif isinstance(other, ss.datedur):
            return ss.date(other.value - self.to_pandas())
        elif isinstance(other, ss.dur):
            return ss.date(other.years - self.to_year())
        elif isinstance(other, (ss.date, dt.date, dt.datetime)):
            if not isinstance(other, pd.Timestamp):
                other = pd.Timestamp(other)
            return ss.datedur(other - self.to_pandas())
        else:
            errormsg = f'Attempted to subtract "{self}" from "{other}" ({type(other)}), which is not supported. Only dates can be subtracted from dates e.g., datetime.date(2020, 1, 1) - ss.date(2010), which will return ss.datedur.'
            raise TypeError(errormsg)

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
            kwargs = dict(day_round=day_round, allow_zero=allow_zero) # These arguments are only valid for ss.date, not ss.datedur
        return DateArray(np.vectorize(date_type)(array, **kwargs))

    @classmethod
    def arange(cls, start, stop, step=1.0, inclusive=True, day_round=None, allow_zero=True):
        """
        Construct an array of dates

        Functions similarly to np.arange, but returns date objects

        Example usage:

        >>> date.arange(2020, 2025)
            array([<2020.01.01>, <2021.01.01>, <2022.01.01>, <2023.01.01>,
                   <2024.01.01>, <2025.01.01>], dtype=object)

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
            if step.value == int(step.value): # e.g. ss.days(2) not ss.days(2.5)
                step = ss.datedur(step) # TODO: check if this does exact rather than to-years-and-back unit conversion
            else:
                step = step.years # Don't try to convert e.g. ss.years(0.1) to a datedur, you get rounding errors
                # start = float(start)
                # stop = float(stop)
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

            tvec = []
            t = start

            compare = (lambda t: t < stop or within_day(t.years, stop.years)) if inclusive else (lambda t: t < stop)
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

    @staticmethod
    def from_json(json):
        """ Reconstruct a date from a JSON; reverse of `to_json()` """
        if not (isinstance(json, dict) and len(json) == 1 and list(json.keys())[0] == 'ss.date'):
            errormsg = f'Expecting a dict with a single key "ss.date", not {json}'
            raise ValueError(errormsg)
        return date(json['ss.date'])



#%% Define base time units and conversion factors
valid_bases = ['years', 'months', 'weeks', 'days']

def normalize_unit(base): # TODO: use lookup on class_map instead of this manual approach of adding 's'
    """ Allow either e.g. 'year' or 'years' as input -- i.e. add an 's' """
    if isinstance(base, str):
        if base[-1] != 's':
            base = base+'s'
        if base not in valid_bases:
            errormsg = f'Invalid base {base}; valid bases are {sc.strjoin(valid_bases)}. Note that not all time units (e.g. seconds) are valid as bases.'
            raise ValueError(errormsg)
    elif isinstance(base, TimePar):
        base = base.base
    else:
        errormsg = f'Base must be str or ss.TimePar, not {base}'
        raise TypeError(errormsg)
    return base


class UnitFactors(sc.dictobj):
    """ Define conversion factors for a given unit, e.g. 'years' """
    def __init__(self, base_factors, unit):
        unit_items = base_factors.copy()
        for k,v in unit_items.items():
            unit_items[k] /= base_factors[unit]
        self.unit = unit
        self.unit_items = unit_items # Precompute for computational efficiency
        self.unit_values = np.array(list(unit_items.values()))
        self.unit_keys = list(unit_items.keys())
        return

    def __getattr__(self, attr):
        return self.unit_items[attr]

    def __getitem__(self, key):
        return self.unit_items[key]

    def __repr__(self):
        return f'factors.{self.unit}:\n{repr(sc.objdict(self.unit_items))}'

    def disp(self):
        return sc.pr(self)


class Factors(sc.dictobj):
    """ Define time unit conversion factors for all factors """
    def __init__(self):
        base_factors = sc.dictobj( # Use dictobj since slightly faster than objdict
            years   = 1, # Note 'years' instead of 'year', since referring to a quantity instead of a unit
            months  = 12,
            weeks   = 365/7, # If 52, then day/week conversion is incorrect
            days    = 365,
            hours   = 365*24,
            minutes = 365*24*60,
            seconds = 365*24*60*60,
        )
        for key in valid_bases:
            unit_factors = UnitFactors(base_factors, key)
            self[key] = unit_factors
        return

    def __repr__(self):
        string = ''
        for key in valid_bases:
            entry = repr(self[key])
            entry = sc.indent(text=entry, n=4).lstrip() # Don't indent the first line
            string += entry
        return string

    def __getattr__(self, attr):
        """ Be a little flexible, and then give helpful warnings """
        attr = normalize_unit(attr)
        try:
            return super().__getattribute__(attr)
        except AttributeError as e:
            errormsg = f'Attribute "{attr}" is not valid; choices are: {sc.strjoin(self.keys())}'
            raise AttributeError(errormsg) from e

# Preallocate for performance
factors = Factors()


#%% Define TimePars

class TimePar:
    """ Parent class for all TimePars -- dur, Rate, etc. """
    base = None # e.g. 'years', 'days', etc
    timepar_type = None # 'dur' or 'rate'
    timepar_subtype = None # e.g. 'datedur' or 'prob'

    def __new__(cls, *args, **kwargs):
        """Special handling for ss.Dist

        This is so e.g. ss.years(ss.normal(3)) is the same as ss.normal(3, unit=ss.years)
        """
        dist = cls._check_dist_arg(*args, **kwargs)
        return dist if dist is not None else super().__new__(cls)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._set_factors()
        return

    @classmethod
    def _check_dist_arg(cls, *args, **kwargs):
        """ Enables e.g. ss.years(ss.normal(...)) instead of ss.normal(..., unit=ss.years) """
        if len(args) and isinstance(args[0], ss.Dist):
            dist = args[0]
            dist.unit = cls
            return dist
        else:
            return None

    @classmethod
    def _set_factors(cls):
        """ Resets the default factors

        For example, with base = 'year', years=1.0 and days=1/365. With base='day',
        years=365.0 and days=1.0.
        """
        if cls.base is not None:
            cls.factors = factors[cls.base].unit_items
            cls.factor_keys = factors[cls.base].unit_keys
            cls.factor_vals = factors[cls.base].unit_values
        return

    def __setattrr__(self, attr, value):
        if object.__getattribute__(self, '_locked'):
            errormsg = f'Cannot set attributes of {self}; object is read-only'
            raise AttributeError(errormsg)
        else:
            super().__setattrr__(self, attr, value)
            return

    def __getitem__(self, index):
        """ For indexing and slicing, e.g. TimePar[inds] """
        if self.is_array:
            return self.__class__(self.value[index]) # NB: this assumes that unit, base, etc are set correctly
        else:
            if index == 0:
                return self
            else:
                errormsg = f'{type(self)} is a scalar'
                raise TypeError(errormsg)

    def __setitem__(self, index, value):
        """ For indexing and slicing, e.g. TimePar[inds] """
        if self.is_array:
            self.value[index] = value
            return
        else:
            if index == 0:
                self.value = value
                return
            else:
                errormsg = f'{type(self)} is a scalar'
                raise TypeError(errormsg)

    def __bool__(self):
        return True

    def __len__(self):
        if self.is_scalar:
            errormsg = f'{self} is a scalar' # This error is needed so NumPy doesn't try to iterate over the array
            raise TypeError(errormsg)
        else:
            return len(self.value)

    def __hash__(self):
        years = self.years
        if self.is_scalar:
            return years
        else:
            return tuple(years)

    def to_numpy(self):
        return self.to_array()

    def to_array(self, *args, **kwargs):
        """ Force conversion to an array """
        return sc.toarray(self.value, *args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ Disallow array operations by default, as they create arrays of objects (basically lists) """
        if len(inputs) == 1 and inputs[0] is self: # With a single input, operate on the value
            value = inputs[0].value
            return getattr(ufunc, method)(value, **kwargs)

        if len(inputs) == 2: # Handle most operations: addition, multiplication, etc.
            arg1,arg2 = inputs # Expect self,other as order of inputs, but can change
            is_tp1 = isinstance(arg1, ss.TimePar)
            is_tp2 = isinstance(arg2, ss.TimePar)
            use_left = is_tp1 or not is_tp2 # Use left unless arg1 isn't a timepar and arg2 is
            if use_left: # For operations where order doesn't matter
                tp_arg,o_arg = arg1,arg2 # Timepar argument and other argument (which may or may not be a timepar)
            else:
                tp_arg,o_arg = arg2,arg1

        # Commutative operations
        if   ufunc == np.add:      return tp_arg.__add__(o_arg)
        elif ufunc == np.multiply: return tp_arg.__mul__(o_arg)
        elif ufunc == np.equal:    return tp_arg.__eq__(o_arg)

        # Non-commutative
        elif ufunc == np.subtract:      return arg1.__sub__(arg2)     if use_left else arg2.__rsub__(arg1)
        elif ufunc == np.divide:        return arg1.__truediv__(arg2) if use_left else arg2.__rtruediv__(arg1)
        elif ufunc == np.less:          return arg1.__lt__(arg2)      if use_left else arg2.__gt__(arg1)
        elif ufunc == np.greater:       return arg1.__gt__(arg2)      if use_left else arg2.__lt__(arg1)
        elif ufunc == np.less_equal:    return arg1.__le__(arg2)      if use_left else arg2.__ge__(arg1)
        elif ufunc == np.greater_equal: return arg1.__ge__(arg2)      if use_left else arg2.__le__(arg1)

        # Fall back to standard ufunc, or raise an exception
        else:
            if any([isinstance(inp, TimePar) for inp in inputs]): # This is likely to be a confusing error, so let's go into detail about it
                errormsg = f'''
Cannot perform array operation "{ufunc}" on timepar {self}.

This can be caused by e.g. forgetting to multiply a rate by dt, or multiplying a scalar (rather than a rate) by dt. Another cause is operating on the timepar rather than its value.

Examples:
• np.exp(-waning*dt) will cause this error if e.g. waning=0.1 instead of ss.peryear(0.1)
• np.minimum(beta, 0.1) will cause this error if e.g. beta=ss.perday(0.2) instead use np.minimum(beta*dt, 0.1) or np.minimum(beta.value, 0.1) depending on what you intend.
'''
                raise ValueError(errormsg)
        return getattr(ufunc, method)(*inputs, **kwargs) # TODO: not sure if this would ever get called

    def array_mul_error(self, ufunc=None):
        ufuncstr = f"{ufunc} " if ufunc is not None else ''
        errormsg = f'Cannot perform array operation {ufuncstr}on {self}: this would create an array of {self.__class__} objects, which is very inefficient. '
        errormsg += 'Are you missing parentheses? If you convert to a unitless quantity by multiplying a rate and a duration, then you can multiply by an array. '
        errormsg += 'If you really want to do this, use an explicit loop instead.'
        raise TypeError(errormsg)

    @property
    def is_scalar(self):
        return not isinstance(self.value, np.ndarray)

    @property
    def is_array(self):
        return isinstance(self.value, np.ndarray)

    def to(self, unit):
        """ Convert this TimePar to one of a different class """
        unit = get_unit_class(self.timepar_type, unit)
        return unit(self)

    @classmethod
    def to_base(cls, other):
        """ Class method to convert another TimePar object to this TimePar's base units; in most cases """
        if isinstance(other, TimePar):
            return getattr(other, cls.base) # e.g. other.years
        else:
            errormsg = f'Can only convert TimePars to a different base, not {other}'
            raise AttributeError(errormsg)


class dur(TimePar):
    """
    Base class for durations

    Note: this class should not be used by the user directly; instead, use ss.years(),
    ss.days(), etc. These classes can be interconverted using `.to()`, e.g. `ss.years(3).to('days')`.
    """
    base = None
    factors = None
    timepar_type = 'dur'
    timepar_subtype = 'dur'

    def __new__(cls, value=None, base=None, **kwargs):
        """ Return the correct type based on the inputs """
        dist = cls._check_dist_arg(value)
        if dist is not None:
            return dist
        elif cls is dur: # The dur class itself, not a subclass: return the correct subclass and initialize it
            if kwargs:
                errormsg = f'Invalid arguments {kwargs} for ss.dur; valid arguments are "value" and "base". If you are trying to construct a datedur, call it directly.'
                raise ValueError(errormsg)
            if base is not None: # e.g. ss.dur(3, 'years'
                new_cls = get_dur_class(base)
            else: # e.g. ss.dur(3)
                new_cls = ss.years # Default
                if ss.options.warn_convert:
                    warnmsg = f'Creating {cls} with no unit specified is valid, but defaults to {new_cls}. Set ss.options.warn_convert = False to disable this warning.'
                    ss.warn(warnmsg)
            self = super().__new__(new_cls)
            self.__init__(value=value)
            return self
        else:
            return super().__new__(cls) # Otherwise, do default initialization

    def __init__(self, value=1, base=None):
        """
        Construct a value-based duration

        Args:
            value (float/`ss.dur`): the value to use
            base (str): the base unit, e.g. 'years'
        """
        super().__init__()
        if isinstance(value, dur):
            if self.base is not None:
                self.value = self.to_base(value)
            else:
                errormsg = f'Cannot convert value since base is None; {self} is not a valid timepar'
                raise ValueError(errormsg)
        elif sc.isnumber(value) or isinstance(value, np.ndarray) or isinstance(value, list):
            if isinstance(value, list):
                value = np.array(value)
            self.value = value
            if base is not None:
                if self.base == base:
                    pass # We're not actually changing the base, even though it's supplied
                elif self.base is None:
                    self.base = base
                    self._set_factors()
                else:
                    errormsg = f'Cannot change the base of `ss.dur` from {self.base} to {base}; use `dur.mutate()` instead'
                    raise AttributeError(errormsg)
        else:
            errormsg = f'Expecting scalar, array, or ss.dur as value, not {value}'
            raise TypeError(errormsg)
        return

    def __repr__(self):
        if self.is_scalar and self.value == 1:
            return f'{self.base.removesuffix("s")}()' # e.g. 'year()'
        else:
            if self.is_scalar:
                valstr = f'{self.value:g}'  # e.g. 2031.35
            else:
                valstr = f'{self.value}' # e.g. 2000, or an array
            return f'{self.base}({valstr})'

    # NB. Durations are considered to be equal if their year-equivalent duration is equal
    # That would mean that dur(years=1)==dur(1) returns True - probably less confusing than having it return False?
    def __hash__(self):
        return hash(self.years)

    def __float__(self):
        return float(self.value)

    @property
    def years(self):
        return self.value*self.factors.years # Needs to return float so matplotlib can plot it correctly

    @property
    def months(self):
        return self.value*self.factors.months # Needs to return float so matplotlib can plot it correctly

    @property
    def weeks(self):
        return self.value*self.factors.weeks # Needs to return float so matplotlib can plot it correctly

    @property
    def days(self):
        return self.value*self.factors.days # Needs to return float so matplotlib can plot it correctly

    def __add__(self, other):
        if isinstance(other, dur):
            return self.__class__(self.value + self.to_base(other))
        elif isinstance(other, date): # If adding to a date, convert to years
            return date.from_year(other.to_year() + self.years)
        elif isinstance(other, DateArray):
            return DateArray(np.vectorize(self.__add__)(other))
        else:
            return self.__class__(self.value + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, dur):
            out = self.__class__(self.value - self.to_base(other))
        elif isinstance(other, date):
            return date.from_year(other.to_year() - self.years)
        else:
            out = self.__class__(self.value - other)
            if sc.isnumber(out) and out < 0:
                warnmsg = f'Subtracting {self} and {other} yields {out}. Durations are rarely negative; are you sure this is intentional?'
                ss.warn(warnmsg)
        return out

    def __mul__(self, other):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, dur):
            raise Exception('Cannot multiply a duration by a duration')
        elif isinstance(other, date):
            raise Exception('Cannot multiply a duration by a date')
        return self.__class__(self.value*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            raise Exception('Cannot divide a duration by a rate')
        else:
            return self.__class__(self.value / other)

    def __neg__(self):
        return -1*self

    def _compare_args(self, other):
        if isinstance(other, (ss.dur, ss.date)):
            return self.years, other.years, True
        elif sc.isnumber(other) or isinstance(other, np.ndarray):
            return self.value, other, True
        else:
            return None, None, False

    def __lt__(self, other):
        a,b,impl = self._compare_args(other)
        return a < b if impl else NotImplemented

    def __gt__(self, other):
        a,b,impl = self._compare_args(other)
        return a > b if impl else NotImplemented

    def __le__(self, other):
        a,b,impl = self._compare_args(other)
        return a <= b if impl else NotImplemented

    def __ge__(self, other):
        a,b,impl = self._compare_args(other)
        return a >= b if impl else NotImplemented

    def __eq__(self, other):
        a,b,impl = self._compare_args(other)
        return a == b if impl else NotImplemented

    def __ne__(self, other):
        a,b,impl = self._compare_args(other)
        return a != b if impl else NotImplemented

    def __rtruediv__(self, other):
        # If a dur is divided by a dur then we will call __truediv__
        # If a float is divided by a dur, then we should return a rate
        # If a rate is divided by a dur, then we will call Rate.__truediv__
        # We also need divide the duration by the numerator when calculating the rate
        if self.years == 0:
            raise ZeroDivisionError('Cannot divide by a duration of zero') # TODO: consider Rate(dur(np.inf))
        elif (sc.isnumber(other) and other == 0) or isinstance(other, dur) and other.years == 0:
            return ss.freq(0)
        else:
            return ss.freq(other, self)

    def __abs__(self):
        return self.__class__(abs(self.value))

    @classmethod
    def arange(cls, start, stop, step, inclusive=True): # TODO: remove? ss.dur(arr) is preferable to array(ss.dur) for performance
        """
        Construct an array of dur instances

        For this function, the start, stop, and step must ALL be specified, and they must
        all be dur instances. Mixing dur types (years and datedur) is permitted.

        Args:
            start (ss.dur): Starting point, e.g., ss.years(0)
            stop (ss.dur): Ending point, e.g. ss.years(20)
            step (ss.dur): Step size, e.g. ss.years(2)
        """
        args = [start, stop, step]
        if not all([isinstance(arg, ss.dur) for arg in args]):
            errormsg = f'All inputs must be ss.dur, not {args}'
            raise TypeError(errormsg)

        tvec = []
        t = start
        compare = (lambda t: t <= stop) if inclusive else (lambda t: t < stop)
        while compare(t):
            tvec.append(t)
            t += step
        out = np.empty(len(tvec), dtype=object) # To stop it converting the array to 2D
        out[:] = tvec
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
            else:
                errormsg = f'Unsupported input {args}.\nExpecting number, ss.dur, ss.datedur, or pd.DateOffset'
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

    def __float__(self):
        return float(self.years)

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

    def to_numpy(self):
        # TODO: This is a quick fix to get plotting to work, but this should do something more sensible
        return self.years

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
        return sum([self.value.kwds.get(k, 0)/v for k, v in ss.time.factors[unit].unit_items.items()])

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

    @property
    def is_variable(self):
        if self.value.kwds.get('years',0) or self.value.kwds.get('months',0):
            return True
        else:
            return False

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
            raise Exception('Cannot multiply a duration by a duration')
        return self.__class__(self.scale(self.value, other))

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
            unit = np.argmax((self_array != 0) | (other_array != 0))
            a = 0
            b = 0
            for i, v in enumerate(self.factor_vals):
                if i < unit:
                    continue
                a += self_array[i] / v
                b += other_array[i] / v
            return a/b
        elif isinstance(other, dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            raise Exception('Cannot divide a duration by a rate')
        return self.__class__(self.scale(self.value, 1/other))

    def __add__(self, other):
        if isinstance(other, date):
            return other + self.value
        elif isinstance(other, datedur):
            return self.__class__(**self._as_args(self.to_array() + self._as_array(other.value)))
        elif isinstance(other, pd.DateOffset):
            return self.__class__(**self._as_args(self.to_array() + self._as_array(other)))
        elif isinstance(other, dur):
            kwargs = {k: v for k, v in zip(self.factor_keys, self.to_array())}
            kwargs['years'] += other.years
            return self.__class__(**kwargs)
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


class Rate(TimePar):
    """
    Store a value per unit time e.g., 2 per day
    - self.value - the numerator (e.g., 2) - a scalar float
    - self.unit - the denominator (e.g., 1 day) - a dur object
    """
    base = None
    factors = None
    timepar_type = 'rate'
    timepar_subtype = 'rate'

    def __init__(self, value, unit=None):
        if unit is not None and self.base is not None:
            unitstr = unit.base if isinstance(unit, ss.TimePar) else unit
            if unitstr != self.base:
                errormsg = f'Cannot specify incompatible unit "{unit}" for a rate when base "{self.base}" is already specified'
                raise ValueError(errormsg)

        if unit is None:
            if self.base is not None:
                dur_class = get_dur_class(self.base)
                self.unit = dur_class(1)
            else:
                if isinstance(self, ss.prob): # ss.prob is the only one for which not having a unit is fine
                    self.unit = None
                else:
                    self.unit = ss.years(1)
                    if ss.options.warn_convert:
                        warnmsg = f'Creating {type(self)} with no unit specified is valid, but defaults to {self.unit}. Set ss.options.warn_convert = False to disable this warning.'
                        ss.warn(warnmsg)
        elif isinstance(unit, ss.dur):
            self.unit = unit
        elif isinstance(unit, str):
            dur_class = get_dur_class(unit)
            self.unit = dur_class(1)
        else: # e.g. number
            self.unit = ss.years(unit) # Default of years

        # Convert from one rate to another
        if isinstance(value, ss.Rate):
            factor = self.unit/value.unit
            value = value.value*factor

        if isinstance(value, list):
            value = np.array(value)

        if not (sc.isnumber(value) or isinstance(value, np.ndarray)) and value is not None:
            errormsg = f'Value must be a scalar number or array, not {type(value)}'
            raise TypeError(errormsg)

        self.value = value
        self.default_dur = None # Used by modules to set dt
        self._set_base()
        return

    def _set_base(self):
        """ Rates can have either unit or base set; handle either """
        if self.base is None and isinstance(self.unit, ss.TimePar):
            self.base = self.unit.base
        elif self.unit is not None and self.base != self.unit.base: # Should not be possible, but check just in case
            errormsg = f'Inconsistent definition: base = {self.base} but unit = {self.unit.base}'
            raise ValueError(errormsg)
        return self

    def set_default_dur(self, dur):
        """ Set the default duration, e.g. module.dt, so .to_prob() can be used with no input """
        self.default_dur = dur
        return self

    def __repr__(self):
        name = self.__class__.__name__
        if name in ['Rate', 'prob', 'per', 'freq']: # If it's a base class, show the unit, e.g. Rate(3/years())
            valstr = f'{self.value:g}' if self.is_scalar else f'{self.value}' # Can't use g format with arrays
            unitstr = f'/{self.unit}' if self.unit is not None else '' # For unitless probabilities
            out = f'{name}({valstr}{unitstr})'
        else: # Otherwise, it's in the class name, e.g. peryear(3)
            if self.is_scalar:
                valstr = f'{self.value:g}'  # e.g. 2031.35
            else:
                valstr = f'{self.value}' # e.g. 2000, or an array
            out = f'{name}({valstr})'
        return out

    def __float__(self):
        return float(self.value)

    def __mul__(self, other):
        errormsg = 'ss.Rate() does not implement multiplication; this is implemented differently by its subclasses. '
        errormsg += warn_deprecation(output=True)
        raise NotImplementedError(errormsg)

    def __rmul__(self, other):
        """ e.g. 3*rate """
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

    def __add__(self, other):
        if isinstance(other, Rate):
            if self.timepar_subtype == other.timepar_subtype:
                return self.__class__(self.value+other*self.unit, self.unit)
            else:
                errormsg = f'Can only add rates with the same subtype (e.g., Rate+Rate, prob+prob); you added {self} + {other}'
                raise TypeError(errormsg)
        else:
            if sc.isnumber(other) or isinstance(other, np.ndarray):
                errormsg = f'Only rates can be added to rates, not {other}. This error most commonly occurs if the rate needs to be multiplied by `self.dt` to get a number of events per timestep.'
                raise TypeError(errormsg)
            else:
                errormsg = 'Only rates can be added to rates, not {other}'
                raise TypeError(errormsg)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if self.__class__ == other.__class__: # TODO: make more flexible, e.g. ss.perday(1) - ss.peryear(1) could work in theory
            return self.__class__(self.value-other*self.unit, self.unit)
        elif not isinstance(other, Rate):
            if sc.isnumber(other) or isinstance(other, np.ndarray):
                raise TypeError(f'Only rates can be subtracted from rates, not {other}. This error most commonly occurs if the rate needs to be multiplied by `self.t.dt` to get a number of events per timestep.')
            else:
                raise TypeError(f'Only rates can be subtracted from rates, not {other}')
        else:
            raise TypeError(f'Can only subtract rates of identical types (e.g., ss.per+ss.per, prob+prob), not {other}')

    def __eq__(self, other):
        if self.unit is None:
            try:
                assert other.unit == None
                self.value == other.value
                return True
            except:
                return False
        else:
            try:
                assert self.value == other.value/other.unit*self.unit
                return True
            except:
                return False

    def __truediv__(self, other):
        """ This is for <rate>/<other> """
        if isinstance(other, Rate):
            # Convert the other rate onto our dt, then divide the value
            if self.unit is None and other.unit is None:
                return self.value/other.value
            else:
                return self.value/(other.value/other.unit*self.unit)
        elif isinstance(other, dur):
            raise Exception('Cannot divide a rate by a duration')
        else:
            return self.__class__(self.value/other, self.unit)

    def __rtruediv__(self, other):
        """ This is for <other>/<rate>, e.g. if a float is divided by a rate """
        if self.unit is not None:
            return other * (self.unit/self.value)
        else:
            return other/self.value

    @property
    def _base_prob(self):
        """ Define the default conversion to probability; used in `to_prob()` """
        errormsg = 'ss.Rate() does not implement _base_prob; see ss.prob, ss.per, or ss.freq'
        raise NotImplementedError(errormsg)

    def to_prob(self, dur=None, scale=1.0):
        """ Convert from one time probability to another

        Args:
            dur (`ss.dur`): the duration over which to convert the probability to
            scale (float): an optional additional mutliplicative scale factor to incorporate in the calculation

        **Example**:

            p_month = ss.probpermonth(0.05)
            p_year = p_month.to_prob(ss.year) # Slightly less than 0.05*12
        """
        if dur is None:
            dur = self.default_dur # May also be None

        if dur is None:
            if scale == 1.0:
                return self._base_prob
            else:
                errormsg = f'Cannot convert to a probability with dur=None and non-unity {scale = }. Use simple multiplication instead, e.g. prob*2 rather than prob.to_prob(scale=2).'
                raise ValueError(errormsg)
        elif isinstance(dur, ss.dur): # Main use case # TODO: make array calculations robust to branches
            if sc.isnumber(self.rate) and self.rate == 0:
                return 0
            elif sc.isnumber(self.rate) and not np.isfinite(self.rate):
                return 1
            else: # Main use case
                if self.unit is None: # Final check before we proceed.
                    errormsg = f'You are trying to convert a unitless probability "{self}" to a different probability with units "{dur}". Are you using ss.prob() instead of e.g. ss.probperyear() or ss.peryear()?'
                    raise TypeError(errormsg)
                factor = (dur/self.unit)*scale # Main calculation step: cancel units and scale
                if sc.isnumber(factor) and factor == 1:
                    return self._base_prob # Avoid expensive calculation and precision issues
                return 1 - np.exp(-self.rate*factor) # Main use case
        elif sc.isnumber(dur) or isinstance(dur, np.ndarray):
            rate = self.rate*dur*scale # Scale the rate rather than the value
            rate_kw = dict(rate=rate) if isinstance(self, ss.prob) else dict(value=rate)
            out = self.__class__(unit=self.unit, **rate_kw) # e.g. ss.prob(unit=ss.year, rate=0.1) or ss.per(unit=ss.year, value=0.1)
            return out
        else:
            errormsg = f'Cannot multiply {type(self)} by {type(dur)}: expecting ss.dur or scalar'
            raise TypeError(errormsg)

    def to_events(self, dur=None):
        """ Simple multiplication: calculate the number of events over the time period """
        if dur is None:
            dur = self.default_dur # May also be None

        if isinstance(dur, ss.dur):
            return self.value*(dur/self.unit)
        elif dur is None and self.unit is None:
            return self.value
        else: # Handle float, array, etc.
            return self.__class__(self.value*dur, self.unit)


class prob(Rate):
    """
    `prob` represents the probability of an event occurring during a
    specified period of time.

    The class is designed to allow conversion of a probability from one
    duration to another through multiplication. However, the behavior of this
    conversion depends on the data type of the the object being multiplied.

    When multiplied by a duration (type ss.dur), the underlying constant rate is
    calculated as
        `rate = -np.log(1 - self.value)`.
    Then, the probability over the new duration is
        `p = 1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the new duration to the original duration.

    For example,
    >>> p = ss.prob(0.8, ss.years(1))
    indicates a 80% chance of an event occurring in one year.

    >>> p*ss.years(1)
    When multiplied by the original denominator, 1 year in this case, the
    probability remains unchanged, 80%.

    >>> p * ss.years(2)
    Multiplying `p` by `ss.years(2)` does not simply double the
    probability to 160% (which is not possible), but rather returns a new
    probability of 96% representing the chance of the event occurring at least
    once over the new duration of two years.

    However, the behavior is different when a `prob` object is multiplied
    by a scalar or array. In this case, the probability is simply scaled. This scaling
    may result in a value greater than 1, which is not valid. For example,
    >>> p * 2
    raises an AssertionError because the resulting probability (160%) exceeds 100%.

    Use `per` instead if `prob` if you would prefer to directly
    specify the instantaneous rate.
    """
    base = None # Can inherit, but clearer to specify
    timepar_type = 'rate'
    timepar_subtype = 'prob'

    def __init__(self, value=None, unit=None, rate=None):
        self._value = None
        self._rate = None
        if value is not None:
            if rate is not None:
                errormsg = f'You can supply value or rate, but not both ({value = } and {rate = })'
                raise ValueError(errormsg)
            try:
                assert 0 <= value <= 1, 'Value must be between 0 and 1'
            except Exception: # Something went wrong, let's figure it out -- either a value error, or value is an array instead of a scalar
                if sc.isnumber(value):
                    valstr = f'Value provided: {value}'
                    lt0 = value < 0
                    gt1 = value > 1
                else:
                    if isinstance(value, ss.TimePar):
                        value = value.value
                    else:
                        value = sc.toarray(value)
                    valstr = f'Values provided:\n{value}'
                    lt0 = np.any(value<0)
                    gt1 = np.any(value>1)

                if lt0:
                    errormsg = f'Negative values are not permitted for rates or probabilities. {valstr}'
                elif gt1:
                    errormsg = f'Values greater than 1 are not permitted for probabilities, please use ss.per() or ss.freq() instead. {valstr}'

                raise ValueError(errormsg)
        Rate.__init__(self, value, unit) # Can't use super() since potentially mutating the class
        if rate is not None:
            self.rate = rate
        return

    @property
    def value(self):
        return self._value

    @property
    def rate(self):
        return self._rate

    @value.setter
    def value(self, v):
        """ Set the rate as well as the value """
        if v is None:
            self._value = None
            self._rate = None
        else:
            if (sc.isnumber(v) and (v<0 or v>1)) or isinstance(v, np.ndarray) and (np.any(v<0) or np.any(v>1)):
                errormsg = f'Probabilities must be in [0, 1], not {v}'
                raise ValueError(errormsg)
            elif sc.isnumber(v) and v == 1: # Don't handle arrays, just raise the warning in the next step if it comes to it
                self._rate = np.inf
            else:
                self._rate = -np.log(1 - v) # Will raise a NumPy warning if it's an array with 1.0 values
            self._value = v # Store the raw value as well
        return

    @rate.setter
    def rate(self, rate):
        """ Set both the rate and the value """
        if sc.isnumber(v) and v<0:
            errormsg = f'Cannot set a negative rate/probability ({v})'
            raise ValueError(errormsg)
        self._rate = rate

        # Set the value
        self._value = 1.0 - np.exp(-self._rate) if np.isfinite(self._rate) else 1.0
        return

    @property
    def _base_prob(self):
        """ Used by to_prob() """
        return self.value

    def disp(self):
        return sc.pr(self)

    def __mul__(self, other):
        return self.to_prob(other)

    def to_prob(self, dur=None, scale=1.0):
        """ Convert from one time probability to another

        Args:
            dur (`ss.dur`): the duration over which to convert the probability to
            scale (float): an optional additional mutliplicative scale factor to incorporate in the calculation

        **Example**:

            p_month = ss.probpermonth(0.05)
            p_year = p_month.to_prob(ss.year) # Slightly less than 0.05*12
        """
        if dur is None:
            if scale == 1.0:
                return self.value
            else:
                errormsg = f'Cannot convert to a probability with dur=None and non-unity {scale = }. Use simple multiplication instead, e.g. prob*2 rather than prob.to_prob(scale=2).'
                raise ValueError(errormsg)
        elif isinstance(dur, np.ndarray):
            self.array_mul_error()
        elif isinstance(dur, ss.dur): # Main use case
            if self._rate == 0:
                return 0
            elif not np.isfinite(self._rate):
                return 1
            else: # Main use case
                if self.unit is None: # Final check before we proceed.
                    errormsg = f'You are trying to convert a unitless probability "{self}" to a different probability with units "{dur}". Are you using ss.prob() instead of e.g. ss.probperyear() or ss.peryear()?'
                    raise TypeError(errormsg)
                factor = (dur/self.unit)*scale # Main calculation step: cancel units and scale
                if factor == 1:
                    return self.value # Avoid expensive calculation and precision issues
                return 1 - np.exp(-self._rate*factor) # Main use case
        elif sc.isnumber(dur):
            rate = self._rate*dur*scale # Scale the rate rather than the value
            out = self.__class__(rate=rate, unit=self.unit) # Placeholder
            return out
        else:
            errormsg = f'Cannot multiply {type(self)} by {type(dur)}: expecting ss.dur or scalar'
            raise TypeError(errormsg)

    @classmethod
    def array_to_prob(cls, arr, dur, v=1):
        arr = sc.promotetoarray(arr)

        if isinstance(dur, np.ndarray):
            assert arr.shape == dur.shape, 'dur must be either a scalar, or the same size as arr'
            def to_prob(a, b, _v=v):
                return a.to_prob(b, v=_v)

            vectorize = np.vectorize(to_prob)
            return vectorize(arr, dur)

        elif isinstance(arr[0], prob):
            factor = np.array([dur / a.unit for a in arr])
            scaled_vals = np.array([a.value * v for a in arr])
            rate = - np.log(1 - scaled_vals)
            return 1-np.exp(-rate*factor)

        else: # Assume arr is an array of values, that would be the values of a prob with unit=ss.years(1)
            factor = dur / ss.years(1)
            scaled_vals = arr * v
            rate = - np.log(1 - scaled_vals)
            return 1 - np.exp(-rate * factor)

    def __truediv__(self, other):
        try:
            return self.__mul__(1.0/other)
        except Exception as e:
            errormsg = f'It is only valid to divide a probability by a quantity that can be inverted, e.g. a scalar, i.e. p/2 = p*(1/2). But this did not work with {other}.'
            raise ValueError(errormsg) from e

    def __rtruediv__(self, other):
        errormsg = f'You cannot divide anything by a probability ({self})'
        raise NotImplementedError(errormsg)


class per(Rate):
    """
    A `per` represents an instantaneous rate of an event occurring. Rates
    must be non-negative, but need not be less than 1.

    Through multiplication, rate can be modified or converted to a probability,
    depending on the data type of the object being multiplied.

    When a `per` is multiplied by a scalar or array, the rate is simply
    scaled. Such multiplication occurs frequently in epidemiological models,
    where the base rate is multiplied by "rate ratio" or "relative rate" to
    represent agents experiencing higher (multiplier > 1) or lower (multiplier <
    1) event rates.

    Alternatively, when a `per` is multiplied by a duration (type
    ss.dur), a probability is calculated. The conversion from rate to
    probability on multiplication by a duration is
        `1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the multiplied duration to the original
    period (denominator).

    For example, consider
    >>> p = ss.per(0.8, ss.years(1))
    When multiplied by a duration of 1 year, the calculated probability is
        `1 - np.exp(-0.8)`, which is approximately 55%.
    >>> p*ss.years(1)

    When multiplied by a scalar, the rate is simply scaled.
    >>> p*2

    The difference between `prob` and `per` is subtle, but important. `per` works directly
    with the instantaneous rate of an event occurring. In contrast, `prob` starts with a probability and a duration,
    and the underlying rate is calculated. On multiplication by a duration,
    * per: rate -> probability
    * prob: probability -> rate -> probability

    The behavior of both classes is depending on the data type of the object being multiplied.

    `ss.per` is identical to `ss.freq`, except by default multiplication returns
    a probability rather than a number of events. Use `ss.per` for probabilities
    (e.g., probability of death per year), and `ss.freq` for events (e.g., number of acts per year).
    """
    base = None # Can inherit, but clearer to specify
    timepar_type = 'rate'
    timepar_subtype = 'per'

    def __init__(self, value, unit=None):
        if sc.isnumber(value):
            assert value >= 0, f'Value must be >= 0, not {value}'
        return super().__init__(value, unit)

    @property
    def rate(self):
        """ Alias to self.value """
        return self.value

    @rate.setter
    def rate(self, rate):
        self.value = rate
        return

    @property
    def _base_prob(self):
        """ Used by to_prob() """
        return 1.0 - np.exp(-self.rate)

    def __mul__(self, other):
        return self.to_prob(other)

    def __truediv__(self, other):
        if isinstance(other, dur):
            raise NotImplementedError()
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        raise NotImplementedError()


class freq(Rate):
    """ Class for the number of events (rather than probability) in a specified period

    Identical to `ss.per`, except by default multiplication returns a number of events
    rather than a probability. Use `ss.freq` for events (e.g., number of acts per year)
    and `ss.per` for probabilities (e.g., probability of death per year).
    """
    base = None
    timepar_type = 'rate'
    timepar_subtype = 'freq'

    @property
    def rate(self):
        """ Alias to self.value """
        return self.value

    @rate.setter
    def rate(self, rate):
        self.value = rate
        return

    @property
    def _base_prob(self):
        """ Used by to_prob() """
        return 1.0 - np.exp(-self.rate)

    def __mul__(self, other):
        return self.to_events(other)


#%% Convenience classes
__all__ += ['years', 'months', 'weeks', 'days', 'year', 'month', 'week', 'day', # Durations
            'perday', 'perweek', 'permonth', 'peryear', # probs
            'probperday', 'probperweek', 'probpermonth', 'probperyear', # prob aliases
            'freqperday', 'freqperweek', 'freqpermonth', 'freqperyear'] # Rates

# Durations
class years(dur):  base = 'years'
class months(dur): base = 'months'
class weeks(dur):  base = 'weeks'
class days(dur):   base = 'days'

# Duration shortcuts
year = years(1)
month = months(1)
week = weeks(1)
day = days(1)
for obj in [year, month, week, day]:
    object.__setattr__(obj, '_locked', True) # Make immutable

# per
class perday(per):   base = 'days'
class perweek(per):  base = 'weeks'
class permonth(per): base = 'months'
class peryear(per):  base = 'years'

# prob
class probperday(prob):   base = 'days'
class probperweek(prob):  base = 'weeks'
class probpermonth(prob): base = 'months'
class probperyear(prob):  base = 'years'

# freq
class freqperday(freq):   base = 'days'
class freqperweek(freq):  base = 'weeks'
class freqpermonth(freq): base = 'months'
class freqperyear(freq):  base = 'years'

#%% Class mappings

# Define a mapping of all the options -- dictionary lookup is O(1) so it's OK to have a lot of keys
reverse_class_map = {
    days:   ['d', 'day', 'days', day, days],
    weeks:  ['w', 'week', 'weeks', week, weeks],
    months: ['m', 'month', 'months', month, months],
    years:  ['y', 'year', 'years', year, years],

    perday:   ['perday', 'probperday', perday],
    perweek:  ['perweek', 'probperweek', perweek],
    permonth: ['permonth', 'probpermonth', permonth],
    peryear:  ['peryear', 'probperyear', peryear],

    freqperday:   ['freqperday', freqperday],
    freqperweek:  ['freqperweek', freqperweek],
    freqpermonth: ['freqpermonth', freqpermonth],
    freqperyear:  ['freqperyear', freqperyear],
}

# Convert to the actual class map
class_map = sc.objdict(dur=sc.objdict(), rate=sc.objdict()) # TODO: make a class with an informative error message for invalid lookups (e.g. dur vs rate)
for v, keylist in reverse_class_map.items():
    for key in keylist:
        if issubclass(v, dur):
            class_map.dur[key] = v
        elif issubclass(v, Rate):
            class_map.rate[key] = v
        else:
            errormsg = f'Unexpected entry: {v}'
            raise TypeError(errormsg)
class_map.full = sc.mergedicts(class_map.dur, class_map.rate) # TODO: do we need this?

def get_unit_class(which, unit):
    """ Take a string or class and return the corresponding TimePar class """
    if isinstance(unit, type) and issubclass(unit, ss.TimePar):
        return unit
    elif isinstance(unit, ss.TimePar):
        if sc.isnumber(unit.value) and unit.value == 1:
            return unit.__class__
        else:
            errormsg = f'TimePar instances can only be used as base classes when the value is one, e.g. ss.years(1), but you supplied {unit}. '
            errormsg += 'Please convert to the equivalent base unit, e.g. instead of ss.Rate(value=4, unit=ss.weeks(2)), do ss.Rate(value=2, unit=ss.weeks).'
            raise ValueError(errormsg)
    elif isinstance(unit, str):
        this_map = class_map[which]
        if which == 'dur': # Only durations have multiple names
            unit = normalize_unit(unit)
        unit = this_map[unit]
        return unit
    elif unit is None:
        return None
    else:
        errormsg = f'Unit must be str (e.g. "years") or TimePar (e.g. ss.years), not "{unit}"'
        raise TypeError(errormsg)

def get_dur_class(unit):
    """ Helper function to get a Dur class """
    return get_unit_class('dur', unit)

def get_rate_class(unit):
    """ Helper function to get a Rate class """
    return get_unit_class('rate', unit)

def get_timepar_class(unit):
    """ Helper function to get any Timepar class (either Dur or Rate) """
    return get_unit_class('full', unit)

#%% Backwards compatibility functions

__all__ += ['rate', 'time_prob', 'rate_prob']

def warn_deprecation(old=None, value=None, unit=None, with_s=False, output=False):
    msg = 'The subclasses of ss.Rate() are: '
    msg += '\n• ss.per() (rate at which an event happens, e.g. mortality rate), '
    msg += '\n• ss.prob() (unitless probability, e.g. symptomatic infection), and '
    msg += '\n• ss.freq() (number of events per time, e.g. acts per year). '
    msg += '\nAlthough ss.freq() is the closest equivalent to ss.rate() in Starsim 2.0, '
    msg += 'in most cases (e.g. birth, death, or infection rates) you actually want ss.per().'
    if ss.options.warn_convert:
        unitstr = str(unit) if unit is not None else 'year'
        if with_s: unitstr = unitstr + 's'
        warnmsg = f'The Starsim v2 class ss.{old}() is deprecated. Please use an ss.Rate() subclass, e.g. ss.{unitstr}({value}) instead. '
        warnmsg += msg
        ss.warn(warnmsg)
    if output:
        return msg
    return

def rate(value, unit=None):
    """ Backwards compatibility function for Rate """
    warn_deprecation('rate', value, unit)
    return ss.freq(value, unit)

def time_prob(value, unit=None):
    """ Backwards compatibility function for prob """
    warn_deprecation('time_prob', value, unit)
    return ss.prob(value, unit)

def rate_prob(value, unit=None):
    """ Backwards compatibility function for per """
    warn_deprecation('rate_prob', value, unit)
    return ss.per(value, unit)
