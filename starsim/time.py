"""
Functions and classes for handling time

Hierarchy of TimePars:
TimePar  # All time parameters
├── Dur  # All durations, units of *time*
│   ├── days  # Duration with units of days
│   ├── weeks
│   ├── months
│   ├── years
│   └── DateDur  # Calendar durations
└── Rate  # All rates, units of *per time*
    ├── rateperday  # Number of events happening per day
    ├── rateperweek
    ├── ratepermonth
    ├── rateperyear
    ├── TimeProb  # Probability of an event happening in a given time
    │   ├── perday # Probability of an event happening in a day
    │   ├── perweek
    │   ├── permonth
    │   └── peryear
    └── RateProb  # Instantaneous probability of an event happening
"""
import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss

# General classes; specific classes are listed below
__all__ = ['date', 'TimePar', 'Dur', 'DateDur', 'Rate', 'TimeProb', 'RateProb']

#%% Base classes
class DateArray(np.ndarray):
    """ Lightweight wrapper for an array of dates """
    def __new__(cls, arr=None):
        if arr is None:
            arr = np.array([])
        if isinstance(arr, np.ndarray): # Shortcut to typical use case, where the input is an array
            return arr.view(cls)
        else:
            errormsg = f'Argument must be an array, not {type(arr)}'
            raise TypeError(errormsg)


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
            elif sc.isnumber(arg): # e.g. 2020
                single_year_arg = True
            elif isinstance(arg, ss.DateDur): # e.g. ss.DateDur(years=2020)
                kwargs.update(arg.to_dict())
            elif isinstance(arg, ss.Dur): # e.g. ss.years(2020)
                arg = arg.years
                single_year_arg = True

        year_kwarg = len(args) == 0 and len(kwargs) == 1 and 'year' in kwargs
        if single_year_arg:
            return cls.from_year(arg)
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
        elif isinstance(other, DateDur):
            return self._timestamp_add(other.value)
        elif isinstance(other, years):
            return date(self.to_year() + other.value)
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
            return date(self.to_pandas() - other.value)
        elif isinstance(other, years):
            return date(self.to_year() - other.value)
        elif isinstance(other, pd.DateOffset):
            return date(self.to_pandas() - other)
        elif isinstance(other, date):
            return years(self.years-other.years)
        else:
            errormsg = f'Unsupported type {type(other)}'
            raise TypeError(errormsg)

    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other) # TODO: check if this should be reversed

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
        return DateArray(np.vectorize(cls)(array))

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
        # Convert this first
        if isinstance(step, ss.Dur):
            step = DateDur(step)

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
            return DateArray(np.array(tvec))
        elif sc.isnumber(step):
            low = low.years if isinstance(low, date) else low
            high = high.years if isinstance(high, date) else high
            return cls.from_array(np.arange(low, high, step))
        else:
            errormsg = f'Cannot construct date range from {low = }, {high = }, and {step = }. Expecting ss.date(), ss.Dur(), or numbers as inputs.'
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



class UnitFactors(sc.dictobj):
    def __init__(self, base, unit):
        unit_items = base.copy()
        for k,v in unit_items.items():
            unit_items[k] /= base[f'{unit}s'] # e.g. 'year' -> 'years'
        self.unit = unit
        self.items = unit_items
        self.values = np.array(list(unit_items.values())) # Precompute for computational efficiency
        self.keys = list(unit_items.keys())
        return

    def __repr__(self):
        return f'{self.unit}:\n{repr(sc.objdict(self.items))}'

    def disp(self):
        return sc.pr(self)


class Factors(sc.dictobj):
    """ Define time unit conversion factors """
    unit_keys = ['year', 'month', 'week', 'day']

    def __init__(self):
        base = sc.dictobj( # Use dictobj since slightly faster than objdict
            years   = 1, # Note 'years' instead of 'year', since referring to a quantity instead of a unit
            months  = 12,
            weeks   = 365/7, # If 52, then day/week conversion is incorrect
            days    = 365,
            hours   = 365*24,
            minutes = 365*24*60,
            seconds = 365*24*60*60,
        )
        for key in self.unit_keys:
            unit_factors = UnitFactors(base, key)
            self[key] = unit_factors
        return

    def __repr__(self):
        string = ''
        for key in self.unit_keys:
            string += 'factors.'
            entry = repr(self[key])
            entry = sc.indent(text=entry, n=4).lstrip() # Don't indent the first line
            string += entry
        return string

# Preallocate for performance
factors = Factors()

class TimePar:
    """ Parent class for all TimePars -- Dur, Rate, etc. """
    base = None # e.g. 'year', 'day', etc
    timepar_type = None # 'dur' or 'rate'
    timepar_subtype = None # e.g. 'datedur' or 'timeprob'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._set_factors()
        return

    @classmethod
    def _set_factors(cls):
        if cls.base is not None:
            cls.basekey = cls.base + 's' # 'year' -> 'years'
            cls.factors = factors[cls.base].items
            cls.factor_keys = factors[cls.base].keys
            cls.factor_vals = factors[cls.base].values
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
        if isinstance(self.value, np.ndarray):
            return self.value[index]
        else:
            errormsg = f'Can only index {type(self)} if value is an array, not a scalar'
            raise TypeError(errormsg)

    def __setitem__(self, index, value):
        """ For indexing and slicing, e.g. TimePar[inds] """
        if isinstance(self.value, np.ndarray):
            self.value[index] = value
            return
        else:
            errormsg = f'Can only index {type(self)} if value is an array, not a scalar'
            raise TypeError(errormsg)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ Disallow array operations by default, as they create arrays of objects (basically lists) """
        a,b = inputs
        a_b = True
        if b is self: # Swap order if needed
            b,a = a,b
            a_b = True

        if   ufunc == np.add:      return self.__add__(b)
        elif ufunc == np.subtract: return self.__sub__(b) if a_b else self.__rsub__(b)
        elif ufunc == np.multiply: return self.__mul__(b)
        elif ufunc == np.divide:   return self.__truediv__(b) if a_b else self.__rtruediv__(b)
        else:
            self.array_mul_error(ufunc)
        return

    def array_mul_error(self, ufunc=None):
        ufuncstr = f"{ufunc} " if ufunc is not None else ''
        errormsg = f'Cannot perform array operation {ufuncstr}on {self}: this would create an array of {self.__class__} objects, which is very inefficient. '
        errormsg += 'Are you missing parentheses? If you convert to a unitless quantity by multiplying a rate and a duration, then you can multiply by an array. '
        errormsg += 'If you really want to do this, use an explicit loop instead.'
        raise TypeError(errormsg)

    def to(self, unit):
        """ Convert this TimePar to one of a different class """
        return class_map[self.timepar_type][unit](self)

    @classmethod
    def to_base(cls, other):
        """ Convert another TimePar object to this TimePar's base units """
        try:
            return getattr(other, cls.basekey) # e.g. other.years
        except AttributeError as e:
            errormsg = f'Cannot get property {cls.basekey} from object {other}. Is it a TimePar?'
            raise AttributeError(errormsg) from e

    def mutate(self, unit):
        """ Mutate a TimePar in place to a new base unit -- see `TimePar.to()` to return a new instance (much more common) """
        self.base = unit
        self._set_factors()
        self.__class__ = class_map[self.timepar_type][unit] # Mutate the class in place
        return self


class Dur(TimePar):
    """
    Base class for durations

    Note: this class should not be used by the user directly; instead, use ss.years(),
    ss.days(), etc.

    Note that although they are different classes, `ss.Dur` objects can be modified
    in place if needed via the `ss.Dur.mutate()` method.
    """
    base = None
    timepar_type = 'dur'
    timepar_subtype = 'dur'

    def __new__(cls, *args, **kwargs):
        # Return
        if cls is Dur:
            if args:
                if isinstance(args[0], (pd.DateOffset, DateDur)):
                    return super().__new__(DateDur)
                elif isinstance(args[0], years):
                    return super().__new__(years)
                else:
                    assert len(args) == 1, f'Dur must be instantiated with only 1 arg (which is in years), or keyword arguments. {len(args)} args were given.'
                    return super().__new__(years)
            else:
                return super().__new__(DateDur) # TODO: do not make the default, but needs new classes
        return super().__new__(cls)

    def __init__(self, value=1):
        """
        Construct a value-based duration

        Args:
            value (float/`ss.Dur`): the value to use
        """
        super().__init__()
        if isinstance(value, Dur):
            self.value = self.to_base(value)
        else:
            self.value = value

    def __repr__(self):
        if self.value == 1:
            return f'{self.base}()'
        else:
            return f'{self.basekey}({self.value})'

    # NB. Durations are considered to be equal if their year-equivalent duration is equal
    # That would mean that Dur(years=1)==Dur(1) returns True - probably less confusing than having it return False?
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
        return self.value*self.factors.day # Needs to return float so matplotlib can plot it correctly

    def to_numpy(self):
        return sc.toarray(self.value)

    def __add__(self, other):
        if isinstance(other, Dur):
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
        if isinstance(other, Dur):
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
        elif isinstance(other, Dur):
            raise Exception('Cannot multiply a duration by a duration')
        elif isinstance(other, date):
            raise Exception('Cannot multiply a duration by a date')
        return self.__class__(self.value*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            raise Exception('Cannot divide a duration by a rate')
        else:
            return self.__class__(self.value / other)

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

    def __rtruediv__(self, other):
        # If a Dur is divided by a Dur then we will call __truediv__
        # If a float is divided by a Dur, then we should return a rate
        # If a rate is divided by a Dur, then we will call Rate.__truediv__
        # We also need divide the duration by the numerator when calculating the rate
        if self.years == 0:
            raise ZeroDivisionError('Cannot divide by a duration of zero') # TODO: consider Rate(Dur(np.inf))
        elif (sc.isnumber(other) and other == 0) or isinstance(other, Dur) and other.years == 0:
            return Rate(0)
        else:
            return Rate(other, self)

    def __abs__(self):
        return self.__class__(abs(self.value))

    @classmethod
    def arange(cls, low, high, step): # TODO: remove? ss.Dur(arr) is preferable to array(ss.Dur) for performance
        """
        Construct an array of Dur instances

        For this function, the low, high, and step must ALL be specified, and they must
        all be Dur instances. Mixing Dur types (years and DateDur) is permitted.

        Args:
            low (ss.Dur): Starting point, e.g., ss.years(0)
            high (ss.Dur): Ending point, e.g. ss.years(20)
            step (ss.Dur): Step size, e.g. ss.years(2)
        """
        args = [low, high, step]
        if not all([isinstance(arg, ss.Dur) for arg in args]):
            errormsg = f'All inputs must be ss.Dur, not {args}'
            raise TypeError(errormsg)

        tvec = []
        t = low
        while t <= high:
            tvec.append(t)
            t += step
        return np.array(tvec)


class DateDur(Dur):
    """ Date based duration e.g., if requiring a week to be 7 calendar days later """
    base = 'year' # DateDur uses 'year' as the default unit for conversion
    timepar_type = 'dur'
    timepar_subtype = 'datedur'

    def __init__(self, *args, **kwargs):
        """
        Create a date-based duration

        Supported arguments are
            - Single positional argument
                - A float number of years
                - A pd.DateOffset
                - A Dur instance
            - Keyword arguments
                - Argument names that match time periods in `ss.time.factors` e.g., 'years', 'months'
                  supporting floating point values. If the value is not an integer, the values
                  will be cascaded to smaller units using `ss.time.factors`

        If no arguments are specified, a DateDur with zero duration will be produced

        Args:
            args:
            kwargs:
        """
        super().__init__()
        if args:
            assert not kwargs, 'DateDur must be instantiated with only 1 arg (which is in years), or keyword arguments.'
            assert len(args) == 1, f'DateDur must be instantiated with only 1 arg (which is in years), or keyword arguments. {len(args)} args were given.'
            if isinstance(args[0], pd.DateOffset):
                self.value = self._round_duration(args[0])
            elif isinstance(args[0], DateDur):
                self.value = sc.dcp(args[0].value) # pd.DateOffset is immutable so this should be OK
            elif isinstance(args[0], Dur):
                self.value = self._round_duration({'years': args[0].years})
            elif sc.isnumber(args[0]):
                self.value = self._round_duration({'years': args[0]})
            else:
                errormsg = f'Unsupported input {args}.\nExpecting number, ss.Dur, ss.DateDur, or pd.DateOffset'
                raise TypeError(errormsg)
        else:
            self.value = self._round_duration(kwargs)

    def __float__(self):
        return float(self.years)

    @classmethod
    def _as_array(cls, dateoffset):
        """
        Return array representation of a pd.DateOffset

        In the array representation, each element corresponds to one of the time units
        in `ss.time.factors` e.g., a 1 year, 2 week duration would be [1, 0, 2, 0, 0, 0, 0, 0]

        Args:
            dateoffset: A pd.DateOffset object

        Returns:
            A numpy array with as many elements as `ss.time.factors`
        """
        return np.array([dateoffset.kwds.get(k, 0) for k in cls.factor_keys])

    @classmethod
    def _as_args(cls, x):
        """
        Return a dictionary representation of a DateOffset or DateOffset array

        This function takes in either a pd.DateOffset, or the result of `_as_array`, and returns
        a dictionary with the same keys as `ss.time.factors` and the values from the input. The
        output of this function can be passed to the `Dur()` constructor to create a new DateDur object.

        Args:
            x: A pd.DateOffset or an array with the same number of elements as `ss.time.factors`

        Returns:
            Dictionary with keys from `ss.time.factors` and values from the input
        """
        if isinstance(x, pd.DateOffset):
            x = cls._as_array(x)
        return {k: v for k, v in zip(cls.factor_keys, x)}

    def to_array(self):
        """ Convert to a Numpy array (NB, different than to_numpy() which converts to fractional years """
        return self._as_array(self.value)

    def to_numpy(self):
        # TODO: This is a quick fix to get plotting to work, but this should do something more sensible
        return self.years

    def to_dict(self):
        """ Convert to a dictionary """
        return self._as_args(self.to_array())

    @property
    def years(self):
        """
        Return approximate conversion into years

        This allows interoperability with years objects (which would typically be expected to
        occur if module parameters have been entered with `DateDur` durations, but the simulation
        timestep is in `years` units).

        The conversion is based on `ss.time.factors` which defines the conversion from each time unit
        to the next

        Returns:
            A float representing the duration in years
        """
        years = 0
        for k, v in self.factors.items():
            years += self.value.kwds.get(k, 0)/v
        return years

    @property
    def is_variable(self):
        if self.value.kwds.get('years',0) or self.value.kwds.get('months',0):
            return True
        else:
            return False

    @classmethod
    def _round_duration(cls, vals):
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
        if isinstance(vals, np.ndarray):
            d = sc.objdict({k:v for k,v in zip(cls.factor_keys, vals)})
        elif isinstance(vals, pd.DateOffset):
            d = sc.objdict.fromkeys(cls.factor_keys, 0)
            d.update(vals.kwds)
        elif isinstance(vals, dict):
            d = sc.objdict.fromkeys(cls.factor_keys, 0)
            d.update(vals)
        else:
            raise TypeError()

        if all([isinstance(val, int) for val in d.values()]):
            if isinstance(vals, pd.DateOffset): return vals  # pd.DateOffset is immutable so this should be OK
            return pd.DateOffset(**d)

        for i in range(len(cls.factors)-1):
            remainder, div = np.modf(d[i])
            d[i] = int(div)
            d[i+1] += remainder * cls.factor_vals[i+1]/cls.factor_vals[i]
        d[-1] = round(d[-1])

        return pd.DateOffset(**d)


    def _scale(self, dateoffset, scale):
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
        return self._round_duration(self._as_array(dateoffset) * scale)

    def __pow__(self, other):
        raise Exception('Cannot multiply a duration by a duration')

    def __mul__(self, other):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, Dur):
            raise Exception('Cannot multiply a duration by a duration')
        return self.__class__(self._scale(self.value, other))

    def __truediv__(self, other):
        if isinstance(other, DateDur):
            # We need to convert both dates into a common timebase to do the division
            # We *can* just do self.years/other.years, however this can cause numerical precision losses
            # yet at the same time is not necessary, because the common timebase depends on the durations.
            # For example, DateDur(weeks=1)/DateDur(days=1) should return 7, but we get 6.9999999 if we convert
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
        elif isinstance(other, Dur):
            return self.years/other.years
        elif isinstance(other, Rate):
            raise Exception('Cannot divide a duration by a rate')
        return self.__class__(self._scale(self.value, 1/other))

    def __repr__(self):
        if self.years == 0:
            return '<DateDur: 0>'
        else:
            labels = self.factor_keys
            vals = self.to_array().astype(float)

            time_portion = vals[4:]
            time_str = ':'.join(f'{np.round(v,1):02g}' for v in time_portion[:3])

            return '<DateDur: ' +  ','.join([f'{k}={int(v)}' for k, v in zip(labels[:4], vals[:4]) if v!=0]) + (f', +{time_str}' if time_str != '00:00:00' else '') + '>'

    def str(self):
        # Friendly representation e.g., 'day', '1 year, 2 months, 1 day'
        vals = self.to_array()

        # If we have only one nonzero value, return 'year', 'month', etc.
        if np.count_nonzero(vals == 1) == 1:
            for unit, val in zip(self.factor_keys, vals):
                if val == 1:
                    return unit[:-1]

        strs = [f'{v} {k[:-1] if abs(v) == 1 else k}' for k, v in zip(self.factor_keys, vals) if v != 0]
        return ', '.join(strs)

    def __add__(self, other):
        if isinstance(other, date):
            return other + self.value
        elif isinstance(other, DateDur):
            return self.__class__(**self._as_args(self.to_array() + self._as_array(other.value)))
        elif isinstance(other, pd.DateOffset):
            return self.__class__(**self._as_args(self.to_array() + self._as_array(other)))
        elif isinstance(other, years):
            kwargs = {k: v for k, v in zip(self.factor_keys, self.to_array())}
            kwargs['years'] += other.years
            return self.__class__(**kwargs)
        elif isinstance(other, DateArray):
            return DateArray(np.vectorize(self.__add__)(other))
        else:
            errormsg = f'For DateDur, it is only possible to add/subtract dates, Dur objects, or pd.DateOffset objects, not {type(other)}'
            raise TypeError(errormsg)

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __abs__(self):
        # Cannot implement this because it's ambiguous how to resolve cases like
        # DateDur(months=-1,days=1) - this is a sensible DateDur interpreted as 'go back 1 month, then go forward 1 day'
        # but just taking the absolute value of all of the components wouldn't work because this would be on average 1 month + 1 day
        # whereas it should be 1 month - 1 day. This could probably be resolved? But is an edge case, unlikely to be needed
        # (whereas abs(years) arises when comparing dates, which automatically return years)
        raise NotImplementedError('The absolute value of a DateDur instance is undefined as components (e.g., months, days) may have different signs.')


class Rate(TimePar):
    """
    Store a value per unit time e.g., 2 per day
    - self.value - the numerator (e.g., 2) - a scalar float
    - self.unit - the denominator (e.g., 1 day) - a Dur object
    """
    base = None
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
                dur = class_map.dur[self.base]
                self.unit = dur(1)
            else:
                self.unit = years(1)
        elif isinstance(unit, Dur):
            self.unit = unit
        elif isinstance(unit, str):
            dur = class_map.dur[unit]
            self.unit = dur(1)
        else: # e.g. number
            self.unit = ss.years(unit) # Default of years

        if not (sc.isnumber(value) or isinstance(value, np.ndarray)):
            errormsg = f'Value must be a scalar number or array, not {type(value)}'
            raise TypeError(errormsg)
        self.value = value
        self._set_base()
        return

    def _set_base(self):
        """ Rates can have either unit or base set; handle either """
        if self.base is None and isinstance(self.unit, ss.TimePar):
            self.base = self.unit.base
        elif self.base != self.unit.base: # Should not be possible, but check just in case
            errormsg = f'Inconsistent definition: base = {self.base} but unit = {self.unit.base}'
            raise ValueError(errormsg)
        return

    def __repr__(self):
        name = self.__class__.__name__
        if name == 'Rate':
            return f'{name}({self.value}/{self.unit})'
        else:
            return f'{name}({self.value})'

    def __float__(self):
        return float(self.value)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self*other
        elif isinstance(other, Dur):
            return self.value*other/self.unit
        else:
            return self.__class__(self.value*other, self.unit)

    def __neg__(self):
        return -1*self

    def __add__(self, other):
        if isinstance(other, Rate):
            if self.timepar_subtype == other.timepar_subtype:
                print('hiii', self.value, self.unit, other.value, other.unit)
                return self.__class__(self.value+other*self.unit, self.unit)
            else:
                errormsg = f'Can only add rates with the same subtype (e.g., Rate+Rate, TimeProb+TimeProb); you added {self} + {other}'
                raise TypeError(errormsg)
        else:
            if sc.isnumber(other) or isinstance(other, np.ndarray):
                errormsg = 'Only rates can be added to rates, not {other}. This error most commonly occurs if the rate needs to be multiplied by `self.dt` to get a number of events per timestep.'
                raise TypeError(errormsg)
            else:
                errormsg = 'Only rates can be added to rates, not {other}'
                raise TypeError(errormsg)

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
            raise TypeError('Can only subtract rates of identical types (e.g., Rate+Rate, TimeProb+TimeProb)')

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
        (e.g., `TimeProb`) and therefore it is important to apply the factor prior to multiplication by duration.
        This function avoids creating an intermediate array of rates, and is therefore much higher performance.

        e.g.

        >>> p = ss.TimeProb(0.05)*self.cd4*self.t.dt

        and

        >>> p = ss.TimeProb(0.05).scale(self.cd4,self.t.dt)

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


class TimeProb(Rate):
    """
    `TimeProb` represents the probability of an event occurring during a
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
    >>> p = ss.TimeProb(0.8, ss.years(1))
    indicates a 80% chance of an event occurring in one year.

    >>> p*ss.years(1)
    When multiplied by the original denominator, 1 year in this case, the
    probability remains unchanged, 80%.

    >>> p * ss.years(2)
    Multiplying `p` by `ss.years(2)` does not simply double the
    probability to 160% (which is not possible), but rather returns a new
    probability of 96% representing the chance of the event occurring at least
    once over the new duration of two years.

    However, the behavior is different when a `TimeProb` object is multiplied
    by a scalar or array. In this case, the probability is simply scaled. This scaling
    may result in a value greater than 1, which is not valid. For example,
    >>> p * 2
    raises an AssertionError because the resulting probability (160%) exceeds 100%.

    Use `RateProb` instead if `TimeProb` if you would prefer to directly
    specify the instantaneous rate.
    """
    base = None # Can inherit, but clearer to specify
    timepar_type = 'rate'
    timepar_subtype = 'timeprob'

    def __init__(self, value, unit=None):
        try:
            assert 0 <= value <= 1, 'Value must be between 0 and 1'
        except Exception: # Something went wrong, let's figure it out -- either a value error, or value is an array instead of a scalar
            if sc.isnumber(value):
                valstr = f'Value provided: {value}'
                lt0 = value < 0
                gt1 = value > 1
            else:
                valstr = f'Values provided:\n{value}'
                lt0 = np.any(value<0)
                gt1 = np.any(value>1)
            if lt0:
                errormsg = f'Negative values are not permitted for rates or probabilities. {valstr}'
            elif gt1:
                if self.base is not None:
                    correct = f'ss.rateper{self.base}()'
                    self.__class__ = class_map.rate[f'rateper{self.base}'] # Mutate the class to the correct class
                    if ss.options.warn_convert:
                        warnmsg = f'Probabilities cannot be greater than one, so converting to a rate. Please use {correct} instead, or set ss.options.warn_convert=false. {valstr}'
                        ss.warn(warnmsg)
                else:
                    errormsg = 'Probabilities are >1, and no base was unit provided. Please use e.g. ss.rateperyear() instead of doing whatever you did. {valstr}'
                    raise ValueError(errormsg)
        Rate.__init__(self, value, unit) # Can't use super() since potentially mutating the class
        return

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            self.array_mul_error()
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

    @classmethod
    def array_to_prob(cls, arr, dur, v=1):
        arr = sc.promotetoarray(arr)

        if isinstance(dur, np.ndarray):
            assert arr.shape == dur.shape, 'dur must be either a scalar, or the same size as arr'
            def to_prob(a, b, _v=v):
                return a.to_prob(dur=b, v=_v)

            vectorize = np.vectorize(to_prob)
            return vectorize(arr, dur)

        elif isinstance(arr[0], TimeProb):
            factor = np.array([dur / a.unit for a in arr])
            scaled_vals = np.array([a.value * v for a in arr])
            rate = - np.log(1 - scaled_vals)
            return 1-np.exp(-rate*factor)

        else: # Assume arr is an array of values, that would be the values of a TimeProb with unit=ss.years(1)
            factor = dur / ss.years(1)
            scaled_vals = arr * v
            rate = - np.log(1 - scaled_vals)
            return 1 - np.exp(-rate * factor)

    def __truediv__(self, other): raise NotImplementedError()
    def __rtruediv__(self, other): raise NotImplementedError()


class RateProb(Rate):
    """
    A `RateProb` represents an instantaneous rate of an event occurring. Rates
    must be non-negative, but need not be less than 1.

    Through multiplication, rate can be modified or converted to a probability,
    depending on the data type of the object being multiplied.

    When a `RateProb` is multiplied by a scalar or array, the rate is simply
    scaled. Such multiplication occurs frequently in epidemiological models,
    where the base rate is multiplied by "rate ratio" or "relative rate" to
    represent agents experiencing higher (multiplier > 1) or lower (multiplier <
    1) event rates.

    Alternatively, when a `RateProb` is multiplied by a duration (type
    ss.Dur), a probability is calculated. The conversion from rate to
    probability on multiplication by a duration is
        `1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the multiplied duration to the original
    period (denominator).

    For example, consider
    >>> p = ss.RateProb(0.8, ss.years(1))
    When multiplied by a duration of 1 year, the calculated probability is
        `1 - np.exp(-0.8)`, which is approximately 55%.
    >>> p*ss.years(1)

    When multiplied by a scalar, the rate is simply scaled.
    >>> p*2

    The difference between `TimeProb` and `RateProb` is subtle, but important. `RateProb` works directly
    with the instantaneous rate of an event occurring. In contrast, `TimeProb` starts with a probability and a duration,
    and the underlying rate is calculated. On multiplication by a duration,
    * RateProb: rate -> probability
    * TimeProb: probability -> rate -> probability

    The behavior of both classes is depending on the data type of the object being multiplied.
    """
    base = None # Can inherit, but clearer to specify
    timepar_type = 'rate'
    timepar_subtype = 'rateprob'

    def __init__(self, value, unit=None):
        assert value >= 0, 'Value must be >= 0'
        return super().__init__(value, unit)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            self.array_mul_error()
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


#%% Convenience classes
__all__ += ['years', 'months', 'weeks', 'days', 'year', 'month', 'week', 'day', # Durations
            'perday', 'perweek', 'permonth', 'peryear', # TimeProbs
            'probperday', 'probperweek', 'probpermonth', 'probperyear', # TimeProb aliases
            'rateperday', 'rateperweek', 'ratepermonth', 'rateperyear'] # Rates

# Durations
class years(Dur):  base = 'year'
class months(Dur): base = 'month'
class weeks(Dur):  base = 'week'
class days(Dur):   base = 'day'

# Duration shortcuts
year = years(1)
month = months(1)
week = weeks(1)
day = days(1)
for obj in [year, month, week, day]:
    object.__setattr__(obj, '_locked', True) # Make immutable

# TimeProbs
class perday(TimeProb):   base = 'day'
class perweek(TimeProb):  base = 'week'
class permonth(TimeProb): base = 'month'
class peryear(TimeProb):  base = 'year'

# Aliases
probperday = perday
probperweek = perweek
probpermonth = permonth
probperyear = peryear

# Rates
class rateperday(Rate):   base = 'day'
class rateperweek(Rate):  base = 'week'
class ratepermonth(Rate): base = 'month'
class rateperyear(Rate):  base = 'year'

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

    rateperday:   ['rateperday', rateperday],
    rateperweek:  ['rateperweek', rateperweek],
    ratepermonth: ['ratepermonth', ratepermonth],
    rateperyear:  ['rateperyear', rateperyear],
}

# Convert to the actual class map
class_map = sc.objdict(dur=sc.objdict(), rate=sc.objdict()) # TODO: make a class with an informative error message for invalid lookups (e.g. dur vs rate)
for v, keylist in reverse_class_map.items():
    for key in keylist:
        if issubclass(v, Dur):
            class_map.dur[key] = v
        elif issubclass(v, Rate):
            class_map.rate[key] = v
        else:
            errormsg = f'Unexpected entry: {v}'
            raise TypeError(errormsg)