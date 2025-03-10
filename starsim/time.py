"""
Functions and classes for handling time
"""
import sciris as sc
import numpy as np
import pandas as pd
import datetime as dt
import dateutil as du
import starsim as ss


#%% Helper functions

# Classes and objects that are externally visible
# __all__ = ['time_units', 'time_ratio', 'date_add', 'date_diff']

# Define available time units
# time_units = sc.objdict(
#     day   = 1.0,
#     week  = 7.0,
#     month = 30.4375, # 365.25/12
#     year  = 365.25, # For consistency with months
# )

# Base classes for points in time and durations that behave sensibly

class Date(pd.Timestamp):
    """
    Define a point in time, based on ``pd.Timestamp``

    Args:
        date (int/float/str/datetime): Any type of date input (ints and floats will be interpreted as years)
        kwargs (dict): passed to pd.Timestamp()

    **Examples**::

        ss.date(2020) # Returns <2020-01-01>
        ss.date(year=2020) # Returns <2020-01-01>
        ss.date(year=2024.75) # Returns <2024-10-01>
        ss.date('2024-04-04') # Returns <2024-04-04>
        ss.date(year=2024, month=4, day=4) # Returns <2024-04-04>
    """

    def __new__(cls, *args, **kwargs):
        # Check if a year was supplied, and preprocess it
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
        out = super(Date, cls).__new__(cls, *args, **kwargs)
        out = cls._reset_class(out)
        return out

    @classmethod
    def _reset_class(cls, obj):
        """ Manually reset the class from pd.Timestamp to ss.date """
        obj.__class__ = Date
        return obj

    def __reduce__(self):
        # This function is internally used when pickling (rather than getstate/setstate)
        # due to Pandas implementing C functions for this. We can wrap the Pandas unpickler
        # with a function that wraps calls _reset_class to support unpickling Date
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
            string += ' ' + self._time_repr

        if bracket:
            string = '<' + string + '>'
        return string

    def __str__(self):
        """ Like repr, but just the date, e.g. 2024.04.04 """
        return self.__repr__(bracket=False)

    def disp(self, **kwargs):
        """ Show the full object """
        return sc.pr(self, **kwargs)

    @classmethod
    def from_year(cls, year):
        """
        Convert an int or float year to a date.

        **Examples**::

            ss.date.from_year(2020) # Returns <2020-01-01>
            ss.date.from_year(2024.75) # Returns <2024-10-01>
        """

        if isinstance(year, int):
            return cls(year=year, month=1, day=1)
        else:
            year_start = pd.Timestamp(year=int(year),month=1,day=1).timestamp()
            year_end = pd.Timestamp(year=int(year),month=12,day=31).timestamp()
            timestamp = year_start + year%1*(year_end-year_start)
            return cls(pd.Timestamp(timestamp, unit='s'))

    def to_year(self):
        """
        Convert a date to a floating-point year

        **Examples**::

            ss.date('2020-01-01').to_year() # Returns 2020.0
            ss.date('2024-10-01').to_year() # Returns 2024.7486
        """
        year_start = pd.Timestamp(year=self.year,month=1,day=1).timestamp()
        year_end = pd.Timestamp(year=self.year,month=12,day=31).timestamp()
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


    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return np.vectorize(self.__add__)(other)

        if isinstance(other, DateDur):
            return Date(self.to_pandas() + other.period)
        elif isinstance(other, YearDur):
            return Date(self.to_year() + other.period)
        elif isinstance(other, pd.DateOffset):
            return Date(self.to_pandas() + other)
        elif isinstance(other, pd.Timestamp):
            raise TypeError('Cannot add a date to another date')
        else:
            raise TypeError('Unsupported type')

    def __sub__(self, other):

        if isinstance(other, np.ndarray):
            return np.vectorize(self.__sub__)(other)

        if isinstance(other, DateDur):
            return Date(self.to_pandas() - other.period)
        elif isinstance(other, YearDur):
            return Date(self.to_year() - other.period)
        elif isinstance(other, pd.DateOffset):
            return Date(self.to_pandas() - other)
        elif isinstance(other, Date):
            # TODO: Could replace this with a DateDur but need to set the components carefully
            return YearDur(self.years-other.years)
        else:
            raise TypeError('Unsupported type')
    #
    def __radd__(self, other): return self.__add__(other)
    # def __iadd__(self, other): return self.__add__(other) # I think pd.Timestamp is immutable so these shouldn't be implemented?
    def __rsub__(self, other): return self.__sub__(other) # TODO: check if this should be reversed
    # def __isub__(self, other): return self.__sub__(other)

class Dur():
    # Base class for durations/periods
    # Subclasses for date-durations and fixed-durations

    # Conversion ratios from date-based durations to fixed durations
    ratios = sc.objdict(
        years=1,
        months=12,
        weeks=52/12,
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
                    assert len(args) == 1
                    return super().__new__(YearDur)
            else:
                return super().__new__(DateDur)
        return super().__new__(cls)

    # NB. Durations are considered to be equal if their year-equivalent duration is equal
    # That would mean that Dur(years=1)==Dur(1) returns True - probably less confusing than having it return False?
    def __hash__(self):
        return hash(self.years)

    def __eq__(self):
        return self.years == other.years

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




    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)

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
        return Rate(self/other)

class YearDur(Dur):
    # Year based duration e.g., if requiring 52 weeks per year

    def __init__(self, years=0):
        """
        Construct a year-based fixed duration

        Supported inputs are
            - A float number of years (so can use values like 1/52 for 1 week)
            - A YearDur instance (as a copy-constructor)
            - A DateDur instance (to convert from date-based to fixed representation)

        :param years: float, YearDur, DateDur
        """
        if isinstance(years, Dur):
            self.period = years.years
        else:
            self.period = years

    def to_numpy(self):
        return self.period

    @property
    def years(self):
        return self.period # Needs to return float so matplotlib can plot it correctly

    @property
    def is_variable(self):
        return False

    def __add__(self, other):
        if isinstance(other, Dur):
            return self.__class__(self.period + other.years)
        elif isinstance(other, Date):
            return Date.from_year(other.to_year() + self.period)
        else:
            return self.__class__(self.period + other)

    def __sub__(self, other):
        if isinstance(other, Dur):
            return self.__class__(max(self.period - other.years, 0))
        elif isinstance(other, Date):
            return Date.from_year(other.to_year() - self.period)
        else:
            return self.__class__(max(self.period - other, 0))

    def __mul__(self, other: float):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, Dur):
            raise Exception('Cannot multiply a duration by a duration')
        elif isinstance(other, Date):
            raise Exception('Cannot multiply a duration by a date')
        return self.__class__(self.period*other)

    def __truediv__(self, other):
        if isinstance(other, Dur):
            return self.years/other.years
        else:
            return self.__class__(self.period / other)

    def __repr__(self):
        return f'<YearDur: {self.period} years>'

class DateDur(Dur):
    # Date based duration e.g., if requiring a week to be 7 calendar days later

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

        :param args:
        :param kwargs:
        """
        if args:
            assert not kwargs
            assert len(args) == 1
            if isinstance(args[0], pd.DateOffset):
                self.period = self._round_duration(args[0])
            elif isinstance(args[0], DateDur):
                self.period = args[0].period # pd.DateOffset is immutable so this should be OK
            elif isinstance(args[0], YearDur):
                self.period = self._round_duration({'years': args[0].years})
            elif sc.isnumber(args[0]):
                self.period = self._round_duration({'years': args[0]})
            else:
                raise TypeError('Unsupported input')
        else:
            self.period = self._round_duration(kwargs)

    @classmethod
    def _as_array(cls, dateoffset: pd.DateOffset) -> np.ndarray:
        """
        Return array representation of a pd.DateOffset

        In the array representation, each element corresponds to one of the time units
        in self.ratios e.g., a 1 year, 2 week duration would be [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]

        :param dateoffset: A pd.DateOffset object
        :return: A numpy array with as many elements as `DateDur.ratios`
        """
        return np.array([dateoffset.kwds.get(k, 0) for k in cls.ratios.keys()])

    @classmethod
    def _as_args(cls, x) -> dict:
        """
        Return a dictionary representation of a DateOffset or DateOffset array

        This function takes in either a pd.DateOffset, or the result of `_as_array`, and returns
        a dictionary with the same keys as `DateDur.ratios` and the values from the input. The
        output of this function can be passed to the `Dur()` constructor to create a new DateDur object.

        :param x: A pd.DateOffset or an array with the same number of elements as `DateDur.ratios`
        :return: Dictionary with keys from `DateDur.ratios` and values from the input
        """
        if isinstance(x, pd.DateOffset):
            x = self._as_array(x)
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

        :return: A float representing the duration in years
        """
        denominator = 1
        years = 0
        for k, v in self.ratios.items():
            denominator *= v
            years += self.period.kwds.get(k, 0)/denominator
        return years

    @property
    def is_variable(self):
        if self.period.kwds.get('years',0) or self.period.kwds.get('months',0):
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

        :return: A pd.DateOffset
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

        for i in range(len(cls.ratios)-1):
            if d[i] < 0:
                d[i], remainder = divmod(d[i], 1)
                if remainder:
                    d[i]+=1
                    remainder -=1
            else:
                d[i], remainder = divmod(d[i], 1)
            d[i] = int(d[i])
            d[i+1] += remainder * cls.ratios[i+1]
        d[-1] = int(d[-1])

        return pd.DateOffset(**d)


    def _scale(self, dateoffset: pd.DateOffset, scale: float) -> pd.DateOffset:
        """
        Scale a pd.DateOffset by a factor

        This function will automatically cascade remainders to finer units using `DateDur.ratios` so for
        example 2.5 weeks would first become 2 weeks and 0.5*7 = 3.5 days,
        and then become 3 days + 0.5*24 = 12 hours.

        :param dateoffset: A pd.DateOffset instance
        :param scale: A float scaling factor (must be positive)
        :return: A pd.DateOffset instance scaled by the requested amount
        """
        return self._round_duration(self._as_array(dateoffset) * scale)

    def to_numpy(self):
        # TODO: This is a quick fix to get plotting to work, but this should do something more sensible
        return self.years
        # return self.period.to_numpy()

    def __mul__(self, other: float):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, Dur):
            raise Exception('Cannot multiply a duration by a duration')
        return self.__class__(self._scale(self.period, other))

    def __truediv__(self, other):
        if isinstance(other, Dur):
            return self.years/other.years
        return self.__class__(self._scale(self.period, 1/other))

    def __repr__(self):
        if self.years == 0:
            return '<DateDur: 0>'
        else:
            return '<DateDur: ' +  ','.join([f'{k}={v}' for k, v in zip(self.ratios, self._as_array(self.period)) if v!=0]) + '>'

    def __add__(self, other):
        if isinstance(other, Date):
            return other + self.period
        elif isinstance(other, DateDur):
            return self.__class__(**self._as_args(self._as_array(self.period) + self._as_array(other.period)))
        elif isinstance(other, pd.DateOffset):
            return self.__class__(**self._as_args(self._as_array(self.period) + self._as_array(other)))
        elif isinstance(other, YearDur):
            kwargs = {k: v for k, v in zip(self.ratios, self._as_array(self.period))}
            kwargs['years'] += other.years
            return self.__class__(**kwargs)
        else:
            raise TypeError('Can only add dates, Dur objects, or pd.DateOffset objects')

    def __sub__(self, other):
        return self.__add__(-1*other)

class Rate():
    def __init__(self, dur:Dur):
        self._dur = dur

    def __repr__(self):
        if isinstance(self._dur, YearDur):
            return f'<{self.__class__.__name__}: {self * YearDur(1)} per year>'
        else:
            return f'<{self.__class__.__name__}: per{str(self._dur).split(":")[1]} (approx. {self * YearDur(1)} per year)'

    def __mul__(self, other):
        if isinstance(other, Dur):
            return other.years/self._dur.years
        elif other == 0:
            return 0 # Or should this be a rate?
        else:
            return Rate(self._dur/other)

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        # This is for <rate>/<other>
        if isinstance(other, Rate):
            # 2 per year divided by 4 per year would be 0.5 as in it's half the rate
            # The corresponding periods would be 0.5 and 0.25, so we want to return other._dur/self._dur
            return other._dur/self._dur
        elif isinstance(other, Dur):
            raise Exception('Cannot divide a rate by a duration')
        else:
            return Rate(self._dur*other)

    def __rtruediv__(self, other):
        # If a float is divided by a rate, we should get back the duration
        return other * self._dur


class RateProb(Rate):
    """
    Class to represent probability rates

    These values are probabilities per unit time. The probability for a given
    period of time can be calculated by multiplying by a duration. The cumulative
    hazard rate conversion will be used when calculating the probability. For multiplicative
    scaling, use a ``Rate`` instead.

    >>> p = ss.RateProb(0.1, ss.Dur(years=1))
    >>> p*ss.Dur(years=2)
    """
    def __init__(self, p, dur=None):
        if dur is None:
            dur = YearDur(1)
        super().__init__(dur)
        self.p = p

    def __mul__(self, other):
        if isinstance(other, Dur):
            if self.p == 0:
                return 0
            elif self.p == 1:
                return 1
            elif 0 <= self.p <= 1:
                rate = -np.log(1 - self.p)
                factor = self._dur/other
                return 1 - np.exp(-rate/factor)
            else:
                errormsg = f'Invalid value {self.v} for {self}: must be 0-1. If using in a calculation, use .values instead.'
                raise ValueError(errormsg)

        else:
            return self.__class__(self.p*other, self._dur)




#%% Time - simulation vectors

class Time(sc.prettyobj):
    """
    Handle time vectors for both simulations and modules.

    Each module can have its own time instance, in the case where the time vector
    is defined by absolute dates, these time vectors are by definition aligned. Otherwise
    they can be specified using Dur objects which express relative times (they can be added
    to a Date to get an absolute time)

    Args:
        start : ss.Date or ss.Dur
        stop : ss.Date if start is an ss.Date, or an ss.Dur if start is an ss.Dur
        dt (ss.Dur): Simulation step size
        pars (dict): if provided, populate parameter values from this dictionary
        parent (obj): if provided, populate missing parameter values from a 'parent" ``Time`` instance
        name (str): if provided, name the ``Time`` object
        init (bool): whether or not to immediately initialize the Time object
        sim (bool/Sim): if True, initializes as a sim-specific ``Time`` instance; if a Sim instance, initialize the absolute time vector

    The ``Time`` object, after initialization, has the following attributes:

    - ``ti`` (int): the current timestep
    - ``npts`` (int): the number of timesteps
    - ``tvec`` (array): time starting at 0, in self units (e.g. ``[0, 0.1, 0.2, ... 10.0]`` if start=0, stop=10, dt=0.1)
    - ``yearvec`` (array): time represented as floating-point years (e.g. ``[2000, 2000.1, 2000.2, ... 2010.0]`` if start=2000, stop=2010, dt=0.1)

    **Examples**::

        t1 = ss.Time(start=2000, stop=2020, dt=1.0, unit='year') # Years, numeric units
        t2 = ss.Time(start='2021-01-01', stop='2021-04-04', dt=2.0, unit='day') # Days, date units
    """

    # Allowable time arguments
    time_args = ['start', 'stop', 'dt']

    def __init__(self, start=None, stop=None, dt=None, unit=None, pars=None, parent=None,
                 name=None, init=True, sim=None):


        self.name = name
        self.start = start
        self.stop = stop
        self.dt = dt
        self.ti = 0 # The time index, e.g. 0, 1, 2


        # Prepare for later initialization

    # - ``tvec`` (array): time starting at 0, in self units (e.g. ``[0, 0.1, 0.2, ... 10.0]`` if start=0, stop=10, dt=0.1)
    # - ``absvec`` (array): time relative to sim start, in units of sim units (e.g. ``[366, 373, 380, ...]`` if sim-start=2001, start=2002, sim-unit='day', unit='week')
    # - ``yearvec`` (array): time represented as floating-point years (e.g. ``[2000, 2000.1, 2000.2, ... 2010.0]`` if start=2000, stop=2010, dt=0.1)
    # - ``datevec`` (array): time represented as an array of ``ss.date`` objects (e.g. ``[<2000.01.01>, <2000.02.07>, ... <2010.01.01>]`` if start=2000, stop=2010, dt=0.1)
    # - ``timevec`` (array): the "native" time vector, which always matches one of ``tvec``, ``yearvec``, or ``datevec``

        self.tvec    = None # The time vector for this instance in Date or Dur format
        self.absvec  = None # Time vector relative to sim_start OR dur=0
        self.datevec = None # Time vector as calendar dates or DateDur
        self.yearvec = None # Time vector as floating point years or YearDur

        # self.abstvec = None
        self.initialized = False

        # Finalize
        self.update(pars=pars, parent=parent)
        if init and self.ready:
            self.init(sim=sim)
        return

    def __repr__(self):
        if self.initialized:
            return f'<Time t={self.tvec[self.ti]}, ti={self.ti}, {self.start}-{self.stop} dt={self.dt}>'
        else:
            return f'<Time (uninitialized)>'

    @property
    def timevec(self):
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

        A time vector is absolute if the start is a Date rather than a Dur
        A relative time vector can be made absolute by adding a Date to it
        """
        try:
            return isinstance(self.start, Date)
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
        # Initial validation
        # self.unit = validate_unit(self.unit)

        # Copy missing values from sim
        if isinstance(sim, ss.Sim):
            self.dt = sc.ifelse(self.dt, sim.t.dt)
            self.start = sc.ifelse(self.start, sim.t.start)
            self.stop = sc.ifelse(self.stop, sim.t.stop)

            if self.is_absolute != sim.t.is_absolute:
                raise Exception('Cannot mix absolute/relative times across modules')

        # TODO: Reimplement validation logic
        if isinstance(self.stop, Dur) and self.start is None:
            self.start = self.stop.__class__()


        # # Handle start and stop
        # if start is None:
        #     start = 0


        # self.start = self.start if self.is_numeric else date(self.start)
        # self.stop  = self.stop  if self.is_numeric else date(self.stop)

        # Convert start and stop to dates
        # date_start = self.start
        # date_stop = self.stop
        # date_unit = 'year' if not has_units(self.unit) else self.unit # Use year by default
        # offset = 0
        # if self.is_numeric and date_start == 0:
        #     date_start = ss.date(ss.time.default_start[date_unit])
        #     date_stop = date_start + ss.dur(date_stop, unit=date_unit)
        #     offset = date_start.year



        # # If unitless, just use that
        # if self.is_unitless:
        #     timevec = round_tvec(sc.inclusiverange(self.start, self.stop, self.dt))
        #     yearvec = round_tvec(timevec)
        #     datevec = timevec

        # If the unit is years, handle that
        # elif date_unit == 'year': # For years, the yearvec is the most robust representation
        #     start_year = sc.datetoyear(date_start.date())
        #     stop_year = sc.datetoyear(date_stop.date())
        #     yearvec = round_tvec(sc.inclusiverange(start_year, stop_year, self.dt))
        #     datevec = years_to_dates(yearvec)
        #     timevec = datevec

        # Otherwise, use dates as the ground truth
        # else:
        #     if int(self.dt) == self.dt: # The step is integer-like, use exactly
        #         key = date_unit + 's' # e.g. day -> days
        #         datelist = sc.daterange(date_start, date_stop, interval={key:int(self.dt)})
        #     else: # Convert to the sim unit instead
        #         day_delta = time_ratio(unit1=date_unit, dt1=self.dt, unit2='day', dt2=1.0, as_int=True)
        #         if day_delta >= 1:
        #             datelist = sc.daterange(date_start, date_stop, interval={'days':day_delta})
        #         else:
        #             errormsg = f'Timestep {dt} is too small; must be at least 1 day'
        #             raise ValueError(errormsg)
        #     datevec = np.array([ss.date(d) for d in datelist])
        #     yearvec = dates_to_years(datevec)
        #     timevec = datevec

        # Store things
        # self.dt_year = dt_year
        # self.npts = len(timevec) # The number of points in the sim

        if isinstance(self.dt, YearDur):
            # Preference setting fractional years
            self.yearvec = np.arange(self.start.years, self.stop.years, self.dt.years)
            if isinstance(self.stop, Dur):
                self.tvec = np.array([self.stop.__class__(x) for x in self.yearvec])
            else:
                self.tvec = np.array([Date(x) for x in self.yearvec])
        else:
            # Preference setting dates
            tvec = []
            t = self.start
            while t <= self.stop:
                tvec.append(t)
                t += self.dt

            self.tvec = np.array(tvec)
            self.yearvec = np.array([x.years for x in tvec])


        # if self.is_absolute:
        #     self.datevec = self.tvec
        # else:
        #     self.datevec = None

        #
        #
        #
        # self.tvec = round_tvec(np.arange(self.npts)*self.dt) # Absolute time array
        # self.timevec = timevec
        # self.datevec = datevec
        # self.yearvec = yearvec
        # if sim == True: # It's the sim itself, the tvec is the absolute time vector
        #     self.abstvec = self.tvec
        # elif sim is not None:
        #     self.make_abstvec(sim)
        # else:
        #     self.abstvec = None # Intentionally set to None, cannot be used in the sim loop until populated
        self.initialized = True
        return
    #
    # def make_abstvec(self, sim):
    #     """ Convert the current time vector into sim units """
    #     # Validation
    #     if self.is_unitless != sim.t.is_unitless:
    #         errormsg = f'Cannot mix units with unitless time: sim.unit={sim.t.unit} {self.name}.unit={self.unit}'
    #         raise ValueError(errormsg)
    #
    #     # Both are unitless or numeric
    #     both_unitless = self.is_unitless and sim.t.is_unitless
    #     both_numeric = self.is_numeric and sim.t.is_numeric
    #     if both_unitless or both_numeric:
    #         abstvec = self.tvec.copy() # Start by copying the current time vector
    #         ratio = time_ratio(unit1=self.unit, dt1=1.0, unit2=sim.t.unit, dt2=1.0) # tvec has sim units, but not dt
    #         if ratio != 1.0:
    #             abstvec *= ratio # TODO: CHECK
    #         start_diff = self.start - sim.t.start
    #         if start_diff != 0.0:
    #             abstvec += start_diff
    #
    #     # The sim uses years; use yearvec
    #     elif sim.t.unit == 'year':
    #         abstvec = self.yearvec.copy()
    #         abstvec -= sim.t.yearvec[0] # Start relative to sim start
    #
    #     # Otherwise (days, weeks, months), use datevec and convert to days
    #     else:
    #         dayvec = dates_to_days(self.datevec, start_date=sim.t.datevec[0])
    #         ratio = time_ratio(unit1='day', dt1=1.0, unit2=sim.t.unit, dt2=1.0)
    #         abstvec = dayvec*ratio # Convert into sim time units
    #
    #     self.abstvec = round_tvec(abstvec) # Avoid floating point inconsistencies
    #     return

    def now(self, key=None):
        """
        Get the current simulation time

        Args:
            which (str): which type of time to get: default (None), "year", "date", "tvec", or "str"

        **Examples**::

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

        now = vec[min(self.ti, len(vec)-1)]
        if key == 'str':
            now = f'{now:0.1f}' if isinstance(now, float) else str(now)
        return now



#%% TimePar classes

# Use Pandas pd.TimeStamp and durs are in pd.DateOffset

def years(x: float) -> Dur:
    return Dur(years=x)

def months(x: float) -> Dur:
    return Dur(months=x)

def weeks(x: float) -> Dur:
    return Dur(weeks=x)

def days(x: float) -> Dur:
    return Dur(days=x)

## TIME AWARE PARAMETERS

#
# __all__ += ['TimePar', 'dur', 'days', 'years', 'rate', 'perday', 'peryear',
#             'time_prob', 'beta', 'rate_prob']

# class TimePar(ss.BaseArr):
#     """
#     Base class for time-aware parameters, durations and rates
#
#     Stores either pd.Timestamp/ss.Date objects or
#
#     NB, because the factor needs to be recalculated, do not set values directly.
#     """
#     def __new__(cls, v=None, *args, **kwargs):
#         """ Allow TimePars to wrap distributions and return the distributions """
#
#         # Special distribution handling
#         if isinstance(v, ss.Dist):
#             if len(args):
#                 errormsg = f'When wrapping a distribution with a TimePar, args not allowed ({args}); use kwargs'
#                 raise ValueError(errormsg)
#             dist = v
#             dist.pars[0] = cls(dist.pars[0], **kwargs) # Convert the first parameter to a TimePar (the same scale is applied to all parameters)
#             return dist
#
#         # Otherwise, do the usual initialization
#         else:
#             return super().__new__(cls)
#
#     def __init__(self, v):
#         """
#
#         :param v: Input quantity. A pd.Timestamp, pd.DateOffset,
#         :param unit:
#         :param parent_unit:
#         :param parent_dt:
#         :param self_dt:
#         """
#
#         self.v = v
#         self.values = None
#         self.initialized = False
#         # self.validate_units()
#         return
#
#     # def validate_units(self):
#     #     """ Check that the units entered are valid """
#     #     try:
#     #         self.unit = unit_mapping[self.unit]
#     #     except KeyError:
#     #         errormsg = f'Invalid unit "{self.unit}"; must be one of: {sc.strjoin(time_units.keys())}'
#     #         raise ValueError(errormsg)
#     #     try:
#     #         self.parent_unit = unit_mapping[self.parent_unit]
#     #     except KeyError:
#     #         errormsg = f'Invalid parent unit "{self.parent_unit}"; must be one of: {sc.strjoin(time_units.keys())}'
#     #         raise ValueError(errormsg)
#     #     return
#
#     def init(self, parent=None, parent_unit=None, parent_dt=None, update_values=True, die=True):
#         """ Link to the sim and/or module units """
#         # if parent is None:
#         #     parent = sc.dictobj(unit=parent_unit, dt=parent_dt)
#         # else:
#         #     if parent_dt is not None:
#         #         errormsg = f'Cannot override parent {parent} by setting parent_dt; set in parent object instead'
#         #         raise ValueError(errormsg)
#
#         # Set defaults if not yet set
#         self.unit = sc.ifelse(self.unit, self.parent_unit) # If unit isn't defined but parent is, set to parent
#
#         # Calculate the actual conversion factor to be used in the calculations
#         self.update_cached(update_values=update_values, die=die)
#         self.initialized = True
#         self.validate_units()
#         return self
#
#     def __repr__(self):
#         name = self.__class__.__name__
#
#         if self.initialized:
#             if self.factor == 1.0:
#                 xstr = ''
#             else:
#                 xstr = f', values={self.values}'
#         else:
#             xstr = ', initialized=False'
#
#         if (self.parent_unit is not None) and (self.unit != self.parent_unit):
#             parentstr = f', parent={self.parent_unit}'
#         else:
#             parentstr = ''
#
#         default_dt = sc.ifelse(self.self_dt, 1.0) == 1.0
#         if not default_dt:
#             dtstr = f', self_dt={self.self_dt}'
#         else:
#             dtstr = ''
#
#         # Rather than ss.dur(3, unit='day'), dispaly as ss.days(3)
#         prefixstr = 'ss.'
#         key = (name, self.unit)
#         mapping = {
#             ('dur',  'day'):  'days',
#             ('dur',  'year'): 'years',
#             ('rate', 'day'):  'perday',
#             ('rate', 'year'): 'peryear',
#         }
#
#         if key in mapping and default_dt:
#             prefixstr += mapping[key]
#             unitstr = ''
#         else:
#             prefixstr += name
#             unitstr = f', unit={self.unit}'
#
#         suffixstr = unitstr + parentstr + dtstr + xstr
#
#         return f'{prefixstr}({self.v}{suffixstr})'
#
#     @property
#     def isarray(self):
#         """ Check if the value is an array """
#         return isinstance(self.v, np.ndarray)
#
#     def set(self, v=None, unit=None, parent_unit=None, parent_dt=None, self_dt=None, force=False):
#         """ Set the specified parameter values (ignoring None values) and update stored values """
#         if v           is not None: self.v           = v
#         if unit        is not None: self.unit        = unit
#         if parent_unit is not None: self.parent_unit = parent_unit
#         if parent_dt   is not None: self.parent_dt   = parent_dt
#         if self_dt     is not None: self.self_dt     = self_dt
#         if self.initialized or force: # Don't try to set these unless it's been initialized
#             self.update_cached()
#         self.validate_units()
#         return self
#
#     def update_cached(self, update_values=True, die=True):
#         """ Update the cached factor and values """
#         try:
#             self.update_factor()
#             if update_values:
#                 self.update_values()
#         except TypeError as E: # For a known error, skip silently if die=False
#             if die:
#                 errormsg = f'Update failed for {self}. Argument v={self.v} should be a number or array; if a function, use update_values=False. Error: {E}'
#                 raise TypeError(errormsg) from E
#         except Exception as E: # For other errors, raise a warning
#             if die:
#                 raise E
#             else:
#                 tb = sc.traceback(E)
#                 warnmsg = f'Uncaught error encountered while updating {self}, but die=False. Traceback:\n{tb}'
#                 ss.warn(warnmsg)
#
#         return self
#
#     def update_factor(self):
#         """ Set factor used to multiply the value to get the output """
#         self.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=self.parent_unit, dt2=self.parent_dt)
#         return
#
#     def update_values(self):
#         """ Convert from self.v to self.values based on self.factor -- must be implemented by derived classes """
#         raise NotImplementedError
#
#     def to(self, unit=None, dt=None):
#         """ Create a new timepar based on the current one but with a different unit and/or dt """
#         new = self.asnew()
#         unit = sc.ifelse(unit, self.parent_unit, self.unit)
#         parent_dt = sc.ifelse(dt, 1.0)
#         new.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=unit, dt2=parent_dt) # Calculate the new factor
#         new.update_values() # Update values
#         new.v = new.values # Reset the base value
#         new.factor = 1.0 # Reset everything else to be 1
#         new.unit = unit
#         new.self_dt = parent_dt
#         new.parent_unit = unit
#         new.parent_dt = parent_dt
#         return new
#
#     def to_parent(self):
#         """ Create a new timepar with the same units as the parent """
#         unit = self.parent_unit
#         dt = self.parent_dt
#         return self.to(unit=unit, dt=dt)
#
#     def to_json(self):
#         """ Export to JSON """
#         attrs = ['v', 'unit', 'parent_unit', 'parent_dt', 'self_dt', 'factor']
#         out = {'classname': self.__class__.__name__}
#         out.update({attr:getattr(self, attr) for attr in attrs})
#         out['values'] = sc.jsonify(self.values)
#         return out
#
#     # Act like a float -- TODO, add type checking
#     def __add__(self, other): return self.values + other
#     def __sub__(self, other): return self.values - other
#     def __mul__(self, other): return self.asnew().set(v=self.v * other)
#     def __pow__(self, other): return self.values ** other
#     def __truediv__(self, other): return self.asnew().set(v=self.v / other)
#
#     # ...from either side
#     def __radd__(self, other): return other + self.values
#     def __rsub__(self, other): return other - self.values
#     def __rmul__(self, other): return self.asnew().set(v= other * self.v)
#     def __rpow__(self, other): return other ** self.values
#     def __rtruediv__(self, other): return other / self.values # TODO: should be a rate?
#
#     # Handle modify-in-place methods
#     def __iadd__(self, other): return self.set(v=self.v + other)
#     def __isub__(self, other): return self.set(v=self.v - other)
#     def __imul__(self, other): return self.set(v=self.v * other)
#     def __itruediv__(self, other): return self.set(v=self.v / other)
#
#     # Other methods
#     def __neg__(self): return self.asnew().set(v=-self.v)

# class dur(TimePar):
#     """ Any number that acts like a duration """
#     def update_values(self):
#         self.values = self.v*self.factor
#         return self.values
#
#
# def days(v, parent_unit=None, parent_dt=None):
#     """ Shortcut to ss.dur(value, units='day') """
#     return dur(v=v, unit='day', parent_unit=parent_unit, parent_dt=parent_dt)
#
#
# def years(v, parent_unit=None, parent_dt=None):
#     """ Shortcut to ss.dur(value, units='year') """
#     return dur(v=v, unit='year', parent_unit=parent_unit, parent_dt=parent_dt)


# class rate(TimePar):
#     """ Any number that acts like a rate; can be greater than 1 """
#     def update_values(self):
#         self.values = self.v/self.factor
#         return self.values


def perday(v):
    """
    Shortcut to specify rate per calendar day

    :param v:
    :return:
    """
    return Rate(Dur(days=1/v))


def peryear(v):
    """
    Shortcut to specify rate per numeric year

    :param v:
    :return:
    """
    return Rate(YearDur(years=1/v))

#
# class time_prob(Rate):
#     """
#     A probability over time (a.k.a. a cumulative hazard rate); must be >=0 and <=1.
#
#     Note: ``ss.time_prob()`` converts one cumulative hazard rate to another with a
#     different time unit. ``ss.rate_prob()`` converts an exponential rate to a cumulative
#     hazard rate.
#     """
#
#     def update_values(self):
#         v = self.v
#         if self.isarray:
#             self.values = v.copy()
#             inds = np.logical_and(0.0 < v, v < 1.0)
#             if inds.sum():
#                 rates = -np.log(1 - v[inds])
#                 self.values[inds] = 1 - np.exp(-rates/self.factor)
#             invalid = np.logical_or(v < 0.0, 1.0 < v)
#             if invalid.sum():
#
#                 errormsg = f'Invalid value {self.v} for {self}: must be 0-1. If using in a calculation, use .values instead.'
#                 raise ValueError(errormsg)
#         else:
#             if v == 0:
#                 self.values = 0
#             elif v == 1:
#                 self.values = 1
#             elif 0 <= v <= 1:
#                 rate = -np.log(1 - v)
#                 self.values = 1 - np.exp(-rate/self.factor)
#             else:
#                 errormsg = f'Invalid value {self.v} for {self}: must be 0-1. If using in a calculation, use .values instead.'
#                 raise ValueError(errormsg)
#         return self.values
#
#
# class rate_prob(TimePar):
#     """
#     An instantaneous rate converted to a probability; must be >=0.
#
#     Note: ``ss.time_prob()`` converts one cumulative hazard rate to another with a
#     different time unit. ``ss.rate_prob()`` converts an exponential rate to a cumulative
#     hazard rate.
#     """
#     def update_values(self):
#         v = self.v
#         if self.isarray:
#             self.values = v.copy()
#             inds = v > 0.0
#             if inds.sum():
#                 self.values[inds] = 1 - np.exp(-v[inds]/self.factor)
#             invalid = v < 0.0
#             if invalid.sum():
#                 errormsg = f'Invalid value {self.v} for {self}: must be >=0. If using in a calculation, use .values instead.'
#                 raise ValueError(errormsg)
#         else:
#             if v == 0:
#                 self.values = 0
#             elif v > 0:
#                 self.values = 1 - np.exp(-v/self.factor)
#             else:
#                 errormsg = f'Invalid value {self.value} for {self}: must be >=0. If using in a calculation, use .values instead.'
#                 raise ValueError(errormsg)
#         return self.values
#
#
# class beta(time_prob):
#     """ A container for beta (i.e. the disease transmission rate) """
#     pass


if __name__ == '__main__':

    from starsim.time import *   # Import the classes from Starsim so that Dur is an ss.Dur rather than just a bare Dur etc.

    DateDur(weeks=1) - DateDur(days=1)

    Date(2020.1)

    ss.Date(2050) - ss.Date(2020)


    print(Dur(weeks=1)+Dur(1/52))
    print(Dur(1/52)+Dur(weeks=1))
    print(Dur(weeks=1)/Dur(1/52))
    print(DateDur(YearDur(2/52)))

    # Date('2020-01-01') + Dur(weeks=52)  # Should give us 30th December 2020
    Date('2020-01-01') + 2.5*Dur(weeks=1)  # Should give us 30th December 2020
    Date('2020-01-01') + 52*Dur(weeks=1)  # Should give us 30th December 2020
    Date('2020-01-01') + 52*Dur(1/52) # Should give us 1st Jan 2021


    import pickle
    x = Date('2020-01-01')
    s = pickle.dumps(x)
    pickle.loads(s)

    t = Time(Date('2020-01-01'), Date('2020-06-01'), Dur(days=1))
    t = Time(Date('2020-01-01'), Date('2020-06-01'), Dur(months=1)) # Allowed

    t = Time(Dur(days=0), Dur(days=30), Dur(days=1))
    t = Time(Dur(days=0), Dur(months=1), Dur(days=30))
    t = Time(Dur(days=0), Dur(years=1), Dur(weeks=1))
    t = Time(Dur(days=0), Dur(years=1), Dur(months=1)) # Not allowed
    t = Time(Dur(days=0), Dur(years=1), Dur(months=1)) # Not allowed


    t = Time(Dur(0), Dur(1), Dur(1/12))
    t.tvec+Date(2020)

    # Date('2020-01-01')+50*Dur(days=1)
    t = Time(Date('2020-01-01'), Date('2030-06-01'), Dur(days=1))

    t = Time(Date(2020), Date(2030.5), Dur(0.1))
    t.tvec + YearDur(1)

    print(1/YearDur(1))
    print(2/YearDur(1))
    print(4/YearDur(1))
    print(4/DateDur(1))
    print(0.5/DateDur(1))

    print(perday(5)*Dur(days=1))


    # time_prob
    p = RateProb(0.1, Dur(years=1))
    p*Dur(years=2)
    p * Dur(0.5)
    p * Dur(months=1)

    p = RateProb(0.1, Dur(1))
    p*Dur(years=2)
    p * Dur(0.5)
    p * Dur(months=1)

    2/Rate(YearDur(4))
    1/(2*Rate(YearDur(4)))
    Rate(YearDur(2))/Rate(YearDur(1))


    # Dists
    module = sc.objdict(t=sc.objdict(dt=Dur(days=1)))
    d = ss.normal(Dur(days=6), Dur(days=1), module=module, strict=False)
    d.init()
    d.rvs(5)

    module = sc.objdict(t=sc.objdict(dt=Dur(weeks=1)))
    d = ss.normal(Dur(days=6), Dur(days=1), module=module, strict=False)
    d.init()
    d.rvs(5)

    # Another example
    # 5/DateDur(weeks=1)
    # Out[3]: <Rate: per days=1,hours=9,minutes=36> (approx. 259.99999999999994 per year)
    # 259/365
    # Out[4]: 0.7095890410958904
    # 5/7*365
    # Out[5]: 260.7142857142857
    # r = 5/DateDur(weeks=1)
    # r*DateDur(days=1)
    # Out[7]: 0.7142857142857142
    # r*DateDur(days=5)
    # Out[8]: 3.5714285714285707
    # r*DateDur(days=7)
    # Out[9]: 4.999999999999999
    # r*DateDur(weeks=1)