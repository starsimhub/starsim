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
__all__ = ['time_units', 'time_ratio', 'date_add', 'date_diff']

# Define defaults
default_unit = 'year'
default_start_date = '2000-01-01'
default_dur = 50
time_args = ['start', 'stop', 'dt', 'unit'] # Allowable time arguments

# Define available time units
time_units = sc.dictobj(
    day = 1,
    week = 7,
    month = 30.4375, # 365.25/12
    year = 365.25, # For consistency with months
)


def time_ratio(unit1='day', dt1=1.0, unit2='day', dt2=1.0, as_int=False):
    """
    Calculate the relationship between two sets of time factors

    Args:
        unit1 (str): units for the numerator
        dt1 (float): timestep for the numerator
        unit2 (str): units for the denominator
        dt2 (float): timestep for the denominator
        as_int (bool): round and convert to an integer

    **Example**::
        ss.time_ratio(unit1='week', dt1=2, unit2='day', dt2=1, as_int=True) # Returns 14
    """
    if dt1 == dt2:
        dt_ratio = 1.0
    else:
        if dt1 is None or dt2 is None:
            errormsg = f'Cannot convert between dt when one is None ({dt1}, {dt2})'
            raise ValueError(errormsg)
        dt_ratio = dt1/dt2

    if unit1 == unit2:
        unit_ratio = 1.0
    else:
        if unit1 in ['none', None] or unit2 in ['none', None]:
            errormsg = f'Cannot convert between units when one is None ({unit1}, {unit2})'
            raise ValueError(errormsg)
        u1 = time_units[unit1]
        u2 = time_units[unit2]
        unit_ratio = u1/u2

    factor = dt_ratio * unit_ratio
    if as_int:
        factor = int(round(factor))
    return factor


def date_add(start, dur, unit):
    """ Add two dates (or integers) together """
    if sc.isnumber(start):
        stop = start + dur
    else:
        if unit in time_units:
            ndays = int(round(time_units[unit]*dur))
            stop = sc.datedelta(start, days=ndays)
        else:
            errormsg = f'Unknown unit {unit}, choices are: {sc.strjoin(time_units.keys())}'
            raise ValueError(errormsg)
    return stop


def date_diff(start, stop, unit):
    """ Find the difference between two dates (or integers) """
    if sc.isnumber(start) and sc.isnumber(stop):
        dur = stop - start
    else:
        if unit == 'year':
            dur = sc.datetoyear(stop) - sc.datetoyear(start) # TODO: allow non-integer amounts
        elif unit == 'day':
            dur = (stop - start).days
        else:
            raise NotImplementedError
    return dur


def round_tvec(tvec):
    """ Round time vectors to a certain level of precision, to avoid floating point errors """
    decimals = int(-np.log10(ss.options.time_eps))
    tvec = np.round(tvec, decimals=decimals)
    return tvec



#%% Time classes

__all__ += ['date', 'Time']


class date(pd.Timestamp):
    """
    Define a single date; based on ``pd.Timestamp``

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
        single_arg = False
        if len(args) == 1:
            if args[0] is None:
                return pd.Timestamp(None)
            elif sc.isnumber(args[0]):
                single_arg = True
        year_kwarg = len(args) == 0 and len(kwargs) == 1 and 'year' in kwargs
        if single_arg:
            return cls.from_year(args[0])
        if year_kwarg:
            return cls.from_year(kwargs['year'])

        # Otherwise, proceed as normal
        out = super().__new__(cls, *args, **kwargs)
        out.__class__ = cls
        return out

    def __repr__(self):
        return f'<{self.year:04d}-{self.month:02d}-{self.day:02d}>'

    def __str__(self):
        return repr(self)

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
            dateobj = sc.datetoyear(year, reverse=True)
            return cls(dateobj)

    def to_pydate(self):
        """ Convert to datetime.date """
        return self.to_pydatetime().date()

    def to_year(self):
        """
        Convert a date to a floating-point year

        **Examples**::

            ss.date('2020-01-01').to_year() # Returns 2020.0
            ss.date('2024-10-01').to_year() # Returns 2024.7486
        """
        return sc.datetoyear(self.to_pydate())

    def to_pandas(self):
        """ Convert to a standard pd.Timestamp instance """
        return pd.Timestamp(self.to_numpy()) # Need to convert to NumPy first or it doesn't do anything

    @staticmethod
    def _convert_other(other):
        if isinstance(other, du.relativedelta.relativedelta):
            tu = ss.time.time_units
            days = other.days + tu.month*other.months + tu.year*other.years
            int_days = int(round(days))
            other = dt.timedelta(days=int_days)
        if isinstance(other, ss.dur):
            factor = ss.time.time_units[other.unit]
            int_days = int(round(factor*other.v))
            other = dt.timedelta(days=int_days)
        return other

    def __add__(self, other):
        other = self._convert_other(other)
        out = super().__add__(other)
        out = date(out)
        return out

    def __sub__(self, other):
        other = self._convert_other(other)
        out = super().__sub__(other)
        out = date(out)
        return out

    def __radd__(self, other): return self.__add__(other)
    def __iadd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)
    def __isub__(self, other): return self.__sub__(other)


class Time(sc.prettyobj):
    """
    Handle time vectors for both simulations and modules
    """
    def __init__(self, start=None, stop=None, dt=None, unit=None, pars=None, parent=None, init=True):
        self.start = start
        self.stop = stop
        self.dt = dt
        self.unit = unit
        self.update(pars=pars, parent=parent, start=start, stop=stop, dt=dt, unit=unit)
        self.start = self.start if self.is_numeric else date(self.start)
        self.stop  = self.stop  if self.is_numeric else date(self.stop)
        self.unit = validate_unit(self.unit)
        self.ti = 0 # The time index, e.g. 0, 1, 2
        if init:
            self.initialize()
        return

    @property
    def is_numeric(self):
        try:
            return sc.isnumber(self.start)
        except:
            return False

    def update(self, pars=None, parent=None, reset=True, force=None, **kwargs):
        """ Reconcile different ways of supplying inputs """
        pars = sc.mergedicts(pars)
        stale = False

        for key in time_args:
            current_val = getattr(self, key, None)
            parent_val = getattr(parent, key, None)
            kw_val = kwargs.get(key)
            par_val = pars.get(key)

            if force == False: # Only update missing (None) values
                val = sc.ifelse(current_val, kw_val, par_val, parent_val)
            elif force is None: # Prioritize current value
                val = sc.ifelse(kw_val, par_val, current_val, parent_val)
            elif force == True: # Prioritize parent value
                val = sc.ifelse(kw_val, par_val, parent_val, current_val)
            else:
                errormsg = f'Invalid value {force} for force: must be False, None, or True'
                raise ValueError(errormsg)

            if val != current_val:
                setattr(self, key, val)
                stale = True

        if stale and reset:
            self.initialize()
        return

    def initialize(self):
        """ Initialize all vectors """
        # Convert start and stop to dates
        date_start = self.start
        date_stop = self.stop
        date_unit = 'year' if no_unit(self.unit) else self.unit
        dt_year = ss.time_ratio(date_unit, self.dt, 'year', 1.0)
        offset = 0
        if self.is_numeric and date_start == 0:
            date_start = ss.date(ss.time.default_start_date)
            date_stop = date_start + ss.dur(date_stop, unit=date_unit)
            offset = date_start.year

        # If numeric, treat that as the ground truth
        if self.is_numeric:
            timevec = sc.inclusiverange(self.start, self.stop, self.dt)
            yearvec = timevec*dt_year + offset
            datevec = np.array([date(sc.datetoyear(y, reverse=True)) for y in yearvec])

        # Otherwise, use dates as the ground truth
        else:
            if int(self.dt) == self.dt: # The step is integer-like, use exactly
                key = date_unit + 's' # e.g. day -> days
                datelist = sc.daterange(date_start, date_stop, interval={key:int(self.dt)})
            else: # Convert to days instead
                day_delta = time_ratio(unit1=date_unit, dt1=dt, unit2='day', dt2=1.0, as_int=True)
                if day_delta > 0:
                    datelist = sc.daterange(date_start, date_stop, interval={'days':day_delta})
                else:
                    errormsg = f'Timestep {dt} is too small; must be at least 1 day'
                    raise ValueError(errormsg)

            # Tidy
            datevec = np.array([ss.date(d) for d in datelist])
            yearvec = np.array([sc.datetoyear(d.to_pydate()) for d in datevec])
            timevec = datevec

        # Store things
        self.dt_year = dt_year
        self.npts = len(timevec) # The number of points in the sim
        self.tvec = np.arange(self.npts)*self.dt # Absolute time array
        self.timevec = timevec
        self.datevec = datevec
        self.yearvec = yearvec
        return

    def now(self, key=None):
        """
        Get the current simulation time

        Args:
            which (str): which type of time to get: default (None), or "year", "date", or "tvec"

        **Examples**::

            t = ss.Time(start='2021-01-01', stop='2022-02-02', dt=1, unit='week')
            t.ti = 25
            t.now() # Returns <2021-06-25>
            t.now('date') # Returns <2021-06-25>
            t.now('year') # Returns 2021.479
        """
        if key in [None, 'none', 'time']:
            vec = self.timevec
        elif key == 'tvec':
            vec = self.tvec
        elif key == 'date':
            vec = self.datevec
        elif key == 'year':
            vec = self.yearvec
        else:
            errormsg = f'Invalid key "{key}": must be None, abs, date, or year'
            raise ValueError(errormsg)
        return vec[min(self.ti, len(vec)-1)]


#%% TimePar classes

__all__ += ['TimePar', 'dur', 'days', 'years', 'rate', 'perday', 'peryear',
            'time_prob', 'beta', 'rate_prob']

class TimePar(ss.BaseArr):
    """
    Base class for time-aware parameters, durations and rates

    NB, because the factor needs to be recalculated, do not set values directly.
    """
    def __new__(cls, v=None, *args, **kwargs):
        """ Allow TimePars to wrap distributions and return the distributions """

        # Special distribution handling
        if isinstance(v, ss.Dist):
            dist = v
            dist.pars[0] = cls(dist.pars[0], *args, **kwargs) # Convert the first parameter to a TimePar (the same scale is applied to all parameters)
            return dist

        # Otherwise, do the usual initialization
        else:
            return super().__new__(cls)

    def __init__(self, v, unit=None, parent_unit=None, parent_dt=None, self_dt=1.0):
        self.v = v
        self.unit = unit
        self.parent_unit = parent_unit
        self.parent_dt = parent_dt
        self.self_dt = self_dt
        self.factor = None
        self.values = None
        self.initialized = False
        self.validate_units()
        return

    def validate_units(self):
        """ Check that the units entered are valid """
        try:
            self.unit = unit_mapping[self.unit]
        except KeyError:
            errormsg = f'Invalid unit "{self.unit}"; must be one of: {sc.strjoin(time_units.keys())}'
            raise ValueError(errormsg)
        try:
            self.parent_unit = unit_mapping[self.parent_unit]
        except KeyError:
            errormsg = f'Invalid parent unit "{self.parent_unit}"; must be one of: {sc.strjoin(time_units.keys())}'
            raise ValueError(errormsg)
        return

    def init(self, parent=None, parent_unit=None, parent_dt=None, update_values=True, die=True):
        """ Link to the sim and/or module units """
        if parent is None:
            parent = sc.dictobj(unit=parent_unit, dt=parent_dt)
        else:
            if parent_dt is not None:
                errormsg = f'Cannot override parent {parent} by setting parent_dt; set in parent object instead'
                raise ValueError(errormsg)

        if parent.unit is not None:
            self.parent_unit = parent.unit

        if parent.dt is not None:
           self.parent_dt = parent.dt

        # Set defaults if not yet set
        self.unit = sc.ifelse(self.unit, self.parent_unit) # If unit isn't defined but parent is, set to parent
        self.parent_unit = sc.ifelse(self.parent_unit, self.unit) # If parent isn't defined but unit is, set to self
        self.parent_dt = sc.ifelse(self.parent_dt, self.self_dt, 1.0) # If dt isn't defined, assume 1 (self_dt is defined by default)

        # Calculate the actual conversion factor to be used in the calculations
        self.update_cached(update_values=update_values, die=die)
        self.initialized = True
        self.validate_units()
        return self

    def __repr__(self):
        name = self.__class__.__name__

        if self.initialized:
            if self.factor == 1.0:
                xstr = ''
            else:
                xstr = f', values={self.values}'
        else:
            xstr = ', initialized=False'

        if (self.parent_unit is not None) and (self.unit != self.parent_unit):
            parentstr = f', parent={self.parent_unit}'
        else:
            parentstr = ''

        default_dt = sc.ifelse(self.self_dt, 1.0) == 1.0
        if not default_dt:
            dtstr = f', self_dt={self.self_dt}'
        else:
            dtstr = ''

        # Rather than ss.dur(3, unit='day'), dispaly as ss.days(3)
        prefixstr = 'ss.'
        key = (name, self.unit)
        mapping = {
            ('dur',  'day'):  'days',
            ('dur',  'year'): 'years',
            ('rate', 'day'):  'perday',
            ('rate', 'year'): 'peryear',
        }

        if key in mapping and default_dt:
            prefixstr += mapping[key]
            unitstr = ''
        else:
            prefixstr += name
            unitstr = f', unit={self.unit}'

        suffixstr = unitstr + parentstr + dtstr + xstr

        return f'{prefixstr}({self.v}{suffixstr})'

    @property
    def isarray(self):
        return isinstance(self.v, np.ndarray)

    def set(self, v=None, unit=None, parent_unit=None, parent_dt=None, self_dt=None, force=False):
        """ Set the specified parameter values (ignoring None values) and update stored values """
        if v           is not None: self.v           = v
        if unit        is not None: self.unit        = unit
        if parent_unit is not None: self.parent_unit = parent_unit
        if parent_dt   is not None: self.parent_dt   = parent_dt
        if self_dt     is not None: self.self_dt     = self_dt
        if self.initialized or force: # Don't try to set these unless it's been initialized
            self.update_cached()
        self.validate_units()
        return self

    def update_cached(self, update_values=True, die=True):
        """ Update the cached factor and values """
        try:
            self.update_factor()
            if update_values:
                self.update_values()
        except TypeError as E: # For a known error, skip silently if die=False
            if die:
                errormsg = f'Update failed for {self}. Argument v={self.v} should be a number or array; if a function, use update_values=False. Error: {E}'
                raise TypeError(errormsg) from E
        except Exception as E: # For other errors, raise a warning
            if die:
                raise E
            else:
                tb = sc.traceback(E)
                warnmsg = f'Uncaught error encountered while updating {self}, but die=False. Traceback:\n{tb}'
                ss.warn(warnmsg)

        return self

    def update_factor(self):
        """ Set factor used to multiply the value to get the output """
        self.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=self.parent_unit, dt2=self.parent_dt)
        return

    def update_values(self):
        """ Convert from self.v to self.values based on self.factor -- must be implemented by derived classes """
        raise NotImplementedError

    def to(self, unit=None, dt=None):
        """ Create a new timepar based on the current one but with a different unit and/or dt """
        new = self.asnew()
        unit = sc.ifelse(unit, self.parent_unit, self.unit)
        parent_dt = sc.ifelse(dt, 1.0)
        new.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=unit, dt2=parent_dt) # Calculate the new factor
        new.update_values() # Update values
        new.v = new.values # Reset the base value
        new.factor = 1.0 # Reset everything else to be 1
        new.unit = unit
        new.self_dt = parent_dt
        new.parent_unit = unit
        new.parent_dt = parent_dt
        return new

    def to_parent(self):
        """ Create a new timepar with the same units as the parent """
        unit = self.parent_unit
        dt = self.parent_dt
        return self.to(unit=unit, dt=dt)

    def to_json(self):
        """ Export to JSON """
        attrs = ['v', 'unit', 'parent_unit', 'parent_dt', 'self_dt', 'factor']
        out = {'classname': self.__class__.__name__}
        out.update({attr:getattr(self, attr) for attr in attrs})
        out['values'] = sc.jsonify(self.values)
        return out

    # Act like a float -- TODO, add type checking
    def __add__(self, other): return self.values + other
    def __sub__(self, other): return self.values - other
    def __mul__(self, other): return self.asnew().set(v=self.v * other)
    def __pow__(self, other): return self.values ** other
    def __truediv__(self, other): return self.asnew().set(v=self.v / other)

    # ...from either side
    def __radd__(self, other): return other + self.values
    def __rsub__(self, other): return other - self.values
    def __rmul__(self, other): return self.asnew().set(v= other * self.v)
    def __rpow__(self, other): return other ** self.values
    def __rtruediv__(self, other): return other / self.values # TODO: should be a rate?

    # Handle modify-in-place methods
    def __iadd__(self, other): return self.set(v=self.v + other)
    def __isub__(self, other): return self.set(v=self.v - other)
    def __imul__(self, other): return self.set(v=self.v * other)
    def __itruediv__(self, other): return self.set(v=self.v / other)

    # Other methods
    def __neg__(self): return self.asnew().set(v=-self.v)


class dur(TimePar):
    """ Any number that acts like a duration """
    def update_values(self):
        self.values = self.v*self.factor
        return self.values


def days(v, parent_unit=None, parent_dt=None):
    """ Shortcut to ss.dur(value, units='day') """
    return dur(v=v, unit='day', parent_unit=parent_unit, parent_dt=parent_dt)


def years(v, parent_unit=None, parent_dt=None):
    """ Shortcut to ss.dur(value, units='year') """
    return dur(v=v, unit='year', parent_unit=parent_unit, parent_dt=parent_dt)


class rate(TimePar):
    """ Any number that acts like a rate; can be greater than 1 """
    def update_values(self):
        self.values = self.v/self.factor
        return self.values


def perday(v, parent_unit=None, parent_dt=None):
    """ Shortcut to ss.rate(value, units='day') """
    return rate(v=v, unit='day', parent_unit=parent_unit, parent_dt=parent_dt)


def peryear(v, parent_unit=None, parent_dt=None):
    """ Shortcut to ss.rate(value, units='year') """
    return rate(v=v, unit='year', parent_unit=parent_unit, parent_dt=parent_dt)


class time_prob(TimePar):
    """
    A probability over time (a.k.a. a cumulative hazard rate); must be >=0 and <=1.

    Note: ``ss.time_prob()`` converts one cumulative hazard rate to another with a
    different time unit. ``ss.rate_prob()`` converts an exponential rate to a cumulative
    hazard rate.
    """
    def update_values(self):
        v = self.v
        if self.isarray:
            self.values = v.copy()
            inds = np.logical_and(0.0 < v, v < 1.0)
            if inds.sum():
                rates = -np.log(1 - v[inds])
                self.values[inds] = 1 - np.exp(-rates/self.factor)
            invalid = np.logical_or(v < 0.0, 1.0 < v)
            if invalid.sum():
                errormsg = f'Invalid value {self.v} for {self}: must be 0-1. If using in a calculation, use .values instead.'
                raise ValueError(errormsg)
        else:
            if v == 0:
                self.values = 0
            elif v == 1:
                self.values = 1
            elif 0 <= v <= 1:
                rate = -np.log(1 - v)
                self.values = 1 - np.exp(-rate/self.factor)
            else:
                errormsg = f'Invalid value {self.v} for {self}: must be 0-1. If using in a calculation, use .values instead.'
                raise ValueError(errormsg)
        return self.values


class rate_prob(TimePar):
    """
    An instantaneous rate converted to a probability; must be >=0.

    Note: ``ss.time_prob()`` converts one cumulative hazard rate to another with a
    different time unit. ``ss.rate_prob()`` converts an exponential rate to a cumulative
    hazard rate.
    """
    def update_values(self):
        v = self.v
        if self.isarray:
            self.values = v.copy()
            inds = v > 0.0
            if inds.sum():
                self.values[inds] = 1 - np.exp(-v[inds]/self.factor)
            invalid = v < 0.0
            if invalid.sum():
                errormsg = f'Invalid value {self.v} for {self}: must be >=0. If using in a calculation, use .values instead.'
                raise ValueError(errormsg)
        else:
            if v == 0:
                self.values = 0
            elif v > 0:
                self.values = 1 - np.exp(-v/self.factor)
            else:
                errormsg = f'Invalid value {self.value} for {self}: must be >=0. If using in a calculation, use .values instead.'
                raise ValueError(errormsg)
        return self.values


class beta(time_prob):
    """ A container for beta (i.e. the disease transmission rate) """
    pass


#%% Final helper functions

# Map different units onto the time units -- placed at the end to include the functions
unit_mapping_reverse = {
    'none': [None, 'none'],
    'day': ['d', 'day', 'days', 'perday', days, perday],
    'year': ['y', 'yr', 'year', 'years', 'peryear', years, peryear],
    'week': ['w', 'wk', 'week', 'weeks'],
    'month': ['m', 'mo', 'month', 'months'],
}
unit_mapping = {v:k for k,vlist in unit_mapping_reverse.items() for v in vlist}


def validate_unit(unit):
    try:
        unit = unit_mapping[unit]
    except KeyError as E:
        errormsg = f'Invalid unit "{unit}". Valid units are:\n{sc.pp(unit_mapping_reverse, output=True)}'
        raise sc.KeyNotFoundError(errormsg) from E
    return unit


def no_unit(unit):
    """ Check if the unit is None-like """
    return unit in unit_mapping_reverse['none']