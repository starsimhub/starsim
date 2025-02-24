"""
Functions and classes for handling time
"""
import copy
import sciris as sc
import numpy as np
import pandas as pd
import datetime as dt
import dateutil as du
import starsim as ss


#%% Helper functions

# Classes and objects that are externally visible
__all__ = ['time_units', 'time_ratio', 'date_add', 'date_diff']

# Allowable time arguments
time_args = ['start', 'stop', 'dt', 'unit']

# Define available time units
time_units = sc.objdict(
    day   = 1.0,
    week  = 7.0,
    month = 30.4375, # 365.25/12
    year  = 365.25, # For consistency with months
)

# Define defaults
default_dur = 50
default_unit = 'year'
default_start_year = 2000
default_start_date = '2000-01-01'
default_start = sc.objdict(
    {k:default_start_date for k in ['day', 'week', 'month']} |
    {k:default_start_year for k in ['year', 'unitless']}
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
        if is_unitless(unit1) or is_unitless(unit2):
            errormsg = f'Cannot convert between units if only one is unitless (unit1={unit1}, unit2={unit2})'
            raise ValueError(errormsg)
        if unit1 is None or unit2 is None:
            errormsg = f'Cannot convert between units when only one has been initialized (unit1={unit1}, unit2={unit2})'
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
        else:
            dur = (ss.date(stop) - ss.date(start)).days
            if unit != 'day':
                dur *= time_ratio(unit1='day', unit2=unit)
    return dur


def round_tvec(tvec):
    """ Round time vectors to a certain level of precision, to avoid floating point errors """
    decimals = int(-np.log10(ss.options.time_eps))
    tvec = np.round(np.array(tvec), decimals=decimals)
    return tvec


def years_to_dates(yearvec):
    """ Convert a numeric year vector to a date vector """
    datevec = np.array([date(sc.datetoyear(y, reverse=True)) for y in yearvec])
    return datevec


def dates_to_years(datevec):
    """ Convert a date vector to a numeric year vector"""
    yearvec = round_tvec([sc.datetoyear(d.date()) for d in datevec])
    return yearvec


def dates_to_days(datevec, start_date):
    """ Convert a date vector into relative days since start date """
    start_date = date(start_date)
    dayvec = np.array([(d - start_date).days for d in datevec])
    return dayvec


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
        """ Check if a year was supplied, and preprocess it; complex due to pd.Timestamp implementation """
        single_year_arg = False
        if len(args) == 1:
            arg = args[0]
            if arg is None:
                return pd.Timestamp(None)
            elif sc.isnumber(arg):
                single_year_arg = True
            elif isinstance(arg, pd.Timedelta):
                return pd.Timedelta(arg)
        year_kwarg = len(args) == 0 and len(kwargs) == 1 and 'year' in kwargs
        if single_year_arg:
            return cls.from_year(args[0])
        if year_kwarg:
            return cls.from_year(kwargs['year'])

        # Otherwise, proceed as normal
        out = super(date, cls).__new__(cls, *args, **kwargs)
        out = cls._reset_class(out)
        return out

    def __copy__(self):
        """ Required due to pd.Timestamp implementation; pd.Timestamp is immutable, so create new object """
        out = self.__class__(self)
        return out

    def __deepcopy__(self, *args, **kwargs):
        """ Required due to pd.Timestamp implementation; pd.Timestamp is immutable, so create new object """
        out = self.__class__(self)
        return out

    @classmethod
    def _reset_class(cls, obj):
        """ Manually reset the class from pd.Timestamp to ss.date """
        try:
            obj.__class__ = date
        except:
            warnmsg = f'Unable to convert {obj} to ss.date(); proceeding with pd.Timestamp'
            ss.warn(warnmsg)
        return obj

    def __repr__(self, bracket=True):
        """ Show the date in brackets, e.g. <2024.04.04> """
        _ = ss.options.date_sep
        y = f'{self.year:04d}'
        m = f'{self.month:02d}'
        d = f'{self.day:02d}'
        string = y + _ + m + _ + d
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

        **Examples**::

            ss.date.from_year(2020) # Returns <2020-01-01>
            ss.date.from_year(2024.75) # Returns <2024-10-01>
        """
        if isinstance(year, int):
            return cls(year=year, month=1, day=1)
        else:
            dateobj = sc.datetoyear(year, reverse=True)
            return cls(dateobj)

    def to_year(self):
        """
        Convert a date to a floating-point year

        **Examples**::

            ss.date('2020-01-01').to_year() # Returns 2020.0
            ss.date('2024-10-01').to_year() # Returns 2024.7486
        """
        return sc.datetoyear(self.date())

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
        elif isinstance(other, ss.dur):
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
    def __rsub__(self, other): return self.__sub__(other) # TODO: check if this should be reversed
    def __isub__(self, other): return self.__sub__(other)


class Time(sc.prettyobj):
    """
    Handle time vectors for both simulations and modules.

    Args:
        start (float/str/date): the start date for the simulation/module
        stop (float/str/date): the end date for the simulation/module
        dt (float): the step size, in units of "unit"
        unit (str): the time unit; choices are "day", "week", "month", "year", or "unitless"
        pars (dict): if provided, populate parameter values from this dictionary
        parent (obj): if provided, populate missing parameter values from a 'parent" ``Time`` instance
        name (str): if provided, name the ``Time`` object
        init (bool): whether or not to immediately initialize the Time object
        sim (bool/Sim): if True, initializes as a sim-specific ``Time`` instance; if a Sim instance, initialize the absolute time vector

    The ``Time`` object, after initialization, has the following attributes:

    - ``ti`` (int): the current timestep
    -  ``dt_year`` (float): the timestep in units of years
    - ``npts`` (int): the number of timesteps
    - ``tvec`` (array): time starting at 0, in self units (e.g. ``[0, 0.1, 0.2, ... 10.0]`` if start=0, stop=10, dt=0.1)
    - ``absvec`` (array): time relative to sim start, in units of sim units (e.g. ``[366, 373, 380, ...]`` if sim-start=2001, start=2002, sim-unit='day', unit='week')
    - ``yearvec`` (array): time represented as floating-point years (e.g. ``[2000, 2000.1, 2000.2, ... 2010.0]`` if start=2000, stop=2010, dt=0.1)
    - ``datevec`` (array): time represented as an array of ``ss.date`` objects (e.g. ``[<2000.01.01>, <2000.02.07>, ... <2010.01.01>]`` if start=2000, stop=2010, dt=0.1)
    - ``timevec`` (array): the "native" time vector, which always matches one of ``tvec``, ``yearvec``, or ``datevec``

    **Examples**::

        t1 = ss.Time(start=2000, stop=2020, dt=1.0, unit='year') # Years, numeric units
        t2 = ss.Time(start='2021-01-01', stop='2021-04-04', dt=2.0, unit='day') # Days, date units
    """
    def __init__(self, start=None, stop=None, dt=None, unit=None, pars=None, parent=None,
                 name=None, init=True, sim=None):
        self.name = name
        self.start = start
        self.stop = stop
        self.dt = dt
        self.unit = unit
        self.ti = 0 # The time index, e.g. 0, 1, 2

        # Prepare for later initialization
        self.dt_year = None
        self.npts    = None
        self.tvec    = None
        self.timevec = None
        self.datevec = None
        self.yearvec = None
        self.abstvec = None
        self.initialized = False

        # Finalize
        self.update(pars=pars, parent=parent)
        if init and self.ready:
            self.init(sim=sim)
        return

    def __bool__(self):
        """ Always truthy """
        return True

    def __len__(self):
        """ Length is the number of timepoints """
        return sc.ifelse(self.npts, 0)

    @property
    def ready(self):
        """ Check if all parameters are in place to be initialized """
        return not any([getattr(self, k) is None for k in time_args])

    @property
    def is_numeric(self):
        """ Check whether the fundamental simulation unit is numeric (as opposed to date-based) """
        try:
            return sc.isnumber(self.start)
        except:
            return False

    @property
    def is_unitless(self):
        return is_unitless(self.unit)

    def __setstate__(self, state):
        """ Custom setstate to unpickle ss.date() instances correctly """
        self.__dict__.update(state)
        self._convert_timestamps()
        return

    def _convert_timestamps(self):
        """ Replace pd.Timestamp instances with ss.date(); required due to pandas limitations with pickling """
        objs = [self.start, self.stop]
        objs += sc.tolist(self.datevec, coerce='full')
        objs += sc.tolist(self.timevec, coerce='full')
        objs = [obj for obj in objs if type(obj) == pd.Timestamp]
        for obj in objs:
            date._reset_class(obj)
        return

    def update(self, pars=None, parent=None, reset=True, force=None, **kwargs):
        """ Reconcile different ways of supplying inputs """
        pars = sc.mergedicts(pars)
        stale = False

        for key in time_args:
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
        self.unit = validate_unit(self.unit)

        # Copy missing values from sim
        if isinstance(sim, ss.Sim):
            self.unit = sc.ifelse(self.unit, sim.t.unit)
            if self.unit == sim.t.unit: # Units match, use directly
                sim_dt = sim.t.dt
                sim_start = sim.t.start
                sim_stop = sim.t.stop
            else: # Units don't match, use datevec instead
                sim_dt = 1.0 # Don't try to reset the dt if the units don't match
                sim_start = sim.t.datevec[0]
                sim_stop = sim.t.datevec[-1]
            self.dt = sc.ifelse(self.dt, sim_dt)
            self.start = sc.ifelse(self.start, sim_start)
            self.stop = sc.ifelse(self.stop, sim_stop)

        # Handle start and stop
        self.start = self.start if self.is_numeric else date(self.start)
        self.stop  = self.stop  if self.is_numeric else date(self.stop)

        # Convert start and stop to dates
        date_start = self.start
        date_stop = self.stop
        date_unit = 'year' if not has_units(self.unit) else self.unit # Use year by default
        dt_year = time_ratio(unit1=date_unit, dt1=self.dt, unit2='year', dt2=1.0) # Timestep in units of years
        offset = 0
        if self.is_numeric and date_start == 0:
            date_start = ss.date(ss.time.default_start[date_unit])
            date_stop = date_start + ss.dur(date_stop, unit=date_unit)
            offset = date_start.year

        # If numeric, treat that as the ground truth
        if self.is_numeric:
            ratio = time_ratio(unit1=date_unit, unit2='year')
            timevec = round_tvec(sc.inclusiverange(self.start, self.stop, self.dt))
            yearvec = round_tvec((timevec-timevec[0])*ratio + offset + timevec[0]) # TODO: simplify
            datevec = years_to_dates(yearvec)

        # If unitless, just use that
        elif self.is_unitless:
            timevec = round_tvec(sc.inclusiverange(self.start, self.stop, self.dt))
            yearvec = round_tvec(timevec)
            datevec = timevec

        # If the unit is years, handle that
        elif date_unit == 'year': # For years, the yearvec is the most robust representation
            start_year = sc.datetoyear(date_start.date())
            stop_year = sc.datetoyear(date_stop.date())
            yearvec = round_tvec(sc.inclusiverange(start_year, stop_year, self.dt))
            datevec = years_to_dates(yearvec)
            timevec = datevec

        # Otherwise, use dates as the ground truth
        else:
            if int(self.dt) == self.dt: # The step is integer-like, use exactly
                key = date_unit + 's' # e.g. day -> days
                datelist = sc.daterange(date_start, date_stop, interval={key:int(self.dt)})
            else: # Convert to the sim unit instead
                day_delta = time_ratio(unit1=date_unit, dt1=self.dt, unit2='day', dt2=1.0, as_int=True)
                if day_delta >= 1:
                    datelist = sc.daterange(date_start, date_stop, interval={'days':day_delta})
                else:
                    errormsg = f'Timestep {dt} is too small; must be at least 1 day'
                    raise ValueError(errormsg)
            datevec = np.array([ss.date(d) for d in datelist])
            yearvec = dates_to_years(datevec)
            timevec = datevec

        # Store things
        self.dt_year = dt_year
        self.npts = len(timevec) # The number of points in the sim
        self.tvec = round_tvec(np.arange(self.npts)*self.dt) # Absolute time array
        self.timevec = timevec
        self.datevec = datevec
        self.yearvec = yearvec
        if sim == True: # It's the sim itself, the tvec is the absolute time vector
            self.abstvec = self.tvec
        elif sim is not None:
            self.make_abstvec(sim)
        else:
            self.abstvec = None # Intentionally set to None, cannot be used in the sim loop until populated
        self.initialized = True
        return

    def make_abstvec(self, sim):
        """ Convert the current time vector into sim units """
        # Validation
        if self.is_unitless != sim.t.is_unitless:
            errormsg = f'Cannot mix units with unitless time: sim.unit={sim.t.unit} {self.name}.unit={self.unit}'
            raise ValueError(errormsg)

        # Both are unitless or numeric
        both_unitless = self.is_unitless and sim.t.is_unitless
        both_numeric = self.is_numeric and sim.t.is_numeric
        if both_unitless or both_numeric:
            abstvec = self.tvec.copy() # Start by copying the current time vector
            ratio = time_ratio(unit1=self.unit, dt1=1.0, unit2=sim.t.unit, dt2=1.0) # tvec has sim units, but not dt
            if ratio != 1.0:
                abstvec *= ratio # TODO: CHECK
            start_diff = self.start - sim.t.start
            if start_diff != 0.0:
                abstvec += start_diff

        # The sim uses years; use yearvec
        elif sim.t.unit == 'year':
            abstvec = self.yearvec.copy()
            abstvec -= sim.t.yearvec[0] # Start relative to sim start

        # Otherwise (days, weeks, months), use datevec and convert to days
        else:
            dayvec = dates_to_days(self.datevec, start_date=sim.t.datevec[0])
            ratio = time_ratio(unit1='day', dt1=1.0, unit2=sim.t.unit, dt2=1.0)
            abstvec = dayvec*ratio # Convert into sim time units

        self.abstvec = round_tvec(abstvec) # Avoid floating point inconsistencies
        return

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

        ti = np.clip(self.ti, 0, len(vec)-1)
        now = vec[ti]
        if key == 'str':
            now = f'{now:0.1f}' if isinstance(now, float) else str(now)
        return now


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
            if len(args):
                errormsg = f'When wrapping a distribution with a TimePar, args not allowed ({args}); use kwargs'
                raise ValueError(errormsg)
            dist = v
            dist.pars[0] = cls(dist.pars[0], **kwargs) # Convert the first parameter to a TimePar (the same scale is applied to all parameters)
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
        """ Check if the value is an array """
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


#%% Unit handling

# Map different units onto the time units -- placed at the end to include the functions
unit_mapping_reverse = {
    None: [None],
    'unitless': ['unitless', 'none'],
    'day': ['d', 'day', 'days', 'perday', days, perday],
    'year': ['y', 'yr', 'year', 'years', 'peryear', years, peryear],
    'week': ['w', 'wk', 'week', 'weeks'],
    'month': ['m', 'mo', 'month', 'months'],
}
unit_mapping = {v:k for k,vlist in unit_mapping_reverse.items() for v in vlist}


def validate_unit(unit):
    """ Check that the unit is valid, and convert to a standard type """
    try:
        unit = unit_mapping[unit]
    except KeyError as E:
        errormsg = f'Invalid unit "{unit}". Valid units are:\n{sc.pp(unit_mapping_reverse, output=True)}'
        raise sc.KeyNotFoundError(errormsg) from E
    return unit


def is_unitless(unit):
    """ Check if explicitly unitless (excludes None; use not has_units() for that) """
    return unit in unit_mapping_reverse['unitless']


def has_units(unit):
    """ Check that the unit is valid and is not unitless or None """
    unit = validate_unit(unit)
    if unit is None:
        return False
    elif is_unitless(unit):
        return False
    else:
        return True
