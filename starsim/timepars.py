"""
Functions and classes for handling time
"""
import numpy as np
import sciris as sc
import starsim as ss

# Classes that are externally visible
__all__ = ['time_units', 'time_ratio', 'date_add', 'date_diff', 'make_timevec', 'make_abs_tvec',
           'TimePar', 'dur', 'days', 'years', 'rate', 'perday', 'peryear', 'time_prob', 'beta']


#%% Helper functions

# Define available time units
time_units = sc.dictobj(
    day = 1,
    week = 7,
    month = 30.4375, # 365.25/12 -- more accurate and nicer fraction
    year = 365, # For simplicity with days
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
        if unit1 is None or unit2 is None:
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


def make_timevec(start, stop, dt, unit):
    """ Parse start, stop, and dt into an appropriate time vector """
    if sc.isnumber(start):
        try:
            timevec = sc.inclusiverange(start=start, stop=stop, step=dt) # The time points of the sim
        except Exception as E:
            errormsg = f'Incompatible set of time inputs: start={start}, stop={stop}, dt={dt}. You can use dates or numbers but not both.'
            raise ValueError(errormsg) from E
    else:
        day_delta = time_ratio(unit1=unit, dt1=dt, unit2='day', dt2=1.0, as_int=True)
        if day_delta > 0:
            timevec = sc.daterange(start, stop, interval={'days':day_delta})
        else:
            errormsg = f'Timestep {dt} is too small; must be at least 1 day'
            raise ValueError(errormsg)
    return timevec


def make_abs_tvec(tv, unit, sim_unit):
    """ Convert a module time vector into a numerical time array with the same units as the sim """

    # It's an array of days or years: easy
    if sc.isarray(tv):
        ratio = time_ratio(unit1=unit, unit2=sim_unit)
        abstv = tv*ratio # Get the units right
        abstv -= abstv[0] # Start at 0

    # It's a date: convert to fractional years and then subtract the
    else:
        yearvec = [sc.datetoyear(d) for d in tv]
        absyearvec = np.array(yearvec) - yearvec[0] # Subtract start date
        abstv = absyearvec*time_ratio(unit1='year', unit2=sim_unit)

    # Round to the value of epsilon; alternative to np.round(abstv/eps)*eps, which has floating point error
    decimals = int(-np.log10(ss.options.time_eps))
    abstv = np.round(abstv, decimals=decimals)

    return abstv


#%% Time classes

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
        self.parent_name = None
        self.initialized = False
        return

    def init(self, parent=None, parent_unit=None, parent_dt=None, update_values=True, die=True):
        """ Link to the sim and/or module units """
        if parent is None:
            parent = sc.dictobj(unit=parent_unit, dt=parent_dt)
        else:
            if parent_dt is not None:
                errormsg = f'Cannot override parent {parent} by setting parent_dt; set in parent object instead'
                raise ValueError(errormsg)
            self.parent_name = parent.__class__.__name__ # TODO: or parent.name for Starsim objects?

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

        return f'ss.{name}({self.v}, unit={self.unit}{parentstr}{xstr})'

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
        new.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=unit, dt2=parent_dt) # Calculaet the new factor
        new.update_values() # Update values
        new.v = new.values # Reset the base value
        new.factor = 1.0 # Reset everything else to be 1
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
        attrs = ['v', 'unit', 'parent_unit', 'parent_dt', 'parent_name', 'self_dt', 'factor']
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


class days(dur):
    """ Shortcut to ss.dur(value, units='day') """
    def __init__(self, v, parent_unit=None, parent_dt=None):
        super().__init__(v=v, unit='day', parent_unit=parent_unit, parent_dt=parent_dt)
        return


class years(dur):
    """ Shortcut to ss.dur(value, units='year') """
    def __init__(self, v, parent_unit=None, parent_dt=None):
        super().__init__(v=v, unit='year', parent_unit=parent_unit, parent_dt=parent_dt)
        return


class rate(TimePar): # TODO: should all rates just be time_prob?
    """ Any number that acts like a rate; can be greater than 1 """
    def update_values(self):
        self.values = self.v/self.factor
        return self.values


class perday(rate):
    """ Shortcut to ss.rate(value, units='day') """
    def __init__(self, v, parent_unit=None, parent_dt=None):
        super().__init__(v=v, unit='day', parent_unit=parent_unit, parent_dt=parent_dt)
        return


class peryear(rate):
    """ Shortcut to ss.rate(value, units='year') """
    def __init__(self, v, parent_unit=None, parent_dt=None):
        super().__init__(v=v, unit='year', parent_unit=parent_unit, parent_dt=parent_dt)
        return


class time_prob(TimePar):
    """ A probability over time (a.k.a. a cumulative hazard rate); must be >0 and <1 """
    def update_values(self):
        v = self.v
        if self.isarray:
            self.values = v.copy()
            inds = np.logical_and(0.0 < v, v < 1.0)
            if inds.sum():
                rates = -np.log(1 - v[inds])
                self.values[inds] = 1 - np.exp(-rates/self.factor)
        else:
            if v == 0:
                self.values = 0
            elif v == 1:
                self.values = 1
            elif 0 <= v <= 1:
                rate = -np.log(1 - v)
                self.values = 1 - np.exp(-rate/self.factor)
            else:
                errormsg = f'Invalid value {self.value} for {self}: must be 0-1'
                raise ValueError(errormsg)
        return self.values


class beta(time_prob):
    """ A container for beta (i.e. the disease transmission rate) """
    pass
