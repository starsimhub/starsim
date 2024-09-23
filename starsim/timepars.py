"""
Functions and classes for handling time
"""

import numpy as np
import sciris as sc

# Classes that are externally visible
__all__ = ['time_units', 'time_ratio', 'date_add', 'date_diff', 'make_timevec', 'make_timearray',
           'TimePar', 'dur', 'days', 'years', 'rate', 'time_prob', 'beta']

    
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


def make_timearray(tv, unit, sim_unit):
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
        
    return abstv


#%% Time classes

class TimePar:
    """
    Base class for time-aware parameters, durations and rates
    
    NB, because the factor needs to be recalculated, do not set values directly.
    """
    def __init__(self, value, unit=None, parent_unit=None, parent_dt=None):
        self.value = value
        self.unit = unit
        self.parent_unit = parent_unit
        self.parent_dt = parent_dt
        self.factor = None
        self.x = np.nan
        self.parent_name = None
        self.initialized = False
        return
    
    def init(self, parent=None, parent_unit=None, parent_dt=None):
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
        self.unit = sc.ifelse(self.unit, self.parent_unit)
        self.parent_dt = sc.ifelse(self.parent_dt, 1.0)
        
        # Calculate the actual conversion factor to be used in the calculations
        self.set_factor()
        self.set_x()
        self.initialized = True
        return self
        
    def __repr__(self):
        name = self.__class__.__name__
        if self.initialized:
            if self.x == self.value:
                xstr = ''
            else:
                xstr = f', x={self.x:n}'
        else:
            xstr = ', initialized=False'
        return f'ss.{name}({self.value}, unit={self.unit}{xstr})'
    
    def disp(self):
        return sc.pr(self)
    
    def set(self, value=None, unit=None, parent_unit=None, parent_dt=None):
        """ Reset the parameter values """
        if value       is not None: self.value       = value
        if unit        is not None: self.unit        = unit
        if parent_unit is not None: self.parent_unit = parent_unit
        if parent_dt   is not None: self.parent_dt   = parent_dt
        if self.initialized: # Don't try to set these unless it's been initialized
            self.set_factor()
            self.set_x()
        return self
    
    def set_factor(self):
        """ Set factor used to multiply the value to get the output """
        self.factor = time_ratio(unit1=self.unit, dt1=1.0, unit2=self.parent_unit, dt2=self.parent_dt)
        return
    
    # Act like a float
    def __add__(self, other): return self.x + other
    def __sub__(self, other): return self.x - other
    def __mul__(self, other): return self.x * other
    def __pow__(self, other): return self.x ** other
    def __truediv__(self, other): return self.x / other
    
    # ...from either side
    def __radd__(self, other): return other + self.x
    def __rsub__(self, other): return other - self.x
    def __rmul__(self, other): return other * self.x
    def __rpow__(self, other): return other ** self.x
    def __rtruediv__(self, other): return other / self.x
    
    # Handle modify-in-place methods
    def __iadd__(self, other): self.value += other; return self
    def __isub__(self, other): self.value -= other; return self
    def __imul__(self, other): self.value *= other; return self
    def __itruediv__(self, other): self.value /= other; return self
    
    # Other methods
    def __neg__(self): return -self.x
    
    # Unfortunately, floats don't define the above methods, so we can't *just* use this
    def __getattr__(self, attr):
        """ Make it behave like a regular float mostly """
        if attr in ['__deepcopy__', '__getstate__', '__setstate__']:
            return self.__getattribute__(attr)
        else:
            return getattr(self.x, attr)
        
    # Similar to Arr -- required for doing efficient array operations
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        args = [(arg.x if arg is self else arg) for arg in args] # Use self.x for the operation
        if method == '__call__':
            return ufunc(*args, **kwargs)
        else:
            return self.x.__array_ufunc__(ufunc, method, *args, **kwargs) # Probably not needed


class dur(TimePar):
    """ Any number that acts like a duration """
    def set_x(self):
        self.x = self.value*self.factor
        return


class days(dur):
    """ Shortcut to ss.dur(value, units='day') """
    def __init__(self, value, parent_unit=None, parent_dt=None):
        super().__init__(value=value, unit='day', parent_unit=parent_unit, parent_dt=parent_dt)
        return


class years(dur):
    """ Shortcut to ss.dur(value, units='year') """
    def __init__(self, value, parent_unit=None, parent_dt=None):
        super().__init__(value=value, unit='year', parent_unit=parent_unit, parent_dt=parent_dt)
        return


class rate(TimePar):
    """ Any number that acts like a rate; can be greater than 1 """
    def set_x(self):
        self.x = self.value/self.factor
        return


class time_prob(TimePar):
    """ A probability over time (a.k.a. a "true" rate, cumulative hazard rate); must be >0 and <1 """
    def set_x(self):
        rate = -np.log(1 - self.value)
        self.x = 1 - np.exp(-rate/self.factor)
        return
        
    
class beta(time_prob):
    """ A container for beta (i.e. the disease transmission rate) """
    pass
    