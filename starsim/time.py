"""
Functions and classes for handling time
"""

import numpy as np
import sciris as sc

# What classes are externally visible
__all__ = ['time_units', 'time_ratio', 'date_add', 'date_diff', 'make_timevec', 'TimeUnit', 'dur', 'rate', 'time_prob']

    
#%% Helper functions

# Define available time units
time_units = sc.dictobj(
    day = 1,
    week = 7,
    month = 30.4375, # 365.25/12 -- more accurate and nicer fraction
    year = 365, # For simplicity with days
)

def time_ratio(unit1='day', dt1=1.0, unit2='day', dt2=1.0):
    """
    Calculate the relationship between two sets of time factors
    
    Args:
        unit1 and dt1 are the numerator, unit2 and dt2 are the denominator
    
    **Example**::
        ss.time_ratio(unit1='week', dt1=2, unit2='day', dt2=1) # Returns 14.0
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
    return factor


def date_add(start, dur, unit):
    """ Add two dates (or integers) together """
    if sc.isnumber(start):
        end = start + dur
    else:
        if unit in time_units:
            ndays = int(round(time_units[unit]*dur))
            end = sc.datedelta(start, days=ndays)
        else:
            errormsg = f'Unknown unit {unit}, choices are: {sc.strjoin(time_units.keys())}'
            raise ValueError(errormsg)
    return end
        

def date_diff(start, end, unit):
    """ Find the difference between two dates (or integers) """
    if sc.isnumber(start) and sc.isnumber(end):
        dur = end - start
    else:
        if unit == 'year':
            dur = sc.datetoyear(end) - sc.datetoyear(start) # TODO: allow non-integer amounts
        elif unit == 'day':
            dur = (end - start).days
        else:
            raise NotImplementedError
    return dur


def make_timevec(start, end, dt, unit):
    """ Parse start, end, and dt into an appropriate time vector """
    if sc.isnumber(start):
        try:
            stop = date_add(end, dt, unit) # Potentially convert to a date
            timevec = np.arange(start=start, stop=stop, step=dt) # The time points of the sim
        except Exception as E:
            errormsg = f'Incompatible set of time inputs: start={start}, end={end}, dt={dt}. You can use dates or numbers but not both.'
            raise ValueError(errormsg) from E
    else:
        day_delta = int(np.round(time_ratio(unit1=unit, dt1=dt, unit2='day', dt2=1.0)))
        if day_delta > 0:
            timevec = sc.daterange(start, end, interval={'days':day_delta})
        else:
            errormsg = f'Timestep {dt} is too small; must be at least 1 day'
            raise ValueError(errormsg)
    return timevec


#%% Time classes

class TimeUnit:
    """
    Base class for durations and rates
    
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
        self.RMULC = 0
        return
    
    def initialize(self, parent=None, parent_unit=None, parent_dt=None):
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
        xstr = f'x={self.x:n}' if self.initialized else 'initialized=False'
        return f'ss.{name}({self.value}, unit={self.unit}, {xstr})'
    
    def disp(self):
        return sc.pr(self)
    
    def set(self, value=None, unit=None, parent_unit=None, parent_dt=None):
        """ Reset the parameter values """
        if value       is not None: self.value       = value
        if unit        is not None: self.unit        = unit
        if parent_unit is not None: self.parent_unit = parent_unit
        if parent_dt   is not None: self.parent_dt   = parent_dt
        self.set_factor()
        self.set_x()
        return self
    
    def set_factor(self):
        """ Set factor used to multiply the value to get the output """
        self.factor = time_ratio(unit1=self.unit, dt1=1.0, unit2=self.parent_unit, dt2=self.parent_dt)
        return
        
    # @property
    # def x(self):
    #     """ The actual value used in calculations -- the key step! """
    #     raise NotImplementedError
        
    # @property
    # def f(self):
    #     """ Return the factor, with a helpful error message if not set """
    #     if self.factor is not None:
    #         return self.factor
    #     else:
    #         errormsg = f'The factor for {self} has not been set. Have you called initialize()?'
    #         raise RuntimeError(errormsg)
    
    # Act like a float
    def __add__(self, other): return self.x + other
    def __sub__(self, other): return self.x - other
    def __mul__(self, other): return self.x * other
    def __pow__(self, other): return self.x ** other
    def __truediv__(self, other): return self.x / other
    
    # ...from either side
    def __radd__(self, other): return other + self.x
    def __rsub__(self, other): return other - self.x
    # def __rmul__(self, other): return other * self.x
    def __rpow__(self, other): return other ** self.x
    def __rtruediv__(self, other): return other / self.x
    
    def __rmul__(self, other):
        dem = 1000
        self.RMULC += 1
        if not self.RMULC % dem:
            print('hi i am rmul', self, other)
            if np.random.rand() < 0.1:
                raise Exception('you FUAIILED')
        return other * self.x
    
    # Handle modify-in-place methods
    def __iadd__(self, other): self.value += other; return self
    def __isub__(self, other): self.value -= other; return self
    def __imul__(self, other): self.value *= other; return self
    def __itruediv__(self, other): self.value /= other; return self
    
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

class dur(TimeUnit):
    """ Any number that acts like a duration """
    def set_x(self):
        self.x = self.value*self.factor
        return


class rate(TimeUnit):
    """ Any number that acts like a rate; can be greater than 1 """
    def set_x(self):
        self.x = self.value/self.factor
        return
    

class time_prob(TimeUnit):
    """ A probability over time (a.k.a. a "true" rate, cumulative hazard rate); must be >0 and <1 """
    def set_x(self):
        rate = -np.log(1 - self.value)
        self.x = 1 - np.exp(-rate/self.factor)
        return
        
    