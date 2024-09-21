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


# def int_to_date(x, unit='year'): # TODO: use sc.datetoyear(..., reverse=True)
#     """ Convert an integer to a date """
#     if unit == 'year':
#         date = sc.date(f'{x}-01-01')
#     else:
#         raise NotImplementedError
#     return date
        

def date_add(start, dur, unit):
    """ Add two dates (or integers) together """
    if sc.isnumber(start):
        end = start + dur
    else:
        if unit == 'year':
            end = sc.datedelta(start, years=dur)
        elif unit == 'day':
            end = sc.datedelta(start, days=dur)
        else:
            raise NotImplementedError
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
        if unit == 'year':
            day_delta = int(np.round(time_ratio(unit1='year', dt1=dt, unit2='day', dt2=1.0)))
            if day_delta == 0:
                errormsg = f'Timestep {dt} is too small; must be at least 1 day'
                raise ValueError(errormsg)
        else:
            if dt < 1:
                errormsg = f'Cannot use a timestep of less than a day ({dt}) with date-based indexing'
                raise ValueError(errormsg)
            day_delta = int(np.round(dt))
        timevec = sc.daterange(start, end, interval={'days':day_delta})
    return timevec


#%% Time classes

class TimeUnit:
    """ Base class for durations and rates """
    def __init__(self, value, unit=None, parent_unit=None, parent_dt=None):
        self.value = value
        self.unit = unit
        self.parent_unit = parent_unit
        self.parent_dt = parent_dt
        self.factor = None
        self.parent_name = None
        self.initialized = False
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
        self.initialized = True
        return self
        
    def __repr__(self):
        name = self.__class__.__name__
        initstr = '' if self.initialized else ', initialized=False'
        return f'ss.{name}({self.value}, unit={self.unit}, {initstr})'
    
    def disp(self):
        return sc.pr(self)
    
    def set(self, **kwargs):
        """ Reset the parameter values (NB, attributes can also be set directly) """
        for k,v in kwargs.items():
            setattr(self, k, v)
        return self
    
    def set_factor(self):
        """ Set factor used to multiply the value to get the output """
        self.factor = time_ratio(unit1=self.unit, dt1=1.0, unit2=self.parent_unit, dt2=self.parent_dt)
        return
        
    @property
    def x(self):
        """ The actual value used in calculations -- the key step! """
        raise NotImplementedError
        
    @property
    def f(self):
        """ Return the factor, with a helpful error message if not set """
        if self.factor is not None:
            return self.factor
        else:
            errormsg = f'The factor for {self} has not been set. Have you called initialize()?'
            raise RuntimeError(errormsg)
    
    # Act like a float
    def __add__(self, other): return self.x + other
    def __sub__(self, other): return self.x - other
    def __mul__(self, other): return self.x * other
    def __pow__(self, other): return self.value ** other
    def __truediv__(self, other): return self.x / other
    
    # ...from either side
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rpow__(self, other): return self.__pow__(other)
    def __rtruediv__(self, other): return self.__truediv__(other)
    
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
        

class dur(TimeUnit):
    """ Any number that acts like a duration """
    @property
    def x(self):
        return self.value*self.f


class rate(TimeUnit):
    """ Any number that acts like a rate; can be greater than 1 """
    @property
    def x(self):
        return self.value/self.f
    

class time_prob(TimeUnit):
    """ A probability over time (a.k.a. a "true" rate, cumulative hazard rate); must be >0 and <1 """
    @property
    def x(self):
        rate = -np.log(1 - self.value)
        out = 1 - np.exp(-rate/self.f)
        return out
        
    