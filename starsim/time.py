"""
Functions and classes for handling time
"""

import sciris as sc


# What classes are externally visible
__all__ = ['time_units', 'time_ratio', 'dur', 'rate']

    
    
#%% Helper functions


# Define available time units
time_units = sc.dictobj(
    day = 1,
    week = 7,
    year = 365,
)

def time_ratio(unit1='day', dt1=1.0, unit2='day', dt2=1.0):
    """
    Calculate the relationship between two sets of time factors
    
    unit1 and dt1 are the numerator, unit2 and dt2 are the denominator    
    """
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



#%% Time classes

class TimeUnit:
    """ Base class for durations and rates """
    def __init__(self, value, unit=None, dt=None):
        self.value = value
        self.unit = unit
        self.dt = dt
        self.factor = None
        self.initialized = False
        return
    
    def initialize(self, parent=None, base_unit=None, base_dt=None): # TODO: should the parent actually be linked to? Seems excessive
        """ Link to the sim and/or module units """
        if parent is None:
            if base_dt is None:
                base_dt = 1.0
            parent = sc.dictobj(unit=base_unit, dt=base_dt)
        else:
            if base_dt is not None:
                errormsg = f'Cannot override parent {parent} by setting base_dt'
                raise ValueError(errormsg)

        if self.unit is None and parent.unit is not None:
            self.unit = parent.unit
            
        if self.dt is None and parent.dt is not None:
            self.dt = parent.dt
            
        self.factor = time_ratio(unit1=self.unit, dt1=parent.dt, unit2=parent.unit, dt2=self.dt) # TODO: check
        self.initialized = True
        return self
        
    def __repr__(self):
        name = self.__class__.__name__
        return f'ss.{name}({self.value}, unit={self.unit}, dt={self.dt})'
    
    def disp(self):
        return sc.pr(self)
    
    @property
    def x(self):
        raise NotImplementedError
    
    # Act like a float
    def __add__(self, other): return self.x + other
    def __sub__(self, other): return self.x - other
    def __mul__(self, other): return self.x * other
    def __truediv__(self, other): return self.x / other
    
    # ...from either side
    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other): return self.__truediv__(other)


class dur(TimeUnit):
    """ A number that acts like a duration """
    @property
    def x(self):
        return self.value*self.factor


class rate(TimeUnit):
    """ A number that acts like a rate """
    @property
    def x(self):
        return self.value/self.factor


        
    