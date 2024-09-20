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



#%% Time classes

class TimeUnit:
    """ Base class for durations and rates """
    def __init__(self, value, unit=None, self_dt=None, parent_unit=None, parent_dt=None):
        self.value = value
        self.unit = unit
        self.self_dt = self_dt
        self.parent_unit = parent_unit
        self.parent_dt = parent_dt
        self.factor = None
        self.parent_name = None
        self.initialized = False
        return
    
    def initialize(self, parent=None, parent_unit=None, parent_dt=None): # TODO: should the parent actually be linked to? Seems excessive
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
            if self.unit is None: 
                self.unit = parent.unit
            
        if parent.dt is not None:
           self.parent_dt = parent.dt
           if self.self_dt is None:
                self.self_dt = parent.dt
        
        # Set defaults if not yet set -- TODO, is there a better way?
        self.self_dt = sc.ifelse(self.self_dt, 1.0)
        self.parent_dt = sc.ifelse(self.parent_dt, 1.0)
        
        self.set_factor()
        self.initialized = True
        return self
        
    def __repr__(self):
        name = self.__class__.__name__
        return f'ss.{name}({self.value}, unit={self.unit}, dt={self.self_dt})'
    
    def disp(self):
        return sc.pr(self)
    
    def set_factor(self):
        """ Set factor used to multiply the value to get the output """
        raise NotImplementedError
    
    @property
    def x(self):
        """ The actual value used in calculations """
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
    
    def set_factor(self):
        self.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=self.parent_unit, dt2=self.parent_dt)
        
    @property
    def x(self):
        return self.value*self.factor


class rate(TimeUnit):
    """ A number that acts like a rate """
    
    def set_factor(self):
        self.factor = time_ratio(unit1=self.unit, dt1=self.self_dt, unit2=self.parent_unit, dt2=self.parent_dt)
        
    @property
    def x(self):
        return self.value/self.factor # TODO: implement an optional 1 - np.exp(-value * dt) version, probably in a different class


        
    