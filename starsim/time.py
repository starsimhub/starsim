"""
Functions and classes for handling time
"""

import sciris as sc


#%% Time classes

# What classes are externally visible
__all__ = ['dur', 'rate']


class TimeUnit:
    """ Base class for durations and rates """
    def __init__(self, value, unit=None, dt=None):
        self.value = value
        self.unit = unit
        self.dt = sc.ifelse(dt, 1.0) # TEMP
        return
    
    def __repr__(self):
        name = self.__class__.__name__
        return f'ss.{name}({self.value}, unit={self.unit}, dt={self.dt})'
    
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
        return self.value/self.dt


class rate(TimeUnit):
    """ A number that acts like a rate """
    @property
    def x(self):
        return self.value*self.dt