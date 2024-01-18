"""
Numerical utilities
"""

# %% Housekeeping

import numpy as np
import sciris as sc
from stisim.utils.ndict import *
from stisim.utils.actions import *
from stisim.settings import *

# What functions are externally visible -- note, this gets populated in each section below
__all__ = []

# System constants
__all__ += ['INT_NAN']

INT_NAN = np.iinfo(np.int32).max  # Value to use to flag invalid content (i.e., an integer value we are treating like NaN, since NaN can't be stored in an integer array)


# %% Helper functions
__all__ += ['ndict']


class ndict(sc.objdict):
    """
    A dictionary-like class that provides additional functionalities for handling named items.

    Args:
        name (str): The items' attribute to use as keys.
        type (type): The expected type of items.
        strict (bool): If True, only items with the specified attribute will be accepted.

    **Examples**::

        networks = ndict(mf(), maternal())
        networks = ndict([mf(), maternal()])
        networks = ndict({'mf':mf(), 'maternal':maternal()})

    """

    def __init__(self, *args, name='name', type=None, strict=True, **kwargs):
        self.setattribute('_name', name)  # Since otherwise treated as keys
        self.setattribute('_type', type)
        self.setattribute('_strict', strict)
        self._initialize(*args, **kwargs)
        return
    
    def append(self, arg, key=None):
        valid = False
        if arg is None:
            return # Nothing to do
        elif hasattr(arg, self._name):
            key = key or getattr(arg, self._name)
            valid = True
        elif isinstance(arg, dict):
            if self._name in arg:
                key = key or arg[self._name]
                valid = True
            else:
                for k,v in arg.items():
                    self.append(v, key=k)
                valid = None # Skip final processing
        elif not self._strict:
            key = key or f'item{len(self)+1}'
            valid = True
        else:
            valid = False
        
        if valid is True:
            self._check_type(arg)
            self[key] = arg
        elif valid is None:
            pass # Nothing to do
        else:
            errormsg = f'Could not interpret argument {arg}: does not have expected attribute "{self._name}"'
            raise ValueError(errormsg)
            
        return
        
    def _check_type(self, arg):
        """ Check types """
        if self._type is not None:
            if not isinstance(arg, self._type):
                errormsg = f'The following item does not have the expected type {self._type}:\n{arg}'
                raise TypeError(errormsg)
        return
    
    def _initialize(self, *args, **kwargs):
        args = sc.mergelists(*args)
        for arg in args:
            self.append(arg)
        for key,arg in kwargs.items():
            self.append(arg, key=key)
        return
    
    def copy(self):
        new = self.__class__.__new__(name=self._name, type=self._type, strict=self._strict)
        new.update(self)
        return new
    
    def __add__(self, dict2):
        """ Allow c = a + b """
        new = self.copy()
        new.append(dict2)
        return new

    def __iadd__(self, dict2):
        """ Allow a += b """
        self.append(dict2)
        return self

    """
    Returns the indices of the values of the array that are not-nan.

    Args:
        arr (array): any array

    **Example**::

        inds = defined(np.array([1,np.nan,0,np.nan,1,0,1]))
    """
    # return np.isnan(arr).nonzero()[-1]