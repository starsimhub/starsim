"""
Result structures.
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['Result', 'Results']


class Result(ss.BaseArr):
    """
    Array-like container for holding sim results.

    Args:
        module (str): the name of the parent module, e.g. 'hiv'
        name (str): the name of this result, e.g. 'new_infections'
        shape (int/tuple): the shape of the result array (usually module.npts)
        scale (bool): whether or not the result scales by population size (e.g. a count does, a prevalence does not)
        label (str): a human-readable label for the result
        values (array): prepopulate the Result with these values
        low (array): values for the lower bound
        high (array): values for the upper bound

    In most cases, ``ss.Result`` behaves exactly like ``np.array()``, except with
    the additional fields listed above. To see everything contained in a result,
    you can use result.disp().
    """
    def __init__(self, module=None, name=None, shape=None, dtype=None, scale=None, label=None, values=None, low=None, high=None):
        if values is not None:
            self.values = np.array(values, dtype=dtype)
        else:
            self.values = np.zeros(shape=shape, dtype=dtype)
        self.name = name
        self.module = module
        self.scale = scale
        self.label = label
        self.low = low
        self.high = high
        return
    
    def __repr__(self):
        modulestr = f'{self.module}.' if (self.module is not None) else ''
        cls_name = self.__class__.__name__
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f'{cls_name}({modulestr}{self.name}):\narray{arrstr}'
        return out
    
    def to_df(self):
        data = {self.name:self}
        if self.low is not None:
            data['low']  = self.low
        if self.high is not None:
            data['high'] = self.high
        df = sc.dataframe(data)
        return df
    

class Results(ss.ndict):
    """ Container for storing results """
    def __init__(self, module, strict=True, *args, **kwargs):
        super().__init__(type=Result, strict=strict)
        if hasattr(module, 'name'):
            module = module.name
        self.setattribute('_module', module)
        return
    
    def append(self, arg, key=None):
        if isinstance(arg, (list, tuple)):
            result = ss.Result(self._module, *arg)
        elif isinstance(arg, dict):
            result = ss.Result(self._module, **arg)
        else:
            result = arg
        if result.module != self._module:
            result.module = self._module
        
        super().append(result, key=key)
        return
    
    def flatten(self, sep='_'):
        return sc.objdict(sc.flattendict(self, sep=sep))
    
    def to_df(self):
        pass
    
    def __repr__(self, *args, **kwargs): # TODO: replace with dataframe summary
        return super().__repr__(*args, **kwargs)
        
    def plot(self):
        pass
