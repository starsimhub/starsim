"""
Result structures.
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['Result', 'Results']


class Result(np.ndarray):
    
    # Define the custom attributes
    _custom_attrs = [
        'name', # The name of the result (e.g. n_infections)
        'module', # The name of the module (e.g. hiv)
        'scale', # Whether or not the result scales with population size (e.g. True)
        'label', # The human-readable label for the result (e.g. Number of infections)
        'low', # The lower bound for the values (used with MultiSim)
        'high', # The upper bound for the values
    ]
    
    def __new__(cls, module=None, name=None, shape=None, dtype=None, scale=None, label=None, low=None, high=None):
        arr = np.zeros(shape=shape, dtype=dtype).view(cls)
        arr.name = name
        arr.module = module
        arr.scale = scale
        arr.label = label
        arr.low = low
        arr.high = high
        return arr
    
    
    def __repr__(self):
        modulestr = f'{self.module}.' if (self.module is not None) else ''
        cls_name = self.__class__.__name__
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f'{cls_name}({modulestr}{self.name}):\narray{arrstr}'
        return out

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in self._custom_attrs:
            value = getattr(obj, attr, None)
            setattr(self, attr, value)
        return

    def __array_wrap__(self, obj, **kwargs):
        if obj.shape == ():
            return obj[()]
        else:
            return super().__array_wrap__(obj, **kwargs)
    
    def to_df(self):
        data = {self.name:self}
        if self.low is not None:
            data['low']  = self.low
        if self.high is not None:
            data['high'] = self.high
        df = sc.dataframe(data)
        return df
    

class Results(ss.ndict):
    
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
