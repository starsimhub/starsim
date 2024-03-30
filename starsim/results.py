"""
Result structures.
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['Result', 'Results']


class Result(np.ndarray):
    
    def __new__(cls, module=None, name=None, shape=None, dtype=None, scale=None):
        arr = np.zeros(shape=shape, dtype=dtype).view(cls)
        arr.name = name
        arr.module = module
        arr.scale = scale
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
        self.name = getattr(obj, 'name', None)
        self.module = getattr(obj, 'module', None)
        self.scale = getattr(obj, 'scale', None)
        return

    def __array_wrap__(self, obj, **kwargs):
        if obj.shape == ():
            return obj[()]
        else:
            return super().__array_wrap__(obj, **kwargs)
    
    def to_df(self):
        return sc.dataframe({self.name:self})
    

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
    
    def to_df(self):
        pass
    
    def __repr__(self, *args, **kwargs): # TODO: replace with dataframe summary
        return super().__repr__(*args, **kwargs)
        
    def plot(self):
        pass
