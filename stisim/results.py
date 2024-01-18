"""
Result structures.
"""

import numpy as np
import sciris as sc


__all__ = ['Result']


class Result(np.ndarray):
    
    def __new__(cls, module=None, name=None, shape=None, dtype=None):
        arr = np.zeros(shape=shape, dtype=dtype).view(cls)
        arr.name = name
        arr.module = module
        return arr
    
    def __repr__(self):
        modulestr = f'{self.module}.' if self.module else ''
        cls_name = self.__class__.__name__
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f'{cls_name}({modulestr}{self.name}):\narray{arrstr}'
        return out

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.module = getattr(obj, 'module', None)
        return

    def __array_wrap__(self, obj, **kwargs):
        if obj.shape == ():
            return obj[()]
        else:
            return super().__array_wrap__(obj, **kwargs)
    
    def to_df(self):
        return sc.dataframe({self.name:self})
