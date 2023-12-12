"""
Result structures.
"""

import numpy as np
import sciris as sc


__all__ = ['Result']


class Result(np.ndarray):
    
    def __new__(cls, module=None, name=None, shape=None, dtype=None, **kwargs):
        arr = np.zeros(shape=shape, dtype=dtype).view(cls)
        arr.name = name
        arr.module = module
        return arr
    
    def __repr__(self):
        if hasattr(self, 'module') and hasattr(self, 'name'):
            modulestr = f'{self.module}.' if self.module else ''
            cls_name = self.__class__.__name__
            arrstr = super().__repr__().removeprefix(cls_name)
            out = f'{cls_name}({modulestr}{self.name}):\narray{arrstr}'
            return out
        else:
            return np.ndarray.__repr__(self)
    
    def to_df(self):
        return sc.dataframe({self.name:self})

