"""
Results and associated structures. Currently only holds results, could move to base.py
unless other things get added (e.g. Resultsets, MultiResults, other...)
"""

import numpy as np
import sciris as sc
from . import utils as ssu


class Results(ssu.ndict):
    def __init__(self, *args, **kwargs):
        super().__init__(self, _type=Result, *args, **kwargs)


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
    
    def to_df(self):
        return sc.dataframe({self.name:self})