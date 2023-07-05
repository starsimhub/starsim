"""
Results and associated structures. Currently only holds results, could move to base.py
unless other things get added (e.g. Resultsets, MultiResults, other...)
"""

import numpy as np


class Result(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, name, module, shape, dtype):
        self.name = name
        self.module = module if module else None
        self.values = np.zeros(shape, dtype=dtype)

    def __getitem__(self, *args, **kwargs):  return self.values.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs): return self.values.__setitem__(*args, **kwargs)

    def __len__(self): return len(self.values)

    def __repr__(self): return f'Result({self.module},{self.name}): {self.values.__repr__()}'

    # These methods allow automatic use of functions like np.sum, np.exp, etc.
    # with higher performance in some cases
    @property
    def __array_interface__(self): return self.values.__array_interface__

    def __array__(self): return self.values

    def __array_ufunc__(self, *args, **kwargs):
        args = [(x if x is not self else self.values) for x in args]
        kwargs = {k: v if v is not self else self.values for k, v in kwargs.items()}
        return self.values.__array_ufunc__(*args, **kwargs)
