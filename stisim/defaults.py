"""
Set the defaults
"""

import numpy as np
import sciris as sc
import pylab as pl
from .settings import options as sso  # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. hpv.defaults.default_int
__all__ = ['datadir', 'default_float', 'default_int']

# Define paths
datadir = sc.path(sc.thisdir(__file__)) / 'data'

# %% Specify what data types to use

result_float = np.float64  # Always use float64 for results, for simplicity
if sso.precision == 32:
    default_float = np.float32
    default_int = np.int32
elif sso.precision == 64:  # pragma: no cover
    default_float = np.float64
    default_int = np.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {sso.precision}')


# %% Default result settings

# Flows
class Flow:
    def __init__(self, name, label=None, color=None):
        self.name = name
        self.label = label or name
        self.color = color
