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

# %% Default result settings

# Flows
class Flow:
    def __init__(self, name, label=None, color=None):
        self.name = name
        self.label = label or name
        self.color = color
