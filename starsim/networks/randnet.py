"""Random networks"""

import starsim as ss
import numpy as np
import numba as nb
from .networks import Network
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Union

__all__ = ['RandomNetwork', 'random']

