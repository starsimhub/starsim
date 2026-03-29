"""
Analyzes results from the other languages
"""

import sciris as sc
import matplotlib.pyplot as plt


# Data -- from running manually
d = sc.objdict()
d.starsim   = dict(flex=9, lines=25, times=[2.48, 2.53, 2.50])
d.python    = dict(flex=8, lines=239, times=[1.83, 2.18, 1.88])
d.numba     = dict(flex=, lines=, times=[])
d.numba_jax = dict(flex=, lines=, times=[])
d.jax_cpu   = dict(flex=, lines=, times=[])
d.jax_gpu   = dict(flex=, lines=, times=[])
d.julia     = dict(flex=, lines=, times=[])
d.rust      = dict(flex=, lines=, times=[])