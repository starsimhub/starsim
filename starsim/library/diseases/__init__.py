"""
Example disease models: cholera, Ebola, HIV, and measles.

These are illustrative, lightly-parameterized models intended as starting points
for building your own disease modules, not as validated models for any specific
setting. Core disease base classes (`ss.Disease`, `ss.Infection`, `ss.SIR`, etc.)
live in `starsim.diseases`.
"""
from .cholera import Cholera
from .ebola   import Ebola
from .hiv     import HIV, ART, CD4_analyzer
from .measles import Measles
