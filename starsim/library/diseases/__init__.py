"""
Example disease models: cholera, Ebola, HIV, and measles, and Starsim 3's SIR, SEIR, and SIS written as code.

These are illustrative, lightly-parameterized models intended as starting points
for building your own disease modules, not as validated models for any specific
setting. Core disease base classes (`ss.Disease`, `ss.Infection`, `ss.SIR`, etc.)
live in `starsim.diseases`.
"""
from .coded   import CodedSIR, CodedSEIR, CodedSIS
from .cholera import Cholera
from .ebola   import Ebola
from .hiv     import HIV, ART, CD4_analyzer
from .measles import Measles
