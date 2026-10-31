"""
Example disease models: cholera, COVID, dengue, Ebola, gonorrhea, HIV, influenza, malaria, measles, pertussis,
polio, RSV, syphilis, TB, typhoid, and a non-communicable disease (NCD), and Starsim 3's SIR, SEIR, and SIS
written as code.

These are illustrative, lightly-parameterized models intended as starting points
for building your own disease modules, not as validated models for any specific
setting. Each one says in its docstring where its structure and its default
parameters came from. Core disease base classes (`ss.Disease`, `ss.Infection`,
`ss.SIR`, etc.) live in `starsim.diseases`.
"""
from .agents      import AgentSIR, AgentSEIR, AgentSIS
from .cholera     import Cholera
from .covid       import COVID
from .dengue      import Dengue
from .ebola       import Ebola
from .enteric     import Typhoid, Polio
from .hiv         import HIV, ART, CD4_analyzer
from .malaria     import Malaria, MalariaImmunity
from .measles     import Measles
from .ncd         import NCD
from .respiratory import Influenza, RSV, Pertussis
from .sti         import Syphilis, Gonorrhea
from .tb          import TB
