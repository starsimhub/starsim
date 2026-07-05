"""
The Starsim library: a collection of example and reference modules that build on the
core Starsim package but are not part of it.

Import as `import starsim.library as ssl`. Classes can be accessed either via their
submodule (e.g. `ssl.networks.HouseholdNet`, `ssl.diseases.Cholera`) or, like core
Starsim, at the top level (e.g. `ssl.HouseholdNet`, `ssl.Cholera`).
"""
from . import diseases
from . import mnch
from . import networks

# Also export the contents of each submodule at the top level
from .diseases import *
from .mnch import *
from .networks import *
