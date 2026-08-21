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
from .diseases import Cholera, Ebola, HIV, ART, CD4_analyzer, Measles
from .mnch     import FetalHealth, fetal_infection, treat_pregnant, CongenitalDisease, NeonatalSepsis
from .networks import HouseholdNet, DiskNet, ErdosRenyiNet, NullNet

__all__ = ['diseases', 'mnch', 'networks'] + diseases.__all__ + mnch.__all__ + networks.__all__
