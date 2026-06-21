"""
The Starsim library: a collection of example and reference modules that build on the
core Starsim package but are not part of it.

Import as ``import starsim.library as ssl``. Unlike core Starsim, the library does not
export everything at the top level; classes are accessed via their submodule, e.g.
``ssl.mnch.FetalHealth``, ``ssl.networks.HouseholdNet``, ``ssl.diseases.Cholera``.
"""
from . import diseases
from . import mnch
from . import networks
