"""
The Starsim library: a collection of example and reference modules that build on the
core Starsim package but are not part of it.

Import as `import starsim.library as ssl`. Classes can be accessed either via their
submodule (e.g. `ssl.networks.HouseholdNet`, `ssl.diseases.Cholera`) or, like core
Starsim, at the top level (e.g. `ssl.HouseholdNet`, `ssl.Cholera`).

Each class appears in exactly two places: where it is defined, and in its subpackage's
`__init__.py`. The top-level exports below, and the API docs, follow automatically.
"""
import inspect
from . import diseases
from . import mnch
from . import networks

# Also export the contents of each subpackage at the top level, e.g. ssl.Cholera
__all__ = ['diseases', 'mnch', 'networks']
for _sub in [diseases, mnch, networks]:
    for _key,_val in vars(_sub).items():
        if not _key.startswith('_') and not inspect.ismodule(_val):
            globals()[_key] = _val
            __all__.append(_key)
del inspect, _sub, _key, _val
