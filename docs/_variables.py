"""
Values for the docs (e.g. {{< var version >}}): `qpyd prerender` writes them to _variables.yml.

The version is read from starsim/version.py rather than by importing Starsim, since this runs on every render.
"""
import runpy as _runpy
import importlib.util as _util
from pathlib import Path as _Path

_version = _runpy.run_path(_Path(_util.find_spec('starsim').origin).parent / 'version.py')
version = _version['__version__']
versiondate = _version['__versiondate__']
