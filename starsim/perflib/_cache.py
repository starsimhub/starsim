"""
Numba's on-disk cache, made stale by the files a module's functions call into

Numba checks only the file of a cached function, so a change to a function it calls in
another file (e.g. a draw in `_draws.py`, called by the kernels) would leave the old
compiled code in use. `depends()` adds those files to the stamp Numba checks.
"""

import os
import numba.core.caching as caching

deps = {} # The extra files each source file's cache depends on


def depends(py_file, *others):
    """ Make the cache of the functions in py_file stale whenever any of the other files changes; call before the functions are decorated """
    deps[os.path.abspath(py_file)] = [os.path.abspath(f) for f in others]
    if not getattr(caching.CacheImpl, '_starsim_deps', False):
        caching.CacheImpl._locator_classes = [with_deps(cls) for cls in caching.CacheImpl._locator_classes]
        caching.CacheImpl._starsim_deps = True
    return


def with_deps(cls):
    """ A locator that adds the stamps of the files its source file depends on (see deps) """
    class Locator(cls):
        def get_source_stamp(self):
            stamp = super().get_source_stamp()
            others = deps.get(os.path.abspath(getattr(self, '_py_file', '')))
            if others:
                stamp = (stamp, tuple((os.stat(f).st_mtime, os.stat(f).st_size) for f in others))
            return stamp
    Locator.__name__ = Locator.__qualname__ = cls.__name__
    return Locator
