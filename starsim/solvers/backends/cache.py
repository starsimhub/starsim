"""
Not for the user: Numba's on-disk cache, made stale by the contents of a module's file and of the files its functions call into, and saying when something is compiled

Numba checks only the file of a cached function, so a change to a function it calls in
another file (e.g. a draw in `draws.py`, called by the kernels) would leave the old
compiled code in use. `depends()` adds those files to the stamp Numba checks. Numba's
stamp is also a file's time and size, which a fresh checkout or install changes and an edit of
the same size may not, so the stamp here is of the contents: a cache is used exactly when
the code it was compiled from is the code that is here, wherever the cache has come from.

Compiling takes seconds the first time (and after a change to the code), so `compiling()` says
what is being compiled and about how long it takes, once a session for each thing: for Numba,
when it starts compiling (not when it loads from its cache) one of the sets of functions in
`slow`, or a function from `ss.jit`; for JAX, where Starsim compiles something.

Used with: agents and compartments; declared and coded modules
"""

import os
import hashlib
import numba.core.caching as caching
from numba.core import event
import starsim as ss

deps = {} # The extra files each source file's cache depends on
hashes = {} # The hash of each file's contents, with the time and size it was read at, so that each file is read once

# Starsim's functions that take long enough to compile to say so, by module and function (or by module): what they are, and how long compiling them takes on a laptop (s)
slow = {
    'starsim.solvers.backends.kernels.run':             ('the compartmental solvers', 60, 130),
    'starsim.solvers.backends.kernels.run_many':        ('the compartmental solvers for many runs', 4, 15),
    'starsim.solvers.backends.kernels.one_step':        ('a step of the compartmental solvers', 3, 10),
    'starsim.solvers.backends.kernels.advance':         ("the particle filter's steps", 10, 30),
    'starsim.solvers.backends.kernels.advance_threads': ("the particle filter's steps in threads", 15, 45),
    'starsim.solvers.backends.passes':         ('the agent solver', 2, 10),
}
said = set() # What has been said to be compiling, this session


def depends(py_file, *others):
    """ Make the cache of the functions in py_file stale whenever any of the other files changes; call before the functions are decorated """
    deps[os.path.abspath(py_file)] = [os.path.abspath(f) for f in others]
    if not getattr(caching.CacheImpl, '_starsim_deps', False):
        caching.CacheImpl._locator_classes = [with_deps(cls) for cls in caching.CacheImpl._locator_classes]
        caching.CacheImpl._starsim_deps = True
    return


def contents(path):
    """ A hash of the contents of a file """
    stat = os.stat(path)
    key = (stat.st_mtime_ns, stat.st_size)
    if hashes.get(path, (None,))[0] != key:
        with open(path, 'rb') as f:
            hashes[path] = (key, hashlib.blake2b(f.read(), digest_size=16).hexdigest())
    return hashes[path][1]


def with_deps(cls):
    """ A locator whose stamp is the contents of its source file and of the files that file depends on (see deps) """
    class Locator(cls):
        """ The locator, with that stamp """
        def get_source_stamp(self):
            path = os.path.abspath(getattr(self, '_py_file', ''))
            if not os.path.isfile(path): # Code that is not in a file, as in a notebook
                return super().get_source_stamp()
            return tuple(contents(f) for f in [path] + deps.get(path, []))
    Locator.__name__ = Locator.__qualname__ = cls.__name__
    return Locator


def compiling(what, engine, low, high):
    """ Say that something is being compiled, and about how long it takes (in seconds), once a session for each thing """
    if what not in said and ss.options.verbose:
        said.add(what)
        print(f'  Compiling {what} with {engine}, estimated time: {low}-{high} s')
    return


class Compiling(event.Listener):
    """ Says when Numba starts compiling a function in `slow`, or one that `ss.jit` gave `ss_compiling`; the functions it calls are compiled within it, and not said again """
    depth = 0

    def on_start(self, ev):
        if not Compiling.depth:
            func = ev.data['dispatcher'].py_func
            info = getattr(ev.data['dispatcher'], 'ss_compiling', None) or slow.get(f'{func.__module__}.{func.__name__}') or slow.get(func.__module__)
            if info:
                compiling(info[0], 'Numba', *info[1:])
        Compiling.depth += 1
        return

    def on_end(self, ev):
        Compiling.depth -= 1
        return


event.register('numba:compile', Compiling())
