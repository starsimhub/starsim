"""
Numba's on-disk cache, made stale by the files a module's functions call into, and saying when something is compiled

Numba checks only the file of a cached function, so a change to a function it calls in
another file (e.g. a draw in `draws.py`, called by the kernels) would leave the old
compiled code in use. `depends()` adds those files to the stamp Numba checks.

Compiling takes seconds the first time (and after a change to the code), so `compiling()` says
what is being compiled and about how long it takes, once a session for each thing: for Numba,
when it starts compiling (not when it loads from its cache) one of the sets of functions in
`slow`, or a function from `ss.jit`; for JAX, where Starsim compiles something.
"""

import os
import numba.core.caching as caching
from numba.core import event
import starsim as ss

deps = {} # The extra files each source file's cache depends on

# Starsim's functions that take long enough to compile to say so, by module and function (or by module): what they are, and how long compiling them takes on a laptop (s)
slow = {
    'starsim.perflib.kernels.run':             ('the compartmental solvers', 30, 90),
    'starsim.perflib.kernels.run_many':        ('the compartmental solvers for many runs', 4, 15),
    'starsim.perflib.kernels.one_step':        ('a step of the compartmental solvers', 3, 10),
    'starsim.perflib.kernels.advance':         ("the particle filter's steps", 10, 30),
    'starsim.perflib.kernels.advance_threads': ("the particle filter's steps in threads", 15, 45),
    'starsim.perflib.passes':         ('the agent solver', 2, 10),
    'starsim.perflib.jit':            ('the random draws of ss.jit', 1, 4),
}
said = set() # What has been said to be compiling, this session


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
