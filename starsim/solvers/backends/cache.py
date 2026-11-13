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
`slow`, or a function from `ss.jit`, with a bar of the time taken so far, redrawn until it is
done (see `Ticker`), or for `ss.compile()`, one bar for all of them (see `Batch`); for JAX, where Starsim compiles something.

Used with: agents and compartments; declared and coded modules
"""

import os
import time
import contextlib
import hashlib
import threading
import numba.core.caching as caching
from numba.core import event
import starsim as ss

deps = {} # The extra files each source file's cache depends on
hashes = {} # The hash of each file's contents, with the time and size it was read at, so that each file is read once

# Starsim's functions that take long enough to compile to say so, by module and function (or by module): what they are, and how long compiling them takes (s), erring long (measured 2026-11-07: 34 to 43 s for many runs, 32 to 46 and 33 to 55 s for the particle filter's)
slow = {
    'starsim.solvers.backends.passes':                  ('the agent solver', 2, 15), # And for ss.compile(), the rest of what sims with agents compile (30 s measured), as 'agents'
    'agents':                                           ('the agent solver and what sims with agents use', 20, 60),
    'starsim.solvers.backends.kernels.run':             ('the compartmental solvers', 60, 150), # 63 s measured
    'starsim.solvers.backends.kernels.one_step':        ('a step of the compartmental solvers', 5, 20),
    'starsim.solvers.backends.kernels.run_many':        ('the compartmental solvers for many runs', 30, 70),
    'starsim.solvers.backends.kernels.advance':         ("the particle filter's steps", 30, 70),
    'starsim.solvers.backends.kernels.advance_threads': ("the particle filter's steps in threads", 40, 80),
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
    """ Say that something is being compiled, and about how long it takes (in seconds), once a session for each thing; True if it was said (see `timed()` to say how long it took, too) """
    if what not in said and ss.options.verbose:
        said.add(what)
        print(f'  Compiling {what} with {engine}, estimated time: {low}-{high} s')
        return True
    return False


@contextlib.contextmanager
def timed(what, engine, low, high):
    """ Say that something is being compiled within the block, with the estimate (see `compiling()`), and then how long it took """
    said_now = compiling(what, engine, low, high)
    start = time.time()
    yield
    if said_now:
        print(f'  Compiled {what} with {engine} in {time.time() - start:.1f} s')
    return


def timed_first(func, what, engine, low, high):
    """ `func`, whose first call (when JAX compiles it) is said with its estimate and then how long it took (see `timed()`) """
    first = [True]
    def call(*args, **kwargs):
        if not first[0]:
            return func(*args, **kwargs)
        first[0] = False
        with timed(what, engine, low, high):
            return func(*args, **kwargs)
    return call


class Ticker:
    """
    While Numba compiles something that takes long (see `Compiling`), a bar of the time taken so far against the most it is
    estimated to take, redrawn twice a second from a thread of its own, in a terminal or a notebook (see `ss.internals.display.Progress`);
    to a file, nothing until it is done. Then a line saying how long it took.
    """
    def __init__(self, what, engine, low, high):
        self.what, self.engine, self.low, self.high = what, engine, low, high
        self.progress = ss.internals.display.Progress()
        self.start = time.time()
        self.stop = threading.Event()
        self.thread = None
        if self.progress.mode != 'lines':
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        return

    def text(self, elapsed):
        """ The line drawn while it compiles """
        return f'  {self.progress.bar(min(elapsed/self.high, 1))} {elapsed:.0f} s (estimated {self.low}-{self.high} s)'

    def run(self):
        while not self.stop.wait(0.5):
            self.progress.draw(self.text(time.time() - self.start))
        return

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.progress.finish(f'  Compiled {self.what} with {self.engine} in {time.time() - self.start:.1f} s')
        return


class Batch(Ticker):
    """
    One bar for compiling several of the sets of functions in `slow` in turn, as `ss.compile()` does: each part takes a fraction
    of the bar by how long it is estimated to take, and is not said on its own (see `Compiling`). A part that takes longer
    than estimated holds the bar at the end of its fraction until it is done.

    Args:
        what (str): what is compiled, as said when it is done
        keys (list): the keys of `slow` that are compiled, in the order they are; `part()` says when each starts and ends
    """
    current = None # The batch being compiled, if any

    def __init__(self, what, keys):
        self.fracs = {key:(slow[key][1] + slow[key][2])/2 for key in keys} # Each part's fraction of the bar
        self.finished = set() # The parts done
        self.key = None # The part being compiled, and when it started
        self.part_start = None
        said.update(slow[key][0] for key in keys) # Not said again on its own, later in the session
        Batch.current = self
        super().__init__(what, 'Numba', sum(slow[key][1] for key in keys), sum(slow[key][2] for key in keys))
        return

    @contextlib.contextmanager
    def part(self, key):
        """ Compile a part (a key of `slow`) within the block """
        self.key, self.part_start = key, time.time()
        try:
            yield
        finally:
            self.finished.add(key)
            self.key = None
        return

    def text(self, elapsed):
        done = sum(self.fracs[key] for key in self.finished if key in self.fracs)
        if self.key in self.fracs and self.key not in self.finished:
            done += min(time.time() - self.part_start, self.fracs[self.key])
        frac = min(done/sum(self.fracs.values()), 1)
        return f'  {self.progress.bar(frac)} {frac:4.0%}  {elapsed:.0f} s'

    def finish(self):
        Batch.current = None
        super().finish()
        return


class Compiling(event.Listener):
    """ Says when Numba starts compiling a function in `slow`, or one that `ss.jit` gave `ss_compiling`, with its progress until it is done (see `Ticker`); the functions it calls are compiled within it, and not said again """
    depth = 0
    ticker = None

    def on_start(self, ev):
        if not Compiling.depth:
            func = ev.data['dispatcher'].py_func
            info = getattr(ev.data['dispatcher'], 'ss_compiling', None) or slow.get(f'{func.__module__}.{func.__name__}') or slow.get(func.__module__)
            if info and Batch.current is None and compiling(info[0], 'Numba', *info[1:]): # Within a batch, its bar says it
                Compiling.ticker = Ticker(info[0], 'Numba', *info[1:])
        Compiling.depth += 1
        return

    def on_end(self, ev):
        Compiling.depth -= 1
        if not Compiling.depth and Compiling.ticker is not None:
            Compiling.ticker.finish()
            Compiling.ticker = None
        return


event.register('numba:compile', Compiling())
