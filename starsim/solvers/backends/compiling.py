"""
Not for the user: compiling the kernels, and choosing which a compartmental run takes

Which compartmental kernels a run takes (`choose()`): Rust's, if they are in use; else Numba's, compiled, or with
`ss.options.backend = 'python'`, the same code run as plain Python (`uncompiled()`). `ss.compile()` compiles Numba's
ahead of time, and `compile_status()` says whether they are compiled.

Numba's on-disk cache is made stale by the contents of a module's file and of the files its functions call into.
Numba checks only the file of a cached function, so a change to a function it calls in
another file (e.g. a draw in `draws.py`, called by the kernels) would leave the old
compiled code in use. `depends()` adds those files to the stamp Numba checks. Numba's
stamp is also a file's time and size, which a fresh checkout or install changes and an edit of
the same size may not, so the stamp here is of the contents: a cache is used exactly when
the code it was compiled from is the code that is here, wherever the cache has come from.

Compiling takes seconds the first time (and after a change to the code), so `announce()` says
what is being compiled and about how long it takes, once a session for each thing: for Numba,
when it starts compiling (not when it loads from its cache) one of the sets of functions in
`slow`, or a function from `ss.jit`, with a bar of the time taken so far, redrawn until it is
done (see `Ticker`), or for `ss.compile()`, one bar for all of them (see `Batch`); for JAX, where Starsim compiles something.

Used with: agents and compartments; declared and coded modules
"""

import os
import sys
import time
import importlib.util
import contextlib
import hashlib
import threading
import multiprocessing as mp
import numpy as np
import numba as nb
import numba.core.caching as caching
from numba.core import event
import sciris as sc
import starsim as ss

deps = {} # The extra files each source file's cache depends on
hashes = {} # The hash of each file's contents, with the time and size it was read at, so that each file is read once

def takes(what, low, high):
    """ What a set of functions is, and the least and most seconds compiling it takes: the entries of `slow`, and of `ss.jit`'s functions, which Compiling says while they compile """
    return sc.obj(what=what, low=low, high=high)


# Starsim's functions that take long enough to compile to say so, by module and function (or by module): what they are, and how long compiling them takes (s), erring long (measured 2026-11-07: 34 to 43 s for many runs, 32 to 46 and 33 to 55 s for the particle filter's)
slow = {
    'starsim.solvers.numba.passes':                         takes('the agent solver', 2, 15), # And for ss.compile(), the rest of what sims with agents compile (30 s measured), as 'agents'
    'starsim.solvers.numba.agent_events.step_events':       takes('the agent solver by events', 3, 15), # Not compiled by ss.compile(): on the first run with abm(step='events')
    'agents':                                               takes('the agent solver and what sims with agents use', 20, 60),
    'starsim.solvers.numba.compartmental.run':              takes('the compartmental solvers', 60, 150), # 63 s measured
    'starsim.solvers.numba.compartmental.one_step':         takes('a step of the compartmental solvers', 5, 20),
    'starsim.solvers.numba.compartmental.run_many':         takes('the compartmental solvers for many runs', 30, 70),
    'starsim.solvers.numba.compartmental.advance':          takes("the particle filter's steps", 30, 70),
    'starsim.solvers.numba.compartmental.advance_threads':  takes("the particle filter's steps in threads", 40, 80),
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


def announce(what, engine, low, high):
    """ Say that something is being compiled, and about how long it takes (in seconds), once a session for each thing; True if it was said (see `timed()` to say how long it took, too) """
    if what not in said and ss.options.verbose:
        said.add(what)
        print(f'  Compiling {what} with {engine}, estimated time: {low}-{high} s')
        return True
    return False


@contextlib.contextmanager
def timed(what, engine, low, high):
    """ Say that something is being compiled within the block, with the estimate (see `announce()`), and then how long it took """
    said_now = announce(what, engine, low, high)
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


#%% Choosing and compiling the compartmental kernels

compile_all_seconds = 360 # About how long `compile()` takes (360 s measured on a busy machine, 2026-10-31; 153 s on another, 2026-11-07); the time of each part is in `slow`
plain_copy = None # The copy of the kernels that runs as plain Python, once made: see uncompiled()


def uncompiled():
    """
    The compartmental kernels as plain Python: the code of `ss.solvers.numba.compartmental`, run without compiling it, which gives the same results with the
    same random numbers (the draws are the same code, `ss.solvers.numba.draws`, run as plain Python too, and Numba follows NumPy's
    algorithms for the uniforms). Much slower for a large model, but there is
    nothing to wait for. Its entry points ignore the overflow of the random streams' integer arithmetic, as compiled
    code does.
    """
    global plain_copy
    if plain_copy is None:
        K = ss.solvers.numba.compartmental
        spec = importlib.util.spec_from_file_location('starsim.solvers.numba.plain_compartmental', K.__file__)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        for name in K.entries:
            setattr(copy, name, quiet(getattr(copy, name)))
        plain_copy = copy
    return plain_copy


def quiet(func):
    """ A kernel run as plain Python, with NumPy's warnings about integer overflow off (see `uncompiled()`) """
    def wrapper(*args, **kwargs):
        with np.errstate(over='ignore'):
            return func(*args, **kwargs)
    return wrapper


def numba_kernel(module, name):
    """ A Numba kernel of `ss.solvers.numba.<module>`, also while Rust's copy is in its place (see `ss.solvers.backends.rust.use()`) """
    held = ss.solvers.backends.rust.numba
    return held[module, name] if held and (module, name) in held else getattr(getattr(ss.solvers.numba, module), name)


def agent_kernels():
    """ The Numba functions that stand for the agent solver (see `compile_status()`) """
    return [ss.solvers.numba.passes.bounds]


def cached(func, in_session=True):
    """ Whether a function compiled by Numba is compiled already: in Numba's cache on disk (an index that is out of date reads as empty), or in this session if `in_session` """
    return bool((in_session and func.signatures) or func._cache._cache_file._load_index())


def choose():
    """
    The compartmental kernels to take a run with: the Rust ones (see `ss.solvers.backends.rust`) if they are in use; else Numba's
    (`ss.solvers.numba.compartmental`), compiled by Numba on their first run (once per install, or ahead of time with `ss.compile()`),
    or with `ss.options.backend = 'python'`, the same code as plain Python (see `uncompiled()`)
    """
    if ss.solvers.backends.rust.on: # Compiled ahead of time
        return ss.solvers.backends.rust.compartmental()
    if ss.options.backend == 'python':
        return uncompiled()
    return ss.solvers.numba.compartmental


def compile(check=False, verbose=True):
    """
    Compile the kernels now (those of the compartmental solvers, the passes over the agents and the rest of what sims with agents use), so that no run waits for them: e.g. after installing Starsim (as `starsim init compile` does), in a Docker image, in CI, or before a calibration

    Compiling takes about five minutes (`compile_all_seconds`), once per install (and after an upgrade); the result is kept in Numba's cache,
    beside the package (or, if that cannot be written to, in the user's cache folder, then a temporary folder, or in
    `NUMBA_CACHE_DIR` if it is set). Otherwise, the first run that needs them compiles them. With the
    Rust kernels in use, there is nothing to compile: they are compiled already. To see which is the case, use `ss.compile(check=True)`.

    Args:
        check (bool): rather than compile, say which backend is in use and whether it is compiled (as `starsim init compile --check` does), and return that as an `sc.obj` of `backend`, `compiled`, `missing`, and `summary` (see `compile_status()`)
        verbose (bool): whether to print what it does (with `check=True`, the summary)

    Examples:
        ```python
        ss.compile()
        ss.compile(check=True) # e.g. "Starsim backend: Rust, compiled"
        ```
    """
    if check:
        status = compile_status()
        if verbose:
            print(status.summary)
        return status
    if ss.solvers.backends.rust.on:
        if verbose:
            print('Nothing to compile: the Rust kernels are in use, and are compiled already (for Numba\'s, set ss.options.backend="numba" first)')
        return
    T = sc.timer()
    K = ss.solvers.numba.compartmental
    parts = {'agents':agent_kernels()} | {f'{K.__name__}.{name}':[getattr(K, name)] for name in ['run', 'one_step', 'run_many', 'advance', 'advance_threads']}
    cold = [key for key,funcs in parts.items() if not all(map(cached, funcs))] # In the order compiled below
    batch = None
    if verbose and cold:
        announce('the solver kernels', 'Numba', sum(slow[k].low for k in cold), sum(slow[k].high for k in cold))
        batch = Batch('the solver kernels', cold) # One bar for all the parts
    part = batch.part if batch is not None else lambda key: contextlib.nullcontext()
    def make(solver='binomial'):
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.1), name='sir')
        return ss.Sim(sir, total_pop=100, unit='day', stop=3, solver=solver, verbose=0)
    try:
        with ss.options.context(backend='numba', verbose=ss.options.verbose if verbose else 0):
            with part('agents'): # The passes over the agents (ss.solvers.numba.passes), which are cached apart from the kernels, and the rest of what sims with agents compile on their first run; the quickest
                make('abm').run()
                for sim in typical_sims():
                    sim.run()
            with part('starsim.solvers.numba.compartmental.run'):
                make().run() # A whole run, in one call: run(), which has every solver's code
            with part('starsim.solvers.numba.compartmental.one_step'):
                make().run(until=1) # A run taken a step at a time: one_step()
            sim = make().init()
            engine = sim.diseases[0].engine
            engine.init_program()
            with part('starsim.solvers.numba.compartmental.run_many'):
                engine.run_many([0, 1]) # An ensemble: run_many()
            for n,key in [(1, 'advance'), (2, 'advance_threads')]: # The particles of a particle filter: advance(), and advance_threads()
                with part(f'starsim.solvers.numba.compartmental.{key}'):
                    engine.restart()
                    ss.solvers.Particles(engine, n, np.random.default_rng()).advance(2)
            y, p = engine.y.copy(), engine.program # And those called on their own
            K.apply_events(engine.make_rng(), y, engine.register, 0, True)
            K.apply_shifts(engine.make_rng(), y, engine.shifts, 0, True, np.zeros((0, 2, y.shape[1])))
            K.settle_places(y, p, K.workspace(y, p), engine.dt, np.zeros((p.n_trans + p.n_movers, y.shape[1])))
    finally:
        if batch is not None:
            batch.finish() # Says how long it took
    if verbose and batch is None:
        print(f'Compiled the solver kernels in {T.toc(output=True):.0f} s')
    return


def compile_status():
    """
    Whether Starsim's numerical code is compiled, without compiling anything (it takes a few milliseconds)

    From the shell, `starsim init compile --check` says the same, and exits with 1 if it isn't compiled.

    Returns:
        An `sc.obj` of `backend` ('rust', 'numba', or 'python': what runs the kernels), `compiled` (True only if they run at full speed from the first run, from the current code), `missing` (the parts for which they don't), and `summary` (a line saying so)

    Example:
        ```python
        print(ss.solvers.compile_status().summary) # Or ss.compile(check=True)
        ```
    """
    rust = ss.solvers.backends.rust
    N = ss.solvers.numba
    parts = {'the compartmental solvers':[N.compartmental.step], 'the agent solver':agent_kernels(), 'the draws':[N.draws.key]} # One function for each file, since a file's functions go out of date together; not one inlined everywhere (e.g. uniform), which is never compiled on its own
    cold = [part for part,funcs in parts.items() if not all(map(cached, funcs))]
    if rust.on:
        backend = 'rust'
        missing = ['the Rust kernels'] if rust.source_hash() not in [None, rust.load().source_hash] else [] # None in a wheel, which has no source beside it
        if missing:
            summary = 'Starsim backend: Rust, OUT OF DATE (built before its source last changed, so results may be wrong): run "./build" in starsim/solvers/rust/'
        elif cold:
            summary = 'Starsim backend: Rust, compiled (Numba\'s copy is not, which only comparing the two needs)'
        else:
            summary = 'Starsim backend: Rust, compiled'
    elif ss.options.backend == 'python':
        backend = 'python'
        missing = list(parts)
        summary = 'Starsim backend: plain Python, NOT COMPILED, so runs are slow: set ss.options.backend = "auto"'
    else:
        backend = 'numba'
        missing = cold
        summary = f'Starsim backend: Numba, NOT COMPILED ({sc.strjoin(cold)}), so the first runs are slow: run "starsim init compile" or ss.compile()' if cold else 'Starsim backend: Numba, compiled'
    return sc.obj(backend=backend, compiled=not missing, missing=missing, summary=summary)


def note_uncompiled():
    """
    On import, say that Starsim's kernels aren't compiled (see `compile_status()`), unless `ss.options.uncompiled_warning` is
    False (`STARSIM_UNCOMPILED_WARNING=0`); not in the workers of `ss.parallel()`, nor for the `starsim` command
    """
    command = os.path.basename(getattr(sys, 'argv', [''])[0]) in ['-m', 'starsim', 'starsim.exe', 'starsim-script.py'] # The command (or python -m starsim), which says what it does itself
    if ss.options.uncompiled_warning and mp.parent_process() is None and not command:
        status = compile_status()
        if not status.compiled:
            print(status.summary)
    return



def typical_sims():
    """ Short sims with agents that between them use the Numba code that sims with agents compile on their first run (networks, demographics, pregnancy, interventions, and diseases written as code), for `compile()` """
    import starsim.library as ssl
    small = dict(n_agents=1000, start=2000, stop=2002, verbose=0)
    sims = [
        ss.Sim(diseases=[ss.SIS(beta=dict(randomnet=0.05, sexualnet=ss.peract(0.05))), ss.SIR(beta=dict(randomnet=0.1, sexualnet=ss.peract(0.1)))], networks=['random', 'sexual'], demographics=True, **small),
        ss.Sim(diseases=ssl.HIV(), networks='sexual', demographics=True, **small),
        ss.Sim(diseases=ssl.AgentSIR(), networks='random', interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5), **small),
        ss.Sim(demographics=[ss.Pregnancy(), ss.Deaths()], networks='prenatal', **small),
    ]
    if ss.options.threads != 1 and nb.get_num_threads() > 1: # The threaded passes, for a sim large enough to share among threads
        sims.append(ss.Sim(diseases='sir', networks='random', n_agents=ss.solvers.numba.passes.min_threaded, dur=ss.days(5), verbose=0))
    return sims


#%% Saying what is being compiled

class Ticker:
    """
    While Numba compiles something that takes long (see `Compiling`), a bar of the time taken so far against the most it is
    estimated to take, redrawn twice a second from a thread of its own, in a terminal or a notebook (see `ss.progress.Progress`);
    to a file, nothing until it is done. Then a line saying how long it took.
    """
    def __init__(self, what, engine, low, high):
        self.what, self.engine, self.low, self.high = what, engine, low, high
        self.progress = ss.progress.Progress()
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
        self.fracs = {key:(slow[key].low + slow[key].high)/2 for key in keys} # Each part's fraction of the bar
        self.finished = set() # The parts done
        self.key = None # The part being compiled, and when it started
        self.part_start = None
        said.update(slow[key].what for key in keys) # Not said again on its own, later in the session
        Batch.current = self
        super().__init__(what, 'Numba', sum(slow[key].low for key in keys), sum(slow[key].high for key in keys))
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
            if info and Batch.current is None and announce(info.what, 'Numba', info.low, info.high): # Within a batch, its bar says it (info is from takes(), in slow or from ss.jit)
                Compiling.ticker = Ticker(info.what, 'Numba', info.low, info.high)
        Compiling.depth += 1
        return

    def on_end(self, ev):
        Compiling.depth -= 1
        if not Compiling.depth and Compiling.ticker is not None:
            Compiling.ticker.finish()
            Compiling.ticker = None
        return


event.register('numba:compile', Compiling())
