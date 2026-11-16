"""
Not for the user: compiling the kernels, and choosing between compiled and plain Python for a compartmental run

Which compartmental kernels a run takes (`choose()`): Rust's, if they are in use; else Numba's, compiled, or the same code run
as plain Python (`uncompiled()`), from about how long the run would take as plain Python (`plain_seconds()`). `ss.compile()`
compiles Numba's ahead of time, and `compile_status()` says whether they are compiled.

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
import importlib
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

# Starsim's functions that take long enough to compile to say so, by module and function (or by module): what they are, and how long compiling them takes (s), erring long (measured 2026-11-07: 34 to 43 s for many runs, 32 to 46 and 33 to 55 s for the particle filter's)
slow = {
    'starsim.solvers.numba.passes':                         ('the agent solver', 2, 15), # And for ss.compile(), the rest of what sims with agents compile (30 s measured), as 'agents'
    'agents':                                               ('the agent solver and what sims with agents use', 20, 60),
    'starsim.solvers.numba.compartmental.run':              ('the compartmental solvers', 60, 150), # 63 s measured
    'starsim.solvers.numba.compartmental.one_step':         ('a step of the compartmental solvers', 5, 20),
    'starsim.solvers.numba.compartmental.run_many':         ('the compartmental solvers for many runs', 30, 70),
    'starsim.solvers.numba.compartmental.advance':          ("the particle filter's steps", 30, 70),
    'starsim.solvers.numba.compartmental.advance_threads':  ("the particle filter's steps in threads", 40, 80),
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

compile_seconds = 120 # About how long compiling the kernels for a first run takes, once per install (86 to 128 s measured, 2026-10-31, with nothing in Numba's cache): runs that would take longer than this as plain Python are compiled first (by default: see choose())
compile_all_seconds = 360 # About how long `compile()` takes (360 s measured on a busy machine, 2026-10-31; 153 s on another, 2026-11-07); the time of each part is in `slow`
note_after = 1 # The seconds of a plain run above which choose() says how long it would take compiled
plain_total = 0.0 # The seconds that the runs taken as plain Python in this session would have taken, as estimated: see choose()
noted = False # Whether choose() has said that a run is taken as plain Python, in this session
chosen = False # Whether choose() has chosen to compile the kernels, in this session
plain_copy = None # The copy of the kernels that runs as plain Python, once made: see uncompiled()
on_disk = None # Whether the compiled kernels were found in Numba's cache, once looked for: see is_compiled()


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


def is_compiled():
    """ Whether the kernels are compiled already: in this session, or in Numba's cache from an earlier one (an index that is out of date reads as empty) """
    global on_disk
    step = ss.solvers.numba.compartmental.step
    if on_disk is None:
        on_disk = bool(step._cache._cache_file._load_index())
    return on_disk or bool(step.signatures)


def work(p, y):
    """
    About how many operations working out a program's rates once takes, from its size alone: each group times each
    flow, target, operation, compartment, and compartment of an infectious state; for each exposure, through each
    mixing term, each group and cell, each entry of the time fraction between places that it uses (times the levels in
    each place), and each pair of cells that mix in a place (a quarter as many for a whole matrix of more than 16,
    which BLAS takes); and each entry of a mover times its compartments. For `plain_seconds()`.
    """
    nc, ng = y.shape
    size = ng*(len(p.src) + len(p.tgt_comp) + len(p.ops) + nc + len(p.set_comps))
    if len(p.ops): # Exposure is worked out only for rates that depend on the compartments
        mixing = 0.0
        for t in range(len(p.sizes)):
            cells, block = int(p.sizes[t]), int(p.block[t])
            pairs = len(p.sh_val)*block
            mixing += 3*ng + cells + (2*p.use[0, t] + p.use[2, t])*pairs + p.use[1, t]*cells*block*(0.25 if block == cells > 16 else 1)
        size += np.sum(p.exp_set >= 0)*mixing
    if p.n_movers:
        size += np.sum(np.diff(p.mov_starts[0])*np.diff(p.mov_starts[1]))
    return size


def evals(npts, solver, n_sub):
    """ How many times a run works out the rates: at each step, each substep (see `substeps()`), and each stage (four for RK4 and tau-leaping) """
    solvers = ss.solvers.numba.compartmental.solvers
    stages = 4 if solver in [solvers['rk4'], solvers['tau']] else 1
    return npts*stages*max(n_sub, 1)


def plain_seconds(p, y, npts, solver, n_sub=1):
    """
    About how long a run would take as plain Python, in seconds, from the program's size alone, so that it is the same
    for a model whatever has run before and whatever else the machine is doing: 10 µs each time the rates are worked
    out (see `evals()`), plus 0.6 µs for each operation that takes (see `work()`); the exact solver takes about one
    event for each person and flow. Measured once (2026-10-25) on 13 plain runs of the ODE (declared SIRs with 1
    to 100 groups, over 20 and 50 places, measles metapop over 260 districts, MEmilio's 50 counties, two SEIRs with
    stages tracked jointly, a stiff SIS, and an SIS with aging), it is within 0.75 to 1.6 times the time taken, and
    0.35 for the SIS with aging, whose epidemic needs more substeps than its rates at the start (see `substeps()`):
    enough to choose whether to compile. The binomial and tau-leaping solvers add 70 µs for each draw, one for each flow, group, and
    step or substep (measured 2026-10-31 on SEIRSs with 1 to 40 groups, within 0.7 to 1.1 times the time taken; without it, 0.1 to 0.2).

    Args:
        n_sub (int): the substeps in each step (see `substeps()`)
    """
    solvers = ss.solvers.numba.compartmental.solvers
    if solver == solvers['gillespie']:
        return y.sum()*len(p.src)*(1e-5 + 3e-7*work(p, y))
    draws = npts*max(n_sub, 1)*len(p.src)*y.shape[1]*7e-5 if solver in [solvers['binomial'], solvers['whole'], solvers['tau']] else 0.0
    return evals(npts, solver, n_sub)*(1e-5 + 6e-7*work(p, y)) + draws


def substeps(rates, dt, n_sub, solver):
    """ About how many substeps a solver takes in each step, for `plain_seconds()`: those given (`n_sub`), or else for the ODE, enough for the fastest of the rates out of a compartment given, as `step()` chooses them """
    if n_sub > 0 or solver >= 2: # The stochastic solvers that take steps of their own length count them in their own way
        return max(n_sub, 1)
    rates = np.ravel(rates)
    fastest = np.max(rates[np.isfinite(rates)], initial=0.0)
    return max(int(np.ceil(fastest*dt/0.25)), 1)


def choose(seconds=0.0):
    """
    The compartmental kernels to take a run with: the Rust ones (see `ss.solvers.backends.rust`) if they are in use, else Numba's (`ss.solvers.numba.compartmental`), compiled by
    Numba, or the same code as plain Python (see `uncompiled()`), as `ss.options.backend` says

    Compiling them takes about two minutes (`compile_seconds`), once per install (or ahead of time, with `ss.compile()`). By default
    (`ss.options.backend = 'auto'`, without Rust) they are compiled first if the runs taken as plain Python in this session, with
    this one, would take longer than that (as a long run, or a calibration's many small runs, would), with a note
    saying so; and otherwise the run is taken as plain Python, with a note (once a session, for a run of more than
    `note_after` seconds) of how long it would take each way. A note is printed whatever the sim's `verbose`, since
    minutes of silence look like a hang.

    Args:
        seconds (float): about how long the run (or runs) would take as plain Python: see `plain_seconds()`
    """
    global plain_total, noted, chosen
    if ss.solvers.backends.rust.on: # Compiled ahead of time
        return ss.solvers.backends.rust.compartmental()
    mode = ss.options.backend
    if mode == 'python':
        return uncompiled()
    K = ss.solvers.numba.compartmental
    if chosen or is_compiled():
        return K
    if mode == 'numba':
        chosen = True # Each part says so as it is compiled: see announce()
        return K
    total = plain_total + seconds
    if total < compile_seconds:
        plain_total = total
        if seconds >= note_after and not noted:
            noted = True
            print(f'Note: running the solver as plain Python (about {seconds:.0f} s; 15 to 300 times faster compiled, after compiling once per install for about {compile_seconds/60:.0f} minutes: ss.compile())')
        return uncompiled()
    chosen = True
    which = 'this run' if seconds >= compile_seconds else 'the runs in this session'
    print(f'Note: compiling the solver (once per install), since {which} would take about {total:.0f} s as plain Python')
    return K


def compile(check=False, verbose=True):
    """
    Compile the kernels now (those of the compartmental solvers, the passes over the agents and the rest of what sims with agents use), so that no run waits for them: e.g. after installing Starsim (as `starsim init compile` does), in a Docker image, in CI, or before a calibration

    Compiling takes about five minutes (`compile_all_seconds`), once per install (and after an upgrade); the result is kept in Numba's cache,
    beside the package (or, if that cannot be written to, in the user's cache folder, then a temporary folder, or in
    `NUMBA_CACHE_DIR` if it is set). Until the kernels are compiled, small runs are taken as plain Python: see `ss.options.backend`. With the
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
    parts = {'agents':ss.solvers.numba.passes.bounds} | {f'{K.__name__}.{name}':getattr(K, name) for name in ['run', 'one_step', 'run_many', 'advance', 'advance_threads']}
    cold = [key for key,func in parts.items() if not (func.signatures or func._cache._cache_file._load_index())] # In the order compiled below
    batch = None
    if verbose and cold:
        announce('the solver kernels', 'Numba', sum(slow[k][1] for k in cold), sum(slow[k][2] for k in cold))
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
    parts = {'the compartmental solvers':N.compartmental.step, 'the agent solver':N.passes.bounds, 'the draws':N.draws.key} # One function for each file, since a file's functions go out of date together; not one inlined everywhere (e.g. uniform), which is never compiled on its own
    cold = [part for part,func in parts.items() if not (func.signatures or func._cache._cache_file._load_index())]
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
            if info and Batch.current is None and announce(info[0], 'Numba', *info[1:]): # Within a batch, its bar says it
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
