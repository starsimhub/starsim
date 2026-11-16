"""
Not for the user: the workers of `ss.parallel()` and of a calibration: how they are started (by fork where it is safe, else by forkserver), with one thread each for the kernels, and a clear error if they run the main script again

Used with: agents and compartments; declared and coded modules
"""
import os
import sys
import ast
import importlib
import contextlib
import functools as ft
import multiprocessing as mp
import numba as nb
import starsim as ss

__all__ = ['worker', 'starting_workers', 'start_method', 'workers_failed']


def one_thread(func, *args, main_modules=None, backend=None, **kwargs):
    """
    Run a function with one thread for the kernels (the passes over the agents, see `ss.options.threads`, and the runs of a particle filter): for the workers of `ss.parallel()` and of a calibration, which share the cores already

    With `backend`, the main process's `ss.options.backend`, which a worker started afresh would not otherwise have

    With `main_modules` (see `worker()`), the modules imported in the main process's `__main__` are imported in this one's
    first, for a worker started afresh (by forkserver, see `start_method()`)
    """
    ss.options.threads = 1 # Every threaded pass asks this first (see `ss.solvers.numba.passes.n_chunks()`), so Numba's threads are never started (as nb.set_num_threads(1) would start them)
    if backend is not None and backend != ss.options.backend:
        ss.options.backend = backend
    if main_modules: # A function defined in a notebook or script looks up its names (e.g. cv of import covasim as cv) in __main__, which in a worker started afresh lacks them
        main = sys.modules['__main__']
        for alias,name in main_modules.items():
            if not hasattr(main, alias):
                try:
                    setattr(main, alias, importlib.import_module(name))
                except Exception: # A module that can't be imported by name (e.g. made at run time) is left out, as it was
                    pass
    return func(*args, **kwargs)


def worker(func):
    """ A function to run in each worker of `ss.parallel()` or of a calibration: with one thread for the kernels, the main process's backend, and the modules imported in `__main__` (see `one_thread()`) """
    main = sys.modules.get('__main__')
    modules = {alias:val.__name__ for alias,val in vars(main).items() if type(val) is type(sys) and not alias.startswith('_')} if main is not None else {}
    return ft.partial(one_thread, func, main_modules=modules, backend=ss.options.backend)


def unguarded_main():
    """
    Whether this is called from the top level of the main script outside `if __name__ == '__main__':`, which a worker started
    afresh (by forkserver, see `start_method()`) would run again, as Python's multiprocessing runs the main script in each
    """
    main = sys.modules.get('__main__')
    path = getattr(main, '__file__', None)
    if path is None or not os.path.isfile(path): # E.g. a notebook, which is not run again
        return False
    frame = sys._getframe()
    line = None
    while frame is not None: # The line of the main script's top level that this was called from
        if frame.f_globals is vars(main) and frame.f_code.co_name == '<module>':
            line = frame.f_lineno
        frame = frame.f_back
    if line is None:
        return False
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return False
    guards = [node for node in tree.body if isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)]
    return not any(node.lineno <= line <= node.end_lineno for node in guards)


@contextlib.contextmanager
def starting_workers(method=None):
    """
    While the workers of `ss.parallel()` or of a calibration are started: if they are started afresh (by `method`, as `start_method()` gives) from the main script outside `if __name__ == '__main__':` (see
    `unguarded_main()`), the script hidden from them

    A forked worker is a copy of this process, so it keeps the script as it is. A worker started afresh (by forkserver, once threads that are not safe to fork have started; see `start_method()`) would otherwise
    run the script again, redoing everything before this call, and stop at this call with a RuntimeError ("An attempt has been made
    to start a new process before the current process has finished its bootstrapping phase"). Hidden, the workers start as from a
    notebook: what they run is pickled by value, and the modules the script imports are imported in them (see `one_thread()`).
    """
    main = sys.modules.get('__main__')
    hide = not forked(method) and unguarded_main()
    hidden = dict(__file__=main.__file__, __spec__=getattr(main, '__spec__', None)) if hide else {}
    try:
        if hide:
            del main.__file__
            main.__spec__ = None
        yield
    finally:
        for key,val in hidden.items():
            setattr(main, key, val)
    return


def fork_unsafe():
    """
    Why a fork of this process could deadlock, if it could: threads started that are not safe to fork (JAX's, or Numba's
    OpenMP layer, as GNU OpenMP is not; see Numba's `threading_layer='forksafe'`). Numba's TBB and workqueue layers are safe
    to fork, and so is the pool of the Rust kernels, which a forked worker starts again (it checks the pid).
    """
    if 'jax' in sys.modules:
        return 'JAX has been imported, and its threads are not safe to fork'
    try:
        layer = nb.threading_layer() # Only once Numba has started its threads
    except ValueError:
        return None
    return f"Numba's threads have started, with its {layer} layer, which is not safe to fork" if layer == 'omp' else None


def start_method():
    """
    How to start the workers of `ss.parallel()` and of a calibration: on Linux, by fork, the fastest, unless this process
    has started threads that a fork can deadlock on (see `fork_unsafe()`); then by forkserver, whose workers start clean.
    Elsewhere (macOS, Windows), as the platform does. Python 3.14 made forkserver the default on Linux, so fork is
    asked for by name.
    """
    if sys.platform == 'darwin' or 'fork' not in mp.get_all_start_methods(): # Fork is not safe on macOS, and Windows has none
        return None
    return 'forkserver' if fork_unsafe() else 'fork'


def forked(method):
    """ Whether workers started by `method` (as `start_method()` gives; None for the platform's default) are forked """
    return (method or mp.get_start_method()) == 'fork'


def workers_failed(E, method):
    """
    Raise a clear error for a pool whose workers could not start because they ran the main script again: a worker started
    afresh (by forkserver or spawn) imports the script, so its code must be under `if __name__ == '__main__':`. Otherwise
    the parent sees only an EOFError ("unexpected EOF"), and the worker prints multiprocessing's "bootstrapping" RuntimeError
    """
    bootstrap = isinstance(E, EOFError) or (isinstance(E, RuntimeError) and 'bootstrapping' in str(E))
    if forked(method) or not bootstrap:
        return
    method = method or mp.get_start_method()
    why = fork_unsafe() if method == 'forkserver' else None
    why = f', since {why}' if why else ''
    errormsg = (f'The workers could not start: they were started by {method} rather than by fork{why}, and a worker started this way runs the main script again, '
                f'including the call that starts them. Put the script\'s code under if __name__ == "__main__": (or run it in a notebook).')
    raise RuntimeError(errormsg) from E
