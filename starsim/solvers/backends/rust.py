"""
Not for the user: Starsim's compiled kernels in Rust, as the other kernels to Numba's: with `ss.options.backend = 'auto'` (the default) when they are there (in
the wheels, or built), or `'rust'` (or `STARSIM_BACKEND=rust`), the
serial and threaded passes over the agents and a network's edges (`ss.solvers.numba.passes`), the pass of `solver=ss.solvers.abm(step='events')` (`ss.solvers.numba.agent_events`), the small kernels of `ss.arrays`, `ss.networks`,
`ss.dists`, `ss.distributions` and `ss.People`, the draws of `ss.Dist` (`draws.many`), and the compartmental kernels (`ss.solvers.numba.compartmental`: see
`compartmental()`) run in Rust, with the same arguments and results (the same
numbers, bit for bit, whatever the number of threads). The Rust kernels are compiled ahead of time, so a first run takes no compiling, and
they run at least as fast as Numba's in nearly every case. A threaded pass takes as many threads as Numba has (see `ss.options.threads`),
from a pool of its own that starts the first time a pass is threaded (so the workers of `ss.parallel()`, with one thread each, start none)
and waits between passes without spinning. Only `ss.jit` (users' own code) stays in Numba. See ../rust/README.md.

The wheels include them (see `pyproject.toml`); from a clone, build them from the crate in `starsim/solvers/rust/` (needs a Rust toolchain; about 90 s the first time):

    cd starsim/solvers/rust && ./build

Used with: agents and compartments; declared and coded modules
"""
import zlib
import importlib.util
import collections
import sciris as sc
import starsim as ss
from ..numba import passes, agent_events, agent_moves, draws

__all__ = ['use', 'built', 'on', 'fell', 'compartmental']

on = False # Whether the Rust kernels are in use (see use()): with ss.options.backend = 'rust', or 'auto' when they have been built
fell = collections.Counter() # How often each kernel has gone to Numba, by kernel, error, and the types of its arrays

def built():
    """ Whether the Rust kernels have been built (see starsim/solvers/rust/README.md) """
    return importlib.util.find_spec('starsim.solvers.backends._rs') is not None

def fallback(rs, nb):
    """ The Rust kernel, or the Numba one for arrays of a type the Rust one does not take; a panic in Rust (e.g. an index out of bounds) raises a RuntimeError """
    def kernel(*args):
        try:
            return rs(*args)
        except TypeError as e:
            fell[nb.__name__, str(e), tuple(str(getattr(a, 'dtype', '')) for a in args)] += 1
            return nb(*args)
        except BaseException as e: # PyO3's PanicException derives from BaseException, so "except Exception" would not catch it
            if type(e).__name__ != 'PanicException':
                raise
            raise RuntimeError(f'{nb.__name__} (Rust) failed: {e}') from None
    return kernel

def modules():
    """ Where the kernels are: modules, a class (whose kernels are static methods), and `draws.many` (a dict) """
    return sc.obj(passes=passes, agent_events=agent_events, agent_moves=agent_moves, arrays=ss.arrays, networks=ss.networks, dists=ss.dists,
                      multi_random=ss.distributions.multi_random, many=draws.many)

single = ['levels', 'crossed', 'infection_probs', 'put_count', 'set_exits', 'edge_index', 'candidates_indexed', 'partners'] # The passes of ss.solvers.numba.passes with no threaded version
drivers = ['census', 'flatnonzero', 'exit_due', 'group_sums', 'infect', 'due', 'keep_alive', 'due_alive', 'count_alive', 'keep_raw', 'random_edges', 'candidates', 'can_transmit', 'ending', 'touching',
           'women', 'debuts', 'compress', 'competing_exits', 'competing_agent_exits'] # Those with a serial and a threaded driver (X_drivers), both the Rust kernel X (the number of chunks says which)
names = dict(passes=single + [f'{k}_drivers' for k in drivers],
             agent_events=['step_events'], agent_moves=['due_agents', 'step_moves'], arrays=['nb_true_raw', 'nb_indexer', 'nb_true', 'nb_count'], networks=['nb_transmit', 'fisher_yates_shuffle', '_mark_partners'], dists=['_hash_uniforms_fill'],
             multi_random=['combine_rvs', 'combine2_rvs'], many=['expon', 'weibull', 'binomial', 'poisson', 'nbinom'])
numba = None # The Numba kernels, once replaced

def get(where, k):
    """ A kernel as it is stored: an entry of a dict, a static method of a class, or a function of a module """
    return where[k] if isinstance(where, dict) else where.__dict__[k] if isinstance(where, type) else getattr(where, k)

def put(where, k, f):
    """ Put a kernel in place, as get() finds it """
    if isinstance(where, dict):
        where[k] = f
    elif isinstance(where, type):
        setattr(where, k, f if isinstance(f, staticmethod) else staticmethod(f))
    else:
        setattr(where, k, f)
    return

def source_hash():
    """ The hash of the crate's source beside this file (as `starsim/solvers/rust/build.rs` works it out when it builds the library), or None if there is no source (a wheel has none, and nothing to go stale) """
    crate = sc.thispath(__file__).parent/'rust'
    if not (crate/'Cargo.toml').exists():
        return None
    files = sorted([crate/'Cargo.toml', *(crate/'src').rglob('*.rs')], key=lambda path: path.relative_to(crate).as_posix())
    crc = 0
    for path in files:
        crc = zlib.crc32(path.read_bytes(), crc)
    return crc

def load():
    """ The compiled Rust kernels, with a clear error if they have not been built or cannot be loaded """
    try:
        from . import _rs
    except ImportError as e:
        if built():
            errormsg = f'The Rust kernels (ss.options.backend="rust") have been built but cannot be loaded: rebuild them with "./build" in the starsim/solvers/rust/ folder of the Starsim repository ({e})'
        else:
            errormsg = f'The Rust kernels (ss.options.backend="rust") have not been built: run "./build" in the starsim/solvers/rust/ folder of the Starsim repository, which needs a Rust toolchain (https://rustup.rs), or use ss.options.backend="auto" ({e})'
        raise ImportError(errormsg) from e
    return _rs

def auto():
    """ For ss.options.backend = 'auto': the Rust kernels if they have been built, else Numba's, also with a warning if a build cannot be used (e.g. one older than the Python that calls it, or than its source) """
    if built():
        try:
            return use(True)
        except (ImportError, AttributeError) as e:
            ss.warn(f'Using the Numba kernels, since the Rust ones cannot be used: {e}')
    return use(False)

def use(rust=True):
    """ Run the Rust kernels (True) or the Numba ones (False); called by `ss.options(backend=...)` """
    global numba, on
    mods = modules()
    if numba is None:
        numba = {(m, k): get(mods[m], k) for m, ks in names.items() for k in ks}
    if rust:
        rs = load()
        missing = [k for k in ss.solvers.numba.compartmental.entries + ['evaluate'] if not hasattr(rs.compartmental, k)]
        if missing: # Checked now rather than at the first run that needs them, as for every other kernel
            errormsg = f'The Rust kernels are older than the Python that calls them (missing {sc.strjoin(missing)}): rebuild them with "./build" in the starsim/solvers/rust/ folder'
            raise AttributeError(errormsg)
        source = source_hash()
        if source is not None and source != getattr(rs, 'source_hash', None): # An old library would give old numbers, with nothing to say so
            errormsg = 'The Rust kernels were built from different source than is here, so may give different numbers: rebuild them with "./build" in the starsim/solvers/rust/ folder'
            raise AttributeError(errormsg)
        kernels = {('passes', k): fallback(getattr(rs, k), numba['passes', k]) for k in single}
        kernels.update({('passes', f'{k}_drivers'): tuple(fallback(getattr(rs, k), nb) for nb in numba['passes', f'{k}_drivers']) for k in drivers})
        for m in ['agent_events', 'agent_moves', 'arrays', 'networks', 'dists', 'multi_random']: # Each Rust kernel has the name of the Numba one
            for k in names[m]:
                nb = numba[m, k]
                kernels[m, k] = fallback(getattr(rs, k), nb.__func__ if isinstance(nb, staticmethod) else nb) # A static method of multi_random, as its function
        for k in names['many']: # The draws of ss.Dist (see ss.dists.inverse())
            kernels['many', k] = fallback(getattr(rs, 'many_' + k), numba['many', k])
    else:
        kernels = numba
    for (m, k), f in kernels.items():
        put(mods[m], k, f)
    on = bool(rust)
    return


class Compartmental:
    """
    The compartmental kernels in Rust (`ss.solvers.numba.compartmental` when the Rust kernels are in use): the entries that the engines call
    (`run()`, `run_many()`, `one_step()`, `advance()`, `advance_threads()`, `apply_events()`, `apply_shifts()`,
    `settle_places()`, `evaluate()`), with the same arguments and results as Numba's; every other name is that of `ss.solvers.numba.compartmental`

    It exists so that the engine can call the Rust kernels and the Numba ones alike.
    """
    def __init__(self, rs):
        for name in ss.solvers.numba.compartmental.entries + ['evaluate']:
            setattr(self, name, getattr(rs.compartmental, name))
        return

    def __getattr__(self, name):
        return getattr(ss.solvers.numba.compartmental, name)

compartmental_set = None # The compartmental kernels in Rust, once loaded

def compartmental():
    """ The compartmental kernels in Rust (see `Compartmental`), which need no compiling """
    global compartmental_set
    if compartmental_set is None:
        compartmental_set = Compartmental(load())
    return compartmental_set
