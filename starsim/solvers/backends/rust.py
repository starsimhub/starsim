"""
Not for the user: Starsim's compiled kernels in Rust, as the other kernels to Numba's: with `ss.options.kernels = 'auto'` (the default) when they have been
built, or `'rust'` (or `STARSIM_KERNELS=rust`), the
serial and threaded passes over the agents and a network's edges (`ss.solvers.backends.passes`), the small kernels of `ss.arrays`, `ss.networks`,
`ss.distributions`, `ss.people` and `ss.utils`, the draws of `ss.Dist` (`draws.many`), the compartmental kernels (`ss.kernels`: see
`compartmental()`), and the chain binomial's draws in JAX (`ss.solvers.backends.chain`) run in Rust, with the same arguments and results (the same
numbers, bit for bit, whatever the number of threads). The Rust kernels are compiled ahead of time, so a first run takes no compiling, and
they run at least as fast as Numba's in nearly every case. A threaded pass takes as many threads as Numba has (see `ss.options.threads`),
from a pool of its own that starts the first time a pass is threaded (so the workers of `ss.parallel()`, with one thread each, start none)
and waits between passes without spinning. Only `ss.jit` (users' own code) stays in Numba. See ../rust/README.md.

Build them from the crate in `starsim/solvers/rust/` (needs a Rust toolchain; about 15 s):

    cd starsim/solvers/rust && ./build

Used with: agents and compartments; declared and coded modules
"""
import sys
import importlib
import collections
import sciris as sc
import starsim as ss
from . import passes
from . import draws

__all__ = ['use', 'built', 'on', 'fell', 'compartmental']

on = False # Whether the Rust kernels are in use (see use()): with ss.options.kernels = 'rust', or 'auto' when they have been built
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
    return sc.obj(passes=passes, arrays=ss.arrays, networks=ss.networks, dists=ss.dists,
                      people=sys.modules['starsim.people'], multi_random=ss.distributions.multi_random, many=draws.many)

single = ['levels', 'crossed', 'infection_probs'] # The passes of ss.solvers.backends.passes with no threaded version
pairs = ['census', 'flatnonzero', 'infect', 'due', 'count_people', 'keep_alive', 'due_alive', 'count_alive', 'keep_raw'] # Those with a serial and a threaded driver by name, both the same Rust kernel (the number of chunks says which)
tuples = dict(edges='random_edges', candidates='candidates', can='can_transmit', ending='ending', touching='touching', women='women', debut='debuts', compress='compress', exits='competing_exits', agent_exits='competing_agent_exits') # Those whose two drivers are a pair, by the Rust kernel
names = dict(passes=single + [f'{k}_{kind}' for k in pairs for kind in ['serial', 'threaded']] + [f'{k}_drivers' for k in tuples],
             arrays=['nb_true_raw', 'nb_indexer', 'nb_true', 'nb_count'], networks=['nb_transmit', 'fisher_yates_shuffle', '_mark_partners'], dists=['_hash_uniforms_fill'],
             people=['count_gone'], multi_random=['combine_rvs', 'combine2_rvs'], many=['expon', 'weibull', 'binomial', 'poisson', 'nbinom'])
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

def load():
    """ The compiled Rust kernels, with a clear error if they have not been built or cannot be loaded """
    try:
        from . import _rs
    except ImportError as e:
        if built():
            errormsg = f'The Rust kernels (ss.options.kernels="rust") have been built but cannot be loaded: rebuild them with "./build" in the starsim/solvers/rust/ folder of the Starsim repository ({e})'
        else:
            errormsg = f'The Rust kernels (ss.options.kernels="rust") have not been built: run "./build" in the starsim/solvers/rust/ folder of the Starsim repository, which needs a Rust toolchain (https://rustup.rs), or use kernels="auto" ({e})'
        raise ImportError(errormsg) from e
    return _rs

def auto():
    """ For ss.options.kernels = 'auto': the Rust kernels if they have been built, else Numba's, also with a warning if a build cannot be used (e.g. one older than the Python that calls it) """
    if built():
        try:
            return use(True)
        except (ImportError, AttributeError) as e:
            ss.warn(f'Using the Numba kernels, since the Rust ones cannot be used: {e}')
    return use(False)

def use(rust=True):
    """ Run the Rust kernels (True) or the Numba ones (False); called by `ss.options(kernels=...)` """
    global numba, on
    mods = modules()
    if numba is None:
        numba = {(m, k): get(mods[m], k) for m, ks in names.items() for k in ks}
    if rust:
        rs = load()
        missing = [k for k in ss.kernels.entries + ['evaluate'] if not hasattr(rs.kernels, k)] + [k for k in ['chain_handler'] if not hasattr(rs, k)]
        if missing: # Checked now rather than at the first run that needs them, as for every other kernel
            errormsg = f'The Rust kernels are older than the Python that calls them (missing {sc.strjoin(missing)}): rebuild them with "./build" in the starsim/solvers/rust/ folder'
            raise AttributeError(errormsg)
        kernels = {('passes', k): fallback(getattr(rs, k), numba['passes', k]) for k in single}
        kernels.update({('passes', f'{k}_{kind}'): fallback(getattr(rs, k), numba['passes', f'{k}_{kind}']) for k in pairs for kind in ['serial', 'threaded']})
        kernels.update({('passes', f'{k}_drivers'): tuple(fallback(getattr(rs, r), nb) for nb in numba['passes', f'{k}_drivers']) for k,r in tuples.items()})
        kernels.update({
            ('arrays', 'nb_true_raw'):        fallback(rs.true_raw, numba['arrays', 'nb_true_raw']),
            ('arrays', 'nb_indexer'):         fallback(rs.indexer, numba['arrays', 'nb_indexer']),
            ('networks', 'nb_transmit'):      fallback(rs.transmit, numba['networks', 'nb_transmit']),
            ('dists', '_hash_uniforms_fill'): fallback(rs.hash_uniforms, numba['dists', '_hash_uniforms_fill']),
            ('arrays', 'nb_true'):            fallback(rs.true_vals, numba['arrays', 'nb_true']),
            ('arrays', 'nb_count'):           fallback(rs.count, numba['arrays', 'nb_count']),
            ('networks', 'fisher_yates_shuffle'): fallback(rs.shuffle, numba['networks', 'fisher_yates_shuffle']),
            ('people', 'count_gone'):         fallback(rs.count_gone, numba['people', 'count_gone']),
            ('networks', '_mark_partners'):   fallback(rs.mark_partners, numba['networks', '_mark_partners']),
            ('multi_random', 'combine_rvs'):  fallback(rs.combine_rvs, numba['multi_random', 'combine_rvs'].__func__),
            ('multi_random', 'combine2_rvs'): fallback(rs.combine2_rvs, numba['multi_random', 'combine2_rvs'].__func__),
        })
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
    The compartmental kernels in Rust (`ss.kernels` when the Rust kernels are in use): the entries that the engines call
    (`run()`, `run_many()`, `one_step()`, `advance()`, `advance_threads()`, `apply_events()`, `apply_shifts()`,
    `settle_places()`, `evaluate()`), with the same arguments and results as Numba's; every other name is ss.kernels' own

    It exists so that the engine can call the Rust kernels and the Numba ones alike.
    """
    def __init__(self, rs):
        for name in ss.kernels.entries + ['evaluate']:
            setattr(self, name, getattr(rs.kernels, name))
        return

    def __getattr__(self, name):
        return getattr(ss.kernels, name)

compartmental_set = None # The compartmental kernels in Rust, once loaded

def compartmental():
    """ The compartmental kernels in Rust (see `Compartmental`), which need no compiling """
    global compartmental_set
    if compartmental_set is None:
        compartmental_set = Compartmental(load())
    return compartmental_set
