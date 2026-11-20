"""
The backends: how the solvers' numbers are computed (see `ss.options.backend`), and the machinery around the kernels (the compiled functions that the solvers run)

The kernels themselves are Numba's, in `ss.solvers.numba` (the reference), and Rust's, built from `../rust/` and swapped in by `rust`. What's here:

    ss.jit                                               # Compile a function (see its docstring)
    ss.solvers.backends.rust                             # Swaps in the Rust kernels, once built (see ss.options.backend)
    ss.solvers.backends.compiling                        # Choosing the kernels for a compartmental run, ss.compile(), and Numba's cache
    ss.solvers.backends.workers                          # Starting the workers of ss.parallel() and of a calibration
    ss.solvers.backends.keep_memory(), release_memory()  # Keep large freed arrays for reuse (the default), or give them back
    ss.solvers.backends.timed_compile()                  # Say what is being compiled within a block, about how long it takes, and then how long it took (timed_first_call() for a function's first call)

Used with: agents and compartments; declared and coded modules
"""
from . import compiling
from . import workers
from .memory import keep as keep_memory, release as release_memory
keep_memory() # Large arrays freed are kept for reuse, so new ones take no page faults: see memory.py
from .jit import jit # ss.solvers.backends.jit is the function; its module is starsim.solvers.backends.jit
from .compiling import timed as timed_compile, timed_first as timed_first_call
