"""
The machinery for speed: the kernels that the solvers run, and what they share

What's here is used under the hood (by the solver kernels, the JAX engine, networks, and `ss.Dist`), so that
the common draws are all fast and none is a performance cliff, plus `ss.jit` for users' own code. The same kernels
come in several sets (Numba's here, and Rust's, built from `../rust/`), which give the same numbers:

    ss.jit                                               # Compile a function (see its docstring)
    ss.kernels                                           # The compartmental kernels (kernels.py)
    ss.solvers.backends.draws                            # Exact draws in Numba from counter-based uniforms, scalar and as vectorized rows, from Python or compiled code
    ss.solvers.backends.passes                           # Compiled passes over the agents and a network's edges
    ss.solvers.backends.rust                             # Swaps in the Rust kernels, once built (see ss.options.kernels)
    ss.solvers.backends.keep_memory(), release_memory()  # Keep large freed arrays for reuse (the default), or give them back
    ss.solvers.backends.compiling()                      # Say what is being compiled, and about how long it takes

Used with: agents and compartments; declared and coded modules
"""
import starsim as ss
from . import draws
from . import passes
from .memory import keep as keep_memory, release as release_memory
keep_memory() # Large arrays freed are kept for reuse, so new ones take no page faults: see memory.py
from .jit import jit # ss.solvers.backends.jit is the function; its module is starsim.solvers.backends.jit
from .cache import compiling
