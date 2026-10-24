"""
Machinery for speed that no user needs to read

What's here is used under the hood (by the solver kernels, the JAX engine, networks, and `ss.Dist`), so that
the common draws are all fast and none is a performance cliff, plus `ss.jit` for users' own code:

    ss.jit                                              # Compile a function, a module's method, or a module class (see its docstring)
    ss.perflib.key, uniform                     # Starsim's counter-based uniforms, from Python or compiled code
    ss.perflib.binomial, poisson                # Draws from those uniforms, one each, from Python or compiled code (scalars or arrays)
    ss.perflib.draws                            # The draw library they wrap: exact draws in Numba, scalar and as vectorized rows
    ss.perflib.keep_memory(), release_memory()  # Keep large freed arrays for reuse (the default; see ss.options.keep_memory), or give them back
    ss.perflib.compiling()                      # Say what is being compiled, and about how long it takes
"""
import starsim as ss
from . import _draws as draws
from . import _passes
from ._memory import keep as keep_memory, release as release_memory
if ss.options.keep_memory:
    keep_memory() # Large arrays freed are kept for reuse, so new ones take no page faults: see _memory.py
from ._jit import jit, key, uniform, binomial, poisson
from ._cache import compiling
