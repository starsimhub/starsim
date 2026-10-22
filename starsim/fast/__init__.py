"""
Machinery for speed that no user needs to read

What's here is used under the hood (by the solver kernels, the JAX engine, and `ss.Dist`), so that
the common draws are all fast and none is a performance cliff, plus `ss.jit` for users' own code:

    ss.jit                                    # Compile a function, a module's method, or a module class (see its docstring)
    ss.fast.key, ss.fast.uniform              # Starsim's counter-based uniforms, from Python or compiled code
    ss.fast.binomial, ss.fast.poisson         # Draws from those uniforms, one each, from Python or compiled code (scalars or arrays)
    ss.fast.draws                             # The draw library they wrap: exact draws in Numba, scalar and as vectorized rows
"""
from . import _draws as draws
from . import _passes
from . import _memory
_memory.keep() # Large arrays freed are kept for reuse, so new ones take no page faults: see _memory.py
from ._jit import jit, key, uniform, binomial, poisson
