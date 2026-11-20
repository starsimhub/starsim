"""
Numba's kernels: the compiled functions that the solvers run, the reference copy of the kernels in Rust (`../rust/src/`, each file named like its Numba one)

    ss.solvers.numba.compartmental  # The compartmental kernels (compartmental.rs): a whole run, or a step, of every compartmental solver
    ss.solvers.numba.passes         # Passes over the agents and a network's edges, serial and threaded (passes.rs)
    ss.solvers.numba.agent_events   # The agent engine taken event by event within a step (agent_events.rs)
    ss.solvers.numba.draws          # Exact draws from counter-based uniforms, which the others and ss.Dist use (draws.rs); also ss.draws

The machinery around them (choosing and compiling them, swapping in Rust's, `ss.jit`) is in `ss.solvers.backends`. See README.md.

Used with: agents and compartments; declared and coded modules
"""
from . import draws
from . import compartmental
from . import passes
from . import agent_events
