# Backends

The kernels that the solvers run, and what they share. None of this is needed to use Starsim, but it is where most of the time of a run goes. The same kernels come in several sets, which give the same numbers: Numba's (here, the reference), Rust's (built from `../rust/`), plain Python's, and, where it applies, JAX's. `ss.options.kernels` chooses between them.

- `kernels.py`: the compartmental kernels (`ss.kernels`), and `ss.compile()`, which compiles them ahead of time
- `ode.py`: the ODE's steps in NumPy, vectorized over groups
- `passes.py`: compiled passes over the agents and a network's edges, for the agent solver
- `draws.py`: exact random draws from counter-based uniforms, which the other kernels and `ss.Dist` use
- `chain.py`: the chain binomial and tau-leaping in JAX (imported when first used)
- `rust.py`: swaps in the Rust kernels, once they have been built
- `jit.py`: `ss.jit`, to compile users' own code with Numba or JAX
- `cache.py`: Numba's on-disk cache, and saying when something is being compiled
- `memory.py`: keeping large freed arrays for reuse (see `ss.options.keep_memory`)

To change a kernel, change the Numba code here first, then its Rust copy to match, and run `tests/test_kernel_sets.py` (see `../rust/README.md`).
