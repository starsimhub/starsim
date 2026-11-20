# Backends

How the solvers' numbers are computed (`ss.options.backend`), and the machinery around the kernels. None of this is needed to use Starsim. The kernels themselves are in `../numba/` (Numba's, the reference, with a table of each file and its Rust copy) and `../rust/` (Rust's, built from source).

- `rust.py`: swaps in the Rust kernels, once they have been built, and falls back to Numba's for arrays of a type Rust's do not take
- `compiling.py`: which compartmental kernels a run takes (Rust's, Numba's compiled, or the same code as plain Python), `ss.compile()`, Numba's on-disk cache, and saying when something is being compiled
- `workers.py`: starting the workers of `ss.parallel()` and of a calibration, with one thread each for the kernels
- `jit.py`: `ss.jit`, to compile users' own code with Numba or JAX
- `memory.py`: keeping large freed arrays for reuse
- `_rs.abi3.so`: the built Rust library (gitignored), put here by `../rust/build`
