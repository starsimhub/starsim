# Starsim's kernels in Rust

This crate holds a second copy of Starsim's compiled kernels, in Rust, which give the same numbers as the Numba ones, bit for bit. They are compiled ahead of time, so a first run takes no compiling (with an empty Numba cache, a first compartmental run takes 0.01 s rather than about two minutes), and they run at least as fast as Numba's in nearly every case. With `ss.options.kernels = 'auto'` (the default), Starsim uses them whenever they have been built; see the table of kernels in the solvers' architecture page (`docs/architecture/solvers.qmd`).

## Building

The wheels on PyPI include the compiled kernels: one abi3 wheel per platform, for Python 3.11 and later (settings in `pyproject.toml`, built by `.github/workflows/pypi_release.yml` through setuptools-rust and cibuildwheel), and the sdist includes this crate. Without a wheel for the platform, `pip` builds from the sdist, with the kernels if it finds a Rust toolchain, and otherwise without them, and Numba's kernels run.

From a clone, install Rust (https://rustup.rs), then:

```bash
cd starsim/solvers/rust && ./build
```

This takes about 90 s the first time (about 15 s after that) and puts the library at `starsim/solvers/backends/_rs.abi3.so` (gitignored), where `ss.solvers.backends.rust` loads it. For CPUs with AVX2 (nearly all x86 CPUs since 2013): `RUSTFLAGS="-C target-cpu=x86-64-v3" ./build`. The default build checks for AVX2 at run time where it matters (`avx2!` in `py.rs`), so it is as fast for the vectorized draws. The build compiles in a hash of the source (`build.rs`); if the source beside the library has changed since, it isn't used: `ss.options.kernels='rust'` is an error, and `'auto'` warns and uses Numba's.

## Keeping the kernels in sync

The Numba kernels are the reference, since they are the easier to read and the plain Python copy is the same code. To change a kernel:

1. Change the Numba code (`starsim/solvers/backends/kernels.py`, `passes.py`, `draws.py`, or a small kernel in `arrays.py`, `networks.py`, `distributions.py`, `people.py` or `utils.py`).
2. Make the same change in its Rust copy here (each Rust function is named after the Numba one, and its docstring names it), and where it applies in JAX (`ss.gradients`, for the ODE).
3. Rebuild (`./build`), and run `cd tests && pytest test_kernel_sets.py`, which compares Numba's, Rust's, plain Python's and JAX's kernels on the main paths, and every kernel of `ss.solvers.backends.rust.names` call by call.
4. After a change to the compartmental kernels, also run the tests that use them with every call compared: `cd tests && python -m pytest -p devtests.rust_twin test_kernels.py test_dimensions.py test_deliveries.py -n 2`.

Bit for bit means following Numba's arithmetic where Rust's differs: Python's `max(a, b)` keeps a NaN `a` (`draws::max`, not `f64::max`), a float out of range becomes -2^63 as an integer (`draws::to_i64`), and the draws use Starsim's own `exp()` and `log()`, not libm's. `ss.solvers.backends.rust.fell` counts any call a Rust kernel refused (an array of a type it doesn't take) and passed to Numba; the tests check that it stays empty.

## Layout

Each file holds its kernels and, at the bottom, their bindings, added to the module by its `register()`:

- `draws.rs`: the exact draws (`starsim/solvers/backends/draws.py`), which the other kernels use, and `draws.many` for `ss.Dist`
- `passes.rs`: the passes over agents and network edges (`starsim/solvers/backends/passes.py`), serial and threaded, with a thread pool of its own
- `small.rs`: the small kernels of `arrays.py`, `networks.py`, `distributions.py`, `people.py` and `utils.py`
- `kernels.rs` and `kernels/`: the compartmental kernels (`starsim/solvers/backends/kernels.py`)
- `py.rs`: what the bindings share (NumPy arrays as slices, a body run for each element type, `avx2!`)

`ss.jit` (users' own code) has no Rust copy: it is compiled by Numba (or traced by JAX).
