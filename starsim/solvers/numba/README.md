# Numba's kernels

The kernels are the compiled functions that the solvers run. Numba's, here, are the reference: their Rust copy (`../rust/src/`) gives the same numbers, bit for bit, and plain Python runs this same code uncompiled. Each Rust file is named after the Numba file (or, for the compartmental kernels, the section) it copies. The machinery around them (choosing compiled or plain Python, `ss.compile()`, swapping in Rust's, `ss.jit`) is in `../backends/`.

| Numba (`numba/`) | Rust (`rust/src/`) | Called by |
|---|---|---|
| `compartmental.py`, section "program" (the program's fields, `workspace()`) | `compartmental.rs` | `ss.solvers.Compartmental.init_program()`, which makes the program |
| `compartmental.py`, section "rates" | `compartmental/rates.rs` | The entry points (section "run") |
| `compartmental.py`, section "chain" (chain binomial, tau-leaping, moving between groups) | `compartmental/chain.rs` | The entry points (section "run") |
| `compartmental.py`, section "exact" (Gillespie, by place, settling the counts) | `compartmental/exact.rs` | The entry points (section "run") |
| `compartmental.py`, section "at once" (infinite rates, conveyors) | `compartmental/at_once.rs` | The entry points (section "run") |
| `compartmental.py`, section "step" (`step()`) | `compartmental.rs` (`step_groups()`) | The entry points (section "run") |
| `compartmental.py`, section "register" (movement registers, deliveries) | `compartmental/register.rs` | The entry points (section "run") |
| `compartmental.py`, section "run" (the entry points, `entries`) | `compartmental.rs`, bound in `compartmental/bind.rs` | `ss.solvers.Compartmental` (`../compartmental.py`), `JointStates`, `Particles`, `ss.fitting` |
| `passes.py` | `passes.rs` | `ss.solvers.Agents` (`../agents*.py`), `ss.People`, the networks, `ss.Births`, `ss.dimensions` |
| `agent_moves.py` | `agent_moves.rs` | `ss.solvers.Compiled` (`../agents_compiled.py`), the default engine of `solver='abm'` |
| `agent_events.py` | `agent_events.rs` | `ss.solvers.Events` (`../agents_events.py`), for `ss.solvers.abm(step='events')` |
| `draws.py` | `draws.rs` | The other kernels, and `ss.Dist` |
| Small kernels in `internals/arrays.py`, `networks.py`, `internals/dists.py`, `distributions.py`, `people.py` | `small.rs` | Those modules |

The compartmental solvers (RK4, Euler, Dormand-Prince, binomial, tau-leaping, Gillespie, whole steps) are modes of one step, so the compartmental kernels are split by what they do rather than by solver. They stay in one Numba file, since the plain Python copy is made by running that file again (`ss.solvers.backends.compiling.uncompiled()`), the kernels are inlined into each other, and Numba's cache goes stale by file. The Rust files with no Numba twin are what the copies are built on: `compartmental/views.rs` (arrays), `compartmental/rng.rs` (NumPy's Generators, and the draws on views), `compartmental/ziggurat.rs`, and `py.rs` (what the bindings share).

To change a kernel, change it here first, then its Rust copy to match, in the same change, and run `tests/test_kernel_sets.py` (see `../rust/README.md`).
