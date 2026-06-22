# Starsim Rust backend — status and roadmap

The goal: let users build and think in Python, and opt into Rust-accelerated
execution where it pays off, with a validation harness guaranteeing the Rust
path matches the Python one. This document records what exists today and what
remains.

## TL;DR

- The hard problem — reproducing numpy's RNG bit-for-bit in Rust — is **solved
  and validated** (PCG64 generation, jump-ahead, the CRN combine, and
  permutation all reproduce numpy exactly).
- Two execution strategies were built and measured:
  1. **Per-module kernels inside the Python loop** (`ssr.SIS`): correct and
     byte-identical, but **~1.1x** whole-sim — numpy is already vectorized C and
     per-timestep round-trips cancel the gains.
  2. **Native whole-sim Rust loop** (`SisRandomNetSim`): byte-identical and
     **7.1x faster at n=2000**, shrinking to ~1.1x at n=100000 (scalar Rust
     ≈ vectorized numpy at scale; the win is eliminating per-timestep Python
     overhead, which dominates at small/moderate populations).
- The native loop is the real prize. It currently covers a restricted SIS +
  RandomNet config; generalizing it and parallelizing it are the main remaining
  work.

## Architecture

A `ssr` module subclasses its `ss` counterpart and overrides only its hot
methods, operating on the existing numpy state buffers via zero-copy views;
everything else inherits the Python implementation (which also gives the
validator a free reference). Mixing Python and Rust modules is cheap because the
loop boundary is per-timestep, not per-agent, and state is shared zero-copy.

The native loop goes further: Rust owns all state, edges, and RNG for the whole
run, with no per-timestep round-trips. Python lifts the post-`init()` state in
and reads results out.

**Key enabler:** rather than reproduce numpy's `SeedSequence` seeding, we lift a
bit generator's 128-bit state and reproduce only the deterministic
step/output/jump algorithms — which is tractable and exact.

## Validation: tiered equivalence

`starsim.rust.compare(ref, test)` reports the weakest tier across all results:

| Tier | Meaning | When |
|------|---------|------|
| `identical` | byte-for-byte | integer / uniform / Bernoulli / CRN / permutation paths |
| `allclose` | within tolerance | `exp`/`log`/reductions (rate→prob, lognormal, means) |
| `discrete` | same rounded trajectory | same agents/states despite last-ULP drift |
| `mismatch` | genuine divergence | a bug |

The boundary is set by the math: anything touching transcendental functions
(lognormal durations, rate→prob conversions) or float reductions can only reach
`allclose`/`discrete`, never `identical`. See `SUPPORTED_SUBSET.md`.

## What's done

### Phase 0 — validation foundation
- `validate.py` → `ss.rust.compare(...)`, tiered, with `require=`.
- `SUPPORTED_SUBSET.md` — portability rules and the tier-boundary rationale.

### Phase 1 — per-module kernels (mixed mode)
- `ssr.SIS` overrides `compute_transmission` with a Rust kernel; byte-identical.
- Finding: porting one sub-function is worthless (boundary overhead cancels it);
  the profiler (`sim.loop.cpu_df`) must guide which whole methods to port.

### Phase 3 — RNG reproduced in Rust
- **PCG64** generation (float64, float32, and the buffered `has_uint32` case).
- **`jumped()`** jump-ahead (lets Rust reproduce Starsim's `jump_dt`/`jump`).
- **`multi_random`** CRN combine (the transmission RNG), bit-identical on 22k+
  edges. Gotcha: `combine_rvs` uses `@njit(fastmath=True)`, so `x/int_max`
  becomes a reciprocal-multiply — Rust must match it or ~0.1% of values drift 1 ULP.
- **`permutation`** (numpy Fisher-Yates with masked-rejection bounded ints).
- Native `MultiRandomRng` integrated into `ssr.SIS` (owns the transmission RNG):
  byte-identical end-to-end, ~1.14x on `sis.step`, ~1.1x whole-sim.

### Phase 4 — native whole-sim loop
- `SisRandomNetSim` (Rust) runs the entire loop for SIS(constant `dur_inf`,
  `waning=0`, `imm_boost=0`) + RandomNet, owning all state/edges/RNG.
- Byte-identical to `ss.SIS` across all results.
- **7.1x @ n=2000, 2.1x @ 20000, 1.1x @ 100000** (see scaling note in TL;DR).

## Performance findings (honest)

- **numpy is already C.** Per-kernel swaps inside the Python loop don't beat it;
  the round-trip + output-array allocation per timestep roughly cancels the gain.
- **The win is eliminating per-timestep Python overhead**, which is a *fixed*
  cost — so the native loop's speedup is largest at small/moderate populations
  and decays toward 1x as O(n) array work comes to dominate.
- **cProfile misleads here** — it inflates Python-heavy functions (the RNG
  wrapper looked like 0.83s under cProfile but is far less in reality). Use
  `cpu_df` wall-time, interleave Python vs Rust runs, and keep an untouched
  function as a sanity control (benchmark noise floor is ~15%).

## Tooling (reusable for any module)

- `rust-portability-linter` agent — read-only GO/NO-GO analysis of a module.
- `starsim-rust-port` skill — the guided port workflow + Rust/wrapper templates.
- `starsim-rust-validate` skill — building and interpreting the harness.

## File map

```
starsim/rust/
  __init__.py            # ssr namespace: compare, SIS, RandomNet, available
  validate.py            # tiered equivalence harness (ss.rust.compare)
  modules.py             # ssr.SIS (+ _NativeTransRng shim), ssr.RandomNet
  SUPPORTED_SUBSET.md    # portability rules + tier rationale
  STATUS.md              # this file
  _crate/                # PyO3/maturin crate -> starsim_rust_kernels
    src/lib.rs           # kernels: compute_transmission, PCG64, jumped,
                         #   multi_random_rvs, permutation, MultiRandomRng,
                         #   SisRandomNetSim (native loop)
    Cargo.toml, pyproject.toml

tests/devtests_rust/     # spikes, validation tests, benchmarks
```

### Building the crate
No virtualenv here, so `maturin develop` fails; use:
```bash
cd starsim/rust/_crate
maturin build --release
pip install --force-reinstall --no-deps target/wheels/*.whl
```
Library `ssr` modules are meant to ship as precompiled abi3 wheels (no user
toolchain). A user porting their own module needs a local Rust toolchain.

## Roadmap (what remains)

### Near term — make the native loop generally useful
1. **Large-n optimization.** Parallelize the transmission and edge-generation
   loops (rayon) and cut per-step allocations, to push the 100k case past 1.1x.
   This is where the scaling story is currently weakest.
2. **Generalize the model coverage.** Add immunity/waning dynamics, then
   lognormal `dur_inf`. Lognormal needs numpy's ziggurat normal — its ~2% tail
   uses libm `exp`/`log`, so lognormal-based runs validate at the `discrete`
   tier (same agents/states), not `identical`. Plan the ziggurat tables + tail.
3. **Clean public API.** Wrap the native loop behind something like
   `sim.run_native()` or an `ssr` engine flag, with the state-lift and
   `compare()` validation built in, so it isn't a hand-assembled driver.

### Medium term — broaden coverage
4. **More modules.** Demographics (births/deaths → dynamic state arrays, slot
   growth), more networks (mixing pools), more diseases (SIR/SEIR). Each needs
   its hot path ported and validated; use the portability linter to gate.
5. **Native results object.** Return a real `ss.Results`-shaped object from the
   native loop so it drops into existing plotting/analysis unchanged.

### Longer term — the full vision
6. **`sim.to_rust()` for library-only sims.** Assemble the native loop
   automatically from a sim built entirely of supported `ssr` modules.
7. **Skill-assisted porting of custom modules.** The linter + port skill +
   validate harness already exist; exercise them end-to-end on a user-written
   module and harden the templates.
8. **Distribution.** CI to build and publish abi3 wheels per platform via
   maturin, so `import starsim.rust` works without a toolchain.

## Known correctness details (so they aren't rediscovered)
- Starsim draws uniforms as **float32** (combined as uint32 in CRN).
- `multi_random` per-timestep index: `ind = dt_jump_size*(ti+1) + call_index`;
  state at draw = `jumped(initial_state, ind)`.
- `asnew` scatters into a **full-length** `.raw`, so UID-indexed state arrays can
  be passed as `.raw` and indexed by UID in Rust.
- Transmission multiply order must match numpy: `(rel_trans*rel_sus)*beta`.
- In the native loop: snapshot rel_trans/rel_sus at the *start* of `infect`
  (collect new cases, then apply); guard `NaN` before `round(ti_infected) as i64`
  (Rust casts NaN to 0).
- For contact-network transmission, `beta` is a bare per-contact probability,
  not a rate (`ss.peryear`).
```
