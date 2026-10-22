# Fast stochastic draws, and JAX for the stochastic compartmental solvers: findings and plan

Written 2026-10-22, from the experiments in [numba_vs_jax.md](numba_vs_jax.md) (measurements, tables, and method) and the prototypes in [numba_vs_jax/](numba_vs_jax/). This file is the summary and the plan; numba_vs_jax.md has the detail. The [constitution](../constitution.md) governs it: where they differ, the constitution wins, and this plan should be changed to match.

Sizes: S is under a day of agent work, M a few days, L more.

## In one page

- **The draws were almost all of the time.** In a whole-run JAX chain binomial, replacing each random draw by its mean made the run 6–14× faster than the Numba kernels; with JAX's own samplers it was 3–5× slower. JAX's samplers can't be made fast on a CPU (they run every lane's loop until the slowest is done), but a native sampler called from JAX can.
- **Exact draws can be much cheaper than NumPy's.** Most draws in a real model are small or already determined (Ghana: 45% have n = 0, 31% have p = 1, under 1% have a mean above 30). Drawing each count from one uniform by inversion, with a test that settles most zeros before any `exp` or `log`, and then making a row of draws (one per group) in SIMD lockstep, cut the draws from about 650 ms to about 220 ms for Ghana, with identical counts to the scalar version and the exact distribution.
- **With those draws, JAX runs the chain binomial 4× faster than the kernels.** Ghana (260 districts, 16 years in days, one core): kernels 1.54 s; v4's program in JAX with the vectorized draws and a leaner rate step 0.38 s; the hand-written model 0.82 s in NumPy, 0.23 s compiled. The draws are now about 60% of the run, from 85–93%.
- **JAX and Numba are equally fast for the same step** (the hand-written loop: 0.378 s in JAX, 0.357 s in Numba, before vectorized draws), and JAX compiles faster (about 1 s against 3–5 s for one model, or 42–56 s for the generic kernels). What made v4's JAX run slower than hand-written code was v4's generic program, mainly dense products with one-hot matrices (12 of 13 matrix products in a Ghana step), not JAX.
- **Recommendation**: (1) one exact draw library in Numba (scalar and vectorized, one uniform a draw, counter-based uniforms), used by the kernels, by JAX through an FFI call, and by modelers' own NumPy or Numba code; (2) the lean fixes to `ss.gradients.Run`, which help the ODE and gradients too; (3) a JAX chain-binomial (and tau-leaping) engine on `Run` for the declared models it supports, chosen when it would be faster, with the kernels for the rest; (4) Numba stays for the exact solvers, agents, and the draws. The decisions are Cliff's; see [Decisions](#decisions-for-cliff).

## Key findings

### 1. Where the time goes

Ghana measles, v4's program in JAX (JAX 0.12, one core, ms a run):

| Draws | Run | Draws' share |
|---|---|---|
| `jax.random` (JAX 0.11.2) | 7,974 | 97% |
| `jax.random` (JAX 0.12) | 4,750 | 95% |
| `method='approximate'` (JAX 0.12, with the binomial pull request) | 4,589 | 95% |
| Native Numba draws, NumPy's algorithms | 873 | 74% |
| One uniform a draw, one FFI call a draw | 666 | 66% |
| One FFI call a step, uniforms made in the call | 549 | 59% |
| Vectorized draws, one call a step | 434 | 48% |
| Vectorized draws, lean rate step | 378 | 59% (of a leaner run) |
| Draws replaced by their means (lean) | 155 | 0% |
| Numba kernels (for reference) | 1,540 | |

The small models (1–50 groups) gain the same way, to 0.26–0.48× the kernels' time (`sir`, `sir_age`, `spatial`), but their best draws are the scalar ones: the vectorized draws were slower at 1, 16, and 50 groups, and faster at 260.

### 2. JAX's own samplers

- JAX 0.12 alone makes `jax.random.binomial` about 2× faster than 0.11.2, but on a CPU rejection samplers in XLA stay at about 2–3× NumPy a draw: a `while_loop` over an array runs until its slowest lane is done, and the cost grows with the array.
- The loop-free `method='approximate'` samplers (Poisson, gamma, beta in JAX; binomial in the pull request on branch `binomial-approximate`, not yet submitted) don't help on a CPU, since they compute every branch for every lane. On the GPU they are 2–5× faster than the exact ones, batched; with them the small models run in 1–3 ms each, but Ghana is still 2.3× slower than the CPU kernels (RTX 3050 laptop card; only 2 Ghana runs fit in 4 GB).
- `jax.pure_callback` to NumPy costs about 120 µs a call: worse than JAX's samplers.

### 3. Native draws from JAX

- An XLA FFI custom call can be a Numba `cfunc` that reads XLA's call frame by byte offset: no C++ build, compiled in 2–5 s, and as fast as NumPy inside a jitted run. The offsets follow XLA's versioned C API structs (API 0.4 here), so a test must guard them. It runs on the CPU only.
- Philox4x32-10 (the Random123 test vectors pass) or a splitmix64 hash of the key and the draw's index give counter-based uniforms, so streams don't depend on how the work is split; Philox costs 5–15 ns a draw more.

### 4. Cheaper exact draws

- **Skip what's determined** (n = 0, p = 0 or 1): the kernels already do; the first JAX prototype didn't (31% of Ghana's binomials have p = 1: the last flow out of a compartment).
- **One uniform per draw, by inversion** (BINV for a binomial, the pmf recurrence for a Poisson, which also replaces NumPy's λ + 1 uniforms a draw below a mean of 10), with fallback to BTPE or PTRS above a mean of 30. Counts are monotone in their uniforms, which suits common random numbers.
- **Zero test**: since (1 − p)^n ≥ 1 − np and e^−λ ≥ 1 − λ, a uniform below 1 − mean is a 0, with no `exp` or `log`; for those who leave a compartment, the chance of none is exp(−n·h), so no `log1p` or `expm1`.
- **One call a step**: all of a step's draws (who leaves each compartment, how they split, the arrivals) in one FFI call, with the uniforms made in the call.
- **Vectorized rows**: every lane of a row walks up its pmf in lockstep (a multiply, an add and a compare a step, with a reciprocal shared by all lanes), from a chance of 0 computed with hand-written float64 `exp`, `expm1` and `log1p` (no SVML needed). 7–9 ns a binomial draw against 11–17 ns scalar, and identical counts for the same uniforms. Numba vectorizes it only with `error_model='numpy'` (its division-by-zero check blocks LLVM) and with no gathering of lanes.
- **All checked against the exact distributions** (binomial means 0.005–20,000, p up to 1; Poisson 0.001–100) within sampling noise, and over 100 Ghana seeds against NumPy's algorithms within 0.8 standard errors.

### 5. v4's generic program

- In a Ghana step, 12 of the 13 products of a 260×260 matrix with a vector are one-hot matrices: summing people by place and spreading the sums back (`Run.place`, 10), and picking each stratification's levels (`cells`, 2). Here they are the identity. As sums and indexing (O(groups)), the place ones save 70 ms of Ghana's run, with identical results.
- Keeping every compartment and flow by group at every step costs about 30 ms of 225 (keep only the results asked for).
- The table of rate constants (`fixed`, `mult`, `consts`, `rel_trans`) is stored at three points a step for every group (146 MB for Ghana) though only seasonal transmission varies, and in time only. It costs only 3% of the time, but storing each value at the level it varies would cut it to under 1 MB.
- The hand-written model does less: 10 draws a district a day against 14 (deaths after the other transitions, not as a competing risk), one seasonal factor, national totals. Its compiled loop (0.23 s) is a ceiling; v4's lean JAX run is within 1.7× of it.

### 6. What didn't pay

- **Clocks for rare flows** (skip draws until the next event, from an Exp(1) threshold): exact, but slower on these models, since the flows they cover have means of 0.3–1.6 and the zero test already makes rare ones cheap. They may pay with short steps, many small places, or agents (not measured).
- **A float32 first pass** (vectorized `exp`/`log` in float32): 4× faster a lane, but no gain, since the scalar code only computes them for the few lanes the zero test doesn't settle.
- **SVML** (`icc_rt`): not on Apple Silicon, and not needed once the math is hand-written.
- **A Numba step generated per model**: no faster than JAX for the same step; JAX's tracing already specializes the generic program per model.

### 7. Open

- Over 20–100 seeds, every JAX draw mode ends Ghana with about 1,800 in I against the kernels' 1,631 (z ≈ 2); S agrees. Since it doesn't depend on the draws, if real it is in the JAX run, e.g. its tables evaluated at the middle of the step. To check before J5.
- GPU: batched runs with fast draws could be much faster (Ghana with draws as means: 54 ms a run, 30× the kernels), but only with GPU draws (the loop-free samplers or the same lockstep inversion in JAX), and only measured on a 4 GB laptop card.

## Recommendations for v4

Against the constitution: performance (primary 2) is the reason; simplicity (primary 1) is unaffected for modelers, who don't choose an engine; flexibility (primary 3) is kept by falling back to the kernels for what the JAX engine doesn't support; correctness (primary 4) is kept, since every draw is exact; maintainability (secondary 2) is the cost, a second stochastic engine, kept in agreement by tests; startup time (secondary 4) improves, since a JAX run compiles in about 1 s.

1. **One draw library, in Numba.** Exact binomial and Poisson (then the others `draw()` and `ss.Dist` use) from one uniform each, scalar and vectorized, with counter-based uniforms. The kernels, the JAX engine (through FFI), and modelers' own code (NumPy or Numba, as in the hand-written Ghana model, which ran 3.6× faster compiled with them: 0.82 s to 0.23 s) all use it. This is also the natural home for CRN: a uniform per (seed, call, slot) and a monotone inverse.
2. **Lean fixes to `ss.gradients.Run`.** One-hot products as sums and indexing; one key a step; keep only the results asked for; tables stored at the level they vary. These help the ODE run and gradients as much as a stochastic engine, and need no new engine.
3. **A JAX engine for the chain binomial and tau-leaping**, built on `Run`: rates from `Run.rates()`, draws from the library through one FFI call a step, `vmap` over seeds and parameter sets. It covers what `ss.kernels.draw()` does for the models it supports, with a `reason()` for what it doesn't (as `ss.gradients.reason()`), and the kernels run the rest.
4. **Numba stays** for the exact solvers (one event at a time, the worst case for JAX), agents (dynamic shapes), and the draws. Dropping Numba would mean C++ draws built for every platform or 2–3× slower JAX samplers, and new exact and agent engines; dropping JAX would lose gradients.
5. **Choose the draw form by size**: scalar rows for small numbers of groups, vectorized for large (the crossover is somewhere between 50 and 260 groups; to measure in J1).

## The plan

- [x] **J1 The draw library (M).** A module (name for Cliff; e.g. `starsim/draws.py`) with exact one-uniform binomial and Poisson, scalar and vectorized (from [ffi/draws_ffi.py](numba_vs_jax/ffi/draws_ffi.py) and [ffi/vdraws.py](numba_vs_jax/ffi/vdraws.py)), the vectorizable `exp`, `expm1` and `log1p`, counter-based uniforms (splitmix64 or Philox, see decision 4), and BTPE/PTRS fallbacks. Tests: the exact distributions (total variation within noise over the regimes above), scalar and vectorized identical for the same uniforms, the Philox test vectors if used, and a check that the vectorized loops compile to vector code (so a Numba change can't silently undo it). Done when the tests pass and the Ghana rows run at the speeds above.
- [x] **J2 The kernels use the library (M).** `ss.kernels.binomial()`/`draw()` for the chain binomial and tau-leaping, with the vectorized rows where there are enough groups (the crossover measured in J1). Streams change, so results change by seed (the same in distribution): update the baselines (decision 2). Done when the switching tests and the perf guard pass and the S0.3 stochastic benchmarks are no slower; measure how much faster.
- [x] **J3 Lean `Run` (S–M).** In `ss.gradients.Run`: `place` and `cells` as segment sums and indexing (the identity case free); one key a step; tables stored at the level each value varies, one point a step for the stochastic solvers; results kept as asked. Done when the ODE and gradient tests pass unchanged (to 1e-12) and the JAX ODE benchmarks are no slower.
- [x] **J4 The FFI bridge (S–M).** The Numba `cfunc` handler for one call a step (from [ffi/step_ffi.py](numba_vs_jax/ffi/step_ffi.py)), registered on first use, compiled with Numba's cache, with a test that guards the call-frame offsets against the installed JAX's C API header. Done when the bridge runs on Linux and macOS (ARM) with the JAX version v4 pins, and the offset test fails cleanly on a mismatch.
- [x] **J5 The JAX stochastic engine (L).** First resolve the I discrepancy (J7). Then the chain binomial and tau-leaping on `Run` with J1–J4: stages and conveyors, reservoirs, credits, maps between groups, registers and deliveries, and diseases tracked jointly (as `CrossRun`), or a `reason()` naming what's missing. Tests: in distribution against the kernels over many seeds (final sizes, peaks, totals by group), and conservation. Done when the S0.3 stochastic compartmental benchmarks pass on it and it is faster than the kernels on each.
- [x] **J6 Choosing the engine (S).** `ss.solvers.choose_solver()` (or the engine) picks the JAX engine for a supported stochastic compartmental model when it would be faster (e.g. more than a few groups, or many runs), and says why; decision 1. Done when the choice is explained by `choose_solver()` and covered by a test.
- [x] **J7 The I discrepancy (S).** Find why the JAX chain binomial ends Ghana with more in I than the kernels (z ≈ 2 over 20–100 seeds), starting with the tables evaluated at the middle of the step. Done when it is explained, and fixed or recorded.
- [x] **J8 Draws for modelers' own code (S).** Public binomial and Poisson (and the uniforms) usable from NumPy and from `@nb.jit` code, with an example of a hand-written step (the Ghana loop) and a way to compile a module's step (decision 3). Done when the Ghana loop runs with them, compiled, at about 0.23 s.

Deferred (to [admin/deferred.md](../admin/deferred.md) if Cliff agrees): GPU runs (batched, with GPU draws; needs a real GPU to measure), clocks for rare flows (for agents or short steps), and gamma, beta and negative binomial in the library (after binomial and Poisson). Won't do: float32 draws, SVML, a Numba step generated per model, and `pure_callback`.

## Decisions for Cliff

1. **Which engine by default** for a supported stochastic compartmental model: the JAX engine (faster for more than a few groups, compiles in about 1 s, but a second engine to keep in agreement), the kernels (now uncompiled by default, per S5.11), or chosen by size? Or should the JAX engine replace the kernels' chain binomial altogether, once it covers everything (one engine, not two)? [CK: i'm hopeful that JAX can replace the kernel chain binomial, if it's faster for big cases and not too much slower for small ones]
2. **New streams**: J2 changes every stochastic result for a given seed (not its distribution). Accept and rebaseline? [CK: sure]
3. **Modelers' own loops**: how much should v4 offer for code like the hand-written Ghana model: only the draws (J8), or also a way to compile a module's step? [CK: would love a compiler: @ss.jit would be the dream! which could be jax, numba, or a combination]
4. **Uniforms**: splitmix64 (cheaper, a good hash) or Philox (the standard counter-based generator, as JAX and NumPy offer; 5–15 ns a draw more)? [CK: let's go with fast, unless it's <5% overall runtime, but i feel 15 ns per draw will add up]
5. **The JAX pull request** (`binomial(method='approximate')`, on the fork's `binomial-approximate` branch, uncommitted): submit it? It helps GPU runs, not CPU ones. [CK: no not yet]

## Definition of done

This plan is done when J1–J4 and J7 are done, J5 and J6 are done or deferred by Cliff's answer to decision 1, J8 is done or deferred by decision 3, and decisions 1–5 are answered. A new idea found on the way goes to deferred.md, not here, unless it blocks an item.

**Estimate**: about 90% done (2026-10-22, overnight, in a separate worktree): J1–J8 are built and tested (see [oct22/jax/README.md](../oct22/jax/README.md)); what is left is Cliff's answers to decisions 1–5 as the build interpreted them, and his review. Before that: about 10%. Update this line with every change to the plan; it should only go up, or say what was added if it goes down.
