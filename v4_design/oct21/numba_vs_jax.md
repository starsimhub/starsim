# Numba vs JAX (and NumPy) for the kernels

A feasibility study, 2026-10-21. The question: the first use of a compartmental model takes about a minute of Numba compilation; is JAX (with preallocated arrays) an alternative? Cliff added a second question: since the hand-written NumPy Ghana measles model is about as fast as v4, could v4 build models to NumPy instead? Related plan items: S5.11 (cold Numba compile) and S1.6 (JAX consolidation spike) in [simplify_plan.md](../oct20/simplify_plan.md). Scripts and how to run them are in [numba_vs_jax/](numba_vs_jax/); nothing in `starsim/` was changed.

## Summary

- **Why it takes a minute.** `ss.kernels.run()` → `step()` is one function for every solver, and the solver is a number passed at run time. So the first run of anything, even a three-state ODE, compiles the ODE, chain-binomial, tau-leaping, and both exact solvers, with every helper inlined. Measured cold: 51–56 s for every compartmental workload (ODE, Euler, binomial, Gillespie, 50 places). After that the on-disk cache loads it in 0.1–0.4 s. Agent-based runs compile about 6 s of their own kernels.
- **Numba can be cut, but not to "fast".** In a copy of the kernels, compiling only the path a solver uses took 18 s for the ODE and 27 s for the chain binomial, against 42 s for the whole (quiet machine, same conditions). Turning off `inline='always'` made no difference once the other solvers were gone. So splitting by solver saves half to two-thirds, and what remains is the generic `step()` interpreter itself.
- **JAX is a good replacement for the ODE only.** The JAX ODE run v4 already has (`ss.gradients.Run`) compiles in 0.2–0.5 s, or 0.05–0.1 s from JAX's persistent cache. It runs a small model faster than Numba (0.45 ms against 0.7 ms) and 100 runs batched at 0.05 ms each. It is slower on big models: 2–3× slower at 50 places and at Ghana's 260 districts, mostly because the time share between places is a scatter-add, which is slow on XLA's CPU backend (fixable; see below).
- **JAX can beat Numba for the stochastic solvers on a CPU, but only with native draws.** A chain-binomial run built the same way, as one `lax.scan`, is 3.3–5.1× slower than the kernels with JAX's own samplers in JAX 0.11.2 (6–13× in the first measurement, under load), and 0.6–3.1× in JAX 0.12 (faster for one group, 3.1× for Ghana; the loop-free samplers don't improve on that on a CPU), and the draws are almost all of that: with each draw replaced by its mean, the same JAX run is 6–14× *faster* than the kernels. With `jax.random.binomial` at 5.5–6.7× NumPy a draw (2.7× in JAX 0.12, 1.2–2.5× with a loop-free binomial, now a pull request to JAX), JAX can't run NumPy's per-draw algorithms itself, but a native kernel called from JAX can: Numba-compiled binomial and Poisson draws registered with `jax.ffi` (no C++ build) make the whole JAX run 1.2–2.4× faster than the kernels in either version (Ghana: 0.88 s against 1.55 s), with the same results in distribution. Part of that lead may be that the JAX run is traced per model while the kernels interpret a general program and keep more (the register, tallies, results by group); a Numba kernel generated per model was not tried. On the GPU (an RTX 3050 laptop card) single runs are slow, about 10 ms of step overhead even without draws and 15–80× the kernels with JAX's draws, but batches are where it pays: with the draws as means, 100 runs of the small models take 0.1 ms each and 8 Ghana runs 56 ms each, 30× faster than the kernels. The exact solver is one event at a time, which is the worst case for JAX (a `while_loop` of tiny steps). Agents would need a new whole-run engine with preallocated arrays, as [gpu_gradabm_feasibility.md](../oct20/gpu_gradabm_feasibility.md) found; that is a separate project, not a swap.
- **NumPy works, with no compile at all, and is close at scale.** The same program run with whole-array NumPy operations (the JAX ODE code with NumPy swapped in, plus a 40-line chain-binomial draw) matches the kernels' ODE to 1e-16. It is 1.7–1.8× slower than Numba at 260 districts, 3–5× at 50 places, and 15–75× for one group. In absolute terms, though, a small model takes 20–90 ms, against a first Numba run of 55 s. The hand-written Ghana model is faster than both because it is specialized: no generic stack of operations, no tables interpolated every step. A NumPy step built per model, rather than interpreted, would close part of that gap (not measured).
- **Recommendation** (for Cliff to decide): (1) split the kernels so that each solver compiles only its own path, which roughly halves or thirds the wait for everyone at little cost. (2) Run the ODE from the array-form code that `ss.gradients.Run` already is: in NumPy by default (no compile), and in JAX for batches and gradients. That removes the ODE, and its half of the compile, from Numba. (3) Keep Numba for the chain binomial, tau-leaping, the exact solvers, and agents for now; JAX with native draws (option 5) is now measured as faster for the chain binomial, so moving the stochastic solvers to it is a real alternative, for Cliff to weigh. Add an explicit `ss.compile()` (Cliff's idea) that compiles them ahead of time, e.g. in a Docker image, in CI, or after install. (4) Optionally, use NumPy for a stochastic run while Numba compiles in a background process, so that no one waits; this needs the NumPy draw to cover everything `draw()` does. Details and costs are below.

## Measurements

All runs use one thread (`OMP_NUM_THREADS=1`, XLA single-threaded) on the shared 20-core machine. Timings were taken at load 1–5 unless noted. Another agent's test suite was running at times, and timings taken during it (load 11–16) are left out. The models ([models.py](numba_vs_jax/models.py)) are a declared SIR in one group (`sir`), with 16 age groups and a contact matrix (`sir_age`), over 50 places with commuting (`spatial`), and Ghana measles (`ghana`: SEIR with births, deaths, importation, and seasonality, in 260 districts, for 16 years in days; from LFD's `vignettes/ghana_measles`). Each runs for a year in days, except Ghana.

### Where the Numba compile goes

[numba_cold.py](numba_vs_jax/numba_cold.py) runs each workload in a fresh process with an empty, private Numba cache (so the shared cache is untouched). It records the compile time of each function through Numba's event API. Times include the functions each one calls.

| Workload | First run (s) | Warm (s) | Largest compiles (s, inclusive) |
|---|---|---|---|
| SIR, ODE | 56.3 | 0.015 | `run` 56.3, `step` 43.0, `gillespie_places` 15.1, `sparse_events` 7.9, `draw` 3.8, `place_events` 3.7 |
| SIR, Euler | 54.1 | 0.018 | the same |
| SIR, binomial | 54.0 | 0.018 | the same |
| SIR, Gillespie | 51.6 | 0.017 | the same |
| 50 places, binomial | 56.0 | 0.029 | the same |
| Declared SIR as agents | 6.2 | 0.19 | `census_serial` 2.0, `count_alive_serial` 2.0, `infect_serial` 0.7 |
| Coded SIR as agents | 6.8 | 0.93 | `count_alive_serial` 1.9, `census_serial` 1.9, `due_alive_serial` 0.8 |

`import starsim` took 1.2–1.3 s. Every compartmental workload compiles the same `run()`, so the cost doesn't depend on the model or the solver.

[numba_split.py](numba_vs_jax/numba_split.py) writes a copy of `kernels.py` to a scratch folder, removes from `step()` the calls into the solvers that a run doesn't use, and times compiling `run()` with caching off. All four variants ran back to back at load ~1:

| Variant | Compile + first run (s) |
|---|---|
| Everything (as now) | 41.9 |
| No exact solver (so binomial and tau) | 27.2 |
| ODE only | 17.9 |
| ODE only, no helper inlined | 17.7 |

(These are lower than the 54–56 s above because the copy doesn't write a cache; both were measured, but the relative sizes are what matter.) So a split by solver would cut a first ODE run to about 18 s and a first binomial run to about 27 s. That is better, but still not acceptable by itself. The ODE path is mostly `step()`'s generic interpreter: the stack of operations, the mixing terms, the substeps, and RK4. It is also passed a 53-field `Program` named tuple, which every function that takes it must type. Whether a smaller `Program` (or passing only the fields used) would shrink the compile further was not tested.

### JAX: the ODE

[jax_ode.py](numba_vs_jax/jax_ode.py) times `ss.gradients.Run` (unchanged) in fresh processes: first with no persistent cache, then with one being filled, then with it read back.

| Model | Trace (s) | XLA compile (s), no cache / cached | JAX run (ms) | Numba run (ms) | Batched, a run (ms) |
|---|---|---|---|---|---|
| sir | 0.04–0.06 | 0.18 / 0.17 | 0.45 | 0.7 | 0.05 (×100) |
| sir_age | 0.05 | 0.21 / 0.017 | 3.6 | 1.4 | 0.6 (×100) |
| spatial | 0.06–0.08 | 0.43 / 0.02 | 32–50 | 18 | 36 (×100) |
| ghana | 0.08–0.10 | 0.40 / 0.03 | 7,300–12,800 | 3,500 | 6,500 (×8) |

`import jax` took 0.2–0.4 s. With the persistent cache, the first JAX run of a model in a session takes about 0.1 s plus the import. JAX compiles once per model structure, not once for all models. So a new model pays 0.2–0.5 s, which is fine.

Two things in the current integration need fixing before this could be the default. First, `sim.jax = True` compiled again for every new sim, even an identical one: 0.17–0.55 s a sim for the small models, and 14–25 s for Ghana against 7–13 s for the bare run. Each engine keeps its own `Run`, and a new function object means a new trace. A module-level cache keyed by `Run.key()` would fix it. Second, `Run.share()` adds the time share in with `.at[].add()`, one entry per pair of places (67,600 for Ghana), which XLA runs slowly on a CPU. The JAX chain-binomial prototype below uses a dense matrix product for it instead. For a sparse share, a segment sum would do the same. That is likely most of the 2–3× gap at 50 and 260 places, but it was not measured separately for the ODE.

Update (2026-10-21, S1.6): with a dense time share taken as a matrix, made once a run, the JAX run takes 6.5 ms for `spatial` (against 17.5 ms for a whole Numba `sim.run()`) and 1.7 s for Ghana (against 3.7 s); `sir` 0.4 against 1.4 ms, `sir_age` 3.0 against 2.2 ms.

### JAX: the chain binomial

[jax_binomial.py](numba_vs_jax/jax_binomial.py) builds the whole run as one `lax.scan`, with the rates from `ss.gradients.Run.rates()` and the draws from `jax.random.binomial` and `poisson`. Everything is preallocated (fixed shapes), and the time share is a dense matrix. `DRAWS=native` takes the draws from the Numba FFI kernels below instead (binomial and Poisson, Philox streams), and `DRAWS=mean` puts each draw's mean in its place, to time everything else. Remeasured on 2026-10-21 (JAX 0.11.2, one core, load ~5, ms a run; batched is per run, ×100, or ×8 for Ghana):

| Model | Kernels | JAX's draws | Native draws | Native, batched | Means | Means, batched |
|---|---|---|---|---|---|---|
| sir | 1.5–1.8 | 5.0 | 1.4 | 0.4 | 0.1 | 0.01 |
| sir_age | 2.1–2.6 | 9.8 | 2.0 | 1.2 | 0.4 | 0.1 |
| spatial | 7.4–9.3 | 27.7 | 4.1 | 3.8 | 0.9 | 0.2 |
| ghana | 1,610–1,960 | 8,240 | 1,097 | 1,134 | 277 | 76 |

A repeat on another core gave the same ratios to the kernels (native draws 0.45–0.91).

The same with JAX 0.12 (the `jax` env: a build of `main` with the binomial pull request; run with Starsim, Numba, and Sciris taken from the `v4` env), interleaved with JAX 0.11.2 on the same core, load ~1.4, ms a run (ratio to the kernels in brackets; the kernels took 1.3, 1.9, 7.2, and 1,550 ms). `approx` is `method='approximate'` for both the binomial and the Poisson:

| Model | 0.11.2, JAX's draws | 0.12, JAX's draws | 0.12, approx | 0.12, native | 0.11.2, native | 0.12, means |
|---|---|---|---|---|---|---|
| sir | 4.6 (3.3) | 0.8 (0.57) | 0.9 (0.69) | 0.5 (0.42) | 1.0 (0.79) | 0.1 (0.07) |
| sir_age | 8.9 (4.6) | 3.6 (1.8) | 5.8 (3.1) | 1.6 (0.81) | 1.5 (0.81) | 0.3 (0.14) |
| spatial | 28.1 (3.7) | 13.1 (1.8) | 19.9 (2.8) | 3.2 (0.44) | 3.2 (0.44) | 0.6 (0.08) |
| ghana | 7,974 (4.9) | 4,750 (3.1) | 4,589 (2.9) | 879 (0.57) | 889 (0.58) | 222 (0.14) |

JAX 0.12 alone takes JAX's own samplers from 3.3–4.9× the kernels to 0.6–3.1×, and beats them for one group. The loop-free samplers don't help further on a CPU: they are slower than 0.12's exact ones except for Ghana (3% faster), since they compute both branches for every lane. With native draws, 0.11.2 and 0.12 are the same (the rest of the run doesn't depend on the version), at 0.42–0.81× the kernels. Batched (×100, or ×8 for Ghana), 0.12 with native draws takes 0.3, 0.9, 2.9, and 901 ms a run. Compiles take 0.4–1.5 s with native draws (plus about 3 s, once, for the Numba kernels themselves), against 1.4–4.5 s with JAX's samplers. Batching doesn't help the native draws, since `vmap` calls them once per run. Over five seeds, Ghana's final susceptibles are 2.306M (sd 0.022M) from the kernels and 2.321M (sd 0.049M) from JAX with native draws.

On the GPU (RTX 3050 laptop, 4 GB, JAX 0.11.2 with CUDA; native draws are CPU-only, so not included):

| Model | Kernels (CPU) | JAX's draws | JAX's draws, batched | Means | Means, batched |
|---|---|---|---|---|---|
| sir | 1.5 | 128 | 2.6 | 9.7 | 0.1 |
| sir_age | 2.3 | 182 | 3.3 | 9.8 | 0.1 |
| spatial | 8.9 | 242 | 7.4 | 10.3 | 0.1 |
| ghana | 1,900 | 28,800 | 13,300 (×2; ×8 ran out of memory) | 476 | 56 |

A single run on the GPU pays about 25 µs a step in launches, so it loses to the CPU for anything small. The GPU's case is batches, and there the draws are again what matters: JAX's exact samplers are rejection loops, worst on a GPU, which is what the loop-free samplers (and the binomial pull request) are for. They were not measured on the GPU: the JAX build with the pull request is CPU-only.

The first measurement (2026-10-21, before the remeasure, JAX's draws, load ~5) was:

| Model | Compile (s) | JAX (ms) | Numba (ms) | Ratio | Batched, a run (ms) |
|---|---|---|---|---|---|
| sir | 0.6 | 13.9 | 1.5 | 9.3 | 4.3 (×100) |
| sir_age | 1.5 | 18.7 | 1.9 | 9.9 | 14.7 (×100) |
| spatial | 0.7 | 40.0 | 6.7 | 6.0 | 52.1 (×100) |
| ghana | 2.1 | 13,745 | 1,915 | 7.2 | 27,170 (×8) |

(A second run during the other agent's tests gave ratios of 6–13.) The results agree in distribution: final sizes and peaks differ only by seed. The draws are nearly all of the gap (see the means above). [dists.py](numba_vs_jax/dists.py) times a draw from each distribution Starsim uses (260 a step, parameters changing every step, interleaved, best of 5, load ~2), in ns, with the ratio to NumPy:

| Distribution | NumPy | JAX 0.11.2 | JAX 0.12 | JAX 0.12, loop-free |
|---|---|---|---|---|
| binomial, p ~ 1e-4 | 110 | 734 (6.7×) | 306 (2.8×) | 125 (1.2×) |
| binomial, p ~ 0.01 | 53 | 296 (5.6×) | 141 (2.7×) | 126 (2.4×) |
| binomial, p ~ 0.2 | 51 | 281 (5.5×) | 132 (2.7×) | 125 (2.5×) |
| multinomial, n ~ 1,000, 4 outcomes | 219 | 1,574 (7.2×) | 972 (4.4×) | 757 (3.4×) |
| Poisson, λ ~ 5 / 50 | 50 / 45 | 371 / 179 | 145 / 99 | 109 / 108 |
| gamma, a ~ 2 | 42 | 313 (7.4×) | 298 (7.0×) | 87 (2.0×) |
| beta, a ~ 2, b ~ 5 | 56 | 639 (11.4×) | 576 (11.2×) | 183 (3.5×) |
| negative binomial (gamma–Poisson) | 109 | 795 (7.3×) | 459 (4.0×) | 204 (1.8×) |

The transform-based draws (uniform, normal, lognormal, exponential, Weibull, Bernoulli) are within 2× of NumPy in 0.11.2 and at or below it in 0.12, so the slow ones are those built on rejection loops: binomial, multinomial (a scan of binomials), Poisson, gamma, beta, and the negative binomial. JAX's exact samplers run every lane's loop until the slowest of the 260 lanes is done, and the binomial runs both of its algorithms (inversion and BTRS rejection) on every lane. JAX already has loop-free `method='approximate'` versions of Poisson, gamma, and beta (since July 2026). The binomial was the gap, listed in JAX's umbrella issue ([jax-ml/jax#39314](https://github.com/jax-ml/jax/issues/39314)): the branch `binomial-approximate` (for a pull request to jax-ml/jax) adds it, and `multinomial` uses it. Its total variation distance from the exact distribution is below 1e-4 (computed exactly, not sampled). It helps most for small means (2.4× faster than exact), and only 5–10% for large ones on a CPU; on a GPU, where a rejection loop's lanes wait for each other, the gain should be larger (not measured: no GPU was usable here, see the oct20 GPU study).

An earlier version of this table (from [draws.py](numba_vs_jax/draws.py)) was measured under load and with fixed parameters, which XLA folds into the sampler; both made it misleading. The whole run with JAX 0.12 and the loop-free binomial was not measured (that JAX build is in an env without Starsim); the native draws above settle the question on a CPU anyway.

### JAX: why its draws are slower than NumPy's, and a native kernel

NumPy and Numba draw a binomial with a scalar loop per draw that stops as soon as that draw is done: inversion (one uniform, walking up the pmf) for n·p ≤ 30, and BTPE rejection otherwise, which usually accepts on the first try. XLA has only whole-array operations, so the same algorithm in JAX is a `while_loop` over the array that does every lane's work until the slowest lane accepts. JAX's exact sampler is already this kind of algorithm (inversion and BTRS), and it gets slower as the array grows, since the slowest lane gets slower ([scaling.py](numba_vs_jax/scaling.py), JAX 0.12, ns a draw):

| Draws a step | 26 | 260 | 2,600 | 26,000 | 260,000 |
|---|---|---|---|---|---|
| NumPy | 180 | 52 | 37 | 38 | 33 |
| JAX exact | 133 | 138 | 167 | 225 | 276 |
| JAX loop-free | 134 | 127 | 117 | 52 | 32 |

The fixed cost of a step is small (a uniform draw costs 4–8 ns at every size). The loop-free sampler's fall at large sizes is XLA using about 8 threads despite the single-thread flags; on one core it costs 120–135 ns at every size, since it computes both branches, three Peizer–Pratt evaluations, and several logs for every lane. So in plain JAX on a CPU, rejection-based draws stay at about 2–3× NumPy. Calling NumPy from the jitted run with `jax.pure_callback` is worse ([callback.py](numba_vs_jax/callback.py)): each call costs about 120 µs, so 526 ns a draw at 260 a step and 99 ns at 2,600.

A native kernel registered as an XLA custom call (`jax.ffi`) does get NumPy's speed. [ffi/binomial_ffi.py](numba_vs_jax/ffi/binomial_ffi.py) prototypes it with NumPy's algorithm (BINV and BTPE, ported from `numpy/random/src/distributions/distributions.c`), one draw at a time, in C++ ([ffi/binomial_ffi.cc](numba_vs_jax/ffi/binomial_ffi.cc), for reference) and in Numba: a `cfunc` that reads XLA's call frame by byte offset and answers its API-version query, so there is no C++ build and no new dependency. Each draw has its own random stream, from the key and the draw's index: splitmix64, or Philox4x32-10 (counter = draw index and block number, key = the JAX key; it matches the Random123 known answers). C++ and Numba give identical draws, including for NaN, inf, negative n, and p outside [0, 1]; the total variation from the exact pmf over 10^6 draws matches NumPy's own sampling noise from n = 10 to 10^8; and it works under `jit`, `lax.scan`, and `vmap` over keys (one key at a time). Inside a jitted scan, n and p changing every step, one core, ns a draw (load 15–20 from other users, so the absolute values are noisy; the order is consistent across runs):

| Draws a step, p | NumPy | Numba | JAX 0.11.2 exact | C++ kernel | Numba kernel, splitmix64 | Numba kernel, Philox |
|---|---|---|---|---|---|---|
| 260, p ~ 1e-4 | 117 | 99 | 621 | 98 | 98 | 102 |
| 260, p ~ 0.01 | 59 | 45 | 251 | 40 | 44 | 52 |
| 260, p ~ 0.2 | 55 | 41 | 237 | 37 | 40 | 48 |
| 2,600, p ~ 0.01 | 57 | 58 | 290 | 50 | 55 | 68 |
| 26, p ~ 0.01 | 239 | 49 | 446 | 72 | 74 | 82 |

So from 260 draws a step the kernels match NumPy and Numba and are 5–6× faster than `jax.random.binomial`; at 26 a step, the call costs about 20–30 ns a draw more than Numba. Philox costs 5–15 ns a draw more than splitmix64. The Numba kernel compiles in about 2 s (not cached here; it could be, like the other kernels). What it doesn't do: GPUs (where JAX's loop-free samplers are the right tool), a batch of keys in one call (`vmap` runs one key at a time), or gradients (a binomial draw has none anyway). The byte offsets follow the FFI C API's versioned structs (API 0.4 here), so a test should guard them. The Poisson now follows the same pattern (NumPy's multiplication method below a mean of 10 and PTRS above, about 40 lines; it matches the exact distribution within sampling noise from a mean of 0.3 to 10^6). Gamma, beta, and the negative binomial would too, at about 50–150 lines apiece.

### NumPy

[numpy_run.py](numba_vs_jax/numpy_run.py) runs the same `Program` with NumPy. It reuses `ss.gradients.Run`'s `rates()` and `evaluate()` unchanged, by handing them NumPy in place of `jax.numpy`. It rewrites `share()` and `derivs()` (which use JAX's `.at[]`) and adds a chain-binomial draw vectorized over groups. That draw follows `ss.kernels.draw()` for what these models use: no stages, reservoirs, credits, deliveries, conveyors, or maps between groups. The steps are a Python loop. Measured at load ~2:

| Model | Solver | Numba (ms) | NumPy (ms) | Ratio | Check |
|---|---|---|---|---|---|
| sir | ODE | 0.7 | 52 | 75 | totals identical |
| sir | binomial | 1.4 | 22 | 15 | final S 50,180 vs 50,097 |
| sir_age | ODE | 1.4 | 73 | 51 | 2e-16 |
| sir_age | binomial | 1.8 | 26 | 15 | final S 2,006 vs 1,916 |
| spatial | ODE | 18 | 86 | 4.9 | 3e-16 |
| spatial | binomial | 9.3 | 29 | 3.1 | peak I 802k vs 785k |
| ghana | ODE | 3,486 | 6,218 | 1.8 | 2e-16 |
| ghana | binomial | 1,533 | 2,606 | 1.7 | peak I 16,355 vs 16,636 |

The sim's own `init()` is 12–20 ms (70 ms for Ghana), so for a small model the user would see about 15 ms with Numba and 35–90 ms with NumPy. The one-group ODE is the worst case: RK4 with automatic substeps is about 4 stages × 2 substeps × 30 NumPy calls a day.

On "compiling to NumPy": the hand-written Ghana model (0.84 s, from [deferred.md](../admin/deferred.md) on 2026-10-16) beats both this generic NumPy run (2.6 s) and the kernels (1.5 s here). It does about 15 whole-array calls a day, with the rates written out. The generic run instead goes through the stack of operations, interpolates four tables, and records every flow by group. A step built per model, as Python source or closures written from the graph, would get close to the hand-written loop for large models. It would not help the one-group case, where the per-call cost of NumPy (about 1 µs) is the floor. That was estimated, not measured.

## What each option would take

1. **Split the kernels by solver (S5.11).** Give `run()`, `run_many()`, and `advance()` one entry per solver family (ODE, stochastic, exact), each calling only its own path, so `solver` is fixed at compile time. It is a mechanical change to `kernels.py` and the callers in `solvers/`, with no change to results. It cuts a first run from about 42–56 s to 18 s (ODE) or 27 s (binomial). Check that the warm run doesn't get slower: the inlining was added for speed. Without the other solvers' code it made no difference to the compile, so try removing it and measure.
2. **The ODE from array code, without Numba (S1.6).** Make `ss.gradients.Run` work on either NumPy or JAX (the `jax()` → `xp` change, plus `share()` and `derivs()` without `.at[]`: dense or sparse products, which also speed up the JAX run), and make it the ODE engine. Then the Numba ODE path in `step()` can go: that is the RK4/Euler stages, the substep choice, and `release_rates()`, a few hundred lines. JAX is used when installed for batches (`ss.parallel()` already switches at 300 runs), gradients, and NUTS; NumPy otherwise. The cost is that a single small ODE run goes from about 1 ms to 50 ms, and with JAX to 0.5 ms plus a 0.2–0.5 s compile per model and session (0.1 s from the persistent cache). Before deciding, measure what happens to calibrations that run one ODE at a time (Optuna), which would feel the 50 ms.
3. **`ss.compile()`.** An explicit call that compiles every kernel into the cache, e.g. in a Docker build, a CI step, or a notebook's first cell. It is about 20 lines. If the kernels are split, it could compile the solver families in parallel processes, since Numba's cache is one file per function, so the wall time is that of the largest family. Also check that the cache works on a read-only install (S5.11 already lists it): Numba writes the cache beside the package, and if it can't, every process compiles again. Setting `NUMBA_CACHE_DIR`, or a cache folder Starsim chooses, would handle that.
4. **NumPy while Numba compiles (optional).** On a cold cache, run a stochastic model in NumPy and start a background process that compiles the kernels, so the next run uses them. No one waits a minute, and the first run is 1.7–15× slower, but tens of milliseconds for small models. The cost is a NumPy chain-binomial draw that covers everything `draw()` does (stages, reservoirs, credits, conveyors, maps between groups, registers, deliveries): perhaps 200–300 lines, kept in agreement with the kernels by cross-backend tests. Its results would differ from Numba's for the same seed, unless the draws were taken from the same generator in the same order. That would be possible, since both use NumPy's `Generator` algorithms, but it would constrain both.
5. **Native draws for JAX (if JAX is to run the stochastic solvers).** Numba FFI kernels, as in the prototype, for each distribution `draw()` uses, with Philox streams from the JAX key, a batch of keys in one call, cached compiles, and a test of the struct offsets: perhaps 500–800 lines in all. With the binomial and Poisson alone, the whole-run chain binomial is already 1.1–2.2× faster than the kernels on these models ([jax_binomial.py](numba_vs_jax/jax_binomial.py), `DRAWS=native`). Beyond the draws, a JAX stochastic engine would need everything `draw()` does (stages, reservoirs, credits, conveyors, registers, deliveries), which this prototype skips, and a decision on the GPU, where the native draws don't run.
6. **Not recommended:** JAX for the stochastic and exact solvers on CPU with JAX's own samplers (3–13× slower with JAX 0.11.2, nearly all of it the draws, which stay at 2–3× NumPy even in JAX 0.12 with loop-free samplers), and JAX for agents as a swap (it needs a new engine; see the oct20 GPU study).

## Open questions for Cliff

- Is about 18–27 s on first use, once per install, acceptable if `ss.compile()` exists and the docs say so? If not, option 4 is the only one measured here that removes the wait for stochastic runs.
- Should the ODE move to NumPy/JAX (option 2)? That means accepting 50 ms for a single small ODE run without JAX, or making JAX a base dependency. jaxlib is 353 MB installed, against 206 MB for Numba with llvmlite.
- Is a JAX engine for the stochastic solvers worth pursuing (option 5)? With native draws it is 1.1–2.2× faster than the kernels on a CPU and compiles in seconds, but it is a second engine to keep in agreement with the kernels, unless it replaces them for the chain binomial; and how much of its lead is per-model tracing (which a generated Numba kernel could also get) is unmeasured.
- The agent kernels' 6 s (the `*_serial` and threaded variants of `census`, `count_alive`, `infect`, ...) is a separate, smaller issue: the same split applies (compile the threaded variants only when threads are used).
