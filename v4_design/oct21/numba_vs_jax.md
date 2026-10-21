# Numba vs JAX (and NumPy) for the kernels

A feasibility study, 2026-10-21. The question: the first use of a compartmental model takes about a minute of Numba compilation; is JAX (with preallocated arrays) an alternative? Cliff added a second question: since the hand-written NumPy Ghana measles model is about as fast as v4, could v4 build models to NumPy instead? Related plan items: S5.11 (cold Numba compile) and S1.6 (JAX consolidation spike) in [simplify_plan.md](../oct20/simplify_plan.md). Scripts and how to run them are in [numba_vs_jax/](numba_vs_jax/); nothing in `starsim/` was changed.

## Summary

- **Why it takes a minute.** `ss.kernels.run()` → `step()` is one function for every solver, and the solver is a number passed at run time. So the first run of anything, even a three-state ODE, compiles the ODE, chain-binomial, tau-leaping, and both exact solvers, with every helper inlined. Measured cold: 51–56 s for every compartmental workload (ODE, Euler, binomial, Gillespie, 50 places). After that the on-disk cache loads it in 0.1–0.4 s. Agent-based runs compile about 6 s of their own kernels.
- **Numba can be cut, but not to "fast".** In a copy of the kernels, compiling only the path a solver uses took 18 s for the ODE and 27 s for the chain binomial, against 42 s for the whole (quiet machine, same conditions). Turning off `inline='always'` made no difference once the other solvers were gone. So splitting by solver saves half to two-thirds, and what remains is the generic `step()` interpreter itself.
- **JAX is a good replacement for the ODE only.** The JAX ODE run v4 already has (`ss.gradients.Run`) compiles in 0.2–0.5 s, or 0.05–0.1 s from JAX's persistent cache. It runs a small model faster than Numba (0.45 ms against 0.7 ms) and 100 runs batched at 0.05 ms each. It is slower on big models: 2–3× slower at 50 places and at Ghana's 260 districts, mostly because the time share between places is a scatter-add, which is slow on XLA's CPU backend (fixable; see below).
- **JAX is not a replacement for the stochastic solvers on a CPU.** A chain-binomial run built the same way was 6–13× slower than Numba on every model, and batching did not help. The cause is the draws: on a quiet machine, `jax.random.binomial` costs 0.34–0.97 µs a draw on a CPU, 5–6× NumPy or Numba (and up to 10× under load). This is fixable in JAX: its binomial, Poisson, gamma, beta, and multinomial samplers are rejection loops, and a loop-free binomial (prototyped here, as JAX already has for Poisson, gamma, and beta) is as fast as NumPy; see below. The exact solver is one event at a time, which is the worst case for JAX (a `while_loop` of tiny steps). Agents would need a new whole-run engine with preallocated arrays, as [gpu_gradabm_feasibility.md](../oct20/gpu_gradabm_feasibility.md) found; that is a separate project, not a swap.
- **NumPy works, with no compile at all, and is close at scale.** The same program run with whole-array NumPy operations (the JAX ODE code with NumPy swapped in, plus a 40-line chain-binomial draw) matches the kernels' ODE to 1e-16. It is 1.7–1.8× slower than Numba at 260 districts, 3–5× at 50 places, and 15–75× for one group. In absolute terms, though, a small model takes 20–90 ms, against a first Numba run of 55 s. The hand-written Ghana model is faster than both because it is specialized: no generic stack of operations, no tables interpolated every step. A NumPy step built per model, rather than interpreted, would close part of that gap (not measured).
- **Recommendation** (for Cliff to decide): (1) split the kernels so that each solver compiles only its own path, which roughly halves or thirds the wait for everyone at little cost. (2) Run the ODE from the array-form code that `ss.gradients.Run` already is: in NumPy by default (no compile), and in JAX for batches and gradients. That removes the ODE, and its half of the compile, from Numba. (3) Keep Numba for the chain binomial, tau-leaping, the exact solvers, and agents, where nothing else is as fast on a CPU. Add an explicit `ss.compile()` (Cliff's idea) that compiles them ahead of time, e.g. in a Docker image, in CI, or after install. (4) Optionally, use NumPy for a stochastic run while Numba compiles in a background process, so that no one waits; this needs the NumPy draw to cover everything `draw()` does. Details and costs are below.

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

[jax_binomial.py](numba_vs_jax/jax_binomial.py) builds the whole run as one `lax.scan`, with the rates from `ss.gradients.Run.rates()` and the draws from `jax.random.binomial` and `poisson`. Everything is preallocated (fixed shapes), and the time share is a dense matrix.

| Model | Compile (s) | JAX (ms) | Numba (ms) | Ratio | Batched, a run (ms) |
|---|---|---|---|---|---|
| sir | 0.6 | 13.9 | 1.5 | 9.3 | 4.3 (×100) |
| sir_age | 1.5 | 18.7 | 1.9 | 9.9 | 14.7 (×100) |
| spatial | 0.7 | 40.0 | 6.7 | 6.0 | 52.1 (×100) |
| ghana | 2.1 | 13,745 | 1,915 | 7.2 | 27,170 (×8) |

(Measured at load ~5; a second run during the other agent's tests gave ratios of 6–13.) The results agree in distribution: final sizes and peaks differ only by seed. [draws.py](numba_vs_jax/draws.py) shows why the run is slow. For 260 binomials a step, on a CPU, the time a draw takes is (updated with [draws_loopfree.py](numba_vs_jax/draws_loopfree.py), interleaved, best of 5, at load ~4; the first version of this table was measured under load and was 2–3× higher for every column):

| p | NumPy (ns) | Numba (ns) | JAX (ns) | JAX, loop-free (ns) |
|---|---|---|---|---|
| 0.0001 | 158 | 129 | 968 | 103 |
| 0.01 | 77 | 60 | 393 | 70 |
| 0.2 | 70 | 53 | 342 | 57 |

The cause is in `jax.random.binomial`'s algorithm, not the bit generator (threefry costs 37 ns for a normal draw). It runs both of its algorithms, inversion and BTRS rejection, on every lane and selects one per lane, and each is a `while_loop` that runs until the slowest of the 260 lanes is done: inversion draws a uniform and takes a log for every success, so about 20 rounds when means are near 10. The last column is a loop-free sampler ([binom_approx.py](numba_vs_jax/binom_approx.py)) that maps one normal draw through the Peizer–Pratt approximation of the binomial cdf, as JAX's `poisson(method='approximate')` does for the Poisson. Its total variation distance from the exact distribution was at the sampling-noise floor of 2 million draws for every (n, p) tried, from (10, 0.3) to (300,000, 0.2). Binomial is listed as still to do in JAX's umbrella issue for loop-free samplers ([jax-ml/jax#39314](https://github.com/jax-ml/jax/issues/39314)), so this is going to JAX as a pull request. Whether it closes the whole-run gap above was not measured: the experiment scripts need updating to the current `ss.kernels.run()` and `ss.geo` first.

The other distributions Starsim draws from, timed the same way with [dists.py](numba_vs_jax/dists.py) (load 13–18, so only the ratios are meaningful): the transform-based ones (uniform, normal, lognormal, exponential, Weibull, Bernoulli, randint, choice) are within 1–4× of NumPy, mostly the per-step cost of splitting the key. The rejection-loop ones are 5–15× slower: binomial, multinomial (a scan of binomials), Poisson, gamma, beta, and the negative binomial (as a gamma–Poisson mixture). With JAX's existing `method='approximate'` for Poisson, gamma, and beta, and the loop-free binomial, these come to 0.6–4×. On a GPU this would be different, but no GPU was usable here (see the oct20 GPU study).

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
5. **Not recommended:** JAX for the stochastic and exact solvers on CPU (6–13× slower, set by the draws, until JAX has a loop-free binomial; then remeasure), and JAX for agents as a swap (it needs a new engine; see the oct20 GPU study).

## Open questions for Cliff

- Is about 18–27 s on first use, once per install, acceptable if `ss.compile()` exists and the docs say so? If not, option 4 is the only one measured here that removes the wait for stochastic runs.
- Should the ODE move to NumPy/JAX (option 2)? That means accepting 50 ms for a single small ODE run without JAX, or making JAX a base dependency. jaxlib is 353 MB installed, against 206 MB for Numba with llvmlite.
- The agent kernels' 6 s (the `*_serial` and threaded variants of `census`, `count_alive`, `infect`, ...) is a separate, smaller issue: the same split applies (compile the threaded variants only when threads are used).
