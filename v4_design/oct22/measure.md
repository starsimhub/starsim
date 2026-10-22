# F1: performance after the October 22 work

Measured 2026-10-22, 04:15–04:35, on the shared 20-core machine at a 1-minute load of 1.6–4.7, single-threaded (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, ...). Before is `wt_head` (the checkout before tonight's merges), after is `wt_measure` (everything merged tonight). Both trees had their compartmental kernels compiled in Numba's cache before timing (the after tree had none, so `ss.compile()` was run first: 74 s), so the ODE and binomial workloads compare compiled against compiled. Before and after were interleaved, in alternating order, over three rounds; each figure is the best over the rounds of each benchmark's own measure (bench.py: cold is the first run in a fresh process, warm the best of three after it; perf guard: min of 8 inits or 3 runs after a warm-up). Raw logs: `_scratch/ab.log`, `_scratch/run2.log` in the measuring worktree.

## Summary

- **Faster**: `by_age` (`ss.by` age tables) 6.2× faster (8.5 s to 1.4 s in the guard, 8.5 s to 1.6 s in bench.py), from B0/B6's tables worked out once and indexed each step. The network SIR (`abm`) 19% faster (0.78 to 0.63 s) and the coded model about 10–15% faster, most likely from D1's int64 auids (indexing with UIDs 2–4× faster). The counts model 8–9% faster (D1: what holds for a step worked out once, not at each of RK4's four derivative calls). `fseis10` 5% faster.
- **Unchanged within noise**: everything else, including all the millisecond-scale perf guard entries, import time (about 1.0 s both), `multi`, the ODE, binomial ensemble, spatial and calibration workloads, Ghana measles and LASER speed.
- **Slower**: nothing beyond noise on the seconds scale. The expected small costs (FP55's stochastic rounding of dwells, about 2%; B1's check; B3/MA6's count substeps, about 3%) are not visible here: they are under the run-to-run noise (5–15%) and outweighed in the models where they apply (`counts`, `by_age`) by the speedups.
- **Memory regressed**: LASER's peak memory per agent went from 45.7 to 51.2 bytes (deterministic), so `tests/switching/test_laser.py::test_laser_memory` now fails its 50-byte target. Cause: D1/D2 made `people.auids` int64 again (`ss.dtypes.uid` int32 to int64): 4 bytes per agent, times 1.2 for the room kept for births, plus temporaries. This was a deliberate choice (worklog d1d2_speed.md, "int64 auids": performance ranks above memory), but its worklog didn't note the switching test. Not fixed: it is Cliff's call whether to keep int64 (and move the target to about 55 bytes, still within 2× of LASER's 30) or go back to int32 auids.

## S0.3 benchmarks (`v4_design/oct20/simplify/bench.py`)

| Workload | Before cold | After cold | Before warm | After warm | Change (warm) |
|---|---|---|---|---|---|
| `ode` (declared SIR, ODE) | 0.07 s | 0.07 s | 15 ms | 13 ms | same (noise) |
| `abm` (coded SIR, network, 20k, 20 y) | 0.84 s | 0.69 s | 0.78 s | 0.63 s | 19% faster |
| `multi` (SIS + SIR, demographics, 10k, 30 y) | 1.79 s | 1.68 s | 1.60 s | 1.61 s | same |
| `spatial` (50 places, binomial) | 0.09 s | 0.09 s | 28 ms | 27 ms | same |
| `ensemble` (100 binomial runs) | 0.40 s | 0.39 s | 0.153 s | 0.145 s | same (5%) |
| `calib` (Optuna, 20 trials) | 0.24 s | 0.22 s | 0.206 s | 0.187 s | 9% faster (noise?) |
| `fseis10` (malariasim, coded) | 0.85 s | 0.81 s | 0.82 s | 0.78 s | 5% faster |
| `counts` (declared, counts as agents) | 0.80 s | 0.75 s | 0.78 s | 0.71 s | 8% faster |
| `coded` (`ssl.AgentSIS`, demographics) | 1.20 s | 1.16 s | 1.12 s | 1.08 s | 4% faster |
| `by_age` (`ss.by` age tables) | 8.48 s | 1.67 s | 8.51 s | 1.65 s | 5.2× faster |
| Import time | 0.98–1.10 s | 0.94–1.00 s | | | same |

## Perf guard cases (`tests/test_baselines.py`, raw seconds, not normalized)

| Case | Before | After | Change |
|---|---|---|---|
| `bare_init` | 12.0 ms | 11.0 ms | same |
| `bare_run` | 140 ms | 138 ms | same |
| `bare_run_no_people` | 73 ms | 74 ms | same |
| `het_init` | 36.6 ms | 36.6 ms | same |
| `births_run` | 28.4 ms | 26.1 ms | same |
| `crn_false_run` | 126 ms | 127 ms | same |
| `fseis10_init` / `_run` | 32 ms / 0.78 s | 30 ms / 0.74 s | run 5% faster |
| `counts_init` / `_run` | 50 ms / 0.72 s | 48 ms / 0.66 s | run 9% faster |
| `coded_init` / `_run` | 23 ms / 0.95 s | 22 ms / 0.82 s | run 15% faster |
| `by_age_init` / `_run` | 110 ms / 8.54 s | 107 ms / 1.38 s | run 6.2× faster |

## Perf guard on the after state

`pytest test_baselines.py::test_performance_guard`, run serially at load 4.7: passed, all 14 entries OK (normalized against `tests/perf_guard.yaml`, limit 1.5× the baseline). Closest to its limit: `crn_false_run` 124 ms against a baseline of 93 ms (limit 139 ms); it is the same before tonight's work (126 ms raw before, 127 ms after), so its baseline is just tight. `by_age_run` 1,278 ms against 1,278 ms; `coded_run` 857 against 825; the rest are at or below their baselines. `tests/perf_guard.yaml` and `tests/benchmark.yaml` were not changed.

## Against hand-written code

| Model | Hand-written | Starsim before | Starsim after |
|---|---|---|---|
| Ghana measles, 260 districts, 2010–2025 daily, binomial, one run (warm) | 0.98–1.06 s (NumPy, `ghana-measles/gm-port`) | 2.18–2.29 s | 2.09–2.33 s |
| LASER England and Wales, 3.3M agents, 2 years (init / run) | 2.75 s (LASER, 20 threads, from `test_laser.py`) | 1.62 s / 9.76 s | 1.55 s / 9.61 s |
| LASER, marginal ns per agent-step (100k vs 300k agents, 20 days) | 1.1 ns (LASER, 20 threads) | 4.2–4.4 ns | 3.3–4.6 ns |
| LASER, peak bytes per agent | 30 (LASER) | 45.7 | 51.2 |

- Re-timed on 2026-10-22 at 10:45 on a quiet machine (load 0.2; warm, three runs each, interleaved twice): the code from before the night's work 1.73–1.74 s; the branch after the JAX track's merge 1.25 s on the kernels and 0.54–0.56 s in JAX, which it now chooses by default. The 2.2 s above was the same code as the 1.74 s, timed at load 4–5.
- Ghana measles: Starsim's binomial run (the model from `oct21/numba_vs_jax/models.py`, compiled kernels) is about 2.2× the hand-written NumPy loop, unchanged by tonight's work. (October 17 found them about equal, at about 2 s each, on a busier machine; today the NumPy loop runs in 1.0 s.) numba_vs_jax.md's JAX run (1.7 s) and its one-uniform-per-draw study (2.3× faster draws) are the known routes to closing the gap.
- LASER: unchanged by tonight's work. The full model here takes 9.6 s single-threaded (the brief's load rules keep Numba to one thread); `test_laser.py` quotes 7–8.5 s threaded against LASER's 2.75 s on 20 threads, so about 3×.

## Fixes made to the benchmarks (not to Starsim)

Two benchmarks no longer ran on the after state; both are fixed in the measuring worktree:

- `tests/benchmark_models.py`, `make_counts()`: raised `UnknownState` at init, since B2 (decision 3) requires an explicit first state when none is named `S` or `susceptible` (its states are `Sh`, `Eh`, ...). This also made the perf guard's `counts_init`/`counts_run` error. Fix: `first = 'Sh'` (accepted by the before tree too, so both trees run the same model).
- `v4_design/oct20/simplify/bench.py`, `spatial`: already broken before tonight (`ss.geo(...)`, `ss.Mobility`, `ss.gravity` are now `ss.geo.location`, `ss.geo.Mobility`, `ss.geo.gravity`); and after B2, a count of people goes in `init_n=` rather than `init=`. Fixed to the current spellings; the before numbers come from a copy with `init=`.

## Regressions to look at

- None on the seconds scale.
- Memory: LASER 45.7 to 51.2 bytes per agent from int64 auids (above), failing `test_laser_memory`; a decision for Cliff, not a local fix.
