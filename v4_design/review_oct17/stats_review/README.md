# Review 2: a statistical and scaling audit

Written 2026-10-18 against commit `03741ba2` of the `v4` branch, run from a clean worktree. Its `starsim/` is identical to `a20c4f7e`, which the earlier reviews used, so the IDs in [../README.md](../README.md), [../findings/](../findings/) and [../new-findings.md](../new-findings.md) apply unchanged. This is review 2 of [../next-reviews.md](../next-reviews.md), in two halves:

- [accuracy/](accuracy/README.md): are inference and the stochastic solvers statistically right?
- [cost/](cost/README.md): does cost grow the way it should?

Nothing in `starsim/` was edited. The machine was shared and ran at a load of 20 to 60 throughout. That doesn't affect the accuracy results. For cost, it affects absolute times (1.5 to 3 times slow) but not the exponents; the headline comparisons were re-measured in a quieter hour.

## In one page

**Inference is right where it was built to be right, and both known inference bugs now have a one-line cause.**

- **What is right:**
  - Away from the triggers below, the JAX log-density equals `sim.loglik()` to 1e-13 across 12 rate forms and 5 model variants.
  - NUTS on one parameter passes simulation-based calibration.
  - L-BFGS-B and Nelder-Mead recover one parameter to 0.1%.
  - The particle filter's likelihood estimate is unbiased, and its variance falls with the particle count.
  - Euler is order 1 and RK4 order 4.
  - `binomial` and `tau` match exact Gillespie at a fine step.
- **N1 (the particle filter) is much wider than filed, with a one-line cause.** Any value that varies smoothly within a step breaks it: `ss.seasonal` alone, or a linear `ss.timeseries` alone, on any flow, fitted or not. Inside the filter, time runs at a third of its speed, because `kernels.advance()` reads row `ti-1` of tables that hold 3 rows per step (`calibration.py:1595`, `kernels.py:1241`, `solvers.py:2570`). Every seasonal PMCMC fit is affected. The filter's test uses a constant `beta`, which is why it never caught this. (The test is not skipped, as the earlier review said.)
- **N2 (the JAX density) also has a single cause.** The density is wrong whenever a fitted parameter times any time-varying value is the rate of a flow that is not transmission, such as recovery or detection, because one value is written into every row of the per-step table (`gradients.py:946-956`). `fit()` notices and falls back to the slow path: on this corpus the posterior was right but took 22 times as long.
- **Four new inference problems, three of them silent.** A two-parameter optimizer fit stops at a worse local optimum 9 to 13% of the time, with SciPy reporting success (A2). NUTS chains all start at the same point, so they can agree confidently on the wrong mode (A4). The density caps its substeps at what the starting values needed, so fits with a high R0 fall back and slow down 5 to 90 times (A1). `solver='sde'` gets worse as `dt` shrinks, and never dies out where Gillespie does 14% of the time (A3).

**Most axes scale linearly, but places do not, and that is where the specialists live.** Agents, steps, runs, particles, groups without a matrix and diseases are linear (47 ns per agent-step for a plain SIR). These are the exceptions:

| Axis | Exponent, where 1 is expected | Cause | Finding |
|---|---|---|---|
| Places with sparse mobility | 2.1 to 2.3 | mobility is dense in every compartmental solver, even with no commuting | SC1 (widens N23) |
| Ages × places | 1.5 to 2.6 | the dense (places × ages)² matrix | SC1, N23 |
| A contact matrix on a `patch` dimension | 3.7 to 4.0 | it is expanded to places² × places²: 40 GB at 300 places, and 7 TiB requested at 1,000 | SC2 (the N4 mechanism; also the wrong answer) |
| The particle filter, in data length | 1.7 to 2.0 | every derived result is recomputed over the whole history at each data point | SC3 (a cause of N27) |
| Gillespie, interacting groups | quadratic | every event rescans every flow in every group | SC6 (widens N26) |
| Building with a matrix over groups | 1.6 to 2.2 | one `ss.constant` Dist per matrix entry, deep-copied twice: 54 s and 3.3 GB to build 600 places, for a 0.05 s run | N28, refined |

NumPy's BLAS threads are also uncapped inside the kernels (SC4). On a loaded machine, or with runs in parallel, one 200-place model took 34 s of wall time and 429 s of CPU, against 4.4 s with one thread.

**Against the specialists:** MEmilio is 16 to 24× ahead, SimInf 38× on the run, odin's particle filter 11× per particle-step, and LASER about 14× on one thread. v4 is still ahead of flepiMoP at 3,000 places. Fixing SC1, SC3 and SC4 would close most of the MEmilio and odin gaps without any new design.

## Findings

Accuracy findings are A1 to A4; cost findings are SC1 to SC7. Each has a repro and, where the problem is structural or can be timed stably, a strict xfail. The details, with causes and fixes, are in each half's README. "Checked" means I re-ran the repro myself.

### New

| ID | Finding | Tier | Status |
|---|---|---|---|
| A2 | Two-parameter optimizer fits (`b`, `dur`) stop at a worse local optimum in 9% (L-BFGS-B) and 13% (Nelder-Mead) of SBC fits. They end 1.6 to 15 log-likelihood units low, and SciPy reports success every time. SciPy's flag also says `ABNORMAL` on 5 to 38% of fits that are right, so R5's `converged` should not be built on it. Restart from the best 2 or 3 scan points and report any disagreement. | 2 | Checked: both optimizers stop at a negative log-likelihood of 58.86, against 54.25 at the real optimum |
| A4 | NUTS can return a confident posterior hundreds of log-likelihood units below the truth with R-hat 1.01 and no divergences: 3 of 96 two-parameter fits. This happens because every chain starts within 0.01 of the initial values (`calibration.py:2333`, `:1832`), so the chains cannot disagree. The spec says NUTS starts from the optimum, but no optimum is found first. | 2 | Agent |
| A3 | `solver='sde'` does not converge in `dt`; its bias grows as `dt` shrinks. Each flow is `max(mean + sqrt(mean) Z, 0)` (`kernels.py:651`), which is biased upward when a flow is small. SIR final size is +1.0% at `dt=1` and +3.5% at `dt=0.01`. The SIS never goes extinct, against 14.3% under Gillespie. Related to X5 and X17. | 2 | Checked: +1.03%, +1.29%, +2.60%, +3.52% at `dt` 1, 0.25, 0.05, 0.01 |
| A1 | The JAX density caps its substeps at what the run at the initial values needed (`gradients.py:890`, `:299`), so faster dynamics get a coarser ODE than the sim: 37 log-likelihood units off on a measles-like fit. `fit()` catches it and falls back in 6% of one-parameter and 21% of two-parameter SBC fits, which are 5 to 90 times slower. One fit ran for more than two hours. | 3 | Agent |
| SC2 | A contact matrix over a dimension named `patch` is expanded to places² × places². That costs 40 GB at 300 places and a `MemoryError` at 1,000, on top of N4's wrong answer. | 1 (as N4) | Checked: at 60 places, a 3,600 × 3,600 matrix and a final R of 15,579, against 933,211 with the dimension named `place` |
| SC3 | At each data point, the particle filter recomputes every derived result over the whole history (`calibration.py:1597-1610`), so it is quadratic in run length with daily data. Threads would not remove this. | 3 | Checked: 11.6 of 14.6 s is in `collect()` |
| SC4 | BLAS threads are uncapped inside the kernels. Cap them when sims run in parallel (on a quiet machine, threads helped one 800-place run). | 3 | Checked: 34.2 s wall and 429 s CPU, against 4.4 s with one thread |
| SC1 | Mobility is dense everywhere, so every compartmental solver is quadratic in places even with sparse mobility (`transitions.py:2693-2714`, `solvers.py:163-183, 2207-2224`, `kernels.py:679`). | 3 | Agent |
| SC5 | The register is read and resolved one row at a time in Python (`transitions.py:2619-2680`), at 12 to 54 µs per event. The build takes longer than the binomial, tau and ODE runs, which explains N26's build time. | 3 | Agent |
| SC6 | Gillespie costs O(groups) per event when groups interact (`kernels.py:439-470, 589-635`): 27 µs at 10 strata and 83 µs at 40, against a flat 4 µs for independent places. | 3 | Agent |
| SC7 | `init()` grows as D^1.5 in the number of diseases, from a per-module `sc.search`. | 6 | Agent |

### Known, with new evidence

| ID | What this review adds |
|---|---|
| N1 | Much wider trigger: any smoothly time-varying value, alone, on any flow, fitted or not. The root cause is one line, and the fix is to pass `per_step` to `advance()`, as `step_at()` does. Slicing the tables `[::3]` removes the error. Tier 1. Checked: a held series agrees exactly (−239.6 both); a linear one is −431.4 against −239.5. |
| N2 | Wider trigger: `ss.seasonal` as well, and any flow that is not transmission. The root cause is in `gradients.py:946-956`. N2's report of all draws divergent didn't reproduce on this corpus; it may be specific to summer2's stiff TB model. Checked: −651.5 against −76.7 at the initial values when the recovery's multiplier is fitted, and exact when it is not. |
| N16 | Optuna on a binomial model overestimates `b` by +8.7% ± 1.8% over 100 prior draws. |
| X4 | Numbers for the docs. Binomial at `dt=1` has SIS extinction at 9.3% against 14.3%, and the SIR peak 1.6 days late. At `dt=7`, the peak is +32% and extinction is 0.4%. Agents at `dt=1` put the peak 3.4 days late. Both are order 1 in `dt`, but the coefficient for agents is large. |
| N23 | Wider: there is no need for commuting or for ages. Measured at 2.1 to 2.6. |
| N26 | SC5 explains the build time, and SC6 widens the run cost. |
| N27 | SC3 is a further cause. 14.5 µs per particle-step, against odin's 1.35 µs. |
| N28 | Refined cause (one Dist per entry, deep-copied twice), plus memory: 3.3 GB at 600 places. Checked: 9.0 s to build 200 places for a 0.02 s run. |
| N24, N25 | Both confirmed as linear: 123 bytes per group-step, and 47 ns per agent-step with 170 bytes per agent for a plain SIR. |
| X8 | Each solver needs its own cold kernel build: 36 s for the ODE, 38 s for SDE, and 62 s for Gillespie at low load. |
| R5 | SciPy's `success` flag is wrong in both directions (A2). |

Two smaller notes:
- `parameters.py:326` says binomial's default substeps are one per stage of the slowest dwell time, while `solvers.py:2846` uses 1 (tier 5).
- `ss.perday(1/6) * ss.timeseries(...)` raises `TypeError` (N21 family).

## Open questions

- **Two-parameter NUTS covers only 86% (`b`) and 89% (`dur`) at the 95% level, while its rank tests pass.** A 60 × 60 grid posterior matches NUTS, and the data generator matches the likelihood. Removing the A4 cases still leaves the shortfall unexplained. It needs more replicates and longer chains; these were 2 × 200.
- **PMCMC covers 90% ± 3.1% at 95% over 50 replicates,** on a constant-rate model, which is borderline.
- **The binomial register at 12,800 herds needed over 16 GB,** where Gillespie used 2.9 GB, and the cause wasn't found.
- **Not reached:** N29, scaling of the JAX path and NUTS, the `tau` and `sde` particle filters, and Gillespie at 1,000 commuting places (it timed out).

## What is left behind

- **The coverage tables** for each method ([accuracy/README.md](accuracy/README.md#sbc-coverage-per-method)), and **the scaling table**, axis × solver, with exponents and the size at which each becomes a problem ([cost/README.md](cost/README.md#the-scaling-table)). Together they are the honest version of the north-star time claim (P8).
- `accuracy/tests/test_accuracy.py`: 31 guards that pass, including the likelihood equivalence across rate forms and a 35 s SBC smoke test, plus 8 strict xfails (N1 ×3, N2 ×2, A1, A2, A3). The 50-replicate NUTS SBC and A4 are under `-m slow`.
- `cost/tests/test_scaling.py`: two-point slopes on the cheap axes (7 pass) and strict xfails for the superlinear ones and the structural checks (8). The full curves are under `-m slow`.
- `cost/perf_guard_scaling.yaml`: a budget per axis, as an exponent bound plus a time at one reference size, so a superlinear regression fails a test.
- Rerunnable, seeded scripts for every table, in `*/scripts/`, with raw results in `*/results/`.

To run anything here, put a checkout of `03741ba2` (or later) on `PYTHONPATH` and set `STARSIM_WT` to it. Every script asserts that `starsim` was imported from there. The cost tests set `OPENBLAS_NUM_THREADS=1`.

## What to do with it

1. **Fix N1 and N2 first.** Each is a few lines now that the cause is known, and each comes with an equivalence test that guards every rate form.
2. **Then fix A2, A4 and R5 together.** Restart optimizers from the best scan points, start NUTS chains from distinct optima, and report disagreement instead of SciPy's flag. That turns two silent failure modes into a warning.
3. **For performance, do SC3, SC4 and N28 first,** because each is small. Then factored mobility (SC1 and N23, which also fixes SC2's cost) and Gillespie's per-event cost (SC6).
4. **Decide what to do with `sde`.** Fix its flow distribution, or cut it as X17 proposes; today it is the one solver that gets worse as `dt` shrinks.
5. **Rerun both halves on a quiet machine before quoting absolute times,** and add the two-parameter NUTS coverage question to the SBC grid with longer chains.
