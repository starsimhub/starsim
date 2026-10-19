# Review 1: invariance fuzzing

Written 2026-10-18, against commit `03741ba2` of the `v4` branch, run from a clean worktree. Its `starsim/` is identical to `a20c4f7e`, which the earlier reviews used, so the IDs in [../README.md](../README.md), [../findings/](../findings/) and [../new-findings.md](../new-findings.md) apply unchanged. This is review 1 of [../next-reviews.md](../next-reviews.md). Nothing in `starsim/` was edited.

## In one page

**What was done.** A generator draws small declared models (2 to 6 states, every rate form, 0 to 2 dimensions, deliveries, modifiers, data and outcomes). 38 transformations each change the model in a way that should leave the answer alone, or change it in a known way, and the result is checked. Each failure was shrunk to a standalone script, grouped by root cause, and checked against the registers. Every confirmed failure has a strict xfail test.

**How much was run.** About 230 models per cheap transformation, 180 per round trip, 400 for the supplementary set, and 52 to 150 for the cross-solver comparisons. That is far short of the planned few thousand. Other agents' test suites kept the machine at a load of 40 to 60 for the whole run. The runs are resumable, and a larger `--n` continues them ([RUNS.md](RUNS.md)).

**What held on every model tried.** These need no further work:
- the string, operator, explicit, positional and lookup spellings;
- `1` against `1.0`;
- typed values in other units;
- renamed transitions and modules;
- the order of transitions, and exit order under the ODE;
- a one-level stratification, and `stratify=` against `sir.stratify()`;
- `init()` before `run()`, and `save()`/`load()` before a run;
- an unrelated analyzer under agents;
- `binomial` against the ODE at a fine step.

v4 rejected no generated model that the spec calls valid.

**What broke.** The fuzzer found 9 new problems, 6 of them silent wrong answers. It widened 5 known ones, including N18, which becomes a tier-1 silent wrong answer reaching `calibrate()`. It re-found 7 more known ones with nothing new, which shows that it can see what the hand reviews saw.

## New findings

IDs are V1 to V9 (V for invariance). The cluster IDs in brackets point into [CLUSTERS.md](CLUSTERS.md) and the two triage write-ups, [triage/wrong_answers.md](triage/wrong_answers.md) and [triage/crn_roundtrip_class.md](triage/crn_roundtrip_class.md), which have the root causes with file and line, and the suggested fixes. I re-ran the repros marked "checked".

| ID | Finding | Tier | Repro | Status |
|---|---|---|---|---|
| V1 | **Agents fix a dwell time when someone enters a state, so later changes to it are ignored for people already waiting.** All three should give `n_I` 498, as the ODE does. `ss.modify(sir.pars.dur, multiply=0.5)` gives 835. A time-varying rate on a split branch is frozen at entry, giving 4,302. A time-varying `dur=` is frozen too, giving 4,733. The same rate on a plain transition is a hazard and agrees (497). Any time-varying treatment or care-seeking speed run as agents is wrong. (C17) | 2 (arguably 1) | `repros/C17_agents_dwell_frozen.py` | Checked |
| V2 | **A split branch records a flow of 0 when its rate is an `ss.par` times a time series or seasonal factor**, under `ode` and `binomial`. The people do move (`n_H` 2,500), but `cum_severe` is 0, so an outcome, data, or a fit built on that branch sees nothing. The constant rate, the bare time series, and the same product on a plain transition are all correct. Root cause not yet located. (WA1) | 2 | `repros/WA1_branch_product_flow_zero.py` | Checked |
| V3 | **A time that sits exactly on a step boundary in another unit lands one step late.** With `unit='year'` and `dt=1/12`, a delivery or a modifier at `ss.months(7)` acts at step 8, and 40 of every 119 months are affected. `ss.weeks(19/7) >= ss.days(19)` is False. Times are rounded to 9 decimals and compared with no tolerance (`timeline.py:512-516`, `connectors.py:155-161, 278-285`, `time.py:1240-1270`). Delivery start, stop and rounds, `ss.change(at=)` and modifier windows are hit; outcome delays, product delays and time series are not. (C11) | 2 | `repros/C11b_monthly_step_late.py` | Checked |
| V4 | **`ss.date + ss.days(k)` is a day late after a leap day.** `ss.date('2000-01-01') + ss.days(200)` gives 2000-07-20, where it should be 2000-07-19, so a delivery at `ss.days(200)` lands at step 201. Dates are added through fractional years (`time.py:461`). (C11c) | 2 | `repros/C11b_monthly_step_late.py` | Checked |
| V5 | **`to_code()` silently drops `ss.contacts` matrices.** The file has only a comment, and there is no warning. Infections by risk group come to 4,853 and 4,947 after the round trip, against 749 and 749 before. `sim.hash()` does change. (C09b) | 2 | `repros/C09b_to_code_contacts.py` | Agent |
| V6 | **Adding or swapping a disease changes another disease's draws under agents.** Distribution traces are named `declared_<index>_…`, because `distributions.py:187-195` finds `sim.declared` (`sim.py:302`) before the module's own path. Adding a disease earlier in the list moves `n_I` of the other by 645. The fix is to skip `declared` there; the agent checked that the names then become `diseases_sir_…`. This is the agents' counterpart of X3. (C12d) | 2 | `repros/C12d_crn_module_order.py` | Agent |
| V7 | **Under `binomial` and `gillespie`, reordering the states changes the draws.** `11-random.md` promises one stream per transition, but each engine draws from one sequential generator, in source-state order (`solvers.py:585, 2277`). X3's proposed fix (one Generator per engine) does not cover this. (C12a) | 2 | `repros/C12a_crn_state_order.py` | Agent |
| V8 | **`to_code()` writes time-series keys and modifier `start`/`stop` as strings, which load back as dates.** The rebuilt model has the same `sim.hash()`, then fails at `run()`. (C06, C07) | 3 | `repros/C06_to_code_timeseries.py`, `repros/C07_to_code_modifier_time.py` | Agent |
| V9 | **The class form cannot use a module-level `ss.par`.** In a product or a branch it raises "Could not find a name for the parameter". Otherwise the parameter is renamed by its role, so `ss.modify(sir.pars.<variable>)` raises `AttributeError`. The cause is that `diseases.py:84` deep-copies the class members before parameters are named by identity. The class example in `00-overview.md` itself works. (C04) | 4 | `repros/C04a_class_par_product.py`, `repros/C04b_class_par_names.py` | Agent |

Two tier 6 notes:
- `ss.seasonal(peak=142)` is accepted silently, because `peak` is a fraction of the year and is taken mod 1.
- `check_dt()` can't see the ODE's step error, because the ODE picks its own substep count (`solvers.py:2922-2925`).

## Known findings, widened

| ID | What the fuzzer adds | Tier |
|---|---|---|
| N18 | **It is a silent wrong answer, not a failure, and it reaches inference** (C02). Two triggers: a sim stratified with `sim.stratify()`, where no contact matrix names the dimension, and any value set on `sim.pars` after construction (`stop`, `total_pop`, `dt`, `rand_seed`). Both are lost by `new()`, so everything built on `new()` runs a different model with no warning: `ss.parallel` gives `n_I` 219 against 13,264 in one run, and `calibrate()` fits `beta` 0.44 to data made at 0.25 (checked). `sim.R0()` gives 1.50 against 2.64, and `gradient()`, `check_dt()` and `ss.parallel(pars=)` are also wrong. The `stratify=` keyword is not affected. The cause is that `sim.py:544` writes the strata into `self.pars`, while `new()` (`sim.py:431`) rebuilds from `self.inputs`. | 1 |
| X3 | **Cheaper triggers** (C12c/d, C19). Under `binomial` and `gillespie`, any delivery puts the run on the stepped path, where adding or reordering an unrelated disease changes the draws. X3's own trigger is `run(profile=True)`. `ss.parallel` draws every run's outcomes from the base seed's generator (`solvers.py:2799-2816`), so outcomes differ from the matching single runs while states agree. | 1 (as ranked) |
| X7 | **Larger than "small", and the proposed fix would make it worse** (C15). As agents at `dt=1`, a `rate=0.25` exit peaks at 1,757 and `dur=4` at 2,420, against the ODE's 2,227. The gap is the same at 10,000 and 100,000 agents and shrinks with `dt`. The cause is that a `rate=` exit is drawn before transmission in the same step (`solvers.py:3655`), while a `dur=` exit gets a half-step correction (`:3571`). X7 proposes making an exponential `dur=` a per-step hazard; without a half-step correction, that would move `dur=` to the worse number. | 2 |
| X9, N22 | **It is common, and a whole-model switch.** Any analyzer, even a no-op, makes every compartmental solver raise E804. Under `auto`, it silently turns a 1,000,000-person ODE into 10,000 agents (C13; 166 of 181 models). | 3 |
| R10 | **Two new constructs:** contact matrices (V5) and time-series keys and modifier times (V8). The function-rate (C05) and `seasonal`/`timeseries` prefix (C08) cases are already in R10. | 2 |

**Re-found, nothing new:**
- N5: `init()` then `ss.parallel` gives identical runs, on 180 of 181 models (C01).
- D4 and N4: the dimension's name decides mixing (C03). D4 without a contact matrix is the commonest trigger.
- R15: `save()` after `run()` fails on every model (C16).
- The other R10 cases (C05, C08, C09a).

These show that the harness sees what the hand reviews saw.

**Dismissed:**
- 1e-10 differences under the ODE, which are summation order on two code paths (C14). Raise the harness's ODE tolerance to 1e-8.
- `dt_half` non-convergence (C18), a harness artefact that has been fixed: switch times now sit on the coarse grid and `substeps` is fixed.
- Agent-solver CRN changing with state or dimension names (C12b, C12f), because streams are named after declarations, as the spec says.
- Six agent and binomial gaps from the extra runs, on explosive early epidemics. Each gap shrinks with `dt` (X4 family). The harness tolerance there should scale with the fastest rate times `dt`.

## What is in this folder

- `fuzz/`: the harness, with instructions in [fuzz/README.md](fuzz/README.md). The files are `generate.py` (the seeded grammar, which writes each model as a script), `transforms.py` (38 transformations, each with its expected effect), `compare.py` (1e-10 for deterministic paths; exact draws where common random numbers promise it, distributions otherwise), `shrink.py`, `run.py` (parallel, resumable, JSONL), `summarize.py` and `clusters.py`.
- `runs/`: every result, and [RUNS.md](RUNS.md), the counts per transformation.
- [CLUSTERS.md](CLUSTERS.md): one row per root cause, with counts, signatures, causes, register matches and tiers.
- `repros/`: a standalone script for each cluster, which prints the two numbers that should agree, with the shrunk model as `.spec.json`.
- `triage/`: the two triage write-ups and their scripts.
- `tests/`:
  - `test_fuzz_smoke.py`: a fixed-seed smoke set that replays the shrunk specs, about 60 s under load. The fixed-seed long run is under `-m slow`.
  - `test_triage_wrong_answers.py`: 21 tests, 6 passing and 15 strict xfail.
  - `test_triage_crn_roundtrip.py`: 26 tests, 10 passing and 16 strict xfail, plus 3 under `slow`.

  Each xfail names its finding ID, so it fails loudly once the bug is fixed.
- `NOTES.md`: the harness agent's running log.

To run anything here, put a checkout of `03741ba2` (or later) on `PYTHONPATH` and set `STARSIM_WT` to it. Every script asserts that `starsim` was imported from there.

## What to do with it

1. **Fix the silent ones and their regression tests first,** with phase 3 of [../implementation.md](../implementation.md): N18 as widened, V1, V2, V3, V4 and V5. N18 and V3 are a few lines each; V1 needs agents to redraw or rescale dwell times when their inputs change, as `rescale_exits()` already does for modifiers keyed on the transition.
2. **Move the harness into `tests/fuzz/` for phase 2.** Make three changes first:
   - loosen the ODE tolerance to 1e-8;
   - scale the stochastic tolerance by the fastest rate × `dt`;
   - expect "same distribution", not "same draws", for agent-solver renames.

   Then put the smoke set in CI and keep the long run under a marker.
3. **Run it at the planned scale** once the machine is quiet: a few thousand models per transformation, and several hundred cross-solver models. At about 230 models per transformation, a failure that hits one model in 200 would be missed about a third of the time. The grammar also has no births, no vector or reservoir counts, and no agent networks (its only mixing is contact matrices and mobility), so widening it is the next step.
