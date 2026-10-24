# Invariance fuzzing: running notes

Worktree: `$WT=/tmp/claude-1000/-home-cliffk-idm-ssv4/a7dce0b8-ac27-4bef-aad0-2a880d856f55/scratchpad/wt` (v4 at `03741ba2`, `starsim/` identical to `a20c4f7e`). Python `/software/conda/envs/v4/bin/python` with `PYTHONPATH=$WT`; every script asserts `ss.__file__` is under `$WT`.

## Checklist

- [x] Learn the API interactively (`/tmp/.../scratchpad/fuzz/t*.py`)
- [x] `fuzz/generate.py`, `transforms.py`, `compare.py`, `run.py`, `shrink.py`, `summarize.py`
- [x] Harness sees N5 (`init_parallel`: runs identical) and N18 (`new_simstrat`: E1102 or a missing outcome), and N4 (`rename_dim_space`, seeds 90 and 111)
- [x] Pilot run (61 models x cheap, scratch only); fixed generator mistakes it exposed (seasonal `peak` is a fraction of a year, migration matrices are rates, modifiers on a duration invert under `dur_rate`, `rename_states` must rename a data column, Gillespie limited to 10,000 people, model functions exec'd in a real module so they pickle)
- [x] Main runs: `runs/cheap.jsonl` (about 220 models x 32 cheap transforms; `to_code_core` records removed, a harness bug), `runs/roundtrip.jsonl` (about 180 models x 5), `runs/supp.jsonl`, `runs/cross.jsonl`
- [x] Clusters (`fuzz/clusters.py`), shrunk repros (`repros/`), register match (`CLUSTERS.md`)
- [x] `CLUSTERS.md`, `RUNS.md`, `fuzz/README.md`, `tests/test_fuzz_smoke.py`

## Throughput

The machine ran at load 40 to 55 on 20 cores the whole time (other agents' test suites). A model through all 32 cheap transforms took about 13 s of CPU, but wall-clock throughput with 4 workers was about 3 models a minute, so the cheap run was stopped at about 220 models rather than the few thousand planned. Everything is resumable: rerun the same `run.py` command with a larger `--n` to extend it.

## Things learned while triaging

- C11: off-grid times hide it; it was 36 of 61 `units_sim` failures in the pilot.
- C18a: a function or step series switching between grid points converges at first order but not monotonically, so the Richardson test misfires (`scratchpad/fuzz/dt2.py`). A better test would use dt/2, dt/4, dt/8 with switch times on the coarsest grid.
- C02e: N18 is silent, not only a crash: `sim.new()` and `ss.parallel(sim, n_runs=)` drop `sim.stratify(...)` with its `adjust=`.
- C15b: X7 as agents at `dt=1`: peak 2,432 (dur) against 1,764 (rate) against 2,227 (ODE).

## Observations while learning the API

- A function or class analyzer that does nothing makes any compartmental solver raise E804 (the module "only works on agents"). Known: X9 (the probe does not cover function analyzers).
- Under `solver='abm'`, `ss.Sim([sis, sir])` gives a different `sir` result from `ss.Sim([sir, sis])` (8,430 against 8,630 final R), with two diseases that do not interact. 11-random.md: reordering modules does not change any distribution's draws. Candidate finding (X3 family?).
- An `ss.par(...)` used inline in a product (`ss.par(0.6) * 0.5 * I`) raises "Could not find a name for the parameter"; bound to a variable it works. The generator always binds.
- `ss.Mobility` needs `kind='space'` unless the name is `patch` or `region`: the generator gives it.
- `to_code()` writes a function rate as `ss.par({'fn': 'rate8'})` (broken code; R10) and leaves interventions out with a comment saying so.
- Class form: a bare number after `@` works; what fails is a module-level `ss.par` used inside the class body in a product or branch ("Could not find a name for the parameter 0.1282"), and a module-level `ss.par` is renamed by role (`dur3` becomes `dur`), so `sir.pars.dur3` raises. Cluster C04.
- The ABM module-order observation above is cluster C12 (`C12d_crn_module_order.py`).
- Two diseases plus a delivery without `disease=` raises E506 (by design); the transforms that add a disease add `disease='sir'`.

## Triage of the silent wrong answers (second pass)

Write-up in `triage/wrong_answers.md`; tests in `tests/test_triage_wrong_answers.py` (19 tests, about 8 s: 5 pass, 14 strict xfail); scripts in `triage/`.

- [x] C02: silent through `calibrate()` (beta 0.44 for 0.25, `repros/C02g_calibrate_simstrat.py`), `R0()`, `gradient()`, `ss.parallel(pars=)`, `check_dt()`; any `sim.pars` value set after construction is lost by `new()`; the `stratify=` keyword is not affected (generator renders it as `sim.stratify()` when it has options)
- [x] C11: monthly steps in a year sim put 40 of 119 months a step late (deliveries, `ss.change`, modifiers); C11c: `ss.date + ss.days(k)` a day late after a leap day; delays and time series fine (`repros/C11b_monthly_step_late.py`)
- [x] C15: O(dt), population-independent; `rate=` as agents is 21% low at the peak at `dt=1` because a per-step exit is drawn before transmission; `dur=` has a half-step correction
- [x] C17: three mechanisms, all "dwell fixed at entry" (`repros/C17_agents_dwell_frozen.py`)
- [x] C01, C03, C13: confirmed, no new trigger (C13 under `auto`: 1M people become 10,000 agents)
- [x] Extra runs (stopped at 96 and 90 models): `runs/triage_cross_abm.jsonl` (cross_abm, seeds 115-210), `runs/triage_misc.jsonl` (dur_rate, new_simstrat, cross_binomial, seeds 231-430). Seed 232 cross_binomial: an explosive model (two contact matrices multiply, growth 1.6/day), gap shrinks with dt (−18%, −6.5%, −4%): X4 family and the harness tolerance, not new (`triage/seed232_binomial.py`)
- [x] WA1 (new): a split branch with rate `par * timeseries` records its flow as 0 under ode/binomial (seeds 240, 307; `repros/WA1_branch_product_flow_zero.py`); other extra-run failures are C02, C15, or discretization on explosive models
