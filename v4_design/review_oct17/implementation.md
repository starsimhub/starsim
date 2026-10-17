# Implementation plan

This file turns the review into work. [README.md](README.md) says what is wrong and why it matters; this file says what to change, where, in what order, and how each change is checked. Finding numbers point to the merged registers in [findings/](findings/) (L, D, X, R, P) and to [new-findings.md](new-findings.md) (N), where each has its repro, the reviewers' positions, and the cost and breakage in full. Code locations are at `a20c4f7e`. When an item is done, check it off here and record it in [../admin/implementation.md](../admin/implementation.md). When one is looked at and set aside, move it to [../admin/deferred.md](../admin/deferred.md) with what it would take and why it waits.

## How to work through it

- **A test with every fix.** Each fix lands with a regression test that fails before it and passes after. A silent wrong answer is tested on the number the user would see (a peak, a final size, a log-likelihood), not on internals.
- **No fix slows the kernels.** Run the benchmarks (`tests/perf_guard.yaml` and `tests/benchmark_*.py`) before and after each phase, with repeated runs. A slowdown is a bug in the fix.
- **Solver agreement at a fine step.** Agents and stepped solvers differ from the ODE by an amount of order `dt`, so solver-agreement tests use a fine step, or a loosened tolerance with a comment saying why.
- **Cliff picks changes to what a solver means.** X4, X7, X13, and X22 change results across the board. Each is put to Cliff with the numbers and the options, not changed quietly.
- **Breaking changes are fine.** Each gets a line in `CHANGELOG.md`, and, if a Starsim 3 user would hit it, a row on the migration page (phase 6).
- **Decisions.** "Decision n" means decision n in [README.md](README.md#decisions-for-cliff), and "Decision lang 4" means decision 4 at the end of [findings/language.md](findings/language.md) (likewise dims, exec, results, docs). Each item is written for the recommended option, so it is ready once Cliff agrees. Items without a decision mark need none.
- **Sizes.** The findings' costs ("a day", "a week") are a human developer's; this kind of work has run 20 to 50 times faster. Sizes here are S (under half an hour with tests), M (an hour or two), and L (half a day or more).

## Phase 1: unblock

A short phase that removes the dead ends a new user meets first. No decisions.

- [ ] **R can build a model** (N17, S). `sys._getframe(1)` at `diseases.py:74` and `sim.py:73` raises `ValueError: call stack is not deep enough` under reticulate. Catch it and fall back to the names that need no frame (endpoint names, `name=`). Test: build and run an `ss.Disease` from code with no Python caller frame (as `me/noframe.py` does), and add a CI job that runs R-Starsim's README example.
- [ ] **Say when the kernels are being built** (X8, S). Print `building the solver kernels for first use (about 30 s; once per install)` when Numba's cache is cold.
- [ ] **Distinct runs after `init()`** (N5, S). `run.py:1216-1218` copies an initialized sim and sets `rand_seed` when it no longer has any effect, and the warning at `run.py:822` is skipped. Build each run with `sim.new(rand_seed=...)` whenever the sim has `inputs`; otherwise raise, saying to pass an uninitialized sim. This also lets `ss.parallel` accept a sim that has already run (P1b), and keeps the fast restart path (`run.py:1122`). Test: after `init()`, `ss.parallel(sim, n_runs=4)` gives four different final sizes, the same four as without `init()`.
- [ ] **`sim.stratify()` survives `sim.new()`** (N18, S). Record the call in `sim.inputs` (`sim.py:535`) so `new()` replays it (`:86`, `:431`). Test: after `sim.stratify(age=...)`, `sim.new()` has stratified results, and `calibrate()` and `ss.parallel` run on it.
- [ ] **`sim.save()` after a run** (R15, S). `Loop.shrink` shrinks `funcs`, which `plan` then iterates. Give the engines a `shrink()` that drops their arrays. Test: a save and load round trip under each solver.
- [ ] **The spec's headline examples run** (P1a, P1b, S). The two-line SIR gets `beta=0.3` and `init_prev=0.01` wherever it appears. `AlreadyRunError` names `sim.new()`. `msim.plot()` works for an outcome with `every=` (today it raises "No legend handles found"). The test is P1c in phase 2.
- [ ] **Portable test paths** (P12, S). Replace the hard-coded `/home/cliffk` paths in `tests/test_interop_*.py` with `STARSIM_LFD`, skipping with a message that names it. `test_samples` uses `tmp_path`.

## Phase 2: contract tests

These come before the bug fixes in phase 3, so each fix has a safety net and the next bug of the same kind is caught. Together they would have caught most of the 74 silent wrong answers, including N1 and N2, before any user did. No decisions.

- [ ] **A model corpus** (P10b, M). Put the models in `tests/corpus.py`, each as a named constructor, imported by every test that needs a model:
    - the nine build-set models and the eight wave-2 models;
    - one small model for each seat's standard pattern, at least: `beta` as a fitted par times `ss.seasonal` times `ss.timeseries`; an age-stratified model with an age-targeted vaccine; two places with an explicit contact matrix over a dimension named `patch`; a Poisson likelihood on weekly averages; and a Starsim 3 SIR.

  Solvers come in through `@pytest.mark.parametrize('solver', ...)`, replacing the per-file loops.
- [ ] **The spec's code runs** (P1c, S). One test extracts every `python` block from `v4_design/design/00` to `09` and runs it. Blocks marked as fragments are skipped. A disease example must show growth (`n_I.max() > n_I[0]`).
- [ ] **The likelihoods match the run** (N1, N2, M). For every corpus model with data:
    - the JAX log-density equals `sim.loglik()` at three parameter points, to 1e-8;
    - the particle filter's estimate, at a large population with many particles, is within its Monte Carlo error of the likelihood of the ODE limit, and it peaks at the same parameter as direct runs.

  Include a fitted par times `ss.timeseries`, times `ss.seasonal`, and times both.
- [ ] **Simulation-based calibration** (P10, R5, M). For each method (Nelder-Mead, L-BFGS-B, Optuna, NUTS, PMCMC):
    - simulate from known values on a small corpus model, and fit;
    - posterior methods must cover the truth, and optimizers must recover it within a tolerance;
    - `converged` must be true.

  A seeded version runs in CI in seconds; a longer one runs under a marker.
- [ ] **Streams are independent** (X3, X19, S). Adding an unrelated module (an analyzer, or a disease that shares no states) leaves a stochastic disease's draws unchanged under `binomial`, `gillespie`, and `abm`. No two engines share a seed. Renaming a module changes only names.
- [ ] **The solvers agree** (S). Each corpus model under `ode`, `binomial`, `gillespie`, and `abm` at a fine step: stochastic means within a stated tolerance of the ODE. Add the three-exit permutation test (X5): reordering competing exits leaves the results unchanged.
- [ ] **Conservation and exact identities** (P10, S). Without demographics, the population is conserved. Constant-rate decay matches the exponential. A fixed branch share matches `p`.
- [ ] **The guesses are marked** (L27, S). For each row of the "what is guessed" table in `03`, one model triggers the guess and checks that `describe()` marks it, and one does not trigger it and checks that it is not marked.
- [ ] **The switching demos** (README §1, M to start). `tests/switching/` holds one script per seat, scaled down to run in CI, asserting the original framework's reference numbers. The numbers are stored as numbers; the originals stay in LFD. Each demo starts as `xfail`, naming what blocks it (see [the acceptance table](#acceptance-the-thirteen-switching-demos)), and the goal of the plan is that all thirteen pass.

## Phase 3: the silent wrong answers

Fix and test each, in this order within each group. Items marked with a decision wait for it; the rest need none.

### Parameters, modifiers, and fitting

- [ ] **Modifiers on time-varying parameters** (D1, S). `Engine.value` returns at `solvers.py:847` for a callable before `self.modified()` at `:857`. Use one path: evaluate the raw value, convert it, then apply modifiers, for scalars, callables, `ss.timeseries`, and `ss.seasonal`. Test: each of the four forms under `ode` and `abm`, where `multiply=0` gives no infections.
- [ ] **JAX density with a fitted multiplier of a time series** (N2, M). Start from the `fitdm_ts` variant of `summer2/nuts_bisect.py`. Check whether N1 has the same root before fixing either.
- [ ] **Particle filter with seasonal times time series** (N1, M). Start from `epydemix/t2_pf_repro.py`. Un-skip the filter's test.
- [ ] **Count likelihoods on non-integer data** (N6, S). Use the continuous extension (`gammaln(k + 1)`) or raise, naming the column. A fit whose log-likelihood is `-inf` at the start point raises.
- [ ] **One kind per result** (R1, S). Set `kind` (`'flow'`, `'stock'`, or `'ratio'`) when each result is created. `compare()` (`calibration.py:1330`, `:1359`), `check_dt()` (`sim.py:476`), `resample()`, and `summarize()` (`sim.py:2042`) read it, and the name heuristics go. Test: a `new_infections` data column is summed over each data interval.
- [ ] **One resampler** (R3, S). Driven by `kind`: flows sum, stocks take the last value, and ratios divide resampled parts. Bins anchor at the sim's start. `annualize()` becomes `resample('year')`.
- [ ] **Stochastic fits: the score, and the luckiest draw** (R4, N16, S). `fit.mismatch` is the ensemble score of the chosen point. Stochastic fits default to `n_runs=4` under CRN, printed (R25).
- [ ] **`fit.converged` and `fit.why`** (R5, S). Each method fills them from what it already knows. W1301 is computed on the searched scale.
- [ ] **Inputs that change what is fitted** (R6, S). An unknown `gof` raises. Optimizers minimize the negative log-posterior when every series has a likelihood and priors exist.
- [ ] **Drop `every=` on outcomes** (R7, S, Decision results 9).
- [ ] **The hash covers what changes results** (R8, M). It covers modifiers, function source, literals at full precision, and numerical options, plus a fifth `data` part. Do this together with R29, so every hash changes only once.
- [ ] **`compare_models` uses one measure** (R9, S).
- [ ] **`to_code()` says what it could not write** (R10, S). It raises `W1401` listing those parts, and prefixes every Starsim name with `ss.`.
- [ ] **Parameter values are checked** (L10, S). At `init()`, raise `E604` for an unvalued parameter and `E606` for one outside its bounds (Decision lang 16), and require finite, non-negative initial values.
- [ ] **`sir.pars.beta = x` keeps the `ss.par`** (L22, S). `Pars.__setitem__` (`parameters.py:73`) sets `.value` on the existing `ss.par`.
- [ ] **Only a rate as a whole carries a unit** (L2, R12, S). A bare factor inside a product is dimensionless. Importers mark known dimensionless parameters through `Source.add_par`.
- [ ] **An imported `N` equal to the population is the total** (R11, S).

### Dimensions, mixing, and space

- [ ] **An explicit contact matrix is always used** (N4, S). `ss.contacts(<dim>=matrix)` applies whatever the dimension is called. The full rule, kind from the value, is D4 in phase 4. Test: the same model with the dimension named `patch` and named `zone` gives the same results (today [9,396, 0] against [9,882, 9,882]).
- [ ] **Age comparisons on a stratified model use ages** (N3, S). Compare a binned dimension against its band edges, and raise when a threshold is not an edge. Test: `sir.age < 5` selects the same people as `sir.age == '0-4'`, under both engines.
- [ ] **Partial cross-immunity runs as agents under `auto`** (D13a, S).
- [ ] **Contact matrices aggregate by population, not band width** (D14, S). Validate the matrix entries and band edges.
- [ ] **A `sex` dimension is `people.female`** (D15, S).
- [ ] **Gravity: shape from gravity, amount from `away=`** (D16, S, Decision dims 12). Warn when the cap binds.
- [ ] **Note `I.n` used literally alongside a route** (D17, S).
- [ ] **Per-route `beta` in one place** (D21, S). Check the keys in both directions with `E403`.
- [ ] **Networks under a compartmental solver** (D10, S, Decision dims 8). Print the approximation, or raise `E804`.
- [ ] **`stratify(immunity=...)` makes a dimension or raises** (N14, S). An explicit `split=` is honored.
- [ ] **One `stratify` parser** (D5, S). An unknown keyword is an error.
- [ ] **Compartmental aging uses the split's density at the band edge** (D12b, S).
- [ ] **`ss.by(stage=...)` on an inflow raises instead of giving 0** (N11, S). The feature itself is in phase 7.

### Deliveries and effects

- [ ] **A waning product is not re-delivered every step** (D6, S). `remembers` is true when the product can be lost.
- [ ] **`set=` and `add=` on per-person quantities raise** (D7, S, Decision dims 6).
- [ ] **`ss.change` returns an `ss.modify`** (D8, S). `start=`, `stop=`, and `when=` then work.
- [ ] **A pending test result blocks re-testing** (D9, S, Decision dims 7).
- [ ] **`ss.Treatment(cure=, efficacy=)`** (D18, S, Decision dims 15). `ss.Vaccine` requires `efficacy=`.
- [ ] **A follow-up with its own schedule raises** (D19, S).
- [ ] **`capacity=` units** (D20, S, Decision dims 16).
- [ ] **"Routine at 9 months" vaccinates as people reach the age** (N8, S). A routine delivery by age skips people already past the window at the start; with `at_age=` (phase 7) it is spelled directly. Fix `08-interventions.md:91`.

### Execution

- [ ] **A distribution on `beta`** (X1, S, Decision exec 7). Sampled per agent under `abm`; agents under `auto`; the mean with a note under an explicit compartmental solver; `E804` under `approx=False`.
- [ ] **A solver asked of a coded model raises** (X2, S).
- [ ] **One random generator per engine** (X3, M, Decision exec 12). Pass an `np.random.Generator` into the kernels, seeded from a hash of `(rand_seed, full module path)`. Stochastic baselines move once.
- [ ] **`sde`** (X5, S, Decision 8). Cut it; if it stays, split exits in proportion to their hazards.
- [ ] **Tau-leaping clips per compartment** (X6, S).
- [ ] **A failure on groups is an error, not a change to agents** (N22, X9, S).
- [ ] **Language items, one or two lines each** (S). `@` with a second driver raises (L6). `>>` and `@` copy (L7). `ss.by` level counts are checked (L11). `ss.transmission` requires `beta=` (L12). `ss.Sim(*models, ...)` takes the rest by keyword only (L13). A typo in an arrow string raises (L15).
- [ ] **Put to Cliff with numbers:**
    - the chain-binomial step at a daily `dt` (X4, Decision 9);
    - exponential `dur=` against `rate=` as agents, and seeded people leaving a step late (X7, L14, Decision lang 17).

### Importers

- [ ] **The EMULSION importer** (N12, S). `interop/emulsion.py:225` divides `proba` by `delta_t`. The metapopulation level and `productions` go in `W1601`.
- [ ] **The other importers on real models** (N13, M).
    - odin: contact matrices, `lambda`, and `if`.
    - camdl: interventions that are off by default (`with_sia`).
    - Atomica: the 6 of 16 library workbooks that fail, and `sir_vaccine`.
    - summer2: named `Parameter`s.
    - flepiMoP: `subpop_setup.selected`, and the last day.

  Add each framework's real models to its interop test.
- [ ] **An incomplete import refuses to run** (S, Decision 6). It runs only when asked, with `allow_incomplete=True` printing the `W1601` list.
- [ ] **Starsim 3 `beta` dicts** (N7, S). `skills/import-model/references/starsim3.md:17` calls the mapping "Exact". Correct it, and note when a dict of bare numbers reaches a declared transmission with `dt` below one unit.

## Phase 4: the guessing rules

This phase needs Decision 1 (principle 3 restated: a guess fills in what is missing and never reinterprets what was given). Each item is small once the rule is chosen.

- [ ] **The principle** (P9b, S). Restate it in `design/` and `best-practices/`, with the guess table.
- [ ] **The unit rule** (L1, M, Decision lang 1). Reorder `Sim.init_unit` (`sim.py:1026-1074`), and add an error when typed values disagree and bare numbers are present. Delete the `assume_cal_year` path, so a calendar year is written `'2020'`.
- [ ] **Keyword numbers** (L3, S, Decision lang 6). In `Graph.collect` (`transitions.py:1151`), a keyword that names an existing parameter sets it; any other bare-number keyword raises `E602`.
- [ ] **The people anchor** (L4, S, Decision lang 13). In `resolve_people`, anchor on the transmission's states.
- [ ] **Default seed and the R0 line** (L5, S, Decision lang 2). Guess `init_prev=0.01`, marked as a guess. At `init()`, print one line with R0 and the number infected at the start.
- [ ] **`source=` for SEIR-shaped strings** (L8, S, Decision lang 11).
- [ ] **A dwell shorter than the step gets a note** (L9, S).
- [ ] **Initial values by size** (L17, S, Decision lang 3). Hash `init` and `init_prev` as one field.
- [ ] **Times outside the span raise** (D2, S, Decision dims 2). In `connectors.to_ti()`, a bare time is in the sim's unit.
- [ ] **A transfer-fed or boolean dimension starts at its first level** (D3, S, Decision dims 3).
- [ ] **A dimension's kind comes from its value** (D4, S, Decision dims 4). Delete the name list, and keep `kind=` as the override.
- [ ] **One default age distribution; named data brings its population** (D11, S, Decision dims 9).
- [ ] **Aging only with demographics** (D12a, S, Decision dims 10).
- [ ] **A real step unit** (L18, M). For a sim with no unit, in `Timeline`.
- [ ] **`E102`: dimensions checked through expressions** (L19, P4, M). In rate and `dur=` slots. This is also the camdl and summer2 feature in phase 7.
- [ ] **Inline parameter names** (L23, M, Decision lang 5). `<transition>.<slot>`, with a unique short alias.
- [ ] **Names without frames** (L20, L21, L36, M, Decision lang 4). `ss.states('S I R')`; structural module names; no lookup at `ss.Sim`. This also finishes N17 properly.

## Phase 5: one language

This needs Decisions 2, 8, and 11. It is the large breaking step, and it is what makes the docs writable. Do it after phase 2, whose contracts are the safety net for the deletions.

- [ ] **Declared `ss.SIR`, `SIS`, and `SEIR`** (X11, L31, L). Use the class form in F2 06's twelve lines. Keep the Starsim 3 keywords, and `infected` and `susceptible` as aliases for a release. `ss.Disease` becomes declared-only, and the code path lives only in `ss.Infection`.
- [ ] **A hidden `ss.v3` namespace** (P3, D31, R13, R27, L28, M, Decision docs 4). Use a module `__getattr__` that warns and names the v4 spelling. It holds:
    - the 18 intervention and product classes, and `ss.seasonality`;
    - `ss.Calibration`, `CalibComponent`, and the likelihood classes;
    - `Samples` and `Dataset`;
    - the time leftovers.

  Check HPVsim, STIsim, and the four packages that use `ss.Calibration` before moving each.
- [ ] **v4 verbs reach coded modules** (D23, N19, R13, L). This is the Starsim 3 seat's bridge:
    - `Selector` and `parse_target` accept any module;
    - a coded `rel_sus` or `rel_trans` is wrapped as `ss.ModifiableArr`;
    - conditions resolve coded arrays (`conditions.py:295-315`);
    - `calibrate()` walks `mod.pars`;
    - `sim.new()` works for any unrun sim (keep `inputs`, X30).

  Verbs that need structure (`equations()`, compartmental solvers) raise `E804`, naming the declared form.
- [ ] **One `ss.Result`** (R18, R2, R19, R20, L, Decision 11). A result has a kind and a run axis. A MultiSim is its results. A flow is named after its transition. `ss.parallel` is the one way to run many sims, and `sim.run(n_runs=)` goes.
- [ ] **Delete the Python solvers and `use_kernels`** (X24, M, Decision exec 3). This removes `solvers.py:2365-3137`. A Python function in a rate is evaluated into the program's tables each step.
- [ ] **The solver menu** (X17, X14, X21, M, Decision 8). `auto`, `ode`, `binomial`, `gillespie` (with tau as its noted fallback, sized by an ODE run), and `abm`. `integrator=` is a scheme only.
- [ ] **`auto`'s reasons** (X9, D26, X10, M, Decision exec 6). One capability record at `init()`, read by `auto`, `solvers()`, `describe()`, and `E804`. Probe step overrides on groups, with `compartmental =` as the override both ways. One function gives the reason a run cannot be taken whole.
- [ ] **`sim.people` in a compartmental run** (X12, M, Decision exec 10). It is `None`, and `ss.People.age < 15` gives a lazy condition.
- [ ] **One time index during a step, and one delivery slot** (X13, X22, M, Decision 12). Put these to Cliff with the shifted numbers first.
- [ ] **Errors and notes** (P6, P5, M, Decision 13). Exception classes named as today's codes; the catalog generated from them; `ss.note()` as the one channel; `verbose` as one integer.
- [ ] **One compartmental engine per sim** (X25, L, Decision exec 8). This replaces `Crossing`, and it unblocks whole runs, the particle filter, and JAX for multi-disease models. Do it last in this phase.

## Phase 6: docs

This overlaps with phase 5. The two short pages can be written now.

- [ ] **"v4 in ten minutes"** (P2a, S). Build it from the overview as it actually runs. It goes under `docs/`, so `test_docs.yml` runs it.
- [ ] **A v3-to-v4 migration page** (P2a, D22, M). Use F 12 §9's table of jobs and spellings, and add the traps the Starsim 3 seat found:
    - the `beta` dict (N7);
    - "routine" (N9);
    - the one-step shift (N10);
    - `step()` starting at `ti == 1` (D22, unless X13 lands);
    - `stop=2020` becoming `stop='2020'` (L1);
    - the `ss.v3` namespace;
    - migrating one disease at a time in a mixed sim.

  Correct `CHANGELOG.md:7` and line 79 of the import skill, which say Starsim 3 results line up (N10).
- [ ] **The tutorial and user-guide port** (P2b, L). Do it after phase 5, lifting spec chapters 00 to 09 (Decision docs 9).
- [ ] **"Coming from X" pages** (README §1, M). One per framework, each built from its `switching/` report: the side-by-side code, the agreement, the timings, and the switching demo. They are published with any failing rows, which the north-star script below produces.
- [ ] **The R path** (Decision 7, S). Document and test the arrow-string form with `name=` from R.
- [ ] **The north star restated** (P8, Decision 5, S). A script reads the vignettes and writes the table, failing rows included: lines as the hard target, and time as two numbers (first result and repeated runs) that fail at more than 2x and more than 1 s.
- [ ] **The spec matches the code** (P7, P11, R25, R30, R31, X23, D42, L32, L). Fold in the departures, strike what is promised but absent (or move it to `deferred.md`), and generate `15-errors.md` and the counts in `17-vocabulary.md` from the code, with a test that compares them (P6, L29). The spec and checklist edits are made in LFD too, while the two copies are kept in step (P15, Decision docs 16).

## Phase 7: what the seats need

These are in the order of [new-findings.md](new-findings.md), most seats first. None is a new concept; each attaches to something v4 already has, named in the last column.

| Done | Feature | Seats | Attaches to | Size |
|---|---|---|---|---|
| [ ] | A fitted value per place or group with a shared prior: `ss.par(ss.by(patch=...), bounds=)`; a par as a modifier's amount, inside `ss.seasonal(mean=)`, in `ss.perday(par/8)`, or in `capacity=`; typed bounds; `away=` fittable; a par used only by an intervention settable by `pars=` (N20) | flepiMoP, MEmilio, Ghana, Atomica | `ss.par`, `ss.by`, `Program` tables | L |
| [ ] | `ss.parallel(sim, priors=True)`, `fit.project(scenarios, n=)`, `to_df(quantiles=)`, `to_hub()` | flepiMoP, MEmilio, epydemix, Ghana, summer2 | `ss.parallel`, `Fit.samples`, R18's run axis | S each |
| [ ] | One-off moves as a delivery (`move='S -> V'`), `ss.Vaccine(leaky=False)` (D40), `at_age=` for routine doses (N8) | camdl, Ghana, LASER, EMULSION | `Delivery`, `Transition(move=)` (D33) | M |
| [ ] | ABC-SMC (as specified in `13-inference.md`), IF2 as a loop over `ParticleFilter`, binomial and Bernoulli observations | epydemix, SimInf, camdl, EpiModel | `calibrate()`'s method registry (R32), `ParticleFilter` | M each |
| [ ] | The JAX log-density and run exposed as functions, with reverse-mode gradients; `sim.gradient()` as a `jacfwd` (X26) | summer2, odin | `Density`, `gradients.py` | M |
| [ ] | A per-place condition on a result with a hold: `when=` on 7-day incidence by place, `hold=` | MEmilio, flepiMoP, SimInf | `ss.modify(when=)`, results by place | S to M |
| [ ] | `PartnerNet` with per-group degree, several `match=` properties, a mixing matrix, `acts=`, and duration in `net.check()` (N15, D39) | EpiModel | `ss.PartnerNet` | M (about 300 lines) |
| [ ] | Declared programs (spending to coverage to effect), `ss.optimize` over spending, and a databook generated from the model and read back | Atomica | `cost=`, `capacity=`, `calibrate()`'s optimizers | L (500 to 800 lines) |
| [ ] | Minimum stays (`after=`/`before=` on a transition), `offspring=` on a transition, inflows from one stratum into another (N11) | EMULSION | `Transition`, `ss.birth` | M each |
| [ ] | Declared life-table deaths, fixed-count seeding, gravity scaled to a mean share with a cap | LASER, Ghana | `ss.death`, `init=`, `ss.gravity` (D16) | S each |
| [ ] | An exact fixed delay (a conveyor, not 200 Erlang stages), and a declared auxiliary variable updated in the kernel | odin | dwell times, `Program` | M |
| [ ] | Dimensional checking through expressions at `init()` (L19's `E102`) | camdl, summer2 | typed values | M (phase 4) |
| [ ] | Data: US states (Mistry 2021), German level 2 with Destatis populations and commuters | epydemix, MEmilio, flepiMoP | `starsim_data`, `ss.people`, `ss.contacts`, `ss.geo` | M |
| [ ] | A thin R layer: `transition("S -> E", rate = ~ beta * I)`, results as data frames | EpiModel, SimInf, odin | R-Starsim, arrow strings | M |
| [ ] | The importers as a front door: `starsim-interop` (R22, R23, R34, Decision 6), plus a MEmilio graph importer, a summer2 live-model importer, and an odin array importer | all | `interop/core.py` | L |

## Phase 8: performance

These run at the scale the seats work at. Each target comes from a seat's measurement. Prototype the best case first for N25 and N26, to confirm the target can be reached before building it in. N23, N24, and N28 are bugs and come first.

| Done | Item | Where | Target |
|---|---|---|---|
| [ ] | Factored mixing within places plus commuting, in place of the dense (places × ages)² Kronecker product (N23); movers as `(dim, R)` contracted along one axis (D44) | `solvers.py:181-182`, `init_movers`/`mover_rates`/`step_movers` and the kernel | MEmilio's 400 counties × 6 ages within 3x of 1.5 s; memory linear in places |
| [ ] | Stratified results kept at a reporting interval by default, and `results=[...]` (N24, R17) | the solvers' `store`, `Result` | Ghana with ages and a campaign under 0.5 GB per run; 100 runs in memory |
| [ ] | Building a sim with many places: no deep copies at `ss.Sim()` (N28) | `sim.py:65`, `parameters.py:79` | 774 patches build in under 5 s |
| [ ] | Particles in parallel in the kernel, one stream per particle (N27; needs X3) | `ParticleFilter`, kernels | Within 3x of odin's 1.35 µs per particle-step on one thread, and scaling with threads |
| [ ] | Gillespie by place that steps only the herds an event touched, with threads across herds (N26) | the Gillespie kernel, the register | SimInf's 16,000 herds within 3x of 1.9 s; build within 2x of 2.1 s |
| [ ] | One fused pass per agent step instead of 10 to 15, then threads; int32 UIDs and int16 timers (N25) | `Agents`, `People` | 3.3M agents within 2 to 3x of LASER's 2.75 s on 20 threads; about 50 bytes per agent |
| [ ] | Declared diseases as agents no slower than Starsim 3 code (N29) | the agent engine | Parity with Starsim 3 on its own models |

## Phase 9: surface cuts, saved form, and code structure

Most of this is easier after phase 5 has deleted the second system. The merged files give the details.

- [ ] **The time vocabulary** (L28, S, Decision lang 9). `dur`, `per`, and `prob`, with `days` to `years` and `perday` to `peryear`.
- [ ] **The namespace in three tiers** (L29, P3, S). Enforced by a test.
- [ ] **Small spellings** (L30, L25, L34, L35, S, Decision lang 10, lang 15).
- [ ] **Movement, schedule, and product surface** (D32 to D39, M, Decisions dims 21 to 29). `ss.contacts` as a small matrix route. `ss.transfer` as a transition. `at=` everywhere. Products as real `ss.modify` objects. The code's base and effective rule. One `stratify` spelling. A pruned network catalogue.
- [ ] **Births and deaths** (D28, S, Decision dims 32). `ss.birth` and `ss.death` go in `demographics=`.
- [ ] **The saved form** (R21, R29, M, Decision 10). `to_code()` as `.py` plus a working pickle. `from_spec` deleted unless there is a non-Python consumer.
- [ ] **The `Fit` object, and `fit()` as a `Problem` plus a method registry** (R26, R32, M).
- [ ] **Results code in one place** (R33, M).
- [ ] **Structure** (X28, P13, L33, L37, X27, L). `sim.groups` (`ss.Groups`) comes off `Sim`. `solvers.py` becomes a package (`engine`, `mixing`, `compartmental`, `agents`). `transitions.py` is split along the spec. Calendar code moves out of `time.py`. These are pure moves once the deletions are done.
- [ ] **Dependencies** (P14, S, Decision docs 17). Declare `starsim_data`, drop `seaborn`, and add an `[interop]` extra.

## Acceptance: the thirteen switching demos

The plan is done when these pass in `tests/switching/` (phase 2). Each demo is the five-step script from [README.md](README.md#1-one-demo-thirteen-communities), written in that field's terms and sized to run in CI. Each passes against the reference numbers in its [switching/](switching/) report.

| Seat | The demo | Passes when | Blocked today by |
|---|---|---|---|
| [EpiModel](switching/epimodel.md) | The HIV network model by risk group, fitted, run from R | Prevalence by risk group near EpiModel's 21%/77% | Per-group degree (phase 7), N17, N16 |
| [SimInf](switching/siminf.md) | SIR in 16,000 herds with the movement register, fitted by PMCMC or ABC-SMC | Within 3x of SimInf's time, with agreement as today | N26, N27, ABC-SMC, N17 |
| [odin](switching/odin.md) | The stochastic SIR from R, fitted by PMCMC, then as agents | Distinct runs; the filter within 3x of dust per particle-step | N17, N5, N27, the odin importer (N13) |
| [camdl](switching/camdl.md) | The polio model with SIAs over `patch=`, fitted by IF2 | Peak 130,643 on day 145, as camdl's ODE; unit slips caught | N4, `move=`, IF2, `E102`, N28 |
| [LASER](switching/laser.md) | 3.3M agents across 149 places with life tables and seeding | Within 2 to 3x of LASER threaded, at about 50 bytes per agent | N25, life tables, seeding |
| [Ghana measles](switching/ghana_measles.md) | Fit, then routine MCV1 at 9 months plus a targeted campaign, 100 runs by district | Routine starts with newborns; 100 runs fit in memory | N24, N8 and `at_age=`, `leaky=False`, a fittable `away=` |
| [MEmilio](switching/memilio.md) | 400 counties × 6 ages with commuting and per-county NPIs, projected from priors | Within 3x of 1.5 s; agreement as today | N23, N18, the per-place hold, `priors=True`, the graph importer |
| [flepiMoP](switching/flepimop.md) | The 3,000-place config, fitted with a value per place, output for the hub | The per-place fit runs; Poisson on weekly averages is finite | N20, N6, N24, `priors=True`, `to_hub()` |
| [summer2](switching/summer2.md) | The TB model, imported live, fitted by NUTS with time-varying detection | NUTS on the JAX path with no divergences | N2, the summer2 importer (N13), the JAX log-density |
| [Atomica](switching/atomica.md) | A library workbook imported, calibrated, and optimized over spending | Agreement as today; an allocation from `ss.optimize` | The Atomica importer (N13), programs and `ss.optimize` |
| [EMULSION](switching/emulsion.md) | The published herd model with minimum stays, offspring, and births between stages | Matches EMULSION's published output | `after=`, `offspring=`, N11, N12 |
| [epydemix](switching/epydemix.md) | COVID in Kenya, seasonal × time-varying `beta`, fitted, projected, with an age-targeted vaccine | The filter peaks at the truth; `sir.age < 5` selects 6.9M | N1, N3, ABC-SMC, `fit.project()`, US states |
| [Starsim 3](switching/starsim3.md) | Their HIV and SIS model unchanged, then v4 verbs on it, then one disease redeclared | v4 verbs work on the coded model; the port matches within seed noise | D23, N19, N7, N9, N10, N29, the migration page |

## Order and dependencies

- **Phases 1 to 3** need no design decisions, apart from the marked items in phase 3. Phase 2 comes before phase 3, because the contracts find the bugs that the individual fixes miss.
- **Phase 4** needs Decision 1.
- **Phase 5** needs Decisions 2, 8, and 11, and runs after phase 2.
- **Phases 6 to 8** can overlap with phase 5. The two short doc pages and N23, N24, and N28 can start at once.
- **Phase 9** waits for phase 5.
