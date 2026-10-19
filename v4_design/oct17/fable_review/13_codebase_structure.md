# Codebase structure

I reviewed the shape of the repository rather than any module's semantics: what v4 added, what it kept, how the 44k lines of `starsim/` are divided, what the 12k lines of `tests/` cost to run, what a `pip install starsim` pulls in, and what the `import starsim as ss` namespace exposes. Files: all of `starsim/*.py` (38,243 lines in 36 files), `starsim/interop/` (5,739 lines in 19 files, including four `dump_*.py` scripts and one `.R` script), `starsim/library/` (2,035 lines), `tests/test_*.py` (11,934 lines, 41 files, 281 test functions), `pyproject.toml`, `starsim/__init__.py`, and the spec's `00-overview.md`, `17-vocabulary.md`, `19-open-questions.md`, plus `admin/implementation.md` and `admin/deferred.md`. I ran `git diff main...v4 --stat`, `git log main..v4`, an AST pass over every module for top-level definition sizes, a repo-wide reference count for every public name (over `starsim/`, `tests/`, `docs/`, `v4_design/examples/`, and the read-only vignettes), the full test suite once (`pytest test_*.py -n auto --durations=0`, 8 GB limit, 20-minute timeout), `test_samples.py` once alone, and two short scripts that exercise the Starsim 3 and v4 forms of each parallel class family against each other. Scripts and raw outputs are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/structure/` (`sizes_all.txt`, `usage.txt`, `neardead.txt`, `surface.txt`, `families.txt`, `test_run.log`).

## Summary

1. **v4 added 43,466 lines and removed 151.** Every Starsim 3 class that v4 supersedes is still there, exported, and documented, and the two halves do not interoperate: a v3 `ss.SIR` cannot take `ss.vaccinate`, `ss.modify`, `sim.calibrate()`, or `solver='ode'` (which it silently ignores), and a v4 `ss.Disease` cannot run with `ss.Births`, `ss.seasonality`, or `ss.routine_vx` under any compartmental solver. Decide per family (section below); my recommendation is to delete `seasonality`, `routine_vx`/`campaign_vx`/`Base*`/`routine_*`/`campaign_*`/`treat_num`/`simple_vx`, `Dx`/`Tx`, `Samples`/`Dataset`, and `Calibration`/`CalibComponent` and their five likelihoods, and to keep `Infection`/`SIR`, `Births`/`Deaths`/`Pregnancy`, and `MixingPool(s)` as library modules. Breaking for v3 users of those names; about 2,300 lines and 40 public names go.
2. **The `ss.` namespace has 285 public names (253 excluding submodules) against the 94 the spec counts.** A third are internals (`Engine`, `Crossing`, `CountState`, `StockCtx`, `module_map`, `find_contacts`, ...) or v3 duplicates. Move internals to their submodules and stop exporting them; the spec's own list is the target. Breaking only for code that reaches for internals via `ss.`.
3. **`sim.py` doubled (1,396 to 2,692 lines) by absorbing stratification geometry and solver choice.** About 420 lines of grid and group arithmetic (`init_strats`, `init_ragged`, `init_crossings`, `group_raw`, `fold*`, `grid_index`, ...) and 240 lines of solver selection (`init_solver`, `agent_reasons`, `try_functions`, `kernel_reason`, `solvers`) belong in `transitions.py`/a `dimensions.py` and `solvers.py`. Not breaking.
4. **`solvers.py` (4,136 lines, one `Compartmental` class of 1,570 lines and 70 methods) and `transitions.py` (2,972 lines, six unrelated families) should each become a package or be split by concern.** The seams are clean and already named in the code. Not breaking if the `ss.` exports stay.
5. **Make `optuna` optional and declare `starsim_data`.** `optuna` is a hard dependency imported lazily in one branch of one function; `starsim_data` is what `ss.contacts('Kenya')`, `ss.people('Kenya')`, and `ss.geo` need and appears nowhere in `pyproject.toml`. Not breaking.
6. **The test suite runs in 89 s on 20 cores but 952 s serially, and 870 of those seconds are v4 tests, five of which take 37 to 51 s each because they run every scenario with 50,000 agents under three solvers.** One failing test (`test_samples::test_seed_result`) is a race between four tests writing the same zip file under xdist. Cut the agent counts in `test_deliveries.py`/`test_dimensions.py` by 10x and give each `test_samples` test its own file.
7. **Strictly dead code is tiny (one 18-line function); near-dead code is the v3 families above, which only their own tests and the Starsim 3 docs reference.** Deleting by reference count alone would remove almost nothing; the gain is in the duplicate families, not in orphans.

## Simpler for the user

### 1. Two vocabularies for the same six things, and they do not mix

**Issue.** A user who opens the docs today sees `ss.SIR(beta=0.1)` with `ss.RandomNet`, `ss.Births`, `ss.routine_vx(product=ss.simple_vx(...))`, `ss.Calibration(calib_pars=..., components=[ss.BetaBinomial(...)])`, `ss.MixingPool(src=ss.AgeGroup(0, 15), ...)`, and `ss.seasonality(diseases='sir')`; a user who reads the spec sees `ss.Disease(S >> I @ beta*I)`, `ss.birth`, `ss.vaccinate(ss.Vaccine(...))`, `sim.calibrate()`, `ss.contacts('Kenya')`, and `ss.seasonal(...)`. Both sets are exported from `ss.` and both work, but only within their own half. I ran each form against the other (`families.py`; output in `families.txt`):

| What I tried | Result |
|---|---|
| v3 `ss.SIR()` with `solver='ode'` | runs silently as agents: `sim.pars.solver == 'ode'`, `sim.solver == 'abm'`, no warning (`v3ode.py`) |
| v4 `ss.Disease` + `ss.Births`/`ss.Deaths`, `solver='ode'` | `E804 CapabilityError: the module "births" (Births) only works on agents` |
| v4 `ss.Disease` + `ss.seasonality`, `solver='ode'` | `E804 ... "seasonality" only works on agents` |
| v4 `ss.Disease` + `ss.routine_vx(...)`, `solver='ode'` | `E804 ... "routine_vx" only works on agents` |
| v3 `ss.SIR()` + `ss.vaccinate(ss.Vaccine(...))` | `E506 UnknownDisease: "vaccinate" needs to know which disease it is for, and the sim has no declared diseases` |
| v3 `ss.SIR()` + `ss.modify('sir.pars.beta', multiply=0.5)` | `E504 UnmodifiableTarget: sir.pars.beta was not found. Quantities that can be modified: randomnet.pars.n_contacts` |
| v3 `ss.SIR()` then `sim.calibrate()` | `ValueError: sim.new() is only available for sims with declared states and transitions` |
| v4 `ss.Disease` + v3 `ss.RandomNet`, `ss.MixingPool` | works (networks are the one family that spans both) |

**Why it matters.** The spec's first property is "one vocabulary, several lengths". Today there are two vocabularies, and the second one is the one the 35 tutorial and user-guide pages teach (zero of them use `ss.transmission`/`ss.progression`/`ss.states`; five use `routine_vx`, four `ss.Births`, six `ss.Infection`, three `ss.MixingPool`, two `ss.Calibration`). The error messages tell a v3 user to "use `disease=`" or to "leave the solver as auto" rather than telling them the two halves do not meet.

**Proposal.** Per family, in the "Simpler codebase" section below: delete the ones with no users outside their own tests and the v3 docs, keep as library modules the ones that are genuine models written in the language (which is how `17-vocabulary.md:33` already describes `ss.SIR`, `ss.Births`, `ss.Deaths`, `ss.Pregnancy`). For what is kept, either make the pairs interoperate (a `Births` module exposes a `birth` transition to the graph; a v3 `Infection` exposes `rel_sus`/`rel_trans`/`beta` as modifiable quantities, which `E504` already lists for networks) or make the first error a user hits say "this is a Starsim 3 module; the v4 form is X".

**Cost.** Breaking for v3 scripts that use the deleted names; a migration page lists the replacements (most are one line). The docs have to be rewritten regardless, since they currently teach only the v3 forms.

**Confidence.** High on the facts; medium on the keep/delete line for `Infection`/`SIR` (see finding 6 under "Simpler codebase").

### 2. 285 public names where the spec promises 94

**Issue.** `len([n for n in dir(ss) if not n.startswith('_')])` is 285: 32 submodules, 179 classes, 65 functions, 9 other (`surface.txt`). `17-vocabulary.md:144` counts 94 `ss.` names and 133 total with methods, and says "60 of the 93 `ss.` names are ones a Starsim user already knows". The gap is made of:

- **Solver internals** exported from `__init__.py:149`: `solver_names, make_engine, Mixing, Engine, Compartmental, Agents, CountState, Crossing`. Nothing outside `starsim/` references `Engine`, `CountState`, `Mixing`, `make_engine`; `Crossing` is used once in `test_gradients.py`.
- **Module plumbing**: `module_map, module_types, register_modules, find_modules, required, Base` (`__init__.py:103`); `link_dists, make_dist, dist_list, scale_types, Dists, rand_raw, multi_random` (`__init__.py:88`); `ndict, find_contacts, standardize_netkey, parse_age_range, apply_age_range, standardize_data, validate_sim_data, plot_args, return_fig` (`__init__.py:65`).
- **Time**: 15 one-line classes (`months`, `weeks`, `days`, `perday`, `probperweek`, `freqpermonth`, ...; `time.py:2279-2305`), plus three deprecated v2 shims `rate`, `time_prob`, `rate_prob` (`time.py:2401-2416`) that still warn about "Starsim 2.0".
- **Twelve debugging and mocking helpers** (`Profile, Debugger, Diagnostics, check_requires, check_version, metadata, mock_time, mock_sim, mock_people, mock_module`).
- **The v3 duplicates** in finding 1 (about 40 names).

Grouped (my judgment, from the full list in `surface.txt`):

| Group | Count | Examples |
|---|---|---|
| A model author needs | ~95 | `Sim, Disease, State, states, Transition, transmission, progression, birth, death, par, outcome, by, stratify, contacts, MixingPool, RandomNet, MFNet, modify, change, vaccinate, screen, treat, Vaccine, Test, Treatment, parallel, MultiSim, date, days, weeks, years, perday, peryear, prob, dur, the 20 distributions, People, data, fit, compare_models, load, save, options` |
| A module author needs | ~60 | `Module, Intervention, Analyzer, Connector, Network, DynamicNetwork, Route, Arr, FloatArr, BoolArr, BoolState, IntArr, IndexArr, uids, Dist, Pars, Result, Results, Timeline, Loop, Cond, Expr, Graph, Transfer, Strat, Requires, warn, error, note, CodedError, infection_log, mock_*` |
| Internal or superseded | ~95 | the solver internals, module plumbing, `link_dists`, `dist_list`, the fifteen one-line unit classes, the v2 shims, `Samples, Dataset, MultiResult, MultiResults, Calibration, CalibComponent, BetaBinomial, Binomial, DirichletMultinomial, GammaPoisson, Normal, ParticleFilter, ModelComparison, linear_interp, linear_accum, step_containing, loglik, mismatch, compare`, `routine_vx` and its eleven siblings, `Dx, Tx, Vx, simple_vx, seasonality, Person, people, AgeGroup, demo, diff_sims, check_sims_match, single_run, multi_run, module_map...` |

**Proposal.** Export what the spec counts, plus the module-author set. Everything else stays reachable as `ss.solvers.Engine`, `ss.utils.find_contacts`, `ss.time.perday` (the sixteen unit classes could become a single `ss.per('day')`, but that is time's reviewer's call). Delete the three v2 shims and `warn_deprecation` (`time.py:2384-2416`, 33 lines): v4 is the moment.

**Cost.** Breaking for code that reached internals through `ss.`; a grep of the vignettes, examples, and library shows none does.

**Confidence.** High.

### 3. The docs teach the superseded half

**Issue.** Of the 35 `.qmd` pages in `docs/tutorials/` and `docs/user_guide/`, none uses a v4 constructor. `git diff main...v4 --stat -- docs` shows 14,709 insertions, but those are `v4_design/`'s neighbors (migration notes, examples), not the tutorials. A reader of docs.starsim.org after v4 ships learns `ss.routine_vx`, `ss.Calibration`, `ss.MixingPool(src=ss.AgeGroup(...))`, and `ss.Samples`.

**Proposal.** Not to rewrite them in this review (the brief says so), but the decision in finding 1 should be made before the rewrite, since every deleted family is one fewer page.

**Confidence.** High.

## Simpler codebase

### 1. What v4 added, changed, and did not remove

`git log main..v4 --oneline | wc -l` is 90 commits. `git diff main...v4 --stat` is 164 files, 43,466 insertions, 151 deletions. Restricted to `starsim/` and `tests/`: 71 files, 27,422 insertions, 150 deletions. The deletions are all incidental edits inside files that grew (`sim.py` −14 of +1,310; `networks.py` −15 of +564; `distributions.py` −27 of +188). No file, class, or function was removed.

New modules: `solvers.py` (4,136), `transitions.py` (2,972), `kernels.py` (1,372), `gradients.py` (1,121), `spec.py` (782), `conditions.py` (446), `named.py` (184), and `interop/` (5,739 across 19 files). Grown: `sim.py` +1,310, `calibration.py` +1,420 (to 2,587), `interventions.py` +741 (to 1,266), `networks.py` +564, `run.py` +343, `products.py` +330, `connectors.py` +293, `diseases.py` +275, `distributions.py` +188.

Tests: 10 new v4 test files totaling 4,100 lines (`test_transitions` 1,018, `test_deliveries` 757, `test_dimensions` 526, `test_kernels` 252, `test_gradients` 226, `test_named` 124, nine `test_interop*` 1,467). The v3 test files are untouched apart from `test_networks` (+85), `test_results` (+56), `test_calibration` (+37), `test_distributions` (+16), `test_time` (+11), and `baseline.yaml`. That is the right shape for a regression suite; it also means every v3 class is still tested and so cannot be removed without touching tests.

### 2. The superseded families, one by one

Reference counts are from `usage.txt` (repo-wide word matches, excluding the defining file; `docs` is `docs/**/*.qmd` and `.py`; `vig` is the 22 vignette models). "Can v4 express it" is from reading both forms and from `families.txt`.

| v3 form (lines) | v4 form (lines) | v3 users: tests / library / docs / vignettes | Can v4 express all of v3? | Recommendation |
|---|---|---|---|---|
| `seasonality` connector (`connectors.py:24`, 83) | `seasonal` time value (`transitions.py:2369`, 59) | 0 / 0 / 0 / 2 (both vignettes use `ss.seasonal`; `seasonality` appears only in comments) | Yes: `seasonality` is a cosine on `rel_trans` with `scale` and `shift`; `ss.seasonal(mean, amplitude, peak)` on `beta` is the same curve, and works under every solver (the connector is agents-only, `families.txt`) | **Delete.** 83 lines, one name. |
| `RoutineDelivery`, `CampaignDelivery`, `BaseTest`, `BaseScreening`, `BaseTriage`, `routine_screening`, `campaign_screening`, `routine_triage`, `campaign_triage`, `BaseTreatment`, `treat_num`, `BaseVaccination`, `routine_vx`, `campaign_vx` (`interventions.py:77-527`, 450 lines, 14 classes) | `Delivery` + `vaccinate`/`treat`/`screen` (`interventions.py:535-1266`, 731 lines) | `routine_vx` 8 / 0 / 12 / 0; `routine_screening` 8 / 0 / 6 / 0; `campaign_*` 6 / 0 / 3 / 0; `treat_num` 2 / 0 / 4 / 0; the `Base*` classes 0 / 0 / 1 / 0 | Yes: `Delivery` has `years`/`dates` (campaign), `start`/`stop`/`interval`/`coverage` (routine), `capacity` (`treat_num`), `on_positive`/`on_negative` (triage), `eligible`, `trace`. The one thing it lacks is `prob` as an array over years, which `coverage=ss.timeseries(...)` covers (unverified for the exact `prob_array` test) | **Delete** all 14. 450 lines, 14 names. `tests/test_interventions.py` (402 lines) becomes a port of its ten tests onto `vaccinate`/`screen`, half of which `test_deliveries.py` already has. |
| `Dx`, `Tx`, `Vx`, `simple_vx` (`products.py:31-227`, 187 lines) | `Held`, `Vaccine`, `Treatment`, `Test` (`products.py:229-533`) | `Dx` 4 / 0 / 5 / 0; `Tx` 3 / 0 / 4 / 0; `Vx` 3 / 0 / 4 / 0; `simple_vx` 5 / 0 / 6 / 0 | `Vx()` already returns a `Vaccine` (`products.py:173-176`); `Dx` is a result-probability table that `ss.Test(sensitivity, specificity, detects)` covers for the two-outcome case (a three-outcome `Dx` table is not expressible: `08-interventions.md:53` says the short names remain as aliases) | **Delete** `simple_vx`, `Tx`; keep `Vx`/`Dx` as one-line aliases only if the alias promise in the spec is kept, otherwise delete. |
| `Calibration`, `CalibComponent`, `BetaBinomial`, `Binomial`, `DirichletMultinomial`, `GammaPoisson`, `Normal`, `linear_interp`, `linear_accum`, `step_containing` (`calibration.py:26-1172`, 1,147 lines) | `fit`, `Fit`, `data`, `ParticleFilter`, `pmmh`, `nuts`, `nuts_jax`, `ModelComparison`, `compare_models` (`calibration.py:1174-2587`, 1,413 lines) | `Calibration` 9 / 0 / 13 / 0 (one test, two doc pages); the five likelihoods 2 each / 0 / 1-4 / 0 | Mostly: `fit(method='optuna')` gives the Optuna best fit; `likelihood=dict(series=ss.normal(...))` or `ss.outcome(likelihood=...)` gives per-series likelihoods. `fit`'s Optuna branch (`calibration.py:2247-2256`) does **not** reuse `Calibration`; it is a second, 10-line Optuna loop. What `fit` lacks: `build_fn` (an arbitrary function from parameters to a sim; v4 fits only `ss.par(bounds=)`), the Optuna database/`n_workers`/`continue_db` machinery, `prune_fn`, and the beta-binomial/Dirichlet-multinomial/gamma-Poisson conjugate likelihoods (`fit` has normal and whatever a distribution's `logpdf` gives) | **Delete** `Calibration` and the component family, and fold the one real gap (a `build=` callable for agent-based sims whose parameters are not `ss.par`) into `fit`. The spec already says (`17-vocabulary.md:110`) that `sim.calibrate()` returns "the calibration object that this specification calls `fit`", i.e. one class, not two. Saves about 1,100 lines, 10 names, and the lazy `seaborn` import (`calibration.py:18`), which is the only use of `seaborn` in the package. |
| `MixingPool`, `MixingPools`, `AgeGroup` (`networks.py:1842-2159`, 293 lines) | `contacts` (`networks.py:2161`, 75 lines; returns `MixingPools`) | `MixingPool` 16 / 0 / 9 / 0; `MixingPools` 4 / 0 / 7 / 0; `AgeGroup` 6 / 0 / 5 / 0; `contacts` 51 / 6 / 32 / 20 | `contacts` is built on `MixingPools`, so this is not a parallel family but a loader over the class. `AgeGroup`/`LevelGroup` are the group selectors `MixingPools` uses | **Keep**; the spec lists `ss.MixingPool` as the homogeneous-mixing class (`06-routes.md:178`). Consider folding `AgeGroup` into `Cond` (`conditions.py`) since `ppl.age < 15` is the v4 spelling of `ss.AgeGroup(0, 15)` (`LevelGroup` already is a condition on strata). |
| `Births`, `Deaths` (`demographics.py:29-328`, 297 lines); `Pregnancy` (796) | `birth`, `death`, `immigration`, `emigration` transitions (`transitions.py:844-916`, 66 lines) | `Births` 28 / 2 / 19 / 1; `Deaths` 18 / 7 / 11 / 4; `Pregnancy` 22 / 28 / 20 / 0 | Partly. `ss.birth(rate=...)` is a crude rate into one state; `ss.Births` takes a year-indexed dataframe of crude birth rates (`birth_rate` as data, `rate_units`, `metadata`), and `ss.Deaths` an age- and sex-specific table. `ss.timeseries` covers the time-varying scalar, but there is no v4 spelling of a death rate by age and sex except `ss.death(rate=ss.by(age=[...]))` on a stratified model, and no v4 pregnancy at all | **Keep**, as `03-states-and-transitions.md:360` says ("for anything beyond the simplest case a `ss.Demographics` module keeps them in one place"). But they are agents-only (`E804` above), so a declared model with real demography cannot run as an ODE; giving `Births`/`Deaths` a `compartmental = True` form that contributes a `birth`/`death` transition to the graph would remove the wall. |
| `Infection`, `SIR`, `SIS`, `SEIR`, `NCD` (`diseases.py:390-1353`, ~800 lines) | `Disease` with transitions (`diseases.py:18` + `transitions.py`) | `SIR` 79 / 12 / 124 / 13; `SIS` 54 / 0 / 124 / 2; `Infection` 7 / 4 / 25 / 0; `NCD` 3 / 0 / 0 / 0 | `SIR`/`SIS`/`SEIR` are two-line declared models; `Infection` is the base for the library's HIV, cholera, Ebola, measles, and the MNCH modules (2,035 lines, none of which uses a v4 form) and for every external Starsim 3 model | **Keep** `Infection` and `Disease` (the base classes); **rewrite** `SIR`/`SIS`/`SEIR` as declared models (`class SIR(ss.Disease): S, I, R = ss.states(3); ...`), so the three names the docs and 124 test and doc references use become v4 models, and so `solver='ode'` works for them. `NCD` (84 lines, 3 test references, no docs) can go. |
| `Samples`, `Dataset` (`samples.py`, 432 lines) | `MultiSim`/`MultiResults` (`run.py:931-1093`) | `Samples` 11 / 0 / 43 / 0 (one test file, one doc page) | `Samples` is a zip-of-CSVs store for thousands of runs' summaries with identifier-keyed lookup; `MultiResults` holds results in memory with a run dimension. They overlap for anything that fits in memory | **Delete**, or move to a separate package: nothing in `starsim/` imports it (core count 1 = `__init__.py`), and the one test is the one that fails under xdist (Bugs below). |
| `MultiResult` vs `MultiResults` (`run.py:931`, 123 lines; `run.py:1056`, 38 lines) | | 0 / 0 / 0 / 0 outside `run.py` | These are one result across runs and the collection of them; the names differ by one letter | **Rename** `MultiResult` to something that is not a typo away from its container, or make it `Result` with a run axis (the results reviewer's call). |

Totals if the "delete" rows are taken: about 2,300 lines of `starsim/`, about 600 lines of tests (`test_interventions.py` ported, `test_samples.py` gone), 40 public names, and the `seaborn` import.

### 3. `sim.py`: 1,396 to 2,692 lines, 38 new methods on one class

`Sim` is 2,442 lines and 85 methods (`sim.py:16`). The 38 methods new in v4 (`comm` against `main:starsim/sim.py`) fall into three groups that have nothing to do with orchestrating a run:

- **Stratification geometry** (about 420 lines): `stratify` (L535), `init_strats` (L547, 105 lines), `init_ragged` (L653, 40), `fold`/`fold_matrix`/`group_vals`/`grid_index` (L694-731), `init_crossings` (L733, 57), `get_declared`/`dim_index`/`groups_changed` (L791-820), `group_raw` (L822, 76), `folded`/`level_raw` (L899-906), `init_strat_vals` (L908, 29), `init_named_people` (L938, 24). This is the arithmetic of a grid of groups; it is used by `solvers.py` and `gradients.py` through `sim.`, and it has a natural home next to `Strat` (`transitions.py:2712`) and `Mobility` (`transitions.py:2554`).
- **Solver choice** (about 240 lines): `init_solver` (L292, 109), `agent_reasons` (L963, 32), `try_functions` (L996, 29), `solvers` (L1165, 43), `kernel_reason`/`kernel_only` (L1866-1893). `solvers.py` already owns `solver_names` and `make_engine`; the rules for choosing belong beside the engines they choose between.
- **Model description** (about 300 lines): `describe` (L1076, 88), `requires` (L1209, 48), `R0` (L1258), `hash` (L1272), `to_spec` (L1290, 41), `to_code` (L1332), `what_modifies` (L493, 41), `check_dt` (L445, 47), `init_unit` (L1026, 49). `spec.py` already has `to_code`, `from_spec`, and `diff`; `describe` on a `Sim` is a wrapper over `Graph.describe` (`transitions.py:1501`).

**Proposal.** A `Dimensions` (or `Grid`) object owned by the sim (`sim.dims`) in a new `dimensions.py` that also takes `Strat`, `Mobility`, `Transfer`, `geo`, and `gravity` from `transitions.py`; solver selection as a function in `solvers.py` (`choose_solver(sim) -> (name, reason)`); and the spec/describe methods as thin delegates. `Sim` returns to roughly its Starsim 3 size with a dozen v4 methods.

**Cost.** None to users if the method names on `Sim` are kept as one-line delegates. Internal call sites in `solvers.py`, `gradients.py`, `kernels.py` (which reach `sim.group_raw`, `sim.grid_index`, etc.) change to `sim.dims.*`.

**Confidence.** High that the split is clean (the methods are already grouped in the file); medium on the exact boundaries.

### 4. `solvers.py` (4,136 lines) is three files, and `Compartmental` is two classes

Top level (`sizes_big5.txt`): `Mixing` (L46, 265 lines), `CountState` (L313, 36), `Crossing` (L351, 183), `Engine` (L556, 966 lines, 56 methods), `StockCtx` (L1524, 39), `Compartmental(Engine)` (L1567, 1,570 lines, 70 methods), `Agents(Engine)` (L3141, 996 lines, 47 methods).

Inside `Compartmental`, the methods separate into (a) building the program the kernels run (`init_movers` 55, `init_register` 50, `build_layout` 84, `init_program` 160, `pack_targets` 29, `refresh_program` 18, `tables` 69, `run_args` 12, `init_shifts` 16, `run_all` 58, `run_many` 29, `collect` 45, `derived` 25, `derived_outcomes` 49: about 700 lines, the kernel interface), and (b) stepping from Python (`step` 39, `step_ode` 41, `step_binomial` 9 + `step_binomial_once` 51, `step_tau` 26, `step_gillespie` 32, `step_sde` 9 + `step_sde_once` 21, `derivs` 35, `next_generation` 45, `draw_targets` 20: about 330 lines, one method per solver name). The Python steppers duplicate what `kernels.py:705` (`step`, 270 lines) does in Numba: `implementation.md` ("Known issues") says the kernels draw from Numba's generator "so a stochastic compartmental run gives different numbers with `engine.use_kernels = False`". Two implementations of five solvers is the largest duplication in the codebase.

**Proposal.** (1) Make `starsim/solvers/` a package: `engine.py` (`Engine`, `CountState`, `StockCtx`), `mixing.py` (`Mixing`, `Crossing`), `compartmental.py`, `agents.py`, with `kernels.py` beside them. (2) Decide whether the Python steppers are a reference implementation (then move them to `tests/` or a `reference.py` that the conformance tests import) or the fallback for when Numba is unavailable (then say so and test equality to the kernels within tolerance). Either way they should not be a second code path that gives different random numbers under the same seed.

**Cost.** None to users (`ss.Compartmental` etc. are internals; see "Simpler for the user" 2). Removing the Python steppers makes `kernels.py` load-bearing for every compartmental run, which it already is by default.

**Confidence.** High on the split; medium on removing the steppers (I did not check whether `scipy`'s `solve_ivp` path at `solvers.py:2940` is user-facing via `integrator='scipy'`; `test_kernels.py::test_scipy` suggests it is).

### 5. `transitions.py` (2,972 lines) holds six unrelated families

By line range: errors and notes (L44-75); the expression algebra `Expr`/`par`/`Count`/`Flow`/`Func`/`BinOp`/`linear`/`evaluate` (L84-427, ~340 lines); `State`/`StateSet`/`states`/`Null` (L428-608); `Transition` and its six constructors plus `outcome`/`FlowSum`/dwell helpers (L610-1098, ~490 lines); `Graph` (L1099-1925, 824 lines: the compiler from a `Disease`'s members to a resolved model, including `describe`, `diagram`, `equations`, `to_json`, `stratify`); formatting helpers (L1927-2043); `by` (L2045, 170 lines); the time values `TimeValue`/`timeseries`/`seasonal` (L2229-2432); `geo`/`gravity`/`Mobility` (L2434-2710, ~280 lines); `Strat`/`Transfer`/`transfer`/`stratify` (L2712-2972, ~260 lines).

**Proposal.** `expr.py` (algebra, `par`, `by`, time values), `transitions.py` (states, transitions, outcomes), `graph.py` (`Graph`, `TransitionTable`, formatting), `dimensions.py` (`Strat`, `Transfer`, `Mobility`, `geo`, `gravity`, plus the grid arithmetic from `Sim`). `__init__.py` keeps every `ss.` name.

**Cost.** None to users. Internal imports change.

**Confidence.** High.

### 6. `calibration.py` (2,587 lines): two systems, one shared word

Covered in the family table above. Beyond deletion, `fit` itself is a 304-line function (`calibration.py:2095-2398`) that dispatches on seven method strings (`nuts`, `l-bfgs-b`, `nelder-mead`, `asd`, any SciPy method, `pmcmc`, `optuna`) with the encode/decode of bounds, the Latin-hypercube `scan`, and the posterior summary inline. A `Fitter` base with one subclass per method (or a dict of method functions with a shared signature) would make `default_method` (L2072), the `scan`, and the summary one place each. `pmmh` (L1632), `nuts` (L1709), and `nuts_jax` (L1781) are already that shape.

**Confidence.** Medium (the calibration reviewer owns the semantics).

### 7. `time.py` (2,612 lines): a plotting subsystem and sixteen one-line classes

`date(pd.Timestamp)` is 430 lines (L265), `datedur` 321, `Rate` 283, `TimePar` 267, `dur` 244, `prob` 226, `DateArray` 199. Then: fifteen one-line unit classes plus `years` (L2260-2305), the v2 deprecation shims (L2384-2416, 33 lines), and a matplotlib converter subsystem (`MockAxis`, `FloatYearLocator`, `FloatYearFormatter`, `DateConverter`, L2418-2612, 195 lines) that registers itself at import (`time.py:2610-2612`). `approx_compare` (L43, 18 lines) is referenced nowhere in the repo, docs, or vignettes (`dead.txt`): the only strictly dead top-level definition I found.

**Proposal.** Move the matplotlib plumbing to `settings.py` (which owns `style`) or a `plotting.py`; delete `approx_compare` and the v2 shims; let time's reviewer decide on the fifteen classes.

**Confidence.** High.

### 8. Dead and near-dead code

Strictly unreferenced (name appears nowhere else in `starsim/`, `tests/`, `docs/`, `v4_design/`, or the vignettes; `dead.txt`): one top-level function, `time.py:43 approx_compare` (18 lines), and 19 methods totaling 142 lines, the largest being `Sim.to_yaml` (`sim.py:2328`, 21), `DateConverter.axisinfo`/`default_units` and `FloatYearLocator.set_axis` (matplotlib protocol methods, so not really dead), `RandomSafeNet.plot_matrix` (`networks.py:1287`, 9), `Arr._to_values`/`_to_raw` (`arrays.py:791-809`, 18), `Module.alias_state_dict` (`modules.py:836`, 8), `People.scale_flows` (`people.py:457`, 6), `Sim.reset_time_pars` (`sim.py:1622`, 5), `Pars.dict_update` (`parameters.py:255`), `Pregnancy.nulliparous`/`tri1_uids`/`tri3_uids`/`dur_gestation_at_birth` (`demographics.py:506-534`), `Results.is_msim` (`results.py:654`).

Public top-level definitions referenced only from their own file, tests, or docs (not from other core modules, the library, the examples, or the vignettes; `neardead.txt`): 118 definitions, 2,664 lines. Almost all are legitimately private helpers that happen to lack a leading underscore (`RParser`/`REval` in `siminf.py`, `disease_code` in `spec.py`, `place_rates`/`settle`/`diffuse` in `kernels.py`, `pmmh`/`nuts_jax`/`product_pairs` in `calibration.py`). The point is not that they are dead but that the module boundary is invisible: `interop/siminf.py` has 727 lines of R-parsing (`RParser` 144, `REval` 122, `read_script` 44) with no underscore anywhere, and `__all__` is used in only `interop/export.py` and `interop/__init__.py`. A convention (underscore, or `__all__` per module) would let a reader and a tool tell the 94 names from the 500.

**Confidence.** High on the counts; the lists are mechanical.

### 9. Dependency surface

`pyproject.toml` hard dependencies: `numpy, pandas, sciris, numba, scipy, networkx, matplotlib, seaborn, optuna`. Observed:

| Package | Imported at module level | Imported lazily | Used by |
|---|---|---|---|
| `numba` | `arrays, debugtools, distributions, networks, settings, diseases, utils, people, kernels` | | 11 `@nb.njit` functions in the v3 core; 45 jitted functions in `kernels.py` (`jit = nb.jit(nopython=True, cache=True, error_model='numpy')`, `kernels.py:116`), which have left 487 cache entries in `starsim/__pycache__/` (45 functions, many signatures). `implementation.md` puts the first compile at about 12 s |
| `scipy` | `distributions, networks, calibration, interop/atomica` | `solvers.py:2940` (`solve_ivp`), `transitions.py:2337` | core |
| `networkx` | | `networks.py:351-598`, `diseases.py:971` | networks, `infection_log` |
| `optuna` | | `calibration.py:2248` (one branch of `fit`); the v3 `Calibration` class via `op` | calibration only |
| `seaborn` | | `calibration.py:18` (`sc.importbyname(..., lazy=True)`) | two plots in the v3 `CalibComponent` family |
| `jax`, `blackjax` | | `gradients.py:43`, `calibration.py:1730,1797` | `[jax]` extra; `sim.gradient()`, `method='nuts'`, `integrator='jax'` |
| `starsim_data` | | `named.py:22` | `ss.contacts('Kenya')`, `ss.people('Kenya')`, `ss.geo`; **not in `pyproject.toml` at all**, even as an extra |
| `openpyxl`, `libsbml` | | `interop/` | importers; `openpyxl` is in `[lock]` only |

So a user installing `starsim` gets and compiles: Numba (the big one, unavoidable for the kernels), SciPy, NetworkX, Matplotlib, pandas, plus Optuna (with SQLAlchemy, alembic, colorlog, and a database layer, visible in `[lock]`) and seaborn, neither of which the core ever imports. Import time with a warm cache is 1.27 s (`surface.txt`); the Numba cache is written into the package's own `__pycache__`, which fails silently on a read-only site-packages and recompiles every process (unverified here; it is Numba's documented behavior, and nothing in the package sets `NUMBA_CACHE_DIR`, although `gradients.py:47-48` does set JAX's equivalent).

**Proposal.** Move `optuna` and `seaborn` to extras (`[calibration]` or drop them with the v3 `Calibration` family); add `starsim_data` as an extra (`[data]`) and make the `ImportError` message in `named.py` say `pip install starsim[data]`; list `openpyxl` under an `[interop]` extra. Consider shipping the Numba cache or warming it at install, since a 12 s first run is the first thing a new user sees.

**Confidence.** High on the facts; medium on shipping the cache (Numba caches are keyed to the exact Numba/LLVM version).

### 10. Tests: 952 serial seconds, 870 of them v4

One run of `pytest test_*.py -n auto --durations=0` on 20 cores: **88.9 s wall, 282 passed, 1 failed, 4 skipped, 87 warnings**; peak RSS of the controlling process 1.1 GB (workers not measured). Summed durations: 952 s over 260 timed entries. By file (`test_run.log`): `test_deliveries.py` 278 s, `test_gradients.py` 117, `test_dimensions.py` 116, `test_calibration.py` 72, the nine `test_interop_*` 215, `test_kernels.py` 39, `test_transitions.py` 30; every v3 file is under 12 s (`test_networks` 11.6, `test_demographics` 10.5, `test_diseases` 10.0).

The slowest single tests: `test_calibration::test_particle_mcmc` 68 s; `test_dimensions::test_carriage` 51 s and `test_states` 47 s; `test_deliveries::test_change` 46 s, `test_transfer` 46 s, `test_coinfection_compartments` 45 s, `test_treat_any_transition` 44 s, `test_short_names` 37 s; `test_gradients::test_posterior_data` 41 s, `test_jax_run` 39 s; four interop tests 38 to 40 s. The deliveries and dimensions tests are slow because `kw = dict(verbose=0, unit='day', total_pop=50_000)` (`test_deliveries.py:12`) and `run_each` runs every scenario under `['ode', 'binomial', 'abm']` (`test_deliveries.py:13,37`): each test is five to eight 150-day agent-based sims of 50,000 agents. `total_pop=5_000` would make them 10x faster with the same assertions (the tolerances are `rtol=0.1`, `check_close` at `test_deliveries.py:42`).

Duplication between v3 and v4 tests of the same behavior is modest because the two halves test different classes: `test_interventions.py` (10 tests, 402 lines) tests `routine_vx`/`simple_vx`/`Dx`/`RoutineDelivery`, and `test_deliveries.py::test_vaccinate` tests the same four behaviors (campaign, routine, waning, capacity) on `vaccinate`/`Vaccine`; `test_mixingpools.py` (8 tests) and `test_dimensions.py::test_mixing` both test age-structured mixing; `test_diseases.py::test_sir/test_sis/test_seir` and `test_transitions.py::test_forms/test_solvers` both test SIR dynamics. Each pair collapses when the v3 family does (section 2).

Test shape: 281 `def test_` across 41 files, two files use `parametrize`, and most tests are 30-to-100-line scripts with several assertions each (the Starsim convention), so pytest's failure report names the test but not the scenario. `pytest.ini` has `env = SCIRIS_BACKEND=agg`; `run_tests` also sets `STARSIM_SHOW=0`. Apart from the race below, the suite is clean: four skips (the skip conditions in the suite are camdl tests that need the `camdlc`/`pycamdl` tools, interop tests that need the `lingua_franca_design` examples, the timing-sensitive performance guard under xdist, and JAX/BlackJAX availability; `test_interop_camdl.py:114,130`, `test_interop_flepimop.py:17`, `test_baselines.py:229`, `test_gradients.py:15`) and warnings that are `os.fork()`-after-JAX notices from xdist workers, which `implementation.md` already records as a known deadlock risk.

**Confidence.** High.

### 11. `interop/`: 5.7k lines, five generator scripts in the wheel

`starsim/interop/` is 19 files: eleven importers (227 to 727 lines each), `core.py` (464), `export.py` (411), `__init__.py` (107), and five `dump_*` scripts (`dump_epimodel.R` 172, `dump_epydemix.py` 174, `dump_memilio.py` 199, `dump_pycamdl.py` 132, `dump_summer2.py` 149 = 826 lines) whose docstrings say they "need neither Starsim nor Sciris, so [they] can be copied into any ... environment". They are shipped inside the package (the `.R` file by the `[tool.setuptools.package-data]` glob only if its extension is listed; `*.R` is not, so it is probably **not** in the wheel while the four `.py` are: an inconsistency either way). All eleven importers are imported eagerly by `interop/__init__.py:11-22` at `import starsim`.

**Proposal.** Move `dump_*` to `v4_design/` or a `tools/` folder outside the package (they are documented in `16-interop.md` by path, so one path change); import each importer lazily from the `formats` dict (`interop/__init__.py:25`) so that `import starsim` does not parse 4,800 lines of importer code it will almost never use.

**Confidence.** High on the layout; medium on the import-time saving (I did not time the interop import separately).

## Spec and implementation

- **Vocabulary count.** `17-vocabulary.md:144-146` counts 94 `ss.` names; the implementation exports 253 non-module names. The spec's rule "when a name is added or removed, update this document and its count" (L262) has not been applied to the Starsim 3 names that stayed; the document says (L33) library modules "are not counted here", but the interventions, products, calibration, samples, and solver internals are not library modules.
- **`sir.solvers()`.** `00-overview.md` ("`sir.solvers()` shows which solvers can run a model before anything runs") and `01-objects.md:41` put `solvers()` on the model; the implementation has it only on `Sim` (`sim.py:1165`); `ss.Disease(...).solvers()` raises `AttributeError` (`families.txt`). Either is fine; the spec and the code should agree.
- **`solver=` on an undeclared sim.** `implementation.md` ("Departures") says "sims with no declared states keep Starsim 3's time defaults" and `sim.py:296` says such a sim "is run as agents, as in earlier"; neither says that an explicit `solver='ode'` is accepted and ignored (see Bugs).
- **`fit` versus `Calibration`.** `17-vocabulary.md:110` lists `ss.Calibration` among "types you get back" and says `sim.calibrate()` returns "the calibration object that this specification calls `fit`": one object. The implementation has `ss.Fit` (returned by `fit`) and `ss.Calibration` (Starsim 3's Optuna class) as unrelated classes, and `fit(method='optuna')` reimplements Optuna rather than using `Calibration`.
- **Aliases.** `08-interventions.md:53` promises `ss.Vx`, `ss.Dx`, `ss.Tx` "remain as aliases". `Vx()` is an alias (`products.py:173`); `Dx` and `Tx` are the unchanged Starsim 3 classes with their dataframe-of-probabilities interface, not aliases of `Test` and `Treatment`.
- **Open questions answered by the structure.** `19-open-questions.md` B12 ("names and spellings that may change") is effectively answered by the fact that nothing was renamed or removed: the implementation chose "add beside" over "replace" for every family, which is the structural question this review is about and which the spec never poses explicitly.

## Bugs

### `test_samples.py` races under xdist

`test_samples_no_identifier`, `test_samples`, `test_verbose`, and `test_seed_result` (`tests/test_samples.py:31-69`) all write `tests/temp/samples_results/0.2.zip` via `ss.Samples.new(..., folder=resultsdir)` with the same identifier `p_death=0.2`. With `-n auto` they land on different workers and one reads a half-written zip:

```
FAILED test_samples.py::test_seed_result - zipfile.BadZipFile: Truncated file header
   starsim/samples.py:413: in __getitem__  with self.zipfile.open(self._seedfile(item)) as f:
```

Alone, `pytest test_samples.py` gives `6 passed in 1.53s` (`test_samples_alone.txt`). Fix: a per-test folder (or `tmp_path`), or delete the class (section 2).

### `solver='ode'` is silently ignored for a Starsim 3 disease

```python
sim = ss.Sim(diseases=ss.SIR(), networks='random', n_agents=500, dur=2, solver='ode').run()
# solver attr: abm | pars.solver: ode | engine: NoneType | reason: None
```

(`v3ode.py`.) No warning or note is printed at `verbose=1`. `sim.py:329-332` sets `self.solver = 'abm'` and `self.solver_reason = None` and returns when `not self.declared`, before the requested solver is compared with what can run. A one-line `ss.note` or `E804` ("the module `sir` (SIR) only works on agents"), which the declared path already raises for the same situation, would do.

## Keep

- **The regression discipline.** No v3 test was weakened; `baseline.yaml` changed by 24 lines. Whatever is deleted, keep the rule that the v3 tests stay green until their classes go.
- **`kernels.py` as the single numerical core**, with `cache=True` and the `Known issues` list in `implementation.md` being honest about where the Python path differs. The 45 small jitted functions with explicit layouts are the readable form of a solver; keep that shape when splitting `solvers.py`.
- **`Delivery` as one class with keywords** (`interventions.py:535`) rather than the 14-class v3 lattice. It is the pattern the spec's "Adding to the vocabulary" rule asks for ("routine and campaign delivery are keywords on one function, not two classes").
- **`Graph` as the compiler** (`transitions.py:1099`): one place that resolves names, states, people, parameters, and exits, and that `describe`, `diagram`, `equations`, `to_json`, and `stratify` all read from. Move it to its own file; do not break it up.
- **Lazy imports of JAX, BlackJAX, `starsim_data`, Optuna, NetworkX**, and the `interop` design of one `Source` and one `build()` for eleven formats (`interop/core.py`).
- **The `__init__.py` import-timing scaffold** (`debug = True` prints per-module import times); cheap and useful when the surface shrinks.
- **The vignettes as the acceptance test for the vocabulary.** They use `ss.Disease`, `stratify`, `contacts`, `Mobility`, `timeseries`, `seasonal`, `outcome`, `vaccinate`, `fit` and no v3 interventions or calibration (`usage.txt`); they are the evidence that the v4 half is sufficient for 22 real models.

## Outside my area

- `ss.Disease` has no `solvers()` method although the spec puts it there (`families.txt`); the diseases or solvers reviewer should pick one spelling.
- `E504`'s "did you mean `randomnet.pars.n_contacts`" when the target was `sir.pars.beta` on a v3 `SIR` is a misleading suggestion; the error should say the module is not declared.
- `routine_vx(start_year=5)` on a sim with `stop=20` and no dates fails with "Years must be within simulation start and end dates" (`families.txt`); if `routine_vx` is kept it should accept the v4 bare-number convention from `implementation.md` ("a bare number ... is a year only if the sim has dates").
