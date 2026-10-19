# Starsim v4: a review of the spec and its implementation

Reviewed on 2026-10-17 at commit `a20c4f7e` (branch `v4`, clean tree), against the spec in `v4_design/design/` and the code in `starsim/`. The brief was Cliff's: look for improvements, treat nothing as sacrosanct, and aim for simplicity, for the user first and the codebase second.

Thirteen reviewers each covered one area and wrote a report (listed at the end, about 67,000 words in all, with roughly 130 findings, 70 verified bugs, and 100 "keep" notes). Every claim about behavior in those reports was run, and the scripts are in `scripts/<area>/`. I read all thirteen summaries and reran the six claims that matter most myself; those are marked **(checked)** below. Everything else is the reviewers' finding, with a repro in their report.

This page is the synthesis: what I think the review adds up to, in order of importance, and a suggested order of work. The area reports hold the detail, the evidence, and the smaller findings.

## The one-paragraph version

The language is good. The declarative core (`S >> I @ beta * I`, one transition class, states that are inferred or declared, `describe()` marking every guess, one model running under several solvers) works, agrees with itself across the four spellings, and reproduced 30 held-out models from eleven frameworks. What is not good is everything around it: v4 added 43,000 lines and removed 151, so Starsim 3 and Starsim 4 now sit side by side in one package as two languages that do not interoperate; the public surface is 253 names where the spec counts 94; the overview's headline example fails at two of its nine steps; the two-line SIR runs with nobody infected; and there is a set of core semantic bugs (units of bare numbers in rates, a shared random stream between diseases, `@` arithmetic that drops terms) that would undermine trust in results the moment a user met one. The spec, meanwhile, promises about forty things the code does not do. The work ahead is less about building and more about choosing and deleting.

## Part 1: the eight things I would do first

### 1. Make it one language

**(checked)** `ss.Sim(diseases=ss.SEIR(), solver='ode')` runs as agents, prints nothing, and `approx=False` raises nothing. A Starsim 3 module cannot take `ss.vaccinate`, `ss.modify`, `sim.calibrate()`, or any compartmental solver; a v4 `ss.Disease` cannot run with `ss.Births`, `ss.seasonality`, or `ss.routine_vx` under one. Each half has its own interventions (19 old names vs 7 new), products (`Dx`/`Tx`/`Vx` vs `Test`/`Treatment`/`Vaccine`), calibration (`Calibration` + 5 component classes vs `fit`), seasonality (`seasonality` vs `seasonal`), result containers (`Samples` vs `MultiResults`), and demographics (`Births` vs `ss.birth`). Reports 04, 08, 11, 13.

The recommendation, with the structure reviewer's per-family table (13 §Simpler codebase) as the detail:

- **Rewrite `ss.SIR`, `ss.SIS`, `ss.SEIR` as declared diseases** in the class form. The spec already promises this (`01-objects.md` §Library models) and it is what makes "the library model and the one-page model are the same model" true. Today `ss.SIR().equations()` raises and `ss.Sim(ss.SIR())` with no network simulates nothing (11 §1).
- **Delete the superseded families**: the `routine_vx`/`campaign_*`/`Base*`/`treat_num`/`simple_vx` lattice (755 lines, zero library, vignette, or v4-test use), `Dx`/`Tx` (keep `Vx` as a one-line alias if at all), `ss.seasonality` (it overwrites `rel_trans`, the anti-pattern the spec warns against), `Samples`/`Dataset`, and `Calibration`/`CalibComponent` with its five likelihood classes (1,150 lines). About 2,300 lines and 40 public names. Breaking for Starsim 3 scripts, which is what a major version is for; a migration page covers it.
- **Keep as library modules written in the language**: `Infection` (the base for coded diseases with real `step()` logic), `Births`/`Deaths`/`Pregnancy` (agent-only, data-driven; the declared `ss.birth`/`ss.death` cover every solver), and `MixingPool(s)` as the class `ss.contacts` returns.
- **Before deleting the old interventions**, make `ss.vaccinate` work on a coded disease; today it refuses with `E506` (04 §1).

### 2. Fix the core semantics bugs

These are small in code and large in consequence. None is a design question.

- **(checked) A bare number inside a rate expression is typed per unit.** `S >> E @ R0 * gamma * I` with `R0 = 2.5`, `gamma = 0.2` peaks at 1,168 infected with `unit='day'` and 38 with `unit='week'`, because both numbers are read as rates in the sim's unit and multiplied. A product of two rates is not a rate; `02-types.md`'s own `E102 WrongDimension` would catch it, and is unimplemented. Only the outermost bare factor can carry the unit; the rest are dimensionless. Every importer writes dimensionless parameters as bare numbers, so every import is right only at the unit it was imported at (10 §Bugs 1, 01 §4).
- **(checked) Stepped stochastic kernels share one global Numba random stream**, so adding an unrelated disease changes another disease's draws (`[502, 573, 648, 738]` alone vs `[505, 574, 653, 744]` with a second disease present). The whole-run kernel and the agent-based path are fine. This breaks the first rule of `11-random.md` for any stochastic compartmental run that is not taken whole, which includes `verbose=1`. Give each engine its own stream (07 §1).
- **(checked) `@` is not strict.** `I >> R @ ss.days(10) * 2` silently drops the `* 2`; `@ ss.days(10) @ 0.1` sets both `dur` and `rate`; `@ ss.by(age=[ss.days(5), ss.days(20)])` is read as rates of 5 and 20 per day and runs. The rule "the type of what follows `@` decides the slot" cannot see inside `ss.by`, functions, or `ss.par`; make `__matmul__` refuse a second driver and make arithmetic on a duration-driven transition raise (02 §3).
- **Arrow-string names resolve against the caller's variables, not the module's states.** `ss.Transition('I', 'R')` grabs whatever *variable* is called `I` in the calling frame, so it can take another module's state (`E206`) or miss a state named `I` that is bound to `I3` (`E204`) (02 §1).
- **The people-vs-count guess anchors on creation order.** `W, S, I, R = ss.states('W S I R')` makes `S`, `I`, `R` counts and `W` the people. Anchor on the states of contact-mediated transitions, then the largest connected component (02 §2).
- **(checked) `sir.pars.beta = 0.1` replaces the `ss.par` with a float**: bounds and units are gone and `ss.modify(sir.pars.beta)` then fails with `E504`. Route `Pars.__setitem__` through `par.value` (01 §2).
- **`beta=ss.lognorm_ex(...)` silently runs at the mean under every solver**, agents included (`solvers.py:840`), where the spec says agents under `auto` and a note under `ode` (06 §3).

### 3. Fix the defaults that produce a wrong model silently

- **(checked) The two-line SIR runs flat.** With no `init_prev`, nobody is infected, `n_I` is 0.0 at every step, and `plot()` draws six horizontal lines without a word. A disease with a contact-mediated transition and no initial condition should seed one agent or 1% and mark it `[guessed]`, or warn (12 §2).
- **A dimension that only a transfer feeds starts half full.** `stratify(vaccinated=[False, True])` with `[guessed: equal across levels]` puts 499,500 of 999,000 susceptibles at `vaccinated=True` on day 0; the spec's own "vaccination as a stratification plus a transfer" example is five to ten times off the `ss.vaccinate` answer for what reads as the same model (04 §6, 12 §4). A boolean or transfer-only dimension should start everyone at its first level.
- **The string-form `source=` guess is wrong for SEIR and cannot be checked.** Every SEIR vignette writes `source='I'`; the one that forgets gets a different epidemic (peak 1,657 vs 2,012) that only `describe()` reveals. Require `source=` when the destination has an onward progression (02 §4).
- **Names are picked from caller frames at two moments.** Transitions are named at `ss.Disease()`, modules at `ss.Sim()`, so `ss.Sim(make_model())` gives `sim.results.disease` and inline transitions get `S_to_E`; a later `adjust={'infection': ...}` then fails with `E301`. Name every object once, at its own construction (12 §3). Separately, `ss.states(3)` is the one spelling that can fail hard (`E210`, states have no fallback name) when states are built in one function and the module in another; `ss.states('S I R')` costs six characters and never fails (01 §3). I would keep `ss.states(3)` only if the frame lookup is made reliable; otherwise drop it from the documentation and keep it as a convenience.

### 4. Make the overview example run, and write the two missing pages

**The overview's "whole language in one model" fails at steps 8 and 9** (12 §1): `ss.parallel(baseline=fit.sim, ...)` raises `AlreadyRunError` (copying doesn't help; `fit.sim.new()` works and nothing says so), and `msim.plot()` on an `n_runs=` MultiSim raises `ValueError: No legend handles found`. Also `ss.Sim(sir, pars=fit.pars)` fails whenever a fitted parameter is named `dur`, because `ss.Sim` decides what `pars=` means by sniffing its keys (08 §2, 11 §2). Let `ss.parallel` take a run sim, fix the plot, and give `pars=` one meaning.

**The docs contain zero v4 terms.** 34 of 35 tutorial and user-guide pages execute unchanged under v4 and not one mentions `ss.Disease(`, `>>`, `solver=`, `describe()`, `ss.modify`, or `data=`; the only v4 documentation is a changelog entry (12 §7, with a page-by-page table). A user who finds the docs writes v3 code; a user who finds the spec meets an example that doesn't run. A "v4 in ten minutes" page and a v3-to-v4 migration page are worth more than any API change in this review, and they are also the forcing function for Part 1.

### 5. Cut the solver menu and its knobs

Six solvers plus `'stochastic'`, plus `integrator=`, `substeps=`, `approx=`, `use_kernels`. The solvers reviewer ran one SIR under all of them (06): `sde` has one test, zero vignettes, the worst agreement with the ODE (+3%, peak 18% high on an Erlang SEIR), and the spec itself rejects it as a default; `tau` is what `'stochastic'` falls back to for large populations and has no reason to be chosen by hand. Cut to **`ode`, `gillespie`, `binomial`, `abm`** plus `auto`, with tau-leaping as `gillespie`'s automatic, noted fallback above an event-count threshold (and fix the threshold: `total_pop × n_transitions` ignores rates and run length and undercounted a 50k endemic SIS by 24x).

**Drop `integrator=` from the public surface** (06 §7, 07 §4). It mixes schemes (RK4/Euler), a library (SciPy), and a gradients backend (JAX); `integrator='jax'` is 150 times slower than the kernels for a single run with identical results, and JAX should be chosen internally for `sim.gradient()`, NUTS, and parameter batches only. Keep `substeps=`.

**Delete the Python solver path** (`derivs`, `step_*_once`, `step_tau`, `step_gillespie`, about 300 lines, `solvers.py:2365-3137`) and with it `use_kernels` (06 §4, 07 §7). It is taken for the whole engine whenever a user function sits in a rate (26x slower Gillespie, different random numbers), and the function can be evaluated once per step into the program's tables, as time-varying values already are. Condition from the kernels reviewer: the stochastic kernels are today tested only against this path, so first test them against ODE expected values, as `test_stages` already does.

Also: `verbose=1`, `profile=True`, `until=`, and `loop.insert()` each silently switch a compartmental sim from the whole-run kernel to the Python loop, 30 to 100 times slower (1 ms vs 28 ms; 11 ms vs 1,091 ms) with identical results. A print flag should not be a performance cliff (07 §2). And the first-use kernel compile is 29 s on this machine, not the documented 12 s, and silent (07 §3).

### 6. Shrink the public surface to what the spec counts

`ss.` exports 253 non-module names; `17-vocabulary.md` counts 94 (01 §1, 13 §2). About 60 are v4 internals (`Graph`, `Engine`, `Crossing`, `CountState`, `Strat`, `Cond`, `Expr`, `Fit`, `ParticleFilter`, `make_engine`, `StockCtx`, `module_map`, ...) and about 40 are the Starsim 3 duplicates of Part 1. Beyond those two groups, the reviewers propose these cuts, each argued in its report:

| Cut | Today | Proposed | Report |
|---|---|---|---|
| Time constructors | 17 (`probper*`, `freqper*` used in two network defaults and nowhere in v4) | 9: `days`..`years`, `perday`..`peryear`, `date`; `ss.prob(0.1, 'day')` spells the rest | 01 §6 |
| Ways to stratify | 4 (`sir.stratify`, `ss.stratify(sir)`, `sim.stratify`, `ss.Sim(stratify=)`) | 2 or 3; the function form is one line calling the method; the sim-level form silently turns `states=`, `immunity=`, `inplace=` into dimensions named `states`, `immunity`, `inplace` | 03 §2, §7 |
| Dimension `kind` | a five-name list (`patch` is space, `place`/`herd`/`location` are plain); every vignette wrote `kind='space'` by hand | the kind lives on the value (`ss.geo(...)`, a strain marker); only `age` by name | 03 §1 |
| Age mixing matrix | 5 spellings (`MixingPools(src=AgeGroup...)`, `LevelGroup`, `contacts('Kenya')`, `contacts(M, age=edges)`, `contacts(age=M)`; `age=` means edges in one and a matrix in another) | `ss.contacts` only; retire `AgeGroup`/`LevelGroup` | 03 §4 |
| Network classes | 18; `RandomSafeNet`, `MaternalNet`, `PostnatalNet`, `BreastfeedingNet`, `MSMNet`, `HybridNet`, `ClusterNet` each used by one test or one library module | merge the three random nets; move single-use nets to `starsim/library/networks/` | 03 §5 |
| Result names per flow | `infection`, `new_infections`, `cum_infection`, `cum_infections` (a plain SIR has 10 results, 4 duplicates) | the transition name, with `.cum` a property; keep `new_infections` for library diseases only | 05 §2 |
| MultiSim result containers | 3 that disagree (`MultiResult` is not a `Result`; `reduce()` writes a fourth copy; `reduce(); plot()` crashes for every declared model) | one `ss.Result` with a leading run axis | 05 §1 |
| Model spec loader | `from_spec`, a restricted expression `Parser`, schema versioning (~270 lines and a security surface); no vignette, test, or doc reads a spec back; it cannot read interventions, most networks, functions, or coded modules | `to_code()` is the saved form; `to_spec()` stays as the input to `hash()` and `diff()` | 09 §1 |
| Exports | Mermaid (twice: `export.py` and `Graph.diagram('mermaid')`), SBML, odin with `W1401 LossyExport`; a stratified SIR through odin and back is off by 9,211 on a peak of 9,387 | Mermaid only, in one place | 10 §4 |
| Importers | 11 frameworks, 5,000 lines in `starsim/interop/`, 5 needing the other framework installed, tests hard-coding one machine's paths | keep `core.py` (`Source` → `build` → `to_code`, `ss.diff`); importers to a companion package | 10 §1 |
| Modifier plumbing | `track_writes=` and the direct-writes half of `what_modifies()` (40 lines of frame inspection; the open question said drop it if the vignettes never use it, and they don't) | keep the declared-modifiers listing | 04 §3 |
| Demographics shortcuts | 4 ways to add births (`ss.birth`, `ss.Births`, `birth_rate=`, `demographics=True`); the `SimPars` shortcuts silently turn an ODE model into a 10,000-agent one | 2: the declared transition and the module | 11 §6 |
| Output channels | `print()`, `ss.note` (drops its `code=`), `ss.warn` with hand-pasted codes, `describe()`; three notes are lost under `verbose=0` so `meta.notes` depends on verbosity | exceptions and one `ss.note(code, msg)` that always records | 09 §3 |
| Error catalog | 57 codes in `15-errors.md` (header says 54); 7 never raised, 5 absent from code, 2 uncataloged, `E605` nameless, no note code tested | the catalog's own rule ("a code with no test is removed") gives about 28 | 09 §4 |

### 7. Settle spec vs code where they disagree

Beyond the departures `implementation.md` already lists, the reviewers found about forty places where the spec promises what the code does not do. Most are the spec over-promising; a few are the code being wrong. The ones that need a decision rather than a correction:

| Question | Spec | Code | I'd pick |
|---|---|---|---|
| Reading a modified quantity | effective value; `.base` for the base | base value; the engine applies modifiers where used; `.effective` as agents only; `.base` is `None` | **code**: one array, no cache invalidation; rewrite 07 in one sentence (04 §2) |
| `beta=ss.Dist` under `auto` | agents | ODE at the mean, no note, even under `abm` | **spec** (06 §3) |
| Counts across stratifications in a rate | aligned on every shared dimension; `I.n.sum('age')`/`select()` exist | `I.n` is local for space, global otherwise; no `sum`/`select` on counts | **code**, stated plainly, plus one explicit spelling (03 §3) |
| Time-varying rates under Gillespie | thinning, MUST | held for a quarter step, with a note (`deferred.md`) | **code**; strike the MUST |
| Sexual network under `ode` | partner-change-rate approximation | `E804`, correctly | **code**; strike the row (03 §6, 06) |
| Fitting method | table with ABC-SMC, "cheap enough" timing, simplex first | five-line rule: ODE no likelihood → L-BFGS-B; ODE + likelihood → NUTS; stochastic compartmental + likelihood → PMCMC; else Optuna | **code**; the rule is better than the table (08 §3) |
| `per × dur` | an expected count (0.3) | a probability (0.259), relied on at 46 call sites | **code**; fix the table in 02 (01 §4) |
| A declared disease with a `step()` override | the canonical class-form example; `compartmental = True` opts in | forced to `abm` at `sim.py:982`; `compartmental` ignored for declared diseases | **spec**: try it on groups like any function (11 §3) |
| Agent-only attribute under requested `ode` | `E804` naming the attribute, offering `stratify` | passes `init()`, raw `AttributeError` at step 1 | **spec** (06 §6) |
| Delivery order | in the loop | after progressions and before transmission as agents; before all flows compartmentally (known issue) | state it as the rule and stop calling it an issue |

Promised and absent, to finish or strike: `sim.sensitivity()`, staged fitting (`method=[...]` raises `AttributeError`), `ss.Sim(results=[...])`, `hiv.solvers()` on a disease, `describe(verbose=True)`, `ss.func`, `sir.infection.beta` (gives `None`), `msim.plot(lines=True)`, the infection log for declared diseases (0 rows), `dependence='density'`/`n_ref`, `E403` on a misspelled `beta` key, `W1203`/`W1204`/`W1303`, convergence reporting for optimizers (the SciPy result is discarded; `maxiter=1` passes silently). The reviewers' view, which I share: strike most, because each is a feature nobody has asked for yet, and move the two or three that matter (`E804` for attributes, `E403`, `describe(verbose=True)`) to a short list.

Of the open questions in `19-open-questions.md` §B1, the implementation has effectively answered about twenty; each report lists the ones in its area. `19` should be updated to say so.

### 8. Reshape the three big files

Not user-visible, but it is where the maintenance cost lives (13 §3-4, 02 §5, 11 §7, 06 §5, 08 §6):

- `sim.py` doubled (1,396 to 2,692). About 420 lines of stratification geometry (`init_strats`, `init_ragged`, `init_crossings`, `fold*`, `grid_index`, `group_raw`, ...) and 240 of solver choice (`init_solver`, `agent_reasons`, `try_functions`, `kernel_reason`) sit on `Sim`, which has 76 public methods. A `sim.groups` object and the engine should own them; the `init_*` phases should be private.
- `transitions.py` (2,972) holds six unrelated families; the ~930 lines of `by`, `Strat`, `geo`, `gravity`, `Mobility`, `Transfer`, `stratify` and the 53 lines of error codes belong elsewhere (a `dimensions.py`).
- `solvers.py` (4,136) has one `Compartmental` class of 1,570 lines and 70 methods. After the Python path goes, the remaining seam is `Crossing`: N engines each holding a full copy of the joint state with an `owner`/`version`/`claim()`/`refresh()` coherence protocol and 28 `self.cross` checks. The math is right (agrees with `abm` within 1-4%); make it one engine over the product graph.
- `calibration.py`: after `Calibration` goes, split the 304-line `fit()` (seven method branches, three coordinate systems) into one function per method, and drop one of the two NUTS paths and the likelihood code duplicated between `calibration.py` and `gradients.Density`.
- `ss.Disease` is both the declared-model class and the base of `Infection`/`NCD`, with 15 `graph is None` branches, and `sir.S` changes type from `ss.State` to `BoolState` at init. The cheapest fix is to stop rebinding `sir.S` (02 §6).

## Part 2: what to keep

Every report has a "Keep" section; these recur:

- **`describe()`.** Three reviewers independently called it excellent. It is the thing that makes guessing defaults safe, and it should be the center of the docs. (It lacks `verbose=True`, mislabels "per step" when `dt=0.1`, prints module order rather than loop order, and says nothing about randomness; all small.)
- **The declarative core and its spellings.** Arrow strings, operators, explicit `ss.Transition`, and the class form build identical specs and bit-identical results (verified in 02). The explicit form earns its place in the vignettes (19 uses, for `p=` lists and `where=`). Only `b @ x` (a transition with no source yet) and "a finished transition times a number scales its rate" (which conflicts with the outcome meaning) could go.
- **One transition class whose kind follows from its rate; `ss.null`; states that are not people; `I.n` vs bare `I`.** The reservoir and vector models in the spec work as written.
- **In-place `stratify`, `where=` for ragged stratifications, `ss.by`, `ss.transfer`.**
- **The whole-run kernels.** 1 ms for an SIR; one compile covers every solver and model shape. Run-level CRN pairing works (it cut the standard deviation of an intervention effect from 52 to 10 infections) and should be reported in one line of `describe()`.
- **`Crossing`'s math, the chain-binomial `substeps` fix, and the Erlang-stage expansion.**
- **`to_code()`.** For simple models it is near hand-written (EMULSION: 37 lines vs the vignette's 11, the difference mostly comments). It should be the product of import and the saved form of a model; it needs a companion data file for large arrays and rounding to the source's precision.
- **`ss.fit()`'s method rule, NUTS with the `W1302` identifiability note, and run-level pairing.** The inference reviewer timed a 1-parameter ODE fit at 1.15 s (JAX compile) and NUTS with a correct correlation note at 4.9 s.
- **The interop `core.py` pipeline and `ss.diff`.**

## Part 3: a suggested order

1. **Core bugs first** (Part 1 §2): units in rates, the shared stream, strict `@`, name resolution, the people guess, `pars` assignment, `beta=ss.Dist`. Days of work; none changes the language.
2. **Defaults and the headline example** (§3, §4 first half): seed infections, transfer-fed dimensions start empty, `source=` required for SEIR shapes, name once at construction, `parallel` takes a run sim, `pars=` has one meaning. Then the overview runs and the two-line SIR shows an epidemic.
3. **One language** (§1): `SIR`/`SIS`/`SEIR` as declared diseases, `ss.vaccinate` on coded diseases, then delete the superseded families. This is the big breaking step and the one that makes the docs writable.
4. **The two docs pages** (§4 second half), written against the language that survives step 3.
5. **Surface cuts** (§5, §6): solvers to four, `integrator=` gone, the table in §6 family by family, each with its vignettes rerun.
6. **Spec reconciliation** (§7): one pass through `design/` striking what the code does not do and recording the decisions; update `19-open-questions.md` and `17-vocabulary.md`'s count to the real one.
7. **Structure** (§8), last, because it is invisible to users and easiest once the deletions have happened.

The test suite is a constraint on all of this: 952 s serial (870 s of it v4), with five tests at 37 to 51 s each because they run every scenario with 50,000 agents under three solvers, and one xdist race (`test_samples::test_seed_result`). Cutting those agent counts tenfold (13 §6) would make the rest of the work faster.

## The reports

| File | Area | Findings (user / codebase / spec / bugs / keep) |
|---|---|---|
| [01_objects_types_vocabulary.md](01_objects_types_vocabulary.md) | Objects, types, names, the vocabulary count | 10 / 4 / 12 / 8 / 10 |
| [02_states_transitions_parameters.md](02_states_transitions_parameters.md) | States, transitions, parameters: the declarative core | 11 / 6 / 13 / 6 / 9 |
| [03_dimensions_routes.md](03_dimensions_routes.md) | Stratification, mixing, networks, mobility, geography | 8 / 5 / 25 / 5 / 8 |
| [04_effects_interventions.md](04_effects_interventions.md) | Modifiers, conditions, products, deliveries | 10 / 7 / 12 / 5 / 9 |
| [05_results_runs.md](05_results_runs.md) | Results, data, outcomes, MultiSim | 11 / 7 / 13 / 9 / 8 |
| [06_solvers.md](06_solvers.md) | The solver menu and the compartmental and agent engines | 8 / 5 / 12 / 3 / 8 |
| [07_execution_random_kernels.md](07_execution_random_kernels.md) | Loop order, randomness, Numba kernels, JAX | 6 / 6 / 10 / 2 / 7 |
| [08_inference.md](08_inference.md) | Calibration, likelihoods, gradients, model comparison | 9 / 7 / 12 / 7 / 10 |
| [09_model_spec_errors.md](09_model_spec_errors.md) | The JSON model spec, `to_code`, hashing, error codes | 7 / 3 / 9 / 8 / 6 |
| [10_interop.md](10_interop.md) | Importers and exporters for eleven frameworks | 6 / 5 / 4 / 4 / 7 |
| [11_sim_people_modules.md](11_sim_people_modules.md) | `ss.Sim`, `ss.People`, modules, demographics | 9 / 6 / 12 / 4 / 7 |
| [12_user_experience.md](12_user_experience.md) | The overview example and a new user's path, end to end; the docs gap | 10 / 3 / 9 / 6 / 8 |
| [13_codebase_structure.md](13_codebase_structure.md) | Size, duplication, dependencies, tests, dead code | 3 / 11 / 6 / 2 / 7 |

Repro scripts: `scripts/<area>/`. They were run with `/software/conda/envs/v4/bin/python` at the commit above; `scripts/check.py` and `scripts/check2.py` are my own six spot-checks.

Nothing in this review has been applied to the code or the spec.
