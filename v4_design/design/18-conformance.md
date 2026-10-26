# Conformance

What has to be tested for the specification's claims to hold, and the models the tests run on.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md) and by the north star in the [root README](../README.md).

## The idea

Each claim the specification makes ([README.md](README.md#what-the-specification-claims)) has a test suite here. A claim with no test should be treated as a hope.

The tests are of three kinds:

- **Asserted**: the test fails if the behavior is wrong (the three spellings give the same model spec; a modifier restores the base value).
- **Recorded**: the result is a published number or table, not a pass or fail (how far an ODE approximation of a network model is from the agent-based run).
- **Measured against the original**: lines of code and run time for each vignette, compared with the framework it came from.

## Levels

An implementation can be complete in stages. Each level includes the ones before it.

| Level | Covers |
|---|---|
| **Core** | Objects ([01](01-objects.md)), types ([02](02-types.md)), states and transitions ([03](03-states-and-transitions.md)), parameters ([04](04-parameters.md)), the ordered loop ([10](10-execution.md)), the model spec ([14](14-model-spec.md)), errors as classes ([15](15-errors.md)), and `solver='abm'` |
| **Compartmental** | `solver='ode'` and `'auto'`; mixing pools and contact matrices; Erlang stages; `equations()`; states that are not people |
| **Stochastic** | `'gillespie'` (exact), `'tau'` (tau-leaping), and `'binomial'`; `ss.MultiSim` for many runs |
| **Structured** | Stratification ([05](05-dimensions.md)), `ss.modify` and conditions ([07](07-effects.md)), interventions ([08](08-interventions.md)), results and data ([09](09-results.md)) |
| **Spatial** | Patches, mobility models, commuting, geography |
| **Inference** | Calibration, sensitivity, and model comparison ([13](13-inference.md)) |
| **Interoperable** | Import and export for at least SBML and one framework ([16](16-interop.md)) |

## Test suites

Each suite ends with what tests it now. "Not yet tested" means the claim is a hope until a test is written; it is not a claim that the behavior is wrong. The test files are in `tests/`; the model corpus below is `tests/corpus.py`.

### 1. Form equivalence

For every corpus model, the two spellings of its transitions (operators and explicit `ss.Transition`) and the two ways of building the module (constructor and class) MUST produce the same model spec and the same model hash, once names agree. ([01-objects.md](01-objects.md#the-equivalence-rule)) The same holds for:

- the ways of declaring states: `S, I, R = ss.states(3)`, `ss.states('S I R')`, and `S = ss.State()` one at a time;
- results and outcomes passed as keywords and passed through `transitions=`, `results=`, and `outcomes=`.

Tested by `test_transitions.py::test_forms` on the SIR (the spellings, a class, and the ways of declaring states, by their hashes). Not yet tested on the rest of the corpus.

### 2. The model spec and the saved form

- Reordering keywords, and the order in which transitions and parameters are passed, does not change the spec or any hash. Reordering states, modules, or stratifications does, because that order has meaning ([14-model-spec.md](14-model-spec.md#normalization)).
- The code `sim.to_code()` writes builds a sim with an identical spec and hash, and the same results (`test_transitions.py::test_load`, on a stratified cholera model under `gillespie`, and with a likelihood). Over every corpus model, the code builds the same hash or lists what it leaves out, such as a network or a function (`test_contracts.py::test_code_round_trip`).
- A sim saved with `ss.save('run.sim', sim)` after a run loads with the same results, under every solver (`test_transitions.py::test_load`).
- A saved model spec is not read back, and `ss.load()` of one says what to do instead (`test_transitions.py::test_load`).
- Each export lists what it changed and left out (`ss.errors.LossyExport`): `test_interop_export.py` checks the list for Erlang stages, a network, an agent-based sim, and a stratification, and that SBML and odin exports read back with the same results. That the list matches everything missing from the file is not checked.

### 3. Agreement between solvers, on simple models

For each **simple** corpus model (marked below), every solver is run and compared with the ODE. Agents and the stepped solvers differ from the ODE by an amount of order `dt`, so the comparison is at a fine step (a quarter of a day). The contracts are layered, from exact to statistical:

| Contract | What is checked | Test |
|---|---|---|
| Conservation | Without births, deaths, or flows to `ss.null`, the number of people is constant, to 1e-9, under `ode`, `binomial`, `gillespie`, and `abm` | `test_contracts.py::test_conservation` |
| Exact identities | Constant-rate decay under `ode` matches the exponential to 1e-5; a fixed branch share is `p` (exactly under `ode`, within 0.01 under `binomial` and `abm`) | `test_contracts.py::test_identities` |
| Destinations | The flow to each destination of a transition with several sums to its total, by group too, under every solver; deaths by age along one destination agree with the ODE within 3 standard errors over 20 runs of `binomial`, `gillespie`, and `abm` | `test_results.py::test_destinations`, `test_contracts.py::test_destinations` |
| Competing exits | Reordering three equal competing exits leaves a third leaving by each, under every solver (including tau-leaping) | `test_contracts.py::test_exit_order` |
| Stochastic agreement | The mean over time of each state, averaged over a few runs of `binomial`, `gillespie`, and `abm`, is within 5% of the ODE's (wider for some models, each with its reason); a small state is compared with a tenth of the largest | `test_contracts.py::test_solvers_agree` (the simple models; every compartmental corpus model with `STARSIM_SLOW=1`) |
| Time in state | A branch probability given the time in the state (an exponential dwell of mean 1 and `p = 1 - exp(-P.time)`, so half go each way) within 3 standard errors under `abm`, and a compartmental solver refuses it (`CapabilityError`, and `auto` runs agents); a hazard that grows with it, against a death rate, clears the exact share within 0.01; the agents' expression and a function reading `hpv.P.time[uids]` give the same run | `test_transitions.py::test_given_dwell` |
| Convergence in `dt` | The difference between a stepped solver and the ODE shrinks with `dt`, and `sim.check_dt()` reports it | `test_transitions.py::test_time_values` (one model) |
| Backend equivalence | A stepped run and a whole run in the kernels draw the same numbers; so do a run stopped and continued, and a run with another sim run in between | `test_contracts.py::test_streams`, `test_streams_across_sims` |

The tolerances are stated per test with the reasoning, so that a failure means something. Not yet tested: agreement against known laws with a correction for repeated comparisons, convergence with its observed order, and distributions (peak size and time, dwell times) rather than means. A test at every output time against Monte Carlo standard errors, as an earlier draft asked, is too strong: a finite population and a finite `dt` give a real bias that many runs make significant.

**Complex models are not asserted to agree.** For the corpus models that are approximated under some solver (a lognormal dwell time as stages), the difference SHOULD be recorded in a published table (peak size, peak time, final size, and endemic prevalence under each solver), so that a change in it is reviewed like any other change in behavior. Not yet done.

### 4. Approximation notes

For every approximation in [12-backends.md](12-backends.md#the-standard-approximations):

- under `solver='auto'`, an approximation of the mixing is never made: the model runs as agents, and the printed reason names the piece that made it agent-based;
- under `solver='auto'`, a dwell time that is not exponential or Erlang is the one exception: the model stays an ODE with fitted Erlang stages, and one note (`ss.errors.StagesUnderAuto`) names the transition, the number of stages, and the mean and standard deviation kept;
- under an explicit solver, the run completes and prints one line naming the piece and what it was approximated by;
- with `approx=True`, the same run prints nothing, and the note is still kept in `sim.notes`;
- with an explicit solver and `approx=False`, the same request raises `ss.CapabilityError` naming the piece and the approximation that would have been used;
- with `solver='auto'` and `approx=False`, the dwell-time model runs as agents, where the dwell time is sampled directly;
- with `verbose=0` nothing is printed, and the note is still kept in `sim.notes`;
- `sim.describe()` shows `approx` for that ingredient, before anything is run;
- a distribution on `beta` or `rel_sus` runs as agents under `'auto'`, and under an explicit `solver='ode'` uses the mean and says so.

For the things with no approximation (contact tracing, transmission trees, agent-only custom code), a compartmental solver raises `ss.CapabilityError` with a message in the shape [15-errors.md](15-errors.md) asks for.

Tested in part by `test_transitions.py::test_auto` and `test_solvers`, `test_kernels.py::test_stages`, and `test_deliveries.py` (agent-only deliveries such as `trace=`); not yet as one suite over every row of the table in 12.

### 5. Functions: the try-it rule

- A function that uses only counts, parameters, time, and stratification coordinates (`fn(sim)` reading `sim.people.age` on an age-stratified model) gives the same model under `'abm'` and `'ode'`, within the tolerance of suite 3.
- A function that takes `uids`, or reads an attribute that is not a stratification, runs under `'abm'` and stops a compartmental run with `ss.CapabilityError`, naming the function and what it asked for.
- Under `'auto'`, the first kind of function leaves the model an ODE and the second makes it agent-based, with the function named in the printed reason.
- The same three checks for a lazy condition (`eligible = (ppl.age > 65) & ~sir.vaccinated`). A function of the sim that compares ages with a number that is not a bin edge runs in a compartmental model without complaint (each group has the age of its midpoint); the lazy form is checked.

Tested by `test_transitions.py::test_functions` and `test_auto`, `test_deliveries.py::test_conditions`, and the `custom` corpus model in `test_contracts.py::test_solvers_agree` (with `STARSIM_SLOW=1`).

### 6. The ordered loop

- **The loop is shown as a schedule.** `sim.describe()` prints the loop grouped by timestep, one line per module in the order they run; the unrolled list is `sim.loop.to_df()`.
- **Order is as specified.** `sim.loop.to_df()` matches the table in [10-execution.md](10-execution.md#the-order-of-calls-in-one-step), with modules of one type in the order given.
- **A within-step chain.** In the screening example of [10-execution.md](10-execution.md#a-within-step-chain), by default interventions come before transmission; after `sim.loop.insert(screen, label='sir.step')`, the screen runs after transmission in the same step, so a person can be infected, tested, and treated on one step (`test_deliveries.py::test_loop_order`, from the order of calls and the results).
- **Each module sees what earlier ones did.** With `diseases=[syphilis, hiv]` and a modifier of HIV transmissibility conditional on syphilis, a new syphilis infection changes HIV transmission in the same step; with the order reversed, one step later. Not yet tested.
- **Mutual dependence runs.** Two diseases that each modify the other build and run with no error (the `coinfection` corpus model; `test_deliveries.py::test_coinfection`).
- **Competing hazards are joint.** Reversing the order of competing transitions out of one state does not change results (suite 3, `test_exit_order`).
- **Removals are deferred.** An agent whose death is requested during a step is still visible to every later module in that step and is gone at the next. Not yet tested as such.
- **Ordering effects are O(dt).** Swapping the order of two modules changes results by an amount that shrinks with `dt` for models where the effect of one module on another within a step is of order `dt` (not, for example, a screen moved before or after the transmission it acts on). Not yet tested.

`test_loop.py` covers the loop's plan, `insert()`, and its plots.

### 7. `ss.modify`

- **Base and effective.** While a modifier is active, reading the array (`sir.rel_sus`) gives the base, and `sir.rel_sus.effective` the value with the active modifiers applied (agents only). When the modifier's condition stops holding, or `stop=` passes, the value the engine uses is the base again, exactly, with no bookkeeping by the user (`test_deliveries.py::test_effective`, `test_change`).
- **Combination.** Set, then multiply, then add, whatever order the modifiers were declared in. Two `set=` modifiers active at the same time give the note `ss.errors.OverlappingSet`, and the later one in the loop is used (`test_deliveries.py::test_modify_where`).
- **Assignment goes to the base.** `arr[uids] = x` while a modifier is active changes the base, and the modifier still applies on top (not yet tested).
- **Any named numeric quantity.** A parameter, `rel_sus` and `rel_trans`, a transition, and a network parameter as a path (`'school.pars.n_contacts'`) work as targets (`test_deliveries.py::test_modify_where`, `test_network_contacts`); a boolean state or a plain number raises `ss.errors.UnmodifiableTarget` (`test_transitions.py::test_modify`, `test_deliveries.py::test_network_contacts`).
- **Nothing accepted, then ignored.** A modifier, change, or efficacy that no solver would apply raises `ss.errors.Unapplied` when the sim is built; a disease written as code that overrides `transmit()` gets its modifiers, and a change to an inflow applies under the ODE (`test_deliveries.py::test_applied`).
- **Acquisition by any rate.** On the Ross-Macdonald model (corpus 12), the bednet idiom `ss.modify(malaria.rel_sus, multiply=0)` and a vaccine of efficacy 1 stop every human infection, `ss.treat` reaches the infected (`Eh`, `Ih`), and `R0()` is the next-generation value with the mosquitoes at their disease-free equilibrium, under `ode` and `abm` (`test_contracts.py::test_acquisition`).
- **`what_modifies()`** lists every modifier on the target; code that assigns `rel_sus` or `rel_trans` directly is not a modifier, and is not tracked (`test_deliveries.py::test_effective`).

### 8. Stratification

- **In place by default.** After `sir.stratify(age=[0, 15, 65])`, `sir` is stratified, and the call returned `sir` itself, so `ss.Sim(sir.stratify(age=[0, 15, 65]))` works. Running the same line again does nothing more (with a note); stratifying by the same dimension in another way raises `ss.errors.DuplicateStructuralDimension`.
- **A copy on request.** After `sir_age = sir.stratify(age=[0, 15, 65], inplace=False)`, the spec and hash of `sir` are what they were before, and `sir_age` is identical to a model stratified in place.
- **Stratifying the sim.** `sim.stratify(age=[0, 5, 18, 65])` and `ss.Sim(..., stratify=dict(age=[0, 5, 18, 65]))` give the same spec, and every module in the sim has the same age groups. Two diseases stratified by age with different groups raise `ss.errors.InconsistentDimension`.
- **States that are not people** are not split by a people dimension (age, sex, risk), are split by `patch`, and are split by their own dimensions when named in `states=`.
- **A contact matrix with other age groups** is aggregated to the model's groups by population in a compartmental run, with the note `ss.errors.MatrixAggregated`.
- **Coverage delivers `f`.** `coverage=0.7` on a model with `P` strata reaches 70% of the eligible people in total, not `1 − (1 − 0.7)^P`; realized coverage is read from the delivery's results.
- **Capacity is a total.** `capacity=50_000` delivers 50,000 doses however many strata there are, shared in proportion to the eligible population.
- **Continuous properties stay continuous in an agent-based run** and are binned only for adjustments, conditions, and results.
- **Space without mobility** builds and runs with no warning.

Tested by `test_transitions.py::test_stratify` (in place, twice, `ss.stratify`, `ss.errors.InconsistentDimension`), `test_dimensions.py` (states that are not people, space, mixing, strains), and `test_deliveries.py::test_vaccinate` and `test_coverage_by_group` (coverage reached in a round, a capacity per day, and coverage over several groups). Not yet tested: `inplace=False` leaving the original's hash unchanged, the aggregation of a contact matrix against one aggregated by hand, and coverage and capacity on 1, 3, and 16 age groups.

### 9. Defaults are printed

Every default the library guessed appears in `describe()`, marked `[guessed]` with the rule that produced it. `test_contracts.py::test_guesses_marked` checks, for each row of the guess table in [03](03-states-and-transitions.md#what-is-guessed), one model that triggers the guess (and is marked) and one that does not (and is not); `test_transitions.py::test_describe` checks the text of `describe()` for the SIR. A published expected `describe()` output for each corpus model, diffed so that a change is reviewed, is not yet written.

**Time with no unit.** A model in which nothing has a unit (no `unit=`, no dates, no typed values) runs in timesteps: `stop=100` is 100 steps, `beta=0.3` is per step, results are indexed by step, `describe()` shows no unit, and nothing is printed about days or years. Giving `unit=`, dates, or a typed value gives the sim that unit, and the note `ss.errors.UnitInferred` says where it came from when it was not given. Typed values with different units alongside bare numbers raise `ss.errors.IncompatibleTypes`, since the unit of the bare numbers is not known (`test_transitions.py::test_time`).

### 10. Reproducibility

- The same model, settings, and seed give identical results on repeated runs on the same machine and library versions, in serial and in parallel (`test_baselines.py`, `test_run.py`).
- Adding an unrelated module (an analyzer, or a disease that shares no states) does not change another module's draws, whether the run is whole or stepped; two identical modules with different names have streams of their own (`test_contracts.py::test_streams`).
- With common random numbers, which are keyed on each agent's slot and nothing else, the variance of the difference between a baseline and an intervention scenario is lower than with `ss.options.crn = False` for small perturbations; the pairing weakens once trajectories diverge (`test_randomness.py` checks the streams and `ss.options.crn`; `test_contracts.py::test_crn_pairing` checks, for a declared disease as agents, that the pairing makes the variance of the difference well below that of independent runs. Against `ss.options.crn = False` it is not yet shown: for a declared disease, plain streams keep the scenarios in step too, and over 30 seeds of a small change the two variances were about the same).
- A sim is one run. `ss.parallel(sim, n_runs=100)`, `sim.run(n_runs=100)`, and `ss.MultiSim(sim, n_runs=100).run()` give runs of the base sim with different seeds, and a result has a run dimension (`.values` is runs by time, `.quantiles()` summarizes across runs) (`test_run.py`, `test_results.py`).
- `sim.hash()` returns an `sc.objdict` with `model`, `pars`, `config`, `scenario`, `data`, and `seed`, and changes in exactly the part that was changed: a parameter value changes `pars` only, an added state changes `model`, a numerical option changes `config`, an added intervention changes `scenario`, attached data changes `data`, and a different `rand_seed` changes `seed` (`test_transitions.py::test_forms`, `test_modify`).

### 11. Errors

The catalog in [15-errors.md](15-errors.md) is generated from the classes in `starsim/errors.py` (their docstrings), and `test_spec.py::test_error_catalog` checks that the page matches the code and that each class is `ss.errors.<Name>` with a docstring. Each class SHOULD have a minimal example that triggers exactly it, with a message that contains the name the user gave the thing and a suggested fix; about half have one today (as `pytest.raises(ss.errors.<Name>)` in the feature tests), and there is not yet one parametrized test over every class.

### 12. Calibration

- On a corpus model with data simulated from known values, each fitting method recovers them: posterior methods cover the truth, and optimizers recover it within a tolerance, with `converged` true (simulation-based calibration, `test_contracts_inference.py::test_sbc`; longer runs with `STARSIM_SLOW=1`).
- The JAX log-density equals `sim.loglik()` at three parameter points, to 1e-8, for every corpus model with data; the particle filter's estimate is within its Monte Carlo error of the likelihood and peaks at the same parameter as direct runs (`test_contracts_inference.py::test_jax_loglik`, `test_particle_filter`).
- The printed output names the fitting method and the reason.
- For an ODE run, the gradient of the results with respect to each parameter, from automatic differentiation, agrees with a finite-difference gradient (`test_gradients.py::test_gradient`).
- `fit.sim` is a run sim with the fitted values, and the original sim is unchanged.
- `ss.fit.compare_models` compares models by their prediction of held-out data, with one measure for all (`test_transitions.py::test_likelihood`).
- Staged fitting and `sim.sensitivity()` run (`test_contracts_inference.py::test_staged_and_sensitivity`).

### 13. North star

The claim: **a model should never be easier to write without the lingua franca than with it.** For each vignette in [vignettes](../vignettes), which translates one of a framework's own examples, `tests/north_star.py` writes a table of lines and time, original against Starsim, with the failing rows:

| Measure | How | Requirement |
|---|---|---|
| **Lines of model code** | Count non-blank, non-comment lines needed to define and run the model, in both versions. Docstrings, imports, plotting, data loading, and the `if __name__ == '__main__'` block are excluded. Lines longer than 100 characters count as two, so that length cannot be hidden in one long line. Where the original calls a built-in written for one situation (`interventions=close_schools_uk()`), the lines of that built-in's body count as part of the original; the lingua franca version writes the same thing out and is counted in full | Lingua franca ≤ original: the hard target |
| **Time** | Two numbers for each: the first run in a fresh process, and the best of 5 repeated runs, on the same machine, at the same population size, timestep, and duration. Each Starsim run builds and runs the sim (`sim.new().run()`) | Reported. A time fails if the lingua franca is both more than 2x slower and more than 1 s slower |
| **Results** | Compare the vignette's results with the original's, to a tolerance stated in the vignette | Agree |

The table has one row per vignette, failing rows included. A row where the lingua franca is longer or slower is a defect in the design or the implementation and is listed as such, with the reason. It is not hidden, and it is not a reason to drop the vignette. The table is published on the "Coming from" pages of the documentation (`docs/coming_from/`).

The table SHOULD cover at least one example from each main framework: MEmilio, EpiModel, epydemix, summer2, SimInf, odin, EMULSION, flepiMoP, Atomica, LASER, and Starsim itself.

The acceptance test is the thirteen switching demos in `tests/switching/` (run with `cd tests && pytest switching`): one per framework, each the five steps of a switch (bring the model in, describe it, fit it, ask the policy question, run it in another paradigm or at another scale), scaled down to run in CI and asserting the original framework's reference numbers. A demo that does not pass yet is marked `xfail`, naming what blocks it.

Three notes on fairness. Time is compared for the solver closest to the original's (an ODE against an ODE, a network model against agents); where the original is compiled C++ or Rust, matching its speed from Python is a real requirement, which is why the compartmental solvers run in compiled kernels ([12-backends.md](12-backends.md)). The allowance for built-ins is there because the comparison is of what each framework lets a modeler say, not of its library of ready-made cases. And the line count is of the model, not of the surrounding analysis, since that is what the claim is about.

### 14. Documentation

Every python block in this folder (chapters 00 to 19) is run by `test_spec.py::test_spec`, in order within each chapter, with each disease example checked to grow; a block that cannot run is marked as a fragment, and a block that fails because of a known library bug is listed in the test. The counts in [17-vocabulary.md](17-vocabulary.md) are generated from the names `ss` exports and compared by `test_spec.py`. The user documentation (`docs/`) is executed when it is built. Comparing printed output with what a page shows is not yet done.

### Performance

Not one of the claims, but checked: `test_baselines.py::test_performance_guard` compares the time of a set of runs with `tests/perf_guard.yaml`, and the `benchmark_*.py` scripts (with their `.yaml` records) time the larger cases. A slowdown is treated as a bug.

## The corpus

The models are the functions of `tests/corpus.py`, each returning an unrun sim sized to run in seconds, with the solver as a keyword (`corpus.sir(solver='binomial')`); the file also has a small model of each standard pattern of the users in the switching review. Models marked **simple** are used for the solver-agreement test in suite 3 by default; the others with `STARSIM_SLOW=1`.

| # | Model | What it tests |
|---|---|---|
| 1 | **SIR**, with no unit anywhere (simple); its spellings and a class are in `test_forms` | Form equivalence; every solver; time in timesteps when nothing has a unit |
| 2 | **SEIR**, in the operator form (simple) | The source of infection, in the rate |
| 3 | **SIR with an Erlang dwell time**, `ss.erlang(mean=10, k=3)` (simple) | Sub-stages, exact in every solver |
| 4 | **SIR with a lognormal dwell time** | `'auto'` keeps it an ODE with fitted Erlang stages and prints `StagesUnderAuto`; `solver='ode'` prints the stage approximation; `approx=True` silences it; `solver='ode'` with `approx=False` raises `CapabilityError`; `'auto'` with `approx=False`, and `solver='abm'`, sample the dwell time directly; the difference is recorded |
| 5 | **A branching model** with per-branch dwell times and a competing death hazard (simple) | Split on entry versus competing hazards |
| 6 | **An age-stratified SIR with a real contact matrix** (simple) | `stratify` in place and on the sim, `ss.contacts`, aging, a contact matrix aggregated to the model's age groups (`MatrixAggregated`); `inplace=False` leaves the original untouched |
| 7 | **A model on patches with gravity mobility**, and the same patches with no mobility | Space as a stratification; commuting as a share of time |
| 8 | **Vaccination as a stratification plus a transfer** | The disease never mentions vaccination; coverage delivers `f`; capacity is a total |
| 9 | **Two diseases with `ss.modify` in both directions** (HIV and syphilis) | Mutual dependence; combination rules; base and effective values; `what_modifies()` |
| 10 | **A sexual-network model** (`ss.SexualNet`) | `'auto'` runs it as agents; `solver='ode'` uses the partner-change model, with a note, and `approx=False` raises `CapabilityError` |
| 11 | **Cholera**: one disease whose states include a water reservoir `W` | Counts in rates; `W` declared a count (`ss.State(people=False)`); inflow from and outflow to `ss.null`; `'auto'` gives an ODE; adding `n_agents=` makes people agents while the water stays a count; `beta * W` raises `ExposureToCount` |
| 12 | **A vector-borne model** (Ross-Macdonald malaria): humans and mosquitoes as states of one disease | Two populations coupled by counts; a seasonal inflow; humans as agents with mosquitoes as three counts; mosquito states left out of `total_pop` and of an age stratification, and split by species with `states=` |
| 13 | **Screening with follow-up treatment** | The within-step chain: a treatment triggered by the test's outcome (`who=screen.positive`), run after it in the same step; moving a module with `sim.loop.insert` |
| 14 | **A model with custom code**: a function of the sim in a branch probability that works on age groups | The try-it rule; what `'auto'` chooses. A function that needs `uids` and a `step()` override are in `test_transitions.py::test_functions`, not the corpus |
| 15 | **A model with outcomes, data, and bounds** (cases, hospitalizations, deaths) | Outcomes as weighted sums of flows with `delay=` and `duration=`; results by name; data matching with `ss.simdata(aliases=)`; `sim.fit()` end to end; `ss.fit.compare_models` |
| 16 | **A threshold-triggered intervention**, with `when=` as a function of the sim | `when=` functions. The same policy held for four weeks by an intervention's `step()` (memory) is not yet in the corpus |
| 17 | **A TB model with a resistance stratification** and acquired resistance as a transfer with `states=` | Strain as a dimension; `ss.transfer` restricted to some states |
| 18 | **Two diseases in one sim** (flu and RSV) **with age stratified on the sim** | `sim.stratify`; every module gets the same groups; `InconsistentDimension` when two diseases disagree |

## What these tests do not show

- That a model is a good model. Nothing here checks the epidemiology.
- That a particular fit converged. Each fit says whether it did (`fit.converged`, with the reason in `fit.why`).
- That an approximation is good enough for the question being asked. The library says what was approximated and the recorded tables say roughly how much it matters; the judgment is the modeler's.
