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
| **Core** | Objects ([01](01-objects.md)), types ([02](02-types.md)), states and transitions ([03](03-states-and-transitions.md)), parameters ([04](04-parameters.md)), the ordered loop ([10](10-execution.md)), the model spec ([14](14-model-spec.md)), coded errors ([15](15-errors.md)), and `solver='abm'` |
| **Compartmental** | `solver='ode'` and `'auto'`; mixing pools and contact matrices; Erlang stages; `equations()`; states that are not people |
| **Stochastic** | `'gillespie'` (and its alias `'ctmc'`), `'tau'`, `'binomial'`, `'sde'`, and `'stochastic'`; `ss.MultiSim` for many runs; extinction reporting |
| **Structured** | Stratification ([05](05-dimensions.md)), `ss.modify` and conditions ([07](07-effects.md)), interventions ([08](08-interventions.md)), results and data ([09](09-results.md)) |
| **Spatial** | Patches, mobility models, commuting, geography |
| **Inference** | Calibration, sensitivity, and model comparison ([13](13-inference.md)) |
| **Interoperable** | Import and export for at least SBML and one framework ([16](16-interop.md)) |

## Test suites

### 1. Form equivalence

For every corpus model, the three spellings of its transitions (arrow-string constructors, operators, explicit `ss.Transition`) and the two ways of building the module (constructor and class) MUST produce the same model spec and the same model hash, once names agree. ([01-objects.md](01-objects.md#the-equivalence-rule)) The same holds for:

- the ways of declaring states: `S, I, R = ss.states(3)`, `ss.states('S I R')`, and `S = ss.State()` one at a time;
- the spellings of an arrow string: `'I -> R'`, `'I->R'`, `'I >> R'`, and `'I>>R'`;
- results and outcomes passed as keywords and passed through `transitions=`, `results=`, and `outcomes=`;
- `solver='ctmc'` and `solver='gillespie'`.

### 2. Normalization and round trip

- Normalizing a spec twice gives the same document as normalizing it once.
- Reordering keywords, and the order in which transitions and parameters are passed, does not change the spec or any hash. Reordering states, modules, or stratifications does, because that order has meaning ([14-model-spec.md](14-model-spec.md#normalization)).
- `ss.load(sim.to_json())` gives a sim with an identical spec, for every corpus model.
- For every other export format, the printed list of what was left out matches what is actually missing from the file.

### 3. Agreement between solvers, on simple models

For each **simple** corpus model (marked below), run every solver that `solvers()` marks `yes` and compare. A sim is one run, so every comparison that needs many runs uses an `ss.MultiSim` (`ss.parallel(sim, n_runs=1000)`).

| Comparison | Agreement expected | How it is checked |
|---|---|---|
| `ode` and `gillespie` | The mean of 1,000 `gillespie` runs is within 2 Monte Carlo standard errors of the ODE at every output time, for `total_pop` of 100,000 or more | per time point |
| `ode` and `abm` with homogeneous mixing or a mixing pool | The same, with the mean over agent-based runs | per time point |
| `gillespie` and `tau` | Distributions of peak size and peak time agree (two-sample Kolmogorov-Smirnov, α = 0.01) | distributions |
| `binomial` and `gillespie` | The difference shrinks as `dt` shrinks, and `sim.check_dt()` reports it | convergence |
| Erlang dwell time, `ode` and `abm` | Dwell-time distributions agree (Kolmogorov-Smirnov, α = 0.01) | distributions |
| A stratification in `ode` and the same property per agent in `abm` | As the `ode` and `abm` row | per time point |
| A state that is not people (the cholera reservoir) in `ode`, and in `abm` with the people as agents and the reservoir as a count | As the `ode` and `abm` row | per time point |
| `ss.MixingPool(n_contacts=10)` in `ode` and `ss.RandomNet(n_contacts=10)` redrawn every step in `abm` | Agree in distribution as the population grows ([06-routes.md](06-routes.md#routes)) | convergence in `n_agents` |

These are assertions. The tolerances depend on population size and number of runs and are stated per test with the reasoning, so that a failure means something.

**Complex models are not asserted to agree.** For the corpus models that are approximated under some solver (the sexual-network model as an ODE, a lognormal dwell time as stages), the suite runs both and **records** the difference in a published table: peak size, peak time, final size, and endemic prevalence under each solver. An agent-based HIV model and its ODE approximation are expected to differ; the table says by how much, and a change in the table is reviewed like any other change in behavior.

### 4. Approximation notes

For every approximation in [12-backends.md](12-backends.md#the-standard-approximations):

- under `solver='auto'`, an approximation of the mixing is never made: the model runs as agents, and the printed reason names the piece that made it agent-based;
- under `solver='auto'`, a dwell time that is not exponential or Erlang is the one exception: the model stays an ODE with fitted Erlang stages, and one line (`W1205`) names the transition, the number of stages, and the mean and standard deviation kept;
- under an explicit solver, the run completes and prints exactly one line naming the piece and what it was approximated by;
- with `approx=True`, the same run prints nothing, and the note is still in `sim.describe()` and `sim.solvers()`;
- with an explicit solver and `approx=False`, the same request raises `E804 CapabilityError` naming the piece and the approximation that would have been used;
- with `solver='auto'` and `approx=False`, the dwell-time model runs as agents, where the dwell time is sampled directly, and the printed reason names the transition;
- with `verbose=0` nothing is printed, and the note is still in `sim.describe()` and `sim.solvers()`;
- `solvers()` shows `approx` for that ingredient under that solver, before anything is run;
- a distribution on `beta` or `rel_sus` runs as agents under `'auto'`, and under an explicit `solver='ode'` uses the mean and says so.

For the things with no approximation (contact tracing, transmission trees, agent-only custom code), a compartmental solver raises `E804` with a message in the shape [15-errors.md](15-errors.md) asks for.

### 5. Functions: the try-it rule

- A function that uses only counts, parameters, time, and stratification coordinates (`fn(sim)` reading `sim.people.age` on an age-stratified model) gives the same model under `'abm'` and `'ode'`, within the tolerance of suite 3.
- A function that takes `uids`, or reads an attribute that is not a stratification, runs under `'abm'` and stops a compartmental run with `E804`, naming the function and what it asked for.
- Under `'auto'`, the first kind of function leaves the model an ODE and the second makes it agent-based, with the function named in the printed reason.
- The same three checks for a lazy condition (`eligible = (ppl.age > 65) & ~sir.vaccinated`), including the case where the threshold falls inside an age bin.

### 6. The ordered loop

- **The loop is shown as a schedule.** `sim.describe()` prints the loop grouped by timestep ("every day: ...; every year: ..."), one line per module in the order they run; the unrolled list is `sim.loop.to_df()`.
- **Order is as specified.** `sim.loop.to_df()` for each corpus model matches the table in [10-execution.md](10-execution.md#the-order-of-calls-in-one-step), with modules of one type in the order given.
- **A within-step chain.** In the screening example of [10-execution.md](10-execution.md#a-within-step-chain), testing and treatment (`on_positive=`) happen in the same step. With the default order, a person infected on step `t` is first screened on step `t + 1`; after `sim.loop.insert(screen, label='sir.step')`, a person can be infected, tested, and treated on step `t`. Both are asserted from the event times.
- **Each module sees what earlier ones did.** With `diseases=[syphilis, hiv]` and a modifier of HIV transmissibility conditional on syphilis, a new syphilis infection changes HIV transmission in the same step; with the order reversed, one step later.
- **Mutual dependence runs.** Two diseases that each modify the other build and run with no error.
- **Competing hazards are joint.** Reversing the order in which two transitions out of one state are declared does not change results in distribution; a sequential-draw implementation fails this at large `rate × dt`.
- **Removals are deferred.** An agent whose death is requested during a step is still visible to every later module in that step and is gone at the next.
- **Ordering effects are O(dt).** Swapping the order of two modules changes results by an amount that shrinks in proportion to `dt`, and `sim.check_dt()` reports it.

### 7. `ss.modify`

- **Base and effective.** While a modifier is active, reading the array gives the effective value and `.base` gives the base. When the modifier's condition stops holding, or `stop=` passes, the effective value equals the base again, exactly, with no bookkeeping by the user.
- **Combination.** Set, then multiply, then add, whatever order the modifiers were declared in. Two `set=` modifiers active for the same people give `W502` and the later one in the loop is used.
- **Assignment goes to the base.** `arr[uids] = x` while a modifier is active changes the base, and the modifier still applies on top.
- **Any named numeric quantity.** A parameter, a float state, a transition's slot, a network parameter, and a string path all work as targets; a boolean state or a plain number raises `E504`.
- **Cost.** A modifier outside its `start`/`stop` period is not evaluated. A constant modifier with an unchanged active set does not trigger a recompute. Checked by counting recomputations.
- **`what_modifies()`** lists every modifier on the target. With `track_writes=True` it also lists every module observed assigning it; with tracking off, which is the default, it says that direct writes were not tracked.

### 8. Stratification

- **In place by default.** After `sir.stratify(age=[0, 15, 65])`, `sir` is stratified, and the call returned `sir` itself, so `ss.Sim(sir.stratify(age=[0, 15, 65]))` works. `ss.stratify(sir, age=[0, 15, 65])` does the same. Running the same line again raises `E305`.
- **A copy on request.** After `sir_age = sir.stratify(age=[0, 15, 65], inplace=False)`, the spec and hash of `sir` are what they were before, and `sir_age` is identical to a model stratified in place.
- **Stratifying the sim.** `sim.stratify(age=[0, 5, 18, 65])` and `ss.Sim(..., stratify=dict(age=[0, 5, 18, 65]))` give the same spec, and every module in the sim has the same age groups. A disease stratified by the same groups afterwards only gains its adjustments. Two diseases stratified by age with different groups raise `E308`.
- **States that are not people** are not split by a people dimension (age, sex, risk), are split by `patch`, and are split by their own dimensions when named in `states=`.
- **A contact matrix with other age groups** is aggregated to the model's groups in a compartmental run, with `W404`, and agrees with a hand-aggregated matrix; in an agent-based run it is used as it is, with no note.
- **Chained stratifications commute where they should.** Stratifying by age then risk, or risk then age, with multiplicative adjustments gives the same results.
- **Coverage delivers `f`.** `coverage=0.7` on a model with `P` strata reaches 70% of the eligible people in total, not `1 − (1 − 0.7)^P`. Asserted on the vaccination corpus model with 1, 3, and 16 age groups; realized coverage is read from the delivery's results.
- **Capacity is a total.** `capacity=50_000` delivers 50,000 doses however many strata there are, shared in proportion to the eligible population.
- **Continuous properties stay continuous in an agent-based run** and are binned only for adjustments, conditions, and results.
- **Space without mobility** builds and runs with no warning.

### 9. Defaults are printed

Each corpus model has a published expected `describe()` output. Every default the library guessed appears in it, marked `[guessed]` with the rule that produced it. A change to that output is reviewed, because it is a change in what the model means. Specific checks: the inferred unit and where it came from, the inferred states, the states inferred to be counts and not people, the guessed source of infection in the string form, the exponential dwell time from a scalar duration, the homogeneous mixing default, and the placement of the initial population.

**Time with no unit.** A model in which nothing has a unit (no `unit=`, no unit on `dt`, no dates, no typed values) runs in timesteps: `stop=100` is 100 steps, `beta=0.05` is per step, results are indexed by step, `describe()` shows no unit, and nothing is printed about days or years. Giving any one of `unit=`, a unit on `dt`, dates, or a typed value gives the sim that unit, and `W101` says where it came from. Typed values that disagree, with bare numbers present, give the smallest unit and `W103`.

### 10. Reproducibility

- The same model, settings, and seed give identical results on repeated runs on the same machine and library versions, in serial and in parallel. ([10-execution.md](10-execution.md#determinism))
- Adding an unrelated module does not change another module's draws.
- With common random numbers, which are keyed on each agent's slot and nothing else, the variance of the difference between a baseline and an intervention scenario is lower than with `ss.options.crn = False`, and the draws `describe()` lists as paired are unchanged by a perturbation elsewhere in the model.
- A sim is one run. `ss.parallel(sim, n_runs=100)` and `ss.MultiSim(sim, n_runs=100).run()` give the same 100 runs; each has the base sim's spec with a different seed, and a result has a run dimension (`.values` is runs by time, `.quantiles()` summarizes across runs).
- `sim.hash()` returns an `sc.objdict` with `model`, `pars`, `config`, `scenario`, and `seed`, and changes in exactly the part that was changed: a parameter value changes `pars` only, an added state changes `model`, a different solver changes `config`, an added intervention changes `scenario`, and a different `rand_seed` changes `seed`. The runs of a MultiSim differ only in `seed`.

### 11. Errors

Every code in [15-errors.md](15-errors.md) has a minimal example that triggers exactly it, and the message contains the name the user gave the thing and a suggested fix. A code with no test is removed from the catalog.

### 12. Calibration

- On the calibration corpus model, `sim.calibrate()` with no arguments recovers known parameter values from synthetic data to a stated tolerance, under each solver, using the fitting method the library chose.
- The printed output names the fitting method and the reason. An Optuna fit reports no interval.
- For an ODE run, the gradient of the fit with respect to each fitted parameter, from automatic differentiation, agrees with a finite-difference gradient, including through a function in a rate that is written with array operations.
- `fit.sim` is a run sim with the fitted values, and the original sim is unchanged.
- `ss.compare_models` ranks the true generating model first on synthetic data from an SEIR.

### 13. North star

The claim: **a model should never be easier to write without the lingua franca than with it.** For each vignette in [vignettes](../vignettes), which translates one of a framework's own examples:

| Measure | How | Requirement |
|---|---|---|
| **Results** | Compare the vignette's results with the original's, to a tolerance stated in the vignette | Agree |
| **Lines of model code** | Count non-blank, non-comment lines needed to define and run the model, in both versions. Imports, plotting, and data loading are excluded from both. Lines longer than 100 characters count as two, so that length cannot be hidden in one long line. Where the original calls a built-in written for one situation (`interventions=close_schools_uk()`), the lines of that built-in's body count as part of the original; the lingua franca version writes the same thing out and is counted in full | Lingua franca ≤ original |
| **Wall-clock time** | Time one run of each on the same machine, at the same population size, timestep, and duration, taking the median of five runs after one warm-up run | Lingua franca ≤ original |

The result is a published table with one row per vignette: framework, model, lines (original and lingua franca), time (original and lingua franca), and whether results agree. A row where the lingua franca is longer or slower is a defect in the design or the implementation and is listed as such, with the reason. It is not hidden, and it is not a reason to drop the vignette.

The table SHOULD cover at least one example from each main framework: MEmilio, EpiModel, epydemix, summer2, SimInf, odin, EMULSION, flepiMoP, Atomica, LASER, and Starsim itself.

Three notes on fairness. Time is compared for the solver closest to the original's (an ODE against an ODE, a network model against agents); where the original is compiled C++ or Rust, matching its speed from Python is a real requirement, which is why the compartmental solvers generate array code ([12-backends.md](12-backends.md#how-a-model-is-built)). The allowance for built-ins is there because the comparison is of what each framework lets a modeler say, not of its library of ready-made cases. And the line count is of the model, not of the surrounding analysis, since that is what the claim is about.

### 14. Documentation

The documentation ships with the library version and is readable offline. Every code example in this folder and in the user documentation is run in continuous integration, and its printed output is compared with what the page shows.

## The corpus

Models marked **simple** are used for the asserted solver-agreement tests in suite 3.

| # | Model | What it tests |
|---|---|---|
| 1 | **SIR**, in all three spellings and as a class, with no unit anywhere and with `unit='day'` (simple) | Form equivalence; every solver; time in timesteps when nothing has a unit |
| 2 | **SEIR**, in the operator form and in the string form with `source='I'` (simple) | The source of infection; the printed guess when `source=` is left out |
| 3 | **SIR with an Erlang dwell time**, as `ss.gamma` with an integer shape and as `stages=3` (simple) | Sub-stages, exact in every solver |
| 4 | **SIR with a lognormal dwell time** | `'auto'` keeps it an ODE with fitted Erlang stages and prints `W1205`; `solver='ode'` prints the stage approximation; `approx=True` silences it; `solver='ode'` with `approx=False` raises `E804`; `'auto'` with `approx=False`, and `solver='abm'`, sample the dwell time directly; the difference is recorded |
| 5 | **A branching model** with per-branch dwell times and a competing death hazard (simple) | Split on entry versus competing hazards |
| 6 | **An age-stratified SIR with a real contact matrix** (simple) | `stratify` in place and on the sim, `ss.contacts`, aging, a contact matrix aggregated to the model's age groups (`W404`); `inplace=False` leaves the original untouched |
| 7 | **A model on patches with gravity mobility**, and the same patches with no mobility | Space as a stratification; commuting as a share of time |
| 8 | **Vaccination as a stratification plus a transfer** | The disease never mentions vaccination; coverage delivers `f`; capacity is a total |
| 9 | **Two diseases with `ss.modify` in both directions** (HIV and syphilis) | Mutual dependence; combination rules; base and effective values; `what_modifies()` |
| 10 | **A sexual-network model** (`ss.MFNet`) | `'auto'` runs it as agents; `solver='ode'` prints the partner-change approximation note and runs; `approx=False` raises `E804`; the difference is recorded |
| 11 | **Cholera**: one disease whose states include a water reservoir `W` | Counts in rates; `W` inferred to be a count and printed (`W212`); inflow from and outflow to `ss.null`; `'auto'` gives an ODE; adding `n_agents=` makes people agents while the water stays a count; `beta * W` raises `E211` |
| 12 | **A vector-borne model** (Ross-Macdonald malaria): humans and mosquitoes as states of one disease | Two populations coupled by counts; a seasonal inflow; humans as agents with mosquitoes as three counts; mosquito states left out of `total_pop` and of an age stratification, and split by species with `states=` |
| 13 | **Screening with follow-up treatment** | The within-step chain; `on_positive=`; moving a module with `sim.loop.insert` |
| 14 | **A model with custom code**: a function in a rate slot that works on groups, one that needs `uids`, and a `step()` override | The try-it rule; what `'auto'` chooses; `E804` messages |
| 15 | **A model with outcomes, data, and bounds** (cases, hospitalizations, deaths) | Outcomes as weighted sums of flows with `delay=`, `duration=`, and `every=`; results by name; data matching with `ss.data(aliases=)`; `sim.calibrate()` end to end; `ss.compare_models` |
| 16 | **A threshold-triggered intervention**, once with `when=` as a function of the sim and once as an intervention whose `step()` holds the policy for four weeks | `when=` functions; memory kept in `step()`, with no switching on and off every step |
| 17 | **A TB model with a resistance stratification** and acquired resistance as a transfer with `states=` | Strain as a dimension; `ss.transfer` restricted to some states |
| 18 | **Two diseases in one sim** (flu and RSV) **with age stratified on the sim** | `sim.stratify`; every module gets the same groups; `E308` when two diseases disagree |

Each corpus model has a published expected `describe()` output (suite 9).

## What these tests do not show

- That a model is a good model. Nothing here checks the epidemiology.
- That a particular fit converged. `fit.summarize()` reports that for each fit.
- That an approximation is good enough for the question being asked. The library says what was approximated and the recorded tables say roughly how much it matters; the judgment is the modeler's.
