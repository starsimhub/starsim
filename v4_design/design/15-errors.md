# Errors, warnings, and notes

What the library says when something is wrong, doubtful, or guessed, and the list of every coded message.

Justified by [best-practices/principles.md](../best-practices/principles.md).

## The policy

**Prefer guessing a default and printing it to raising an error.** Most things a user leaves out have a sensible default, and stopping the run to ask for it helps nobody. An error is for the cases where there is no sensible guess: a name that does not exist, lengths that do not match, a request the solver cannot meet.

A message is read once, by an epidemiologist or by an AI writing a model for one. Either way it should say what is wrong in the user's own terms, name the thing it is about and what exists instead, and say what to do.

## Message shape

Every named error SHOULD have:

1. **A name**, the exception class, such as `ss.errors.UnknownParameter`. Names are stable across versions so they can be searched for and documented, and caught: `except ss.errors.UnknownParameter`, or `except ss.CodedError` for any of them. (Drafts of this specification numbered them, e.g. `E602 UnknownParameter`; the numbers were dropped.)
2. **One line saying what is wrong**, using the names the user gave things.
3. **What to do about it**, as code the user could paste where possible, with the closest matching names for anything misspelled (and none when nothing is close).

```text
starsim.internals.errors.WrongDimension: the rate of the transition "recovery" is a plain number (a rate times a duration), not a rate:
       rate = peryear(0.1) * days(3) = 0.00082158
     A rate times a duration is a probability over that duration, not a rate; give the rate itself, e.g. rate=ss.peryear(0.1).
```

Avoid `ValueError: shape mismatch`, `KeyError: 'beta'`, and a traceback through library internals as the main message. Notes are one line each and need only say what was assumed or done.

## Levels

| Level | Means | What happens |
|---|---|---|
| **Error** (a subclass of `ss.CodedError`) | There is no sensible guess, or the request cannot be met | Raises an exception |
| **Warning** (a subclass of `ss.StarsimWarning`) | The run can proceed, but something probably needs attention | Issued through Starsim's `ss.warn`, so Python's `warnings` filters and `ss.options.warnings='error'` apply; the run continues |
| **Note** (`ss.note()`, some with a subclass of `ss.Note`) | Something was assumed, approximated, or is worth knowing | Kept in `sim.notes` whatever the verbosity; printed as one line when `verbose` is 1 or more |

`verbose` is one integer for every solver: 0 prints nothing, 1 (the default) the solver chosen, the notes, and the progress of a run, and 2 everything, including each step.

Most guessed defaults are not named at all: `describe()` marks them `[guessed]` ([01-objects.md](01-objects.md#describe)). The few that also print a line when the sim is built (the unit, a state treated as a count) are named notes so the line can be looked up. Approximations made when a solver is requested explicitly are unnamed one-line notes ([12-backends.md](12-backends.md#approximations)); `approx=True` keeps them without printing them, and they turn into `CapabilityError` under `approx=False`. The one approximation `solver='auto'` makes without being asked, Erlang stages for a dwell time, is named (`StagesUnderAuto`).

## The catalog

The table below is made from the code (`ss.errors.catalog()`, from the docstring of each class), and a test checks that it matches; to update it, run `ss.errors.check_catalog('v4_design/design/15-errors.md', write=True)`.

<!-- catalog start -->

66 names: 49 errors, 7 warnings, and 10 notes. Each is `ss.errors.<Name>` (and `ss.CapabilityError` and `ss.AlreadyRunError` are also at the root); the errors are subclasses of `ss.CodedError` (itself a `ValueError`), and the warnings and notes of `ss.StarsimWarning` (itself a `RuntimeWarning`).

| Name | Kind | When |
|---|---|---|
| `IncompatibleTypes` | error (also a `TypeError`) | Typed quantities are combined in a way that has no meaning, such as adding an `ss.per` to an `ss.prob` or `ss.freq`, or a number to a rate; names both types |
| `WrongDimension` | error (also a `TypeError`) | Typed arithmetic whose result has no type (a rate times a rate, a duration times a duration), or typed values that give a slot the wrong dimension: a rate times a duration, or a duration, in `rate=` or after `@`; a rate in `dur=`; both `beta` and `n_contacts` carry a time unit; or a typed rate as `beta` on a network with acts (acts per unit time times a rate per unit time is not a hazard), or inside `ss.peract` |
| `ReservedName` | error | A state, transition, parameter, or result is named like an attribute of `ss.Module` (`step`, `results`, `pars`, `describe`, `stratify`, `name`, `dt`, ...) |
| `UnknownState` | error (also a `KeyError`) | A state is named (in `init=`, `first=`, `states=`, a product, or a delivery) that the module does not have. Lists the closest names |
| `BadBranchSplit` | error | Constant branch probabilities out of one state do not sum to 1, or `ss.null` is one of several destinations (a branch out of the model is a transition of its own) |
| `DuplicateDeclaration` | error | Two members of a module are given the same explicit name, or a state is an endpoint of transitions in two modules |
| `NonlinearExposure` | error | A rate is not linear in its bare states (`I ** 2`, `I / (I + K)`, or a term containing two bare states). Says to use `I.n` if a literal count was meant |
| `UndrivenTransition` | error | A transition reaches build time with no rate, no duration, and no `p=` |
| `UnnamedState` | error | A state made by `ss.states(3)` or `ss.State()` cannot be matched to a variable name when the module is built (two names for one object, a state built inline). Suggests `ss.states('S I R')` or `ss.State('S')` |
| `ExposureToCount` | error | A bare state that does not hold people appears in a rate (`beta * W`). There is no contact with a reservoir to work out. Suggests `W.n` |
| `StrainsByCount` | error | In a disease with strains, an acquisition's rate counts its own states (`beta*I.frac`), which carry no strain. Suggests the transmission `beta*I` |
| `UnknownTransition` | error (also a `KeyError`) | `adjust=` names a transition that does not exist. Lists the ones that do |
| `BadSplit` | error | `split=` has the wrong number of entries. (A split that does not sum to 1 is normalized, and `describe()` says so) |
| `DuplicateStructuralDimension` | error | A model or sim is stratified by a dimension it already has, in another way. `stratify` works in place, so a notebook cell that is run twice stratifies it in the same way again, which does nothing (with a note) and is not this error. (Stratifying a disease by a people property the sim already has, with the same levels, only adds the adjustments and is not an error) |
| `InvalidAdjustment` | error | A negative multiplier in `adjust=`. (A multiplier that pushes a probability above 1 is clipped, with a note) |
| `PartialStratificationMixing` | error | Contacts are given by a dimension that some of the people who mix do not have: a state the dimension does not apply to (`states=`) is the origin or the source of a transmission, or the dimension has `where=`. There is no coordinate for those people to mix as |
| `InconsistentDimension` | error | Two modules in one sim are stratified by the same people property (age, sex, place) with different groups. Names both modules and their groups, and suggests stratifying the sim: `sim.stratify(age=[...])` |
| `NoGeography` | error | A movement model such as `ss.geo.gravity()` is used on a dimension with no geography to supply distances |
| `RouteMismatch` | error | A key of a per-layer `beta` dict matches no network or layer, or a network or layer has no key; a route has a `beta` of its own in a declared model; a `beta` is a positional pair (directions are named: `ss.peract(m2f=..., f2m=...)`); `ss.peract` is used on a route without acts, or names a direction the network does not have (or leaves one out, or has keys on a network with no sides), listing its directions; or a modifier's `route=` names no route |
| `UnmodifiableTarget` | error | The target of `ss.modify` or `ss.change` cannot be resolved to a named numeric quantity (a plain number, a boolean state, a name that does not exist). Lists the closest names |
| `Unapplied` | error | A modifier, change, or efficacy of a product on a quantity that no solver would apply: a plain parameter of a declared module (read only by its own code), the `rel_sus` or `rel_trans` of a disease that is not transmitted, a condition on people on an inflow or a transition of counts, or an efficacy on a transition with several destinations (rather than on one of them); or a stratification of a sim in which nothing uses it. Names it and why |
| `UnknownEffect` | error (also a `KeyError`) | A product names something it cannot act on: a key of a product's `efficacy` that is not something it can act on in the disease, a `transition=` of a treatment that the disease does not have (or a cure with no single transition to leave by), or a sensitivity of a test for a state the disease does not have. Lists the valid names |
| `UnknownDisease` | error (also a `KeyError`) | An intervention needs to know which disease it is for: `disease=` names one that is not a declared disease in the sim, or is not given and the sim has more than one |
| `UnknownParameter` | error (also a `KeyError`) | A parameter name the module does not have is set, directly or through `ss.Sim(pars=...)`; or, when the sim is initialized, a parameter was set before it (`pars.x = v`) that the module never defined (a renamed or misspelt one). Lists the closest names |
| `ShapeError` | error | A value's length does not match its stratification (four values for three age groups). Names the parameter, the transition, and both lengths |
| `UnvaluedParameter` | error | An `ss.par` has no value, no bounds to start from, and nothing supplied through `pars=` |
| `UnknownLocation` | error (also a `KeyError`) | A place name cannot be resolved by the data provider (`starsim_data`), or a source has no contact matrix for it. Says where it can be found, if anywhere |
| `OutOfBounds` | error | A parameter's value is outside its `bounds`; or a probability per act (`ss.peract`) is outside 0 to 1 |
| `UnknownOutcomeSource` | error (also a `KeyError`) | The source of an `ss.outcome` names nothing in the model |
| `OutcomeCycle` | error | Outcomes chain back to themselves |
| `DataMisaligned` | error | The data lies entirely outside the simulated period. (Data partly outside is used where it overlaps, with a note) |
| `UnknownDataColumn` | error (also a `KeyError`) | A data column matches no result, when a fit compares it with the results (at `init()` it is a note). Lists the closest result names and suggests `ss.simdata(..., aliases=dict(result='Column'))` to rename it or `ignore_extra=True` to drop it. Also data by stratum for a level the dimension does not have, or for a result that is not kept by group, or kept by group only every few steps |
| `OverlappingSources` | error | A treatment moves people in one state along two transitions of the same disease (`ss.Treatment(moves=)`); each person treated can move along only one transition of a disease |
| `TriggerLoop` | error | Moves with no wait (branches nobody waits in) lead back to a state they started from, so people passed through at once would never settle; names the loop |
| `ClockOutsideDomain` | error | A clock (`X.time`, `(A | B).time`, `X.progress`) is read where it has no value: by a transition out of a state outside its states, or by a modifier whose `where=` is not inside them (at build; a `where=` that is a function is checked when it runs, where a NaN amount is an error) |
| `NoScheduledDwell` | error | `X.progress` is read, but `X` has only rates out of it, so no dwell time is scheduled when people enter it |
| `PeopleAndCounts` | error | A transition moves between states of people and states that are counts (not people); names them, and says to leave out `people=False` if they are people |
| `InitWhereNobodyIs` | error | `init=` gives people a state only at levels of the stratifications where the split puts nobody, so nobody would start there (given at some such levels only, it is a note) |
| `CapabilityError` | error | The requested solver cannot run part of the model. Three cases: there is no sensible approximation (contact tracing, transmission trees, agent-only custom code); a function or condition asked for something that exists only per agent in a compartmental run; or an approximation would be needed and `approx=False`. Names the part, what it asked for, and what would fix it |
| `TimeNotKept` | error | A declared disease run as agents keeps the time each agent entered a state (`ti_X`) only for the states whose times the engine reads (for scheduled exits, clocks, and `after=`), those declared with `ss.State(keep_time=True)`, and all of them if `ss.options.keep_times = True`. Raised when code reads one of the others, e.g. an analyzer or a script after the run; names the array and says how to keep it |
| `TimeOutsideSim` | error | A time given to a delivery (`start=`, a round in `at=`), `ss.modify` (`start=`), or `ss.change` (`at=`) is before the start of the sim or after its end. A `stop=` after the end is allowed, and means it never stops. A number is a time in the sim's unit (in a sim with dates, that long after the start, or a whole year within the sim), and the message says so |
| `MissingDimension` | error (also a `KeyError`) | An `ss.transfer` names a stratification the model does not have (a boolean coordinate that does not exist yet is created instead); or a transition uses a value by level of a dimension (`ss.by()`) that the states it moves have no level of; or a value by position along a dimension (`ss.by(age=[...])`) that the sim is not stratified by |
| `AlreadyRunError` | error (also a `RuntimeError`) | A sim that has already run (or been finalized) is stepped, finalized, or added to again; `sim.run()` runs it again from the start |
| `DistNotInitializedError` | error (also a `RuntimeError`) | A distribution made with `strict=True` is sampled before it is initialized (by the sim, or with `dist.init()` or `dist.mock()`) |
| `DistSeedRepeatError` | error (also a `RuntimeError`) | Two distributions of a sim have the same seed, or a distribution is jumped back to a state it has passed, so that it would draw the same random numbers twice |
| `BooleanOperationError` | error (also a `NotImplementedError`) | A logical operation (and, or, xor, not) is applied to an array of people's values that is not Boolean, e.g. `ppl.age & ppl.female` |
| `NothingToFit` | error | `fit()` or `sensitivity()` is called and no parameter has `bounds` |
| `NoData` | error | `fit()`, `compare_models()`, or `log_density()` is called and no result has data |
| `NoPosterior` | error | `method='nuts'` or `method='pmcmc'` is asked for, and not every data series has a likelihood, or the sim cannot be sampled that way (no gradients, for NUTS; not a stochastic compartmental sim that the kernels run whole, for particle MCMC) |
| `TargetStatsNotRecovered` | warning | `net.check()` finds that a network does not reproduce a target statistic it was given |
| `AtBound` | warning | A best-fit value is at or near one of its bounds |
| `StageStopped` | warning | A stage of a staged fit did not converge; the sequence stops and the last completed fit is returned |
| `NotConverged` | warning | The chains of a posterior disagree (R-hat above 1.01), or some draws diverged |
| `UndrivenUnused` | warning | A transition with no rate, duration, or `p=` (undriven) is named by no product, delivery, or `on=` in the sim, so only custom code can move people along it |
| `LossyExport` | warning | An export to another format, or to code, left something out or changed it. Lists each item |
| `ImportIncomplete` | warning | An import could not translate something in the source. Lists each item by name, says what to ask an AI assistant with the `import-model` skill to finish it, and warns again when the sim is first run |
| `UnitInferred` | note | The sim's unit was not given with `unit=` and was taken from `dt`, from `start` and `stop`, or from the typed values. Says which unit and where it came from (`unit='day' (from start and stop dates)`). Not said when nothing has a unit: time is then in timesteps and nothing was inferred |
| `LargeStep` | note | `rate × dt > 0.1` for a converted rate, or a mean dwell time shorter than the step; says how far `rate × dt` is from `1 − exp(−rate × dt)` and points to `sim.check_dt()` |
| `MatrixAggregated` | note | In a compartmental run, a contact matrix whose age groups are not the model's was aggregated (or split) to the model's groups, weighted by population. Names the matrices (once for all of them, e.g. the layers of a country's) and both sets of groups. Agents use the matrix's own groups and get no note |
| `OverlappingSet` | note | Two `set=` modifiers are active for the same people at once; the one later in the loop is used. Names both |
| `ClippedToRange` | note | The modifiers of a rate, duration, or multiplier made it negative, so it was set to 0 where it was; or made the probabilities of the branches out of a state come to more than 1, so they were scaled to sum to 1. Said once per quantity |
| `NotCompiled` | note | A function given to `ss.jit` can't be compiled (or traced by JAX), so it runs as written; says why (e.g. what Numba couldn't compile). Said once per function |
| `SmallCounts` | note | `ode` with a disease that starts with only a few infected, so that whether it takes off is a matter of chance |
| `ExtinctionPossible` | note | `ode` on a model where the number infected falls so low that the epidemic could plausibly die out by chance |
| `StagesUnderAuto` | note | A compartmental run represented a dwell time that is not exponential or Erlang by fitted Erlang stages. Names the transition, the number of stages, and the mean and standard deviation kept. `stages=` on the transition silences it, and `approx=True` keeps it without printing it; with `approx=False` it is a `CapabilityError` instead, and `solver='auto'` runs the model as agents |
| `Unidentifiable` | note | Two fitted parameters appear in the model only as a product (or a ratio), or are almost perfectly correlated in the fit: over the draws of a posterior, or in how they change the fitted results at the best fit |

<!-- catalog end -->

## Conditions without a name

Some mistakes are ordinary Python usage errors and raise a plain `TypeError` or `ValueError` with a message in the same shape. They can be given names later if they turn out to be common:

- giving both `rate=` and `dur=` to a transition, or more than one of `multiply=`, `add=`, `set=` to `ss.modify`;
- a state's name where the state is meant (`ss.Transition(I, 'R', dur=10)`), which says to declare the states;
- a set of sources without parentheses (`S | I | R >> D @ mu`);
- using a lazy condition where Python needs a single truth value (`if`, `and`, `or`, `not`); the message says to use `&`, `|`, `~`;
- reading a result from a module that was given to a sim (`sir.results.I` after `ss.Sim(sir).run()`): the sim ran its own copy, so the `AttributeError` (or `KeyError`) says where the result is (`sim.results.sir.I`).

## Removed since the first draft

The first draft had about 75 codes and the second 54. Most of the ones that went were errors for mechanisms that no longer exist (queued updates, the closure ladder, declared reads and writes, the dict shorthand, predicate rank) or refusals that are now a guessed default or a printed approximation. In particular: a missing `n_agents` is guessed; a spatial dimension without mobility is an ordinary model; coverage and capacity on a stratified model are behaviors the library is required to get right ([05](05-dimensions.md#coverage-and-counts-on-a-stratified-model)), not errors; and running a network model as an ODE prints a note.

Since the second draft: `E901 ResultExists` went with result caching, which is deferred. `W101` was `UnitAssumed` ("no unit given, assuming days"); a model with no unit now runs in timesteps and the code reports an inferred unit instead. An unmatched data column was a warning (`W705`) and is an error (`E705`), now that `ss.simdata(aliases=, ignore_extra=)` gives a way to deal with it. Seven codes were added, for states that are not people, sim-level stratification, units, and `solver='auto'`: `W103`, `E210`, `E211`, `W212`, `E308`, `W404`, and `W1205`.

Since the review of October 2026: the numbers were dropped, and each error, warning, and note is a class named as before (`E804 CapabilityError` is `ss.CapabilityError`), so the catalog is made from the code. The codes that nothing raised went: `W103 UnitsDisagree`, `W207 UnreachableState`, `W309 LargeStateSpace`, `W1002 CoarseFlow`, `W1203 LeapConditionMarginal`, `W1204 DiscretizationCoarse`, `E1003 RunAfterFinish` (a finished sim raises `ss.AlreadyRunError`), and `E1401 SchemaTooNew`. Three that were raised but not listed were added: `UnknownEffect`, `UnknownDisease`, and `UnknownLocation`. `OverlappingSet` is a note, as the code has it, not a warning. Every note is kept in `sim.notes`, including under `verbose=0` and `approx=True`, which only stop it being printed.

## Documentation and tests

The catalog above is the documentation of each name: the docstring of its class, which `help(ss.errors.UnknownState)` also shows. A page per name with a minimal example that triggers it is not written yet. Every name SHOULD have a test that triggers exactly it ([18-conformance.md](18-conformance.md)); most errors have one, but about twenty names (most of the notes and warnings) are not yet triggered by name in any test ([deferred.md](../admin/deferred.md)).
