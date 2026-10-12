# Errors, warnings, and notes

What the library says when something is wrong, doubtful, or guessed, and the list of every coded message.

Justified by [best-practices/principles.md](../best-practices/principles.md).

## The policy

**Prefer guessing a default and printing it to raising an error.** Most things a user leaves out have a sensible default, and stopping the run to ask for it helps nobody. An error is for the cases where there is no sensible guess: a name that does not exist, lengths that do not match, a request the run method cannot meet.

A message is read once, by an epidemiologist or by an AI writing a model for one. Either way it should say what is wrong in the user's own terms, show the line, and say what to do.

## Message shape

Every coded error SHOULD have:

1. **A code and a name**, such as `E602 UnknownParameter`. Codes are stable across versions so they can be searched for and documented.
2. **One line saying what is wrong**, using the names the user gave things.
3. **The declaration involved**, quoted, with a caret under the relevant part where that helps.
4. **What to do about it**, as code the user could paste where possible, with the closest matching names for anything misspelled.

```text
E102 WrongDimension: the rate for transition 'recovery' has no time dimension.
       rate = ss.peryear(0.1) * ss.days(3)
                               ^^^^^^^^^^^ multiplying by a duration removes the "per time"
     A rate times a duration is an expected count, not a rate. Did you mean p= instead of rate=?
```

Avoid `ValueError: shape mismatch`, `KeyError: 'beta'`, and a traceback through library internals as the main message. Notes are one line each and need only say what was assumed or done.

## Levels

| Level | Means | What happens |
|---|---|---|
| **Error** (`E`) | There is no sensible guess, or the request cannot be met | Raises an exception |
| **Warning** (`W`) | The run can proceed, but something probably needs attention | Printed through Starsim's `ss.warn`; the run continues |
| **Note** (`W`, marked "note" below) | Something was assumed, approximated, or is worth knowing | One printed line; `verbose=0` hides it |

Guessed defaults are not coded at all: `describe()` marks them `[guessed]` ([01-objects.md](01-objects.md#describe)). Approximations made when a method is requested explicitly are uncoded one-line notes as well ([12-backends.md](12-backends.md#approximations)); they turn into `E804` only under `approx=False`.

Warnings and notes use Python's `warnings` machinery, as Starsim does today, so they can be filtered or turned into errors in the usual way. Notes keep a `W` code so that they can be looked up.

## The catalog

Every code cited anywhere in this folder is listed here.

### 1xx: types and units

| Code | Name | Level | When |
|---|---|---|---|
| `E101` | `IncompatibleTypes` | error | Typed quantities are combined in a way that has no meaning, such as adding an `ss.per` to an `ss.prob` or `ss.freq` ([02](02-types.md#arithmetic)) |
| `E102` | `WrongDimension` | error | A fully typed expression has the wrong dimension for its slot (a rate times a duration in `rate=`), or both `beta` and `n_contacts` carry a time unit ([06](06-routes.md#units)) |
| `W101` | `UnitAssumed` | note | No unit was given anywhere; days were assumed |
| `W102` | `LargeStep` | note | `rate × dt > 0.1` for a converted rate; says how far `rate × dt` is from `1 − exp(−rate × dt)` and points to `sim.check_dt()` |

### 2xx: states and transitions

| Code | Name | Level | When |
|---|---|---|---|
| `E201` | `ReservedName` | error | A state, transition, parameter, or result is named like an attribute of `ss.Module` (`step`, `results`, `pars`, `describe`, `stratify`, `name`, `dt`, ...) |
| `E204` | `UnknownState` | error | States were declared explicitly and an arrow string or `source=` names one that was not. Lists the closest declared names. Not raised when states are inferred |
| `E205` | `BadBranchSplit` | error | Constant branch probabilities out of one state do not sum to 1 |
| `E206` | `DuplicateDeclaration` | error | Two members of a module are given the same explicit name, or a state is an endpoint of transitions in two modules |
| `E208` | `NonlinearExposure` | error | A rate is not linear in its bare states (`I ** 2`, `I / (I + K)`, or a term containing two bare states). Says to use `I.n` if a literal count was meant |
| `E209` | `UndrivenTransition` | error | A transition reaches build time with no rate, no duration, and no `p=` |
| `W207` | `UnreachableState` | note | A state has no transition into it and starts empty, which is usually a typo |

### 3xx: stratification

| Code | Name | Level | When |
|---|---|---|---|
| `E301` | `UnknownTransition` | error | `adjust=` names a transition that does not exist. Lists the ones that do |
| `E302` | `BadSplit` | error | `split=` has the wrong number of entries. (A split that does not sum to 1 is normalized, and `describe()` says so) |
| `E305` | `DuplicateStructuralDimension` | error | The model is stratified by age twice, or by any dimension it already has |
| `E306` | `InvalidAdjustment` | error | A negative multiplier in `adjust=`. (A multiplier that pushes a probability above 1 is clipped, with a note) |
| `E307` | `PartialStratificationMixing` | error | A state that a dimension does not apply to is the origin or the source of a transmission, so there is no coordinate for those people to mix as |
| `W309` | `LargeStateSpace` | warning | The stratified model has more than about 10,000 compartments. Names the dimensions and sizes and notes that an agent-based run does not grow this way |

### 4xx: mixing and geography

| Code | Name | Level | When |
|---|---|---|---|
| `E402` | `NoGeography` | error | A movement model such as `ss.gravity()` is used on a dimension with no geography to supply distances |
| `E403` | `RouteMismatch` | error | A key of a per-layer `beta` dict matches no network or layer |
| `W402` | `TargetStatsNotRecovered` | warning | `net.check()` finds that a network does not reproduce a target statistic it was given |

### 5xx: modifiers

| Code | Name | Level | When |
|---|---|---|---|
| `E504` | `UnmodifiableTarget` | error | The target of `ss.modify` or `ss.change` cannot be resolved to a named numeric quantity (a plain number, a boolean state, a name that does not exist). Lists the closest names |
| `W502` | `OverlappingSet` | warning | Two `set=` modifiers are active for the same people at once; the one later in the loop is used. Names both |

### 6xx: parameters and data

| Code | Name | Level | When |
|---|---|---|---|
| `E602` | `UnknownParameter` | error | A parameter name the module does not have is set, directly or through `ss.Sim(pars=...)`. Lists the closest names |
| `E603` | `ShapeError` | error | A value's length does not match its stratification (four values for three age groups). Names the parameter, the transition, and both lengths |
| `E604` | `UnvaluedParameter` | error | An `ss.par` has no value, no bounds to start from, and nothing supplied through `pars=` |
| `E605` | `UnknownLocation` | error | A place name cannot be resolved by any data provider. Lists near matches and the providers searched |
| `E606` | `OutOfBounds` | error | A parameter's value is outside its `bounds` |

### 7xx: results and data

| Code | Name | Level | When |
|---|---|---|---|
| `E702` | `UnknownOutcomeSource` | error | The source of an `ss.outcome` names nothing in the model |
| `E703` | `OutcomeCycle` | error | Outcomes chain back to themselves |
| `E704` | `DataMisaligned` | error | The data lies entirely outside the simulated period. (Data partly outside is used where it overlaps, with a note) |
| `W705` | `UnknownDataColumn` | warning | A data column matches no result. It is reported once, with the closest result names, and ignored |

### 8xx and 12xx: run methods

| Code | Name | Level | When |
|---|---|---|---|
| `E804` | `CapabilityError` | error | The requested run method cannot run part of the model. Three cases: there is no sensible approximation (contact tracing, transmission trees, agent-only custom code); a function or condition asked for something that exists only per agent in a compartmental run; or an approximation would be needed and `approx=False`. Names the part, what it asked for, and what would fix it |
| `W1201` | `SmallCounts` | note | `ode` or `sde` with a compartment holding only a few people |
| `W1202` | `ExtinctionPossible` | note | `ode` on a model where the epidemic could plausibly die out by chance |
| `W1203` | `LeapConditionMarginal` | note | `tau` with rates changing quickly within a leap |
| `W1204` | `DiscretizationCoarse` | note | `binomial` or `abm` with `rate × dt` large |

Each of the four notes names the quantity involved and a method that would not have the problem ([12-backends.md](12-backends.md#the-methods)).

### 9xx: reproducibility

| Code | Name | Level | When |
|---|---|---|---|
| `E901` | `ResultExists` | error | Only when result caching is on: a different result would overwrite a stored one with the same run hash |

### 10xx: execution

| Code | Name | Level | When |
|---|---|---|---|
| `E1003` | `RunAfterFinish` | error | `run()` is called on a sim that has already finished. Copy the sim or build a new one |
| `W1002` | `CoarseFlow` | note | A flow from a module with a coarse `dt` is requested on a finer grid and has been spread evenly across the interval |

### 11xx: interventions

| Code | Name | Level | When |
|---|---|---|---|
| `E1102` | `MissingDimension` | error | An `ss.transfer` names a stratification the model does not have. (A boolean coordinate that does not exist yet is created instead) |

### 13xx: calibration

| Code | Name | Level | When |
|---|---|---|---|
| `E1301` | `NothingToFit` | error | `calibrate()` is called and no parameter has `bounds` |
| `E1302` | `NoData` | error | `calibrate()` is called and no result has data |
| `W1301` | `AtBound` | warning | A best-fit value is at or near one of its bounds |
| `W1302` | `Unidentifiable` | note | Two fitted parameters appear in the model only as a product, or are almost perfectly correlated in the fit |
| `W1303` | `StageStopped` | warning | A stage of a staged fit did not converge; the sequence stops and the last completed fit is returned |

### 14xx and 16xx: model spec and other formats

| Code | Name | Level | When |
|---|---|---|---|
| `E1401` | `SchemaTooNew` | error | A model spec was written by a newer major version than the reader |
| `W1401` | `LossyExport` | warning | An export to another format left something out or changed it. Lists each item |
| `W1601` | `ImportIncomplete` | warning | An import could not translate something in the source. Lists each item by name |

## Conditions without a code

Some mistakes are ordinary Python usage errors and raise a plain `TypeError` or `ValueError` with a message in the same shape. They can be given codes later if they turn out to be common:

- giving both `rate=` and `dur=` to a transition, or more than one of `multiply=`, `add=`, `set=` to `ss.modify`;
- a positional value after an arrow string (`ss.transmission('S -> I', 0.05)`);
- a set of sources without parentheses (`S | I | R >> D @ mu`);
- an unknown key in a product's `efficacy` dict (the message lists the valid keys);
- using a lazy condition where Python needs a single truth value (`if`, `and`, `or`, `not`); the message says to use `&`, `|`, `~`.

## Removed since the first draft

The first draft had about 75 codes. Most of the ones that went were errors for mechanisms that no longer exist (queued updates, the closure ladder, declared reads and writes, the dict shorthand, predicate rank) or refusals that are now a guessed default or a printed approximation. In particular: a missing `n_agents` is guessed; a spatial dimension without mobility is an ordinary model; coverage and capacity on a stratified model are behaviors the library is required to get right ([05](05-dimensions.md#coverage-and-counts-on-a-stratified-model)), not errors; and running a network model as an ODE prints a note.

## Documentation

Every code SHOULD have a documentation page, shipped with the library and readable offline, giving a minimal example that triggers it and the fix. Every code MUST have a test that triggers exactly it ([18-conformance.md](18-conformance.md)); a code with no test is removed from this list.
