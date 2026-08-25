# Errors and warnings

Error messages are part of the language. A refusal that does not say what to do instead is a bug.

Justified by [best-practices/principles.md](../best-practices/principles.md) §Error messages are part of the language.

## Why this is a specification and not a style guide

A substantial fraction of models in this language will be written by an AI that will otherwise produce a wrong answer with complete confidence, and by epidemiologists who will read the message once and either fix the model or give up. Both audiences are served by the same thing: a message that names the rule, shows the offending declaration, and states the fix.

The evidence that this pays is that the one framework in the landscape with numbered, documented, epidemiologically-specific error codes uses them to prevent real modeling bugs — stratified coverage delivering `1 − (1 − f)^P` instead of `f`, a count applied per cell instead of in total — that other frameworks commit silently.

## Required shape

Every error and warning MUST have:

1. **A code and a name.** `E303 StratifiedCoverage`. The code is stable across versions and is documentable, searchable, and citable.
2. **A one-line statement of what is wrong**, in the user's vocabulary, naming the declaration by the name the user gave it.
3. **The offending declaration**, quoted, with its source location where available and a caret under the offending part.
4. **The rule**, stated as a sentence about epidemiology or arithmetic, not about the implementation.
5. **The fix**, as code the user could paste, or an explicit statement that there is no fix and why.

```text
E102 WrongDimension: rate expression for transition 'recovery' has dimension
     [dimensionless], expected [count/time].

       recovery = ss.progression('I -> R', rate=ss.peryear(0.1) * ss.days(3))
                                                                 ^^^^^^^^^^^
       Multiplying a rate by a duration removes the time dimension.

     A rate times a duration is an expected count, not a rate.
     Did you mean p= rather than rate=?
```

Forbidden: `ValueError: shape mismatch`, `KeyError: 'beta'`, a traceback through library internals as the primary message, and any message that names an internal class the user did not write.

## Severity

| Level | Means | Behavior |
|---|---|---|
| **Error** (`E`) | A guess would be wrong quietly | Raises. Never suppressible. |
| **Warning** (`W`) | A guess was made that is usually right and occasionally load-bearing | Printed in `summary()`, collected in `sim.warnings`, suppressible individually by code |
| **Note** | A default was filled in | Marked `[inferred]` in `summary()` only |

The rule for choosing: **an error is a claim that no guess is safe, and that claim has to be true.** Refusing to run when the answer could be guessed correctly 99% of the time discourages exactly the audience this language exists for.

Warnings are suppressible **by code, individually, with a reason**: `ss.Sim(..., allow=['W102'])`. Blanket suppression is not available.

## The catalogue

### E1xx — types and units

| Code | Name | Raised when |
|---|---|---|
| `E101` | `IncompatibleTypes` | An arithmetic combination not in the type lattice ([02-types.md](02-types.md)) |
| `E102` | `WrongDimension` | An expression's dimension does not match its slot |
| `E103` | `UntypedInTypedExpression` | A bare number combined with a typed quantity where the unit cannot be inferred |
| `E104` | `InconsistentParameterKind` | One `ss.Par` used in slots of two different kinds |
| `W101` | `UnitAssumed` | No unit was declared anywhere; days assumed |
| `W102` | `LargeStep` | `rate × dt > 0.1`; the exponential conversion matters here |

### E2xx — structure

| Code | Name | Raised when |
|---|---|---|
| `E201` | `ReservedName` | A declaration name collides with a module attribute |
| `E202` | `AmbiguousShorthand` | A dict-form tuple matches zero or two coercion rules |
| `E203` | `AmbiguousEntryState` | More than one state has no inbound transition and none has `init` |
| `E204` | `UnknownState` | A name in an arrow, `source=`, or `states=` does not resolve |
| `E205` | `AmbiguousBranchSplit` | `p=` cannot be resolved as timing or as a split ([03](03-states-and-transitions.md)) |
| `E206` | `DuplicateDeclaration` | Two declarations share a name |
| `E207` | `EmptyModel` | A module declares no states and no transitions |
| `W204` | `LikelyLatentState` | A transmission destination has an onward progression and sources no transmission — the SEIR shape with `source=` omitted |
| `W206` | `AbsorbingState` | A state has no outbound transition and is not declared `absorbing=True` |
| `W207` | `UnreachableState` | A state has no inbound transition and no `init` |

### E3xx — dimensions and stratification

| Code | Name | Raised when |
|---|---|---|
| `E301` | `UnknownTransition` | `adjust=` names a transition that does not exist |
| `E302` | `BadSplit` | A population split does not sum to 1, or has the wrong length |
| `E303` | `StratifiedCoverage` | A `coverage=` would be applied per stratum, delivering `1 − (1 − f)^P` |
| `E304` | `StratifiedCount` | A `count` on a bare stratified transfer would multiply the intended total |
| `E305` | `DuplicateStructuralDimension` | A second `age` or `strain` dimension |
| `E306` | `InvalidAdjustment` | A negative multiplier, or one that drives a probability above 1 |
| `E307` | `PartialStratificationMixing` | A mixing matrix over a dimension that does not cover every state |
| `E308` | `StrainMixing` | A mixing matrix on a strain dimension |
| `E309` | `StateSpaceTooLarge` | Over 10⁶ states in a compartmental backend |
| `W304` | `IsolatedPatches` | A spatial dimension with no mobility |
| `W309` | `LargeStateSpace` | Over 10⁴ states; names the dimensions and their sizes |

### E4xx — routes, mixing, and geography

| Code | Name | Raised when |
|---|---|---|
| `E401` | `NoInfectiousSource` | A transmission whose `source` states can never be occupied |
| `E402` | `NoGeography` | A mobility model with no geography to supply distances |
| `E403` | `RouteMismatch` | A transmission restricted to a route that does not exist |
| `E404` | `MalformedMixingMatrix` | Non-square, negative, or wrongly-sized for the dimension |
| `W401` | `UnanchoredDensity` | Density dependence with no `n_ref` and a changing population |
| `W402` | `TargetStatsNotRecovered` | `net.check()` finds realized statistics outside their interval |

### E5xx — effects and composition

| Code | Name | Raised when |
|---|---|---|
| `E501` | `UndeclaredWrite` | A module writes a quantity not in its `writes` |
| `E502` | `IncompatibleCombiners` | Two active `set=` effects on one target |
| `E503` | `EffectCycle` | Effects form a cycle within one step |
| `E504` | `UnmodifiableTarget` | The target is not in the closed target list |
| `E505` | `InvalidClamp` | `min=` above `max=` |
| `W503` | `MixedCombiners` | `multiply=` and `add=` on one target; names the resolution order |

### E6xx — parameters and data

| Code | Name | Raised when |
|---|---|---|
| `E601` | `MissingRequirement` | A declared `ss.Needs` cannot be satisfied |
| `E602` | `UnknownParameter` | An undeclared parameter name is assigned; lists near matches |
| `E603` | `ShapeError` | A value's shape does not fit the model's dimensions |
| `E604` | `UnvaluedParameter` | A declared `ss.Par` has no value from any source |
| `E605` | `UnknownLocation` | A named location does not resolve; lists near matches and providers |
| `E606` | `OutOfBounds` | A value outside its declared `bounds` |
| `W601` | `StaleData` | A cached dataset is older than its provider's update interval |

### E7xx — results and observation

| Code | Name | Raised when |
|---|---|---|
| `E701` | `OutcomeUsedAsState` | Something in the process reads an outcome |
| `E702` | `UnknownOutcomeSource` | An outcome's source names nothing |
| `E703` | `OutcomeCycle` | Outcomes chain in a cycle |
| `E704` | `DataMisaligned` | Observed data has no overlap with the simulated period |
| `W701` | `NoDenominator` | A characteristic that looks like a proportion has no denominator |

### E8xx — capability and backend

| Code | Name | Raised when |
|---|---|---|
| `E801` | `ResolutionUnset` | `n > 10⁶` with no `n_agents` given |
| `E802` | `DwellTimeUnrepresentable` | No exact realization in the chosen backend; lists the alternatives |
| `E803` | `TimeVaryingRateUnsupported` | A backend cannot handle a time-varying rate and would otherwise freeze it |
| `E804` | `CapabilityError` | Any refused (element × method) combination; names the element and the reason |
| `E805` | `UnderivedClosure` | A network-to-compartmental conversion for which no closure has been derived |
| `W1201` | `SmallCounts` | `sde` or `ode` with a compartment below ~20 |
| `W1202` | `ExtinctionPossible` | `ode` where the epidemic could plausibly go extinct |
| `W1203` | `LeapConditionMarginal` | Tau-leaping with propensities changing appreciably over τ |
| `W1204` | `DiscretizationCoarse` | `binomial` with `rate × dt` above 0.1 |
| `W1205` | `MeanFieldSubstituted` | A network route run under a compartmental method with a named closure |
| `W1206` | `ScaledPopulation` | `n_agents < n`; names the scale factor and the variance inflation |

### E9xx — reproducibility

| Code | Name | Raised when |
|---|---|---|
| `E901` | `ResultExists` | Writing a different result to an existing run identity |
| `E902` | `CRNContractViolated` | A CRN-keyed draw that is not a per-agent inverse CDF, when `crn_strict=True` |
| `W901` | `CRNNotHeld` | The same, as the default warning; names the draw |
| `W902` | `CacheHit` | A run was served from cache rather than recomputed |
| `W903` | `UncheckedConvergence` | A method with a convergence condition, run without `check_dt()` |

### E10xx — execution

| Code | Name | Raised when |
|---|---|---|
| `E1001` | `ConflictingUpdate` | Two queued updates write different values to the same quantity and index |
| `E1002` | `ModuleCycle` | Two modules each read the other's output within a step |
| `E1003` | `RunAfterFinish` | `run()` on a completed sim without `reset=True` |
| `W1001` | `ExplicitOrdering` | `substep()` was called; the model has an ordering dependence |
| `W1002` | `CoarseFlow` | A module's `dt` is coarser than the output grid |

### E11xx — interventions and delivery

| Code | Name | Raised when |
|---|---|---|
| `E1101` | `UnknownProduct` | A delivery names a product that does not exist |
| `E1102` | `MissingDimension` | A transfer names a dimension the model does not have |
| `W1101` | `CoverageUnreachable` | Coverage cannot be met from the eligible population; reports the achieved fraction |
| `W1102` | `NoHysteresis` | A threshold trigger with no `min_duration`; policies will flap at the step frequency |

### E13xx — inference

| Code | Name | Raised when |
|---|---|---|
| `E1301` | `NothingToFit` | `calibrate()` with no parameter carrying `bounds` or `prior` |
| `E1302` | `NoLikelihood` | `calibrate()` with no outcome carrying `data` and `likelihood` |
| `E1303` | `StageGateFailed` | A staged fit failed its convergence gate; the sequence stops |
| `W1301` | `AtBound` | A best-fit value at or near a declared bound |
| `W1302` | `Unidentifiable` | Free parameters that enter the model only as a product |

### E14xx — schema and IR

| Code | Name | Raised when |
|---|---|---|
| `E1401` | `SchemaTooNew` | The document's major version exceeds the reader's |
| `E1402` | `NormalizationNotIdempotent` | An internal invariant failed; always a bug in the implementation |
| `W1401` | `LossyExport` | An export dropped or approximated declarations; lists them |

### E16xx — interoperation

| Code | Name | Raised when |
|---|---|---|
| `E1601` | `UnclassifiableRate` | An imported rate expression cannot be classified into the transition vocabulary |
| `E1602` | `AmbiguousDenominator` | An imported model's dependence convention cannot be determined |
| `W1601` | `ImportIncomplete` | Constructs in the source could not be translated; lists them by name |

## Documentation obligation

Every code in this catalogue MUST have a documentation page reachable offline at the shipped version, containing the rule, a minimal failing example, the fix, and the reasoning. `ss.explain('E303')` prints it. A code whose page says only what the message says has not been documented.

This is the AI-native requirement applied to errors: an agent that reads stale or absent documentation writes broken models, and the error catalogue is the part of the documentation it will read most.

