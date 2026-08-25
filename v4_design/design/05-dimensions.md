# Dimensions and stratification

Age, space, risk, strain, vaccination status, and any agent property are one concept. Stratification is an operator applied to a finished model.

Justified by [best-practices/stratification.md](../best-practices/stratification.md).

## The operator

```python
Model.stratify(
    <name>  = levels,           # exactly one dimension per call
    *,
    split   = None,             # population split across levels; defaults to the demography
    adjust  = None,             # {transition_name: [multiplier per level]}
    states  = 'all',            # which states this dimension applies to
    mixing  = None,             # a route over this dimension's levels
    mobility = None,            # a movement model (space only)
    immunity = None,            # a cross-protection matrix (strain only)
    kind    = None,             # override the inferred dimension kind
)
```

```python
sir.stratify(
    age    = [0, 15, 65],
    split  = [0.30, 0.55, 0.15],
    adjust = {'recovery': [1.5, 1.0, 0.5]},
    mixing = ss.contacts('Kenya'),
)
```

One call per dimension. Multiple calls compose, each seeing the model as the previous left it:

```python
sir.stratify(age=[0, 15, 65], mixing=ss.contacts('Kenya'))
sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
```

`stratify()` returns the model, so calls may be chained, and it mutates in place. It is one of exactly two structural mutators in the language; the other is `observe()`.

### What it does

`stratify()` MUST:

1. Split every state in `states` into one state per level.
2. Split every transition touching those states, preserving its kind, its name, and its rate or dwell-time declaration.
3. Apply each `adjust` multiplier to the named transition in the corresponding level.
4. Extend the mixing categories, if a `mixing` is given.
5. Generate the structural transitions implied by the dimension's kind (below).
6. Leave the base model's declarations textually untouched.

The base model says "recovery happens at `dur_inf`"; the stratification says "×1.5 for children". Neither knows about the other. That factoring is what lets one model text serve many structures, and it is the reason the transition vocabulary is closed: the operator can split a transition correctly only if it knows what the transition means.

An `adjust` naming a transition that does not exist raises `E301 UnknownTransition` and lists the transitions that do. A `split` that does not sum to 1, or whose length does not match the level count, raises `E302 BadSplit`.

## Levels

```python
sir.stratify(age=[0, 15, 65])                 # bin edges: 0-14, 15-64, 65+
sir.stratify(age=[0, 15, 65], top=100)        # explicit upper bound
sir.stratify(risk=['low', 'high'])            # labels
sir.stratify(patch=ss.geo('Kenya', level='county'))   # a named source
```

A numeric list is bin edges when the dimension is continuous (age, viral load) and labels otherwise. Where this is ambiguous, `kind=` settles it.

**Bin at the boundary, not in the model.** In an agent-based backend a continuous dimension need not be binned at all: `age` is a per-agent float, and the bins are applied only when a compartmental backend requires them or an output requests them. `sim.results.infection.sel(age=0)` therefore works in both backends and means the same thing, computed differently.

## Dimension kinds

Dimensions are not interchangeable. Some carry structure, and generating that structure is the operator's job — asking the user to write ageing flows by hand is asking them to make an error the framework can prevent.

| Kind | Inferred from name | Implied structure |
|---|---|---|
| `age` | `age` | Ageing transfers between adjacent levels, at the rate implied by the bin widths |
| `space` | `patch`, `node`, `region`, `district`, `county`, or an `ss.geo` value | Movement between levels via a mobility model ([06-routes.md](06-routes.md)) |
| `strain` | `strain`, `variant`, `genotype`, `serotype` | Force of infection restructured per strain; a cross-immunity matrix |
| `plain` | anything else | Nothing implied |

The kind is inferred from the name, because `age` is always age. `kind=` overrides. An `age` dimension that generated no ageing transfers would be wrong; a `patch` dimension with no mobility is legal but emits `W304 IsolatedPatches`, because unconnected patches are almost always an oversight.

### Age

Ageing transfers are generated as `ss.transfer(age='0-14 -> 15-64', rate=1/width)` for each adjacent pair, applied to all states. The generated transfers appear in `summary()` and in the IR, and are ordinary transitions thereafter — they can be adjusted, observed, and effected like any other. An age dimension MAY be applied only once per model; a second raises `E305 DuplicateStructuralDimension`.

### Space

See [06-routes.md](06-routes.md) §Mobility. Space is a dimension carrying movement structure, not a separate kind of model, and a metapopulation is therefore a one-line addition to any model:

```python
sir.stratify(patch=ss.geo('Kenya', level='county'), mobility=ss.gravity(k=0.5, a=1, b=1, c=2))
```

### Strain

A strain dimension restructures the force of infection: each strain has its own infectious states and its own λ, and susceptibility depends on infection history through the cross-immunity matrix.

```python
sir.stratify(strain=['wild', 'alpha'], immunity=[[1.0, 0.7], [0.7, 1.0]])
```

`immunity[i][j]` is the protection conferred by prior infection with strain `i` against strain `j`. This restructuring MUST be derived from the declaration rather than special-cased per model, and it is the one dimension kind whose semantics extend the force-of-infection expression in [06-routes.md](06-routes.md). Whether strain is best treated as a dimension or as a second disease module is settled here in favor of a dimension, with the reasoning and the residual risk recorded in [19-open-questions.md](19-open-questions.md).

## Moving between coordinates

`ss.transfer` ([03-states-and-transitions.md](03-states-and-transitions.md)) is the transition kind that exists for dimensions:

```python
ss.transfer(vaccinated='no -> yes', rate=ss.peryear(0.2))
```

The individual does not change disease state; they change dimension coordinate, for all disease states at once. This is the mechanism underneath most of [08-interventions.md](08-interventions.md), and it is why vaccination need not be represented as extra compartments in the disease model.

## Adjustments are multiplicative only

`adjust` values are multipliers. There is no overwrite.

Multiplicative adjustments commute, so the result does not depend on the order in which dimensions were applied. Overwrites do not commute, so with two stratifications the result depends on which was applied first — a dependence that is invisible in the model text. Anyone who needs an overwrite is describing a different transition and should write one.

An adjustment value that is negative, or that would drive a probability above 1, raises `E306 InvalidAdjustment`. Adjustments compose with `ss.Effect` multipliers on the same transition by multiplication, in either order ([07-effects.md](07-effects.md)).

## Restrictions, enforced and explained

These are correctness properties, not implementation limits, and each is a named error:

| Restriction | Error | Why |
|---|---|---|
| A mixing matrix requires a complete stratification | `E307 PartialStratificationMixing` | There is no coordinate for the unstratified people to mix as |
| At most one `age` dimension | `E305 DuplicateStructuralDimension` | Two ageing processes on one population is not a model |
| At most one `strain` dimension | `E305` | The force-of-infection restructuring is not composable with itself |
| A strain dimension carries no mixing matrix | `E308 StrainMixing` | Strains do not meet each other; hosts do |
| `coverage=` on a stratified transfer is not applied per stratum | `E303 StratifiedCoverage` | Realized coverage would be `1 − (1 − f)^P`, not `f` |
| A `count` on a bare stratified transfer | `E304 StratifiedCount` | It would multiply the intended total by the number of cells |

```text
E307 PartialStratificationMixing: mixing matrix given for dimension 'age',
     but 'age' does not cover every state.
       'D' (deaths) is unstratified.
     A mixing matrix over a partial stratification is undefined: there is no age
     for the people in 'D' to mix as.
     Fix: sir.stratify(age=..., states='all'), or drop mixing= from this call.
```

`E303` and `E304` encode real modeling bugs that most frameworks commit silently. They are worth their implementation cost on that basis alone.

## Realization per backend

The same declaration, two realizations, and this unification is the most useful one available in the design:

| Backend | Realization |
|---|---|
| ODE / CTMC / tau / SDE | Splits every state into one per level; splits every transition; applies the adjustment |
| Agent-based | Adds a per-agent property with the given split; scales the corresponding per-agent quantity by the adjustment |

An agent property and a compartment axis are the same declaration at different aggregation levels. A categorical agent attribute targetable by interventions — EMOD's Individual Properties, EpiHiper's traits — is a stratification dimension, and treating them as one concept is a genuine simplification available at no cost.

## Size

Strata multiply. Age (16) × risk (2) × strain (3) × patch (47) is 4,512 states before anything interesting happens, and someone will do this by accident.

- The build MUST report the resolved state count in `summary()`.
- Above 10,000 states, `W309 LargeStateSpace` names the dimensions and their sizes, and states that the agent-based backend does not pay this cost.
- Above 1,000,000 states, the compartmental backends raise `E309 StateSpaceTooLarge` rather than attempting it.

The last point is a statable, quantitative reason to switch paradigm, which is exactly the kind of guidance the capability system exists to give.

## Rejected

- **Manual array indexing as the mechanism.** Maximally general, maximally laborious, and it makes the model unstratifiable by any tool. The operator is not sugar over user-written indices; the indices are an implementation detail the user never sees.
- **`Overwrite` adjustments.** Order-dependent.
- **Stratification by writing out the compartments.** `S_child`, `S_adult`, and every process between them, by hand.
- **A separate metapopulation concept.** Space is a dimension that carries movement.
- **Required ontology identifiers on strata.** Optional.
