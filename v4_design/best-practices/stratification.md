# Stratification

## Recommendation

1. **One concept covers age, space, risk, strain, vaccination status, and any agent property.** They are all dimensions; the difference is which structure they carry, not what kind of thing they are.
2. **Stratification is an operator applied to a finished model, not a rewrite of it.** summer2 proves this works and shows exactly what it costs.
3. **The base model stays stratum-agnostic; the stratification supplies multiplicative adjustments.**
4. **Some dimensions carry structure — age implies ageing, space implies movement, strain implies a restructured force of infection.** Mark which kind a dimension is; derive the rest.
5. **In an agent-based backend, a dimension is an agent property. In a compartmental backend, it is a compartment axis. Same declaration.** This is the most useful unification available here.
6. **Enforce the restrictions that follow from the structure**, with named errors.

## Why

### Four frameworks, one operation

[MetaCast](../approaches/metacast.md) states the general case most plainly: stratification, metapopulation structure, and any other partition are the same operation — a dimension the model is replicated over. [summer2](../approaches/summer2.md) implements it as a transformation (`stratify_with`). [camdl](../approaches/camdl.md) has `dimensions { age = [child, adult] }` plus `stratify(by = age)`. [flepiMoP](../approaches/flepimop.md) declares compartments as a Cartesian product of named axes. [MIRA](../approaches/notes.md#askem-model-representation-amr--mira) attaches a `context` dict to each Concept and operates on it with a generic `stratify()`.

And [Kendrick](../approaches/notes.md#kendrick), in 2015, made the same argument as the organizing principle for the whole field: an epidemiological model is a composition of independent concerns, combined mechanically rather than interleaved.

The other end of the spectrum is [odin](../approaches/odin.md), where stratification means array indices and "adding an age dimension to an odin model means rewriting every equation with indices". Its own review: "The lingua franca probably needs both — an operator for the common cases, explicit indexing underneath it — with the operator documented as sugar that expands to the indexed form."

### Why the operator needs a closed transition vocabulary

From the [summer2 review](../approaches/summer2.md):

> **A closed vocabulary of named flow kinds.** … An open "rate expression" cannot be stratified automatically; a named flow kind can.

This is the load-bearing connection between this document and [model-structure.md](model-structure.md). The framework can split a transition correctly only if it knows what the transition *means*. That is the entire justification for `ss.transmission` and `ss.progression` being named kinds rather than free rate expressions, and it is why `ss.process` escape hatches are not stratifiable.

### The ABM/compartmental unification nobody states

[EMOD](../approaches/emod.md)'s Individual Properties are user-defined categorical agent attributes (`Risk: HIGH/MEDIUM/LOW`) declared in demographics and targetable by interventions. [EpiHiper](../approaches/epihiper.md)'s traits are the same. [HPVsim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) added genotype "as a state dimension rather than by duplicating the disease module". [MetaCast](../approaches/metacast.md)'s coordinates and [camdl](../approaches/camdl.md)'s strata are the compartmental version.

From the [EMOD implications](../approaches/emod.md): "This is the agent-based counterpart to a stratification dimension, and the lingua franca should treat them as the same concept expressed at different aggregation levels."

Nobody currently does. This is a genuine contribution available cheaply.

## The proposal

### The operator

**summer2** ([approaches/summer2.md](../approaches/summer2.md)):

```python
age_strat = Stratification("age", agegroup_keys, ["S", "I", "R"])
age_strat.set_population_split({...})
age_strat.set_flow_adjustments("recovery", rec_adj)
age_strat.add_infectiousness_adjustments("I", {...})
age_strat.set_mixing_matrix(mm)
m.stratify_with(age_strat)
```

**camdl:**

```camdl
dimensions { age = [child, adult] }
stratify(by = age)
```

**Lingua franca** — one call, keyword per adjustment:

```python
sir.stratify(
    age = [0, 15, 65],                       # bin edges, or a list of labels
    split = [0.3, 0.55, 0.15],               # population split; defaults to the demography
    adjust = {'recovery': [1.5, 1.0, 0.5]},  # per-stratum multipliers on named transitions
    mixing = ss.contacts('Kenya'),           # who meets whom
)
```

and then, because the base model was never rewritten, this still works and now means "incidence in children":

```python
sim.results.infection[age=0]
```

The base model says "recovery happens at `dur_inf`"; the stratification says "×1.5 for children". Neither knows about the other. That factoring is summer2's, and it is the reason one model text serves many structures.

Multiple stratifications compose, each seeing the model as the previous left it:

```python
sir.stratify(age=[0, 15, 65], mixing=ss.contacts('Kenya'))
sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
```

### Dimensions that carry structure

summer2 special-cases `AgeStratification` (generates ageing flows) and `StrainStratification` (restructures the force of infection). camdl distinguishes population strata from *residence structure* (`via erlang`). [LASER](../approaches/laser.md) puts `node_id` in the core rather than treating space as one dimension among many.

Both summer2 and camdl are recognizing the same thing: **dimensions are not interchangeable, and the language needs a way to say which kind a dimension is.** The kind should be inferred from the name where possible, because `age` is always age:

| Dimension kind | Implied structure | Declared as |
|---|---|---|
| `age` | Ageing flows between adjacent strata, at the right rate | `sir.stratify(age=[0, 15, 65])` |
| `patch` / space | Movement between coordinates; a mobility model | `sir.stratify(patch=..., mobility=ss.gravity(...))` |
| `strain` | Force of infection restructured per strain; cross-immunity | `sir.stratify(strain=['wild','alpha'], immunity=...)` |
| plain | Nothing implied | `sir.stratify(risk=['low','high'])` |

An `age` dimension that generated no ageing flows would be wrong, and asking the user to write the ageing flows by hand is asking them to make an error that the framework can prevent.

### Moving between coordinates

The transition kind from [model-structure.md](model-structure.md) that exists specifically for this. **MetaCast** ([approaches/metacast.md](../approaches/metacast.md)):

```python
{"from_coordinates": ["high", "unvaccinated"],
 "to_coordinates":   ["high", "vaccination_lag"],
 "states":           "all",
 "parameter":        "nu_unvaccinated"}
```

**Lingua franca:**

```python
ss.transfer(vaccinated='no -> yes', rate=ss.peryear(0.2))
```

The individual does not change disease state; they change dimension coordinate, for all disease states at once. This is what vaccination, ageing, and migration *are*, and it means the disease model never mentions vaccination. See [interventions.md](interventions.md), where this becomes the mechanism for most of the intervention vocabulary.

### The ABM and compartmental readings

The same declaration, two realizations:

```python
sir.stratify(risk=['low', 'high'], adjust={'infection': [1.0, 3.0]})
```

| Backend | What it does |
|---|---|
| ODE / CTMC | Splits `S`, `I`, `R` into `S_low`, `S_high`, … and splits every transition, applying the adjustment |
| Agent-based | Adds a per-agent `risk` property with the given split, and scales `rel_sus` by the adjustment |

Nothing in the model text changes. This is the same idea as [EMULSION](../approaches/emulsion.md)'s one-word paradigm switch, applied to the dimension rather than the model, and it is why an EMOD Individual Property and a camdl stratum are the same declaration.

The agent-based case gets a bonus: a dimension can be continuous. `age` in an ABM need not be binned at all, and the binning is only applied when the backend requires it or the output requests it. **Bin at the boundary, not in the model.**

### Restrictions, enforced and explained

summer2 enforces: a mixing matrix requires a *complete* stratification; `AgeStratification` may be applied once; `StrainStratification` may be applied once; strains cannot carry a mixing matrix. Its review is right that "a mixing matrix requires a complete stratification is a correctness property, not an implementation limit".

camdl's errors are the model for how to deliver these. E238 rejects a `count` on a bare stratified transfer "because it would multiply the intended total by the number of cells"; E239 rejects a bare endpoint inside an indexed family "because realised coverage would be `1 − (1 − f)^P` rather than `f`". From its review: "Each of these is a real modelling bug that other frameworks silently commit."

```text
StratifyError: mixing matrix given for dimension 'age', but 'age' does not cover every state.
  'D' (deaths) is unstratified. A mixing matrix over a partial stratification is undefined:
  there is no age for the people in 'D' to mix as.
  Fix: sir.stratify(age=..., states='all') or drop mixing= from this call.
```

### Adjustments are multiplicative only

summer2 has `Multiply` and `Overwrite`. Its own implications section asks "consider whether `Overwrite` should exist at all, given how it interacts with multiple stratifications" — and the answer is no. Multiplicative adjustments commute, so the result does not depend on stratification order; overwrites do not, so it does. Anyone who needs an overwrite is describing a different transition, and should write one.

## Trade-offs

- **Stratification is order-dependent even with multiplicative adjustments**, because the *set of transitions* being adjusted changes as strata are added. summer2 enforces ordering constraints "but they are not derivable from the model text — you have to know them". Mitigation: print the resolved stratified model, and refuse the combinations that are genuinely wrong.
- **The operator only supports what it implements.** odin's arrays are maximally general and maximally laborious. The escape hatch here is explicit per-stratum transitions, written out; the printed summary shows a hand-written transition alongside the generated ones, so the model stays legible.
- **Strata multiply.** Age (16) × risk (2) × strain (3) × patch (47) is 4,512 compartments, and someone will do it by accident. Warn on the compartment count at build time, and note that the agent-based backend does not pay this cost — which is a real, statable reason to switch paradigm.
- **Continuous dimensions in ABM and binned dimensions in ODE are not the same model.** Binning is an approximation and the conversion should say so ([paradigm-conversion.md](paradigm-conversion.md)).

## Rejected

- **Manual array indexing as the only mechanism** (odin). General, laborious, and it makes the model unstratifiable by any tool.
- **`Overwrite` adjustments** (summer2), for the reason above.
- **Stratification by writing out the compartments** ([epipack](../approaches/epipack.md), [Epydemix](../approaches/epydemix.md)). "Age structure means writing out `S_child`, `S_adult` and every process between them by hand."
- **A separate metapopulation concept.** Space is a dimension that carries movement structure, not a different kind of model. See [metapopulation.md](metapopulation.md).
- **Ontology-identified strata as a requirement** (MIRA's `context` keyed to ontologies). Optional.

## Open questions

- **Is strain a dimension or a second disease?** summer2 special-cases it; HPVsim made it a state dimension; Starsim would make it a second `Disease` module. Probably a dimension, but cross-immunity between strains is an effect between coordinates of the same dimension, which is a shape the [effects mechanism](composition-and-effects.md) does not obviously have.
- **Where does per-agent continuous heterogeneity sit?** `rel_sus ~ lognorm(1, 0.3)` is not a dimension and not a state. In an ODE backend it has no representation without adding a dimension or accepting a mean-field approximation. This needs a named answer, not silence.
- **Partial stratification of transitions**, not just states: camdl's `only = [R]`. Does the operator need it, or is it a sign the model wanted two transitions?
