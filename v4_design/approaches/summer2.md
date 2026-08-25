# summer2

|  |  |
|---|---|
| **Language** | Python, with a JAX computational backend |
| **Paradigms** | Compartmental — ODE (default) and stochastic |
| **Specification style** | Declarative model object; structure built by method calls, then *transformed* by stratification |
| **Version reviewed** | `summerepi2` (Monash EMU) |
| **Licence** | BSD-2-Clause |
| **Code** | <https://github.com/monash-emu/summer2> |
| **Docs** | <https://summer2.readthedocs.io/> |
| **Paper** | — (documented; used across Monash EMU's TB, COVID-19 and other work) |

## What it's for

summer2 builds stratified compartmental models. Its distinctive claim is not the compartments — those are ordinary — but **stratification as a transformation applied to a finished model**. You build an unstratified SIR, then apply an age stratification, then a vaccination stratification, then a strain stratification, and summer2 rewrites the compartments, the flows, the mixing, and the outputs accordingly.

This is the most developed answer anywhere in the review to the question "how do you add a dimension to a model without rewriting it", and that question is close to the centre of what the lingua franca has to solve.

## How a model is specified

Build the base model:

```python
from summer2 import CompartmentalModel, Stratification
from computegraph.types import param

m = CompartmentalModel(times=[0, 100], compartments=["S", "I", "R"], infectious_compartments="I")
m.set_initial_population({"S": 990.0, "I": 10.0})
m.add_infection_frequency_flow("infection", param("contact_rate"), "S", "I")
m.add_transition_flow("recovery", param("recovery_rate"), "I", "R")

incidence = m.request_output_for_flow("incidence", "infection")
m.request_function_output("notifications", incidence * param("cdr"))
m.set_default_parameters({"contact_rate": 0.4, "recovery_rate": 0.1, "cdr": 0.2})
```

Then stratify:

```python
age_strat = Stratification("age", agegroup_keys, ["S", "I", "R"])
age_strat.set_population_split({...})                  # how to split the existing population
age_strat.set_flow_adjustments("recovery", rec_adj)    # per-stratum flow multipliers
age_strat.add_infectiousness_adjustments("I", {...})   # per-stratum infectiousness
age_strat.set_mixing_matrix(mm)                        # heterogeneous mixing between strata
m.stratify_with(age_strat)

# outputs, requested after stratification, can select strata
m.request_output_for_flow("incidenceXage_00", "infection", source_strata={"age": "00"})
```

`stratify_with()` is where the work happens. It splits every listed compartment, splits every flow (`flow.stratify(strat)`), applies the flow adjustments, extends the mixing categories, and — for an `AgeStratification` — generates the inter-stratum ageing flows automatically. The user does not touch the flows they already wrote.

## Core abstractions

| Concept | summer2 name | Notes |
|---|---|---|
| Compartment | `Compartment(name, strata)` | Names carry their strata; `_original_compartment_names` tracks the pre-stratified set |
| Flow | `add_transition_flow`, `add_infection_frequency_flow`, `add_infection_density_flow`, `add_crude_birth_flow`, `add_replacement_birth_flow`, `add_importation_flow`, `add_death_flow`, `add_universal_death_flows` | A closed vocabulary of **named flow kinds**, not an open rate expression |
| Stratification | `Stratification(name, strata, compartments)` | Plus `AgeStratification` (adds ageing flows) and `StrainStratification` (multi-strain force of infection) |
| Stratum adjustment | `Adjustment` — `Multiply`, `Overwrite` | Applied to flows and to infectiousness, per stratum |
| Mixing | `set_mixing_matrix(matrix_or_function)` | Only allowed on a full stratification; builds up `_mixing_categories` |
| Parameter | `param("name")`, `set_default_parameters({...})` | Symbolic — parameters are graph nodes, resolved at run time |
| Derived output | `request_output_for_flow`, `request_output_for_compartments`, `request_function_output` | Tracked in a `networkx` DAG of dependencies, with a whitelist for selective evaluation |
| Computed value | `add_computed_value_func(name, func)` | Runtime-computed intermediate available to flow rates |
| Backend | `BackendType.JAX` (default) | The whole model compiles to JAX, giving autodiff and JIT |

## The key idea: stratification as a transformation

Three properties of `stratify_with()` are worth spelling out, because they are what the lingua franca would need to reproduce.

**It is validated against existing structure.** A flow adjustment naming a flow that does not exist is rejected. An infectiousness adjustment naming a compartment not in the model is rejected. Strata named in a source or destination are checked to exist.

**It is order-dependent but composable.** Stratifications apply in sequence, each seeing the model as the previous ones left it. `AgeStratification` may be applied only once; `StrainStratification` may be applied only once; a mixing matrix requires a *complete* stratification (all compartments), which is a real and correct restriction — you cannot mix on an axis that only some compartments have.

**Adjustments are the interface between the base model and the stratum.** The base model says "recovery happens at `recovery_rate`"; the stratification says "×1.5 for children, ×0.5 for the elderly". Neither knows about the other. This factoring — a generic process plus a per-stratum modifier — is the mechanism that lets one model text serve many structures.

## Parameters and the computation graph

Parameters are symbolic (`param("contact_rate")`), and derived outputs are nodes in a `networkx` DAG. The model is not evaluated when it is built; it is *compiled*, and the JAX backend then JIT-compiles the whole thing.

This buys two things that matter well beyond speed: **automatic differentiation of the entire model with respect to its parameters** (so gradient-based calibration and sensitivity analysis are free), and a **dependency graph of derived outputs** that can be pruned to only what is requested.

It is the same architectural move as camdl's IR and odin's dependency sort, arrived at through a different mechanism: separate the description of the model from its evaluation.

## Time

`CompartmentalModel(times=(start, end), timestep=1.0, ref_date=datetime)`. Times may be given as datetimes if `ref_date` is set, and are converted to numbers internally. The constructor checks that the timestep divides the period exactly.

There are **no dimensional types**. Rates are floats and their units are the modeller's responsibility. `add_infection_frequency_flow` versus `add_infection_density_flow` is the one place where a dimensional distinction is made explicit in the API — frequency-dependent versus density-dependent transmission — and it is made by *choosing a different function*, which is a coarse but effective form of typing.

## Strengths

- **Stratification as a first-class, validated transformation.** The single best treatment of the problem in the review.
- **A closed vocabulary of named flow kinds.** `add_infection_frequency_flow` versus `add_infection_density_flow` versus `add_transition_flow` versus `add_crude_birth_flow` encodes real epidemiological distinctions in the API, so the framework knows what a flow *means* and can transform it correctly under stratification. An open "rate expression" cannot be stratified automatically; a named flow kind can.
- **Adjustments (`Multiply` / `Overwrite`) as the base-model/stratum interface** — generic process, per-stratum modifier, neither aware of the other.
- **Special-cased stratification types.** `AgeStratification` generating ageing flows and `StrainStratification` restructuring the force of infection are exactly the cases that are laborious and error-prone by hand.
- **Correct restrictions, enforced.** Mixing matrices only on complete stratifications; one age stratification; one strain stratification; strains cannot carry a mixing matrix.
- **Symbolic parameters and a compiled computation graph**, with a JAX backend giving JIT and autodiff over the whole model.
- **Derived outputs as a requested DAG**, with source/destination stratum filters, so `incidence in children` is one line rather than a post-processing script.
- **Runtime validation toggleable** (`set_validation_enabled`) — thorough checks by default, switchable off for production runs.
- **A build tracker** (`ModelBuildTracker`) recording the actions taken to construct the model, which is a step toward introspectable model provenance.

## Limitations

- **Compartmental only.** No agents, no networks, no individual state.
- **No dimensional types.** Rates are floats; frequency-versus-density is the only dimensional distinction the API makes.
- **Flow kinds are closed.** The vocabulary is rich, but a flow whose semantics are not in the list has to be approximated by one that is. This is the price of the stratification machinery: the transformation only works because the framework knows what each flow kind means.
- **Stratification is order-dependent**, and the ordering constraints (age once, strain once, mixing needs completeness) are enforced but not derivable from the model text — you have to know them.
- **Model structure is built imperatively**, by a sequence of method calls, so the specification is a Python script rather than a document. The `ModelBuildTracker` records what was done, but the model is not itself data.
- **Interventions have no vocabulary.** Time-varying parameters go through `summer2.functions.time`; interventions are parameter changes. There is no coverage, delivery, eligibility, or product concept.
- **`_finalized` state and a two-phase lifecycle** (build, then finalize, then run) that the user has to be aware of.
- **Comments in the source flag work deferred to a hypothetical summer3** — parameterizable flow adjustments, arbitrary output dimensions, xarray output. The current design has known edges.

## Implications for the lingua franca

1. **Stratification must be an operator, not a rewrite.** summer2 proves this works and shows exactly what it requires: the framework must know what each flow *means* in order to know how it transforms. This is a strong argument for a **closed, semantically meaningful set of transition kinds** in the lingua franca — infection (frequency), infection (density), progression, birth, death, importation, ageing — rather than only an open rate expression.
2. **The adjustment mechanism is the right base-model/stratum interface.** `Multiply` and `Overwrite` per stratum, applied to flows and to infectiousness, keeps the base model stratum-agnostic. Adopt this shape; consider whether `Overwrite` should exist at all, given how it interacts with multiple stratifications.
3. **Special-case the stratifications that carry structure.** Age (which implies ageing flows), strain (which implies a restructured force of infection), and space (which implies movement) are not generic dimensions. camdl makes the same distinction differently, by separating population strata from residence structure. Both are recognising that dimensions are not interchangeable, and the lingua franca needs a way to say which kind a dimension is.
4. **Enforce the restrictions that follow from the structure.** "A mixing matrix requires a complete stratification" is a correctness property, not an implementation limit, and summer2 checks it. A multi-paradigm language will have many such properties; each should be a named error.
5. **Compile, do not evaluate.** summer2's `networkx` DAG, odin's topological sort, and camdl's IR are three routes to the same requirement. Whatever the lingua franca's surface syntax, the model must become a graph before it becomes a computation.
6. **Requested outputs with stratum filters** — `request_output_for_flow("incidence", "infection", source_strata={"age": "00"})` — is a good declarative output vocabulary and pairs naturally with camdl's `incidence(transition[age = child])`. Both name a transition and select strata; the lingua franca should have one such form.
7. **Note the missing intervention layer.** summer2 and odin, the two strongest compartmental specification designs here, both have essentially no intervention vocabulary. Starsim, EMOD, and EpiHiper have the strongest intervention vocabularies and the weakest structural specification. That split is a gap in the field, and closing it is one of the more concrete contributions available to this project.
