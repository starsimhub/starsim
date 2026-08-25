# epidemics (Epiverse-TRACE)

|  |  |
|---|---|
| **Language** | R, with models written in the odin DSL |
| **Paradigms** | Deterministic compartmental (ODE via odin); one discrete-time stochastic model |
| **Specification style** | **Model structure is fixed.** The user composes a scenario from *composable elements* |
| **Version reviewed** | 0.5.0 |
| **Licence** | MIT |
| **Code** | <https://github.com/epiverse-trace/epidemics> |
| **Docs** | <https://epiverse-trace.github.io/epidemics/> |
| **Paper** | — |
| **Ecosystem** | Epiverse-TRACE (data.org, LSHTM, Universidad de los Andes) — ~18 interoperating R packages |

## What it's for

`epidemics` is deliberately the opposite of a modelling framework. Its scope statement is unusually clear:

> *epidemics* aims to help public health practitioners — rather than research-focussed modellers — to rapidly simulate disease outbreak scenarios […] *epidemics* **trades away some flexibility in defining model structures for a gain in the ease of defining epidemic scenario components** such as affected populations and model events.

So the compartmental structure is **fixed**: a library of published models (`model_default` — an age-structured SEIR-V; `model_vacamole` — a two-dose leaky-vaccination model; `model_ebola` — a stochastic Erlang sub-compartment model; `model_diphtheria` — a camp-setting model). Users cannot create new links between compartments. What they *can* do is compose the scenario around the model.

This is in the review because it is the clearest statement of a position the lingua franca has to take one way or the other: **which parts of a model should be user-definable, and which should be a library?**

## How a model is specified

```r
uk_population <- population(
  name               = "UK",
  contact_matrix     = contact_matrix,
  demography_vector  = demography_vector,
  initial_conditions = initial_conditions   # rows = age groups, cols = compartments
)

output <- model_default(population = uk_population, time_end = 600, increment = 1.0)
```

Then compose elements onto it:

```r
close_schools <- intervention(
  type = "contacts", time_begin = 200, time_end = 260,
  reduction = matrix(c(0.5, 0.01, 0.01))     # per demography group
)

mask_mandate <- intervention(
  type = "rate", time_begin = 200, time_end = 300,
  reduction = 0.163                           # on transmission_rate
)

vaccinate <- vaccination(
  name = "vaccinate all", time_begin = ..., time_end = ..., nu = ...
)

output <- model_default(
  population    = uk_population,
  intervention  = list(contacts = close_schools, transmission_rate = mask_mandate),
  vaccination   = vaccinate,
  time_dependence = list(transmission_rate = function(time, x) x * seasonality(time)),
  time_end = 600
)
```

Every composable element is optional except `population`. Passing lists of elements produces a **scenario grid**, run and returned as a single tidy data frame with scenario identifiers — so scenario comparison is a built-in output shape, not a loop the user writes.

## Core abstractions

| Concept | epidemics name | Notes |
|---|---|---|
| Population | `<population>` S3 class | Contact matrix + demography vector + initial conditions matrix, with rows = demography groups throughout |
| Model | `model_default`, `model_vacamole`, `model_ebola`, `model_diphtheria` | Fixed structures from the literature |
| Intervention | `<intervention>` superclass → `<contacts_intervention>`, `<rate_intervention>` | Time window + reduction, per demography group |
| Vaccination | `<vaccination>` | Time window + per-group rate `nu`, possibly multi-dose |
| Time dependence | named list of `function(time, x)` | Seasonality and other forcing on named parameters |
| Model event | `<event>` | Discrete changes |
| Output | `<data.frame>`-inheriting, tidy | Long format with scenario columns |

## The design decisions, as stated

The package ships a `design-principles.Rmd` vignette, which is itself notable — very few frameworks in this review write down *why*. Some of its decisions matter here.

**Interventions compose additively, not multiplicatively.** Two interventions reducing contacts by X% and Y% give `C × (1 − (X + Y))`, not `C × (1 − X)(1 − Y)`. The stated reason is that additive effects were considered easier for users to understand.

This is a striking choice. Multiplicative composition is the more defensible modelling assumption and the more common one; the package chose the other and **documented the choice and its rationale**. Whether or not one agrees, the fact that it is written down is exactly right, and it stands in contrast to the many frameworks where the composition rule is whatever the implementation happens to do.

**Model-specific composable-element sets.** The Ebola model does not accept a `vaccination` element, because vaccination is considered unlikely in its use case. The Ebola model also refuses rate interventions on `infectiousness_rate` and `removal_rate`, because those parameters determine the number of Erlang sub-compartments and cannot be changed mid-run without redistributing individuals. That is a **capability restriction derived from the model's internal structure, enforced and explained** — the same discipline camdl applies through its capability checks.

**Two-level function structure.** Each model has an internal implementation (`.model_ebola_internal()`) and a user-facing wrapper (`model_ebola()`) that does input checking, cross-checking between elements, and scenario combination. The checking layer is substantial: `assert_*()` functions verify each element and verify that elements are mutually compatible with the population.

**Demography-in-rows everywhere.** Initial conditions, vaccination rates, and contact reductions all use rows = demography groups. A convention, stated once, applied throughout.

**Reproducibility across scenarios via `withr`.** Seeds are preserved across parameter sets and interventions so that differences between scenario runs are attributable to the scenario rather than to the RNG — the same goal as common random numbers, achieved by seed management.

## Strengths

- **A stated scope, and design decisions that follow from it.** The trade — less structural flexibility, more scenario expressiveness — is explicit, justified, and consistently applied.
- **A published design-principles document.** Rare, and valuable.
- **Composable elements as classed objects**: `<population>`, `<intervention>`, `<vaccination>`, plus time-dependence functions and events. Each optional, each checkable.
- **The intervention type split** — `<contacts_intervention>` versus `<rate_intervention>` — distinguishes "changed who meets whom" from "changed a parameter", which is a real and usually implicit distinction.
- **Scenario grids as a native output shape**, returned tidy with scenario identifiers.
- **Composition rule stated explicitly** (additive), with the reasoning.
- **Model-specific capability restrictions**, enforced with explanations grounded in the model's structure.
- **Extensive cross-checking** between composable elements and the population.
- **Built on odin**, so the fixed models are compiled and fast, and their source is a readable DSL rather than hand-written C.
- **Interoperates with the Epiverse-TRACE stack** — `socialmixr` contact matrices, `epiparameter` delay distributions, `<incidence2>` data.

## Limitations

- **Model structure is not user-definable.** New compartments or new links require contributing a model to the package. For this project's purposes that is the whole point of the entry, but it is a real limit on the tool.
- **Additive intervention composition is a questionable default**, however well documented. Two 60% reductions give a 120% reduction, which has to be clamped somewhere.
- **The library is small** — four models — and each has its own parameter names and its own allowed element set, so the "unified API" is thinner than it appears.
- **Deterministic apart from the Ebola model**, and that one is written in R rather than odin for a specific performance reason.
- **No dimensional types**; rates are per-day numerics.
- **No observation model and no inference.** The package simulates; fitting is somebody else's package, by ecosystem design.
- **Composable elements do not compose across models uniformly** — knowing which elements a model accepts is knowledge held in the vignettes.

## Implications for the lingua franca

1. **This is the clearest articulation in the review of a choice the design must make explicitly.** `epidemics` fixes the structure and opens the scenario; camdl, odin, and summer2 open the structure and largely omit the scenario. A lingua franca that opens both has to be careful that the scenario vocabulary does not become unusable in the presence of arbitrary structure — for instance, `<contacts_intervention>` only means something if the model has a contact matrix. **The design needs a way for a scenario element to declare which structural features it requires**, which is again epymorph's requirement contract, applied to interventions.
2. **State the composition rule, and make it settable.** `epidemics` deserves credit for writing down that interventions compose additively and why. The lesson is not to copy the rule but to make it **explicit and per-effect** — Vivarium's declared combiners are the general mechanism, and `epidemics` is the evidence that a global default silently chosen would be wrong for someone.
3. **The contacts/rate intervention split is worth keeping.** An intervention either changes the contact structure or changes a parameter. These have different semantics, different data requirements, and different validity conditions, and separating them at the type level (as `epidemics` does, and as Epydemix and MEmilio do implicitly by only supporting the first) makes both checkable.
4. **Capability restrictions should be derived and explained.** "You cannot apply a rate intervention to `infectiousness_rate` because it determines the number of Erlang sub-compartments" is a restriction that follows from the model's structure, and the package says so. In a language where dwell-time structure is declared, this restriction becomes derivable automatically — which is a nice illustration of what declared structure buys.
5. **Scenario grids as a native result shape.** Passing lists of elements and getting back one tidy frame with scenario identifiers is the right ergonomics, and it matches camdl's `batch` and Starsim's `MultiSim`. The lingua franca should treat "a set of scenarios" as the default unit of a run, not a single trajectory.
6. **Write the design-principles document.** `epidemics` has one; almost nothing else here does. For a project whose deliverable is a specification, the reasoning behind each decision *is* part of the deliverable, and the `best-practices` folder is where it goes.
