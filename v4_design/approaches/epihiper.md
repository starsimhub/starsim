# EpiHiper

|  |  |
|---|---|
| **Language** | C++ (MPI/OpenMP); models specified in JSON against published schemas |
| **Paradigms** | Agent-based on explicit dynamic contact networks, at national scale (300M+ agents) |
| **Specification style** | Five JSON documents against published schemas: config, disease model, initialization, intervention, scenario |
| **Version reviewed** | 2.0.0 |
| **Licence** | MIT |
| **Code** | <https://github.com/NSSAC/EpiHiper>, schemas at <https://github.com/NSSAC/EpiHiper-Schema> |
| **Docs** | <https://epihiper.readthedocs.io/> |
| **Paper** | [10.1093/pnasnexus/pgae557](https://doi.org/10.1093/pnasnexus/pgae557) |
| **Users** | Biocomplexity Institute, University of Virginia; CDC Scenario Modeling Hub at national scale |

## What it's for

EpiHiper runs US-scale agent-based epidemics — hundreds of millions of agents on a synthetic contact network — on HPC. Its relevance here is not the scale but the **specification**: unlike EMOD, EpiHiper makes the *disease model itself* data, and its intervention format is the most fully realised declarative intervention language in the review.

## How a model is specified

### The disease model is JSON

```json
{
  "$schema": ".../diseaseModelSchema.json",
  "states": [
    { "id": "S",      "ann:label": "Susceptible",  "infectivity": 0, "susceptibility": 1 },
    { "id": "I",      "ann:label": "Infectious",   "infectivity": 1, "susceptibility": 0 },
    { "id": "R",      "ann:label": "Recovered",    "infectivity": 0, "susceptibility": 0 },
    { "id": "V",      "ann:label": "Vaccinated",   "infectivity": 0, "susceptibility": 0.1 },
    { "id": "wanedS", "ann:label": "Waned natural immunity", "infectivity": 0, "susceptibility": 0.7 },
    { "id": "wanedV", "ann:label": "Waned vaccine immunity",  "infectivity": 0, "susceptibility": 0.5 }
  ],
  "initialState": "S",
  "transmissions": [
    { "id": "infection_I_S", "entryState": "S", "exitState": "I",
      "contactState": "I", "transmissibility": 1.0 },
    { "id": "infection_I_V", "entryState": "V", "exitState": "I",
      "contactState": "I", "transmissibility": 1.0 }
  ],
  "transitions": [
    { "id": "I_to_R", "entryState": "I", "exitState": "R",
      "probability": 1, "dwellTime": { "gamma": { "alpha": 1.0, "beta": 5.0 } } },
    { "id": "R_to_wanedS", "entryState": "R", "exitState": "wanedS",
      "probability": 1, "dwellTime": { "gamma": { "alpha": 1.0, "beta": 183.0 } } }
  ]
}
```

Three things stand out.

**States carry `infectivity` and `susceptibility` as numbers.** The entire relative-transmission structure of the model — vaccine efficacy, waning, partial immunity — is expressed as two scalars per state rather than as multipliers scattered through transmission code. `V` has susceptibility 0.1; `wanedV` has 0.5; that *is* the vaccine model.

**Transmission and transition are separate, named, typed objects.** A `transmission` names an `entryState`, an `exitState`, a `contactState`, and a `transmissibility`. A `transition` names an `entryState`, an `exitState`, a `probability`, and a **`dwellTime` distribution**. This is precisely epipack's spontaneous/neighbour-induced split, expressed as data instead of tuples.

**Dwell times are distributions, declared per transition.** `{"gamma": {"alpha": 1.0, "beta": 5.0}}`. Not a rate, not an implicit exponential — a named distribution with parameters. MEmilio needed separate LCT and IDE model families to get this; EpiHiper gets it by making dwell time a field.

### Interventions are a small declarative language

The intervention file has four sections — `variables`, `sets`, `triggers`, `interventions` — and together they are a genuine rule language.

**Sets** are computed by set algebra over predicates on nodes and edges:

```json
{
  "id": "school_edges",
  "content": {
    "operation": "union",
    "sets": [
      { "elementType": "edge",
        "left":  { "edge": { "property": "targetActivity", "feature": "activityType" } },
        "operator": "==",
        "right": { "value": { "trait": "activityTrait", "feature": "activityType", "enum": "school" } } },
      { "elementType": "edge",
        "left":  { "edge": { "property": "sourceActivity", "feature": "activityType" } },
        "operator": "==",
        "right": { "value": { "trait": "activityTrait", "feature": "activityType", "enum": "school" } } }
    ]
  }
}
```

**Variables** are named mutable state with scope and reset behaviour:

```json
{ "id": "week_day", "initialValue": 0, "scope": "local", "reset": 7 }
```

**Triggers** are boolean expressions over observables and variables, mapped to intervention ids:

```json
{ "trigger": { "and": [
      { "left": { "observable": "time" }, "operator": ">=", "right": { "value": { "number": 0 } } },
      { "left": { "variable": { "idRef": "week_day" } }, "operator": "==", "right": { "value": { "number": 5 } } }
  ] },
  "interventionIds": ["close_schools"] }
```

**Interventions** bind a target set to a list of operations:

```json
{ "ann:id": "maintain_week_day",
  "trigger": { "value": true },
  "target": { "set": { "idRef": "%empty%" } },
  "once": [ { "operations": [
      { "target": { "variable": { "idRef": "week_day" } }, "operator": "+=",
        "value": { "number": 1 } } ] } ] }
```

So "close schools on Fridays" is: a variable counting the day of week that resets every 7, a set computed from edge activity types, a trigger that is a boolean expression, and an intervention that operates on the set. **No code.** The whole thing is data.

## Core abstractions

| Concept | EpiHiper name | Notes |
|---|---|---|
| Disease state | `states[]` entry with `infectivity`, `susceptibility` | Relative transmission encoded as state properties |
| Transmission | `transmissions[]`: entry × exit × contact state × transmissibility | Named |
| Transition | `transitions[]`: entry × exit × probability × `dwellTime` distribution | Named; dwell time is a distribution |
| Contact network | edge list with activity types, durations, and traits | External file; supports temporal networks |
| Trait | `traits.json` | Typed enumerations for agent and edge features |
| Set | set algebra over node/edge predicates | Computed, named, reusable |
| Variable | named, scoped (`local`/`global`), with `reset` period | Mutable model-level state |
| Trigger | boolean expression over observables, variables, and sets | AND/OR trees |
| Intervention | `target` set × `once`/`sampling` × `operations` | Operations mutate variables, node/edge properties, or health states |
| Scenario | `scenario.json` | Binds config, disease model, network, initialization, and interventions |

## Strengths

- **Both the disease model and the intervention layer are data.** The only framework in the review of which this is true. EMOD makes interventions data; camdl and odin make the disease data; EpiHiper does both.
- **Dwell-time distributions per transition, declared.** `{"gamma": {"alpha": ..., "beta": ...}}` — the concept MEmilio needs four model families for.
- **Relative transmission as state properties.** `infectivity` and `susceptibility` per state expresses vaccination, waning, and partial immunity uniformly, without multipliers threaded through code.
- **A real set algebra over the network.** Sets of nodes and edges computed from predicates over typed traits, composable with union/intersection, named and reusable. This is the most expressive targeting mechanism in the review — more so than EMOD's `TargetingLogic`, because it operates on *edges* as well as nodes and supports set operations rather than only conjunctions of predicates.
- **Named mutable variables with scope and automatic reset** — enough state to express day-of-week logic, cumulative counters, and stateful policies declaratively.
- **Triggers as boolean expression trees** over observables and variables, decoupled from the interventions they fire.
- **Published JSON schemas** for every document type, with `$schema` references in the files themselves.
- **Annotation convention.** `ann:label`, `ann:id` — a namespaced prefix distinguishing human annotation from semantics, so documentation lives in the model without being confused for it.
- **National scale**, with MPI and OpenMP builds and CDC Scenario Modeling Hub use.

## Limitations

- **JSON verbosity, at its worst in the review.** The "close schools on Fridays" example is roughly 200 lines. A boolean comparison is a nine-line object (`left` / `operator` / `right`, each nested). The expressiveness is real and the ergonomics are hostile.
- **The transmission model is fixed in form.** Transmissibility is a scalar per transmission rule, scaled by state infectivity/susceptibility and edge weight. Anything else — dose response, within-host viral load, density dependence — is not expressible.
- **No dimensional types.** Dwell-time parameters, transmissibilities, and times are bare numbers with conventions.
- **Requires a synthetic contact network as input**, which for most users means the Biocomplexity Institute's US datasets. There is no network *specification* vocabulary — the network is a file, not a model.
- **Heavy build**: C++, CMake, PostgreSQL, MPI/OpenMP, git submodules.
- **Small documented user base outside UVA** — 6 stars, 2 forks at last count, notwithstanding its national-scale deployments.
- **No observation model and no inference layer.**
- **Hand-authoring is impractical.** The format is realistically machine-generated, and there is no published authoring language above it.

## Implications for the lingua franca

1. **EpiHiper is the closest existing thing to a fully declarative agent-based model.** Both structure and policy are data. That combination is unique in the review, and it is a proof that the "no code" goal is reachable on the agent-based side and not only for compartmental models.
2. **Take the `infectivity` / `susceptibility` state-property idea.** Expressing vaccine efficacy, waning immunity, and partial protection as two scalars attached to a *state* rather than as multipliers in a transmission calculation is compact, uniform, and introspectable. It also composes correctly: a new state with new values needs no changes anywhere else. Starsim's `rel_sus` / `rel_trans` are the per-agent version of the same idea; EpiHiper's per-state version is more declarative.
3. **Take the sets-variables-triggers-operations decomposition for interventions.** Combined with EMOD's four-part campaign structure, this gives a nearly complete design: named computed **sets** as targets (over agents *and* contacts), named **variables** for policy state, **triggers** as boolean expressions over observables, and **operations** as the actions. That is a rule language, and it is what an intervention vocabulary needs to be to cover real policies.
4. **Dwell time as a declared distribution per transition** is the single cleanest solution in the review to the exponential-dwell-time problem. Adopt it, and let the backends decide how to realise it — Erlang sub-stages for an ODE backend (as camdl's `via erlang` does), direct sampling for an agent-based one, a memory kernel for an IDE one. **This is a case where one declaration maps onto different implementations per paradigm, which is exactly the lingua franca's job.**
5. **Adopt the annotation-prefix convention.** `ann:label` and `ann:id` cleanly separate human-facing documentation from machine-facing semantics inside the same document. Better than a comment convention, and it survives serialisation.
6. **EpiHiper is the strongest evidence for the "readable grammar, data serialisation underneath" architecture.** It has arguably the best intervention *semantics* in the review and unquestionably the worst intervention *syntax*. Everything in that file could be five lines of a readable DSL. The lesson repeats from EMOD and EMULSION and should now be treated as settled.
