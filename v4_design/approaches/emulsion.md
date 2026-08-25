# EMULSION

|  |  |
|---|---|
| **Language** | YAML DSL; engine in Python |
| **Paradigms** | Compartmental, individual-based (IBM), **hybrid**, and metapopulation — multi-level and multi-scale |
| **Specification style** | A single human-readable YAML document. No code required |
| **Version reviewed** | 1.2.1 |
| **Licence** | Apache-2.0 |
| **Code** | <https://forgemia.inra.fr/dynamo/software/emulsion-public> |
| **Docs** | Ships with the repository; quickstart and feature model libraries |
| **Paper** | Picault et al. (2019), *PLOS Comp Biol* 15(9): e1007342, [10.1371/journal.pcbi.1007342](https://doi.org/10.1371/journal.pcbi.1007342) |

## What it's for

EMULSION is a modelling framework for veterinary and plant epidemiology in which **the model is a document**, not a program. Its stated motivation is explicitly about communication: models should be "explicit, intelligible and revisable" so that modellers can discuss them with biologists, veterinarians, and economists throughout the modelling process, rather than translating between a shared conceptual model and private code.

For this project, EMULSION is the most important framework in the review after camdl and Starsim, for one reason given below.

## The finding

**EMULSION already does the thing this project is trying to do.** Compare two of its shipped example models:

```yaml
# compart_SIR.yaml
levels:
  population:
    desc: 'level of the population'
    aggregation_type: 'compartment'
    contains:
      - individuals
  individuals:
    desc: 'level of individuals'
```

```yaml
# IBM_SIR.yaml
levels:
  population:
    desc: 'level of the population'
    aggregation_type: 'IBM'
    contains:
      - individuals
  individuals:
    desc: 'level of the individuals'
```

The rest of both files — the state machine, the transitions, the parameters, the initial conditions, the outputs — is identical. **A one-word change switches the model between a stochastic compartmental model and an individual-based model.** A third value, `hybrid`, gives a mixed formulation where some processes are tracked individually and others in aggregate.

This is the paradigm-independence goal, shipped, in an Apache-2.0 package, published in 2019. Any design work on this project should start by understanding exactly what EMULSION can and cannot express under that switch, because that boundary is the real research question.

## How a model is specified

One YAML document with named top-level sections:

```yaml
model_name: IBM_SIR

time_info:
  time_unit: 'days'
  delta_t: 1
  origin: 'January 1'
  total_duration: '100'

levels:
  population:
    aggregation_type: 'IBM'
    contains: [individuals]
  individuals:
    desc: 'level of the individuals'

processes:
  individuals:
    - health_state              # names a state machine

state_machines:
  health_state:
    states:
      - S: {name: 'Susceptible', fillcolor: 'wheat'}
      - I: {name: 'Infectious',  fillcolor: 'maroon'}
      - R: {name: 'Resistant',   fillcolor: 'deepskyblue'}
    transitions:
      - {from: S, to: I, rate: 'force_of_infection'}
      - {from: I, to: R, rate: 'recovery'}

parameters:
  transmission_I:
    desc: 'transmission rate from infectious individuals (/day)'
    value: 0.5
  recovery:
    desc: 'recovery rate (/day)'
    value: 0.1
  force_of_infection:
    desc: 'infection function'
    value: 'transmission_I * total_I / total_population'
    source: 'classical function assuming frequency dependence'

prototypes:
  individuals:
    - healthy:  {health_state: S}
    - infected: {health_state: I}

initial_conditions:
  population:
    - {prototype: healthy,  amount: 'initial_population_size - initial_infected'}
    - {prototype: infected, amount: 'initial_infected'}

outputs:
  type: csv
  population:
    period: 1
    extra_vars: [percentage_prevalence, total_population]
```

Every element carries a `desc:`, and parameters may carry a `source:` naming the assumption or citation behind the value. Documentation is part of the model, not adjacent to it.

## Core abstractions

| Concept | EMULSION name | Notes |
|---|---|---|
| Organisation level | `levels:` with `aggregation_type` and `contains` | `compartment`, `IBM`, `hybrid`, `metapopulation`; levels nest (individual → herd → metapopulation) |
| Process | `processes:` per level | Names the state machines (or grouped processes) that run at that level |
| State machine | `state_machines:` | States and transitions — the core model structure |
| Transition | `{from:, to:, rate:}` or `{... proba:}` or `{... duration:}` | **Three ways to specify timing**, converted to the time step automatically |
| Parameter | name with `value`, `desc`, `source` | `value` may be a number or an expression over other parameters and aggregates |
| Aggregate | `total_I`, `total_population` | Automatically available names for state totals |
| Prototype | `prototypes:` | Named bundles of initial attribute values — a template for creating individuals |
| Initial condition | `initial_conditions:` | Prototype + amount, where amount is an expression |
| Output | `outputs:` with `period` and `extra_vars` | Declarative |
| Code add-on | optional Python module | The escape hatch for behaviour the YAML cannot express |

## Time

`time_info` declares `time_unit`, `delta_t`, `origin` (a calendar date), and `total_duration`. Transitions may be specified as a **rate**, a **probability**, or a **duration**, and the engine converts:

> Rates are automatically converted into probabilities w.r.t the duration of the time step (delta_t), assuming a classical exponential distribution of durations in the states.

This is the same three-way distinction Starsim later encoded as `per` / `prob` / `dur`, expressed here as three alternative keys on a transition. It is not a type system — a rate is still a number and the unit lives in the `desc:` string — but the *distinction* is made in the model text, and the conversion is the engine's job rather than the modeller's.

## Multi-level and multi-scale

`levels` with `contains` builds a hierarchy: individuals inside herds inside a metapopulation, each with its own `aggregation_type` and its own processes. A herd-level process (culling, movement, testing) operates on herds; an individual-level process (infection, recovery) operates on individuals; the engine reconciles them.

This is a genuinely different structural idea from anything else in the review. Starsim, EpiModel, and epiworld all have one population level. epymorph has nodes and individuals-in-aggregate. EMULSION lets **each level independently choose its paradigm**, which is why the hybrid mode is possible at all.

## Strengths

- **Paradigm as a declared property of a level.** `aggregation_type: compartment | IBM | hybrid` on an otherwise identical model. This is the review's existence proof for the project's central premise.
- **The model is a document.** YAML, human-readable, diffable, version-controllable, and reviewable by non-programmers. Transitions are data; the state machine is data; initial conditions are data.
- **Multi-level structure with per-level paradigm choice.** Individual → herd → metapopulation, each aggregated as appropriate.
- **Three timing specifications per transition** — rate, probability, duration — with engine-side conversion.
- **Documentation embedded**: `desc:` on everything, `source:` on parameters recording where a value came from.
- **Prototypes** as named bundles of initial attributes — a clean way to say "create 100 healthy individuals and 1 infected one".
- **Parameters as expressions** over other parameters and automatic aggregates (`total_I`, `total_population`), so the force of infection is a parameter definition rather than hidden engine behaviour.
- **State machines are diagrammable** — `fillcolor` on states is there because the engine draws the model.
- **A stated communication goal**, and a design that follows from it.

## Limitations

- **YAML is a poor host for a language.** No syntax checking beyond schema validation, expressions embedded in strings (`value: 'transmission_I * total_I / total_population'`), no editor support, no type checking, and error messages that come from a YAML parser rather than a compiler. Everything camdl gets from having a real grammar, EMULSION gives up.
- **No dimensional types.** A rate's units live in its `desc:` string.
- **The force of infection is a hand-written expression.** `transmission_I * total_I / total_population` is correct here and is the modeller's responsibility; nothing checks it, and nothing knows that this transition is an infection rather than a progression. That is the same missing-`/N` class of bug camdl catches.
- **Aggregate names are magic.** `total_I` and `total_population` appear without declaration.
- **Code add-ons are the escape hatch**, and any model using them loses the "model is a document" property.
- **Small community, hosted on INRAE's GitLab rather than GitHub**, primarily veterinary/plant epidemiology. Discoverability and adoption outside that community are limited.
- **The paradigm switch has undocumented limits.** The shipped examples show `compartment`/`IBM`/`hybrid` on the same simple SIR. What happens when the model uses individual-level attributes that have no compartmental analogue is not stated in the material reviewed here, and is exactly the question this project needs answered.

## Implications for the lingua franca

1. **Start by mapping EMULSION's paradigm-switch boundary.** This is a concrete, tractable piece of research: take the shipped feature models, push each one across `compartment` / `IBM` / `hybrid`, and record where the switch stops working and why. That boundary is the actual scope of the lingua franca's paradigm-independence claim, and someone has already built the apparatus for measuring it.
2. **Paradigm belongs to a *level*, not to a model.** EMULSION's insight is that the choice is per-level and hierarchical: individuals may be aggregated while herds are explicit, or vice versa. That is more expressive than a single per-model paradigm flag and matches how hybrid models are actually built. The lingua franca should support declaring the aggregation level of each structural level.
3. **Three timing forms per transition — rate, probability, duration — is the right user-facing vocabulary.** It matches how epidemiologists actually have their parameters (a hazard, a per-step probability, a mean duration), and it converts mechanically. Starsim's type hierarchy is the rigorous version of the same idea; EMULSION's is the readable version. The lingua franca wants both: `rate`/`prob`/`dur` as *typed* alternatives on a transition.
4. **Embed provenance in the model.** `source:` on a parameter — "classical function assuming frequency dependence" — is a one-line field that answers the question every reviewer asks. Combined with camdl's `#'` doc comments and `@symbol` annotations, this suggests the language should have a standard place for *why this value*, not just *what value*.
5. **Prototypes are a good initial-conditions primitive.** Naming a bundle of initial attribute values and then saying how many of each to create is cleaner than either a compartment vector (which cannot express correlated attributes) or per-agent initialisation code.
6. **Do not build on YAML.** EMULSION demonstrates the value of model-as-document and simultaneously demonstrates the cost of choosing a data format as the syntax. Expressions in strings, no type checking, and parser-level errors are exactly what a purpose-built grammar avoids. The lingua franca should be as readable as EMULSION's YAML and as checkable as camdl's DSL — which means its own grammar with a canonical data serialisation underneath, not a data format used as a language.
