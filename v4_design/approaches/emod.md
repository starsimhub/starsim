# EMOD

|  |  |
|---|---|
| **Language** | C++ engine; models specified entirely in JSON; Python tooling (`emod-api`, `emodpy-*`) |
| **Paradigms** | Agent-based (individual-based), with node-level metapopulation structure and migration |
| **Specification style** | Three JSON documents against a published schema: `config.json`, `campaign.json`, `demographics.json` |
| **Version reviewed** | EMOD-Hub community fork |
| **Licence** | MIT |
| **Code** | <https://github.com/EMOD-Hub/EMOD> |
| **Docs** | <https://emod-hub.github.io/> |
| **Paper** | [10.1093/femspd/fty059](https://doi.org/10.1093/femspd/fty059) |
| **Status** | **Retired by IDM**; maintained by the community EMOD-Hub organisation |

## What it's for

EMOD is a multi-disease individual-based transmission platform — malaria, HIV, TB, typhoid, polio, and generic respiratory — in which the *entire model is data*. There is one C++ executable; a model is three JSON files. Adding a vaccination campaign, changing the demographic structure, or switching disease is editing JSON, not writing code.

It is included here despite being retired because it is **the largest and longest-running experiment in fully schema-driven model specification** in infectious disease modelling, and this project needs to know both what that bought and what it cost.

## How a model is specified

**`config.json`** — simulation parameters, validated against a published schema.

**`demographics.json`** — nodes, populations, age distributions, and *Individual Properties*: user-defined categorical attributes (`Risk: HIGH/MEDIUM/LOW`, `Accessibility: YES/NO`) that agents carry and that interventions can target.

**`campaign.json`** — everything that happens to people. This is the interesting one:

```json
{
  "Events": [{
    "class": "CampaignEvent",
    "Start_Day": 10,
    "Nodeset_Config": { "class": "NodeSetAll" },
    "Event_Coordinator_Config": {
      "class": "StandardInterventionDistributionEventCoordinator",
      "Number_Repetitions": 5,
      "Timesteps_Between_Repetitions": 20,
      "Target_Demographic": "Everyone",
      "Demographic_Coverage": 0.2,
      "Targeting_Config": {
        "class": "TargetingLogic",
        "Logic": [
          [ { "class": "HasIntervention", "Is_Equal_To": 0, "Intervention_Name": "MyVaccine" } ],
          [ { "class": "HasIP", "Is_Equal_To": 1, "IP_Key_Value": "Risk:HIGH" } ]
        ]
      },
      "Intervention_Config": {
        "class": "SimpleVaccine",
        "Intervention_Name": "MyVaccine",
        "Vaccine_Take": 1,
        "Vaccine_Type": "AcquisitionBlocking",
        "Waning_Config": { "class": "WaningEffectConstant", "Initial_Effect": 1.0 }
      }
    }
  }]
}
```

## The campaign grammar

This is EMOD's real contribution and the reason it is in this review. A campaign event decomposes into four orthogonal parts:

| Part | Question it answers | Examples |
|---|---|---|
| **Nodeset_Config** | *Where?* | `NodeSetAll`, `NodeSetNodeList`, `NodeSetPolygon` |
| **Event_Coordinator_Config** | *When, to how many, how often?* | `Start_Day`, `Number_Repetitions`, `Timesteps_Between_Repetitions`, `Demographic_Coverage`, `Target_Demographic`; coordinators include `NChooser` (target a number, not a fraction), triggered coordinators, and reference-tracking coordinators |
| **Targeting_Config** | *To whom?* | `TargetingLogic` — an AND-of-ORs over predicates: `HasIntervention`, `HasIP`, `IsPregnant`, `HasRelationship`, … |
| **Intervention_Config** | *What is delivered?* | `SimpleVaccine`, `OutbreakIndividual`, `SimpleDiagnostic`, `AntimalarialDrug`, plus a `Waning_Config` describing how efficacy decays |

Each part is independently substitutable. Change `Targeting_Config` and the same vaccine goes to a different group; change `Intervention_Config` and the same schedule delivers a different product.

Two pieces deserve particular attention.

**`Targeting_Config` is a declarative eligibility predicate.** `TargetingLogic` with `Logic` as a list of lists is a conjunction of disjunctions over named, checkable conditions. "People who do not already have MyVaccine, AND are Risk:HIGH" is *data* — inspectable, serialisable, diffable. This is exactly what Starsim's `eligibility` callable and EpiModel's scenario grammar cannot express, and it is the answer to a gap flagged in both of those reviews.

**`Waning_Config` separates efficacy from its decay.** `WaningEffectConstant`, `WaningEffectExponential`, `WaningEffectMapPiecewise` with a `Durability_Map` of times and values — a vaccine's initial effect and its waning profile are separate, named, composable objects. Starsim's `ss.Product` reaches for this; EMOD's version is more complete.

## Core abstractions

| Concept | EMOD name | Notes |
|---|---|---|
| Agent | Individual | Age, sex, and disease state fixed by the C++ build |
| Agent attribute | Individual Property (IP) | User-defined categorical, declared in demographics, targetable and transition-able |
| Node attribute | Node Property (NP) | Same, for nodes |
| Disease | compiled into the executable | **Not** specifiable in JSON — this is the boundary |
| Node | entry in demographics | Population, geography, and migration links |
| Campaign event | `CampaignEvent` | Nodeset × coordinator × targeting × intervention |
| Product | `Intervention_Config` class | With `Waning_Config` and `Cost_To_Consumer` |
| Eligibility | `Targeting_Config` | Declarative predicate logic |
| Trigger | broadcast/observe event strings | Interventions can listen for events (`NewInfectionEvent`) and fire in response |
| Schema | published JSON schema | `emod-api` reads it; tooling is generated from it |

## The boundary: what is data and what is not

The disease model itself is C++. Compartments, progression, and transmission for malaria or HIV are compiled in; JSON configures them. So EMOD is schema-driven *above* the disease and hard-coded *at* it — which is exactly the inverse of camdl, odin, and summer2, where the disease structure is the specification and interventions are absent.

This is the sharpest illustration of the split that runs through the whole review, and EMOD sits at one extreme of it.

## Strengths

- **The model is data.** Three JSON documents, a published schema, no compilation to run a new scenario. Tooling, validation, and generation all follow from that.
- **The four-part campaign grammar** — where × when/how many × to whom × what — is the most complete decomposition of an intervention in the review.
- **`Targeting_Config` as declarative eligibility logic.** Named predicates over agent state, composed as AND-of-ORs, as data.
- **`Waning_Config` as a separate, named efficacy-decay object**, including piecewise maps.
- **Individual Properties**: user-defined categorical agent attributes, declared in demographics, targetable by interventions and transitionable between values. A user-extensible state dimension without touching the engine.
- **Event broadcast and observation**, so interventions can be triggered by model events rather than only by schedule.
- **A published schema**, which is what made the whole `emod-api` / `emodpy` Python layer possible.
- **Multi-disease from one engine** with a shared configuration vocabulary.
- **National-scale deployment** in real programme planning.

## Limitations

- **Retired by its originating institution.** IDM has confirmed it is no longer supported; commits since come from the community EMOD-Hub organisation.
- **The disease model is not specifiable.** New disease structure means C++ and a rebuild. The schema covers everything except the thing this project most wants to specify.
- **JSON is a bad language.** No comments (the regression files use `"COMMENT1"` keys), no expressions, deep nesting, and `"class"` as a discriminator string. A three-level campaign event is hard to read and harder to review. This is EMOD's version of EMULSION's YAML problem, and it is worse because the documents are bigger.
- **Enormous parameter surface.** The config schema runs to hundreds of parameters with subtle interdependencies, and `"Use_Defaults": 1` means much of a model's behaviour is not visible in its files.
- **No dimensional types.** `Start_Day`, `Timesteps_Between_Repetitions`, and rates are all bare numbers.
- **Verbose.** The two-event campaign quoted above is 60 lines of JSON for "seed an outbreak and vaccinate 20% of high-risk people who are not already vaccinated, five times, twenty days apart".
- **No observation model** in the specification; reporters are configured separately.
- **Heavy build and run**, with a C++ executable, MPI, and a substantial Python workflow layer.

## Implications for the lingua franca

1. **Adopt the four-part intervention decomposition.** *Where* (spatial/node scope) × *when and how much* (schedule and coverage) × *to whom* (eligibility predicate) × *what* (product with efficacy and waning). EMOD has the most complete version; Starsim's delivery/product split is a partial version of parts two and four; camdl's `transfer` is part four only. This decomposition should go into the design essentially as EMOD has it.
2. **Eligibility must be a declarative predicate, and EMOD shows what it looks like.** Named conditions over agent state (`HasIntervention`, `HasIP`, `IsPregnant`), composed with explicit logic. This closes the gap flagged in the Starsim and EpiModel reviews, and it means an intervention can be diffed, diagrammed, and generated from prose — which an opaque Python callable cannot.
3. **Efficacy and its waning are separate objects.** `Vaccine_Take`, `Vaccine_Type` (acquisition- / transmission- / mortality-blocking), and `Waning_Config` are three orthogonal facts about a vaccine. Any product vocabulary should have this shape.
4. **User-defined categorical agent properties are the right extensibility mechanism.** EMOD's Individual Properties let a modeller add `Risk`, `Accessibility`, or `Insurance` without touching the engine, and make them targetable. This is the agent-based counterpart to a stratification dimension, and the lingua franca should treat them as the same concept expressed at different aggregation levels — which is also MetaCast's and EMULSION's insight.
5. **Triggered interventions need a place in the vocabulary.** EMOD's event broadcast/observe mechanism means "when a new infection is observed, trigger contact tracing" is expressible declaratively. Most frameworks make this a callback. camdl has scheduled and reactive interventions; the lingua franca needs both.
6. **EMOD is the cautionary half of the model-as-data argument.** Everything good about it follows from the schema; everything painful about it follows from JSON being the surface syntax. Combined with EMULSION's YAML, the conclusion is the same twice over: **a data format is the right thing to compile *to*, and the wrong thing to write *in*.** The lingua franca needs a readable grammar with a canonical serialisation underneath — which is precisely camdl's architecture.
7. **Note where the boundary fell, and why.** EMOD made everything data except the disease model, because the disease model is where the performance and the complexity are. That is the boundary this project has to cross, and EMOD's history is evidence that crossing it is the hard part rather than an oversight.
