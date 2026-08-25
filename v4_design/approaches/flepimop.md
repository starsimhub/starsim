# flepiMoP

|  |  |
|---|---|
| **Language** | Python (simulation) and R (inference and post-processing) |
| **Paradigms** | Metapopulation compartmental — deterministic (RK4) or stochastic, with a full inference and forecasting pipeline |
| **Specification style** | **One YAML configuration** spanning compartments, transitions, seeding, modifiers, outcomes, and fitting |
| **Version reviewed** | current `main` |
| **Licence** | (see repository) |
| **Code** | <https://github.com/HopkinsIDD/flepiMoP> |
| **Docs** | `documentation/` in the repository |
| **Paper** | — |
| **Users** | Johns Hopkins IDD; US COVID-19 and influenza Scenario Modeling Hub submissions |

## What it's for

flepiMoP (FLexible EPIdemic MOdeling Pipeline) is the production pipeline behind a large share of US COVID-19 and influenza scenario-hub output. It is not primarily a modelling *framework* — it is a **configuration language for a modelling pipeline**, and that is precisely why it belongs in this review: it is the most complete example of an entire modelling workflow, from model structure through interventions to likelihood and inference, expressed as a single declarative document.

## How a model is specified

One YAML file. The whole SIR-in-two-subpopulations example:

```yaml
name: sample_2pop
start_date: 2020-02-01
end_date: 2020-08-31
nslots: 1

subpop_setup:
  geodata: model_input/geodata_sample_2pop.csv
  mobility: model_input/mobility_sample_2pop.csv

initial_conditions:
  method: SetInitialConditions
  initial_conditions_file: model_input/ic_2pop.csv
  allow_missing_subpops: TRUE
  allow_missing_compartments: TRUE

compartments:
  infection_stage: ["S", "E", "I", "R"]

seir:
  integration:
    method: rk4
    dt: 1
  parameters:
    sigma: {value: 1 / 4}
    gamma: {value: 1 / 5}
    Ro:    {value: 2.5}
  transitions:
    - source: ["S"]
      destination: ["E"]
      rate: ["Ro * gamma"]
      proportional_to: [["S"], ["I"]]
      proportion_exponent: ["1", "1"]
    - source: ["E"]
      destination: ["I"]
      rate: ["sigma"]
      proportional_to: ["E"]
      proportion_exponent: ["1"]
    - source: ["I"]
      destination: ["R"]
      rate: ["gamma"]
      proportional_to: ["I"]
      proportion_exponent: ["1"]
```

### Compartments are a product of axes

`compartments: {infection_stage: [S, E, I, R]}` declares one axis. Adding more axes gives the Cartesian product:

```yaml
compartments:
  infection_stage: ["S", "E", "I", "R"]
  vaccination_status: ["unvaccinated", "first_dose", "second_dose"]
  variant: ["wild", "alpha", "delta"]
```

and transitions are written over axis *lists*, so `source: [["S"], ["unvaccinated", "first_dose"], ["wild"]]` names a set of compartments rather than one. This is the MetaCast dimension algebra and the camdl `dimensions` block, expressed in YAML — stratification is declared, not written out.

### The transition form is unusually explicit

```yaml
- source: ["S"]
  destination: ["E"]
  rate: ["Ro * gamma"]
  proportional_to: [["S"], ["I"]]
  proportion_exponent: ["1", "1"]
```

A transition is a rate **times a product of compartment terms raised to exponents**. `proportional_to: [["S"], ["I"]]` with exponents `[1, 1]` gives `Ro * gamma * S * I` (frequency-normalised by the pipeline). Setting an exponent to a parameter name gives density-dependent or sub-exponential mixing.

This is more structured than an arbitrary rate expression and less structured than summer2's named flow kinds: the *form* of the rate is fixed (a coefficient times a product of powers of compartments), so the pipeline can reason about it, while the coefficient stays an expression. It is a genuinely interesting middle position, and it is the form that makes automatic stratification over the compartment axes tractable.

### Modifiers are the intervention grammar

```yaml
seir_modifiers:
  scenarios: [Ro_lockdown, Ro_all]
  modifiers:
    Ro_lockdown:
      method: SinglePeriodModifier
      parameter: Ro
      period_start_date: 2020-03-15
      period_end_date: 2020-05-01
      subpop: "all"
      value: 0.4
    Ro_relax:
      method: SinglePeriodModifier
      parameter: Ro
      period_start_date: 2020-05-01
      period_end_date: 2020-08-31
      subpop: "all"
      value: 0.8
    Ro_all:
      method: StackedModifier
      modifiers: ["Ro_lockdown", "Ro_relax"]
```

A modifier names a **parameter**, a **date range**, a **subpopulation set**, and a **value**. `StackedModifier` composes them. `scenarios:` names which combinations to run. Modifier values can be given distributions, in which case they become fitted quantities.

This is EpiModel's tabular scenario grammar generalised: parameter × time window × spatial scope × value, with composition and with a scenario set declared in the same file.

### Outcomes are a declarative observation model

```yaml
outcomes:
  method: delayframe
  outcomes:
    incidCase:
      source: {incidence: {infection_stage: "I"}}
      probability: {value: 0.5}
      delay:      {value: 5}
    incidHosp:
      source: {incidence: {infection_stage: "I"}}
      probability: {value: 0.05}
      delay:      {value: 7}
      duration:   {value: 10, name: currHosp}
    incidDeath:
      source: incidHosp
      probability: {value: 0.2}
      delay:       {value: 14}
```

This is the best thing in flepiMoP. An observed outcome is declared as **source × probability × delay × duration**, and outcomes can **chain**: `incidDeath` takes `incidHosp` as its source rather than the infection state. So the whole surveillance cascade — infections → cases → hospitalisations → deaths, each with its own ascertainment probability and reporting delay — is a dozen lines of declaration, and `duration` with a `name` automatically produces the prevalence series (`currHosp`) alongside the incidence one.

`outcome_modifiers` then apply the same modifier grammar to outcome parameters, so "case ascertainment was lower before June 2020" is `parameter: incidCase::probability` over a date range.

## Core abstractions

| Concept | flepiMoP name | Notes |
|---|---|---|
| Compartment | Cartesian product of named axes | Stratification by declaration |
| Transition | source × destination × rate × `proportional_to` × `proportion_exponent` | Structured rate form |
| Subpopulation | `geodata` + `mobility` CSVs | Metapopulation from data files |
| Seeding | `initial_conditions` / `seeding` with named methods | Including importation draws |
| Modifier | parameter × date range × subpop × value; `StackedModifier` | The intervention grammar |
| Scenario | named list of modifiers | Declared in the config |
| Outcome | source × probability × delay × duration, chainable | The observation model |
| Inference | `inference:` block with data, likelihoods, and priors | Distributions on parameters and modifier values |
| Slot | `nslots` | Parallel inference chains |

## Strengths

- **The entire workflow is one document**: structure, geography, seeding, interventions, observation model, scenarios, priors, and fitting. Nothing else in the review has this scope in a single declarative artefact.
- **The outcomes block.** Source × probability × delay × duration, chainable, with automatic prevalence series from `duration`. The clearest treatment of the reporting cascade anywhere here.
- **The modifier grammar.** Parameter × time × space × value, composable via stacking, with named scenarios and fittable values — one mechanism serving interventions, time-varying parameters, and inference targets.
- **Compartments as a Cartesian product of named axes**, with transitions written over axis sublists, so stratification is declarative.
- **A structured rate form** (`rate` × ∏ `proportional_to`^`proportion_exponent`) that is analysable without being closed.
- **Calendar dates throughout.**
- **Production-proven at national scale**, in a setting where the model is rebuilt weekly against new data.
- **Deterministic and stochastic from the same configuration**, selected by the integration method.

## Limitations

- **YAML, again, with the same consequences**: expressions in strings (`rate: ["Ro * gamma"]`), no type checking, no editor support, deep nesting, and errors surfacing from the pipeline rather than a parser.
- **`proportional_to` / `proportion_exponent` is opaque.** Getting frequency- versus density-dependence right through parallel lists of compartment sets and exponent strings is error-prone, and the meaning is not evident from the notation.
- **Configurations are large.** A production scenario-hub config runs to hundreds of lines across included partials, and the partial-file mechanism (`*_part.yml`) is convention rather than language.
- **Two languages.** Python simulates, R infers; a user needs both toolchains.
- **Compartmental metapopulation only.** No agents, no networks.
- **No dimensional types.** `rate: ["Ro * gamma"]` is a string; `delay: {value: 5}` is 5 of something.
- **Heavy operational footprint** — batch scripts, cluster configuration, and a build system, reflecting its origin as a pipeline rather than a library.
- **Documentation is oriented to hub participants**, and the configuration reference is spread across the repository.

## Implications for the lingua franca

1. **Adopt the outcomes block, close to as-is.** Source × probability × delay × duration, chainable, is the observation model that most epidemiological data actually requires, and flepiMoP's version handles the cascade and the incidence/prevalence duality in one declaration. Combined with camdl's `observations` block (which supplies the likelihood, `~ neg_binomial(...)`, and multiple streams), this is close to a complete design: **flepiMoP declares how the latent process becomes an observed quantity; camdl declares how that quantity relates to the data.** Both halves are needed.
2. **Adopt the modifier grammar as the time-varying-parameter and scenario mechanism.** Parameter × time window × spatial or population scope × value, with stacking, with named scenario sets, and with values that may be distributions and therefore fitted. This single mechanism covers what EpiModel's scenario table does, what Epydemix's and MEmilio's contact dampings do, and what most "intervention" needs in a compartmental model amount to. It does *not* cover EMOD-style targeted individual interventions — those need the eligibility predicate — and the design should be explicit that these are two different layers.
3. **The scenario set belongs in the model file.** `scenarios: [Ro_lockdown, Ro_all]` declares what to run. camdl does this with named scenarios; flepiMoP does it with modifier composition. Either way, the set of counterfactuals is part of the artefact rather than a shell script, which is what makes a scenario comparison reproducible.
4. **Consider a structured rate form.** flepiMoP's coefficient × ∏ compartment^exponent is analysable — you can tell frequency-dependence from density-dependence by inspection, and you can stratify automatically — without closing the vocabulary. It sits between summer2's named flow kinds and epipack's free reaction. Whether the lingua franca wants this middle position, or wants named kinds over a free form, is a real design decision that this comparison sharpens.
5. **One document for the whole workflow is the right ambition.** flepiMoP demonstrates that structure, geography, interventions, observation, scenarios, and inference *can* live in one declarative artefact, and that doing so is what makes a weekly-rebuilt production model tractable. camdl splits this across `.camdl` and `fit.toml` deliberately, to keep structural identity separate from fitting configuration. Both are defensible; the lingua franca needs to decide where the seam goes, and flepiMoP is the evidence that putting it too far toward "the pipeline is a script" costs reproducibility.
6. **And YAML, for the fourth time in this review, is the wrong surface.** EMULSION, EMOD, EpiHiper, flepiMoP: four frameworks with good declarative semantics and four instances of the same ergonomic problem. This should now be treated as a settled finding rather than a repeated observation.
