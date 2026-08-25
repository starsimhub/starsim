# Epydemix

|  |  |
|---|---|
| **Language** | Python (optional Numba JIT) |
| **Paradigms** | Compartmental — stochastic (default) and deterministic, with age/contact-matrix structure |
| **Specification style** | Imperative construction of a model object: compartments, then transitions, then population |
| **Version reviewed** | 1.x |
| **Licence** | GPL-3.0 |
| **Code** | <https://github.com/epistorm/epydemix> |
| **Docs** | <https://epydemix.readthedocs.io/>, <https://www.epydemix.org/> |
| **Paper** | Gozzi et al. (2025), *PLOS Comp Biol* 21(11): e1013735, [10.1371/journal.pcbi.1013735](https://doi.org/10.1371/journal.pcbi.1013735) |
| **Ecosystem** | Epistorm — `epydemix-data` (400+ locations of population and contact matrices), plus dashboard and forecast applications |

## What it's for

Epydemix is compartmental epidemic modelling with the data already attached. It is named in the project brief as an interconversion target, and it represents the "batteries included" end of the design space: the model vocabulary is small and unsurprising, and the value is in `epydemix_data` — population structures and contact matrices for over 400 locations — plus a built-in Approximate Bayesian Computation calibration stack.

It is the clearest example in the review of a framework whose *data bundling* is the product and whose *specification language* is deliberately minimal.

## How a model is specified

```python
from epydemix import EpiModel

model = EpiModel(name="SIR Model", compartments=["S", "I", "R"])
model.add_transition(source="S", target="I", params=(0.3, "I"), kind="mediated")
model.add_transition(source="I", target="R", params=0.1, kind="spontaneous")

results = model.run_simulations(start_date="2024-01-01", end_date="2024-04-10", Nsim=100)
df = results.get_quantiles_compartments()
```

Two transition kinds, `spontaneous` and `mediated`, distinguished by whether the rate depends on another compartment. `params=(0.3, "I")` means "rate 0.3, mediated by compartment I". New kinds can be registered:

```python
model.register_transition_kind("my_kind", my_rate_function)
```

which is a small but real extension point — the transition vocabulary is closed by default and openable by the user, rather than being either fixed or arbitrary.

Population and mixing come from the data package:

```python
from epydemix.population import load_epydemix_population

population = load_epydemix_population(
    population_name="United_States",
    contacts_source="mistry_2021",
    layers=["home", "work", "school", "community"],
)
model.set_population(population)
```

## Core abstractions

| Concept | Epydemix name | Notes |
|---|---|---|
| Compartment | string in `compartments` | `add_compartments`, `clear_compartments` |
| Transition | `Transition(source, target, kind, params)` | An object with four fields — data, not code |
| Transition kind | `"spontaneous"`, `"mediated"`, or user-registered | The extension point |
| Parameter | `add_parameter`, `override_parameter` | Named, with a separate *override* mechanism for time-varying values |
| Population | `Population` with age groups and layered contact matrices | From `epydemix_data` or user-supplied |
| Intervention | `add_intervention(layer_name, start_date, end_date, reduction_factor=/new_matrix=)` | Contact-layer reductions, date-scheduled |
| Calibration | ABC (rejection, SMC) with metrics in `epydemix.calibration` | The headline feature |
| Results | `SimulationResults` with `get_quantiles_compartments()` | Quantiles across stochastic replicates as the default output form |

## Time

Simulations run on **calendar dates** (`start_date="2024-01-01"`), not abstract steps. Interventions are specified by date. This is a small thing that matters a great deal in practice: real interventions have real dates, and a framework that requires converting them to step indices creates an entire class of off-by-one errors.

There are no dimensional types; rates are floats interpreted per time step.

## Interventions

One mechanism, narrowly scoped and well matched to its use case:

```python
model.add_intervention(
    layer_name="school",
    start_date="2020-03-15",
    end_date="2020-06-01",
    reduction_factor=0.2,
)
```

An intervention is a **contact-layer modification over a date range** — either a multiplicative reduction or a wholesale replacement matrix. That covers school closures, workplace restrictions, and lockdowns, which is most of what a COVID-era scenario model needs, and nothing else. Pharmaceutical interventions are modelled as extra compartments and transitions.

The `override_parameter` / `delete_override` / `clear_overrides` mechanism handles time-varying parameters separately.

## Calibration

ABC is first-class rather than bolted on: rejection ABC and ABC-SMC, with a metric library, running against the stochastic simulator directly. For a framework whose models are cheap and stochastic and whose likelihood is awkward, this is the right inference choice, and having it built in with the data package means a user can go from "United States, mistry_2021" to a calibrated model without writing inference code.

## Agent-facing CLI

Worth noting for this project specifically: Epydemix ships a CLI explicitly designed for LLM agents and automation, covering discovery, config validation, running, and result inspection, with a documented contract (`AGENT.md`). It is one of the few frameworks in the review that has thought about being driven by a machine, and the shape it chose — a CLI with a documented contract rather than a Python API — is a data point for how AI-native tooling gets built in practice.

## Strengths

- **Transitions are objects with four fields**, so the model is inspectable data rather than code.
- **A closed-but-extensible transition vocabulary.** `spontaneous` / `mediated` covers the common cases; `register_transition_kind` opens it without making it arbitrary. This is a good middle position between summer2's closed set and epipack's fully general reaction.
- **Calendar dates throughout**, including for interventions.
- **Population and contact data for 400+ locations bundled**, with multiple contact-matrix sources and layers. Nothing else in the review removes this much setup friction.
- **ABC calibration built in**, matched to the stochastic simulator.
- **Quantiles across replicates as the default result form**, which is the right default for a stochastic model and is usually left to the user.
- **An explicit agent-facing CLI contract.**
- **Small API.** The whole model vocabulary is compartments, transitions, parameters, population, interventions.

## Limitations

- **Model structure is built imperatively.** `add_transition` calls in sequence; the model is a Python object, not a document. No serialisation of structure.
- **No stratification operator.** Age structure comes from the `Population`'s contact matrix, which stratifies mixing but not compartments — a model where progression differs by age needs the compartments written out.
- **No dimensional types**, no units, no rate/probability distinction.
- **Interventions are contact-layer-only.** Vaccination, treatment, screening, and testing have no vocabulary; they are compartments and transitions.
- **No observation model.** Calibration compares simulated compartment counts to data through a metric function; there is no declared reporting process, delay, or ascertainment.
- **Deterministic mode is secondary.** The design centre is stochastic simulation plus ABC.
- **GPL-3.0**, which is more restrictive than most of the rest of the reviewed set and may matter for interconversion tooling.

## Implications for the lingua franca

1. **Closed-but-registerable transition kinds are a good middle position.** Epydemix's `spontaneous` / `mediated` plus `register_transition_kind` gets most of summer2's benefit (the framework knows what a transition means) without summer2's ceiling (a flow whose semantics are not in the list is unrepresentable). Worth considering as the shape of the lingua franca's transition vocabulary: a small named core, an escape hatch, and the escape hatch's cost stated explicitly (things that use it lose automatic stratification, symbolic analysis, and so on).
2. **Calendar dates should be the default, not an option.** Epydemix, Starsim, camdl (with `origin`), and flepiMoP all support real dates; odin, summer2, EpiModel, and epipack do not, or do so awkwardly. Interventions have dates, data has dates, and the conversion to step indices is a recurring source of error.
3. **Data bundling is a first-class design concern, not packaging.** Epydemix's real contribution is that `load_epydemix_population("United_States", "mistry_2021")` works. A lingua franca that can express a contact matrix but cannot say *which* contact matrix, from *which* source, for *which* place, has moved the problem rather than solved it. Consider whether population and mixing references belong in the language — as camdl's `read("data/lga_pop.tsv", column = "patch")` partly does — with provenance attached.
4. **Quantiles-across-replicates as the default output.** For any stochastic paradigm, the natural result object is a distribution over trajectories, not a trajectory. Epydemix makes this the default; most frameworks make the user aggregate.
5. **Note the agent-facing CLI.** Epydemix chose a documented CLI contract as its machine interface rather than exposing its Python API. camdl chose the same (CLI plus offline versioned docs). For an AI-native language, the interface an agent drives may matter as much as the syntax it writes.
6. **The intervention gap repeats.** Epydemix's interventions are contact reductions and nothing else; summer2 and odin have none; Starsim and EMOD have rich ones with no structural specification underneath. The pattern across the whole review is that structural rigour and intervention expressiveness have not yet appeared in the same framework.
