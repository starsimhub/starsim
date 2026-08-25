# epiworld (epiworldR / epiworldpy)

|  |  |
|---|---|
| **Language** | Header-only C++ core; R and Python bindings |
| **Paradigms** | Agent-based — networks, fully-connected mixing, and mixing-matrix models |
| **Specification style** | Model built from states, transition functions, viruses, and tools, either from templates or with a model builder |
| **Version reviewed** | epiworldR (CRAN) |
| **Licence** | MIT |
| **Code** | <https://github.com/UofUEpiBio/epiworld>, <https://github.com/UofUEpiBio/epiworldR> |
| **Docs** | <https://uofuepibio.github.io/epiworldR/> |
| **Paper** | Meyer & Vega Yon (2023), *JOSS* 8(90): 5781, [10.21105/joss.05781](https://doi.org/10.21105/joss.05781) |
| **Ecosystem** | `epiworld` (C++ core), `epiworldR`, `epiworldpy`, `epiworldRShiny` — all over the same engine |

## What it's for

epiworld is a fast agent-based epidemic engine — the documentation claims ~30 million agent-days per second — packaged as a header-only C++ library with thin bindings in R and Python. Its two distinguishing features are **multiple simultaneous diseases** and **a virus/tool separation** that treats pathogens and interventions as the same kind of object attached to agents.

Its relevance to this project is the binding story: `epiworldR` and `epiworldpy` are not parallel implementations, they are two front ends over one engine, so **model semantics are identical across host languages**. Nothing else in the review can claim that.

## How a model is specified

Templates cover the common cases — `ModelSIR`, `ModelSEIR`, `ModelSIRCONN`, `ModelSEIRCONN`, `ModelSEIRD`, `ModelSEIRMixing`, `ModelSEIRMixingQuarantine`, `ModelDiffNet` — with `CONN` meaning fully connected, `Mixing` meaning a contact matrix over groups, and the plain forms meaning a network.

Beyond templates, the model builder assembles a model from parts:

```r
model <- Model()

add_param(model, "Rec Rate", 0.1)
add_param(model, "Trans Rate", 0.5)

# States, each with an update function
update_fun_s <- update_fun_susceptible()
add_state(model, "S", update_fun_s)

update_fun_i <- update_fun_rate(param_names = "Rec Rate", target_states = 2L)
add_state(model, "I", update_fun_i)

add_state(model, "R", NULL)     # absorbing

# A virus is a separate object with its own rates
flu <- virus("Flu", 0, .5, .2, .01, prevalence = 0, as_proportion = TRUE)
virus_set_state(flu, 1, 2, 2)   # which states the virus moves agents into
set_prob_infecting_ptr(flu, model, "Trans Rate")
set_distribution_virus(flu, distribute_virus_randomly(1L, FALSE))
```

Two features of this are worth extracting.

**States carry update functions.** A state is a name plus a function describing what happens to an agent in it. `update_fun_rate(param_names = "Rec Rate", target_states = 2L)` is a *prefab* — a parameterised, named update rule — so the common cases are declarative even though the mechanism is functional. `set_prob_infecting_ptr(flu, model, "Trans Rate")` binds a virus's infection probability to a named model parameter by pointer, so calibration can move the parameter and every virus using it follows.

**States are indexed by integer.** `target_states = 2L`, `virus_set_state(flu, 1, 2, 2)`. This is fast and it is a readability and correctness hazard: the example needs a comment (`0 = S, 1 = I, 2 = R`) to be intelligible.

## Core abstractions

| Concept | epiworld name | Notes |
|---|---|---|
| Agent | agent | With a status (state index), a set of viruses, and a set of tools |
| State | `add_state(model, name, update_fun)` | Name plus update function; indexed by integer |
| **Virus** | `virus(name, ...)` with its own rates and `virus_set_state` | A first-class transmissible object attached to agents; a model can carry many |
| **Tool** | `tool(...)` | A first-class *protective* object attached to agents — vaccine, mask, treatment — with its own effects on susceptibility, transmissibility, recovery, and death |
| Entity | `entity` | Groups agents (households, schools) for mixing-matrix models |
| Parameter | `add_param(model, name, value)` | Named; bindable by pointer into virus/tool rates |
| Distribution | `distribute_virus_randomly`, `distribute_tool_*` | How viruses and tools are seeded across the population |
| Network | edge list, or `CONN` (fully connected), or `Mixing` (contact matrix) | Three transmission structures over one engine |
| Event | `events.R`, global actions | Scheduled model-level actions |
| Inference | `LFMCMC` | Likelihood-free MCMC, built in |
| Diagram | `ModelDiagram` | Generates a state-transition diagram from a run |

## The virus/tool duality

The design idea most worth taking. epiworld models an agent as carrying two collections:

- **Viruses**: transmissible objects with infection probability, recovery rate, death probability, and a mapping to states.
- **Tools**: protective objects with multiplicative effects on susceptibility, transmissibility, recovery, and death.

An agent's effective susceptibility is a function of the tools it carries; its effective transmissibility is a function of the viruses it carries and the tools it carries. Multi-disease and multi-intervention composition fall out of this automatically: adding a second virus does not change the first, and a mask reduces susceptibility to all of them.

This is a genuinely different factoring from Starsim's (where diseases are modules with their own state, and interventions modify module state) and is arguably cleaner for the specific case of "several pathogens, several protective measures, all multiplicative". It is less general — a tool is a bundle of four multipliers, not arbitrary behaviour — and that is exactly why it composes.

## Time

Discrete steps, unitless. No dates, no dimensional types. Rates are per-step probabilities and the modeller holds the interpretation.

## Strengths

- **One engine, two host languages, identical semantics.** A model in `epiworldR` and the same model in `epiworldpy` are the same C++ computation. This is the strongest multi-language story in the review, and header-only C++ is why it was affordable.
- **Multi-disease by construction**, through the virus collection rather than as a framework feature.
- **The virus/tool duality.** Pathogens and protective measures as first-class agent-attached objects with declared multiplicative effects. Composition is automatic.
- **Three transmission structures over one engine** — network, fully-connected, and mixing-matrix — selected by model class. The same conceptual disease runs as an ABM or as a mixing-matrix model, which is Starsim's `Route` idea arrived at independently.
- **Prefab update functions** (`update_fun_rate`, `update_fun_susceptible`) that make the common state behaviours declarative.
- **Parameters bound by pointer** into virus and tool rates, so calibration targets propagate.
- **Fast.** The performance claims are the reason the C++ core is header-only and the bindings are thin.
- **`ModelDiagram`** — automatic state-transition diagram generation.
- **Likelihood-free MCMC built in.**
- **Small, comprehensible API.** The whole vocabulary is states, viruses, tools, entities, and parameters.

## Limitations

- **Integer state indices.** `target_states = 2L` and `virus_set_state(flu, 1, 2, 2)` are unreadable without a comment mapping indices to names, and a wrong index is a silently different model. This is the clearest argument in the review for named references over positional ones.
- **Update functions are C++/R closures.** The prefabs are declarative; anything custom is code, and custom code in the R binding crosses the language boundary.
- **Tools are limited to four multiplicative effects.** That is why they compose, and it means anything else — a diagnostic with a result, a treatment with a duration, a screening cascade — needs different machinery.
- **No dimensional types, no dates, unitless steps.**
- **Networks are inputs, not specifications.** An edge list is supplied; there is no vocabulary for describing the network to be generated (contrast EpiModel's `netest`).
- **No observation model.** LFMCMC compares summary statistics.
- **Single institution, modest adoption**, though CRAN presence and JOSS publication give it more visibility than most of its size.
- **Model structure is imperative.** `add_state`, `add_param`, `virus_set_state` in sequence; the model is not a document.

## Implications for the lingua franca

1. **The virus/tool duality is worth serious consideration as a primitive.** "Things an agent carries that make it transmit" and "things an agent carries that protect it", both with declared multiplicative effects on a fixed set of quantities (susceptibility, transmissibility, recovery, death), gives automatic and correct composition for multi-disease, multi-intervention models. It is more constrained than Starsim's connectors and, for exactly that reason, does not have Starsim's last-write-wins problem. Where it runs out — cascades, diagnostics, durations — is well defined and can be handled by other machinery.
2. **One engine, many front ends, is the right distribution architecture.** epiworld shows this is achievable cheaply with a header-only core. For a lingua franca whose whole point is that a model has one canonical representation, having R and Python front ends that are provably the same computation (rather than parallel implementations that drift) is close to a requirement. Starsim's `rstarsim` achieves the same thing by embedding Python; epiworld's is the cleaner version.
3. **Names, never indices.** `target_states = 2L` is the anti-pattern. Every reference in the language — to a state, a transition, a parameter, a stratum — must be by name, and the compiler must check it exists. camdl's E332/E263 errors are the positive version of this rule.
4. **Prefab parameterised behaviours are a good middle layer.** `update_fun_rate(param_names = "Rec Rate", target_states = 2L)` is a named, parameterised rule rather than either a fixed template or arbitrary code. This is the same shape as Epydemix's registerable transition kinds and summer2's named flow kinds, and the convergence across three frameworks is worth noting: **a closed set of named, parameterised behaviours covers most real models, and the escape hatch is where the language's guarantees stop.**
5. **Note the third independent arrival at route abstraction.** Starsim (`Network` / `MixingPool` behind one `Route`), epiworld (`ModelSEIR` / `ModelSEIRCONN` / `ModelSEIRMixing` over one engine), and epipack (network versus well-mixed) have all separated the disease process from the contact structure. Three independent arrivals is strong evidence this is the correct seam, and the differences between how they cut it are the detail the design needs to get right.
6. **Automatic model diagrams should be a deliverable, not an extra.** `ModelDiagram` in epiworld, `fillcolor` in EMULSION, `comp_plot` in EpiModel, and `plot_step_order` in Starsim all point the same way: if the model is data, a picture of it is nearly free, and it is the artefact non-modellers actually read.
