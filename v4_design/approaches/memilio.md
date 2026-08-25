# MEmilio

|  |  |
|---|---|
| **Language** | C++ (core), Python bindings |
| **Paradigms** | ODE, integro-differential (IDE), linear-chain-trick (LCT / GLCT), SDE, metapopulation (graph-ODE), agent-based, and hybrid agent–metapopulation |
| **Specification style** | C++ template model classes; parameters as typed tag structs |
| **Version reviewed** | current `main` |
| **Licence** | Apache-2.0 |
| **Code** | <https://github.com/SciCompMod/memilio> |
| **Docs** | <https://memilio.readthedocs.io/> |
| **Paper** | Bicker et al. (2026), [arXiv:2602.11381](https://doi.org/10.48550/arXiv.2602.11381), plus a paper per model family |
| **Ecosystem** | DLR / Forschungszentrum Jülich; `memilio-simulation`, `memilio-epidata`, `memilio-generation`, `memilio-surrogatemodel` |

## What it's for

MEmilio is a high-performance C++ library implementing many model families for the same epidemiological problem, so that the *formulation* can be varied and compared. Its model directory is the review's clearest map of the paradigm space:

```
ode_sir  ode_seir  ode_secir  ode_secirvvs  ode_secirts  ode_seair  ode_seirdb  ode_seirv  ode_mseirs4
ide_seir  ide_secir                         # integro-differential
lct_secir  glct_secir  lct_secir_2_diseases # linear chain trick
sde_sir  sde_sirs  sde_seirvv               # stochastic differential
ode_seir_metapop                            # metapopulation
abm  d_abm  graph_abm  smm                  # agent-based variants
hybrid                                      # agent ↔ metapopulation
```

For this project MEmilio is important less for its API than for what that list encodes: **the paradigms are not two or three, they are a lattice**, and moving between them changes what the model can express in specific, documented ways.

## How a model is specified

A model is a C++ class instantiated and configured through typed parameter tags:

```cpp
mio::osir::Model<ScalarType> model(1);   // 1 age group

model.populations[{mio::AgeGroup(0), mio::osir::InfectionState::Infected}]  = 1000;
model.populations[{mio::AgeGroup(0), mio::osir::InfectionState::Recovered}] = 1000;
model.populations[{mio::AgeGroup(0), mio::osir::InfectionState::Susceptible}] = total - ...;

model.parameters.set<mio::osir::TimeInfected<ScalarType>>(2);
model.parameters.set<mio::osir::TransmissionProbabilityOnContact<ScalarType>>(0.5);

auto& contact_matrix = model.parameters.get<mio::osir::ContactPatterns<ScalarType>>().get_cont_freq_mat();
contact_matrix[0].get_baseline().setConstant(2.7);
contact_matrix[0].add_damping(0.6, mio::SimulationTime<ScalarType>(12.5));

model.check_constraints();

auto integrator = std::make_unique<mio::EulerIntegratorCore<ScalarType>>();
auto sir = mio::simulate<ScalarType>(t0, tmax, dt, model, std::move(integrator));
auto interpolated = mio::interpolate_simulation_result<ScalarType>(sir);
```

Three features of this are worth extracting from the C++ noise.

**Parameters are types.** `TimeInfected`, `TransmissionProbabilityOnContact`, `ContactPatterns` are tag structs, and `parameters.set<T>(v)` is compile-time-checked. A misspelled parameter is a compilation error rather than a silently ignored key — the strongest parameter-name checking in the review, achieved through the type system rather than a schema.

**Populations are indexed multidimensionally by typed indices.** `populations[{AgeGroup(0), InfectionState::Infected}]` — a compile-time-checked coordinate in an (age × state) space, the same conceptual structure as MetaCast's coordinates or camdl's strata, expressed as a C++ multi-index.

**Constraints are checked explicitly.** `model.check_constraints()` is a separate, callable validation step.

## Model families and what distinguishes them

| Family | What it adds | Cost |
|---|---|---|
| **ODE** | Baseline compartmental | Exponential dwell times |
| **LCT / GLCT** | Linear chain trick: sub-compartments give Erlang (and generalised) dwell-time distributions | More state |
| **IDE** | Integro-differential: arbitrary dwell-time distributions via memory kernels | Non-Markovian, harder to solve and to initialise |
| **SDE** | Diffusion-approximation stochasticity | Approximation valid only at moderate-to-large populations |
| **Graph-ODE / metapop** | Nodes with explicit mobility, including traveller states in transit | Mobility formulation matters (there is a paper on exactly this) |
| **ABM** | Individual agents with locations and activity-driven contacts | Cost, and calibration difficulty |
| **Hybrid** | Agents where resolution is needed, metapopulation elsewhere | Coupling semantics at the boundary |

That MEmilio has a published paper for nearly every one of these, including one revisiting *"the Linear Chain Trick in epidemiological models: implications of underlying assumptions for numerical solutions"*, is itself the point: the choice between these formulations is a modelling decision with consequences, and the library is built to make the comparison possible.

## SBML import

`cpp/sbml_model_generation/sbml_to_memilio.cpp` reads SBML and generates MEmilio model files. This is the only **model-format import path** in the reviewed set other than MIRA's, and it is a direct precedent for the lingua franca's interconversion goal: a compiler from an interchange format to a framework's native representation.

## Time

Simulation time is `ScalarType` with an explicit `dt` passed to `simulate()`, and integrators are pluggable (`EulerIntegratorCore`, adaptive RK). `mio::SimulationTime<ScalarType>` is a distinct type used for scheduling (contact-matrix dampings), which is a partial dimensional typing — time is a type, rates are not.

`interpolate_simulation_result()` resamples adaptive-integrator output onto a regular grid, which is a small but frequently-needed operation that most frameworks leave to the user.

## Interventions

Interventions are **dampings on the contact matrix**: `contact_matrix[0].add_damping(0.6, SimulationTime(12.5))` reduces contacts to 60% from day 12.5. Multiple dampings compose. Vaccination and testing appear as extra compartments in the `secirvvs` and `secirts` model variants rather than as an intervention vocabulary.

This is the same design as Epydemix's contact-layer interventions, with the same scope and the same limit.

## Strengths

- **The widest paradigm coverage in the review by a wide margin**, with each family implemented properly rather than approximated.
- **Dwell-time distributions treated as a first-class modelling axis**: exponential (ODE), Erlang (LCT), generalised (GLCT), arbitrary (IDE). This is the single most under-served concept in every other framework here.
- **Hybrid agent–metapopulation models**, with the coupling as an explicit design object and a paper justifying it.
- **Parameters as compile-time-checked types.** Nothing else in the review checks parameter names at build time.
- **Typed multi-dimensional population indices** (age × infection state), checked.
- **An SBML import path** — a working precedent for format-to-framework compilation.
- **Explicit `check_constraints()`** as a separate validation phase.
- **Result interpolation** onto regular grids from adaptive integrators.
- **Serious performance and parallelism**, with published national-scale applications.
- **Python bindings and a surrogate-model package**, so the C++ core is reachable from an analysis workflow.

## Limitations

- **The model is C++ template code.** Adding a compartment means writing a new model class. There is no model-definition language, no serialisable structure, and no way to introspect a model as data.
- **Every model family is a separate implementation.** `ode_seir`, `lct_secir`, `ide_seir`, and `sde_sir` are different C++ classes with different parameter sets. There is no single specification from which the paradigms are generated — which is precisely the gap this project is trying to fill, and MEmilio is the clearest demonstration that the gap is real and expensive: this library is a large, well-funded, multi-year effort, and the paradigm coverage was bought by writing each one separately.
- **High barrier to entry.** C++17 templates, CMake, and a large API.
- **No intervention vocabulary** beyond contact-matrix dampings.
- **No observation model.** Comparison to data happens outside the library.
- **Rates are not dimensionally typed**, though time partially is.

## Implications for the lingua franca

1. **MEmilio is the cost estimate for *not* having a lingua franca.** Twenty-plus model families, each implemented independently, each with its own paper, sharing a parameter convention but not a specification. If one typed specification could generate the ODE, LCT, SDE, and metapopulation forms of the same model, that is the value proposition, quantified.
2. **Dwell-time distribution is a first-class modelling axis and the language must express it.** ODE means exponential; LCT means Erlang with *k* stages; IDE means an arbitrary kernel. camdl handles this with `via erlang` / `via hyper_erlang` and — importantly — classifies it as *residence structure* rather than population stratification. That distinction is right, and MEmilio's model list is the evidence for how much rides on it. Most frameworks in this review silently assume exponential dwell times and never say so.
3. **Adopt the paradigm lattice as the design's map.** Not "compartmental / metapopulation / SDE / ABM" as four options, but a lattice with axes: aggregation (compartment ↔ agent), stochasticity (deterministic ↔ diffusion ↔ jump), dwell time (exponential ↔ Erlang ↔ arbitrary), and space (single ↔ patch ↔ explicit mobility). A model specification selects a point; a conversion moves along an axis. This is a more useful framing than a list of paradigms, and it makes the "which conversions are meaningful" question answerable per axis.
4. **Compile-time parameter checking is achievable and worth it.** MEmilio gets it from C++ types; camdl gets it from its compiler; Starsim gets a runtime version from `define_pars` locking. EpiModel, Epydemix, and MetaCast get nothing. A misspelled parameter name should be a compile error.
5. **Study the SBML importer.** It is a working instance of the exact artefact this project needs (interchange format → executable framework model), including the parts that do not map cleanly.
6. **Traveller state in transit is a real modelling concern.** MEmilio has a paper on how to represent people who are in the process of moving between patches. A metapopulation vocabulary that only has "in patch i" or "in patch j" cannot say this, and the lingua franca should decide deliberately whether it can.
