# Vivarium

|  |  |
|---|---|
| **Language** | Python (pandas-backed state table) |
| **Paradigms** | Agent-based microsimulation, discrete time; disease as an explicit state machine |
| **Specification style** | Components assembled in a YAML model specification; behaviour written as Python `Component` subclasses |
| **Version reviewed** | 4.1.6 |
| **Licence** | BSD-3-Clause |
| **Code** | <https://github.com/ihmeuw/vivarium-suite> (the `ihmeuw/vivarium` repository is **archived**) |
| **Docs** | <https://vivarium-engine.readthedocs.io/> |
| **Paper** | — |
| **Note** | The package was renamed and moved during this review: `vivarium` → `vivarium-engine`, import path `vivarium` → `vivarium.engine`, source now in the `vivarium-suite` monorepo. The archived `v4.1.6` is the code reviewed here. |

## What it's for

Vivarium is IHME's microsimulation framework: individual-level simulation of health outcomes driven by Global Burden of Disease data, used for intervention and forecasting analyses. It is not primarily an infectious disease framework — its centre of gravity is risk factors, chronic disease, and health-system outcomes — and it is in this review for its **architecture**, which solves a problem that the infectious disease frameworks handle much less well.

That problem is: **how do independently-written components modify each other's quantities without stepping on each other?**

## How a model is specified

A YAML specification lists components; the components are Python classes.

```yaml
components:
    vivarium_examples.disease_model:
        population:
            - BasePopulation()
            - Mortality()
        disease:
            - SIS_DiseaseModel('diarrhea')
        risk:
            - Risk('child_growth_failure')
            - RiskEffect('child_growth_failure', 'infected_with_diarrhea.incidence_rate')
        intervention:
            - TreatmentEffect('treatment', 'infected_with_diarrhea.remission_rate')
configuration:
    randomness:
        key_columns: ['entrance_time', 'age']
```

Note `RiskEffect('child_growth_failure', 'infected_with_diarrhea.incidence_rate')` — a component whose entire declared purpose is *"this risk modifies that rate"*, with the target named as a string. That is the design idea.

A component:

```python
class DiseaseTransition(Transition):
    def setup(self, builder: Builder) -> None:
        self.base_rate = builder.configuration[self.cause_key][self.measure]
        builder.value.register_rate_producer(
            self.rate_name,
            source=self._risk_deleted_rate,
            required_resources=[self.joint_paf_pipeline],
        )

    def _probability(self, index) -> pd.Series:
        effective_rate = self.population_view.get(index, self.rate_name)
        return pd.Series(rate_to_probability(effective_rate))
```

## The value pipeline — the idea worth taking

Vivarium's core mechanism is the **value pipeline**. A component *registers a producer* for a named quantity:

```python
builder.value.register_rate_producer("diarrhea.incidence_rate", source=self._base_rate)
```

Other components *register modifiers* on that name:

```python
builder.value.register_value_modifier("diarrhea.incidence_rate", modifier=self._apply_risk_effect)
```

When the value is requested, the framework calls the source, then applies every registered modifier in a defined order, combining them with a declared **combiner** and finishing with a declared **post-processor** (`list_combiner`, `union_post_processor`, and others).

The consequences:

- A component that produces a rate does not know who modifies it.
- A component that modifies a rate does not know who produces it — only its name.
- **Multiple modifiers compose by a declared rule**, not by whoever ran last.
- The dependency graph between components is explicit (`required_resources=[...]`) and used to order setup.

Compare this to Starsim's connectors, where a connector writes `hiv.rel_trans[coinf] = 2.0` directly — general, and last-write-wins. Vivarium's mechanism is more constrained and, for exactly that reason, composes: two risk factors both raising an incidence rate combine correctly without either knowing about the other.

The `population_attributable_fraction` handling in the example above shows the pattern at work: several risks each contribute a PAF to a joint pipeline via `list_combiner`, the `union_post_processor` combines them into a single joint PAF, and the rate producer divides it out. That is a real epidemiological combination rule, declared once.

## The lifecycle

Vivarium's other structural idea is a **named lifecycle with priorities**. A component may implement:

`setup` → `on_post_setup` → then per timestep: `on_time_step_prepare` → `on_time_step` → `on_time_step_cleanup` → `on_collect_metrics` → finally `on_simulation_end`.

Each phase has a priority (`time_step_priority`, `collect_metrics_priority`, …, default 5), so ordering *within* a phase is declared rather than implicit in registration order. Lifecycle state is tracked (`lifecycle_states`) and violations raise `LifeCycleError` — a component cannot, for example, register a producer after setup has closed.

This is more structured than Starsim's `Loop` (which is inspectable and reorderable but has one step phase per module) and vastly more structured than EpiModel's `module.order`.

## Core abstractions

| Concept | Vivarium name | Notes |
|---|---|---|
| Population | a pandas DataFrame — the **state table**, one row per simulant | Columns are attributes |
| Access control | `PopulationView` | A component declares which columns it may read and write |
| Component | `Component` subclass | With `sub_components`, `CONFIGURATION_DEFAULTS`, lifecycle hooks and priorities |
| Value | named pipeline with a source, modifiers, a combiner and a post-processor | The composition mechanism |
| Lookup table | `builder.lookup`, `data_sources` in config | Data indexed by age, sex, year, location — the GBD-data interface |
| Disease | `Machine` / `State` / `Transition` / `TransientState` | An explicit state machine in `framework/state_machine.py` |
| Randomness | `builder.randomness` with `key_columns` | Common random numbers keyed on simulant attributes |
| Observer | `Component` registering observations | The results system |
| Artifact | `builder.data` over an HDF artifact | Pre-extracted input data, versioned |

## Randomness

`randomness: key_columns: ['entrance_time', 'age']` configures a **common-random-numbers scheme keyed on simulant attributes**: a simulant's random draw for a given decision is a deterministic function of its key columns and the decision's name. Two scenarios that differ only in an intervention give the same simulant the same draw.

This is the same capability Starsim gets from per-agent slots and hashed distribution names, arrived at independently, and configured declaratively rather than built in.

## Strengths

- **The value pipeline.** Named quantities with a source, registered modifiers, a declared combiner and a declared post-processor. The best answer in the review to "how do independent components modify each other's quantities", and directly applicable to a gap identified in the Starsim review.
- **Declared resource dependencies** (`required_resources=[...]`), used to order setup and detect cycles.
- **A structured lifecycle with priorities and enforced state.** Phases are named, ordering within a phase is declared, and lifecycle violations are errors.
- **`PopulationView` as declared read/write access** to state-table columns — a component states which columns it touches, and the framework enforces it. This is capability-based access control for model state, and nothing else in the review has it.
- **An explicit disease state machine** (`Machine` / `State` / `Transition` / `TransientState` / `Trigger`) separate from the component system.
- **Common random numbers keyed on declared attributes**, configured in the model specification.
- **Lookup tables and data artifacts** as the standard data interface, with `data_sources` declarable in `CONFIGURATION_DEFAULTS`.
- **Components listed in YAML**, so model composition is declarative even though component behaviour is code.
- **Sub-components**, giving hierarchical model composition.

## Limitations

- **Recently renamed and relocated.** `vivarium` → `vivarium-engine`, moved into a monorepo, with the reviewed repository archived. Anything built on the old import path needs migration.
- **Not an infectious disease framework.** No transmission, no contact structure, no networks, no force of infection. `SIS_DiseaseModel` in the examples is driven by an incidence *rate*, not by contact with infectious simulants. Adding transmission means writing it.
- **Component behaviour is Python.** The YAML lists components; it does not define them. The disease structure, the rates, and the effects are all code.
- **pandas state table.** Readable and slow relative to the array-based frameworks (Starsim, LASER, epiworld), and memory-hungry at scale.
- **Steep learning curve.** Value pipelines, lifecycle phases, population views, lookup tables, artifacts, and the resource system are a lot of machinery before the first line of epidemiology.
- **IHME/GBD-shaped.** The data interfaces assume GBD-style artifacts indexed by age, sex, year, and location.
- **No observation model or inference layer.** Simulation and observation only.
- **Sparse public documentation** of the value-pipeline semantics relative to their importance.

## Implications for the lingua franca

1. **Adopt the value-pipeline pattern for cross-module effects.** This is the most important thing in the entry. The lingua franca needs a way for a co-infection, a risk factor, an intervention, or a seasonal forcing to modify a quantity owned by another module *declaratively*, with a stated combination rule. Vivarium's shape — **named quantity, one producer, many registered modifiers, declared combiner** — solves the Starsim connector problem (last-write-wins, invisible dependencies) without giving up composability. It also makes the effect inspectable: you can ask "what modifies `diarrhea.incidence_rate`?" and get an answer.
2. **Declare the combination rule, because it is a modelling decision.** Two interventions that each reduce transmission by 40% might compose multiplicatively (`0.6 × 0.6`) or additively (`1 − 0.8`), and the choice changes the answer. Vivarium makes the combiner an explicit, named object; epidemics (Epiverse-TRACE) chose additive globally and documented why; most frameworks leave it implicit in the order of operations. **The lingua franca must let this be stated.**
3. **Adopt declared read/write access to state.** `PopulationView` means a component announces which state it reads and which it writes, and the framework enforces it. In a language whose goal is no hidden behaviour, a module that can silently write any other module's state is a hole. This is the same requirement as epymorph's declared `AttributeDef` requirements, extended from inputs to writes.
4. **A lifecycle with named phases and declared priorities beats a flat step list.** `time_step_prepare` → `time_step` → `time_step_cleanup` → `collect_metrics` gives components a place to put things without fighting over ordering, and priorities settle the remainder declaratively. Combined with `individual`'s queued-update semantics (which make within-phase order irrelevant), this is a good target: named phases, declared priorities, order-independence inside a phase.
5. **Common random numbers keyed on declared attributes is the portable formulation.** Starsim keys on an agent slot; Vivarium keys on `['entrance_time', 'age']` — configurable, and meaningful across populations that are not the same set of agents. The latter generalises better to comparing scenarios with different population sizes, which is a case Starsim's slot scheme handles awkwardly.
6. **Note that a framework can be declarative about *composition* while remaining imperative about *behaviour*.** Vivarium's YAML lists components and configuration; the components are Python. That is a coherent position and a common one (Starsim's dict form, EMOD's campaigns over a C++ engine). The lingua franca's ambition is stronger — behaviour declarative too — and Vivarium is a useful reminder of what the fallback looks like and how far it gets you.
