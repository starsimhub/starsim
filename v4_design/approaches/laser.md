# LASER

|  |  |
|---|---|
| **Language** | Python with Numba (and C/OpenMP where needed) |
| **Paradigms** | Agent-based, spatial/metapopulation, discrete time; large-scale (national to continental) |
| **Specification style** | A model is a `LaserFrame` of agent properties plus a list of **components**, each with an init and a step function |
| **Version reviewed** | `laser-core` 0.6.x |
| **Licence** | MIT |
| **Code** | <https://github.com/InstituteforDiseaseModeling/laser-core> |
| **Docs** | <https://docs.idmod.org/projects/laser/en/latest/> |
| **Paper** | — |
| **Ecosystem** | `laser-core` plus disease packages: `laser-polio`, `laser-measles` |
| **Note** | Not in the landscape database; included because it is an IDM framework whose design point differs materially from Starsim's |

## What it's for

LASER is IDM's engine for spatial agent-based models at eradication scale — national polio and measles models over thousands of nodes and hundreds of millions of agent-timesteps. Its design is derived almost entirely from a performance argument, and that argument produces a specification style worth contrasting with Starsim's.

The stated principle:

> The core principle of LASER's design is to optimize computational efficiency by aligning the system with what modern CPUs and GPUs excel at — performing billions of floating-point operations per second — while minimizing costly operations like runtime memory allocation and random memory access.

## How a model is specified

The population is a `LaserFrame` — a mutable columnar table, "similar to a db table or a Pandas DataFrame":

```python
from laser_core import LaserFrame

class Model:
    pass

model = Model()
model.population = LaserFrame(capacity=1000)
model.population.add_scalar_property("disease_state", dtype=np.int32, default=0)
model.population.add_vector_property("position", length=3, dtype=np.float32)
start, end = model.population.add(10)
```

Behaviour is added as **components**, each an object with an initialiser and a `step` function operating on one or more properties. The top-level script assembles them:

```python
model.components = [VitalDynamics_ABM, DiseaseState_ABM, Transmission_ABM, RI_ABM, SIA_ABM]
```

(the class names above are from `laser-polio`, which is the reference downstream model). Each component processes the columns it cares about, once per timestep, sequentially over preallocated arrays.

Parameters are a `PropertySet` — a dict with dot access and `+` composition:

```python
ps1 = PropertySet({'immunity': 'high', 'region': 'north'})
ps2 = PropertySet({'infectivity': 0.7})
combined = ps1 + ps2
```

## The four design principles

Stated in the architecture document, and they explain nearly everything else:

1. **Preallocate memory.** All arrays allocated at initialisation; nothing allocated during the run.
2. **Sequential array access.** Iterate through preallocated arrays, ideally once per timestep, for cache friendliness.
3. **Fixed data structures.** Agents are marked dead rather than removed; "preborn" agents are in the array from the start and activated at the right timestep; the array never resizes.
4. **Time-specific data slots.** Reports and outputs allocate a slot per timestep and location in advance.

The consequences for *specification* are direct: the population has a fixed capacity computed up front (`calc_capacity`), births are activations of pre-allocated rows, deaths are flags, and results are preallocated arrays indexed by (timestep, node). A model must therefore declare its maximum size and its output shape before it runs — which is a constraint, and also a form of declaration.

## Core abstractions

| Concept | LASER name | Notes |
|---|---|---|
| Population | `LaserFrame` | Columnar; `add_scalar_property`, `add_vector_property`, `add`, `sort`, `squash`; HDF5-backed |
| Agent property | a column | Typed (`dtype`), with a default |
| Node / patch | `node_id`, a built-in property | Geospatial focus is in the core; single-node models set it uniformly |
| Component | init function + `step` function | The unit of behaviour |
| Parameters | `PropertySet` | Dot-access dict, composable with `+`, JSON-serialisable |
| Migration | `laser_core.migration` | `gravity`, `competing_destinations`, `stouffer`, `radiation`, `distance`, `row_normalizer` — a library of published spatial interaction models |
| Scheduling | `SortedQueue` | Priority queue for scheduled events (non-disease deaths) |
| Demographics | `laser_core.demographics` | Age pyramids, Kaplan–Meier estimators, date-of-birth / expected-date-of-death initialisation |

## What distinguishes it from Starsim

Both are IDM Python agent-based frameworks. The differences are instructive because they are design choices, not accidents:

| | Starsim | LASER |
|---|---|---|
| **Agent lifecycle** | UIDs added and removed; `auids` view of the living | Fixed capacity; preborn rows activated; dead rows flagged |
| **State ownership** | Each module declares its own states | One shared `LaserFrame`; components add columns to it |
| **Time** | Typed (`dur`/`per`/`prob`/`freq`), per-module timelines, calendar dates | Unitless integer timesteps |
| **Transmission** | `Route` abstraction — networks and mixing pools | Node-level force of infection with a migration matrix |
| **Spatial** | Optional, via spatial networks | Built in: `node_id` is a core property, and the migration library is first-class |
| **Contract** | `ss.Module` with a declared lifecycle and locked state | A component is an object with a `step` function |
| **Results** | Declared `ss.Result` objects with labels and timevecs | Preallocated arrays by (timestep, node) |
| **Scale target** | 10⁴–10⁶ agents | 10⁷–10⁹ agent-timesteps |

The migration library is the clearest single differentiator: `gravity`, `competing_destinations`, `stouffer`, and `radiation` are the four standard published spatial-interaction models, implemented, parameterised, and sanity-checked. Nothing else in this review ships that.

## Time

Unitless integer timesteps, with dates handled by the downstream disease packages (`laser-polio` has date utilities in `utils.py`). There are no dimensional types.

Scheduling is via `SortedQueue`, a priority queue of future events — the same mechanism as `individual`'s `TargetedEvent` and Starsim's `ti_*` arrays, chosen here for the same reason (avoid scanning all agents for something that applies to few).

## Strengths

- **Explicit, stated performance principles**, with the whole design derived from them. Rare, and it makes the trade-offs auditable.
- **A migration model library**: gravity, competing destinations, Stouffer, radiation, with distance calculation, row normalisation, and parameter sanity checks. This is the best spatial-interaction vocabulary in the review.
- **Spatial structure in the core.** `node_id` is a built-in property rather than a modelling convention.
- **Columnar agent storage** (`LaserFrame`) with typed scalar and vector properties, HDF5 persistence, and `sort`/`squash` operations.
- **Preallocation as a discipline**: no runtime allocation, no array resizing, cache-friendly sequential access, and therefore predictable performance at scale.
- **Preborn agents** as a births mechanism — an elegant way to avoid dynamic growth.
- **`PropertySet`** — composable, dot-accessible, JSON-serialisable parameter sets.
- **Demographic initialisation utilities**: age pyramids, Kaplan–Meier, date-of-death sampling.
- **Genuinely small core.** `laser-core` is a handful of modules; the disease models live outside it.

## Limitations

- **No epidemiological vocabulary in the core.** No disease, transmission, network, or intervention concepts — those are in `laser-polio` and `laser-measles`, written per model. `laser-core` is closer to `individual` than to Starsim in what it provides.
- **Components are code with no declared contract.** A component is an object with a `step` function; there is no declaration of which properties it reads or writes, no lifecycle beyond init-and-step, and no ordering mechanism beyond list position.
- **Order dependence is total and undeclared.** The component list is the execution order, and nothing states the dependencies that ordering satisfies.
- **Shared mutable state.** All components write columns of one `LaserFrame`. This is fast and it is exactly the hazard Vivarium's `PopulationView` and value pipelines exist to prevent.
- **Unitless time**, no dates in the core, no dimensional types.
- **Fixed capacity must be estimated up front**, and a model that exceeds it fails.
- **The architecture document suggests "declarative behavior is encouraged, with step functions optionally described in SQL-like syntax"** — an interesting idea, but the implemented interface is Numba/NumPy code.
- **Small user base**, essentially IDM plus collaborators, with two disease packages.
- **No inference layer in the core**; calibration lives in the disease packages.

## Implications for the lingua franca

1. **The migration library is directly reusable and should be referenced by name.** Gravity, radiation, Stouffer, and competing-destinations are the published models for spatial interaction, and a metapopulation vocabulary needs to be able to say "the mobility matrix is a gravity model with parameters k, a, b, c" rather than only "here is a matrix". This is the movement-model counterpart to EpiModel's network-by-target-statistics point: **structure should be specifiable by its generating model, not only by its realisation.**
2. **LASER and Starsim together bound the agent-based design space at IDM**, and the differences are the design questions. Fixed-capacity preallocation versus dynamic UIDs; shared state table versus module-owned states; unitless steps versus typed time; spatial-in-core versus spatial-as-network. A lingua franca that has to compile to both needs to know which of these are *semantic* (typed time, state ownership) and which are purely *implementation* (preallocation, dead-flagging). Most are implementation — which is encouraging, and worth confirming deliberately.
3. **Preallocation is an implementation strategy that the language can enable.** If the model declares its maximum population, its node set, and its output shape — which a declarative language naturally does — a backend can preallocate without the modeller thinking about it. The constraint LASER imposes on its users is information the lingua franca would already have.
4. **The shared-mutable-state hazard appears here in its strongest form**, and reinforces the recommendation from the Vivarium and Starsim entries: components writing arbitrary columns of a shared table is fast and unanalysable. Declared read/write sets cost nothing at run time — they are compile-time information — and would let a backend keep LASER's performance while recovering the analysis.
5. **"Declarative behaviour with SQL-like step functions" is worth pursuing, because LASER's authors already identified the need.** A step function that filters agents by a predicate, samples a subset, and updates a column is exactly the "select a set, thin it, act" primitive identified in the `individual` entry, and it is expressible declaratively. That LASER's own architecture document reaches for this is corroborating evidence for the shape of the primitive.
6. **Spatial structure deserves first-class status.** LASER puts `node_id` in the core; epymorph puts geography in the RUME; MEmilio has graph-ODE and traveller states; Starsim treats space as one network among several. For an eradication-scale or subnational-planning audience, space is not one stratification dimension among many, and the design should decide deliberately whether it gets special treatment.
