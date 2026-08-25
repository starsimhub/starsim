# epymorph

|  |  |
|---|---|
| **Language** | Python (SymPy for the disease model) |
| **Paradigms** | Metapopulation — compartmental dynamics within nodes, discrete movement between them |
| **Specification style** | Composition of independently declared parts: IPM (disease) × MM (movement) × scope (geography) × initializer × parameters |
| **Version reviewed** | current `main` |
| **Licence** | (see repository) |
| **Code** | <https://github.com/NAU-CCL/epymorph> |
| **Docs** | <https://docs.epimorph.org> |
| **Paper** | — (EpiMoRPH project, NAU) |

## What it's for

epymorph is a spatial metapopulation framework whose organising idea is that **the disease process, the movement process, and the geography are three independent things that get combined**. You can swap the disease without touching the movement model, swap the movement model without touching the disease, and re-run either over a different geographic scope with the same code.

This is the cleanest separation of concerns in the review, and it is directly relevant: the lingua franca's paradigm-independence goal is a similar factoring, one axis over.

## How a model is specified

Four parts and a container.

**The IPM (Intra-Population Model)** — the disease process, declared as compartments plus symbolic edges:

```python
from epymorph.kit import *
from sympy import Max

class SIRS(CompartmentModel):
    compartments = [
        compartment("S"),
        compartment("I"),
        compartment("R"),
    ]

    requirements = [
        AttributeDef("beta",  type=float, shape=Shapes.TxN, comment="infectivity"),
        AttributeDef("gamma", type=float, shape=Shapes.TxN, comment="progression I→R"),
        AttributeDef("xi",    type=float, shape=Shapes.TxN, comment="progression R→S"),
    ]

    def edges(self, symbols):
        [S, I, R] = symbols.all_compartments
        [β, γ, ξ] = symbols.all_requirements
        N = Max(1, S + I + R)   # avoid division by zero
        return [
            edge(S, I, rate=β * S * I / N),
            edge(I, R, rate=γ * I),
            edge(R, S, rate=ξ * R),
        ]
```

**The MM (Movement Model)** — one or more `MovementClause`s, each declaring what it needs, when it fires, which sub-step travellers leave on, and when they return:

```python
class CentroidsClause(MovementClause):
    requirements = (
        AttributeDef("population", int, Shapes.N),
        AttributeDef("centroid", CentroidType, Shapes.N),
        AttributeDef("phi", float, Shapes.Scalar, default_value=40.0),
        AttributeDef("commuter_proportion", float, Shapes.Scalar, default_value=0.1),
    )

    predicate = EveryDay()
    leaves  = TickIndex(step=0)
    returns = TickDelta(step=1, days=0)
```

**The scope** — a `GeoScope` (US states, counties, census tracts) — and **ADRIOs**, which fetch the data each requirement needs from real sources (US Census, ACS, and others), with caching.

**The RUME** (Runnable Modeling Experiment) binds them:

```python
rume = SingleStrataRUME.build(ipm=SIRS(), mm=Centroids(), scope=..., init=..., params=...)
```

A `MultiStrataRUME` composes several IPM/MM/params triples (a "GPM") into one experiment — the mechanism for multi-strain or multi-population models.

## Core abstractions

| Concept | epymorph name | Notes |
|---|---|---|
| Disease process | `CompartmentModel` (IPM) | Compartments + symbolic edges; validated at subclass definition time |
| Transition | `edge(src, dst, rate=expr)` | SymPy expression; also `fork(...)` for branching |
| Data requirement | `AttributeDef(name, type, shape, default_value, comment)` | **Declared, typed, and shaped** — a real interface contract |
| Shape | `Shapes.Scalar`, `Shapes.N`, `Shapes.T`, `Shapes.TxN`, `Shapes.NxN`, … | Time × node × arm shape algebra, checked |
| Movement | `MovementModel` of `MovementClause`s | Each with `requirements`, `predicate`, `leaves`, `returns` |
| Timing | `EveryDay()`, `TickIndex(step=)`, `TickDelta(step=, days=)` | Sub-daily ticks; travellers leave on one and return on another |
| Geography | `GeoScope` — states, counties, tracts | With real census hierarchies |
| Data source | ADRIO | Fetches an attribute from an external source, with caching and an inspection report |
| Stratum | `GPM` (Geo-Population Model) | An (IPM, MM, init, params) bundle |
| Experiment | `RUME` — `SingleStrataRUME` / `MultiStrataRUME` | The runnable whole |
| Fitting | `epymorph.parameter_fitting` | Particle-filter-based |

## The requirement/ADRIO contract

The piece most worth taking. An IPM does not read data; it **declares what it needs**: a name, a type, a shape, an optional default, and a comment. So does every movement clause. The RUME then resolves those requirements against parameters the user supplied or against ADRIOs that fetch them from real sources.

The effects are worth listing:

- A model states its data interface *as data*, so it can be checked before anything runs.
- The same disease model runs over any geography for which the requirements can be satisfied.
- Shape checking (`Shapes.TxN` versus `Shapes.N` versus `Shapes.Scalar`) catches an entire class of broadcasting bug at bind time.
- Data provenance is explicit: an attribute came from a named ADRIO with a named source, or from the user.
- `data_usage.py` can estimate how much data an experiment will fetch before fetching it.

Nothing else in the review makes a model's data dependencies declarative in this way. Most frameworks have the model reach out and get what it needs.

## Time

The timeline is `TimeFrame`, and movement runs on **sub-daily ticks**: a clause declares which tick travellers leave on (`TickIndex(step=0)`) and when they come back (`TickDelta(step=1, days=0)`). This gives commuting — leave in the morning, mix at the destination, return in the evening — without a continuous-time formulation.

Disease rates are per-day floats with declared shapes. There are no dimensional types, but the **shape** system does a related job: `Shapes.TxN` says a parameter varies by time and node, and passing a scalar where `TxN` is required is an error.

## Strengths

- **Disease, movement, and geography are genuinely independent.** Swap any one without touching the others. This is the review's cleanest separation of concerns.
- **Declared, typed, shaped data requirements.** A model's data interface is data, checkable before the run.
- **ADRIOs.** Attribute fetching from real sources (census, ACS) with caching, inspection, and provenance — the geographic-data equivalent of Epydemix's population bundling, but *pull*-based and declarative rather than bundled.
- **Symbolic transitions.** IPM edges are SymPy expressions over declared symbols, so the disease model is inspectable, differentiable, and diagrammable.
- **Validation at class-definition time.** `__init_subclass__` runs `_validate_compartment_model`, so a malformed IPM fails at import, not at run.
- **Sub-daily movement timing** with leave/return semantics — the right model for commuting, and unusual.
- **Multi-strata composition** via `MultiStrataRUME`.
- **A shape algebra** that catches broadcasting errors at bind time.

## Limitations

- **The IPM is Python subclassing.** `edges()` is a method returning a list; the model is a class, not a document. Serialisation of structure is not the design.
- **No dimensional types.** Shapes are checked, units are not.
- **Movement clauses are code.** The `predicate` / `leaves` / `returns` declaration is elegant, but the movement mechanism itself (the dispersal kernel, the row normalisation) is an ordinary Python method.
- **US-centric.** GeoScope and the ADRIO library are built around US Census geography and US data sources. Applying it elsewhere means writing ADRIOs.
- **Compartmental within nodes only.** No individual agents; heterogeneity is by node and by stratum.
- **No intervention vocabulary.** Interventions are time-varying parameters (which the `TxN` shape supports well) or bespoke code.
- **Narrow adoption.** A well-designed framework with a small user base, so its abstractions have not been stress-tested at scale.

## Implications for the lingua franca

1. **Adopt the declared-requirement contract.** A model that says "I need `beta`, a float, shaped time × node" and is bound to data separately is checkable, portable, and provenanced. This is the single most transferable idea in epymorph, and it generalises past geography: every module in a lingua franca model — disease, network, demography, intervention — could declare its data interface the same way. It also solves a problem Starsim's connectors have, since a declared requirement makes cross-module dependencies visible.
2. **Adopt a shape algebra.** `Scalar` / `N` / `T` / `TxN` / `NxN` as checked shapes catches broadcasting bugs that dimensional types alone would not. Dimensional types say *what a number means*; shapes say *how many of them there are and along which axis*. A language spanning agents, patches, strata, and time needs both.
3. **The three-way factoring is the model to follow.** Disease × movement × geography for metapopulations generalises to disease × contact structure × population for the whole design space. Starsim's `Route` abstraction is reaching for the same thing from the agent-based side; epymorph shows what it looks like when the separation is total rather than partial.
4. **Sub-tick movement timing is worth expressing.** Leave on tick 0, return on tick 1 of the next day is a compact way to say "commuting", and it is a genuinely different time structure from either a single `dt` or Starsim's per-module timelines. The language should be able to express within-step scheduling of this kind.
5. **Validate at definition, not at run.** epymorph checks IPM well-formedness when the class is defined. camdl checks at compile. The general principle — the earliest possible check — should be explicit in the design.
6. **Data resolution deserves a vocabulary.** Between epymorph's ADRIOs (pull with provenance), Epydemix's bundled data, and camdl's `read(...)`, there is a clear need for the language to express *where a parameter's value comes from* rather than only its value. That belongs in the design alongside the model structure.
