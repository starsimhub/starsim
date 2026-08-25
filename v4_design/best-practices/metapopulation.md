# Metapopulation

## Recommendation

1. **Space is a dimension that carries movement structure** — not a separate kind of model.
2. **Separate the disease process from the movement process from the geography.** epymorph's three-way factoring is the cleanest separation in the review.
3. **Mobility is specifiable by its generating model.** Gravity, radiation, Stouffer, competing-destinations — named, parameterized, not just a matrix.
4. **Real movement data is a first-class input**, not a fallback. Where a register exists, use it.
5. **Distinguish commuting from migration.** People who return are not people who move.
6. **Geography resolves from a named source with provenance.**

## Why

### The three-way factoring

[epymorph](../approaches/epymorph.md) composes an IPM (disease), an MM (movement), and a GEO scope independently: "You can swap the disease without touching the movement model, swap the movement model without touching the disease, and re-run either over a different geographic scope with the same code."

Its review calls this "the cleanest separation of concerns in the review", and notes the generalization: "Disease × movement × geography for metapopulations generalises to disease × contact structure × population for the whole design space. Starsim's `Route` abstraction is reaching for the same thing from the agent-based side; epymorph shows what it looks like when the separation is total rather than partial."

That is the connection to [population-and-mixing.md](population-and-mixing.md): a metapopulation is not a new paradigm, it is a route with a spatial dimension.

### Mobility models exist and are published

[LASER](../approaches/laser.md) ships `gravity`, `competing_destinations`, `stouffer`, and `radiation` "with distance calculation, row normalisation, and parameter sanity checks. This is the best spatial-interaction vocabulary in the review."

Against that: [EpiHiper](../approaches/epihiper.md), [epiworld](../approaches/epiworld.md), and [EoN](../approaches/eon.md) take structure as an input file only. [flepiMoP](../approaches/flepimop.md) reads `geodata` and `mobility` CSVs. The general finding (11 in the [landscape review](../approaches/README.md#cross-cutting-findings)): "A language that only accepts an edge list or a matrix has moved the problem rather than solved it — because the description, not the realisation, is what modellers actually have."

### And sometimes the data is the model

[SimInf](../approaches/siminf.md) was built for livestock disease, where the data is a national animal-movement register: "a real animal-movement register — 200,000 dated transfers between holdings — is loaded directly as the events data frame, and the model runs against actual movement data rather than a mobility model. That is a genuinely different way to drive a metapopulation, and it is the one the data often supports."

### Commuting is not migration

[epymorph](../approaches/epymorph.md)'s movement clauses declare `leaves=TickIndex(step=0)` and `returns=TickDelta(step=1, days=0)` — sub-daily ticks, so travellers leave in the morning, mix at the destination, and return in the evening. [MEmilio](../approaches/memilio.md) has a paper on representing people *in transit* between patches, and its review notes: "A metapopulation vocabulary that only has 'in patch i' or 'in patch j' cannot say this."

## The proposal

### Space is a dimension

```python
sir.stratify(patch=ss.geo('Kenya', level='county'),
             mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))
```

That is the whole metapopulation declaration. It uses the [stratification operator](stratification.md) unchanged; the only thing space adds is that it carries movement structure, exactly as `age` carries ageing flows.

The disease is untouched, which is epymorph's point made structural:

```python
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05),
                 recovery=ss.progression('I -> R', dur=ss.days(10)))

sir.stratify(patch=ss.geo('Kenya', level='county'), mobility=ss.gravity(...))
ss.Sim(sir, mixing=ss.contacts('Kenya')).run(method='ode')
```

Disease, movement, geography, and mixing are four independent declarations. Any one can be swapped.

### Mobility models by name

**LASER** ([approaches/laser.md](../approaches/laser.md)):

```python
from laser_core.migration import gravity, row_normalizer
M = row_normalizer(gravity(pops, distances, k=0.5, a=1.0, b=1.0, c=2.0), 0.1)
```

**Lingua franca** — the same models, as route descriptions rather than matrices:

```python
ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0)   # distances and populations come from the geo
ss.radiation()                            # parameter-free
ss.stouffer(k=..., a=...)
ss.competing_destinations(...)
ss.Mobility(matrix=M)                     # the realization, still allowed
ss.Mobility(data='movements.csv')         # SimInf's register
```

Distances and populations are supplied by the geography, so `ss.gravity(...)` does not need them passed in — which is the requirement contract from [parameters-and-distributions.md](parameters-and-distributions.md) doing its job.

### Movement data as events

**SimInf** encodes movement as an events data frame plus a select matrix `E` and a shift matrix `N`. Its review is blunt about the surface: "The `E` and `N` matrices are opaque… it is the least legible part of an otherwise clear design, and it is the part encoding the most important semantics."

**MetaCast**'s transfer dictionaries are the readable version of the same semantics. **Lingua franca** — a movement register is a table of `ss.transfer` events:

```python
ss.Mobility(data='animal_movements.csv')
# columns: date, from, to, n, [states]
```

`states` defaults to all disease states, sampled proportionally — which is what SimInf's select matrix encodes, made a default instead of a matrix. Naming states explicitly is for the cases where a movement is selective (only healthy animals are sold).

### Commuting versus migration

Two different things, and the distinction is real:

```python
ss.Mobility(ss.gravity(...), kind='commute', away=ss.hours(9))  # leave, mix, return
ss.Mobility(ss.gravity(...), kind='migrate')                    # relocate
```

Under `commute`, an individual's residence patch does not change; they contribute to the destination's force of infection for the away period and return. Under `migrate`, their patch coordinate changes — it is an `ss.transfer` on the patch dimension.

The distinction matters epidemiologically: commuting couples patches without mixing their demography, migration does both. epymorph's sub-tick leave/return semantics is the mechanism for the first, and [time-and-units.md](time-and-units.md) covers the sub-step scheduling it needs.

### Geography with provenance

[epymorph](../approaches/epymorph.md)'s ADRIOs fetch attributes from the US Census and ACS with caching, an inspection report, and a `data_usage` estimate before fetching. [Epydemix](../approaches/epydemix.md) bundles 400+ locations of population and contact matrices.

```python
ss.geo('Kenya', level='county')            # names, populations, centroids, adjacency
ss.geo('USA', level='county', year=2020)
ss.geo(shapefile='my_districts.shp', pop='pop.csv')
```

with the source recorded in `sim.summary()` and in the [run hash](stochasticity-and-reproducibility.md). epymorph's US-centricity is a limitation to avoid, not to copy: the geography interface should be a protocol with pluggable sources, and the bundled set should include somewhere that is not the United States on day one.

### Per-patch parameters

[SimInf](../approaches/siminf.md)'s `gdata`/`ldata` and [MetaCast](../approaches/metacast.md)'s `universal_params`/`subpop_params` are the same distinction arrived at twice; [epymorph](../approaches/epymorph.md)'s shapes are the general form. This is already covered — `beta=ss.by(patch=[...])` — by [parameters-and-distributions.md](parameters-and-distributions.md), and it is worth noting here only because the metapopulation case is where it comes up most.

## Trade-offs

- **Patch counts multiply compartments.** 47 counties × 16 age groups × 3 states is 2,256 compartments before anything interesting happens. Report the count; note that an agent-based backend does not pay it.
- **A mobility model needs distances, and distances need geography.** So `ss.gravity()` without a geo is an error — one of the few places where a guess is not available, because a gravity model over unknown distances is nothing.
- **Commuting requires sub-step time structure**, which is genuine additional machinery in the execution loop. Justified by how common commuting metapopulations are.
- **Travellers in transit are a third location** that neither `commute` nor `migrate` represents. MEmilio has a paper on it. We should decide deliberately whether we can say it, and the honest v1 answer is probably no, stated in the capability list.

## Rejected

- **Metapopulation as a separate model class** (MEmilio's `ode_seir_metapop`, EpiModel's separate paths). It is a dimension with movement.
- **Mobility as a matrix only** (flepiMoP, EpiHiper, epiworld). Accept a matrix; do not require one.
- **Opaque select/shift matrices** (SimInf). The semantics are right and the surface is not.
- **US-only geography** (epymorph). A protocol with sources, not a hard-coded hierarchy.
- **Force of infection by string-concatenated parameter names** (MetaCast's `beta_[high,vaccinated]`). Shapes.
- **A separate "patch" concept distinct from other dimensions.** LASER puts `node_id` in the core for performance reasons that are real and are implementation, not semantics.

## Open questions

- **Should space get special treatment after all?** LASER's review argues for it from an eradication-scale perspective, and its `node_id`-in-the-core choice is not obviously wrong. The position taken here — space is a dimension carrying movement — may be too uniform for a subnational planning audience whose every question is spatial.
- **What does a patch mean in an agent-based backend?** An agent property, per the [stratification](stratification.md) unification. But LASER's node-level force of infection with a migration matrix is a *third* thing: agents in patches with aggregate coupling. That is probably `ss.MixingPool` over a patch dimension, and it should be confirmed rather than assumed.
- **Adaptive or behavior-driven mobility** — people stop travelling during an epidemic — is an [effect](composition-and-effects.md) on the mobility matrix, which the mechanism supports but nobody in the review does declaratively.
