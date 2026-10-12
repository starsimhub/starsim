# Metapopulation

## Recommendation

1. **Space is a stratification dimension like any other**, not a separate kind of model. Movement between patches is optional: space is useful with it (commuting, migration) and without it (wastewater catchments, malaria driven by local environment).
2. **Separate the disease process from the movement process from the geography.** epymorph's three-way factoring is the cleanest separation in the review.
3. **Mobility is specifiable by its generating model.** Gravity, radiation, Stouffer, competing-destinations — named, parameterized, not just a matrix.
4. **Real movement data is a first-class input**, not a fallback. Where a register exists, use it.
5. **Distinguish commuting from migration.** People who return are not people who move.
6. **Geography resolves from a named source with provenance.**

## Why

### The three-way factoring

[epymorph](../approaches/epymorph.md) composes an IPM (disease), an MM (movement), and a GEO scope independently: "You can swap the disease without touching the movement model, swap the movement model without touching the disease, and re-run either over a different geographic scope with the same code."

Its review calls this "the cleanest separation of concerns in the review", and notes the generalization: "Disease × movement × geography for metapopulations generalizes to disease × contact structure × population for the whole design space. Starsim's `Route` abstraction is reaching for the same thing from the agent-based side; epymorph shows what it looks like when the separation is total rather than partial."

That is the connection to [population-and-mixing.md](population-and-mixing.md): a metapopulation is not a new paradigm, it is a model stratified by place, with mobility as one more description of who meets whom.

### Mobility models exist and are published

[LASER](../approaches/laser.md) ships `gravity`, `competing_destinations`, `stouffer`, and `radiation` "with distance calculation, row normalization, and parameter sanity checks. This is the best spatial-interaction vocabulary in the review."

Against that: [EpiHiper](../approaches/epihiper.md), [epiworld](../approaches/epiworld.md), and [EoN](../approaches/eon.md) take structure as an input file only. [flepiMoP](../approaches/flepimop.md) reads `geodata` and `mobility` CSVs. The general finding (11 in the [landscape review](../approaches/README.md#cross-cutting-findings)): "A language that only accepts an edge list or a matrix has moved the problem rather than solved it — because the description, not the realization, is what modelers actually have."

### And sometimes the data is the model

[SimInf](../approaches/siminf.md) was built for livestock disease, where the data is a national animal-movement register: "a real animal-movement register — 200,000 dated transfers between holdings — is loaded directly as the events data frame, and the model runs against actual movement data rather than a mobility model. That is a genuinely different way to drive a metapopulation, and it is the one the data often supports."

### Commuting is not migration

[epymorph](../approaches/epymorph.md)'s movement clauses declare `leaves=TickIndex(step=0)` and `returns=TickDelta(step=1, days=0)` — sub-daily ticks, so travelers leave in the morning, mix at the destination, and return in the evening. [MEmilio](../approaches/memilio.md) has a paper on representing people *in transit* between patches, and its review notes: "A metapopulation vocabulary that only has 'in patch i' or 'in patch j' cannot say this." Travelers in transit are out of scope for now; see Trade-offs.

### Space without movement

Not every spatial model is about people moving. Wastewater surveillance needs infections grouped by sewer catchment so that shedding can be summed per treatment plant. Malaria transmission depends on local rainfall, temperature, and mosquito density, which differ between districts whether or not anyone travels. In both, place matters because parameters, environmental quantities, and data differ by place. [SimInf](../approaches/siminf.md)'s per-node local data (`ldata`) and per-node continuous state are there for exactly this, and its `SISe` models couple each node's infections to that node's environment.

## The proposal

### Space is a dimension

```python
kenya = ss.geo('Kenya', level='county')

sir_kenya = sir.stratify(patch=kenya)                                                   # patches, nobody moves
sir_kenya = sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))  # patches coupled by travel
```

That is the whole metapopulation declaration. It uses the [stratification operator](stratification.md) unchanged. `stratify` returns a new model and leaves `sir` as it was, so the spatial and non-spatial versions can be run side by side. `mobility=` is optional, in the same way that an age stratification may or may not come with aging. Patches without mobility are a valid model and produce no warning.

The disease is untouched, which is epymorph's point:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

sir_kenya = sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))
ss.Sim(sir_kenya, networks=ss.contacts('Kenya'), unit='day').run()
```

Disease, movement, geography, and mixing are four independent declarations. Any one can be swapped. Every part of this model has a compartmental form, so `auto` runs it as an ODE; `method='abm'` runs the same model with agents, where the patch is an agent property.

### Mobility models by name

**LASER** ([approaches/laser.md](../approaches/laser.md)):

```python
from laser_core.migration import gravity, row_normalizer
M = row_normalizer(gravity(pops, distances, k=0.5, a=1.0, b=1.0, c=2.0), 0.1)
```

**Lingua franca**: the same models, given by name and parameters instead of as matrices:

```python
ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0)   # distances and populations come from the geo
ss.radiation()                            # parameter-free
ss.stouffer(k=0.5, a=1.0)
ss.competing_destinations(k=0.5, a=1.0, b=1.0, c=2.0, delta=0.1)
ss.Mobility(matrix=M)                     # a matrix you already have, still allowed
ss.Mobility(data='movements.csv')         # SimInf's register
```

Distances and populations are supplied by the geography, so `ss.gravity(...)` does not need them passed in.

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
gravity = ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0)

ss.Mobility(gravity, kind='commute', away=0.4)   # leave, mix, return; 40% of the time is spent away
ss.Mobility(gravity, kind='migrate')             # relocate
```

Under `commute`, a person's home patch does not change. They are exposed, and expose others, at home for part of the time and at the destination for the rest, in proportion to `away`. Under `migrate`, their patch changes: it is an `ss.transfer` on the patch dimension.

The distinction matters epidemiologically: commuting couples patches without mixing their demography, migration does both. epymorph implements commuting with sub-daily ticks (leave in the morning, return in the evening). Weighting exposure by the share of time spent in each place gives the same coupling without changing the time step, and it is the standard form in commuting metapopulation models, so that is the default here.

### Geography with provenance

[epymorph](../approaches/epymorph.md)'s ADRIOs fetch attributes from the US Census and ACS with caching, an inspection report, and a `data_usage` estimate before fetching. [Epydemix](../approaches/epydemix.md) bundles 400+ locations of population and contact matrices.

```python
ss.geo('Kenya', level='county')            # names, populations, centroids, adjacency
ss.geo('USA', level='county', year=2020)
ss.geo(shapefile='my_districts.shp', pop='pop.csv')
```

with the source recorded in `sim.describe()` and in the [run hash](stochasticity-and-reproducibility.md). epymorph's US-centricity is a limitation to avoid, not to copy: the geography interface should be a protocol with pluggable sources, and the bundled set should include somewhere that is not the United States on day one.

### Per-patch parameters

[SimInf](../approaches/siminf.md)'s `gdata`/`ldata` and [MetaCast](../approaches/metacast.md)'s `universal_params`/`subpop_params` are the same distinction arrived at twice; [epymorph](../approaches/epymorph.md)'s shapes are the general form. This is already covered by [parameters-and-distributions.md](parameters-and-distributions.md) (`beta=ss.by(patch=[...])`), and it is worth noting here only because the metapopulation case is where it comes up most. It is also most of what a spatial model without movement needs: the same disease in every patch, with parameters that differ by patch.

### Per-patch environmental quantities

The other thing a spatial model without movement needs is a quantity that lives in each patch and is not a person: virus in a catchment's wastewater, mosquitoes in a district. **SimInf** holds these in its continuous per-node state `v`, updated by a C function after each step. **Lingua franca**: an `ss.Compartments` module, stratified by the same geography. It holds counts per patch, uses the same transition syntax as a disease, and its counts can appear in disease rates and in results.

```python
S, I, R = ss.states('S I R')
W       = ss.states('W')                  # viral copies in wastewater, per catchment

infection = S >> I @ ss.perday(0.3) * I
recovery  = I >> R @ ss.days(10)
shedding  = None >> W @ ss.perday(1e9) * I.n   # copies shed per infectious person per day
decay     = W >> None @ ss.perday(0.5)

catchments = ss.geo(shapefile='catchments.shp', pop='catchment_pop.csv')
covid      = ss.Disease(infection, recovery).stratify(patch=catchments)
wastewater = ss.Compartments(shedding, decay).stratify(patch=catchments)

sim = ss.Sim(modules=[covid, wastewater], unit='day', data='wastewater.csv')
```

Nobody moves in this model. `I.n` in the shedding rate is the count of infectious people in the same catchment, because both modules are stratified by `patch`. `n_W` is produced per catchment and can be compared with measured concentrations by name ([results-and-observation.md](results-and-observation.md)). The same pattern, with mosquito states in place of `W` and rainfall as a per-patch time series, is environmentally driven malaria.

## Trade-offs

- **Patch counts multiply compartments.** 47 counties × 16 age groups × 3 states is 2,256 compartments before anything interesting happens. Report the count; note that an agent-based run does not have this problem.
- **A mobility model needs distances, and distances need geography.** So `ss.gravity()` without a geo is an error. This is one of the few places where a guess is not available, because a gravity model over unknown distances is nothing.
- **Time-share commuting ignores the time of day.** It cannot say that commuters mix with workplace contacts by day and household contacts by night. A model that needs that uses a sub-daily `dt` and separate daytime and nighttime mixing.
- **Travelers in transit are out of scope for now.** People on the train are in neither their home patch nor their destination, and neither `commute` nor `migrate` represents them. MEmilio has a paper on it. It is too complex for v1; a model that needs it can add a transit patch by hand.
- **Making mobility optional removes a safety check.** A user who meant to couple the patches and forgot `mobility=` gets independent epidemics and no warning. `describe()` lists the mobility (or "none"), which is where to look.

## Rejected

- **Metapopulation as a separate model class** (MEmilio's `ode_seir_metapop`, EpiModel's separate paths). It is a stratification by place, with optional movement.
- **Requiring movement for a spatial model.** Wastewater and environmentally driven models have patches and no travel.
- **Mobility as a matrix only** (flepiMoP, EpiHiper, epiworld). Accept a matrix; do not require one.
- **Opaque select/shift matrices** (SimInf). The semantics are right and the surface is not.
- **US-only geography** (epymorph). A protocol with sources, not a hard-coded hierarchy.
- **Force of infection by string-concatenated parameter names** (MetaCast's `beta_[high,vaccinated]`). Shapes.
- **A separate "patch" concept distinct from other dimensions.** LASER puts `node_id` in the core for performance reasons that are real and are implementation, not semantics.

## Open questions

- **Does space need conveniences that other dimensions do not?** Space is a dimension, with or without movement; that much is settled. But LASER's review argues from an eradication-scale perspective that a subnational planning audience asks spatial questions all the time, and its `node_id`-in-the-core choice is not obviously wrong. Maps as a default plot and per-patch data loading are the likely candidates.
- **What does a patch mean in an agent-based run?** An agent property, per [stratification.md](stratification.md). But LASER's node-level force of infection with a migration matrix is a *third* thing: agents in patches with aggregate coupling. That is probably `ss.MixingPool` grouped by patch, and it should be confirmed on a LASER example.
- **How are counts matched between modules with different stratifications?** In the wastewater example both modules are stratified by patch, so `I.n` means "in this catchment". If the disease is also stratified by age and the wastewater is not, the natural rule is to match on shared dimensions and sum over the rest. The same question is raised in [stochastic-compartmental.md](stochastic-compartmental.md).
- **Adaptive or behavior-driven mobility** (people stop traveling during an epidemic) is an `ss.modify` on the mobility ([composition-and-effects.md](composition-and-effects.md)), which the mechanism supports but nobody in the review does declaratively.
