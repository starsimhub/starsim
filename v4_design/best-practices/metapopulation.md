# Metapopulation

## Recommendation

1. **Space is a stratification dimension like any other**, not a separate kind of model. Movement between patches is optional: space is useful with it (commuting, migration) and without it (wastewater catchments, malaria driven by local environment).
2. **Separate the disease process from the movement process from the geography.** epymorph's three-way factoring is the cleanest separation in the review.
3. **Mobility is specifiable by its generating model**, named and parameterized, not just a matrix. Gravity is built (`ss.gravity()`); radiation, Stouffer, and competing destinations are deferred from v1, and a matrix covers them.
4. **Real movement data is a first-class input**, not a fallback. Where a register exists, use it.
5. **Distinguish commuting from migration.** People who return are not people who move.
6. **Geography resolves from a named source with provenance** (`ss.geo('Kenya', level=1)`, from `starsim_data`), or is given directly (names, populations, and positions or distances).

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

## The specification

Space as a dimension (`sim.stratify(patch=ss.geo(...))`, with or without `mobility=`), `ss.geo`, `ss.gravity`, and `ss.Mobility` (commuting, visiting, migration, a movement register, and LASER's moving force of infection, `kind='force'`) are specified in [06-routes.md](../design/06-routes.md) and [05-dimensions.md](../design/05-dimensions.md). Values by patch are `ss.by(patch=...)` ([04-parameters.md](../design/04-parameters.md)); a quantity that lives in each patch and is not people (wastewater, mosquitoes) is a state that holds a count ([03-states-and-transitions.md](../design/03-states-and-transitions.md#states-that-are-not-people)), split by place and not by properties of people.

## Trade-offs

- **Patch counts multiply compartments.** 47 counties × 16 age groups × 3 states is 2,256 compartments before anything interesting happens. Report the count; note that an agent-based run does not have this problem.
- **A mobility model needs distances, and distances need geography.** So `ss.gravity()` without a geo is an error (`ss.NoGeography`). This is one of the few places where a guess is not available, because a gravity model over unknown distances is nothing.
- **Time-share commuting ignores the time of day.** It cannot say that commuters mix with workplace contacts by day and household contacts by night. A model that needs that uses a sub-daily `dt` and separate daytime and nighttime mixing.
- **Travelers in transit are out of scope for now.** People on the train are in neither their home patch nor their destination, and neither `commute` nor `migrate` represents them. MEmilio has a paper on it. It is too complex for v1; a model that needs it can add a transit patch by hand.
- **Making mobility optional removes a safety check.** A user who meant to couple the patches and forgot `mobility=` gets independent epidemics and no warning. `describe()` lists the mobility (or "none"), which is where to look.
- **Time-share commuting is a choice of form.** epymorph commutes with sub-daily ticks (leave in the morning, return in the evening). Weighting exposure by the share of time spent in each place gives the same coupling without changing the time step, and it is the standard form in commuting metapopulation models, so it is the default.

## Rejected

- **Metapopulation as a separate model class** (MEmilio's `ode_seir_metapop`, EpiModel's separate paths). It is a stratification by place, with optional movement.
- **Requiring movement for a spatial model.** Wastewater and environmentally driven models have patches and no travel.
- **Mobility as a matrix only** (flepiMoP, EpiHiper, epiworld). Accept a matrix; do not require one.
- **Opaque select/shift matrices** (SimInf). The semantics are right and the surface is not.
- **US-only geography** (epymorph). A protocol with sources, not a hard-coded hierarchy.
- **Force of infection by string-concatenated parameter names** (MetaCast's `beta_[high,vaccinated]`). Shapes.
- **A separate "patch" concept distinct from other dimensions.** LASER puts `node_id` in the core for performance reasons that are real and are implementation, not semantics.

## Open questions

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
