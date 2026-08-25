# Routes: population, mixing, and the force of infection

How contact happens. The route is the only thing that varies between mass-action, contact-matrix, and network transmission; the disease declaration does not change.

Justified by [best-practices/population-and-mixing.md](../best-practices/population-and-mixing.md) and [best-practices/metapopulation.md](../best-practices/metapopulation.md).

## The factoring

> **`beta` is the probability, or hazard, of transmission per contact between an infectious and a susceptible individual. It is a property of the pathogen and the pair. The route supplies who contacts whom, and how often.**

This is the decision the rest of the document follows from. In most compartmental frameworks β is a composite — transmission probability × contact rate, fused into one number — and that fusion is exactly what makes mass-action and edge-based transmission look like different mathematics. Unfused, they are the same declaration with a different route.

The cost, stated plainly: **β and the contact rate are not separately identifiable from incidence data.** Only their product is. The language MUST therefore allow the product to be the fitted quantity ([13-inference.md](13-inference.md)). This is not a new problem; it is the existing problem, made visible.

## The force of infection

The force of infection is computed by the framework, always. No model text contains `/N`.

Let

- `i`, `j` index **mixing groups**: the combinations of dimension coordinates the route resolves over (a single group, if the model is unstratified);
- `X[s,j]` be the number in state `s` in group `j`, and `N[j] = Σ_s X[s,j]`;
- `T` be a transmission with per-contact rate `β`, arrow `from(T) -> to(T)`, and infecting states `inf(T)` — the states named by `source=`;
- `c[i,j]` be the contact rate: expected contacts per unit time that one individual in group `i` has with individuals in group `j`.

Note that `from(T)` and `inf(T)` are different sets and are spelled differently in the declaration: in `ss.transmission('S -> E', beta=..., source='I')`, `from(T) = {S}`, `to(T) = {E}`, and `inf(T) = {I}`.

Then the infectious pressure exerted by group `j` on transmission `T` is

```
P[j] = Σ_{s ∈ inf(T)}  rel_trans[s] · X[s,j]
```

and the hazard of infection for a susceptible individual of state `s ∈ from(T)` in group `i` is

```
λ[s,i] = rel_sus[s] · Σ_j  c[i,j] · β · P[j] / D[j]
```

where the denominator `D[j]` is the route's **dependence convention**:

| `dependence=` | `D[j]` | Means |
|---|---|---|
| `'frequency'` (default) | `N[j]` | Contact rate is fixed; the *fraction* of contacts that are infectious matters |
| `'density'` | `N_ref` | Contact rate scales with density; `N_ref` is a declared reference size, defaulting to the initial total population |
| `'total'` | `Σ_j N[j]` | Contacts drawn from the whole population regardless of group |

The flow through `T` out of state `s` in group `i` over a step is then `X[s,i]` thinned by `1 − exp(−λ[s,i]·dt)`, split across `to(T)` by the branch probabilities if the arrow branches.

Frequency dependence is the default because it is right for directly transmitted human infections, which is what most users are modeling. The convention in force MUST be printed in `summary()`, because a model that silently chose the other is wrong in a way that only shows up when population size changes — and then shows up as a factor, not as an error.

Defining density dependence against an explicit `N_ref` rather than by omitting the denominator makes the two conventions coincide at `N = N_ref` and makes the scaling assumption visible. A density-dependent route with no `N_ref` and a changing population emits `W401 UnanchoredDensity`.

### Per-agent and edge-based form

In an agent-based backend with an edge-based route, the same quantities apply per edge rather than per group. For an edge `e = (u, v)` between infectious `u` and susceptible `v`, carrying `acts[e]` acts per unit time:

```
hazard[e] = β · acts[e] · rel_trans[u] · rel_sus[v]
p[e]      = 1 − exp(−hazard[e] · dt)
p[v]      = 1 − Π_e (1 − p[e])         over all edges incident on v
```

`rel_trans[u]` is the product of `u`'s state-level `rel_trans` and any per-agent modifiers ([07-effects.md](07-effects.md)); likewise `rel_sus[v]`. The product form for `p[v]` is exact under independence of edges within a step and MUST be used rather than summing hazards, which double-counts at high degree.

### Environmental

For a transmission whose `source` is an `ss.Continuous` quantity `W` rather than a state, the group form becomes

```
λ[i] = rel_sus · β_env · W[i] / (W[i] + K)
```

with `K` the half-saturation constant, declared on the transmission as `k=`. This is the standard dose-response form for reservoir transmission, and it replaces the contact term because there is no contact to be *per*. A transmission with an environmental source MUST NOT be given a route, and `beta` in that slot is `β_env` — a different quantity, with different units, and `summary()` labels it as such.

This is the most likely place the per-contact factoring needs extending; see [19-open-questions.md](19-open-questions.md).

## The route classes

```
ss.Route                        base: supplies c[i,j] or an edge set
 ├── ss.Homogeneous(n_contacts=, dependence=, n_ref=)
 ├── ss.MixingPool(matrix=, dependence=, groups=)
 ├── ss.RandomNet(n_contacts=, dur=)
 ├── ss.SexualNet(mean_degree=, dur=, acts=, concurrency=, ...)
 ├── ss.Network(edges=)
 └── ss.Mobility(model=, kind=, ...)          # a route over a spatial dimension
```

| Route | `c[i,j]` | ODE | CTMC | ABM |
|---|---|---|---|---|
| `ss.Homogeneous(n_contacts=c)` | `c` for all pairs | exact | exact | exact |
| `ss.MixingPool(matrix=M)` | `M[i,j]` | exact | exact | exact |
| `ss.RandomNet(n_contacts=c)` | realized edges, redrawn each step | mean-field, exact as `n → ∞` | same | exact |
| `ss.SexualNet(...)` | persisting partnerships × acts | **refused** | **refused** | exact |
| `ss.Network(edges=E)` | a fixed edge set | refused unless `closure=` given | refused | exact |

`ss.Homogeneous` and `ss.MixingPool` have exact compartmental forms, which is what makes the compartmental limit exact rather than an approximation reached by simulating homogeneous agents. This is the largest single change from Starsim's current behavior.

`ss.SexualNet` has no compartmental limit, and this is a fact about concurrent partnerships rather than a limitation of the implementation. The compiler MUST refuse rather than approximate ([12-backends.md](12-backends.md)).

### The claim, stated so it can be tested

```python
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05),
                 recovery=ss.progression('I -> R', dur=ss.days(10)))

ss.Sim(sir, mixing=ss.Homogeneous(n_contacts=10)).run(method='ode')
ss.Sim(sir, mixing=ss.contacts('Kenya')).run(method='ode')
ss.Sim(sir, mixing=ss.RandomNet(n_contacts=10), n_agents=100_000).run()
ss.Sim(sir, mixing=ss.SexualNet()).run()
```

The disease object is byte-identical across all four. Runs 1 and 3 MUST agree in distribution as `n → ∞`; [18-conformance.md](18-conformance.md) states the tolerance and the test.

## Specifying structure by its generating model

A route is *described*, not supplied. "A network where mean degree is 0.7 and partnerships last 50 days" is what modelers have; an edge list is not.

```python
ss.RandomNet(n_contacts=ss.poisson(10))
ss.SexualNet(mean_degree=0.7, dur=ss.days(50), acts=ss.freqperweek(2), concurrency=0.22)
ss.contacts('Kenya', source='prem_2021', layer='home')
ss.Network(edges=my_edgelist)              # the realization: still accepted, never required
```

An edge list MUST be accepted and MUST NOT be required. A route given only an edge list reports reduced capability, because a realization cannot be resampled, rescaled to a different population, or diagnosed.

### Partnership and act are distinct

```python
ss.SexualNet(
    mean_degree = 0.7,                # how many partners
    dur         = ss.days(50),        # how long partnerships last
    acts        = ss.freqperweek(2),  # how often, within a partnership
)
```

Three numbers with three different data sources. Collapsing them into a single "contact rate" is lossy for anything sexually transmitted, and the separation is what makes per-act transmission probability meaningful.

### Diagnostics

```python
net.check()      # simulate the route alone; do the target statistics recover?
net.check(plot=True)
```

Network diagnostics are treated as mandatory in the one framework that does them properly, and the discipline is worth importing: a route fitted to target statistics that does not reproduce them is a silent modeling error. `check()` MUST report, for each declared target statistic, the target, the realized mean, and the interval across replicates.

## Layers

A list of routes is a layered population, and each layer is separately named and separately targetable.

```python
home      = ss.contacts('Kenya', layer='home')
school    = ss.contacts('Kenya', layer='school')
work      = ss.contacts('Kenya', layer='work')
community = ss.contacts('Kenya', layer='community')

ss.Sim(sir, mixing=[home, school, work, community])
```

A layer is bound to a name so that it can be targeted later; `sim.mixing.school` reaches the same object for a route that was declared inline.

The force of infection sums over layers:

```
λ[i] = Σ_layers Σ_j c_layer[i,j] · β_layer · P[j] / D[j]
```

`β` may be declared per layer (`ss.transmission(..., beta=ss.by(layer=...))`) when transmissibility differs by setting; otherwise it is shared and only the contact rates differ. Layers are what school-closure interventions act on:

```python
ss.Effect(ss.date('2020-03-15') <= ss.time <= ss.date('2020-06-01'),
          school.contacts, multiply=0.2)
```

A transmission MAY restrict itself to particular layers with `route=`; by default a transmission uses every route in the sim, which is the behavior that makes adding a layer a one-line change.

## Mobility

Space is a dimension ([05-dimensions.md](05-dimensions.md)); mobility is the route that connects its levels.

```python
sir.stratify(patch=ss.geo('Kenya', level='county'),
             mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))
```

Movement models are named and parameterized, not supplied as matrices:

```python
ss.gravity(k=, a=, b=, c=)        # distances and populations come from the geo
ss.radiation()                     # parameter-free
ss.stouffer(k=, a=)
ss.competing_destinations(k=, a=, b=, c=, delta=)
ss.Mobility(matrix=M)              # the realization, still accepted
ss.Mobility(data='movements.csv')  # a movement register
```

`ss.gravity()` without a geography raises `E402 NoGeography`: a gravity model over unknown distances is nothing, and this is one of the few places where no guess is available.

### Movement registers

Where a real movement register exists — livestock transfers, flight data, mobile-phone-derived flows — it is the model, not a fallback:

```python
ss.Mobility(data='animal_movements.csv')
# columns: date, from, to, n, [states]
```

`states` defaults to all disease states, sampled in proportion to their occupancy at the origin, which is the common case and is what opaque select matrices encode elsewhere. Naming states explicitly covers selective movement — only healthy animals are sold.

### Commuting versus migration

```python
ss.Mobility(ss.gravity(...), kind='commute', at=ss.hours(8), returns=ss.hours(18))
ss.Mobility(ss.gravity(...), kind='migrate')
```

- Under `commute`, the individual's patch coordinate does not change. They contribute to and are exposed to the destination's force of infection for the away period, then return. This requires sub-step scheduling ([10-execution.md](10-execution.md) §Sub-steps).
- Under `migrate`, the patch coordinate changes. It is an `ss.transfer` on the spatial dimension and needs no special machinery.

The distinction is epidemiological, not technical: commuting couples patches without mixing their demography; migration does both. A `kind=` that is not declared defaults to `commute` for a gravity or radiation model and to `migrate` for a movement register, and the choice is printed.

Individuals *in transit* — neither at origin nor at destination — are not representable in v1 and are listed as a refusal in `capabilities()`.

## Geography

```python
ss.geo('Kenya', level='county')
ss.geo('USA', level='county', year=2020)
ss.geo(shapefile='my_districts.shp', pop='pop.csv')
```

A geography supplies names, populations, centroids, distances, and adjacency. It is a pluggable provider protocol, not a hard-coded hierarchy, and the source is recorded in `summary()` and in the parameter hash. US-centricity is a limitation to avoid rather than to copy: the bundled providers MUST include non-US geographies from the first release.

## Route capability reporting

Every route MUST report, for each backend method, whether it is exact, approximate with a named closure, or refused. This is the row structure of `capabilities()` ([12-backends.md](12-backends.md)), and it is what makes the central claim falsifiable rather than rhetorical.

## Rejected

- **Making the modeler write the force of infection.** It is the single most commonly mis-written line in epidemiological modeling.
- **Frequency versus density as different transition kinds.** It is a route property; putting it on the transition means every transition repeats it.
- **Networks as inputs only.** Accept an edge list; never require one.
- **A separate transmission vocabulary for networks.** This is precisely the seam the route abstraction removes.
- **Node-level force of infection as the only spatial mechanism.** Correct for eradication-scale spatial models; not general.
- **Opaque select and shift matrices** for movement. The semantics are right; the surface is not.
