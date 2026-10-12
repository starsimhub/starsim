# Routes: mixing, networks, and the force of infection

How infection gets from one person to another. The disease says who infects whom (`S >> I @ beta * I`); the routes say who meets whom and how often. Changing from homogeneous mixing to a contact matrix to a network does not change the disease.

Justified by [best-practices/population-and-mixing.md](../best-practices/population-and-mixing.md) and [best-practices/metapopulation.md](../best-practices/metapopulation.md).

## What a state in a rate means

```python
S, I, R   = ss.states('S I R')
infection = S >> I @ beta * I
```

**A bare state inside a rate means exposure to that state through the model's routes.** The user does not write `/N`; the library works out the exposure from whatever mixing the sim has.

- **No network given.** Mixing is homogeneous with one contact per unit time. `beta * I` is then the textbook `beta * I / N`, and `beta` is the classical transmission rate (per unit time, in the sim's unit).
- **Networks or mixing pools given** (`ss.Sim(sir, networks=...)`). They supply who meets whom and how often, and `beta` is the transmission probability per contact.

The two readings are the same rule: the hazard is `beta` times the contact rate times the fraction of contacts that are infectious, and the default contact rate is 1. Which reading applies is printed by `describe()`, with the resulting units of `beta`.

In the string form the source is a keyword: `ss.transmission('S -> E', beta=0.05, source='I')` ([03-states-and-transitions.md](03-states-and-transitions.md)).

## The force of infection

Let

- `i`, `j` index **mixing groups**: the combinations of dimension levels the routes distinguish (a single group if the model is unstratified);
- `X[s,j]` be the number of people in state `s` in group `j`, and `N[j]` the number of people in group `j`;
- `T` be a transition `from(T) >> to(T)` whose rate is `beta * (source states)`;
- `c[i,j]` be the contact rate supplied by a route: contacts per unit time that one person in group `i` has with people in group `j`.

The infectious pressure from group `j` is

```
P[j] = Σ_{s ∈ source(T)}  rel_trans[s] · X[s,j]
```

and the hazard of infection for a person in state `s ∈ from(T)` in group `i` is

```
λ[s,i] = rel_sus[s] · Σ_routes Σ_j  c[i,j] · β · P[j] / D[j]
```

where `D[j]` depends on the route's `dependence` option ([below](#frequency-and-density-dependence)) and is `N[j]` by default. With no route given there is one term with `c = 1`, so `λ = β · I / N`. The probability of infection over a step of length `dt` is `1 − exp(−λ · dt)`; if the transition branches, the infected are split across the destinations by the branch probabilities.

`rel_sus` and `rel_trans` are the state-level values declared on `ss.State` multiplied by any per-agent or per-group modifiers ([07-effects.md](07-effects.md)).

### Per-edge form

On a network that keeps individual contacts, the same quantities apply per edge. For an edge between infectious `u` and susceptible `v` carrying `acts` acts per unit time (1 if the network does not define acts):

```
hazard[e] = β · acts[e] · rel_trans[u] · rel_sus[v]
p[e]      = 1 − exp(−hazard[e] · dt)
p[v]      = 1 − Π_e (1 − p[e])         over all edges into v
```

The product form for `p[v]` MUST be used; adding the per-edge probabilities overcounts for people with many contacts.

### Units

`beta` times the contact rate is a hazard. Bare numbers always work: a bare `beta` with the default mixing is a rate in the sim's unit; with a route it is a probability per contact, and a bare `n_contacts` is contacts per unit time. One of the two may carry a time unit (`beta=ss.perday(0.03)` with `n_contacts=10`, or `beta=0.03` with `n_contacts=ss.freqperday(10)`). If both carry one, the product is not a hazard and `E102 WrongDimension` is raised.

## Counts used literally

`I.n` is the plain number of people in a state, and `I.prev` is its fraction of the population. A rate may be any arithmetic expression of counts and parameters, and it is then used exactly as written, without going through the routes:

```python
S >> I @ beta * I                      # exposure to I through the routes (the usual case)
S >> I @ beta * I.n / N.n              # the same thing written out for homogeneous mixing; N = S + I + R
S >> I @ beta_w * W.n / (W.n + K)      # environmental: a saturating dose-response on a reservoir W
Sh >> Eh @ a * b * Iv.n / Nh.n         # vector-borne: bites per person from infectious mosquitoes
```

This is how environmental and vector-borne transmission are written. There is no separate environmental route and no special transmission kind: the reservoir or the vector population is an **`ss.Compartments`** module, which holds counts and uses the same transition syntax as a disease, and its counts appear in the disease's rates.

Rules:

- Each term of a rate is either an exposure (it contains one bare state) or literal (it contains only `.n`, `.prev`, parameters, and numbers). A rate may add the two kinds: `S >> I @ beta * I + beta_w * W.n / (W.n + K)` is direct plus environmental transmission. A term containing two bare states has no defined meaning and is an error.
- A sum of states is a group: with `N = S + I + R`, `N.n` is the total.
- In a stratified model, a count is aligned with the person at risk on every dimension the two share and summed over the rest. With cholera and a water reservoir both stratified by `patch`, `W.n` is the reservoir in the susceptible person's own patch; mosquitoes that are not stratified by age see `Ih.n` summed over human age groups. `I.n.sum('age')` and `I.n.sel(age=...)` override the alignment.
- A transition whose rate mentions a state is still an infection for the purposes of results and the transmission log; the mentioned state is its source.

### Environmental transmission: cholera

```python
S, I, R = ss.states('S I R')
W       = ss.states('W')                         # vibrios in the water

cholera = ss.Disease(
    infection = S >> I @ ss.perday(0.2) * W.n / (W.n + 1e6),
    recovery  = I >> R @ ss.days(5),
)
water = ss.Compartments(
    shedding = None >> W @ ss.perday(10) * I.n,  # shed per infectious person per day
    decay    = W >> None @ ss.days(30),
)

sim = ss.Sim(modules=[cholera, water], n_agents=10_000, unit='day')
```

Because `n_agents` is given, people are agents and the water is a count updated alongside them in the same sim. Without it, every piece has a compartmental form and `method='auto'` runs the pair as an ODE. Stratifying both modules by the same geography gives one reservoir per patch ([05-dimensions.md](05-dimensions.md#space)).

### Vector-borne transmission

```python
Sh, Eh, Ih, Rh = ss.states('Sh Eh Ih Rh')
Sv, Ev, Iv     = ss.states('Sv Ev Iv')
Nh = Sh + Eh + Ih + Rh

a = ss.perday(0.3)     # bites per mosquito per day
b, c = 0.5, 0.5        # transmission probability per bite: mosquito to human, human to mosquito

malaria = ss.Disease(
    infection  = Sh >> Eh @ a * b * Iv.n / Nh.n,
    incubation = Eh >> Ih @ ss.days(12),
    recovery   = Ih >> Rh @ ss.days(60),
)
mosquitoes = ss.Compartments(
    emergence  = None >> Sv @ ss.perday(5000),
    infection  = Sv >> Ev @ a * c * Ih.n / Nh.n,
    incubation = Ev >> Iv @ ss.days(10),
    death      = (Sv | Ev | Iv) >> None @ ss.days(14),
)

sim = ss.Sim(modules=[malaria, mosquitoes], unit='day')
```

The humans can be agents (`n_agents=`) while the mosquitoes stay counts.

## Routes

Routes are passed with Starsim's keyword `networks=`, which accepts networks and mixing pools alike, singly or as a list.

| Route | What it supplies | Agent-based run | Compartmental run |
|---|---|---|---|
| none given | homogeneous mixing, 1 contact per unit time | exact | exact |
| `ss.MixingPool(n_contacts=c)` | homogeneous mixing, `c` contacts per unit time | exact | exact |
| `ss.MixingPools(...)`, `ss.contacts('Kenya')` | a contact matrix between groups | exact | exact |
| `ss.RandomNet(n_contacts=c)` | individual contacts, redrawn or persisting for `dur` | exact | approximated by homogeneous mixing with `c` contacts; exact as the population grows when contacts are redrawn every step |
| `ss.MFNet(...)` and other sexual networks | partnerships with durations, acts, and concurrency | exact | approximated by partner-change rates between sex/risk/age groups |
| `ss.StaticNet(edges=...)` | a fixed set of contacts | exact | approximated by group-to-group contact rates computed from the edges (homogeneous mixing at the mean degree if the nodes have no groups) |
| counts used literally (`W.n`, `Iv.n`) | nothing; the rate is the hazard | exact | exact |

There is no `ss.Homogeneous` class. Homogeneous mixing is what happens when no network is given; to set a contact rate use `ss.MixingPool(n_contacts=10)`, which is Starsim's class and is homogeneous when given no groups.

How the columns are used follows the approximation policy in [12-backends.md](12-backends.md):

- Under `method='auto'` nothing is approximated. A model with only "exact" compartmental entries runs as an ODE; a model containing an individual-level network runs as agents.
- When the user asks for a compartmental method on a model with a network, the library uses the approximation in the table and says so in one line (for example `sexual network 'mf' approximated by partner-change rates between sex and risk groups`). It does not refuse. A network model run as an ODE is a coarser model, and there are many valid ODE models of sexually transmitted infections.
- `approx=False` turns any approximation into an error. `sir.methods()` shows this table for the model at hand.

**The sexual-network approximation.** The model is stratified by the attributes the network forms partnerships on (sex, plus risk group and age band where the network uses them), if it is not already. The contact rate `c[i,j]` is the rate at which a person in group `i` starts partnerships with people in group `j`, taken from the network's own parameters, and the per-contact probability becomes the probability of transmission over a whole partnership, `1 − (1 − β)^(acts × dur)`. What is lost is concurrency, repeated exposure to the same partner, and variation in partner numbers within a group. The note printed at build time lists the groups and the rates used.

Things that need to know who a person's contacts are (contact tracing, partner notification, the transmission tree) only work on routes that keep contacts, and so only in an agent-based run.

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

ss.Sim(sir, unit='day').run()                                                   # ODE; beta is a rate per day
ss.Sim(sir, networks=ss.MixingPool(n_contacts=10), unit='day').run()            # ODE; beta is per contact
ss.Sim(sir, networks=ss.contacts('Kenya'), unit='day').run()                    # age-structured ODE
ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), n_agents=100_000, unit='day').run()   # agents
```

The disease object is the same in all four. The second and fourth MUST agree in distribution as the population grows; [18-conformance.md](18-conformance.md) states the tolerance.

### Describing a network

A network is usually described by how it is generated, since that is what modelers know: "mean degree 0.7, partnerships last 50 days". An edge list is accepted (`ss.StaticNet(edges=...)`) but never required. Partnerships and acts are separate numbers with separate data sources, and sexual networks keep them separate (number of partners, partnership duration, acts per unit time), which is what gives a per-act transmission probability its meaning.

`net.check()` simulates the network alone and reports, for each target statistic it was given, the target, the realized mean, and the spread across replicates (`W402 TargetStatsNotRecovered` if a target is missed). Running it is recommended before trusting results from a fitted network.

## Frequency and density dependence

This is an option on the mixing, not on the transition, and `describe()` prints which is in force:

| `dependence=` | `D[j]` | Meaning |
|---|---|---|
| `'frequency'` (default) | `N[j]` | A person's number of contacts is fixed; the fraction of them that are infectious matters |
| `'density'` | `n_ref` | Contacts scale with population size; `n_ref` is a reference size, by default the initial population (printed) |

```python
ss.Sim(sir, networks=ss.MixingPool(dependence='density'))
```

Frequency dependence is the default because it is the usual assumption for directly transmitted human infections. Defining density dependence against a reference size makes the two conventions give the same answer when the population equals `n_ref`, so `beta` keeps the same meaning under both. The option applies to mixing pools; a network's contacts are already explicit.

## Layers

A list of routes is a layered population. Each layer has a name and can be targeted separately.

```python
home      = ss.contacts('Kenya', layer='home')
school    = ss.contacts('Kenya', layer='school')
work      = ss.contacts('Kenya', layer='work')
community = ss.contacts('Kenya', layer='community')

sim = ss.Sim(sir, networks=[home, school, work, community])
```

The force of infection sums over layers. `sim.networks.school` reaches a layer that was declared inline. `beta` is shared across layers unless given per layer, as a dict keyed by layer name as in Starsim (`beta=dict(home=0.1, school=0.05)`); a key that matches no route raises `E403 RouteMismatch`.

Closing schools is a modifier on the school layer's contacts ([07-effects.md](07-effects.md)):

```python
closure = ss.modify(school.pars.n_contacts, multiply=0.2, start='2020-03-15', stop='2020-06-01')
sim = ss.Sim(sir, networks=[home, school, work, community], interventions=closure)
```

## Mobility

Space is a dimension ([05-dimensions.md](05-dimensions.md#space)), and mobility is the optional movement between its levels. Without a mobility model, people mix within their own patch and patches do not affect each other.

```python
kenya     = ss.geo('Kenya', level='county')
sir_kenya = sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))
```

Movement models are given by name and parameters; a matrix or a data file is also accepted:

```python
ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0)    # distances and populations come from the geography
ss.radiation()                            # parameter-free
ss.stouffer(k=0.5, a=1.0)
ss.competing_destinations(k=0.5, a=1.0, b=1.0, c=2.0, delta=0.1)
ss.Mobility(matrix=M)                     # a matrix you already have
ss.Mobility(data='movements.csv')         # a movement register: columns date, from, to, n, [states]
```

`ss.gravity()` on a dimension with no geography raises `E402 NoGeography`: there are no distances to compute it from, and no reasonable guess. In a movement register, `states` defaults to all disease states, sampled in proportion to how many are in each at the origin; naming states covers selective movement (only healthy animals are sold).

### Commuting and migration

```python
gravity = ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0)

ss.Mobility(gravity, kind='commute', away=0.4)   # people keep their home patch; 40% of their time is spent away
ss.Mobility(gravity, kind='migrate')             # people relocate
```

Under `migrate`, a person's patch changes: it is an `ss.transfer` on the spatial dimension at the rates the mobility model gives.

Under `commute`, a person's home patch does not change. They spend a share `away` of their time in other patches, split according to the mobility model, and they are exposed and expose others in each patch in proportion to the time they spend there. With `T[i,k]` the share of time residents of `i` spend in `k` (rows sum to 1):

```
λ[i] = Σ_k  T[i,k] · β · (Σ_j T[j,k] · P[j]) / (Σ_j T[j,k] · N[j])
```

This is the standard time-share form of a commuting model. It needs no change to the timestep and works in every method. It does not represent the time of day; a model that needs daytime and nighttime mixing uses a sub-daily `dt` with separate layers.

Commuting couples patches without mixing their populations; migration does both. If `kind=` is not given, a gravity, radiation, or similar model is treated as `commute` and a movement register as `migrate`, and the choice is printed.

Travelers in transit (in neither their home patch nor their destination) are out of scope for v1. A model that needs them can add a transit patch by hand.

## Geography

```python
ss.geo('Kenya', level='county')
ss.geo('USA', level='county', year=2020)
ss.geo(shapefile='my_districts.shp', pop='pop.csv')
```

A geography supplies names, populations, centroids, distances, and adjacency. Sources are pluggable, the source used is recorded in `describe()` and in the run hash, and the bundled sources MUST include non-US geographies from the first release.

## Rejected

- **Making the modeler write `/N`.** It is one of the most commonly mis-written parts of a model. Writing counts literally remains available for the cases that need it.
- **A special route kind for environmental or vector-borne transmission.** Counts in rates plus `ss.Compartments` cover both with no new vocabulary.
- **Refusing to run a network model as an ODE.** The standard approximation is used and reported.
- **Frequency versus density as different transition kinds.** It is a property of the mixing.
- **Requiring an edge list.** Accept one; never require one.
- **Sub-step scheduling for commuting.** The time-share form gives the same coupling without it.
- **Opaque select and shift matrices for movement.** A register with named columns says the same thing.
