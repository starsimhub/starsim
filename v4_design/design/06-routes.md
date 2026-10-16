# Routes: mixing, networks, and the force of infection

How infection gets from one person to another. The disease says who infects whom (`S >> I @ beta * I`); the routes say who meets whom and how often. Changing from homogeneous mixing to a contact matrix to a network does not change the disease.

Justified by [best-practices/population-and-mixing.md](../best-practices/population-and-mixing.md) and [best-practices/metapopulation.md](../best-practices/metapopulation.md).

## What a state in a rate means

```python
S, I, R   = ss.states(3)
infection = S >> I @ beta * I
```

**A bare state inside a rate means contact with people in that state, through the model's routes.** The user does not write `/N`; the library works out the contact from whatever mixing the sim has.

There are two ways to mention a state in a rate, and they answer different questions:

- **`I.n` is how many people are in `I`.** It is a number, the same one that appears in the results as `n_I`, and it is used exactly as written ([below](#counts-used-literally)).
- **A bare `I` is the chance that a contact is with someone in `I`.** With no network that is `I.n / N`, so `beta * I` and `beta * I.n / N` are the same model. With a contact matrix or a network it is the same quantity worked out for each person from whom they actually meet, which cannot be written as one formula; that is why the bare form exists.

So a bare state is written when people catch the infection from people, and `.n` when the formula needs the number itself (a reservoir, a vector, a threshold). Three consequences:

- **Several infectious states are a sum of exposures**, each with its own coefficient: `S >> I @ beta * I + beta_vax * I_vax`, or `S >> E @ beta * (I + 0.5 * A)`.
- **A rate with no state is not transmission.** `S >> I @ beta` is legal and is a constant hazard that does not depend on anyone being infectious (an importation rate, say), exactly as `I >> R @ gamma` is a constant hazard of recovery. In the short form, the word `transmission` in `ss.transmission('S -> I', beta=0.05)` is what stands for `* I`.
- **A bare state that is not people is an error.** There is no contact with a reservoir to work out, so `beta * W` (with `W` bacteria in water) raises `E211 ExposureToCount`, which says to write `W.n`.

`describe()` prints every rate with its bare states expanded, for example `infection: S -> I at 0.3 × I.n / N  [homogeneous mixing]`, so what a rate resolved to can always be checked.

What `beta` means follows from the mixing:

- **No network given.** Mixing is homogeneous with one contact per unit time. `beta * I` is then the textbook `beta * I / N`, and `beta` is the classical transmission rate (per unit time, in the sim's unit).
- **Networks or mixing pools given** (`ss.Sim(sir, networks=...)`). They supply who meets whom and how often, and `beta` is the transmission probability per contact (per act, on a sexual network that defines acts).

The two readings are the same rule: the hazard is `beta` times the contact rate times the fraction of contacts that are infectious, and the default contact rate is 1. Which reading applies is printed by `describe()`, with the resulting units of `beta`.

In the string form the source is a keyword: `ss.transmission('S -> E', beta=0.05, source='I')` ([03-states-and-transitions.md](03-states-and-transitions.md)).

## The force of infection

Let

- `i`, `j` index **mixing groups**: the combinations of dimension levels the routes distinguish (a single group if the model is unstratified);
- `X[s,j]` be the number of people in state `s` in group `j`, and `N[j]` the number of people in group `j`;
- `T` be a transition `from(T) >> to(T)` whose rate is `beta * (source states)`, with `β[r]` its `beta` on route `r` (the same number on every route unless `beta` was given [per route](#beta-per-route-and-direction));
- `c[i,j]` be the contact rate supplied by a route: contacts per unit time that one person in group `i` has with people in group `j`.

The infectious pressure from group `j` is

```
P[j] = Σ_{s ∈ source(T)}  rel_trans[s] · X[s,j]
```

and the hazard of infection for a person in state `s ∈ from(T)` in group `i` is

```
λ[s,i] = rel_sus[s] · Σ_r Σ_j  c[r,i,j] · β[r] · P[j] / D[j]
```

where the outer sum is over routes `r`, `c[r,i,j]` is route `r`'s contact rate, and `D[j]` depends on the route's `dependence` option ([below](#frequency-and-density-dependence)) and is `N[j]` by default. With no route given there is one term with `c = 1`, so `λ = β · I / N`. The probability of infection over a step of length `dt` is `1 − exp(−λ · dt)`; if the transition branches, the infected are split across the destinations by the branch probabilities.

`rel_sus` and `rel_trans` are the state-level values declared on `ss.State` multiplied by any per-agent or per-group modifiers ([07-effects.md](07-effects.md)).

### Per-edge form

On a network that keeps individual contacts, the same quantities apply per edge. For an edge between infectious `u` and susceptible `v` carrying `acts` acts per unit time (1 if the network does not define acts):

```
hazard[e] = β · acts[e] · rel_trans[u] · rel_sus[v]
p[e]      = 1 − exp(−hazard[e] · dt)
p[v]      = 1 − Π_e (1 − p[e])         over all edges into v
```

The product form for `p[v]` MUST be used; adding the per-edge probabilities overcounts for people with many contacts.

### Beta per route and direction

Infectiousness belongs to the disease and contact to the routes, but infectiousness can itself depend on the route: HIV is far more transmissible through a shared needle than through sex, and more from men to women than the reverse. Those are still facts about the pathogen, so they stay on the disease, as one `beta` per network. This is Starsim's existing convention:

```python
infection = ss.transmission('S -> I', beta=dict(mf=[0.008, 0.004], needles=0.03))
sim = ss.Sim(hiv, networks=[ss.MFNet(), needles])
```

- A single number applies to every route.
- A dict is keyed by network or layer name. A key that matches no route raises `E403 RouteMismatch`.
- A pair is the two directions of a directed network (for `ss.MFNet`, male to female and then female to male).
- The operator form takes the same dict as its `beta`.

### Units

`beta` times the contact rate is a hazard. Bare numbers always work: a bare `beta` with the default mixing is a rate in the sim's unit; with a route it is a probability per contact, and a bare `n_contacts` is contacts per unit time. One of the two may carry a time unit (`beta=ss.perday(0.03)` with `n_contacts=10`, or `beta=0.03` with `n_contacts=ss.freqperday(10)`). If both carry one, the product is not a hazard and `E102 WrongDimension` is raised.

## Counts used literally

`I.n` is the plain number in a state (the same number as the result `n_I`), and `I.frac` is its fraction of the population. A rate may be any arithmetic expression of counts and parameters, and it is then used exactly as written, without going through the routes:

```python
S >> I @ beta * I                      # exposure to I through the routes (the usual case)
S >> I @ beta * I.n / N.n              # the same thing written out for homogeneous mixing; N = S + I + R
S >> I @ beta_w * W.n / (W.n + K)      # environmental: a saturating dose-response on a reservoir W
Sh >> Eh @ a * b * Iv.n / Nh.n         # vector-borne: bites per person from infectious mosquitoes
```

This is how environmental and vector-borne transmission are written. There is no separate environmental route, no special transmission kind, and no separate module class: the reservoir or the vector population is made of **ordinary states in the same disease**, written like any other, and their counts appear in the rates.

**States that are not people.** The one thing the library needs to know about such a state is that it is not people, because in an agent-based run the people become agents and the other states stay numbers, and because `total_pop`, age, and sex apply only to people.

- **It is worked out from the transitions.** A state is people if transitions connect it to the disease's first state, in either direction. No transition joins `W` to `S`, or `Sv` to `Sh`, so `W` and `Sv` are not people. `describe()` prints this as a guess (`W: not people [guessed]`), and the build notes it once (`W212 StateNotPeople`).
- **`ss.State(people=False)` says it outright.**
- **A state that is not people holds a plain count.** It is never represented as agents, it is left out of `total_pop` and of `N`, and it is skipped by people dimensions (age, sex, risk) when the model is stratified. It is stratified by `patch`, or by its own dimensions with `states=` ([05-dimensions.md](05-dimensions.md#states-that-are-not-people)).
- **Inflows and outflows use `ss.null`**, which stands for outside the model: `ss.null >> W @ rate` is an inflow and `W >> ss.null @ rate` an outflow ([03-states-and-transitions.md](03-states-and-transitions.md)). An inflow rate is a total per unit time, not a per-capita hazard.

In a compartmental run none of this changes anything, because every state is a number anyway.

Rules:

- Each term of a rate is either an exposure (it contains one bare state) or literal (it contains only `.n`, `.prev`, parameters, and numbers). A rate may add the two kinds: `S >> I @ beta * I + beta_w * W.n / (W.n + K)` is direct plus environmental transmission. A term containing two bare states has no defined meaning and is an error.
- A sum of states is a sum, not a new state: with `N = S + I + R`, `N.n` is the total, usable as a source or a denominator.
- In a stratified model, a count is aligned with the person at risk on every dimension the two share and summed over the rest. With cholera and a water reservoir both stratified by `patch`, `W.n` is the reservoir in the susceptible person's own patch; mosquitoes, which are not stratified by age, see `Ih.n` summed over human age groups. `I.n.sum('age')` and `I.n.select(age=...)` override the alignment.
- A transition whose rate mentions a state is still an infection for the purposes of results and of the infection log when one is recorded ([09-results.md](09-results.md)); the mentioned state is its source.

### Environmental transmission: cholera

```python
S, I, R, W = ss.states(4)                         # W is vibrios in the water

cholera = ss.Disease(
    infection = S >> I @ ss.perday(0.2) * W.n / (W.n + 1e6),
    recovery  = I >> R @ ss.days(5),
    shedding  = ss.null >> W @ ss.perday(10) * I.n,   # shed per infectious person per day
    decay     = W >> ss.null @ ss.days(30),
)

sim = ss.Sim(cholera, n_agents=10_000, unit='day')
```

No transition leads from `S` to `W`, so `W` is not people. Because `n_agents` is given, people are agents and the water is a count updated alongside them in the same module. Without it, every piece has a compartmental form and `solver='auto'` runs the model as an ODE. Stratifying by a geography gives one reservoir per patch ([05-dimensions.md](05-dimensions.md#space)).

### Vector-borne transmission

```python
Sh, Eh, Ih, Rh = ss.states(4)
Sv, Ev, Iv     = ss.states(3)
Nh = (Sh + Eh + Ih + Rh).n

a = ss.perday(0.3)     # bites per mosquito per day
b, c = 0.5, 0.5        # transmission probability per bite: mosquito to human, human to mosquito

malaria = ss.Disease(
    infection  = Sh >> Eh @ a * b * Iv.n / Nh,
    latency    = Eh >> Ih @ ss.days(12),
    recovery   = Ih >> Rh @ ss.days(60),
    emergence  = ss.null >> Sv @ ss.perday(5000),       # a total per unit time
    biting     = Sv >> Ev @ a * c * Ih.n / Nh,
    incubation = Ev >> Iv @ ss.days(10),
    mortality  = (Sv | Ev | Iv) >> ss.null @ ss.days(14),
)

sim = ss.Sim(malaria, n_agents=5_000, unit='day')
```

Nobody can move from `Sh` into `Sv`, so the mosquito states are not people: `n_agents=` makes the humans agents while the mosquitoes stay three numbers. Mosquito species are a stratification of those states (`malaria.stratify(species=[...], states=[Sv, Ev, Iv])`).

## Routes

Routes are passed with Starsim's keyword `networks=`, which accepts networks and mixing pools alike, singly or as a list.

| Route | What it supplies | Agent-based run | Compartmental run |
|---|---|---|---|
| none given | homogeneous mixing, 1 contact per unit time | exact | exact |
| `ss.MixingPool(n_contacts=c)` | homogeneous mixing, `c` contacts per unit time | exact | exact |
| `ss.MixingPools(...)`, `ss.contacts('Kenya')` | a contact matrix between groups | exact | exact |
| `ss.RandomNet(n_contacts=c)` | individual contacts, redrawn or persisting for `dur` | exact | approximated by homogeneous mixing with `c` contacts; exact as the population grows when contacts are redrawn every step |
| `ss.MFNet(...)` and other sexual networks | partnerships with durations and acts; `beta` is per act | exact | approximated by partner-change rates between sex/risk/age groups |
| `ss.StaticNet(edges=...)` | a fixed set of contacts | exact | approximated by group-to-group contact rates computed from the edges (homogeneous mixing with the same mean number of contacts if the nodes have no groups) |
| counts used literally (`W.n`, `Iv.n`) | nothing; the rate is the hazard | exact | exact |

There is no `ss.Homogeneous` class. Homogeneous mixing is what happens when no network is given; to set a contact rate use `ss.MixingPool(n_contacts=10)`, which is Starsim's class and is homogeneous when given no groups.

How the columns are used follows the approximation policy in [12-backends.md](12-backends.md):

- `solver='auto'` never approximates the mixing. A model with only "exact" compartmental entries runs as an ODE; a model containing an individual-level network runs as agents, and the choice is printed. (The one approximation `auto` may make is Erlang stages for a non-exponential dwell time, which it reports; [12-backends.md](12-backends.md).)
- When the user asks for a compartmental solver on a model with a network (`solver='ode'`), the library uses the approximation in the table and says so in one line (for example `sexual network 'mf' approximated by partner-change rates between sex and risk groups`). It does not refuse. A network model run as an ODE is a coarser model, and there are many valid ODE models of sexually transmitted infections.
- `approx=False` turns any approximation into an error (`E804 CapabilityError`), and `approx=True` silences the note. `sir.solvers()` shows this table for the model at hand.

The table has one approximation per route. A network with strong variation in the number of contacts is approximated by homogeneous mixing with the same mean like any other, which is a poor approximation for it; there is no degree-grouped ODE form ([Rejected](#rejected)), and the agent-based run is the way to keep that variation.

**The sexual-network approximation.** The model is stratified by the attributes the network forms partnerships on (sex, plus risk group and age band where the network uses them), if it is not already. The contact rate `c[i,j]` is the rate at which a person in group `i` starts partnerships with people in group `j`, derived from the network's own parameters (number of partners, duration, acts) so that the user does not supply them twice, and the per-contact probability becomes the probability of transmission over a whole partnership, `1 − (1 − β)^(acts × dur)`. What is lost is concurrency, repeated exposure to the same partner, and variation in partner numbers within a group. The note printed at build time lists the groups and the rates used.

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

A network is usually described by how it is generated, since that is what modelers know: "0.7 partners on average, partnerships last 50 days". An edge list is accepted (`ss.StaticNet(edges=...)`) but never required.

```python
ss.MFNet(
    n_contacts  = ss.poisson(0.7),     # how many partners at a time
    duration    = ss.days(50),         # how long partnerships last
    acts        = ss.freqperweek(2),   # how often, within a partnership
    concurrency = 0.22,                # the share of people with more than one partner at once
)
```

- **Partnerships and acts are separate numbers** with separate data sources (number of partners, partnership duration, acts per unit time), and sexual networks keep them separate. That is what gives a per-act transmission probability its meaning.
- **`n_contacts` takes a distribution.** "0.7 partners" on its own is ambiguous: one partner 70% of the time (`ss.bernoulli(0.7)`) and a Poisson number of partners with mean 0.7 (`ss.poisson(0.7)`) are different networks. A bare number is read as the mean of a Poisson, and `describe()` says so. `n_contacts` is the name Starsim's other networks already use, and it takes a distribution there too.
- **`n_contacts`, `concurrency`, and `max_contacts` are targets**, as for `ss.PartnerNet` (below): partnerships form between men and women so that the number each person has at a time settles at the target distribution, and anyone left without a partner of the other sex waits for one. Without them, `ss.MFNet` forms partnerships from `participation` and `debut`, as in Starsim v3. `net.check()` reports whether a generated network recovers the targets.

`net.check()` simulates the network alone and reports, for each target statistic it was given, the target, the realized mean, and the spread across replicates (`W402 TargetStatsNotRecovered` if a target is missed). Running it is recommended before trusting results from a fitted network.

A one-mode network of partnerships is described the same way, by `ss.PartnerNet`, with the targets EpiModel's introductory network is fitted to:

```python
ss.PartnerNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)   # 175 partnerships among 500, 110 people with two or more, nobody with four or more
```

The targets are cross-sectional: nobody has a fixed number of partners. Partnerships end at a constant rate, and people look for partners at a rate that depends on how many they already have, chosen so that the target distribution is the network's stable state. This matches the number of partners people have, which is what most published models target. Mixing by a property of people is a target too: `match=dict(risk=0.9)` makes 90% of partnerships between people at the same level of `risk` (EpiModel's `nodematch`), and `net.check()` reports the share realized. Triangles (two of a person's partners who are partners of each other), which an ERGM can also target, are not covered: they need the network fitted as a whole, not partner by partner.

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

A list of routes is a layered population. Each layer has a name and can be targeted separately. Layers are Starsim's, as of v3.7.2; nothing here is new machinery.

```python
home      = ss.contacts('Kenya', layer='home')
school    = ss.contacts('Kenya', layer='school')
work      = ss.contacts('Kenya', layer='work')
community = ss.contacts('Kenya', layer='community')

sim = ss.Sim(sir, networks=[home, school, work, community], stratify=dict(age=[0, 5, 18, 65]))
```

The force of infection sums over layers. `sim.networks.school` reaches a layer that was declared inline. `beta` is shared across layers unless given per layer, as a dict keyed by layer name (`beta=dict(home=0.1, school=0.05)`; [above](#beta-per-route-and-direction)).

Age is a property of the people and not of one disease, so it is the sim that is stratified, and every disease in it gets the same age groups ([05-dimensions.md](05-dimensions.md#stratifying-the-sim-or-the-disease)).

**Contact matrices with other age groups.** The matrices come with their own age groups (five-year bands for the Prem matrices, single years for Mistry), which need not be the model's.

- In an agent-based run that does not matter: each agent has an exact age and is counted in whichever band the matrix uses.
- In a compartmental run the matrix is aggregated to the model's groups, weighted by population, which is the standard calculation. Where a model group cuts through a matrix band, the band is split in proportion to population. A one-line note says so, and `describe()` shows the aggregated matrix.

Closing schools is a modifier on the school layer's contacts ([07-effects.md](07-effects.md)):

```python
closure = ss.modify(school.pars.n_contacts, multiply=0.2, start='2020-03-15', stop='2020-06-01')
sim = ss.Sim(sir, networks=[home, school, work, community], stratify=dict(age=[0, 5, 18, 65]), interventions=closure)
```

## Mobility

Space is a dimension ([05-dimensions.md](05-dimensions.md#space)), and mobility is the optional movement between its levels. Without a mobility model, people mix within their own patch and patches do not affect each other.

```python
kenya     = ss.geo('Kenya', level='county')
sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0))
```

Movement models are given by name and parameters; a matrix or a data file is also accepted:

```python
ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0)    # distances and populations come from the geography
ss.Mobility(matrix=M)                     # a matrix you already have
ss.Mobility(data='movements.csv')         # a movement register: columns date, from, to, n, [states], [to_states]
```

Other named models (radiation, Stouffer, competing destinations, the set LASER ships) are deferred until a vignette needs one ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)); a matrix covers them meanwhile. The generators are implemented in the library itself. It does not call LASER's migration module or any other framework's package for them: an outside dependency is harder to keep working than a few lines of code we control. `ss.gravity()` on a dimension with no geography raises `E402 NoGeography`: there are no distances to compute it from, and no reasonable guess. In a movement register, `states` defaults to all disease states, sampled in proportion to how many are in each at the origin; naming states covers selective movement (only healthy animals are sold). `to_states` moves the people chosen from each named state to the matching state of its list, in the same place or another, which is SimInf's shift matrix: `states='S1, I1'`, `to_states='S2, I2'` ages calves into young stock, keeping each one's infection status.

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

This is the standard time-share form of a commuting model. It needs no change to the timestep and works under every solver. It does not represent the time of day; a model that needs daytime and nighttime mixing uses a sub-daily `dt` with separate layers.

A third form moves neither people nor time: `kind='force'` moves the force of infection itself, as LASER does. With `M[i,k]` the movement from `i` to `k` (scaled by `away=`), each patch keeps `1 − Σ_k M[i,k]` of its own force of infection and receives `M[k,i]` of each other patch's, unweighted by the patches' sizes. It is less mechanistic than commuting, but it is what LASER models are written in, and the same gravity matrix gives different epidemics under the two forms (in LASER's hub-and-spoke example, 88% infected against 80%).

Commuting couples patches without mixing their populations; migration does both. If `kind=` is not given, a gravity model or matrix is treated as `commute` and a movement register as `migrate`, and the choice is printed.

Travelers in transit (in neither their home patch nor their destination) are out of scope for v1. A model that needs them can add a transit patch by hand.

## Geography

```python
ss.geo('Kenya', level='county')
ss.geo('USA', level='county', year=2020)
ss.geo(shapefile='my_districts.shp', pop='pop.csv')
```

A geography supplies names, populations, centroids, distances, and adjacency. Sources are pluggable, and the source used is recorded in `describe()` and in the `pars` part of `sim.hash()`. Geographic data follows the same policy as the rest of the named data ([04-parameters.md](04-parameters.md#named-data)): what is small and common is bundled, the rest is fetched on demand and cached, each data set is adapted once with its provenance recorded, and nothing depends on another framework's data package. The bundled set MUST include non-US geographies from the first release.

## Rejected

- **Making the modeler write `/N`.** It is one of the most commonly mis-written parts of a model. Writing counts literally remains available for the cases that need it.
- **A special route kind for environmental or vector-borne transmission.** Counts in rates plus ordinary states that are not people cover both with the ordinary transition syntax.
- **A separate module class for quantities that are not people** (an earlier draft's `ss.Compartments`). The reservoir's states use the same syntax as the disease's, so a second class added a name without adding anything to say.
- **A degree-grouped ODE approximation** for networks with strong variation in the number of contacts. Homogeneous mixing with the same mean is a poor approximation for these, and the standard improvement groups people by degree. It is not worth building: the agent-based run is one keyword away and is exact.
- **A separate `ss.Homogeneous` class, or a new keyword for mixing.** Homogeneous mixing is the default, and `networks=` takes networks and mixing pools today.
- **Depending on other frameworks' packages for mobility models or data.** The generators and the data are ours.
- **Refusing to run a network model as an ODE.** The standard approximation is used and reported.
- **Frequency versus density as different transition kinds.** It is a property of the mixing.
- **Requiring an edge list.** Accept one; never require one.
- **Sub-step scheduling for commuting.** The time-share form gives the same coupling without it.
- **Opaque select and shift matrices for movement.** A register with named columns says the same thing.
