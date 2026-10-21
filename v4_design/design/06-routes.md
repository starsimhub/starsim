# Routes: mixing, networks, and the force of infection

How infection gets from one person to another. The disease says who infects whom (`S >> I @ beta * I`); the routes say who meets whom and how often. Changing from homogeneous mixing to a contact matrix to a network does not change the disease.

Justified by [best-practices/population-and-mixing.md](../best-practices/population-and-mixing.md) and [best-practices/metapopulation.md](../best-practices/metapopulation.md).

## What a state in a rate means

<!-- fragment -->
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
- **A rate with no state is not transmission.** `S >> I @ beta` is legal and is a constant hazard that does not depend on anyone being infectious (an importation rate, say), exactly as `I >> R @ gamma` is a constant hazard of recovery.
- **A bare state that is not people is an error.** There is no contact with a reservoir to work out, so `beta * W` (with `W` bacteria in water) raises `ExposureToCount`, which says to write `W.n`.

`describe()` prints every rate with its bare states expanded, for example `rate = 0.3 × I.n / N  [homogeneous mixing]` under `infection`, so what a rate resolved to can always be checked.

What `beta` means follows from the mixing:

- **No network given.** Mixing is homogeneous with one contact per unit time. `beta * I` is then the textbook `beta * I / N`, and `beta` is the classical transmission rate (per unit time, in the sim's unit).
- **Networks or mixing pools given** (`ss.Sim(sir, networks=...)`). They supply who meets whom and how often, and `beta` is the transmission probability per contact (per act, on a sexual network that defines acts).

The two readings are the same rule: the hazard is `beta` times the contact rate times the fraction of contacts that are infectious, and the default contact rate is 1. Which reading applies is printed by `describe()`, with the resulting units of `beta`.

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

where the outer sum is over routes `r`, `c[r,i,j]` is route `r`'s contact rate, and `D[j] = N[j]`: transmission through the routes is frequency-dependent ([below](#frequency-and-density-dependence)). With no route given there is one term with `c = 1`, so `λ = β · I / N`. The probability of infection over a step of length `dt` is `1 − exp(−λ · dt)`; if the transition branches, the infected are split across the destinations by the branch probabilities.

`rel_sus` and `rel_trans` are the state-level values declared on `ss.State` multiplied by any per-agent or per-group modifiers ([07-effects.md](07-effects.md)).

### Per-edge form

On a network that keeps individual contacts, the same quantities apply per edge. On a network with acts (`ss.MFNet`, `ss.PartnerNet(acts=...)`, a subclass of `ss.SexualNetwork`), for an edge `e` from infectious `u` to susceptible `v` with `a[e]` acts per unit time and edge weight `w[e]`:

```
h[d]      = −ln(1 − p[d])                         the hazard per act, for the edge's direction d (from u's side to v's)
M[e]      = Π_k (1 − s_k · (1 − m_k))             the modifiers of the transmission on this route (s_k = 1 without share=)
hazard[e] = w[e] · a[e] · h[d] · w_s · rel_trans[u] · rel_sus[v] · M[e]     (w_s the weight of u's state in the rate, 0.5 for A in beta * (I + 0.5 * A))
p[e]      = 1 − exp(−hazard[e] · dt)            (= 1 − (1 − p[d])^(a[e]·dt) when nothing else applies)
p[v]      = 1 − Π_e (1 − p[e])                    over all edges into v
```

`rel_trans`, `rel_sus`, source weights, edge weights, and modifiers (of `beta`, of the transmission, with or without `route=`) all multiply the hazard of each act, so `beta=0.03` with `rel_trans=10` is `1 − 0.97^10 = 0.263` per act; the product never passes 1, and factors compose in any order. Over a year the edge transmits with `1 − (1 − p)^a`, whatever the step. A network written as code that overrides `net_beta()` keeps Starsim 3's contract instead (`net.custom_beta`): it is called once for each direction of the edges with the probability per act of each edge, and what it returns is the probability for the partnership in the step, which `rel_trans`, `rel_sus` and the modifiers on the route then multiply. A disease written as code (an `ss.Infection`) always has Starsim 3's contract, over the same mechanics (`ss.Network.transmit()`): a probability per act in each direction (`ss.peract(m2f=..., f2m=...)`) on any network with sides, and the modifiers of the network's contacts and of its `rel_sus` on the route.

On a network without acts, `β` is a probability (or rate) per contact: `p[e] = 1 − exp(−β·dt)` times the edge weight, `rel_trans[u]`, and `rel_sus[v]`.

The product form for `p[v]` MUST be used; adding the per-edge probabilities overcounts for people with many contacts.

### Beta per route and direction

Infectiousness belongs to the disease and contact to the routes, but infectiousness can itself depend on the route: HIV is far more transmissible through a shared needle than through sex, and more from men to women than the reverse. Those are still facts about the pathogen, so they stay on the disease, as one `beta` per network. This is Starsim's existing convention:

<!-- fragment -->
```python
S, I = ss.states('S I')
infection = ss.Transition(S, I, rate=dict(mf=ss.peract(m2f=0.008, f2m=0.004), needles=ss.peract(0.006)) * I)
sim = ss.Sim(hiv, networks=[ss.MFNet(), needles])
```

- A single value applies to every route.
- A dict is keyed by network or layer name, and names every route: a key that matches no route, or a route with no key, raises `RouteMismatch` when the sim is initialized. Under every solver the dict is applied in one place, to the exposure through each route.
- A route has no `beta` of its own in a declared model (`ss.MixingPool(beta=...)` other than 1 raises `RouteMismatch`; `ss.contacts()` takes none): the `beta` belongs to the transmission.
- On a network with acts, `beta` is a probability per act: `ss.peract(p)`, or a bare number (as in Starsim 3). A typed rate there (`ss.permonth(b)`, `ss.probpermonth(b)`) raises `WrongDimension`, since a rate per unit time times acts per unit time is not a hazard. `ss.peract` on a route without acts raises `RouteMismatch`: there is no act to be per.
- Directions are named, never positional. A network says what its **sides** are (`sides=`, a dict of a label to a condition over people, or a class attribute): `ss.MFNet` has `m` and `f` (people's sex), `ss.PartnerNet(sides='sex')` has `m` and `f` with partnerships between anyone, `ss.PartnerNet(between=('fsw', 'client'))` has its two groups, and the mother-to-child networks have `mother` (`p1`) and `child` (`p2`). Its directions, `net.directions`, are `<side of the infectious partner>2<side of the susceptible one>` for each pair of sides its edges can join: `m2f` and `f2m` on `ss.MFNet`; `m2f`, `f2m`, `m2m`, `f2f` with `sides='sex'`; `fsw2client` and `client2fsw`; and `mother2child` only, since the mother-to-child networks transmit one way. A network with no sides has no directions and takes one value. `ss.peract(m2f=0.3, f2m=0.15)` gives one for each (a direction may also be written as a pair of sides, `{('m', 'f'): 0.3}`); every direction needs a value, unless one is given for all, and a direction the network does not have raises `RouteMismatch`, listing those it has. The engine labels each edge's ends by side once a step, so the order of a network's partners does not matter.
- A pair (`[b1, b2]`) raises `RouteMismatch`, giving the `ss.peract` spelling.
- The operator form takes the same dict as its `beta`.

### Units

`beta` times the contact rate is a hazard. Bare numbers always work: a bare `beta` with the default mixing is a rate in the sim's unit; with a route it is a probability per contact (per act, on a network with acts), and a bare `n_contacts` is contacts per unit time. On a network with acts, `acts=` is acts per unit time in each partnership: a rate (`ss.freqperweek(2)`), or a distribution of rates drawn once for each partnership (`ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30))`, the default of `ss.MFNet`); the edge keeps its acts in a step, `a·dt`, as a real number, never truncated. A bare number, or a whole number of acts per step drawn once (`ss.poisson(lam=ss.freqperyear(80))`), would depend on the step, so raises on lasting partnerships. One of the two may carry a time unit (`beta=ss.perday(0.03)` with `n_contacts=10`, or `beta=0.03` with `n_contacts=ss.perday(10)`). If both carry one, the product is not a hazard and `WrongDimension` is raised.

## Counts used literally

`I.n` is the plain number in a state (the same number as the result `n_I`), and `I.frac` is its fraction of the population. A rate may be any arithmetic expression of counts and parameters, and it is then used exactly as written, without going through the routes:

<!-- fragment -->
```python
S >> I @ beta * I                      # exposure to I through the routes (the usual case)
S >> I @ beta * I.n / N.n              # the same thing written out for homogeneous mixing; N = S + I + R
S >> I @ beta_w * W.n / (W.n + K)      # environmental: a saturating dose-response on a reservoir W
Sh >> Eh @ a * b * Iv.n / Nh.n         # vector-borne: bites per person from infectious mosquitoes
```

This is how environmental and vector-borne transmission are written. There is no separate environmental route, no special transmission kind, and no separate module class: the reservoir or the vector population is made of **ordinary states in the same disease**, written like any other, and their counts appear in the rates.

**States that are not people.** The one thing the library needs to know about such a state is that it is not people, because in an agent-based run the people become agents and the other states stay numbers, and because `total_pop`, age, and sex apply only to people.

- **It is declared.** Every state is people unless it is declared a count with `ss.State(people=False)` (or `ss.states(3, people=False)`); it is not worked out from the transitions ([03-states-and-transitions.md](03-states-and-transitions.md)).
- **A state that is not people holds a plain count.** It is never represented as agents, it is left out of `total_pop` and of `N`, and it is skipped by people dimensions (age, sex, risk) when the model is stratified. It is stratified by `patch`, or by its own dimensions with `states=` ([05-dimensions.md](05-dimensions.md#states-that-are-not-people)).
- **Inflows and outflows use `ss.null`**, which stands for outside the model: `ss.null >> W @ rate` is an inflow and `W >> ss.null @ rate` an outflow ([03-states-and-transitions.md](03-states-and-transitions.md)). An inflow rate is a total per unit time, not a per-capita hazard.

In a compartmental run none of this changes anything, because every state is a number anyway.

Rules:

- Each term of a rate is either an exposure (it contains one bare state) or literal (it contains only `.n`, `.frac`, parameters, and numbers). A rate may add the two kinds: `S >> I @ beta * I + beta_w * W.n / (W.n + K)` is direct plus environmental transmission. A term containing two bare states has no defined meaning and is an error.
- A sum of states is a sum, not a new state: with `N = S + I + R`, `N.n` is the total, usable as a source or a denominator.
- In a stratified model, a count is the total, except within a place: with cholera and a water reservoir both stratified by `patch`, `W.n` is the reservoir in the susceptible person's own patch, while on any other dimension (age, risk) `I.n` is summed over its levels. There is no spelling yet for a count at the person's own level of another dimension.
- A count is used literally even when the model has routes: `S >> I @ beta * I.n / N.n` under `ss.contacts(...)` or a network is homogeneous mixing, not mixing through them. The build notes this once for each transition that uses a count of people while there are routes, and says to write the bare state for exposure through them.
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
    mortality  = Sv + Ev + Iv >> ss.null @ ss.days(14),
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
| `ss.MFNet(...)`, `ss.PartnerNet(...)`, and other networks of partnerships | partnerships with durations and acts; `beta` is per act | exact | the partnership approximation (below), marked `approx` |
| `ss.PartnerNet(between=(a, b), n_contacts=dict(b=rate))` with no `dur=` | one-off contacts between two groups, with acts per contact | exact | exact: a contact matrix between the two groups (levels of one stratification), with `1 − E[exp(−K·h)]` per contact for `K` acts |
| `ss.contacts(matrix, dims=['sex', 'risk'])` | a contact matrix over several stratifications together | exact | exact |
| `ssl.StaticNet(graph=...)` (the library's) | a fixed set of contacts | exact | approximated by homogeneous mixing with the mean degree of its graph (or, for a random graph made when the sim starts, its mean number of contacts); `CapabilityError` for a graph made by another function |
| counts used literally (`W.n`, `Iv.n`) | nothing; the rate is the hazard | exact | exact |

There is no `ss.Homogeneous` class. Homogeneous mixing is what happens when no network is given; to set a contact rate use `ss.MixingPool(n_contacts=10)`, which is Starsim's class and is homogeneous when given no groups.

How the columns are used follows the approximation policy in [12-backends.md](12-backends.md):

- `solver='auto'` never approximates the mixing. A model with only "exact" compartmental entries runs as an ODE; a model containing an individual-level network runs as agents, and the choice is printed. (The one approximation `auto` may make is Erlang stages for a non-exponential dwell time, which it reports; [12-backends.md](12-backends.md).)
- When the user asks for a compartmental solver on a model with a network (`solver='ode'`), the library uses the approximation in the table and says so in one line (for example `ssl.StaticNet` is `StaticNet approximated by homogeneous mixing with 2 contacts per day (the mean degree of its graph)`). A network with no approximation in the table raises `CapabilityError`.
- `approx=False` turns any approximation into an error (`CapabilityError`), and `approx=True` silences the note. `sim.describe()` shows this for the sim at hand, with a row for each network (`approx (contact rates)` for `ss.RandomNet`, `no (no approximation)` for `ss.MFNet`).

The table has one approximation per route. A network with strong variation in the number of contacts is approximated by homogeneous mixing with the same mean like any other, which is a poor approximation for it; there is no degree-grouped ODE form ([Rejected](#rejected)), and the agent-based run is the way to keep that variation.

**Networks of partnerships.** A partnership lasts, so its number of partners is not a number of contacts per unit time, and homogeneous mixing at that rate ignores how long partnerships last (with `ss.PartnerNet`, SIS prevalence 0.71 under an ODE against 0.17 as agents when partnerships last 500 days). Under a compartmental solver they use the partnership approximation instead, marked `approx` (a note, and `CapabilityError` with `approx=False`), and they run as agents under `auto`. A person in group `i` forms partnerships at `φ_i = n_i/D̄` (partners at a time over how long partnerships last), a share `m_ij` of them with people in group `j` (from the network's sides, `match=`, and partners at a time by level, with the side with fewer partners in all setting how many partnerships there are); a partnership with someone in a source state transmits before it ends, the partner leaves that state, or the person leaves theirs (`c`, the total rate of those exits) with `P = H/(H + c + 1/D̄)` for partnerships that end at a constant rate and `P = H/(H + c)·(1 − E[exp(−(H + c)·T)])` over 32 points of any other duration distribution, where `H = a·h_d` is the mean acts per unit time times the hazard per act in the direction from the partner's side (or the beta itself, as a rate, on a network without acts); and the force of infection is `λ_i = φ_i·Σ_j m_ij·Σ_s w_s·P_{ij,s}·I_{j,s}/N_j`. The sides of the network (sex for `ss.MFNet` and `sides='sex'`, the two levels of `between=`) and the property it matches or counts partners by must be dimensions of the model (`sim.stratify(sex=['female', 'male'])`); `where=` and sides given as conditions raise. `rel_trans`, `rel_sus`, and modifiers on the route multiply `P` rather than `H`, and `P` is worked out when the run starts. It leaves out what the network remembers: partnerships under way when the run starts, and partnerships under way when a partner is infected, so it is too low with concurrency, and too high for long monogamous partnerships with a long infection, where infection stays within pairs (see the measured ratios in [`12-backends.md`](12-backends.md)).

Things that need to know who a person's contacts are (contact tracing, partner notification, the transmission tree) only work on routes that keep contacts, and so only in an agent-based run.

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.05 * I),
    recovery  = ss.Transition(I, R, dur=ss.days(10)),
    init_prev = 0.01,
)

ss.Sim(sir, unit='day').run()                                                   # ODE; beta is a rate per day
ss.Sim(sir, networks=ss.MixingPool(n_contacts=10), unit='day').run()            # ODE; beta is per contact
ss.Sim(sir, networks=ss.contacts('Kenya'), unit='day').run()                    # age-structured ODE
ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), n_agents=100_000, unit='day').run()   # agents
```

The disease object is the same in all four. The second and fourth MUST agree in distribution as the population grows; [18-conformance.md](18-conformance.md) states the tolerance.

### Describing a network

A network is usually described by how it is generated, since that is what modelers know: "0.7 partners on average, partnerships last 50 days". A fixed set of contacts is accepted as a NetworkX graph, or a function that makes one (`ssl.StaticNet(graph=...)`, from the library, `import starsim.library as ssl`), but never required.

```python
ss.PartnerNet(
    n_contacts = 0.7,                  # the mean number of partners at a time
    dur        = ss.days(50),          # how long partnerships last
    acts       = ss.freqperweek(2),    # how often, within a partnership (acts per unit time)
)
```

- **Partnerships and acts are separate numbers** with separate data sources (number of partners, partnership duration, acts per unit time), and networks of partnerships keep them separate. That is what gives a per-act transmission probability its meaning.
- **Partnerships are not only sexual.** `ss.PartnerNet` is lasting partnerships between anyone, among some people (`where=`, a condition over people, e.g. needle sharing among people who inject), or between two groups (`between=(a, b)`: each partnership joins one person of each group, levels of a stratification or boolean properties of people, which are its sides; `n_contacts` can be a number for each, `dict(fsw=3, client=0.5)`, which balance only in the mean, the surplus of one group waiting). People look for a new partner with the same chance on every step, however many they have, and those looking are paired at random; partnerships end at a constant rate (or after a drawn duration). So `n_contacts` is the mean number of partners at a time, which in a closed population is Poisson, and the network starts there; these are the partnerships the partnership approximation assumes (below). With `between=` and no `dur=`, contacts are one-off: they last a step, `n_contacts` is a rate for one group (`dict(client=ss.permonth(1))`), whose contacts are each with someone of the other group at random, and `acts=` is acts per contact (default 1).
- **`ss.MFNet`** is partnerships between men and women formed from `participation` and `debut`, as in Starsim v3: people with no partner are paired, so nobody has more than one at a time. `ss.SexualNetwork` is the base class that STI models subclass.

Targets for the number of partners at a time, as EpiModel's networks fitted to target statistics have, are the library's (`import starsim.library as ssl`): `ssl.DegreeNet` takes the arguments of `ss.PartnerNet` (except one-off contacts), with `n_contacts` as a target distribution (a bare number is the mean of a Poisson; "0.7 partners" on its own is ambiguous, since one partner 70% of the time, `ss.bernoulli(0.7)`, and `ss.poisson(0.7)` are different networks), `concurrency` (the share with two or more) and `max_contacts`; `ss.MFNet` takes these three as well (without them, people with no partner are paired, as in Starsim v3). The EpiModel importer makes an `ssl.DegreeNet`. EpiModel's introductory network:

```python
import starsim.library as ssl
ssl.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)   # 175 partnerships among 500, 110 people with two or more, nobody with four or more
```

The targets are cross-sectional: nobody has a fixed number of partners. Partnerships end at a constant rate, and people look for partners at a rate that depends on how many they already have, chosen so that the target distribution is the network's stable state; partnerships ended by a partner leaving are allowed for (EpiModel's d.rate correction). Mixing by a property of people is a target too: `match=dict(risk=0.9)` makes 90% of partnerships between people at the same level of `risk` (EpiModel's `nodematch`). Several properties each have a target of their own (`match=dict(risk=0.9, region=0.7)`): each person looks within their level of each with its own chance, and is paired with someone looking within the same levels; this gives each target when the properties are independent. A target below the share that pairing at random gives cannot be reached, and gives that share. `n_contacts` can be given for each level of a property of people (`n_contacts=ss.by(risk=[0.4, 1.6])`, EpiModel's nodefactor). `net.check()` runs the sim again (three times by default) and reports, for each target, the target, the mean over the second half of each run, and the spread across runs (`TargetStatsNotRecovered` if a target is missed by more than the spread). Triangles (two of a person's partners who are partners of each other), which an ERGM can also target, are not covered: they need the network fitted as a whole, not partner by partner.

## Frequency and density dependence

Transmission through the routes is frequency-dependent: a person's number of contacts is fixed, and the fraction of them that are infectious matters (`D[j] = N[j]` above). This is the usual assumption for directly transmitted human infections. There is no option for it on the mixing. A density-dependent rate, where contacts grow with the size of the population, is written with the count itself, which is used literally ([above](#counts-used-literally)):

```python
S, I, R = ss.states(3)
sir_density = ss.Disease(
    infection = S >> I @ ss.perday(3e-5) * I.n,    # grows with the number infected, not their share
    recovery  = I >> R @ ss.days(10),
    init      = dict(I=10),
)
ss.Sim(sir_density, total_pop=10_000, unit='day').run()
```

The importers write a rate such as `beta * S * I` with no division by the total this way (`beta * I.n`). Since a rate per person that grows with a count of people is often meant as a share (`I.frac`), the build notes it as density-dependent, and says how to write a share.

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
closure = ss.modify('school.pars.n_contacts', multiply=0.2, start='2020-03-15', stop='2020-06-01')
sim = ss.Sim(sir, networks=[home, school, work, community], stratify=dict(age=[0, 5, 18, 65]), interventions=closure, start='2020-01-01', stop='2020-12-31')
```

The target is the parameter's path as a string, since `school.pars.n_contacts` itself is a plain value (the matrix). It works under every solver: in a compartmental run the contacts are scaled while the modifier is on, and as agents each contact happens with that probability in a step.

## Mobility

Space is a dimension ([05-dimensions.md](05-dimensions.md#space)), and mobility is the optional movement between its levels. Without a mobility model, people mix within their own patch and patches do not affect each other. The routes act within each patch: a contact matrix by age under every solver, and, as agents, a mixing pool of everyone and a random network too, as in a compartmental run (people contact others where they live, or where they commute to).

```python
kenya     = ss.geo('Kenya', level=1)
sir.stratify(patch=kenya, mobility=ss.gravity(away=0.1, a=1.0, b=1.0, c=2.0))
```

Movement models are given by name and parameters; a matrix or a data file is also accepted:

<!-- fragment -->
```python
ss.gravity(away=0.1, a=1.0, b=1.0, c=2.0)    # distances and populations come from the geography
ss.Mobility(matrix=M)                     # a matrix you already have
ss.Mobility(data='movements.csv')         # a movement register: columns date, from, to, n, [states], [to_states]
```

Other named models (radiation, Stouffer, competing destinations, the set LASER ships) are deferred until a vignette needs one ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)); a matrix covers them meanwhile. The generators are implemented in the library itself. It does not call LASER's migration module or any other framework's package for them: an outside dependency is harder to keep working than a few lines of code we control. `ss.gravity()` on a dimension with no geography raises `NoGeography`: there are no distances to compute it from, and no reasonable guess. In a movement register, `states` defaults to all disease states, sampled in proportion to how many are in each at the origin; naming states covers selective movement (only healthy animals are sold). `to_states` moves the people chosen from each named state to the matching state of its list, in the same place or another, which is SimInf's shift matrix: `states='S1, I1'`, `to_states='S2, I2'` ages calves into young stock, keeping each one's infection status. A row with no `from` adds people to `to` (births, purchases), and one with no `to` removes them (deaths, sales). The events dated at a time point are in the numbers at that time point, in the order of the table.

### Commuting and migration

```python
gravity = ss.gravity(a=1.0, b=1.0, c=2.0)

ss.Mobility(gravity, kind='commute', away=0.4)                  # people keep their home patch; 40% of their time is spent away
ss.Mobility(ss.gravity(k=2e-5, c=2.0), kind='migrate')          # people relocate, at rates scaled by a calibrated k
```

The gravity model gives the shape of the movement, and `away=` its amount: the share of their time people spend away, split between the other patches by gravity (10% if neither `away=` nor `k=` is given, which is printed). `k` has units of distance^c / pop^(a+b−1), so no default fits both three towns and a country; it is for a calibrated amount, and for the rates of migration. Without `away=`, nobody can spend more than all their time away, and a warning names how many places hit that cap.

Under `migrate`, a person's patch changes: it is an `ss.transfer` on the spatial dimension at the rates the mobility model gives.

Under `commute`, a person's home patch does not change. They spend a share `away` of their time in other patches, split according to the mobility model, and they are exposed and expose others in each patch in proportion to the time they spend there. With `T[i,k]` the share of time residents of `i` spend in `k` (rows sum to 1):

```
λ[i] = Σ_k  T[i,k] · β · (Σ_j T[j,k] · P[j]) / (Σ_j T[j,k] · N[j])
```

This is the standard time-share form of a commuting model. It needs no change to the timestep and works under every solver. It does not represent the time of day; a model that needs daytime and nighttime mixing uses a sub-daily `dt` with separate layers.

The time-share form is the limit in which commuters come back at once. Commuters who come back at a finite rate (Keeling and Rohani's commuter model 7.3, whose `X[i,j]` are the residents of `i` now in `j`) are written with the general pieces: home is a dimension of its own, where people are now is the spatial one, and they move between its levels at rates by home. Results by home count people by home, wherever they are (`res.n_I.by('home')`), as the book's `Y_i = Σ_j Y_ij` does:

<!-- fragment -->
```python
l, r = 0.05, 0.45 # Leaving home, and coming back, per day: 10% of time away
sir.stratify(home=['a', 'b'])
sir.stratify(patch=['a', 'b'], kind='space', split=ss.by([[0.72, 0.08], [0.02, 0.18]], dims=['home', 'patch']), # Each home's people in each place for their share of time there
             out=ss.transfer(patch='a -> b', rate=ss.by(home=[ss.perday(l), ss.perday(r)])),
             back=ss.transfer(patch='b -> a', rate=ss.by(home=[ss.perday(r), ss.perday(l)])))
```

Under `visit`, people also keep their home patch and spend a share `away` of their time elsewhere, but visitors are only exposed: they are exposed to the residents of the patches they go to, and do not expose anyone there (as for short trips, where the visitor's own infectiousness away from home is small).

`kind='visit'` is commuting in which people are exposed to the residents of the places they go to but expose no one there. It is kept because flepiMoP's force of infection has this form, and its importer matches flepiMoP's runs to 1e-6.

A third form moves neither people nor time: `kind='force'` moves the force of infection itself, as LASER does. With `M[i,k]` the movement from `i` to `k` (scaled by `away=`), each patch keeps `1 − Σ_k M[i,k]` of its own force of infection and receives `M[k,i]` of each other patch's, unweighted by the patches' sizes. It is the same as a contact matrix over the patches, `ss.contacts(patch=M.T + np.diag(1 - M.sum(axis=1)))`, but kept as its own form because as agents a contact matrix makes a mixing pool for each pair of patches, which is far slower for many patches (LASER's England and Wales, 954 patches: 34 s to build and 2.9 s a step, against 1.3 s and 0.35 s). It is less mechanistic than commuting, and is kept as the form LASER models are written in (a model written for Starsim uses `commute`), and the same gravity matrix gives different epidemics under the two forms (in LASER's hub-and-spoke example, 88% infected against 80%).

`away=` gives every patch the same share. To keep the differences between patches that the movement model gives, or to cap any patch's share (as LASER scales its gravity coupling), scale the matrix in NumPy and give it as `matrix=` with no `away=`.

Commuting couples patches without mixing their populations; migration does both. If `kind=` is not given, a gravity model or matrix is treated as `commute` and a movement register as `migrate`, and `describe()` marks the choice as a guess.

Travelers in transit (in neither their home patch nor their destination) are out of scope for v1. A model that needs them can add a transit patch by hand.

## Geography

<!-- fragment -->
```python
ss.geo('Kenya', level=1)                       # its 47 counties, bundled
ss.geo('Kenya', level=1, places=['Nairobi', 'Kiambu'])
ss.geo(['Kenya', 'Uganda'], level=1)           # several countries, named 'Kenya:Busia' and so on
ss.geo('USA', level=2, year=2015)              # US counties, populations for 2015: fetched
ss.geo('districts.csv')                        # the user's own: name, pop, and lat and lon (or x and y)
ss.geo(names=['a', 'b'], pop=[1e4, 2e4], x=[0, 1], y=[0, 0])    # or given directly (or with distances=)
```

A geography supplies names, populations, centroids, distances, and which places share a border (`geo.adjacency`). Levels are numbers, as the boundary sources number them (0 the country, 1 its provinces, states, or counties, 2 the next), since what a level is called differs from country to country. The source used is recorded in `describe()`, in the model spec, and in `sim.hash()`. Geographic data follows the same policy as the rest of the named data ([04-parameters.md](04-parameters.md#named-data)): what is small and common is bundled, the rest is fetched on demand and cached, each data set is adapted once with its provenance recorded, and nothing depends on another framework's data package. Every country's first two levels are bundled with `starsim_data` (`pip install starsim[data]`); finer levels and other years are fetched (which needs `pip install starsim_data[spatial]`).

## Rejected

- **Making the modeler write `/N`.** It is one of the most commonly mis-written parts of a model. Writing counts literally remains available for the cases that need it.
- **A special route kind for environmental or vector-borne transmission.** Counts in rates plus ordinary states that are not people cover both with the ordinary transition syntax.
- **A separate module class for quantities that are not people** (an earlier draft's `ss.Compartments`). The reservoir's states use the same syntax as the disease's, so a second class added a name without adding anything to say.
- **A degree-grouped ODE approximation** for networks with strong variation in the number of contacts. Homogeneous mixing with the same mean is a poor approximation for these, and the standard improvement groups people by degree. It is not worth building: the agent-based run is one keyword away and is exact.
- **A separate `ss.Homogeneous` class, or a new keyword for mixing.** Homogeneous mixing is the default, and `networks=` takes networks and mixing pools today.
- **Depending on other frameworks' packages for mobility models or data.** The generators and the data are ours.
- **Refusing to run a network model as an ODE.** The standard approximation is used and reported.
- **Frequency versus density as different transition kinds.** The rate says which: a bare state for exposure through the routes, a count for density dependence.
- **A `dependence=` option on the mixing, with a reference population `n_ref`.** Writing the count (`beta * I.n`) says the same thing in the rate, where it can be read, and is what the importers produce.
- **Requiring an edge list.** Accept one; never require one.
- **Sub-step scheduling for commuting.** The time-share form gives the same coupling without it.
- **Opaque select and shift matrices for movement.** A register with named columns says the same thing.
