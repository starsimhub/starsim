# Population and mixing

Finding 4 of the [landscape review](../approaches/README.md#cross-cutting-findings) calls this "the hardest unsolved problem in this review, and where the project has to do original work." This document proposes the solution.

## Recommendation

1. **Factor transmission into two facts: how infectious the pathogen is, and how contact happens.** The disease states the first; the networks state the second. Where infectiousness itself depends on the route (HIV through sex or through shared needles, male to female or female to male), the disease gives one `beta` per network.
2. **Keep Starsim's route abstraction and its keyword.** `networks=` accepts individual-level networks and mixing pools alike.
3. **Homogeneous mixing is the default.** With no network given, `S >> I @ beta * I` is the textbook `beta * S * I / N`, and `beta` is the classical transmission rate. The user does not write `/N`.
4. **A bare state in a rate means contact with people in that state, through the model's networks.** `I.n` means the number of people in it. The first covers person-to-person transmission; the second covers everything else.
5. **Name the denominator convention**, because frequency- and density-dependent transmission are different models and the difference is invisible in most frameworks.
6. **Let structure be specified by its generating model, not only its realization.** "Mean degree 0.7, partnerships lasting 50 days" is what modelers have; an edge list is not.
7. **Keep partnership and act distinct.** Duration of relationship and acts per unit time have different data sources.
8. **A network model can be run as a compartmental model when the user asks for it**, using the standard approximation and saying so.

## Why: three independent arrivals at route abstraction

[Starsim](../approaches/starsim.md) puts `ss.Network` (edge-based) and `ss.MixingPool` (aggregate, contact-matrix) behind one `ss.Route`, so "the *same disease module* can be run as an individual-network ABM or as a mixing-matrix metapopulation model without editing the disease". [epiworld](../approaches/epiworld.md) has `ModelSEIR` / `ModelSEIRCONN` / `ModelSEIRMixing` over one engine. [epipack](../approaches/epipack.md) has network versus well-mixed.

Three independent arrivals is strong evidence that this is the right place to divide things. Starsim's version is the most developed and is the starting point.

But every version is partial, and the reviews say where:

- **Starsim**: "the disease module is still written as agent code, so the compartmental limit is reached by making the population homogeneous rather than by solving equations… paradigm conversion is therefore one-directional."
- **epipack**: "`StochasticEpiModel` uses a different process vocabulary from `EpiModel`. The network paradigm is adjacent to the others, not the same specification reinterpreted." Its review is explicit that this is because "edge-mediated transmission genuinely is different from mass-action — the rate is per-edge rather than per-population-pair."

## The proposal

### What the user writes

```python
beta = 0.3                       # per day
S, I, R   = ss.states(3)
infection = S >> I @ beta * I    # "S goes to I at rate beta times I"
recovery  = I >> R @ ss.days(10)
sir = ss.Disease(infection, recovery)
```

or, in the short form, `ss.transmission('S -> I', beta=0.3)`. Either way the model reads like the equation, and there is no `/N`.

The two forms line up like this: the word `transmission` in the short form is what `* I` is in the operator form. `S >> I @ beta` on its own is legal and means something else: a constant hazard of infection that does not depend on anyone being infectious (an importation rate, say), exactly as `I >> R @ gamma` is a constant hazard of recovery. So no operator form with a bare `beta` means transmission. The infectious state has to appear, and that is also what makes the source of infection explicit. Several infectious states are a sum of terms, each with its own coefficient: `S >> I @ beta * I + beta_vax * I_vax`.

**A bare state inside a rate means "contact with people in that state, through the model's networks".** The library works out what that contact is from how the population mixes:

| What is given | Who meets whom | What `beta` is |
|---|---|---|
| Nothing | Everyone, equally; one contact per unit time | The classical transmission rate: `beta * I` is `beta * I / N` |
| `ss.MixingPool(n_contacts=10)` | Everyone, equally; 10 contacts per unit time | Per contact |
| `ss.contacts('Kenya')` (a set of `ss.MixingPools`) | Groups, by a contact matrix | Per contact |
| `ss.RandomNet(n_contacts=10)` | Individual edges, redrawn each step | Per contact |
| `ss.MFNet(...)` and other sexual networks | Persisting partnerships, with acts within each | Per act |

The first row and the others are one definition, not two: with no network there is one contact per unit time, so "per contact" and "per unit time" are the same number. A textbook model works with textbook values; adding a network gives `beta` the meaning the network needs.

There is no `ss.Homogeneous` class. Homogeneous mixing is what you get by default, and `ss.MixingPool(n_contacts=10)` (Starsim's class, with no groups) is how to set the contact rate.

Underneath, the force of infection is always the same expression, and only the contact pattern varies:

λᵢ = Σⱼ cᵢⱼ · β · Iⱼ / Nⱼ

```python
ss.Sim(sir, unit='day').run()                                        # homogeneous; runs as an ODE
ss.Sim(sir, networks=ss.MixingPool(n_contacts=10), unit='day').run() # homogeneous, 10 contacts a day
ss.Sim(sir, stratify=dict(age=[0, 15, 65]), networks=ss.contacts('Kenya'), unit='day').run()   # age-structured
ss.Sim(sir, networks=ss.RandomNet(n_contacts=10), unit='day').run()  # agents on a random network
```

The disease is the same object in all four. `solver='auto'` runs the first three as ODEs and the fourth as agents, because an individual-level network is something only agents have, and prints which it chose.

[EpiModel](../approaches/epimodel.md) already separates infectiousness from contact, and its review flags this as a strength worth keeping: `inf.prob` is a per-act transmission probability and `act.rate` is acts per partnership per time step, "an unusually explicit separation of *relationship* from *exposure*", and "collapsing them into a single 'contact rate' is lossy for anything sexually transmitted".

### Compare: what the frameworks make you write

**MetaCast** ([approaches/metacast.md](../approaches/metacast.md)) computes the force of infection for you, which is the right instinct, "implemented in the most fragile possible way, through string concatenation of parameter names (`'beta_[high,vaccinated]'`)". It also has `foi_population_focus` (`None` / `'i'` / `'j'`), which selects the denominator. That option is a real modeling decision that almost nothing else in the review exposes.

**EMULSION** ([approaches/emulsion.md](../approaches/emulsion.md)) makes the modeler write it:

```yaml
force_of_infection:
  value: 'transmission_I * total_I / total_population'
```

and its review notes: "nothing checks it, and nothing knows that this transition is an infection rather than a progression. That is the same missing-`/N` class of bug camdl catches."

**summer2** ([approaches/summer2.md](../approaches/summer2.md)) makes it a choice of function name: `add_infection_frequency_flow` versus `add_infection_density_flow`, "a coarse but effective form of typing".

**Lingua franca**: the convention is an option on the mixing, with a default:

```python
ss.MixingPool(n_contacts=10)                         # frequency-dependent (default)
ss.MixingPool(n_contacts=10, dependence='density')   # density-dependent
```

Frequency-dependent is the default because it is right for directly transmitted human infections, which is what most users are modeling. `describe()` prints which one is in use, because a model that silently chose the other is wrong in a way that only shows up when population size changes.

### When infectiousness depends on the route

The split between pathogen and contact is not always clean. HIV is far more transmissible through a shared needle than through sex, and more from men to women than the reverse. Those are still facts about the pathogen, so they stay on the disease, as one `beta` per network:

```python
infection = ss.transmission('S -> I', beta=dict(mf=[0.008, 0.004], needles=0.03))
sim = ss.Sim(hiv, networks=[ss.MFNet(), needles])
```

This is Starsim's existing convention. The keys are network names, and a pair is the two directions of a directed network (here male to female, then female to male). A single number applies to every network. The operator form takes the same dict as its `beta`.

### Transmission that is not person to person

Cholera has no contact for `beta` to be per, and malaria has two host populations. Neither needs a new kind of route. **`I.n` is the plain number of people in a state**, and a rate may be any expression of counts and parameters, used exactly as written.

Environmental transmission, where [SimInf](../approaches/siminf.md) keeps the reservoir in a per-node continuous variable updated by a C function:

```python
beta, K, xi = 0.5, 1e5, 10            # per day; half-saturation concentration; shedding per person per day
S, I, R, W = ss.states(4)            # W is bacteria in the water

cholera = ss.Disease(
    infection = S >> I @ beta * W.n / (W.n + K),
    recovery  = I >> R @ ss.days(5),
    shedding  = ss.null >> W @ xi * I.n,
    decay     = W >> ss.null @ ss.days(30),
)
sim = ss.Sim(cholera, total_pop=100_000, unit='day')
```

`ss.null` is outside the model: `ss.null >> W` is an inflow and `W >> ss.null` an outflow.

Vector-borne transmission (Ross-Macdonald), with `a` the biting rate and `b`, `c` the two transmission probabilities:

```python
a, b, c = 0.3, 0.5, 0.5
Sh, Eh, Ih, Rh = ss.states(4)
Sv, Ev, Iv     = ss.states(3)
Nh = (Sh + Eh + Ih + Rh).n

malaria = ss.Disease(
    infection  = Sh >> Eh @ a * b * Iv.n / Nh,
    latency    = Eh >> Ih @ ss.days(12),
    recovery   = Ih >> Rh @ ss.days(100),
    emergence  = ss.null >> Sv @ ss.perday(5_000),   # an inflow rate is a total per unit time, not a per-capita hazard
    biting     = Sv >> Ev @ a * c * Ih.n / Nh,
    incubation = Ev >> Iv @ ss.days(10),
    mortality  = (Sv | Ev | Iv) >> ss.null @ ss.days(10),
)
sim = ss.Sim(malaria, n_agents=5_000, unit='day')
```

`W` and the mosquito states are ordinary states, written like any other, in the same disease. In a compartmental run there is nothing more to say: every state, human or mosquito, is a number. The one thing the library needs to know is which states are people, because in an agent-based run those become agents and the others stay numbers, and because `total_pop`, age, and sex apply only to people. It reads that from the transitions: none connects the mosquito states to the human ones, so the mosquito states are not people. `describe()` prints the guess, and `ss.State(people=False)` says it outright. In the malaria example `n_agents=` therefore makes the humans agents while the mosquitoes stay three numbers. See [model-structure.md](model-structure.md) and [agent-based.md](agent-based.md).

### `I` and `I.n`

There are two ways to mention a state in a rate, and they answer different questions.

- **`I.n` is how many people are in `I`.** It is a number, the same one that appears in the results as `n_I`. `xi * I.n` says each infectious person sheds `xi`, and nothing more is done to it.
- **A bare `I` is the chance that a contact is with someone in `I`.** With no network that is `I.n / N`, so `beta * I` and `beta * I.n / N` are the same model. With a contact matrix or a network it is the same quantity worked out for each person from whom they actually meet. That cannot be written as one formula, which is why the bare form exists.

So write the bare state when people catch the infection from people, and `.n` when the formula needs the number itself (a reservoir, a vector, a threshold). `describe()` prints every rate with the bare states expanded, for example `infection: S -> I at 0.3 × I.n / N  [homogeneous mixing]`, so what a rate resolved to can always be checked. A bare state that is not people (`beta * W`) is an error that suggests `W.n`, since there is no contact with a reservoir to work out.

### Infectious sources come from the states

`S >> I @ beta * I` names the source of infection directly. Where several states are infectious, they are a sum, and relative infectiousness is a property of the state:

```python
I  = ss.State(infectious=True)
Ia = ss.State(infectious=True, rel_trans=0.5)   # asymptomatic, half as infectious
```

[MetaCast](../approaches/metacast.md) declares `infected_states`, `infectious_states`, and `symptomatic_states` as separate sets, with `asymptomatic_transmission_modifier` applying to states in the second but not the third. Its review: "That is real epidemiological semantics encoded in a set relationship, which is the kind of thing a language should be able to say."

Our version says it more directly, with `rel_trans` on the state ([EpiHiper](../approaches/epihiper.md)'s `infectivity` per state) and no parallel sets.

### Specify structure by its generating model

Finding 11 of the [landscape review](../approaches/README.md#cross-cutting-findings). The evidence:

**EpiModel** ([approaches/epimodel.md](../approaches/epimodel.md)) fits a network from target statistics and validates it before any disease runs on it:

```r
formation    <- ~edges + concurrent + degrange(from = 4)
target.stats <- c(175, 110, 0)
coef.diss    <- dissolution_coefs(dissolution = ~offset(edges), duration = 50)
est          <- netest(nw, formation, target.stats, coef.diss)
dx           <- netdx(est, nsims = 5, nsteps = 500)   # diagnostics are treated as mandatory
```

**LASER** ([approaches/laser.md](../approaches/laser.md)) ships `gravity`, `competing_destinations`, `stouffer`, and `radiation`, "the best spatial-interaction vocabulary in the review".

**EoN**, **epiworld**, and **EpiHiper** all take a network as an *input file* and have no specification vocabulary at all. From the EpiModel review: "A lingua franca that only accepts an edgelist cannot express 'a network where mean degree is 0.7 and partnerships last 50 days' — and that description, not the edgelist, is what modelers actually have."

**Lingua franca**: a network is described, not supplied:

```python
ss.MFNet(n_contacts=ss.poisson(0.7), duration=ss.days(50), concurrency=0.22)
ss.Mobility(ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0), distances=d)
ss.contacts('Kenya', source='prem2021')
ss.StaticNet(edges=my_edgelist)        # still allowed: the realization, not the description
```

(`n_contacts` and `concurrency` are proposed additions to Starsim's `ss.MFNet`, which today takes `duration`, `acts`, `participation`, and `debut`. `n_contacts` is the name Starsim's other networks already use.) The diagnostic step EpiModel treats as mandatory becomes one call:

```python
net.check()   # simulate the network alone; do the target statistics recover?
```

### Partnership and act

```python
ss.MFNet(
    n_contacts  = ss.poisson(0.7),   # how many partners at a time
    duration    = ss.days(50),       # how long partnerships last
    acts        = ss.freqperweek(2), # how often, within a partnership
)
```

Three numbers with three different data sources. Collapsing them into "contact rate" is lossy for anything sexually transmitted. Starsim's sexual networks already carry `duration` and `acts` separately.

"0.7 partners" on its own is ambiguous: it could be one partner 70% of the time, or a Poisson number of partners with a mean of 0.7, and those are different networks. So `n_contacts` takes a distribution, as it does on Starsim's other networks (`ss.bernoulli(0.7)` for the first reading). A bare number is read as the mean of a Poisson, and `describe()` says so.

### Sexual networks in a compartmental run

Persisting, concurrent partnerships are not mass action at any population size, so a model with `ss.MFNet` runs as agents under `solver='auto'`. There are also plenty of valid ODE models of sexually transmitted infections, and they describe the same population more coarsely: people are grouped by sex and risk, and groups form new partnerships with each other at given rates. When the user asks for a compartmental run, the library builds that model and says so:

```text
>>> ss.Sim(hiv, networks=ss.MFNet(), solver='ode').run()
note: sexual network approximated by partner-change rates between sex/risk groups
```

The partner-change rates are derived from the network's own parameters (number of partners, duration, acts), so the user does not supply them twice. The result is a different, coarser model, not an invalid one: it loses concurrency and the persistence of individual partnerships, which matter for early growth and for anything that depends on who exactly is connected to whom. `approx=False` turns the note into an error for users who want only exact runs, and `approx=True` silences it. The general policy is in [paradigm-conversion.md](paradigm-conversion.md).

The same applies to `ss.RandomNet(n_contacts=10)`: asked for as an ODE, it is approximated by homogeneous mixing with the same mean number of contacts.

### Layers

Household, school, work, community: [Epydemix](../approaches/epydemix.md) has them, Covasim had them, [MEmilio](../approaches/memilio.md) has them via contact-matrix dampings, and they are what school-closure interventions act on. Starsim has them as of v3.7.2. A list of networks is a layered population, and each layer can be changed separately:

```python
home   = ss.contacts('Kenya', layer='home')
school = ss.contacts('Kenya', layer='school')
work   = ss.contacts('Kenya', layer='work')

sim = ss.Sim(sir, networks=[home, school, work], unit='day',
    stratify      = dict(age=[0, 5, 18, 65]),
    interventions = ss.modify(school.pars.n_contacts, multiply=0.2, start='2020-03-15', stop='2020-06-01'),
)
```

The `ss.modify` line is a school closure: school contacts are multiplied by 0.2 between the two dates and return to their original value afterwards ([interventions.md](interventions.md), [composition-and-effects.md](composition-and-effects.md)).

Age is a property of the people and not of one disease, so here it is the sim that is stratified (`stratify=` on the sim, or `sim.stratify(age=...)`), and every disease in it gets the same age groups ([stratification.md](stratification.md)).

The contact matrices come with their own age groups (five-year bands, for the Prem matrices), which need not be the model's. In an agent-based run that does not matter: each agent has an exact age and is counted in whichever band the matrix uses. In a compartmental run the matrix is aggregated to the model's groups, weighted by population, which is the standard calculation. Where a model group cuts through a matrix band, the band is split in proportion to population, and a note says so.

## Trade-offs

- **`beta` means "per contact" once a network is given, and the contact rate is then a second number.** Incidence data identifies only their product. This is not a new problem; it is the existing one made visible, and [calibration.md](calibration.md) covers fitting it.
- **`I` and `I.n` look similar and mean different things.** With no network they differ only by the `/N`; with one, the bare state is something only the library can compute. `describe()` prints the resolved rate for every transition, including the denominator, so the difference is visible. A longer spelling for the bare state (`I.contact`, say) would be clearer and would stop `S >> I @ beta * I` reading like the textbook equation.
- **An approximated network model is a different model.** The one-line note says what was done; it does not say how much the answer changed. Running both and comparing is the only way to know.
- **A network library is a maintenance surface.** Every named generator (gravity, radiation, contact matrices for 180 countries) is code and data we own. That is accepted: the generators live in the library and the data in [`starsim_data`](https://github.com/starsimhub/starsim_data), which bundles what is small and common (national demographics, the Prem and Mistry contact matrices) and fetches the rest on demand with a cache. We do not depend on other frameworks' packages (LASER's migration module, `epydemix-data`, `socialmixr`) for them: an outside dependency is harder to keep working than code we control, and adapting a data set once, with its provenance recorded, is cheap.
- **A contact matrix and a network are not interchangeable even when both are available.** The matrix is the mean of a distribution whose variance matters for threshold behavior.

## Rejected

- **Making the modeler write the denominator** (EMULSION, MetaCast's subpop model, [`individual`](../approaches/individual.md)'s `foi <- beta * I / N`, [odin](../approaches/odin.md)). It is the most commonly mis-written line in epidemiological modeling. Writing it out stays possible with `I.n`.
- **A separate `ss.Homogeneous` class.** It is the default, and `ss.MixingPool` already exists.
- **A new keyword for mixing.** Starsim's `networks=` takes networks and mixing pools today.
- **Special route kinds for environmental and vector-borne transmission.** Count-based rates and ordinary states that are not people cover them with the ordinary transition syntax.
- **A degree-grouped ODE approximation for networks with strong degree heterogeneity.** Homogeneous mixing with the same mean is a poor approximation for these, and the standard improvement groups people by degree. It is not worth building: the agent-based run is one keyword away and is exact.
- **Refusing to run a network model as an ODE.** The approximation is standard and widely published; it is used when asked for and reported.
- **Frequency versus density as different transition kinds** (summer2). It is a property of the mixing; putting it on the transition means every transition repeats it.
- **Networks as inputs only** (EoN, epiworld, EpiHiper, LASER). Accept an edge list; do not require one.
- **A separate transmission vocabulary for networks** (epipack's `set_link_transmission_processes` versus `set_processes`).
- **Node-level force of infection as the only spatial mechanism** (LASER). Correct for eradication-scale spatial models and not general.

## Open questions

- **How are partner-change rates derived from a network's parameters?** For a simple network (number of partners, duration) there is a standard formula. For a network with age mixing, concurrency, and several partnership types, the grouping and the rates need choosing, and the choice should be printed.
- **Dose response.** [EpiHiper](../approaches/epihiper.md)'s review notes its transmission form is fixed and "anything else — dose response, within-host viral load, density dependence — is not expressible". Here a rate can be a function ([parameters-and-distributions.md](parameters-and-distributions.md)), which covers it in principle; a worked example with viral load is needed to confirm.
