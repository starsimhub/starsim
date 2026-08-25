# Population and mixing

Finding 4 of the [landscape review](../approaches/README.md#cross-cutting-findings) calls this "the hardest unsolved problem in this review, and where the project has to do original work." This document proposes the solution.

## Recommendation

1. **Factor transmission into two independent facts: how infectious the pathogen is per contact, and how contact happens.** The disease declares the first; the route declares the second.
2. **Keep Starsim's `Route` abstraction and extend it so the compartmental limit is exact rather than an approximation with homogeneous agents.**
3. **The force of infection is the framework's job, always.** No model text should contain `/N`.
4. **Name the denominator convention**, because frequency- and density-dependent are genuinely different models and the difference is invisible in most frameworks.
5. **Structure is specifiable by its generating model, not only its realization.** "Mean degree 0.7, partnerships lasting 50 days" is what modelers have; an edge list is not.
6. **Keep partnership and act distinct.** Duration-of-relationship and acts-per-unit-time have different data sources.

## Why: three independent arrivals at route abstraction

[Starsim](../approaches/starsim.md) puts `ss.Network` (edge-based) and `ss.MixingPool` (aggregate, contact-matrix) behind one `ss.Route`, so "the *same disease module* can be run as an individual-network ABM or as a mixing-matrix metapopulation model without editing the disease". [epiworld](../approaches/epiworld.md) has `ModelSEIR` / `ModelSEIRCONN` / `ModelSEIRMixing` over one engine. [epipack](../approaches/epipack.md) has network versus well-mixed.

Three independent arrivals is strong evidence that this is the correct seam. Starsim's version is the most developed and is the right starting point.

But every version is partial, and the reviews say where:

- **Starsim**: "the disease module is still written as agent code, so the compartmental limit is reached by making the population homogeneous rather than by solving equations… paradigm conversion is therefore one-directional."
- **epipack**: "`StochasticEpiModel` uses a different process vocabulary from `EpiModel`. The network paradigm is adjacent to the others, not the same specification reinterpreted." Its review is explicit that this is because "edge-mediated transmission genuinely is different from mass-action — the rate is per-edge rather than per-population-pair."

## The proposal: β is per contact, and the route supplies the contacts

The unification follows from one decision about what β means.

In most compartmental frameworks, β is a *composite*: transmission probability × contact rate, fused into one number. That fusion is exactly what makes mass-action and edge-based transmission look like different mathematics. Unfuse it, and they are the same declaration.

[EpiModel](../approaches/epimodel.md) already separates them and its review flags this as a strength worth keeping: `inf.prob` is a per-act transmission probability and `act.rate` is acts per partnership per time step — "an unusually explicit separation of *relationship* from *exposure*", and "collapsing them into a single 'contact rate' is lossy for anything sexually transmitted".

So:

> **`beta` is the probability (or hazard) of transmission per contact between an infectious and a susceptible individual. It is a property of the pathogen and the pair. The route supplies who contacts whom, how often.**

The force of infection is then always the same expression, and the route is the only thing that varies:

λᵢ = Σⱼ cᵢⱼ · β · Iⱼ / Nⱼ

| Route | cᵢⱼ | Backend |
|---|---|---|
| `ss.Homogeneous(n_contacts=c)` | `c` for all pairs | Exact in ODE, CTMC, and ABM |
| `ss.MixingPool(matrix=M)` | `M[i,j]` between groups | Exact in ODE, CTMC, and ABM |
| `ss.RandomNet(n_contacts=c)` | Realized edges, drawn each step | ABM; mean-field limit is `Homogeneous(c)` |
| `ss.SexualNet(...)` | Persisting partnerships × acts per partnership | ABM only; no exact compartmental limit |

### The declaration

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),   # per contact
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

ss.Sim(sir, mixing=ss.Homogeneous(n_contacts=10)).run(method='ode')  # exact ODE
ss.Sim(sir, mixing=ss.contacts('Kenya')).run(method='ode')           # age-structured ODE
ss.Sim(sir, mixing=ss.RandomNet(n_contacts=10)).run()                # ABM, same disease
ss.Sim(sir, mixing=ss.SexualNet()).run()                             # ABM only
```

**The disease is byte-identical across all four.** That is the claim, and it is testable: the first and third must agree in distribution as `n_agents → ∞`, and [EoN](../approaches/eon.md)'s analytic hierarchy is the tool that says by how much they do not.

The one honest complication: `β·c` is what is identifiable from incidence data, not β and c separately. The language must let you fit the product, and [calibration.md](calibration.md) covers it. This is a cost of the factoring and it is worth paying, because β is the quantity that transfers between settings and `β·c` is not.

### Compare: what the frameworks make you write

**MetaCast** ([approaches/metacast.md](../approaches/metacast.md)) computes the force of infection for you — the right instinct — "implemented in the most fragile possible way, through string concatenation of parameter names (`'beta_[high,vaccinated]'`)". It also has `foi_population_focus` (`None` / `'i'` / `'j'`), which selects the denominator. That option is a real modeling decision that almost nothing else in the review exposes.

**EMULSION** ([approaches/emulsion.md](../approaches/emulsion.md)) makes the modeler write it:

```yaml
force_of_infection:
  value: 'transmission_I * total_I / total_population'
```

and its review notes: "nothing checks it, and nothing knows that this transition is an infection rather than a progression. That is the same missing-`/N` class of bug camdl catches."

**summer2** ([approaches/summer2.md](../approaches/summer2.md)) makes it a choice of function name: `add_infection_frequency_flow` versus `add_infection_density_flow` — "a coarse but effective form of typing".

**Lingua franca** — the convention is a route property, named and defaulted:

```python
ss.Homogeneous(n_contacts=10)                        # frequency-dependent (default)
ss.Homogeneous(n_contacts=10, dependence='density')  # density-dependent
```

Frequency-dependent is the default because it is right for directly transmitted human infections, which is what most users are modeling. The summary prints which one is in force, because a model that silently chose the other is wrong in a way that only shows up when population size changes.

### Infectious sources come from the states

The route needs to know who is infectious and how infectious. That is already declared:

```python
ss.State('I',  infectious=True)
ss.State('Ia', infectious=True, rel_trans=0.5)   # asymptomatic, half as infectious
```

[MetaCast](../approaches/metacast.md) declares `infected_states`, `infectious_states`, and `symptomatic_states` as separate sets, with `asymptomatic_transmission_modifier` applying to states in the second but not the third. Its review: "That is real epidemiological semantics encoded in a set relationship, which is the kind of thing a language should be able to say."

Our version says it more directly — `rel_trans` on the state, [EpiHiper](../approaches/epihiper.md)'s `infectivity` per state — without needing three parallel sets. `infected` (any non-susceptible disease state) and `infectious` (transmits) stay distinguishable because `infectious=` is a flag, not an inference.

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

**LASER** ([approaches/laser.md](../approaches/laser.md)) ships `gravity`, `competing_destinations`, `stouffer`, and `radiation` — "the best spatial-interaction vocabulary in the review".

**EoN**, **epiworld**, and **EpiHiper** all take a network as an *input file* and have no specification vocabulary at all. From the EpiModel review: "A lingua franca that only accepts an edgelist cannot express 'a network where mean degree is 0.7 and partnerships last 50 days' — and that description, not the edgelist, is what modelers actually have."

**Lingua franca** — a route is described, not supplied:

```python
ss.SexualNet(mean_degree=0.7, dur=ss.days(50), concurrency=0.22)
ss.Mobility(ss.gravity(k=0.5, a=1.0, b=1.0, c=2.0), distances=d)
ss.contacts('Kenya', source='prem_2021')
ss.Network(edges=my_edgelist)          # still allowed; the realization, not the description
```

and the diagnostic step EpiModel treats as mandatory becomes one call:

```python
net.check()   # simulate the route alone; do the target statistics recover?
```

### Partnership and act

```python
ss.SexualNet(
    mean_degree = 0.7,               # how many partners
    dur         = ss.days(50),       # how long partnerships last
    acts        = ss.freqperweek(2), # how often, within a partnership
)
```

Three numbers with three different data sources. Collapsing them into "contact rate" is lossy for anything sexually transmitted, and it is a distinction that Starsim's `ss.Network` edges (with `beta` and `dur`) already partly carries.

### Layers

Household, school, work, community — [Epydemix](../approaches/epydemix.md) has them, Covasim had them, [MEmilio](../approaches/memilio.md) has them via contact-matrix dampings, and they are what school-closure interventions act on. A list of routes is a layered population, and each layer is separately targetable:

```python
ss.Sim(sir, mixing=[ss.contacts('Kenya', layer='home'),
                    ss.contacts('Kenya', layer='school'),
                    ss.contacts('Kenya', layer='work')])
```

Then `ss.Effect(..., 'school.contacts', multiply=0.2)` is a school closure ([interventions.md](interventions.md)).

## Trade-offs

- **Unfusing β from contact rate makes β unidentifiable on its own.** Real, and the mitigation is that the fitted quantity can be the product. Note that this is not a new problem: it is the existing problem, made visible.
- **`ss.SexualNet` has no exact compartmental limit** — persisting concurrent partnerships are not mass-action at any population size. The compiler must refuse rather than approximate. See [paradigm-conversion.md](paradigm-conversion.md).
- **A route library is a maintenance surface.** Every named generator (gravity, radiation, ERGM-fitted, contact matrices for 400 locations) is code and data we own. Prefer wrapping the existing implementations — LASER's migration module, `socialmixr`/`epydemix-data` contact matrices, `ergm` for fitted networks — over reimplementing them.
- **A contact matrix and a network are not interchangeable even when both are "available".** The matrix is the mean of a distribution whose variance matters for threshold behavior. The route names which one you have.

## Rejected

- **Making the modeler write the force of infection** (EMULSION, MetaCast's subpop model, [`individual`](../approaches/individual.md)'s `foi <- beta * I / N`, [odin](../approaches/odin.md)). This is the single most commonly mis-written line in epidemiological modeling.
- **Frequency versus density as different transition kinds** (summer2). It is a route property; putting it on the transition means every transition repeats it.
- **Networks as inputs only** (EoN, epiworld, EpiHiper, LASER). Accept an edge list; do not require one.
- **A separate transmission vocabulary for networks** (epipack's `set_link_transmission_processes` versus `set_processes`). This is precisely the seam we are claiming not to need.
- **Node-level force of infection as the only spatial mechanism** (LASER). Correct for eradication-scale spatial models and not general.

## Open questions

- **Does the β-per-contact factoring survive environmental transmission?** Cholera, and anything with a reservoir, has no contact to be per. Probably a fourth route kind (`ss.Environmental`) with its own λ form, coupled to [SimInf](../approaches/siminf.md)-style continuous node state. This is the most likely place the unification breaks and it should be tested early.
- **Vector-borne transmission** is two-host and does not fit λᵢ = Σⱼ cᵢⱼ β Iⱼ/Nⱼ without a second population. Does it need a route kind, a second disease, or a second `People`?
- **What is the exact mean-field limit of `ss.RandomNet`?** For a Poisson-degree random network regenerated each step it is `Homogeneous`; for a static network with degree heterogeneity it is not, and EoN's ladder names the corrections. The compiler should know which case it is in.
- **Dose response.** [EpiHiper](../approaches/epihiper.md)'s review notes its transmission form is fixed and "anything else — dose response, within-host viral load, density dependence — is not expressible". Ours has the same limit unless β can be a function of exposure.
