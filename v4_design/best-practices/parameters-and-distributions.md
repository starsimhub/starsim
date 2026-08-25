# Parameters and distributions

## Recommendation

1. **Declare parameters; refuse unknown names.** A misspelled parameter should be an error at build time, not a silent behavioral difference.
2. **A parameter may be a scalar, a typed quantity, a distribution, a time series, or a callable — and the slot decides which are legal.**
3. **Keep values with the model by default, and make it possible to separate them.** camdl is right that structural identity should be separable from values; it is wrong that this must be mandatory.
4. **Declare what a module needs, as data.** epymorph's requirement contract is the most transferable idea in the review after typed time.
5. **Record where a value came from.** One optional field, `source=`, answers the question every reviewer asks.
6. **Parameters have shapes, and the shape is checked.**

## Why

### Declaration catches typos; no framework without it does

[EpiModel](../approaches/epimodel.md): "`param.net(inf.prob = 0.4, act.rate = 2, my.new.param = 0.7)` is legal… There is no parameter schema and no declaration step; a misspelled parameter name is not an error until something looks for it." Its own review lists "No declaration step of any kind — parameters, attributes, and outputs all spring into existence on first use. Typos become silent behavioral differences" as a limitation.

Against that: [MEmilio](../approaches/memilio.md) gets compile-time checking from C++ tag types (`parameters.set<TimeInfected>(2)` — "the strongest parameter-name checking in the review"); [camdl](../approaches/camdl.md) gets it from its compiler; **Starsim gets a runtime version from `define_pars` locking**, where `self.foo = 3` raises unless `foo` was declared. Starsim's mechanism is already the right one for a Python host. Keep it.

### Where values live

[camdl](../approaches/camdl.md) takes the strongest position in the review:

> **parameter values are never declared in the `parameters` block**. That block carries names, kinds, dimensions, and optional priors only. Values come from a `params.toml`, a CLI `--param`, or a named scenario's `set = { ... }`.

with a factored run identity (model hash, params hash, config hash, scenario hash, fit hash) so that "a reviewer asking 'is this a structural change or a parameter sweep?' reads the level hashes rather than diffing files."

The mechanism is excellent and the *mandatory separation* is wrong for our audience. An epidemiologist writing an SIR should not need two files. The resolution: **values default to living inline, and the split is available**, with the hashes computed either way.

### Requirements as data

[epymorph](../approaches/epymorph.md)'s `AttributeDef` is the standout:

```python
requirements = [
    AttributeDef("beta", type=float, shape=Shapes.TxN, comment="infectivity"),
]
```

From its review: "A model states its data interface *as data*, so it can be checked before anything runs… Data provenance is explicit… Shape checking catches an entire class of broadcasting bug at bind time. Nothing else in the review makes a model's data dependencies declarative in this way. Most frameworks have the model reach out and get what it needs."

[Atomica](../approaches/atomica.md) does the same thing in a different medium: the databook — a data-entry spreadsheet with exactly the fields the model needs, per population, with the right units and labels — is **generated from the framework**.

## The proposal

### The default: values inline, typed by slot

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.perday(0.05)),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
```

There is no separate parameters block, because in a two-transition model a parameters block is pure ceremony. Naming a parameter is for when it is used twice, fit, or documented:

```python
sir = ss.Disease(
    beta      = ss.Par(ss.perday(0.05), bounds=[0.001, 0.5], source='Ferguson et al. 2020, Table 2'),
    dur_inf   = ss.Par(ss.days(10),     bounds=[3, 30]),
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=dur_inf),
)
```

`bounds=` is camdl's `in [lo, hi]`, and it is what makes calibration declarative ([calibration.md](calibration.md)) and what lets the inference layer transform automatically — camdl fits probabilities on a logit scale and rates on a log scale "without the user specifying it", because the parameters are typed.

`source=` is [EMULSION](../approaches/emulsion.md)'s field, and it is one line for the question every reviewer asks:

```yaml
force_of_infection:
  value: 'transmission_I * total_I / total_population'
  source: 'classical function assuming frequency dependence'
```

### The split, when you want it

**camdl:**

```camdl
parameters {
  #' transmission rate
  beta : rate in [0.001, 2.0]
}
```
plus a separate `params.toml`.

**Lingua franca** — the same separation, opt-in:

```python
sir = ss.Disease(
    beta      = ss.Par(rate=True, bounds=[0.001, 2.0], doc='transmission rate'),
    infection = ss.transmission('S -> I', beta=beta),
    ...
)

sim = ss.Sim(sir, pars='params/kenya_2024.csv')
```

and either way:

```text
>>> sim.hash()
model=8f3a21  pars=c47e90  config=1b6d55  scenario=baseline
```

That is [camdl](../approaches/camdl.md)'s factored run identity, and it costs nothing to compute. See [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md).

### Every parameter slot accepts the whole ladder

Starsim already does most of this: a parameter "may be a scalar, a `TimePar`, a `Dist`, or a callable of `(module, sim, uids)`". Formalize the ladder and make it uniform:

```python
beta = 0.05                                       # bare number, unit from context
beta = ss.perday(0.05)                            # typed
beta = ss.lognorm(mean=0.05, std=0.01)            # a distribution: heterogeneity or a prior
beta = ss.timeseries({2020: 0.05, 2021: 0.02})    # time-varying
beta = ss.seasonal(mean=0.05, amplitude=0.3)      # a named forcing function
beta = lambda mod, sim, uids: ...                 # escape hatch
```

[epipack](../approaches/epipack.md) is the standard to meet here: it accepts a number, a callable of `(t, y)`, or a SymPy expression, and does the right thing in ODE, Gillespie, *and* inhomogeneous-Poisson modes. From its review: "a rate is an expression in time and state, and each backend is responsible for handling that correctly or refusing."

[odin](../approaches/odin.md)'s `interpolate(time, value, mode)` with `constant` / `linear` / `spline` is the right set of options for a time series; `ss.timeseries(..., mode='linear')` should carry them.

### Distributions do double duty, and that is fine

A distribution in a parameter slot means one of two things, and the slot disambiguates:

- **Heterogeneity**: `dur=ss.lognorm(mean=ss.days(10), std=ss.days(3))` — each agent draws its own dwell time. In an ODE backend this becomes Erlang sub-compartments ([model-structure.md](model-structure.md)).
- **Uncertainty**: `beta=ss.Par(prior=ss.lognorm(...))` — a prior for calibration. camdl's `~ prior` syntax.

Keeping them in separate slots (`dur=` versus `prior=`) is what stops the ambiguity that "a vector-valued parameter means a time series here and a sensitivity sweep there" produces in [EpiModel](../approaches/epimodel.md).

### Shapes

[epymorph](../approaches/epymorph.md)'s `Shapes.Scalar` / `N` / `T` / `TxN` / `NxN`, checked at bind time. Its review states the complement precisely: "Dimensional types say *what a number means*; shapes say *how many of them there are and along which axis*. A language spanning agents, patches, strata, and time needs both."

The same distinction appears as [SimInf](../approaches/siminf.md)'s `gdata` versus `ldata` (global versus per-node parameters) and [MetaCast](../approaches/metacast.md)'s `universal_params` versus `subpop_params`. Two independent arrivals at a coarse version of what epymorph does properly.

**Lingua franca** — infer the shape from the value, check it against the slot:

```python
beta = 0.05                                    # scalar: same everywhere
beta = ss.by(age=[0.08, 0.05, 0.03])           # one per age stratum → shape (age,)
beta = ss.by(patch=gravity_matrix)             # shape (patch, patch)
```

```text
ShapeError: 'beta' for transition 'infection' has shape (age=3,)
  but the model is stratified by (age=3, patch=47).
  Use ss.by(age=..., patch=...) or leave 'patch' out to broadcast.
```

Note the last line: broadcasting is allowed and named. epymorph errors; we say what to do.

### Requirements

Every module declares what it needs, so a model's data interface is checkable before anything runs, and so the data-entry form can be generated:

```python
class MyNetwork(ss.Network):
    requires = [
        ss.Needs('population', shape='patch', source='UN WPP 2024'),
        ss.Needs('contact_matrix', shape='age x age', source='socialmixr:POLYMOD'),
    ]
```

```python
sim.needs()             # what this model requires, as a table
sim.needs(to='form.csv')  # Atomica's databook: the data-entry contract, generated
```

This is also the mechanism that makes cross-module dependencies visible, which is a hole in Starsim's connectors — see [composition-and-effects.md](composition-and-effects.md).

### Named data, not just values

[Epydemix](../approaches/epydemix.md)'s real contribution is that this works:

```python
population = load_epydemix_population("United_States", contacts_source="mistry_2021",
                                      layers=["home","work","school","community"])
```

Its review: "A lingua franca that can express a contact matrix but cannot say *which* contact matrix, from *which* source, for *which* place, has moved the problem rather than solved it."

epymorph's ADRIOs are the pull-based version with caching and provenance; camdl has `read("data/lga_pop.tsv", column="patch")`; [LASER](../approaches/laser.md) ships a library of migration models. All four say the same thing: **a reference to data is part of the model.**

```python
sim = ss.Sim(sir, people=ss.People('Kenya', year=2024),
                  mixing=ss.contacts('Kenya', source='prem_2021'))
```

with the provenance carried into `sim.summary()` and the run hash, and the fetch cached.

## Trade-offs

- **Declaration means adding a parameter is two edits** (declare, use). Starsim already pays this and it is worth it. Mitigated by the fact that most parameters live in their transition and are never named at all.
- **A generated data form is only as good as the requirement declarations.** A module that reads something it did not declare breaks the contract silently. This argues for the requirements being *enforced* (a module gets a view containing only what it declared), which is Vivarium's `PopulationView` — see [composition-and-effects.md](composition-and-effects.md).
- **Bundled data goes stale and is large.** Prefer epymorph's pull-with-cache over Epydemix's bundle, and pin the source in the hash so a re-run with a changed upstream is detectable rather than silent.
- **Optional `source=` will mostly be empty.** True, and it costs nothing when unused, and the models that matter most are the ones where someone bothered.

## Rejected

- **Open parameter lists** (EpiModel's `...`). The cost is documented.
- **Mandatory value/model separation** (camdl). The capability is right; requiring two files for an SIR is not.
- **String-concatenated parameter names** ([MetaCast](../approaches/metacast.md)'s `parameters['p' + subpop_suffix]`, `'beta_[high,vaccinated]'`). Its review: "the lesson is in the split, not the mechanism." Shapes do this job properly.
- **`Use_Defaults: 1`** ([EMOD](../approaches/emod.md)), where "much of a model's behaviour is not visible in its files". We guess defaults aggressively and *print every one*, which is the opposite policy despite sounding similar.
- **A distribution registry with 20+ named distributions as core vocabulary.** Starsim's review flags "~20 distributions" as part of the surface-area problem. Ship the ten that get used (`bernoulli`, `uniform`, `normal`, `lognorm`, `expon`, `gamma`, `erlang`, `poisson`, `binomial`, `weibull`) and let the rest be `ss.dist(scipy_frozen)`.

## Open questions

- Should `bounds=` and `prior=` be the same field? camdl has one (`in [lo, hi] ~ prior`); they serve calibration and validation respectively, and collapsing them may be right.
- How much of the ladder does the *ODE* backend accept? An agent-level `ss.lognorm` dwell time has an Erlang expansion; an agent-level `ss.lognorm` on `beta` has no compartmental analogue without adding a dimension. This is a capability check, and the rule needs stating in [paradigm-conversion.md](paradigm-conversion.md).
