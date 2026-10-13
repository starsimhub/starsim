# Parameters and distributions

## Recommendation

1. **Declare parameters; refuse unknown names.** A misspelled parameter should be an error at build time, not a silent behavioral difference.
2. **A parameter may be a number, a typed quantity, a distribution, a time series, a seasonal curve, a per-group table, or a function.** Every slot accepts all of them.
3. **Keep values with the model by default, and make it possible to separate them.** camdl is right that structural identity should be separable from values; it is wrong that this must be mandatory.
4. **Let a module say what data it needs, as data.** epymorph's requirement contract is the most transferable idea in the review after typed time. Here it is optional metadata, used to generate data-entry forms and check inputs early; it is not enforced.
5. **Record where a value came from.** One optional field, `source=`, answers the question every reviewer asks.
6. **Parameters have shapes, and the shape is checked.**

## Why

### Declaration catches typos; no framework without it does

[EpiModel](../approaches/epimodel.md): "`param.net(inf.prob = 0.4, act.rate = 2, my.new.param = 0.7)` is legal… There is no parameter schema and no declaration step; a misspelled parameter name is not an error until something looks for it." Its own review lists "No declaration step of any kind — parameters, attributes, and outputs all spring into existence on first use. Typos become silent behavioral differences" as a limitation.

Against that: [MEmilio](../approaches/memilio.md) gets compile-time checking from C++ tag types (`parameters.set<TimeInfected>(2)` — "the strongest parameter-name checking in the review"); [camdl](../approaches/camdl.md) gets it from its compiler; **Starsim gets a runtime version from `define_pars`**: passing a parameter that was not declared (`ss.SIR(foo=3)`) raises an error naming it. Starsim's mechanism is already the right one for Python. Keep it.

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

### The default: values inline

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.perday(0.05)),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
```

There is no separate parameters block, because in a two-transition model it would add lines and no information. Bare numbers work too (`beta=0.05, dur=10`), taking the sim's unit; see [time-and-units.md](time-and-units.md).

`ss.par` is for a parameter that is reused, fitted, or documented:

```python
beta    = ss.par(ss.perday(0.05), bounds=[0.001, 0.5], source='Ferguson et al. 2020, Table 2')
dur_inf = ss.par(ss.days(10),     bounds=[3, 30])

sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=dur_inf),
)
```

A parameter used in one place can also be wrapped where it is used: `beta=ss.par(0.05, bounds=[0.001, 0.5])`.

`bounds=` is camdl's `in [lo, hi]`. It is what makes calibration one call ([calibration.md](calibration.md)): every parameter with bounds is fitted. Bounds are in the same unit as the value. Because the slot says what kind of quantity a parameter is, the fitting code can pick a scale automatically; camdl fits probabilities on a logit scale and rates on a log scale "without the user specifying it".

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
beta = ss.par(bounds=[0.001, 2.0], doc='transmission rate')   # no value here

sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

sim = ss.Sim(sir, pars='params/kenya_2024.csv')   # values come from the file
```

and either way:

```text
>>> sim.hash()
model=8f3a21  pars=c47e90  config=1b6d55  scenario=baseline
```

That is [camdl](../approaches/camdl.md)'s factored run identity. See [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md).

### Every parameter slot accepts the whole ladder

Starsim already does most of this: a parameter "may be a scalar, a `TimePar`, a `Dist`, or a callable of `(module, sim, uids)`". Formalize the ladder and make it uniform:

```python
beta = 0.05                                       # bare number; the sim's unit applies
beta = ss.perday(0.05)                            # typed quantity
beta = ss.lognorm_ex(mean=0.05, std=0.01)         # a distribution: varies between people
beta = ss.timeseries({2020: 0.05, 2021: 0.02})    # time-varying
beta = ss.seasonal(mean=0.05, amplitude=0.3)      # a named forcing function
beta = ss.by(age=[0.08, 0.05, 0.03])              # one value per group
beta = lambda sim: 0.05 * (1 + 0.3 * np.cos(2 * np.pi * sim.t.now('year')))   # a function
```

The function is a normal option, not a last resort. It is called as `fn(sim)` (or `fn(sim, uids)` when it needs individuals) and is tried on whatever the run provides. In an agent-based run it sees agents; in a compartmental run it sees groups, one row per stratum. A function that only uses things that exist in both (state counts, parameters, time, stratification coordinates) works in both, like the one above. A function that reaches for something only agents have (`uids`, or an attribute that is not a stratification) works in an agent-based run, and a compartmental run stops with a message naming what it asked for. There is no analysis of the function in advance; it is tried, and the result is reported.

[epipack](../approaches/epipack.md) is the standard to meet here: it accepts a number, a callable of `(t, y)`, or a SymPy expression, and does the right thing in ODE, Gillespie, *and* inhomogeneous-Poisson modes. From its review: "a rate is an expression in time and state, and each backend is responsible for handling that correctly or refusing."

[odin](../approaches/odin.md)'s `interpolate(time, value, mode)` with `constant` / `linear` / `spline` is the right set of options for a time series; `ss.timeseries(..., mode='linear')` should carry them.

### Distributions do double duty, and that is fine

A distribution in a parameter slot means one of two things, and the slot disambiguates:

- **Variation between people**: `dur=ss.lognorm_ex(mean=10, std=3, unit='days')`. Each agent draws its own dwell time. In a compartmental run a dwell-time distribution becomes sub-stages ([model-structure.md](model-structure.md)).
- **Uncertainty**: `beta=ss.par(0.05, prior=ss.lognorm_ex(mean=0.05, std=0.02))`. A prior for calibration; camdl's `~ prior` syntax.

Keeping them in separate slots (`dur=` versus `prior=`) is what stops the ambiguity that "a vector-valued parameter means a time series here and a sensitivity sweep there" produces in [EpiModel](../approaches/epimodel.md).

The distributions are Starsim's, with Starsim's names: `ss.normal`, `ss.lognorm_ex` (given the mean and standard deviation of the result), `ss.lognorm_im` (given those of the underlying normal), `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, `ss.choice`.

Units are given once. Either on the distribution, or on one parameter, with the others in the same dimension inheriting it:

```python
ss.lognorm_ex(mean=10, std=3, unit='days')
ss.lognorm_ex(mean=ss.days(10), std=3)        # the same
```

Writing `mean=ss.days(10), std=ss.days(3)` is allowed and never required.

### Shapes

[epymorph](../approaches/epymorph.md)'s `Shapes.Scalar` / `N` / `T` / `TxN` / `NxN`, checked when data is attached. Its review states the complement precisely: "Dimensional types say *what a number means*; shapes say *how many of them there are and along which axis*. A language spanning agents, patches, strata, and time needs both."

The same distinction appears as [SimInf](../approaches/siminf.md)'s `gdata` versus `ldata` (global versus per-node parameters) and [MetaCast](../approaches/metacast.md)'s `universal_params` versus `subpop_params`. Two independent arrivals at a coarse version of what epymorph does properly.

**Lingua franca**: infer the shape from the value, and check it against the model's stratification:

```python
beta = 0.05                                    # scalar: same everywhere
beta = ss.by(age=[0.08, 0.05, 0.03])           # one per age group: shape (age,)
beta = ss.by(patch=gravity_matrix)             # shape (patch, patch)
```

```text
ShapeError: 'beta' for transition 'infection' has shape (age=3,)
  but the model is stratified by (age=3, patch=47).
  Use ss.by(age=..., patch=...) or leave 'patch' out to broadcast.
```

Note the last line: broadcasting is allowed and named. epymorph errors; we say what to do.

### Requirements

A module can say what data it needs, so that inputs can be checked before anything runs and a data-entry form can be generated:

```python
class MyNetwork(ss.Network):
    requires = [
        ss.Needs('population', shape='patch', source='UN WPP 2024'),
        ss.Needs('contact_matrix', shape='age x age', source='socialmixr:POLYMOD'),
    ]
```

```python
sim.needs()             # what this model requires, as a table
sim.needs(to='form.csv')  # Atomica's databook: the data-entry contract, generated  (deferred from v1; see design/19)
```

This is optional metadata. A module without `requires` works exactly as before, and a module with it is not restricted to what it listed: it can still read anything in the sim. The declaration exists to produce the table and the form, not to police access. (Which modules write to which quantities is found by observing a run, not by declaration; see [composition-and-effects.md](composition-and-effects.md).) [CK: let's just keep `requires()`, no need to have `requires` attribute, `ss.Needs` class, and `sim.needs()` method. we can do `requires` and `ss.Requires()` if necessary]

### Named data, not just values

[Epydemix](../approaches/epydemix.md)'s real contribution is that this works:

```python
population = load_epydemix_population("United_States", contacts_source="mistry_2021",
                                      layers=["home","work","school","community"])
```

Its review: "A lingua franca that can express a contact matrix but cannot say *which* contact matrix, from *which* source, for *which* place, has moved the problem rather than solved it."

epymorph's ADRIOs are the pull-based version with caching and provenance; camdl has `read("data/lga_pop.tsv", column="patch")`; [LASER](../approaches/laser.md) ships a library of migration models. All four say the same thing: **a reference to data is part of the model.**

```python
sim = ss.Sim(sir, people=ss.People('Kenya', year=2024), # [CK: capitalization should be consistent, i think ss.people() here?]
                  networks=ss.contacts('Kenya', source='prem_2021'))
```

with the provenance carried into `sim.describe()` and the run hash, and the fetch cached.

## Trade-offs

- **Declaration means adding a parameter to a custom module is two edits** (declare, use). Starsim already works this way and it is worth it. Most parameters live in their transition and are never named at all.
- **A generated data form is only as good as the requirement declarations.** A module that reads something it did not list will be missing from the form. Enforcing the list (Vivarium's `PopulationView`, where a module sees only what it declared) would close that gap, but people forget declarations and they drift from the code, so it is left as optional metadata.
- **A function in a parameter slot is only known to work in the run methods it has been tried in.** `sir.methods()` reports what happened; nothing is promised in advance.
- **Bundled data goes stale and is large.** Prefer epymorph's pull-with-cache over Epydemix's bundle, and pin the source in the hash so a re-run with a changed upstream is detectable rather than silent. [CK: yes, but ... we can do a bit of both. see /home/cliffk/idm/starsim_data for current thinking on this; happy to revisit, but i think 'bundle small and common, pull everything else' is the right compromise.]
- **Optional `source=` will mostly be empty.** True, and it costs nothing when unused, and the models that matter most are the ones where someone bothered.

## Rejected

- **Open parameter lists** (EpiModel's `...`). The cost is documented.
- **Mandatory value/model separation** (camdl). The capability is right; requiring two files for an SIR is not.
- **String-concatenated parameter names** ([MetaCast](../approaches/metacast.md)'s `parameters['p' + subpop_suffix]`, `'beta_[high,vaccinated]'`). Its review: "the lesson is in the split, not the mechanism." Shapes do this job properly.
- **`Use_Defaults: 1`** ([EMOD](../approaches/emod.md)), where "much of a model's behavior is not visible in its files". We guess defaults aggressively and *print every one*, which is the opposite policy despite sounding similar.
- **New distribution names.** Starsim's names stay (`ss.lognorm_ex`, not a new `ss.lognorm`; `ss.nbinom`, not `ss.neg_binomial`), and there is no separate `ss.erlang`: use `ss.gamma` with an integer shape, or `stages=` on the transition. Starsim's review flags "~20 distributions" as part of the surface-area problem, so the documentation teaches the dozen listed above; the others, and any SciPy distribution via `ss.Dist`, remain available. [CK: i think we can revisit the list of starsim distributions, i'm not wedded to them necessarily.]
- **Enforced requirement views** (Vivarium). See the trade-off above.
- **Repeating the unit on every distribution parameter.** Once is enough.

## Open questions

- Should `bounds=` and `prior=` be the same field? camdl has one (`in [lo, hi] ~ prior`); they serve calibration and validation respectively, and collapsing them may be right. [CK: hm, this is a good question! i think pros and cons either way. i think keep both to start and then we can decide.]
- What does a compartmental run do with a distribution that varies between people? A dwell-time distribution has a standard answer (sub-stages). A distribution on `beta` or `rel_sus` does not, short of adding a dimension or using the mean. Under `method='auto'` such a model runs as agents; what an explicit `method='ode'` should do is a question for [paradigm-conversion.md](paradigm-conversion.md). [CK: either stratify, or simplify .... right?]
