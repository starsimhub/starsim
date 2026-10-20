# Parameters and data

Where values live, `ss.par`, the forms a value can take, shapes, optional requirements, and named data.

Justified by [best-practices/parameters-and-distributions.md](../best-practices/parameters-and-distributions.md).

## The default: values live inline

A parameter used once lives in the transition that uses it. There is no parameter block to fill in.

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
    init_prev = 0.01,
)
```

Both values become parameters of the module, named after their slots, `sir.pars.beta` and `sir.pars.dur` (qualified only on a clash, e.g. `dur_recovery`), for modifiers ([07-effects.md](07-effects.md)), `ss.change` ([08-interventions.md](08-interventions.md)), `pars=`, and inspection. Name a parameter when it is reused, fitted, or documented.

## `ss.par`

<!-- fragment -->
```python
ss.par(
    value  = None,        # any form on the ladder below
    bounds = None,        # [lo, hi]: the allowed range, and the search range for fitting
    prior  = None,        # a distribution: the prior, for posterior methods
    source = None,        # free text: where the number came from
    doc    = None,        # free text: what it means
    name   = None,        # usually taken from the keyword or variable
    label  = None,        # human-readable
)
```

```python
beta = ss.par(0.05, bounds=[0.001, 0.5], source='Ferguson et al. 2020, Table 2')
dur  = ss.par(ss.days(10), bounds=[3, 30])

S, I, R = ss.states(3)
sir = ss.Disease(
    infection = S >> I @ beta * I,
    recovery  = I >> R @ dur,
)
```

An `ss.par` behaves like its value in arithmetic, so it can be used anywhere a number can, including after `@` and inside rates. In the string form it is passed the same way: `ss.transmission('S -> I', beta=beta)`.

- **Names.** A named parameter is collected into the module's `pars`, Starsim's parameter container, under its name: `sir.pars.beta`. The name is the keyword if it was passed as one (`ss.Disease(beta=ss.par(0.05), ...)`), else the variable it is bound to, by the same lookup as for transitions ([01-objects.md](01-objects.md#names)). A plain Python number used in a rate is just a number; wrap it in `ss.par` to make it addressable by name.
- **`bounds`** is what makes a parameter fittable: `sim.calibrate()` fits every parameter that has bounds to every result that has data, wherever it is used (a rate, an initial value, a seasonal mean, a modifier's amount) ([13-inference.md](13-inference.md)). Bounds that are positive and span a factor of 10 or more are searched on a log scale, and others on a linear one. Bounds given as bare numbers are in the unit of the value, and typed bounds are converted (`ss.par(ss.days(8), bounds=[ss.days(3), ss.weeks(4)])`). A value outside its bounds raises `OutOfBounds` when the sim is initialized.
- **A value per level.** `ss.par(ss.by(patch=[...]), bounds=[...])` is fitted level by level, each with the same bounds and prior; the fitted value is an `ss.by`, and one level is set by name, `pars={'beta[a]': 0.3}`.
- **`prior`** is needed only for posterior methods. Without one, a parameter with bounds is treated as uniform over them. `bounds` and `prior` are separate fields: bounds say which values are valid and where to search, a prior says what is believed. Whether they should be one field is an open question ([19-open-questions.md](19-open-questions.md)); both are kept.
- **`source`** is one optional line that answers the question every reviewer asks. It is shown by `describe()` and saved in the model spec.
- **The type** of a parameter (rate, duration, probability) comes from the slot it is used in ([02-types.md](02-types.md)) or from its typed value. There is no `kind=` argument.
- **No value.** An `ss.par` with bounds and no value starts at the middle of its bounds (on the search scale), and `describe()` says so. One with neither a value nor bounds, and no value supplied through `ss.Sim(pars=...)`, raises `UnvaluedParameter` when the sim is initialized.

Setting a parameter name that the module does not have raises `UnknownParameter`, listing the parameters it has. This is Starsim's existing behavior and it is kept, because a misspelled parameter that is silently ignored is one of the hardest bugs to find. A keyword number on `ss.Disease` sets the parameter of that name:

```text
>>> ss.SIR(dur_inff=3)
UnknownParameter: "dur_inff=3" was given to "sir", but it has no parameter called "dur_inff"; its parameters are beta, dur_inf, p_death, init_prev.
     To add a parameter, use it in a transition, e.g. infection = S >> I @ dur_inff * I with dur_inff = ss.par(3).
```

## The ladder

Every slot that takes a value accepts every form below. The slot fixes what the value means; the form is free.

```python
beta = 0.05                                          # a number; unit from the sim
beta = ss.perday(0.05)                               # a typed quantity
beta = ss.lognorm_ex(mean=0.05, std=0.01)            # a distribution: varies between agents
beta = ss.timeseries({2020: 0.05, 2021: 0.02})       # varies in time
beta = ss.timeseries('beta.csv', mode='linear')      # the same, from a file
beta = ss.seasonal(mean=0.05, amplitude=0.3, peak='January')
beta = ss.by(age=[0.08, 0.05, 0.03])                 # one value per stratum
beta = lambda sim: 0.05 if sim.now < ss.date('2020-03-15') else 0.03      # a function
```

- `ss.timeseries(..., period=ss.years(1))` repeats with the period given, as a seasonal pattern written out over one year does.
- `ss.timeseries` interpolates with `mode='constant'`, `'linear'`, `'cosine'` (an S-shaped change between each value and the next), or `'spline'`. A bare number for a time is a time in the sim's unit counted from the start ([02-types.md](02-types.md#bare-numbers)).
- `ss.seasonal(mean=, amplitude=, peak=)` is `mean × (1 + amplitude × cos(2π(t − peak)))` with `t` in years; a peak given as a month is the middle of that month, and in a sim with no dates time starts at the beginning of a year. Its `mean` may be an `ss.par`, which is then fitted or set like any other.
- A function is `fn(sim)`, or `fn(sim, uids)` for agent-only logic. It is tried on whatever the solver provides, agents or groups, and if it needs something that only exists per agent the compartmental run stops and says what it asked for. The rule is in [03-states-and-transitions.md](03-states-and-transitions.md#functions-in-any-slot).
- The forms combine: `ss.by(age=[...])` may hold time series, a time series or a seasonal value may be multiplied by a number or a parameter, and a function may return any of the others' values.
- Any of these can be wrapped in `ss.par(...)` to give it a name, bounds, or a source.

Every solver handles every form or says that it cannot, and no solver freezes a value that varies in time at its starting value. The ODE takes it at each stage of a step; the chain-binomial solver at the middle of each substep; and Gillespie holds it for a quarter of a step at a time, at its middle, with events exact within each quarter, and with a note (`approx=False` refuses it instead; [12-backends.md](12-backends.md)).

### A distribution as a value, a distribution as a prior

A distribution means two different things depending on where it is put, and the place decides:

```python
ss.progression('I -> R', dur=ss.lognorm_ex(mean=10, std=3, unit='days'))    # variation between people
ss.par(0.05, prior=ss.lognorm_ex(mean=0.05, std=0.01))                      # uncertainty about one number
```

The first gives each agent their own dwell time. The second is a prior over a single value and is used only by calibration.

A compartmental run has no individuals to draw for, so variation between people is represented in one of two ways, depending on the slot:

| Where the distribution is | `solver='auto'` | An explicit compartmental solver (`solver='ode'`) |
|---|---|---|
| A dwell time (`dur=`) | Erlang stages matched to the distribution's mean and variance, with a note ([03-states-and-transitions.md](03-states-and-transitions.md)) | the same |
| Anything else (`beta`, `rel_sus`, `rel_trans`) | the model runs as agents | the mean is used, with a note naming the parameter |

To keep heterogeneity in `beta` or `rel_sus` in a compartmental run, bin it into a dimension (`sir.stratify(rel_sus=[...])`, [05-dimensions.md](05-dimensions.md)). Replacing a distribution by its mean is an approximation, so `approx=False` makes it `CapabilityError` and `approx=True` silences the note ([12-backends.md](12-backends.md)).

The distributions are Starsim's, under Starsim's names: `ss.normal`, `ss.lognorm_ex` (given the mean and standard deviation of the result), `ss.lognorm_im` (given those of the underlying normal), `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, and `ss.choice`. Starsim's other distributions, and any SciPy distribution through `ss.Dist`, remain available. Two are added: `ss.erlang(k=3, mean=...)`, which is `ss.gamma` with an integer shape (`ss.gamma(a=3, scale=...)`), given by its number of stages and its mean (sub-stages are also `stages=3` on the transition); and `ss.truncnorm`, a normal between `low` and `high`, as flepiMoP draws its parameters.

The unit of a distribution is given once, either with `unit=` or on one of its parameters; the other parameters of the same dimension take it:

```python
ss.lognorm_ex(mean=10, std=3, unit='days')
ss.lognorm_ex(mean=ss.days(10), std=3)        # the same
```

## Shapes

Types say what a number means; shapes say how many there are and along which stratification.

<!-- fragment -->
```python
beta = 0.05                                    # the same everywhere
beta = ss.by(age=[0.08, 0.05, 0.03])           # one per age group
beta = ss.by(age=[0.08, 0.05, 0.03], sex=[1.0, 0.8])     # one per age group and sex: the product of the two
beta = ss.by(table)                            # a dataframe with a column per stratification and a value column
pop  = ss.by([[3000, 1000], [5000, 2000]], dims=['age', 'patch'])   # an array with one axis per stratification
mu   = ss.by(sex=dict(female=0.008, male=0.011))  # by label: from people.female, with or without a dimension by sex
p    = ss.by(dict(age=range(15, 50), value=vals)) # a table over a property: a column of numbers holds lower bin edges
dur  = ss.by(sex=dict(female=ss.lognorm_ex(6, 2), male=ss.lognorm_ex(12, 4)))  # a distribution for each level
mix  = ss.by(dict(age=[15, 15, 30, 30], to=['pill', 'iud', 'pill', 'iud'], value=[0.7, 0.3, 0.4, 0.6]))  # in p=, by destination (to)
act  = ss.by(dict(since_birth=[None, 0, 3, 12], value=[1.0, 0.0, 0.5, 1.0]))  # over a clock: None is its level none
```

`ss.by` names the stratification each axis belongs to, so values never depend on the order strata happen to be stored in. A value with fewer axes than the model is used for every level of the missing ones, and `describe()` prints the value with its axes (`beta = by(age=[0.08, 0.05, 0.03])`). Arithmetic works on `ss.by` values (over the same levels; different numbers of levels raise an error), and any of their values can be a time series. A value whose length does not match its stratification, such as four values for three age groups, raises `ShapeError` when the sim runs, naming the dimension and both lengths.

**Values by a property of people.** `ss.by` may be over a property that is not a dimension of the sim, which is then read from each person, as agents: under `'auto'` the model runs as agents, with a note, and an explicit compartmental solver raises `CapabilityError` naming `stratify()` (the rule for a distribution across people, and for `ss.ppl.<name>` in a rate). Its values are given by label (`sex=dict(female=..., male=...)`, read from `people.female`) or as a table; values by position (`ss.by(sex=[a, b])`) need the dimension's order, so without the dimension they raise `MissingDimension`, naming the keyed form or `stratify()`. In a table, a column of numbers holds lower bin edges, the last bin open-ended, so a table by single year of age works whatever the model's age groups: agents read each person's own age, and a compartmental run averages it over each group, weighting each bin by how much of the group it covers, with a note. A row whose key is `None` gives the value for people with no value of the property; without one, they get the mean over the levels, weighted by the people at each. Values by different dimensions multiply as their outer product (`ss.by(age=[...]) * ss.by(sex=dict(...))` is `ss.by(age=[...], sex=dict(...))`, and a table times a table or a value by label is a table of every combination of their rows), and a typed value times an `ss.by` applies to each of its values (`ss.peryear(0.0016) * ss.by(sex=dict(...))`). Over a clock ([05](05-dimensions.md#counters-and-clocks)), a table's numbers are times in the clock's unit, its `None` row is for the people at `none`, and values by position are for the levels as given, also when a compartmental run keeps the clock a step at a time. A dimension called `to` is the destination of a transition with several, in `p=` ([03](03-states-and-transitions.md#branching)). A distribution's parameter may be a property or an expression of properties (`ss.lognorm_ex(mean=0.2*ss.ppl.age + 1, std=0.1)`), read for each person as they draw, as agents. A dwell time may be an `ss.by` of distributions of one kind: as agents, each person draws from their level's; as compartments, each group has its level's mean, with one number of Erlang stages for all, noted.

`ss.by` replaces the separate global and local parameter concepts of other frameworks (SimInf's `gdata`/`ldata`, MetaCast's `universal_params`/`subpop_params`) and replaces string-built parameter names such as `'beta_[high,vaccinated]'`, which are not part of the language.

## Values kept outside the model

Values live inline by default. They can also be supplied separately, which is useful for a set of country-specific values or a fitted parameter set:

```python
beta = ss.par(bounds=[0.001, 2.0], doc='transmission rate')
sir  = ss.Disease(
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
sim = ss.Sim(sir, pars='params/kenya_2024.csv', unit='day')
```

`pars=` takes a dict or a file of values by name (`beta`, or `sir.beta` when there is more than one module): a CSV with a column of names and a column of values (`name,value`) or one row with a column per parameter, or a TOML or JSON table. One level of a value per level is named with brackets (`beta[a]`). Names it does not recognize raise `UnknownParameter`.

Either way, `sim.hash()` returns the parts of the run's identity as an `sc.objdict` ([11-random.md](11-random.md)):

```text
>>> sim.hash()
objdict(model='8f3a21', pars='c47e90', config='1b6d55', scenario='baseline', data='a90f3e', seed=1)
```

A change of values changes `pars` and nothing else, so the question "did the model change, or did the numbers?" has an answer whether or not the values were in a separate file. The hash is a cheap by-product, useful for telling apart the many runs of a sweep or a calibration made from one commit; the git hash or tag of the code remains the main record of what was run.

## Requirements

A module MAY list the data it needs. One word, `requires`, names the list on the module, the entries in it, and the table. This is optional metadata, used to tell the user what to supply and to check inputs before anything runs. It is not enforced: a module can read anything, and a module that lists nothing works as before.

```python
class MyNetwork(ss.Network):
    requires = [
        ss.Requires('population',     shape='patch',     source='UN WPP 2024'),
        ss.Requires('contact_matrix', shape='age x age', source='Prem et al. 2021'),
    ]
```

`ss.Requires(name, shape=, source=, doc=, default=)` describes one item. For modules made only of states, transitions, and parameters, the list is generated from the parameters, so nothing has to be written.

```python
sim.requires()                 # the table of what this sim requires, and which items are still missing
```

`sim.requires()` works without initializing the sim. A data-entry form generated from the list (`sim.requires(to='form.csv')`, Atomica's databook) serves one use case and is deferred from v1 ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)). Because the list is optional, the table is as complete as the declarations are; items a module reads without listing do not appear in it.

## Named data

A reference to data is part of the model: not just a contact matrix, but which one, from which source, for which place.

```python
ss.people('Kenya', year=2024)                     # size, age structure, sex ratio; returns an ss.People
ss.contacts('Kenya', source='prem2021')           # an age contact matrix; returns a set of ss.MixingPools (layer='home' for one setting)
ss.geo('Kenya', level=1)                          # its counties: names, populations, centroids, distances, borders
ss.Sim(sir, data='weekly_cases.csv')              # observations, attached to results by column name
```

```python
sim = ss.Sim(sir, people=ss.people('Kenya', year=2024), networks=ss.contacts('Kenya', source='prem2021'))
```

`ss.people`, `ss.contacts`, and `ss.geo` are lowercase because they are loaders: each looks up named data and returns the ordinary object. `ss.People(n_agents=10_000)` remains the class.

A named population gives the sim its size and the split of an age stratification that does not have one of its own, so it serves a compartmental model as well as agents; it makes the model agent-based only if it is given `n_agents=`. Without `year=`, it is the population in the year the sim starts, if the sim has dates, and otherwise in the year of the data's revision.

**Bundled and pulled.** The data comes from [`starsim_data`](https://github.com/starsimhub/starsim_data) (`import starsim_data as ssd`), which does some of each:

- **Bundled**: what is small and common. UN World Population Prospects population, birth, death, and migration rates for every country; the Prem (2021) age contact matrices by setting (`all`, `home`, `work`, `school`, `community`), for 177 countries (with Prem 2017 for the ten it lacks); and the first two administrative levels of every country, with their populations, centroids, areas, and borders. Reading it (`ssd.load('kenya', which='prem2021', kind='home')`) never touches the network, so it works offline and is reproducible from a pinned version.
- **Pulled**: everything else, fetched on demand and cached locally (`ssd.fetch('kenya', which='deaths', years=(2000, 2025))`): annual resolution, life tables, other projection variants, the earlier Prem (2017) and the Mistry (2021, by single year of age) contact matrices (`ss.contacts('Japan', source='mistry2021')`), and finer administrative levels or other years of population (which need the geospatial extra). The cache key includes the source and its version.
- **Versioned**: each UN revision is kept as a separate store, so a new release adds data and does not change what an existing model loads; `revision=` pins one.
- **Provenance**: every data set records its source and license (`ssd.provenance('prem2017')`).

The library does not depend on other frameworks' packages (LASER's migration module, `epydemix-data`, `socialmixr`) for generators or data. A data set is adapted once, with its provenance recorded, and then it is ours to keep working.

- Where the data came from is shown by `describe()` and written in the model spec, so it is part of `sim.hash()` and `sim.to_code()` looks it up again.
- Providers are pluggable, so a group can add its own sources, and a geography can always come from the user's own table instead ([06-routes.md](06-routes.md#geography)).
- Place names are forgiving (`'Kenya'`, `'kenya'`, `'KE'`, `'KEN'`), but a name that cannot be resolved is never matched to the nearest guess: it raises `UnknownLocation`, listing near matches.

How data is matched to results is in [09-results.md](09-results.md).

## Not included

- **Parameters that spring into existence on first use.** A typo then becomes a silent change in behavior; see `UnknownParameter` above.
- **A required split between model and values.** The split is available; two files for an SIR are not required.
- **String-built parameter names.** `ss.by` does this job.
- **Required, enforced data declarations.** Requirements are metadata.
- **A `use_defaults` switch** that hides a model's behavior. Defaults here are guessed and printed.
- **New distribution names, other than `ss.erlang` and `ss.truncnorm`.** Starsim's names are used.
- **Repeating the unit on every parameter of a distribution.** Once is enough.
- **Depending on outside packages for data.** Bundle what is small and common, pull the rest with a cache, and own both.
