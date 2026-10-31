# Parameters and data

Where values live, `ss.par`, the forms a value can take, shapes, optional requirements, and named data.

Justified by [best-practices/parameters-and-distributions.md](../best-practices/parameters-and-distributions.md).

## The default: values live inline

A parameter used once lives in the transition that uses it. There is no parameter block to fill in.

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=ss.days(10)),
    init_prev = 0.01,
)
```

Both values become parameters of the module, named after their slots, `sir.pars.beta` and `sir.pars.dur` (qualified only on a clash, e.g. `dur_recovery`), for modifiers ([07-effects.md](07-effects.md)), `ss.change` ([08-interventions.md](08-interventions.md)), `pars=`, and inspection. Name a parameter when it is reused, fitted, or documented.

## `ss.par`

```python
ss.par(
    value  = None,        # any form on the ladder below
    bounds = None,        # [lo, hi]: the allowed range, and the search range for fitting
    prior  = None,        # a distribution: the prior, for posterior methods
    source = None,        # free text: where the number came from
    doc    = None,        # free text: what it means
    name   = None,        # usually taken from the keyword or variable
    label  = None,        # human-readable; shown in its repr, in describe() (`beta (Transmission rate) = ...`), and in the table of a fit
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

An `ss.par` behaves like its value in arithmetic, so it can be used anywhere a number can, including after `@` and inside rates.

- **Names.** A named parameter is collected into the module's `pars`, Starsim's parameter container, under its name: `sir.pars.beta`. The name is the keyword if it was passed as one (`ss.Disease(beta=ss.par(0.05), ...)`), else the variable it is bound to, by the same lookup as for transitions ([01-objects.md](01-objects.md#names)). A plain Python number used in a rate is just a number; wrap it in `ss.par` to make it addressable by name.
- **`bounds`** is what makes a parameter fittable: `sim.fit()` fits every parameter that has bounds to every result that has data, wherever it is used (a rate, an initial value, a seasonal mean, a modifier's amount) ([13-inference.md](13-inference.md)). Bounds that are positive and span a factor of 10 or more are searched on a log scale, and others on a linear one. Bounds given as bare numbers are in the unit of the value, and typed bounds are converted (`ss.par(ss.days(8), bounds=[ss.days(3), ss.weeks(4)])`). A value outside its bounds raises a `ValueError` when the sim is initialized.
- **A value per level.** `ss.par(ss.by(patch=[...]), bounds=[...])` is fitted level by level, each with the same bounds and prior; the fitted value is an `ss.by`, and one level is set by name, `pars={'beta[a]': 0.3}`.
- **`prior`** is needed only for posterior methods. Without one, a parameter with bounds is treated as uniform over them. `bounds` and `prior` are separate fields: bounds say which values are valid and where to search, a prior says what is believed. Whether they should be one field is an open question ([19-open-questions.md](19-open-questions.md)); both are kept.
- **`source`** is one optional line that answers the question every reviewer asks. It is shown by `describe()`.
- **The type** of a parameter (rate, duration, probability) comes from the slot it is used in ([02-types.md](02-types.md)) or from its typed value. There is no `kind=` argument.
- **No value.** An `ss.par` with bounds and no value starts at the middle of its bounds (on the search scale), and `describe()` says so. One with neither a value nor bounds, and no value supplied through `ss.Sim(pars=...)`, raises a `ValueError` when the sim is initialized.

Setting a parameter name that the module does not have raises a `KeyError`, listing the parameters it has. This is Starsim's existing behavior and it is kept, because a misspelled parameter that is silently ignored is one of the hardest bugs to find. A keyword number on `ss.Disease` sets the parameter of that name:

```text
>>> ss.SIR(dur_inff=3)
KeyError: "dur_inff=3" was given to "sir", but it has no parameter called "dur_inff"; its parameters are beta, dur_inf, p_death, init_prev.
     To add a parameter, use it in a transition, e.g. infection = S >> I @ dur_inff * I with dur_inff = ss.par(3).
```

### Setting a parameter

A value given for a parameter (as a keyword, in `pars=`, or by `pars.update()`) replaces it. There are three exceptions, and each keeps the parameter rather than changing what it means:

- **A declared parameter** (an `ss.par`) takes the new value and keeps its name and bounds (`sir.pars.beta = 0.1`), since the module's transitions use that same object. A new `ss.par` gives its bounds, prior and description as well.
- **A bare number for a typed value** keeps the unit of the default: `ss.SexualNet(dur=5)` on a default in years is `ss.years(5)`, with a warning if the sim runs in another unit.
- **A single value for a distribution** sets its main parameter and keeps the rest, as in Starsim 3: for `ss.bernoulli` its probability (`init_prev=0.05`), for `ss.lognorm_ex` its mean (`dur=5` is a lognormal of mean 5 with the default's standard deviation), in the default's unit if the value is a bare number (a network of partnerships is the exception: `ss.SexualNet(dur=ss.months(6))` is a mean, with partnerships ending at a constant rate, as for `ss.DegreeNet`); for `ss.choice` it is that value always (`embryos_per_pregnancy=2` is `ss.constant(2)`). A typed value (`dur=ss.days(200)`) does the same, with a warning that says it set the mean of the default distribution and how to replace the distribution instead (`ss.expon(ss.days(200))`), since in a declared model a duration alone is an exponential (MS8); a bare number does not warn. In a declared model a single value for a parameter whose default is a distribution (other than an exponential) replaces it, with a warning that says how to keep the distribution. A fixed value is `ss.constant(...)`. A dict with a `type` is a distribution (`n_contacts=dict(type='poisson', lam=4)`).

To change one parameter of a distribution and keep the others, use the distribution's own method, `pars.dur_inf.set(mean=8)`. A list or another dict given for a distribution raises `TypeError` and names that method, since it could mean the distribution's arguments or the values themselves.

Code reads a parameter as `self.pars.x` (or `disease.pars.x`). Arithmetic and NumPy on a declared parameter (`ss.par`) build an expression while the model is being made and use its current value once the model is built (from `init_post()` on), whatever the other operand (`1 - p` and `np.ones(2) - p` alike), so code beside a model (`acts*(1 - disease.pars.eff_condom)`) reads the current value: a typed value as a number in the sim's unit, a value that varies in time at the current time, a value by level of one dimension as one value for each level, and the modifiers active on it applied (one value for each group or agent when they differ between them). A parameter whose value is a distribution raises there (it has no single value: use `.mean()` or draw from it). In a module written as code, a plain value is itself (a number, or an `ss.par`'s value, which the module keeps while fitting and `sim.new()` find the `ss.par`), a typed value stays typed (its own arithmetic and `.to_prob()`/`.to_dt()` keep its unit), and a value that varies (`ss.timeseries`, `ss.seasonal`, `ss.by`) is its current value in arithmetic, NumPy, comparisons, and `float()`: of numbers, a number in the sim's unit (one for each level), as for a declared parameter; of typed values, the typed value now (an `ss.by` of one dimension: one typed array, one value per level), so `self.pars.rate.to_prob()` is the same code whether `rate` is `ss.peryear(0.1)` or a time series of them, and its arithmetic is typed arithmetic. `float(sir.pars.beta)` and `par.raw` are the value the parameter holds, without the modifiers. The engines read the same values through `ss.parameters.current_value()` and `Engine.value()`; `mod._value()` is internal.

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
- `ss.timeseries` interpolates with `mode='constant'`, `'linear'`, or `'cosine'` (an S-shaped change between each value and the next). A bare number for a time is a time in the sim's unit counted from the start ([02-types.md](02-types.md#bare-numbers)).
- `ss.seasonal(mean=, amplitude=, peak=, period=)` is `mean × (1 + amplitude × cos(2π(t − peak)/period))` with `t` in years and the period a year by default (`period=ss.years(0.5)` for two peaks a year); a peak given as a month is the middle of that month, and in a sim with no dates time starts at the beginning of a year. Its `mean` and `amplitude` may be an `ss.par`, which is then fitted or set like any other; one with no name or variable of its own is named after the parameter it is in and its field, e.g. `beta_mean`.
- A function is `fn(sim)`, or `fn(sim, uids)` for agent-only logic. It is tried on whatever the solver provides, agents or groups, and if it needs something that only exists per agent the compartmental run stops and says what it asked for. The rule is in [03-states-and-transitions.md](03-states-and-transitions.md#functions-in-any-slot).
- The forms combine: `ss.by(age=[...])` may hold time series, a time series or a seasonal value may be multiplied by a number or a parameter, and a function may return any of the others' values.
- Any of these can be wrapped in `ss.par(...)` to give it a name, bounds, or a source.

Every solver handles every form or says that it cannot, and no solver freezes a value that varies in time at its starting value. The ODE takes it at each stage of a step; the chain-binomial solver at the middle of each substep; and Gillespie holds it for a quarter of a step at a time, at its middle, with events exact within each quarter, and with a note (`approx=False` refuses it instead; [12-backends.md](12-backends.md)).

### A distribution as a value, a distribution as a prior

A distribution means two different things depending on where it is put, and the place decides:

```python
I, R = ss.states('I R')
ss.Transition(I, R, dur=ss.lognorm_ex(mean=10, std=3, unit='days'))    # variation between people
ss.par(0.05, prior=ss.lognorm_ex(mean=0.05, std=0.01))                      # uncertainty about one number
```

The first gives each agent their own dwell time. The second is a prior over a single value and is used only by calibration.

A compartmental run has no individuals to draw for, so variation between people is represented in one of two ways, depending on the slot:

| Where the distribution is | `solver='auto'` | An explicit compartmental solver (`solver='ode'`) |
|---|---|---|
| A dwell time (`dur=`) | Erlang stages matched to the distribution's mean and variance, with a note ([03-states-and-transitions.md](03-states-and-transitions.md)) | the same |
| Anything else (`beta`, `rel_sus`, `rel_trans`) | the model runs as agents | the mean is used, with a note naming the parameter |

To keep heterogeneity in `beta` or `rel_sus` in a compartmental run, bin it into a dimension (`sir.stratify(rel_sus=[...])`, [05-dimensions.md](05-dimensions.md)). Replacing a distribution by its mean is an approximation, so `approx=False` makes it `CapabilityError` and `approx=True` silences the note ([12-backends.md](12-backends.md)).

The distributions are Starsim's, under Starsim's names: `ss.normal`, `ss.lognorm_ex` (given the mean and standard deviation of the result), `ss.lognorm_im` (given those of the underlying normal), `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, and `ss.choice`. Starsim's other distributions, and any SciPy distribution through `ss.Dist`, remain available. Two are added: `ss.erlang(k=3, mean=...)`, which is `ss.gamma` with an integer shape (`ss.gamma(a=3, scale=...)`), given by its number of stages and its mean (sub-stages are also `stages=3` on the transition); and `ss.truncnorm`, a normal between `low` and `high`, as flepiMoP draws its parameters. `ss.choice` takes `p` as a row of probabilities per agent (a 2D array, or a function returning one; each row normalized), drawn with one random number per agent through the inverse CDF of that agent's row, so a choice whose probabilities differ by agent is one draw that keeps common random numbers.

The unit of a distribution is given once, either with `unit=` or on one of its parameters; the other parameters of the same dimension take it:

```python
ss.lognorm_ex(mean=10, std=3, unit='days')
ss.lognorm_ex(mean=ss.days(10), std=3)        # the same
```

## Shapes

Types say what a number means; shapes say how many there are and along which stratification.

```python
beta = 0.05                                    # the same everywhere
beta = ss.by(age=[0.08, 0.05, 0.03])           # one per age group
beta = ss.by(age=[0.08, 0.05, 0.03], sex=[1.0, 0.8])     # one per age group and sex: the product of the two
beta = ss.by(table)                            # a dataframe with a column per stratification and a value column
pop  = ss.by([[3000, 1000], [5000, 2000]], dims=['age', 'patch'])   # an array with one axis per stratification
mu   = ss.by(sex=dict(female=0.008, male=0.011))  # by label: from people.female, with or without a dimension by sex
p    = ss.by(dict(age=range(15, 50), value=vals)) # a table over a property: a column of numbers holds lower bin edges
dur  = ss.by(sex=dict(female=ss.lognorm_ex(6, 2), male=ss.lognorm_ex(12, 4)))  # a distribution for each level
```

`ss.by` names the stratification each axis belongs to, so values never depend on the order strata happen to be stored in. A value with fewer axes than the model is used for every level of the missing ones, and `describe()` prints the value with its axes (`beta = by(age=[0.08, 0.05, 0.03])`). Arithmetic works on `ss.by` values (over the same levels, or tables with the same rows; different numbers of levels raise an error), and any of their values can be a time series. Anything else is an expression in which the `ss.by` is one value, worked out where it is used, as for a parameter: tables by different bins (`p_a + p_b`), an `ss.by` with a count or a property of people, and NumPy functions of it (`np.minimum(p, 1)`, `np.log(p)`). Time series and seasonal values are expressions in the same way, and so is an `ss.by` holding them times a number or a typed value (`ss.perday(0.1) * ss.by(age=[school, 1, 1])`, as it is written inside a rate); an `ss.by` cannot hold an expression itself (`ss.by(age=[ss.perday(0.1)*school, ...])` raises a `TypeError` that says to put the factor outside). A value whose length does not match its stratification, such as four values for three age groups, raises a `ValueError` when the sim runs, naming the dimension and both lengths. A function of the sim may give the values of one dimension, `ss.by(patch=fn)` with `fn(sim)` one value per level (or one number), worked out once a step and read as a value by level is, by every solver and whatever other dimensions the sim has (MS2): a value for each place from code beside a declared model, e.g. the biting rate from mosquitoes written as code.

**Values by a property of people.** `ss.by` may be over a property that is not a dimension of the sim, which is then read from each person, as agents: under `'auto'` the model runs as agents, with a note, and an explicit compartmental solver raises `CapabilityError` naming `stratify()` (the rule for a distribution across people, and for `ss.ppl.<name>` in a rate). Its values are given by label (`sex=dict(female=..., male=...)`, read from `people.female`) or as a table; values by position (`ss.by(sex=[a, b])`) need the dimension's order, so without the dimension they raise a `KeyError`, naming the keyed form or `stratify()`. In a table, a column of numbers holds lower bin edges, the last bin open-ended, so a table by single year of age works whatever the model's age groups: agents read each person's own age, and a compartmental run averages it over each group, weighting each bin by how much of the group it covers, with a note. A row whose key is `None` gives the value for people with no value of the property; without one, they get the mean over the levels, weighted by the people at each. Values by different dimensions multiply as their outer product (`ss.by(age=[...]) * ss.by(sex=dict(...))` is `ss.by(age=[...], sex=dict(...))`, and a table times a table or a value by label is a table of every combination of their rows), and a typed value times an `ss.by` applies to each of its values (`ss.peryear(0.0016) * ss.by(sex=dict(...))`). A distribution's parameter may be a property or an expression of properties (`ss.lognorm_ex(mean=0.2*ss.ppl.age + 1, std=0.1)`), read for each person as they draw, as agents. A dwell time may be an `ss.by` of distributions of one kind: as agents, each person draws from their level's; as compartments, each group has its level's mean, with one number of Erlang stages for all, noted.

`ss.by` replaces the separate global and local parameter concepts of other frameworks (SimInf's `gdata`/`ldata`, MetaCast's `universal_params`/`subpop_params`) and replaces string-built parameter names such as `'beta_[high,vaccinated]'`, which are not part of the language.

## Values kept outside the model

Values live inline by default. They can also be supplied separately, which is useful for a set of country-specific values or a fitted parameter set:

```python
beta = ss.par(bounds=[0.001, 2.0], doc='transmission rate')
S, I, R = ss.states('S I R')
sir  = ss.Disease(
    infection = ss.Transition(S, I, rate=beta * I),
    recovery  = ss.Transition(I, R, dur=ss.days(10)),
)
sim = ss.Sim(sir, pars=sc.loadjson('params/kenya_2024.json'), unit='day')
```

`pars=` takes a dict of values by name (`beta`, or `sir.beta` when there is more than one module); values kept in a file are read into one first (`sc.loadjson()`, `pd.read_csv()`), since reading files is not the sim's job. One level of a value per level is named with brackets (`beta[a]`). Names it does not recognize raise a `KeyError`.

## Named data

A reference to data is part of the model: not just a contact matrix, but which one, from which source, for which place.

```python
ss.people('Kenya', year=2024)                     # size, age structure, sex ratio; returns an ss.People
ss.contacts('Kenya', source='prem2021')           # an age contact matrix; returns a set of ss.MixingPools (layer='home' for one setting)
ss.geo.geography('Kenya', level=1)                          # its counties: names, populations, centroids, distances, borders
ss.Sim(sir, data='weekly_cases.csv')              # observations, attached to results by column name
```

```python
sim = ss.Sim(sir, people=ss.people('Kenya', year=2024), networks=ss.contacts('Kenya', source='prem2021'))
```

`ss.people`, `ss.contacts`, and `ss.geo.geography` are lowercase because they are loaders: each looks up named data and returns the ordinary object. `ss.People(n_agents=10_000)` remains the class.

A named population gives the sim its size and the split of an age stratification that does not have one of its own, so it serves a compartmental model as well as agents; it makes the model agent-based only if it is given `n_agents=`. Without `year=`, it is the population in the year the sim starts, if the sim has dates, and otherwise in the year of the data's revision.

**Bundled and pulled.** The data comes from [`starsim_data`](https://github.com/starsimhub/starsim_data) (`import starsim_data as ssd`), which does some of each:

- **Bundled**: what is small and common. UN World Population Prospects population, birth, death, and migration rates for every country; the Prem (2021) age contact matrices by setting (`all`, `home`, `work`, `school`, `community`), for 177 countries (with Prem 2017 for the ten it lacks); and the first two administrative levels of every country, with their populations, centroids, areas, and borders. Reading it (`ssd.load('kenya', which='prem2021', kind='home')`) never touches the network, so it works offline and is reproducible from a pinned version. The bundled birth and death rates are estimates for single years, every ten years, so `ss.Births` and `ss.Deaths` given a place's name interpolate them linearly between those years (holding each for its decade overshot the UN's population of Kenya by 4–5% from 2000 to 2030); rates given as data of one's own hold each row until the next, as for data by period.
- **Pulled**: everything else, fetched on demand and cached locally (`ssd.fetch('kenya', which='deaths', years=(2000, 2025))`): annual resolution, life tables, other projection variants, the earlier Prem (2017) and the Mistry (2021, by single year of age) contact matrices (`ss.contacts('Japan', source='mistry2021')`), and finer administrative levels or other years of population (which need the geospatial extra). The cache key includes the source and its version.
- **Versioned**: each UN revision is kept as a separate store, so a new release adds data and does not change what an existing model loads; `revision=` pins one.
- **Provenance**: every data set records its source and license (`ssd.provenance('prem2017')`).

The library does not depend on other frameworks' packages (LASER's migration module, `epydemix-data`, `socialmixr`) for generators or data. A data set is adapted once, with its provenance recorded, and then it is ours to keep working.

- Where the data came from is shown by `describe()`.
- Providers are pluggable, so a group can add its own sources, and a geography can always come from the user's own table instead ([06-routes.md](06-routes.md#geography)).
- Place names are forgiving (`'Kenya'`, `'kenya'`, `'KE'`, `'KEN'`), but a name that cannot be resolved is never matched to the nearest guess: it raises a `KeyError`, listing near matches.

How data is matched to results is in [09-results.md](09-results.md).

## Not included

- **Parameters that spring into existence on first use.** A typo then becomes a silent change in behavior; see a `KeyError` above.
- **A required split between model and values.** The split is available; two files for an SIR are not required.
- **String-built parameter names.** `ss.by` does this job.
- **Required, enforced data declarations.** Requirements are metadata.
- **A `use_defaults` switch** that hides a model's behavior. Defaults here are guessed and printed.
- **New distribution names, other than `ss.erlang` and `ss.truncnorm`.** Starsim's names are used.
- **Repeating the unit on every parameter of a distribution.** Once is enough.
- **Depending on outside packages for data.** Bundle what is small and common, pull the rest with a cache, and own both.
