# Parameters and data

Where values live, `ss.par`, the forms a value can take, shapes, optional requirements, and named data.

Justified by [best-practices/parameters-and-distributions.md](../best-practices/parameters-and-distributions.md).

## The default: values live inline

A parameter used once lives in the transition that uses it. There is no parameter block to fill in.

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
```

Both values are addressable through their transition, as `sir.infection.beta` and `sir.recovery.dur`, for modifiers ([07-effects.md](07-effects.md)), `ss.change` ([08-interventions.md](08-interventions.md)), and inspection. Name a parameter when it is reused, fitted, or documented.

## `ss.par`

```python
ss.par(
    value  = None,        # any form on the ladder below
    bounds = None,        # [lo, hi]: the allowed range, and the search range for fitting
    prior  = None,        # a distribution: the prior, for posterior methods
    source = None,        # free text: where the number came from
    doc    = None,        # free text: what it means
)
```

```python
beta = ss.par(0.05, bounds=[0.001, 0.5], source='Ferguson et al. 2020, Table 2')
dur  = ss.par(ss.days(10), bounds=[3, 30])

S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = S >> I @ beta * I,
    recovery  = I >> R @ dur,
)
```

An `ss.par` behaves like its value in arithmetic, so it can be used anywhere a number can, including after `@` and inside rates. In the string form it is passed the same way: `ss.transmission('S -> I', beta=beta)`.

- **Names.** A named parameter is collected into the module's `pars`, Starsim's parameter container, under its name: `sir.pars.beta`. The name is the keyword if it was passed as one (`ss.Disease(beta=ss.par(0.05), ...)`), else the variable it is bound to, by the same lookup as for transitions ([01-objects.md](01-objects.md#names)). A plain Python number used in a rate is just a number; wrap it in `ss.par` to make it addressable by name.
- **`bounds`** is what makes a parameter fittable: `sim.calibrate()` fits every parameter that has bounds to every result that has data ([13-inference.md](13-inference.md)). The scale used for the search (log for rates and durations, logit for probabilities) follows from the slot the parameter is used in. Bounds given as bare numbers are in the unit of the value. A value outside its bounds raises `E606 OutOfBounds`.
- **`prior`** is needed only for posterior methods. Without one, a parameter with bounds is treated as uniform over them.
- **`source`** is one optional line that answers the question every reviewer asks. It is shown by `describe()` and saved in the model spec.
- **The type** of a parameter (rate, duration, probability) comes from the slot it is used in ([02-types.md](02-types.md)) or from its typed value. There is no `kind=` argument.
- **No value.** An `ss.par` with bounds and no value starts at the middle of its bounds (on the search scale), and `describe()` says so. One with neither a value nor bounds, and no value supplied through `ss.Sim(pars=...)`, raises `E604 UnvaluedParameter`.

Setting a parameter name that the module does not have raises `E602 UnknownParameter`, listing the closest names. This is Starsim's existing behavior and it is kept, because a misspelled parameter that is silently ignored is one of the hardest bugs to find:

```text
E602 UnknownParameter: 'sir' has no parameter 'dur_inff'.
     Did you mean 'dur_inf'?
     Parameters: beta, dur_inf
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

- `ss.timeseries` interpolates with `mode='constant'`, `'linear'`, or `'spline'`.
- A function is `fn(sim)`, or `fn(sim, uids)` for agent-only logic. It is tried on whatever the run method provides, agents or groups, and if it needs something that only exists per agent the compartmental run stops and says what it asked for. The rule is in [03-states-and-transitions.md](03-states-and-transitions.md#functions-in-any-slot).
- The forms combine: `ss.by(age=[...])` may hold time series, and a function may return any of the others' values.
- Any of these can be wrapped in `ss.par(...)` to give it a name, bounds, or a source.

Every run method handles every form or says that it cannot. In particular a time-varying rate under an exact continuous-time method (`'ctmc'`) needs the time-inhomogeneous algorithm; a method MUST NOT freeze the rate at its starting value.

### A distribution as a value, a distribution as a prior

A distribution means two different things depending on where it is put, and the place decides:

```python
ss.progression('I -> R', dur=ss.lognorm_ex(mean=10, std=3, unit='days'))    # variation between people
ss.par(0.05, prior=ss.lognorm_ex(mean=0.05, std=0.01))                      # uncertainty about one number
```

The first gives each agent their own dwell time (and becomes stages in a compartmental run). The second is a prior over a single value and is used only by calibration.

## Shapes

Types say what a number means; shapes say how many there are and along which stratification.

```python
beta = 0.05                                    # the same everywhere
beta = ss.by(age=[0.08, 0.05, 0.03])           # one per age group
beta = ss.by(age=[0.08, 0.05, 0.03], sex=[1.0, 0.8])     # one per age group and sex: the product of the two
beta = ss.by(table)                            # a dataframe with a column per stratification and a value column
```

`ss.by` names the stratification each axis belongs to, so values never depend on the order strata happen to be stored in. A value with fewer axes than the model is used for every level of the missing ones; `describe()` shows this (`beta: by age (3); the same for every patch (47)`). A value whose length does not match its stratification, such as four values for three age groups, raises `E603 ShapeError`, naming the parameter, the transition, and both lengths.

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

`pars=` takes a dict or a file (CSV, TOML, or JSON) of values by name (`beta`, or `sir.beta` when there is more than one module). Names it does not recognize raise `E602 UnknownParameter`.

Either way, `sim.hash()` reports the factored run identity ([11-random.md](11-random.md)):

```text
model=8f3a21  pars=c47e90  config=1b6d55  scenario=baseline  fit=none  seed=1
```

so the question "did the model change, or did the numbers?" has an answer whether or not the values were in a separate file.

## Requirements

A module MAY list the data it needs. This is optional metadata, used to tell the user what to supply and to generate a data-entry form. It is not enforced: a module can read anything, and a module that lists nothing works as before.

```python
class MyNetwork(ss.Network):
    requires = [
        ss.Needs('population',     shape='patch',     source='UN WPP 2024'),
        ss.Needs('contact_matrix', shape='age x age', source='socialmixr:POLYMOD'),
    ]
```

`ss.Needs(name, shape=, source=, doc=, default=)` describes one item. For modules made only of states, transitions, and parameters, the list is generated from the parameters, so nothing has to be written.

```python
sim.needs()                 # the table of what this sim needs, and which items are still missing
sim.needs(to='form.csv')    # a data-entry form with those fields, units, and labels
```

`needs(to=...)` is Atomica's databook, generated from the model. Because the list is optional, the form is as complete as the declarations are; items a module reads without listing do not appear in it.

## Named data

A reference to data is part of the model: not just a contact matrix, but which one, from which source, for which place.

```python
ss.People('Kenya', year=2024)                     # size, age structure, sex ratio
ss.contacts('Kenya', source='prem_2021')          # an age contact matrix, as mixing pools
ss.geo('Kenya', level='county')                   # names, populations, centroids, adjacency
ss.Sim(sir, data='weekly_cases.csv')              # observations, attached to results by column name
```

- Data is fetched on first use and cached locally; the cache key includes the source and its version.
- The resolved source, version, and retrieval date are shown by `describe()` and contribute to the parameter hash, so a re-run against changed upstream data can be detected.
- Providers are pluggable, so a group can add its own sources.
- A place name that cannot be resolved raises `E605 UnknownLocation`, listing near matches and the providers searched.

How data is matched to results is in [09-results.md](09-results.md).

## Not included

- **Parameters that spring into existence on first use.** A typo then becomes a silent change in behavior; see `E602` above.
- **A required split between model and values.** The split is available; two files for an SIR are not required.
- **String-built parameter names.** `ss.by` does this job.
- **Required, enforced data declarations.** Requirements are metadata.
- **A `use_defaults` switch** that hides a model's behavior. Defaults here are guessed and printed.
