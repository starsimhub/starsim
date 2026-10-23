# The complete vocabulary

Every public name the specification uses, in one place, with a count.

Justified by [best-practices/principles.md](../best-practices/principles.md): a vocabulary nobody can hold in their head is a vocabulary nobody uses correctly.

The count is here so that the size of the language is something that can be looked at. The base is Starsim, so most of the names are Starsim's own and are marked as such; names this specification adds are in **bold**.

## Capitalization

The rule is Starsim's existing one (`ss.Dist` and `ss.normal`, `ss.Rate` and `ss.perday`):

> **Modules, containers, and base classes are CamelCase. The concrete pieces you put inside a module are lowercase.**

So `ss.Sim`, `ss.MultiSim`, `ss.Disease`, `ss.RandomNet`, `ss.MixingPool`, and `ss.Intervention` are CamelCase, along with the two graph classes `ss.State` and `ss.Transition` and the products `ss.Vaccine`, `ss.Diagnostic`, `ss.Treatment`. Time values, distributions, `ss.par`, `ss.states`, `ss.null`, the transition constructors, `ss.modify`, `ss.outcome`, the parameter forms, the deliveries, and functions such as `ss.parallel` are lowercase. So are the loaders, which look up named data and return an ordinary object: `ss.people('Kenya')` returns an `ss.People`, and `ss.contacts('Kenya')` a set of `ss.MixingPools`. The tables they read are in `ss.data` (`ss.data.age('Kenya')`, `ss.data.population()`, `ss.data.rates()`, `ss.data.contacts()`, `ss.data.geography()`), which is not counted here, as `ss.geo`'s other names are not.

## The names

The names below are checked against the code: each one is exported by `starsim`, each method exists, and the counts in the headings and in [the total](#the-total) are made from one list in `tests/test_spec.py`, which a test compares with this page. To update the counts after changing the list, run `check_vocabulary(write=True)` from that file. `ss.` also exports names that are not counted: the classes and helpers for writing modules as code (the second tier, listed in `tests/test_spec.py`), and the base classes of the errors (`ss.CodedError`, `ss.StarsimWarning`, `ss.Note`), `ss.note`, and `ss.CapabilityError`. The other errors, warnings, and notes are in `ss.errors` (`ss.errors.UnknownState`), the testing and profiling tools in `ss.debugtools`, and everything else in its own submodule (`ss.solvers.Compartmental`, `ss.transitions.Expr`).

### Sim and modules: 12

| Name | What it is |
|---|---|
| `ss.Sim` | Modules, a population, mixing, a timeline, and a solver. One sim is one run |
| `ss.People` | The population: size, demography, per-agent attributes |
| **`ss.people`** | A named population: `ss.people('Kenya', year=2024)` returns an `ss.People` |
| **`ss.ppl`** | The people's properties before there are people: `ss.ppl.age < 15` is a condition, and `expit(b0 + b1*ss.ppl.urban)` a rate, with each person's value (each group's, as compartments) |
| `ss.Module` | The one module abstraction; everything below is one |
| `ss.Disease` | Any disease: declared states and the transitions between them, code (`define_states()`, `step()`), or both. Most states hold people; a reservoir or a vector population is a state that holds a count |
| **`ss.Strains`** | The strains of a disease, `ss.Disease(..., strains=ss.Strains(['DS', 'RIF'], ...))`, giving `dis.strains`; unpacked into strain objects (`DS, RIF = strains`) for conditions and `move=DS >> (DS, RIF)`. States are shared (one course, sets carried up to `max_strains`) or each strain's own (`shared=False`, or `shared=['Ca']`); strains can also be diseases of their own sharing a natural history (`ss.Strains([hpv16, hpv18])`) |
| `ss.Infection` | An `ss.Disease` with transmission written as code (`beta` over networks, `rel_sus`, `rel_trans`, `set_prognoses()`), as in Starsim 3 |
| `ss.Demographics` | Births, deaths, pregnancy, migration |
| `ss.Intervention` | Base class for interventions with custom `step()` code |
| `ss.Connector` | Base class for cross-module `step()` code |
| `ss.Analyzer` | Base class for extra results |

Starsim's library modules (`ss.SIR`, `ss.SIS`, `ss.SEIR`, `ss.Births`, `ss.Deaths`, `ss.LegacyPregnancy`, the `ss.infection_log` analyzer, and the rest) remain available and are not counted here; they are models written in the language, not part of it. Nor are the error, warning, and note classes (`ss.errors.UnknownState`, `ss.errors.ImportIncomplete`, and the rest), which are listed in [15-errors.md](15-errors.md): they are what the library says, not what a user writes.

### States, transitions, parameters, results: 14

| Name | What it is |
|---|---|
| **`ss.State`** | A state, with `people`, `init`, `rel_sus`, `rel_trans`, `label` |
| **`ss.states`** | Several states at once, named from the variables they are assigned to: `S, I, R = ss.states(3)`. `ss.states('S I R')` when the names are wanted up front |
| **`ss.null`** | Outside the model: the source of an inflow and the destination of an outflow (`ss.null >> W @ rate`, `I >> ss.null @ 0.1`) |
| **`ss.Transition`** | The one transition class; also built by `a >> b @ x` |
| **`ss.birth`** | Shorthand: an inflow of newborns, at a rate per person |
| **`ss.death`** | Shorthand: an outflow; `lifetable=` draws each person's age at death from a life table |
| **`ss.immigration`** | Shorthand: an inflow of a number of people per unit time, at ages like the population's |
| **`ss.emigration`** | Shorthand: an outflow, recorded as people who left rather than died |
| **`ss.lifetable`** | A life table (ages and deaths) for `ss.death(lifetable=...)`, or by year and sex for `ss.Deaths(lifetable=...)` |
| **`ss.transfer`** | Movement between levels of a stratification |
| **`ss.par`** | A named parameter with `bounds`, `prior`, `source`, `doc` |
| **`ss.pars`** | Several parameters at once, from defaults and the values given, for a disease whose transitions are made in `__init__()`: `p = ss.pars(defaults, **kwargs)` |
| **`ss.outcome`** | A reported quantity: a weighted sum of flows that is delayed, given a duration, or totaled by reporting period |
| **`ss.simdata`** | A data file or data frame with its columns matched to result names: `ss.simdata(df, aliases=..., ignore_extra=True)` |

The operators `>>`, `@`, `|`, and arithmetic on states are part of this group and add no names. A custom per-step function needs no name of its own: a plain function is given as a keyword, or written in `step()`.

### Time: 27

All Starsim's, and **`ss.peract`**, a probability per act, which is not a time value.

| Kind | Type | Constructors |
|---|---|---|
| Duration | `ss.dur` | `ss.days` `ss.weeks` `ss.months` `ss.years` `ss.datedur`, and one unit as `ss.day` `ss.week` `ss.month` `ss.year` |
| Rate (hazard) | `ss.per` | `ss.perday` `ss.perweek` `ss.permonth` `ss.peryear` |
| Probability per period | `ss.prob` | `ss.probperday` `ss.probperweek` `ss.probpermonth` `ss.probperyear` (or `ss.prob(0.1, 'day')`) |
| Frequency (an expected count) | `ss.freq` | `ss.freqperday` `ss.freqperweek` `ss.freqpermonth` `ss.freqperyear` |
| Calendar date | | `ss.date` |
| Probability per act, on a network with acts | **`ss.peract`** | `ss.peract(0.01)`, or one value for each of the network's directions, `ss.peract(m2f=0.3, f2m=0.15)` (also `{('m', 'f'): 0.3}`); no unit and no conversion |

That is four types, each with a constructor per unit, and `ss.date`: two choices (which kind, which unit), and a bare number works in every slot for anyone who does not want to use them. In a declared transition a frequency gives the same result as a rate ([02-types.md](02-types.md#conversion)); the difference matters to modules written as code, which convert them to a step themselves.

### Distributions: 16

Starsim's, with Starsim's names: `ss.Dist` (the base class, which also wraps any SciPy distribution) and `ss.normal`, `ss.lognorm_ex`, `ss.lognorm_im`, `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, `ss.choice`; and three new ones, **`ss.erlang`** (a number of exponential stages with a mean, which a compartmental run takes as stages), **`ss.truncnorm`**, and **`ss.binom`** (mostly as a likelihood). The rest of Starsim's set is available and not listed.

### Parameter forms: 3

**`ss.by`** (one value per stratum), **`ss.timeseries`** (varies in time), **`ss.seasonal`** (a yearly cycle). The other forms a parameter can take (a number, a typed quantity, a distribution, a function) need no names of their own.

### Mixing and space: 9

Everything spatial is in the submodule `ss.geo`.

| Name | What it is |
|---|---|
| `ss.Network` | Base class for networks |
| `ss.RandomNet` `ss.SexualNet` | Individual-level networks (Starsim's; its other networks are available too, and the library's, e.g. `ssl.StaticNet` for a NetworkX graph). `ss.SexualNet` is partnerships between two groups (`between=`): men and women (`'mf'`, the default), men (`'mm'`), or e.g. sex workers and clients (`('fsw', 'client')`), with acts per unit time in each, as in Starsim 3 (people with no partner are paired), or with a target number of partners at a time (`n_contacts`, `concurrency`, `max_contacts`). Partnerships between anyone (needle sharing), and EpiModel's targets (`match=`), are the library's `ssl.DegreeNet`; more structured sexual networks are STIsim's |
| `ss.MixingPool` `ss.MixingPools` | Group-level mixing; `ss.MixingPool(n_contacts=10)` with no groups is homogeneous |
| **`ss.contacts`** | A contact matrix as mixing pools: a named one, `ss.contacts('Kenya')`, or one given as a matrix (over several dimensions together with `dims=`) |
| **`ss.geo.geography`** | A geography: names, populations, centroids, distances; named (`ss.geo.geography('Kenya', level=1)`) or given directly |
| **`ss.geo.Mobility`** | Movement between places from a matrix, a data file of moves, or a movement model |
| **`ss.geo.gravity`** | Movement model (radiation, Stouffer, and competing destinations are deferred; a matrix covers them) |

### Interactions and interventions: 9

| Name | What it is |
|---|---|
| **`ss.modify`** | While a condition holds, multiply, add to, or set a named quantity |
| **`ss.change`** | A parameter change at given times; shorthand for a modifier |
| **`ss.immunity`** | Protection between strains, combined by a rule: what protects is `due_to=` (a state, an attribute, `strains.last`, `strains.ever`), the quantity it lowers is the keyword and the protection its value, a matrix (rows from, columns against) or a number (`ss.immunity(due_to=R, rel_sus=m, combine='sum')`, `rel_trans=`, `dur=dict(In=m)`, `p=dict(to_symp=m)`); for `ss.Strains(immunity=)` |
| **`ss.Vaccine`** **`ss.Diagnostic`** **`ss.Treatment`** | Products: nouns for things (Starsim 3's `ss.Vx`, `ss.Dx`, `ss.Tx` remain only as base classes for products written as code, not as aliases; `ss.Diagnostic` was `ss.Test`, renamed so that no product and delivery differ only by case) |
| **`ss.vaccinate`** **`ss.screen`** **`ss.treat`** | Deliveries: verbs (`ss.Delivery` with no product is a step people attend, a visit) |

### Running, comparing, saving: 7

`ss.MultiSim` (several runs of a sim, or several sims, whose results have a run dimension), `ss.parallel` (run them in parallel: `ss.parallel(sim, n_runs=100)` for replicates, with `sim.run(n_runs=100)` as its shortcut, and `ss.parallel(baseline=..., vaccine=...)` for scenarios), `ss.options` (Starsim's global options; `ss.options.crn = False` turns common random numbers off), **`ss.diff`** (what differs between two models), `ss.load` (a sim saved by `ss.save`, or a model in another framework's format), `ss.save` (a sim with its results, or a model as code with `ss.save('model.py', sim)`, or in another format), **`ss.compile`** (compile the compartmental solvers' kernels ahead of time, e.g. in a Docker image or before a calibration; until then, small runs are taken uncompiled, with the same results).

### Types you get back or subclass: 5

`ss.Result`, `ss.Loop`, `ss.Timeline`, `ss.Rate`, `ss.TimePar`, all Starsim's. Users rarely construct these.

Fitting is the submodule `ss.fit`, which `import starsim` does not import (its first use does), so its names are not counted here: `ss.fit.Fit`, what `sim.calibrate()` returns (called `fit` below), `ss.fit.compare_models` (fit and compare by out-of-sample prediction), and `ss.fit.optimize` (reallocate spending between programs to maximize or minimize a result). (Starsim 3's `ss.Calibration` is removed.)

### Methods and attributes: 43 distinct names

| On | Names |
|---|---|
| A model (`ss.Disease`, any module) | **`describe`** **`diagram`** **`equations`** **`transitions`** **`stratify`** **`to_spec`** `copy` `pars` **`R0`** |
| A state or a sum of states | **`n`** **`frac`** **`total`** |
| A modifiable array (`sir.rel_sus`) | **`effective`** |
| `ss.Sim` | `init` `run` `plot` `summarize` `to_df` `loop` `results` **`describe`** **`stratify`** **`to_spec`** **`to_code`** `to_json` **`hash`** **`calibrate`** **`sensitivity`** **`check_dt`** **`what_modifies`** **`new`** **`gradient`** |
| `sim.loop` | `insert` `to_df` `plot` `plot_step_order` |
| A result | **`select`** **`by`** **`cum`** `resample` `sum` |
| A result of an `ss.MultiSim`, which has a run dimension | **`quantiles`** `values` |
| An `ss.MultiSim`, which is what `ss.parallel` returns | `run` `plot` `summarize` `to_df` `results` **`diff`** |
| `fit` | **`sim`** `pars` `plot` `summarize` `to_df` **`project`** **`samples`** |
| What `ss.fit.compare_models` returns | **`fits`** `plot` `summarize` `to_df` |

Names that repeat across rows (`plot`, `summarize`, `to_df`, `describe`, `stratify`, `to_spec`, `pars`, `run`, `results`) are counted once. The split is deliberate: `describe`, `diagram`, `equations` are about the model; `plot`, `summarize`, `to_df` are about results, as in Starsim. `values` is Starsim's name for the array behind a result; on a MultiSim result it is the full runs-by-time array. `.n` is the number in a state in each place, and `.total` in the whole population; `.frac` is the share; `.dwell` is the time each person has spent in it (in `p=` and rates out of it). On a transition with several destinations, `.X` is the flow to its destination `X` (`cin.X`, read `res.cin.X`), not a new name. Reading `sir.rel_sus` gives the base value; `.effective` gives it with the active modifiers applied (agents only). `msim.check_runs()` is deferred and not counted.

## The total

<!-- counts start -->

| Group | Names | Of which new |
|---|---|---|
| Sim and modules | 12 | 3 |
| States, transitions, parameters, results | 14 | 14 |
| Time | 27 | 1 |
| Distributions | 16 | 3 |
| Parameter forms | 3 | 3 |
| Mixing and space | 9 | 4 |
| Interactions and interventions | 9 | 9 |
| Running, comparing, saving | 7 | 2 |
| Types you get back or subclass | 5 | 0 |
| **`ss.` names** | **102** | **39** |
| Methods and attributes | 43 | 28 |
| **Total** | **145** | **67** |

<!-- counts end -->

A simple SIR uses four of them: `ss.states`, `ss.Disease`, `ss.Sim`, and `run` (five with `plot`, and `ss.days` if the duration is typed). A stratified, calibrated national model with interventions uses perhaps forty.

The total is about the first draft's (134). What changed is what the names are: most of the `ss.` names are ones a Starsim user already knows, and the ones removed (below) were mostly new concepts, while the ones added are mostly methods for looking at a model.

## Keywords

Keywords are not counted above, but they are vocabulary too, so the main sets are listed here.

| On | Keywords |
|---|---|
| `ss.Sim` | the model (a module or a list of modules) as the first argument; `networks` `people` `total_pop` `n_agents` `start` `stop` `dt` `unit` `solver` `integrator` `substeps` `approx` `stratify` `rand_seed` `verbose` `interventions` `connectors` `analyzers` `data` `pars` `results` `use_aging` `label`; plus `diseases`, `demographics`, `modules` as in Starsim. `solver` may also be given to `sim.run` before the sim is initialized, and `n_runs` to `sim.run` |
| `ss.MultiSim`, `ss.parallel` | `n_runs` `pars` `priors`, and one keyword per scenario |
| `ss.Disease` | `name` `dt` `init_prev` `init` `init_n` `first` `transitions` `results` `outcomes`, and one keyword per member |
| `ss.State` | `name` `people` `init` `rel_sus` `rel_trans` `label` |
| `ss.Transition` | `rate` `dur` `p` `stages` `where` `move` `increment` `after` `before` `name` `label` |
| `ss.birth`, `ss.immigration` | `rate` |
| `ss.death`, `ss.emigration` | `rate` `dur`; `lifetable` (`ss.death`) |
| `ss.transfer` | the stratification's name (or `increment`, for a counter), `rate` `dur` `p` `states`, and the delivery keywords |
| `ss.par` | `bounds` `prior` `source` `doc` `literal` |
| `stratify` (on a model or a sim) | the dimension's name, `split` `adjust` `states` `where` `inplace` `mobility` `kind` (`'age'`, `'space'`, `'clock'`, `'plain'`) |
| `ss.Strains` | the strains (names, a dict of values by strain, or diseases), `shared` `immunity` `max_strains` `transmit` `split` `cross_immunity` `name` `label`, and values by parameter |
| `ss.immunity` | `due_to` `scale` `curve` `rel_sus` `rel_trans` `dur` `p` `combine` |
| `ss.modify` | `multiply` `add` `set` `efficacy` `share` `route` `where` `when` `start` `stop` `hold` `name` |
| `ss.change` | `at` `by` `to` |
| Deliveries | `what` `who` `operator` `start` `stop` `at` `interval` `at_age` `where` `coverage` `capacity` `deliver` `on` `when` `trace` `move` `increment` `disease` |
| Products | `efficacy` `waning` `cost` (all); `leaky` (`ss.Vaccine`); `sensitivity` `specificity` `delay` `detects` (`ss.Diagnostic`); `cure` (a number, or a dict by state) `transition` `to` (`ss.Treatment`) |
| `ss.outcome` | `delay` `duration` `likelihood` |
| `ss.simdata` | `aliases` `ignore_extra` |
| `ss.MixingPool` | `n_contacts` `src` `dst` `beta` |
| `ss.SexualNet` | `between` (`'mf'`, `'mm'`, or a pair of groups) `dur` `acts` `participation` `debut` `n_contacts` `concurrency` `max_contacts` `match` |
| `ss.geo.Mobility` | `matrix` `data` `kind` `away` |
| `ss.contacts` | `bands` `name` `source` `layer` `dims`, and a matrix for each stratification by its name |
| Networks | `sides` (and `net.directions`, which follow from them; `between` on `ss.SexualNet`) |
| `sim.calibrate` | `pars` `data` `weights` `likelihood` `gof` `method` `n_runs` `rand_seed`, and the method's own settings |
| `ss.fit.optimize` | `maximize` `minimize` `at` `start` `total` `spend` `bounds` |
| `ss.fit.compare_models` | `data` `holdout` `folds` |

## Slots and their types

A bare number in any of these slots takes the slot's type and the sim's unit ([02-types.md](02-types.md#bare-numbers)). In a sim where nothing has a unit, "per unit time" below is per timestep and a duration is a number of steps. Every slot also accepts the rest of the parameter ladder ([04-parameters.md](04-parameters.md#the-ladder)), including a function.

| Slot | A bare number is | Appears on |
|---|---|---|
| `beta=` | a rate per unit time with no network; a probability per contact on a network without acts; a probability per act on a network with acts (`ss.SexualNet`, `ssl.DegreeNet(acts=...)`), as `ss.peract`. A dict gives one value per network; directions are named with `ss.peract(m2f=..., f2m=...)` (a pair is an error) | `ss.SIR`, `ss.SIS`, `ss.SEIR` and the library's diseases |
| `rate=`, and a bare number after `@` | a rate (hazard) per unit time | `ss.Transition` and every shorthand constructor |
| `dur=` | a duration; a scalar means an exponential dwell time with that mean | `ss.Transition`, `ss.transfer`, networks |
| `p=` | a branch probability | `ss.Transition`, `ss.transfer` |
| `stages=` | a whole number of sub-stages | `ss.Transition` |
| `init=` | a share of people, up to 1 (a whole number too, with a warning to write a float); on the disease, a dict of these by state name | `ss.State`, `ss.Disease` |
| `init_n=` | a number of people; on the disease, a dict of these by state name | `ss.State`, `ss.Disease` |
| `init_prev=` | a fraction | `ss.Disease` |
| `n_contacts=` | contacts per unit time; on a sexual network, the mean number of partners at a time | `ss.MixingPool`, `ss.RandomNet`, `ss.SexualNet` |
| `acts=` | on lasting partnerships, an error (it would be acts per step): give acts per unit time, e.g. `ss.freqperweek(2)` | `ss.SexualNet` |
| `share=` | the share of acts (or contacts) a modifier acts on | `ss.modify` |
| `coverage=` | a fraction of the eligible people | deliveries, `ss.transfer` |
| `capacity=` | a number per round (or step), or a rate per unit time (required for routine delivery) | deliveries |
| `efficacy=`, `sensitivity=`, `specificity=` | a probability | products; `efficacy=` also on `ss.modify` (`multiply=1 - efficacy`) |
| `waning=` | a duration: the product is lost at an exponentially distributed time with that mean (a function of the time since delivery fades the efficacy instead, as agents only, and a new dose restarts it); other shapes are written as states ([08](08-interventions.md#products)) | products |
| `delay=`, `duration=` | a duration | `ss.outcome`, `ss.Diagnostic` |
| `interval=`, `away=` | a duration; `away` is a fraction of time | deliveries, `ss.geo.Mobility` |
| `multiply=`, `add=`, `set=`, `by=`, `to=` | a multiplier, or a value in the target's own type | `ss.modify`, `ss.change` |
| `at=`, `start=`, `stop=` | a time in the sim's unit, counted from the start (a calendar year only in a sim whose own `start` and `stop` are year-like numbers); dates are written as dates. A time outside the sim raises `ss.errors.TimeOutsideSim` | `ss.change`, `ss.modify`, deliveries |
| `bounds=` | a pair, in the unit of the parameter's value | `ss.par` |
| `cost=` | a plain number | products |

## Removed since the first draft

| Removed | Why | Use instead |
|---|---|---|
| `ss.Effect` | Renamed and split: scaling a quantity is one thing, complex logic another | `ss.modify`, or Python in `step()` |
| `ss.Expr`, and `ss.age`, `ss.sex`, `ss.uid`, `ss.time`, `ss.year`, `ss.weekday`, `ss.has`, `ss.scheduled` | A second set of names for things the sim already has | Conditions on the real attributes: `(ppl.age > 65) & ~sir.vaccinated`, or a function of the sim |
| `ss.Continuous` | A special kind of state and route for reservoirs | An ordinary `ss.State` that is not people, with counts (`W.n`) in rates |
| `ss.radiation`/`ss.stouffer`/`ss.competing_destinations`, `msim.check_runs()` | Deferred from v1: one use case each, and the vocabulary is smaller without them ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)) | A mobility matrix; `check_dt` |
| `ss.process` | A penalty-box name for custom code | A plain function as a keyword, or `step()` |
| `ss.Homogeneous` | Not needed | No network given, or `ss.MixingPool(n_contacts=10)` |
| `ss.Network(edges=)` | Starsim already has names | Starsim's networks; `ssl.StaticNet(graph=...)`, with a NetworkX graph or a function that makes one |
| `ss.Par`, `ss.Outcome` | Capitalization rule | `ss.par`, `ss.outcome` |
| `ss.cascade`, `observe()`, `Model.results()` | A separate observation step (`ss.cascade` came back as a group of steps, and was removed again on 2026-10-20: it ran as its steps would as a list) | Results declared as keywords with the model; the steps as a list |
| `ss.visit`, `ss.Resource`, `ss.share` | Delivery extras, removed 2026-10-20 with stock targets (`target=`, `above=`), shared resources (`uses=`), and `count_events=` | `ss.Delivery(who=...)` with no product; `capacity=` or `deliver=` as a function of the sim; an analyzer |
| `ss.N`, `ss.count` | Extra names for a sum and a keyword | `N = S + I + R`; `capacity=` |
| `ss.dist(...)` | Duplicate | `ss.Dist(dist=...)` |
| `ss.lognorm`, `ss.binomial`, `ss.neg_binomial`, `ss.rate` | Not Starsim's names | `ss.lognorm_ex` / `ss.lognorm_im`, Starsim's set, `ss.nbinom`, `ss.per` |
| `ss.exp_decay`, `ss.linear_decay` | Covered by existing forms | `waning=` as a duration, a function, or `ss.timeseries` |
| `ss.act`, `queue()`, `phase=`, `sim.substep()` | Queued updates and named phases are gone | Starsim's ordered loop, `sim.loop.insert()` |
| `ss.Stage`, `fit.apply()` | Simpler spellings | `calibrate(method=[...])`; `fit.sim` |
| `capabilities()`, `closure=`, `to_ode()`, `scale=` | The closure ladder and per-transition hybrid solvers are out of v1 | `describe()`, `solver=`, printed approximations |
| `jacobian()`, `fixed_points()`, `sim.equations()` | Symbolic analysis is not a v1 feature | `sir.equations()` |
| `summary()` on a model, `to_dict()` | `summarize` is about results in Starsim; the model as data is its model spec, and `to_json` is a sim's results | `describe()`, `to_spec()` |
| The dict tuple shorthand `dict(infection=('S -> I', 0.05))` | Ambiguous: it guesses the kind of transition from a bare value | The operator form |
| `ss.transmission`, `ss.progression` (arrow strings) | A second spelling of a transition, with states made from their names and the source of infection guessed | `S >> I @ beta * I`, `ss.Transition(I, R, dur=10)` |
| `rate=ss.on(...)`, `after=` an event, `undriven=` | Duplicates of `on=` (`on=event, dur=delay`), and a flag that only silenced a warning | `on=`; a transition with no driver |
| Weighted destinations (`S >> (p*A + (1-p)*B)`), `p=` a function giving a row | Duplicates of the list form of branches | `ss.Transition(S, [A, B], p=[p, 1 - p])`, a function for each destination |
| `ss.any` | A reduce of `\|` | `a \| b`, or a list in `who=` |
| `ss.Route` | Starsim's name; the base class is `ss.networks.Route`, not exported | `ss.Network`, passed with `networks=` |
| `ss.stratify(sir, ...)`, `ss.RandomExactNet`, `birth_rate=`/`death_rate=` on `ss.Sim`, a delivery's `years=`/`dates=`/`eligible=`, `track_writes=` | Duplicates of other spellings (D34, D36, D38, D39) | `sir.stratify(...)`, `ss.RandomNet(method='exact')`, `demographics=[ss.birth(...), ss.death(...)]`, `at=`/`where=`, `sim.what_modifies()` |
| `ss.explain`, `ss.conformance`, `mixing=`, `n=`, `route=`, `ontology=`, `kind=` on parameters, `take=`/`doses=`, `min=`/`max=` clamps, `crn_strict`, `run(force=)`, `run(reset=)` | Small things that did not earn a name | Documentation pages; `networks=`; `total_pop=`; inline provenance in `ss.par(source=)` |
| An `'ide'` solver | Deferred | `stages=`, or agents |

The second draft's names that went in this revision:

| Removed | Why | Use instead |
|---|---|---|
| `ss.Compartments` | A second module class for states that are not people. The states use the same syntax as the disease's, and the name was confusing, since `S`, `I`, and `R` are compartments too | An ordinary `ss.State` in the disease; `people=False` is inferred, or given |
| `None` as a source or sink (`None >> W`) | Python evaluates `None @ 0.1` before it sees the arrow | `ss.null` |
| `method=` on `ss.Sim` and `sim.run`, `methods()`, `'ctmc'` as the name | "Solver" is the word; `method=` stays on `sim.calibrate` for the fitting method | `solver=`, `'gillespie'`; `integrator=` for the ODE scheme |
| `n_reps` on `ss.Sim`, `.trajectories`, `check_reps` | A sim is one run | `ss.MultiSim(sim, n_runs=100)` or `ss.parallel(sim, n_runs=100)`; `.values` and `.quantiles` on its results; `check_runs` (deferred) |
| `crn_key` | No case found where keying common random numbers on agent attributes beats Starsim's slots | Slots; `ss.options.crn = False` to turn CRN off |
| `ss.export` | A model is a Python script, which is already the thing to share | The script, which `ss.save('model.py', sim)` writes (`sim.to_code()`); `ss.save` of the sim |
| `ss.Needs`, `sim.needs()` (later `ss.Requires`, `sim.requires()`) | Metadata that was never enforced and that no model used (removed 2026-10-21) | A docstring; `doc=` and `source=` on `ss.par` |
| `ss.People('Kenya')` | A loader is lowercase, like `ss.contacts` | `ss.people('Kenya')` |
| `.sel()` | Spelled out | `.select()`, and `.by()` for one series per group |
| `sim.transmissions` | The log is large and slows the run, so it is recorded on request | `analyzers='infection_log'`, Starsim's analyzer |
| `mean_degree` | Starsim's name | `n_contacts`, which may be a distribution |
| `min_duration=` | A policy with memory is fragile as a keyword | `when=` as a function of the sim; `step()` or an intervention class for anything that has to remember |
| `p=` on `ss.outcome` | A number times a flow already means that fraction of it | `ss.outcome(0.5 * infection)` |
| `summary()` | Starsim's name | `summarize()` |

Since the review of October 2026, these names went, or were promised and never built:

| Removed | Why | Use instead |
|---|---|---|
| `ss.update` | A coded module does it: a connector's `step()` keeps the value, read by a function in a rate | A connector, or `step()` |
| `ss.clock()` | `(A \| B).time` is the same clock | `(A \| B).time` |
| `X.dwell` | The same clock as `X.time`, half a step later | `X.time` |
| `ss.spread` | Back-dating a module's start is a longer run; only pregnancy's burn-in keeps it, inside the module | Start the sim earlier |
| `offspring=` | A birth in the same disease does it | `ss.birth(NB, per=farrowing, n=3)` |
| `ss.func` | Never built; a plain function given as a keyword already does what it was for | A plain function, or `step()` |
| `sim.crn_report()` | Never built | `ss.options.crn`, and paired runs ([11-random.md](11-random.md)) |
| `dependence=` and `n_ref=` on `ss.MixingPool` | Never built; density dependence is a count in the rate | `beta * I.n` |
| `.prev` on a state | The code's name is `.frac` | `.frac`, and `.n` or `.total` for counts |
| `.base` on a modifiable array | Reading the array gives the base value | `sir.rel_sus` for the base, `sir.rel_sus.effective` for the value with modifiers applied |
| `ss.rate`, `ss.time_prob`, `ss.rate_prob` | Starsim 2 names, deprecated in Starsim 3 (Starsim's other time names are kept) | `ss.peryear(x)`, `ss.probperyear(x)` |
| `solver='sde'`, `'stochastic'`, `'ctmc'` | One solver per method | `'gillespie'` (exact), `'tau'` (tau-leaping), `'binomial'` |
| `ss.Calibration` as the result of a fit | `sim.calibrate()` returns an `ss.fit.Fit` | `ss.fit.Fit` (Starsim 3's `ss.Calibration` is removed) |
| `ss.from_spec()`, and `ss.load()` of a model spec | The spec is the normalized form for `sim.hash()` and `ss.diff()`, and is not read back | `ss.save('model.py', sim)` and `ss.save('run.sim', sim)` |
| `msim.reduce()`, `msim.mean()`, `msim.median()`, `ss.MultiResult`, `ss.multi_run()` | A MultiSim is its results, with a run dimension | `msim.results.sir.n_I.quantiles()` (and `.median()`, `.mean()`); `ss.parallel(pars=...)` |
| `ss.outcome(every=)`, `annualize()` | Resampling belongs to a result | `result.resample('year')` |
| `ss.routine_vx`, `ss.campaign_vx`, `ss.simple_vx`, `ss.routine_screening`, `ss.campaign_screening`, `ss.routine_triage`, `ss.campaign_triage`, `ss.seasonality`, `ss.rate`, `ss.time_prob`, `ss.rate_prob` | Starsim 3 and 2 duplicates of the language | Products with deliveries, `ss.seasonal`, `ss.per` and `ss.prob`; each is mapped in `skills/migrate-starsim3` |

## Adding to the vocabulary

There is no fixed budget. Before adding a name, check:

- **Does Starsim already have a name for it?** Use that one.
- **Can it be written with what exists?** A bednet is an `ss.modify`; a metapopulation is a stratification by patch; a junction is a branching transition; a reservoir is a state that is not people.
- **Would more than one kind of model use it?** Something only one framework's specialty needs probably belongs in a library built on top.
- **Does it make some real model shorter than it is in its original framework?** That is the test the whole design is held to.
- **Is a keyword enough?** Routine and campaign delivery are keywords on one function, not two classes.

When a name is added or removed, update the list in `tests/test_spec.py` and this document; `check_vocabulary(write=True)` there rewrites the counts, and `test_vocabulary` fails until both agree with the code.

## Not in the language

Each of these exists in at least one reviewed framework. The reasons are in [best-practices/principles.md](../best-practices/principles.md).

| Not included | Instead |
|---|---|
| A standalone modeling language with its own syntax | Python objects |
| Models written in YAML, JSON, or spreadsheets | Python; the model spec is produced from it |
| Many named flow kinds | One transition class whose kind follows from its rate |
| A separate class per solver | `solver=` |
| A sim that is many runs | `ss.MultiSim` |
| A default time unit | Timesteps, when nothing in the model has a unit |
| A separate metapopulation model class | A stratification by patch |
| A special route, or a separate module class, for environmental or vector-borne transmission | Counts in rates, and ordinary states that are not people |
| A class per combination of delivery mode and product | Keywords |
| A rule language for triggers and targeting | Conditions and functions in Python |
| Overwrite adjustments in stratification | Multipliers |
| Positional or integer references to states | Names |
| Required declarations of what a module reads and writes | `sim.what_modifies()`, which lists the declared modifiers |
| Common random numbers keyed on agent attributes | Starsim's slots |
