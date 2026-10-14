# The complete vocabulary

Every public name the specification uses, in one place, with a count.

Justified by [best-practices/principles.md](../best-practices/principles.md): a vocabulary nobody can hold in their head is a vocabulary nobody uses correctly.

The count is here so that the size of the language is something that can be looked at. The base is Starsim, so most of the names are Starsim's own and are marked as such; names this specification adds are in **bold**.

## Capitalization

The rule is Starsim's existing one (`ss.Dist` and `ss.normal`, `ss.Rate` and `ss.perday`):

> **Modules, containers, and base classes are CamelCase. The concrete pieces you put inside a module are lowercase.**

So `ss.Sim`, `ss.MultiSim`, `ss.Disease`, `ss.RandomNet`, `ss.MixingPool`, and `ss.Intervention` are CamelCase, along with the two graph classes `ss.State` and `ss.Transition` and the products `ss.Vaccine`, `ss.Test`, `ss.Treatment`. Time values, distributions, `ss.par`, `ss.states`, `ss.null`, the transition constructors, `ss.modify`, `ss.outcome`, the parameter forms, the deliveries, and functions such as `ss.parallel` and `ss.stratify` are lowercase. So are the loaders, which look up named data and return an ordinary object: `ss.people('Kenya')` returns an `ss.People`, and `ss.contacts('Kenya')` a set of `ss.MixingPools`.

## The names

### Sim and modules: 9

| Name | What it is |
|---|---|
| `ss.Sim` | Modules, a population, mixing, a timeline, and a solver. One sim is one run |
| `ss.People` | The population: size, demography, per-agent attributes |
| **`ss.people`** | A named population: `ss.people('Kenya', year=2024)` returns an `ss.People` |
| `ss.Module` | The one module abstraction; everything below is one |
| `ss.Disease` | States and the transitions between them. Most states hold people; a reservoir or a vector population is a state that holds a count |
| `ss.Demographics` | Births, deaths, pregnancy, migration |
| `ss.Intervention` | Base class for interventions with custom `step()` code |
| `ss.Connector` | Base class for cross-module `step()` code |
| `ss.Analyzer` | Base class for extra results |

Starsim's library modules (`ss.SIR`, `ss.Births`, `ss.Deaths`, `ss.Pregnancy`, the `ss.infection_log` analyzer, and the rest) remain available and are not counted here; they are models written in the language, not part of it.

### States, transitions, parameters, results: 14

| Name | What it is |
|---|---|
| **`ss.State`** | A state, with `people`, `init`, `infectious`, `rel_sus`, `rel_trans`, `label` |
| **`ss.states`** | Several states at once, named from the variables they are assigned to: `S, I, R = ss.states(3)`. `ss.states('S I R')` when the names are wanted up front |
| **`ss.null`** | Outside the model: the source of an inflow and the destination of an outflow (`ss.null >> W @ rate`, `I >> ss.null @ 0.1`) |
| **`ss.Transition`** | The one transition class; also built by `a >> b @ x` |
| **`ss.transmission`** | Shorthand: an arrow string with `beta=` and optionally `source=` |
| **`ss.progression`** | Shorthand: an arrow string with `dur=` or `rate=`, optionally `p=` |
| **`ss.birth`** | Shorthand: an inflow |
| **`ss.death`** | Shorthand: an outflow |
| **`ss.transfer`** | Movement between coordinates of a stratification |
| **`ss.par`** | A named parameter with `bounds`, `prior`, `source`, `doc` |
| **`ss.outcome`** | A reported quantity: a weighted sum of flows that is delayed, given a duration, or totaled by reporting period |
| **`ss.data`** | A data file or data frame with its columns matched to result names: `ss.data(df, aliases=..., ignore_extra=True)` |
| **`ss.func`** | Attaches options to a custom per-step function; rarely needed |
| **`ss.stratify`** | `ss.stratify(sir, age=...)`, the same as `sir.stratify(age=...)` |

The operators `>>`, `@`, `|`, and arithmetic on states are part of this group and add no names.

### Time: 22

All Starsim's, unchanged.

| Kind | Type | Constructors |
|---|---|---|
| Duration | `ss.dur` | `ss.days` `ss.weeks` `ss.months` `ss.years` `ss.datedur` |
| Rate (hazard) | `ss.per` | `ss.perday` `ss.perweek` `ss.permonth` `ss.peryear` |
| Probability per period | `ss.prob` | `ss.probperday` `ss.probperweek` `ss.probpermonth` `ss.probperyear` |
| Frequency (expected count per time) | `ss.freq` | `ss.freqperday` `ss.freqperweek` `ss.freqpermonth` `ss.freqperyear` |
| Calendar date | | `ss.date` |

That is 4 types, 17 constructors, and `ss.date`. It is the largest block, but it is two choices (which of four kinds, which unit), not 22 things to memorize, and a bare number works in every slot for anyone who does not want to use them.

### Distributions: 13

All Starsim's, with Starsim's names: `ss.Dist` (the base class, which also wraps any SciPy distribution) and `ss.normal`, `ss.lognorm_ex`, `ss.lognorm_im`, `ss.expon`, `ss.gamma`, `ss.weibull`, `ss.poisson`, `ss.nbinom`, `ss.uniform`, `ss.bernoulli`, `ss.constant`, `ss.choice`. The rest of Starsim's set is available and not listed.

### Parameter forms: 3

**`ss.by`** (one value per stratum), **`ss.timeseries`** (varies in time), **`ss.seasonal`** (a yearly cycle). The other forms a parameter can take (a number, a typed quantity, a distribution, a function) need no names of their own.

### Mixing and space: 11

| Name | What it is |
|---|---|
| `ss.Network` | Base class for networks |
| `ss.RandomNet` `ss.MFNet` `ss.StaticNet` | Individual-level networks (Starsim's; its other networks are available too). `ss.MFNet` gains two keywords, `n_contacts` and `concurrency` |
| `ss.MixingPool` `ss.MixingPools` | Group-level mixing; `ss.MixingPool(n_contacts=10)` with no groups is homogeneous |
| **`ss.PartnerNet`** | Partnerships that form and dissolve, described by targets for the number of partners at a time (`n_contacts`, `concurrency`, `max_contacts`) and their duration |
| **`ss.contacts`** | A named contact matrix as mixing pools: `ss.contacts('Kenya')` |
| **`ss.geo`** | A named geography: names, populations, centroids, distances |
| **`ss.Mobility`** | Movement between patches from a matrix, a data file, or a movement model |
| **`ss.gravity`** | Movement model (radiation, Stouffer, and competing destinations are deferred; a matrix covers them) |

### Interactions and interventions: 8

| Name | What it is |
|---|---|
| **`ss.modify`** | While a condition holds, multiply, add to, or set a named quantity |
| **`ss.change`** | A parameter change at given times; shorthand for a modifier |
| **`ss.Vaccine`** **`ss.Test`** **`ss.Treatment`** | Products (Starsim's `ss.Vx`, `ss.Dx`, `ss.Tx`, which remain as aliases) |
| **`ss.vaccinate`** **`ss.screen`** **`ss.treat`** | Deliveries |

### Running, comparing, saving: 7

`ss.MultiSim` (several runs of a sim, or several sims), `ss.parallel` (run them in parallel: `ss.parallel(sim, n_runs=100)` for replicates, `ss.parallel(baseline=..., vaccine=...)` for scenarios), `ss.options` (Starsim's global options; `ss.options.crn = False` turns common random numbers off), **`ss.compare_models`** (fit and compare by out-of-sample prediction), **`ss.diff`** (what differs between two sims), `ss.load`, `ss.save`.

### Optional metadata: 1

**`ss.Requires`**: one entry in a module's `requires` list, saying what data the module needs (`ss.Requires('population', shape='patch', source='UN WPP 2024')`). `sim.requires()` prints the table. It is never enforced. Generating a data-entry form from it (`sim.requires(to='form.csv')`) is deferred ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)).

### Types you get back or subclass: 6

`ss.Result`, `ss.Loop`, `ss.Timeline`, `ss.Rate`, `ss.TimePar`, `ss.Calibration`. All Starsim's. Users rarely construct these; `sim.calibrate()` returns the calibration object that this specification calls `fit`.

### Methods and attributes: 39 distinct names

| On | Names |
|---|---|
| A model (`ss.Disease`, any module) | **`describe`** **`diagram`** **`equations`** **`solvers`** **`transitions`** **`stratify`** `to_json` `copy` `pars` **`R0`** |
| A state or a sum of states | **`n`** **`prev`** |
| A modifiable array | **`base`** |
| `ss.Sim` | `init` `run` `plot` `summarize` `to_df` `loop` `results` **`describe`** **`solvers`** **`stratify`** **`requires`** `to_json` **`hash`** **`calibrate`** **`sensitivity`** **`check_dt`** **`what_modifies`** **`crn_report`** |
| `sim.loop` | `insert` `to_df` `plot` `plot_step_order` |
| A result | **`select`** **`by`** **`sum`** `resample` |
| A result of an `ss.MultiSim`, which has a run dimension | **`quantiles`** `values` |
| A network | **`check`** |
| An `ss.MultiSim`, which is what `ss.parallel` returns | `run` `plot` `summarize` `to_df` `results` **`diff`** |
| `fit` | **`sim`** `pars` `plot` `summarize` `to_df` |
| What `ss.compare_models` returns | **`fits`** `plot` `summarize` `to_df` |

Names that repeat across rows (`plot`, `summarize`, `to_df`, `describe`, `solvers`, `stratify`, `to_json`, `pars`, `run`, `results`) are counted once. The split is deliberate: `describe`, `diagram`, `equations`, `solvers` are about the model; `plot`, `summarize`, `to_df` are about results, as in Starsim. `R0` and `crn_report` are optional. `values` is Starsim's name for the array behind a result; on a MultiSim result it is the full runs-by-time array. `msim.check_runs()` is deferred and not counted.

## The total

| Group | Names | Of which new |
|---|---|---|
| Sim and modules | 9 | 1 |
| States, transitions, parameters, results | 14 | 14 |
| Time | 22 | 0 |
| Distributions | 13 | 0 |
| Parameter forms | 3 | 3 |
| Mixing and space | 11 | 5 |
| Interactions and interventions | 8 | 8 |
| Running, comparing, saving | 7 | 2 |
| Optional metadata | 1 | 1 |
| Types you get back or subclass | 6 | 0 |
| **`ss.` names** | **94** | **34** |
| Methods and attributes | 39 | 25 |
| **Total** | **133** | **59** |

The two-line SIR uses five of them: `ss.Disease`, `ss.transmission`, `ss.progression`, `ss.Sim`, and `run` (six with `plot`). The operator form swaps the two constructors for `ss.states` and adds `ss.days` if the duration is typed. A stratified, calibrated national model with interventions uses perhaps forty.

The total is about the first draft's (134). What changed is what the names are: 60 of the 93 `ss.` names are ones a Starsim user already knows, and the ones removed (below) were mostly new concepts, while the ones added are mostly methods for looking at a model.

## Keywords

Keywords are not counted above, but they are vocabulary too, so the main sets are listed here.

| On | Keywords |
|---|---|
| `ss.Sim` | `networks` `people` `total_pop` `n_agents` `start` `stop` `dt` `unit` `solver` `integrator` `approx` `stratify` `rand_seed` `verbose` `interventions` `connectors` `analyzers` `data` `pars` `results` `track_writes` `label`; plus `diseases`, `demographics`, `modules` as in Starsim. `solver` may also be given to `sim.run` |
| `ss.MultiSim`, `ss.parallel` | `n_runs`, and one keyword per scenario; `lines` on `msim.plot` |
| `ss.Disease` | `name` `dt` `init_prev` `init` `transitions` `results` `outcomes`, and one keyword per member |
| `ss.State` | `name` `people` `init` `infectious` `rel_sus` `rel_trans` `label` |
| `ss.Transition` | `rate` `dur` `p` `stages` `name` `label` |
| `ss.transmission` | `beta` `source` `p` |
| `ss.progression`, `ss.birth`, `ss.death` | `rate` `dur` `p` `stages` |
| `ss.transfer` | the stratification's name, `rate` `dur` `p` `states`, and the delivery keywords |
| `ss.par` | `bounds` `prior` `source` `doc` |
| `stratify` (on a model or a sim) | the dimension's name, `split` `adjust` `states` `where` `inplace` `mobility` `immunity` `kind` |
| `ss.modify` | `multiply` `add` `set` `where` `when` `start` `stop` `name` |
| `ss.change` | `at` `by` `to` |
| Deliveries | `start` `stop` `years` `dates` `interval` `eligible` `coverage` `capacity` `deliver` `on` `when` `on_positive` `on_negative` |
| Products | `efficacy` `waning` `cost` (all); `sensitivity` `specificity` `delay` `detects` (`ss.Test`); `transition` (`ss.Treatment`) |
| `ss.outcome` | `delay` `duration` `every` `likelihood` |
| `ss.data` | `aliases` `ignore_extra` |
| `ss.Requires` | `shape` `source` |
| `ss.MixingPool` | `n_contacts` `dependence` `n_ref` |
| `ss.MFNet` | `n_contacts` `concurrency`, added to Starsim's `duration` `acts` `participation` `debut` |
| `ss.Mobility` | `matrix` `data` `kind` `away` |
| `sim.calibrate` | `pars` `data` `weights` `likelihood` `gof` `method` `n_runs` `rand_seed` |
| `ss.compare_models` | `data` `holdout` `folds` |

## Slots and their types

A bare number in any of these slots takes the slot's type and the sim's unit ([02-types.md](02-types.md#bare-numbers)). In a sim where nothing has a unit, "per unit time" below is per timestep and a duration is a number of steps. Every slot also accepts the rest of the parameter ladder ([04-parameters.md](04-parameters.md#the-ladder)), including a function.

| Slot | A bare number is | Appears on |
|---|---|---|
| `beta=` | a rate per unit time with no network; a probability per contact with one. A dict gives one value per network, and a pair one per direction | `ss.transmission` |
| `rate=`, and a bare number after `@` | a rate (hazard) per unit time | `ss.Transition` and every shorthand constructor |
| `dur=` | a duration; a scalar means an exponential dwell time with that mean | `ss.Transition`, `ss.progression`, `ss.transfer`, networks |
| `p=` | a branch probability | `ss.Transition`, `ss.progression`, `ss.transmission`, `ss.transfer` |
| `stages=` | a whole number of sub-stages | `ss.Transition`, `ss.progression` |
| `init=` | below 1 a fraction, above 1 a count; on the disease, a dict of these by state name | `ss.State`, `ss.Disease` |
| `init_prev=` | a fraction | `ss.Disease` |
| `n_contacts=` | contacts per unit time; on a sexual network, the mean of a Poisson number of partners at a time | `ss.MixingPool`, `ss.RandomNet`, `ss.MFNet` |
| `coverage=` | a fraction of the eligible people | deliveries, `ss.transfer` |
| `capacity=` | a number per round, or per unit time for routine delivery | deliveries |
| `efficacy=`, `sensitivity=`, `specificity=` | a probability | products |
| `waning=` | a duration (exponential decay with that mean) | products |
| `delay=`, `duration=`, `every=` | a duration | `ss.outcome`, `ss.Test` |
| `interval=`, `away=` | a duration; `away` is a fraction of time | deliveries, `ss.Mobility` |
| `multiply=`, `add=`, `set=`, `by=`, `to=` | a multiplier, or a value in the target's own type | `ss.modify`, `ss.change` |
| `at=` | a year (dates and durations are written as such) | `ss.change` |
| `bounds=` | a pair, in the unit of the parameter's value | `ss.par` |
| `cost=` | a plain number | products |

## Removed since the first draft

| Removed | Why | Use instead |
|---|---|---|
| `ss.Effect` | Renamed and split: scaling a quantity is one thing, complex logic another | `ss.modify`, or Python in `step()` |
| `ss.Expr`, and `ss.age`, `ss.sex`, `ss.uid`, `ss.time`, `ss.year`, `ss.weekday`, `ss.has`, `ss.scheduled` | A second set of names for things the sim already has | Conditions on the real attributes: `(ppl.age > 65) & ~sir.vaccinated`, or a function of the sim |
| `ss.Continuous` | A special kind of state and route for reservoirs | An ordinary `ss.State` that is not people, with counts (`W.n`) in rates |
| `ss.radiation`/`ss.stouffer`/`ss.competing_destinations`, `msim.check_runs()`, `sim.requires(to=)` | Deferred from v1: one use case each, and the vocabulary is smaller without them ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)) | A mobility matrix; `check_dt`; `sim.requires()` |
| `ss.process` | A penalty-box name for custom code | A plain function as a keyword, or `step()` |
| `ss.Homogeneous` | Not needed | No network given, or `ss.MixingPool(n_contacts=10)` |
| `ss.SexualNet`, `ss.Network(edges=)` | Starsim already has names | `ss.MFNet` and Starsim's other networks; `ss.StaticNet(edges=...)` |
| `ss.Par`, `ss.Outcome` | Capitalization rule | `ss.par`, `ss.outcome` |
| `ss.cascade`, `observe()`, `Model.results()` | A separate observation step | Results declared as keywords with the model |
| `ss.N`, `ss.count` | Extra names for a sum and a keyword | `N = S + I + R`; `capacity=` |
| `ss.dist(...)` | Duplicate | `ss.Dist(dist=...)` |
| `ss.lognorm`, `ss.binomial`, `ss.neg_binomial`, `ss.rate` | Not Starsim's names | `ss.lognorm_ex` / `ss.lognorm_im`, Starsim's set, `ss.nbinom`, `ss.per` |
| `ss.exp_decay`, `ss.linear_decay` | Covered by existing forms | `waning=` as a duration, a function, or `ss.timeseries` |
| `ss.act`, `queue()`, `phase=`, `sim.substep()` | Queued updates and named phases are gone | Starsim's ordered loop, `sim.loop.insert()` |
| `ss.Stage`, `fit.apply()` | Simpler spellings | `calibrate(method=[...])`; `fit.sim` |
| `capabilities()`, `closure=`, `to_ode()`, `scale=` | The closure ladder and per-transition hybrid solvers are out of v1 | `solvers()`, `solver=`, printed approximations |
| `jacobian()`, `fixed_points()`, `sim.equations()` | Symbolic analysis is not a v1 feature | `sir.equations()` |
| `summary()` on a model, `to_dict()` | `summarize` is about results in Starsim; `to_json` is Starsim's name | `describe()`, `to_json()` |
| The dict tuple shorthand `dict(infection=('S -> I', 0.05))` | Ambiguous: it guesses the kind of transition from a bare value | The two-line constructor form |
| `ss.Route` | Starsim's name | `ss.Network`, passed with `networks=` |
| `ss.explain`, `ss.conformance`, `mixing=`, `n=`, `route=`, `ontology=`, `kind=` on parameters, `take=`/`doses=`, `min=`/`max=` clamps, `crn_strict`, `run(force=)`, `run(reset=)` | Small things that did not earn a name | Documentation pages; `networks=`; `total_pop=`; inline provenance in `ss.par(source=)` |
| An `'ide'` solver | Deferred | `stages=`, or agents |

The second draft's names that went in this revision:

| Removed | Why | Use instead |
|---|---|---|
| `ss.Compartments` | A second module class for states that are not people. The states use the same syntax as the disease's, and the name was confusing, since `S`, `I`, and `R` are compartments too | An ordinary `ss.State` in the disease; `people=False` is inferred, or given |
| `None` as a source or sink (`None >> W`) | Python evaluates `None @ 0.1` before it sees the arrow | `ss.null` |
| `method=` on `ss.Sim` and `sim.run`, `methods()`, `'ctmc'` as the name | "Solver" is the word; `method=` stays on `sim.calibrate` for the fitting method | `solver=`, `solvers()`, `'gillespie'` (with `'ctmc'` as an alias); `integrator=` for the ODE library |
| `n_reps` on `ss.Sim`, `.trajectories`, `check_reps` | A sim is one run | `ss.MultiSim(sim, n_runs=100)` or `ss.parallel(sim, n_runs=100)`; `.values` and `.quantiles` on its results; `check_runs` (deferred) |
| `crn_key` | No case found where keying common random numbers on agent attributes beats Starsim's slots | Slots; `ss.options.crn = False` to turn CRN off |
| `ss.export` | A model is a Python script, which is already the thing to share | The script; `to_json()`, `ss.save` |
| `ss.Needs`, `sim.needs()` | One word for data requirements | `requires = [ss.Requires(...)]`, `sim.requires()` |
| `ss.People('Kenya')` | A loader is lowercase, like `ss.contacts` | `ss.people('Kenya')` |
| `.sel()` | Spelled out | `.select()`, and `.by()` for one series per group |
| `sim.transmissions` | The log is large and slows the run, so it is recorded on request | `analyzers='infection_log'`, Starsim's analyzer |
| `mean_degree` | Starsim's name | `n_contacts`, which may be a distribution |
| `min_duration=` | A policy with memory is fragile as a keyword | `when=` as a function of the sim; `step()` or an intervention class for anything that has to remember |
| `p=` on `ss.outcome` | A number times a flow already means that fraction of it | `ss.outcome(0.5 * infection)` |
| `summary()` | Starsim's name | `summarize()` |

## Adding to the vocabulary

There is no fixed budget. Before adding a name, check:

- **Does Starsim already have a name for it?** Use that one.
- **Can it be written with what exists?** A bednet is an `ss.modify`; a metapopulation is a stratification by patch; a junction is a branching transition; a reservoir is a state that is not people.
- **Would more than one kind of model use it?** Something only one framework's specialty needs probably belongs in a library built on top.
- **Does it make some real model shorter than it is in its original framework?** That is the test the whole design is held to.
- **Is a keyword enough?** Routine and campaign delivery are keywords on one function, not two classes.

When a name is added or removed, update this document and its count.

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
| Required declarations of what a module reads and writes | Writes observed when `track_writes=True`; `sim.what_modifies()` |
| Common random numbers keyed on agent attributes | Starsim's slots |
| Budget optimization | `cost=` now; optimization later |
