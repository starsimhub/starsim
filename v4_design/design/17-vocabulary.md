# The complete vocabulary

Every public name the specification uses, in one place, with a count.

Justified by [best-practices/principles.md](../best-practices/principles.md): a vocabulary nobody can hold in their head is a vocabulary nobody uses correctly.

The count is here so that the size of the language is something that can be looked at. The base is Starsim, so most of the names are Starsim's own and are marked as such; names this specification adds are in **bold**.

## Capitalization

The rule is Starsim's existing one (`ss.Dist` and `ss.normal`, `ss.Rate` and `ss.perday`):

> **Modules, containers, and base classes are CamelCase. The concrete pieces you put inside a module are lowercase.**

So `ss.Sim`, `ss.Disease`, `ss.RandomNet`, `ss.MixingPool`, and `ss.Intervention` are CamelCase, along with the two graph classes `ss.State` and `ss.Transition` and the products `ss.Vaccine`, `ss.Test`, `ss.Treatment`. Time values, distributions, `ss.par`, `ss.states`, the transition constructors, `ss.modify`, `ss.outcome`, the parameter forms, the deliveries, and functions such as `ss.parallel` and `ss.stratify` are lowercase.

## The names

### Sim and modules: 9

| Name | What it is |
|---|---|
| `ss.Sim` | Modules, a population, mixing, a timeline, and a run method |
| `ss.People` | The population: size, demography, per-agent attributes |
| `ss.Module` | The one module abstraction; everything below is one |
| `ss.Disease` | States people are in, and the transitions between them |
| **`ss.Compartments`** | States that are counts (reservoirs, vectors, wastewater), with the same transition syntax |
| `ss.Demographics` | Births, deaths, pregnancy, migration |
| `ss.Intervention` | Base class for interventions with custom `step()` code |
| `ss.Connector` | Base class for cross-module `step()` code |
| `ss.Analyzer` | Base class for extra results |

Starsim's library modules (`ss.SIR`, `ss.Births`, `ss.Deaths`, `ss.Pregnancy`, and the rest) remain available and are not counted here; they are models written in the language, not part of it.

### States, transitions, parameters, results: 12

| Name | What it is |
|---|---|
| **`ss.State`** | A state, with `init`, `infectious`, `rel_sus`, `rel_trans`, `label` |
| **`ss.states`** | Several states from one string: `S, I, R = ss.states('S I R')` |
| **`ss.Transition`** | The one transition class; also built by `a >> b @ x` |
| **`ss.transmission`** | Shorthand: an arrow string with `beta=` and optionally `source=` |
| **`ss.progression`** | Shorthand: an arrow string with `dur=` or `rate=`, optionally `p=` |
| **`ss.birth`** | Shorthand: an inflow |
| **`ss.death`** | Shorthand: an outflow |
| **`ss.transfer`** | Movement between coordinates of a stratification |
| **`ss.par`** | A named parameter with `bounds`, `prior`, `source`, `doc` |
| **`ss.outcome`** | A reported quantity: a thinned, delayed flow |
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

### Mixing and space: 13

| Name | What it is |
|---|---|
| `ss.Network` | Base class for networks |
| `ss.RandomNet` `ss.MFNet` `ss.StaticNet` | Individual-level networks (Starsim's; its other networks are available too) |
| `ss.MixingPool` `ss.MixingPools` | Group-level mixing; `ss.MixingPool(n_contacts=10)` with no groups is homogeneous |
| **`ss.contacts`** | A named contact matrix as mixing pools: `ss.contacts('Kenya')` |
| **`ss.geo`** | A named geography: names, populations, centroids, distances |
| **`ss.Mobility`** | Movement between patches from a matrix, a data file, or a movement model |
| **`ss.gravity`** **`ss.radiation`** **`ss.stouffer`** **`ss.competing_destinations`** | Movement models |

### Interactions and interventions: 8

| Name | What it is |
|---|---|
| **`ss.modify`** | While a condition holds, multiply, add to, or set a named quantity |
| **`ss.change`** | A parameter change at given times; shorthand for a modifier |
| **`ss.Vaccine`** **`ss.Test`** **`ss.Treatment`** | Products (Starsim's `ss.Vx`, `ss.Dx`, `ss.Tx`, which remain as aliases) |
| **`ss.vaccinate`** **`ss.screen`** **`ss.treat`** | Deliveries |

### Running, comparing, saving: 6

`ss.parallel` (run several sims as scenarios), **`ss.compare_models`** (fit and compare by out-of-sample prediction), **`ss.diff`** (what differs between two sims), **`ss.export`** (a folder that reproduces a run), `ss.load`, `ss.save`.

### Optional metadata: 1

**`ss.Needs`**: one item of input data a module expects, for generating data-entry forms. Never enforced.

### Types you get back or subclass: 6

`ss.Result`, `ss.Loop`, `ss.Timeline`, `ss.Rate`, `ss.TimePar`, `ss.Calibration`. All Starsim's. Users rarely construct these; `sim.calibrate()` returns the calibration object that this specification calls `fit`.

### Methods and attributes: 40 distinct names

| On | Names |
|---|---|
| A model (`ss.Disease`, `ss.Compartments`, any module) | **`describe`** **`diagram`** **`equations`** **`methods`** **`transitions`** **`stratify`** `to_json` `copy` `pars` **`R0`** |
| A state or a sum of states | **`n`** **`prev`** |
| A modifiable array | **`base`** |
| `ss.Sim` | `init` `run` `plot` `summary` `to_df` `loop` `results` **`describe`** **`methods`** `to_json` **`hash`** **`calibrate`** **`sensitivity`** **`check_dt`** **`check_reps`** **`what_modifies`** **`needs`** **`transmissions`** **`crn_report`** |
| `sim.loop` | `insert` `to_df` `plot` `plot_step_order` |
| A result | **`sel`** **`sum`** `resample` **`quantiles`** **`trajectories`** |
| A network | **`check`** |
| What `ss.parallel` returns | `plot` `to_df` **`diff`** |
| `fit` | **`sim`** `pars` `plot` `summary` `to_df` |
| What `ss.compare_models` returns | **`fits`** `plot` `summary` `to_df` |

Names that repeat across rows (`plot`, `summary`, `to_df`, `describe`, `methods`, `to_json`, `pars`) are counted once. The split is deliberate: `describe`, `diagram`, `equations`, `methods` are about the model; `plot`, `summary`, `to_df` are about results, as in Starsim. `R0` and `crn_report` are optional.

## The total

| Group | Names | Of which new |
|---|---|---|
| Sim and modules | 9 | 1 |
| States, transitions, parameters, results | 12 | 12 |
| Time | 22 | 0 |
| Distributions | 13 | 0 |
| Parameter forms | 3 | 3 |
| Mixing and space | 13 | 7 |
| Interactions and interventions | 8 | 8 |
| Running, comparing, saving | 6 | 3 |
| Optional metadata | 1 | 1 |
| Types you get back or subclass | 6 | 0 |
| **`ss.` names** | **93** | **35** |
| Methods and attributes | 40 | 27 |
| **Total** | **133** | **62** |

The two-line SIR uses five of them: `ss.Disease`, `ss.transmission`, `ss.progression`, `ss.Sim`, and `run` (six with `plot`). The operator form swaps the two constructors for `ss.states` and adds `ss.days` if the duration is typed. A stratified, calibrated national model with interventions uses perhaps forty.

The total is about the same as the first draft's (134). What changed is what the names are: 58 of the 93 `ss.` names are ones a Starsim user already knows, and the ones removed (below) were mostly new concepts, while the ones added are mostly methods for looking at a model.

## Keywords

Keywords are not counted above, but they are vocabulary too, so the main sets are listed here.

| On | Keywords |
|---|---|
| `ss.Sim` | `networks` `people` `total_pop` `n_agents` `start` `stop` `dt` `unit` `method` `approx` `solver` `rand_seed` `n_reps` `crn_key` `verbose` `interventions` `connectors` `analyzers` `data` `pars` `results` `track_writes` `label`; plus `diseases`, `demographics`, `modules` as in Starsim |
| `ss.Disease`, `ss.Compartments` | `name` `dt` `init_prev` (`ss.Disease`) `method` (`ss.Compartments`), and one keyword per member |
| `ss.State` | `name` `init` `infectious` `rel_sus` `rel_trans` `label` |
| `ss.Transition` | `rate` `dur` `p` `stages` `name` `label` |
| `ss.transmission` | `beta` `source` `p` |
| `ss.progression`, `ss.birth`, `ss.death` | `rate` `dur` `p` `stages` |
| `ss.transfer` | the stratification's name, `rate` `dur` `p` `states`, and the delivery keywords |
| `ss.par` | `bounds` `prior` `source` `doc` |
| `stratify` | the dimension's name, `split` `adjust` `states` `mobility` `immunity` `kind` |
| `ss.modify` | `multiply` `add` `set` `where` `when` `start` `stop` `min_duration` `name` |
| `ss.change` | `at` `by` `to` |
| Deliveries | `start` `stop` `years` `dates` `interval` `eligible` `coverage` `capacity` `deliver` `on` `when` `min_duration` `on_positive` `on_negative` |
| Products | `efficacy` `waning` `cost` (all); `sensitivity` `specificity` `delay` `detects` (`ss.Test`); `transition` (`ss.Treatment`) |
| `ss.outcome` | `p` `delay` `duration` `likelihood` |
| `ss.MixingPool` | `n_contacts` `dependence` `n_ref` |
| `ss.Mobility` | `matrix` `data` `kind` `away` |
| `sim.calibrate` | `pars` `data` `weights` `likelihood` `gof` `method` `n_reps` `rand_seed` |
| `ss.compare_models` | `data` `holdout` `folds` |

## Slots and their types

A bare number in any of these slots takes the slot's type and the sim's unit ([02-types.md](02-types.md#bare-numbers)). Every slot also accepts the rest of the parameter ladder ([04-parameters.md](04-parameters.md#the-ladder)), including a function.

| Slot | A bare number is | Appears on |
|---|---|---|
| `beta=` | a rate per unit time with no network; a probability per contact with one | `ss.transmission` |
| `rate=`, and a bare number after `@` | a rate (hazard) per unit time | `ss.Transition` and every shorthand constructor |
| `dur=` | a duration; a scalar means an exponential dwell time with that mean | `ss.Transition`, `ss.progression`, `ss.transfer`, networks |
| `p=` | a branch probability on a transition; a reported fraction on an outcome | `ss.Transition`, `ss.progression`, `ss.transmission`, `ss.transfer`, `ss.outcome` |
| `stages=` | a whole number of sub-stages | `ss.Transition`, `ss.progression` |
| `init=` | below 1 a fraction, above 1 a count | `ss.State` |
| `init_prev=` | a fraction | `ss.Disease` |
| `n_contacts=` | contacts per unit time | `ss.MixingPool`, `ss.RandomNet` |
| `coverage=` | a fraction of the eligible people | deliveries, `ss.transfer` |
| `capacity=` | a number per round, or per unit time for routine delivery | deliveries |
| `efficacy=`, `sensitivity=`, `specificity=` | a probability | products |
| `waning=` | a duration (exponential decay with that mean) | products |
| `delay=`, `duration=` | a duration | `ss.outcome`, `ss.Test` |
| `interval=`, `min_duration=`, `away=` | a duration; `away` is a fraction of time | deliveries, `ss.modify`, `ss.Mobility` |
| `multiply=`, `add=`, `set=`, `by=`, `to=` | a multiplier, or a value in the target's own type | `ss.modify`, `ss.change` |
| `at=` | a year (dates and durations are written as such) | `ss.change` |
| `bounds=` | a pair, in the unit of the parameter's value | `ss.par` |
| `cost=` | a plain number | products |

## Removed since the first draft

| Removed | Why | Use instead |
|---|---|---|
| `ss.Effect` | Renamed and split: scaling a quantity is one thing, complex logic another | `ss.modify`, or Python in `step()` |
| `ss.Expr`, and `ss.age`, `ss.sex`, `ss.uid`, `ss.time`, `ss.year`, `ss.weekday`, `ss.has`, `ss.scheduled` | A second set of names for things the sim already has | Conditions on the real attributes: `(ppl.age > 65) & ~sir.vaccinated`, or a function of the sim |
| `ss.Continuous` | A special kind of state and route for reservoirs | `ss.Compartments`, with counts (`W.n`) in rates |
| `ss.process` | A penalty-box name for custom code | A plain function as a keyword, or `step()` |
| `ss.Homogeneous` | Not needed | No network given, or `ss.MixingPool(n_contacts=10)` |
| `ss.SexualNet`, `ss.Network(edges=)` | Starsim already has names | `ss.MFNet` and Starsim's other networks; `ss.StaticNet(edges=...)` |
| `ss.Par`, `ss.Outcome` | Capitalization rule | `ss.par`, `ss.outcome` |
| `ss.cascade`, `observe()`, `Model.results()` | A separate observation step | Results declared as keywords with the model |
| `ss.N`, `ss.count` | Extra names for a sum and a keyword | `N = S + I + R`; `capacity=` |
| `ss.erlang`, `ss.dist(...)` | Duplicates | `ss.gamma` with an integer shape or `stages=`; `ss.Dist(dist=...)` |
| `ss.lognorm`, `ss.binomial`, `ss.neg_binomial`, `ss.rate` | Not Starsim's names | `ss.lognorm_ex` / `ss.lognorm_im`, Starsim's set, `ss.nbinom`, `ss.per` |
| `ss.exp_decay`, `ss.linear_decay` | Covered by existing forms | `waning=` as a duration, a function, or `ss.timeseries` |
| `ss.act`, `queue()`, `phase=`, `sim.substep()` | Queued updates and named phases are gone | Starsim's ordered loop, `sim.loop.insert()` |
| `ss.Stage`, `fit.apply()` | Simpler spellings | `calibrate(method=[...])`; `fit.sim` |
| `capabilities()`, `closure=`, `to_ode()`, `scale=` | The closure ladder and per-transition hybrid methods are out of v1 | `methods()`, `method=`, printed approximations |
| `jacobian()`, `fixed_points()`, `sim.equations()` | Symbolic analysis is not a v1 feature | `sir.equations()` |
| `summary()` on a model, `to_dict()` | `summary` is about results in Starsim; `to_json` is Starsim's name | `describe()`, `to_json()` |
| The dict tuple shorthand `dict(infection=('S -> I', 0.05))` | Ambiguous: it guesses the kind of transition from a bare value | The two-line constructor form |
| `ss.Route` | Starsim's name | `ss.Network`, passed with `networks=` |
| `ss.explain`, `ss.conformance`, `ss.data`, `mixing=`, `n=`, `route=`, `ontology=`, `kind=` on parameters, `take=`/`doses=`, `min=`/`max=` clamps, `crn_strict`, `run(force=)`, `run(reset=)` | Small things that did not earn a name | Documentation pages; `networks=`; `total_pop=`; inline provenance in `ss.par(source=)` |
| `method='ide'` | Deferred | `stages=`, or agents |

## Adding to the vocabulary

There is no fixed budget. Before adding a name, check:

- **Does Starsim already have a name for it?** Use that one.
- **Can it be written with what exists?** A bednet is an `ss.modify`; a metapopulation is a stratification by patch; a junction is a branching transition; a reservoir is an `ss.Compartments`.
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
| A separate class per run method | `method=` |
| A separate metapopulation model class | A stratification by patch |
| A special route for environmental or vector-borne transmission | Counts in rates, and `ss.Compartments` |
| A class per combination of delivery mode and product | Keywords |
| A rule language for triggers and targeting | Conditions and functions in Python |
| Overwrite adjustments in stratification | Multipliers |
| Positional or integer references to states | Names |
| Required declarations of what a module reads and writes | Observed writes, `sim.what_modifies()` |
| Budget optimization | `cost=` now; optimization later |
