# Results and data

What comes out of a run, and how it is connected to data. There is one results concept: everything below is a result, lives in `sim.results`, and can be plotted, exported, selected by stratum, and compared with data. A sim is one trajectory; many runs are a MultiSim, whose results have a run dimension.

Justified by [best-practices/results-and-observation.md](../best-practices/results-and-observation.md).

## Automatic results

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=10),
    init_prev = 0.01,
)
sim = ss.Sim(sir, unit='day').run()

res = sim.results.sir
res.S, res.I, res.R        # one count per state
res.infection, res.recovery      # one flow per transition: the number of events in each step
res.infection.cum                # its cumulative total, computed when asked for
```

Nothing was declared. Results live under the module that produced them, `sim.results.<module>.<result>`, as in Starsim, and this access form is used throughout the specification. A module's results are the same under every solver. The sim's own results about people (`sim.results.alive`, `female`, `new_deaths`, and the rest) exist only in an agent-based run, since a compartmental run has no people of its own.

A build MUST generate, for every module with states and transitions:

- `<state>` for every state: the number of people in it at each time point (a stock), named exactly as the state is;
- `<transition>` for every named transition: the number of people who made that transition during each step (a flow, not a stock), named exactly as the transition is;
- and `<transition>.<state>` for each destination of a transition with several (`cin = ss.Transition(C, [X, R], ...)` gives `cin.X` and `cin.R`): the number who made it and arrived in that state, whose sum is the transition's flow. It is read `res.cin.X` or `res['cin.X']`, and `res.cin.select(to='X')` is the same result; it is kept by group like the total, is a source of an outcome (`ss.outcome(0.8*cin.X)`) and a series in a result expression, and is matched by a data column named `cin.X`. A destination may not be named like an attribute of a result (`cum`, `values`, ...), which raises when the disease is made.

Each flow's running total is `res.<transition>.cum`, a result (a stock) computed when it is asked for, not a second stored series.

Results follow the convention of other modeling frameworks, under every solver: a result has one value for each time point of the sim, from `start` to `stop`. `<state>[ti]` is the number in the state at time point `ti` (so `<state>[0]` is the number at the start), and `<transition>[ti]` is the number who made the transition during the step that ends at time point `ti` (so it is 0 at the start). So if nobody has recovered at the start and 10 people recover in the first step, `R[0]` is 0 and `R[1]` and `recovery[1]` are both 10, and `R[ti] - n_R[ti-1]` is the net flow into the state in the step that ends at `ti`. A sim with `stop=T` takes a step from each time point but the last, so it ends at `T`. A declared model takes values that vary in time at the start of each step (between the time points, an ODE solver takes them where it needs them); a delivery at a time point happens there, between the steps, and is counted there.

A flow has one name, the transition's, which matches the model text, `ss.modify()` targets, `ss.outcome()`, and data columns. A declared disease has no other names for its flows: Starsim 3's `new_infections` is the flow of the transition that infects (`infection` in `ss.SIR`), and `cum_infections` is its running total (`res.infection.cum`). A ratio such as the prevalence is a result that the model declares (`prevalence = I / (S + I + R)`), which in a stratified run is by group as any declared ratio is (`res.prevalence.select(risk='high')`, ST20); the library's `ss.SIR`, `ss.SIS`, and `ss.SEIR` declare it this way, as the infected over everyone, so it is nothing special.

A state that is not people (a water reservoir, a mosquito population; [03-states-and-transitions.md](03-states-and-transitions.md)) gets the same results: `W` is the count it holds.

A disease whose strains have courses of their own (`ss.Strains(shared=...)`, [05-dimensions.md](05-dimensions.md#strains)) has results made from its strains' results of the same name: for a state or a named sum of states that every member has (a stock), the number in it in at least one member, a union that counts each person once (`sim.results.hpv.infected` is the number infected with any genotype, where a sum would count the co-infected twice); for a transition (or outcome) that every member has (a flow), the sum, since each event belongs to one member (`sim.results.hpv.cancer`). `res.by('strain')` (the strains' name) gives each strain's own result (`sim.results.hpv.infected.by('strain')`). The strains keep their own results (`sim.results.hpv16.infected`). With strains in shared states carried together, `res.I` counts each person once and `res.I.by('strain')` counts each in every strain they carry; a set is `res.I.select(strain='DS+RIF')`. An outcome over unrelated diseases is still not declared.

`ss.SIR`, `ss.SIS`, and `ss.SEIR` are declared diseases, so their results are named after their states and transitions (`sim.results.sir.I` and `sim.results.sir.infection`, not Starsim 3's `infected` and `new_infections`), with the `prevalence` they declare. Starsim 3's versions, written as code (`ssl.AgentSIR` and the others), keep Starsim 3's names.

## `ss.Result`

A result carries its name, label, module, values, time vector, unit, and how it is summarized over time (`summarize_by`): `'sum'` for a flow, `'last'` for a stock, `'mean'` for a ratio, or `None` for a result that is not summarized; other summarizers can be added to `ss.Result.summarizers`. This is Starsim's `ss.Result`, with two additions: selection by stratum, and a run dimension when it holds the results of many runs.

During a run, a result is read from inside the sim (by a trigger, a modifier, or custom code) as in Starsim: the storage is preallocated for the whole run, and during a step `sim.ti` is the step's own index, where its results will go, so the results so far (those of the steps before it) are `res.I[:sim.ti]`, and a result written by the step itself is `res.x[sim.ti]`. A shorter spelling for "the steps so far" (`res.I.so_far`, or a slicing helper) would be convenient in triggers; it is listed as open and not specified here ([19-open-questions.md](19-open-questions.md)).

Because a result knows how it is summarized (`res.infection.summarize_by` is `'sum'`), converting it to a coarser time step does the right thing without being told: `res.infection.resample('week')` sums, `res.I.resample('week')` takes the value at the end of the week, and a ratio is recomputed from its resampled parts. Weeks (and days, or any duration) are counted from the sim's start; months and years are calendar ones for a sim with dates. As for any period, a time point closes the step that ends there, so the point on 1 January is in the year before, and the sim's start is in the first period: a run from 2000 to 2003 has three years. The same `summarize_by` decides how the result is compared with data and summarized by `sim.summarize()` (the total of a flow, the final value of a stock, the mean of a ratio, which for a ratio with a numerator and denominator is each summed and then divided). It is Starsim 3's spelling, kept as the only one, since it says what is done. A result has one time vector; other time representations are conversions.

### Selecting strata

After stratifying, a result is still **one result**, now with a dimension for each stratification. The strata are coordinates of that result, not separate results: there is no `I_female_high`. Used as it is, the result is the total; `.select()` picks groups by name, and `.by()` gives one series per group:

```python
res = sim.results.sir
res.I                                              # everyone in I
res.I.select(sex='female')                         # women in I, all ages and risk groups
res.I.select(sex='female', risk='high')            # high-risk women in I
res.I.by('age')                                    # one series per age group
res.infection.select(age=0)                          # by position: incidence in the youngest age group
res.infection.select(age='0-14', patch='Nairobi')    # by label
res.prevalence.select(risk='high')
```

- `.select(**dims)` keeps the named groups along each dimension given and sums over every dimension not given. It returns a result, so it can be plotted, resampled, selected again, or compared with data.
- `.by(dim)` returns the result with that one dimension kept and the rest summed: one series per group. `.select(...).by(...)` chains, in either order (`.by('age').select(area='rural')` is the same).
- A dimension the result does not have is an error that lists the dimensions it does have.
- A state that is not people has only the dimensions it was stratified by (`patch`, or its own), so `res.W.select(age=0)` is an error and `res.W.select(patch='Nairobi')` is not.

A ratio is never averaged over groups. It keeps its numerator and denominator, each is summed over the selected groups, and then they are divided: prevalence among women is the women in `I` over the women in `N` (see [Ratios and combinations](#ratios-and-combinations)). In an agent-based run, selecting on a continuous property uses the bins it was stratified with ([05-dimensions.md](05-dimensions.md)).

A module written as code keeps a result by group in the same way when it is made with `groups=True`, so it can be selected as a declared module's is, instead of one result per stratum (`prevalence_f_15_20`). As agents, stratifying the sim by a property people already have (`sex`, `age`) changes no draws, so it can be done for the results alone. `sim.groups.count(where)` counts the people alive for whom a condition holds, in each group, in one pass (each agent by its weight, if they differ), and `res.set_groups(ti, vals, denom=None)` writes a count by group, or a ratio from its numerator and denominator by group:

```python
sim.stratify(sex=['female', 'male'], age=[0, 15, 20, 25, 30, 35, 50, 65])

def init_results(self):                             # in the module written as code
    super().init_results()
    self.define_results(ss.Result('infected', groups=True), ss.Result('prevalence', scale=False, groups=True))

def update_results(self):
    super().update_results()
    alive, infected = self.sim.groups.count(), self.sim.groups.count(self.infected)
    self.results.infected.set_groups(self.ti, infected)
    self.results.prevalence.set_groups(self.ti, infected, alive)

sim.results.ng.prevalence.select(sex='female', age='15-19')   # infected women aged 15-19 over all women aged 15-19
```

## Declaring more results

Extra results are declared with the model. There is no separate observation step.

```python
S, I, R   = ss.states(3)
N         = S + I + R
infection = S >> I @ 0.3 * I
recovery  = I >> R @ ss.days(10)

prevalence    = I / N
ever_infected = (I + R) / N
cases  = ss.outcome(0.5  * infection, delay=ss.days(5))
hosp   = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
deaths = ss.outcome(0.2  * hosp,      delay=ss.days(14))

sir = ss.Disease(
    transitions = [infection, recovery],
    results     = [prevalence, ever_infected],
    outcomes    = [cases, hosp, deaths],
    init_prev   = 0.01,
)
```

`transitions=`, `results=`, and `outcomes=` each take a list and keep the three kinds of thing apart; the members take their names from the variables, as transitions do ([03-states-and-transitions.md](03-states-and-transitions.md)). In a short model the same things can be given to the disease directly, as positional transitions and keywords:

```python
sir = ss.Disease(infection, recovery, prevalence=I / N, cases=ss.outcome(0.5 * infection, delay=ss.days(5)), init_prev=0.01)
```

Either way they end up in the same place: `sim.results.sir.prevalence`, `sim.results.sir.cases`, `sim.results.sir.n_hosp`.

### Ratios and combinations

Arithmetic on states never makes a new state: nobody is "in" `N`. `S + I + R` is a sum of states, and what a sum does depends on where it is put. In a rate it is a source of infection or a denominator ([03-states-and-transitions.md](03-states-and-transitions.md)). Given to the disease as a result, it is a result, in which a state stands for the number of people in it. So a sum of states is a combined count and a ratio of two is a proportion: `I / N` is a time series of prevalence. Flows can be used the same way (`infection / S` is the per-capita incidence). `N` itself is not a result of the model above, because it was not given to the disease. `I.frac`, the fraction of the whole population in `I`, exists for the most common case.

The rules:

- **A weighted sum** (`I + 0.5 * A`) is allowed, in a rate and in a result.
- **A ratio aggregates by summing its numerator and its denominator separately** over the selected groups, and over each interval when resampled in time, and then dividing. It is never an average of the groups' ratios.
- **Anything nonlinear** (a product of states, a saturating function) is written with counts, `I.n * S.n / N.n`, and is computed from the totals of whichever groups are selected.

### `ss.outcome`

An outcome is a reported quantity: a flow worked out from other flows. It covers reported cases, hospital admissions, deaths, and similar series that are counted from the epidemic without being part of transmission. It moves nobody: no one changes state because they were counted.

```python
ss.outcome(
    source,               # a transition, another outcome, or a weighted sum of them
    delay      = None,    # time from the source event to this one: a duration, a distribution, or a function of the sim
    duration   = None,    # if given, also produces n_<name>: the number currently in this outcome; a duration (an exponential stay) or a distribution (ss.constant for a fixed stay)
    likelihood = None,    # optional; see below
)
```

- **The source is a weighted sum of flows.** A number times a flow means that fraction of those events: `0.5 * infection` is half of infections. There is no `p=` argument; the weight is the probability. Reporting that depends on severity is a sum, `ss.outcome(0.01 * asymptomatic + 0.3 * symptomatic + 0.8 * severe, delay=ss.days(3))`, where the three names are the transitions into those states. A weight follows the parameter ladder ([04-parameters.md](04-parameters.md)): it can be an `ss.par` with `bounds` and be fitted like any other parameter, or `ss.by(age=[...])`.
- **Outcomes chain.** The source of `deaths` above is the outcome `hosp`, so infection → hospitalization → death is three lines. A chain that loops back on itself raises a `ValueError`. Where a transition was declared as a keyword and there is no variable to refer to, the source may be its name (`ss.outcome('infection', delay=ss.days(5))`); a name that matches nothing raises a `KeyError`.
- **`delay=`** is a duration, a distribution (reporting delays are not a single number), or a function of the sim for anything irregular.
- **`duration=` gives both series.** `hosp` is admissions per step and `n_hosp` is the number currently in hospital.
- **There is no reporting period.** An outcome is a flow on the sim's time step, like a transition. Weekly data is compared with its total over each week automatically (see [Data](#data)), and `res.cases.resample('week')` gives the weekly series to plot.

An outcome belongs to one disease, and its source is made of that disease's flows. There is no outcome on the sim for results that span modules (all-cause deaths, a syndromic count over several diseases): arithmetic on the results after the run gives the series (`0.3 * (sim.results.flu.infection + sim.results.rsv.infection)` is an `ss.Result`), but it is not a declared result, so data cannot be matched to it.

An outcome is worked out from the counts of the flows it comes from, in each group, under every solver (it records no individuals, as agents or otherwise). How the weight and the delay are realized depends on whether the run counts whole people:

| Solver | A weight on a flow | `delay=` and `duration=` |
|---|---|---|
| `ode` | multiplies the flow | the flow convolved with the delay distribution, taken in whole steps |
| `binomial`, `gillespie`, `abm` | a weight between 0 and 1 is a binomial draw from the step's events (any other weight multiplies) | each step's reports are spread over the following steps by a multinomial draw from the delay distribution |

### Outcome, intervention, or state

An outcome is a shortcut for an observation process that the model does not represent. Which of the three to use depends on what the model contains:

| The situation | Use |
|---|---|
| Testing and reporting are not in the model: "about half of infections are reported, five days later" | An outcome |
| Testing is modeled, by one or more interventions | The interventions' own results. Each testing intervention reports the diagnoses it made, and reported cases are their sum: `cases = symptomatic_testing.positive + screening.positive` |
| Being counted changes what happens to the person (hospitalized people are isolated and stop transmitting) | A state, with transitions in and out |

So an outcome never competes with a testing intervention over who produces `cases`: if the model has tests, the tests produce the cases ([08-interventions.md](08-interventions.md)), and `ss.outcome` is not used for them.

An outcome is a result, and anything can read a result: a [trigger](08-interventions.md#triggers) can close schools when `n_hosp` passes a threshold, and a rate can depend on it. What an outcome cannot do is single out individuals, because it never recorded who they were. If being hospitalized changes what happens to a person, hospitalization is a state.

## Plotting

`sim.plot()` plots each result in axes of its own, leaving out the sim's own results that never change (such as the number alive in a closed population). A module's `plot()` plots its results by what they are:

```python
sir.plot()            # the number in each state, together in one axes (the default; 'states')
sir.plot('flows')     # each transition's flow, in axes of its own
sir.plot('all')       # every result, each in axes of its own
sir.plot(['S', 'I'])  # S and I, together
sir.plot('S', 'I')    # S and I, apart
```

A list within the list of results given to `sim.plot()` is plotted in one axes too (`sim.plot([['sir.S', 'sir.I'], 'sir.infection'])`).

## Data

Data is attached to results by name:

```python
sim = ss.Sim(sir, data='kenya_2020.csv', start='2020-01-01', stop='2020-12-31').run()
sim.plot()
```

- `data=` takes a file name, a data frame, or an `ss.simdata(...)` object (below). It has a time column (`date`, `year`, `t`, or `time`) and one column per data series, **named like the result it corresponds to** (`cases`, `deaths`, `prevalence`, or `cin.X` for the flow to one destination of a transition). With several modules, a column is matched to the one module that has a result of that name, or is written `sir.cases` when the name is ambiguous. This works the same for modules written as code (Starsim 3's diseases among them), so it is a way into fitting for a Starsim 3 model.
- A column that matches no result is noted when the sim is initialized, with the result names closest to it, since data on a sim can be just for plotting (or read by a fit's own `gof=`); when a fit compares every column with its result, it is an error (a `KeyError`), so a typo in a header is caught. `ss.simdata(..., ignore_extra=True)` drops such columns. Data entirely outside the simulated period raises a `ValueError` when it is compared with the results (by `sim.fit()`, for instance); data partly outside is used where it overlaps.
- Data for particular strata is given in long form: extra columns named after dimensions (`age`, `patch`) say which stratum each row belongs to, by the level's label (or position). Each result and stratum is then one data series, named like `cases[age=0-14]` and compared with `cases.select(age='0-14')`; a likelihood or weight given for `cases` applies to every stratum of it.
- For a disease whose strains have courses of their own, a strain's own result is its column (`hpv16.cancer`), and in long form a column named after the strains (`strain`, or the name given to `ss.Strains`) gives it: `cancer` in a row for `strain=hpv16` is `hpv16.cancer`. `hpv.cancer[strain=hpv16]` is an error naming `hpv16.cancer`.
- The data need not be on the model's time step. A flow (weekly reported cases) is compared with the model's flow summed over each data interval; a stock or ratio (a prevalence survey) is compared with the model's value at the data's time. The result already knows which it is. The interval assumed for flow data is the spacing of the data. Data by calendar year, an integer `year` column (as in UNAIDS-style data and `res.resample('year')`), is the year each row names, not 1 January of it: a flow is summed over the year and a stock or ratio averaged over it, with a note; only whole years inside the run are compared. A result of a module with a timestep of its own is compared on that module's timeline.
- Results that have data are plotted with it automatically: `sim.plot()` (and `msim.plot()`, with the data of its sims) draws the model as a line and the data as points. A flow with data at a longer period than the step (weekly cases from a daily model) is drawn as the fit compares it, its total over each period of the data, so a good fit looks like one; `fit.plot()` and its posterior band do the same.

Real files rarely arrive with matching headers, so `ss.data` renames on the way in:

```python
ss.simdata(
    source,                 # a file name or a data frame
    aliases      = None,    # dict of result (or time-column) name -> the column that holds it
    ignore_extra = False,   # True: drop columns that match no result instead of raising
)
```

```python
data = ss.simdata('kenya_moh_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
sim  = ss.Sim(sir, data=data, start='2020-01-01', stop='2020-12-31')
```

An alias that names a column the file does not have, or a result the model does not have, is an error.

This is all that calibration needs: `sim.fit()` fits every parameter that has `bounds` to every result that has data ([13-inference.md](13-inference.md)). Several data series (clinical cases, wastewater, seroprevalence) are more columns.

### An optional likelihood

Nobody has to write a likelihood to compare a model with data or to fit it. The default measure of fit is the normalized absolute error: for each series, the mean absolute difference between model and data, divided by the mean of the data. It assumes no noise model, and one bad reporting week does not dominate it the way it dominates a squared error ([13-inference.md](13-inference.md#goodness-of-fit)). When a likelihood is wanted, it is attached to the result:

```python
cases = ss.outcome(0.5 * infection, delay=ss.days(5),
                   likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
```

The model's value is the mean of the distribution and the user supplies the rest (here the dispersion, which is fitted). The likelihoods are `ss.poisson`, `ss.nbinom`, `ss.binom`, `ss.bernoulli`, `ss.normal`, and `ss.lognorm_ex`; any other is a function `fn(observed, model)` returning the log-likelihood of each value ([13-inference.md](13-inference.md#likelihoods-when-wanted)). A result that is not an outcome (a ratio such as `prevalence = I / N`) is given one at calibration time, `sim.fit(likelihood=dict(prevalence=ss.normal(scale=0.02)))`, keyed by the result's name or by the data column (`sir.prevalence`, or one stratum, `prevalence[age=0-14]`); a stratum without one of its own takes its result's. Where a likelihood for such a result should be declared with the model is open. A likelihood makes posterior methods available; [13-inference.md](13-inference.md) says what is done with it.

## One run, many runs

A sim is one trajectory, whatever the solver. Many runs are a MultiSim, as in Starsim, and going from one to the other is one call:

```python
sim  = ss.Sim(sir, solver='gillespie', total_pop=1000, unit='day')
msim = ss.parallel(sim, n_runs=100)          # or sim.run(n_runs=100), the shortcut for one sim

msim.plot()                                  # median and interval
msim.plot(lines=True)                        # every run
n_I = msim.results.sir.I                   # the same result, with a run dimension
n_I.quantiles([0.025, 0.5, 0.975])           # one series per quantile
n_I.values                                   # the full array, runs by time
```

- `n_runs` belongs to the MultiSim, not to the sim. `ss.Sim` has no replicate count, and `sim.results` always holds one series per result. Each run has its own seed ([11-random.md](11-random.md)), which `msim.runs` (one row per run: its scenario, any values swept or drawn, and its `rand_seed`) gives. A set of values swept with `pars=` is a scenario of its own, and has the same seeds as every other set, so they are compared with the same chance events; seeds swept (`pars=dict(rand_seed=[1, 5, 9])`) and draws (`priors=`, `fit.project()`) are runs of their scenario, and each draw has a seed of its own.
- **A MultiSim is its results, and a MultiSim result is the same `ss.Result` with a `run` dimension first.** `.select()` and `.by()` work as on a single sim (`msim.results.sir.I.select(age=0).quantiles([0.5])`), `.select(run=3)` is one run (a result of one sim), and arithmetic, `.resample()`, `.to_df()`, and `.plot()` work on it as on any result. `len(msim)` is the number of runs, whichever way they were taken (restarted compartmental runs, runs in one call to the kernels or to JAX, or sims run in parallel).
- `.quantiles(q)` takes a list of quantiles and returns a result with one series per quantile (for each scenario), over the runs; `.median()` and `.mean()` return the series over the runs (of each scenario). `.values` is the full array with one row per run.
- `msim.summarize()` is a table of each result over the whole of each run (a flow's total, a stock's final value), across the runs: the median, an interval (95% by default, as `msim.plot()` draws), the mean, and the standard deviation, for each scenario, rounded for reading. `msim.diff()` and `msim.compare()` work from the results too. `ss.io.to_hub(msim, ...)` writes them as a submission to a forecast hub.
- `msim.plot()` draws the median and an interval. `msim.plot(lines=True)` draws every run, because extinction, bimodality, and threshold behavior cannot be seen in a median.
- `ss.parallel({'Baseline': sim_a, 'With vaccine': sim_b}, n_runs=100)` (or `ss.parallel(baseline=sim_a, vaccine=sim_b)`, for labels that are names) returns one object whose results are indexed by scenario, run, and time ([08-interventions.md](08-interventions.md)): `msim.results.sir.I.select(scenario='With vaccine')`. A sweep, `ss.parallel(sim, pars=dict(beta=[0.2, 0.4]), n_runs=10)`, has the values as coordinates of the runs: `msim.results.sir.I.select(beta=0.2).quantiles()`.
- `ss.parallel()` is the one way to run many sims, and `sim.run(n_runs=100)` is its shortcut for running one sim many times; both end in the same results.


How often a disease died out, or caused a major outbreak, is read from the runs' results, for each declared disease and scenario. A disease that nobody can catch again dies out in every run that is long enough, after a major outbreak as after a minor one, so for it the question is how often there was a major outbreak, counted from the acquisitions over the run (so the people infected at the start are not counted):

```python
died = (msim.results.sir.I.values[:, -1] == 0).mean()                  # the share of runs with nobody infected at the end
major = (msim.results.sir.infection.values.sum(axis=1) > 100).mean()     # the share of runs with more than 100 infected
```

## The transmission log

The log of who infected whom is usually much larger than all the other results put together, recording it slows the run, and most models never look at it. So it is not recorded by default. It is requested with Starsim's existing analyzer:

```python
sim = ss.Sim(sir, networks='random', analyzers='infection_log').run()
sim.analyzers.infection_log.to_df()   # one row per infection: infector, infectee, time, and the network it happened on
```

Every infection event in an agent-based run, as a table. It supports transmission trees, phylodynamic comparison, and counting secondary cases directly. The infector is known when transmission happened on a route that keeps contacts; for infections through a mixing pool the row records the source group, and where infection does not come from a person (a mosquito count, a water reservoir, `W.n`) it records the infectee and the source state, with no infector. The log exists only in an agent-based run: a compartmental run has no individuals to record, so asking for it makes a model agent-based under `solver='auto'` and is an error (`CapabilityError`) under an explicitly compartmental solver ([12-backends.md](12-backends.md)).

## Requesting fewer results

```python
ss.Sim(sir, results=['I', 'cases'])
```

keeps only the named results by group (by name, or as `'sir.I'`); every other result keeps only its total over the groups, which is small. A name that is not a result is an error. The default is everything, which is the right default until stratification multiplies it: a result per state and per transition, for every stratum, at every step, can be a lot of memory.

So by default the values by group are kept at every step only while they fit in about 100 MB for a module in one run (500 MB over all the runs of `sim.run(n_runs=)`). Beyond that, they are kept every few steps: the shortest of a week, two weeks, four weeks, a quarter, half a year, or a year that fits. A row of values by group then holds a flow added up over its steps and a state as it is at the end of them, so `res.I.select(district='north')` is a series at those times, its flows add up to the totals, and `resample()` works to periods no shorter than those. The totals are still kept at every step. This is a guess, so it is said in a note and marked in `describe()`; naming fewer results with `results=` keeps those more often. Data by group given to the sim keep every step, since they are compared with the results at their own times.

## Rejected

- **A separate `observe()` step and a separate kind of object for derived quantities.** Ratios and outcomes are declared with the model and are results like any other.
- **`data=` on each outcome.** One data file with columns named like results is simpler and is what gets plotted.
- **A required likelihood.** Optional.
- **Results created on first use.** Declared results give typo protection and make it possible to request fewer.
- **Reporters configured separately from the model.** What is reported is part of the model.
- **Post-processing scripts for incidence.** Every named transition has a flow result.
- **A separate cascade analysis object.** An ordered list of results is a plot.
- **A sim that is many trajectories** (a replicate count on the sim, with a distribution as its result). One sim, one run is easier to reason about, and Starsim's MultiSim already holds many.
- **A `p=` argument on outcomes.** A number times a flow already says "this fraction of those events", and it extends to weighted sums.
- **A reporting period on outcomes (`every=`).** It made the outcome zero between reporting days, a comb that compared wrongly with data on any other schedule. Data by period is compared with the total over each period, and `resample()` gives the series by period.
- **Recording every transmission by default.** The log is large and slows the run; it is one keyword when wanted.
- **One named result per stratum** (`I_female_high`). A stratified result is one result with dimensions.
- **Averaging ratios over strata.** Numerator and denominator are summed separately.
- **Several parallel time representations on every result.** One time vector, converted on request.
- **Four names for each flow** (`infection`, `cum_infection`, `new_infections`, `cum_infections`). A flow has the transition's name, and its running total is `.cum`.
- **A MultiSim reduced to one sim** (Starsim's `msim.reduce()`, `msim.mean()`, `msim.median()`, with bounds as `low`/`high` on flattened results). The results across runs are summarized directly: `res.quantiles()`, `res.median()`, `res.mean()`.
