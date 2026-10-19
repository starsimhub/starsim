# Results and data

What comes out of a run, and how it is connected to data. There is one results concept: everything below is a result, lives in `sim.results`, and can be plotted, exported, selected by stratum, and compared with data. A sim is one trajectory; many runs are a MultiSim, whose results have a run dimension.

Justified by [best-practices/results-and-observation.md](../best-practices/results-and-observation.md).

## Automatic results

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
    init_prev = 0.01,
)
sim = ss.Sim(sir, unit='day').run()

res = sim.results.sir
res.n_S, res.n_I, res.n_R        # one count per state
res.infection, res.recovery      # one flow per transition: the number of events in each step
res.infection.cum                # its cumulative total, computed when asked for
```

Nothing was declared. Results live under the module that produced them, `sim.results.<module>.<result>`, as in Starsim, and this access form is used throughout the specification. A module's results are the same under every solver. The sim's own results about people (`sim.results.n_alive`, `n_female`, `new_deaths`, and the rest) exist only in an agent-based run, since a compartmental run has no people of its own.

A build MUST generate, for every module with states and transitions:

- `n_<state>` for every state: the number of people in it at each time point;
- `<transition>` for every named transition: the number of people who made that transition during each step (a flow, not a stock), named exactly as the transition is.

Each flow's running total is `res.<transition>.cum`, a result (a stock) computed when it is asked for, not a second stored series.

Results follow the convention of other modeling frameworks, under every solver: a result has one value for each time point of the sim, from `start` to `stop`. `n_<state>[ti]` is the number in the state at time point `ti` (so `n_<state>[0]` is the number at the start), and `<transition>[ti]` is the number who made the transition during the step that ends at time point `ti` (so it is 0 at the start). So if nobody has recovered at the start and 10 people recover in the first step, `n_R[0]` is 0 and `n_R[1]` and `recovery[1]` are both 10, and `n_R[ti] - n_R[ti-1]` is the net flow into the state in the step that ends at `ti`. A sim with `stop=T` takes a step from each time point but the last, so it ends at `T`. A declared model takes values that vary in time at the start of each step (between the time points, an ODE solver takes them where it needs them); a delivery at a time point happens there, between the steps, and is counted there.

A flow has one name, the transition's, which matches the model text, `ss.modify()` targets, `ss.outcome()`, and data columns. Starsim's standard disease results are kept under their existing names on the library's `ss.SIR`, `ss.SIS`, and `ss.SEIR` (or any disease class with `standard_results = True`), so code written for Starsim carries over: for a disease with transmission, `new_infections` is the total flow through its transmission transitions, `cum_infections` its running total, and `prevalence` the fraction of the population in its infected states. A disease declared with `ss.Disease(...)` does not have them; a result declared under one of these names gives it one (e.g. `prevalence = I / N`). A data column called `new_infections` is matched to the flow of the transmission of a declared disease that has one, so Starsim data files still line up.

A state that is not people (a water reservoir, a mosquito population; [03-states-and-transitions.md](03-states-and-transitions.md)) gets the same results: `n_W` is the count it holds.

`ss.SIR`, `ss.SIS`, and `ss.SEIR` are declared diseases, so their results are named after their states (`sim.results.sir.n_I`, not Starsim 3's `n_infected`), with the three standard results above added. Starsim 3's versions, written as code (`ssl.AgentSIR` and the others), keep Starsim 3's names.

## `ss.Result`

A result carries its name, label, module, values, time vector, unit, and kind: a flow, a stock, or a ratio. This is Starsim's `ss.Result`, with two additions: selection by stratum, and a run dimension when it holds the results of many runs.

During a run, a result is read from inside the sim (by a trigger, a modifier, or custom code) as in Starsim: the storage is preallocated for the whole run, and during a step `sim.ti` is the time point at its start, so the results so far are `res.n_I[:sim.ti + 1]`. A shorter spelling for "the steps so far" (`res.n_I.so_far`, or a slicing helper) would be convenient in triggers; it is listed as open and not specified here ([19-open-questions.md](19-open-questions.md)).

Because a result knows whether it is a flow, a stock, or a ratio (`res.infection.kind` is `'flow'`), converting it to a coarser time step does the right thing without being told: `res.infection.resample('week')` sums, `res.n_I.resample('week')` takes the value at the end of the week, and a ratio is recomputed from its resampled parts. Weeks (and days, or any duration) are counted from the sim's start; months and years are calendar ones for a sim with dates. As for any period, a time point closes the step that ends there, so the point on 1 January is in the year before, and the sim's start is in the first period: a run from 2000 to 2003 has three years. The same kind decides how the result is compared with data and summarized by `sim.summarize()` (the total of a flow, the final value of a stock, the mean of a ratio). A result has one time vector; other time representations are conversions.

### Selecting strata

After stratifying, a result is still **one result**, now with a dimension for each stratification. The strata are coordinates of that result, not separate results: there is no `n_I_female_high`. Used as it is, the result is the total; `.select()` picks groups by name, and `.by()` gives one series per group:

<!-- fragment -->
```python
res = sim.results.sir
res.n_I                                              # everyone in I
res.n_I.select(sex='female')                         # women in I, all ages and risk groups
res.n_I.select(sex='female', risk='high')            # high-risk women in I
res.n_I.by('age')                                    # one series per age group
res.infection.select(age=0)                          # by position: incidence in the youngest age group
res.infection.select(age='0-14', patch='Nairobi')    # by label
res.prevalence.select(risk='high')
```

- `.select(**dims)` keeps the named groups along each dimension given and sums over every dimension not given. It returns a result, so it can be plotted, resampled, selected again, or compared with data.
- `.by(dim)` returns the result with that one dimension kept and the rest summed: one series per group. `.select(...).by(...)` chains.
- A dimension the result does not have is an error that lists the dimensions it does have.
- A state that is not people has only the dimensions it was stratified by (`patch`, or its own), so `res.n_W.select(age=0)` is an error and `res.n_W.select(patch='Nairobi')` is not.

A ratio is never averaged over groups. It keeps its numerator and denominator, each is summed over the selected groups, and then they are divided: prevalence among women is the women in `I` over the women in `N` (see [Ratios and combinations](#ratios-and-combinations)). In an agent-based run, selecting on a continuous property uses the bins it was stratified with ([05-dimensions.md](05-dimensions.md)).

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

<!-- fragment -->
```python
ss.outcome(
    source,               # a transition, another outcome, or a weighted sum of them
    delay      = None,    # time from the source event to this one: a duration, a distribution, or a function of the sim
    duration   = None,    # if given, also produces n_<name>: the number currently in this outcome; a duration (an exponential stay) or a distribution (ss.constant for a fixed stay)
    likelihood = None,    # optional; see below
)
```

- **The source is a weighted sum of flows.** A number times a flow means that fraction of those events: `0.5 * infection` is half of infections. There is no `p=` argument; the weight is the probability. Reporting that depends on severity is a sum, `ss.outcome(0.01 * asymptomatic + 0.3 * symptomatic + 0.8 * severe, delay=ss.days(3))`, where the three names are the transitions into those states. A weight follows the parameter ladder ([04-parameters.md](04-parameters.md)): it can be an `ss.par` with `bounds` and be fitted like any other parameter, or `ss.by(age=[...])`.
- **Outcomes chain.** The source of `deaths` above is the outcome `hosp`, so infection → hospitalization → death is three lines. A chain that loops back on itself raises `OutcomeCycle`. Where a transition was declared as a keyword and there is no variable to refer to, the source may be its name (`ss.outcome('infection', delay=ss.days(5))`); a name that matches nothing raises `UnknownOutcomeSource`.
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
| Testing is modeled, by one or more interventions | The interventions' own results. Each testing intervention reports the diagnoses it made, and reported cases are their sum: `cases = symptomatic_testing.diagnosed + screening.diagnosed` |
| Being counted changes what happens to the person (hospitalized people are isolated and stop transmitting) | A state, with transitions in and out |

So an outcome never competes with a testing intervention over who produces `cases`: if the model has tests, the tests produce the cases ([08-interventions.md](08-interventions.md)), and `ss.outcome` is not used for them.

An outcome is a result, and anything can read a result: a [trigger](08-interventions.md#triggers) can close schools when `n_hosp` passes a threshold, and a rate can depend on it. What an outcome cannot do is single out individuals, because it never recorded who they were. If being hospitalized changes what happens to a person, hospitalization is a state.

## Data

Data is attached to results by name:

```python
sim = ss.Sim(sir, data='kenya_2020.csv', start='2020-01-01', stop='2020-12-31').run()
sim.plot()
```

- `data=` takes a file name, a data frame, or an `ss.data(...)` object (below). It has a time column (`date`, `year`, `t`, or `time`) and one column per data series, **named like the result it corresponds to** (`cases`, `deaths`, `prevalence`). With several modules, a column is matched to the one module that has a result of that name, or is written `sir.cases` when the name is ambiguous. This works the same for modules written as code (Starsim 3's diseases among them), so it is a way into fitting for a Starsim 3 model.
- A column that matches no result is an error (`UnknownDataColumn`) that lists the result names closest to it, so a typo in a header is caught. `ss.data(..., ignore_extra=True)` drops such columns instead. Data entirely outside the simulated period raises `DataMisaligned` when it is compared with the results (by `sim.calibrate()`, for instance); data partly outside is used where it overlaps.
- Data for particular strata is given in long form: extra columns named after dimensions (`age`, `patch`) say which stratum each row belongs to, by the level's label (or position). Each result and stratum is then one data series, named like `cases[age=0-14]` and compared with `cases.select(age='0-14')`; a likelihood or weight given for `cases` applies to every stratum of it.
- The data need not be on the model's time step. A flow (weekly reported cases) is compared with the model's flow summed over each data interval; a stock or ratio (a prevalence survey) is compared with the model's value at the data's time. The result already knows which it is. The interval assumed for flow data is the spacing of the data.
- Results that have data are plotted with it automatically: `sim.plot()` draws the model as a line and the data as points.
- The data's content is hashed as a part of `sim.hash()` of its own (`data`), apart from the model's ([11-random.md](11-random.md#run-identity)).

Real files rarely arrive with matching headers, so `ss.data` renames on the way in:

<!-- fragment -->
```python
ss.data(
    source,                 # a file name or a data frame
    aliases      = None,    # dict of result (or time-column) name -> the column that holds it
    ignore_extra = False,   # True: drop columns that match no result instead of raising
)
```

```python
data = ss.data('kenya_moh_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
sim  = ss.Sim(sir, data=data, start='2020-01-01', stop='2020-12-31')
```

An alias that names a column the file does not have, or a result the model does not have, is an error.

This is all that calibration needs: `sim.calibrate()` fits every parameter that has `bounds` to every result that has data ([13-inference.md](13-inference.md)). Several data series (clinical cases, wastewater, seroprevalence) are more columns.

### An optional likelihood

Nobody has to write a likelihood to compare a model with data or to fit it. The default measure of fit is the normalized absolute error: for each series, the mean absolute difference between model and data, divided by the mean of the data. It assumes no noise model, and one bad reporting week does not dominate it the way it dominates a squared error ([13-inference.md](13-inference.md#goodness-of-fit)). When a likelihood is wanted, it is attached to the result:

```python
cases = ss.outcome(0.5 * infection, delay=ss.days(5),
                   likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
```

The model's value is the mean of the distribution and the user supplies the rest (here the dispersion, which is fitted). The likelihoods are `ss.poisson`, `ss.nbinom`, `ss.binom`, `ss.bernoulli`, `ss.normal`, and `ss.lognorm_ex`. A result that is not an outcome (a ratio such as `prevalence = I / N`) is given one at calibration time, `sim.calibrate(likelihood=dict(prevalence=ss.normal(scale=0.02)))`, keyed by the result's name or by the data column (`sir.prevalence`, or one stratum, `prevalence[age=0-14]`); a stratum without one of its own takes its result's. Where a likelihood for such a result should be declared with the model is open. A likelihood makes posterior methods available; [13-inference.md](13-inference.md) says what is done with it.

## One run, many runs

A sim is one trajectory, whatever the solver. Many runs are a MultiSim, as in Starsim, and going from one to the other is one call:

```python
sim  = ss.Sim(sir, solver='gillespie', total_pop=1000, unit='day')
msim = ss.parallel(sim, n_runs=100)          # or sim.run(n_runs=100), the shortcut for one sim

msim.plot()                                  # median and interval
msim.plot(lines=True)                        # every run
n_I = msim.results.sir.n_I                   # the same result, with a run dimension
n_I.quantiles([0.025, 0.5, 0.975])           # one series per quantile
n_I.values                                   # the full array, runs by time
```

- `n_runs` belongs to the MultiSim, not to the sim. `ss.Sim` has no replicate count, and `sim.results` always holds one series per result. Each run has its own seed ([11-random.md](11-random.md)).
- **A MultiSim is its results, and a MultiSim result is the same `ss.Result` with a `run` dimension first.** `.select()` and `.by()` work as on a single sim (`msim.results.sir.n_I.select(age=0).quantiles([0.5])`), `.select(run=3)` is one run (a result of one sim), and arithmetic, `.resample()`, `.to_df()`, and `.plot()` work on it as on any result. `len(msim)` is the number of runs, whichever way they were taken (restarted compartmental runs, runs in one call to the kernels or to JAX, or sims run in parallel).
- `.quantiles(q)` takes a list of quantiles and returns a result with one series per quantile (for each scenario), over the runs; `.median()` and `.mean()` return the series over the runs (of each scenario). `.values` is the full array with one row per run.
- `msim.summarize()`, `msim.diff()`, `msim.extinction()`, and `msim.compare()` work from the results.
- `msim.plot()` draws the median and an interval. `msim.plot(lines=True)` draws every run, because extinction, bimodality, and threshold behavior cannot be seen in a median.
- `ss.parallel(baseline=sim_a, vaccine=sim_b, n_runs=100)` returns one object whose results are indexed by scenario, run, and time ([08-interventions.md](08-interventions.md)): `msim.results.sir.n_I.select(scenario='vaccine')`. A sweep, `ss.parallel(sim, pars=dict(beta=[0.2, 0.4]), n_runs=10)`, has the values as coordinates of the runs: `msim.results.sir.n_I.select(beta=0.2).quantiles()`.
- `ss.parallel()` is the one way to run many sims, and `sim.run(n_runs=100)` is its shortcut for running one sim many times; both end in the same results.


Where extinction is possible, `msim.extinction()` reports it, for each declared disease and scenario: the share of runs with nobody infected at the end (not counting runs that started with nobody infected), and the median time it happened (`output=True` gives the table as a data frame):

```text
sir: died out in 41% of 500 runs (median time 23 days)
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
ss.Sim(sir, results=['n_I', 'cases'])
```

keeps only the named results by group (by name, or as `'sir.n_I'`); every other result keeps only its total over the groups, which is small. A name that is not a result is an error. The default is everything, which is the right default until stratification multiplies it: a result per state and per transition, for every stratum, at every step, can be a lot of memory.

So by default the values by group are kept at every step only while they fit in about 100 MB for a module in one run (500 MB over all the runs of `sim.run(n_runs=)`). Beyond that, they are kept every few steps: the shortest of a week, two weeks, four weeks, a quarter, half a year, or a year that fits. A row of values by group then holds a flow added up over its steps and a state as it is at the end of them, so `res.n_I.select(district='north')` is a series at those times, its flows add up to the totals, and `resample()` works to periods no shorter than those. The totals are still kept at every step. This is a guess, so it is said in a note and marked in `describe()`; naming fewer results with `results=` keeps those more often. Data by group given to the sim keep every step, since they are compared with the results at their own times.

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
- **One named result per stratum** (`n_I_female_high`). A stratified result is one result with dimensions.
- **Averaging ratios over strata.** Numerator and denominator are summed separately.
- **Several parallel time representations on every result.** One time vector, converted on request.
- **Four names for each flow** (`infection`, `cum_infection`, `new_infections`, `cum_infections`). A flow has the transition's name; its running total is `.cum`; the Starsim names are kept on the library's diseases.
- **A MultiSim reduced to one sim** (Starsim's `msim.reduce()`, `msim.mean()`, `msim.median()`, with bounds as `low`/`high` on flattened results). The results across runs are summarized directly: `res.quantiles()`, `res.median()`, `res.mean()`.
