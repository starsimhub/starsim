# Results and data

What comes out of a run, and how it is connected to data. There is one results concept: everything below is a result, lives in `sim.results`, and can be plotted, exported, selected by stratum, and compared with data. A sim is one trajectory; many runs are a MultiSim, whose results have a run dimension.

Justified by [best-practices/results-and-observation.md](../best-practices/results-and-observation.md).

## Automatic results

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
sim = ss.Sim(sir, unit='day').run()

res = sim.results.sir
res.n_S, res.n_I, res.n_R        # one count per state
res.infection, res.recovery      # one flow per transition: the number of events in each step
res.cum_infection                # cumulative
```

Nothing was declared. Results live under the module that produced them, `sim.results.<module>.<result>`, as in Starsim, and this access form is used throughout the specification.

A build MUST generate, for every module with states and transitions:

- `n_<state>` for every state: the number of people in it at each time point;
- `<transition>` for every named transition: the number of people who made that transition during each step (a flow, not a stock), named exactly as the transition is;
- `cum_<transition>`: its running total.

Starsim's standard disease results are kept under their existing names, so code and data files written for Starsim carry over. For any disease with transmission, `new_infections` is the total flow through all its transmission transitions, `cum_infections` its running total, and `prevalence` the fraction of the population in its infected states. In the SIR above, `res.new_infections` and `res.infection` are the same series. A result declared by the user under one of these names replaces the default.

What a flow result should be called is open ([19-open-questions.md](19-open-questions.md)). Here it has the transition's name, which matches the model text exactly; Starsim's `new_<transition>` reads better. Until the vignettes have been written both ways, the transition's name is the specified one and `new_infections` exists alongside it.

A state that is not people (a water reservoir, a mosquito population; [03-states-and-transitions.md](03-states-and-transitions.md)) gets the same results: `n_W` is the count it holds.

Library diseases keep the result names they have today (`sim.results.sir.n_infected` for `ss.SIR`), because their states have those names.

## `ss.Result`

A result carries its name, label, module, values, time vector, unit, and whether it is a flow or a stock. This is Starsim's `ss.Result`, unchanged, with two additions: selection by stratum, and attached data.

During a run, a result is read from inside the sim (by a trigger, a modifier, or custom code) as in Starsim: the storage is preallocated for the whole run, and the steps completed so far are `res.n_I[:sim.ti]`. A shorter spelling for "the steps so far" (`res.n_I.so_far`, or a slicing helper) would be convenient in triggers; it is listed as open and not specified here ([19-open-questions.md](19-open-questions.md)).

Because a result knows whether it is a flow or a stock, converting it to a coarser time step does the right thing without being told: `res.infection.resample('week')` sums, `res.n_I.resample('week')` takes the value at the end of the week, and a ratio is recomputed from its resampled parts. A result has one time vector; other time representations are conversions.

### Selecting strata

After stratifying, a result is still **one result**, now with a dimension for each stratification. The strata are coordinates of that result, not separate results: there is no `n_I_female_high`. Used as it is, the result is the total; `.select()` picks groups by name, and `.by()` gives one series per group:

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
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)

prevalence    = I / N
ever_infected = (I + R) / N
cases  = ss.outcome(0.5  * infection, delay=ss.days(5), every=ss.weeks(1))
hosp   = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
deaths = ss.outcome(0.2  * hosp,      delay=ss.days(14))

sir = ss.Disease(
    transitions = [infection, recovery],
    results     = [prevalence, ever_infected],
    outcomes    = [cases, hosp, deaths],
)
```

`transitions=`, `results=`, and `outcomes=` each take a list and keep the three kinds of thing apart; the members take their names from the variables, as transitions do ([03-states-and-transitions.md](03-states-and-transitions.md)). In a short model the same things can be given to the disease directly, as positional transitions and keywords:

```python
sir = ss.Disease(infection, recovery, prevalence=I / N, cases=ss.outcome(0.5 * infection, delay=ss.days(5)))
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
    duration   = None,    # if given, also produces n_<name>: the number currently in this outcome
    every      = None,    # a reporting period (ss.weeks(1)): the outcome is totaled over each period
    likelihood = None,    # optional; see below
)
```

- **The source is a weighted sum of flows.** A number times a flow means that fraction of those events: `0.5 * infection` is half of infections. There is no `p=` argument; the weight is the probability. Reporting that depends on severity is a sum, `ss.outcome(0.01 * asymptomatic + 0.3 * symptomatic + 0.8 * severe, delay=ss.days(3))`, where the three names are the transitions into those states. A weight follows the parameter ladder ([04-parameters.md](04-parameters.md)): it can be an `ss.par` with `bounds` and be fitted like any other parameter, or `ss.by(age=[...])`.
- **Outcomes chain.** The source of `deaths` above is the outcome `hosp`, so infection → hospitalization → death is three lines. A chain that loops back on itself raises `E703 OutcomeCycle`. Where a transition was declared as a keyword and there is no variable to refer to, the source may be its name (`ss.outcome('infection', delay=ss.days(5))`); a name that matches nothing raises `E702 UnknownOutcomeSource`.
- **`delay=`** is a duration, a distribution (reporting delays are not a single number), or a function of the sim for anything irregular.
- **`duration=` gives both series.** `hosp` is admissions per step and `n_hosp` is the number currently in hospital.
- **`every=` totals the outcome by reporting period**, so `cases` above is a weekly count whether or not there is data to compare it with. It also takes a function of the sim that says whether the current step is a reporting day; events wait for the next one, which covers "nothing is reported at weekends".

For results that span modules (all-cause deaths, a syndromic count over several diseases), declare the outcome on the sim: `ss.Sim([flu, rsv], ili=ss.outcome(0.3 * (flu.infection + rsv.infection)))`; it appears as `sim.results.ili`. (Whether such a result belongs on the sim or on one of the modules is open; see [19-open-questions.md](19-open-questions.md).)

How the weight, the delay, and the reporting period are realized depends on the solver:

| Solver | A weight on a flow | `delay=` | `every=` |
|---|---|---|---|
| Agent-based | a draw per event | a delay sampled per event; the report is scheduled | events are counted into the period they fall in |
| Deterministic compartmental | multiplies the flow | a convolution of the flow with the delay distribution, or Erlang stages | the flow is integrated over each period |
| Stochastic compartmental | a binomial draw from the flow | as deterministic, applied to the drawn counts | the drawn counts are summed over each period |

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

- `data=` takes a file name, a data frame, or an `ss.data(...)` object (below). It has a time column (`date`, `year`, `t`, or `time`) and one column per data series, **named like the result it corresponds to** (`cases`, `deaths`, `prevalence`). With several modules, a column is matched to the one module that has a result of that name, or is written `sir.cases` when the name is ambiguous.
- A column that matches no result is an error (`E705 UnknownDataColumn`) that lists the result names closest to it, so a typo in a header is caught. `ss.data(..., ignore_extra=True)` drops such columns instead. Data entirely outside the simulated period raises `E704 DataMisaligned`.
- Data for particular strata is given in long form: extra columns named after dimensions (`age`, `patch`) say which stratum each row belongs to.
- The data need not be on the model's time step. A flow (weekly reported cases) is compared with the model's flow summed over each data interval; a stock or ratio (a prevalence survey) is compared with the model's value at the data's time. The result already knows which it is. The interval assumed for flow data is the spacing of the data and is printed.
- Results that have data are plotted with it automatically: `sim.plot()` draws the model as a line and the data as points.
- The data's source and content are hashed with the fitting settings, not with the model ([11-random.md](11-random.md#run-identity)).

Real files rarely arrive with matching headers, so `ss.data` renames on the way in:

```python
ss.data(
    source,                 # a file name or a data frame
    aliases      = None,    # dict of result (or time-column) name -> the column that holds it
    ignore_extra = False,   # True: drop columns that match no result instead of raising
)

data = ss.data('kenya_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
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

The model's value is the mean of the distribution and the user supplies the rest (here the dispersion, which is fitted). A result that is not an outcome (a ratio such as `prevalence = I / N`) is given one at calibration time, `sim.calibrate(likelihood=dict(prevalence=ss.normal(scale=0.02)))`; where a likelihood for such a result should be declared with the model is open. A likelihood makes posterior methods available; [13-inference.md](13-inference.md) says what is done with it.

## One run, many runs

A sim is one trajectory, whatever the solver. Many runs are a MultiSim, as in Starsim, and going from one to the other is one call:

```python
sim  = ss.Sim(sir, solver='gillespie', total_pop=1000, unit='day')
msim = ss.parallel(sim, n_runs=100)          # or sim.run(n_runs=100), or ss.MultiSim(sim, n_runs=100).run()

msim.plot()                                  # median and interval
msim.plot(lines=True)                        # every run
n_I = msim.results.sir.n_I                   # the same result, with a run dimension
n_I.quantiles([0.025, 0.5, 0.975])           # one series per quantile
n_I.values                                   # the full array, runs by time
```

- `n_runs` belongs to the MultiSim, not to the sim. `ss.Sim` has no replicate count, and `sim.results` always holds one series per result. Each run has its own seed ([11-random.md](11-random.md)).
- **A MultiSim result is the same result with a `run` dimension.** `.select()` and `.by()` work as on a single sim (`msim.results.sir.n_I.select(age=0).quantiles([0.5])`), and `.select(run=3)` is one run.
- `.quantiles(q)` takes a list of quantiles and returns one series per quantile, over the runs. `.values` is the full array with one row per run.
- `msim.plot()` draws the median and an interval. `msim.plot(lines=True)` draws every run, because extinction, bimodality, and threshold behavior cannot be seen in a median.
- `ss.parallel(baseline=sim_a, vaccine=sim_b, n_runs=100)` returns one object whose results are indexed by scenario, run, and time ([08-interventions.md](08-interventions.md)): `msim.results.sir.n_I.select(scenario='vaccine')`.

The names `lines=True` and `.values` for "every run" and "the full array" are placeholders and are listed as open.

Where extinction is possible, the MultiSim's results summary (`msim.summarize()`, Starsim's name) SHOULD report it:

```text
extinction: 41% of 500 runs (median time 23 days)
peak prevalence (among non-extinct): 87 [61, 118]
```

## The transmission log

The log of who infected whom is usually much larger than all the other results put together, recording it slows the run, and most models never look at it. So it is not recorded by default. It is requested with Starsim's existing analyzer:

```python
sim = ss.Sim(sir, networks='random', analyzers='infection_log').run()
sim.analyzers.infection_log.to_df()   # one row per infection: infector, infectee, time, and the network it happened on
```

Every infection event in an agent-based run, as a table. It supports transmission trees, phylodynamic comparison, and counting secondary cases directly. The infector is known when transmission happened on a route that keeps contacts; for infections through a mixing pool the row records the source group, and where infection does not come from a person (a mosquito count, a water reservoir, `W.n`) it records the infectee and the source state, with no infector. The log exists only in an agent-based run: a compartmental run has no individuals to record, so asking for it makes a model agent-based under `solver='auto'` and is an error (`E804 CapabilityError`) under an explicitly compartmental solver ([12-backends.md](12-backends.md)).

## Requesting fewer results

```python
ss.Sim(sir, results=['n_I', 'cases'])
```

keeps only the named results and whatever they are computed from. The default is everything, which is the right default until stratification multiplies it: a result per state and per transition, for every stratum, at every step, can be a lot of memory. Since results are declared, the library knows what each depends on and can drop the rest.

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
- **Recording every transmission by default.** The log is large and slows the run; it is one keyword when wanted.
- **One named result per stratum** (`n_I_female_high`). A stratified result is one result with dimensions.
- **Averaging ratios over strata.** Numerator and denominator are summed separately.
- **Several parallel time representations on every result.** One time vector, converted on request.
