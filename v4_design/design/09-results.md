# Results and data

What comes out of a run, and how it is connected to data. There is one results concept: everything below is a result, lives in `sim.results`, and can be plotted, exported, selected by stratum, and compared with data.

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

Library diseases keep the result names they have today (`sim.results.sir.n_infected` for `ss.SIR`), because their states have those names.

## `ss.Result`

A result carries its name, label, module, values, time vector, unit, and whether it is a flow or a stock. This is Starsim's `ss.Result`, unchanged, with two additions: selection by stratum, and attached data.

Because a result knows whether it is a flow or a stock, converting it to a coarser time step does the right thing without being told: `res.infection.resample('week')` sums, `res.n_I.resample('week')` takes the value at the end of the week, and a ratio is recomputed from its resampled parts. A result has one time vector; other time representations are conversions.

### Selecting strata

On a stratified model every result has the model's dimensions, and `.sel()` selects along them:

```python
sim.results.sir.new_infections.sel(age=0)                  # by position
sim.results.sir.infection.sel(age='0-14', patch='Nairobi') # by label
sim.results.sir.n_I.sel(strain='alpha')
sim.results.sir.prevalence.sel(risk='high')
```

Without `.sel()`, a result is the total over all strata (for a ratio, the ratio of the totals). In an agent-based run, selecting on a continuous property uses the bins it was stratified with ([05-dimensions.md](05-dimensions.md)).

## Declaring more results

Extra results are declared with the model, as keywords, next to the transitions. There is no separate observation step.

```python
S, I, R   = ss.states('S I R')
N         = S + I + R
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)

sir = ss.Disease(
    infection, recovery,
    prevalence    = I / N,
    ever_infected = (I + R) / N,
    cases  = ss.outcome(infection, p=0.5,  delay=ss.days(5)),
    hosp   = ss.outcome(infection, p=0.05, delay=ss.days(7), duration=ss.days(10)),
    deaths = ss.outcome('hosp',    p=0.2,  delay=ss.days(14)),
)
```

### Ratios and combinations

In a result, a state stands for the number of people in it. A sum of states is a combined count, and a ratio of two is a proportion: `I / N` is a time series of prevalence. Flows can be used the same way (`infection / S` is the per-capita incidence). A ratio keeps its numerator and denominator, so selecting a stratum or resampling in time recomputes it correctly instead of averaging ratios.

These are ordinary results afterwards: `sim.results.sir.ever_infected`.

### `ss.outcome`

An outcome is a reported quantity: a flow that has been thinned and delayed. It covers reported cases, hospital admissions, deaths, and similar series that are counted from the epidemic without being part of transmission.

```python
ss.outcome(
    source,               # a transition, another outcome, or a result expression; the object or its name
    p          = 1.0,     # the fraction that is reported, or that goes on to this outcome
    delay      = None,    # time from the source event to this one
    duration   = None,    # if given, also produces n_<name>: the number currently in this outcome
    likelihood = None,    # optional; see below
)
```

- **Outcomes chain.** The source of `deaths` above is the outcome `hosp`, so infection → hospitalization → death is three lines. A chain that loops back on itself raises `E703 OutcomeCycle`; a source that names nothing raises `E702 UnknownOutcomeSource`.
- **`duration=` gives both series.** `hosp` is admissions per step and `n_hosp` is the number currently in hospital.
- **`p=` and `delay=` accept the parameter ladder**, including distributions (reporting delays are not a single number) and `ss.by(age=[...])`. They can carry `bounds` and be fitted like any other parameter.
- **An outcome does not change anyone's state.** If hospitalized people stop transmitting, hospitalization should be a state in the disease; if it only needs counting, it is an outcome. Both are legitimate, and the choice is a modeling decision. Outcomes can be read like any other result, for example by a [trigger](08-interventions.md#triggers) that responds to reported cases.

For results that span modules (all-cause deaths, a syndromic count over several diseases), declare the outcome on the sim: `ss.Sim(modules=[flu, rsv], ili=ss.outcome(flu.infection + rsv.infection, p=0.3))`; it appears as `sim.results.ili`.

How the thinning and the delay are realized depends on the method:

| Method | `p=` | `delay=` |
|---|---|---|
| Agent-based | a draw per event | a delay sampled per event; the report is scheduled |
| Deterministic compartmental | multiplies the flow | a convolution of the flow with the delay distribution, or Erlang stages |
| Stochastic compartmental | a binomial draw from the flow | as deterministic, applied to the drawn counts |

## Data

Data is attached to results by name:

```python
sim = ss.Sim(sir, data='kenya_2020.csv', start='2020-01-01', stop='2020-12-31').run()
sim.plot()
```

- `data=` takes a file name or a data frame. It has a time column (`date`, `year`, `t`, or `time`) and one column per data series, **named like the result it corresponds to** (`cases`, `deaths`, `prevalence`). With several modules, a column is matched to the one module that has a result of that name, or is written `sir.cases` when the name is ambiguous.
- A column that matches no result is reported (`W705 UnknownDataColumn`, listing the result names closest to it) and ignored. Data entirely outside the simulated period raises `E704 DataMisaligned`.
- Data for particular strata is given in long form: extra columns named after dimensions (`age`, `patch`) say which stratum each row belongs to.
- The data need not be on the model's time step. A flow (weekly reported cases) is compared with the model's flow summed over each data interval; a stock or ratio (a prevalence survey) is compared with the model's value at the data's time. The result already knows which it is. The interval assumed for flow data is the spacing of the data and is printed.
- Results that have data are plotted with it automatically: `sim.plot()` draws the model as a line and the data as points.
- The data's source and content enter the run hash ([11-random.md](11-random.md)).

This is all that calibration needs: `sim.calibrate()` fits every parameter that has `bounds` to every result that has data ([13-inference.md](13-inference.md)). Several data series (clinical cases, wastewater, seroprevalence) are more columns.

### An optional likelihood

Nobody has to write a likelihood to compare a model with data or to fit it; the default measure of fit is a normalized root-mean-square error per series. When a likelihood is wanted, it is attached to the result:

```python
cases = ss.outcome(infection, p=0.5, delay=ss.days(5),
                   likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
seroprevalence = ss.outcome(R / N, likelihood=ss.normal(scale=0.02))
```

The model's value is the mean of the distribution and the user supplies the rest (here the dispersion, which is fitted). Any result can be given one by wrapping it in `ss.outcome`. A likelihood makes posterior methods available; [13-inference.md](13-inference.md) says what is done with it.

## Stochastic results are distributions

For a stochastic method the natural result is a distribution over trajectories.

```python
sim = ss.Sim(sir, method='stochastic', n_reps=100, unit='day').run()
sim.plot()                                         # median and interval
sim.results.sir.n_I.quantiles([0.025, 0.5, 0.975])
sim.results.sir.n_I.trajectories                   # every replicate
sim.plot(style='trajectories')
```

With `n_reps` above 1, each result holds all replicates; plotting and summaries show the median and an interval by default, and the individual trajectories stay available, because extinction, bimodality, and threshold behavior cannot be seen in a median.

Where extinction is possible, the results summary (`sim.summarize()` in Starsim) SHOULD report it:

```text
extinction: 41% of 500 replicates (median time 23 days)
peak prevalence (among non-extinct): 87 [61, 118]
```

## The transmission log

```python
sim.transmissions      # one row per infection: infector, infectee, time, disease, and network
```

Every infection event in an agent-based run, as a table. It supports transmission trees, phylodynamic comparison, and counting secondary cases directly. The infector is known when transmission happened on a route that keeps contacts; for infections through a mixing pool or from a count (`W.n`), the row records the source group or state instead. A compartmental run has no individuals to record, and there `sim.transmissions` is absent, not empty, so that an empty table is never mistaken for "no transmission".

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
- **Post-processing scripts for incidence.** Every named transition has a flow result.
- **A separate cascade analysis object.** An ordered list of results is a plot.
- **Several parallel time representations on every result.** One time vector, converted on request.
