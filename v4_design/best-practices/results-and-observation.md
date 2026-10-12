# Results and observation

## Recommendation

1. **One results concept.** Everything the model reports is a result, declared with the model. There is no separate observation step.
2. **Auto-generate the obvious ones.** A count for every state and a flow for every named transition, with no boilerplate.
3. **Adopt Atomica's characteristics: a named combination of states with an optional denominator.** Written as arithmetic on states: `prevalence = I / N`.
4. **Take flepiMoP's outcome cascade essentially as-is**: source × probability × delay × duration, chainable. A reported quantity is `ss.outcome(...)`.
5. **Data attaches to results by name.** `ss.Sim(sir, data='cases.csv')`, with columns named like results, as in Covasim. Results that have data are plotted against it.
6. **Incidence is a result, not something derived by post-processing.** odin's `zero_every` accumulator plus camdl's `incidence(transition)` is a complete answer.
7. **For any stochastic run, the default result is a distribution over trajectories, not a trajectory.**
8. **Record transmission events.** Cheap, generally useful, and it enables phylodynamic comparison.

## Why

### Declared results remove real boilerplate

[Starsim](../approaches/starsim.md) declares results (`define_results`) rather than creating them on demand, and "Boolean states automatically generate an `n_<state>` result, which removes the most common piece of result boilerplate". A `ss.Result` carries a name, label, module, shape, timevec, and unit, so plots and exports are self-describing.

Against that: [EpiModel](../approaches/epimodel.md)'s `set_epi(dat, name, at, value)` creates outputs on first use, with no declaration and therefore no protection from typos; [`individual`](../approaches/individual.md) has "no results vocabulary beyond `Render` collecting numbers a process pushes into it".

The extension is obvious once [transitions are data](model-structure.md): a named transition generates a flow result exactly as a state generates a count. From the Starsim implications: "With declared transitions, flows (`new_infections`, `si.flow`) can be auto-generated the same way."

### Incidence is the most-needed derived output and almost nobody declares it

[odin](../approaches/odin.md):

```r
initial(incidence, zero_every = 1) <- 0
```

Its review: "Incidence-per-unit-time is the single most commonly needed derived output in the field, and this is the cleanest declaration of it anywhere in the review."

[camdl](../approaches/camdl.md) supplies the other half — `incidence(transition)` names a *transition*, not a compartment, so "the flow through a named edge is a first-class observable. That only works because transitions are named objects."

[summer2](../approaches/summer2.md) has `request_output_for_flow("incidence", "infection", source_strata={"age": "00"})` — a named transition with a stratum filter.

### Characteristics

[Atomica](../approaches/atomica.md)'s framework workbook has a Characteristics sheet:

| Code Name | Components | Denominator |
|---|---|---|
| ch_all | sus, inf, rec | |
| ch_prev | inf | ch_all |
| ch_infrec | inf, rec | |

From its review: "Prevalence is not a compartment; it is a named ratio of a compartment set to a denominator set, declared once and available to functions, data entry, and output. This is a cleaner treatment of derived outputs than most code-based frameworks manage."

It unifies three things that are otherwise separate: what to fit to, what to plot, and what to report.

### Reported quantities

What surveillance sees is not what the model simulates: cases are a fraction of infections, reported late. Two frameworks describe that step; the rest compare simulated compartment counts to data directly.

**flepiMoP** ([approaches/flepimop.md](../approaches/flepimop.md)) — "the best thing in flepiMoP":

```yaml
outcomes:
  incidCase:  {source: {incidence: {infection_stage: "I"}}, probability: {value: 0.5},  delay: {value: 5}}
  incidHosp:  {source: {incidence: {infection_stage: "I"}}, probability: {value: 0.05}, delay: {value: 7},
               duration: {value: 10, name: currHosp}}
  incidDeath: {source: incidHosp, probability: {value: 0.2}, delay: {value: 14}}
```

Outcomes **chain** — `incidDeath` takes `incidHosp` as its source — so the whole surveillance cascade is a dozen lines, and `duration` with a name automatically produces the prevalence series alongside the incidence one.

**camdl** ([approaches/camdl.md](../approaches/camdl.md)) ties each reported quantity to a data file and a likelihood, and multiple data series are ordinary:

```camdl
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

[odin](../approaches/odin.md) does the likelihood in two lines (`cases <- data()`, `cases ~ Poisson(incidence)`) and gets a particle filter from dust.

flepiMoP has the cascade and camdl has the link to data. Nobody has both, and both camdl and odin require a likelihood before any comparison to data can happen. [Covasim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) shows the simplest link that works: a data file whose column names match result names, plotted against them automatically.

## The proposal

### Automatic results

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
sim = ss.Sim(sir, unit='day').run()

res = sim.results.sir
res.n_S, res.n_I, res.n_R        # one per state
res.infection, res.recovery      # one per transition (new events per step)
res.cum_infection                # cumulative
```

Nothing was declared. The names are the ones in the model text, which is the point of transitions having names. Results live under the module that produced them, as in Starsim.

### Ratios and combinations

Extra results are declared with the model, as keywords. With the states in scope they are plain arithmetic:

```python
S, I, R   = ss.states('S I R')
N         = S + I + R
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)

sir = ss.Disease(
    infection, recovery,
    prevalence    = I / N,
    ever_infected = (I + R) / N,
)
```

This is Atomica's Components-and-Denominator in Python. A sum of states is a named combination; a ratio of two is a characteristic. In a result, a state stands for the number of people in it, so `I / N` is a time series of prevalence. (`I.prev`, the fraction of the whole population in `I`, exists for the most common case.)

After [stratifying](stratification.md), any result can be selected by stratum, which is summer2's `source_strata` filter:

```python
sim.results.sir.infection.sel(age=0)      # incidence in the youngest age group
sim.results.sir.prevalence.sel(risk='high')
```

### Outcomes: the reporting cascade

**flepiMoP** (above) → **lingua franca**, in the same place as the other results:

```python
sir = ss.Disease(
    infection, recovery,
    prevalence = I / N,
    cases  = ss.outcome(infection, p=0.5,  delay=ss.days(5)),
    hosp   = ss.outcome(infection, p=0.05, delay=ss.days(7), duration=ss.days(10)),
    deaths = ss.outcome('hosp',    p=0.2,  delay=ss.days(14)),
)
```

An outcome is a flow that has been thinned (`p=`) and delayed (`delay=`). Its source is a transition or another outcome, given as the object or by name, so outcomes chain. `duration=` produces both `hosp` (admissions) and `n_hosp` (people currently in hospital), which is flepiMoP's automatic `currHosp` and is the incidence/prevalence pair that gets hand-rolled everywhere else. `p=` and `delay=` accept distributions, because ascertainment varies and reporting delays are not point masses.

This replaces adding hospitalization and death compartments to the disease model when they are not part of transmission. The same disease model then serves a hospital-burden question and a transmission question. Outcomes are results like any other: `sim.results.sir.cases`, `sim.results.sir.n_hosp`.

### Data, by name

```python
sim = ss.Sim(sir, data='kenya_2020.csv', start='2020-01-01', stop='2020-12-31').run()
sim.plot()
```

`kenya_2020.csv` has a date column and columns named `cases`, `deaths`, and `prevalence`. Each column is attached to the result of the same name; a column that matches no result is an error that lists the result names. `sim.plot()` then draws each of those results with its data points. A data frame works in place of a file name.

The data does not need to be on the model's timestep. Weekly case counts are compared with the model's cases summed over each week, because `cases` is a flow; a prevalence survey is compared with the model's value on the survey date, because `prevalence` is a stock. The result already knows which it is. (Starsim's calibration components ask the user for this as `conform='incident'` or `'prevalent'`.)

That is all [calibration](calibration.md) needs: `sim.calibrate()` fits to every result that has data, using a normalized root-mean-square error per series. Nobody has to write a likelihood. When one is wanted, it goes on the result:

```python
cases = ss.outcome(infection, p=0.5, delay=ss.days(5),
                   likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
```

which is camdl's `weekly_cases ~ neg_binomial(...)` and odin's `cases ~ Poisson(incidence)`. Several data series (clinical cases, wastewater, seroprevalence) are just more columns.

### Stochastic results are distributions

[Epydemix](../approaches/epydemix.md) makes `get_quantiles_compartments()` the default result form. Its review: "For any stochastic paradigm, the natural result object is a distribution over trajectories, not a trajectory. Epydemix makes this the default; most frameworks make the user aggregate."

```python
sim = ss.Sim(sir, method='stochastic', n_reps=100).run()
sim.plot()                        # median and interval, not 100 spaghetti lines
sim.results.sir.n_I.quantiles([0.025, 0.5, 0.975])
sim.results.sir.n_I.trajectories  # still there when you want it
```

And [`epidemics`](../approaches/epidemics.md)'s scenario grid is the same idea one level up: passing lists of elements returns one tidy frame with scenario identifiers. `ss.parallel(baseline=..., vaccine=...)` should return a result object indexed by scenario, replicate, and time — the [default unit of a run is a set of scenarios](interventions.md#scenarios-in-the-file), not a trajectory.

### Transmission provenance

[EpiModel](../approaches/epimodel.md)'s `get_transmat()` records every infection event with infector, infectee, time, and probability, and converts to a phylogeny via `as.phylo.transmat()`. Its review: "cheap, generally useful, and enables phylodynamic comparison. It should be part of the standard result set rather than an opt-in."

```python
sim.transmissions   # infector, infectee, time, route, and the layer it happened on
```

Available in any agent-based run at no extra cost. A compartmental run has no individuals to record, so the attribute is absent rather than empty.

### Requesting less

Starsim's `sim.to_df()` flattens everything, and at scale that is a memory problem. summer2's graph of derived outputs "can be pruned to only what is requested"; [LASER](../approaches/laser.md) preallocates a slot per timestep and location, which requires knowing the output shape in advance.

```python
ss.Sim(sir, results=['n_I', 'cases'])   # only these, and whatever they depend on
```

The results are already declared, so the library knows what each one depends on.

## Trade-offs

- **Auto-generating a result per state and per transition is memory the user did not ask for.** Bounded by `n_states + n_transitions` per timestep, which is small until stratification multiplies it. The pruning above is the answer, and the default should stay generous.
- **The outcome cascade duplicates what compartments can express.** A hospitalization can be a state (if hospitalized people transmit differently) or an outcome (if they do not). Both are legitimate and the choice is a modeling decision. Say so in the documentation rather than picking one.
- **Delays in an ODE run need convolution or extra state.** flepiMoP's delayframe method is discrete-time convolution. In an ODE run a delay distribution becomes Erlang stages, the same expansion as [dwell times](model-structure.md).
- **Matching data by column name depends on the names matching.** A typo in a column header is caught (unknown columns are errors), but the user does have to rename `Reported Cases (weekly)` to `cases`.
- **Quantiles-by-default hides individual trajectories**, and sometimes the spaghetti is the point (extinction, bimodality). `sim.plot(style='trajectories')` must be one word away.

## Rejected

- **Results created on first use** (EpiModel's `set_epi`, `individual`'s `Render`). No declaration means no typo protection and no way to prune.
- **Reporters configured separately from the model** (EMOD). What is reported is part of the model; camdl is right about this.
- **A separate observation step** (an earlier draft's `sim.observe(...)`). Outcomes are results, so they are declared where results are declared.
- **A required likelihood** (camdl, odin). Optional; see [calibration.md](calibration.md).
- **Post-processing scripts for incidence.** `zero_every` and `incidence(transition)` between them close this.
- **A separate cascade analysis object** (Atomica's Cascades sheet). A cascade is an ordered list of characteristics; once characteristics exist, plotting them in order is a plotting function, not a language feature.
- **Six parallel time representations as an output requirement.** Starsim's `tvec`/`tivec`/`timevec`/`yearvec`/`datevec`/`relvec` is convenient and, per its own review, "a lot of parallel representations". Results should carry one timevec and convert on request.

## Open questions

- **Should an outcome be able to feed back into the model?** "Hospitalized people are isolated and stop transmitting" makes the outcome causal, at which point it should have been a state. A clean rule is needed, and "outcomes are observational only" is probably it.
- **Where do agent-level outputs go?** A line list, an age-at-infection distribution, and a partnership history are not time series. Starsim's `ss.Analyzer` handles this and stays imperative; that may be correct.
- **Under-reporting, reporting delay, and reporting schedule** are three different things. flepiMoP handles the first two with `probability` and `delay`; camdl handles the third with `emit_schedule`. Here `p=` and `delay=` are on the outcome and the schedule comes from the dates in the data. Whether a schedule is ever needed without data (for example, to output weekly totals) is not settled.
- **The denominator.** `N = S + I + R` is explicit and needs no new name; a built-in total population would save a line and would also count people outside the disease's states. Which is the better default is open.
