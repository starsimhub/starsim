# Results and observation

## Recommendation

1. **One results concept.** Everything the model reports is a result, declared with the model. There is no separate observation step.
2. **Auto-generate the obvious ones.** A count for every state and a flow for every named transition, with no boilerplate.
3. **Adopt Atomica's characteristics: a named combination of states with an optional denominator.** Written as arithmetic on states: `prevalence = I / N`.
4. **Take flepiMoP's outcome cascade**: a reported quantity is `ss.outcome(...)`, a weighted sum of flows that is delayed, given a duration, or totaled by reporting period, and outcomes chain. It is a shortcut for a reporting process the model does not represent, not a replacement for one it does.
5. **Data attaches to results by name.** `ss.Sim(sir, data='cases.csv')`, with columns named like results, as in Covasim. Results that have data are plotted against it.
6. **Incidence is a result, not something derived by post-processing.** odin's `zero_every` accumulator plus camdl's `incidence(transition)` is a complete answer.
7. **A sim is one trajectory. Many runs are a MultiSim, whose results are distributions**, and going from one to the other is one call.
8. **Record transmission events on request.** It enables phylodynamic comparison, and the log is too large to keep by default.

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
S, I, R   = ss.states(3)
N         = S + I + R
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)

sir = ss.Disease(
    infection, recovery,
    prevalence    = I / N,
    ever_infected = (I + R) / N,
)
```

This is Atomica's Components-and-Denominator in Python. Arithmetic on states never makes a new state: nobody is "in" `N`. `S + I + R` is a sum of states, and what a sum does depends on where it is put. In a rate it is a source of infection or a denominator. Given to the disease by name, as `prevalence` and `ever_infected` are here, it is a result, in which a state stands for the number of people in it, so `I / N` is a time series of prevalence. `N` itself is not a result of this model, because it was never given to the disease. (`I.prev`, the fraction of the whole population in `I`, exists for the most common case.)

The cases that need a rule:

- **A weighted sum** (`I + 0.5 * A`) is allowed, in a rate and in a result.
- **A ratio in a stratified model** is the sum of the numerator over the selected groups divided by the sum of the denominator. Prevalence among women is the women in `I` over the women in `N`; it is never an average of the age groups' prevalences.
- **Anything nonlinear** (a product of states, a saturating function) is written with counts, `I.n * S.n / N.n`, and is computed from the totals of whichever groups are selected.

After [stratifying](stratification.md), a result is still one result, now with a dimension for each stratification. Used as it is, it is the total; a group is selected by name, and `by()` gives one series per group. This is summer2's `source_strata` filter:

```python
res = sim.results.sir
res.n_I                                     # everyone in I
res.n_I.select(sex='female')                # women in I, all ages and risk groups
res.n_I.select(sex='female', risk='high')   # high-risk women in I
res.n_I.by('age')                           # one series per age group
res.infection.select(age=0)                 # incidence in the youngest age group
res.prevalence.select(risk='high')
```

Nobody adds up `I_0_female_low` through `I_55_male_high` by hand, and there are no such names: the strata are coordinates of one result, not separate results.

### Outcomes: the reporting cascade

**flepiMoP** (above) → **lingua franca**:

```python
prevalence = I / N
cases  = ss.outcome(0.5  * infection, delay=ss.days(5), every=ss.weeks(1))
hosp   = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
deaths = ss.outcome(0.2  * hosp,      delay=ss.days(14))

sir = ss.Disease(
    transitions = [infection, recovery],
    results     = [prevalence],
    outcomes    = [cases, hosp, deaths],
)
```

An outcome is a flow worked out from other flows. It moves nobody: no one changes state because they were counted.

- **The source** is a transition, another outcome, or a weighted sum of them. A number times a flow means that fraction of those events (a random fraction, in a stochastic run). Because the source can be another outcome, outcomes chain.
- **`delay=`** is a duration, a distribution (reporting delays are not point masses), or a function of the sim for anything irregular.
- **`duration=`** produces both `hosp` (admissions) and `n_hosp` (people currently in hospital), which is flepiMoP's automatic `currHosp` and is the incidence and prevalence pair that gets hand-rolled everywhere else.
- **`every=`** totals the outcome by reporting period, so `cases` above is a weekly count whether or not there is data to compare it with. It also takes a function that says whether today is a reporting day, and events wait for the next one, which covers "nothing is reported at weekends".

Reporting that depends on severity is a weighted sum, where the three names are the transitions into those states:

```python
cases = ss.outcome(0.01 * asymptomatic + 0.3 * symptomatic + 0.8 * severe, delay=ss.days(3))
```

In a two-transition model, results and outcomes can be given to the disease directly as keywords (`ss.Disease(infection, recovery, prevalence=I / N, cases=ss.outcome(...))`). Beyond that, `transitions=`, `results=`, and `outcomes=` keep the three kinds of thing apart, and the names come from the variables. Either way they all end up in the same place: `sim.results.sir.cases`, `sim.results.sir.n_hosp`.

### Outcome, intervention, or state

An outcome stands in for a process the model does not represent. Which of the three to use depends on what the model contains:

| The situation | Use |
|---|---|
| Testing and reporting are not in the model; "about half of infections are reported, five days later" | An outcome |
| Testing is modeled, by one or more interventions | The interventions' own results. Each testing intervention reports the diagnoses it made, and reported cases are their sum: `cases = symptomatic_testing.diagnosed + screening.diagnosed` |
| Being counted changes what happens to the person (hospitalized people are isolated and stop transmitting) | A state, with transitions in and out |

So an outcome never competes with a testing intervention for who "owns" cases: if the model has tests, the tests produce the cases, and `ss.outcome` is not used for them.

An outcome is a result, and anything can read a result: a trigger can close schools when `n_hosp` passes a threshold, and a rate can depend on it. What an outcome cannot do is single out individuals, because it never recorded who they were. That is the line between an outcome and a state, and it means the same disease model serves a hospital-burden question and a transmission question without hospital compartments that play no part in transmission.

### Data, by name

```python
sim = ss.Sim(sir, data='kenya_2020.csv', start='2020-01-01', stop='2020-12-31').run()
sim.plot()
```

`kenya_2020.csv` has a date column and columns named `cases`, `deaths`, and `prevalence`. Each column is attached to the result of the same name; a column that matches no result is an error that lists the result names. `sim.plot()` then draws each of those results with its data points. A data frame works in place of a file name.

Real files rarely arrive with matching headers, so `ss.data` renames on the way in:

```python
data = ss.data('kenya_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
sim  = ss.Sim(sir, data=data, start='2020-01-01', stop='2020-12-31')
```

`aliases` maps a result name to the column that holds it, and `ignore_extra=True` drops the columns that match nothing instead of raising.

The data does not need to be on the model's timestep. Weekly case counts are compared with the model's cases summed over each week, because `cases` is a flow; a prevalence survey is compared with the model's value on the survey date, because `prevalence` is a stock. The result already knows which it is. (Starsim's calibration components ask the user for this as `conform='incident'` or `'prevalent'`.)

That is all [calibration](calibration.md) needs: `sim.calibrate()` fits to every result that has data. The default measure of fit is the mean absolute difference between model and data, divided by the mean of the data, for each series. It assumes no noise model, and one bad reporting week does not dominate it the way it dominates a squared error. Nobody has to write a likelihood. When one is wanted, it goes on the result:

```python
cases = ss.outcome(0.5 * infection, delay=ss.days(5),
                   likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
```

which is camdl's `weekly_cases ~ neg_binomial(...)` and odin's `cases ~ Poisson(incidence)`. Several data series (clinical cases, wastewater, seroprevalence) are just more columns.

### One run, many runs

A sim is one trajectory, whatever the solver. Many runs are a MultiSim, as in Starsim, and going from one to the other is one call:

```python
sim  = ss.Sim(sir, solver='gillespie', total_pop=1000)
msim = ss.parallel(sim, n_runs=100)

msim.plot()                          # median and interval
msim.plot(lines=True)                # every run, for when the spaghetti is the point
n_I = msim.results.sir.n_I           # the same result, with a run dimension
n_I.quantiles([0.025, 0.5, 0.975])
n_I.values                           # the full array, one row per run
```

[Epydemix](../approaches/epydemix.md) makes `get_quantiles_compartments()` the default result form. Its review: "For any stochastic paradigm, the natural result object is a distribution over trajectories, not a trajectory. Epydemix makes this the default; most frameworks make the user aggregate." That is taken up at the level where there are several trajectories to summarize: a MultiSim plots and summarizes the distribution without a separate `reduce()` step, and every run stays available.

[`epidemics`](../approaches/epidemics.md)'s scenario grid is the same idea one level up: passing lists of elements returns one tidy frame with scenario identifiers. `ss.parallel(baseline=..., vaccine=...)` returns one object whose results are indexed by scenario, run, and time ([interventions.md](interventions.md#scenarios-in-the-file)).

### Transmission provenance

[EpiModel](../approaches/epimodel.md)'s `get_transmat()` records every infection event with infector, infectee, time, and probability, and converts to a phylogeny via `as.phylo.transmat()`. Its review argues it "should be part of the standard result set rather than an opt-in". In practice the log is usually much larger than all the other results put together and recording it slows the run, while most models never look at it. So it is one keyword, and off unless asked for, using Starsim's existing analyzer:

```python
sim = ss.Sim(sir, networks='random', analyzers='infection_log').run()
sim.analyzers.infection_log.to_df()   # infector, infectee, time, and the network it happened on
```

It exists only in an agent-based run; a compartmental run has no individuals to record. Where infection does not come from a person (a mosquito count, a water reservoir) the entry has the infectee and the source state, and no infector.

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
- **Matching data by column name depends on the names matching.** A typo in a column header is caught (unknown columns are errors), and `aliases` handles headers that were never going to match.
- **A median and interval hide individual trajectories**, and sometimes the spaghetti is the point (extinction, bimodality). `msim.plot(lines=True)` is one keyword away.
- **Agent-level outputs are not covered.** A line list, an age-at-infection distribution, and a partnership history are not time series. They are written as analyzers, as in Starsim, and that is plain code.

## Rejected

- **Results created on first use** (EpiModel's `set_epi`, `individual`'s `Render`). No declaration means no typo protection and no way to prune.
- **Reporters configured separately from the model** (EMOD). What is reported is part of the model; camdl is right about this.
- **A sim that is many trajectories.** An earlier draft gave a stochastic sim `n_reps=` and made its result a distribution. One sim, one run is easier to reason about, and Starsim's MultiSim already holds many.
- **Recording every transmission by default.** See above.
- **A separate observation step** (an earlier draft's `sim.observe(...)`). Outcomes are results, so they are declared where results are declared.
- **A required likelihood** (camdl, odin). Optional; see [calibration.md](calibration.md).
- **Post-processing scripts for incidence.** `zero_every` and `incidence(transition)` between them close this.
- **A separate cascade analysis object** (Atomica's Cascades sheet). A cascade is an ordered list of characteristics; once characteristics exist, plotting them in order is a plotting function, not a language feature.
- **Six parallel time representations as an output requirement.** Starsim's `tvec`/`tivec`/`timevec`/`yearvec`/`datevec`/`relvec` is convenient and, per its own review, "a lot of parallel representations". Results should carry one timevec and convert on request.

## Open questions

- **Is `ss.outcome` needed at all?** The line between a modeled state and an observed quantity is clear in the table above and may blur in practice: an outcome with a `duration=` is close to a state with a delay. If real models keep wanting outcomes to act on individuals, the better design may be a state that does not take part in transmission, and no outcome class.
- **Where do results that combine modules go?** `cases = symptomatic_testing.diagnosed + screening.diagnosed` belongs to neither intervention. It could be declared on the sim, or on the disease that both test for.
- **What a flow result is called.** Here it has the transition's name (`res.infection`), which matches the model text exactly. Starsim's `new_infections` reads better and is still generated for any disease with transmission, so one model has both. Whether to keep the exact match, switch to `new_<transition>`, or keep both needs trying on the vignettes.
- **Names for the many-run views.** `msim.plot(lines=True)` and `.values` are placeholders for "every run" and "the full array".
