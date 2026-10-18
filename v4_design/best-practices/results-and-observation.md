# Results and observation

## Recommendation

1. **One results concept.** Everything the model reports is a result, declared with the model. There is no separate observation step.
2. **Auto-generate the obvious ones.** A count for every state and a flow for every named transition, with no boilerplate.
3. **Adopt Atomica's characteristics: a named combination of states with an optional denominator.** Written as arithmetic on states: `prevalence = I / N`.
4. **Take flepiMoP's outcome cascade**: a reported quantity is `ss.outcome(...)`, a weighted sum of flows that is delayed or given a duration, and outcomes chain. Data by reporting period (weekly cases) is compared with the outcome's totals over each period, so no reporting period is declared. It is a shortcut for a reporting process the model does not represent, not a replacement for one it does.
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

### A result knows what kind of quantity it is

Weekly case counts are compared with the model's cases added up over each week, because cases are a flow; a prevalence survey is compared with the model's value on the survey date, because prevalence is a stock. A declared result already knows which it is (a flow, a stock, or a ratio), so the user is not asked; Starsim's calibration components ask for it as `conform='incident'` or `'prevalent'`, and odin's `zero_every` and camdl's `emit_schedule = every 7 'days` exist only to make the model produce the period totals.

### An outcome stands in for a process the model does not represent

If testing is not in the model ("about half of infections are reported, five days later"), cases are an outcome. If testing is modeled, the tests' own results are the cases, and an outcome is not used for them. If being counted changes what happens to the person (hospitalized people stop transmitting), it is a state, with transitions in and out. An outcome is a result, so anything can read it (a trigger on hospital occupancy, a rate), but it never records who was counted, which is the line between an outcome and a state. The same disease model then serves a hospital-burden question and a transmission question, without hospital compartments that play no part in transmission.

### The natural result of a stochastic model is a distribution

[Epydemix](../approaches/epydemix.md) makes `get_quantiles_compartments()` the default result form. Its review: "For any stochastic paradigm, the natural result object is a distribution over trajectories, not a trajectory. Epydemix makes this the default; most frameworks make the user aggregate." That is taken up where there are several trajectories to summarize: the results of a MultiSim have a run axis, and plot and summarize as a distribution without a separate reduction step, with every run kept. [`epidemics`](../approaches/epidemics.md)' scenario grid is the same idea one level up: one tidy frame with scenario identifiers.

### Transmission provenance is large

[EpiModel](../approaches/epimodel.md)'s `get_transmat()` records every infection with infector, infectee, time, and probability, and converts to a phylogeny; its review argues it "should be part of the standard result set rather than an opt-in". In practice the log is usually much larger than all the other results put together and recording it slows the run, while most models never look at it, so it is kept on request (Starsim's `infection_log` analyzer), and only in an agent-based run.

## The specification

[09-results.md](../design/09-results.md) specifies the automatic results (`n_<state>` for each state, and a flow named after each transition, with its running total as `.cum`), results declared as arithmetic on states, stratified results as one result with dimensions (`select()`, `by()`), `ss.outcome`, data attached by name (`ss.data`, with `aliases=` and `ignore_extra=`), results of many runs, and the infection log. Fitting to the data, the default goodness of fit, and likelihoods on outcomes are in [13-inference.md](../design/13-inference.md).

## Trade-offs

- **Auto-generating a result per state and per transition is memory the user did not ask for.** Bounded by `n_states + n_transitions` per timestep, which is small until stratification multiplies it. `ss.Sim(results=['n_I', 'cases'])` keeps only the named results by group (totals are always kept), and results by group too large to keep at every step are kept at a coarser interval, with a note; the default stays generous.
- **The outcome cascade duplicates what compartments can express.** A hospitalization can be a state (if hospitalized people transmit differently) or an outcome (if they do not). Both are legitimate and the choice is a modeling decision. Say so in the documentation rather than picking one.
- **Delays need convolution or extra state.** flepiMoP's delayframe method is discrete-time convolution. A `delay=` on an outcome given as a duration is an exact shift of the flow it comes from (a distribution spreads it out), while a plain `duration=` is an exponential stay, the same as a [dwell time](model-structure.md) given as a number.
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

The open questions from this document are in [19-open-questions.md](../design/19-open-questions.md).
