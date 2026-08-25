# Results and observation

## Recommendation

1. **Declare results; auto-generate the obvious ones.** A count for every state and a flow for every named transition, with no boilerplate.
2. **Adopt Atomica's characteristics: a named combination of states with an optional denominator.** It covers prevalence, coverage, and every cascade indicator in one form.
3. **Separate the latent process from the observation process.** flepiMoP declares how a process becomes an observed quantity; camdl declares how that quantity relates to data. Both halves are needed.
4. **Take flepiMoP's outcome cascade essentially as-is**: source × probability × delay × duration, chainable.
5. **Incidence is declared, not derived by post-processing.** odin's `zero_every` accumulator plus camdl's `incidence(transition)` is a complete answer.
6. **For any stochastic paradigm, the default result is a distribution over trajectories, not a trajectory.**
7. **Record transmission events.** Cheap, generally useful, and it enables phylodynamic comparison.

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

### The observation model

Two frameworks have one; the rest compare simulated compartment counts to data directly.

**flepiMoP** ([approaches/flepimop.md](../approaches/flepimop.md)) — "the best thing in flepiMoP":

```yaml
outcomes:
  incidCase:  {source: {incidence: {infection_stage: "I"}}, probability: {value: 0.5},  delay: {value: 5}}
  incidHosp:  {source: {incidence: {infection_stage: "I"}}, probability: {value: 0.05}, delay: {value: 7},
               duration: {value: 10, name: currHosp}}
  incidDeath: {source: incidHosp, probability: {value: 0.2}, delay: {value: 14}}
```

Outcomes **chain** — `incidDeath` takes `incidHosp` as its source — so the whole surveillance cascade is a dozen lines, and `duration` with a name automatically produces the prevalence series alongside the incidence one.

**camdl** ([approaches/camdl.md](../approaches/camdl.md)) supplies the likelihood, and multiple streams are ordinary:

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

[odin](../approaches/odin.md) does the likelihood in two lines (`cases <- data()`, `cases ~ Poisson(incidence)`) and gets a particle filter from dust for free.

Nobody has both halves. That is the gap.

## The proposal

### Free by default

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
sim = ss.Sim(sir).run()

sim.results.n_S, sim.results.n_I, sim.results.n_R    # one per state
sim.results.infection, sim.results.recovery          # one per transition (incidence)
sim.results.cum_infection                            # cumulative, free
```

Nothing was declared. The names are the ones in the model text, which is the point of transitions being keyword arguments.

### Characteristics

```python
sir.results(
    prevalence   = sir.I / ss.N,
    ever_infected = (sir.I + sir.R) / ss.N,
    incidence_u5 = sir.infection[age=0],
)
```

Atomica's Components-and-Denominator in Python, with summer2's stratum filter on the same line. `ss.N` is the total population, always available, never declared — one of the few pieces of magic worth having, because writing it out is what produces the missing-`/N` error class.

### Outcomes: the reporting cascade

**flepiMoP** (above) → **lingua franca:**

```python
sim.observe(
    cases  = ss.Outcome(sir.infection, p=0.5,  delay=ss.days(5)),
    hosp   = ss.Outcome(sir.infection, p=0.05, delay=ss.days(7), duration=ss.days(10)),
    deaths = ss.Outcome('hosp',        p=0.2,  delay=ss.days(14)),
)
```

`duration=` produces both `hosp` (incidence) and `n_hosp` (prevalence), which is flepiMoP's automatic `currHosp` and is exactly the incidence/prevalence duality that gets hand-rolled everywhere else. Sources chain by name. `p=` and `delay=` accept distributions, not just scalars, because ascertainment varies and reporting delays are not point masses.

Note what this replaces: adding hospitalization and death compartments to the disease model when they are not part of transmission. The disease stays a disease; the surveillance system is declared separately. That separation is the reason the same disease model serves a hospital-burden question and a transmission question.

### Likelihood, next to the outcome

```python
sim.observe(
    cases = ss.Outcome(sir.infection, p=0.5, delay=ss.days(5),
                       data='data/weekly_cases.csv',
                       every=ss.weeks(1),
                       likelihood=ss.neg_binomial(k=ss.Par(bounds=[0.1, 100]))),
)
```

That is camdl's `observations` block and odin's `cases ~ Poisson(incidence)` in the same place as flepiMoP's cascade — the two halves joined. Multiple streams (clinical cases *and* environmental surveillance *and* seroprevalence) are just more keyword arguments, which camdl's review notes should be "ordinary, not a special case".

Once this exists, [calibration.md](calibration.md) has nothing left to declare.

### Stochastic results are distributions

[Epydemix](../approaches/epydemix.md) makes `get_quantiles_compartments()` the default result form. Its review: "For any stochastic paradigm, the natural result object is a distribution over trajectories, not a trajectory. Epydemix makes this the default; most frameworks make the user aggregate."

```python
sim = ss.Sim(sir, n_reps=100).run()
sim.plot()                        # median and interval, not 100 spaghetti lines
sim.results.n_I.quantiles([0.025, 0.5, 0.975])
sim.results.n_I.trajectories      # still there when you want it
```

And [`epidemics`](../approaches/epidemics.md)'s scenario grid is the same idea one level up: passing lists of elements returns one tidy frame with scenario identifiers. `ss.parallel(baseline=..., vaccine=...)` should return a result object indexed by scenario, replicate, and time — the [default unit of a run is a set of scenarios](interventions.md#scenarios-in-the-file), not a trajectory.

### Transmission provenance

[EpiModel](../approaches/epimodel.md)'s `get_transmat()` records every infection event with infector, infectee, time, and probability, and converts to a phylogeny via `as.phylo.transmat()`. Its review: "cheap, generally useful, and enables phylodynamic comparison. It should be part of the standard result set rather than an opt-in."

```python
sim.transmissions   # infector, infectee, time, route, and the layer it happened on
```

Free in any agent-based or network backend; meaningless in a compartmental one, where it should be absent rather than empty.

### Requesting less

Starsim's `sim.to_df()` flattens everything, and at scale that is a memory problem. summer2's derived-output DAG "can be pruned to only what is requested"; [LASER](../approaches/laser.md) preallocates a slot per timestep and location, which requires knowing the output shape in advance.

```python
ss.Sim(sir, results=['n_I', 'cases'])   # only these, and the DAG behind them
```

The declaration that makes this possible already exists; it just needs to be honored.

## Trade-offs

- **Auto-generating a result per state and per transition is memory the user did not ask for.** Bounded by `n_states + n_transitions` per timestep, which is small until stratification multiplies it. The pruning above is the answer, and the default should stay generous.
- **The outcome cascade duplicates what compartments can express.** A hospitalization can be a state (if hospitalized people transmit differently) or an outcome (if they do not). Both are legitimate and the choice is a modeling decision. Say so in the documentation rather than picking one.
- **Delays in an ODE backend need convolution or extra state.** flepiMoP's delayframe method is discrete-time convolution. In a compartmental backend a delay distribution becomes Erlang stages — the same expansion as [dwell times](model-structure.md), which is a nice consistency.
- **Quantiles-by-default hides individual trajectories**, and sometimes the spaghetti is the point (extinction, bimodality). `sim.plot(style='trajectories')` must be one word away.

## Rejected

- **Results created on first use** (EpiModel's `set_epi`, `individual`'s `Render`). No declaration means no typo protection and no way to prune.
- **Reporters configured separately from the model** (EMOD). The observation model is part of the model — camdl is right about this.
- **Post-processing scripts for incidence.** `zero_every` and `incidence(transition)` between them close this.
- **A separate cascade analysis object** (Atomica's Cascades sheet). A cascade is an ordered list of characteristics; once characteristics exist, `ss.cascade([tested, diagnosed, treated, suppressed])` is a plotting function, not a language feature.
- **Six parallel time representations as an output requirement.** Starsim's `tvec`/`tivec`/`timevec`/`yearvec`/`datevec`/`relvec` is convenient and, per its own review, "a lot of parallel representations". Results should carry one timevec and convert on request.

## Open questions

- **Should an outcome be able to feed back into the model?** "Hospitalized people are isolated and stop transmitting" makes the outcome causal, at which point it should have been a state. A clean rule is needed, and "outcomes are observational only" is probably it.
- **Where do agent-level outputs go?** A line list, an age-at-infection distribution, and a partnership history are not time series. Starsim's `ss.Analyzer` handles this and stays imperative; that may be correct.
- **Under-reporting versus reporting delay versus reporting schedule** are three different things that flepiMoP handles with `probability`, `delay`, and camdl handles with `emit_schedule`. Keeping all three distinct is right; whether they belong on one object is not settled.
