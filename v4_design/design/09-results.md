# Results and observation

What comes out, and how the model connects to data. The latent process and the observation process are separate.

Justified by [best-practices/results-and-observation.md](../best-practices/results-and-observation.md).

## Free by default

```python
sim = ss.Sim(sir).run()

sim.results.n_S, sim.results.n_I, sim.results.n_R    # one count per state
sim.results.infection, sim.results.recovery          # one flow per transition
sim.results.cum_infection                            # cumulative, free
```

Nothing was declared. The names are the keywords from the model text, which is why transitions are keyword arguments.

Normatively, a build MUST generate:

- `n_<state>` for every state, as a count at each output time;
- `<transition>` for every named transition, as the **flow through that transition during the interval** — incidence, not a stock;
- `cum_<transition>` for every named transition.

A transition's flow result is zeroed at the start of every output interval, so `sim.results.infection` is incidence per unit output time regardless of the internal timestep. Incidence per unit time is the single most commonly needed derived output in the field, and it MUST NOT require post-processing.

Stratified models generate the same results with dimension indices, selectable by keyword:

```python
sim.results.infection.sel(age=0)
sim.results.infection.sel(age=0, patch='Nairobi')
sim.results.n_I.sel(strain='alpha')
```

## `ss.Result`

A result carries a name, a label, its module, its shape, its timevec, its unit, and its provenance. This is what makes plots, exports, and generated forms self-describing, and it is retained from Starsim unchanged.

`sel()` selects along dimension coordinates and `by()` converts the time frame. Selection is a method rather than a keyword subscript because keyword arguments in a subscript are not valid Python, and the language does not put expressions in strings to work around syntax.

Results carry **one** timevec and convert on request (`result.by('year')`, `result.by('date')`). Parallel time representations belong to the `Timeline`, not to every result.

## Characteristics

A named combination of states with an optional denominator. It covers prevalence, coverage, and every cascade indicator in one form, and it unifies three things that are otherwise separate: what to fit to, what to plot, and what to report.

```python
sir.results(
    prevalence    = sir.I / ss.N,
    ever_infected = (sir.I + sir.R) / ss.N,
    incidence_u5  = sir.infection.sel(age=0),
    hosp_ratio    = sir.hosp / sir.cases,          # outcomes declared below
)
```

`ss.N` is the total population, always available and never declared. It is one of the few pieces of magic worth having, because making the user write the denominator by hand is what produces the missing-denominator class of error.

`Model.results(...)` **declares** characteristics; `sim.results` **holds** results. The two spellings are close and the distinction is real: one is a method on a model, the other an attribute on a run. Nothing else in the language reuses a name this way, and the alternative — a separate `characteristic=` keyword — would put a declaration in a constructor that cannot yet refer to the states it combines.

Characteristics are ordinary `ss.Result` objects thereafter: they can be plotted, exported, observed, and fitted to.

A cascade is an ordered list of characteristics, and `ss.cascade([tested, diagnosed, treated, suppressed])` is a plotting function rather than a language feature.

## The observation model

The disease is a disease; the surveillance system is declared separately. This separation is why the same disease model serves a hospital-burden question and a transmission question.

```python
sir.observe(
    cases  = ss.Outcome(sir.infection, p=0.5,  delay=ss.days(5)),
    hosp   = ss.Outcome(sir.infection, p=0.05, delay=ss.days(7), duration=ss.days(10)),
    deaths = ss.Outcome('hosp',        p=0.2,  delay=ss.days(14)),
)
```

```python
ss.Outcome(
    source,               # a transition, a characteristic, or the name of another outcome
    *,
    p          = 1.0,     # ascertainment or branching probability; may be a distribution
    delay      = None,    # reporting or progression delay; may be a distribution
    duration   = None,    # if given, also produces a prevalence series
    every      = None,    # emission schedule, e.g. ss.weeks(1)
    data       = None,    # observed values to compare against
    likelihood = None,    # how data relate to the projection
    dispersion = None,    # shorthand for a negative-binomial k
)
```

Semantics:

- **Outcomes chain.** A string source names another outcome, so the whole surveillance cascade — infection → case → hospitalization → death — is a few lines.
- **`duration=` produces both series.** `hosp` is the incidence of hospitalization and `n_hosp` is the number currently hospitalized. The incidence/prevalence duality is generated, not hand-rolled.
- **`p=` and `delay=` accept distributions**, because ascertainment varies and reporting delays are not point masses.
- **Outcomes are observational only.** An outcome MUST NOT feed back into the process: nothing in the model may read an outcome. If hospitalized people are isolated and stop transmitting, hospitalization is a *state*, not an outcome, and the language says so with `E701 OutcomeUsedAsState` naming the reader. This rule is what keeps the separation meaningful.

`observe()` is defined on `ss.Module` and is available on `ss.Sim`, which delegates to a sim-level module for outcomes that span several diseases (all-cause mortality, syndromic surveillance).

### Realization per backend

| Backend | Delay realization |
|---|---|
| Agent-based | sample a delay per event, schedule the report |
| Compartmental | discrete-time convolution, or Erlang stages for the delay distribution |
| Stochastic compartmental | as compartmental, with the thinning drawn |

A delay distribution in a compartmental backend becomes Erlang stages — the same expansion as a dwell time ([03-states-and-transitions.md](03-states-and-transitions.md)), which is a consistency worth relying on rather than a coincidence.

## The likelihood

The likelihood lives with the outcome that produces the quantity, because it is a statement about how data relate to that process.

```python
sir.observe(
    cases = ss.Outcome(sir.infection, p=0.5, delay=ss.days(5),
                       data='data/weekly_cases.csv',
                       every=ss.weeks(1),
                       likelihood=ss.neg_binomial(k=ss.Par(bounds=[0.1, 100]))),
)
```

- `data=` accepts a path, a dataframe, or an `ss.data(...)` reference; its provenance enters the parameter hash.
- `likelihood=` accepts `ss.poisson()`, `ss.neg_binomial(k=)`, `ss.normal(std=)`, `ss.binomial(n=)`, or a callable.
- Nuisance parameters of the likelihood — ascertainment `p`, dispersion `k` — are ordinary `ss.Par` objects with bounds, and are fitted alongside the process parameters.
- **Multiple observation streams are ordinary, not a special case.** Clinical cases *and* environmental surveillance *and* seroprevalence are three more keyword arguments.

Once this exists, calibration has nothing left to declare ([13-inference.md](13-inference.md)). Under-reporting (`p=`), reporting delay (`delay=`), and reporting schedule (`every=`) are three different things and MUST remain distinct.

## Stochastic results are distributions

For any stochastic method, the natural result object is a distribution over trajectories, not a trajectory.

```python
sim = ss.Sim(sir, n_reps=100).run()
sim.plot()                                     # median and interval by default
sim.results.n_I.quantiles([0.025, 0.5, 0.975])
sim.results.n_I.trajectories                   # still there
sim.plot(style='trajectories')                 # one word away
```

Quantiles are the default because aggregating is what everyone does anyway; individual trajectories remain reachable because sometimes the spaghetti is the point — extinction, bimodality, and threshold behavior are invisible in a median.

For methods where extinction is possible, the summary MUST report it, because it is the actual output of the paradigm:

```text
extinction: 41% of 500 replicates (median time 23 days)
peak prevalence (among non-extinct): 87 [61, 118]
```

## Transmission provenance

```python
sim.transmissions      # infector, infectee, time, route, layer, and transition
```

Every infection event, recorded. It is cheap, it is generally useful, and it enables phylodynamic comparison and transmission-tree reconstruction. It MUST be part of the standard result set in any agent-based or network backend, and MUST be **absent** rather than empty in a compartmental one — an empty table invites the reader to conclude there were no transmissions.

## Requesting less

```python
ss.Sim(sir, results=['n_I', 'cases'])
```

Restricts the result set to the named results and whatever they depend on. Result generation is a dependency graph, so pruning is exact rather than heuristic. The default is generous; at scale — stratified models with many dimensions — the default becomes a memory problem, and this is the release valve.

## Rejected

- **Results created on first use.** No declaration means no typo protection and no way to prune.
- **Reporters configured separately from the model.** The observation model is part of the model, and it hashes with the model.
- **Post-processing scripts for incidence.** A zeroed accumulator per named transition closes this.
- **A separate cascade analysis object.** An ordered list of characteristics is a plotting function.
- **Six parallel time representations on every result.** One timevec, converted on request.
