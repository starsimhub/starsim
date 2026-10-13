# Stochastic compartmental

Gillespie/CTMC, tau-leaping, chain-binomial, and diffusion (SDE). One model, four ways of being random.

## Recommendation

1. **Stochasticity is a run-time choice with several named solvers**, not a property of the model.
2. **Prefer continuous time where the model allows it**, because it eliminates rate-to-probability conversion entirely.
3. **Name the approximation.** Gillespie, tau-leaping, chain-binomial, and diffusion are different models with different validity conditions, and the result should say which one produced it.
4. **One sim is one trajectory; the distribution comes from many runs**, and running many is one call.
5. **Quantities that are not people are states too.** An environmental reservoir or a vector population is a state holding a count, with its own transitions, and disease rates may use that count directly.
6. **Extinction is the reason this paradigm exists**; make it visible.

## Why

### One model, several stochastic methods, already demonstrated

[camdl](../approaches/camdl.md) runs the same model text on Gillespie SSA (exact continuous-time), chain-binomial (Euler-multinomial discrete-time), and ODE. [epipack](../approaches/epipack.md) runs the same process tuples deterministically and stochastically, including "a correct inhomogeneous-Poisson Gillespie implementation" for time-varying rates — "a real capability that most frameworks skip or approximate". [odin](../approaches/odin.md) shares `initial()` between `deriv()` and `update()`.

[PyRoss](../approaches/notes.md#pyross) is the sharpest evidence for the *ladder* rather than the binary: it implements Gillespie, tau-leaping, and Gaussian/van Kampen approximations of the same model side by side. [MEmilio](../approaches/memilio.md) has a whole `sde_*` family alongside its `ode_*` family, separately implemented.

### Continuous time removes an error class

[SimInf](../approaches/siminf.md) is continuous-time Gillespie throughout, and its review makes the point directly: "Being continuous-time is a real advantage for correctness — the exponential-conversion errors that afflict discrete-time frameworks simply do not arise — and a real limitation for scale, since Gillespie cost grows with event count." [EoN](../approaches/eon.md) is continuous-time for the same reason.

The implication drawn in the SimInf review: "The lingua franca should treat discrete-time versus continuous-time as a backend choice with a declared cost, not as a property of the model."

### Disease state coupled to an environment

SimInf's `u`/`v` split (integer compartment counts governed by the Markov chain, alongside continuous per-node variables updated by a post-timestep function) is used by its `SISe` family for a SIS disease coupled to an environmental compartment. Its review: "Environmental reservoirs, accumulated exposure, and vector abundance are all naturally continuous and coupled to a discrete disease process." The need is real. The lingua franca meets it without a second kind of state: the reservoir is an ordinary state that holds a count instead of people, and rates are written in terms of that count.

## The proposal

### The solver ladder

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.3),
    recovery  = ss.progression('I -> R', dur=10),
)
kw = dict(total_pop=1000, unit='day')

ss.Sim(sir, **kw).run()                    # auto, which for this model is solver='ode'
ss.Sim(sir, **kw).run(solver='gillespie')  # exact continuous-time
ss.Sim(sir, **kw).run(solver='tau')        # tau-leaping
ss.Sim(sir, **kw).run(solver='binomial')   # chain-binomial, discrete time
ss.Sim(sir, **kw).run(solver='sde')        # diffusion approximation
ss.Sim(sir, **kw).run(solver='abm')        # agent-based
```

Same `sir`, same homogeneous mixing (the default when no network is given). Six answers to the same model, each valid under different conditions.

The exact method is called `'gillespie'` because that is the name the field uses for it (EoN, epipack, SimInf, camdl, and PyRoss all say Gillespie). `'ctmc'`, the name of the process it simulates, is accepted as an alias.

`solver='auto'` chooses between the ODE and agents; it does not pick a stochastic compartmental solver by itself. A user who wants one but does not want to choose which asks for `solver='stochastic'`:

```text
>>> ss.Sim(sir, total_pop=1000, unit='day').run(solver='stochastic')
total_pop = 1000, 3 states, 2 transitions
solver: gillespie (exact; the number of events is manageable at this population size)
```

```text
>>> ss.Sim(sir, total_pop=10_000_000, unit='day').run(solver='stochastic')
solver: tau (exact Gillespie would need about 10^7 events)
  note: tau-leaping is an approximation; step chosen adaptively, leap condition 0.03
```

This is the same policy as everywhere else ([paradigm-conversion.md](paradigm-conversion.md)): when the library approximates, it says so in one line. A trajectory that came from tau-leaping is not a trajectory from the CTMC, and the result should record which it was. A user who asks for `solver='gillespie'` gets the exact method however long it takes.

### Validity conditions, checked

Each solver has conditions, and the model is data, so they can be checked rather than documented:

| Solver | Valid when | Check |
|---|---|---|
| Gillespie | Always (it is exact) | Event count feasible |
| Tau-leaping | Propensities roughly constant over τ | Leap condition, adaptively |
| Chain-binomial | `rate × dt` small | Warn above ~0.1; `check_dt()` |
| SDE / diffusion | Counts large in every compartment | Warn when any compartment falls below ~20 |
| ODE | Counts large, no extinction question | Warn when the epidemic could go extinct |

The SDE warning matters because the diffusion approximation is "valid only at moderate-to-large populations" ([MEmilio](../approaches/memilio.md)) and fails exactly where stochasticity was the reason to use it. These are warnings, not errors: the run goes ahead and the user is told.

### Time-varying rates in every solver

[epipack](../approaches/epipack.md) sets the standard: a rate may be a number, a callable of `(t, y)`, or a symbolic expression, and the framework does the right thing in ODE, Gillespie, and inhomogeneous-Poisson modes. Its review: "a rate is an expression in time and state, and each backend is responsible for handling that correctly or refusing."

```python
ss.modify(sir.pars.beta, multiply=ss.seasonal(amplitude=0.3, peak='January'))   # no where=, so it applies to everyone
```

works under every solver. Seasonal forcing under a Gillespie simulator requires the inhomogeneous-Poisson algorithm, and the library uses it without the user having to ask.

### An environmental reservoir

**SimInf** ([approaches/siminf.md](../approaches/siminf.md)) has a continuous variable `v0` plus a `pts_fun` written in C that updates it after each step. **Lingua franca:**

```python
beta  = ss.perday(0.5)     # infection rate when the water is saturated
K     = 1e6                # bacteria in the water at which infection is half its maximum
shed  = ss.perday(10)      # bacteria shed per infectious person

S, I, R, W = ss.states(4)  # W is bacteria in the water: a count, not people

cholera = ss.Disease(
    infection = S >> I @ beta * W.n / (W.n + K),
    recovery  = I >> R @ ss.days(5),
    shedding  = ss.null >> W @ shed * I.n,
    decay     = W >> ss.null @ ss.days(30),
)

sim = ss.Sim(cholera, total_pop=10_000, unit='day')
sim.run(solver='stochastic')
```

`W.n` is the plain count in `W`, and a rate may be any expression of counts and parameters, used as written. Nothing here is a special kind of transmission: infection depends on the amount in the water, and the water is filled by the infectious and decays. The same four transitions describe a vector-borne disease with mosquito states in place of `W`. `W` is an ordinary state; the library sees that no transition connects it to `S`, `I`, or `R` and treats it as a count ([model-structure.md](model-structure.md#transmission-that-is-not-person-to-person)), so in an agent-based sim people are agents and the reservoir stays a number ([agent-based.md](agent-based.md)).

Because the rate uses a count and not a bare state, there is no contact rate and no network involved, which is how the [per-contact `beta`](population-and-mixing.md) is kept out of a place where there is no contact to be per.

### Extinction is the point

The reason to run a stochastic compartmental model rather than an ODE is usually a question the ODE cannot answer: will it die out, when, and how often. Nothing in the review reports this by default.

```python
sim  = ss.Sim(sir, total_pop=1000, unit='day', solver='gillespie')
msim = ss.parallel(sim, n_runs=500)
msim.summarize()
```

```text
extinction: 41% of 500 runs (median time 23 days)
peak prevalence (among non-extinct): 87 [61, 118]
```

Cheap to compute, and it is the main output of the paradigm. Everything else about results (one sim is one trajectory, a MultiSim summarizes many, scenario grids, reported quantities) follows [results-and-observation.md](results-and-observation.md).

### Dwell times and CRN

Non-exponential dwell times under a Gillespie run need sub-states, exactly as under an ODE ([compartmental.md](compartmental.md)): the Erlang expansion is the same expansion, `stages=k` sets it by hand, and a dwell time that is not Erlang is approximated by fitted stages with a one-line note. Common random numbers under a compartmental solver pair at the *run* level rather than the agent level (there are no agents), which is [camdl](../approaches/camdl.md)'s formulation; see [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md).

## Trade-offs

- **Exact Gillespie simulation does not scale**, and the crossover to tau-leaping is population- and rate-dependent. Automatic selection with a printed reason is better than a documentation footnote, and worse than the user understanding it. Both should be possible.
- **Automatic method selection can be actively misleading** if the user is asking an extinction question and gets tau-leaping. The heuristic should bias toward exactness when compartment counts are small — which is exactly when extinction questions arise.
- **A reservoir with very large counts makes exact Gillespie simulation expensive.** Bacteria in water are shed and decay in numbers far larger than the number of infections, and simulating each one as an event is wasted effort. `solver='stochastic'` will pick tau-leaping for such a model. Treating the reservoir deterministically while the infections stay discrete is the hybrid case, which is deferred (see open questions).
- **Count-based rates are literal.** `beta * W.n / (W.n + K)` is used exactly as written, so the library cannot check that it is a sensible force of infection or adjust it for mixing. That is the intent, and it means the user owns the formula.
- **Continuous-time methods make "results at time t" a resampling question.** [MEmilio](../approaches/memilio.md)'s `interpolate_simulation_result()` and SimInf's `tspan` both handle this and most frameworks leave it to the user. Ours should resample onto the requested output grid by default.

## Rejected

- **Stochasticity as a model property** (MEmilio's `sde_*` classes, EpiModel's `icm` versus `dcm`). It is a choice of solver.
- **A single stochastic solver** (SimInf: Gillespie only; Epydemix: chain-binomial only). Each is right for a different population size.
- **Substituting an approximation without saying so.** If tau-leaping was used, the result says so.
- **A separate kind of continuous state for environmental quantities** (SimInf's `v`). Ordinary states holding counts, with count-based rates, cover the same cases with no new concept.
- **A stochastic sim that is many trajectories.** A sim is one run; `ss.parallel(sim, n_runs=500)` is many.
- **The diffusion approximation as the default stochastic solver.** It fails at small counts, which is where stochasticity matters.
- **Requiring the user to choose.** Choose, print, allow override — the same policy as [calibration](calibration.md).

## Open questions

- **Hybrid methods (deferred, not in v1).** Running some transitions as discrete random events and the rest as an ODE in one run: rare importations as chance events, bulk transmission deterministic. [Catalyst.jl](../approaches/notes.md#modelingtoolkitjl--catalystjl) does this per reaction. If added, it would be a `solver=` keyword on the transition. See [paradigm-conversion.md](paradigm-conversion.md).
- **Overdispersion.** camdl's `overdispersed()` for extra-demographic noise is a real need (superspreading) and sits awkwardly between the transitions and the reported quantities. Which one owns it?
- **How does a reservoir stratify alongside the people?** A per-patch environmental reservoir in an age-stratified disease should be split by patch and not by age. The rule proposed in [stratification.md](stratification.md) is that people dimensions skip states that are not people, and a count used in a rate is matched on the dimensions the two sides share and summed over the rest. This needs confirming on a real cholera or malaria example.
