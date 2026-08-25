# Stochastic compartmental

Gillespie/CTMC, tau-leaping, chain-binomial, and diffusion (SDE). One model, four ways of being random.

## Recommendation

1. **Stochasticity is a run-time choice with several named methods**, not a property of the model.
2. **Prefer continuous time where the model allows it**, because it eliminates rate-to-probability conversion entirely.
3. **Name the approximation.** Gillespie, tau-leaping, chain-binomial, and diffusion are different models with different validity conditions, and the result should say which one produced it.
4. **Default to a distribution over trajectories**, not a trajectory.
5. **Discrete and continuous state coexist.** Environmental reservoirs are naturally continuous and coupled to a discrete disease process.
6. **Extinction is the reason this paradigm exists**; make it visible.

## Why

### One model, several stochastic backends, already demonstrated

[camdl](../approaches/camdl.md) runs the same model text on Gillespie SSA (exact continuous-time), chain-binomial (Euler-multinomial discrete-time), and ODE. [epipack](../approaches/epipack.md) runs the same process tuples deterministically and stochastically, including "a correct inhomogeneous-Poisson Gillespie implementation" for time-varying rates — "a real capability that most frameworks skip or approximate". [odin](../approaches/odin.md) shares `initial()` between `deriv()` and `update()`.

[PyRoss](../approaches/notes.md#pyross) is the sharpest evidence for the *ladder* rather than the binary: it implements Gillespie, tau-leaping, and Gaussian/van Kampen approximations of the same model side by side. [MEmilio](../approaches/memilio.md) has a whole `sde_*` family alongside its `ode_*` family, separately implemented.

### Continuous time removes an error class

[SimInf](../approaches/siminf.md) is continuous-time Gillespie throughout, and its review makes the point directly: "Being continuous-time is a real advantage for correctness — the exponential-conversion errors that afflict discrete-time frameworks simply do not arise — and a real limitation for scale, since Gillespie cost grows with event count." [EoN](../approaches/eon.md) is continuous-time for the same reason.

The implication drawn in the SimInf review: "The lingua franca should treat discrete-time versus continuous-time as a backend choice with a declared cost, not as a property of the model."

### Discrete and continuous state together

SimInf's `u`/`v` split — integer compartment counts governed by the Markov chain, alongside continuous per-node variables updated by a post-timestep function — is used by its `SISe` family for a SIS disease coupled to an environmental compartment. Its review: "Environmental reservoirs, accumulated exposure, and vector abundance are all naturally continuous and coupled to a discrete disease process. Frameworks without this force everything into compartments."

## The proposal

### The method ladder

```python
sim = ss.Sim(sir, mixing=ss.Homogeneous(n_contacts=10), n=1000)

sim.run(method='ode')       # deterministic
sim.run(method='ctmc')      # exact continuous-time (Gillespie)
sim.run(method='tau')       # tau-leaping
sim.run(method='binomial')  # chain-binomial, discrete time
sim.run(method='sde')       # diffusion approximation
sim.run()                   # agent-based (Starsim's default)
```

Same `sir`. Same route. Six answers to the same model, each valid under different conditions.

And the default should choose:

```text
>>> sim.run(method='stochastic')
n = 1000, 3 states, 2 transitions
method: ctmc (exact) — event count is tractable at this population size
```

```text
>>> ss.Sim(sir, n=10_000_000).run(method='stochastic')
method: tau  (exact CTMC would require ~10^9 events)
  ! tau-leaping is an approximation; τ chosen adaptively, leap condition ε=0.03
```

Naming the approximation in the output is the point. A trajectory that came from tau-leaping is not a trajectory from the CTMC, and every framework that silently substitutes one for the other is producing a result whose provenance is lost.

### Validity conditions, checked

Each method has conditions, and the model is data, so they can be checked rather than documented:

| Method | Valid when | Check |
|---|---|---|
| CTMC / Gillespie | Always (it is exact) | Event count feasible |
| Tau-leaping | Propensities roughly constant over τ | Leap condition, adaptively |
| Chain-binomial | `rate × dt` small | Warn above ~0.1; `check_dt()` |
| SDE / diffusion | Counts large in every compartment | Warn when any compartment falls below ~20 |
| ODE | Counts large, no extinction question | Warn when the epidemic could go extinct |

The SDE warning matters because the diffusion approximation is "valid only at moderate-to-large populations" ([MEmilio](../approaches/memilio.md)) and fails exactly where stochasticity was the reason to use it. This is [camdl](../approaches/camdl.md)'s capability-check discipline turned toward numerical validity.

### Time-varying rates in every method

[epipack](../approaches/epipack.md) sets the standard: a rate may be a number, a callable of `(t, y)`, or a symbolic expression, and the framework does the right thing in ODE, Gillespie, and inhomogeneous-Poisson modes. Its review: "a rate is an expression in time and state, and each backend is responsible for handling that correctly or refusing."

```python
ss.Effect(sir.beta, multiply=ss.seasonal(amplitude=0.3, peak='January'))   # unconditional: the condition is optional
```

works under every method, or errors by name. Seasonal forcing under a Gillespie simulator requires the inhomogeneous-Poisson algorithm, and getting it right for free is a real capability.

### Continuous state alongside discrete

**SimInf** ([approaches/siminf.md](../approaches/siminf.md)) has `v0` plus a `pts_fun` in C. **Lingua franca:**

```python
cholera = ss.Disease(
    infection  = ss.transmission('S -> I', beta=0.02, source=ss.env.W),
    recovery   = ss.progression('I -> R', dur=ss.days(5)),
    W          = ss.Continuous(shed=ss.perday(10), decay=ss.perday(0.3)),
)
```

`W` is a continuous per-patch quantity, not a compartment: it is shed into by the infectious and decays. It is the one clean answer in the review to environmental transmission, and it is also the most likely place the [β-per-contact factoring](population-and-mixing.md) needs an extension, since there is no contact to be per.

### Extinction is the point

The reason to run a stochastic compartmental model rather than an ODE is usually a question the ODE cannot answer: will it die out, when, and how often. Nothing in the review reports this by default.

```python
sim = ss.Sim(sir, n=1000, n_reps=500).run(method='ctmc')
```

```text
extinction: 41% of 500 replicates (median time 23 days)
peak prevalence (among non-extinct): 87 [61, 118]
```

Cheap to compute, and it is the actual output of the paradigm. Everything else about results — quantiles by default, scenario grids, the observation model — follows [results-and-observation.md](results-and-observation.md), which already recommends distributions over trajectories on [Epydemix](../approaches/epydemix.md)'s evidence.

### Dwell times and CRN

Non-exponential dwell times under a CTMC need sub-states, exactly as under an ODE ([compartmental.md](compartmental.md)) — the Erlang expansion is the same expansion. Common random numbers under a CTMC pair at the *run* level rather than the agent level (there are no agents), which is [camdl](../approaches/camdl.md)'s formulation; see [stochasticity-and-reproducibility.md](stochasticity-and-reproducibility.md).

## Trade-offs

- **Exact CTMC does not scale**, and the crossover to tau-leaping is population- and rate-dependent. Automatic selection with a printed reason is better than a documentation footnote, and worse than the user understanding it. Both should be possible.
- **Automatic method selection can be actively misleading** if the user is asking an extinction question and gets tau-leaping. The heuristic should bias toward exactness when compartment counts are small — which is exactly when extinction questions arise.
- **The `ss.Continuous` state is a new concept** with a maintenance cost, justified only if environmental and vector-borne transmission are in scope. They are, and SimInf's `u`/`v` split is the precedent.
- **Continuous-time backends make "results at time t" a resampling question.** [MEmilio](../approaches/memilio.md)'s `interpolate_simulation_result()` and SimInf's `tspan` both handle this and most frameworks leave it to the user. Ours should resample onto the requested output grid by default.

## Rejected

- **Stochasticity as a model property** (MEmilio's `sde_*` classes, EpiModel's `icm` versus `dcm`). It is a run method.
- **A single stochastic method** (SimInf: Gillespie only; Epydemix: chain-binomial only). Each is right for a different population size.
- **Silent substitution of an approximation.** If tau-leaping was used, the result says so.
- **The diffusion approximation as the default stochastic method.** It fails at small counts, which is where stochasticity matters.
- **Requiring the user to choose.** Choose, print, allow override — the same policy as [calibration](calibration.md).

## Open questions

- **Hybrid methods.** [Catalyst.jl](../approaches/notes.md#modelingtoolkitjl--catalystjl)'s per-reaction `PhysicalScale` — rare importation as a jump, bulk transmission as an ODE, in one system — is the right target and "is what a hybrid epidemiological model actually needs". Does it belong on the transition (`ss.transmission(..., scale='jump')`) or is that too much vocabulary for the gain? See [paradigm-conversion.md](paradigm-conversion.md).
- **Overdispersion.** camdl's `overdispersed()` for extra-demographic noise is a real need (superspreading) and sits awkwardly between the process model and the observation model. Which one owns it?
- **How does `ss.Continuous` stratify?** A per-patch environmental reservoir under an age stratification should not be split by age, which means the dimension has to know which quantities it applies to.
