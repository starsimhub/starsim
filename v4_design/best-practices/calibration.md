# Calibration

## Recommendation

1. **Nothing extra should need declaring.** If parameters carry bounds and priors, and outcomes carry data and a likelihood, calibration is `sim.calibrate()`.
2. **Keep Starsim's Optuna-based `ss.Calibration` as the default**, because it works on any model including agent-based ones. *[CK: No, I think the method should be selected based on the model -- sure use Optuna if it's agent based, but switch to another method if it's not.]*
3. **Offer the method ladder, and pick for the user by default.** Which method is right follows from the model, not from the user's preferences. *[CK: yes, thought this seems to be the opposite of your point above.]*
4. **Fitting configuration is separate from the model.** camdl's split is right: fitting is not part of a model's structural identity. *[CK: yes, though in practice they often live in the same script, which avoids drift.]*
5. **Adopt staged fitting with convergence gates** for the cases that need it.
6. **Report whether it converged**, not just the best-fit parameters.
7. **Common random numbers make calibration cheaper**; say so and use them.

## Why

### The declarations already exist

By the time a model reaches calibration, three things have been said elsewhere in this folder:

- **What can move**: `ss.Par(..., bounds=[0.001, 0.5], prior=...)` from [parameters-and-distributions.md](parameters-and-distributions.md) — camdl's `beta : rate in [0.001, 2.0] ~ prior`.
- **What is compared to what**: `ss.Outcome(..., data=..., likelihood=...)` from [results-and-observation.md](results-and-observation.md) — camdl's `observations` block and odin's `cases ~ Poisson(incidence)`.
- **How much noise a comparison has**: [common random numbers](stochasticity-and-reproducibility.md).

So the entire calibration interface is one method call. Any framework that requires more is asking the user to say something twice.

### Where the likelihood lives

[odin](../approaches/odin.md) puts it in the model file, in two lines, and gets a particle filter from dust and MCMC from monty. Its review: "The likelihood is in the model. Two lines, and the particle-filter machinery follows." Its architecture is also the cleanest split in the review — **odin is a language, dust is an engine, monty is an inference layer, and they meet at defined interfaces** — so monty works against any dust model, DSL-generated or not.

[camdl](../approaches/camdl.md) does the same, plus a separate `fit.toml` with named stages, so that "IF2 to find the mode, then PGAS to characterise the posterior around it" is two declared stages rather than a script.

Against that: [summer2](../approaches/summer2.md), [epipack](../approaches/epipack.md), [MEmilio](../approaches/memilio.md), [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [`individual`](../approaches/individual.md), and [LASER](../approaches/laser.md) have no observation model, so comparison to data happens outside the framework and every project rebuilds it.

### The method depends on the model, and the review shows the mapping

| Model | Method | Framework |
|---|---|---|
| Deterministic ODE, tractable likelihood | Gradient-based optimization, NUTS | [camdl](../approaches/camdl.md) (analytic gradients by source-to-source differentiation), [summer2](../approaches/summer2.md) (JAX autodiff over the whole model) |
| Discrete-time stochastic, latent states | Particle filter + PMCMC | [odin](../approaches/odin.md)/dust/monty, [SimInf](../approaches/siminf.md), [epymorph](../approaches/epymorph.md) |
| Stochastic, awkward likelihood, cheap to simulate | ABC / ABC-SMC | [Epydemix](../approaches/epydemix.md), [SimInf](../approaches/siminf.md) |
| Agent-based, expensive, no likelihood | Optuna / surrogate / LFMCMC | [Starsim](../approaches/starsim.md), [epiworld](../approaches/epiworld.md) |
| Any, for sensitivity rather than fit | LHS + PRCC | [MetaCast](../approaches/metacast.md) |

The mapping is deterministic enough to automate. A user should not have to know that their model's structure makes NUTS available and ABC wasteful.

### What Starsim has

`ss.Calibration` wraps Optuna, with `CalibComponent` objects defining likelihood terms against data. Optuna is the right default for agent-based models: it makes no assumptions, it parallelizes, and it handles the case — expensive, stochastic, no gradient — that defeats everything else in the table. `CalibComponent` is also, in effect, an observation model declared at calibration time rather than in the model, which is the piece that moves. *[CK: I don't particularly care for CalibComponent; it's so complicated and verbose. Let's try to do better with the lingua franca.]*

## The proposal

### The whole interface

```python
sir = ss.Disease(
    beta      = ss.Par(0.05, bounds=[0.01, 0.5]), # [CK: let's use ss.par() for this]
    dur_inf   = ss.Par(ss.days(10), bounds=[3, 30]),
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=dur_inf),
)

sim = ss.Sim(sir, start='2020-01-01', stop='2020-12-31')
sim.observe(cases = ss.Outcome(sir.infection, p=0.5, delay=ss.days(5),
                               data='data/weekly_cases.csv', every=ss.weeks(1),
                               likelihood=ss.neg_binomial(k=ss.Par(bounds=[0.1, 100]))))
# [CK: it's not clear to me that sim.observe should be needed as separate from sim.results]
fit = sim.calibrate()
```

`calibrate()` reads the bounds from the parameters and the likelihood from the outcomes. Nothing is repeated. `sim.calibrate(pars=['beta'])` restricts what moves; `sim.calibrate(method='abc')` overrides the choice.

### Choosing the method

```text
>>> sim.calibrate()
model: deterministic ODE, 2 free parameters, analytic likelihood available
method: L-BFGS-B with analytic gradients  (override with method=)
...
converged: yes (grad norm 3e-7, 41 evaluations)
beta     0.052  [0.048, 0.057]
dur_inf  9.3 d  [8.1, 10.6]
```

versus the same model run as an ABM:

```text
>>> sim.calibrate()
model: agent-based, 2 free parameters, no tractable likelihood
method: Optuna TPE, 200 trials, 4 replicates each  (override with method=)
note: common random numbers on — paired evaluation, ~4× fewer replicates than independent
```

Two things are happening. The method is chosen from properties of the model that the model already declared, which is only possible because the model is data. And the CRN note is not decoration: with paired evaluation the objective is much less noisy, which is the practical payoff of [the CRN machinery](stochasticity-and-reproducibility.md) and is rarely stated as a calibration benefit.

### Staged fitting

camdl's `fit.toml` has named stages with convergence gates. The cases that need this are real — find the mode cheaply, then characterize the posterior around it — and it should not require a script:

```python
fit = sim.calibrate(stages=[
    ss.Stage('optuna', trials=500),
    ss.Stage('mcmc', draws=2000, init='previous'),
])
```

A stage that fails its convergence gate stops the sequence and says so, rather than feeding a bad mode into an expensive sampler.

### Fitting configuration is not part of the model

camdl separates `.camdl` from `fit.toml` deliberately, "to keep structural identity separate from fitting configuration". [flepiMoP](../approaches/flepimop.md) puts everything in one document and its review notes both are defensible.

The resolution follows from [run identity](stochasticity-and-reproducibility.md): the *likelihood* is part of the model (it is a statement about how data relate to the process, and it belongs with the outcome that produces the quantity), while the *fitting procedure* — method, trials, chains, gates — is configuration and hashes separately. Changing the number of Optuna trials does not change the model.

### Report the fit

```python
fit.summary()      # estimates, intervals, convergence diagnostics
fit.plot()         # trajectories against data, per outcome stream
fit.apply(sim)     # a sim with the calibrated values [CK: it's not clear to me what is being "applied" -- why not fit.sim or similar]
```

camdl's discipline is the target: "Convergence is a checked claim, not an assumption." Report R̂ and ESS for MCMC; report gradient norm and iterations for optimization; report acceptance and ESS for particle filters; report whether the best trial was near a bound, because that usually means the bound was wrong rather than the parameter.

### Sensitivity, which is not calibration

[MetaCast](../approaches/metacast.md) ships LHS + PRCC parallelized via dask, and it is the only framework in the review that treats sensitivity analysis as built-in. It answers a different question — which parameters matter — and it is often what the user actually wanted:

```python
sim.sensitivity()   # LHS over declared bounds, PRCC against declared results
```

Free, because the bounds and the results are already declared.

## Trade-offs

- **Automatic method selection can be wrong.** Mitigation: it is printed, and it is one keyword to override. The alternative — the user chooses from five methods on their first day — is worse.
- **Declaring the likelihood in the model file commits to a form.** Negative binomial with an overdispersion parameter covers most surveillance count data; anything else takes a custom likelihood function, which should be allowed and priced. *[CK: Hmm. Seems like something simple like RMS should be the default for simplicity, even if not as rigorous. But I could be convinced.]*
- **Optuna is not an inference method.** It gives a point estimate and a search history, not a posterior, and the summary must not present its trial spread as an interval. This is a real hazard in current practice.
- **Analytic gradients are a large implementation project.** camdl gets them from source-to-source symbolic differentiation; summer2 gets them from JAX. Wrapping JAX for the compartmental backend is the cheap route and worth taking, since it also makes the ODE backend fast. *[CK: Agree]*

## Rejected

- **A separate calibration DSL.** Everything needed is already declared; a second vocabulary would be a second place for it to drift.
- **`CalibComponent`-style likelihood declaration at calibration time** (Starsim today). It duplicates the observation model. Declare it once, with the outcome. *[CK: fine, as long as the user isn't forced into thinking about likelihoods when they just want to build a model.]*
- **ABC as the default** (Epydemix). Right for cheap stochastic models with awkward likelihoods, wasteful when a likelihood exists. *[CK: it's not clear to me that that's not the regime we'll be in a lot of the time. I'd revisit this, though you may be right]*
- **Requiring the user to choose an inference method.** They should be able to; they should not have to.
- **Bundling the inference engine.** [odin](../approaches/odin.md)'s odin/dust/monty split is the architectural precedent — "an engine can be replaced, and the inference layer is reusable by models that were never written in the DSL". Wrap Optuna, JAX, and a particle filter; own none of them.

## Open questions

- **Multi-objective and multi-stream weighting.** Fitting cases *and* deaths *and* seroprevalence requires either a joint likelihood (correct, and requires their correlation) or weights (practical, and arbitrary). camdl's multiple observation streams assume independence. This needs a stated default. *[CK: I think weights is fine, but could be convinced.]*
- **Calibrating structure, not just parameters.** "Does this model need an exposed compartment?" is a model-comparison question, and once models are data it is mechanizable (prequential scoring, which camdl has). Out of scope for v1, worth not foreclosing. *[CK: agree, it would be a superpower to be able to rigorously compare two models: `comparison = ss.calib_compare([sim_sir, sim_seir], against=data)`. If we could include this in v1 -- incredible.]
- **The `β`/`c` identifiability question** raised in [population-and-mixing.md](population-and-mixing.md): the factoring is right for portability and means the two are not separately identifiable from incidence. Fitting the product needs a clean spelling.
