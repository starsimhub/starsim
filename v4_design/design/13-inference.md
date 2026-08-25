# Inference: calibration and sensitivity

Nothing extra should need declaring. If parameters carry bounds and outcomes carry data and a likelihood, calibration is one method call.

Justified by [best-practices/calibration.md](../best-practices/calibration.md).

## The whole interface

```python
sir = ss.Disease(
    beta      = ss.Par(0.05, bounds=[0.01, 0.5]),
    dur_inf   = ss.Par(ss.days(10), bounds=[3, 30]),
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=dur_inf),
)

sim = ss.Sim(sir, start='2020-01-01', stop='2020-12-31')
sir.observe(cases=ss.Outcome(sir.infection, p=0.5, delay=ss.days(5),
                             data='data/weekly_cases.csv', every=ss.weeks(1),
                             likelihood=ss.neg_binomial(k=ss.Par(bounds=[0.1, 100]))))

fit = sim.calibrate()
```

`calibrate()` reads what can move from the parameters' `bounds` and `prior`, and what is compared to what from the outcomes' `data` and `likelihood`. Nothing is said twice. Any interface that requires more is asking the user to repeat a declaration they have already made.

```python
sim.calibrate(
    pars    = None,       # restrict which parameters move; default: all with bounds
    method  = None,       # override the automatic choice
    stages  = None,       # a staged fitting sequence
    n_reps  = None,       # replicates per evaluation for stochastic models
    seed    = None,
    **method_kwargs,
)
```

## Choosing the method

The right method follows from the model, not from the user's preferences, and the mapping is deterministic enough to automate.

| Model properties | Method | Why |
|---|---|---|
| Deterministic, tractable likelihood, gradients available | L-BFGS-B, then NUTS | gradients come free from the compiled backend |
| Discrete-time stochastic, latent states | particle filter + PMCMC | the standard for partially observed Markov processes |
| Stochastic, awkward likelihood, cheap to simulate | ABC / ABC-SMC | no likelihood needed |
| Agent-based, expensive, no likelihood | Optuna TPE | makes no assumptions, parallelizes, handles noise |
| Any, asking which parameters matter | LHS + PRCC | this is sensitivity, not calibration |

```text
>>> sim.calibrate()
model: deterministic ODE, 2 free parameters, analytic likelihood available
method: L-BFGS-B with analytic gradients  (override with method=)
converged: yes (grad norm 3e-7, 41 evaluations)
beta     0.052  [0.048, 0.057]
dur_inf  9.3 d  [8.1, 10.6]
```

```text
>>> sim.calibrate()      # the same model, run as an ABM
model: agent-based, 2 free parameters, no tractable likelihood
method: Optuna TPE, 200 trials, 4 replicates each  (override with method=)
note: common random numbers on — paired evaluation, ~4x fewer replicates than independent
```

Two things are happening in that output. The method is chosen from properties the model already declared, which is only possible because the model is data. And the CRN note is load-bearing rather than decorative: with paired evaluation the objective is far less noisy, which is the practical payoff of the CRN machinery and is rarely stated as a calibration benefit.

The choice MUST be printed with its reason and MUST be overridable. Requiring a user to choose among five inference methods on their first day is worse than choosing for them and saying so.

**Optuna is not an inference method.** It returns a point estimate and a search history, not a posterior, and the summary MUST NOT present the spread of trial values as an interval. This is a real hazard in current practice, and the requirement is normative: a fit whose method has no posterior reports `interval: not available (method='optuna' gives a point estimate)`.

## Staged fitting

Finding the mode cheaply and then characterizing the posterior around it is a real workflow and should not require a script:

```python
fit = sim.calibrate(stages=[
    ss.Stage('optuna', trials=500),
    ss.Stage('mcmc', draws=2000, init='previous'),
])
```

```python
ss.Stage(method, *, gate=None, init=None, **kwargs)
```

A stage that fails its convergence `gate` stops the sequence and reports why, rather than feeding a bad mode into an expensive sampler. `init='previous'` carries the preceding stage's result forward.

## What is model and what is configuration

- The **likelihood is part of the model.** It is a statement about how data relate to the process, it lives with the outcome that produces the quantity, and it hashes into `model=`.
- The **fitting procedure is configuration.** Method, trials, chains, gates, and seeds hash into `fit=` ([11-random.md](11-random.md)).

Changing the number of optimizer trials does not change the model, and the hash levels say so.

## Reporting a fit

```python
fit.summary()      # estimates, intervals, convergence diagnostics
fit.plot()         # trajectories against data, per outcome stream
fit.apply(sim)     # a sim with the calibrated values
fit.to_df()
```

Convergence is a checked claim, not an assumption. `summary()` MUST report:

| Method family | Required diagnostics |
|---|---|
| Optimization | gradient norm, iterations, whether the optimum is at a bound |
| MCMC | R-hat, effective sample size, divergences |
| Particle filter | acceptance rate, effective sample size, particle degeneracy |
| Optuna / TPE | best value, trial count, and an explicit statement that this is not a posterior |
| ABC | acceptance rate, final tolerance, effective sample size |

A best-fit value at or near a declared bound MUST be flagged (`W1301 AtBound`), because in practice that usually means the bound was wrong rather than that the parameter is extreme.

## Multiple streams

Fitting cases *and* deaths *and* seroprevalence is ordinary: three outcomes, three likelihoods. The default combination is a **sum of log-likelihoods, assuming independence between streams**, and this default MUST be printed, because it is a modeling assumption and it is usually wrong in detail:

```text
likelihood: cases + deaths + sero  (independent streams assumed)
```

Weights are available (`ss.Outcome(..., weight=)`) and are honest about what they are: a practical device, not a probability model. A joint likelihood with a declared correlation is out of scope for v1 and is recorded in [19-open-questions.md](19-open-questions.md).

## Fitting a product of parameters

Because β is per contact and the contact rate belongs to the route ([06-routes.md](06-routes.md)), only their product is identifiable from incidence data. The language MUST make fitting the product expressible:

```python
sim.calibrate(pars=[sir.beta * mixing.n_contacts])
```

A calibration whose free parameters are not jointly identifiable MUST emit `W1302 Unidentifiable`, naming the parameters and the combination that is identifiable. This check is available because the model is data: the force-of-infection expression is inspectable, and β and c enter it only as a product.

## Sensitivity, which is not calibration

```python
sim.sensitivity()                       # LHS over declared bounds, PRCC against declared results
sim.sensitivity(results=['peak_n_I'])
```

Free, because the bounds and the results are already declared. It answers a different question — *which parameters matter* — and it is often what the user actually wanted. It is the one analysis in this document that requires no data at all.

## Rejected

- **A separate calibration DSL.** Everything needed is already declared; a second vocabulary is a second place for it to drift.
- **Likelihood declaration at calibration time.** It duplicates the observation model. Declare it once, with the outcome.
- **ABC as the default.** Right for cheap stochastic models with awkward likelihoods, wasteful when a likelihood exists.
- **Requiring the user to choose a method.** They should be able to; they should not have to.
- **Bundling the inference engine.** A language, an engine, and an inference layer meeting at defined interfaces is the right architecture: the engine can be replaced and the inference layer is reusable by models that were never written in this language. Wrap the optimizers, the autodiff library, and a particle filter; own none of them.
- **Structural calibration in v1.** "Does this model need an exposed compartment?" is mechanizable once models are data, and it is not v1 vocabulary. Do not foreclose it.
