# Calibration

## Recommendation

1. **Nothing extra should need declaring.** If parameters carry bounds and results have data, calibration is `sim.calibrate()`.
2. **Nobody has to think about likelihoods to fit a model.** The default goodness of fit is a normalized root-mean-square error per data series, combined with equal weights. A likelihood is optional, and giving one enables posterior methods.
3. **Choose the method from the model, and print the choice.** Least squares for a deterministic model, ABC-SMC for a stochastic compartmental one, Optuna for an expensive agent-based one. One keyword overrides it.
4. **Fitting settings are separate from the model but normally live in the same script.** They do not change what the model is, so they hash separately; keeping them next to the model stops the two drifting apart.
5. **Do better than Starsim's `CalibComponent`.** Data attaches to results by name; the rest is inferred.
6. **Adopt staged fitting with convergence gates** for the cases that need it.
7. **Report whether it converged**, not just the best-fit parameters.
8. **Common random numbers make calibration cheaper**; say so and use them.
9. **Comparing models is part of v1.** `ss.compare_models([sim_sir, sim_seir], data=df)` fits each and compares how well they predict data they were not fitted to.

## Why

### The declarations already exist

By the time a model reaches calibration, three things have been said elsewhere in this folder:

- **What can move**: `ss.par(0.05, bounds=[0.001, 0.5])` from [parameters-and-distributions.md](parameters-and-distributions.md), which is camdl's `beta : rate in [0.001, 2.0]`.
- **What is compared to what**: results have names, and the data file has columns with the same names ([results-and-observation.md](results-and-observation.md)). This is how Covasim attaches data.
- **How much noise a comparison has**: [common random numbers](stochasticity-and-reproducibility.md).

So the calibration interface is one method call. A framework that requires more is asking the user to say something twice.

### What Starsim asks for today

`ss.Calibration` wraps Optuna. Fitting an SIR's prevalence to three data points takes a parameter dictionary, a function that copies those parameters into the sim, a component, and the calibration object (abridged from the Starsim calibration tutorial):

```python
calib_pars = dict(
    beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
)

def build_sim(sim, calib_pars, **kwargs):
    for k, pars in calib_pars.items():
        if k == 'beta':
            sim.pars.diseases.pars.beta = ss.perday(pars['value'])
    return sim

prevalence = ss.Normal(
    name = 'Disease prevalence',
    conform = 'prevalent',
    expected = pd.DataFrame({'x': [0.13, 0.16, 0.06]},
        index=pd.Index([ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']], name='t')),
    extract_fn = lambda sim: pd.DataFrame({'x': sim.results.sir.prevalence},
        index=pd.Index(sim.results.timevec, name='t')),
    sigma2 = 0.05,
)

calib = ss.Calibration(sim=sim, calib_pars=calib_pars, build_fn=build_sim,
                       components=[prevalence], total_trials=100)
calib.calibrate()
```

Every piece of that restates something the model already knows. The bounds belong to the parameter. `build_sim` exists only because the calibration does not know where `beta` lives. `extract_fn` names a result that already has a name. `conform='prevalent'` says prevalence is a stock rather than a flow, which the result already knows. And the user had to pick a likelihood (`ss.Normal`, with a variance) before they could fit anything.

The same fit in the lingua franca, with `prevalence.csv` holding a date column and a `prevalence` column:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.par(0.15, bounds=[0.01, 0.30])),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
sim = ss.Sim(sir, data='prevalence.csv', start='2020-01-01', stop='2020-03-01', unit='day')
fit = sim.calibrate()
```

### The method depends on the model

| Model | Method | Framework |
|---|---|---|
| Deterministic ODE | Least squares (gradient-based or simplex); NUTS if a likelihood is given | [camdl](../approaches/camdl.md) (analytic gradients by source-to-source differentiation), [summer2](../approaches/summer2.md) (JAX autodiff over the whole model) |
| Stochastic compartmental, no likelihood | ABC-SMC | [Epydemix](../approaches/epydemix.md), [SimInf](../approaches/siminf.md) |
| Stochastic compartmental, likelihood given | Particle filter + particle MCMC | [odin](../approaches/odin.md)/dust/monty, [SimInf](../approaches/siminf.md), [epymorph](../approaches/epymorph.md) |
| Agent-based, expensive | Optuna (point estimate); ABC when affordable | [Starsim](../approaches/starsim.md), [epiworld](../approaches/epiworld.md) (LFMCMC) |
| Any, for sensitivity rather than fit | LHS + PRCC | [MetaCast](../approaches/metacast.md) |

The mapping is regular enough to automate. A user should not have to know that their model's structure makes gradients available, or that 200 Optuna trials on a model that runs in a millisecond is a waste when ABC-SMC would also give them uncertainty.

Optuna is the right choice for expensive agent-based models: it makes no assumptions, it parallelizes, and it handles the case (expensive, stochastic, no gradient) that defeats everything else in the table. It is not the default for every model.

ABC-SMC is the default for stochastic compartmental models. This is likely to be a common case: the model is cheap to run, the user has not written a likelihood, and they want uncertainty. ABC needs only a distance between simulated and observed series, and the default goodness of fit is exactly that.

### Likelihoods are optional

[odin](../approaches/odin.md) puts the likelihood in the model file, in two lines (`cases <- data()`, `cases ~ Poisson(incidence)`), and gets a particle filter from dust and MCMC from monty. [camdl](../approaches/camdl.md) does the same, plus a separate `fit.toml` with named stages. Both are right that when there is a likelihood it belongs with the quantity it describes. Both also require one before anything can be fitted, and most epidemiologists fitting a model to case counts do not start from a likelihood. They start from "make the curve go through the points".

So the lingua franca starts there too. A normalized root-mean-square error is less rigorous than a likelihood: it does not weight points by their expected noise and it does not give a posterior on its own. It is simple, it has no settings, and it is what the user would have written by hand. The likelihood is there when wanted.

Against that: [summer2](../approaches/summer2.md), [epipack](../approaches/epipack.md), [MEmilio](../approaches/memilio.md), [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [`individual`](../approaches/individual.md), and [LASER](../approaches/laser.md) have no built-in comparison to data at all, so every project rebuilds it.

## The proposal

### The whole interface

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=ss.par(0.05, bounds=[0.01, 0.5])),
    recovery  = ss.progression('I -> R', dur=ss.par(10, bounds=[3, 30])),
    cases     = ss.outcome('infection', p=0.5, delay=5),
)

sim = ss.Sim(sir, data='weekly_cases.csv', start='2020-01-01', stop='2020-12-31', unit='day')
fit = sim.calibrate()
```

`weekly_cases.csv` has a date column and a `cases` column. The column name matches the result name, so the data is attached to that result: `sim.plot()` shows the two together, and `calibrate()` fits every parameter that has `bounds` to every result that has data. Because `cases` is a flow, simulated cases are summed over each interval between data points before comparing (weekly data against a daily model works with no extra settings); a stock such as prevalence is read at the data's dates.

`sim.calibrate(pars=['beta'])` restricts what moves (parameters are named by their slot, `infection.beta`, or just `beta` when that is unambiguous). `sim.calibrate(method='abc')` overrides the method.

### Goodness of fit

For each data series, the default mismatch is the root-mean-square difference between model and data, divided by the mean of the data. Dividing by the mean puts a series of thousands of weekly cases and a series of prevalence fractions on the same scale, so they can be added. The total is the weighted sum, with equal weights unless told otherwise:

```python
sim  = ss.Sim(sir, data='cases_and_deaths.csv', unit='day')   # columns: date, cases, deaths
fit  = sim.calibrate()                                        # equal weights
fit2 = sim.calibrate(weights=dict(cases=1, deaths=3))         # trust the deaths more
```

To use a likelihood instead, attach one to the result. The model's value is the mean; the distribution supplies the noise:

```python
cases = ss.outcome('infection', p=0.5, delay=5,
                   likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))   # n: dispersion, fitted
```

That is camdl's `weekly_cases ~ neg_binomial(mean = rho * projected, r = k)`. With a likelihood on every fitted result, posterior methods (NUTS, particle MCMC) become available and are chosen automatically.

### Choosing the method

A deterministic model:

```text
>>> fit = sim.calibrate()
model: deterministic (ode), 2 free parameters, 1 data series (cases, 52 points)
fit: normalized RMSE
method: least squares, Nelder-Mead  (override with method=)
...
converged: yes (63 evaluations)
infection.beta   0.052
recovery.dur     9.3 days
note: point estimates only. For uncertainty, add a likelihood or use method='abc'.
```

The same model run with `method='stochastic'`:

```text
model: stochastic compartmental (binomial), 2 free parameters, 1 data series
fit: normalized RMSE
method: ABC-SMC, 5 generations, 500 particles  (override with method=)
...
infection.beta   0.052  [0.047, 0.058]
recovery.dur     9.3 days  [8.0, 10.9]
```

And as an agent-based model:

```text
model: agent-based, 2 free parameters, 1 data series
fit: normalized RMSE
method: Optuna TPE, 200 trials, 4 replicates each  (override with method=)
note: common random numbers on: paired evaluation needs fewer replicates
note: Optuna gives a best estimate, not a posterior. No intervals are reported.
```

The method is chosen from things the library already knows about the model: whether it is deterministic, whether a likelihood was given, and how long one run takes. The note about common random numbers matters in practice: with paired evaluation the objective is much less noisy, which is a benefit of [the CRN machinery](stochasticity-and-reproducibility.md) that is rarely stated.

For deterministic models with more than a few parameters, gradients come from JAX, which is also what makes the ODE runs fast ([summer2](../approaches/summer2.md) does this). Small models use plain SciPy.

### Fitting settings

Method, number of trials, chains, and gates are arguments to `calibrate()`, in the same script as the model:

```python
fit = sim.calibrate(method='optuna', trials=500, reps=4)
```

camdl separates `.camdl` from `fit.toml` "to keep structural identity separate from fitting configuration"; [flepiMoP](../approaches/flepimop.md) puts everything in one document. The lingua franca keeps the conceptual separation and not the second file. Bounds, data, and any likelihood are part of the model, because they are statements about the system and how it is observed. The fitting procedure is not: changing the number of Optuna trials does not change the model, so it has its own entry in the [run hash](stochasticity-and-reproducibility.md). Keeping both in one script means the settings cannot drift from the model they were tuned for.

### Staged fitting

camdl's `fit.toml` has named stages with convergence gates. The case that needs this is real (find the mode cheaply, then characterize the posterior around it) and it should not require a script:

```python
fit = sim.calibrate(method=[
    dict(method='optuna', trials=500),
    dict(method='nuts', draws=2000),
])
```

`method=` takes a list, and each stage starts from where the previous one ended. A stage that does not converge stops the sequence, says so, and returns the last good fit, rather than feeding a bad starting point into an expensive sampler.

### Report the fit

```python
fit.summary()      # estimates, intervals where the method gives them, convergence diagnostics
fit.plot()         # trajectories against data, one panel per data series
fit.pars           # the fitted values
fit.sim            # the calibrated sim, ready to run or to build scenarios from
```

camdl's standard is the target: "Convergence is a checked claim, not an assumption." Report R-hat and effective sample size for MCMC; gradient norm and iterations for optimization; acceptance rate and effective sample size for ABC and particle filters; and whether the best value was near a bound, because that usually means the bound was wrong rather than the parameter.

### Comparing models

"Does this model need an exposed compartment?" is a question about structure, not about parameter values. Because each model is a described object and the data attaches by name, the comparison can be done the same way every time:

```python
comparison = ss.compare_models([sim_sir, sim_seir], data='weekly_cases.csv')
comparison.summary()
```

```text
          free pars   fit (in-sample)   prediction (held-out)
sir               2             0.118                   0.171
seir              3             0.094                   0.102
```

Each model is fitted to the first part of each data series and scored on the rest, so an extra parameter only helps if it improves prediction. This is the idea behind camdl's prequential scoring. The held-out fraction and the score are settings with defaults; the models can differ in any way as long as each produces the results named in the data.

### Sensitivity, which is not calibration

[MetaCast](../approaches/metacast.md) ships LHS + PRCC parallelized via dask, and it is the only framework in the review that treats sensitivity analysis as built-in. It answers a different question (which parameters matter), and it is often what the user wanted:

```python
sim.sensitivity()   # LHS over declared bounds, PRCC against the results
```

It needs nothing new, because the bounds and the results already exist.

## Trade-offs

- **Automatic method selection can be wrong.** It is printed, and it is one keyword to override. The alternative, where the user chooses from five methods on their first day, is worse.
- **Normalized RMSE is not a statistical model of the data.** It treats every point in a series as equally noisy, and dividing by the mean behaves badly for a series that is mostly zeros. It is the default because it is simple and needs no decisions; the summary says which measure was used, and a likelihood is one keyword away.
- **Weights are arbitrary.** Equal weights across series is a convention, not a truth. A joint likelihood is the principled alternative and needs the correlation between series, which is rarely known. Weights are the v1 answer.
- **Optuna is not an inference method.** It gives a point estimate and a search history, not a posterior, and the summary must not present its trial spread as an interval.
- **ABC needs many runs.** Fine for a compartmental model that runs in milliseconds; for an agent-based model it is used only when the run time allows, and the printout says how many runs it will take.
- **Gradients are an implementation project.** camdl gets them from symbolic differentiation; summer2 gets them from JAX. Wrapping JAX for the compartmental methods is the cheaper route and worth taking, since it also makes ODE runs fast.

## Rejected

- **A separate calibration DSL.** Everything needed is already declared; a second vocabulary would be a second place for it to drift.
- **`CalibComponent`-style declaration at calibration time** (Starsim today). It restates the result name, the data alignment, and the bounds, and it makes a likelihood mandatory.
- **A required likelihood** (odin, camdl). Right when the user has one; a barrier when they just want to fit a curve.
- **One method for every model.** Optuna on a cheap ODE wastes runs and gives no gradient-based convergence check; NUTS on an agent-based model is not possible. The method follows from the model.
- **Requiring the user to choose an inference method.** They should be able to; they should not have to.
- **Bundling the inference engine.** [odin](../approaches/odin.md)'s odin/dust/monty split is the precedent: "an engine can be replaced, and the inference layer is reusable by models that were never written in the DSL". Wrap Optuna, JAX, an ABC sampler, and a particle filter; own none of them.

## Open questions

- **The normalization.** Dividing RMSE by the mean of the data is one choice; the range or the standard deviation are others, and count data with many zeros may want a different default.
- **Likelihoods on results that are not outcomes.** `likelihood=` is shown on `ss.outcome`. A ratio such as `prevalence = I / N` needs somewhere to put one too.
- **Joint likelihoods with correlation between series** are out of v1.
- **`beta` and the contact rate.** When a network or mixing pool supplies the contacts ([population-and-mixing.md](population-and-mixing.md)), `beta` is per contact, and incidence data identifies only the product of `beta` and the contact rate. Giving bounds to one and fixing the other is the simple answer; fitting the product by name may be worth a spelling.
