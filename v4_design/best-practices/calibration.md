# Calibration

## Recommendation

1. **Nothing extra should need declaring.** If parameters carry bounds and results have data, calibration is `sim.calibrate()`.
2. **Nobody has to think about likelihoods to fit a model.** The default goodness of fit is the normalized absolute error (the mean absolute difference between model and data, divided by the mean of the data) per data series, combined with equal weights. It assumes no noise model. Other measures are available by name, a likelihood is optional, and giving one enables posterior methods.
3. **Choose the fitting method from the model, and print the choice.** For a deterministic model, a local optimizer with exact gradients from automatic differentiation (L-BFGS-B; Nelder-Mead when the model has no gradient), or NUTS for a posterior when every data series has a likelihood. For a stochastic compartmental model whose data series all have a likelihood, particle MCMC. For any other stochastic model, agent-based ones included, Optuna, which gives a best fit and not a posterior. ABC-SMC gives a posterior without a likelihood when asked for (`method='abc-smc'`). One keyword overrides the choice.
4. **Fitting settings are separate from the model but normally live in the same script.** They do not change what the model is, so they are not part of its hash (`sim.hash()` covers the model, the run settings, and the data); keeping them next to the model stops the two drifting apart.
5. **Do better than Starsim's `CalibComponent`.** Data attaches to results by name; the rest is inferred.
6. **Adopt staged fitting with convergence gates** for the cases that need it: `method=` takes a list of fitting methods, run in order, each starting from the previous one's best fit. A stage that does not converge stops the sequence, so a bad starting point is never handed to an expensive sampler.
7. **Report whether it converged**, not just the best-fit parameters.
8. **Use common random numbers in stochastic fits.** Each point is scored by the mean of a few runs with the same seeds (4 by default), which makes nearby points much less noisy to compare. The pairing is strong for small changes in the parameters and weakens once trajectories diverge, so it reduces the noise of a fit without removing it.
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

The same fit in Starsim 4 is the bounds on `beta`, a data file whose column is named after the result, and `sim.calibrate()` ([13-inference.md](../design/13-inference.md) shows the two side by side).

### The method depends on the model

| Model | Method | Framework |
|---|---|---|
| Deterministic ODE | A local optimizer (gradient-based or simplex); NUTS if a likelihood is given | [camdl](../approaches/camdl.md) (analytic gradients by source-to-source differentiation), [summer2](../approaches/summer2.md) (JAX autodiff over the whole model) |
| Stochastic compartmental, no likelihood | ABC-SMC | [Epydemix](../approaches/epydemix.md), [SimInf](../approaches/siminf.md) |
| Stochastic compartmental, likelihood given | Particle filter + particle MCMC | [odin](../approaches/odin.md)/dust/monty, [SimInf](../approaches/siminf.md), [epymorph](../approaches/epymorph.md) |
| Agent-based, expensive | Optuna (point estimate); ABC when affordable | [Starsim](../approaches/starsim.md), [epiworld](../approaches/epiworld.md) (LFMCMC) |
| Any, for sensitivity rather than fit | LHS + PRCC | [MetaCast](../approaches/metacast.md) |

The mapping is regular enough to automate. A user should not have to know that their model's structure makes gradients available. The choice can be made from what the library already knows (whether the model is deterministic, whether it has gradients, whether every data series has a likelihood), not from timing runs, so it is the same every time and the reason can be printed.

Optuna is the right choice for expensive agent-based models: it makes no assumptions, it parallelizes, and it handles the case (expensive, stochastic, no gradient) that defeats everything else in the table. In practice it also works better than ABC for agent-based models, so it is the default for every stochastic model without a likelihood for each series, stochastic compartmental ones included. ABC-SMC is there for a user who wants a posterior without writing a likelihood: it needs only a distance between simulated and observed series, and the default goodness of fit is exactly that, but it takes many runs.

For an ODE, gradients by automatic differentiation make a local optimizer (L-BFGS-B) the default; a symbolic or source-to-source route, as camdl takes, gives the same thing. Measured on small ODEs, L-BFGS-B with JAX takes about 1.2 to 1.6 s against about 0.15 s for Nelder-Mead with one or two parameters, so the gradient pays off as the number of parameters grows, not for the smallest fits.

### Likelihoods are optional

[odin](../approaches/odin.md) puts the likelihood in the model file, in two lines (`cases <- data()`, `cases ~ Poisson(incidence)`), and gets a particle filter from dust and MCMC from monty. [camdl](../approaches/camdl.md) does the same, plus a separate `fit.toml` with named stages. Both are right that when there is a likelihood it belongs with the quantity it describes. Both also require one before anything can be fitted, and most epidemiologists fitting a model to case counts do not start from a likelihood. They start from "make the curve go through the points".

So the lingua franca starts there too. A normalized absolute error is less rigorous than a likelihood: it does not weight points by their expected noise and it does not give a posterior on its own. It is simple, it has no settings, and it is what [Covasim](../approaches/notes.md#covasim--hpvsim--stisim--fpsim) used by default. Because the differences are not squared, one bad reporting week (a holiday backlog, a batch of late reports) counts in proportion to its size and does not dominate the fit as it would a squared error; dividing by the mean of the data puts a series of thousands of weekly cases and a series of prevalence fractions on one scale, so they can be added. The likelihood is there when wanted, and when one is given on every fitted result, the posterior methods (NUTS, particle MCMC) are chosen automatically.

Against that: [summer2](../approaches/summer2.md), [epipack](../approaches/epipack.md), [MEmilio](../approaches/memilio.md), [EMOD](../approaches/emod.md), [EpiHiper](../approaches/epihiper.md), [`individual`](../approaches/individual.md), and [LASER](../approaches/laser.md) have no built-in comparison to data at all, so every project rebuilds it.

## The specification

[13-inference.md](../design/13-inference.md) specifies all of this: `sim.calibrate()` and its arguments, data attached by name, goodness of fit and likelihoods, how the method is chosen, gradients, staged fitting, what the fit reports, `sim.sensitivity()`, and `ss.compare_models()`. Parameters with bounds and priors are in [04-parameters.md](../design/04-parameters.md), and results and outcomes in [09-results.md](../design/09-results.md).

## Trade-offs

- **Automatic method selection can be wrong.** It is printed, and it is one keyword to override. The alternative, where the user chooses from five methods on their first day, is worse.
- **The normalized absolute error is not a statistical model of the data.** It treats every point in a series as equally noisy, and dividing by the mean behaves badly for a series that is mostly zeros. It is the default because it is simple, needs no decisions, and is hard to throw off; the summary says which measure was used, and `gof=` or a likelihood is one keyword away.
- **An absolute error is not smooth where the model passes through a data point.** Gradient-based optimizers cope with this in practice, and somewhat less cleanly than with a squared error. `gof='rmse'` is the smooth alternative, and the price is sensitivity to outliers.
- **Weights are arbitrary.** Equal weights across series is a convention, not a truth. A joint likelihood is the principled alternative and needs the correlation between series, which is rarely known. Weights are the v1 answer.
- **Optuna is not an inference method.** It gives a point estimate and a search history, not a posterior. The summary gives the range of its best trials as an indication of uncertainty and says that it is not an interval.
- **ABC needs many runs** (by default 200 particles over up to 8 generations). Fine for a compartmental model that runs in milliseconds, slow for an agent-based one, so it is asked for (`method='abc-smc'`) and never chosen by default.
- **Gradients are an implementation project, and they constrain how functions are written.** camdl gets them from symbolic differentiation; summer2 gets them from JAX. Here the ODE runs in compiled kernels, and JAX runs the same steps only when a gradient is wanted. A rate written as a Python function of the sim has no gradient, so a model with one is fitted by Nelder-Mead; one written with NumPy functions of counts keeps it.

## Rejected

- **A separate calibration DSL.** Everything needed is already declared; a second vocabulary would be a second place for it to drift.
- **`CalibComponent`-style declaration at calibration time** (Starsim today). It restates the result name, the data alignment, and the bounds, and it makes a likelihood mandatory.
- **A required likelihood** (odin, camdl). Right when the user has one; a barrier when they just want to fit a curve.
- **One method for every model.** Optuna on a cheap ODE wastes runs and gives no gradient-based convergence check; NUTS on an agent-based model is not possible. The method follows from the model.
- **Requiring the user to choose an inference method.** They should be able to; they should not have to.
- **Bundling the inference engine.** [odin](../approaches/odin.md)'s odin/dust/monty split is the precedent: "an engine can be replaced, and the inference layer is reusable by models that were never written in the DSL". Wrap Optuna, JAX, an ABC sampler, and a particle filter; own none of them.

## Open questions

The open questions are collected in [19-open-questions.md](../design/19-open-questions.md).
