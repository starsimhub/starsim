# Inference: calibration, sensitivity, and model comparison

If parameters have bounds and results have data, calibration is one call. Nobody has to think about likelihoods to build or fit a model.

Justified by [best-practices/calibration.md](../best-practices/calibration.md).

## The whole interface

```python
import starsim as ss

S, I, R = ss.states(3)
beta = ss.par(0.075, bounds=[0.01, 0.30])            # has bounds, so it is fitted

sir = ss.Disease(
    infection  = S >> I @ beta * I,
    recovery   = I >> R @ ss.days(10),
    prevalence = I.frac,                             # a result; the data has a column with this name
    init_prev  = ss.par(0.02, bounds=[0.01, 0.05]),
    name       = 'sir',
)
net = ss.RandomNet(n_contacts=ss.par(4, bounds=[2, 10]))
sim = ss.Sim(sir, networks=net, n_agents=2000, start='2020-01-01', stop='2020-02-12', dt=ss.days(1), data='prevalence.csv')

fit = sim.calibrate()
fit.plot()
```

`calibrate()` fits **every parameter that has `bounds`** to **every result that has data**. Both were already stated where they belong: the bounds on the parameter, the data on the sim. Nothing is said twice.

### Compared with Starsim today

The same calibration with Starsim 3.7's `ss.Calibration` and `CalibComponent` (condensed from the Starsim calibration tutorial):

```python
import pandas as pd
import starsim as ss

def make_sim():
    sir = ss.SIR(beta=ss.perday(0.075), init_prev=ss.bernoulli(0.02))
    net = ss.RandomNet(n_contacts=ss.poisson(4))
    return ss.Sim(n_agents=2000, start=ss.date('2020-01-01'), stop=ss.date('2020-02-12'), dt=ss.days(1), diseases=sir, networks=net, verbose=0)

calib_pars = dict(                                   # the bounds, stated away from the parameters
    beta       = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    init_prev  = dict(low=0.01, high=0.05, guess=0.02),
    n_contacts = dict(low=2, high=10, guess=3, suggest_type='suggest_int'),
)

def build_sim(sim, calib_pars, **kwargs):            # how to put each value back into the sim
    sir = sim.pars.diseases
    net = sim.pars.networks
    for k, pars in calib_pars.items():
        v = pars['value']
        if k == 'beta':
            sir.pars.beta = ss.perday(v)
        elif k == 'init_prev':
            sir.pars.init_prev = ss.bernoulli(v)
        elif k == 'n_contacts':
            net.pars.n_contacts = ss.poisson(v)
    return sim

data = pd.read_csv('prevalence.csv', parse_dates=['t']).set_index('t')   # one column, 'x'

prevalence = ss.Normal(                              # what to compare, how to align it, and a likelihood
    name       = 'Disease prevalence',
    conform    = 'prevalent',
    expected   = data,
    extract_fn = lambda sim: pd.DataFrame({'x': sim.results.sir.prevalence}, index=pd.Index(sim.results.timevec, name='t')),
    sigma2     = 0.05,
)

calib = ss.Calibration(sim=make_sim(), calib_pars=calib_pars, build_fn=build_sim, components=[prevalence], total_trials=100, reseed=True)
calib.calibrate()
calib.check_fit()
```

About thirty lines against three (the three `bounds=`, `data=`, and `sim.calibrate()`), and the thirty repeat things the model already knows: where each parameter lives, what the prevalence result is called, and that a prevalence is read at a point in time while a count of new cases is summed over an interval. The lingua franca version has no build function, no extract function, no alignment keyword, and no likelihood.

### Arguments

```python
fit = sim.calibrate(
    pars       = None,   # which parameters to fit; default: all that have bounds
    data       = None,   # data to fit to; default: the sim's data
    weights    = None,   # relative weight of each data series; default: equal
    likelihood = None,   # dict of result name -> distribution, if not given on the result
    gof        = None,   # the goodness of fit: a name ('rmse') or a function; default: normalized absolute error
    method     = None,   # the fitting method; overrides the automatic choice; a list runs several in sequence
    n_runs     = None,   # runs per evaluation, for stochastic sims
    rand_seed  = None,
    **kwargs,            # passed to the fitting method (e.g. trials=500, draws=2000)
)
```

`method=` here is the fitting method. How the sim itself is run is its `solver=` ([12-backends.md](12-backends.md)), which calibration does not change.

A parameter's `prior=`, if it has one, is used by methods that take priors; otherwise the prior is uniform over the bounds. A parameter whose bounds are positive and span a factor of ten or more is searched on a log scale, and this is printed.

## Data

Data is attached to the sim and matched to results by name, as in Covasim:

```python
sim = ss.Sim(sir, data='cases.csv')       # a path, a dataframe, or ss.data(...)
```

- The time column is `date`, `year`, `t`, or `time`. Every other column is matched to a result with the same name (`cases`, or `sir.cases` when the short name is ambiguous). A column that matches nothing is an error that lists the closest result names (`E705 UnknownDataColumn`).
- Headers that do not match are renamed on the way in, and columns that are not wanted are dropped: `data=ss.data(df, aliases=dict(cases='Reported Cases', year='Years'), ignore_extra=True)` ([09-results.md](09-results.md#data)).
- Results that have data are plotted against it by `sim.plot()` without being asked.
- **Alignment is automatic.** A result that is a state of the system at a point in time (a count, a prevalence) is compared at the data's dates, interpolating if needed. A result that is a flow (new infections, reported cases, deaths) is summed over the interval ending at each data date. The library knows which is which because results are declared ([09-results.md](09-results.md)).
- Stratified data uses extra columns named after the stratification (`age`, `sex`), matched to the result's coordinates.
- Data wholly outside the simulated period is an error (`E704 DataMisaligned`); data partly outside is used where it overlaps, with a note.

## Goodness of fit

The default is deliberately simple:

> **For each data series, the normalized absolute error: the mean of |model − data|, divided by the mean of the data. The series are combined as a weighted average, with equal weights unless `weights=` says otherwise.**

```python
fit = sim.calibrate(weights=dict(cases=1, deaths=2, seroprevalence=1))
```

This is the measure Covasim uses. It assumes no noise model, so there is nothing to choose before fitting, and because the differences are not squared, one bad reporting week does not dominate the fit. Dividing by the mean puts series with different units and sizes (thousands of cases, a prevalence of a few percent) on the same scale, so that equal weights are a reasonable starting point. A series whose mean is zero is divided by its range instead. For a stochastic sim the model value is the mean over `n_runs` runs. The measure, the weights, and each series' contribution are printed by `fit.summarize()`.

Other measures are chosen by name or given as a function:

```python
fit = sim.calibrate(gof='rmse')    # root-mean-square error, normalized by the mean of the data
fit = sim.calibrate(gof=fn)        # fn(sim) returns a number to minimize
```

The default measure is not a likelihood, and it does not need to be: it is enough to find parameters that make the model go through the data, which is what most calibration is for. Which measure should be the default, and which others get names, is open ([19-open-questions.md](19-open-questions.md)): the requirement is a sensible default that needs no noise model, and count data with many zeros may want a different one.

### Likelihoods, when wanted

A likelihood can be attached to a result. It says how the observed number is distributed around the model's value, and it enables methods that give a posterior:

```python
cases = ss.outcome(0.5 * infection, delay=ss.days(5), likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
```

or, for any result, at calibration time: `sim.calibrate(likelihood=dict(prevalence=ss.normal(scale=0.02)))`.

- The model's value is the mean of the distribution; the distribution's other parameters are given by the user. One that is an `ss.par` with bounds (the dispersion above) is fitted with the rest.
- If every series being fitted has a likelihood, the fit uses the sum of their log-likelihoods, treating the series as independent, and says so. `weights=` still applies, as multipliers on each term; with weights other than 1 the result is labeled as weighted.
- If only some series have one, the fit uses the default measure for all of them and notes that the likelihoods were not used.
- A joint likelihood with correlation between series is not in v1 ([19-open-questions.md](19-open-questions.md)).

## Choosing the method

The right fitting method follows from the model, so the library picks it and prints the choice with the reason. `method=` overrides it.

| The model is | No likelihood (default) | With a likelihood |
|---|---|---|
| Deterministic (`solver='ode'`) | minimize the goodness of fit: gradient-based (L-BFGS-B) with exact gradients, simplex (Nelder-Mead) when a gradient is not available | NUTS, started from the optimum |
| Stochastic compartmental | ABC-SMC, using the goodness of fit as the distance; gives uncertainty without a likelihood | particle MCMC |
| Agent-based, cheap enough to run thousands of times | ABC-SMC | ABC-SMC (the likelihood is used as the distance) |
| Agent-based, expensive | Optuna; a point estimate | Optuna; a point estimate |

"Cheap enough" is decided by timing one run against the default budget of evaluations, and the decision is printed.

```text
>>> fit = sim.calibrate()          # the SIR above with solver='ode'
model: deterministic ODE, 3 parameters to fit, 1 data series (prevalence, 3 points)
method: L-BFGS-B with gradients by automatic differentiation  (override with method=)
goodness of fit: normalized absolute error, equal weights
converged: yes (41 evaluations)
beta        0.071
init_prev   0.023
n_contacts  4.6
note: beta and n_contacts enter the model only as a product; their product (0.33) is determined, the individual values are not
```

```text
>>> fit = sim.calibrate()          # the same model with a network, run as agents
model: agent-based (2,000 agents, 0.02 s per run), 3 parameters to fit
method: ABC-SMC, 5 generations of 200 particles, 4 runs each  (override with method=)
goodness of fit: normalized absolute error, equal weights
note: common random numbers on, so runs are paired across parameter sets
beta        0.071  [0.052, 0.094]
init_prev   0.023  [0.012, 0.041]
n_contacts  4.6    [2.9, 7.8]
```

**Optuna gives a point estimate, not a posterior.** It is the right tool when each run is expensive and nothing else is affordable, and the output says what it is: `interval: not available (method='optuna' gives a point estimate)`. The spread of trial values MUST NOT be presented as an interval. [CK: true, but can also sample the best N trials, or do a weighed sample by GOF, or do other workarounds that give at least some indication of uncertainty.]

### Gradients

For a model run with `solver='ode'`, the fit has exact gradients. The ODE solver runs through JAX ([12-backends.md](12-backends.md#how-a-model-is-built)), so `sim.calibrate()` gets the derivative of the goodness of fit (or of the log-likelihood) with respect to every fitted parameter by automatic differentiation, in the same pass that runs the model. This is what camdl gets by symbolic differentiation, without anything symbolic. Two things follow:

- **Gradient-based optimizers** (L-BFGS-B) are the default for ODE models, and converge in far fewer runs than a simplex or a trial-based search when there are more than a few parameters.
- **HMC/NUTS** is available for ODE models that have a likelihood, which gives a posterior at a cost that does not grow badly with the number of parameters.

Conditions and limits:

- A function in a rate is differentiable if it is written with array operations that JAX can trace. One that is not is reported in one line, and the fit uses finite differences or the simplex method instead.
- Without JAX installed, small models are fitted with SciPy using finite differences or the simplex method, and NUTS is not offered.
- Gradients exist for `solver='ode'` only. Stochastic compartmental and agent-based sims are fitted by the methods in the table that need none.
- `sim.sensitivity()` can use the same gradients for local sensitivities.
- Symbolic Jacobians, fixed points, and eigenvalues are not part of this and stay out of v1.

## Staged fitting

Sometimes one fitting method is not enough. A cheap, robust search is good at finding roughly where the best fit is and bad at saying how uncertain it is; a sampler is good at describing the uncertainty and slow, or lost, if started far from the best fit. Staged fitting runs the first and then the second.

`method=` takes a list. **Each entry is a fitting method. They are run in the order given, and each one starts from the best fit the previous one found.**

```python
fit = sim.calibrate(method=['optuna', 'nuts'])
```

Read aloud: "search with Optuna; then, starting from the best parameters Optuna found, sample with NUTS". The returned `fit` is the result of the last stage that ran.

An entry is either a method's name, which uses that method's default settings, or a dict that also gives its settings:

```python
fit = sim.calibrate(method=[
    dict(method='optuna', trials=500),    # stage 1: 500 trials to find a good region
    dict(method='nuts', draws=2000),      # stage 2: 2,000 posterior draws, started from stage 1's best fit
])
```

```text
stage 1 of 2: Optuna TPE, 500 trials
  best fit 0.094 (normalized absolute error); improvement over the last quarter of trials 0.2%: converged
stage 2 of 2: NUTS, 2000 draws, started from stage 1
  R-hat 1.00, effective sample size 1,640: converged
beta        0.071  [0.058, 0.086]
init_prev   0.023  [0.014, 0.036]
```

What "starts from the previous best fit" means depends on the stage that follows: an optimizer takes it as its initial guess, and a sampler (NUTS, particle MCMC, ABC-SMC) starts its chains or its first generation around it.

**A stage that fails to converge stops the sequence.** Each method has its own convergence check ([Convergence reporting](#convergence-reporting)). If a stage does not pass it, the stages after it are not run, the output says which stage stopped and why, and the fit returned is the last good one: the result of the last stage that did converge (`W1303 StageStopped`). So a poor starting point is never fed into an expensive sampler. If the first stage is the one that fails, its best attempt is returned, labeled as not converged.

```text
stage 1 of 2: Optuna TPE, 500 trials
  still improving after 500 trials (4.1% over the last quarter): not converged
W1303 StageStopped: stage 1 (optuna) did not converge, so stage 2 (nuts) was not run.
  Returning stage 1's best attempt. Try more trials, or tighter bounds.
```

A single method is the one-stage case: `method='optuna'` and `method=['optuna']` are the same.

## The result

```python
fit.sim          # the calibrated sim: a copy of the original with the best-fit values, already run
fit.pars         # a table: parameter, initial value, best fit, interval if the method gives one, bounds
fit.plot()       # results against data, before and after, per data series
fit.summarize()    # estimates, goodness of fit per series, convergence diagnostics
fit.to_df()      # every evaluation or sample
```

The original `sim` is not changed. `fit.sim` is an ordinary sim: plot it, add interventions to a copy, pass it to `ss.parallel`. For methods that give a posterior, `fit.plot()` shows the spread of trajectories across samples.

### Convergence reporting

`fit.summarize()` reports whether the fit can be trusted, by method:

| Method family | Reported |
|---|---|
| Least squares | final gradient norm or simplex size, evaluations, whether any value is at a bound |
| NUTS / MCMC | R-hat, effective sample size, divergences |
| Particle MCMC | acceptance rate, effective sample size, particle degeneracy |
| ABC-SMC | acceptance rate, final tolerance, effective sample size |
| Optuna | best value, number of trials, improvement over the last quarter of trials, and the statement that this is a point estimate |

A best-fit value at or near a bound is flagged (`W1301 AtBound`), because that usually means the bound was too tight rather than that the parameter is extreme.

## What is model and what is fitting

- A likelihood declared on a result is part of the model: it says how observations relate to the process.
- Everything else here is a fitting setting: the data, weights, goodness of fit, fitting method, number of trials, and seeds. Changing them does not change the model, so they are not part of `sim.hash()`; the fit records its own hash of them ([11-random.md](11-random.md#run-identity)).

In practice both live in the same script, which is fine and avoids the two drifting apart.

## Fitting a product of parameters

With a network or mixing pool, `beta` is per contact and the number of contacts belongs to the network, so incidence data determine only their product. The library notes this when it can see it (`W1302 Unidentifiable`): from the model, when two fitted parameters appear only multiplied together; or from the fit, when two parameters are almost perfectly correlated across samples or trials.

To fit the product directly:

```python
contacts = ss.par(4, bounds=[2, 10])
net = ss.RandomNet(n_contacts=contacts)

fit = sim.calibrate(pars=[beta * contacts, sir.pars.init_prev])
```

A product in `pars=` is one free quantity. Its bounds are the products of the factors' bounds; the search moves the first factor and holds the others at their current values; and the product is what is reported.

## Sensitivity

```python
sim.sensitivity()                          # every parameter with bounds, against every result
sim.sensitivity(results=['n_I'])           # against chosen results
```

Latin hypercube samples over the bounds, with partial rank correlation coefficients between each parameter and a summary of each result (its peak and its final value, by default). It needs no data. It answers a different question from calibration (which parameters matter?) and is often worth running first, since a parameter that does not matter cannot be fitted.

## Model comparison

Which of two models is better supported by the data? "Does this model need an exposed compartment?" is this kind of question.

```python
comp = ss.compare_models([sim_sir, sim_seir], data=df)
comp.plot()
```

`ss.compare_models` judges models by how well they predict data they were not fitted to:

1. Hold out the last part of each data series (the final 25% of its time points by default; `holdout=` takes a fraction or a date).
2. Fit each model to the rest with `sim.calibrate()`, using the method chosen for that model.
3. Run each fitted model forward over the held-out period.
4. Score the predictions against the held-out data: with the default goodness of fit, or with the log predictive density if every series has a likelihood.
5. Report a table.

```text
>>> comp = ss.compare_models([sim_sir, sim_seir], data=df)
held out: the last 25% of each series (cases: 2020-10-01 to 2020-12-31)
            fitted pars   fit error   prediction error   rank
sim_seir    3             0.08        0.12               1
sim_sir     2             0.15        0.31               2
prediction error is the normalized absolute error on held-out data (lower is better)
```

- "Fit error" is the error on the data used for fitting, shown for reference. A model with more parameters will usually fit better; "prediction error" is the column that matters, and no separate penalty for the number of parameters is needed.
- `data=` is attached to every sim, so each model must have results named like its columns. Without `data=`, each sim's own data is used, and it must be the same for all of them.
- Models are labeled by the sims' `label`, else by position.
- `folds=k` repeats the comparison with `k` different cut points (fit to the first part, predict the next) and reports the mean and spread of the prediction error. With stochastic models the spread across runs is included. When the spread is larger than the difference between models, the table says there is no clear difference rather than ranking them.
- `comp.fits` holds each model's `fit`; `comp.summarize()` and `comp.to_df()` give the table.

The models can differ in any way (structure, stratification, solver), since each is just a sim.

## Rejected

- **A separate calibration vocabulary** (Starsim's `calib_pars`, `build_fn`, `CalibComponent`). Bounds belong on parameters and data belongs with results; restating them elsewhere is verbose and drifts.
- **Requiring a likelihood.** Most users want the model to go through the data. A likelihood is available when the question needs one.
- **A squared error as the default goodness of fit.** One bad reporting week dominates it. It is available as `gof='rmse'`.
- **One method for every model.** Optuna on a three-equation ODE is slow and gives no uncertainty; NUTS on an agent-based model is not possible. The model determines the method.
- **Presenting the spread of optimizer trials as an interval.**
- **`fit.apply(sim)`.** The calibrated sim is `fit.sim`.
- **Writing the inference methods from scratch.** Wrap existing optimizers and samplers (SciPy, Optuna, a NUTS implementation, an ABC-SMC implementation); the library's job is to connect the model to them.
- **Information criteria as the comparison.** They need a likelihood and a parameter count that means something; out-of-sample prediction needs neither.
