# Inference: calibration, sensitivity, and model comparison

If parameters have bounds and results have data, calibration is one call. Nobody has to think about likelihoods to build or fit a model.

Justified by [best-practices/calibration.md](../best-practices/calibration.md).

## The whole interface

<!-- fragment -->
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

`calibrate()` fits **every parameter that has `bounds`** to **every result that has data**, and returns an `ss.fit.Fit`. Both were already stated where they belong: the bounds on the parameter, the data on the sim. Nothing is said twice. A parameter of a module written as code, given as an `ss.par` with bounds (`ssl.HIV(beta=ss.par(0.01, bounds=[0.001, 0.1]))`), is fitted too. A parameter with a value for each level of a stratification (`ss.par(ss.by(patch=[0.2, 0.3]), bounds=[0.1, 1])`) is fitted level by level, with the same bounds and prior, as `beta[a]`, `beta[b]`, and so on, and a typed value takes typed bounds (`ss.par(ss.days(8), bounds=[ss.days(3), ss.weeks(4)])`).

The fitting itself is the submodule `ss.fit`, which an ordinary run never imports: `sim.calibrate()` imports it when it is called, and `import starsim.fit` makes its parts available directly (`ss.fit.loglik()`, `ss.fit.ParticleFilter`, `ss.fit.compare_models()`, and so on).

### Compared with Starsim 3

The same calibration with Starsim 3.7's `ss.Calibration` and `CalibComponent` (condensed from the Starsim calibration tutorial):

<!-- fragment -->
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

<!-- fragment -->
```python
fit = sim.calibrate(
    pars       = None,   # which parameters to fit; default: all that have bounds
    data       = None,   # data to fit to; default: the sim's data
    weights    = None,   # relative weight of each data series; default: equal
    likelihood = None,   # dict of result name -> distribution, if not given on the result
    gof        = None,   # the goodness of fit: 'nae' (the default, normalized absolute error), 'rmse', or a function of the sim; any other name is an error
    method     = None,   # the fitting method; overrides the automatic choice; a list runs several in sequence
    n_runs     = None,   # runs per evaluation: 4 for a stochastic compartmental sim (the same seeds at every point), 1 for agents and for a deterministic one
    rand_seed  = None,
    **kwargs,            # passed to the fitting method (e.g. trials=500, draws=2000)
)
```

`method=` here is the fitting method. How the sim itself is run is its `solver=` ([12-backends.md](12-backends.md)), which calibration does not change.

A parameter's `prior=`, if it has one, is used whenever every data series has a likelihood: the posterior methods sample the posterior, and the methods that find a best fit minimize the negative log-posterior (the most probable values), which is printed. A parameter with no prior is uniform over its bounds. Without likelihoods, priors are not used. A parameter whose bounds are positive and span a factor of ten or more is searched on a log scale, and this is printed.

A stochastic sim is scored at each point by the mean of the scores of `n_runs` runs (4 by default for a stochastic compartmental sim, and 1 for agents, whose runs are slow and whose search, by Optuna, already averages over nearby points; not the score of their mean trajectory), with the same seeds at every point (common random numbers), so that chance does not differ between points; the number of runs is printed. Optuna is the exception: each point takes fresh seeds. Its search learns from nearby points, and with the same seed everywhere it would find the optimum of that one seed's realization, which pooling cannot average out; a local method such as Nelder-Mead keeps the same seeds, or its steps follow the noise. With fresh seeds the lowest single trial is the luckiest, which favors values whose runs vary most, so the best values are those of the trial whose neighbors score lowest: each trial's score averaged with the others' by a Gaussian kernel of their distance on the searched scale, with a width by Scott's rule. The pairing is close for small changes in the values and weak once the runs' trajectories diverge (an epidemic that takes off in one run and not in another), so the score of a stochastic sim is still noisy, and more runs make it less so. Particle MCMC and ABC-SMC take a run of their own for each point instead. `fit.mismatch` is that same score at the best values, so it is the score that was minimized, and `fit.scores` holds the score of each run; `fit.sim` is the run with the first seed, for plotting.

## Data

Data is attached to the sim and matched to results by name, as in Covasim:

<!-- fragment -->
```python
sim = ss.Sim(sir, data='cases.csv')       # a path, a dataframe, or ss.data(...)
```

- The time column is `date`, `year`, `t`, or `time`. Every other column is matched to a result with the same name (`cases`, or `sir.cases` when the short name is ambiguous). A column that matches nothing is an error that lists the closest result names (`UnknownDataColumn`).
- Headers that do not match are renamed on the way in, and columns that are not wanted are dropped: `data=ss.data(df, aliases=dict(cases='Reported Cases', year='Years'), ignore_extra=True)` ([09-results.md](09-results.md#data)).
- Results that have data are plotted against it by `sim.plot()` without being asked.
- **Alignment is automatic.** A result that is a state of the system at a point in time (a count, a prevalence) is compared at the data's dates, interpolating if needed. A result that is a flow (new infections, reported cases, deaths) is summed over the interval ending at each data date. The library knows which is which because results are declared ([09-results.md](09-results.md)).
- Stratified data uses extra columns named after the stratification (`age`, `sex`), matched to the result's coordinates.
- Data wholly outside the simulated period is an error (`DataMisaligned`); data partly outside is used where it overlaps.

## Goodness of fit

The default is deliberately simple:

> **For each data series, the normalized absolute error: the mean of |model − data|, divided by the mean of the data. The series are combined as a weighted average, with equal weights unless `weights=` says otherwise.**

A series whose data are all zero (deaths early in an epidemic, a stratum with no cases) has no mean to divide by; it is divided by the mean of the model instead, so it adds 1 when the model has any of it and 0 when the model has none, and cannot swamp the other series in raw counts. To push a fit toward zero more strongly, give that series a likelihood or a weight.

<!-- fragment -->
```python
fit = sim.calibrate(weights=dict(cases=1, deaths=2, seroprevalence=1))
```

This is the measure Covasim uses. It assumes no noise model, so there is nothing to choose before fitting, and because the differences are not squared, one bad reporting week does not dominate the fit. Dividing by the mean puts series with different units and sizes (thousands of cases, a prevalence of a few percent) on the same scale, so that equal weights are a reasonable starting point. A series whose mean is zero is divided by its range instead (and an all-zero series by 1). For a stochastic sim the score is the mean of the scores of `n_runs` runs. `fit.summarize()` prints the measure before and after the fit.

Other measures are chosen by name or given as a function:

<!-- fragment -->
```python
fit = sim.calibrate(gof='rmse')    # root-mean-square error, normalized by the mean of the data
fit = sim.calibrate(gof=fn)        # fn(sim) returns a number to minimize; then data is not needed
```

The default measure is not a likelihood, and it does not need to be: it is enough to find parameters that make the model go through the data, which is what most calibration is for. Count data with many zeros may want a different measure, given as a function.

### Likelihoods, when wanted

A likelihood can be attached to a result. It says how the observed number is distributed around the model's value, and it enables methods that give a posterior:

<!-- fragment -->
```python
cases = ss.outcome(0.5 * infection, delay=ss.days(5), likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100])))
```

or, for any result, at calibration time: `sim.calibrate(likelihood=dict(prevalence=ss.normal(scale=0.02)))`, keyed by the result's name or the data column ([09-results.md](09-results.md#an-optional-likelihood)). The likelihoods are `ss.poisson`, `ss.nbinom` (`n` the dispersion), `ss.binom` (`n` the number of trials, the model's value the probability), `ss.bernoulli`, `ss.normal` (`scale`), and `ss.lognorm_ex` (`std`).

- The model's value is the mean of the distribution; the distribution's other parameters are given by the user. One that is an `ss.par` with bounds (the dispersion above) is fitted with the rest.
- If every series being fitted has a likelihood, the fit uses the sum of their log-likelihoods, treating the series as independent, and says so. `weights=` still applies, as multipliers on each term; with weights other than 1 the result is labeled as weighted.
- If only some series have one, the fit uses the default measure for all of them and notes that the likelihoods were not used.
- A joint likelihood with correlation between series is not in v1 ([19-open-questions.md](19-open-questions.md)).

## Choosing the method

By default a fit is a best fit (a point estimate), by the method that suits the sim, which the library picks and prints with the reason; `method=` overrides it. A posterior is always asked for by name, so that what a fit answers does not depend on what is installed: the same call asks the same question on every machine.

| The sim | The default: a best fit | A posterior, by name |
|---|---|---|
| `solver='ode'`, with gradients (JAX installed, and the kernels run the sim whole) | L-BFGS-B, with gradients by automatic differentiation (Nelder-Mead for one or two parameters, where it is faster) | `'nuts'`, with a likelihood for every series (needs BlackJAX) |
| `solver='ode'`, without gradients | Nelder-Mead | `'abc-smc'` |
| Stochastic compartmental, run whole by the kernels | Optuna | `'pmcmc'` (particle MCMC), with a likelihood for every series |
| Agent-based, or a stochastic run stepped from Python | Optuna | `'abc-smc'` |

With a likelihood for every series, the best fit is the maximum likelihood (or, with priors, the most probable values), and the printed reason names the posterior method the sim allows. The other methods are chosen by name: `'abc-smc'` (approximate Bayesian computation by sequential Monte Carlo: a posterior without a likelihood, for any sim, with `generations=`, `quantile=`, `tolerance=`, and `particles=`), `'if2'` (iterated filtering, as pomp's and camdl's: the maximum likelihood of a stochastic compartmental sim, from passes of a particle filter in which each particle's values follow a shrinking random walk, with `particles=`, `iterations=`, `cooling=`, `sd=`, and `chains=`), `'asd'` (adaptive stochastic descent, `sc.asd()`), and any method of `scipy.optimize.minimize`. Settings for a method are keywords of `calibrate()`: `trials=` for Optuna (100 by default), `draws=`, `warmup=`, and `chains=` for NUTS (500, 500, and 4) and particle MCMC (1,000, 1,000, and 4), `particles=` for particle MCMC (chosen by each chain in its warm-up by default). An unknown `method=` is an error that lists the methods. For the SciPy methods, a search starts from each of the best three points (`restarts=3`) of a spread of points over the bounds (`scan=`, 10 per parameter by default, with the initial values) that are at least a tenth of the bounds apart, and the best result is kept; NUTS starts each chain from one of the best of the same points, so that chains caught in different modes disagree and R-hat sees it.

The same model with no network, run as an ODE:

```python
S, I, R = ss.states(3)
beta = ss.par(0.2, bounds=[0.05, 0.5])
sir = ss.Disease(
    infection  = S >> I @ beta * I,
    recovery   = I >> R @ ss.days(10),
    prevalence = I.frac,
    init_prev  = ss.par(0.02, bounds=[0.001, 0.05]),
    name       = 'sir',
)
sim = ss.Sim(sir, start='2020-01-01', stop='2020-02-12', dt=ss.days(1), data='prevalence.csv')
fit = sim.calibrate()
```

```text
Searching on a log scale: init_prev, beta
Fitting by nelder-mead (2 fitted parameters, for which it is faster than gradients; override with method=)
Converged: the best two of 3 searches, from points of the scan that are apart, ended within 0.1 of each other (1342, 1342, 1388; 287 evaluations)
```

```text
>>> fit = sim.calibrate()          # the model at the top, with a network, run as agents
Searching on a log scale: beta
Fitting by optuna (a stochastic sim; override with method=)
Scoring each point by the mean of 4 runs, with the same seeds at every point (override with n_runs=)
Converged: the best value improved by 0.4% over the last quarter of the 100 trials
```

Parameters that the data cannot tell apart are noted when the fit ends (`Unidentifiable`; see [Fitting a product of parameters](#fitting-a-product-of-parameters)), and so is a best fit at a bound (`AtBound`).

**Optuna gives a best fit, not a posterior.** It is the right tool when each run is expensive and nothing else is affordable, and the output says what it is. The spread of trial values MUST NOT be presented as an interval, but as an indication of the uncertainty the fit gives the range of the best 10% of the trials (`spread_lower` and `spread_upper` in `fit.table`), labeled as such by `fit.summarize()`.

### Gradients

For a sim with `solver='ode'`, the fit has gradients. The ODE is run by the kernels (RK4 in Numba); for a gradient, the same RK4 steps are taken in JAX and differentiated by automatic differentiation, and the steps either side of the run (from the parameters to the numbers in the program, and from the run to the results and the goodness of fit) by central differences, so a gradient is exact to about 1e-10 ([12-backends.md](12-backends.md)). For NUTS, the whole log-density (the run, the results with data, the likelihood, and the priors) is one function in JAX, differentiated in forward mode, so the sampling is compiled and the chains run on cores of their own; a model it does not cover yet is sampled with Python in the loop, which is much slower, and the fit says so. This is what camdl gets by symbolic differentiation, without anything symbolic. Two things follow:

- **Gradient-based optimizers** (L-BFGS-B) are the default for ODE models, and converge in far fewer runs than a simplex or a trial-based search when there are more than a few parameters.
- **HMC/NUTS** is available for ODE models that have a likelihood, which gives a posterior at a cost that does not grow badly with the number of parameters.

Conditions and limits:

- Gradients need a sim that the kernels run from start to finish: no interventions other than the deliveries the kernels take, no modifier with `when=`, no transfer whose rate varies, and no movement register. A rate written with NumPy functions of counts has gradients; a rate that is a Python function of the sim has none. Without gradients the fit uses Nelder-Mead, and the printed reason says why.
- Without JAX installed, ODE models are fitted by Nelder-Mead; NUTS also needs BlackJAX.
- Gradients exist for `solver='ode'` only. Stochastic compartmental and agent-based sims are fitted by the methods in the table that need none.
- The same machinery is available directly: `sim.gradient()` gives the derivatives of results with respect to parameters, and `sim.log_density()` the log-density as a function in JAX, with its gradient by reverse mode.
- Symbolic Jacobians, fixed points, and eigenvalues are not part of this and stay out of v1.

## Staged fitting

Sometimes one fitting method is not enough. A cheap, robust search is good at finding roughly where the best fit is and bad at saying how uncertain it is; a sampler is good at describing the uncertainty and slow, or lost, if started far from the best fit. Staged fitting runs the first and then the second.

`method=` takes a list. **Each entry is a fitting method. They are run in the order given, and each one starts from the best fit the previous one found.**

<!-- fragment -->
```python
fit = sim.calibrate(method=['optuna', 'nuts'])
```

Read aloud: "search with Optuna; then, starting from the best parameters Optuna found, sample with NUTS". The returned `fit` is the result of the last stage that ran.

An entry is either a method's name, which uses that method's default settings, or a dict that also gives its settings:

<!-- fragment -->
```python
fit = sim.calibrate(method=[
    dict(method='optuna', trials=500),    # stage 1: 500 trials to find a good region
    dict(method='nuts', draws=2000),      # stage 2: 2,000 posterior draws, started from stage 1's best fit
])
```

```text
Stage 1 of 2: optuna
Using the likelihood: the sum of the log-likelihoods of prevalence, treated as independent
Converged: the best value improved by 0.0% over the last quarter of the 500 trials
Stage 2 of 2: nuts, started from stage 1
Using the likelihood: the sum of the log-likelihoods of prevalence, treated as independent
NUTS: 4 chains on 4 cores, each of 500 warm-up draws and then 2000 draws: compiled in 2.3 s, sampled in 3.1 s
Converged: R-hat at most 1.003 and effective sample size at least 2,146, with 0 divergent draws
```

What "starts from the previous best fit" means depends on the stage that follows: an optimizer and NUTS start from it (it is the best point of the scan, and the other searches or chains start from the next best points apart from it), particle MCMC starts its chains from the best of it and a spread of points over the bounds, and ABC-SMC draws its first generation from the prior, so for ABC-SMC an earlier stage changes nothing.

**A stage that fails to converge stops the sequence.** Each method has its own convergence check ([Convergence reporting](#convergence-reporting)). If a stage does not pass it, the stages after it are not run, a warning says which stage stopped and why (`StageStopped`), and the fit returned is the last good one: the result of the last stage that did converge. So a poor starting point is never fed into an expensive sampler. If the first stage is the one that fails, its best attempt is returned, labeled as not converged.

```text
Stage 1 of 2: optuna
Not converged: the best value improved by 4.1% over the last quarter of the 500 trials; try more trials (trials=)
StageStopped: stage 1 (optuna) did not converge (the best value improved by 4.1% over the last quarter of the 500 trials; try more trials (trials=)), so the stages after it were not run; returning its best attempt
```

A single method is the one-stage case: `method='optuna'` and `method=['optuna']` are the same.

## The result

<!-- fragment -->
```python
fit.sim          # the calibrated sim: a copy of the original with the best-fit values, already run
fit.pars         # the best-fit value of each parameter, a dict that can be passed to ss.Sim(pars=...)
fit.table        # a table: parameter, initial value, best fit, bounds, and the interval and diagnostics if the method gives them
fit.samples      # with a posterior method, the draws: one row per draw, with its chain
fit.plot()       # results against data, before and after, per data series
fit.summarize()  # the table, the goodness of fit before and after, and the convergence diagnostics
fit.to_df()      # every evaluation
fit.project(dict(baseline=None, vaccine=ss.vaccinate(...)), n=100)   # scenarios run with the posterior's draws
```

The original `sim` is not changed. `fit.sim` is an ordinary sim: plot it, add interventions to a copy, pass it to `ss.parallel`. For methods that give a posterior, `fit.plot()` shows a band over the posterior (the 2.5th and 97.5th percentiles of the model's value over draws), and `fit.project()` runs each scenario once per draw (or with the best fit, for a method that gives no posterior), returning a MultiSim whose results have a run dimension labeled by scenario. `ss.parallel(sim, priors=True)` draws the parameters from their priors instead, for forward runs before fitting.

### Convergence reporting

Every fit says whether it converged: `fit.converged` is true or false, and `fit.why` says on what grounds, from what the method already knows. It is printed as one line when the fit ends, and again by `fit.summarize()`, which reports, by method:

| Method family | Reported |
|---|---|
| SciPy optimizers | whether the best two searches from the scan's points ended within 0.1 of each other (of the negative log-likelihood; 0.1% of a goodness of fit), with the value each ended at and the number of evaluations; a single search (`restarts=1`) reports SciPy's own rule, which is wrong in both directions often enough not to be relied on |
| `asd` | whether the method stopped by its own rule (its message), and the number of evaluations |
| NUTS | R-hat (rank-normalized) and bulk effective sample size of each parameter, and divergences; converged if R-hat is at most 1.01 with no divergent draws |
| Particle MCMC | R-hat, effective sample size, and the share of proposals each chain accepted; converged if R-hat is at most 1.01 |
| ABC-SMC | the final tolerance, the share of the last proposals accepted, and the effective sample size; converged if the tolerance has stopped shrinking (by less than 10% in the last generation) |
| Optuna | the number of trials and the improvement over the last quarter of them (converged if less than 1%) |

A best-fit value at or near a bound is flagged when the fit ends (`AtBound`), because that usually means the bound was too tight rather than that the parameter is extreme. "Near" is within 1% of the range on the scale the parameter was searched on, so a value near the low end of a log-scaled range is not flagged.

## What is model and what is fitting

- A likelihood declared on a result is part of the model: it says how observations relate to the process.
- The data is not part of the model either, but it is part of the run's identity: `sim.hash()` has a part for it (`data`), apart from the model's ([11-random.md](11-random.md#run-identity)).
- Everything else here is a fitting setting: the weights, goodness of fit, fitting method, number of trials, and seeds. Changing them does not change the model or the sim's hash.

In practice both live in the same script, which is fine and avoids the two drifting apart.

## Fitting a product of parameters

With a network or mixing pool, `beta` is per contact and the number of contacts belongs to the network, so incidence data determine only their product. The library notes this when it can see it (`Unidentifiable`), with the product (or ratio) the data determine: from the model, when two fitted parameters appear in its rates only multiplied together (or one divided by the other); over a posterior, when two are almost perfectly correlated; and in a fit by gradients, when two change the fitted results in almost the same way at the best fit.

The remedy is to fit one of the two and fix the other, which is what the note says. Fitting a product as one free quantity (`pars=[beta * contacts]`) is not built.

## Sensitivity

```python
sim.sensitivity()                          # every parameter with bounds, against every result
sim.sensitivity(results=['n_I'])           # against chosen results
```

Latin hypercube samples over the bounds (`n=100` by default; wide positive bounds on a log scale, as `calibrate()` searches them), with partial rank correlation coefficients between each parameter and a summary of each result (a count by its peak and its final value, a flow by its peak and its total), printed and returned as a data frame. It needs no data. It answers a different question from calibration (which parameters matter?) and is often worth running first, since a parameter that does not matter cannot be fitted.

## Model comparison

Which of two models is better supported by the data? "Does this model need an exposed compartment?" is this kind of question.

<!-- fragment -->
```python
comp = ss.fit.compare_models([sim_sir, sim_seir], data=df)
comp.plot()
```

`ss.fit.compare_models` judges models by how well they predict data they were not fitted to:

1. Hold out the last part of each data series (the final 25% of its time points by default; `holdout=` takes a fraction or a date).
2. Fit each model to the rest with `sim.calibrate()`, by its default method, which is a best fit (an optimizer, not a sampler).
3. Run each fitted model forward over the held-out period.
4. Score the predictions against the held-out data, the same way for every model: with the default goodness of fit, or, if every series of every model has a likelihood, with the negative log-likelihood of the held-out data at the best fit.
5. Report a table.

```text
>>> comp = ss.fit.compare_models([sim_sir, sim_seir], data=df)
>>> comp.summarize()
held out: the last 25% of each series (cases: 2020.10.04 to 2020.12.27)
       fitted_pars  fit_error  prediction_error  rank
model
seir             3   0.075541          0.120583     1
sir              2   0.124173          0.307811     2
fit error and prediction error are the normalized absolute error, on the data fitted to and on the held-out data (lower is better)
```

- "Fit error" is the error on the data used for fitting, shown for reference. A model with more parameters will usually fit better; "prediction error" is the column that matters, and no separate penalty for the number of parameters is needed.
- `data=` is attached to every sim, so each model must have results named like its columns. Without `data=`, each sim's own data is used, and it must be the same for all of them.
- Models are labeled by the sims' `label`, else by position (`model 1`, ...), or by the keys when the sims are given as a dict.
- `folds=k` repeats the comparison with `k` different cut points (fit to the first part, predict the next block) and reports the mean and spread of the prediction error across the folds. When the spread is larger than the difference between the best two models, the summary says there is no clear difference between them.
- `comp.fits` holds each model's `fit` (one per fold, with `folds=`); `comp.summarize()` prints the table and `comp.to_df()` returns it.

The models can differ in any way (structure, stratification, solver), since each is just a sim.

## Rejected

- **A separate calibration vocabulary** (Starsim's `calib_pars`, `build_fn`, `CalibComponent`). Bounds belong on parameters and data belongs with results; restating them elsewhere is verbose and drifts.
- **Requiring a likelihood.** Most users want the model to go through the data. A likelihood is available when the question needs one.
- **A squared error as the default goodness of fit.** One bad reporting week dominates it. It is available as `gof='rmse'`.
- **One method for every model.** Optuna on a three-equation ODE is slow and gives no uncertainty; NUTS on an agent-based model is not possible. The model determines the default method, and whether the answer is a best fit or a posterior is asked with `method=`.
- **A posterior by default when it is possible.** A deterministic sim with a likelihood for every series used to be sampled by NUTS where BlackJAX was installed and fitted by L-BFGS-B where it was not, so the same call gave a posterior on one machine and a best fit on another. The default is a best fit everywhere.
- **Presenting the spread of optimizer trials as an interval.**
- **`fit.apply(sim)`.** The calibrated sim is `fit.sim`.
- **Writing the inference methods from scratch.** Wrap existing optimizers and samplers (SciPy, Optuna, BlackJAX's NUTS); the library's job is to connect the model to them. (Particle MCMC and ABC-SMC are short samplers written in Starsim, `ss.fit.pmmh()` and `ss.fit.abc_smc()`, since they need the sim's own particle filter and runs.)
- **Information criteria as the comparison.** They need a likelihood and a parameter count that means something; out-of-sample prediction needs neither.
