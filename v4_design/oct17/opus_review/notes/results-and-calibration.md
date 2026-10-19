# Review: results, randomness, calibration, and many runs

Reviewer area: `results.py`, `calibration.py`, `gradients.py`, `run.py`, `samples.py`, the outcome/result parts of `transitions.py`, and the CRN/stream parts of `distributions.py`; design docs 09, 11, 13 and the matching best-practices docs. Every repro script named below is in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/results/` and was run with `/software/conda/envs/v4/bin/python`. Every bug claim below was reproduced.

## Summary

The core v4 idea in this area works and is worth keeping: data attached to results by name, automatic flow/stock alignment, `sim.calibrate()` with bounds on `ss.par`, a fast NUTS path through JAX, and per-distribution CRN streams that give identical draws across the three spellings of a model. The problems are mostly about there being two of everything. There are two calibration systems (v3 `ss.Calibration` + `CalibComponent`, 1,145 lines, which the user docs still teach; and v4 `ss.fit`), and `sim.calibrate()` fails with a confusing error on any sim built from library modules such as `ss.SIR`. There are two or three shapes of MultiSim, depending on which path ran it. `ss.parallel(sim, n_runs=20)` on a compartmental sim returns a MultiSim with `len()` 1, whose `plot()` draws one run and whose `summarize()` reports a standard deviation of 0. `ss.MultiSim(sim, n_runs=20).run()` has `results=None`. The spec says these are equivalent. A result's "kind" (flow, stock, or ratio) is spread over `summarize_by`, a name heuristic, `scale`, and `denom`, and that split causes three silent wrong answers: a v3 flow compared to data as if it were a stock, a stratified ratio across runs summed as counts, and `resample()` averaging a stock. Several spec features are silently missing and not listed in `implementation.md`: staged fitting, `sim.sensitivity()`, sim-level outcomes, `ss.Sim(results=[...])`, the CRN listing in `describe()`, and a converged flag. The headline examples in `00-overview.md` and `13-inference.md` fail as written.

Top three: (1) merge the two calibration systems into one `sim.calibrate()` that works for every sim, and delete `CalibComponent` and its family; (2) make a MultiSim a single thing, with results that always carry a run dimension however it was run; (3) give `ss.Result` one explicit `kind` (flow/stock/ratio), and derive summarizing, resampling, data alignment, and reduction across groups and runs from it.

## Findings

### 1. Two calibration systems; `sim.calibrate()` refuses class-based sims

Type: design / API / code simplicity.

Evidence: `starsim/calibration.py` lines 26–1170 are the v3 `ss.Calibration` (Optuna wrapper with `calib_pars`, `build_fn`, `eval_fn`, `components`, sqlite DB handling) plus `CalibComponent`, `BetaBinomial`, `Binomial`, `DirichletMultinomial`, `GammaPoisson`, `Normal`, and `linear_interp`/`linear_accum`/`step_containing`, all still exported from `starsim/__init__.py:166-171`. Lines 1174–2587 are the v4 `ss.data`/`ss.fit`/`ss.Fit`/`ParticleFilter`/`compare_models`. These two halves share nothing: different likelihood objects (`ss.Normal` component vs `ss.normal` dist), different alignment (`conform='prevalent'` vs `summarize_by`), different result objects. `13-inference.md` §Rejected rejects the first half explicitly, yet it stays. `docs/user_guide/workflows_calibration.qmd` and `workflows_sir_calibration.qmd` teach only `ss.Calibration`, and no doc teaches `sim.calibrate()`. `tests/test_calibration.py` tests only the v3 class, while the v4 path is tested in `test_transitions.py` and `test_gradients.py`.

A user with a normal Starsim sim gets a non-answer (`t11_v3.py`):

```text
>>> ss.Sim(diseases=ss.SIR(beta=ss.perday(0.075)), networks='random', data=df, ...).calibrate()
ValueError: sim.new() is only available for sims with declared states and transitions
```

Proposal: one entry point, `sim.calibrate()`, for every sim. Bounds are an `ss.par` wherever a value goes, including the pars of class-based modules. Starsim 3's two real needs become keyword escape hatches on the same call: `build=` (a function that applies values to a fresh sim, for parameters that are not pars) and `gof=` (already exists). Likelihoods are the `ss.poisson`/`ss.nbinom`/`ss.normal`/... dists already used by v4, and a `ss.betabinom` dist covers `BetaBinomial`. Delete `Calibration`, `CalibComponent`, the five component subclasses, and the three conform functions, or move them to `starsim.library`/a compat module for one release.

```python
# Before (v3, still the documented way)
calib = ss.Calibration(sim=sim, calib_pars=dict(beta=dict(low=0.01, high=0.3, guess=0.1, log=True)),
                       build_fn=build_sim, components=[ss.Normal(name='prev', expected=df, extract_fn=..., conform='prevalent', sigma2=0.05)], total_trials=100)
calib.calibrate()
# After (works for ss.SIR too)
sim = ss.Sim(diseases=ss.SIR(beta=ss.par(0.1, bounds=[0.01, 0.3])), networks='random', data=df)
fit = sim.calibrate(trials=100)
```

Cost: breaking for every Starsim 3 calibration script, and for any downstream package that wraps `Calibration` (worth checking STIsim and HPVsim). That is a migration guide plus a deprecation shim, not a reason to keep two systems. The one real extra is a `par` hook on class-based modules: `ss.fit` currently reads `mod.graph.pars`, so it needs a generic walk over `mod.pars` for `ss.par` with bounds. About 1,100 lines leave the core.

### 2. A MultiSim has several incompatible shapes depending on how it was run

Type: design / bug / API.

Evidence (`t4_msim.py`, a Gillespie SIR):

```text
ss.parallel(sim, n_runs=20):   len 1, results MultiResults, n_I.values (20, 61)
  msim.plot():                 1 line per axis (the last run; spec: median and interval)
  msim.summarize():            sir_n_I: mean 145.0, std 0.0, sem 0.0   (one sim, not 20 runs)
ss.MultiSim(sim, n_runs=20).run():  len 20, results None -> msim.results.sir fails
sim.run(n_runs=20):            returns a MultiSim (not a Sim), len 1
ABM ss.parallel(sim, n_runs=4):len 4, results MultiResults
msim.results.sir.n_I.select(run=3) -> numpy.ndarray (select(scenario=...) -> MultiResult)
```

`09-results.md` §One run, many runs says `ss.parallel(sim, n_runs=100)`, `sim.run(n_runs=100)`, and `ss.MultiSim(sim, n_runs=100).run()` are the same thing. In the code they are three. The cause is in `run.py`: `restart_runs()` (line 1110) keeps one runner sim per input sim and puts every run only in `msim.results`, while `parallel()`'s other path keeps one sim per run, and `MultiSim.run()` never builds `MultiResults` at all. On top of that, the v3 surface (`reduce`/`mean`/`median`/`combine`/`summarize`/`compare`/`plot` working from `self.sims`, with `which` and `orig_base_sim` state) sits beside the v4 one (`MultiResult.quantiles/mean/median/select/by/diff/plot`), and each works only on some of the shapes.

Proposal: a MultiSim is its `MultiResults`. Every path (restart, kernel `run_many`, JAX `run_sets`, multiprocessing) ends in the same constructor, which builds results with `run` (and `scenario`) dimensions. `len(msim)` is the number of runs. `msim.sims` is optional, kept with `keep_sims=True`. `plot()`, `summarize()`, `extinction()`, and `diff()` work from the results. `reduce()`/`mean()`/`median()` become methods on results (`msim.results.sir.n_I.median()`) and stop mutating the MultiSim's `base_sim`. `MultiSim.run()` and `ss.parallel()` become the same code. Drop `sim.run(n_runs=)`, since a method whose return type depends on an argument is a trap and `ss.parallel(sim, n_runs=)` is one call. `select(run=3)` returns a `Result`.

```python
# After: these are the same object, whatever the solver
msim = ss.parallel(sim, n_runs=100)
msim = ss.MultiSim(sim, n_runs=100).run()
len(msim)                                   # 100
msim.results.sir.n_I.quantiles([0.1, 0.5, 0.9])
msim.plot()                                 # median and interval, from results
```

Cost: v3 code that reaches into `msim.sims[i]` after a run would need `keep_sims=True` for agent-based sims (the multiprocessing path can keep the current default). `msim.reduce()` mutating `base_sim` is used by HPVsim-style plotting and would need a shim. In exchange, roughly 400 lines of `MultiSim` (`reduce`, `combine`, `summarize`, `compare`, `plot` branches, `merge`/`split`) collapse onto one results object.

### 3. Bug: a stratified ratio across runs is summed as a count

Type: bug (silent wrong answer).

Evidence (`t7_msratio.py`, ABM, prevalence by age):

```text
single sim prev.select(age=0) final       0.0806
msim prev.select(age=0).values[0] final   22.0      # the numerator count, not a prevalence
```

`MultiResults.from_sims()` (`run.py:1076`) copies `strata` but not `denom`, and `MultiResult.select()` (`run.py:966`) sums `strata` over the kept groups. `09-results.md` says a ratio is never averaged or summed over groups. On the kernel path (`t6_strat.py`), the same call raises `AttributeError: 'NoneType' object has no attribute 'shape'` from `_keep()`, because ratios are not kept by group there (a known issue). It should raise the readable "not kept by group" error, which exists in `_strat()` but runs after `_keep()`.

Proposal: carry `denom` into `MultiResult` and reuse `Result._substrata()` instead of reimplementing selection. Better still, make `MultiResult` an `ss.Result` with a leading run axis (finding 4), so `select`/`by` exist once. Cost: small.

### 4. A result's kind lives in four places; make it one `kind` attribute

Type: design / code simplicity (root cause of findings 3, 5, and 6).

Evidence: whether a result is a flow, a stock, or a ratio decides how it is summarized, resampled, aligned to data, scaled, combined, and reduced over groups and runs. It is currently `summarize_by` (`'sum'`/`'mean'`/`'last'`/None), `Result.summary_method()`'s name heuristic (`'new_' in name` → sum, `n_` → mean, `cum_` → last), `scale` (counts vs not), and the presence of `denom` (ratio). Callers disagree about which to read: `sim.py:476` uses `res.summarize_by or res.summary_method()`, while `calibration.py:1330,1359` uses `res.summarize_by == 'sum'` only, and `MultiResult` ignores `denom`. v3 even has `tfr` with `summarize_by='sum'` (`demographics.py:693`).

Proposal: `ss.Result(kind='flow' | 'stock' | 'ratio' | 'cumulative')`, set at declaration (declared results already know it), with the old heuristic used only to fill `kind` for v3 results that do not give it. `summarize_by`, `scale`, and the alignment rule are derived from it, and `resample()`, `compare()`, `select()`, `combine()`, and `reduce()` each switch on `kind`. That is one word for the user to learn (if ever) and one place in the code.

Cost: touches every `ss.Result(...)` in the v3 modules (mechanical), and `summarize_by=` stays as a deprecated alias.

### 5. Bug: v3 flow results are compared to data as stocks

Type: bug (silent wrong answer).

Evidence (`t20_v3flow.py`): weekly `new_infections` data on an `ss.SIR` sim. `res.summarize_by` is None and `res.summary_method()` is `'sum'`, so `ss.compare()` interpolates the one-day value instead of summing the week:

```text
weekly data           [1038.  944.    5.    0.]
model as compared     [ 345.    8.    0.    0.]
```

`ss.compare` and `ss.mismatch` are public exports, and they would become reachable from `sim.calibrate()` once finding 1 is fixed. Proposal: use `kind` (finding 4), or at minimum `res.summarize_by or res.summary_method()` in `compare()` and `lineup()`. Cost: trivial.

### 6. `Result.resample()` does not do what 09 says, and fails on undated sims

Type: doc-code mismatch / bug.

Evidence (`t1_basic.py`, `t5_outcome.py`): with `unit='day'` and no dates, `res.infection.resample('week')` raises `TypeError: Only valid with DatetimeIndex...` from pandas. On a dated sim, `res.n_I.resample('week')` averages the stock over pandas' calendar weeks (anchored to Sunday, so the first bin is 1–5 January), giving 15.5. 09 says it should take the value at the end of the week (the real values are 33.1 at day 6 and 40.3 at day 7). `prevalence.resample('week')` averages the ratio (0.0053) instead of recomputing it from its resampled parts (0.0131 at the end of week 2).

Proposal: implement resampling from `kind` on the sim's own time grid (every k steps), without pandas: flows sum, stocks take the last value, cumulatives take the last value, and ratios resample numerator and denominator and then divide. Keep pandas offsets only for calendar periods (`'month'`, `'year'`) on dated sims. Cost: about 40 lines, and `annualize()` becomes `resample('year')`.

### 7. Bug: `infection_log` records nothing for declared diseases

Type: bug.

Evidence (`t16b.py`):

```text
declared SIR, 500 agents, homogeneous mixing: cum_infections 400, log rows 0
declared SIR, 500 agents, networks='random':  cum_infections 495, log rows 0
v3 ss.SIR, networks='random':                 log rows 499
```

Asking for it does switch `solver='auto'` to agents, as 09 says, but the table is empty. Nothing in `implementation.md` mentions this. Proposal: have the declared engine's infection step report (infector or source group, infectee, network) to the analyzer, as 09 §The transmission log describes. Cost: moderate; it is the one place where the declared agent engine has to hand events to an analyzer.

### 8. The spec's headline examples fail as written

Type: doc-code mismatch.

Evidence:

- `00-overview.md` and `13-inference.md` §The whole interface: `ss.RandomNet(n_contacts=ss.par(4, bounds=[2, 10]))` raises `TypeError: Updating dist from <class 'starsim.distributions.constant'> to <class 'starsim.transitions.par'> is not supported` (`t2_calib.py`). Bounds on a network parameter cannot be given at all.
- `00-overview.md`: `ss.parallel(baseline=fit.sim, masks=...)` raises `AlreadyRunError: Simulation is already complete` (`t17_overview.py`), because `fit.sim` "is already run" (13 §The result) and `parallel` will not copy it. Either `parallel` copies a run sim (it does for initialized ones: `sim.copy() if sim.initialized`, but only on the `n_runs` path), or `fit.sim` stays unrun and `fit.plot()` runs what it needs.
- `09-results.md` §Automatic results: the two-line SIR with no `init=`/`init_prev=` runs with nobody infected. Every result is 0, there is no note, and it prints `RuntimeWarning: Dates with years < 1 are not valid (year = 0.0); returning ss.datedur instead` for a plain `unit='day'` sim (`t1_basic.py`). If "guess the default" means everyone in S, then a transmission model that can never transmit deserves a one-line note (W-level), not silence.
- `09-results.md` §ss.outcome: `ss.Sim([flu, rsv], ili=ss.outcome(...))` raises `KeyNotFoundError: Key(s) ['ili'] not found` (`t15_misc.py`).
- `09-results.md` §Requesting fewer results: `ss.Sim(sir, results=['n_I'])` raises `KeyNotFoundError: ['results']`.
- `ss.Sim(..., solver=None)` raises `Unknown solver "None"`, while `None` means the default everywhere else in the API.

Proposal: add the overview and 13 examples as doctests (principle "documentation shipped with the version, doc-tested"). Fix the two that are bugs (par on network pars, `parallel` of a run sim) and mark the others as not yet implemented in `implementation.md`. Cost: small, and it is exactly the class of drift doctests prevent.

### 9. Fits never say whether they converged; staged fitting depends on that and is missing

Type: bug / doc-code mismatch.

Evidence (`t13_conv.py`, `t3_ode.py`):

- `fit()` throws away `result.success`/`result.message` from `scipy.optimize.minimize` (`calibration.py:2269-2272`), and nothing judges Optuna's trial history. With `maxiter=5`, Nelder-Mead returns a negative log-likelihood of 6,330 against an optimum of 61, with no warning; `Fit` has no `converged` attribute. 13 §Convergence reporting promises gradient norm, simplex size, "converged: yes (41 evaluations)", and, for Optuna, "improvement over the last quarter of trials, and the statement that this is a point estimate". None of these is printed.
- `W1301 AtBound` is a false positive for parameters searched on a log scale. `init_prev` fitted to its true value 0.0008 within bounds [0.0001, 0.1] is flagged "at one of its bounds", because the check uses a linear 1% of the span (`calibration.py:1974-1976`); on the log scale it is 30% of the way along. The check also runs only in `fit.summarize()`, not when the fit finishes.
- `method=['nelder-mead', 'l-bfgs-b']` fails with `AttributeError: 'list' object has no attribute 'lower'`. Staged fitting (13 §Staged fitting, best-practices recommendation 6) is not implemented, and `implementation.md` does not say so.

Proposal: give `Fit` `converged` (bool) and `why` (str), filled by each method from its own diagnostic (SciPy's `success`; Optuna's improvement over the last quarter of trials; R-hat/ESS/divergences for NUTS; acceptance for pMCMC). Print one line at the end of the fit. Do the AtBound check on the searched scale, at fit time. Then staged fitting is a ten-line loop (`x0 = previous.best`, stop when `not previous.converged`). Implement it, or drop it from the spec. My lean is to implement it, since it is cheap once `converged` exists, and `optuna → nuts` is the one combination worth it.

```python
fit = sim.calibrate(method=['optuna', 'nuts'])
fit.converged, fit.why      # (False, 'optuna: still improving 4.1% over the last quarter of 100 trials; nuts not run')
```

Cost: small. Each method needs a few lines to report what it already knows.

### 10. Silent spec gaps in this area

Type: doc-code mismatch.

`implementation.md` checks off 09, 11, and 13 without listing these, all verified absent:

- `sim.sensitivity()` (13 §Sensitivity): no such method.
- Staged fitting (finding 9).
- `describe()` listing CRN coverage (11 §What CRN covers): the output has no "common random numbers" section (`t8_crn.py`).
- `sim.hash().config` is documented to include "whether CRN is on"; `Sim.hash()` (`sim.py:1272`) hashes `spec['config']`, which has no CRN entry, so `ss.options.crn=False` runs share a config hash.
- `sim.crn_report()` (a MAY, fine to omit).
- Sim-level outcomes and `results=` (finding 8).
- The Optuna "interval: not available" line and the CRN note in the calibration printout.

Proposal: either implement or list each under "Not yet" in `implementation.md`. `sim.sensitivity()` is worth reconsidering: LHS + PRCC is about 40 lines on top of `ss.parallel(sim, pars=...)`, which already exists and is fast. Otherwise drop it from the spec, since `ss.parallel(sim, pars=lhs)` plus a correlation is a two-line recipe. Cost: none to list them.

### 11. Fitting-method auto-selection: simplify, and pick the faster method for small models

Type: design / performance.

Evidence (`t3_ode.py`, 2-parameter ODE SIR): the auto choice is L-BFGS-B with JAX gradients, which takes 1.56 s; Nelder-Mead reaches the same answer in 0.16 s. `implementation.md` already measured this ("With 3 it is slower than Nelder-Mead (1.7 s against 0.25)"), and best-practices `calibration.md` §Gradients says "A model with two or three parameters does not need any of this and uses plain SciPy". `default_method()` (`calibration.py:2072`) ignores the parameter count. For stochastic sims, the default is Optuna with `n_runs=1` and the seed held fixed across trials, so the fit is tuned to one realization, and `fit.sim` (the same seed) looks better than it is. The spec's printout says "4 runs each". Also, the reported `fit.mismatch` comes from one run at the default seed, while the trials are averaged over `n_runs` seeds (`t19_misc.py`: best trial 0.176, reported 0.184).

Proposal: keep auto-selection (it is the right idea and it prints its reason), but make the table smaller and closer to what was measured:

| | no likelihood | every series has a likelihood |
|---|---|---|
| ODE | Nelder-Mead if ≤ 4 parameters, else L-BFGS-B | NUTS |
| stochastic compartmental | Optuna | pMCMC |
| agents | Optuna | Optuna |

Default `n_runs` to something like 4 for stochastic sims, and report `fit.mismatch` with the same seeds as the trials. Cost: a slower default for stochastic fits (four times the runs), in exchange for a less overfit answer, and it is one keyword to undo.

### 12. `fit()` is one 300-line function; likelihoods, priors, and outcomes are written twice

Type: code simplicity.

Evidence: `calibration.py:2095-2398` is a single function with about 14 closures (`to_pars`, `measure`, `run`, `objective`, `with_gradient`, `encode`, `decode`, `to_x`, `share_of`, `to_z`, `logpost_pf`, `logpost`, `exact`, `logpost_jax`) and an if-chain over six methods. Three parameter transforms coexist (log/linear `encode`, logit-of-bounds `to_z` for samplers, and Optuna's own `log=`). `gradients.Density` (`gradients.py:797-1121`) reimplements `ss.calibration.logpdf` ("As `ss.calibration.logpdf()`, in JAX"), `logprior`, the outcome convolution/stay/reporting logic of `ss.Compartmental.derived_outcomes()`, and result lookup. Because the two can drift, `fit()` checks the JAX density against a Python run at three draws and resamples the whole posterior if they disagree (`calibration.py:2335-2340`). `compare()` and `lineup()` (`calibration.py:1310-1366`) duplicate the alignment rule, once as values and once as a matrix.

Proposal:

- A `Problem` object (fitted pars, bounds, transforms, `objective(x)`, `logpost(z)`) built once, and a registry `methods = dict(optuna=..., nelder_mead=..., nuts=..., pmcmc=...)`, each a function `(problem, x0, **kw) -> MethodResult(best, converged, why, samples)`. That is the natural shape for staged fitting too.
- `logpdf(dist, observed, model, xp=np)` and `logprior(par, x, xp=np)` written once against an array namespace, with `xp=jax.numpy` and `jax.scipy.stats` passed in by `Density`.
- `compare()` defined as `W @ values` from `lineup()`, so the alignment rule exists once.

Cost: a refactor with no user-visible change. My guess is that `fit()` would end up around 150 lines, and the duplicated JAX code (about 120 lines) would go along with the consistency check.

### 13. `compare_models` compares unlike measures

Type: bug / design.

Evidence (`t14_cmp.py`): an SIR whose `cases` outcome has a Poisson likelihood against an SEIR without one:

```text
       fitted_pars  fit_error  prediction_error  rank
sir              2  50.530645          0.236581     1
seir             2   0.107029          0.938744     2
```

`fit_error` is a negative log-likelihood for one row and a normalized absolute error for the other, under one heading. The table is also printed twice (once by the constructor, once by `summarize()`). Proposal: a comparison uses one measure for every model, the log predictive density only if every series of every model has a likelihood and the default goodness of fit otherwise, and says which. Printing belongs to `summarize()` (or to `verbose`), not to construction. Cost: small.

### 14. Misleading message when `gof=` is given with likelihoods

Type: bug (message).

Evidence (`t12_lik.py`): `calibrate(gof='nae')` on a model whose only series has a likelihood prints "Note: not every data series has a likelihood, so the likelihoods of cases are not used". The real reason is that `gof=` was given (`use_lik = all(...) and gof is None` at `calibration.py:2141`, while the message at 2165 assumes the other branch). Proposal: "gof='nae' was given, so the likelihood of cases is not used". Cost: one line.

### 15. Data-column handling: near-misses get the wrong advice

Type: API.

Evidence (`t5_outcome.py`): a time column named `Date` or `week` is not recognized (`data.time_cols` is case-sensitive and has no `week`), and the error treats it as a mistyped result: "the data column "Date" matches no result. The closest results are cases, n_S, n_R. Rename it with ss.data(..., aliases=dict(result='Date'))". The suggested alias also uses the literal placeholder `result` even when a close match exists (`case` → `cases`).

Proposal: match time columns case-insensitively, and add `week`/`month`. When a time column is not found, say so first ("no time column: expected one of date, year, t, time"). When the best match is close, write the alias out (`aliases=dict(cases='case')`). Cost: trivial. E705 with closest names is otherwise one of the best errors in this area.

### 16. `Fit` carries more than the user needs, with names that mislead

Type: API.

Evidence: `Fit` has `sim`, `before`, `pars`, `table`, `trials`, `samples`, `band`, `divergences`, `acceptance`, `method`, `mismatch`, `mismatch_before`, `measure`, plus `to_df()`, which just returns `trials`. With a likelihood, `mismatch` and the `trials.mismatch` column hold a negative log-likelihood. Proposal: `fit.pars`, `fit.sim`, `fit.table` (estimates, intervals, and diagnostics), `fit.samples` (draws or trials), `fit.converged`/`fit.why`, `fit.summarize()`/`plot()`/`to_df()`, with `before`, `band`, `divergences`, and `acceptance` inside `fit.diagnostics`. Rename the column to `objective` (and say in `measure` what it is). Cost: minor breakage of something only weeks old.

### 17. `samples.py` (`ss.Samples`, `ss.Dataset`) is disconnected v3 machinery

Type: design / vocabulary.

Evidence: `samples.py` (432 lines) stores many runs as zip files keyed by parameter dicts. Nothing in `starsim/` uses it except the export in `__init__.py:175`, and it knows nothing of `MultiResults`, `msim.run_pars`, or `sim.hash()`, which now cover the same need (many runs, indexed by parameters and seed). Proposal: drop it from the top-level namespace (move it to `starsim.library` or delete it). If saving large sweeps is wanted, add `msim.save()`/`ss.load()` of `MultiResults` with `run_pars` and the hashes as the index. Cost: the vocabulary shrinks by two names. Any user of `Samples` (I believe some HPVsim/FPsim workflows) needs the library import.

### 18. CRN and streams: keep, with a note on what users should expect

Type: design (keep).

Evidence (`t8_crn.py`): the same SIR written as arrow strings, operators, and a class gives identical results (2,202 cumulative infections) and identical `sim.hash()` (`model='23435d', pars='7e44f8'`), and adding an unrelated disease leaves the first disease's results unchanged. The splitmix hash per slot (`distributions.py:42`) is simple and fast. That is exactly what 11 asks for. Paired scenarios (`t10_crn.py`) with a 0.1% change in beta differ by 0–11 infections with CRN on a network, against −14 to +24 with CRN off. With homogeneous mixing, CRN on and off pair about equally well, because both reuse the same seed and the draw order hardly changes. For a 10% change, the paired difference had a spread close to that of independent runs (`t9_pair.py`), because the epidemics genuinely diverge.

Proposal: keep the mechanism as is. Do not oversell it in the docs: 13's "common random numbers on, so runs are paired across parameter sets" note and best-practices "CRN makes calibration cheaper" are true for small perturbations and weak once trajectories diverge, so the promised `describe()` CRN section should be the honest statement. One small simplification: `str2int(name) + rand_seed` (`distributions.py:764`) means seed s for stream A and seed s+1 for a stream whose hash is one larger are the same. Hashing `(name, rand_seed)` together costs nothing and removes the possibility. Cost: changes every stream's numbers once, which would need a baseline update.

### 19. Smaller items

- `ss.Result` arithmetic error messages name the wrong result: `(n_I / (n_S + n_I + n_R)).select(age=0)` says 'The result "n_I" is not stratified', because the output takes the first operand's name (`t6_strat.py`). Name it after the expression, or say "this ratio of results".
- An anonymous `ss.par` in an outcome weight (`ss.outcome(ss.par(0.5, bounds=...) * infection)`) errors with "give it one with ss.par(..., name="beta")". The suggestion should name something neutral, or the parameter could take a name from the outcome (`cases_weight`).
- Running `sim.calibrate()` on an ODE once imports JAX, and a later `ss.parallel()` of agent-based sims in the same process forks (seen as `self.pid = os.fork()` warnings in `t17_overview.py`). This is a known issue in `implementation.md`, but the user-facing fix could be automatic: switch `sc.parallelize` to `spawn` when `'jax' in sys.modules`.
- `msim.summarize()` returns mean/std/sem of each sim's scalar summary, which for flows is the mean over time of the per-step flow. Once finding 2 is done, it should report the quantity that 09 describes (e.g. final size and peak, with an interval), plus extinction.

## What is good and should be kept

- Data attached by name, with automatic flow/stock alignment (`t5_outcome.py`: a weekly outcome compared with itself gives exactly 0; biweekly sums of a daily flow align exactly), long-form data by stratum, and E705 with the closest names.
- `ss.outcome` as a weighted sum of flows with `delay`/`duration`/`every`, chaining, and results that are ordinary results. `results=` on the disease for ratios that keep numerator and denominator.
- `select()`/`by()` on single-sim stratified results, including ratios done correctly (`t6_strat.py`).
- NUTS through JAX: 2 parameters, 2 chains × 200 draws in 4 s, including compilation, recovering beta = 0.400 and rho = 0.299 (truth 0.4, 0.3) (`t12_lik.py`). The MLE by L-BFGS-B takes 0.75 s.
- `sim.restart()` and `restart_runs()` make compartmental sweeps very fast (20 Gillespie runs in 0.18 s). The fix in finding 2 is about the object they return, not the speed.
- Identifiability notes (`W1302`) from the model structure and from correlated draws are a real improvement over Starsim 3.
- Per-distribution streams and CRN by slot (finding 18).
