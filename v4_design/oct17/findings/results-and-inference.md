# Results, inference, and the model as a file: merged findings

This merges every finding on results, many runs, data, calibration, the model spec, save/load, hashing, and interop from the four reviews (F: `fable_review/05`, `08`, `09`, `10`; F2: `fable2_review/08`; O: `opus_review/notes/results-and-calibration.md` and the serialization and interop parts of `surface-and-machinery.md`; C: `codex_review/REVIEW.txt` F04, F11, F14–F16, F21, F22 and its evidence). The reviews agree to an unusual degree on the diagnosis: each area is two systems side by side (Starsim 3's `Result`/`reduce()`/`Calibration`/`Samples` next to v4's strata, `MultiResult`, and `fit()`), and the silent wrong answers come from that seam, chiefly from a result's flow/stock/ratio kind living in four places. They disagree mainly on what the saved form of a model should be, on which exporters stay, and on a few fitting defaults. I re-ran every single-review silent-wrong-answer claim and every disputed fact (scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/results/`, `c1_*.py` to `c19_*.py`); all reproduced except one detail of F's literal-precision claim. Verdict: the v4 ideas here (data by name, alignment by result kind, `calibrate()` from bounds, the time-point convention, `to_code()`, the hash, `interop/core.py`) are right and should stay; the work is to give `ss.Result` one kind and one set of axes (which removes three silent wrong answers and the `MultiResult` duplicate at once), to make every fit report the score it optimized and whether it converged, and to settle on one human-readable saved form (`to_code()`) plus one checkpoint (a pickle that works), with the JSON spec kept in memory for `hash()` and `diff()`.

## Findings

### R1. Data alignment, `summarize()`, and resampling read a result's kind from different places, and a `new_infections` column is compared as a stock

Raised by: F2 08 f1, O results 4 and 5, F 05§6 and bug 9 (the three summarizers), C interfaces ("design judgments": stock/flow/ratio metadata must survive every operation).

Kind: silent wrong answer.

Status: F2 and O reproduced; I re-ran both (`c3_flowalign.py`). On a declared SIR with weekly data, a column named `infection` is summed over the week and matches exactly (455.5, 1797, 6533); the same numbers under the column `new_infections` (the name the design keeps "so data files written for Starsim carry over") are compared with the one-day value: 109.2, 425.0, 1470.5, about 4.2 times low. For a Starsim 3 `ss.SIR`, `new_infections` data of 1038, 944 are compared with 345, 8. Because `mismatch()` and `loglik()` go through `compare()`/`lineup()`, `sim.calibrate()` fits to the wrong target with no message. Separately, `sim.summarize()` reports the mean of a flow: `sir_infection` 1019 against a total of 93,767 (`c14_misc.py`; F reported 1.24 vs 7.45), because it matches name prefixes (`sim.py:2042`) while the result itself carries `summarize_by='sum'`. Cause: `compare()` reads only `res.summarize_by == 'sum'` (`calibration.py:1330`, `:1359`), `check_dt()` reads `summarize_by or summary_method()` (`sim.py:476`), `resample()` and `summarize()` use name heuristics, and only transition flows and outcomes get `summarize_by` set (`solvers.py:1218`, `:1325`).

Proposal (all four agree on the shape): one attribute set at creation for every result, `kind` in `'flow' | 'stock' | 'ratio'` (O adds `'cumulative'`; with F's `.cum` property in R18 it is not needed), from which summarizing, resampling, data alignment, `check_dt()`, scaling, and reduction over groups and runs are derived. The name heuristic survives only to fill `kind` for Starsim 3 results that do not give it. Delete `summary_method()` and `summarize(how=)`'s prefix dict.

```python
# before: whether data is summed or interpolated depends on the column name
ss.Sim(sir, data=df)            # df has 'new_infections' -> interpolated (wrong); 'infection' -> summed
# after: depends on the result's kind
sim.results.sir.new_infections.kind   # 'flow' -> summed over each data interval, whatever its name
```

Cost: small for the fix (one line per auto-created result plus the fallback); `summarize_by=` stays as a deprecated alias. Breaks: printed `summarize()` values for flows change from mean to sum.

### R2. The many-run object has three shapes; on the default compartmental path it reports one run as if it were ten, and a stratified ratio across runs is summed as a count

Raised by: F 05§2, §9, bugs 1, 2, 5, 7, c2, c6; O results 2, 3, 19; F2 08 f2, f3, f8 and "other" (`len` lies); O surface 4 and F 12§1 and F2 03 (overview's `ss.parallel(baseline=fit.sim)` and `msim.plot()` legend crash).

Kind: silent wrong answer (summaries, plot, ratio), plus crashes.

Status: all reproduced by two or three reviews; I re-ran (`c2_msratio.py`, `c13_msim.py`). `sim.run(n_runs=10)` on a binomial SIR returns a `MultiSim` with `len 1` whose results hold `(10, 61)`; `msim.summarize()` gives `sir_n_I` mean 304.6, std 0.0, sem 0.0; `msim.plot()` draws 1 line per axis; `msim.plot(lines=True)` raises `KeyNotFoundError`; `ss.MultiSim(sim, n_runs=3).run().results` is `None`; `parallel → reduce() → plot()` raises `AttributeError: 'objdict' object has no attribute 'columns'`; `select(run=3)` returns an `ndarray`; `MultiResult` is not an `ss.Result` and `MultiResult + 1` raises; a `pars=dict(beta=[0.2, 0.4])` sweep labels its one sim `''` and `select(beta=0.2)` raises; `ss.parallel(baseline=<a run sim>)` raises `AlreadyRunError`. O's single-review silent wrong answer reproduced exactly: as agents with three runs, `prev.select(age=0)` is 0.0806 for one sim and 22.0 from the MultiSim (the numerator count, since `MultiResults.from_sims()` copies `strata` but not `denom`, `run.py:1076`); on the restart path the same call raises `AttributeError: 'NoneType' object has no attribute 'shape'` instead of the "not kept by group" message.

Proposal (merged; F, O, and F2 converge): a MultiSim is its results. Every path (restart, kernel `run_many`, JAX, multiprocessing) ends in one constructor that builds `ss.Result`s with a leading `run` axis (and `scenario`, and one coordinate per swept parameter); `len(msim)` is the number of runs; `msim.sims` is optional (`keep_sims=True`). `plot()` draws median and interval from the results, `lines=True` draws the runs; `summarize()`, `extinction()`, `diff()` work from the results; `quantiles()`/`median()`/`mean()` are methods on a result and return a `Result`. `reduce()`, `mean()`, `median()` on the MultiSim, `which`, `orig_base_sim`, `MultiResult`, `MultiResults.merge` go (a `reduce()` shim that returns `self` is cheap). `ss.parallel` accepts a sim that has run (it calls `new()`), which fixes the overview example.

```python
# before
msim = ss.parallel(sim, pars=dict(beta=[0.3, 0.5]), n_runs=10)
rows = msim.run_pars[msim.run_pars.beta == 0.3].index; msim.results.sir.n_I.values[rows]
# after
msim.results.sir.n_I.select(beta=0.3).quantiles()     # a Result; .plot(), .to_df(), arithmetic all work
len(msim)                                              # 20
```

Cost: the largest item in this area, about 400–500 lines of `MultiSim`/`MultiResult` collapsing onto R18's `Result`. Breaks: Starsim 3 code that calls `msim.reduce()` and reads flattened `msim.results.sis_prevalence`, or indexes `msim.sims[i]` after an agent run without `keep_sims=True`.

### R3. `Result.resample()` keeps stale strata, averages stocks and ratios, bins by calendar week, and fails without dates

Raised by: C F11 (stale strata); F 05§7, c4, bug 3; F2 08 f4; O results 6.

Kind: silent wrong answer (and a crash for undated sims).

Status: C's single-review claim reproduced on a real sim, not only a hand-built `Result` (`c4_resample.py`): `sim.results.sir.infection.resample('week')` has 9 values, but `.select(age=0)` on it returns 61 daily values, identical to the daily series, under a weekly time axis. The others reproduced by three reviews and by me: `n_I.resample('week')` gives the weekly mean (153.4) where the spec says the value at the end of the week (217.8 at the first Sunday); prevalence is averaged rather than recomputed from resampled parts; the first bin is a partial calendar week anchored on Sunday rather than the sim start; `unit='day'` with no dates raises `TypeError: Only valid with DatetimeIndex`.

Proposal (agreed): one resampler on the sim's own grid, driven by R1's `kind`: flows sum, stocks take the last value, ratios resample numerator and denominator and then divide; bins anchored at the sim start; pandas offsets only for calendar `'month'`/`'year'` on dated sims; it returns one consistent `Result` (values, time, strata, denominator together) or drops the groups explicitly. `annualize()` becomes `resample('year')`, and `compare()`/`check_dt()` use the same function (F2).

Cost: about 40 lines; deletes `annualize()` and the pandas path. Breaks: printed numbers of resampled stocks (mean to end value).

### R4. A stochastic fit reports a different score from the one it optimized

Raised by: C F21, O results 11 (last paragraph: best trial 0.176, reported 0.184).

Kind: silent wrong answer.

Status: reproduced (`c1_fitscore.py`). With `n_runs=3` and `maxiter=1`, every evaluated objective is 0.7289, `fit.mismatch_before` is 0.7289, and `fit.mismatch` is 0.72, so `summarize()` shows an improvement that the search never made. With Optuna (30 trials, `n_runs=4`) the best trial is 0.2917 and `fit.mismatch` 0.2867. Cause, read at `calibration.py:2221-2227` and `:2390-2396`: trials average `measure()` over seeds 1..`n_runs`, while `fit.sim` and its score come from one fresh run at the base seed.

Proposal (C and O agree): `fit.mismatch` is the ensemble score of the chosen point, computed exactly as in the search; keep the per-replicate scores and seeds; label `fit.sim` as a representative run, not as the measure of fit.

```python
fit.mismatch          # 0.7289, the score that was optimized
fit.sim               # one run at the fitted values, for plotting
```

Cost: a few lines. Breaks: nothing but printed numbers.

### R5. Fits never say whether they converged, and the at-bound check misfires on log-scale parameters

Raised by: C F15; F 08§6 and bug 7; O results 9.

Kind: silent wrong answer (an unconverged fit looks like a fit).

Status: three reviews reproduced the missing convergence state (`calibration.py:2269-2272` keeps only `.x`; O: `maxiter=5` gives a negative log-likelihood of 6,330 against an optimum of 61, no warning). O's single-review W1301 claim reproduced (`c6_fitchecks.py`): `init_prev` fitted to its true 0.0008 within `[0.0001, 0.1]` is reported "at one of its bounds", because the check uses a linear 1% of the span although the search was on a log scale; it also fires only in `summarize()`.

Proposal (merged): `Fit.converged` (bool) and `Fit.why` (str), filled by each method from what it already knows (SciPy `success`/`message`/`nfev`; Optuna improvement over the last quarter of trials; R-hat/ESS/divergences for NUTS; acceptance for PMCMC), printed as one line when the fit ends; W1301 computed on the searched scale, at fit time.

Cost: small. Breaks: nothing.

### R6. Inputs that silently change what is fitted

Raised by: F 08§4 (unknown `gof`, all-zero series), F 08§5 (priors ignored by optimizers), O results 14 (misleading message when `gof=` is given with likelihoods).

Kind: silent wrong answer.

Status: all three single-review claims reproduced (`c6_fitchecks.py`, `c18_zero.py`). `gof='foo'` runs and returns exactly the default normalized absolute error (6.16e-05 both ways). A tight prior `lognorm_ex(mean=0.4, std=0.02)` on `beta` changes nothing under `method='l-bfgs-b'` (0.25 with and without). An all-zero `deaths` series adds 38.4 to the mismatch (scale falls back to 1.0, so raw counts), against 0.17 for a `cases` series 50% off: the zero series dominates by more than 200 times.

Proposal: check `gof` against the known names (or a callable) and raise; when every series has a likelihood and priors exist, optimizers minimize the negative log-posterior (MAP), which is three lines and makes `prior=` mean something everywhere; scale a zero-mean data series by the mean of the model values (or by the other series' scale) and say so in the spec; fix O's message ("`gof='nae'` was given, so the likelihood of cases is not used").

Cost: under 20 lines. Breaks: fits that silently used the wrong measure.

### R7. An `every=` outcome is a comb of zeros, and it is compared wrongly with daily data

Raised by: F 05§4. (O results "keep" lists `every=` among `ss.outcome`'s good options; see disagreements.)

Kind: silent wrong answer.

Status: reproduced (`c5_every.py`). `weekly = ss.outcome(0.5*infection, every=ss.weeks(1))` is 0 on six days of seven and 227.7 on the seventh; compared with daily data (16.6, 20.2, 24.7, ...) the model side is 0, 0, 0, 0, 0, 0, 227.7. Daily outcomes compared with weekly data are already exact without `every=`, because `compare()` sums a flow over each data interval.

Proposal: drop `every=` and the "reporting day" function; say "weekly data is compared with the weekly sum automatically; to see a weekly series, `res.cases.resample('week')`" (R3 makes that agree with the data path).

```python
# before
cases = ss.outcome(0.5*infection, every=ss.weeks(1))
# after
cases = ss.outcome(0.5*infection)          # weekly data is matched by summing; plot weekly with .resample('week')
```

Cost: removes one keyword, `info.every` (`solvers.py:1327`, `:1341-1410`), and a solver-table row. Breaks: the overview model's `every=` and `tests/test_transitions.py:651`.

### R8. The hash misses things that change results, and `diff()` and `hash()` disagree about names

Raised by: C F04 (connectors, CRN, data); O results 10 (CRN not in `config`); F 09§9 (function bodies), F 09 bug 3 (literal precision); F2 08 f10 (`hash()` initializes the sim); F 09§6 and O surface 14 (`diff()` says "same" on a rename).

Kind: silent wrong answer (run identity).

Status: re-ran each (`c9_hashfn.py`, `c10_hashconn.py`, `c11_diff.py`, `c12_prec.py`). Two sims whose only difference is `ss.modify(beta, multiply=0.2)` given under `connectors=` have identical hashes and final `n_R` of 8,912 and 223 (`hash()` hashes `modules`, `networks`, `stratify`, `interventions`, not `spec['modifiers']`, `sim.py:1284-1288`). Two sims whose recovery rate is a function returning 0.1 and 0.5 have identical hashes and `n_R` of 8,912 and 244. `ss.options.crn` on and off hash the same. Attached data is not hashed or saved (C). `hash()` sets `initialized` from False to True. Renaming `beta` to `transmission_rate` gives `model: same / pars: same / config: same / scenario: same` from `ss.diff()` while `model` and `pars` hashes differ. F's precision claim reproduced in part: `beta*(1+1e-9)*I` is written `beta * 1 * I`, and `x=1e-9` and `x=3e-7` both hash the same as `x=0`; but the spec round trip of that model kept the hash equal here, unlike F's report (the loss happens before hashing, so both sides agree).

Proposal (merged): hash a complete, normalized definition: modifiers wherever they are given, function source text (`func_source`), literals with `repr` precision, `options.crn` and other numerical options in `config`, and a fingerprint of attached data in a fifth part (`data`), so that "the model changed" and "the data changed" stay distinguishable. Hash without initializing (normalize from the inputs). `diff()` reports renames (`beta renamed transmission_rate`, `module 'sir' renamed 'flu'`) so that a hash difference always has a line that explains it; print endpoints as `S -> I`.

Cost: small to moderate (function source is about 15 lines; data fingerprint a few). Breaks: every existing hash once.

### R9. `ss.compare_models` ranks models on unlike measures

Raised by: O results 13.

Kind: silent wrong answer.

Status: reproduced (`c6_fitchecks.py`). Two identical SIRs, one whose outcome has a Poisson likelihood: `fit_error` is 43.39 (a negative log-likelihood) for one and 0.000037 (normalized absolute error) for the other under one heading; prediction errors are equal (0.000312) and the ranking puts the likelihood model first. The table also prints at construction and again at `summarize()`.

Proposal: one measure for every model in a comparison (log predictive density only if every series of every model has a likelihood, else the default goodness of fit), named in the table; print only from `summarize()`.

Cost: small. Breaks: nothing.

### R10. `to_code()` writes a different model, or broken code, without saying so

Raised by: O surface 16; F 09§7, §9, bugs 1–2; F2 08 f5(a); C F14 (attached data not saved).

Kind: silent wrong answer (the saved model is not the model).

Status: re-ran a matrix of eleven models through `to_spec → from_spec`, `to_code → exec`, and a pickle (`c7_roundtrip.py`, `c8_tocode.py`). `to_code()` of a sim with `ss.RandomNet()` or `ss.vaccinate(...)` succeeds and the generated sim differs (`model`/`pars` or `scenario` hash), with only a `# Not written as code` comment and nothing printed. A function in a rate is written `rate = ss.par({'fn': 'clearance'})` with no footer; the generated file builds, hashes identically (R8), and then fails at run with `TypeError: unsupported operand type(s) for *: 'dict' and 'float'`. `ss.by(...)` in a rate gives `NameError: name 'by' is not defined` (F2's case; F's `seasonal(...)` case is the same bug in `code_expr`, `spec.py:319`; a lone `ss.seasonal` round-tripped here). Attached data is dropped by both `to_code()` and `to_spec()`, silently.

Proposal: `to_code()` never emits code that builds a different model without saying so: anything it cannot write goes to the footer *and* raises `W1401 LossyExport` at call time with the list; function slots write the function's source when available, else go to the footer; prefix every Starsim name with `ss.`; data is written as a file reference (`data='cases.csv'`) or listed. Closing the gaps (networks as `{type, pars}`, interventions) is O surface 6's generic module spec and is worth it whichever saved form wins (R21).

Cost: small for honesty, moderate for coverage. Breaks: nothing.

### R11. An imported `beta * S * I / N` with `N` a parameter equal to the population is read as density-dependent, so the transmission is not contact-mediated

Raised by: F 10§3.

Kind: silent wrong answer (an import builds a different kind of model).

Status: reproduced (`c16_odin.py`, odin `seir_array_odin.R`): the import log says `not divided by the total: density-dependent, kept as counts => beta * I.n / N`, and `describe()` shows `new_inf S -> E spontaneous`, so no mixing, `rel_sus`, or `rel_trans` applies. The message is also wrong on its face.

Proposal: in `core.Rates`, a parameter whose value equals the initial population (within rounding) is the total, marked `[guessed: N = 1e7 is the population]`; otherwise say "divided by the parameter N, not by the population".

Cost: about 10 lines. Breaks: nothing.

### R12. A product of bare numbers in a rate is typed per unit, so every import is right only at the unit it was imported at

Raised by: F 10 bug 1 (and F README §2, checked there; core area, not mine in origin).

Kind: silent wrong answer.

Status: F and the F synthesis reproduced (`R0*gamma*I`: peak 1,278 at `unit='day'`, 83 at `unit='week'`); not re-run here, since the core-language register owns it. In this area it means `to_code()` of the flepiMoP import writes `Ro = ss.par(ss.perday(2.5))`.

Proposal: whatever the core fix is (only the outermost bare factor carries the unit), let `Source.add_par` mark a parameter dimensionless and let importers that know a type (Atomica durations) pass it.

Cost and breaks: as in the core register.

### R13. Two calibration systems; `sim.calibrate()` refuses every class-based sim

Raised by: F 08§1, §7, c1, bug 6; O results 1; F2 08 f6 and "other" (top-level exports); F README §1; O README §1.

Kind: first-week friction.

Status: reproduced by F and O: `ss.Sim(diseases=ss.SIR(...), ...).calibrate()` raises `sim.new() is only available for sims with declared states and transitions`; `ss.Calibration` on a declared sim without `build_fn` fails with `'NoneType' object is not callable` once per trial. `calibration.py:26-1172` (about 1,150 lines) is the Starsim 3 half and shares nothing with `fit()`; `docs/user_guide/workflows_calibration.qmd` and `workflows_sir_calibration.qmd` teach only it.

Proposal (agreed by F, O, F2): one entry point, `sim.calibrate()`, for every sim. Bounds are an `ss.par` wherever a value goes, including the pars of class-based modules (O: a generic walk over `mod.pars`; this also comes free once `ss.SIR`/`SIS`/`SEIR` are redeclared, which every synthesis recommends). One escape hatch for values that are not pars, `build=` (a function `(sim, pars) -> sim`, the old `build_fn`), plus the existing callable `gof=`. `base.new()` falls back to `sc.dcp(sim)` for an undeclared sim (F: the only blocker, `calibration.py:2130`). Add `ss.binomial(n=)`/`ss.betabinom` to `logpdf()` (about 30 lines) to cover prevalence surveys and `BetaBinomial`. Delete `Calibration`, `CalibComponent`, the five component classes, and the three conform helpers.

```python
# before (Starsim 3, still the documented way)
calib = ss.Calibration(sim=sim, calib_pars=dict(beta=dict(low=0.01, high=0.3, guess=0.1)), build_fn=build_sim,
                       components=[ss.Normal(name='prev', expected=df, extract_fn=..., conform='prevalent', sigma2=0.05)], total_trials=100)
calib.calibrate()
# after, for ss.SIR too
sim = ss.Sim(diseases=ss.SIR(beta=ss.par(0.1, bounds=[0.01, 0.3])), networks='random', data=df)
fit = sim.calibrate(trials=100)
```

Cost: about 1,100 lines leave; a migration page. Breaks: every Starsim 3 calibration script and any downstream wrapper (STIsim, HPVsim to be checked); Optuna's SQLite multi-worker mode (`run_workers`) is lost unless `fit(method='optuna', n_workers=)` grows it.

### R14. Fitted values and fittable parameters: several natural spellings fail

Raised by: F 08§2 and bug 1 (`pars=fit.pars` with `dur`; also F 11§2); O results 8 and C F16 (`ss.RandomNet(n_contacts=ss.par(...))`); F 08§5 and bugs 4–5 (product of parameters, parameter inside a distribution); F2 08 "other" and O results 19 (an anonymous `ss.par` in a likelihood or outcome weight); new here (typed bounds).

Kind: crash / first-week friction.

Status: re-ran (`c15_parsrt.py`, `c14_misc.py`). `ss.Sim(sir, pars=fit.pars)` with a fitted parameter named `dur` raises `KeyNotFoundError: Key(s) ['beta'] not found; available keys are ['label', 'verbose', ...]`; renamed `recovery_time` it round-trips with a max difference of 0.0. `likelihood=ss.nbinom(n=ss.par(10, bounds=[1, 100]))`, the design's own snippet, raises `Could not find a name for the parameter 10 used in "cases"; give it one with ss.par(..., name="beta")` (and suggests `beta`). New: `ss.par(ss.days(10), bounds=[ss.days(3), ss.days(30)])`, the obvious way to bound a typed duration, crashes `calibrate()` with `ValueError: setting an array element with a sequence` (`calibration.py:2170`); the spec's form is bare bounds (`04-parameters.md:34`). C and O reproduced the `RandomNet` failure (`Updating dist from constant to par is not supported`).

Proposal: (a) give model values their own path so nothing is sniffed: F's option (b), document `fit.sim.new(...)` as the way to build the calibrated sim with changes (already works, breaks nothing), and, with the objects register, stop `ss.Sim(pars=)` from guessing by key overlap. (b) Name an anonymous `ss.par` from its slot (`cases_n`, `cases_weight`), as `fit()` already does for externals. (c) Accept typed bounds (convert to the value's unit) or raise a sentence that says "bounds are in the value's unit". (d) One parameter registry with qualified paths (`sir.beta`, `randomnet.n_contacts`) used by discovery, overrides, fitting, hashing, and serialization (C F16), so network and delivery parameters can take bounds; until then, coded errors for the product and inside-a-distribution cases saying they are not supported.

Cost: small for (b)–(c), moderate for (d). Breaks: nothing (a) as documented.

### R15. `sim.save()` crashes for every declared model after a run

Raised by: O surface 7.

Kind: crash (first week).

Status: reproduced (`c17_save.py`, and the pickle column of `c7_roundtrip.py`): `sim.save()` after `run()` raises `TypeError: 'Shrunk' object is not iterable` under `ode` and `Module sir did not successfully shrink: 295 KB > 91 KB` under `abm`, for all ten declared models tried; a Starsim 3 sim saves. `sim.save(shrink=False)` and `ss.save(path, sim)` work (13 KB and 31 KB, results equal after `ss.load`).

Proposal: fix `Loop.shrink` (it shrinks `funcs`, which `plan` then iterates) and give `Disease`/engines a `shrink()` that drops engine arrays; add a save/load test per solver.

Cost: small. Breaks: nothing.

### R16. Attaching data: near-misses get the wrong advice, misalignment is found late, and stratified data is not drawn

Raised by: O results 15 (`Date`/`week` columns); F 05§8, bugs 6 and 8 (`E704` at attach, stratified data not plotted, two matchers); F2 08 "other" (data points over the per-step flow); F 08 spec (seven time-column names vs four in the spec).

Kind: first-week friction.

Status: re-ran (`c14_misc.py`). A column `Date` raises `E705 UnknownDataColumn: the data column "Date" matches no result. The closest results are cases, n_S, n_R. Rename it with ss.data(..., aliases=dict(result='Date'))`, which treats the time column as a mistyped result and writes a placeholder alias. Data from 2023 on a 2020 sim runs without complaint; only `ss.compare()` raises `E704`.

Proposal: one matcher (`ss.data.match`) used by `init_data`, `compare`, and `plot`; raise `E704` at `init_data`; time columns matched case-insensitively, with `week`/`month` added, and "no time column: expected one of ..." said first; alias suggestions written out (`aliases=dict(cases='case')`); stratified data drawn on its result's panel; when data spacing exceeds `dt`, draw the model as `compare()` sees it (or the data per step) so weekly points do not float seven times above a daily line.

Cost: removes code (`sim.py:2432`'s matcher, `utils.py:578-620`'s second time parser). Breaks: nothing.

### R17. Specified in this area and absent

Raised by: F 05§10, §11, bug 4, spec; F 08§6, spec; O results 7, 8, 10; F2 08 "other"; C F16 (staged fitting); F README §7, O README §6.

Kind: spec drift (one is a silent empty result).

Status: each reproduced by two or more reviews. `analyzers='infection_log'` on a declared disease switches to agents and records 0 rows, with or without a network (F, O; 297 and 495 infections). `ss.Sim(results=[...])` raises `KeyNotFoundError`. Sim-level `ss.outcome` (`ss.Sim([flu, rsv], ili=...)`) raises. `method=[...]` raises `'list' object has no attribute 'lower'`. `sim.sensitivity()` does not exist. `describe()` has no CRN section. A result read during the run has no "so far" (`res.n_I[-3:]` returns preallocated zeros, F). None of these is in `implementation.md` or `deferred.md` except staged fitting in a parenthesis.

Proposal: implement the infection log for declared agent diseases (it is the one that returns a plausible empty table) or make asking for it an error; strike `results=`, sim-level outcomes, staged fitting, and `sensitivity()` from the spec and the count until a vignette needs them (`sensitivity` is a recipe: `ss.parallel(sim, pars=lhs)` plus a correlation); add `res.so_far` (a view `values[:sim.ti+1]`); list whatever stays unbuilt in `deferred.md`.

Cost: listing is free; the log is moderate. Breaks: nothing.

### R18. One `ss.Result` with named axes and a kind, instead of three results models

Raised by: F 05§3, §9, c2, c3; F2 08 f3; O results 2, 3, 4; C F11 and interfaces (coordinate-aware results with an explicit reduction algebra).

Kind: surface (the structural fix behind R1–R3).

Status: design; the symptoms are verified above. Today `Result` is an ndarray subclass with Starsim 3 `columns`/`low`/`high` and v4 `strata`/`strats`/`dims`/`denom` set from outside by the solver (`getattr(self, 'strata', None)` nine times in `results.py`); `by()` returns a 2D `Result` that cannot be selected again and renames level `low` to `risk_low`; a ratio written in Python after the run loses its groups; `__array_ufunc__` decides group survival in one 400-character condition (`results.py:217`).

Proposal (agreed): `values` of shape `(run?, time, *groups)` with named `dims` and labeled `coords`, an optional `denom` of the same shape, and R1's `kind`; `select()` indexes coordinates and sums the rest; `by()`, `select()`, `quantiles()`, `resample()` all return a `Result`; `run`, `scenario`, and swept parameters are dims; Starsim 3 multi-strain `columns` become `dims=['strain']` with a `__getitem__` shim; arithmetic is broadcasting plus one rule (sums and scalar products keep groups; a ratio keeps numerator and denominator). No xarray dependency (F). `to_df()` writes wide columns `name[age=0-14]`, which `ss.data.widen` already reads.

Cost: medium-high; touches `reduce`, `combine`, `flatten`, `to_df`, `plot`, and the solvers' `store`. Breaks: `res['alpha']` on multi-strain Starsim 3 results (shimmed).

### R19. Four names for each flow

Raised by: F 05§1; F2 01§1 and F2 08 "other"; F README §6 table. F2 02§8 prefers `new_<transition>` (see disagreements).

Kind: surface.

Status: F verified under every solver: a plain SIR has 10 results of which 4 are copies (`infection == new_infections`, `cum_infection == cum_infections`, `solvers.py:1223-1225`, `1436-1437`, `1519-1520`), stored with different flags (R1).

Proposal: the transition's name only (`infection`, `recovery`), with `res.infection.cum` a property computed on request; `new_infections`, `cum_infections`, and the automatic `prevalence` only on library diseases; a fixed alias so a data column `new_infections` on a declared model still matches the sum of its transmission flows.

```python
# before: n_S n_I n_R infection cum_infection recovery cum_recovery new_infections cum_infections prevalence
# after:  n_S n_I n_R infection recovery          (res.infection.cum when wanted)
```

Cost: small. Breaks: Starsim 3-style code reading `new_infections` from a declared disease (one alias line in the migration page).

### R20. Six ways to run many sims

Raised by: F2 08 f8; O results 2; F 05 c6 (the 50-line `MultiSim.__init__`).

Kind: surface.

Status: F2 and O verified: `ss.MultiSim(sim, n_runs=).run()`, `ss.parallel(sim, n_runs=)`, `sim.run(n_runs=)`, `ss.multi_run(sim, n_runs=)`, `ss.parallel(sim, pars=...)`, `multi_run(iterpars=)`, with results built only on some paths (R2).

Proposal: `ss.parallel(*sims, n_runs=, pars=, **scenarios)` as the one function; `ss.MultiSim` stays as the class it returns but users do not construct it; `iterpars` folds into `pars`; `multi_run` goes. Whether `sim.run(n_runs=)` stays is a disagreement (below).

Cost: small after R2. Breaks: `ss.multi_run` and `iterpars` callers.

### R21. The model as a file: ten entry points for two jobs

Raised by: F 09§1; O surface 8; F2 08 f10 and "other" (`to_code`/`from_spec` serializers); C F14 and F22; O README §4 table; F README §6 table.

Kind: surface.

Status: re-ran the round-trip matrix (`c7_roundtrip.py`, eleven models). `from_spec` rebuilt 6 of 11 with the same hash and refused 5 loudly (`ss.modify`, `ss.vaccinate`, `RandomNet`, a function, `ss.by`). `to_code` rebuilt 8 with the same hash (including `ss.modify`, which `from_spec` refuses), but 2 of those 8 are wrong (a function: builds, crashes at run; data: dropped) and 2 more build different models with only a comment (`RandomNet`, `ss.vaccinate`); `ss.by` fails both; Starsim 3 `ss.SIR` fails both. The pickle failed for all ten declared models via `sim.save()` (R15) and worked via `ss.save()`. `disease.to_json()` returns the spec while `sim.to_json()` returns Starsim 3 results; `sim.to_spec(filename)` returns the filename. A spec labeled `lingua-franca/99.0` loads (C F22, F 09 spec). Note for F's proposal: the spec `Parser` cannot go with `from_spec`, because `interop/core.py:320` uses it to read other frameworks' expressions without running them.

Proposal (my merge; positions below): two files and one in-memory form. `ss.save('model.py', sim)` writes `to_code()` (the readable, diffable, shareable model, made honest by R10); `ss.save('run.sim', sim)` writes a pickle (the checkpoint, with results, made to work by R15); `sim.to_spec()` and `disease.to_spec()` return the normalized dict that `hash()`, `diff()`, and the importers use, and are not a file format. `ss.load('run.sim')` reads the pickle; a `.py` model is run or imported like any script. Delete `from_spec`, `ss.load('*.json')`, `ss.to_code`, `ss.interop.export` at top level, `joint_split`, and the schema-version story; rename `disease.to_json` to `to_spec`; leave `sim.to_json` as the results export or fold it into `to_df()`.

```python
# before
sim.to_spec('m.json'); sir.to_json('sir.json'); sim.to_code('m.py'); ss.save('m.xml', sim, format='sbml'); ss.interop.export(sim, 'odin'); sim.save('m.sim')
# after
ss.save('model.py', sim)      # the model, as code
ss.save('run.sim', sim)       # the run, as a checkpoint
spec = sim.to_spec()          # in memory, for hash/diff/tools
```

Cost: about 200 lines of reader go (`quantity`, `disease`, `strat_kwargs`, `from_spec`; the `Parser` stays). Breaks: `ss.load('sim.json')`, used by two tests (`test_named.py:38`, `:112`) and nothing in the vignettes or docs.

### R22. Eleven framework importers live in the core package

Raised by: F 10§1, §5, c2, c5; O surface 13; F2 08 f9; all three syntheses.

Kind: surface / codebase.

Status: agreed and verified by three reviews: about 5,000–5,700 lines and five `dump_*` scripts shipped in the wheel; five of eleven importers need the other framework (or camdl's `camdlc` binary, or R) anyway; 7–8 of 10 interop test files skip unless `/home/cliffk/idm/lingua_franca_design` exists. Import cost is negligible (3 ms), so this is purely maintenance and promise.

Proposal: keep in `starsim`: `interop/core.py` (`Source`, `build`, `Rates.classify`, the printed reading of each rate), `to_code`, `ss.diff`, and the SBML importer (a standard, file-only, 251 lines). Move the other ten importers, the dump scripts, their tests and fixtures, and the `import-model` reference guides to `starsim-interop`, registered through `ss.interop.formats` with per-format `sniff()` functions; without it, `ss.load(..., format='odin')` says `pip install starsim-interop`. Fix the hard-coded paths either way.

Cost: a second package to release. Breaks: `ss.load(file, format=)` needs one more install for ten formats.

### R23. Exporters: two Mermaid writers, and lossy SBML and odin exports

Raised by: F 10§4, c3, c4; O surface 13 (Mermaid is what `diagram()` should use); F2 08 f9 (SBML out stays, odin export moves).

Kind: surface.

Status: F measured an age-stratified SIR with a contact matrix exported to odin and re-imported at a peak of 175 against 9,387 (max difference 9,211); the loss is listed by `W1401`, so it is not silent. Not re-run here. `Graph.diagram('mermaid')` and `export.to_mermaid` are two writers, the first poorer.

Proposal: one Mermaid writer behind `sir.diagram('mermaid')`; move the SBML and odin exporters with their importers (R22 keeps the SBML importer; its exporter is only useful for unstratified models, which is where `sir.equations()` already serves), and with them `export.flatten()`/`rename` and `W1401` unless R10 reuses it for `to_code()`.

Cost: removes about 250 lines from core. Breaks: `ss.save(..., format='sbml'|'odin')` without the companion package.

### R24. `to_code()` output after an import: right for small models, unusable for data-heavy ones

Raised by: F 10§2, §6, bugs 2–4; F 09§3, §7 (dates with dots, guessed split hard-coded, integers as floats).

Kind: first-week friction (for importers).

Status: F reproduced on five imports: EMULSION 37 lines against the vignette's 11 (mostly docstring); SimInf writes an 8,032-character and a 48,463-character line and turns the event register into a comment; Atomica writes `(contacts - contacts)` from its `floor()` rewrite; dates come out `'2020.02.01'`; `sim.new()` drops `sim.imported`. I saw the dotted dates in `c8_tocode.py`'s output (`start='2020.01.01'`).

Proposal: arrays longer than about ten numbers go to a sidecar file the generated code reads; round to the source's precision; ISO dates; inflows/outflows named `birth_<state>`/`death_<state>`; omit guessed `split`/`kind` (give `Strat.spec` a `guessed` list); provenance as one header line plus the `W1601` list; carry `imported` through `sim.new()`; fix Atomica's dead term.

Cost: moderate, mostly in `spec.py` and `core.py`. Breaks: nothing.

### R25. The fitting method rule: keep it, fix the spec, and reconsider two defaults

Raised by: F 08§3, §9, spec; O results 11; F2 08 f7 and "other" (`n_runs` default); C interfaces (best-practices still recommends ABC-SMC).

Kind: spec drift / first-week friction.

Status: all agree the code's rule (`default_method()`, `calibration.py:2072-2092`) is predictable and prints its reason, and that the spec's table (ABC-SMC, timing-based "cheap enough", simplex first) and both transcripts describe something else. Two reviews timed the ODE case: L-BFGS-B with JAX 1.15 s and 1.56 s against Nelder-Mead 0.15 s and 0.16 s for one or two parameters (`implementation.md:221` records the same). The default `n_runs` for a stochastic fit is 1 (`calibration.py:2178`), so the fit is tuned to one realization under a fixed seed.

Proposal: rewrite `13-inference.md`'s table and transcripts to the code's rule; adopt O's table (ODE without likelihood: Nelder-Mead up to four parameters, else L-BFGS-B; ODE with likelihood: NUTS; stochastic compartmental: Optuna or PMCMC; agents: Optuna); default `n_runs=4` for stochastic sims, printed, under CRN (positions below).

Cost: a spec edit and two lines of code. Breaks: default fits become four times slower for stochastic sims (one keyword to undo).

### R26. The `Fit` object and the comparison names

Raised by: O results 16; O surface 19 (five names for "compare"); F 08 spec (`fit.pars` dict vs `fit.table`).

Kind: surface.

Status: design. `Fit` has 13 attributes plus `to_df()` that returns `trials`; with a likelihood, the `mismatch` column is a negative log-likelihood. `ss.diff`, `ss.diff_sims`, `msim.diff`, `ss.compare`, `ss.compare_models` are five comparison names.

Proposal: `fit.pars`, `fit.sim`, `fit.table`, `fit.samples`, `fit.converged`/`fit.why` (R5), `summarize()`/`plot()`/`to_df()`, with `before`, `band`, `divergences`, `acceptance` under `fit.diagnostics`; rename the column `objective` and say in `measure` what it is. Fold `ss.diff_sims` into `ss.diff`; keep `ss.compare` for data-vs-model (it is what `calibrate()` documents).

Cost: small. Breaks: names only weeks old.

### R27. `ss.Samples` and `ss.Dataset` are disconnected Starsim 3 code

Raised by: F 05§6 (summary), c5; O results 17; F2 08 "other".

Kind: surface.

Status: agreed: `samples.py` (432 lines) has no v4 caller, one test, and one user-guide page that still says `cvv.Samples`.

Proposal: move to `starsim.library` or a compatibility module; if saving sweeps is wanted, `ss.save('sweep.msim', msim)` of R2's results with `run_pars` and hashes as the index.

Cost: none. Breaks: HPVsim/FPsim-style users need a library import.

### R28. `duration=` on an outcome is exponential while `delay=` is exact

Raised by: F 05§5 (a known issue in `implementation.md`).

Kind: first-week friction.

Status: F measured `duration=ss.days(7)` as an implied mean stay of 7.42 with an exponential tail against 6.98 for `ss.constant(7)`; not re-run.

Proposal: a bare duration on an outcome means a fixed stay, matching `delay=`; a distribution for anything else.

Cost: one clause in 09. Breaks: occupancy curves of models that relied on the exponential (three tests).

### R29. The model spec's content: a 36 KB network dump, a dead `functions` field, string leaks, and no version check

Raised by: F 09§2, §8, §9, spec; O surface 17, 11; F2 08 f5(b, c), "other" (`ss.diff` detail); C F22; F 10 spec 3 (contact matrix not in the spec).

Kind: spec drift.

Status: F and O measured `ss.contacts('Kenya')` as 36,270 of 38,496 characters, with `python_class`, `do_cache`, `ti_cache`, all hashed into `model`. C and F reproduced `lingua-franca/99.0` being accepted. F2 showed likelihoods and priors written as `str(dist)` (`'ss.nbinom(n=par(10, bounds=[1, 100]), p=0.5)'`). The documented nested-list grammar is not what the code writes (strings, with some lists left over: `"mean": ["par", "dur"]`).

Proposal: whichever saved form wins (R21), `to_spec()` feeds the hash, so it must be clean: networks write `{type, pars}` or a data reference only; functions carry source; likelihoods and priors go through `quantity_json`; dates ISO; `14-model-spec.md` describes strings and drops the nested-list table, the `fit` field, normalization step 4, and the `sha256` file form. If `from_spec` stays, F2's hoisting rule (every non-scalar factor in a rate becomes a named parameter whose value is typed JSON) closes the grammar, and the version check rejects unknown majors.

Cost: about 40 lines for networks, 15 for functions. Breaks: every hash once (do it with R8).

### R30. Inference and results spec drift beyond R17 and R25

Raised by: F 08 spec, F 05 spec, F2 08 "other" (likelihood in two key spaces), O results 18 (CRN overstated).

Kind: spec drift.

Status: F listed these from the code; each is a sentence. `17-vocabulary.md:110` calls `ss.Calibration` the `fit` object; `fit.pars` is a dict and the table is `fit.table`; `n_runs` averages goodness of fit across runs, not the goodness of fit of the mean trajectory; B1's answers (NAE default, log-scale rule, 25% holdout) are implemented and can be marked settled; module results are the same under every solver but sim-level `n_alive` exists only as agents; `ss.data` works with Starsim 3 sims (worth saying, since it is the migration path); `likelihood=` is keyed one way on the outcome and three ways at `calibrate()`; CRN pairing is strong for small perturbations and weak once trajectories diverge (O measured), so the docs should not oversell it.

Proposal: one editing pass on 09 and 13 with these, and one key space for likelihoods (result name, `name[dim=level]` for strata).

Cost: an afternoon. Breaks: nothing.

### R31. Interop spec drift

Raised by: F 10 spec 1–3.

Kind: spec drift.

Status: F listed: `to_code`, `sim.imported`, `Source`/`build`, and per-format keywords are unspecified; 17 lists `ss.export` as removed; 16's examples return a disease where the code (rightly) returns a sim; the table of importers names ones not built (Starsim 3, epipack) and omits ones that were; 18 claims the lossy list is checked against the file and no test does it; four of the fourteen "gaps found while writing the importers" are real holes in the language (a transition's flow in a rate, a one-off scheduled move, `substeps=` forced by one Erlang stage, the contact matrix not in the spec).

Proposal: spec `to_code`, `sim.imported`, `Source`/`build` in 16 and 17; make the rest private (or companion); `sim = ss.load(...)`; correct the table; soften 18 or write the test; carry the four holes to the language register.

Cost: a spec edit. Breaks: nothing.

### R32. `fit()` is one 300-line function; likelihoods, priors, and alignment are written twice

Raised by: F 08 c2–c5; O results 12; F2 08 f7; F README §8.

Kind: codebase.

Status: agreed: `fit()` (`calibration.py:2095-2398`) has 10–14 closures, three coordinate systems, and seven method branches inline; `logpdf`/`logprior` (NumPy) are mirrored in `gradients.Density` (JAX); `compare()` and `lineup()` re-derive the same alignment. I measured what the NUTS Python-callback fallback costs and when it is reached (`c19_nuts.py`): an `ss.modify` on a fitted `beta` makes `Density` raise `Unsupported`, and NUTS (100 + 100 draws, one chain) takes 63.4 s instead of 10.6 s.

Proposal: a `Problem` (fitted pars, bounds, transforms, `objective(x)`, `logpost(z)`) built once, and a registry `methods = dict(optuna=..., nelder_mead=..., nuts=..., pmcmc=...)` of `(problem, x0, **options) -> MethodResult(best, converged, why, samples)` (which also gives R6's unknown-name check and R5's `converged`); `logpdf(dist, observed, model, xp=np)` and `logprior` written once against an array namespace; `compare()` as `W @ values` from `lineup()`. Keep the NUTS fallback until `Density` covers modifiers (positions below). Un-skip the PMCMC test with a 20-second budget; add one test of the hand-written R-hat/ESS against a known chain.

Cost: a refactor of about 300 lines with the existing tests as the net; O estimates `fit()` at about 150 lines after. Breaks: nothing user-visible.

### R33. Results code is spread over six files, with four `plot`s and two resamplers

Raised by: F 05 c1–c4, c7; F2 08 f4.

Kind: codebase.

Status: F mapped it: `results.py`, `solvers.py` (creation and the outside-set attributes), `run.py` (`MultiResult` mirrors `select`/`by`/`plot`), `calibration.py` (`ss.data`, `compare`, matcher), `sim.py` (its own data matcher and `plot`), `utils.py` (a second time parser). `ss.data` is a lowercase class whose six static methods take `sim` and never use `self`, and it shadows `fit()`'s own `data` argument.

Proposal: `results.py` owns `Result`, `Results`, `data`, `compare`, and the one matcher; `Sim.plot` and `MultiSim.plot` become `self.results.plot(data=self.data, **kw)` after R2 and R18; `ss.data(...)` stays as the user-facing constructor with its helpers as module functions.

Cost: a move plus deletions. Breaks: nothing.

### R34. Interop internals duplicated across importers

Raised by: F 10 c1, c2, c5.

Kind: codebase.

Status: F counted: `siminf.py:706-727` reimplements `core.build()`; seven hand-rolled AST visitors; nine copies of the name walk where `core.names()` exists; eight `getattr(ss, f'{unit}s')(x)` where `ss.dur(x, unit=)` exists; identical unit tables in two importers; `flepimop.py` importing a helper from `emulsion.py`; a 310-line R tokenizer and evaluator in `siminf.py`.

Proposal: move `number()`, a `transform()` helper, the unit table, and `dur_in()` into `core.py`; delete `siminf.build`; replace the R parser with a `dump_siminf.R` (as EpiModel does). Most of this travels with R22.

Cost: about 450 lines saved. Breaks: SimInf import then needs R (its users have it).

## Where the reviews disagree

### The saved form of a model: `to_code()`, `ss.save` by extension, or a complete JSON spec

Positions. F 09§1: make `to_code()` the saved form; keep `to_spec()` only as the input to `hash()`/`diff()`; delete `from_spec`, `ss.load('sim.json')`, the `Parser`, `joint_split`, and schema versioning (confidence medium on removing outright, "Cliff may have a tool consumer in mind"). O surface 8: one verb pair, `ss.save(path, obj)`/`ss.load(path)` with the format from the extension (`.json` spec, `.py` code, `.xml` SBML, `.sim` pickle), `to_spec()` in memory, and (surface 6) a generic `{type, pars}` module spec so `from_spec` can rebuild networks and demographics. F2 08 f5 and its questions: keep string rates, close the grammar by the hoisting rule so `from_spec` and `to_code` share one table, and asks which is primary. C F14: separate a portable declarative specification from a runtime checkpoint, and return an explicit unsupported-elements manifest for any partial export.

Check (`c7_roundtrip.py`, eleven models): `to_code` reproduces more models (8 of 11 build, including `ss.modify`, which `from_spec` refuses) but fails silently in four of them (function, data, `RandomNet`, `ss.vaccinate`); `from_spec` reproduces 6 and fails loudly on 5. Neither handles Starsim 3 modules. The pickle is the only form that carries results and data, and `sim.save()` is broken for declared models (R15). Fact check on F: the `Parser` cannot be deleted, since `interop/core.py:320` uses it; deleting `from_spec` saves about 200 lines, not 270.

Recommendation: C's split, with F's choice of the declarative form and O's verbs. The user needs two files, not four: the model as something to read, edit, and share, and the run as a checkpoint. A Python file is the readable model for this audience (the docs, vignettes, and importers already hand people `.py`), it is git-diffable, and `to_code()` already covers more than `from_spec`. A JSON format earns its keep only for a non-Python consumer, and none exists in the corpus; keeping it means maintaining a second reader and a security-sensitive grammar to the same coverage. So: `ss.save('model.py', sim)` and `ss.save('run.sim', sim)`, `ss.load('run.sim')`, `to_spec()` in memory, `from_spec` deleted (kept if Cliff names a tool consumer, in which case F2's hoisting rule plus O's `{type, pars}` spec is the way to make it complete). Whichever is chosen, R10's honesty rules (warn on anything not written, never emit broken code) apply to it.

### Which interop pieces stay in the core

Positions. All three that covered it agree the framework importers move to a companion package and `interop/core.py` stays. F 10§1, §4: also move SBML and odin exporters out of the public API, keep Mermaid only, perhaps keep the EMULSION and SBML importers in core. F2 08 f9: core keeps `Source`, `build`, `sniff`, load/save dispatch, SBML in and out, Mermaid out; the odin exporter moves. O surface 13: core keeps the spec, `to_code`, `ss.diff`, Mermaid, and "possibly" SBML in and out.

Recommendation: keep the SBML importer (a standard, file-only, the conformance level in 18 names it), Mermaid behind `diagram()`, and `core.py`; move every exporter but Mermaid. The SBML exporter is exact only for unstratified models without networks, outcomes, interventions, or functions (F), which are the models nobody needs to export, so it does not justify core space; it can live beside the odin importer whose round trip tests it. Not checked by me: F's 9,211 odin round-trip difference.

### Dropping `sim.run(n_runs=)`

Positions. O results 2: drop it, since a method whose return type depends on an argument is a trap and `ss.parallel(sim, n_runs=)` is one call. F2 08 f8: keep `sim.run(n_runs=)` and `ss.parallel` as the two spellings. F 05§2 keeps the spec's spellings and fixes the object underneath.

Recommendation: drop it. `sim.run()` returns the sim; `sim.run(n_runs=10)` returns a different class with a different `len`, `plot`, and `summarize`, and the two have to be kept identical to `ss.parallel` forever. `ss.parallel(sim, n_runs=10)` costs eight characters more and is the same function that already takes scenarios and sweeps. If Cliff keeps it for brevity, it must be a literal one-line call to `ss.parallel`.

### `every=` on an outcome

Positions. F 05§4: drop it or make it a step function. O results "keep": lists `every` among the good options of `ss.outcome`.

Check: `c5_every.py`, the comb compared with daily data (R7).

Recommendation: drop it. The data path already sums flows over each data interval, so `every=` adds only a way to get a wrong comparison and a series that plots as a comb; the one unique use, a reporting-day function, has no test.

### One flow name: `infection` or `new_infections`

Positions. F 05§1 and F2 01§1: the transition's name (it matches the model text, `ss.modify` targets, `ss.outcome('infection')`, and data columns), with `new_infections` only on library diseases. F2 02§8: `new_<transition>` reads better and matches Starsim 3. F README and F2 README both report the split.

Recommendation: the transition's name. It is the one name the user wrote; every v4 test and example reads it (F: `tests/test_transitions.py:552`, `:609`, `:713`); and R1's fix plus a fixed alias table keeps Starsim 3 data files matching without storing a second series.

### The default method for small ODE fits, and `n_runs` for stochastic fits

Positions. O results 11: Nelder-Mead for ODE fits with four or fewer parameters; `n_runs=4` by default for stochastic sims. F 08§3: low confidence that a parameter-count threshold is worth an extra rule; the code's mean-of-goodness-of-fit is arguably better than the spec's goodness-of-fit-of-the-mean. F2 08 "other": default `n_runs` to about 4 for `solver != 'ode'` and print it. C: prefer one simple optimization path and one explicit posterior path.

Recommendation: adopt both of O's defaults. The 10x speed difference falls on exactly the first-week case (one to three parameters), and a fit to one stochastic realization with a fixed seed is overfit by construction (R4 shows how its score then misleads). Both are printed in the "Fitting by ..." line, so they stay visible and one keyword undoes them.

### Staged fitting and `sim.sensitivity()`

Positions. F 08§6: strike both from the spec (users compose them). O results 9, 10: implement staged fitting once `converged` exists (about ten lines, `optuna → nuts` is the worthwhile pair); `sensitivity()` either way. C F16: prioritize diagnostics over fitting orchestration.

Recommendation: strike both now, build R5 first, and revisit staged fitting only if a vignette needs it. Simplicity for the user favors one method per call with an honest `converged`; a staged pipeline is a second concept with its own stop rule (`W1303`).

### NUTS: one sampler or two

Positions. F 08 c3: drop the Python-callback `nuts()` and its fallback; when `Density` cannot cover a model, say so and suggest `l-bfgs-b`. F2 08 "good": the callback fallback and the three-draw consistency check are a cheap, good guard. O results 12: write likelihoods once against an array namespace, which removes the drift that the check guards against.

Check: `c19_nuts.py`: an `ss.modify` on a fitted parameter, a common pattern, reaches the fallback; it ran in 63.4 s against 10.6 s, and finished.

Recommendation: do O's unification now (one `logpdf`/`logprior`), keep the fallback until `Density` covers modifiers of fitted parameters, then delete it and the consistency check together. Dropping it today would remove NUTS from any model with an intervention on a fitted rate.

### Calibration escape hatch: `set=`, `build=`, or bounds by name

Positions. F 08§1: `pars=dict(beta=[0.01, 0.3])` (bounds by name for parameters that are not `ss.par`), `set=` (how to put values into a sim with no graph), `gof=`. O results 1: bounds are `ss.par` wherever a value goes, including class-based modules' pars; `build=` and `gof=`.

Recommendation: O's form. One rule for where bounds go ("on the parameter") is simpler than two; bounds by name in `calibrate()` duplicates it. Call the hatch `build=`, since it is the old `build_fn` with the same signature, which makes the migration one rename.

## Keep

- The time-point convention (`n_R[ti] - n_R[ti-1] == recovery[ti]`), the same names and dtypes under every solver (F, F2, O).
- `select()`/`by()` semantics on single-run results, including ratios that sum numerator and denominator (F, O, C).
- `ss.data` by column name, `aliases=`, `ignore_extra=`, long-form strata as `cases[age=0-14]`, and `E705` with the closest names (all four).
- `compare()`'s rule: a flow summed over each data interval, a stock read at the data time (F, F2, O); `lineup()` as a matrix so the same rule runs in JAX (F2).
- `ss.outcome` as a weighted sum of flows with `delay=`, chaining, `E702`/`E703`, and weights as `ss.par`/`ss.by` (F, O).
- `sim.calibrate()` from bounds on `ss.par`, finding parameters inside `init=`, `ss.seasonal`, likelihoods, and shared across modules (F, F2, O).
- `default_method()` with its printed reason (F, F2, O; C on keeping one simple path).
- `sim.restart()` and `restart_runs()` as the fast path for compartmental fits and sweeps (F, F2, O).
- CRN by default across evaluations, and per-distribution streams that agree across the three spellings (F, O).
- The `Fit` object's core (`fit.sim` run, `fit.before`, `fit.pars`, `fit.table`, `fit.plot()`), and `fit.sim.new()` keeping the fitted values (F, O).
- `W1302` identifiability notes from model structure and from the posterior (F, F2, O).
- NUTS through BlackJAX with the log-density as one JAX function; the bootstrap `ParticleFilter` that steps the kernels (F, F2, O).
- `ss.compare_models` by held-out prediction with `folds=` (F, F2), once R9 is fixed.
- Lazy imports of JAX, BlackJAX, Optuna with install hints (F).
- `msim.extinction()` (F, F2), and `ss.parallel(baseline=..., vaccine=...)` naming by keyword with `pars=` sweeps and `msim.diff('baseline')` paired by run (F).
- `sim.hash()` as four short parts plus the seed, unit-normalized, unchanged by labels and docs (F, O, C on keeping component hashes).
- `to_code()`: idiomatic operator-form code that rebuilds the same hash for declared models, and its footer as the honest place for what it cannot write (F, F2, O).
- `ss.diff` in the model's own terms, transitions matched by endpoints (F, F2, O).
- `interop/core.py`: a neutral `Source`, one `Rates.classify` for the five rules, one `build()`, the printed reading of each rate, `W1601` with what was not translated, and parsing that never executes the source (all four that covered interop).
- The dump-script pattern for frameworks whose models are code (F).

## Decisions for Cliff

1. The saved form of a model: `to_code()` `.py` plus a pickle checkpoint, with the JSON spec in memory only (F, and my merge with C's split and O's verbs); or a complete JSON spec as a file (needs F2's hoisting rule and O's generic module spec). Is there a non-Python consumer of the spec in mind? (Reviewers: `.py` primary; F2 asks.)
2. Delete `from_spec` and `ss.load('*.json')`? (F: yes; O: keep as `.json` load; recommendation: yes unless 1 says otherwise. The `Parser` stays for interop either way.)
3. Move the ten framework importers to `starsim-interop`; keep `core.py`, the SBML importer, and Mermaid in core? (All three: move; they differ on SBML export, which I would move.)
4. Drop the SBML and odin exporters from the public API? (F: yes; F2: keep SBML out; recommendation: move both with the companion package.)
5. Drop Starsim 3's `Calibration`/`CalibComponent` and the five likelihood classes, or keep them in a legacy namespace for a release? (F, O, F2: drop, with a migration page and `ss.binomial`/`ss.betabinom` added; check STIsim/HPVsim first.)
6. Escape hatch name for `calibrate()` on sims whose values are not pars: `build=` (O) or `set=` (F)? (Recommendation: `build=`.)
7. One flow result name: `infection` (F, F2 01) or `new_infections` (F2 02)? (Recommendation: the transition's name.)
8. Drop `sim.run(n_runs=)` in favor of `ss.parallel(sim, n_runs=)`? (O: drop; F2: keep; recommendation: drop.)
9. Drop `every=` on outcomes? (F: drop or step function; O listed it as good; recommendation: drop.)
10. Make `duration=` on an outcome a fixed stay, like `delay=`? (F: yes.)
11. Defaults for fitting: Nelder-Mead for ODE fits with up to four parameters, and `n_runs=4` for stochastic fits? (O, F2: yes; F: unsure about the threshold.)
12. Strike staged fitting and `sim.sensitivity()` from the spec? (F: strike; O: implement staged after `converged`; C: diagnostics first; recommendation: strike.)
13. Make priors mean something for optimizers (MAP when every series has a likelihood), or move priors to `calibrate(priors=)` and out of the hash? (F leans to keeping priors on the parameter, excluded from the hash.)
14. Keep the NUTS Python-callback fallback until `Density` covers modifiers? (F: drop now; F2: keep; recommendation: keep for now, measured at 6x slower but working.)
15. Move `ss.Samples`/`ss.Dataset` out of core? (F, O, F2: yes.)
16. Implement the infection log for declared agent diseases, or make asking for it an error? (F, O: one or the other; not left returning an empty table.)
17. Add a `data` part to `sim.hash()` (a fingerprint of attached data), as distinct from `model`/`pars`/`config`/`scenario`? (C asks for data in the run identity; recommendation: a fifth part, so model and data changes stay distinguishable.)
