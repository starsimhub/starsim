# Results, data, outcomes, and many runs

I reviewed `v4_design/design/09-results.md` in full, the results parts of `05-dimensions.md` (reading stratified results) and `13-inference.md` (data), the relevant rows of `17-vocabulary.md`, B1 and B8 of `19-open-questions.md`, and the departures and known issues in `v4_design/admin/implementation.md`. Code read line by line: `starsim/results.py` (778 lines: `Result`, `Results`, the strata machinery at 144-242, the 2D `columns` machinery at 244-265, `resample`/`annualize`/`to_df`/`plot`), `starsim/run.py` (1228 lines: `MultiSim`, `MultiResult`, `MultiResults`, `restart_runs`, `parallel`), `starsim/samples.py` (432), `starsim/analyzers.py` (191), `ss.outcome` and `FlowSum` in `starsim/transitions.py:918-1017`, `ss.data`, `compare`, and `data_times` in `starsim/calibration.py:1174-1340`, the engine's results code in `starsim/solvers.py:1204-1565`, and `sim.init_data`, `sim.run(n_runs=)`, `sim.summarize`, and `sim.plot` in `starsim/sim.py`. I ran eight scripts in `/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/results/` (`t1_names.py` through `t8_misc.py`), each a small SIR under the ODE, binomial, Gillespie, and agent solvers, with and without stratification, data, outcomes, and 3-5 runs; outputs quoted below are from those runs.

## Summary

1. **Make `msim.results.<module>.<result>` a real `ss.Result` with a leading run axis, and drop `MultiResult`, `MultiResults`, and `reduce()`'s copied results.** Today a MultiSim has three result representations that disagree (`MultiResult` is not a `Result`: no `to_df`, no arithmetic, no indexing; `reduce()` writes `low`/`high` into a fourth copy inside `base_sim`), `msim.plot()` after `reduce()` crashes for every declared model, and the restart path leaves a MultiSim with 1 sim but 5 runs so `summarize()` reports `std=0`. Breaking for Starsim 3 code that uses `msim.reduce()`/`msim.results.sis_prevalence`; the biggest single simplification in this area.
2. **One name per flow: drop `new_infections`/`cum_infections` from declared models (keep them only for library diseases), and drop `cum_<transition>` as a stored result in favor of `res.infection.cum` (a property).** A plain SIR today has 10 results of which 4 are duplicates (`infection`, `new_infections`, `cum_infection`, `cum_infections` are two series stored twice). Breaking only for data files written against Starsim 3 names on a declared model, which `aliases=` covers.
3. **Replace the `columns`/`col()`/`by_column()`/`stack()` 2D-result machinery and the `strata`/`strats`/`dims`/`denom` attributes with one `dims` + `coords` model on `Result`**, so that `by()` returns a result that can still be selected, a `by('risk')` column is not renamed `risk_low` because `low` is an attribute, and `MultiResult` needs no second copy of `select`/`by`. Breaking for `res['alpha']` on multi-strain Starsim 3 results; it is the cost of recommendation 1.
4. **Fix `every=` or remove it.** A weekly outcome is a comb of zeros that `compare()` lines up against daily data as 0, 0, 0, 0, 0, 0, 6.5, and `res.daily.resample('week')` bins by calendar week while `every=ss.weeks(1)` bins from the sim start, so the two "weekly cases" disagree (2.7 vs 6.5 in the first week). Since data comparison already sums flows over the data's intervals, `every=` buys only "reporting day" functions; make it store the weekly total on every step of the week (a step function) or drop it. Not breaking in practice.
5. **Make `duration=` on an outcome mean what `delay=` means: a fixed stay, with a distribution for anything else.** Today `delay=ss.days(7)` is an exact shift and `duration=ss.days(7)` is an exponential stay with mean 7 (measured 7.42 vs 6.98 for `ss.constant(7)`), in the same call. Breaking for models that relied on the exponential; one line in the spec.
6. **Move `samples.py` (`Dataset`, `Samples`) out of core.** It is untouched Starsim 3 code, used by one test and one user-guide page, and overlaps `MultiResults`/`msim.to_df()`. Not breaking if it goes to `starsim.library` or a separate package.
7. **Finish or cut three spec promises the code does not keep:** `ss.Sim(results=[...])` (not implemented, not in `deferred.md`), the transmission log for declared diseases (`analyzers='infection_log'` records 0 rows under `abm` with or without a network), and `msim.plot(lines=True)` (fails; only `MultiResult.plot(lines=True)` works). Either is fine; the spec and `implementation.md` should say which.

## Simpler for the user

### 1. Four names for one flow

The issue. A declared SIR produces, under every solver (`t1_names.py`):

```text
disease keys: ['n_S', 'n_I', 'n_R', 'infection', 'cum_infection', 'recovery', 'cum_recovery', 'new_infections', 'cum_infections', 'prevalence']
infection == new_infections: True
cum_infection == cum_infections: True
```

So the user who writes `infection = S >> I @ beta * I` gets `infection`, `new_infections`, `cum_infection`, and `cum_infections`, two of which are byte-identical copies of the others (`solvers.py:1223-1225`, `1436-1437`, `1519-1520`). `sim.summarize()` and `sim.to_df()` list all ten. The spec says the transition name is "the specified one and `new_infections` exists alongside it" (`09-results.md:34`) until the vignettes have been written both ways; the vignettes have now been written and the implementation has answered B1: the transition's name is what every test and example reads (`tests/test_transitions.py:552`, `:609`, `:713`), and `new_infections` is read nowhere in the v4 tests except as a Starsim 3 compatibility check.

Why it matters. Every result listing, `summarize()`, `to_df()`, `plot(show_skipped=True)`, and `msim.compare()` is 40% longer than it needs to be, and a newcomer has to learn that `cum_infection` and `cum_infections` are the same thing while `infection` and `infections` would not be. It also costs memory per stratum (`cum_` results carry their own `strata` array, `solvers.py:1267-1270`).

Proposal. (a) Drop `new_infections`, `cum_infections`, and the automatic `prevalence` from declared models; keep them for library diseases, whose states have those names anyway (`09-results.md:38`). A data column called `new_infections` on a declared model can still be matched through a fixed alias table (`new_infections` → the sum of the transmission flows), so Starsim 3 data files carry over without the result existing. (b) Make `cum_<transition>` a property, `res.infection.cum`, computed on request (`np.cumsum` over the right axis, with `summarize_by='last'`), rather than a stored result. Then an SIR has `n_S`, `n_I`, `n_R`, `infection`, `recovery`: five results, one per model element, which is what the spec's "one count per state, one flow per transition" says.

Cost. Breaking for Starsim 3-style code that reads `sim.results.sir.new_infections` on a declared model (library `ss.SIR` unaffected). `cum_` as a property means `sim.summarize()` no longer reports cumulative totals unless it is taught to (it already special-cases `cum_` by name, `sim.py:2042`). Spec: `09-results.md:24-34` loses two bullets and the B1 row closes.

Confidence: high.

### 2. The many-run surface is three objects where one would do

The issue. After `msim = sim.run(n_runs=5)` (`t5_msim.py`, `t7_restart.py`):

```text
msim.results type MultiResults keys ['sir']
n_I type MultiResult values (5, 41)
has to_df? False has resample? False  isinstance Result? False
nI+1 ERR TypeError unsupported operand type(s) for +: 'MultiResult' and 'int'
nI[0] ERR TypeError 'MultiResult' object is not subscriptable
select(run=0) -> ndarray
by(age) -> objdict          (Result.by() returns a 2D Result)
msim.results.sir type objdict (sim.results.sir is an ss.Results)
```

Then `msim.reduce()` adds a second representation into the same dict, lowercased and flattened, next to the nested one:

```text
after reduce: results keys ['sir', 'sir_n_s', 'sir_n_i', 'sir_n_r', 'sir_infection', ...]
msim.summarize() keys: ['sir_n_S', 'sir_n_I', ...]      (case kept here, lost there)
```

and a third copy lives in `msim.base_sim.results.sir.n_I.low/high`. The spec's own first line for many runs, `msim.plot()  # median and interval` (`09-results.md:207`), is not what happens: unreduced, `msim.plot()` plots each sim as a line (1 line on the restart path, 5 on the copy path), and after `reduce()` it crashes for every declared model:

```text
B: reduced plot ERR AttributeError 'objdict' object has no attribute 'columns'    (run.py:690)
C: ss3 reduced plot ok                                                               (a Starsim 3 sim is fine)
```

because `match_result_keys` flattens the `MultiResults` dict and finds the nested `sir` objdict of `MultiResult`s. And on the restart path (the default for any pure-compartment sim) the MultiSim holds one sim and five runs:

```text
A: len(msim) 1 n sims 1 runs in results 5
A: summarize n_I  mean: 58.2  std: 0.0  sem: 0.0          (over 1 sim)
A: reduced low==high everywhere? True
B (copies path): len(msim) 5, summarize std: 20.9
```

so `msim.summarize()`, `msim.compare()`, `msim.reduce()`, `msim.mean()`, `msim.median()`, `msim.combine()`, `msim.split()` and unreduced `msim.plot()` all silently report on one run while `msim.results` has five. Nothing warns.

Why it matters. This is the headline promise of the design ("A sim is one run; many runs are a MultiSim ... the same result, with a run dimension") and a user's first `msim.plot()` either shows one trajectory or crashes. The three containers (`MultiResult`, reduced flat `Result`s, `base_sim` copies) each answer "what is the median?" differently, and `run.py` carries ~120 lines of `MultiResult.select/by/_strat/_keep/plot` (`run.py:966-1054`) that mirror `Result.select/by/_check_strata/_substrata` (`results.py:144-242`).

Proposal. One object: `ss.Result` with a leading `run` dimension (and `scenario` as a coordinate on it), so `msim.results.sir.n_I` *is* an `ss.Result`, `isinstance` holds, `to_df`, `resample`, arithmetic, `plot`, `select(age=0)`, and `select(run=3)` all work by the same code, and `quantiles()`/`mean()`/`median()` reduce over the run axis and return a `Result` with `low`/`high` set (which is what `reduce()` produces today, without copying the sim). `msim.results` is then an ordinary `ss.Results` with one `Results` per module; `reduce()`, `mean()`, `median()`, `which`, `orig_base_sim`, `_rm_orig_sim`, and `MultiResults.merge` go. `msim.plot()` draws median and interval by default and `lines=True` draws the runs, as the spec says. The restart path should produce a MultiSim whose `sims` is either the list of runs or empty with `len(msim) == n_runs` taken from the results; today's "one sim, five runs" hybrid should not exist.

Cost. Breaking for Starsim 3 code that calls `msim.reduce()` and reads `msim.results.sis_prevalence` (flattened, lowercased); a thin compatibility shim (`reduce()` returning `self` after computing quantiles) is possible. The `Result` class must become n-dimensional (see finding 3); `combine()` (sum sims as one population) is a genuinely different operation and can stay as a function. Spec: `09-results.md:199-220` already describes the proposed behavior; `17-vocabulary.md:122` ("A result of an `ss.MultiSim` ... `quantiles` `values`") stays.

Confidence: high.

### 3. Two kinds of "one result with several series", and neither composes

The issue. A `Result` can be 2D by `columns` (Starsim 3's multi-strain results: `res['alpha']`, `col()`, `by_column()`, `stack()`, `results.py:244-265`) or stratified by `strata`/`strats`/`dims`/`denom` (v4, `results.py:144-242`). `by()` converts the second into the first, and the conversion loses the ability to go back (`t2_strata.py`):

```text
by(age): Result ['0-14', '15-64', '65+'] (21, 3)
by then select? ERR ValueError The result "n_I" is not stratified, so there is nothing to select
by(risk) columns: ['risk_low', 'risk_high']       # 'low' and 'high' collide with Result.low/high (results.py:190-191)
resample by ERR TypeError Only valid with DatetimeIndex ...  (see bugs)
```

A ratio of two results loses its groups unless it was declared on the disease (a known issue), so `res.n_I / (res.n_S + res.n_I + res.n_R)` cannot be selected but `prev = I / N` can; `2 * res.prev` keeps its groups but `res.prev / 2`... depends on the operand order (`results.py:217`, a single 400-character condition). The attributes `strata`, `strats`, `dims`, `denom` are set by the solver from outside (`solvers.py:1267-1270`) rather than being part of `Result`'s constructor, so `getattr(self, 'strata', None)` appears 9 times in `results.py` and a `Result` made any other way has none of them.

Why it matters. The spec's rule is simple ("a stratified result is one result with a dimension for each stratification; the strata are coordinates"), but the user meets two representations with different methods, a renamed column, and arithmetic that works or not depending on how a quantity was written. The implementer carries both.

Proposal. Give `Result` one model: `values` of shape `(n_t, *dims)` with named `dims` and labelled `coords` (what xarray calls a DataArray; no need to depend on xarray, but the same three attributes), with a `denom` of the same shape for ratios. `select()` indexes coordinates and sums the rest; `by(dim)` sums the others and returns the same class (so `by('age').select(age=0)` works and `by('risk')` columns are `low`/`high` because columns are coordinates, not attributes); `run` is just another dim (finding 2); `columns` for multi-strain results is `dims=['strain']`. Arithmetic is then plain broadcasting with one rule written once: a sum or scalar product keeps groups, a ratio keeps numerator and denominator.

Cost. Breaking for `res['alpha']`/`res.alpha` on multi-strain Starsim 3 results (becomes `res.select(strain='alpha')`; a `__getitem__` shim is cheap). `to_df` for n-dim results needs a long or wide choice; wide with `name[age=0-14]` columns matches what `ss.data.widen` already produces (`calibration.py:1225-1236`), which is a nice closure. Spec: `09-results.md:48-68` unchanged in substance; "returns a result" becomes literally true for `by()`.

Confidence: medium-high (the design is clear; the port touches `reduce`, `combine`, `flatten`, `to_df`, `plot`, and the solvers' `store`).

### 4. `every=` makes a comb, and a different comb from `resample()`

The issue (`t8_misc.py`, dated sim, `weekly = ss.outcome(0.5*infection, every=ss.weeks(1))`, `daily = ss.outcome(0.5*infection)`):

```text
weekly[:9]  [0. 0. 0. 0. 0. 0. 0. 6.5 0.]
daily.resample('week')[:3]  [ 2.7 13.  32.7]      weekly nonzero[:3]  [ 6.5 20.5 40.1]
daily data vs every=week outcome -> model [0. 0. 0. 0. 0. 0. 6.5]   (data [0.5 0.6 0.7 0.9 1. 1.3 1.5])
weekly data vs daily outcome -> model [ 6.5 20.5 40.1 37.2]  data [ 6.5 20.5 40.1 37.2]
```

So: (a) `resample('week')` uses pandas calendar weeks (anchored to the calendar, not to the sim start) and `every=` uses weeks from the sim start, so "weekly cases" has two answers; (b) the daily outcome compared with weekly data is exactly right without `every=`, because `compare()` already sums a flow over each data interval (`calibration.py:1329-1333`, as the spec says at `09-results.md:167`); (c) a weekly outcome compared with daily data is wrong and silent. `every=` therefore adds a way to get the wrong comparison and nothing the data path does not do already; its one unique feature, a function saying whether today is a reporting day, is not exercised anywhere in the tests (`grep every= tests/` finds only `ss.weeks(1)`).

Why it matters. The overview's first model (`00-overview.md:18`) uses `every=ss.weeks(1)` with weekly data, which works only because the data happens to fall on the same days; a user with daily data or who plots the series gets a comb.

Proposal. Either drop `every=` (and the "reporting day" function) and say "weekly data is compared with the weekly sum automatically; to see a weekly series call `res.cases.resample('week')`", or keep it and make the stored series the running total within the period (a step function that resets), which plots sensibly and compares correctly with data at any spacing. If `resample()` stays, it should bin from the sim start for unit sims and let the user pick the anchor for dated ones.

Cost. Dropping is breaking for the overview model and `tests/test_transitions.py:651`; it removes one keyword, one row of the solver table (`09-results.md:135-139`), and `info.every` (`solvers.py:1327`, `1341-1410`). Not dropping costs a paragraph in the spec warning about the comb.

Confidence: high that the current behavior is a trap; medium on which fix.

### 5. `duration=` is exponential, `delay=` is exact, in the same call

The issue (`t4_outcome.py`, `hosp = ss.outcome(0.1*infection, ...)`, ODE, daily step):

```text
{'delay': days(7)}                      peak inf t=17, peak hosp t=24
{'duration': days(7)}                   peak hosp t=17, n_hosp peak t=22, implied mean stay=7.42, final n_hosp=1.30
{'duration': ss.constant(v=7)}          peak hosp t=17, n_hosp peak t=20, implied mean stay=6.98, final n_hosp=0.58
```

`implementation.md` lists this under known issues. The spec itself says it (`09-results.md:119`: "a duration (an exponential stay) or a distribution (`ss.constant` for a fixed stay)"), so spec and code agree, but the agreement is on a trap: two keywords of one function take the same type and read it differently, and nothing in the model text or `describe()` says so. Transitions have the same convention (`I >> R @ ss.days(10)` is an exponential dwell), which is the argument for keeping it; but an outcome is a reporting shortcut ("ten days in hospital"), where the user almost always means the number.

Proposal. `duration=ss.days(10)` means a stay of 10 days (`ss.constant`), matching `delay=`; `duration=ss.expon(...)` or `ss.lognorm_ex(...)` for a distribution. Both keywords then read the same way, and the known issue closes.

Cost. Breaking for any model that relied on the exponential (tests: `tests/test_transitions.py:652`, `:705`, `:729`); it changes `n_hosp` curves slightly. Spec: one clause at `09-results.md:119` and the solver table row for `duration`. The inconsistency with transitions is real but transitions have `@ dist` for the same choice, and the spec already decided a bare number means different things in different places (bounds, splits).

Confidence: medium-high.

### 6. `summarize()` and `summarize_by` disagree, and `summarize()` is a name heuristic

The issue. Flows are created with `summarize_by='sum'` (`solvers.py:1218`), which `resample()` and `annualize()` honor, but `sim.summarize()` matches on name prefixes (`sim.py:2042`: `'new_': 'mean'`, `'': 'mean'`), so (`t8_misc.py`):

```text
7. summarize infection (mean): 1.24   sum 7.45
```

`Result.summary_method()` (`results.py:303-322`) is a second heuristic, with different answers from `summarize_by` (`infection` has `summarize_by='sum'` but `summary_method()` says `mean`, `t1_names.py`). So there are three ways a result says how to be summarized.

Proposal. One attribute, set at creation for every result (`'sum'` for flows and outcomes, `'mean'` for stocks and ratios, `'last'` for cumulative), used by `summarize()`, `resample()`, `annualize()`, and `compare()`. Delete `summary_method()` and the `how` dict. The spec's "a result knows whether it is a flow or a stock" (`09-results.md:46`) is exactly this attribute; call it `kind` (`'flow'`/`'stock'`/`'ratio'`) and derive the summary from it, which also fixes the next two findings.

Cost. `sim.summarize(how=...)` loses its prefix-matching dict form (nobody should miss it). Not breaking for values of stock results; `summarize()` of flows changes from mean to sum, which is a behavior change Starsim 3 users would notice in printed summaries.

Confidence: high.

### 7. `resample()` does not do what the spec says for stocks and ratios, and fails without dates

The spec (`09-results.md:46`): "`res.infection.resample('week')` sums, `res.n_I.resample('week')` takes the value at the end of the week, and a ratio is recomputed from its resampled parts." The code (`t6_summarize.py`, dated SIR):

```text
n_I.resample(week)[:3] [30.3 89.1 214.1]   # mean over the week; n_I at the week's end is 72.6
prevalence.resample    [0.030 0.089 0.214]  # mean of the ratio, not a ratio of resampled parts
```

and for a sim with `unit='day'` and no dates (`t2_strata.py`, `t8_misc.py`):

```text
2. declared unit=day resample ERR TypeError Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex
2. ss3 unit=day resample ok
```

with three `Dates with years < 1 are not valid` warnings on the way (`results.py:477-478` turns day numbers into `ss.date(t)`). Either the spec should say "mean" for stocks (defensible, and Starsim 3's behavior) or the code should take the end value; the ratio rule is the same rule as `select()` and should share its code (`denom`). The numeric-time failure is a bug, below.

Proposal. Implement `resample` on the `kind` attribute of finding 6: sum, last (or mean, but say which), and numerator/denominator for ratios; accept numeric time vectors by binning on the sim's own axis rather than converting to pandas dates.

Cost. None for users; `resample` and `annualize` (`results.py:324-436`, two implementations of the same operation) collapse to one.

Confidence: high.

### 8. Data by stratum attaches but is never drawn, and the matcher exists twice

The issue. Long-form data widens correctly (`t3_data.py`):

```text
long -> wide cols ['cases[age=0-14]', 'cases[age=15-64]', 'cases[age=65+]']
compare keys ['cases[age=0-14]', 'cases[age=15-64]', 'cases[age=65+]']
scatter per axis [0, 0, 0, 0, 0, 0, 0]   (sim.plot: nothing drawn)
```

because `sim.plot` has its own matcher (`sim.py:2432`: tries `mod.name`, `mod_name`, `name`) that knows nothing about `[age=...]`, while `ss.data.match` (`calibration.py:1247-1268`) knows about strata and the `module.result`/`module_result` spellings. The spec promises "Results that have data are plotted with it automatically" (`09-results.md:168`). Also, `E704 DataMisaligned` is raised only by `compare()`, so `ss.Sim(..., data=<2023 data>).run()` on a 2020 sim runs and plots nothing (`t7_restart.py` D).

Proposal. One matcher (`ss.data.match`), used by `init_data` (to check), `compare` (to fit), and `plot` (to draw); check `E704` in `init_data`. Stratified data draws on the panel of its result, one marker style per stratum, or on `res.by('age').plot()`.

Cost. None; removes code.

Confidence: high.

### 9. `select(run=0)` and `by()` return different kinds of thing than their siblings

Covered by findings 2 and 3; listed here because it is the user-visible symptom: `Result.by()` returns a 2D `Result`, `MultiResult.by()` an `objdict` of `MultiResult`s, `MultiResult.select(run=0)` a bare `ndarray`, `MultiResult.select(scenario=...)` a `MultiResult`. One class with one `select`/`by` fixes all four.

### 10. A result read during the run has no "so far"

`res.n_I[:sim.ti]` works, `res.n_I[-3:]` silently returns the preallocated zeros (`t8_misc.py`: `r[-3:] -> [0. 0. 0.]` at `ti=5`), and there is no `so_far`. The spec lists it as open (`19-open-questions.md:143`). A trigger written as `sim.results.sir.n_hosp[-1] > 100` is the natural spelling and is always false. Proposal: `res.so_far` (a view `values[:sim.ti+1]`), or make `Result.__getitem__` with negative indices relative to `ti` while the sim runs (magic; the view is better). Separately, a plain `ss.Analyzer` that reads a result forces the model to agents (`E804` in `t8_misc.py` until `compartmental = True` was added) although reading a result needs no agents; the spec says "anything can read a result" (`09-results.md:153`). Low cost either way. Confidence: medium.

### 11. The transmission log is promised for declared diseases and records nothing

`t7_restart.py` E:

```text
E: log rows with network 0 infections 297.0
E: log rows no network 0 infections 45.0 solver abm
```

The spec devotes a section to it (`09-results.md:229-238`, with infector, source group, and source state). The analyzer collects `disease.infection_log` (`analyzers.py:49-53`), which the declared engine never writes. Either implement it for declared diseases (agents only; the agent engine knows the infector where there is a network, so it is one record per infection event) or mark it deferred. Confidence: high that it is missing.

## Simpler codebase

### 1. Results logic is spread over six files

Where a result is defined, filled, aggregated, matched to data, and plotted: `results.py` (class), `solvers.py:1204-1565` and `:2644-2835` (creation and filling, including the `strata`/`denom` attributes set from outside the class), `run.py:931-1095` (the many-run variants), `calibration.py:1174-1340` (`ss.data`, `compare`, `data_times`, `widen`, `parse`, `match`), `transitions.py:918-1017` (`ss.outcome`, `FlowSum`), `sim.py:1703-1715` (`init_data`) and `:2350-2457` (`plot`, with its own column matcher), plus `utils.py:578-620` (`validate_sim_data`, a second time-column parser with the same list as `data.time_cols`). `ss.data` is a results concept and does not belong in `calibration.py`. Proposal: `results.py` owns `Result`, `Results`, `data`, `compare`, and the one matcher; `run.py` keeps `MultiSim`; `outcome` can stay with transitions (it is model text). Confidence: high; cost is a move.

### 2. `MultiResult` duplicates `Result`

`run.py:966-1021` (`select`, `by`, `_strat`, `_keep`) against `results.py:144-242` (`select`, `by`, `_check_strata`, `_check_dim`, `_substrata`); `run.py:1042-1054` (`plot`) against `results.py:550-583`. `MultiResults.from_sims` (`run.py:1076-1094`) and the tail of `restart_runs` (`run.py:1152-1167`) build the same structure twice. All of it goes with recommendation 1.

### 3. `Result.__array_ufunc__` is one unreadable condition

`results.py:217` is a single boolean expression of about 400 characters deciding whether groups survive arithmetic, followed by a second one at `:222-225` for denominators. With a `dims`/`coords` model this becomes broadcasting plus one rule. Until then, it should be a small function with named cases (`is_linear`, `is_scaled_ratio`).

### 4. Two resamplers, three summarizers

`resample()` (`results.py:324-388`, via pandas) and `annualize()` (`:390-436`, via `np.bincount`) do the same job; `summary_method()`, `summarize_by`, and `sim.summarize(how=)` decide the same question. One `kind` attribute and one `resample` (finding 6 and 7 above).

### 5. `samples.py` is Starsim 3 code with no v4 role

432 lines, last meaningful change before v4 (`git log`: "update samples", "Add support for BytesIO"), referenced only by `tests/test_samples.py` and `docs/user_guide/workflows_samples.qmd` (which still says `cvv.Samples`, a Covasim leftover). It stores per-run dataframes and summaries in a zip, filtered by parameter "identifiers", which is what `ss.parallel(sim, pars=..., n_runs=...)` + `msim.run_pars` + `msim.results` now provide in memory, and `sc.save` provides on disk. Proposal: move to `starsim.library` or a separate package; the spec never mentions it. Confidence: high.

### 6. `MultiSim` carries Starsim 3 machinery that the restart path does not populate

`reduce`, `mean`, `median`, `combine`, `compare`, `plot_compare`, `merge`, `split`, `shrink`, `copy`, `reset`, `_has_orig_sim`, `_rm_orig_sim`, `which`, `orig_base_sim` (`run.py:219-260`, `261-466`, `586-612`, `703-777`) all assume `msim.sims` is the list of runs. `restart_runs` returns a MultiSim with one sim (`run.py:1164`), so every one of these is wrong or vacuous on the default path for compartmental sims (verified: `std: 0.0`). Either the restart path makes `n_runs` lightweight sims (results only; `sim.shrink()` already exists for that shape) or these methods move to the results object and `sims` becomes optional. The 50-line `__init__` that accepts a sim, a list, a MultiSim, or a list of MultiSims (`run.py:46-100`) is also more generality than `ss.parallel` needs.

### 7. `plot` is written four times

`Result.plot` (`results.py:550`), `Results.plot` (`:760`), `Sim.plot` (`sim.py:2350`, 107 lines with its own data matcher and figure-size logic), `MultiSim.plot` (`run.py:613`, 88 lines with a copy of the same figure-size logic at `:672-677` and `sim.py:2396-2400`), and `MultiResult.plot` (`run.py:1042`). `Sim.plot` and `MultiSim.plot` (reduced branch) could both be `self.results.plot(data=self.data, **kw)` once results carry their run axis and bounds.

## Spec and implementation

- **B1 flow names: answered.** The implementation keeps both spellings and every v4 test and example uses the transition's name. The spec should pick it and drop `new_infections` for declared models (finding 1).
- **`ss.Sim(results=[...])`** (`09-results.md:240-246`, "Requesting fewer results") is not implemented (`KeyNotFoundError: Key(s) ['results'] not found`, `t6_summarize.py`) and is not in `deferred.md`. It matters most for stratified models, which is where memory grows; should be listed one way or the other.
- **`msim.plot()` as median and interval** (`09-results.md:207`) and **`lines=True`** (`:208`, `17-vocabulary.md:159`): neither is what the code does (finding 2 above; `msim.plot(lines=True)` raises `KeyNotFoundError: Did not successfully convert all plotting keys`).
- **`msim.summarize()` SHOULD report extinction** (`09-results.md:222`): `implementation.md:185` says "not yet"; fine, but `summarize()` itself is wrong on the restart path, so fixing that comes first.
- **Stock resampling** is specified as end-of-period and implemented as the mean; **ratio resampling** is specified as recomputed from parts and implemented as the mean of the ratio (finding 7).
- **Results that have data are plotted with it** (`09-results.md:168`): true for a plain column, false for `cases[age=...]` columns (finding 8).
- **`E704 DataMisaligned`** is raised by `compare()`, not when the data is attached (`09-results.md:165` says it is an error of attaching).
- **The transmission log for declared diseases** (`09-results.md:229-238`) is specified in detail and records nothing (finding 11).
- **A ratio written in Python after the run** (`res.n_I / tot`) "says the result is not stratified" (known issue), while the spec's `.select()` section reads as if any result can be selected; the spec should say the rule (declared ratios keep groups; ad hoc ones do not), or the code should keep them (finding 3 makes that natural).
- **`by()` levels named like attributes** become `risk_low` (departure, `implementation.md`); it is a symptom of columns-as-attributes (finding 3), not something to document.
- **Sim-level results differ by solver.** Under `abm` a declared SIR has `sim.results.n_alive, n_female, new_deaths, new_emigrants, cum_deaths`; under `ode` and `binomial` it has none (`t1_names.py`). The module-level results are the same under all three (same names, same dtypes; only the internal `scale` flag differs), which is good. Whether `n_alive` should exist for a compartmental run (it is `sum(n_<state> for people states)`) is worth a sentence in the spec.
- **`ss.data` with a Starsim 3 sim** works for attaching and `compare()` (`t3_data.py`: `ss3 data ok ['prevalence']`), which the spec does not say; worth saying, since it is the migration path.
- **Open question "Is `ss.outcome` needed at all?"** (`19-open-questions.md:141`): the implementation's `duration=` outcome *is* a state with a dwell time computed as a convolution (`solvers.py:1317-1324`), minus the ability to select people. The two trap findings (4 and 5) are both about `ss.outcome` growing state-like options. A smaller `ss.outcome(source, delay=)` plus "use a state for anything with a duration" would remove `duration=`, `every=`, `n_<outcome>`, `info.stay`, `info.leaving`, and `info.p_leave`. I would not go that far yet (the hospital-occupancy outcome is genuinely common and the spec's own argument for it is good), but the question is still live and the spec should say so.
- **Reading a result from a module forces agents** (`E804` for a plain `ss.Analyzer`, `t8_misc.py`): `09-results.md:153` says "anything can read a result"; `12-backends.md`'s `compartmental = True` departure is the reason. The results spec should mention it or analyzers that only read results should be compartmental by default.

## Bugs

1. **`msim.reduce(); msim.plot()` crashes for every declared model** (`run.py:690`, `AttributeError: 'objdict' object has no attribute 'columns'`), on both the restart and copy paths; a Starsim 3 sim is fine. Repro: `t7_restart.py` B and C. Cause: `MultiResults.__init__` puts the nested `sir` objdict into the same dict that `reduce()` fills with flat results (`run.py:1062-1063`, `:339`), and `match_result_keys(flattened=True)` iterates both.
2. **`sim.run(n_runs=5)` on a compartmental sim gives a MultiSim with `len(msim) == 1`**; `summarize()` reports `std: 0.0`, `reduce()` gives `low == high == values`, `compare()` has one column, unreduced `plot()` draws one line. Repro: `t7_restart.py` A (`run.py:1164`).
3. **`res.resample('week')` fails on a declared sim with `unit='day'` and no dates** (`TypeError: Only valid with DatetimeIndex ...` from `results.py:375`, after three `Dates with years < 1` warnings from `:478`); the same call on a Starsim 3 `unit='day'` sim works. Repro: `t8_misc.py` 2, `t2_strata.py`.
4. **`analyzers='infection_log'` records 0 rows for a declared disease** under `abm`, with or without a network (`t7_restart.py` E).
5. **`msim.plot(lines=True)` raises** `KeyNotFoundError: Did not successfully convert all plotting keys` (`t5_msim.py`); the spec and vocabulary list it.
6. **`sim.plot()` does not draw long-form (stratified) data**; `compare()` sees it (`t3_data.py`, `scatter per axis [0, 0, 0, 0, 0, 0, 0]`).
7. **`MultiResult.select(age=0)` on a result without groups raises `AttributeError: 'NoneType' object has no attribute 'shape'`** (`run.py:1014`, `_keep` runs before `_strat` checks) instead of the friendly "not kept by group" message; hit by `msim.results.sir.prev.select(age=0)` on the restart path, where ratios are not kept by group (`t2_strata.py`).
8. **`E704 DataMisaligned` is not raised at attach time**: `ss.Sim(sir, start='2020-01-01', stop='2020-02-01', data=<2023 rows>).run()` runs and plots without complaint; only `compare()` raises (`t7_restart.py` D).
9. **`sim.summarize()` reports the mean of flows** that carry `summarize_by='sum'` (`infection: 1.24` vs sum `7.45`, `t8_misc.py` 7) because it matches on name prefixes (`sim.py:2042`).

## Keep

- **One count per state and one flow per transition, named exactly as the model text, with the timing convention `n_R[ti] - n_R[ti-1] == recovery[ti]`** under every solver (verified for ODE, binomial, agents: `recovery[0] == 0`, `n_R[1]-n_R[0] == recovery[1]`). This is the core of the design and it is right; same names and dtypes across solvers (`t1_names.py`).
- **`select()` and `by()` on a single-run result.** The semantics (keep the named levels, sum the rest; ratios sum numerator and denominator then divide; select by label or position; chained selection; an unknown dimension lists the ones that exist) are correct and tested: `prev.select(age=0)` equals the hand computation to 1e-16, `select(risk='low') + select(risk='high') == total`, `select(age=0).select(age=1) == 0`. Keep the semantics while changing the representation.
- **`ss.outcome` as a weighted sum of flows with `delay=`**, with chaining and `E702`/`E703`. `0.5 * infection` as the probability is the right spelling; the weight as an `ss.par` or `ss.by` is a real feature. Keep `delay=`; the questions are about `duration=` and `every=`.
- **Data by column name, `ss.data(aliases=, ignore_extra=)`, and `E705` with the closest names and the exact `aliases=` line to type.** The error text is a model of its kind (`t3_data.py`). Long-form by stratum (`cases[age=0-14]`) and the levels-by-label check also work and are well made.
- **`compare()`'s rule**: a flow is summed over the data's interval, a stock interpolated at the data's time; the weekly-data-vs-daily-outcome case reproduces the data exactly. This is why `every=` is not needed for fitting.
- **`msim.extinction()`**: correct, small, and reports per scenario. Keep, and fold into `summarize()` once `summarize()` is right.
- **`ss.parallel(baseline=..., vaccine=...)` naming sims by keyword, `n_runs=`, `pars=` sweeps with `run_pars`, and `msim.diff('baseline')` paired by run.** The surface is right; only the result object underneath needs to change.
- **`restart_runs`** as a performance path (many times faster than copies) is worth keeping; it is the MultiSim it hands back that is wrong.

## Outside my area

- A plain `ss.Analyzer` subclass with a `step()` that only reads results makes the whole model agent-based (`E804` under `solver='binomial'`) unless it sets `compartmental = True`; the default for analyzers (which by construction do not move people) should probably be `True`.
- `ss.Disease(...)` with no variable to take a name from is called `disease`, so a function returning diseases gives `sim.results.disease`; fine, but `describe()` should flag it as a guessed name.
- A `ss.SIS(beta=0.1)` alongside a declared disease in a `unit='day'` sim prints the "beta read in years" warning on every run; correct, but noisy where the user cannot act on it except by writing `ss.perday`.
