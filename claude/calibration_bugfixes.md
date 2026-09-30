# `calibration.py` bug audit

Audit of `starsim/calibration.py` (1156 lines, read in full: the `Calibration` class, the three conformers, `CalibComponent` and its five subclasses) at commit `3d8dc9d5` (branch `rc3.6.2`, file unmodified in the working tree), using Starsim 3.6.1 (editable install), numpy 2.4.6, pandas 3.0.5, optuna 4.9.0, sciris 3.3.0. Every finding below was reproduced with a script in the session scratchpad (`calibration/components.py`, `calibration/calib_repros.py`, `calibration/die_false.py`) run under `MPLBACKEND=agg`, with tiny sims (500 agents, 20 days, 3-8 trials, `debug=True`). Intent was checked against the docstrings, `tests/test_calibration.py`, and `docs/user_guide/workflows_calibration.qmd` / `workflows_sir_calibration.qmd`. Only unambiguous defects are listed; the bootstrap plotting helpers (documented as experimental) were not audited in depth.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Normal.compute_nll()` | With the default `sigma2=None`, the "ML variance" is computed from every expected timepoint against one simulated value, so a worse fit can score better than a perfect fit | 1077 |
| 2 | Medium | `Normal.compute_nll()` | A per-timepoint `sigma2` array (documented) is broadcast across every row instead of being matched to its timepoint | 1079 |
| 3 | Medium | `Calibration.run_trial()` | `die=False` (the default) does not keep the calibration going: a failed sim returns `None`, which then crashes `eval_fn` and aborts `calibrate()` | 109, 182 |
| 4 | Medium | `step_containing()` | Uses `searchsorted(side='left')`, so a data time between two steps picks the *next* step, not the step containing it | 484 |
| 5 | Medium | `CalibComponent._combine_reps_nll()` | `combine_reps` crashes with `ValueError: No group keys passed!` when `t` is in float years (a documented index type) | 603 |
| 6 | Low | `Calibration.check_fit()` | Crashes if `build_fn` returns a MultiSim that was not created with `initialize=True` | 320 |
| 7 | Low | `Binomial.get_p()` / `compute_nll()` | The `'p'`-column branch calls `.values` on a scalar and always crashes | 778 |
| 8 | Low | `Calibration.check_fit()` | The "calibration was interrupted" fallback is unreachable: `self.best_pars` is never initialized, so it raises `AttributeError` | 296 |

## High severity

### 1. With the default `sigma2=None`, the "ML variance" mixes all expected timepoints against one simulated value, so a worse fit can beat a perfect fit — `calibration.py:1077`

When `sigma2` is not given, `compute_nll()` recomputes the variance separately for each merged row as `self.compute_var(expected['x'], a_x)`: the full column of *all* expected values minus the *single* simulated value of the current row. That is not the residual variance described by `compute_var()`'s docstring ("maximum-likelihood variance of the residuals between expected and actual values") or by the user guide ("max likelihood estimates will be used if not provided"). The resulting sigma depends on the spread of the data around each simulated point, so it is smallest for simulated values near the middle of the data and largest at the extremes. This biases the objective towards trajectories that are flattened towards the mean of the data.

```python
t = [2020., 2021., 2022.]
exp = df([10., 20., 30.], t)   # df(x, t) -> DataFrame({'x': x}, index=pd.Index(t, name='t'))
for label, a in [('perfect fit', [10., 20., 30.]), ('worse fit ', [15., 20., 25.])]:
    comp = ss.Normal('x', exp, extract, conform='prevalent')
    print(label, 'nll =', comp(FakeSim(df(a, t))))   # FakeSim just carries .data and pars.rand_seed
```

Actual:

```
perfect fit nll = 3.324221316102688
worse fit  nll = 3.215851406759905
```

Expected: the perfect fit should have a lower NLL than a fit that misses two of the three points by 5.

Blast radius: every `ss.Normal` component created without `sigma2`, which the docs present as the default ("it's optional"). The optimizer is pulled towards the wrong optimum without any warning. Single-timepoint components are unaffected (the variance degenerates to the squared residual).

**Fix**: compute the variance once from all residuals before the loop, e.g. `sigma2 = self.compute_var(combined['x_e'], combined['x_a'])` (after applying the `/n` rate conversion to the actual column if `'n'` is present), and use that for every row. If a per-replicate estimate is wanted, group `combined` by `rand_seed` and compute one variance per replicate.

## Medium severity

### 2. A per-timepoint `sigma2` array is broadcast across every row instead of being matched to its timepoint — `calibration.py:1079`

The user guide says `sigma2` "could be a single float or an array with the same shape as the expected values", and `plot_facet()` explicitly supports this (it indexes `sigma2[ti]` by the timepoint). `compute_nll()` does not: it passes the whole array as `scale=np.sqrt(sigma2)` for each scalar row, so each row returns a vector of log-likelihoods, one per variance. `nlls` becomes an `(n_rows, n_timepoints)` array and `eval()` averages all of it, so each residual is scored against every timepoint's variance.

```python
comp = ss.Normal('x', exp, extract, conform='prevalent', sigma2=np.array([1., 100., 10000.]))
wnll = comp(FakeSim(df([11., 30., 130.], t))) # Every residual is exactly 1 sigma
print('nll shape:', np.shape(comp.nll))
print('returned:', wnll)
right = -sps.norm.logpdf([1., 10., 100.], scale=np.sqrt([1., 100., 10000.]))
print('expected per-timepoint nll:', right, '-> mean', right.mean())
```

Actual:

```
nll shape: (3, 3)
returned: 570.0559736261988
expected per-timepoint nll: [1.41893853 3.72152363 6.02410872] -> mean 3.7215236261987186
```

Blast radius: anyone following the documented per-timepoint variance option. The large-variance point's residual (100) is scored with `sigma=1`, which dominates the objective.

**Fix**: when `sigma2` is array-like, look up the entry for each row's timepoint (e.g. `sigma2[expected.index.get_loc(rep['t'])]`, as `plot_facet()` already does), or vectorize: map `sigma2` onto `combined` by `t` and call `sps.norm.logpdf` once on the columns.

### 3. `die=False` (the default) does not keep the calibration going: a failed sim aborts `calibrate()` anyway — `calibration.py:109`, `calibration.py:182`

`run_sim()` catches the exception when `die=False`, prints the traceback, and returns `None`. `run_trial()` then passes that `None` straight to `self.eval_fn(sim, ...)`. The default `_eval_fit()` calls each component, and `extract_fn(None)` raises. Optuna's `study.optimize()` re-raises exceptions from the objective by default (`catch=()`), so the whole calibration stops. The `die` docstring says "whether to stop if an exception is encountered (default: false)".

```python
def build_sim(sim, calib_pars, **kwargs):
    beta = calib_pars['beta']['value']
    sim.pars.diseases.pars.beta = beta
    if beta > 0.15:
        sim.pars.interventions = lambda sim: 1/0 # Force this sim to fail when run
    return sim

calib = ss.Calibration(sim=make_sim(), calib_pars=dict(beta=dict(low=0.01, high=0.3, guess=0.1)), build_fn=build_sim,
                       components=[prev], total_trials=8, debug=True, die=False, verbose=False, reseed=False)
calib.calibrate()
```

Actual (after the printed ZeroDivisionError traceback from `run_sim()`):

```
calibrate aborted: AttributeError 'NoneType' object has no attribute 'results'
```

Expected: the failed trial is recorded as failed (or given an infinite mismatch) and the remaining trials run. `parse_study()` already has handling for trials with `value is None`.

Blast radius: any calibration where some region of parameter space makes the sim raise, which is exactly the situation `die=False` exists for. In parallel mode the worker that hits the error dies and takes its remaining trials with it.

**Fix**: in `run_trial()`, if `sim is None`, return `np.inf` (or raise `op.exceptions.TrialPruned()`, or `return float('nan')`, which optuna records as a failed trial) instead of calling `eval_fn`.

### 4. `step_containing()` picks the *next* step when the data time falls between steps — `calibration.py:484`

The docstring and the user guide describe this conformer as choosing "the simulated timestep that contains the time indicated in the real data" using a "zero order hold". The step containing `t` is the last sim time `<= t`, i.e. `searchsorted(..., side='right') - 1`. The code uses `searchsorted(..., side='left')`, which returns the first sim time `>= t`. This is correct only when the data times fall exactly on sim times. For any time strictly between two steps it returns the following step. A data time after the last sim step raises `IndexError`, even though the last step contains it.

```python
actual = df([10., 20., 30.], [2020.0, 2021.0, 2022.0])
expected = df([0., 0.], [2020.5, 2021.5])
print(ss.calibration.step_containing(expected, actual))
print(ss.calibration.step_containing(df([0.], [2022.5]), actual))
```

Actual:

```
           x
t           
2020.5  20.0
2021.5  30.0
t past last step: IndexError index 3 is out of bounds for axis 0 with size 3
```

Expected: `10.0` at 2020.5 and `20.0` at 2021.5 (the value held from the containing step), and `30.0` at 2022.5.

Blast radius: users of `conform='step_containing'` whose data dates don't line up exactly with sim steps, e.g. survey dates mid-month or mid-year against a monthly or annual sim (the SIR calibration user guide uses this conformer). Their data is compared against the simulation one step late, with no warning.

**Fix**: `inds = np.searchsorted(actual.index, t, side='right') - 1` (optionally clipping negative indices or raising a clear error for data before the first step).

### 5. `combine_reps` crashes when `t` is in float years — `calibration.py:603`

`_combine_reps_nll()` finds the columns to group by with `isinstance(actual[c].iloc[0], dt.datetime)`. The `CalibComponent` docstring says the index "should be the time 't' in either floating point years or datetime". With float years there are no datetime columns, so `timecols == []` and `groupby([])` raises. The same component works when `t` is a date.

```python
comp = ss.Normal('x', exp, extract, conform='prevalent', sigma2=1.0, combine_reps='mean')
msim.sims = [FakeSim(df([10., 20., 30.], t), seed=1), FakeSim(df([12., 22., 32.], t), seed=2)] # t = float years
comp(msim)
# same data with ss.date() times instead
```

Actual:

```
ValueError No group keys passed!
same with dates: 1.418938533204673
```

Blast radius: any component that combines replicates (`combine_reps='mean'`/`'sum'`, as in the user guide's multi-rep example) against data indexed by float year, which is common for annual data.

**Fix**: group by the time index names from `self.expected` (`[n for n in self.expected.index.names]`, i.e. `['t']` or `['t', 't1']`) instead of sniffing for datetime values.

## Low severity

### 6. `check_fit()` crashes if `build_fn` returns a MultiSim that was not created with `initialize=True` — `calibration.py:320`

`run_sim()` explicitly supports a `build_fn` that returns a MultiSim ("Run the simulation (or MultiSim)"), and calibration trials work fine with `return ss.MultiSim(sim, n_runs=n_reps)`. `check_fit()` wraps a bare `Sim` itself, but for a MultiSim it assumes `.sims` is already populated. An uninitialized MultiSim has `sims = None`, so `self.before_msim.sims + self.after_msim.sims` raises.

```python
def build_sim(sim, calib_pars, n_reps=1, **kwargs):
    ...
    return ss.MultiSim(sim, n_runs=n_reps) # Not initialized

calib = mk(build_kw=dict(n_reps=2))
calib.calibrate()          # runs fine
calib.check_fit(do_plot=False)
```

Actual:

```
check_fit: TypeError unsupported operand type(s) for +: 'NoneType' and 'NoneType'
```

Blast radius: users who build replicate MultiSims without `initialize=True`. The docs and tests always pass `initialize=True`, which is why this rates Low. The calibration itself completes, and the crash happens at the validation step.

**Fix**: before concatenating, call `init_sims()` on any MultiSim whose `sims` is `None`.

### 7. `Binomial`'s `'p'`-column branch always crashes — `calibration.py:778`

`Binomial.get_p()` supports a precomputed probability column (`if 'p' in df: p = df['p'].values`), and `compute_nll()` has a matching branch (`if 'p' in rep`). But `compute_nll()` passes a single row (`rep`, a Series), so `df['p']` is a numpy scalar with no `.values`.

```python
exp = df([30, 40], [2020., 2021.], n=[100, 100])
comp = ss.Binomial('x', exp, lambda s: s.data, conform='prevalent')
comp(FakeSim(pd.DataFrame(dict(p=[0.3, 0.4]), index=pd.Index([2020., 2021.], name='t'))))
```

Actual:

```
AttributeError 'numpy.float64' object has no attribute 'values'
```

Blast radius: only users whose `extract_fn` returns `p` rather than `x`/`n`. The code explicitly supports this path, but it can never have worked.

**Fix**: `p = df['p']` (use `np.asarray(df['p'])` if array output is needed for the DataFrame case).

### 8. The "calibration was interrupted" fallback in `check_fit()` is unreachable — `calibration.py:296`

`check_fit()` does `if self.best_pars is None:` and then tries to load the best parameters from the stored study ("Load in case calibration was interrupted"). But `__init__` never sets `self.best_pars`. It is only assigned in `calibrate()` after the workers finish, or in `make_study()` when a duplicate study is found. When calibration is interrupted (e.g. Ctrl-C during `run_workers()`), or when a fresh `Calibration` object points at a kept database, the check raises `AttributeError` instead of taking the fallback path.

```python
calib = mk(study_name='ck_audit', db_name='ck_audit.db', keep_db=True)
calib.calibrate()
calib2 = mk(study_name='ck_audit', db_name='ck_audit.db', keep_db=True, continue_db=True)
calib2.check_fit(do_plot=False)
```

Actual:

```
check_fit: AttributeError 'Calibration' object has no attribute 'best_pars'
```

Expected: `best_pars` loaded from the kept study and the fit checked.

**Fix**: initialize `self.best_pars = None` in `__init__` (alongside `self.study = None`).

## Verified clean

Tested (or, for `parse_study()` and `to_json()`, traced) and found correct: parallel workers (`sc.parallelize(self.worker, iterarg=n_workers)` with `n_workers=2` runs the right total number of trials into one shared study); `to_df()` and `plot_optuna()` after a default `calibrate()` still work even though `keep_db=False` has deleted the sqlite file; `n_trials = ceil(total_trials/n_workers)`; `_sample_from_trial()` (strips `path`/`guess`, honours `suggest_type`, skips specs that already have `value`); `reseed` injecting `rand_seed` into the pars passed to `build_fn` and `prune_fn`; `parse_study()` handling of failed/`None`-valued trials; `to_json()` ordering (the dataframe is already sorted, so positional `argsort` is consistent); `check_fit()` with a bare-Sim `build_fn` (wrapping, joint run, and in-place result propagation back to `before_msim`/`after_msim`); `linear_interp()`; `linear_accum()`'s (t, t1] accumulation, which is documented and intended ("difference in cumulative counts between the end of step t1 and the end of step t"); the `GammaPoisson` negative-binomial reparameterization (`n=1+a_x`, `p=beta/(beta+T)` is the correct gamma-Poisson predictive); `BetaBinomial` parameters; `eval()`'s `weight=0` and NaN-to-inf handling. Not recorded: `build_fn=None` (the default) crashes with `'NoneType' object is not callable`, so it is effectively required. That is an unhelpful-signature issue, not a logic error. Averaging (rather than summing) the NLL over timepoints in `eval()` is a design choice, not a bug.
