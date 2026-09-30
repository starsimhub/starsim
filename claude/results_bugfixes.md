# `results.py` bug audit

Audit of `starsim/results.py` (all 576 lines) on branch `rc3.6.2` at commit `3d8dc9d5`, using Starsim 3.6.1 (editable install), numpy 2.4.6, pandas 3.0.5. Method: line-by-line reading, then a repro script for every candidate, run against the editable install with `MPLBACKEND=agg` (scripts in the session scratchpad under `results/`). Only defects the repro demonstrated, and that are not documented or intended behavior, are recorded. Style, docs, performance, and contrived corner cases are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `Results.annualize()` | Leaves the `timevec` entry at the original (e.g. monthly) resolution, so the annualized Results is inconsistent and `to_df(descend=True)` crashes | 513-514 |
| 2 | Medium | `Results.append()` | Tuple/dict input passes the module name as the Result's `name`: tuples produce a wrongly named result, and dicts always crash | 455, 457 |
| 3 | Medium | `Result.resample()` | Any `col_names` value other than the default `'vlh'` crashes with `AttributeError` | 215, 280 |
| 4 | Low | `Result.plot()` | `fig=` without `ax=` draws on the current figure, not the supplied one | 380-381 |

## Medium severity

### 1. `Results.annualize()` leaves `timevec` un-annualized — `results.py:513`

`Results.annualize()` annualizes every `Result`, but copies every other entry unchanged (`new[key] = res`). This includes `sim.results.timevec`, which is a `DateArray`, not a `Result`. The returned Results then has a timevec of length 37 (monthly) next to results of length 4 (annual). `Results.to_df(descend=True)` uses `self.timevec`, so it crashes.

```python
import starsim as ss
sim = ss.Sim(n_agents=2000, diseases='sis', networks='random', start=2000, dur=ss.years(3), dt=ss.month, verbose=0)
sim.run()
ann = sim.results.annualize()
print(len(ann.timevec), len(ann.n_alive))
ann.to_df(descend=True)
```

Actual:

```
len(ann.timevec)= 37 len(ann.n_alive)= 4
ann.timevec[:3] [<2000.01.01> <2000.01.31> <2000.03.02>]
descend to_df failed ValueError All arrays must be of the same length
```

Expected: `ann.timevec` is the annual timevec (length 4, the same as each annualized result's `.timevec`), and `to_df(descend=True)` returns a 4-row dataframe.

Blast radius: anyone using `sim.results.annualize()` (the documented way to annualize a whole results group, tested in `tests/test_other.py`) and then using the group-level `timevec` or exporting with `to_df(descend=True)`. The shallow `to_df()` still works because it uses each result's own timevec.

**Fix**: In `Results.annualize()`, replace a `timevec` entry with the annual timevec of any annualized child result (e.g. the first `Result`'s `.annualize().timevec`) instead of copying it.

### 2. `Results.append()` builds tuple/dict results with the wrong `name` — `results.py:455`

`Results.append()` (also called via `results += ...`) explicitly supports tuple and dict inputs. It builds them with `ss.Result(self._module, *arg)` and `ss.Result(self._module, **arg)`, but the first positional parameter of `Result.__init__` is `name`, not `module`. With a tuple, the module name becomes the result's name, the intended name becomes the label, and `module` stays `None`, so the result is stored under the key `'mymod'`. With a dict containing `name` (which a result always needs), it raises `TypeError`.

```python
import starsim as ss
res = ss.Results('mymod')
res += ('new_x', 'New x')
r = list(res.values())[-1]
print(r.name, r.label, r.module, list(res.keys()))
res += dict(name='new_y', label='New y')
```

Actual:

```
tuple: name= mymod label= new_x module= None key= ['mymod']
dict failed: TypeError Result.__init__() got multiple values for argument 'name'
```

Expected: a result with `name='new_x'`, `label='New x'`, `module='mymod'`, stored under the key `'new_x'`. The dict form behaves the same way.

Blast radius: users who add results with the tuple/dict shorthand in custom modules. Built-in code always passes `ss.Result` objects, so it isn't affected.

**Fix**: Pass the module as a keyword: `ss.Result(*arg, module=self._module)` and `ss.Result(**(dict(module=self._module) | arg))`.

### 3. `Result.resample(col_names=...)` crashes for any non-default value — `results.py:280`

`resample()` passes `col_names` to `self.to_df()`, which names the value column `self.name` (for `None`) or `col_names` (for a string). `from_df()` then always reads `df.value` (and `df.low`/`df.high`), which don't exist under those names. So the documented `col_names` argument crashes for every value except `'vlh'`. If it didn't crash, the renamed low/high columns would also be ignored. (`Results.to_df(resample=...)` isn't affected, because it doesn't forward `col_names` into `resample()`.)

```python
import starsim as ss
sim = ss.Sim(n_agents=2000, diseases='sis', networks='random', start=2000, dur=ss.years(3), dt=ss.month, verbose=0)
sim.run()
r = sim.results.sis.new_infections
r.resample('year', col_names=None)
r.resample('year', col_names='foo')
```

Actual:

```
col_names='vlh': ok [11. 14. 36.  0.]
col_names=None failed: AttributeError: 'dataframe' object has no attribute 'value'
col_names='foo' failed: AttributeError: 'dataframe' object has no attribute 'value'
```

Expected: the same resampled Result as with the default (the column naming only affects the intermediate dataframe).

Blast radius: direct callers of `Result.resample()` or `Result.to_series(resample=..., col_names=...)` who pass `col_names`. Low usage, but it is a documented argument that can never work.

**Fix**: In `resample()`, always call `self.to_df(col_names='vlh', ...)` internally, since `from_df()` needs those names. Alternatively, remove the argument from `resample()`.

## Low severity

### 4. `Result.plot(fig=fig)` draws on the current figure, not `fig` — `results.py:380`

If `fig` is given but `ax` isn't, `ax = plt.subplot(111)` creates the axes on matplotlib's *current* figure, which isn't necessarily `fig`. The plot then goes on another figure, while the empty `fig` is returned.

```python
import matplotlib.pyplot as plt
fig1 = plt.figure(); fig2 = plt.figure()
sim.results.sis.prevalence.plot(fig=fig1)
print(len(fig1.axes), len(fig2.axes))
```

Actual: `axes in fig1: 0 axes in fig2: 1`

Expected: `axes in fig1: 1 axes in fig2: 0`

Blast radius: users who pass their own figure to `Result.plot()`. `Results.plot()` passes `ax`, so it isn't affected.

**Fix**: Use `ax = fig.add_subplot(111)` in place of `plt.subplot(111)`.

## Verified clean

The following were hypothesized and tested, and found correct: `Result.annualize()` for `sum`/`mean`/`last` against manual per-year aggregation. For `int`-dtype `n_*` results, the mean is *not* truncated, because `init_values()` skips the dtype cast when values are supplied. `Result.resample()` for `sum`/`last`, agreeing with `annualize()`, and restoring the original `timevec` afterwards; resampling MultiSim results with `low`/`high` bounds; `Result.to_df()` with bounds; `Results.to_df()` shallow, `descend=True`, and with `resample='year'`; `sim.to_df(resample='year')`; `MultiSim.results.to_df(resample='year')`; `Results.flatten()` with `only_auto`/`resample`; `summary_method()` heuristics on the built-in result names (including `cum_*`, `n_*`, and msim-prefixed names); `Result.key`/`full_label`/`has_dates`/`convert_timevec()`; and `Results.__repr__`.
