# `loop.py` bug audit

Audit of `starsim/loop.py` (all 726 lines: `LoopEntry`, and `Loop` plan construction, execution, insertion, dataframe export, plotting, and deep copy) for real, unambiguous bugs only: wrong results on in-contract input, documented arguments that silently do nothing, stale state, and crashes on normal usage. Style, docstrings, performance, test gaps, and contrived corner cases are out of scope. **Method**: line-by-line reading, then a repro script run against the editable install for every candidate (Starsim 3.6.1 working tree on branch `rc3.6.2`, commit `3d8dc9d5` plus Cliff's uncommitted changes; numpy 2.4.6, sciris 3.3.0, `MPLBACKEND=agg`). Repro scripts are in the session scratchpad under `loop/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `Loop.insert()` / `_insert_into_plan()` | An inserted function is bound to the original sim by a lambda closure, so after `sim.copy()` (or in a serial `MultiSim`) it runs on the wrong sim | 431 |
| 2 | Low | `Loop.plot_cpu()` | `bytime=False` places bars at the function order of a *different* entry, so the plot is in scrambled order | 612-613 |
| 3 | Low | `Loop.plot()` | Crashes after two or more functions have been inserted with `insert()` | 570-571 |
| 4 | Low | `Loop.run()` / `to_df()` | Profiling a sim that is run in more than one call (`until=`) gives an all-`NaN` `cpu_time` column | 392-393, 526-530 |
| 5 | Low | `Loop.run()` | `verbose=True` prints `t=n` instead of the time for date-based sims | 397 |

## Medium severity

### 1. An inserted function is bound to the original sim by a lambda closure, so after `sim.copy()` (or in a serial `MultiSim`) it runs on the wrong sim — `loop.py:431`

`_insert_into_plan()` wraps the user function as `sim_func = lambda: func(self.sim)` and stores that lambda in each new `LoopEntry`. `copy.deepcopy` treats plain functions as atomic, so when the sim (and with it `Loop.__deepcopy__`, the plan and every `LoopEntry`) is deep-copied, the copy's plan still holds the *same* lambda, whose closure cell is the *original* `Loop`. The copied sim therefore runs the inserted function against the original sim, never against itself. Regular plan entries are bound methods, which `deepcopy` does rebind, so only inserted entries are affected. Pickling (as in a parallel `MultiSim`) happens to keep the closure consistent, so the same sim gives different results with `parallel=True` and `parallel=False`.

```python
import starsim as ss

def record(sim):
    sim.metadata.setdefault('calls', 0)
    sim.metadata['calls'] += 1

sim = ss.Sim(n_agents=500, start=2000, stop=2010, diseases='sis', networks='random', verbose=0)
sim.init()
sim.loop.insert(record, label='sim.finish_step')
sim2 = sim.copy()
sim2.run()
print('copy  calls:', sim2.metadata.get('calls'))
print('orig  calls:', sim.metadata.get('calls'))

# Same via MultiSim
sim = ss.Sim(n_agents=500, start=2000, stop=2010, diseases='sis', networks='random', verbose=0)
sim.init()
sim.loop.insert(record, label='sim.finish_step')
msim = ss.MultiSim(sim, n_runs=3)
msim.run(parallel=False)
print('msim calls:', [s.metadata.get('calls') for s in msim.sims], 'base:', sim.metadata.get('calls'))
```

Actual:

```
copy  calls: None
orig  calls: 11
msim calls: [None, 11, 22] base: 33
```

The same `MultiSim` with `parallel=True` gives `[11, 11, 11]`.

Expected: `copy calls: 11`, `orig calls: None`, and `[11, 11, 11]` for the serial `MultiSim` with the base sim untouched.

**Blast radius**: anyone who uses `sim.loop.insert()` (the documented way to hook into an arbitrary point of the loop, e.g. the `update_betas` docstring example) and then copies the sim, which includes `ss.MultiSim(..., parallel=False)`, `ss.parallel()` with serial execution, and any scenario workflow that copies an initialized base sim. An intervention-like inserted function (as in the docstring example) silently modifies the wrong sim, so the runs that are reported are not the runs that were intervened on.

**Fix**: don't close over `self`. Either bind the sim as data that `deepcopy` follows, e.g. `sim_func = ft.partial(func, self.sim)` (the partial's `args` are deep-copied through the shared memo, so they resolve to the copied sim), or store the raw `func` on the entry with a flag and have `run()`/`run_one_step()` call `entry.func(self.sim)` for inserted entries.

## Low severity

### 2. `bytime=False` places bars at the function order of a *different* entry, so the plot is in scrambled order — `loop.py:612-613`

`cpu_df` is sorted by `cpu_time`. With `bytime=False`, `y = df.func_order.values` and then `y = y[::-1]`, but `x` (widths) and `ylabels` are not reversed. So entry `i` (bar width `x[i]`, label `ylabels[i]`) is drawn at `y[i]`, the func order of entry `n-1-i`. Bars and their tick labels stay paired (both indexed by `i`), but the vertical position is the func order of an unrelated function, so the plot is neither in time order nor in the documented "actual order". The `y[::-1]` reversal is only correct for the `bytime=True` branch, where `y` is `arange`.

```python
import starsim as ss

sim = ss.Sim(n_agents=2000, start=2000, stop=2010, diseases='sis', networks='random', verbose=0)
sim.run(profile=True)
fig = sim.loop.plot_cpu(bytime=False, max_entries=None)
ax = fig.axes[0]
bars = sorted([(p.get_y() + p.get_height()/2, p.get_width()) for p in ax.patches])
ticks = {t: l.get_text().split('(')[0] for t, l in zip(ax.get_yticks(), ax.get_yticklabels())}
cdf = sim.loop.cpu_df
for y, w in bars:
    label = ticks[y]
    print(f'{y:5.1f} -> {label:28s} | {cdf.loc[label].func_order}')
```

Actual (y position, label drawn there, that label's real func order):

```
  0.0 -> people.step_die              | 6
  1.0 -> randomnet.update_results     | 8
  2.0 -> randomnet.finish_step        | 10
  3.0 -> people.finish_step           | 12
  4.0 -> sis.finish_step              | 11
  5.0 -> sim.finish_step              | 13
  6.0 -> sim.start_step               | 0
  ...
```

Expected: each function drawn at its own func order (the position and the last column equal, or consistently mirrored).

**Blast radius**: anyone using `plot_cpu(bytime=False)` to see CPU time in loop order; the default `bytime=True` is correct.

**Fix**: in the `bytime=False` branch, use `y = -df.func_order.values` (or keep `y = df.func_order.values` and invert the axis) rather than reversing the array, so each bar keeps its own func order; apply `[::-1]` only in the `bytime=True` branch.

### 3. `Loop.plot()` crashes after two or more functions have been inserted with `insert()` — `loop.py:570-571`

`plot()` builds the ticks with `yticks = df.func_order.unique()` and `ylabels = df.label.unique()`, assuming the two columns are 1:1. Inserted entries all have `func_order=None` but distinct labels (the function name), so every insertion after the first adds a label without adding a tick, and `plt.yticks(yticks, ylabels)` fails.

```python
import starsim as ss

def f1(sim): pass
def f2(sim): pass

for funcs in [[f1], [f1, f2]]:
    sim = ss.Sim(n_agents=500, start=2000, stop=2003, diseases='sis', networks='random', verbose=0)
    sim.init()
    for f in funcs:
        sim.loop.insert(f, label='sis.step')
    try:
        sim.loop.plot()
        print(len(funcs), 'insertions: plot OK')
    except Exception as e:
        print(len(funcs), 'insertions:', type(e).__name__, e)
```

Actual:

```
1 insertions: plot OK
2 insertions: ValueError The number of FixedLocator locations (15), usually from a call to set_ticks, does not match the number of labels (16).
```

Expected: the plan diagram, as with no or one insertion. (Even with one insertion, the inserted entry has `y=None` and is silently not drawn.)

**Blast radius**: users of `insert()` who then call `sim.loop.plot()` to check where their functions landed, which is the obvious way to debug an insertion.

**Fix**: derive ticks and labels from the same de-duplicated pairs, e.g. `pairs = df[['func_order', 'label']].drop_duplicates()`, and give inserted entries a plottable y (e.g. the func order of the entry they were inserted next to, plus 0.5).

### 4. Profiling a sim that is run in more than one call (`until=`) gives an all-`NaN` `cpu_time` column — `loop.py:392-393, 526-530`

`run()` appends a starting timestamp to `cpu_time` at the start of *every* call (line 393) as well as after every entry. `to_df()` computes `np.diff(self.cpu_time)` and only fills `cpu_time` if the length equals the plan length exactly; otherwise it silently sets the whole column to `NaN`. After a split run there is one extra timestamp per extra `run()` call, so the lengths never match.

```python
import starsim as ss

sim = ss.Sim(n_agents=500, start=2000, stop=2010, diseases='sis', networks='random', verbose=0)
sim.run(profile=True)
print('single run:  n NaN cpu_time =', sim.loop.to_df().cpu_time.isna().sum(), 'of', len(sim.loop.plan))

sim = ss.Sim(n_agents=500, start=2000, stop=2010, diseases='sis', networks='random', verbose=0)
sim.run(until=2005, profile=True)
sim.run(profile=True)
print('split run:   n NaN cpu_time =', sim.loop.to_df().cpu_time.isna().sum(), 'of', len(sim.loop.plan), '; len(cpu_time) =', len(sim.loop.cpu_time))
```

Actual:

```
single run:  n NaN cpu_time = 0 of 154
split run:   n NaN cpu_time = 154 of 154 ; len(cpu_time) = 156
```

Expected: 0 `NaN` values for the split run as well (timing for all 154 entries).

**Blast radius**: profiling of sims run with `until=` or stepped with `sim.run_one_step()`; single-call runs are fine.

**Fix**: record per-entry durations rather than raw timestamps (take a timestamp before and after `entry.func()` and append the difference, keyed by `self.index`), so repeated `run()` calls simply continue filling the array; `to_df()` then uses it directly.

### 5. `verbose=True` prints `t=n` instead of the time for date-based sims — `loop.py:397`

The verbose line is `f'Running t={entry.time:n}, ...'`. The `n` format spec is a number format; for an `ss.date` it is passed through to date formatting, where `n` is not a directive, so the literal `n` is printed. Year- and duration-based sims print correctly.

```python
import starsim as ss
for start in [2000, '2000-01-01', 0]:
    sim = ss.Sim(n_agents=200, start=start, dur=3, diseases='sis', networks='random', verbose=0)
    sim.init()
    sim.run_one_step(verbose=True)
```

Actual (first line for each):

```
Running t=2000, step=0, sim.start_step()
Running t=n, step=0, sim.start_step()
Running t=0, step=0, sim.start_step()
```

Expected: `Running t=2000.01.01, step=0, sim.start_step()` (or similar) for the date-based sim.

**Blast radius**: only the debug output of `loop.run(verbose=True)`/`sim.run_one_step(verbose=True)`, which is its sole purpose, for date-based sims.

**Fix**: use `{entry.time}` (i.e. `str()`), or `:n` only when the time is numeric.

## Verified clean

Tested with repro scripts and found correct: `run(until=...)` with `until` given as an int, a string, and an `ss.date`, for both year-based and date-based sims (all stop consistently after the `until` step and resume correctly); `insert()` via both `label` and a boolean/indices `match_fn` (placement before/after), and replay of insertions; the uniform (no-sort) and sorted plan paths, including the per-entry `ti` computed from `sim.start_step` boundaries with modules on coarser timesteps; `_null_update_results()` skipping only true no-ops; `to_df()`/`cpu_df` for a single profiled run; `run_one_step()`; parallel `MultiSim` with an inserted function. Read line by line with no candidate found: `collect_funcs()` ordering against the documented loop order, `collect_abs_tvecs()` keys (including `People` subclasses), `_timelines_uniform()`, `_warn_near_identical()`, `plot_step_order()`, and `__deepcopy__`.
