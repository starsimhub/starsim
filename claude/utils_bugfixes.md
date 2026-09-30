# `utils.py` bug audit

Audit of `starsim/utils.py` (all 872 lines of the working-tree version, including Cliff's uncommitted changes) on branch `rc3.6.2` at commit `3d8dc9d5`, using Starsim 3.6.1 (editable install), numpy 2.4.6, pandas 3.0.5. Method: line-by-line reading, then a repro script for every candidate, run against the editable install with `MPLBACKEND=agg` (scripts in the session scratchpad under `utils/`). Only defects the repro demonstrated, and that are not documented or intended behavior, are recorded. Style, docs, performance, and contrived corner cases are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `match_result_keys()` (via `MultiSim.plot()`) | Plotting a reduced MultiSim permanently deletes every `auto_plot=False` result from `msim.results` | 789, 797 |
| 2 | Medium | `match_result_keys()` (via `sim.plot(key)`) | An exact result key fails with "not found" whenever it is also a substring of another key, e.g. `sim.plot('new_deaths')` with any disease that has `new_deaths` | 801-802 |
| 3 | Low | `shrink()` | Crashes with `UnboundLocalError` if none of the requested attributes exist, although missing attributes are an expected, documented case | 684 |

## High severity

### 1. `MultiSim.plot()` deletes non-auto-plot results from `msim.results` — `utils.py:789`

With `flattened=True` (which `MultiSim.plot()` uses for a reduced MultiSim, `run.py:472`), `match_result_keys()` sets `flat = results`, i.e. the caller's own `msim.results` object, not a copy. Then, when `show_skipped` is falsy (the default), line 797 does `flat.pop(k)` for every result with `auto_plot=False`. This removes them from `msim.results` in place. `flat_orig = flat` ("Copy reference before we modify in place") only copies the reference, so it doesn't protect anything. For `Sim.plot()` the dict is a fresh `results.flatten()`, so only MultiSim is affected.

```python
import starsim as ss
sim = ss.Sim(n_agents=1000, diseases='sis', networks='random', dur=10, verbose=0)
msim = ss.MultiSim(sim, n_runs=3)
msim.run(parallel=False)
msim.reduce()
before = list(msim.results.keys())
msim.plot()
after = list(msim.results.keys())
print('missing after plot:', [k for k in before if k not in after])
msim.results['n_alive']
```

Actual:

```
n results before plot: 12
auto_plot=False results: ['randomnet_n_edges', 'n_alive', 'n_female', 'new_deaths', 'new_emigrants', 'cum_deaths']
n results after plot: 6
missing after plot: ['randomnet_n_edges', 'n_alive', 'n_female', 'new_deaths', 'new_emigrants', 'cum_deaths']
Access failed: KeyNotFoundError
```

Expected: `msim.results` is unchanged by plotting, so all 12 results are still there.

Blast radius: anyone who plots a reduced MultiSim (a very common workflow: `msim.reduce(); msim.plot()`) and then reads or exports its results. `n_alive`, `new_deaths`, `cum_deaths`, and network `n_edges` silently disappear from `msim.results`, `msim.results.to_df()`, and later calls like `msim.plot('n_alive')`.

**Fix**: Copy before filtering, e.g. `flat = sc.objdict(results) if flattened else results.flatten()` (or build the filtered dict with a comprehension, not `pop()`). Then `flat_orig` would also keep the full key list for the error message, as its comment intends.

## Medium severity

### 2. An exact result key is rejected if it's a substring of another key — `utils.py:801`

For a string key, `match_result_keys()` keeps every flattened key that *contains* `normkey(key)`, then raises `KeyNotFoundError` unless exactly one is left. An exact match gets no priority. So if the sim-level `new_deaths` and a disease's `<disease>_new_deaths` both exist, `sim.plot('new_deaths')` matches both and fails with "not found", even though `new_deaths` is a valid key and is listed in the error message. `ss.NCD`, `ssl.diseases.Cholera`, and `ssl.diseases.HIV` all define `new_deaths` (Cholera also defines `cum_deaths`), so this happens in ordinary models. The list form, `sim.plot(['new_deaths'])`, works because it uses exact lookup.

```python
import starsim as ss
import starsim.library as ssl
sim = ss.Sim(n_agents=1000, diseases=ssl.diseases.Cholera(), networks='random', dur=10, verbose=0)
sim.run()
sim.plot('new_deaths')
```

Actual:

```
['cholera_new_deaths', 'cholera_cum_deaths', 'new_deaths', 'cum_deaths']
sim.plot('new_deaths') failed: KeyNotFoundError: Key "new_deaths" not found; valid keys are:
sim.plot('cum_deaths') failed: KeyNotFoundError: Key "cum_deaths" not found; valid keys are:
sim.plot(['new_deaths']) ok
```

Expected: `sim.plot('new_deaths')` plots the sim-level `new_deaths` result.

Blast radius: any sim with NCD, Cholera, or HIV (or a custom disease with a `new_deaths`/`cum_deaths` result) whose user plots a sim-level death result by name. The same happens for any custom result name that is a substring of another.

**Fix**: Check for an exact match first: `if normkey(key) in flat: flat = {normkey(key): flat[normkey(key)]}`. Only fall back to substring matching if there is no exact match.

## Low severity

### 3. `shrink()` raises `UnboundLocalError` when no requested attribute exists — `utils.py:684`

`shrunk` is only assigned inside `if hasattr(obj, attr)`. If none of `attrs` exist on `obj`, `return shrunk` raises `UnboundLocalError`. The `verbose` argument ("print warnings about missing attributes") shows that missing attributes are an expected input, so they should be skipped, not cause a crash. In the library, `Module.shrink()` wraps its per-state call in `sc.tryexcept()`, which hides this error.

```python
import starsim as ss
class Obj: pass
o = Obj(); o.a = 1
ss.shrink(o, ['a', 'b'])   # fine
ss.shrink(o, ['missing'])  # crashes
```

Actual: `shrink missing: UnboundLocalError cannot access local variable 'shrunk' where it is not associated with a value`

Expected: no error; nothing is shrunk (and a warning is printed if `verbose=True`).

Blast radius: users writing custom `shrink()` methods for modules whose attributes are only sometimes present.

**Fix**: Initialize `shrunk = None` before the loop (or return `obj`).

## Verified clean

The following were hypothesized and tested, and found correct: `parse_age_range()` and `apply_age_range()` on all documented formats (`'5-9'`, `'5 to 9'`, `'<5'`, `'95+'`, `'>95'`, and all four bracket forms, with the documented inclusive/exclusive bounds; `'5-9'` → `[5, 9)` is the documented convention); `standardize_data()` with a sex mapping, the `Time`→`Year` synonym, `min_year` truncation, and the `-inf` out-of-range age rows; `validate_sim_data()` time-column indexing; `ndict.get()` by name, case-insensitive name, `match_case=True`, and type; `ndict.__add__`/`copy`/`__iadd__` (the original isn't mutated); `warn()` in `'print'` mode; `plot_args()` merging of defaults, shortcuts, `*_kw` dicts, and renamed data keys; `get_result_plot_label()` for `None`/`True`/`False`/`-1`/int; and `match_result_keys()` on every exact key of an SIR+demographics sim, where there are no substring collisions. `combine_rands()`, `find_contacts()`, `nlist_to_dict()`, `resolve_data_col()`, `load()`/`save()`, and `return_fig()` were read and found correct. `plotting_kw` lists `ncols` under both `fig` and `legend`, so the `ncols` shortcut always goes to the figure. This was noted but not recorded as a finding, since the legend can still be set via `legend_kw` and the shortcut is not documented for legends.
