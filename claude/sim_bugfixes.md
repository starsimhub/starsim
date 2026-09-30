# `sim.py` bug audit

Audit of `starsim/sim.py` (all 1250 lines: `Sim`, `AlreadyRunError`, `demo()`, `diff_sims()`, `check_sims_match()`) for real, unambiguous bugs only. Style, naming, docs typos, test gaps, performance, and contrived corner cases are out of scope. **Method**: I read every line, formed hypotheses, and ran each one as a repro script against the editable install (Starsim 3.6.1 working tree on branch `rc3.6.2`, commit `3d8dc9d5` plus Cliff's uncommitted changes, numpy 2.4.6, sciris 3.3.0, `MPLBACKEND=agg`). A finding is recorded only if the repro showed the wrong behaviour and the code, docstrings, and tests show it is not intended. Scripts are in the session scratchpad under `sim/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `Sim.summarize()` | Cumulative results named `cumulative` (e.g. `births.cumulative`, `deaths.cumulative`) are summarized as the time-mean, not the final value | 607 |
| 2 | Medium | `Sim.init()` / `Sim.init_people()` | `sim.init()` cannot be called a second time, though `AlreadyRunError` tells users to "call sim.init() to re-run" | 301, 456, 513 |
| 3 | Medium | `Sim.save()` / `Sim.shrink()` | Saving (or shrinking) a sim that has not been initialized crashes with `UnboundLocalError` | 836, 657 (root cause `utils.py:684`) |
| 4 | Low | `Sim.check_method_calls()` | `verbose=True` crashes: `Sim` has no `_call_required` | 732 |
| 5 | Low | `demo()` | `plot=True` (and `show`) silently do nothing when `summary=False` | 1052-1058 |
| 6 | Low | `check_sims_match()` | Its own docstring example (unrun sims) crashes with `IndexError` | 1232-1236, 1098 |
| 7 | Low | `diff_sims()` | `diff` and `ratio` become NaN, and `change` becomes "N/A", whenever the sim1 value is 0 or negative | 1156 |

## Medium severity

### 1. Cumulative results named `cumulative` are summarized as the time-mean, not the final value — `sim.py:607`

The default `how` mapping picks a summary function by substring match on the flattened result key: `{'n_':'mean', 'new_':'mean', 'cum_':'last', 'timevec':'last', '':'mean'}`. `ss.Births` and `ss.Deaths` name their cumulative result `cumulative` (flattened to `births_cumulative` / `deaths_cumulative`). That key does not contain `cum_`, so it falls through to `''` and gets `mean`. The `Result` objects even declare `summarize_by='last'` (`demographics.py:87`, `:301`), but `summarize()` ignores it. As a result `sim.summary` reports the average of a running total, which is meaningless.

```python
sim = ss.Sim(n_agents=1000, diseases='sis', networks='random', demographics=True, verbose=0).run()
for k in ['births_cumulative', 'deaths_cumulative', 'cum_deaths']:
    res = sc.flattendict(sim.results, sep='_')[k]
    print(k, 'summary =', sim.summary[k], ' last =', res[-1], ' mean =', res.mean(), ' summarize_by =', res.summarize_by)
```

Actual:

```
births_cumulative summary = 594.1960784313726  last = 1278.0  mean = 594.1960784313726  summarize_by = last
deaths_cumulative summary = 301.37254901960785  last = 655.0  mean = 301.37254901960785  summarize_by = last
cum_deaths summary = 655.0  last = 655.0  mean = 301.37254901960785  summarize_by = None
```

Expected: `births_cumulative` = 1278 and `deaths_cumulative` = 655, the final values, matching `cum_deaths`.

Blast radius: every sim with `demographics=True`, `birth_rate`, `death_rate`, or explicit `ss.Births`/`ss.Deaths`. The wrong values show up in `sim.summary`, `print(sim.summary)` in `ss.demo()`, `sim.to_json()`, `diff_sims()`/`check_sims_match()`, and MultiSim summaries built from sim summaries. Any other module result named `cumulative*` without `cum_` is also affected.

**Fix**: When `how='default'`, use the `Result`'s own `summarize_by` when it is set (`'last'`, `'mean'`, `'sum'`), and only fall back to the substring map when it is not. A narrower alternative is to add `'cumulative':'last'` to the default map, but the first fix also covers future results.

### 2. `sim.init()` cannot be called a second time, though `AlreadyRunError` tells users to — `sim.py:301`

Both `start_step()` (line 456) and `run()` (line 513) raise `AlreadyRunError('Simulation is already complete (call sim.init() to re-run)')`. But the first `init()` consumes its inputs. `init_people()` does `self.pars.pop('people')` (line 301). On the second `init()`, `pars.validate()` → `validate_agents()` reads `self.people` and gets `AttributeError`. This also happens on `init(force=True)`, and when `init()` is called twice without running. Other steps would fail next even without that one. `validate_total_pop()` then raises because `total_pop` and `pop_scale` are both filled in by the first validation (`parameters.py:326`). `init_module_attrs()` also replaces `pars.diseases` etc. with objdicts of `Pars`.

```python
kw = dict(n_agents=1000, diseases='sis', networks='random', verbose=0)
sim = ss.Sim(**kw).run()
sim.run()                     # AlreadyRunError: ... (call sim.init() to re-run)
sim.init(); sim.run()         # follow the advice

s = ss.Sim(**kw); s.init(); s.init()
s = ss.Sim(**kw); s.run(); s.init(force=True)
```

Actual:

```
AlreadyRunError Simulation is already complete (call sim.init() to re-run)
Did you mean to copy the sim before running it?
AttributeError 'SimPars' object has no attribute 'people'
init twice AttributeError 'SimPars' object has no attribute 'people'
run, init(force=True) AttributeError 'SimPars' object has no attribute 'people'
```

Once `people` is restored, the next layer also fails:

```python
s = ss.Sim(n_agents=1000, total_pop=1e6, verbose=0); s.init()
s.pars.people = None          # undo the pop() to get past the first failure
s.pars.validate_sim_pars()
```

```
validate_sim_pars again: ValueError You can define total_pop (1000000.0) or pop_scale (1000.0), but not both, since one is calculated from the other
```

Expected: either `sim.init()` re-initializes the sim so it can run again, or the error message stops recommending it.

Blast radius: anyone who follows the error message's advice, which is the only recovery path it offers. The AlreadyRunError default hint ("Did you mean to copy the sim before running it?") is the path that actually works.

**Fix**: Simplest and honest: change the two messages to recommend copying the sim before running it (e.g. `sim.copy()` or `ss.Sim(**original_pars)`) and drop the `sim.init()` advice. Making re-init work would take several changes: read `people` without popping it (or guard `validate_agents` with `self.get('people')`), record whether `total_pop`/`pop_scale` were user-supplied so validation is idempotent, and make `init_module_attrs` re-use the modules already on the sim.

### 3. Saving or shrinking an uninitialized sim crashes with `UnboundLocalError` — `sim.py:836` (root cause `utils.py:684`)

`save()` defaults to `shrink=True` unless the sim is mid-run (line 827). That means an uninitialized sim is always shrunk, via `self.shrink(inplace=False)` → `ss.shrink(sim, 'people')` (line 657). An uninitialized sim has no `people` attribute, so the loop in `ss.utils.shrink()` never assigns `shrunk`, and `return shrunk` raises. The same applies to calling `sim.shrink()` directly before `init()`. The comment at line 659 ("If the sim is not initialized, we're done") shows that this path is meant to work.

```python
s = ss.Sim(n_agents=1000, diseases='sis', networks='random', verbose=0)
s.save('u.sim')
s.shrink()
```

Actual:

```
  File "/home/cliffk/idm/starsim/starsim/sim.py", line 836, in save
    sim = self.shrink(inplace=False) if shrink else self
  File "/home/cliffk/idm/starsim/starsim/sim.py", line 657, in shrink
    ss.shrink(sim, 'people')
  File "/home/cliffk/idm/starsim/starsim/utils.py", line 684, in shrink
    return shrunk
UnboundLocalError: cannot access local variable 'shrunk' where it is not associated with a value
shrink: UnboundLocalError cannot access local variable 'shrunk' where it is not associated with a value
```

Expected: the sim is saved (and can be loaded and run).

Blast radius: anyone who saves a configured-but-unrun sim, e.g. to ship a scenario definition or to run it later on a cluster. Saving after `run()` works, because `people` exists then.

**Fix**: In `ss.utils.shrink()`, initialize `shrunk = None` before the loop, or return `obj` in the `obj`/`attrs` form, since it works in place. This probably belongs to the `utils.py` audit; it is listed here because `Sim.save()` is where users hit it. Alternatively (or as well), `Sim.shrink()` could skip `ss.shrink(sim, 'people')` when `not sim.initialized`.

## Low severity

### 4. `check_method_calls(verbose=True)` crashes — `sim.py:732`

The documented `verbose` argument ("whether to print the number of times each method was called") runs `sc.pp(self._call_required)`. `_call_required` exists only on modules (`modules.py:385`), never on the `Sim`.

```python
sim = ss.Sim(n_agents=1000, diseases='sis', networks='random', verbose=0).run()
sim.check_method_calls(verbose=True)
```

Actual: `AttributeError 'Sim' object has no attribute '_call_required'`. Expected: the per-module call counts are printed.

Blast radius: anyone debugging a "required methods were not called" warning, which is exactly when this option is useful.

**Fix**: Print per module, e.g. `sc.pp({mod.name: mod._call_required for mod in self.modules})`.

### 5. `demo(plot=True)` does not plot when `summary=False` — `sim.py:1052-1058`

`if plot:` (and `if show:` inside it) is nested inside `if summary:`. So `summary=False` silently disables plotting, although the docstring describes `plot` and `summary` as independent options.

```python
plt.close('all')
sim = ss.demo(n_agents=1000, verbose=0, summary=False, plot=True, show=False)
print('figures:', plt.get_fignums())
```

Actual: `figures: []`. Expected: `figures: [1]`.

Blast radius: small. It affects users who want the demo plot without the printed summary.

**Fix**: Dedent the `if plot:` block one level so it sits under `if run:` rather than `if summary:`.

### 6. `check_sims_match()` docstring example crashes — `sim.py:1232-1236`

The example builds three sims and calls `ss.check_sims_match(s1, s2, s3)` without running them. `diff_sims()` calls `summarize()` on each sim. An unrun sim has no results, so the summary is empty, and `multi = isinstance(sim1[0], dict)` (line 1098) indexes an empty objdict.

```python
s1 = ss.Sim(diseases='sir', networks='random')
s2 = ss.Sim(pars=dict(diseases='sir', networks='random'))
s3 = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet())
ss.check_sims_match(s1, s2, s3)
```

Actual: `IndexError index 0 out of range for dict of length 0`. Expected (per the example's `assert`): `True`. `tests/test_sim.py:46` runs the sims first, which is why the tests pass.

Blast radius: people copying the docstring example. The docstring also appears in the API reference.

**Fix**: Add `.run()` to the three sims in the example. Optionally, have `diff_sims()` raise a clear "sim has not been run" error when given an unrun `Sim`.

### 7. `diff_sims()` reports NaN diff and "N/A" change whenever the sim1 value is 0 or negative — `sim.py:1156`

The branch that computes `diff`, `ratio`, and the change arrows is guarded by `numeric and old > 0`. Any summary value that is 0 in sim1 falls through to the "non-numeric" `else` branch, which reports `diff=NaN, ratio=NaN, change='N/A'`. Zero is common, e.g. `new_deaths` or `cum_infections` in a baseline without deaths or transmission. `new - old` and the direction are well defined there. Only the ratio is undefined.

```python
s1 = dict(a=0.0, b=5.0, c=-2.0)
s2 = dict(a=3.0, b=6.0, c=-4.0)
print(ss.diff_sims(s1, s2, output=True))
```

Actual:

```
   sim1  sim2  diff  ratio change
a   0.0   3.0   NaN    NaN    N/A
b   5.0   6.0   1.0    1.2     ↑↑
c  -2.0  -4.0   NaN    NaN    N/A
```

Expected: `a` has `diff=3.0` and an up arrow (ratio may be NaN/inf), and `c` has `diff=-2.0` and a down arrow.

Blast radius: users comparing a scenario against a zero baseline. The mismatch is still detected and counted (`n_mismatch` is computed earlier), so `check_sims_match()` and `die=True` are unaffected. Only the printed/returned table is uninformative.

**Fix**: Guard on `numeric` only. Compute `this_diff` always. Compute `this_ratio` only when `old != 0` (else NaN). For the non-multi case, when the ratio is undefined or negative, choose the arrow from the sign of `this_diff`.

## Verified clean

I tested these and found them correct:

- **`get_modules()`/`get_module()`**: type queries, `*` prefix/suffix/both wildcards, `match_case`, and the uninitialized search over `pars`.
- **`finalize_results()`**: scaling by `pop_scale` exactly once (guarded by `results_ready`), and the `auto_plot=False` detection comparing against the pre-scaled array.
- **`run(until=...)` partial runs followed by `run()`**: the `complete` detection via `loop.index == len(loop.plan)`, and the `AlreadyRunError` guard. The `verbose=` override persisting across a partial run is by design, since it is restored in `finalize()`.
- **`start_step()` progress printing** for fractional, `>=1`, and `'brief'` (-1) verbosity.
- **`init_module_attrs()`**: moving modules to the sim and replacing `pars[modtype]` with module `Pars`.
- **Other `Sim` methods**: `products()` de-duplication, the `modules` iterator order, `label` get/set via `pars`, `__getitem__`/`__setitem__`, `__repr__` before and after init, `to_json()`/`to_yaml()` on a run sim, and `to_df()`.
- **`shrink()`**: `inplace=False` leaves the original sim intact, and the size check works on an initialized sim.
- **`diff_sims()`** on two MultiSims (mean/sem columns, zscore/statsig) and on plain summary dicts with positive values.
- **`AlreadyRunError`**: message concatenation.
- **`plot()`**: the data-column lookup via `label_to_name`.
