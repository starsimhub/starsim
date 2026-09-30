# `run.py` bug audit

Audit of `starsim/run.py` (671 lines, read in full: `MultiSim`, `single_run()`, `multi_run()`, `parallel()`) at commit `3d8dc9d5` (branch `rc3.6.2`, file unmodified in the working tree), using Starsim 3.6.1 (editable install), numpy 2.4.6, sciris 3.3.0. Every finding was reproduced with scripts in the session scratchpad (`run/run_repros.py`, `run/shrink_default.py`, `run/clean_checks.py`) run under `MPLBACKEND=agg` with 1000-agent SIS sims. Intent was checked against the docstrings, the tests, `docs/whatsnew.qmd`, and git history where relevant.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `single_run()` | Reseeding an already-initialized sim only changes `pars['rand_seed']`, so all "replicates" are identical | 540 |
| 2 | Medium | `MultiSim.run()` | `debug=True` with a single sim ignores `n_runs` and runs just one sim | 192, 212 |
| 3 | Low | `MultiSim.plot()` | `fig=` is silently discarded for an unreduced MultiSim | 434 |
| 4 | Low | `MultiSim.plot()` | Passing `alpha=` crashes for an unreduced MultiSim (`got multiple values for keyword argument 'alpha'`) | 452 |
| 5 | Low | `MultiSim.reduce()` | A second `reduce()`/`mean()`/`median()` overwrites `orig_base_sim` with the first reduced sim, so `reset()` no longer restores the original | 349 |
| 6 | Low | `multi_run()` | `shrink=None` default means "don't shrink", which inverts the pre-rename `keep_people=None` default and differs from the `debug=True` path | 562 |

## Medium severity

### 1. Reseeding an already-initialized sim has no effect, so all replicates are identical — `run.py:540`

`single_run()` implements reseeding as `sim.pars['rand_seed'] += ind`. The seed is only used in `Sim.init()` (to seed `np.random` and the `Dist` objects). If the sim passed to `MultiSim`/`multi_run` has already been initialized, `sim.run()` does not re-initialize it. The replicates then run with identical random streams while reporting different `rand_seed` values. The same applies to any `sim_args` (line 547), which silently have no effect on an initialized sim.

```python
sim = make()     # ss.Sim(n_agents=1000, diseases=ss.SIS(beta=0.05), networks='random', dur=10, verbose=0)
sim.init()
msim2 = ss.MultiSim(sim, n_runs=3, parallel=False)
msim2.run()
print('rand_seed pars:', [s.pars.rand_seed for s in msim2.sims])
print('final infected:', [int(s.results.sis.n_infected[-1]) for s in msim2.sims])
```

Actual:

```
rand_seed pars: [1, 2, 3]
final infected: [303, 303, 303]
```

Expected: different trajectories per replicate (as with an uninitialized sim, e.g. `[303, 317, 272]` from `MultiSim(make(), n_runs=3, initialize=True)`), or an error/warning.

Blast radius: anyone who calls `sim.init()` (e.g. to inspect or modify modules) before handing the sim to `MultiSim` or `ss.multi_run()`. Uncertainty bands collapse to zero width and nothing warns about it.

**Fix**: in `single_run()`, when `reseed` or `sim_args` would change an initialized sim, either raise a clear error, or re-seed the distributions directly (e.g. `sim.dists.init(...)`/`sim.dists.reset()` with the new seed). Re-initializing with `sim.init(force=True)` is another option, at the cost of discarding any post-init modifications.

### 2. `debug=True` with a single sim ignores `n_runs` — `run.py:192`, `run.py:212`

The `MultiSim` docstring says `debug (bool): if True, run in serial`. For a single base sim, `run()` sets `sims = [self.base_sim]` for the debug loop and `run_target = self.base_sim` for `multi_run()`. The comment on line 193 notes that passing a one-element list would ignore `n_runs`. But the debug branch iterates over that one-element `sims` list and pops `n_runs`, so it runs exactly one sim.

```python
msim = ss.MultiSim(make(), n_runs=3, debug=True)
msim.run()
print('debug=True  n sims:', len(msim))
msim = ss.MultiSim(make(), n_runs=3, parallel=False)
msim.run()
print('debug=False n sims:', len(msim))
```

Actual:

```
debug=True  n sims: 1
debug=False n sims: 3
```

Blast radius: anyone switching on `debug=True` to get serial execution or readable tracebacks. They get one sim instead of `n_runs`, and a later `reduce()` then reports a zero-width band. (The calibration docs pass `debug=True` together with `initialize=True`, which populates `self.sims` first and so avoids this path.)

**Fix**: in the debug branch, when `self.sims is None`, expand the base sim into `n_runs` reseeded copies (e.g. `[single_run(self.base_sim, ind=i, copy_sim=True, **kwargs) for i in range(n_runs)]`), or simply route debug through `multi_run(run_target, parallel=False, **kwargs)`, which already runs serially with the same reseed/copy semantics.

## Low severity

### 3. `plot(fig=...)` is silently discarded for an unreduced MultiSim — `run.py:434`

The docstring documents `fig (Figure): if provided, plot results into an existing figure`. The reduced branch honours it, but the unreduced branch starts with `fig = None`, which overwrites the argument, so a new figure is always created.

```python
fig0 = plt.figure()
out = msim.plot(fig=fig0)
print('returned fig is the one passed in:', out is fig0, '| axes in passed fig:', len(fig0.axes))
```

Actual:

```
returned fig is the one passed in: False | axes in passed fig: 0
```

**Fix**: delete the `fig = None` line (the loop already threads `fig` through successive `sim.plot(fig=fig, ...)` calls).

### 4. `plot(alpha=...)` crashes for an unreduced MultiSim — `run.py:452`

The unreduced branch reads a default alpha from `kw.plot.get('alpha', ...)` and then calls `sim.plot(..., alpha=alpha, ..., **kwargs)`. `ss.plot_args()` does not remove `alpha` from `kwargs`, so a user-supplied `alpha` is passed twice.

```python
msim.plot(alpha=0.3)
```

Actual:

```
TypeError starsim.sim.Sim.plot() got multiple values for keyword argument 'alpha'
```

**Fix**: `alpha = kwargs.pop('alpha', 0.7 if len(self) < 5 else 0.5)` before calling `sim.plot()`.

### 5. A second `reduce()` makes `reset()` restore the first reduced sim instead of the original — `run.py:349`

`reduce()` unconditionally does `self.orig_base_sim = self.base_sim`. On a second call (e.g. `msim.mean()` then `msim.median()`) `base_sim` is already the first reduced sim, so that sim overwrites the saved original. `reset()`, documented as "Undo reduce() by resetting the base sim", then restores a reduced sim.

```python
orig = msim.base_sim
msim.mean()
msim.median()
msim.reset()
print('base_sim restored to original:', msim.base_sim is orig, '| which:', msim.which)
```

Actual:

```
base_sim restored to original: False | which: None
```

**Fix**: only save the original if it isn't already saved: `if not self._has_orig_sim(): self.orig_base_sim = self.base_sim`.

### 6. `multi_run()`'s `shrink=None` default means "don't shrink", which inverts the old default and differs from the debug path — `run.py:562`

`single_run()` defaults to `shrink=True`, but `multi_run()` defaults to `shrink=None` and forwards it, and `if shrink:` treats `None` as false. Git history (`bbaec916`, "reimplementing shrink") shows this came from a mechanical rename of `keep_people=None` (where `if not keep_people:` meant "shrink by default") to `shrink=None`, which flipped the default. The `ss.parallel()` docstring example still passes `shrink=False` as the opt-out. As a result, `MultiSim.run()` returns shrunk sims with `debug=True` (which calls `single_run()` directly) and full sims otherwise.

```python
for debug in [False, True]:
    msim = ss.MultiSim([make(), make()], debug=debug)
    msim.run()
    print(f'debug={debug}: people kept after run:', isinstance(msim.sims[0].people, ss.People))
```

Actual:

```
debug=False: people kept after run: True
debug=True: people kept after run: False
```

Blast radius: code that works in debug mode but not in parallel mode (or vice versa) if it touches `sim.people` afterwards. In parallel mode, full `People` objects are also pickled back from every worker, which is the memory cost `shrink` exists to avoid.

**Fix**: decide on one default and apply it in both places. Either `multi_run(..., shrink=True)`, matching the original behaviour and the `parallel()` example, or `single_run(..., shrink=False)` plus passing `shrink` explicitly in the debug branch. Note that changing the parallel default back to `True` would break any user code that has come to rely on `msim.sims[i].people` existing.

## Verified clean

Tested and found correct: `multi_run()` with `iterpars` in both serial and parallel mode (per-run values applied, `n_runs` taken from the iterpars length, `rand_seed` iterpars override the `+ind` reseed); `MultiSim(..., initialize=True)` followed by `run()` (distinct seeds, distinct trajectories, no double reseed); in-place updating of the original sim objects for a list of sims; `reduce()` median and custom-quantile bounds and `mean(bounds=1)` bounds against numpy; `reduce()` on integer-typed results (no truncation: the mean of `new_infections` matched exactly); `summarize(method='median', quantiles=[...])`; `ss.parallel(s1, s2, n_runs=5)` correctly running exactly the two given sims; reduced-branch `plot()` after `mean()`; the serial path's explicit `copy_sim=True`, so repeated runs don't mutate the base sim.
