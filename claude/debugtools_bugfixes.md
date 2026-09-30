# `debugtools.py` bug audit

Audit of `starsim/debugtools.py` (all 493 lines: `Profile`, `Debugger`, `Diagnostics`, `check_requires()`, `check_version()`, `metadata()`, and the `mock_*()` helpers) for real, unambiguous bugs, at commit `3d8dc9d5` (branch `rc3.6.2`, working tree as-is) with Starsim 3.6.1, numpy 2.4.6, sciris 3.3.0. `Diagnostics` was audited together with its call sites in `sim.py` (`init()`, `finish_step()`) and `distributions.py` (`Dist.rvs()`), since the class is only used through them. Method: line-by-line reading, then a repro script for every candidate, run against the editable install with `MPLBACKEND=agg` (scripts are in the session scratchpad under `debugtools/`). Every "actual" output below is verbatim from those runs.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Diagnostics` / `Dist.rvs()` | Random-variate diagnostics are never recorded: the empty `rvs` dict is falsy, so `store_rvs()` is never called | `debugtools.py:303`, `distributions.py:1119` |
| 2 | Medium | `Diagnostics.store_states()` | With `detailed=True` and no deaths, every timestep stores the same live array, so all entries show the final state | 342 |
| 3 | Medium | `Diagnostics.compute_stats()` | Summary `hash` is taken of the truncated `repr`, so for more than 1000 agents it misses changes to most agents | 318 |
| 4 | Low | `check_version()` | `'!='` comparator never fails, because equal versions skip the check entirely | 391, 397 |
| 5 | Low | `mock_module()` | Documented `dur` argument is swallowed and not passed to `mock_time()` | 486-490 |

## High severity

### 1. Random-variate diagnostics are never recorded — `distributions.py:1119` / `debugtools.py:303`

`Diagnostics.__init__` sets `self.rvs = sc.objdict()` when `rvs=True`. The only place that records variates, `Dist.rvs()`, checks `if self.sim and self.sim.diagnostics and self.sim.diagnostics.rvs:` — but an empty `objdict` is falsy, and it can only become non-empty via `store_rvs()`, which is behind this very check. So nothing is ever stored, and `export()` (which also checks `if self.rvs:`) silently writes no rvs file. The states path avoids this because `sim.init()` uses `is not None` (with the comment "Need not None since dict is empty at this point"); the rvs path was not given the same treatment. The existing test `test_diagnostics_rvs` passes vacuously: it asserts `len(hashes) == len(set(hashes))` on an empty list.

```python
import starsim as ss
kw = dict(n_agents=100, start=0, stop=10, networks='random', diseases='sis', verbose=0)
sim = ss.Sim(**kw)
sim.set_diagnostics() # Docstring example
sim.run()
print('rvs keys:', list(sim.diagnostics.rvs.keys()))
print('states keys:', list(sim.diagnostics.states.keys()))
print('n dists called:', sum(d.called for d in sim.dists.dists.values()))
```

Actual:

```
rvs keys: []
states keys: ['init', 'ti0', 'ti1', 'ti2', 'ti3', 'ti4', 'ti5', 'ti6', 'ti7', 'ti8', 'ti9', 'ti10']
n dists called: 63
```

Expected: `rvs` has one entry per timestep with stats for each of the dists called.

Blast radius: anyone using `sim.set_diagnostics()` (default `rvs=True`) to find where two runs' random numbers diverge, which is the feature's headline use; they get an empty result with no warning.

**Fix**: in `Dist.rvs()` change the check to `self.sim.diagnostics.rvs is not None` (mirroring `sim.init()`); likewise use `is not None` checks in `Diagnostics.export()` and `Sim.finish_step()` for consistency.

## Medium severity

### 2. `detailed=True` state diagnostics store a live reference, so every timestep shows the final values — `debugtools.py:342`

`store_states()` stores `state.values` for each state. `BaseArr.values` returns `self.raw` itself (not a copy) whenever `raw.size == auids.size`, i.e. whenever nobody has died or been removed (`arrays.py:558-559`). All the stored entries then alias the same array, which the sim keeps mutating, so after the run every timestep's snapshot equals the final state. (The existing test uses `demographics=True`, where deaths make `.values` a fancy-indexed copy, so it doesn't catch this.)

```python
import starsim as ss
kw = dict(n_agents=200, start=0, stop=10, networks='random', diseases='sis', verbose=0)
for detailed in [False, True]:
    sim = ss.Sim(**kw)
    sim.set_diagnostics(detailed=detailed)
    sim.run()
    S = sim.diagnostics.states
    for ti in ['init','ti0','ti5','ti10']:
        v = S[ti]['sis.infected']
        print(' ', ti, (v.sum() if detailed else (v.mean, v.hash)))
    print('  actual final', sim.diseases.sis.infected.sum())
```

Actual:

```
  init (0.005, 'c46acf204c076641')
  ti0 (0.01, '327e25d891c202f0')
  ti5 (0.155, 'b2be9c6a89af19f6')
  ti10 (0.585, '9291564740092055')
  actual final 117
  init 117
  ti0 117
  ti5 117
  ti10 117
  actual final 117
```

Expected (detailed): counts rising from 1 at init to 117 at ti10, consistent with the summary means above.

Blast radius: anyone using `sim.set_diagnostics(detailed=True)` (the docstring's second example) on a sim without demographics, which is the typical small debugging sim. The data looks plausible but is entirely wrong for every timestep except the last.

**Fix**: store a copy: `entry[state_key] = state.values.copy()` (and for symmetry, `rvs.copy()` in `store_rvs()` when `detailed`).

### 3. Summary `hash` ignores most agents for populations over 1000 — `debugtools.py:318`

`compute_stats()` uses `sc.sha(v)`, which hashes `repr(v)` for anything that isn't `str`/`bytes`. Both numpy arrays (rvs) and Starsim `Arr` objects (states) have a truncated `repr` above numpy's 1000-element print threshold (`[31.902851 20.196325 27.718803 ...  1.003933 24.816738 37.11892 ]`), so the hash only covers the first and last three values. Since the non-detailed mode's purpose is to detect *where* two runs diverge, a hash that doesn't change when an agent's value changes defeats it.

```python
import starsim as ss
for n in [200, 500, 1000, 5000]:
    sim = ss.Sim(n_agents=n, networks='random', diseases='sis', verbose=0).init()
    a = sim.people.age
    s1 = ss.Diagnostics.compute_stats(a)
    a.raw[n//2] += 1.0
    s2 = ss.Diagnostics.compute_stats(a)
    print(f'n={n}: hash changes when one mid value changes: {s1.hash != s2.hash}')
```

Actual:

```
n=200: hash changes when one mid value changes: True
n=500: hash changes when one mid value changes: True
n=1000: hash changes when one mid value changes: True
n=5000: hash changes when one mid value changes: False
```

Expected: `True` for all sizes.

Blast radius: default (`detailed=False`) diagnostics on any sim with more than 1000 agents (the default `n_agents` is 10,000). Divergences that don't shift the mean/min/max noticeably (e.g. one agent's infection timing differs) are missed.

**Fix**: hash the raw bytes: `hash = sc.sha(np.ascontiguousarray(np.asarray(v)).tobytes()).hexdigest()[:16]` (using `v.values` for `Arr` inputs).

## Low severity

### 4. `check_version('!=X')` never fails — `debugtools.py:391, 397`

For `'!'` the code sets `valid = [1, -1]`, i.e. only "equal" (`0`) should fail. But the check is inside `if relation:`, and `relation` is `''` exactly when the versions are equal, so the equal case is skipped and the function returns silently even with `die=True`.

```python
import starsim as ss
print(ss.check_version('!=' + ss.__version__, die=True))
```

Actual: `0` (no error, no warning). Expected: `ValueError` (or a warning with `die=False`), since the installed version is exactly the excluded one.

Blast radius: small; `'!='` is handled in code but not mentioned in the docstring, so few users will rely on it.

**Fix**: move the `compare not in valid` test outside `if relation:`, and give the equal case a message such as `f'Starsim version {version} is not allowed'`.

### 5. `mock_module(dur=...)` silently ignores `dur` — `debugtools.py:486-490`

`mock_module(dur=10, **kwargs)` takes `dur` as a named argument, then calls `mock_time(**kwargs)`, so `dur` never reaches `mock_time()` and `mod.t.dur` is always 10. (`mock_sim()` does pass `dur` through, so `Dist.mock(dur=...)`, which forwards kwargs to both, gets inconsistent mock sim and module times.)

```python
import starsim as ss
print(ss.mock_module(dur=20).t.dur)
```

Actual: `10`. Expected: `20`.

Blast radius: small; tests and `Dist.mock()` users passing `dur`.

**Fix**: `t = mock_time(dur=dur, **kwargs)`.

## Verified clean

Tested and found correct: `Profile` / `sim.profile(follow=[net.add_pairs, sis.infect])` (the docstring example) profiles the copied sim's methods correctly and leaves the original sim uninitialized; `Debugger` docstring examples 1 (three identical sims, `func='equal'`, runs to completion and the stepped results match a normal `sim.run()`) and 2 (`die='pause'` stops at the first differing step and `db.results[-1].df` shows the differing rows); `check_version()` with `>=`, `<=` and implied `==`, including `die=True`; `check_requires()` with names and classes (the only in-library use, `HouseholdNet`, passes a name); `metadata()`; `mock_sim()` (including `dt` passthrough) and `mock_people()`; `Diagnostics` state keys (`init`, then `ti0`...`tiN`) and `export()` of states with a custom filename. The `Debugger` step counter is one ahead of `sim.t.ti` when labelling differences (it increments before stepping); this is a labelling convention, not treated as a bug.
