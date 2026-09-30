# `timeline.py` bug audit

Audit of `starsim/timeline.py` (all 598 lines: `ss.Timeline` argument reconciliation, vector construction, lazy `datevec`/`timevec`/`relvec`, `now()`, `update()`, and sharing vectors with the sim timeline) for real, unambiguous bugs only: wrong results on in-contract input, documented arguments that silently do nothing, stale state, and crashes on normal usage. Style, docstrings, performance, test gaps, and contrived corner cases are out of scope. **Method**: line-by-line reading, then a repro script run against the editable install for every candidate (Starsim 3.6.1 working tree on branch `rc3.6.2`, commit `3d8dc9d5` plus Cliff's uncommitted changes; numpy 2.4.6, sciris 3.3.0, `MPLBACKEND=agg`). Repro scripts are in the session scratchpad under `timeline/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `Timeline.relvec` | For any calendar-based timeline (including the default), `relvec` is an array of `datedur` objects, not numbers in the sim's units | 155-160 |
| 2 | Low | `Timeline.update()` | `reset=True` never re-initializes an already-initialized timeline, leaving `start`/`stop`/`dt` out of sync with the vectors | 306-307 |
| 3 | Low | `Timeline.update()` | Passing a `Timeline` as `parent` always raises `AttributeError` (`Timeline` has no `unit`) | 288-290 |

## Medium severity

### 1. For any calendar-based timeline, `relvec` is an array of `datedur` objects, not numbers in the sim's units — `timeline.py:155-160`

The class docstring defines `relvec` as "relative time, in the sim's time units", and the property ends with `self._relvec = dur_vec.to_array() # Only keep the numeric array`. For every timeline whose reference `date0` is an `ss.date` (all calendar-year and date-based timelines, i.e. everything except `start=0`-style relative timelines), the code computes `dur_vec = self.datevec - date0`, which is a `DateArray` of `datedur` objects, then `dur_class(dur_vec)`. The `ss.dur` constructors do not convert an array of `datedur`s; they just wrap it, so `.to_array()` returns the same object array of `datedur`s, not numbers, and `now('relvec')` returns a `datedur`. Separately, going through `datevec` first (fractional years rounded to calendar days) would add rounding error even if the conversion worked: converting element-wise gives `[0, 0.986, 2.033, 3.033]` months for a monthly timeline, not `[0, 1, 2, 3]`.

```python
import starsim as ss

cases = [
    dict(start=2000, stop=2002, dt=0.25),
    dict(start=2000, stop=2002, dt=ss.months(1)),
    dict(start='2000-01-01', stop='2000-03-01', dt=ss.days(7)),
    dict(start=0, stop=10, dt=1),
]
for kw in cases:
    t = ss.Timeline(**kw)
    t.ti = 2
    print(kw, '\n   relvec[:6] =', t.relvec[:6], '\n   now(relvec) at ti=2:', t.now('relvec'))

sim = ss.Sim(n_agents=200, start=2000, stop=2002, dt=ss.months(1), verbose=0)
sim.init()
print('sim relvec[:6]:', sim.t.relvec[:6])
```

Actual:

```
{'start': 2000, 'stop': 2002, 'dt': 0.25} 
   relvec[:6] = [datedur(0) datedur(months=3, days=1) datedur(months=6, days=1)
 datedur(months=9) datedur(years=1) datedur(years=1, months=3, days=1)] 
   now(relvec) at ti=2: datedur(months=6, days=1)
{'start': 2000, 'stop': 2002, 'dt': months(1)} 
   relvec[:6] = [datedur(0) datedur(days=30) datedur(months=2, days=1)
 datedur(months=3, days=1) datedur(months=4, days=1)
 datedur(months=5, days=1)] 
   now(relvec) at ti=2: datedur(months=2, days=1)
{'start': '2000-01-01', 'stop': '2000-03-01', 'dt': days(7)} 
   relvec[:6] = [datedur(0) datedur(days=7) datedur(days=14) datedur(days=21)
 datedur(days=28) datedur(months=1, days=4)] 
   now(relvec) at ti=2: datedur(days=14)
{'start': 0, 'stop': 10, 'dt': 1} 
   relvec[:6] = [0. 1. 2. 3. 4. 5.] 
   now(relvec) at ti=2: 2.0
sim relvec[:6]: [datedur(0) datedur(days=30) datedur(months=2, days=1)
 datedur(months=3, days=1) datedur(months=4, days=1)
 datedur(months=5, days=1)]
```

Expected: numeric arrays in the sim's units, i.e. `[0, 0.25, 0.5, 0.75, 1, 1.25]` (years), `[0, 1, 2, 3, 4, 5]` (months), `[0, 7, 14, 21, 28, 35]` (days), like the relative (`start=0`) case. Note that the unit mismatch also shows in the values: `datedur(days=30)` and `datedur(months=2, days=1)` are what should be exactly 1 and 2 months.

The same code was present in v3.5.2 (eagerly computed then), so this is not a regression from the lazy-vector refactor. Nothing inside `starsim/` or `tests/` reads `relvec`, so it is only visible to users.

**Blast radius**: any user reading `sim.t.relvec`, `mod.t.relvec`, `t.to_dict()['relvec']` or `t.now('rel')` on a calendar-year or date-based sim (the default configuration), e.g. to get "years since start" for a time-varying parameter or for plotting; they get objects that can't be used in arithmetic with floats.

**Fix**: compute `relvec` numerically instead of via `datevec` subtraction. For duration-based timelines, `(self.yearvec - ref_year0) / dur_class(1).years`, rounded to ~9 decimals as elsewhere in `init()` (this gives exactly `[0, 1, 2, ...]` months), where `ref_year0` is the sim's first `yearvec` value (capture it in `_capture_relvec_context()` alongside, or instead of, `date0`). For date-based timelines, use the exact calendar difference (e.g. the day count between each date and `date0`) converted to `dur_class`; don't convert `datedur`s element-wise, since e.g. `ss.days(ss.datedur(months=1, days=4))` gives `34.42`, not `35`.

## Low severity

### 2. `reset=True` never re-initializes an already-initialized timeline, leaving `start`/`stop`/`dt` out of sync with the vectors — `timeline.py:306-307`

`update()` documents `reset (bool): if True and stale, reinitialize after update` (the default), and does `if stale and reset and self.initialized: self.init()`. But `init()` begins with `if self.initialized and not force: return self`, so this call is always a no-op: the new raw values are written to `start`/`stop`/`dt`, while `tvec`/`yearvec`/`npts` still describe the old timeline, and the attributes are not even converted (`stop` becomes a bare `int`, while `start` is still `years(2000)`). A later `init(sim)` from `Module.init_pre()` also returns early, so a module in this state runs on its old timeline.

```python
import starsim as ss

t = ss.Timeline(start=2000, stop=2010, dt=1)
print('before:', t, 'npts =', t.npts)
t.update(stop=2020, dt=0.5)
print('after: start =', repr(t.start), 'stop =', repr(t.stop), 'dt =', repr(t.dt))
print('after: npts =', t.npts, 'tvec[-1] =', t.tvec[-1], 'yearvec[:3] =', t.yearvec[:3])

# Same thing through a module whose timeline was initialized in the constructor
class MyInt(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(start=2000, stop=2010)
        self.define_pars()
        self.update_pars(pars, **kwargs)
    def step(self): pass
m = MyInt(stop=2005)
print('module: stop =', repr(m.t.stop), 'but tvec ends at', m.t.tvec[-1], 'npts =', m.t.npts)
```

Actual:

```
before: Timeline(2000-2010; dt=years(1); now=2000; ti=0/10) npts = 11
after: start = years(2000) stop = 2020 dt = 0.5
after: npts = 11 tvec[-1] = 2010 yearvec[:3] = [2000. 2001. 2002.]
module: stop = 2005 but tvec ends at 2010 npts = 11
```

Expected: `stop = years(2020)`, `dt = years(0.5)`, `npts = 41`, `tvec[-1] = 2020`; and `tvec` ending at 2005 for the module.

**Blast radius**: direct callers of `Timeline.update()` on an initialized timeline, and `Module.update_pars()` when the module's timeline was already initialized in the constructor (see `modules_bugfixes.md` #1; with that fixed, the module path no longer triggers this). The common path, where `update_pars()` runs before the timeline is initialized, is unaffected.

**Fix**: call `self.init(force=True)` in `update()`. Since `reconcile_args()` has already overwritten `dur` with `stop - start` on the first init, also reset `self.dur = None` when `start` or `stop` changed (otherwise the stale `dur` trips the `dur != stop - start` check).

### 3. Passing a `Timeline` as `parent` always raises `AttributeError` — `timeline.py:288-290`

`update()` documents `parent (Timeline): parent timeline to inherit values from`, and its `force` options only make sense with a parent. But the `dt` branch does `if isinstance(parent, Timeline): if parent.unit != self.unit:`, and `Timeline` has no `unit` attribute (it was removed when units moved onto `ss.dur` types), so any call with a `Timeline` parent crashes.

```python
import starsim as ss

parent = ss.Timeline(start=2000, stop=2010, dt=0.5)
child = ss.Timeline()
child.update(parent=parent, force=False)
```

Actual:

```
AttributeError: 'Timeline' object has no attribute 'unit'
```

Expected: `child` gets `start`, `stop`, and `dt` from `parent`.

**Blast radius**: small; nothing in `starsim/` passes `parent`, so this only affects users calling `update(parent=...)` directly.

**Fix**: drop the unit check (units are now carried by the `dt` object itself, so inheriting `parent.dt` is always well defined), or, if the intent is still "don't inherit `dt` across different unit types", compare `type(parent.dt)` with `type(self.dt)` when both are set. Note also that `parent_val = 1.0` in that branch would be a unitless float, not a `dur`.

## Verified clean

Tested with repro scripts and found correct: `reconcile_args()`/`init()` for mixed inputs, including `start=ss.years(0), stop=ss.years(1), dt=ss.days(73)`, `start=ss.days(0), stop=ss.years(1), dt=ss.months(3)`, numeric `start=2000` with `dt=ss.months(3)` and with `dt=ss.datedur(months=3)`, string dates with `dt=ss.months(3)`, `dt=ss.months(1), dur=24` (24 months), and `start=2000, dur=ss.months(6)` (all give the expected endpoints, `npts`, and aligned `yearvec`); relative (`start=0`) `relvec`; `now()` with and without a key, including the out-of-range branch used at the end of a run; `_share_from()` sharing for modules on the sim's timeline and building a separate timeline for modules with a different `dt` (the sorted-plan `ti` values in `loop_bugfixes.md` confirm the module and sim `yearvec`s align). The default `dur` of 50 *units of `dt`* (e.g. 50 days for `dt=ss.days(1)`) looks surprising but is documented in `SimPars` ("default 50 steps of self.unit"), so it was not recorded.
