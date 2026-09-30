# `modules.py` bug audit

Audit of `starsim/modules.py` (all 913 lines: module registry/discovery, `ss.required()`, `ss.Base`, and `ss.Module`) for real, unambiguous bugs only: wrong results on in-contract input, documented arguments that silently do nothing, stale state, and crashes on normal usage. Style, docstrings, performance, test gaps, and contrived corner cases are out of scope. **Method**: line-by-line reading, then a repro script run against the editable install for every candidate (Starsim 3.6.1 working tree on branch `rc3.6.2`, commit `3d8dc9d5` plus Cliff's uncommitted changes; numpy 2.4.6, sciris 3.3.0, `MPLBACKEND=agg`). Repro scripts are in the session scratchpad under `modules/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `Module.__init__()` | Time kwargs passed through `super().__init__(**kwargs)` build the module's timeline immediately, standalone, so it never inherits the sim's `dt` | 279 |
| 2 | Low | `Module.set_metadata()` | A custom `name` given to a subclass (e.g. `ss.SIR(name='sir2')`) never becomes the default label; the label stays the class name | 414-416 |
| 3 | Low | `Module.create()` | Always raises `AttributeError`: calls `ss.all_subclasses()`, which was removed in v0.5 | 427 |

## Medium severity

### 1. Time kwargs passed through `super().__init__(**kwargs)` build the module's timeline immediately, standalone, so it never inherits the sim's `dt` — `modules.py:279`

`Module.__init__()` does `self.t = ss.Timeline(**kwargs, name=self.name)`. `Timeline.__init__()` auto-initializes whenever two of `start`/`stop`/`dur` are given (`timeline.py:96`), so a module constructed with e.g. `start=2005, stop=2010` gets a fully built timeline right there, with no sim, using the Timeline default `dt=1` year. Later, `init_pre()` calls `self.t.init(sim=self.sim)`, which returns immediately because the timeline is already initialized (`timeline.py:470`), so `reconcile_args(sim)` never gets to fill in `dt` from the sim. The documented alternative pattern (`super().__init__()` then `update_pars(pars, **kwargs)`) only stores the values and defers `init()` to `init_pre()`, where the sim's `dt` *is* inherited. The same user input therefore gives a different timestep depending on which constructor pattern the module class happens to use.

```python
import starsim as ss

class Counter(ss.Intervention):
    """ Counts how many times step() is called """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 0
    def step(self):
        self.n += 1

class Counter2(ss.Intervention):
    """ Same, but using the documented super().__init__() + update_pars() pattern """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars()
        self.update_pars(pars, **kwargs)
        self.n = 0
    def step(self):
        self.n += 1

for cls in [Counter, Counter2]:
    c = cls(start=2005, stop=2010)
    sim = ss.Sim(n_agents=500, start=2000, stop=2010, dt=0.25, interventions=c, verbose=0)
    sim.run()
    iv = sim.interventions[0]
    print(cls.__name__, 'dt =', iv.t.dt, 'npts =', iv.t.npts, 'step calls =', iv.n)
```

Actual:

```
Counter dt = year npts = 6 step calls = 6
Counter2 dt = years(0.25) npts = 21 step calls = 21
```

Expected: both modules inherit the sim's quarterly `dt` (21 points, 21 step calls), as `Counter2` does.

The standalone init also means `_capture_relvec_context()` runs without the sim, so the module's `relvec` is measured from its own start rather than the sim's.

**Blast radius**: any module class that forwards `**kwargs` to `super().__init__()`, which is the natural way to write a small custom intervention/analyzer and is also what the built-in `ss.Intervention`, `ss.RoutineDelivery`, `ss.CampaignDelivery`, `ss.BaseTest`, `ss.BaseTreatment`, `ss.treat_num` and `ss.BaseVaccination` do. Triggered whenever the user passes two of `start`/`stop`/`dur` without `dt` in a sim whose `dt` isn't one year; the module then silently steps once a year. It also sets up the stale-update bug in `timeline_bugfixes.md` #2, since the timeline is already initialized when `update_pars()` later calls `self.t.update()`.

**Fix**: construct the module timeline uninitialized, `self.t = ss.Timeline(**kwargs, name=self.name, init=False)`, so initialization always happens in `init_pre()` with the sim available, exactly as for the `update_pars()` path.

## Low severity

### 2. A custom `name` given to a subclass never becomes the default label; the label stays the class name — `modules.py:414-416`

`set_metadata()` intends the default label to follow a custom name (`default_label = self.name if self.name != cls_lower else cls_name`; this was added deliberately in commit `44848223`). But for every subclass that follows the documented pattern (call `super().__init__()` with no arguments, then `update_pars(**kwargs)`), `set_metadata()` runs twice. The first call, from `Module.__init__()`, has no name, so it sets `self.label = 'SIR'`. The second call, from `update_pars()`, gets `name='sir2'` but `label=None`, and `_reconcile('label', None, default_label)` returns the *existing attribute* (`'SIR'`) ahead of the default, so the custom-name default is never applied. A plain `ss.Module(name='foo')` (one call) gets `label='foo'` correctly, which is what the logic intends.

```python
import starsim as ss
m = ss.Module(name='foo')
print('Module(name=foo).label:', m.label)
s = ss.SIR(name='sir2')
print('SIR(name=sir2).name/label:', s.name, s.label)
n = ss.RandomNet(name='net2')
print('RandomNet(name=net2):', n.name, n.label)
sim = ss.Sim(n_agents=500, dur=5, diseases=[ss.SIR(name='sir1'), ss.SIR(name='sir2')], networks='random')
sim.run(verbose=0)
print(sim.results.sir1.prevalence.full_label, '|', sim.results.sir2.prevalence.full_label)
```

Actual:

```
Module(name=foo).label: foo
SIR(name=sir2).name/label: sir2 SIR
RandomNet(name=net2): net2 RandomNet
SIR: Prevalence | SIR: Prevalence
```

Expected: `SIR(name='sir2').label == 'sir2'`, and the two diseases' results labelled `sir1: Prevalence` and `sir2: Prevalence`.

**Blast radius**: anyone running two instances of the same module class (the main reason to give a custom name), e.g. two SIR strains or two random networks. Results are correct, but `result.module` is set from `self.label` in `define_results()`, so result labels and plot titles for the two instances are identical and indistinguishable.

**Fix**: in `set_metadata()`, only keep an existing label if it was set explicitly, e.g. track whether the current label is an auto-generated default (a private `_default_label` flag set when the default was used) and recompute it from the new name if so; or reconcile label as `sc.ifelse(label, self.pars.get('label'), explicit_label, default_label)` without falling back to the attribute that the first call auto-filled.

### 3. `Module.create()` always raises `AttributeError`: it calls `ss.all_subclasses()`, which was removed in v0.5 — `modules.py:427`

`create()` is a public, documented classmethod ("Create a module instance by name"), but its body iterates `ss.all_subclasses(cls)`. That helper was deleted from `utils.py` in commit `049ef76f` (May 2024; the changelog says "Removed `ss.get_subclasses()` and `ss.all_subclasses()` (now handled by `ss.find_modules()`)"), so every call fails before doing anything. Nothing in `starsim/`, `tests/` or `docs/` calls it, which is why it went unnoticed.

```python
import starsim as ss
ss.Disease.create('sis')
```

Actual:

```
AttributeError: module 'starsim' has no attribute 'all_subclasses'
```

Expected: an `ss.SIS()` instance.

**Blast radius**: only users (or downstream packages) calling `SomeModule.create(name)` directly; string-to-module conversion inside the sim uses `find_modules()` and is unaffected.

**Fix**: either delete `create()`, or reimplement it on top of the registry, e.g. look up `name` in `ss.find_modules(flat=True)` and check `issubclass(found, cls)` before instantiating (this also gets the `net`-suffix aliases and registered custom modules for free).

## Verified clean

Tested with repro scripts and found correct: `find_modules()` with custom modules registered as a list (including the `net`-suffix alias and the catch-all `modules` group); `ss.required()` detection of a required method overridden without `super()` (warns at the end of `sim.run()`), and `ss.required('disable')`; `from_func()` modules after `sim.copy()` (the step is called on the copy, not the original); callable aliases (the automatic `n_<alias>` result equals the sum of its parts, and the returned anonymous array is read-only); `finalize_results()` pop-scale scaling. Read line by line with no candidate found: `define_pars()`/`update_pars()` (including the `__init__` frame inspection), `define_states()` with `reset`/`overwrite`, `_remove_state()`, `define_results()`'s duplicate-auto-result check, `init_pre()`/`link_timepars()`/`init_post()`, `finish_step()`/`finalize()` time-index bookkeeping, and `__setattr__` attribute locking. `match_time_inds(inds)` ignores `inds` when the module and sim timelines differ in length, but no caller passes `inds`, so it was not recorded.
