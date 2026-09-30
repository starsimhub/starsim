# `parameters.py` bug audit

Audit of `starsim/parameters.py` (all 495 lines: `Pars`, `SimPars`, and their update, validation, and module-conversion methods) for real, unambiguous bugs only. Style, naming, docs typos, test gaps, performance, and contrived corner cases are out of scope. **Method**: I read every line, formed hypotheses, and ran each one as a repro script against the editable install (Starsim 3.6.1 working tree on branch `rc3.6.2`, commit `3d8dc9d5` plus Cliff's uncommitted changes, numpy 2.4.6, sciris 3.3.0, `MPLBACKEND=agg`). A finding is recorded only if the repro showed the wrong behaviour and the code, docstrings, and tests show it is not intended. Scripts are in the session scratchpad under `sim/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `SimPars.validate_demographics()` | `demographics=True` (or `False`) combined with `birth_rate` and/or `death_rate` always raises `ValueError`, though the code explicitly supports the combination | 376-394 |
| 2 | Low | `Pars._update_module()` | Updating a module-valued parameter with a dict crashes: it indexes the module with its own parameter key (`old[key]`) instead of using `old` | 101 |

## Medium severity

### 1. `demographics=True` plus `birth_rate`/`death_rate` always raises — `parameters.py:385`

Lines 378-382 explicitly handle `demographics=True` combined with a rate. If `birth_rate` is set, they skip the default `ss.Births()` so that the rate-driven one can be added below, and likewise for `death_rate`. But the shortcut replaces `self.demographics` with an `sc.autolist()`. The check on line 385, `valid = isinstance(self.demographics, ss.ndict) and not len(self.demographics)`, is therefore always `False` on this path, so lines 387-389 / 393-395 raise. The same happens with `demographics=False` plus a rate.

```python
for extra in [dict(birth_rate=20), dict(death_rate=15), dict(birth_rate=20, death_rate=15)]:
    s = ss.Sim(n_agents=1000, demographics=True, verbose=0, **extra); s.init()
s = ss.Sim(n_agents=1000, demographics=False, birth_rate=20, verbose=0); s.init()
s = ss.Sim(n_agents=1000, birth_rate=20, death_rate=15, verbose=0); s.init()   # control
```

Actual:

```
{'birth_rate': 20} ValueError You can only specify birth_rate together with (optionally) death_rate, not other demographics modules; add ss.Births() manually
{'death_rate': 15} ValueError You can only specify death_rate together with (optionally) birth_rate, not other demographics modules; add ss.Deaths() manually
{'birth_rate': 20, 'death_rate': 15} ValueError You can only specify birth_rate together with (optionally) death_rate, not other demographics modules; add ss.Births() manually
False+birth_rate ValueError You can only specify birth_rate together with (optionally) death_rate, not other demographics modules; add ss.Births() manually
control: ['births', 'deaths']
```

Expected: `demographics=True, birth_rate=20` gives `['births', 'deaths']`, with births at the given rate and default deaths. This is what the `if self.birth_rate is None` guards on lines 379-382 exist for.

Blast radius: users who turn on default demographics and then customize one rate. `demographics=True` is used widely in tests and docs, and `birth_rate=` is the documented shortcut on `SimPars`, so it is natural to combine them. The error message blames "other demographics modules" the user never supplied.

**Fix**: Compute `valid` before the shortcut replaces `self.demographics`, or treat the shortcut's autolist as valid, e.g. set a local `valid = True` inside the `if demog in [True, False, 1, 0]` branch. The simplest version: in the shortcut branch, set `self.demographics = ss.ndict()` rather than `sc.autolist()` and add modules with `+=` / `.append()`. Line 385 then evaluates correctly, as long as the check happens before the default modules are appended, or it counts only non-rate modules.

## Low severity

### 2. `_update_module()` indexes the module with its own parameter key — `parameters.py:101`

`update()` routes a parameter whose current value is an `ss.Module` to `_update_module(key, old, new)`, which then does `old[key].pars.update(new)`. `old` is already the module. `key` is the name of the parameter holding it (e.g. `'mod'`), not an attribute of the module, so the lookup fails. It should be `old.pars.update(new)`, mirroring `_update_ndict` (line 92), which indexes the ndict by the *sub*-key.

```python
p = ss.Pars(mod=ss.SIR())
p.update(mod=dict(init_prev=0.3))
p.mod.pars.update(init_prev=0.3); print('direct:', p.mod.pars.init_prev)   # control
```

Actual:

```
AttributeError SIR object has no attribute "mod"
direct: ss.bernoulli(p=0.3)
```

Expected: the first call updates `p.mod.pars.init_prev` to `bernoulli(p=0.3)`, the same as the direct call.

Blast radius: low. No built-in Starsim module stores another module as a parameter; I checked all module pars in a demographics+SIS+random sim and `ss.routine_vx(product=...)`. The code comments this branch as "rare". It only affects user-defined modules whose `define_pars()` contains a module, and only when they are updated with a dict.

**Fix**: Replace `old[key].pars.update(new)` with `old.pars.update(new)`.

## Verified clean

I tested these and found them correct:

- **`Pars.__init__`**: merging `pars` with `kwargs`, and rejecting non-dict `pars`.
- **`Pars.update()`**: `create=False` raising `KeyNotFoundError` for new keys, recursive update of nested `Pars`, and the dispatch order (atomic → Pars → ndict → Module → TimePar → Dist → callable → dict).
- **`_update_ndict()`**: overwriting when empty, and per-module `pars.update` when populated.
- **`_update_timepar()`**: number sets `.value`, list becomes an array, a TimePar/dict/Series replaces the value.
- **`_update_dist()`**: a Dist replaces the value, bernoulli is protected against a type change, and number, list, dict-with/without-`type`, and function inputs all work.
- **`validate_agents()`**: a pre-built People with default vs non-default `n_agents`, and int conversion.
- **`validate_total_pop()`** on first validation, for `total_pop` only, `pop_scale` only, neither, and both (raises as intended).
- **`validate_demographics()`** for `demographics=True` alone, and `birth_rate`/`death_rate` without `demographics`.
- **`validate_modules()`/`convert_modules()`**: strings, classes, dicts with string or class `type`, functions via `from_func`, and `modules=` sorting by type into the right containers or `custom`.

`validate_total_pop()` does raise when it is run a second time on already-validated pars, because both `total_pop` and `pop_scale` are filled in by then. That only matters for re-initializing a sim, and it is covered as part of finding #2 in `bugfixes/sim_bugfixes.md`.
