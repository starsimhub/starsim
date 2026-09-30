# `settings.py` bug audit

Audit of `starsim/settings.py` (all 491 lines: `dtypes`, `rc_starsim`, the `Options` class, `load_fonts()`, and `style()`) for real, unambiguous bugs, at commit `3d8dc9d5` (branch `rc3.6.2`, working tree as-is) with Starsim 3.6.1, numpy 2.4.6, sciris 3.3.0. Method: line-by-line reading, then a repro script for every candidate, run against the editable install with `MPLBACKEND=agg` (scripts are in the session scratchpad under `settings/`). Every "actual" output below is verbatim from those runs.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Options.get_orig_options()`, `Options.set()` | Default `precision=64` does not match the actual 32-bit default dtypes, so `ss.options.set('defaults')` (or `precision='default'`) silently switches the whole library to 64-bit and changes sim results; `STARSIM_PRECISION=64` does nothing | 128, 281-304 |
| 2 | Medium | `Options.refresh_references()` | Changing precision leaves stale dtype copies; after the switch `ss.randint()` with default bounds crashes | 385-399 |
| 3 | Medium | `Options.context()` / `__exit__()` | Nested `ss.options.context()` blocks raise `AttributeError` on the outer exit and leak the outer setting | 312-338, 175-187 |
| 4 | Low | `Options` docstrings | Docstring examples call methods/options that don't exist (`with_style`, `save`, `load`, `dpi`) | 48, 52-54, 276, 327 |

## High severity

### 1. Default `precision=64` contradicts the real 32-bit default, so resetting options to defaults silently changes numerics — `settings.py:128`

`dtypes` is initialised with 32-bit floats and random ints (`float=np.float32`, `rand_int=np.int32`, `rand_uint=np.uint32`, lines 15-25; the CHANGELOG confirms float32 is the intended default). But `options.precision` defaults to `64` (line 128), and `set_precision()` is never called at import. So the option lies about the current state, and anything that re-applies the "default" precision actually changes it: `ss.options.set('defaults')` iterates over `orig_options` and, on `precision`, calls `set_precision()` with 64, switching `dtypes.float` to `float64` and `rand_int`/`rand_uint` to 64-bit. This changes the random streams and therefore sim results, even though the user asked for no change. Conversely `STARSIM_PRECISION=64` is silently ignored (the dtypes stay 32-bit), and `ss.options.help('precision')` reports "64" for what is really 32-bit.

```python
import starsim as ss
def run():
    sim = ss.Sim(n_agents=2000, diseases='sis', networks='random', rand_seed=1, verbose=0).run()
    return sim.results.sis.cum_infections[-1]
a = run()
ss.options.set('defaults') # Should be a no-op: nothing has been changed
b = run()
print('cum_infections before reset:', a)
print('cum_infections after reset: ', b)
```

Actual:

```
cum_infections before reset: 4997.0
cum_infections after reset:  4982.0
```

And directly (with `STARSIM_PRECISION=64` in the environment for the second run):

```
options.precision = 64
dtypes.float = float32 | arrays.ss_float = float32
age dtype = float32
--- after ss.options.set("defaults") ---
options.precision = 64
dtypes.float = float64 | rand_int = int64
age dtype = float64
======
env STARSIM_PRECISION = 64
options.precision = 64
dtypes.float = float32 | arrays.ss_float = float32
age dtype = float32
```

Expected: `set('defaults')` leaves the dtypes at float32 and results unchanged (`4997.0` both times); `STARSIM_PRECISION=64` gives float64.

Blast radius: anyone who calls `ss.options.set('defaults')` or `ss.options(precision='default')` (the class docstring recommends the former to "reset all values to default"), e.g. at the top of a script or between test cases, silently gets different results and ~2x memory for float arrays; also triggers finding 2. Anyone relying on the `STARSIM_PRECISION` environment variable gets nothing.

**Fix**: make the default `precision` 32 (matching `dtypes`), and apply the env-var value at import by calling `options.set_precision()` after `options = Options()` (only needed when it differs from the dtypes baked into `dtypes`, but calling it unconditionally is harmless since it runs before the other modules are imported).

## Medium severity

### 2. `refresh_references()` misses several cached dtype copies, so `ss.randint()` crashes after switching precision — `settings.py:385-399`

`refresh_references()` only patches attributes named `ss_<key>` in five modules (`arrays`, `demographics`, `diseases`, `distributions`, `networks`). It misses: the default argument `dtype=ss.dtypes.rand_int` of `ss.randint.__init__` (`distributions.py:1518`, bound at class-definition time), `ss_float_` in `library/networks/theoretical.py` and `library/networks/spatial.py`, and `ss_float` in `time.py`. The `randint` case is a crash: after `precision=64`, `high` defaults to `np.iinfo(ss.dtypes.rand_int).max` (read at call time, now int64 max) while `dtype` is still the stale `np.int32`.

```python
import starsim as ss
import starsim.library as ssl
ss.options(precision=64)
print('dtypes.float =', ss.dtypes.float.__name__, 'rand_int =', ss.dtypes.rand_int.__name__)
d = ss.randint(name='r', strict=False).init()
print('randint dtype par:', d.pars.dtype, 'high:', d.pars.high)
try:
    print(d.rvs(5))
except Exception as E:
    print('randint.rvs raised', type(E).__name__, E)
print('theoretical ss_float_:', ssl.networks.theoretical.ss_float_.__name__)
```

Actual:

```
dtypes.float = float64 rand_int = int64
randint dtype par: <class 'numpy.int32'> high: 9223372036854775807
randint.rvs raised ValueError high is out of bounds for int32
theoretical ss_float_: float32
```

Expected: `randint` uses `int64` and draws successfully; library networks build `beta` in float64.

Blast radius: anyone using `precision=64` (or hitting finding 1 via `set('defaults')`) who uses `ss.randint()` without an explicit `high` (e.g. `ssl.HouseholdNet` creates `ss.randint()` and sets `high` later, but still with the stale int32 `dtype`), or the `ErdosRenyiNet`/`DiskNet` library networks (edge `beta` stays float32 while other networks use float64).

**Fix**: have `randint.__init__` default `dtype=None` and resolve `ss.dtypes.rand_int` at call time; in the library networks use `ss.dtypes.float` at call time (or rename to `ss_float` and add `library.networks.*` to the refresh list). Generally, prefer reading `ss.dtypes.*` at call time over module-level copies for anything not on a hot path.

### 3. Nested `ss.options.context()` blocks crash on exit and leak the outer setting — `settings.py:312-338`

`context()` stores the previous values in a single attribute `on_entry`, overwriting any existing one, and `__exit__` deletes it. With two nested `with ss.options.context(...)` blocks, the inner block overwrites the outer's `on_entry`, the inner exit deletes it, and the outer exit then raises `AttributeError` ("Please use ss.options.context() if using a with block") without restoring its option.

```python
import starsim as ss
print('before: verbose =', ss.options.verbose, 'warnings =', ss.options.warnings)
try:
    with ss.options.context(verbose=0):
        with ss.options.context(warnings='error'):
            pass
        print('between: verbose =', ss.options.verbose, 'warnings =', ss.options.warnings)
except Exception as E:
    print('outer exit raised', type(E).__name__, E)
print('after: verbose =', ss.options.verbose, 'warnings =', ss.options.warnings)
```

Actual:

```
before: verbose = 0.1 warnings = warn
between: verbose = 0 warnings = warn
outer exit raised AttributeError Please use ss.options.context() if using a with block
after: verbose = 0 warnings = warn
```

Expected: no exception; `verbose` restored to `0.1` after the outer block.

Blast radius: any code that nests contexts, including indirectly (a user's `with ss.options.context(verbose=0):` around a helper function that itself uses a context). The failure also leaves the global option permanently modified for the rest of the session.

**Fix**: keep a stack of `on_entry` dicts (e.g. `self.getattribute('_on_entry_stack')`, appended in `context()` and popped in `__exit__`), or have `context()` return a small separate context-manager object holding its own `on_entry`.

## Low severity

### 4. Docstring examples reference nonexistent methods and options — `settings.py:48, 52-54, 276, 327`

The `Options` class docstring says to use `ss.options.set(dpi='default')`, `ss.options.save()`/`ss.options.load()`, and `ss.options.with_style()`; the `set()` docstring example is `ss.options.set(dpi=50)`; the `context()` docstring example is `with ss.options.with_style(dpi=50):`. None of these exist on Starsim's `Options` (they are carried over from Sciris/Covasim), so copying any of these examples fails.

```python
import starsim as ss
for label, fn in [
    ("ss.options.with_style(dpi=50)", lambda: ss.options.with_style(dpi=50)),
    ("ss.options.set(dpi=50)", lambda: ss.options.set(dpi=50)),
    ("ss.options.set(dpi='default')", lambda: ss.options.set(dpi='default')),
    ("ss.options.save()", lambda: ss.options.save()),
    ]:
    try:
        fn(); print(label, '-> OK')
    except Exception as E:
        print(label, '->', type(E).__name__, str(E).splitlines()[0])
```

Actual:

```
ss.options.with_style(dpi=50) -> AttributeError 'Options' object has no attribute 'with_style'
ss.options.set(dpi=50) -> KeyNotFoundError Option "dpi" not recognized; options are "defaults" or:
ss.options.set(dpi='default') -> KeyNotFoundError Option "dpi" not recognized; options are "defaults" or:
ss.options.save() -> AttributeError 'Options' object has no attribute 'save'
```

Expected: the documented examples run.

Blast radius: users following the docstrings (rendered in the API docs).

**Fix**: replace the examples with real ones, e.g. `ss.options.set(verbose='default')`, `ss.options.set(verbose=0)`, and `with ss.style(): ...` (the module-level `ss.style()` is the actual plotting context manager); drop the `save()`/`load()` sentence.

## Verified clean

Tested and found correct: `ss.options(key=value)`, `ss.options[key] = value` and `ss.options.key = value` all route through `set()` and invalid keys raise; `set(key, 'default')`/`None` resets a single key; `changed()`; `help('verbose')` shows current/default/env name and "(modified)" correctly; `is_jupyter` for `jupyter` = -1/0/1; `set_style()` for `'starsim'` (dict copy of `rc_starsim` with Mulish font) and for a named Sciris style (`'simple'`), and `style='default'` resetting to the Starsim dict; `ss.style()` with no args uses `options._style`; `numba_indexing` changes propagate to `ss.arrays.numba_indexing`; `context()` restores values after a single non-nested block, including when the body raises. `load_fonts()` runs before `set_style()` at import so `_style` picks up Mulish (confirmed `options._style['font.family'] == 'Mulish'`).
