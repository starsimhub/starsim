# `arrays.py` bug audit

Audit of `starsim/arrays.py` (all 1135 lines: `BaseArr`, `Arr`, `FloatArr`, `IntArr`, `BoolArr`, `BoolState`, `IndexArr`, `uids`, and the Numba helpers) for real, unambiguous defects, at commit `3d8dc9d5` (branch `rc3.6.2`, working tree as-is), Starsim 3.6.1 (editable install), numpy 2.4.6. Method: line-by-line reading, then a repro script for every candidate, run with `MPLBACKEND=agg`; all "actual" output below is verbatim from those runs. Intent was checked against docstrings, `tests/`, and `CHANGELOG.md` / `docs/whatsnew.qmd`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `Arr._boolmath()` | Comparing an `Arr` with a NumPy array crashes with `UnboundLocalError` whenever the array is raw-sized (including the normal case of a sim with no deaths/births yet) | 355-356 |
| 2 | Medium | `BaseArr.__array_ufunc__()` / `Arr.convert()` | A NumPy scalar or array on the left of a comparison returns a `FloatArr` of booleans instead of a `BoolArr`, so `.uids` and `&` fail | 104, 663-666 |
| 3 | Medium | `Arr.astype()` | Crashes for every input: `cls.dtype`/`cls.nan` are not class attributes, and it shadows NumPy's `astype(int)` | 715-728 |
| 4 | Low | `Arr.__init__()` | `dtype=None` ("infer from value") gives `RecursionError`; the class docstring example crashes | 234, 246 |
| 5 | Low | `BoolArr` docstring | Docstring example `infected[[0, 2, 4]] = True` raises `TypeError` | 790 |
| 6 | Low | `Arr` docstring | Docstring says integer keys index `values`; code (intentionally) indexes `raw` by UID | 190-196 |
| 7 | Low | `uids.__new__()` | A NumPy integer scalar (e.g. `u[0]`) produces a 0-d `uids`, unlike a Python `int` | 973 |

## High severity

### 1. `Arr` vs NumPy-array comparison crashes with `UnboundLocalError` — `arrays.py:355-356`

In `_boolmath()`, the `np.ndarray` branch only sets `both_raw = self_raw.size == other.size` and never assigns `other_raw`. If the sizes match, the code then reaches `b = other_raw` (line 363) and crashes. The sizes match whenever the caller passes a raw-sized array, and also whenever `len(values) == len(raw)`, which is the state of every sim before anyone dies or is born (i.e. all sims without demographics, and every sim during `init`/first step). So `arr > np_array` works only if agents have died, and otherwise crashes.

```python
import numpy as np, starsim as ss
sim = ss.Sim(n_agents=1000).init()
age = sim.people.age
(age > np.full(len(age), 30.0)).count()
```

Actual: `UnboundLocalError cannot access local variable 'other_raw' where it is not associated with a value`. Expected: a count (the same call returns `85` in a sim with deaths, where `len(values) != len(raw)`). With deaths, passing a raw-sized array (`np.full(age.raw.size, 30.0)`) crashes the same way.

Blast radius: any module/intervention/analyzer comparing a state to a per-agent NumPy array (thresholds, per-agent draws) — `==`, `!=`, `<`, `<=`, `>`, `>=`, plus `&`, `|`, `^` on `BoolArr`, and in-place `&=`, `|=`, `^=`. It fails deterministically in the default (no demographics) configuration.

**Fix**: in the `np.ndarray` branch, set `other_raw = other` (so the raw path uses it when sizes match); the values path already uses `other` directly.

## Medium severity

### 2. NumPy scalar/array on the left of a comparison returns a `FloatArr` of booleans — `arrays.py:104`, `arrays.py:663-666`

`np.float64(30) < age` is dispatched by NumPy to `BaseArr.__array_ufunc__`, which calls `self.convert(result)`; `Arr.convert()` calls `self.asnew(obj)`, which keeps `self.__class__` (here `FloatArr`) even though the result dtype is bool. The result then lacks `BoolArr` methods: `.uids` falls through `__getattr__` to the ndarray and fails, and `&` raises `BooleanOperationError`. The Python-float version (`30.0 < age`) goes through `Arr.__gt__` and correctly returns a `BoolArr`. The same applies to arithmetic (e.g. `np.ones(n) * boolarr` returns a `BoolState` holding float64 values, with `nan=False`), unlike `_math()`, which explicitly picks the result class via `_math_result_type()`.

```python
sim = ss.Sim(n_agents=1000, verbose=0).init()
age = sim.people.age
x = np.float64(30)
type(x < age).__name__          # actual: 'FloatArr'    expected: 'BoolArr' (type(30.0 < age) -> 'BoolArr')
(x < age).uids                  # actual: AttributeError 'numpy.ndarray' object has no attribute 'uids'
(x < age) & sim.people.female   # actual: BooleanOperationError Logical operations are only valid on Boolean arrays, not bool
(age > x).uids                  # works: 520
```

Blast radius: NumPy scalars are everywhere (values pulled out of arrays, `np.mean(...)`, parameters taken from arrays, time values), so writing the comparison "constant first" silently gives the wrong type and then crashes downstream.

**Fix**: in `BaseArr.__array_ufunc__` (or `Arr.convert`), choose the result class and `nan` from the result dtype, using `Arr._math_result_type()` as `_math()` does, and set `nan`/`nan_eq` on the new object.

### 3. `Arr.astype()` crashes for every input — `arrays.py:715-728`

`astype(cls)` reads `cls.dtype` and `cls.nan`, but no `Arr` subclass has these as class attributes; they are set as instance attributes in `__init__`. So `astype(ss.IntArr)` always raises `AttributeError`. Because the method shadows NumPy's `astype` (which `Arr` otherwise forwards via `__getattr__`), the common NumPy idiom `arr.astype(int)` also crashes, in `object.__new__(int)`. Even if the attributes existed, `np.array(..., dtype=new_dtype, copy=False)` (line 727) raises `ValueError` in NumPy 2 whenever a dtype change requires a copy, which is always the case here.

```python
sim = ss.Sim(n_agents=1000).init()
sim.people.age.astype(ss.IntArr)  # actual: AttributeError type object 'IntArr' has no attribute 'dtype'
sim.people.age.astype(int)        # actual: TypeError object.__new__(int) is not safe, use int.__new__()
```

Expected: an `IntArr` for the first call; an integer array for the second (as with any ndarray). Blast radius: anyone calling `.astype()` on a state. The method is not used inside Starsim, which is why this has gone unnoticed.

**Fix**: take dtype/nan from an instance (e.g. `tmp = cls()`, then `tmp.dtype`, `tmp.nan`) or add class-level `dtype`/`nan` to `FloatArr`/`IntArr`/`BoolArr`/`IndexArr`; use `copy=None` (or `astype(..., copy=copy)`) instead of `copy=False`; and if `cls` is not an `Arr` subclass, fall back to `self.values.astype(cls)`.

## Low severity

### 4. `Arr(dtype=None)` recurses infinitely; class docstring example crashes — `arrays.py:234`, `arrays.py:246`

The docstring documents `dtype (class): ... (if None, infer from value)`, but nothing infers it: with `dtype=None`, `self.dtype` is never set, and `np.empty(0, dtype=self.dtype)` (line 246) goes to `BaseArr.__getattr__('dtype')`, which reads `values`, which reads `raw` (not yet set), which calls `__getattr__` again, and so on. The first example in the `Arr` docstring hits this.

```python
age = ss.Arr('age', default=0, mock=5)                   # actual: RecursionError maximum recursion depth exceeded
ss.Arr('age', dtype=float, default=0, mock=5).values     # works: array([0., 0., 0., 0., 0.])
```

Blast radius: users following the docstring, or creating a base `Arr` without a dtype. The subclasses always set `dtype`, so they are not affected.

**Fix**: when `dtype is None`, infer it from `default` (e.g. `np.asarray(default).dtype` for scalars, float otherwise), or raise a clear error; and add `dtype=float` to the docstring example.

### 5. `BoolArr` docstring example raises `TypeError` — `arrays.py:790`

The example indexes with a plain list, which `_convert_key()` rejects on purpose (see the changelog: indexing by non-`uids` integer arrays is ambiguous).

```python
infected = ss.BoolArr('infected', mock=5)
infected[[0, 2, 4]] = True
# actual: TypeError Indexing an Arr (infected) by ([0, 2, 4]) is ambiguous or not supported. Use ss.uids() instead, or index Arr.raw or Arr.values.
```

**Fix**: change the example to `infected[ss.uids([0, 2, 4])] = True`.

### 6. `Arr` docstring misdescribes integer indexing — `arrays.py:190-196`

The class docstring says "If indexing by an int or slice, `Arr.values` is used", and that with 100 dead agents `sim.people.age[999]` raises `IndexError`. In fact `_convert_key()` passes integers straight to `raw` (by UID). This is intentional (changelog: "Allows `Arr` objects to be indexed by integer (which are assumed to be UIDs)"), and `_convert_key`'s own docstring says so. With deaths, the two readings give different agents:

```python
# after a sim with deaths; UID 0 is dead
age[0], age.values[0], age.raw[0]   # actual: 32.90285 60.594948 32.90285
```

Following the class docstring, `age[0]` would be `60.594948` (first active agent); it is actually the dead agent with UID 0. Blast radius: users who rely on the docstring to index "the i-th alive agent".

**Fix**: rewrite that paragraph: integers (like `ss.uids`) index `raw` by UID; only a full slice `[:]` or boolean arrays refer to active agents; use `arr.values[i]` for the i-th active agent.

### 7. `ss.uids(numpy_int)` returns a 0-d array — `arrays.py:973`

`__new__` promotes a scalar to a list only when `isinstance(arr, int)`, which excludes NumPy integers. So `ss.uids(u[0])` (and any `np.int64`) produces a 0-d `uids`, which has no `len()` and cannot be concatenated, while `ss.uids(3)` gives `uids([3])`. The comment ("Convert e.g. ss.uids(0) to ss.uids([0])") and `_convert_key`'s handling of `ss_int` show that NumPy ints are meant to be treated like `int`.

```python
u = ss.uids([4, 7, 9])
ss.uids(3)                       # uids([3])
ss.uids(u[0])                    # actual: uids(4)   expected: uids([4])
len(ss.uids(u[0]))               # actual: TypeError len() of unsized object
ss.uids(u[0]) + ss.uids([1])     # actual: ValueError zero-dimensional arrays cannot be concatenated
```

**Fix**: test `isinstance(arr, numbers.Integral)` (or `np.integer`), or apply `np.atleast_1d` before `_ensure_int`.

## Verified clean

These were hypothesised as bugs and tested, and behave correctly:
- `uids` operators `+` (concatenation vs. scalar add), `|`, `&`, `-`, `^`, `+=` with a scalar (in place), and `max()` returning an `int`.
- `uids.concatenate` called as an instance method, with a list, a tuple, `[[]]`, or `None`.
- `intersect([])` (float promotion handled by `_ensure_int`); `int32` input normalised to `int64`.
- `unique(return_index=True)`: the index is wrapped as `uids`, but its only caller (`diseases.py:304`) uses it positionally on plain arrays, so it is harmless.
- `BoolArr` `==`/`!=`/`&`/`|`/`^` with `uids` and with `BoolArr`, in-place versions, `~`, and `split()`.
- `FloatArr` `isnan`/`notnan`/`notnanvals`/`true()`/`false()` with NaNs (NaN is truthy in both the NumPy and Numba paths).
- `IntArr` NaN sentinel defaults; `isin()` for small and large sets.
- Forward and reflected arithmetic (`10 - f`, `f / 2`, `f += 1`) and result-class selection in `_math()`.
- `grow()` over-allocation and NaN-filling of spare capacity.
- The `values` fast path (`raw.size == auids.size`).
- Deepcopy/pickle of states (via `People.__setstate__`).
