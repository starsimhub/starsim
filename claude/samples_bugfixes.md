# `samples.py` bug audit

Audit of `starsim/samples.py` (all 432 lines) on branch `rc3.6.2` at commit `3d8dc9d5`, using Starsim 3.6.1 (editable install), numpy 2.4.6, pandas 3.0.5. Method: line-by-line reading, then a repro script for every candidate, run against the editable install (scripts in the session scratchpad under `samples/`). Only defects the repro demonstrated, and that are not documented or intended behavior, are recorded. Style, docs, performance, and contrived corner cases are out of scope.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Low | `Samples.new()` / `Samples.__getitem__()` | Per-seed dataframes gain a spurious `Unnamed: 0` column on reload, replacing the original `timevec` column position | 309, 414 |
| 2 | Low | `Samples.new()` | With no `identifiers` and no `fname`, the archive is saved as a hidden file named `.zip`, and every such call overwrites it | 339 |

## Low severity

### 1. Reloaded per-seed dataframes contain a junk `Unnamed: 0` column — `samples.py:309`

`new()` writes each dataframe with `df.to_csv()`, which includes the index. `__getitem__` reads it back with `read_csv(f, index_col="timevec")`. For the standard input, `sim.to_df()` (used in `tests/test_samples.py`), `timevec` is a column and the index is a `RangeIndex`. The CSV therefore has an unnamed leading index column, which comes back as a data column called `Unnamed: 0` containing `0, 1, 2, ...`.

```python
import starsim as ss
sim = ss.Sim(n_agents=1000, networks='random', diseases=ss.SIR(p_death=0.2), rand_seed=1, dur=ss.years(3), verbose=0)
sim.run()
outputs = [(sim.to_df(), dict(seed=1, p_death=0.2))]
s = ss.Samples.new(outputs, identifiers=['p_death'], folder=folder, verbose=False)
print(list(sim.to_df().columns[:3]))
print(list(s[1].columns[:3]), s[1].index.name)
```

Actual:

```
orig columns[:3]: ['timevec', 'randomnet_n_edges', 'sir_n_susceptible'] index: RangeIndex
loaded columns[:3]: ['Unnamed: 0', 'randomnet_n_edges', 'sir_n_susceptible'] index name: timevec
```

Expected: the reloaded dataframe has only the original result columns, indexed by `timevec`.

Blast radius: any user who stores `sim.to_df()` output and then does column-wise operations on `Samples[seed]` or `Samples.apply()`, e.g. summing columns, plotting all columns, or `df.max()`. The extra integer column silently contaminates these.

**Fix**: In `new()`, write with `df.to_csv(index=False)` if `timevec` is already a column, or `df.set_index('timevec').to_csv()` otherwise. The goal is for the CSV to have exactly one leading `timevec` column.

### 2. No-identifier archives are saved as the hidden file `.zip` — `samples.py:339`

If `identifiers` is omitted (documented as optional) and `fname` isn't given, the auto-generated name is `"-".join([]) + ".zip"`, i.e. `.zip`. This is a hidden dotfile. Because `Path('.zip').suffix == ''`, `Dataset(folder)` doesn't treat it as a zip archive, and every later no-identifier call in the same folder silently overwrites it.

```python
s = ss.Samples.new(outputs, folder=folder, verbose=False)
print(sorted(p.name for p in folder.iterdir()))
print(len(ss.Dataset(folder)))
```

Actual:

```
files: ['.zip']
Path(".zip").suffix = ''
Dataset len: 0
```

Expected: a visible file with a real `.zip` suffix (e.g. `samples.zip`).

Blast radius: users who call `Samples.new()` without identifiers (as `test_samples_no_identifier` does) and then look for the file on disk or save several of them.

**Fix**: Fall back to a default stem when there are no identifiers, e.g. `stem = '-'.join(...) or 'samples'`.

## Verified clean

The following were hypothesized and tested, and found correct: `Dataset` construction from a folder, `ids`, `__repr__`, `filter()` with scalar and list values, and `get()` with exactly one match; `Samples.identifier`/`id`/`seeds`/`__contains__`; `get(seed)` returning the summary row minus the seed level; `copy()` sharing the cache while letting seeds be dropped from the copy only; and `new()` writing into an `io.BytesIO`, then reading back with `memory_buffer` both on and off. The multi-valued identifier check in `new()` and `__getitem__` for summary columns were read and found correct. (`tests/test_samples.py` gives every run in `get_outputs()` the same `rand_seed=0`, so all three dataframes collide on `seed_0.csv`. This is a test issue, not a `samples.py` defect.)
