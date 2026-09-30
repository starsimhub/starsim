# `people.py` bug audit

Audit of `starsim/people.py` (all 866 lines: `People`, `Person`, `Filter`) for real, unambiguous defects, at commit `3d8dc9d5` (branch `rc3.6.2`, working tree as-is), Starsim 3.6.1 (editable install), numpy 2.4.6. Method: line-by-line reading, then a repro script for every candidate, run with `MPLBACKEND=agg` on small sims (2000-10000 agents); all "actual" output below is verbatim from those runs. Intent was checked against docstrings, `tests/`, and `CHANGELOG.md`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `People.__iter__()` | Iterates UIDs `0..len(people)-1` instead of the active agents, so after deaths it yields dead agents and skips living ones | 125-126 |
| 2 | Medium | `People.plot_ages()` | Percentage denominator is `f_vals.sum() + f_vals.sum()` (female counted twice, males ignored) | 640 |
| 3 | Medium | `People.plot_ages()` | A list of `bins` is silently ignored and replaced by 5-year bins | 620-626 |

## Medium severity

### 1. `for person in people` yields dead agents and skips living ones — `people.py:125-126`

`__iter__` loops `for i in range(len(self)): yield self[i]`. `len(self)` is the number of active agents, but `self[i]` → `person(i)` indexes each state by integer, and integer keys index `raw` by UID (see `Arr._convert_key`). So the loop visits UIDs `0 .. n_alive-1`, whatever their status, rather than `self.auids`. After any deaths, it returns dead agents and never reaches agents with higher UIDs (e.g. newborns).

```python
sim = ss.Sim(n_agents=2000, diseases='sis', networks='random', demographics=[ss.Deaths(death_rate=100), ss.Births(birth_rate=20)], dur=10, verbose=0, rand_seed=1).run()
ppl = sim.people
it_uids = [int(p.uid) for p in ppl]
len(it_uids), len(set(it_uids) - set(ppl.auids.tolist())), len(set(ppl.auids.tolist()) - set(it_uids))
[bool(p.alive) for p in list(ppl)[:10]]
```

Actual: `816` people yielded, of whom `556` are dead, and `556` living agents are never yielded; alive flags `[True, False, False, False, True, False, False, True, False, False]`. Expected: the 816 active agents, all with `alive=True`. Blast radius: any user code that iterates over `sim.people` after a sim with deaths (and with births, newborns are always missed). Without demographics it works by coincidence.

**Fix**: iterate over the active UIDs: `for uid in self.auids: yield self.person(int(uid))`.

### 2. `plot_ages()` percentages use the wrong total — `people.py:640`

`total = f_vals.sum() + f_vals.sum()` counts females twice and ignores males, so the percentages are wrong unless the sexes are exactly balanced, and the bars do not add up to 100%.

```python
fig = ppl.plot_ages()   # same sim as finding 1
sum(abs(p.get_width()) for p in fig.axes[0].patches)
```

Actual: `95.77464788732395` (female share of the population is `0.522`). Expected: `100.0`. Blast radius: every default (`absolute=False`) age pyramid; the error is proportional to the sex imbalance, and gets worse in populations with sex-specific mortality.

**Fix**: `total = f_vals.sum() + m_vals.sum()`.

### 3. `plot_ages(bins=[...])` ignores the supplied bin edges — `people.py:620-626`

The docstring says `bins (list/int)`. For an iterable, line 621 sets `width = None`, but the following `if np.isscalar(bins): ... else: width = 5` unconditionally overwrites it with `5`, and then `bins` is rebuilt as 5-year bins. So a list of edges never takes effect.

```python
fig = ppl.plot_ages(bins=[0, 15, 50, 100])
len(fig.axes[0].patches)//2, [t.get_text() for t in fig.axes[0].get_yticklabels()][:4]
```

Actual: `15` bars per sex, labels `['0–4', '5–9', '10–14', '15–19']`. Expected: 3 bars per sex (`0–14`, `15–49`, `50–99`). Blast radius: anyone plotting custom age groups (e.g. 0-14/15-49/50+), who silently gets the default plot.

**Fix**: make it one chain: `if np.iterable(bins): width = None; elif np.isscalar(bins): width = bins; else: width = 5`. (A separate small issue: the `else f'{bins[i]:n}+'` label branch is unreachable for `i in range(len(bins)-1)`, so the top bin is never labelled "N+". This is cosmetic and not listed as a finding.)

## Verified clean

These were hypothesised as bugs and tested, and behave correctly:
- `find_children()` matches a direct `np.isin` on active parents.
- `to_df()` with dead agents gives active rows only and includes module states.
- `Filter`: the docstring example (`filter('female')`, `f1('age') > 5`, `~f2('sir.infected')`) matches direct boolean counts exactly (2519/2322/599); also `filter('~female')`, `split=True`, `filter(uids=...)` including a nonexistent UID, and chained `f1.filter(f1.age > 5)` without mutating `f1`.
- `get_age_dist()` with a DataFrame, an Nx2 ndarray, or a Series: the upper edge is extrapolated from the last bin width (max age ~30 for bins 0/10/20), and weights are respected.
- `__setstate__`: after `sc.dcp` and after a pickle round-trip, the `_states` registry is rebuilt, so all states still grow with births, and the copies give results identical to the original.
- `update_results()`: `cum_deaths[-1] == new_deaths.sum()`, and `n_alive` matches `len(people)` at the end.
- `step_die`/`remove_dead` NaN handling of `ti_dead`/`ti_removed`.
- `grow()` slot/parent growth.
