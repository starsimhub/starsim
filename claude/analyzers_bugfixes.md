# `analyzers.py` bug audit

Audit of `starsim/analyzers.py` (all 179 lines: `Analyzer`, `infection_log` including `plot()` and `animate()`, `dynamics_by_age`) for real, unambiguous bugs only. Method: line-by-line reading, then repro scripts for each candidate, run against the editable install on branch `rc3.6.2` at commit `3d8dc9d5` (working tree as-is; Starsim 3.6.1, numpy 2.4.6). Repro scripts are in the session scratchpad under `analyzers/`. Style, docstrings, performance, and error-message quality are out of scope.

**Nothing in this document has been applied.**

## Summary

No bugs found.

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| – | – | – | No bugs found | – |

## High severity

None.

## Medium severity

None.

## Low severity

None.

## Verified clean

Hypothesised and tested, found correct: the `dynamics_by_age` docstring example (`ss.dynamics_by_age('sis.infected')`) runs, validates the state key in `init_post()`, records one value per timestep for each bin (51 points for 51 timesteps), and its per-bin counts agree with a manual count over active agents (differences at the final step are due only to ageing after the analyzer runs); the `people.states[...][BoolArr]` indexing counts only active agents; `plot()` aligns `sim.t.timevec` with the history length; `infection_log` enables logging in every disease via `Disease.init_pre()`, collects the logs in `finalize_results()` (7516 entries = `init_prev` seeds + 7500 `cum_infections` for SIS with demographics), and `plot()` renders from `InfectionLog.to_df()`. Excluding agents at or above the top bin edge (100 by default) is documented binning behaviour, not a bug. `animate()` was read but not exercised interactively (it calls `plt.pause`); nothing in its compact UID-to-grid mapping looked wrong.
