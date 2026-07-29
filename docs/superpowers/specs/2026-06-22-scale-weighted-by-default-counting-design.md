# Scale-weighted-by-default counting in Starsim

**Date:** 2026-06-22
**Status:** Approved for implementation
**Author:** Ryan Hull (with Claude)

## Problem

Under multiscale (agents with non-uniform `scale`), every count of agents must be scale-weighted to
be correct. The framework already does this automatically for **state counts** (`n_<state>` results,
filled via the scale-weighted `Arr.count()`). But two places are still hand-written, where a builder
of a new disease/intervention/analyzer can silently produce a raw count under multiscale:

1. **Flow / event results** (`new_infections`, `new_deaths`, `new_screened`, …) — filled manually in
   each module's `update_results`.
2. **Ad-hoc counts** in custom `step()`/`apply()`/analyzer code.

Goal: make turning an agent set into a number **scale-weighted by default**, so component builders
don't have to think about it. Strictly backward compatible: when every agent has `scale == 1`, all
counts equal the raw count and results are unchanged.

## Key findings (grounding)

- `Arr.count()` is already scale-weighted (`scale_flows(self.uids)`), and **works on computed
  `BoolArr` expressions** — `(infected & (age > 50)).count()` retains its `people` link and is
  scaled. So ad-hoc counting via `(condition).count()` is already correct.
- `People.scale_flows(inds)` is the native scale-weighted sum over uids/mask.
- Auto `n_<state>` results are generated from `BoolState`s and filled in the base
  `Module.update_results` loop. **Flows have no equivalent auto-fill** — this is the gap.

## Design

### 1. Declarative flow results

A *flow* is a result whose per-step value is the scale-weighted count of agents who experienced an
event this step. Add a `flow` parameter to `ss.Result`:

```python
ss.Result('new_infections', flow=lambda m: (m.ti_infected.round() == m.ti))
```

`flow(module)` is a callable returning the agents for this step — a `BoolArr` condition or a `uids`
set. Evaluated in `update_results` (after `step()`), so it captures this-step events.

### 2. Framework auto-fills flows, scale-weighted

The base `Module.update_results` already loops `auto_state_list` to fill `n_<state>` via
`state.count()`. Add a parallel loop over flow results:

```python
for res in self._flow_results:           # results created with a flow= callable
    self.results[res.name][self.ti] = self.sim.people.count(res.flow(self))
```

A module's custom `update_results` calls `super().update_results()` (as today), so flows fill
automatically. Builders declare flows and write no counting code.

### 3. `People.count(x)` — one scaled entry point

Dispatching helper, used by the flow auto-fill internally and as the public ad-hoc idiom:

```python
def count(self, x):
    """ Scale-weighted count of agents: a BoolArr/BoolState condition, or a uids set. """
    if isinstance(x, ss.BoolArr):
        return x.count()
    return self.scale_flows(x)   # uids or boolean mask
```

The counting idiom becomes uniform: `(condition).count()` or `people.count(uids)` — both scaled.

### 4. Dogfood + backward compatibility

Refactor the standard disease flows we currently hand-fill (`new_infections`, SIR/SIS `new_deaths`)
to declarative flows, deleting the manual scale-weighting. Demographics and irregular accumulation
flows (neonatal deaths tallied mid-processing) keep using `scale_flows` explicitly — the declarative
mechanism is additive, not mandatory. When `scale == 1`, every flow equals the raw count, so
`test_baselines` is unchanged.

`new_infections` has a special case (subtract initial cases at `ti == 0`); its `flow` callable
encodes the condition, and the ti==0 adjustment stays in the disease's `update_results` after the
`super()` call (it can read/adjust the auto-filled value).

## Non-goals

- No guardrail/enforcement against raw `len`/`count_nonzero` (explicitly deferred).
- No change to result aggregation, cumulative sums, or the global `pop_scale` finalize multiply —
  those operate on already-counted numbers.

## Testing strategy (TDD)

- **Flow auto-fill is scaled:** a sim that splits a cohort mid-run; a declared flow's result is
  conserved (not inflated by the raw fine-agent count). Fails with raw counting.
- **`People.count` dispatch:** returns scale-weighted count for a `BoolArr` condition, a `BoolState`,
  and a `uids` set; equals raw count when `scale == 1`.
- **Analyzer flow:** a custom analyzer declaring a flow result gets it auto-filled and scaled
  (covers the disease/intervention/analyzer breadth).
- **Backward compatibility:** `test_baselines` + full suite unchanged (flows == raw counts at
  `scale == 1`; refactored disease flows produce identical values).
- **Flow timing:** the flow condition is evaluated after `step()`, capturing this-step events
  (verified by matching the refactored `new_infections` to the previous hand-filled values).

## Risks / open questions

- **Flow result registration:** flows must be discoverable by the base `update_results`
  (`_flow_results`), populated when results are defined. Must survive the existing `define_results`
  / auto-`n_<state>` machinery without name clashes.
- **Return-type dispatch:** `People.count` must correctly distinguish a `BoolArr` from a `uids`
  array (both are array-like); use `isinstance(x, ss.BoolArr)`.
- **Irregular flows:** flows that accumulate across multiple sub-steps (not a single per-step
  condition) are out of scope for the declarative path and keep calling `scale_flows`.
