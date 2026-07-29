# `spawn_fine`: rare-outcome resolution as a first-class multiscale primitive

**Date:** 2026-06-23
**Status:** Draft for review
**Author:** Ryan Hull (with Claude)

## Problem

`People.split(uids, ratio)` partitions one body into `ratio` *persistent, equal* fractions that all
continue living. That is the right primitive for the **population-fraction** regime (survey/sampling
weights, oversampling a key population). It is the *wrong* primitive for the **rare-outcome
resolution** regime — the original motivating use case (HPV: resolve the rare CIN→cancer branch at
fine scale; and its cousins TB→death, polio, congenital outcomes, vaccine adverse events, maternal
mortality). There, a *common at-risk body must stay whole* (it keeps transmitting, reproducing, and
recurring) while only its *rare outcome* is resolved at finer scale.

Forcing `split` onto resolution has three costs, all observed porting hpvsim:

1. **Bloat.** To get `ratio` independent draws of the rare event you must materialize `ratio`
   persistent agents — but the event is rare, so almost all of them are non-event waste that must
   then be culled (or accumulate). At ratio=12 in an hpvsim run this was ~22k inert agents (~2.25×
   the population).
2. **Bookkeeping in the disease model.** Because `split` shrinks the body even when no event
   occurs, the model must restore non-event parents to full scale and remove non-event siblings —
   re-implementing the selective logic `split` was supposed to absorb.
3. **CRN slot math leaks into the disease.** Spawning only the event successes (the efficient
   pattern) via `grow` forces the disease model to compute CRN-safe reserved-block slots itself —
   the exact framework-level concern, and the exact thing the original hpvsim attempt got wrong
   (it used default sequential slots, breaking cross-scenario reproducibility).

The rare-outcome pattern is generic, not hpvsim-specific. It deserves its own primitive so the
disease model expresses only the *event policy* (when to resolve, what the event is, the draw) and
the framework owns everything CRN-critical and conserving.

## Design — `People.spawn_fine(parent_uids, n_events, ratio)`

Materialize fine sub-agents **only for events that occur**, leaving each parent a whole body.

**Args:**
- `parent_uids` (uids): the at-risk bodies (e.g. agents in CIN this step).
- `n_events` (int array, aligned with `parent_uids`): how many fine sub-agents to materialize for
  each parent, `0 .. ratio`. The disease model computes this from its own rare-event draw over
  `ratio` sub-resolutions per body. Parents with `0` are untouched.
- `ratio` (int ≥ 2): the resolution. Each fine agent carries weight `parent.scale / ratio`.

**Behavior** (for each parent `i` with `k = n_events[i] > 0`):
1. **CRN-safe slots.** Allocate `k` fine agents via `grow` with reserved-block slots keyed by the
   parent slot, exactly as `split` does: `offset + parent_slot * ratio + j` for `j in 0..k-1`. The
   block width is the constant `ratio`, so blocks are disjoint across parents, collision-free by
   construction, and a pure function of the parent — reproducible across scenarios and independent
   of call order, timing, and volume.
2. **State copy.** Copy all module states parent→fine (the fine agents begin as replicas at the
   resolution point).
3. **Tag the fine agents.** `scale = parent.scale / ratio`, `epi_weight = 0`, `fine = True`,
   `parent` lineage recorded.
4. **Shed the delegated mass from the parent.** `parent.scale *= (1 - k/ratio)`
   (equivalently, `parent.scale -= k * parent.scale/ratio`). The parent's `epi_weight` is
   unchanged and it stays **non-`fine`** — a whole transmitting/reproducing body whose result
   weight is reduced by the outcome mass handed to the fine agents.

**Returns:** `new_uids` (the fine agents), with `people.parent[new_uids]` set.

The primitive is **mechanism only**: it does not draw the event, does not touch the parent's
natural-history trajectory, and does not decide the resolution point. The disease model owns all of
that and routes the parent down its non-event path.

## Conservation contract

Across a `spawn_fine` call, with `S = parent.scale` before:
- **Result axis:** `S` is redistributed as `parent: S(1-k/ratio)` + `k × S/ratio` (fine) `= S`.
  `sum(scale)` is invariant.
- **Epi axis:** fine agents have `epi_weight = 0`; the parent's `epi_weight` is unchanged.
  `sum(epi_weight)` is invariant.

Therefore every scale-weighted result (prevalence, `n_<state>`, flows, screening/treatment tallies)
counts the cohort as exactly the population it represents, and every `epi_weight`-weighted quantity
(births, deaths, transmission participation) counts it as the same whole bodies as before. A
screening program that reaches this woman tallies `(1-k/ratio)` on the parent + `k/ratio` on the
fine agents = **one person**, while the rare cancer outcome is resolved across `k` independent fine
draws.

**Corner case `k = ratio`** (every sub-resolution had the event): `parent.scale → 0`. Valid: the
body's entire outcome mass is delegated to fine agents; the parent contributes 0 to result counts
but remains one whole body (`epi_weight = 1`) for transmission and vital dynamics. Rare (requires
all `ratio` draws to hit) but consistent.

## CRN guarantees (the load-bearing property)

`spawn_fine` reuses `split`'s reserved-block slot scheme, so fine-agent slots are a pure function of
the parent slot and the per-parent index — reproducible across reruns and unperturbed by unrelated
spawns, exactly the property that makes common random numbers work. The parent keeps its own slot,
so its trajectory is bit-identical to a no-resolution run. This is the framework concern the disease
model must NOT re-implement.

**Interaction with `split` / one-ratio-per-sim.** `split` reserves blocks of width `ratio-1`;
`spawn_fine` reserves width `ratio`. Mixing both from the same `_split_slot_offset` in one sim would
overlap blocks. Resolution (to confirm in the plan): share the existing one-ratio guard and require
a sim use **either** `split` **or** `spawn_fine` (a single resolution scheme per sim), or give
`spawn_fine` a disjoint reserved offset band. hpvsim uses only `spawn_fine`, so no mixing occurs
there; the guard is to prevent silent collisions if a future model uses both.

## Implications callers should know (not the primitive's concern, but document)

A disease that delegates its rare outcome entirely to fine agents (parent never itself experiences
the event — hpvsim's intended use) accepts one approximation: **the event's mortality no longer
depletes the `epi_weight` (body) population** — the host parent keeps living/transmitting even when
its fine agent dies of the outcome. This is appropriate when the outcome's demographic feedback is
second-order (rare, late-life events) and is consistent with the existing `level0`-style multiscale
approximation; it should be validated against the model's single-scale baselines. A model that needs
the body itself to carry the outcome would instead pass its own body as one of the resolutions —
out of scope for this primitive.

## Testing strategy (TDD — mirrors the `split` failure-mode suite, `single_rng` off)

- **Conservation:** for random `n_events`, `sum(scale)` and `sum(epi_weight)` are invariant across
  the call.
- **Failure mode A — reproducibility:** identical parents + identical `n_events` ⇒ identical fine
  slots; invariant to call order and to unrelated `spawn_fine` calls earlier in the run.
- **Failure mode B — non-perturbation:** agents not spawned-from are bit-identical to a no-spawn
  run (same slot ⇒ same draws); the parent's own trajectory is unchanged but for its scale.
- **Failure mode C — independence / variance:** a scale-weighted rare-event estimator built via
  `spawn_fine` (draw event for `ratio` sub-resolutions, spawn successes) is ~unbiased with variance
  ~`1/ratio` of the no-resolution estimator; fine siblings of a parent get distinct slots and
  decorrelated outcomes; slot-collision rate stays below threshold at representative volumes.
- **Weighting:** fine agents are excluded from network edges (`fine`) and from vital dynamics
  (`epi_weight = 0`); their `scale` counts in disease results; a screening/treatment tally over the
  cohort equals the single-scale tally.
- **Backward compatibility:** a suite of `scale == epi_weight == 1` example sims with no
  `spawn_fine` call is bit-identical before/after the change.
- **Edge cases:** `k = ratio` ⇒ `parent.scale == 0`, `epi_weight == 1`, parent still in network;
  `k = 0` ⇒ that parent untouched; `n_events` all zero ⇒ no agents created.

## Non-goals

- **The event draw** — the disease model owns it; `spawn_fine` only materializes the counts.
- **`merge` / coarsen** — restoring a parent's shed scale after its fine agents resolve (so repeat
  episodes aren't slightly under-weighted) is a separate future primitive. The residual is bounded
  (`k/ratio` on the rare subset of bodies that had an event) and intentionally not addressed here.
- **Scale-aware transmission intensity** — still deferred (participation only, per the two-axis
  spec).

## Risks / open questions

- **Reserved-slot collision with `split`** if a sim used both; sizing of the reserved range vs spawn
  volume (same concern as `split`, quantify in the plan).
- **`parent.scale == 0`** corner: confirm no result/rate path divides by a per-agent scale in a way
  that this breaks (counts sum fine; should be fine).
- **Recurrence under-weight:** a body that had an event recurs at `scale 1-k/ratio`; bounded and
  rare, but a future `merge` would eliminate it.
- **Relationship to `split`:** `spawn_fine` and `split` are siblings (shared slot scheme, shared
  two-axis weights). Decide whether to factor the common reserved-slot + state-copy machinery into a
  shared internal helper to avoid divergence.
