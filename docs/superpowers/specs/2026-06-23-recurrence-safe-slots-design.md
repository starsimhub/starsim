# Recurrence-safe reserved-block slots (addendum to split / spawn_fine)

**Date:** 2026-06-23
**Status:** Draft for review
**Author:** Ryan Hull (with Claude)
**Amends:** `2026-06-23-spawn-fine-rare-outcome-resolution-design.md`, and the existing `split` reserved-block scheme.

## Problem (verified)

Both `split` and `spawn_fine` key fine-agent slots purely off the parent slot:
`offset + parent_slot*width + j` (split `width = ratio-1`, spawn_fine `width = ratio`). That purity is what
makes slots reproducible across scenarios — the design's core CRN property. But it has **no episode
discriminator**, so a parent that resolves a rare branch *more than once* (any recurrent infection —
HPV/CIN is the motivating case) is handed the **identical slot block** on each occasion. Verified
empirically: parent slot 7, `spawn_fine` twice → both cohorts at slots {1035, 1036}; the first
cohort is still alive when the second spawns → **two live agents share a slot**, the correlated-CRN
failure the reserved-block scheme exists to prevent.

`split` only avoids this by rejecting re-split of *fine* agents; once a non-event parent is restored
to non-fine (the recurrence case), it is re-splittable and hits the same collision.

## The hard constraint: slots must stay dense and small

`Dist.process_size` sizes each draw vector as `slots.max() + 1` (`distributions.py:663`). So any scheme
that widens the slot range inflates **every** RNG draw-vector in the sim. The naive fix — fold the
spawn timestep into the key (`offset + (parent_slot*n_ti + ti)*ratio + j`) — pushes max slots to
`~n_agents × n_timesteps × ratio` (e.g. 2.4e10 for 10k agents × 200 steps × 12), allocating
multi-gigabyte draw arrays. **Infeasible.** The discriminator must be bounded and dense.

## Requirements

1. **Reproducible within a scenario:** identical seed + identical history ⇒ identical fine slots.
2. **No live–live collision:** at any time, no two live agents share a slot.
3. **Disjoint across distinct parents** (existing property, keep).
4. **Dense/bounded slot range:** max slot grows like `n_agents × ratio × (small constant)`, not ×`n_timesteps`.
5. **Backward compatible (core invariant):** a sim that never splits/spawns (scale==epi_weight==1)
   is bit-identical to today. NOTE: giving each parent an inline block of `width*MAX_LIVE_COHORTS`
   re-spaces every parent's reserved block, so `split`-*with*-multiscale slot **values** change (and
   thus exact draws/results for split-based multiscale runs). This is acceptable: no in-tree consumer
   depends on split's exact slot values (the split tests are statistical/relative — reproducibility,
   distinctness, disjointness, variance), and behavior stays statistically equivalent. A separate
   "recurrence band" that preserved episode-0 slots was rejected: births mint unbounded slots over a
   run, so such a band cannot be statically sized without collision risk.

## Design options

### Option A (recommended) — live-descendant-cohort index, with recycling

Each parent owns a contiguous block of `MAX_LIVE_COHORTS` sub-blocks of `width` slots:
```
base    = offset + parent_slot * (width * MAX_LIVE_COHORTS)
# Recycle: pick the lowest sub-block not currently occupied by a LIVE descendant.
occupied = { (slot[d] - base) // width  for d in live fine agents with parent==this parent }
episode  = min(e in 0..MAX_LIVE_COHORTS-1  with e not in occupied)     # raise if none free
slot     = base + episode*width + j                                     # j in 0..count-1
```
Using the occupied-sub-block set (not a simple live-count `// width`) makes it robust to
`spawn_fine`'s *variable* per-call cohort sizes (`k <= width`): each episode claims a whole `width`
sub-block regardless of how many of its slots it fills, and a sub-block frees for reuse only when all
its live descendants have died.
- Reproducible: the live-descendant count is a deterministic function of run state.
- Recurrence-safe: while a prior cohort is alive, `episode` is ≥1 ⇒ a disjoint sub-block; once all
  prior descendants die, `episode` returns to 0 and the block is **reused against dead agents only**
  (the normal, safe slot-reuse pattern) — so the range does not grow without bound.
- Dense: max slot ≈ `offset + n_agents * width * MAX_LIVE_COHORTS`. With `MAX_LIVE_COHORTS = 4` and
  width=ratio=12, that's ~`offset + 48·n` — a small constant ×`n`, on the order of split's existing
  headroom. Draw-vector inflation is the constant, not `n_timesteps`.
- Bounded by `MAX_LIVE_COHORTS` (a `People`/Pars knob, default small). Exceeding it (a parent with
  more than `MAX_LIVE_COHORTS` simultaneously-live cohorts — vanishingly rare for HPV) raises a clear
  error rather than silently colliding.
- Cross-scenario nuance: if scenarios differ in whether a prior cohort is still alive (e.g. an
  intervention averted it), the recurrent cohort's `episode` index — and thus its slots/draws —
  differ between scenarios. Acceptable: those scenarios genuinely differ in the parent's descendant
  state; the primary CRN guarantees (unperturbed agents identical; within-scenario reproducibility;
  variance reduction) are unaffected.

### Option B — monotonic per-parent spawn counter (simpler, less dense)

Track a per-parent cumulative spawn count `n[parent]` (never decremented):
`slot = offset + parent_slot*(width*MAX_SPAWNS) + n[parent]*width + j`.
- Simpler (no live-descendant query), still reproducible and recurrence-safe.
- But never recycles, so it needs `MAX_SPAWNS` ≥ the most episodes any parent ever has over the whole
  run (larger constant ⇒ larger slot range ⇒ more draw-vector inflation), and a parent exceeding
  `MAX_SPAWNS` errors. Strictly worse density than A for long runs with frequent recurrence.

### Option C — explicit free-slot recycling

Maintain a free-list of slots vacated by dead fine agents and draw from it first. Most space-efficient
but the most stateful/complex, and reproducibility hinges on a deterministic free-list ordering.
Not recommended unless A's density proves insufficient.

## Scope

- Apply the chosen scheme to **both** `spawn_fine` (width `ratio`) and `split` (width `ratio-1`) via
  the shared reserved-block helper, so both are recurrence-safe.
- `split`'s existing "reject re-split of fine agents" stays (you cannot re-split an agent that is
  *currently* a fine sub-agent); the new scheme handles re-resolution of a *restored/non-fine* parent.
- Add `MAX_LIVE_COHORTS` (Option A) as a `People` parameter with a small default.

## Testing (add to the failure-mode suite)

- **Recurrence, live descendants:** spawn/split the same parent twice while the first cohort is alive
  ⇒ the two cohorts have **disjoint** slots; no live agent shares a slot. (Currently fails.)
- **Recurrence, dead descendants:** if the first cohort is dead before re-spawn, the block is reused
  (slot range does not grow) and no live collision occurs.
- **Reproducibility under recurrence:** identical scenario rerun ⇒ identical fine slots across both
  episodes.
- **Variance reduction survives recurrence:** the failure-mode-C estimator stays unbiased with
  ~1/ratio variance in a multi-episode setting (no correlation reintroduced).
- **Bound:** exceeding `MAX_LIVE_COHORTS` raises a clear error (no silent collision).
- **Backward compat:** non-recurrent single-resolution sims are bit-identical to the pre-change slots.

## Risks / open questions

- **`MAX_LIVE_COHORTS` default:** small enough to keep slots dense, large enough to never error in
  realistic use. For HPV (fine cancer cohorts die within years; concurrent live cohorts per woman are
  rare) a default of 3–4 is likely ample; confirm against an hpvsim run.
- **Live-descendant query cost:** Option A counts live fine descendants per spawn. Vectorize over the
  spawning parents (group live fine agents by `parent`); should be cheap relative to a step.
- **Interaction with the one-scheme-per-sim guard:** unchanged — still one of split/spawn_fine per sim.
