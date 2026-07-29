# Dense recycled fine-slot allocator — design (2026-06-25)

**Status:** approved, supersedes the reserved-block scheme in
`2026-06-23-recurrence-safe-slots-design.md` for fine-agent slot allocation.

## Problem

`_reserved_fine_slots` keys each parent's fine-agent slots to a sparse block at
`offset + parent_slot · width · MAX_LIVE_COHORTS`. So `people.slot[fine].max()`
scales as **O(n_agents × ratio × MAX_LIVE_COHORTS)** and **grows with sim length**
(births keep raising parent slots). `Dist.process_size` allocates a
`slots.max()+1`-sized RNG array on *every draw that touches a fine agent*
(`size = self.slots[uids].max() + 1`), so fine-agent draws cost O(that max) in
time and memory regardless of how few fine agents are drawn.

Measured (hpvsim, ratio=12):

| config | peak concurrent fine | `slots.max()` (sparse) | array/draw |
|---|---|---|---|
| n=4000, 70yr | 3,225 | 12.1M | 96.9 MB |
| n=40000, 10yr (partial) | 7,800 | 20.3M | 162 MB |

These 50–280× oversized allocations, repeated every step, cause >2 GB / multi-hour
runaways at high n_agents or long horizons. The per-parent `MAX_LIVE_COHORTS=4`
cap (added to bound the block count) separately **trips** under realistic recurrent
infection when long-lived fine cancer agents accumulate, forcing a brittle bump to
16 (which quadruples the band).

## Approach: dense, self-recycling pool

Allocate fine-agent slots as the **lowest free slots ≥ `_split_slot_offset` not
currently occupied by a LIVE fine agent**. Dead fine agents drop out of
`fine & alive`, so their slots are reused on the next allocation automatically —
no freelist, no death-hook. Result: `slots.max() ≈ offset + peak_concurrent_fine`
≈ **O(n)**, flat in ratio, mlc, and sim length. Fine-agent draws then cost the
same as ordinary-agent draws (dense ≈ 0.35 MB at n=4000, 3.3 MB at n=40000).

Trade-off: fine-agent slots are no longer keyed to the parent, so fine-agent RNG
draws are **not CRN-stable across scenarios** (ratio change / intervention toggle).
They remain **reproducible within a run** (deterministic allocation order for a
seed). Cross-scenario CRN for fine agents is explicitly not required for this
project.

## Design

### Replace `_reserved_fine_slots(parent_uids, counts, width)` with `_fine_slots(parent_uids, counts)`

```python
def _fine_slots(self, parent_uids, counts):
    """Dense, recycling slots for fine descendants (split/spawn_fine).

    Allocates counts.sum() slots as the lowest free slots >= _split_slot_offset
    not occupied by a LIVE fine agent; dead fine agents' slots are reused
    automatically. Returns (slots, parent_map) aligned element-wise, with
    parent_map = repeat(parent_uids, counts) driving the state-copy. Slots are
    dense (slots.max() ~ offset + peak concurrent fine, O(n)) and NOT parent-keyed,
    so fine-agent Dist draws cost the same as ordinary draws. Reproducible within
    a run; not CRN-stable across scenarios (fine agents do not need that)."""
    counts = np.asarray(counts, dtype=int)
    keep = counts > 0
    parent_uids = ss.uids(parent_uids)[keep]
    counts = counts[keep]
    n = int(counts.sum())
    if n == 0:
        return np.array([], dtype=int), ss.uids()
    offset = self._split_slot_offset
    live = self.slot[self.fine & self.alive]
    live = live[live >= offset].astype(int)
    hi = offset + len(live) + n            # >= n free slots guaranteed in [offset, hi]
    free_mask = np.ones(hi - offset + 1, dtype=bool)
    free_mask[live - offset] = False
    free = (offset + np.nonzero(free_mask)[0][:n]).astype(int)
    parent_map = ss.uids(np.repeat(parent_uids, counts))
    return free, parent_map
```

Sizing note: at most `len(live)` slots are occupied within any window, so a window
of `len(live) + n + 1` candidate slots always contains ≥ `n` free ones.

### `split` and `spawn_fine`
- `split`: `new_slots, parent_map = self._fine_slots(uids, np.full(len(uids), n_sib))`.
- `spawn_fine`: `new_slots, parent_map = self._fine_slots(par, k)`.
- Keep everything else (the `fine[uids].any()` re-split guard, `grow`, state-copy
  via `parent_map`, scale division, `epi_weight=0`, `fine=True`, parent map).
- **Remove** the `MAX_LIVE_COHORTS` attribute (and its `People.__init__` line and all
  docstring references) and the `_claim_resolution_scheme` method + its two call
  sites — a shared dense pool has no width-collision concern, so split and
  spawn_fine may coexist at any ratio.

### Collision-freeness within a step
After `grow`, new agents are in `auids` (alive) and are tagged `fine=True` before
any subsequent allocation in the same step, so sequential `_fine_slots` calls see
prior same-step allocations as occupied. Verify the `fine[new]=True` assignment
stays immediately after `grow` in both callers (it does).

## Tests (`starsim/tests/test_multiscale.py`)

Update the reserved-block / recurrence / CRN tests to the dense semantics; keep
conservation + variance-reduction tests unchanged:

1. **No cohort cap:** a parent re-infected/re-resolved many times over a long run
   spawns many cohorts with NO error (the `MAX_LIVE_COHORTS` raise is gone).
2. **Dense bound:** after a long, high-recurrence run,
   `slot[fine & alive].max() - _split_slot_offset` stays bounded ~O(n)
   (e.g. `< 2 * n_agents`), NOT O(n·ratio·mlc).
3. **Recycling:** the max live fine slot plateaus over a long run (dead agents'
   slots are reused) rather than growing unboundedly.
4. **Collision-free:** at end of run, `len(unique(slot[fine & alive])) ==
   (fine & alive).sum()` (no two live fine agents share a slot).
5. **Within-run reproducibility:** same seed → identical results across two runs.
6. **Remove** assertions that fine slots sit in a parent-keyed block or are
   CRN-stable across scenarios.

## hpvsim follow-up (separate repo)
Remove the `people.MAX_LIVE_COHORTS = max(.., 16)` block in `hpvsim/sim.py`
`Sim.init()` — the cap no longer exists. Then re-validate the multiscale gates,
including the previously-runaway `test_multiscale_matches_single_scale_mean`
(ratio=12, n=40000), which should now run in the originally-intended minutes.

## Out of scope
The `_split_slot_offset = 10·n_init` heuristic stays as-is (it must clear the
pregnancy newborn-reserved range ~5·n); dense draws at ~10·n are already cheap and
flat. Lowering it is a separate micro-optimization.
