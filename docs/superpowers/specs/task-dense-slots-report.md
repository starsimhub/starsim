# Report — dense recycled fine-slot allocator

Implements `2026-06-25-dense-recycled-fine-slots-design.md` on branch `feat/spawn-fine`.
All changes confined to `starsim/people.py` + `tests/test_multiscale.py` (plus this report
and the spec docs). hpvsim untouched (handled separately by the controller).

## The new allocator (`People._fine_slots`)

Replaced `_reserved_fine_slots(parent_uids, counts, width)` (sparse, parent-keyed band at
`offset + parent_slot·width·MAX_LIVE_COHORTS`) with `_fine_slots(parent_uids, counts)`:

- Allocates `counts.sum()` slots as the **lowest free slots ≥ `_split_slot_offset` not
  occupied by a LIVE fine agent**. Dead fine agents drop out of `fine & alive`, so their
  slots recycle automatically — no freelist, no death-hook.
- Returns `(slots, parent_map)` element-wise aligned, `parent_map = repeat(parent_uids, counts)`.
- Result: `slot[fine].max() ≈ offset + peak_concurrent_fine` = **O(n)**, flat in ratio,
  cohort count, and sim length. Fine-agent `Dist` draws cost the same as ordinary draws.
- Trade-off (intentional): fine slots are no longer parent-keyed → not CRN-stable across
  scenarios; still reproducible within a run (deterministic allocation order per seed).

### Two correctness fixes vs. the spec's literal pseudocode
1. **Slot-value access.** The spec wrote `live = self.slot[self.fine & self.alive]`. In this
   starsim version, indexing an `IndexArr` with a `BoolArr` returns the **uids** of the true
   entries, not the stored slot values. Corrected to
   `live = np.asarray(self.slot[(self.fine & self.alive).uids]).astype(int)`.
2. **Window upper bound.** A live fine slot can sit *above* `offset + len(live) + n` when low
   slots recycled while higher ones survive, over-indexing `free_mask`. Extended
   `hi = max(offset + len(live) + n, live.max())` so the mask always covers every live slot
   while still guaranteeing ≥ n free candidates (≤ `len(live)` of `len(live)+n+1` are occupied).

Window-sizing invariant verified empirically: over 50 randomized allocate/kill/recycle
iterations, `_fine_slots` always returns exactly `n` distinct slots, all disjoint from the
current live-fine set.

## Call-site changes

- `split`: `new_slots, parent_map = self._fine_slots(uids, np.full(len(uids), n_sib))`
  (was `_reserved_fine_slots(uids, …, n_sib)`). Dropped the `_claim_resolution_scheme('split', ratio)`
  call. Kept the `fine[uids].any()` re-split guard, `grow`, per-state copy via `parent_map`,
  `scale /= ratio`, `epi_weight[new]=0`, `fine[new]=True`. Docstring updated to dense semantics.
- `spawn_fine`: `new_slots, parent_map = self._fine_slots(par, k)` (was `_reserved_fine_slots(par, k, ratio)`).
  Dropped the `_claim_resolution_scheme('spawn_fine', ratio)` call. Everything else unchanged.
- Both callers tag `fine[new_uids] = True` immediately after the state-copy (confirmed), so
  sequential same-step allocations see prior allocations as occupied → collision-free within a step.

## Removals

- `People.__init__`: removed `self.MAX_LIVE_COHORTS = 4`.
- Removed the `_claim_resolution_scheme` method and both its call sites.
- Removed `_reserved_fine_slots` entirely.
- Grepped the whole `starsim/` package: zero remaining references to `MAX_LIVE_COHORTS`,
  `_claim_resolution_scheme`, `_reserved_fine_slots`, or `_resolution_scheme`.

## Test changes (`tests/test_multiscale.py`)

Rewritten to dense semantics (old parent-keyed / cap / cross-scenario-CRN assertions dropped):

- `test_split_rejects_mixed_ratio` → **`test_split_allows_mixed_ratio`**: split may run at any
  ratio (no scheme guard); all fine slots stay distinct.
- `test_split_slots_are_deterministic_function_of_parent` → **`test_split_slots_are_dense_and_above_offset`**:
  fine slots are ≥ offset, collision-free, and dense (`max - offset < len(new) + 1`).
- `test_split_invariant_to_unrelated_scenario_change` → **removed** (cross-scenario CRN
  intentionally not provided; within-run reproducibility covered elsewhere).
- `test_spawn_fine_slots_deterministic_function_of_parent` → **`test_spawn_fine_slots_are_dense_and_reproducible_within_run`**:
  fixed-order allocation is identical across runs, ≥ offset, collision-free, dense.
- `test_split_and_spawn_fine_cannot_mix` → **`test_split_and_spawn_fine_can_coexist`**: both
  schemes coexist at any ratio; slots stay distinct.
- `test_all_zero_spawn_fine_does_not_claim_scheme` → **`test_all_zero_spawn_fine_is_noop`**.
- `test_spawn_fine_recurrence_reuses_block_when_descendants_dead` → **`test_spawn_fine_recycles_slots_when_descendants_dead`**:
  after the prior cohort dies, the new cohort reuses *exactly* the freed slots (`s2 == s1`).
- `test_spawn_fine_bound_raises_after_max_live_cohorts` → **`test_spawn_fine_no_cohort_cap`**:
  20 live cohorts on one parent → all materialize, no raise, all collision-free.

New dense-semantics tests added:
- **`test_dense_bound_no_cohort_cap_no_blowup`** — 200 episodes, ratio 12: live fine-slot span
  bounded O(n); asserts `MAX_LIVE_COHORTS` attribute is gone.
- **`test_dense_bound_via_sim_long_recurrence`** — end-to-end Sim, dur=60, repeated spawn_fine:
  `slot[fine & alive].max() - offset < 2 * n_agents`.
- **`test_recycling_plateau`** — late-run high-water mark ≤ early-run (recycled, not growing).
- **`test_collision_free_within_step`** — 5 coexisting cohorts: `len(unique(slot)) == (fine & alive).sum()`.
- **`test_within_run_reproducibility`** — same seed → identical `new_infections` across two runs.
- **`test_ss_only_multiscale_conserves_scale`** — split + rare spawn_fine, `sum(scale)` conserved.

Kept unchanged: conservation tests (`test_split_total_scale_is_conserved`, `test_spawn_fine_*_conservation`,
demographics/pregnancy scale-weighting), variance-reduction tests
(`test_split_estimator_unbiased_and_lower_variance`, `test_spawn_fine_estimator_*`,
`test_spawn_fine_variance_survives_recurrence`), the disjoint-while-live recurrence tests, and
the bit-identical backward-compat guards.

## Measured dense bound vs. old sparse

Long high-recurrence run, n=2000 parents, cohort 2, ratio 12, 300 episodes:

| metric | dense (new) | old sparse |
|---|---|---|
| `slot[fine&alive].max() - offset` | **3999** (≈ peak concurrent fine = 4000) | O(n·ratio·mlc) ≈ **96000**, and *grows with sim length* |
| early-vs-late high-water mark | 3999 / 3999 (flat) | monotonically growing |

The per-draw RNG array (`Dist.process_size` allocates `slots.max()+1`) therefore drops from the
measured 12–20M-element / 97–162 MB sparse allocations to O(n) dense.

## Gate results

- `pytest tests/test_multiscale.py -q` → **46 passed**.
- Dense-bound demo → span 3999 vs old ~96000; plateau early==late==3999; window invariant OK.
- `pytest tests/` (excl. baselines) → **178 passed, 3 skipped**; `tests/test_baselines.py` → **2 passed**.
  No collateral breakage from removing `MAX_LIVE_COHORTS` / `_claim_resolution_scheme`.

## Notes / concerns
- `_split_slot_offset = 10·n_init` heuristic left as-is (out of scope per the design).
- hpvsim's `people.MAX_LIVE_COHORTS = max(.., 16)` block in `Sim.init()` still exists and will
  now reference a removed attribute behavior — but the controller owns that removal + re-validation.
