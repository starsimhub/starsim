# Implementer brief — starsim dense recycled fine-slot allocator

Implement the design in `docs/superpowers/specs/2026-06-25-dense-recycled-fine-slots-design.md` (read it first — it is your complete requirements and contains the exact `_fine_slots` code, the call-site changes, the removals, and the test list).

Repo: `C:/Users/ryanhu/PycharmProjects/starsim` (branch `feat/spawn-fine`, base 4df8a793).
Interpreter: `C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.venv/Scripts/python.exe` (starsim is editable-installed into it).

## Scope (all in `starsim/people.py` + `starsim/tests/test_multiscale.py`)
1. Replace `_reserved_fine_slots(parent_uids, counts, width)` with `_fine_slots(parent_uids, counts)` exactly as specced (dense, recycling, lowest-free-slots ≥ `_split_slot_offset`).
2. Update `split` (drop the `width`/`n_sib*mlc` reasoning; call `_fine_slots(uids, np.full(len(uids), n_sib))`) and `spawn_fine` (`_fine_slots(par, k)`). Keep the re-split guard, grow, state-copy via parent_map, scale division, epi_weight=0, fine=True.
3. Remove the `MAX_LIVE_COHORTS` attribute (the `People.__init__` line + all docstring mentions) and the `_claim_resolution_scheme` method + its two call sites.
4. Update `tests/test_multiscale.py`: rewrite the reserved-block / `MAX_LIVE_COHORTS` / recurrence / cross-scenario-CRN tests to the 6 dense-semantics checks in the spec's Tests section (no cohort cap; dense bound ~O(n); recycling plateau; collision-free within-step; within-run reproducibility; drop parent-keyed/cross-scenario-CRN assertions). KEEP the conservation + variance-reduction tests unchanged.

## Gates (run worktree/standalone; starsim is the dev checkout itself, no path-pinning needed for starsim's own tests)
- `python -m pytest starsim/tests/test_multiscale.py -q` (or wherever the multiscale tests live — confirm path) → green.
- Add/keep a test that demonstrates the win: after a long high-recurrence multiscale run, `slot[fine & alive].max() - _split_slot_offset < 2 * n_agents` (dense bound) AND no `MAX_LIVE_COHORTS`-style error occurs.
- Sanity: a quick `ss`-only multiscale sim (split + a rare event) runs and conserves `sum(scale)`.
- Run the broader starsim suite `python -m pytest starsim/tests/ -q -x` (or the standard fast subset) to confirm no collateral breakage from removing MAX_LIVE_COHORTS/_claim_resolution_scheme.

DO NOT touch hpvsim — the controller handles the hpvsim mlc=16 removal + hpvsim re-validation separately.

Commit on `feat/spawn-fine` (stage `starsim/people.py`, `starsim/tests/test_multiscale.py`, and the spec docs). Message: "feat(multiscale): dense recycled fine-slot pool; drop MAX_LIVE_COHORTS + resolution-scheme guard".

WRITE REPORT to `C:/Users/ryanhu/PycharmProjects/starsim/docs/superpowers/specs/task-dense-slots-report.md`: the new allocator, the call-site/removal diffs, the test changes (what each new dense test asserts + measured dense bound vs old sparse), and the full-suite result.

RETURN ONLY: status; commit SHA+subject; one-line gate summary (multiscale tests / dense-bound demo / broader suite); report path; any concern.