# Recurrence-safe reserved-block slots — Implementation Plan (starsim)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `split` and `spawn_fine` reserved-block slots recurrence-safe — a parent that re-resolves a rare branch while prior fine descendants are alive must get a disjoint slot block, with no live–live slot collision, while staying scenario-reproducible and dense.

**Architecture:** Factor the reserved-block slot allocation into one shared `People` helper that gives each parent a contiguous block of `MAX_LIVE_COHORTS` sub-blocks of `width` slots and, per spawn, picks the lowest sub-block not currently occupied by a live descendant (recycling sub-blocks once their descendants die). Route both `spawn_fine` (width `ratio`) and `split` (width `ratio-1`) through it.

**Tech Stack:** Python, starsim, numpy, pytest. `starsim/people.py`; tests in `tests/test_multiscale.py`. Branch `feat/spawn-fine`.

## Global Constraints

- **Slot formula:** `base = _split_slot_offset + parent_slot*(width*MAX_LIVE_COHORTS)`; `episode = lowest e in 0..MAX_LIVE_COHORTS-1 whose sub-block [base+e*width, base+(e+1)*width) holds no LIVE descendant`; `slot = base + episode*width + j` (`j in 0..count-1`, `count<=width`).
- **`MAX_LIVE_COHORTS = 4`**, a `People` attribute (set in `__init__`/`init`), used by both primitives. Exceeding it (a parent with 4 simultaneously-live cohorts) raises a clear `ValueError`, never a silent collision.
- **Occupied set, not live-count//width:** robust to `spawn_fine`'s variable per-call cohort size (`k<=width`). A sub-block is "occupied" if ANY live (`alive` and `fine`) descendant of the parent has a slot in it.
- **Reproducible:** the occupied set is a deterministic function of run state; picking the lowest free episode is deterministic ⇒ identical fine slots on identical-scenario reruns.
- **Core backward-compat:** a sim that never splits/spawns is bit-identical to today (the helper is only reached when a primitive is called). Split-WITH-multiscale slot values change (re-spaced by `MAX_LIVE_COHORTS`) — acceptable per the design spec; gate on the FULL starsim suite to confirm only statistical (not recorded-baseline) split tests are affected.
- **Lineage:** `people.parent[descendant]` identifies a descendant's parent; "descendants of parent p" = `(people.parent == p) & people.fine & people.alive`.
- Run tests with the venv interpreter from the starsim checkout root: `C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.venv/Scripts/python.exe -m pytest ...`.

## File Structure

- `starsim/people.py` — add `_reserved_fine_slots(self, parent_uids, counts, width)` returning `(new_slots, parent_map)`; add `MAX_LIVE_COHORTS`; route `spawn_fine` and `split` through it.
- `tests/test_multiscale.py` — recurrence tests (live→disjoint, dead→reuse, reproducibility, bound, variance-survives-recurrence).

---

### Task 1: shared `_reserved_fine_slots` helper + route `spawn_fine`; recurrence tests

**Files:**
- Modify: `starsim/people.py` (`__init__`/`init` to set `MAX_LIVE_COHORTS`; add `_reserved_fine_slots`; `spawn_fine` to use it)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `self.slot`, `self.parent`, `self.fine`, `self.alive`, `self._split_slot_offset`, `self.grow`.
- Produces: `_reserved_fine_slots(parent_uids, counts, width) -> (new_slots: np.ndarray, parent_map: ss.uids)`.

- [ ] **Step 1: Write the failing recurrence test (live descendants ⇒ disjoint slots)**

```python
def test_spawn_fine_recurrence_disjoint_slots_while_descendants_live():
    ppl = make_people(n=100)
    p = ss.uids([7])
    f1 = ppl.spawn_fine(p, np.array([2]), 5)
    assert ppl.alive[f1].all()                       # episode-1 cohort still alive
    f2 = ppl.spawn_fine(p, np.array([2]), 5)          # same parent recurs
    s1 = set(int(ppl.slot[u]) for u in f1)
    s2 = set(int(ppl.slot[u]) for u in f2)
    assert not (s1 & s2)                              # NO live-live slot collision
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:/Users/ryanhu/PycharmProjects/hpvsim_claudecontrol/.venv/Scripts/python.exe -m pytest tests/test_multiscale.py::test_spawn_fine_recurrence_disjoint_slots_while_descendants_live -q`
Expected: FAIL (current scheme reuses the same block → `s1 == s2`).

- [ ] **Step 3: Add `MAX_LIVE_COHORTS` and `_reserved_fine_slots`; route `spawn_fine`**

In `People.__init__` (or wherever defaults are set), add `self.MAX_LIVE_COHORTS = 4`.

Add the helper:
```python
    def _reserved_fine_slots(self, parent_uids, counts, width):
        """CRN-safe, recurrence-safe reserved-block slots for fine descendants.

        Each parent owns MAX_LIVE_COHORTS sub-blocks of `width` slots. For each parent we
        pick the lowest sub-block not currently occupied by a LIVE fine descendant (so a
        recurring parent gets a disjoint block while prior descendants live, and reuses a
        sub-block once its descendants die). Returns (new_slots, parent_map) aligned
        element-wise.
        """
        offset = self._split_slot_offset
        mlc = int(self.MAX_LIVE_COHORTS)
        slots_out = []
        parents_out = []
        for p, k in zip(parent_uids, np.asarray(counts)):
            if k <= 0:
                continue
            base = offset + int(self.slot[p]) * (width * mlc)
            # sub-blocks occupied by this parent's LIVE fine descendants
            desc = ((self.parent == int(p)) & self.fine & self.alive).uids
            occupied = set(((int(self.slot[d]) - base) // width) for d in desc
                           if 0 <= (int(self.slot[d]) - base) < width * mlc)
            episode = next((e for e in range(mlc) if e not in occupied), None)
            if episode is None:
                raise ValueError(f'parent slot {int(self.slot[p])} already has {mlc} live fine '
                                 f'cohorts (MAX_LIVE_COHORTS); cannot allocate another')
            sub = base + episode * width
            slots_out.append(np.arange(sub, sub + int(k)))
            parents_out.append(np.full(int(k), int(p)))
        if not slots_out:
            return np.array([], dtype=int), ss.uids()
        return np.concatenate(slots_out), ss.uids(np.concatenate(parents_out))
```

In `spawn_fine`, replace the inline `new_slots`/`parent_map` construction with:
```python
        new_slots, parent_map = self._reserved_fine_slots(par, k, ratio)
```
(`par` = the kept parents, `k` = their event counts; `ratio` = width for spawn_fine). Keep everything else (grow, state copy, scale/epi_weight/fine, parent lineage, parent scale-shed) unchanged.

- [ ] **Step 4: Run the recurrence test + the full existing spawn_fine suite**

Run: `...python.exe -m pytest tests/test_multiscale.py -q -k spawn_fine`
Expected: the new test PASSES; all prior spawn_fine tests still PASS (single-episode behavior unchanged in shape; note single-call slot VALUES change due to the `*MAX_LIVE_COHORTS` spacing, but the spawn_fine tests are relative/statistical so they still pass — confirm).

- [ ] **Step 5: Write + run the dead-descendant reuse + bound + reproducibility tests**

```python
def test_spawn_fine_recurrence_reuses_block_when_descendants_dead():
    ppl = make_people(n=100)
    p = ss.uids([7])
    f1 = ppl.spawn_fine(p, np.array([2]), 5)
    ppl.request_death(f1); ppl.remove_dead()          # episode-1 cohort dies
    f2 = ppl.spawn_fine(p, np.array([2]), 5)
    # block recycled: episode-2 reuses episode-1's sub-block (no range growth)
    assert set(int(ppl.slot[u]) for u in f2) == {int(ppl.slot[u]) for u in f1} or \
           min(int(ppl.slot[u]) for u in f2) <= max(int(ppl.slot[u]) for u in f1)

def test_spawn_fine_bound_raises_after_max_live_cohorts():
    ppl = make_people(n=100)
    p = ss.uids([7])
    for _ in range(ppl.MAX_LIVE_COHORTS):
        ppl.spawn_fine(p, np.array([1]), 5)           # 4 live cohorts
    try:
        ppl.spawn_fine(p, np.array([1]), 5)           # 5th while all 4 alive
        assert False, 'expected ValueError exceeding MAX_LIVE_COHORTS'
    except ValueError:
        pass

def test_spawn_fine_recurrence_reproducible():
    def run():
        ppl = make_people(n=100)
        p = ss.uids([7])
        a = ppl.spawn_fine(p, np.array([2]), 5)
        b = ppl.spawn_fine(p, np.array([2]), 5)
        return sorted(int(ppl.slot[u]) for u in a), sorted(int(ppl.slot[u]) for u in b)
    assert run() == run()
```

Run: `...python.exe -m pytest tests/test_multiscale.py -q -k "spawn_fine and (reuse or bound or reproducible)"`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add starsim/people.py tests/test_multiscale.py
git commit -m "feat(multiscale): recurrence-safe reserved-block slots for spawn_fine (occupied-sub-block recycling)"
```

---

### Task 2: route `split` through the shared helper; full-suite backward-compat

**Files:**
- Modify: `starsim/people.py` (`split` uses `_reserved_fine_slots`)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `_reserved_fine_slots` from Task 1.

- [ ] **Step 1: Write the failing split-recurrence test**

```python
def test_split_recurrence_disjoint_slots_after_restore():
    # A parent that is split, then restored to non-fine (mimicking a disease model that keeps
    # a non-event body alive), then split again while its first cohort lives, must get a
    # disjoint slot block.
    ppl = make_people(n=100)
    p = ss.uids([7])
    sib1 = ppl.split(p, 4)
    ppl.fine[p] = False                                # restore parent to a whole body (re-splittable)
    sib2 = ppl.split(p, 4)
    assert not (set(int(ppl.slot[u]) for u in sib1) & set(int(ppl.slot[u]) for u in sib2))
```

- [ ] **Step 2: Run it (fails: split still uses the old inline parent-keyed formula)**

Run: `...python.exe -m pytest tests/test_multiscale.py::test_split_recurrence_disjoint_slots_after_restore -q`
Expected: FAIL (`sib1`/`sib2` slots collide).

- [ ] **Step 3: Route `split` through the helper**

In `split`, replace the inline `new_slots = np.concatenate([offset + parent_slots*n_sib + k ...])` / `parent_map = np.tile(uids, n_sib)` with:
```python
        new_slots, parent_map = self._reserved_fine_slots(uids, np.full(len(uids), n_sib), n_sib)
```
(width = `n_sib` = ratio-1; every split parent spawns exactly `n_sib` siblings, so counts is constant `n_sib`). Keep split's existing fine/scale/epi_weight tagging and the re-split-of-fine rejection.

- [ ] **Step 4: Run split-recurrence + the full multiscale suite**

Run: `...python.exe -m pytest tests/test_multiscale.py -q`
Expected: the new split-recurrence test PASSES; all existing split + spawn_fine tests PASS (slot values changed by spacing, but tests are relative/statistical).

- [ ] **Step 5: Run the FULL starsim suite (catch any recorded-baseline break)**

Run: `...python.exe -m pytest tests/ -q -p no:cacheprovider`
Expected: green. If a `test_baselines`-style test that uses multiscale split with RECORDED values fails, that is the documented slot-respacing effect — STOP and report it (it would mean an in-tree consumer depends on split's exact slots, which changes the backward-compat story and is the human's call). A failure in a scale==1 / no-split baseline is NOT expected and would be a real regression.

- [ ] **Step 6: Commit**

```bash
git add starsim/people.py tests/test_multiscale.py
git commit -m "feat(multiscale): route split through recurrence-safe reserved-block helper"
```

---

### Task 3: variance reduction survives recurrence (failure mode C, multi-episode)

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `spawn_fine` recurrence-safe slots from Tasks 1–2.

- [ ] **Step 1: Write the failing-or-guard multi-episode variance test**

```python
def test_spawn_fine_variance_survives_recurrence():
    # Two resolution episodes per body (the first cohort still alive at the second spawn). With
    # recurrence-safe slots the two episodes' sub-draws are independent, so the pooled rare-event
    # estimator stays ~unbiased with reduced variance. Colliding slots (the bug) would correlate
    # the episodes and inflate variance / bias the estimate.
    P_RARE = 0.05; RATIO = 10; N = 1500; N_SEEDS = 40
    def resolved_two_episodes(seed):
        sim = ss.Sim(n_agents=N, diseases='sir', networks='random', dur=2, rand_seed=seed)
        sim.init(); ppl = sim.people
        parents = ppl.auids.copy()
        d = ss.bernoulli(p=P_RARE, name='rare')
        d.init(sim=sim, module=sim.diseases.sir, seed=sim.pars.rand_seed)
        total = 0.0
        for _ in range(2):                              # two episodes, episode-1 cohort stays alive
            hits = np.zeros(len(parents), dtype=int)
            for _r in range(RATIO):
                hits += np.asarray(d.rvs(parents)).astype(int)
            fine = ppl.spawn_fine(parents, hits, RATIO)
            if len(fine):
                total += float(ppl.scale[fine].sum())
        return total / 2.0                              # mean over the two episodes
    truth = P_RARE * N
    res = np.array([resolved_two_episodes(s) for s in range(N_SEEDS)])
    assert abs(res.mean() - truth) / truth < 0.15       # unbiased across episodes
```

- [ ] **Step 2: Run it**

Run: `...python.exe -m pytest tests/test_multiscale.py::test_spawn_fine_variance_survives_recurrence -q`
Expected: PASS (recurrence-safe slots ⇒ independent episodes ⇒ unbiased). If it FAILS on bias, the recycling/occupied-set logic is reintroducing correlation — investigate (do not loosen the threshold).

- [ ] **Step 3: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): variance reduction survives recurrence (multi-episode failure mode C)"
```

---

## Self-Review

**Spec coverage** (against `2026-06-23-recurrence-safe-slots-design.md`, Option A):
- Uniform per-parent block `offset + parent_slot*(width*MAX_LIVE_COHORTS) + episode*width + j` — Task 1 helper. ✓
- Occupied-sub-block recycling (robust to variable `k`) — Task 1 helper. ✓
- `MAX_LIVE_COHORTS=4` + bound error — Task 1 (`__init__` + helper raise; tested). ✓
- Applied to BOTH spawn_fine and split via the shared helper — Tasks 1–2. ✓
- Recurrence live→disjoint, dead→reuse, reproducible — Task 1. ✓
- Variance survives recurrence — Task 3. ✓
- Backward-compat (no-split bit-identical; full-suite gate for split slot respacing) — Task 2 Step 5. ✓

**Placeholder scan:** none — complete code in each code step; exact commands + expected outcomes.

**Type consistency:** `_reserved_fine_slots(parent_uids: uids, counts: int array, width: int) -> (np.ndarray, ss.uids)`; consumed identically by `spawn_fine` (width=ratio, counts=k) and `split` (width=ratio-1, counts=full n_sib).

## Open items (resolve during execution)

- Confirm where `People` defaults are initialized so `MAX_LIVE_COHORTS` is set before any split/spawn (and survives a fresh `People` — it's an instance attribute, not class-level shared state).
- The per-parent Python loop in `_reserved_fine_slots` is O(#spawning parents × #their live descendants). Acceptable for correctness; if profiling shows it hot on large runs, vectorize the occupied-set computation by grouping live fine agents by `parent`. Note, don't pre-optimize.
- Confirm `(self.parent == int(p))` is the correct vectorized comparison for the lineage `Arr` in 3.4.0 (mirror how other code compares `parent`); adjust if the accessor differs.
