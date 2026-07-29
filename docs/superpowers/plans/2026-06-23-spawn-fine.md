# `spawn_fine` Implementation Plan (starsim)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `People.spawn_fine(parent_uids, n_events, ratio)` to starsim — a mechanism-only rare-outcome-resolution primitive that materializes fine sub-agents only for events that occur, keeping each parent a whole body that sheds the delegated result mass.

**Architecture:** `spawn_fine` is `split`'s sibling: it reuses the reserved-block CRN-safe slot scheme, the parent→child state copy, and the two-axis (`scale`/`epi_weight`) weights, but (a) materializes a *variable* per-parent count instead of a fixed `ratio-1`, (b) leaves the parent non-`fine` with `epi_weight` intact, and (c) sheds `k/ratio` of the parent's `scale` to conserve `sum(scale)`. The disease model owns the event draw and passes per-parent success counts.

**Tech Stack:** Python, starsim, numpy, pytest. Implementation in `starsim/people.py`; tests in `tests/test_multiscale.py`.

## Global Constraints

- **Conservation:** across a `spawn_fine` call, `sum(scale)` and `sum(epi_weight)` are invariant. Parent: `scale *= (1 - k/ratio)`, `epi_weight` unchanged, stays non-`fine`. Fine agents: `scale = parent.scale/ratio`, `epi_weight = 0`, `fine = True`.
- **CRN-safe slots:** fine slots are a pure function of the parent slot and per-parent index — `offset + parent_slot*ratio + j` for `j in 0..k-1`, `offset = self._split_slot_offset`. Reproducible across reruns, invariant to call order/volume and to unrelated spawns. The parent keeps its own slot.
- **Backward compatibility:** with no `spawn_fine` call, all results are bit-identical (the method only adds; it touches nothing until called).
- **One resolution scheme per sim:** `split` reserves block width `ratio-1`; `spawn_fine` reserves width `ratio`. A sim must use only one of them (guarded), else reserved blocks overlap.
- **Mechanism only:** `spawn_fine` does not draw the event, does not touch parents' natural-history/disease state, and does not decide the resolution point.
- `single_rng` is off for all CRN tests (multi-RNG mode).
- Run tests with the repo's interpreter from the starsim checkout root: `python -m pytest tests/test_multiscale.py -q`.

## File Structure

- `starsim/people.py` — add `spawn_fine` method (next to `split`, ~line 460); add a one-scheme guard shared with `split`.
- `tests/test_multiscale.py` — add a `spawn_fine` test block mirroring the existing `split` failure-mode suite.

---

### Task 1: `spawn_fine` core mechanics, conservation, and edge cases

**Files:**
- Modify: `starsim/people.py` (add `spawn_fine` after `split`)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `self._split_slot_offset` (property), `self.grow(n, new_slots)`, `self.slot`, `self.scale`, `self.epi_weight`, `self.fine`, `self.parent`, `self.states`.
- Produces: `spawn_fine(parent_uids, n_events, ratio) -> ss.uids` (new fine uids).

- [ ] **Step 1: Write the failing mechanics+conservation test**

```python
def test_spawn_fine_mechanics_and_conservation():
    ppl = make_people(n=100)
    parents = ss.uids([3, 7, 42])
    n_events = np.array([2, 0, 1])   # parent 7 gets none
    ratio = 5
    orig_scale = ppl.scale[parents].copy()
    orig_age = ppl.age[parents].copy()
    total_scale_before = ppl.scale[ppl.auids].sum()

    new = ppl.spawn_fine(parents, n_events, ratio)

    assert len(new) == 3                      # 2 + 0 + 1
    # fine agents: scale = parent/ratio, epi_weight 0, fine True, state-copied
    assert np.allclose(ppl.scale[new], np.repeat(orig_scale[[0, 2]] / ratio, [2, 1]))
    assert (ppl.epi_weight[new] == 0).all()
    assert ppl.fine[new].all()
    # parents stay whole bodies (epi_weight unchanged, not fine), scale shed by k/ratio
    assert (ppl.epi_weight[parents] == 1).all()
    assert not ppl.fine[parents].any()
    assert np.allclose(ppl.scale[parents], orig_scale * (1 - n_events / ratio))
    # conservation of sum(scale)
    assert np.isclose(ppl.scale[ppl.auids].sum(), total_scale_before)
    # lineage + state copy
    assert set(int(p) for p in ppl.parent[new]) == {3, 42}
    assert np.allclose(ppl.age[new], np.repeat(orig_age[[0, 2]], [2, 1]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_mechanics_and_conservation -q`
Expected: FAIL with `AttributeError: 'People' object has no attribute 'spawn_fine'`.

- [ ] **Step 3: Implement `spawn_fine`**

Add to `starsim/people.py` after `split`:
```python
    def spawn_fine(self, parent_uids, n_events, ratio):
        """
        Materialize fine sub-agents for rare-event successes, keeping each parent a
        whole body. For rare-outcome resolution (resolve a rare branch at finer
        scale) as opposed to split()'s population partition.

        For each parent with `n_events[i] = k > 0`, create `k` fine agents at
        `scale = parent.scale/ratio` (epi_weight 0, fine=True), with CRN-safe
        reserved-block slots keyed by the parent (`offset + parent_slot*ratio + j`),
        copying all states. The parent stays a whole body (epi_weight unchanged,
        not fine) and sheds the delegated mass: `scale *= (1 - k/ratio)`, conserving
        sum(scale). The disease model owns the event draw and supplies the counts.

        Args:
            parent_uids (uids): at-risk whole bodies (must not be fine)
            n_events (int array): successes per parent, 0..ratio (aligned to parent_uids)
            ratio (int): resolution; each fine agent carries 1/ratio of the parent scale

        Returns:
            new_uids (uids): the newly created fine agents
        """
        parent_uids = ss.uids(parent_uids)
        ratio = int(ratio)
        n_events = np.asarray(n_events, dtype=int)
        if ratio < 2 or len(parent_uids) == 0:
            return ss.uids()
        if len(n_events) != len(parent_uids):
            raise ValueError('n_events must align with parent_uids')
        if (n_events < 0).any() or (n_events > ratio).any():
            raise ValueError('n_events entries must be in [0, ratio]')
        if self.fine[parent_uids].any():
            raise ValueError('spawn_fine received fine agents; only whole bodies can be resolved')

        # One resolution scheme per sim: spawn_fine reserves block width `ratio`
        # (split reserves ratio-1); mixing would overlap reserved blocks.
        self._claim_resolution_scheme('spawn_fine', ratio)

        keep = n_events > 0
        if not keep.any():
            return ss.uids()
        par = parent_uids[keep]
        k = n_events[keep]
        par_slots = np.asarray(self.slot[par])
        par_scale = self.scale[par].copy()

        offset = self._split_slot_offset
        new_slots = np.concatenate([offset + s * ratio + np.arange(kk)
                                    for s, kk in zip(par_slots, k)])
        parent_map = ss.uids(np.repeat(par, k))

        new_uids = self.grow(len(new_slots), new_slots)
        for state in self.states.values():
            state[new_uids] = state[parent_map]
        self.parent[new_uids] = parent_map

        self.scale[new_uids] = np.repeat(par_scale / ratio, k)
        self.epi_weight[new_uids] = 0.0
        self.fine[new_uids] = True
        # Shed the delegated outcome mass from the parents (conserves sum(scale)).
        self.scale[par] = par_scale * (1 - k / ratio)
        return new_uids
```

Add the shared scheme guard (used by `split` too — see Task 5; for now define it):
```python
    def _claim_resolution_scheme(self, scheme, ratio):
        """Enforce one multiscale resolution scheme + one ratio per sim, so the
        reserved-block slot ranges of split (width ratio-1) and spawn_fine (width
        ratio) never overlap."""
        prev = getattr(self, '_resolution_scheme', None)
        if prev is None:
            self._resolution_scheme = (scheme, ratio)
        elif prev != (scheme, ratio):
            raise ValueError(f'this sim already uses resolution scheme {prev}; '
                             f'cannot also use {(scheme, ratio)} (reserved slot blocks would collide)')
        return
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_mechanics_and_conservation -q`
Expected: PASS.

- [ ] **Step 5: Write + run edge-case tests**

```python
def test_spawn_fine_edge_cases():
    ppl = make_people(n=100)
    # k = ratio -> parent scale 0, still epi_weight 1 (a whole body) and not fine
    new = ppl.spawn_fine(ss.uids([5]), np.array([4]), 4)
    assert len(new) == 4
    assert np.isclose(ppl.scale[ss.uids([5])][0], 0.0)
    assert ppl.epi_weight[ss.uids([5])][0] == 1.0
    assert not ppl.fine[ss.uids([5])].any()
    # all-zero counts -> no agents, no-op
    ppl2 = make_people(n=50)
    before = len(ppl2.auids)
    assert len(ppl2.spawn_fine(ss.uids([1, 2]), np.array([0, 0]), 4)) == 0
    assert len(ppl2.auids) == before
    # epi_weight conserved across a real spawn
    ppl3 = make_people(n=100)
    ew_before = ppl3.epi_weight[ppl3.auids].sum()
    ppl3.spawn_fine(ss.uids([1, 2, 3]), np.array([2, 1, 3]), 4)
    assert np.isclose(ppl3.epi_weight[ppl3.auids].sum(), ew_before)
```

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_edge_cases -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starsim/people.py tests/test_multiscale.py
git commit -m "feat(multiscale): add People.spawn_fine rare-outcome-resolution primitive"
```

---

### Task 2: CRN reproducibility and non-perturbation (failure modes A & B)

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `spawn_fine` from Task 1; `ss.normal` probe dist; `people.parent`, `people.slot`.

- [ ] **Step 1: Write the failing reproducibility test (failure mode A)**

```python
def test_spawn_fine_slots_deterministic_function_of_parent():
    def fine_slots_by_parent(order):
        ppl = make_people(n=100)
        new = ppl.spawn_fine(ss.uids(order), np.array([2, 3, 1]), 4)
        out = {}
        for u in new:
            ps = int(ppl.slot[ss.uids([int(ppl.parent[u])])][0])
            out.setdefault(ps, []).append(int(ppl.slot[u]))
        return {k: sorted(v) for k, v in out.items()}
    a = fine_slots_by_parent([10, 20, 30])
    # reversed order + matching counts must give identical slots per parent
    def reversed_run():
        ppl = make_people(n=100)
        ppl.spawn_fine(ss.uids([99]), np.array([2]), 4)        # unrelated spawn first
        new = ppl.spawn_fine(ss.uids([30, 20, 10]), np.array([1, 3, 2]), 4)
        out = {}
        for u in new:
            ps = int(ppl.slot[ss.uids([int(ppl.parent[u])])][0])
            out.setdefault(ps, []).append(int(ppl.slot[u]))
        return {k: sorted(v) for k, v in out.items()}
    b = reversed_run()
    for ps, slots in a.items():
        assert slots == b[ps], 'fine slots must be a pure function of parent slot + index'
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_slots_deterministic_function_of_parent -q`
Expected: PASS (the implementation is already deterministic; this is the guard). If it FAILS, the slot formula is order-dependent — fix before continuing.

- [ ] **Step 3: Write + run the non-perturbation test (failure mode B)**

```python
def test_spawn_fine_does_not_perturb_other_agents():
    def draws_after(spawn):
        sim = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=5, rand_seed=1)
        sim.init()
        ppl = sim.people
        if spawn:
            ppl.spawn_fine(ss.uids([10, 11, 12, 13, 14]), np.array([5, 5, 5, 5, 5]), 5)
        d = ss.normal(loc=0, scale=1, name='probe')
        d.init(sim=sim, module=sim.diseases.sir)
        untouched = ppl.auids[~ppl.fine[ppl.auids]]
        return {int(ppl.slot[u]): float(v) for u, v in zip(untouched, d.rvs(untouched))}
    none = draws_after(False)
    some = draws_after(True)
    common = [s for s in none if s in some]
    assert len(common) > 100
    for s in common:
        assert none[s] == some[s], f'slot {s} draw changed because other agents spawned fine'
```

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_does_not_perturb_other_agents -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): spawn_fine CRN reproducibility + non-perturbation (failure modes A, B)"
```

---

### Task 3: Independence and variance reduction (failure mode C)

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `spawn_fine`; `ss.bernoulli` probe.

- [ ] **Step 1: Write the failing variance-reduction test**

```python
def test_spawn_fine_estimator_unbiased_and_lower_variance():
    # The rare-outcome estimator built via spawn_fine must be ~unbiased and have lower
    # variance than a single draw per body. Model: draw the rare event for `ratio`
    # sub-resolutions per body, spawn one fine agent per success; the scale-weighted
    # count of fine agents estimates the expected rare-event count.
    P_RARE = 0.05
    RATIO = 10
    N = 2000
    N_SEEDS = 40

    def single_count(seed):
        sim = ss.Sim(n_agents=N, diseases='sir', networks='random', dur=2, rand_seed=seed)
        sim.init(); ppl = sim.people
        d = ss.bernoulli(p=P_RARE, name='rare')
        d.init(sim=sim, module=sim.diseases.sir, seed=sim.pars.rand_seed)
        return float((ppl.scale[ppl.auids] * d.rvs(ppl.auids)).sum())

    def resolved_count(seed):
        sim = ss.Sim(n_agents=N, diseases='sir', networks='random', dur=2, rand_seed=seed)
        sim.init(); ppl = sim.people
        parents = ppl.auids.copy()
        d = ss.bernoulli(p=P_RARE, name='rare')
        d.init(sim=sim, module=sim.diseases.sir, seed=sim.pars.rand_seed)
        # draw the rare event RATIO times per body, count successes per body
        hits = np.zeros(len(parents), dtype=int)
        for _ in range(RATIO):
            hits += np.asarray(d.rvs(parents)).astype(int)
        fine = ppl.spawn_fine(parents, hits, RATIO)
        # scale-weighted count of materialized fine (cancer) agents
        return float(ppl.scale[fine].sum()) if len(fine) else 0.0

    truth = P_RARE * N
    base = np.array([single_count(s) for s in range(N_SEEDS)])
    res = np.array([resolved_count(s) for s in range(N_SEEDS)])
    assert abs(base.mean() - truth) / truth < 0.15
    assert abs(res.mean() - truth) / truth < 0.15      # unbiased
    assert res.var() < base.var() * 0.6                # variance materially reduced
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_estimator_unbiased_and_lower_variance -q`
Expected: PASS. If variance is not reduced, the per-success draws are correlated (slot collision) — investigate via the distinct-slots assertion below.

- [ ] **Step 3: Write + run the distinct-slots test**

```python
def test_spawn_fine_siblings_distinct_slots():
    ppl = make_people(n=500)
    parents = ss.uids(np.arange(0, 300))
    new = ppl.spawn_fine(parents, np.full(300, 10), 10)   # all parents, full count
    fine_slots = np.asarray(ppl.slot[new])
    assert len(np.unique(fine_slots)) == len(fine_slots)          # all distinct
    base_slots = np.asarray(ppl.slot[parents])
    assert len(np.intersect1d(fine_slots, base_slots)) == 0       # disjoint from parents
```

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_siblings_distinct_slots -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): spawn_fine independence + variance reduction (failure mode C)"
```

---

### Task 4: Integration — weighting, network/demographics exclusion, backward compatibility

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `spawn_fine`; `ss.Sim` with `networks`, `ss.Deaths`, `ss.Births`; `people.count`, `people.epi_flows`.

- [ ] **Step 1: Write the weighting/exclusion test**

```python
class SpawnFineSusceptibles(ss.Intervention):
    """At ti=2, resolve each susceptible as if a rare event hit 1 of `ratio` sub-resolutions."""
    def __init__(self, ratio=5, name=None):
        super().__init__(name=name); self.ratio = ratio
    def step(self):
        if self.sim.ti == 2:
            uids = self.sim.people.sir.susceptible.uids
            if len(uids):
                self.sim.people.spawn_fine(uids, np.ones(len(uids), dtype=int), self.ratio)

def test_spawn_fine_weighting_and_exclusion():
    sim = ss.Sim(n_agents=400, diseases='sir', networks='random', dur=10, rand_seed=1,
                 interventions=SpawnFineSusceptibles(ratio=5))
    sim.run()
    ppl = sim.people
    fine = ppl.auids[ppl.fine[ppl.auids]]
    assert len(fine) > 0
    # fine agents carry no network edges
    net = sim.networks[0]
    endpoints = set(np.asarray(net.edges.p1).tolist()) | set(np.asarray(net.edges.p2).tolist())
    assert not (set(int(u) for u in fine) & endpoints)
    # fine agents carry no demographic body weight
    assert np.isclose(ppl.epi_flows(fine), 0.0)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_weighting_and_exclusion -q`
Expected: PASS (fine excluded from network via `~fine`; epi_weight 0 ⇒ epi_flows 0). No production code change expected; if it fails, the `fine`/`epi_weight` tagging from Task 1 is wrong.

- [ ] **Step 3: Write + run the backward-compatibility test**

```python
def test_spawn_fine_absent_is_bit_identical():
    s1 = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=20, rand_seed=3); s1.run()
    s2 = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=20, rand_seed=3); s2.run()
    for key in s1.results.sir.keys():
        a = np.asarray(s1.results.sir[key]); b = np.asarray(s2.results.sir[key])
        if np.issubdtype(a.dtype, np.number):
            assert np.array_equal(a, b, equal_nan=True)
        else:
            assert np.array_equal(a, b)
    assert (s1.people.scale.raw == 1).all()
    assert (s1.people.epi_weight.raw == 1).all()
    assert not s1.people.fine.raw.any()
```

Run: `python -m pytest tests/test_multiscale.py::test_spawn_fine_absent_is_bit_identical -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): spawn_fine weighting/exclusion + backward-compat"
```

---

### Task 5: Route `split` through the shared resolution-scheme guard

**Files:**
- Modify: `starsim/people.py` (`split` — use `_claim_resolution_scheme`)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `_claim_resolution_scheme` from Task 1.

- [ ] **Step 1: Write the failing mixing-guard test**

```python
def test_split_and_spawn_fine_cannot_mix():
    ppl = make_people(n=100)
    ppl.split(ss.uids([1, 2]), 5)
    try:
        ppl.spawn_fine(ss.uids([3]), np.array([1]), 5)  # different scheme, same offset region
        assert False, 'expected ValueError mixing split + spawn_fine'
    except ValueError:
        pass
    # spawn_fine-only sim: a second spawn_fine with the same ratio is fine
    ppl2 = make_people(n=100)
    ppl2.spawn_fine(ss.uids([1]), np.array([1]), 5)
    new = ppl2.spawn_fine(ss.uids([2]), np.array([2]), 5)
    assert len(new) == 2
```

- [ ] **Step 2: Run it (fails: split doesn't claim the scheme yet)**

Run: `python -m pytest tests/test_multiscale.py::test_split_and_spawn_fine_cannot_mix -q`
Expected: FAIL (split currently uses its own `_split_n_sib` guard, not the shared one, so mixing is not yet rejected).

- [ ] **Step 3: Route `split` through the shared guard**

In `starsim/people.py` `split`, replace the existing `_split_n_sib` mixed-ratio check with a call to the shared guard, preserving the original ValueError intent:
```python
        # (replaces the prev_n_sib / _split_n_sib block)
        self._claim_resolution_scheme('split', ratio)
```
Keep `_split_n_sib` assignment if other code reads it; otherwise remove it. Verify the existing `test_split_rejects_mixed_ratio` still passes (same-scheme/different-ratio must still raise).

- [ ] **Step 4: Run the guard test + the existing split mixed-ratio test**

Run: `python -m pytest tests/test_multiscale.py::test_split_and_spawn_fine_cannot_mix tests/test_multiscale.py::test_split_rejects_mixed_ratio -q`
Expected: both PASS.

- [ ] **Step 5: Run the full multiscale suite**

Run: `python -m pytest tests/test_multiscale.py -q`
Expected: all PASS (no regression to split's existing behavior).

- [ ] **Step 6: Commit**

```bash
git add starsim/people.py tests/test_multiscale.py
git commit -m "refactor(multiscale): share one-resolution-scheme guard between split and spawn_fine"
```

---

## Self-Review

**Spec coverage** (against `2026-06-23-spawn-fine-rare-outcome-resolution-design.md`):
- `spawn_fine(parent_uids, n_events, ratio)` signature + behavior — Task 1. ✓
- Conservation (both axes) — Task 1 (mechanics + edge). ✓
- CRN-safe reserved-block slots, reproducibility, non-perturbation — Tasks 1–2. ✓
- Independence / variance reduction — Task 3. ✓
- Weighting (fine excluded from network + demographics; scale counts in results) — Task 4. ✓
- Backward compatibility — Task 4. ✓
- `k=ratio` ⇒ `scale 0`; `k=0` no-op — Task 1 edge. ✓
- One-scheme-per-sim guard (split/spawn_fine) — Task 5. ✓
- Non-goals (merge, event draw, transmission intensity) — not implemented, by design. ✓

**Placeholder scan:** none. Each code step shows complete code; each test step shows the full test and the exact command + expected outcome.

**Type consistency:** `spawn_fine(parent_uids: uids, n_events: int array, ratio: int) -> ss.uids`; `_claim_resolution_scheme(scheme: str, ratio: int) -> None`; reuses `grow(n, new_slots) -> uids`, `_split_slot_offset: int`. Consistent across tasks.

## Open items (resolve during execution)

- Confirm `self.states.values()` is the correct iterable of per-agent state arrays to copy parent→child in 3.4.0 (mirror exactly what `split` does — if `split` uses a different accessor, match it).
- Confirm whether any code reads `_split_n_sib` before removing it in Task 5 (grep first).
- Reserved-slot range sizing / `size = slots.max()+1` draw-vector cost at high spawn volume — same concern as `split`; quantify if large-population perf matters.
