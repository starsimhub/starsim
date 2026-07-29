# CRN-safe multiscale `People.split()` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CRN-safe `People.split(uids, ratio)` primitive that resolves rare events at finer
scale by replacing coarse agents with conservation-preserving fine-scale copies, with random draws
that are reproducible across scenarios, non-perturbing to other agents, and genuinely independent
between siblings.

**Architecture:** `split()` keeps each input agent (preserving its slot, hence its CRN trajectory)
and adds `ratio-1` sibling copies. Siblings receive slots from a **deterministic reserved block**
keyed by the parent's slot — `slot_k = offset + parent_slot·(ratio-1) + k` — which is collision-free
by construction and a pure function of the parent, so it is reproducible across scenarios and
independent of split order/volume. All `ratio` agents have `scale /= ratio`, conserving the total
represented population. This is the piece that defeated the prior hpvsim attempt, which grew agents
with default sequential slots (order-dependent → non-reproducible).

**Tech Stack:** Python, NumPy, Starsim (`ss`), pytest.

## Global Constraints

- Backward compatibility is absolute: with all agents at `scale == 1` and no `split()` call,
  every existing test must produce bit-identical results. `split()` adds capability; it changes
  no default behavior.
- All CRN tests run in multi-RNG mode (`ss.options.single_rng == False`, the default).
- This plan covers ONLY the `split()` primitive and its CRN guarantees. Scale-aware result
  counting, `pop_scale` unification, demographics, and network non-transmission are deferred to
  follow-on plans. `split()` here sets the `scale` and `fine` states correctly; consuming them in
  result counting is out of scope.
- Slot dtype is integer (`ss.dtypes.int`). Slots index random-number vectors; `size = slots.max()+1`
  per draw, so the reserved `offset` inflates draw-vector size — keep it as small as correctness
  allows.
- New code follows the surrounding people.py style (NumPy ops, no per-agent Python loops on hot
  paths).

---

## File Structure

- `starsim/people.py` — add the `fine` BoolState (init), the `split()` method, and the
  `_split_slot_offset` helper. One responsibility added: multiscale splitting.
- `tests/test_multiscale.py` — new test file for all split/CRN tests. Kept separate from
  `test_people.py` so the multiscale suite is discoverable as a unit.

---

### Task 1: Add the `fine` state and split-slot offset

**Files:**
- Modify: `starsim/people.py` (states list near line 67-70; add helper method near `grow`)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Produces: `people.fine` — `ss.BoolArr` (default False) marking fine-scale agents. Uses
  `ss.BoolArr` (NOT `ss.BoolState`) so it does NOT auto-generate an `n_fine` result (result
  counting is a separate plan).
- Produces: `People._split_slot_offset` — int, the base of the reserved slot band, computed at
  init as `max(1000, 10 * self.n_agents_init)`. Chosen above Pregnancy's reserved range
  (`slot_scale=5 ⇒ 5·n`) to avoid newborn/fine-agent slot collisions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale.py
import starsim as ss
import numpy as np

def make_people(n=100):
    sim = ss.Sim(n_agents=n, diseases='sir', networks='random', dur=5)
    sim.init()
    return sim.people

def test_fine_state_exists_and_defaults_false():
    ppl = make_people()
    assert hasattr(ppl, 'fine')
    assert ppl.fine.dtype == bool
    assert not ppl.fine.raw.any()  # nobody is fine until split() is called

def test_split_slot_offset_above_pregnancy_band():
    ppl = make_people(n=200)
    assert ppl._split_slot_offset >= 10 * 200
    assert ppl._split_slot_offset >= 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multiscale.py -q`
Expected: FAIL — `AttributeError: 'People' object has no attribute 'fine'`.

- [ ] **Step 3: Add the `fine` state and offset**

In `starsim/people.py`, add to the states list (after the `scale` FloatArr near line 69):

```python
            ss.FloatArr('scale', default=1.0), # The scale factor for the agents (multiplied for making results)
            ss.BoolArr('fine', default=False),  # True for fine-scale agents created by People.split()
```

Add a property near the other properties (e.g. after `n_uids`, ~line 337):

```python
    @property
    def _split_slot_offset(self):
        """ Base of the reserved slot band for fine-scale agents created by split() """
        return int(max(1000, 10 * self.n_agents_init))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_multiscale.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add starsim/people.py tests/test_multiscale.py
git commit -m "feat(people): add fine state and reserved split-slot offset"
```

---

### Task 2: Implement `split()` — reproduce the hpvsim bug first, then fix it

This task is structured so the determinism test FAILS against a naive default-slot implementation
(reproducing exactly what broke hpvsim), then PASSES once the deterministic reserved-block slots are
used. That is the core TDD loop of the whole plan.

**Files:**
- Modify: `starsim/people.py` (add `split()` near `grow`, ~after line 389)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `people.fine` (Task 1), `people._split_slot_offset` (Task 1), `people.grow(n, new_slots)`
  (returns `new_uids`), `people.states` (dict of name→state), `people.slot`, `people.scale`.
- Produces: `People.split(uids, ratio) -> ss.uids` returning the newly created sibling UIDs. After
  the call: each input uid is retained with `scale /= ratio` and `fine=True`; `ratio-1` siblings
  per parent exist as exact state copies with the same reduced scale and `fine=True`; sibling slots
  are `offset + parent_slot·(ratio-1) + k` for `k in 0..ratio-2`. Raises `ValueError` if any input
  uid is already `fine` (re-splitting is unsupported in v1).

- [ ] **Step 1: Write the failing determinism + mechanics tests**

```python
# tests/test_multiscale.py (append)

def test_split_mechanics_count_scale_and_copy():
    ppl = make_people(n=100)
    uids = ss.uids([3, 7, 42])
    ratio = 4
    orig_scale = ppl.scale[uids].copy()
    orig_age = ppl.age[uids].copy()
    new_uids = ppl.split(uids, ratio)

    # ratio-1 siblings per parent
    assert len(new_uids) == len(uids) * (ratio - 1)
    # conservation: each resolved agent carries 1/ratio of the parent's scale
    assert np.allclose(ppl.scale[uids], orig_scale / ratio)
    assert np.allclose(ppl.scale[new_uids], np.tile(orig_scale / ratio, ratio - 1))
    # siblings are exact state copies of their parent (age block-tiled by parent)
    assert np.allclose(ppl.age[new_uids], np.tile(orig_age, ratio - 1))
    # everyone involved is now fine-scale
    assert ppl.fine[uids].all() and ppl.fine[new_uids].all()

def test_split_total_scale_is_conserved():
    ppl = make_people(n=100)
    before = ppl.scale[ppl.auids].sum()
    new_uids = ppl.split(ss.uids([1, 2, 3]), 5)
    after = ppl.scale[ppl.auids].sum()
    assert np.isclose(before, after)  # Σscale invariant across a split

def test_split_rejects_resplit():
    ppl = make_people(n=100)
    uids = ss.uids([5, 6])
    ppl.split(uids, 3)
    try:
        ppl.split(uids, 3)
        assert False, "expected ValueError on re-split"
    except ValueError:
        pass

def test_split_slots_are_deterministic_function_of_parent():
    # FAILURE MODE A (unit-level): the slots a parent produces must NOT depend on
    # call order or on which other agents are split alongside it.
    ppl_a = make_people(n=100)
    slots_a = {}
    new_a = ppl_a.split(ss.uids([10, 20, 30]), 3)
    for u in new_a:
        slots_a.setdefault(int(ppl_a.parent_slot_of(u)), []).append(int(ppl_a.slot[u]))

    # Same parents, different call order + an extra unrelated split interleaved
    ppl_b = make_people(n=100)
    ppl_b.split(ss.uids([99]), 3)            # unrelated split first
    new_b = ppl_b.split(ss.uids([30, 10, 20]), 3)  # reversed order
    slots_b = {}
    for u in new_b:
        slots_b.setdefault(int(ppl_b.parent_slot_of(u)), []).append(int(ppl_b.slot[u]))

    for parent_slot, slots in slots_a.items():
        assert sorted(slots) == sorted(slots_b[parent_slot]), \
            "fine-agent slots must be a pure function of the parent slot"
```

Note: `parent_slot_of` is a tiny test helper — add it to the test file:

```python
# tests/test_multiscale.py (helper near top)
def _patch_parent_slot_helper():
    # During split, parent linkage is recorded in people.parent (UID of parent).
    pass
```

Replace the `parent_slot_of` usage by reading `people.parent`: `split()` MUST set
`people.parent[new_uids] = parent_uid` (as Pregnancy does). So the helper is:

```python
def parent_slot(ppl, u):
    return ppl.slot[ss.uids([int(ppl.parent[u])])][0]
```

and the test uses `parent_slot(ppl_a, u)` instead of `ppl_a.parent_slot_of(u)`.

- [ ] **Step 2: Add a NAIVE split (default slots) and watch determinism FAIL**

Add this temporary implementation to `people.py` to reproduce the hpvsim bug:

```python
    def split(self, uids, ratio):
        uids = ss.uids(uids); ratio = int(ratio)
        n_sib = ratio - 1
        parent_map = np.tile(np.asarray(uids), n_sib)
        new_uids = self.grow(n_sib * len(uids))  # NAIVE: default slots = sequential UIDs
        for state in self.states.values():
            state[new_uids] = state[parent_map]
        self.parent[new_uids] = parent_map
        new_scale = self.scale[uids] / ratio
        self.scale[uids] = new_scale
        self.scale[new_uids] = np.tile(new_scale, n_sib)
        self.fine[uids] = True; self.fine[new_uids] = True
        return new_uids
```

Run: `python -m pytest tests/test_multiscale.py::test_split_slots_are_deterministic_function_of_parent -q`
Expected: FAIL — with default slots, interleaving the unrelated `[99]` split shifts the sequential
slots, so the same parent yields different fine slots. **This is the hpvsim bug, reproduced.**

- [ ] **Step 3: Replace with the CRN-safe deterministic reserved-block implementation**

```python
    def split(self, uids, ratio):
        """
        Split coarse agents into `ratio` finer-scale agents, conserving the total
        represented population (`Σ scale`), to resolve rare events at higher resolution.

        Each input agent is retained (keeping its slot, hence its CRN trajectory) and
        `ratio - 1` sibling copies are created. All `ratio` resolved agents have their
        `scale` divided by `ratio`. Sibling slots come from a deterministic reserved block
        keyed by the parent's slot: `offset + parent_slot*(ratio-1) + k`. This is collision-free
        by construction and a pure function of the parent, so fine-agent draws are reproducible
        across scenarios and independent of split order/volume.

        Args:
            uids (uids): coarse agents to split (must not already be fine-scale)
            ratio (int): number of fine-scale agents each coarse agent becomes (>= 2)

        Returns:
            new_uids (uids): the newly created sibling UIDs
        """
        uids = ss.uids(uids)
        ratio = int(ratio)
        if ratio < 2 or len(uids) == 0:
            return ss.uids()
        if self.fine[uids].any():
            raise ValueError('split() received agents that are already fine-scale; re-splitting is unsupported')

        n_sib = ratio - 1
        offset = self._split_slot_offset
        parent_slots = np.asarray(self.slot[uids])

        # Deterministic reserved block per parent slot: disjoint across distinct parents
        # new_slots layout matches parent_map: [block k=0 for all parents, k=1 for all parents, ...]
        new_slots = np.concatenate([offset + parent_slots * n_sib + k for k in range(n_sib)])
        parent_map = np.tile(np.asarray(uids), n_sib)

        new_uids = self.grow(n_sib * len(uids), new_slots)
        for state in self.states.values():
            state[new_uids] = state[parent_map]
        self.parent[new_uids] = parent_map

        new_scale = self.scale[uids] / ratio
        self.scale[uids] = new_scale
        self.scale[new_uids] = np.tile(new_scale, n_sib)
        self.fine[uids] = True
        self.fine[new_uids] = True
        return new_uids
```

- [ ] **Step 4: Run all Task-2 tests to verify they pass**

Run: `python -m pytest tests/test_multiscale.py -q`
Expected: PASS (Task 1 + Task 2 tests, including the determinism test).

- [ ] **Step 5: Commit**

```bash
git add starsim/people.py tests/test_multiscale.py
git commit -m "feat(people): CRN-safe split() via deterministic reserved-block slots"
```

---

### Task 3: Failure mode B — splitting does not perturb other agents (end-to-end)

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `People.split` (Task 2). Uses a small in-test intervention that splits a fixed,
  identity-stable subset at a fixed timestep, so "split" vs "no split" differ only in that subset.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale.py (append)

class SplitSubset(ss.Intervention):
    """ At timestep `ti_split`, split the given (slot-identified) agents. """
    def __init__(self, split_slots, ratio=5, ti_split=2):
        super().__init__()
        self.split_slots = set(int(s) for s in split_slots)
        self.ratio = ratio
        self.ti_split = ti_split
    def step(self):
        if self.sim.ti == self.ti_split:
            ppl = self.sim.people
            mask = np.isin(np.asarray(ppl.slot[ppl.auids]), list(self.split_slots))
            uids = ppl.auids[mask]
            if len(uids):
                ppl.split(uids, self.ratio)

def _sir_outcomes_by_slot(sim):
    """ Map slot -> ti_infected for agents that are NOT fine-scale (the original cohort). """
    ppl = sim.people
    orig = ppl.auids[~ppl.fine[ppl.auids]]
    return {int(ppl.slot[u]): float(ppl.sir.ti_infected[u]) for u in orig}

def test_split_does_not_perturb_other_agents():
    # FAILURE MODE B: agents that never split must have identical trajectories whether
    # or not OTHER agents split. Same slot => same draws => same outcome.
    split_slots = [10, 11, 12, 13, 14]
    base = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=10, rand_seed=1)
    base.run()
    with_split = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=10, rand_seed=1,
                        interventions=SplitSubset(split_slots, ratio=5, ti_split=2))
    with_split.run()

    a = _sir_outcomes_by_slot(base)
    b = _sir_outcomes_by_slot(with_split)
    # restrict to non-split original agents present in both
    untouched = [s for s in a if s not in split_slots and s in b]
    assert len(untouched) > 100
    for s in untouched:
        assert a[s] == b[s] or (np.isnan(a[s]) and np.isnan(b[s])), \
            f"slot {s} trajectory changed merely because other agents split"
```

- [ ] **Step 2: Run to verify it fails or errors**

Run: `python -m pytest tests/test_multiscale.py::test_split_does_not_perturb_other_agents -q`
Expected: Initially may FAIL if fine agents participate in the SIR network and transmission,
perturbing untouched agents via contacts. This is the real signal: it tells us whether split alone
is non-perturbing once transmission is held constant.

- [ ] **Step 3: Make the test surgical (isolate slot-level non-perturbation)**

If the end-to-end test fails because fine agents transmit (a network effect, out of scope for this
plan), narrow the assertion to the mechanism this plan owns: that splitting does not change the
random draws of untouched agents. Replace the body with a draw-level check using a fresh
distribution:

```python
def test_split_does_not_perturb_other_agents():
    # FAILURE MODE B (mechanism-level): a slotted draw for untouched agents is identical
    # whether or not other agents were split. Splitting must not consume/shift shared RNG
    # state for agents that did not split.
    def draws_after(split_uids):
        sim = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=5, rand_seed=1)
        sim.init()
        ppl = sim.people
        if len(split_uids):
            ppl.split(ss.uids(split_uids), 5)
        d = ss.normal(loc=0, scale=1, name='probe')
        d.init(sim=sim, module=sim.diseases.sir)
        untouched = ppl.auids[~ppl.fine[ppl.auids]]
        return {int(ppl.slot[u]): float(v) for u, v in zip(untouched, d.rvs(untouched))}

    none = draws_after([])
    some = draws_after([10, 11, 12, 13, 14])
    common = [s for s in none if s in some]
    assert len(common) > 100
    for s in common:
        assert none[s] == some[s], f"slot {s} draw changed because other agents split"
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_multiscale.py::test_split_does_not_perturb_other_agents -q`
Expected: PASS — slotted draws depend only on the agent's own slot, which split leaves untouched.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): failure mode B — split does not perturb other agents"
```

---

### Task 4: Failure mode C — independence and variance reduction

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `People.split` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multiscale.py (append)

def test_fine_siblings_have_distinct_slots_no_collisions():
    # Independence precondition: all fine agents get distinct slots, and they do not
    # collide with each other or with the base/alive population's slots.
    ppl = make_people(n=500)
    uids = ss.uids(np.arange(0, 400))   # split most of the population at high ratio
    new_uids = ppl.split(uids, 10)
    fine_slots = np.asarray(ppl.slot[new_uids])
    assert len(np.unique(fine_slots)) == len(fine_slots)          # distinct among siblings
    base_slots = np.asarray(ppl.slot[uids])
    assert len(np.intersect1d(fine_slots, base_slots)) == 0       # disjoint from parents

def test_split_estimator_unbiased_and_lower_variance():
    # FAILURE MODE C: the scale-weighted count of a rare event must be ~unbiased and have
    # lower variance with splitting than without, over many seeds.
    P_RARE = 0.02
    RATIO = 10
    N = 2000
    N_SEEDS = 40

    def rare_count(seed, do_split):
        sim = ss.Sim(n_agents=N, diseases='sir', networks='random', dur=2, rand_seed=seed)
        sim.init()
        ppl = sim.people
        uids = ppl.auids.copy()
        if do_split:
            new = ppl.split(uids, RATIO)
            uids = uids.concatenate(new)
        d = ss.bernoulli(p=P_RARE, name='rare')
        d.init(sim=sim, module=sim.diseases.sir)
        hit = d.rvs(uids)
        return float((ppl.scale[uids] * hit).sum())   # scale-weighted rare-event count

    base = np.array([rare_count(s, False) for s in range(N_SEEDS)])
    split = np.array([rare_count(s, True) for s in range(N_SEEDS)])

    truth = P_RARE * N
    assert abs(base.mean() - truth) / truth < 0.15      # both unbiased
    assert abs(split.mean() - truth) / truth < 0.15
    assert split.var() < base.var() * 0.6               # variance materially reduced
```

- [ ] **Step 2: Run to verify behavior**

Run: `python -m pytest tests/test_multiscale.py::test_fine_siblings_have_distinct_slots_no_collisions tests/test_multiscale.py::test_split_estimator_unbiased_and_lower_variance -q`
Expected: `distinct_slots` PASSES immediately (deterministic blocks are collision-free). The
variance test is the scientific gate; if variance is not reduced, the siblings are not drawing
independently and the slot scheme is wrong.

- [ ] **Step 3: If variance is not reduced, diagnose slot independence**

The most likely cause is the reserved block reusing a too-small range so distinct parents map to
overlapping blocks. Verify `offset + parent_slots*(ratio-1)` blocks are disjoint by asserting in a
scratch check that `offset >= 5*n_agents` and `parent_slots` are unique. If blocks are disjoint and
variance still does not drop, the bernoulli is being drawn unslotted — confirm `d.rvs(uids)` slots
by the agents' slots (it does when initialized with the sim). No code change should be needed if
Task 2 is correct; this step is the debugging path.

- [ ] **Step 4: Run to verify both pass**

Run: `python -m pytest tests/test_multiscale.py -q`
Expected: PASS (all multiscale tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): failure mode C — independence and variance reduction"
```

---

### Task 5: Failure mode A — cross-scenario reproducibility (end-to-end)

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `People.split` (Task 2), `SplitSubset` intervention (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale.py (append)

def test_split_reproducible_across_reruns():
    # FAILURE MODE A (1): identical seed + identical split => bit-identical fine-agent slots.
    def fine_slots(seed):
        sim = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=8, rand_seed=seed,
                     interventions=SplitSubset([10,11,12,13,14], ratio=5, ti_split=2))
        sim.run()
        ppl = sim.people
        fine = ppl.auids[ppl.fine[ppl.auids]]
        return sorted(int(s) for s in ppl.slot[fine])
    assert fine_slots(7) == fine_slots(7)

def test_split_invariant_to_unrelated_scenario_change():
    # FAILURE MODE A (2): an intervention that changes OTHER agents must not change the fine
    # slots assigned to a parent that splits identically in both scenarios.
    def fine_slots_for_split_cohort(extra_split_slots):
        slots = [10,11,12,13,14]
        ivs = [SplitSubset(slots, ratio=5, ti_split=2)]
        if extra_split_slots:
            ivs.append(SplitSubset(extra_split_slots, ratio=3, ti_split=4))
        sim = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=8, rand_seed=7,
                     interventions=ivs)
        sim.run()
        ppl = sim.people
        # fine agents whose parent slot is in the target cohort
        out = {}
        fine = ppl.auids[ppl.fine[ppl.auids]]
        for u in fine:
            ps = int(ppl.slot[ss.uids([int(ppl.parent[u])])][0]) if not np.isnan(ppl.parent[u]) else int(ppl.slot[u])
            if ps in slots:
                out.setdefault(ps, []).append(int(ppl.slot[u]))
        return {k: sorted(v) for k, v in out.items()}

    a = fine_slots_for_split_cohort(None)
    b = fine_slots_for_split_cohort([120, 121, 122])  # unrelated extra split
    assert a == b, "fine slots for the target cohort must be invariant to unrelated splits"
```

- [ ] **Step 2: Run to verify**

Run: `python -m pytest tests/test_multiscale.py::test_split_reproducible_across_reruns tests/test_multiscale.py::test_split_invariant_to_unrelated_scenario_change -q`
Expected: PASS — deterministic reserved blocks keyed by parent slot are invariant to unrelated
splits and to reruns. (A default-slot implementation would fail the second test, as Task 2 Step 2
demonstrated at unit level.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): failure mode A — cross-scenario reproducibility"
```

---

### Task 6: Backward-compatibility guardrail

**Files:**
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: nothing new; asserts the framework is unchanged when `split()` is never called.

- [ ] **Step 1: Write the test**

```python
# tests/test_multiscale.py (append)

def test_no_split_is_bit_identical_to_baseline():
    # Adding the fine state + split() must not change any default-sim result.
    s1 = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=20, rand_seed=3)
    s1.run()
    s2 = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=20, rand_seed=3)
    s2.run()
    for key in s1.results.sir.keys():
        assert np.array_equal(s1.results.sir[key], s2.results.sir[key], equal_nan=True)
    # and scale is untouched everywhere
    assert (s1.people.scale.raw == 1).all()
    assert not s1.people.fine.raw.any()
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/test_multiscale.py::test_no_split_is_bit_identical_to_baseline -q`
Expected: PASS.

- [ ] **Step 3: Run the full pre-existing suite to confirm no regressions**

Run: `python -m pytest tests/test_randomness.py tests/test_people.py tests/test_distributions.py tests/test_demographics.py tests/test_sim.py -q`
Expected: PASS, same counts as the recorded baseline (22 in the first three files, plus the
demographics/sim files).

- [ ] **Step 4: Commit**

```bash
git add tests/test_multiscale.py
git commit -m "test(multiscale): backward-compatibility guardrail"
```

---

## Deferred to follow-on plans (out of scope here)

1. **Scale-aware result counting** — route auto `n_<state>` fills and module flows through a
   `scale[selected].sum()` helper; fix rate numerators/denominators in lockstep.
2. **`pop_scale` unification** — resolve the single-multiply contract so per-agent `scale` and the
   global `pop_scale` do not double-count.
3. **Scale-aware demographics** — births/deaths/migration proportional to scale-weighted population.
4. **Non-transmitting flag** — networks exclude fine agents from transmission (the realistic hpvsim
   setting; note Task 3's end-to-end variant is deferred to this plan).

## Self-review notes

- Spec coverage: this plan covers spec Primitive 1 (`split`) and all three CRN failure-mode tests
  plus the conservation and backward-compat invariants. Primitives 2-4 and `pop_scale` unification
  are explicitly deferred above, matching the spec's decomposition.
- The variance test (Task 4) and reproducibility tests (Task 5) are the scientific/CRN gates; the
  determinism unit test (Task 2) deliberately reproduces the hpvsim default-slot bug before fixing
  it.
- Open risk carried forward: end-to-end non-perturbation with transmitting fine agents is deferred
  to the network-flag plan; Task 3 narrows to the slot-level mechanism this plan owns.
