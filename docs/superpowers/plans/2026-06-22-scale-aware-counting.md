# Scale-aware result counting Implementation Plan

> **For agentic workers:** Implement task-by-task with TDD. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make Starsim's result counting count agents by their scale-weighted sum
(`scale[uids].sum()`) instead of raw `count_nonzero`/`len`, so multiscale (fine-scale) agents are
tallied by the population they represent — while remaining bit-identical for non-multiscale sims.

**Architecture:** One scale-aware choke-point — `Arr.count()` — returns `scale[truthy].sum()`
(equals the raw nonzero count when all scales are 1). Route the auto `n_<state>` fills and the
standard flows/rates through scale-weighted counting. `pop_scale` stays orthogonal: it is a global
factor applied once at finalize (`modules.py:772`, `sim.py:566`); per-agent `scale` is relative
(default 1.0), so the two never double-count. Auto `n_<state>` results become `float` to preserve
fractional scale-weighted counts.

**Tech Stack:** Python, NumPy, Starsim, pytest.

## Global Constraints

- Backward compatibility is absolute: with all `scale == 1`, every existing test — especially
  `tests/test_baselines.py` — produces equal results. `scale[truthy].sum()` must equal
  `count_nonzero` to floating tolerance when scales are 1.
- Do NOT change `pop_scale` handling. It is orthogonal and applied once at finalize.
- Per-agent `scale` is relative (default 1.0); `split()` already conserves `sum(scale)`.
- Conservation: scale-weighted counts of a split cohort equal the coarse-agent count they replaced.

---

## File Structure

- `starsim/arrays.py` — `Arr.count()` becomes scale-weighted (the choke-point).
- `starsim/modules.py` — auto `n_<state>` result dtype int->float; fill via `state.count()`.
- `starsim/people.py` — auto `n_<state>` dtype int->float; fills via scale-weighted counting.
- `starsim/diseases.py` — `n_infections`, `prevalence` (num and denom), `new_deaths`,
  `n_not_at_risk` scale-weighted.
- `tests/test_multiscale.py` — new scale-aware counting tests.

---

### Task 1: Make `Arr.count()` scale-weighted

**Files:**
- Modify: `starsim/arrays.py` (`count`, ~line 493)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Produces: `Arr.count() -> float` = sum of `people.scale` over active agents whose value is truthy.
  Equals raw nonzero count when all scales are 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale.py (append)
def test_count_is_scale_weighted():
    ppl = make_people(n=100)
    # all scale 1 -> count == raw nonzero count
    assert ppl.alive.count() == 100
    # split 10 agents by ratio 5: alive raw count grows to 100 + 10*4 = 140 agents,
    # but the scale-weighted count must stay 100 (population represented is conserved)
    ppl.split(ss.uids(np.arange(10)), 5)
    assert len(ppl.auids) == 140
    assert np.isclose(ppl.alive.count(), 100.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_multiscale.py::test_count_is_scale_weighted -q`
Expected: FAIL — `count()` returns 140 (raw), not 100.

- [ ] **Step 3: Implement scale-weighted count**

In `starsim/arrays.py`, replace `Arr.count`:

```python
    def count(self):
        """
        Scale-weighted count of nonzero (truthy) values among active agents.

        Equals the raw nonzero count when every agent's `scale` is 1 (the default), and
        counts fine-scale agents by the population they represent under multiscale.
        """
        truthy = np.asarray(self.values) != 0
        return float(self.people.scale.values[truthy].sum())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_multiscale.py::test_count_is_scale_weighted -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starsim/arrays.py tests/test_multiscale.py
git commit -m "feat(arrays): scale-weighted Arr.count()"
```

---

### Task 2: Route auto `n_<state>` fills through scale-weighted counting (float dtype)

**Files:**
- Modify: `starsim/modules.py` (result creation ~line 650; fill ~line 754)
- Modify: `starsim/people.py` (result creation ~line 261/276; fill ~line 572)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `Arr.count()` (Task 1).
- Produces: `n_<state>` results are `dtype=float`; module fills use `state.count()`; People fills
  use `getattr(self, state.name).count()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multiscale.py (append)
def test_module_state_counts_are_scale_weighted_in_a_run():
    # A sim where a subset is split mid-run: n_susceptible etc. count represented population.
    class SplitAll(ss.Intervention):
        def __init__(self, ratio=5, ti_split=2, name=None):
            super().__init__(name=name); self.ratio=ratio; self.ti_split=ti_split
        def step(self):
            if self.sim.ti == self.ti_split:
                ppl = self.sim.people
                self._before = ppl.sir.susceptible.count()
                ppl.split(ppl.sir.susceptible.uids, self.ratio)
                self._after = ppl.sir.susceptible.count()
    iv = SplitAll(ratio=5, ti_split=2)
    ss.Sim(n_agents=300, diseases='sir', networks='random', dur=6, rand_seed=1, interventions=iv).run()
    # splitting susceptibles must not change the scale-weighted susceptible count
    assert np.isclose(iv._before, iv._after)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_multiscale.py::test_module_state_counts_are_scale_weighted_in_a_run -q`
Expected: FAIL — `count()` already scale-weighted from Task 1, so this may PASS for the count()
assertion; if it passes, it confirms Task 1 plumbing. The dtype/fill changes below are verified by
Task 3's fractional-precision test. (If it passes here, record that and proceed.)

- [ ] **Step 3: Change dtype to float and fills to count()**

`modules.py` ~650:
```python
            results += ss.Result(f'n_{state.name}', dtype=float, scale=True, label=state.label)
```
`modules.py` ~754:
```python
            self.results[f'n_{state.name}'][self.ti] = state.count()
```
`people.py` ~261 (the `kw` dict): change `dtype=int` to `dtype=float`.
`people.py` ~572:
```python
            res[f'n_{state.name}'][ti] = getattr(self, state.name).count()
```

- [ ] **Step 4: Run the multiscale + baseline suites**

Run: `python -m pytest tests/test_multiscale.py tests/test_baselines.py -q`
Expected: PASS. Baselines unchanged because scale-weighted == raw count when scale==1 (float values
equal the stored integer values).

- [ ] **Step 5: Commit**

```bash
git add starsim/modules.py starsim/people.py tests/test_multiscale.py
git commit -m "feat: scale-weighted auto n_<state> result counting (float dtype)"
```

---

### Task 3: Scale-weight People flows and Disease flows/rates

**Files:**
- Modify: `starsim/people.py` (`update_results`: new_deaths, new_emigrants ~line 573-574)
- Modify: `starsim/diseases.py` (`update_results`: n_infections ~408, prevalence ~416; and the
  SIR-level n_not_at_risk/prevalence/new_deaths ~605-607)
- Test: `tests/test_multiscale.py`

**Interfaces:**
- Consumes: `People.scale`, `Arr.count()`.
- Produces: flows and rates counted scale-weighted; rate denominators use scale-weighted alive count.

- [ ] **Step 1: Write the failing test (fractional precision)**

```python
# tests/test_multiscale.py (append)
def test_rare_flow_is_fractional_under_multiscale():
    # Split a single agent into 10; mark exactly one fine sibling as newly infected.
    # The scale-weighted new-infection flow must be 0.1, not 0 (int) or 1 (raw).
    ppl = make_people(n=100)
    new = ppl.split(ss.uids([0]), 10)
    one = ss.uids([int(new[0])])
    # scale-weighted count of that one fine agent
    val = ppl.scale[one].sum()
    assert np.isclose(val, 0.1)
```

- [ ] **Step 2: Run to verify it passes (precondition) / fails for int-truncation**

Run: `python -m pytest tests/test_multiscale.py::test_rare_flow_is_fractional_under_multiscale -q`
Expected: PASS at the People.scale level (this asserts the scale arithmetic). It guards the
principle that flows must be stored as float; the dtype change in Task 2 enables it.

- [ ] **Step 3: Scale-weight the flow/rate fills**

`people.py` `update_results` (~573-575):
```python
        res.new_deaths[ti]    = self.scale[(self.ti_dead == ti).uids].sum()
        res.new_emigrants[ti] = self.scale[(self.ti_removed == ti).uids].sum()
        res.cum_deaths[ti]    = np.sum(res.new_deaths[:ti])
```
Ensure `new_deaths`, `new_emigrants`, `cum_deaths` Results are `dtype=float` (people.py `kw`).

`diseases.py` (~408, 416):
```python
        infected_now = ss.uids(np.nonzero(np.round(self.ti_infected.values) == ti)[0])  # active-index mask -> uids
        n_infections = self.sim.people.scale[self.infected.uids[np.round(self.infected.uids... )]]  # see note
```
NOTE for implementer: the cleanest form mirrors the existing logic but scale-weights the matched
agents. Replace `np.count_nonzero(np.round(self.ti_infected) == ti)` with the scale-weighted sum
over the agents whose `ti_infected` rounds to `ti`:
```python
        newly = (np.round(self.ti_infected) == ti)            # BoolArr over active agents
        n_infections = self.sim.people.scale[newly.uids].sum()
```
and the prevalence denominator with the scale-weighted alive count:
```python
        res.prevalence[ti] = res.n_infected[ti] / self.sim.people.alive.count()
```
SIR-level (~605-607):
```python
        self.results.n_not_at_risk[ti] = self.not_at_risk.count()
        self.results.prevalence[ti]    = self.affected.count() / self.sim.people.alive.count()
        self.results.new_deaths[ti]    = self.sim.people.scale[(self.ti_dead == ti).uids].sum()
```
Ensure the relevant Disease/SIR Results are `dtype=float`.

- [ ] **Step 4: Run multiscale + baseline + disease suites**

Run: `python -m pytest tests/test_multiscale.py tests/test_baselines.py tests/test_diseases.py -q`
Expected: PASS. Baselines unchanged (scale==1).

- [ ] **Step 5: Commit**

```bash
git add starsim/people.py starsim/diseases.py tests/test_multiscale.py
git commit -m "feat: scale-weighted flows and rates"
```

---

## Deferred to follow-on plans

- **Scale-aware demographics** — births/deaths/migration proportional to scale-weighted population.
- **Network non-transmission flag** — networks exclude fine agents from transmission.

## Self-review notes

- Spec coverage: implements Primitive 2 (scale-aware counting) and confirms `pop_scale`
  orthogonality (no unification needed). Primitives 3-4 deferred.
- The dtype int->float change is the subtle part; the backward-compat guardrail (`test_baselines`)
  is the gate — scale-weighted counts equal stored integer values when scale==1.
- Risk: any result whose dtype stays int will truncate fractional multiscale counts; Task 3's
  fractional test guards the principle. A full audit of every flow's dtype is part of Task 3.
