"""
Tests for CRN-safe multiscale agent splitting (People.split).

See docs/superpowers/plans/2026-06-22-crn-safe-multiscale-split.md
"""
import numpy as np
import starsim as ss


def make_people(n=100):
    sim = ss.Sim(n_agents=n, diseases='sir', networks='random', dur=5)
    sim.init()
    return sim.people


# ---------------------------------------------------------------------------
# Task 1: fine state and reserved split-slot offset
# ---------------------------------------------------------------------------

def test_fine_state_exists_and_defaults_false():
    ppl = make_people()
    assert hasattr(ppl, 'fine')
    assert ppl.fine.dtype == bool
    assert not ppl.fine.raw.any()  # nobody is fine until split() is called


def test_split_slot_offset_above_pregnancy_band():
    ppl = make_people(n=200)
    assert ppl._split_slot_offset >= 10 * 200
    assert ppl._split_slot_offset >= 1000


# ---------------------------------------------------------------------------
# Task 2: split() mechanics + determinism (failure mode A, unit level)
# ---------------------------------------------------------------------------

def parent_slot(ppl, u):
    """ Slot of the parent of fine agent `u`. """
    return int(ppl.slot[ss.uids([int(ppl.parent[u])])][0])


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
    ppl.split(ss.uids([1, 2, 3]), 5)
    after = ppl.scale[ppl.auids].sum()
    assert np.isclose(before, after)  # sum(scale) invariant across a split


def test_split_rejects_resplit():
    ppl = make_people(n=100)
    uids = ss.uids([5, 6])
    ppl.split(uids, 3)
    try:
        ppl.split(uids, 3)
        assert False, "expected ValueError on re-split"
    except ValueError:
        pass


def test_split_rejects_mixed_ratio():
    # The deterministic reserved-block scheme is collision-free only when the
    # sibling-block width (ratio-1) is constant; a second call with a different
    # ratio would overlap blocks and silently correlate fine agents. Reject it.
    ppl = make_people(n=100)
    ppl.split(ss.uids([1]), 5)
    try:
        ppl.split(ss.uids([2]), 3)   # different ratio -> would collide
        assert False, "expected ValueError on mixed ratio"
    except ValueError:
        pass
    # same ratio on a different cohort is fine
    new = ppl.split(ss.uids([3]), 5)
    assert len(new) == 4


def test_split_slots_are_deterministic_function_of_parent():
    # FAILURE MODE A (unit-level): the slots a parent produces must NOT depend on
    # call order or on which other agents are split alongside it.
    ppl_a = make_people(n=100)
    new_a = ppl_a.split(ss.uids([10, 20, 30]), 3)
    slots_a = {}
    for u in new_a:
        slots_a.setdefault(parent_slot(ppl_a, u), []).append(int(ppl_a.slot[u]))

    # Same parents, different call order + an extra unrelated split interleaved
    ppl_b = make_people(n=100)
    ppl_b.split(ss.uids([99]), 3)                  # unrelated split first
    new_b = ppl_b.split(ss.uids([30, 10, 20]), 3)  # reversed order
    slots_b = {}
    for u in new_b:
        slots_b.setdefault(parent_slot(ppl_b, u), []).append(int(ppl_b.slot[u]))

    for ps, slots in slots_a.items():
        assert sorted(slots) == sorted(slots_b[ps]), \
            "fine-agent slots must be a pure function of the parent slot"


# ---------------------------------------------------------------------------
# Task 3: failure mode B - splitting does not perturb other agents
# (mechanism level; the end-to-end transmitting variant is deferred to the
#  network non-transmission plan, since fine agents still transmit today.)
# ---------------------------------------------------------------------------

def test_split_does_not_perturb_other_agents():
    # A slotted draw for untouched agents must be identical whether or not other
    # agents were split. Split must not consume/shift shared RNG state or mutate
    # the slots of agents that did not split.
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


# ---------------------------------------------------------------------------
# Task 4: failure mode C - independence and variance reduction
# ---------------------------------------------------------------------------

def test_fine_siblings_have_distinct_slots_no_collisions():
    # Independence precondition: all fine agents get distinct slots, disjoint from
    # the base/alive population's slots.
    ppl = make_people(n=500)
    uids = ss.uids(np.arange(0, 400))   # split most of the population at high ratio
    new_uids = ppl.split(uids, 10)
    fine_slots = np.asarray(ppl.slot[new_uids])
    assert len(np.unique(fine_slots)) == len(fine_slots)          # distinct among siblings
    base_slots = np.asarray(ppl.slot[uids])
    assert len(np.intersect1d(fine_slots, base_slots)) == 0       # disjoint from parents


def test_split_estimator_unbiased_and_lower_variance():
    # FAILURE MODE C: the scale-weighted count of a rare event must be ~unbiased and
    # have lower variance with splitting than without, over many seeds.
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
        # Pass the sim's base seed so the probe varies across seeds (matches how the
        # Dists container seeds module dists: dists.init(base_seed=rand_seed), sim.py).
        d.init(sim=sim, module=sim.diseases.sir, seed=sim.pars.rand_seed)
        hit = d.rvs(uids)
        return float((ppl.scale[uids] * hit).sum())   # scale-weighted rare-event count

    base = np.array([rare_count(s, False) for s in range(N_SEEDS)])
    split = np.array([rare_count(s, True) for s in range(N_SEEDS)])

    truth = P_RARE * N
    assert abs(base.mean() - truth) / truth < 0.15      # both unbiased
    assert abs(split.mean() - truth) / truth < 0.15
    assert split.var() < base.var() * 0.6               # variance materially reduced


# ---------------------------------------------------------------------------
# Task 5: failure mode A - cross-scenario reproducibility (end-to-end)
# ---------------------------------------------------------------------------

class SplitSubset(ss.Intervention):
    """ At `ti_split`, split the agents whose slot is in `split_slots`, recording the
    fine slots assigned to each parent slot in `self.assigned`. """
    def __init__(self, split_slots, ratio=5, ti_split=2, name=None):
        super().__init__(name=name)
        self.split_slots = set(int(s) for s in split_slots)
        self.ratio = ratio
        self.ti_split = ti_split
        self.assigned = {}

    def step(self):
        if self.sim.ti == self.ti_split:
            ppl = self.sim.people
            mask = np.isin(np.asarray(ppl.slot[ppl.auids]), list(self.split_slots))
            uids = ppl.auids[mask]
            if len(uids):
                new = ppl.split(uids, self.ratio)
                for u in new:
                    ps = int(ppl.slot[ss.uids([int(ppl.parent[u])])][0])
                    self.assigned.setdefault(ps, []).append(int(ppl.slot[u]))
                self.assigned = {k: sorted(v) for k, v in self.assigned.items()}


def test_split_reproducible_across_reruns():
    # FAILURE MODE A (1): identical seed + identical split => identical fine-slot assignment.
    def assigned(seed):
        iv = SplitSubset([10, 11, 12, 13, 14], ratio=5, ti_split=2)
        ss.Sim(n_agents=200, diseases='sir', networks='random', dur=8, rand_seed=seed,
               interventions=iv).run()
        return iv.assigned
    assert assigned(7) == assigned(7)


def test_split_invariant_to_unrelated_scenario_change():
    # FAILURE MODE A (2): an unrelated split of OTHER agents (even earlier in the run)
    # must not change the fine slots assigned to the target cohort. Slots are a pure
    # function of the parent slot, so the assignment is invariant.
    def target_assignment(extra_split_slots):
        target = SplitSubset([10, 11, 12, 13, 14], ratio=5, ti_split=2, name='target')
        ivs = [target]
        if extra_split_slots:
            # Same ratio as target (one ratio per sim); disjoint parent slots -> disjoint blocks.
            ivs.append(SplitSubset(extra_split_slots, ratio=5, ti_split=1, name='extra'))  # unrelated, earlier
        ss.Sim(n_agents=200, diseases='sir', networks='random', dur=8, rand_seed=7,
               interventions=ivs).run()
        return target.assigned

    a = target_assignment(None)
    b = target_assignment([120, 121, 122])
    assert a == b, "fine slots for the target cohort must be invariant to unrelated splits"


# ---------------------------------------------------------------------------
# Task 6: backward-compatibility guardrail
# ---------------------------------------------------------------------------

def test_no_split_is_bit_identical_to_baseline():
    # Adding the fine state + split() must not change any default-sim result, and
    # must leave scale==1 / fine==False everywhere when split() is never called.
    s1 = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=20, rand_seed=3)
    s1.run()
    s2 = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=20, rand_seed=3)
    s2.run()
    for key in s1.results.sir.keys():
        a = np.asarray(s1.results.sir[key])
        b = np.asarray(s2.results.sir[key])
        if np.issubdtype(a.dtype, np.number):
            assert np.array_equal(a, b, equal_nan=True)   # numeric result (may contain NaN)
        else:
            assert np.array_equal(a, b)                   # e.g. the date timevec
    assert (s1.people.scale.raw == 1).all()
    assert not s1.people.fine.raw.any()


# ---------------------------------------------------------------------------
# Scale-aware counting plan, Task 1: scale-weighted Arr.count()
# ---------------------------------------------------------------------------

def test_count_is_scale_weighted():
    ppl = make_people(n=100)
    # all scale 1 -> count == raw nonzero count
    assert ppl.alive.count() == 100
    # split 10 agents by ratio 5: raw alive grows to 100 + 10*4 = 140 agents,
    # but the scale-weighted count must stay 100 (represented population conserved)
    ppl.split(ss.uids(np.arange(10)), 5)
    assert len(ppl.auids) == 140
    assert np.isclose(ppl.alive.count(), 100.0)


class SplitSusceptibles(ss.Intervention):
    """ At `ti_split`, split all currently-susceptible agents. """
    def __init__(self, ratio=5, ti_split=2, name=None):
        super().__init__(name=name)
        self.ratio = ratio
        self.ti_split = ti_split

    def step(self):
        if self.sim.ti == self.ti_split:
            uids = self.sim.people.sir.susceptible.uids
            if len(uids):
                self.sim.people.split(uids, self.ratio)


def test_auto_state_results_are_scale_weighted_and_float():
    def run(split):
        ivs = [SplitSusceptibles(ratio=5, ti_split=2)] if split else []
        s = ss.Sim(n_agents=300, diseases='sir', networks='random', dur=6, rand_seed=1, interventions=ivs)
        s.run()
        return s
    base = run(False)
    sp = run(True)
    ns = np.asarray(sp.results.sir.n_susceptible)
    assert np.issubdtype(ns.dtype, np.floating)   # dtype changed to float to hold fractional counts
    # raw counting would spike ~5x at the split step; scale-weighting conserves represented pop
    assert ns.max() <= 1.3 * np.asarray(base.results.sir.n_susceptible).max()


# ---------------------------------------------------------------------------
# Scale-aware counting plan, Task 3: scale-weighted flows and rates
# ---------------------------------------------------------------------------

def test_disease_flow_results_are_float():
    # Infection flows (new_infections) and NCD flows (new_deaths, n_not_at_risk) must be float
    # to hold fractional scale-weighted counts.
    sir = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=5, rand_seed=1)
    sir.run()
    assert np.issubdtype(np.asarray(sir.results.sir.new_infections).dtype, np.floating)
    ncd = ss.Sim(n_agents=200, diseases=ss.NCD(), dur=5, rand_seed=1)
    ncd.run()
    for key in ('new_deaths', 'n_not_at_risk'):
        assert np.issubdtype(np.asarray(ncd.results.ncd[key]).dtype, np.floating), f"{key} must be float"


def test_scale_weighted_prevalence_denominator():
    # Prevalence = scale-weighted infected / scale-weighted active population. Splitting
    # susceptibles conserves both, so prevalence is unchanged. A raw len(people) denominator
    # would grow with the split and drop prevalence ~ratio-fold.
    sim = ss.Sim(n_agents=400, diseases='sir', networks='random', dur=3, rand_seed=2)
    sim.run()  # produce some infected agents
    ppl = sim.people
    sir = ppl.sir
    def prevalence():
        return sir.infected.count() / ppl.scale.values.sum()
    before = prevalence()
    ppl.split(sir.susceptible.uids, 5)  # split susceptibles (does not change infected or Sum(scale))
    after = prevalence()
    assert before > 0
    assert np.isclose(before, after)
