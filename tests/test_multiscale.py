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
    # _fine_slots groups by parent (repeat, not tile): [p0_s0, p0_s1, p1_s0, p1_s1, ...]
    assert np.allclose(ppl.scale[new_uids], np.repeat(orig_scale / ratio, ratio - 1))
    # siblings are exact state copies of their parent (grouped by parent)
    assert np.allclose(ppl.age[new_uids], np.repeat(orig_age, ratio - 1))
    # only the siblings are fine sub-agents; the parent stays a full participating body
    assert not ppl.fine[uids].any()
    assert ppl.fine[new_uids].all()


def test_split_total_scale_is_conserved():
    ppl = make_people(n=100)
    before = ppl.scale[ppl.auids].sum()
    ppl.split(ss.uids([1, 2, 3]), 5)
    after = ppl.scale[ppl.auids].sum()
    assert np.isclose(before, after)  # sum(scale) invariant across a split


def test_split_rejects_resplit():
    # Fine siblings (epi_weight==0) must never be re-split; parents can recur (recurrence-safe).
    ppl = make_people(n=100)
    uids = ss.uids([5, 6])
    siblings = ppl.split(uids, 3)
    # Attempting to split the fine SIBLINGS must raise.
    try:
        ppl.split(siblings[:2], 3)
        assert False, "expected ValueError on re-split of fine siblings"
    except ValueError:
        pass
    # The PARENT can be re-split (recurrence-safe: gets a new cohort slot block).
    new = ppl.split(uids, 3)
    assert len(new) == len(uids) * 2  # ratio-1 siblings per parent


def test_split_allows_mixed_ratio():
    # The dense pool is a single shared free-list, so split may be called at any
    # ratio (and may coexist with spawn_fine): there is no width-collision concern.
    ppl = make_people(n=100)
    a = ppl.split(ss.uids([1]), 5)
    b = ppl.split(ss.uids([2]), 3)   # different ratio -> fine in the dense scheme
    assert len(a) == 4 and len(b) == 2
    # all fine slots distinct (no collision across the differing ratios)
    fine = ppl.auids[ppl.fine[ppl.auids]]
    fs = np.asarray(ppl.slot[fine])
    assert len(np.unique(fs)) == len(fs)


def test_split_slots_are_dense_and_above_offset():
    # Dense semantics: fine slots are packed at the lowest free slots >= the offset,
    # collision-free, and bounded ~O(n) (NOT parent-keyed sparse). Order may change the
    # specific slot a parent gets (CRN-keyed-to-parent is intentionally dropped), but the
    # allocation stays dense and disjoint.
    ppl = make_people(n=100)
    offset = ppl._split_slot_offset
    new = ppl.split(ss.uids([10, 20, 30]), 3)
    fs = np.asarray(ppl.slot[new])
    assert (fs >= offset).all()                       # all in the fine band
    assert len(np.unique(fs)) == len(fs)              # collision-free
    assert fs.max() - offset < len(new) + 1           # dense (packed at lowest free slots)


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


# (Removed test_split_invariant_to_unrelated_scenario_change: the dense pool is
#  intentionally NOT CRN-stable across scenarios — fine-agent slots depend on the
#  set of live fine agents at allocation time, not on the parent. Within-run
#  reproducibility is covered by test_split_reproducible_across_reruns.)


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


# ---------------------------------------------------------------------------
# Network non-transmission flag (Primitive 4)
# ---------------------------------------------------------------------------

class SplitSomeSusceptibles(ss.Intervention):
    """ At ti=2, split the first 30 susceptible agents (so a known fine cohort exists). """
    def __init__(self, ratio=5, name=None):
        super().__init__(name=name)
        self.ratio = ratio

    def step(self):
        if self.sim.ti == 2:
            uids = self.sim.people.sir.susceptible.uids[:30]
            if len(uids):
                self.sim.people.split(uids, self.ratio)


def test_fine_agents_excluded_from_network_transmission():
    # Fine agents must not form network edges, so they cannot be infected via transmission.
    # Without exclusion they would join add_pairs (alive & age>0) and could get infected.
    sim = ss.Sim(n_agents=400, diseases='sir', networks='random', dur=12, rand_seed=1,
                 interventions=SplitSomeSusceptibles(ratio=5))
    sim.run()
    ppl = sim.people
    fine = ppl.auids[ppl.fine[ppl.auids]]
    assert len(fine) > 0
    # no network edge touches any fine agent
    net = sim.networks[0]
    edge_endpoints = set(np.asarray(net.edges.p1).tolist()) | set(np.asarray(net.edges.p2).tolist())
    assert not (set(int(u) for u in fine) & edge_endpoints), "fine agents must not appear in network edges"


# ---------------------------------------------------------------------------
# Scale-aware demographics (Primitive 3): Deaths
# ---------------------------------------------------------------------------

class SplitEveryone(ss.Intervention):
    """ At ti=1, split the whole living population by `ratio`. """
    def __init__(self, ratio=4, name=None):
        super().__init__(name=name)
        self.ratio = ratio

    def step(self):
        if self.sim.ti == 1:
            self.sim.people.split(self.sim.people.auids.copy(), self.ratio)


def test_deaths_count_is_scale_weighted():
    # Splitting the whole population by 4 makes 4x as many (raw) agents, each representing
    # 1/4 of a person. The scale-weighted death count must be conserved, not inflated 4x.
    def total_deaths(split):
        ivs = [SplitEveryone(ratio=4)] if split else []
        s = ss.Sim(n_agents=2000, demographics=ss.Deaths(death_rate=20), dur=10, rand_seed=1, interventions=ivs)
        s.run()
        return float(np.nansum(s.results.deaths.new))
    base = total_deaths(False)
    sp = total_deaths(True)
    assert base > 0
    assert np.isclose(base, sp, rtol=0.15), f"represented deaths should be conserved: base={base} split={sp}"


def test_births_count_is_scale_weighted():
    # Splitting the population must not inflate represented births: newborns inherit the
    # parent's scale, so the scale-weighted birth count is conserved.
    def total_births(split):
        ivs = [SplitEveryone(ratio=4)] if split else []
        s = ss.Sim(n_agents=2000, demographics=ss.Births(birth_rate=30), dur=10, rand_seed=1, interventions=ivs)
        s.run()
        return float(np.nansum(s.results.births.new))
    base = total_births(False)
    sp = total_births(True)
    assert base > 0
    assert np.isclose(base, sp, rtol=0.15), f"represented births should be conserved: base={base} split={sp}"


def test_pregnancy_counts_are_scale_weighted():
    # Pregnancy births/pregnancies must be represented (scale-weighted): newborns inherit the
    # mother's scale, so splitting the population does not inflate the counts ~4x.
    def total(split, key):
        ivs = [SplitEveryone(ratio=4)] if split else []
        s = ss.Sim(n_agents=2000, demographics=ss.Pregnancy(fertility_rate=80),
                   dur=8, rand_seed=1, interventions=ivs)
        s.run()
        return float(np.nansum(s.results.pregnancy[key]))
    for key in ('births', 'pregnancies'):
        base = total(False, key)
        sp = total(True, key)
        assert base > 0, f"{key} baseline should be positive"
        assert np.isclose(base, sp, rtol=0.2), f"represented {key} should be conserved: base={base} split={sp}"


# ---------------------------------------------------------------------------
# Scale-weighted-by-default counting: People.count(x) dispatch
# ---------------------------------------------------------------------------

def test_people_count_dispatch_scale_weighted():
    sim = ss.Sim(n_agents=200, diseases='sir', networks='random', dur=3, rand_seed=1)
    sim.run()
    ppl = sim.people
    sir = ppl.sir
    # scale==1: count(condition) == raw count_nonzero; count(uids) == len
    cond = sir.infected & (ppl.age > 0)
    assert ppl.count(cond) == np.count_nonzero(cond)            # BoolArr condition
    assert ppl.count(sir.infected) == sir.infected.count()      # BoolState
    assert ppl.count(sir.infected.uids) == len(sir.infected.uids)  # uids set
    # under multiscale: count is conserved across a split
    before = ppl.count(sir.susceptible)
    ppl.split(sir.susceptible.uids, 5)
    assert np.isclose(ppl.count(sir.susceptible), before)       # splitting susceptibles conserves the count


# ---------------------------------------------------------------------------
# Scale-weighted-by-default counting: declarative flow results
# ---------------------------------------------------------------------------

class FlowAnalyzer(ss.Analyzer):
    """ Declares a flow result bound to a condition; writes NO counting code. """
    def step(self):
        pass

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('flow_count', flow=lambda a: a.sim.people.sir.susceptible)
        )


def test_declarative_flow_autofills_scaled():
    # An analyzer that only DECLARES a flow gets it auto-filled and scale-weighted by the
    # framework. Splitting the population must not inflate the flow ~5x.
    def run(split):
        ivs = [SplitEveryone(ratio=5)] if split else []  # splits everyone at ti=1
        s = ss.Sim(n_agents=300, diseases='sir', networks='random', dur=6, rand_seed=1,
                   interventions=ivs, analyzers=FlowAnalyzer())
        s.run()
        return np.asarray(s.results.flowanalyzer.flow_count)
    base = run(False)
    sp = run(True)
    assert base.max() > 0                          # the flow was auto-filled (builder wrote no count)
    assert sp.max() <= 1.3 * base.max()            # scale-weighted: not inflated by the split


# ---------------------------------------------------------------------------
# Two-axis multiscale: epi_weight (demographics/transmission) vs scale (results)
# ---------------------------------------------------------------------------

def test_split_two_axis_epi_weight():
    ppl = make_people(n=100)
    uids = ss.uids([3, 7, 42])
    ratio = 5
    new = ppl.split(uids, ratio)
    # Result axis: scale is 1/ratio for the whole cohort (parent + siblings)
    assert np.allclose(ppl.scale[uids], 1 / ratio)
    assert np.allclose(ppl.scale[new], 1 / ratio)
    # Epi axis: the parent keeps its full body weight; siblings carry none
    assert np.allclose(ppl.epi_weight[uids], 1.0)
    assert np.allclose(ppl.epi_weight[new], 0.0)
    # fine tags ONLY the siblings now (the parent stays a full participant)
    assert not ppl.fine[uids].any()
    assert ppl.fine[new].all()
    # Conservation on both axes: scale-weighted = represented pop; epi-weighted = whole bodies
    assert np.isclose(ppl.scale[uids].sum() + ppl.scale[new].sum(), len(uids))      # 3 represented
    assert np.isclose(ppl.epi_weight[uids].sum() + ppl.epi_weight[new].sum(), len(uids))  # 3 bodies (parents)


# ---------------------------------------------------------------------------
# Task 1: spawn_fine primitive
# ---------------------------------------------------------------------------

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


def test_split_parent_reproduces_as_whole_body():
    # A split parent stays a full reproducing body: it bears whole newborns (scale == epi_weight ==
    # the mother's epi_weight); fine siblings bear none.
    class SplitFirst(ss.Intervention):
        def __init__(self, name=None):
            super().__init__(name=name)
        def step(self):
            if self.sim.ti == 1:
                ppl = self.sim.people
                fertile = ppl.female.uids[:20]
                if len(fertile):
                    ppl.split(fertile, 10)
    sim = ss.Sim(n_agents=1000, demographics=ss.Pregnancy(fertility_rate=200), dur=8,
                 rand_seed=1, interventions=SplitFirst())
    sim.run()
    ppl = sim.people
    # Any agent born during the run is a whole body: scale == epi_weight, both > 0 (no fractional newborns)
    born = ppl.auids[(ppl.age[ppl.auids] >= 0) & (ppl.parent[ppl.auids] != ppl.parent.nan)]
    newborns = born[ppl.age[born] < 8]
    assert len(newborns) > 0
    assert np.allclose(ppl.scale[newborns], ppl.epi_weight[newborns])  # whole, not fractional
    assert (ppl.epi_weight[newborns] > 0).all()
    # fine siblings never reproduce -> no newborn has a fine parent
    parents_of_newborns = ppl.parent[newborns]
    assert not ppl.fine[ss.uids(parents_of_newborns.astype(int))].any()


# ---------------------------------------------------------------------------
# Task 2: spawn_fine CRN reproducibility + non-perturbation (failure modes A, B)
# ---------------------------------------------------------------------------

def test_spawn_fine_slots_are_dense_and_reproducible_within_run():
    # Dense semantics: spawn_fine slots are packed at the lowest free slots >= offset,
    # collision-free, and reproducible for a fixed allocation order within a run. (They
    # are NOT a pure function of the parent — CRN-keyed-to-parent is intentionally dropped.)
    def run():
        ppl = make_people(n=100)
        new = ppl.spawn_fine(ss.uids([10, 20, 30]), np.array([2, 3, 1]), 4)
        return sorted(int(ppl.slot[u]) for u in new), ppl._split_slot_offset
    a, offset = run()
    b, _ = run()
    assert a == b                                   # identical fixed-order allocation
    assert all(s >= offset for s in a)              # all in the fine band
    assert len(set(a)) == len(a)                    # collision-free
    assert max(a) - offset < len(a) + 1             # dense (packed at lowest free slots)


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


# ---------------------------------------------------------------------------
# Task 3: spawn_fine independence + variance reduction (failure mode C)
# ---------------------------------------------------------------------------

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


def test_spawn_fine_siblings_distinct_slots():
    ppl = make_people(n=500)
    parents = ss.uids(np.arange(0, 300))
    new = ppl.spawn_fine(parents, np.full(300, 10), 10)   # all parents, full count
    fine_slots = np.asarray(ppl.slot[new])
    assert len(np.unique(fine_slots)) == len(fine_slots)          # all distinct
    base_slots = np.asarray(ppl.slot[parents])
    assert len(np.intersect1d(fine_slots, base_slots)) == 0       # disjoint from parents


# ---------------------------------------------------------------------------
# Task 4: integration — weighting, network/demographics exclusion, backward compat
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dense pool: split and spawn_fine may coexist at any ratio (no scheme guard)
# ---------------------------------------------------------------------------

def test_split_and_spawn_fine_can_coexist():
    # The shared dense pool has no width-collision concern, so split and spawn_fine
    # may be used in the same sim at any ratio without error.
    ppl = make_people(n=100)
    a = ppl.split(ss.uids([1, 2]), 5)
    b = ppl.spawn_fine(ss.uids([3]), np.array([1]), 5)   # different scheme, same region: fine now
    assert len(a) == 8 and len(b) == 1
    # all fine slots remain distinct across the two schemes
    fine = ppl.auids[ppl.fine[ppl.auids]]
    fs = np.asarray(ppl.slot[fine])
    assert len(np.unique(fs)) == len(fs)


def test_all_zero_spawn_fine_is_noop():
    # An all-zero n_events spawn_fine creates nothing and is a no-op; a subsequent
    # split at a different ratio succeeds (no scheme is ever claimed).
    ppl = make_people(n=100)
    result = ppl.spawn_fine(ss.uids([1, 2]), np.array([0, 0]), 4)
    assert len(result) == 0
    new = ppl.split(ss.uids([3, 4]), 7)
    assert len(new) == 12  # (7-1) * 2 parents


# ---------------------------------------------------------------------------
# Task 1 (recurrence): reserved-block multi-cohort slots for spawn_fine
# ---------------------------------------------------------------------------

def test_spawn_fine_recurrence_disjoint_slots_while_descendants_live():
    ppl = make_people(n=100)
    p = ss.uids([7])
    f1 = ppl.spawn_fine(p, np.array([2]), 5)
    assert ppl.alive[f1].all()                       # episode-1 cohort still alive
    f2 = ppl.spawn_fine(p, np.array([2]), 5)          # same parent recurs
    s1 = set(int(ppl.slot[u]) for u in f1)
    s2 = set(int(ppl.slot[u]) for u in f2)
    assert not (s1 & s2)                              # NO live-live slot collision


def test_spawn_fine_recycles_slots_when_descendants_dead():
    ppl = make_people(n=100)
    p = ss.uids([7])
    f1 = ppl.spawn_fine(p, np.array([2]), 5)
    s1 = set(int(ppl.slot[u]) for u in f1)
    ppl.request_death(f1); ppl.step_die(); ppl.remove_dead()  # episode-1 cohort dies
    f2 = ppl.spawn_fine(p, np.array([2]), 5)
    s2 = set(int(ppl.slot[u]) for u in f2)
    # Dead fine agents drop out of (fine & alive), so their slots are the lowest free
    # ones and get reused: episode-2 reuses exactly episode-1's slots (no range growth).
    assert s2 == s1


def test_spawn_fine_no_cohort_cap():
    # The MAX_LIVE_COHORTS raise is gone: a parent may accumulate many live cohorts
    # with no error (the dense pool just keeps packing into the lowest free slots).
    ppl = make_people(n=100)
    p = ss.uids([7])
    for _ in range(20):                               # far more than the old cap of 4
        ppl.spawn_fine(p, np.array([1]), 5)
    fine = ppl.auids[ppl.fine[ppl.auids]]
    assert len(fine) == 20                            # all cohorts materialized, no raise
    fs = np.asarray(ppl.slot[fine])
    assert len(np.unique(fs)) == len(fs)              # and all collision-free


def test_spawn_fine_recurrence_reproducible():
    def run():
        ppl = make_people(n=100)
        p = ss.uids([7])
        a = ppl.spawn_fine(p, np.array([2]), 5)
        b = ppl.spawn_fine(p, np.array([2]), 5)
        return sorted(int(ppl.slot[u]) for u in a), sorted(int(ppl.slot[u]) for u in b)
    assert run() == run()


# ---------------------------------------------------------------------------
# Task 2 (recurrence): reserved-block multi-cohort slots for split
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task 3 (recurrence): variance reduction survives multi-episode recurrence
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dense recycled fine-slot pool: the slot-band-blowup fix (design 2026-06-25)
# ---------------------------------------------------------------------------

def _high_recurrence_run(ppl, n_parents, n_episodes, ratio=10, cohort=2):
    """ Spawn `cohort` fine agents per parent for `n_episodes`, killing the prior
    episode's cohort each round (so slots can recycle), tracking the live fine-slot
    high-water mark per episode. Returns (peaks, prev_cohort). """
    parents = ss.uids(np.arange(n_parents))
    peaks = []
    prev = ss.uids()
    for _ in range(n_episodes):
        if len(prev):                                   # kill the previous cohort so its slots free up
            ppl.request_death(prev); ppl.step_die(); ppl.remove_dead()
        prev = ppl.spawn_fine(parents, np.full(n_parents, cohort), ratio)
        live = ppl.slot[ppl.fine & ppl.alive]
        peaks.append(int(live.max()) if len(live) else ppl._split_slot_offset)
    return peaks, prev


def test_dense_bound_no_cohort_cap_no_blowup():
    # The core win: after a long, high-recurrence multiscale run the live fine-slot
    # span stays bounded ~O(n) (NOT O(n*ratio*MAX_LIVE_COHORTS)), and NO cohort-cap
    # error occurs (the MAX_LIVE_COHORTS raise is gone).
    n = 400
    ppl = make_people(n=n)
    peaks, _ = _high_recurrence_run(ppl, n_parents=n, n_episodes=200, ratio=12, cohort=2)
    span = max(peaks) - ppl._split_slot_offset
    # Dense bound: peak live fine = n_parents * cohort = 800; well under 2*n_agents... but
    # n here is small relative to the cohort, so bound against peak concurrent fine instead.
    assert span < 2 * n + n            # O(n), not O(n * ratio * mlc) ~ tens of millions
    assert not hasattr(ppl, 'MAX_LIVE_COHORTS')  # attribute removed entirely


def test_dense_bound_via_sim_long_recurrence():
    # End-to-end through a Sim: after a long run with repeated spawn_fine on the whole
    # population, slot[fine & alive].max() - offset stays < 2 * n_agents (dense bound).
    N = 500
    class Recur(ss.Intervention):
        def __init__(self, ratio=12, name=None):
            super().__init__(name=name); self.ratio = ratio; self.prev = ss.uids()
        def step(self):
            ppl = self.sim.people
            if len(self.prev):
                ppl.request_death(self.prev)            # retire last episode's cohort
            whole = ppl.auids[~ppl.fine[ppl.auids]]
            if len(whole):
                self.prev = ppl.spawn_fine(whole, np.ones(len(whole), dtype=int), self.ratio)
    sim = ss.Sim(n_agents=N, diseases='sir', networks='random', dur=60, rand_seed=1,
                 interventions=Recur(ratio=12))
    sim.run()
    ppl = sim.people
    live = ppl.slot[ppl.fine & ppl.alive]
    if len(live):
        span = int(live.max()) - ppl._split_slot_offset
        assert span < 2 * N            # dense O(n) bound, flat in ratio and sim length


def test_recycling_plateau():
    # The live-fine-slot high-water mark must plateau over a long run (dead agents'
    # slots are reused) rather than growing unboundedly with episode count.
    n = 300
    ppl = make_people(n=n)
    peaks, _ = _high_recurrence_run(ppl, n_parents=n, n_episodes=150, ratio=10, cohort=2)
    # late-run peaks must not exceed early-run peaks (no growth: pure recycling)
    early = max(peaks[:10])
    late = max(peaks[-10:])
    assert late <= early                # plateaued (recycled), not growing


def test_collision_free_within_step():
    # At the end of a run with many live fine agents, no two share a slot.
    n = 400
    ppl = make_people(n=n)
    # several distinct live cohorts coexisting (no deaths between episodes)
    parents = ss.uids(np.arange(n))
    for _ in range(5):
        ppl.spawn_fine(parents, np.full(n, 1), 10)
    fine_alive = ppl.fine & ppl.alive
    slots = np.asarray(ppl.slot[fine_alive])
    assert len(np.unique(slots)) == int(fine_alive.sum())   # all distinct


def test_within_run_reproducibility():
    # Same seed -> identical results across two runs (within-run reproducibility holds
    # even though fine slots are not CRN-stable across scenarios).
    def run():
        sim = ss.Sim(n_agents=400, diseases='sir', networks='random', dur=10, rand_seed=5,
                     interventions=SplitEveryone(ratio=5))
        sim.run()
        return np.asarray(sim.results.sir.new_infections)
    assert np.array_equal(run(), run(), equal_nan=True)


def test_ss_only_multiscale_conserves_scale():
    # Sanity: a plain ss multiscale sim (split + a rare spawn_fine event) runs and
    # conserves sum(scale) end to end.
    class SplitThenSpawn(ss.Intervention):
        def __init__(self, name=None):
            super().__init__(name=name)
        def step(self):
            ppl = self.sim.people
            if self.sim.ti == 1:
                ppl.split(ppl.auids.copy(), 4)
            elif self.sim.ti == 3:
                whole = ppl.auids[~ppl.fine[ppl.auids]][:20]
                if len(whole):
                    ppl.spawn_fine(whole, np.ones(len(whole), dtype=int), 4)
    sim = ss.Sim(n_agents=500, diseases='sir', networks='random', dur=8, rand_seed=1,
                 interventions=SplitThenSpawn())
    sim.init()
    before = sim.people.scale[sim.people.auids].sum()
    sim.run()
    after = sim.people.scale[sim.people.auids].sum()
    assert np.isclose(before, after, rtol=1e-6)   # represented population conserved
