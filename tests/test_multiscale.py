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
