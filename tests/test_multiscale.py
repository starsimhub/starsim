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
