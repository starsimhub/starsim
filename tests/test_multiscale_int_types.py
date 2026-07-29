"""
Multiscale should keep integer-valued quantities as ints where exact, while
scale-weighted (genuinely fractional) results stay float.

Two-axis recap:
  - `scale`      : fractional result weight (1/ratio after a split) -> float, always.
  - `epi_weight` : whole-body participation flag, only ever 0 or 1 -> integer.

Body-weighted vital-dynamics flows (filled via People.epi_flows) are always
integer-valued, so their result arrays should be integer-typed; with an integral
pop_scale the finalized result stays integer. Scale-weighted flows (People.scale_flows
/ count(), filled fractionally under split) must remain float.
"""
import numpy as np
import starsim as ss


# ---------------------------------------------------------------------------
# epi_weight is an integer participation flag (0/1), not a float
# ---------------------------------------------------------------------------

def test_epi_weight_is_integer_typed():
    sim = ss.Sim(n_agents=100, diseases='sir', networks='random', dur=3)
    sim.init()
    assert np.issubdtype(sim.people.epi_weight.dtype, np.integer)
    # default value is 1 (a whole body) for everyone before any split/spawn_fine
    assert (sim.people.epi_weight.raw == 1).all()


# ---------------------------------------------------------------------------
# Body-weighted vital-dynamics flows are integer-typed (default pop_scale = 1.0)
# ---------------------------------------------------------------------------

def test_births_results_are_integer():
    sim = ss.Sim(n_agents=500, demographics=ss.Births(birth_rate=30), dur=10, rand_seed=1)
    sim.run()
    for key in ('new', 'cumulative'):
        arr = np.asarray(sim.results.births[key])
        assert np.issubdtype(arr.dtype, np.integer), f'births.{key} should be int, got {arr.dtype}'


def test_pregnancy_body_weighted_results_are_integer():
    sim = ss.Sim(n_agents=800, demographics=ss.Pregnancy(fertility_rate=120), dur=8, rand_seed=1)
    sim.run()
    int_keys = ('pregnancies', 'births', 'n_preterm', 'n_very_preterm',
                'miscarriages', 'stillbirths', 'neonatal_deaths', 'maternal_deaths')
    for key in int_keys:
        arr = np.asarray(sim.results.pregnancy[key])
        assert np.issubdtype(arr.dtype, np.integer), f'pregnancy.{key} should be int, got {arr.dtype}'


# ---------------------------------------------------------------------------
# Scale-weighted (fractional under split) results must stay float
# ---------------------------------------------------------------------------

def test_deaths_result_stays_float():
    # Deaths are scale-weighted (a death removes the agent's fractional `scale`).
    sim = ss.Sim(n_agents=500, demographics=ss.Deaths(death_rate=20), dur=10, rand_seed=1)
    sim.run()
    assert np.issubdtype(np.asarray(sim.results.deaths.new).dtype, np.floating)


def test_pregnancy_scale_weighted_derived_results_stay_float():
    sim = ss.Sim(n_agents=500, demographics=ss.Pregnancy(fertility_rate=80), dur=5, rand_seed=1)
    sim.run()
    for key in ('n_fecund', 'n_fertile', 'n_susceptible'):
        arr = np.asarray(sim.results.pregnancy[key])
        assert np.issubdtype(arr.dtype, np.floating), f'pregnancy.{key} should be float, got {arr.dtype}'


# ---------------------------------------------------------------------------
# pop_scale: integer-preserving when integral, float when fractional, exact either way
# ---------------------------------------------------------------------------

def test_integral_pop_scale_preserves_integer_results():
    base = ss.Sim(n_agents=500, demographics=ss.Births(birth_rate=30), dur=10, rand_seed=1)
    base.run()
    scaled = ss.Sim(n_agents=500, demographics=ss.Births(birth_rate=30), dur=10, rand_seed=1, pop_scale=1000)
    scaled.run()
    new = np.asarray(scaled.results.births.new)
    assert np.issubdtype(new.dtype, np.integer)                       # integral scale keeps int
    assert np.array_equal(new, np.asarray(base.results.births.new) * 1000)  # and is exactly scaled


def test_fractional_pop_scale_yields_float_results():
    sim = ss.Sim(n_agents=500, demographics=ss.Births(birth_rate=30), dur=10, rand_seed=1, pop_scale=1.5)
    sim.run()
    assert np.issubdtype(np.asarray(sim.results.births.new).dtype, np.floating)


# ---------------------------------------------------------------------------
# Accuracy: declaring body-weighted flows int must not change the reported counts,
# even under multiscale splitting (newborns inherit whole-body epi_weight == 1).
# ---------------------------------------------------------------------------

class _SplitEveryone(ss.Intervention):
    def __init__(self, ratio=4, name=None):
        super().__init__(name=name)
        self.ratio = ratio
    def step(self):
        if self.sim.ti == 1:
            self.sim.people.split(self.sim.people.auids.copy(), self.ratio)


def test_births_integer_and_conserved_under_split():
    def total_births(split):
        ivs = [_SplitEveryone(ratio=4)] if split else []
        s = ss.Sim(n_agents=2000, demographics=ss.Births(birth_rate=30), dur=10, rand_seed=1, interventions=ivs)
        s.run()
        arr = np.asarray(s.results.births.new)
        assert np.issubdtype(arr.dtype, np.integer)   # still integer even with splitting active
        return float(np.nansum(arr))
    base = total_births(False)
    sp = total_births(True)
    assert base > 0
    assert np.isclose(base, sp, rtol=0.15)
