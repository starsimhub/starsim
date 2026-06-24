"""
Rust-backed Starsim modules (the ``ssr`` classes) and the bridge that runs a
normal ``ss.Sim`` on the fast native engine (``ssr_engine``).

Each ``ssr`` module subclasses its ``ss`` counterpart, so a sim built from them
behaves like a normal Starsim sim for prototyping. They additionally carry an
``_engine_spec()`` describing themselves to the Rust engine. When the sim is run
with ``engine='rust'`` (or auto-detected because every module is ssr-native),
:func:`run_engine` extracts effective parameters, runs the whole loop natively
in Rust, and writes the results back into ``sim.results`` so ``sim.plot()`` works.

Note: the engine is FAST but not byte-identical -- it uses a quick RNG and lazy
per-edge transmission draws (no CRN), matching Starsim statistically. For exact
reproducibility use the pure-Python path (the default) or the byte-identical
kernels in ``starsim/rust/_crate``.
"""
import numpy as np
import starsim as ss

from .validate import compare, ValidationReport, TIERS  # noqa: F401 (re-exported)

try:
    import ssr_engine
    available = True
except ImportError: # pragma: no cover
    ssr_engine = None
    available = False

__all__ = ['SIS', 'SIR', 'RandomNet', 'Births', 'Deaths', 'run_engine', 'all_native',
           'available', 'compare', 'ValidationReport', 'TIERS']


# ---------------------------------------------------------------------------
# Parameter extraction helpers (Python params -> effective engine params)
# ---------------------------------------------------------------------------
def _prob(rate_or_val, dt=None):
    """ Convert an ss.Rate to its per-step probability, else pass through as float """
    if isinstance(rate_or_val, ss.Rate):
        return float(rate_or_val.to_prob(dt) if dt is not None else rate_or_val.to_prob())
    return float(rate_or_val)


def _scalar(dist_or_val):
    """ Pull a representative scalar from a Dist (mean of a sample) or a bare value """
    if isinstance(dist_or_val, ss.Dist):
        return float(np.mean(dist_or_val.rvs(np.arange(1000))))
    return float(dist_or_val)


# ---------------------------------------------------------------------------
# ssr modules: normal ss modules + an engine spec
# ---------------------------------------------------------------------------
class SSRNative:
    """ Marker base for modules the Rust engine knows how to run """
    pass


class SIS(ss.SIS, SSRNative):
    """ SIS that can run on the fast Rust engine (see module docstring) """

    def _engine_spec(self):
        dt = self.t.dt
        d = self.pars.dur_inf.rvs(np.arange(min(2000, len(self.sim.people))))
        return ('sis', dict(
            beta=_prob(self.pars.beta, dt),
            init_prev=float(self.pars.init_prev.pars.get('p', 0.01)),
            dur_inf=float(d.mean()),
            dur_inf_std=float(d.std()),
            waning=_prob(self.pars.waning),
            imm_boost=float(self.pars.imm_boost),
        ))


class SIR(ss.SIR, SSRNative):
    """ SIR that can run on the fast Rust engine """

    def _engine_spec(self):
        dt = self.t.dt
        d = self.pars.dur_inf.rvs(np.arange(min(2000, len(self.sim.people))))
        return ('sir', dict(
            beta=_prob(self.pars.beta, dt),
            init_prev=float(self.pars.init_prev.pars.get('p', 0.01)),
            dur_inf=float(d.mean()),
            dur_inf_std=float(d.std()),
        ))


class RandomNet(ss.RandomNet, SSRNative):
    """ RandomNet that can run on the fast Rust engine """

    def _engine_spec(self):
        return ('randomnet', dict(
            n_contacts=_scalar(self.pars.n_contacts),
            dur=0.0,
            beta=float(self.pars.beta),
        ))


def _rate_prob(rate, units):
    """ Per-step probability for a demographic crude rate (e.g. CBR per 1000).
    Assumes a 1-unit timestep (the common dt=1 year case). """
    return float(rate) * float(units)


class Births(ss.Births, SSRNative):
    """ Births that can run on the fast Rust engine """

    def _engine_spec(self):
        return ('births', dict(birth_prob=_rate_prob(self.pars.birth_rate, self.pars.rate_units)))


class Deaths(ss.Deaths, SSRNative):
    """ Deaths that can run on the fast Rust engine """

    def _engine_spec(self):
        return ('deaths', dict(death_prob=_rate_prob(self.pars.death_rate, self.pars.rate_units)))


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------
def all_native(sim):
    """ True if every disease, network, and demographics module is ssr-native """
    mods = list(sim.diseases()) + list(sim.networks()) + list(sim.demographics())
    return len(mods) > 0 and all(isinstance(m, SSRNative) for m in mods)


def run_engine(sim, verbose=None):
    """
    Run an initialized ss.Sim on the native Rust engine and map results back.

    The sim must be built from ssr-native modules (ssr.SIS, ssr.RandomNet, ...).
    Returns the sim, with ``sim.results`` populated, so ``sim.plot()`` works.
    """
    if ssr_engine is None:
        raise ImportError('ssr_engine is not installed; build starsim/rust/_engine and pip install the wheel.')
    if not sim.initialized:
        sim.init()

    diseases = [(m, *m._engine_spec()) for m in sim.diseases()]
    networks = [m._engine_spec() for m in sim.networks()]
    demographics = [m._engine_spec() for m in sim.demographics()]
    n_agents = len(sim.people)
    n_steps = sim.t.npts
    seed = int(sim.pars.rand_seed or 0)

    out = ssr_engine.run(
        n_agents=n_agents, n_steps=n_steps, seed=seed,
        networks=networks,
        diseases=[(name, pars) for (_m, name, pars) in diseases],
        demographics=demographics,
    )

    # Map engine results -> sim.results[module.name][key]
    for (mod, engine_name, _pars) in diseases:
        modres = sim.results[mod.name]
        for key in list(modres.keys()):
            ekey = f'{engine_name}_{key}'
            if ekey in out:
                arr = np.asarray(out[ekey])
                if len(arr) == len(modres[key]):
                    modres[key][:] = arr

    sim.complete = True
    return sim
