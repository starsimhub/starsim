"""
Rust-backed Starsim modules (the ``ssr`` classes).

Each class subclasses its ``ss`` counterpart and overrides only its hot methods
with calls into the compiled kernel (``starsim_rust_kernels``), operating on
zero-copy views of the existing numpy state buffers. Everything else inherits
the Python implementation -- which means (a) any unported method still works,
and (b) ``starsim.rust.compare`` has a free Python reference to validate against.

Phase 1 keeps all RNG in Python: the kernels receive the already-drawn random
arrays, so results are byte-identical to the pure-Python path by construction.
"""
import numpy as np
import starsim as ss

try:
    import starsim_rust_kernels as _kernels
    available = True
except ImportError: # pragma: no cover - depends on the compiled wheel being installed
    _kernels = None
    available = False

__all__ = ['SIS', 'RandomNet', 'available']

_M64 = (1 << 64) - 1


def _initial_state(seed):
    """ The PCG64 state right after default_rng(seed) -- Starsim's reset() target """
    rng = np.random.default_rng(seed)
    st = rng.bit_generator.state['state']
    return st['state'], st['inc']


class _NativeTransRng:
    """
    Native-Rust replacement for an ``ss.multi_random`` transmission RNG.

    Built once from the two underlying ``ss.random`` dists' seeds; thereafter the
    RNG state lives entirely in Rust (see ``MultiRandomRng``), reproducing
    Starsim's per-timestep jumping without any numpy RNG calls during the run.
    """
    def __init__(self, multi):
        self.sd, self.td = multi.dists
        s0, sinc = _initial_state(self.sd.seed)
        t0, tinc = _initial_state(self.td.seed)
        self.rng = _kernels.MultiRandomRng(
            (s0 >> 64) & _M64, s0 & _M64, (sinc >> 64) & _M64, sinc & _M64,
            (t0 >> 64) & _M64, t0 & _M64, (tinc >> 64) & _M64, tinc & _M64,
            int(self.sd.dt_jump_size),
        )

    def set_timestep(self, ti):
        self.rng.set_timestep(int(ti))

    def rvs(self, *args):
        src, trg = args
        slots_raw = np.asarray(self.sd.slots.raw, dtype=np.int64) # indexed by UID in Rust
        return self.rng.rvs(np.asarray(src, dtype=np.int64), np.asarray(trg, dtype=np.int64), slots_raw)


class SIS(ss.SIS):
    """
    Rust-accelerated SIS (see :class:`starsim.SIS`).

    Overrides ``compute_transmission`` (the RNG-free transmission kernel) and
    routes the transmission RNG (``trans_rng``) through a native Rust generator
    that owns its state. All other behaviour is inherited from ``ss.SIS``. Falls
    back to the Python implementation if the compiled kernel is unavailable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_rng = None # lazily built on first step (needs initialized seeds)
        return

    def step(self):
        """ Reset the native RNG timestep, then run the inherited transmission step """
        if _kernels is not None:
            if self._native_rng is None:
                self._native_rng = _NativeTransRng(self.trans_rng)
                self.trans_rng = self._native_rng # route rvs() through Rust
            self._native_rng.set_timestep(self.ti)
        return super().step()

    @staticmethod
    def compute_transmission(src, trg, rel_trans, rel_sus, beta_per_dt, randvals):
        """ Rust kernel mirror of ``Infection.compute_transmission`` """
        if _kernels is None: # Graceful fallback to the inherited Python implementation
            return ss.SIS.compute_transmission(src, trg, rel_trans, rel_sus, beta_per_dt, randvals)

        target_uids, source_uids = _kernels.compute_transmission(
            np.asarray(src, dtype=np.int64),
            np.asarray(trg, dtype=np.int64),
            np.asarray(rel_trans.raw, dtype=np.float64), # full-length raw buffer, indexed by UID
            np.asarray(rel_sus.raw, dtype=np.float64),
            np.asarray(beta_per_dt, dtype=np.float64),
            np.asarray(randvals, dtype=np.float64),
        )
        return target_uids.view(ss.uids), source_uids.view(ss.uids)


# Not yet accelerated: a pass-through to the Python network so the documented
# `ss.Sim(networks=ssr.RandomNet())` API works today. Port target for Phase 1+.
class RandomNet(ss.RandomNet):
    """ Pass-through to :class:`starsim.RandomNet` (not yet Rust-accelerated) """
    pass
