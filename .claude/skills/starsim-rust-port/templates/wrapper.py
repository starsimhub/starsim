"""
TEMPLATE: a Phase-1 ssr wrapper module.

Add classes like this to `starsim/rust/modules.py`. The wrapper subclasses the
`ss` module and overrides ONLY the hot method(s) you ported, calling the kernel
on zero-copy views of the state arrays. Everything else inherits the Python
implementation, which (a) keeps unported methods working and (b) gives
`starsim.rust.compare` a free Python reference to validate against.

See `starsim/rust/modules.py::SIS` for the worked SIS+RandomNet example.
"""
import numpy as np
import starsim as ss

try:
    import starsim_rust_kernels as _kernels
    available = True
except ImportError:
    _kernels = None
    available = False


class MyModule(ss.MyModule):  # subclass the ss counterpart
    """ Rust-accelerated MyModule (see starsim.MyModule) """

    def my_method(self, *args, **kwargs):
        """ Rust kernel mirror of ss.MyModule.my_method """
        if _kernels is None:  # graceful fallback to inherited Python
            return ss.MyModule.my_method(self, *args, **kwargs)

        # 1. Draw any randoms in Python (Phase 1), exactly as the parent does.
        # randvals = self.some_dist.rvs(uids)

        # 2. Pass numpy buffers/views into the kernel. For UID-indexed state
        #    arrays, pass `.raw` (full-length) and the UID arrays separately.
        out = _kernels.my_kernel(
            np.asarray(uids, dtype=np.int64),
            np.asarray(self.some_state.raw, dtype=np.float64),
            np.asarray(randvals, dtype=np.float64),
        )

        # 3. Re-wrap outputs into Starsim types as the parent returns them.
        return out.view(ss.uids)
