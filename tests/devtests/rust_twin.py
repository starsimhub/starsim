"""
Run every call to the compartmental kernels through both Numba's and Rust's, and fail on any difference

A pytest plugin: Numba's kernels take the real arguments (so each sim goes on as with Numba), and Rust's take deep copies,
then the results, every array the call changed, and the random generators' states are compared. Use it after changing
ss.kernels (and the Rust copy to match: see starsim/solvers/rust/README.md), over any tests, e.g.:

    cd tests && python -m pytest -p devtests.rust_twin test_kernels.py test_dimensions.py devtests/test_gradients.py -n 2

A difference fails the test that made the call, saying which kernel and which of its arguments or results differ.
"""
import copy
import pytest
import numpy as np
import numba as nb
import starsim as ss

K = ss.kernels
names = K.entries + ['evaluate']
calls = {} # The calls compared, by kernel


def clone(x):
    """ A deep copy of a kernel's arguments: arrays, Generators, typed lists, and (named) tuples of them """
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, np.random.Generator):
        return copy.deepcopy(x)
    if isinstance(x, nb.typed.List):
        return nb.typed.List([clone(v) for v in x])
    if isinstance(x, tuple):
        items = [clone(v) for v in x]
        return type(x)(*items) if hasattr(x, '_fields') else tuple(items)
    return x


def differ(a, b, path):
    """ Where two arguments or results differ (bit for bit, with NaN equal to NaN) """
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape:
            return [f'{path} (shape {a.shape} vs {getattr(b, "shape", None)})']
        bad = (a != b) & ~(np.isnan(a) & np.isnan(b)) if a.dtype.kind == 'f' else a != b
        return [f'{path} ({bad.sum()} of {a.size} differ, e.g. {a[bad][:3]} vs {b[bad][:3]})'] if bad.any() else []
    if isinstance(a, np.random.Generator):
        return [] if a.bit_generator.state == b.bit_generator.state else [f'{path} (generator state)']
    if isinstance(a, (tuple, list, nb.typed.List)):
        fields = getattr(a, '_fields', range(len(a)))
        return [d for f, u, v in zip(fields, a, b) for d in differ(u, v, f'{path}.{f}')]
    if isinstance(a, float):
        return [] if a == b or (a != a and b != b) else [f'{path}: {a} vs {b}']
    return []


def twin(name, rust):
    numba = getattr(K, name)
    def call(*args):
        copies = clone(args)
        out = numba(*args)
        rout = getattr(rust, name)(*copies)
        found = differ(args, copies, 'args') + differ(out, rout, 'out')
        calls[name] = calls.get(name, 0) + 1
        assert not found, f'The Rust kernel {name}() differs from Numba\'s: {found[:6]}'
        return out
    return call


class Twin:
    """ The compartmental kernels, each call taken by both Numba's and Rust's """
    def __init__(self):
        rust = ss.solvers.backends.rust.compartmental()
        for name in names:
            setattr(self, name, twin(name, rust))

    def __getattr__(self, name):
        return getattr(K, name)


def pytest_configure(config):
    ss.options.kernels = 'numba' # Numba's, compiled, take the run; Rust's are called beside them
    pair = Twin()
    K.choose = lambda seconds=0.0: pair
    for mod in [ss.solvers.compartmental, ss.solvers.crossing]:
        mod.choose_kernels = lambda owner, runs, seconds: pair
    return


def pytest_sessionfinish(session):
    """ In a worker of pytest-xdist, send its counts to the main process """
    if hasattr(session.config, 'workeroutput'):
        session.config.workeroutput['rust_twin'] = calls
    return


@pytest.hookimpl(optionalhook=True) # A hook of pytest-xdist
def pytest_testnodedown(node, error):
    """ In the main process, add up the workers' counts """
    for k, n in getattr(node, 'workeroutput', {}).get('rust_twin', {}).items():
        calls[k] = calls.get(k, 0) + n
    return


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f'rust_twin: calls compared, by kernel: {calls}')
    return
