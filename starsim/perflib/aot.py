"""Optional ahead-of-time compilation of the existing Numba kernels.

Build once with ``ss.aot.build(folder)`` or ``starsim compile --aot FOLDER``;
call ``ss.aot.load(folder)`` in each process that will use the artifact.
Whole compartmental runs and expression evaluation use the native extension.
Other entry points keep their normal JIT/plain-Python behavior, and
``ss.options.compile = 'never'`` still selects plain Python.

The default CPU target is generic within the build's architecture. Artifacts
require matching Python, NumPy, Numba, llvmlite, platform, and kernel source.
Building requires a C compiler and setuptools; loading does not. Numba's pycc
is pending deprecation, so AOT remains optional.
"""
import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import subprocess
import tempfile
import time
from types import SimpleNamespace

__all__ = ['build', 'load', 'unload']


MODULE = 'starsim_kernels_aot'
FORMAT = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import llvmlite
    import numba
    import numpy
    import starsim as ss
    return dict(
        python=sys.version, cache_tag=sys.implementation.cache_tag,
        platform=sys.platform, machine=platform.machine(),
        numpy=numpy.__version__, numba=numba.__version__, llvmlite=llvmlite.__version__,
        # Include helper modules (including a future program representation)
        # rather than assuming that only draws.py can be compiled into run().
        sources={p.name:digest(p) for p in sorted(Path(ss.kernels.__file__).parent.glob('*.py'))},
    )


def sample_arguments():
    """Capture a normal run's arguments before its kernel is compiled or called."""
    import numpy as np
    import starsim as ss
    saved = {name:getattr(ss.kernels, name) for name in ['run', 'run_ode']
             if hasattr(ss.kernels, name)}
    captured = {}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        bound = inspect.signature(saved['run'].py_func).bind(*args, **kwargs)
        bound.apply_defaults()
        captured['run'] = tuple(bound.arguments.values())
        raise Captured()

    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.1), name='sir')
    sim = ss.Sim(sir, total_pop=100, unit='day', stop=30, solver='ode', verbose=0)
    sim.jax = False
    try:
        for name in saved:
            setattr(ss.kernels, name, capture)
        with ss.options.context(compile='always'):
            sim.run()
    except Captured:
        pass
    finally:
        for name, original in saved.items():
            setattr(ss.kernels, name, original)
    if 'run' not in captured:
        raise RuntimeError('The example did not reach the whole-run kernel')

    # The tape's types are independent of its length and number of groups.
    k = ss.kernels
    ints = lambda x: np.array(x, dtype=np.int64)
    tape = k.Tape(ints([k.CONST, k.COUNT, k.MUL]), ints([0, 0, 0]), ints([-1, -1, 1]),
                  np.array([True]), ints([0, 1]), ints([0, 1]), ints([1]), ints([0, 1]),
                  2, True, ints([0, 1]), False)
    captured['evaluate'] = (np.array([[9., 18.], [1., 2.]]), tape,
                            np.array([[0.1, 0.2]]), np.zeros((3, 2)), np.zeros(3, dtype=bool))
    return captured


def build(folder, cpu='generic'):
    """Build an AOT artifact in ``folder``, without activating it in this process.

    A fresh Python process sets Numba's CPU target before importing Starsim.
    ``cpu='generic'`` is portable within the same architecture and OS;
    ``cpu='native'`` targets this CPU and checks its features when loaded.
    Returns the build manifest. Rebuilding leaves any loaded binary intact.
    """
    if cpu not in ['generic', 'native']:
        raise ValueError('cpu must be "generic" or "native"')
    folder = Path(folder).expanduser().resolve()
    env = os.environ.copy()
    if cpu == 'generic':
        env['NUMBA_CPU_NAME'] = 'generic'
    else:
        env.pop('NUMBA_CPU_NAME', None)
    env.pop('NUMBA_CPU_FEATURES', None)
    env['NUMBA_DISABLE_JIT'] = '0'
    env['STARSIM_UNCOMPILED_WARNING'] = '0'
    # Build the same checkout/install as this process, including in an editable install.
    root = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [root, env.get('PYTHONPATH')]))
    subprocess.run([sys.executable, '-m', 'starsim.perflib.aot', str(folder), '--cpu', cpu], env=env, check=True)
    return json.loads((folder/'manifest.json').read_text())


def _build(folder, cpu='generic'):
    """Export fixed signatures from the same functions used by njit."""
    import numba as nb
    from numba.pycc import CC
    import llvmlite.binding as llvm
    import starsim as ss

    target = llvm.get_host_cpu_name() if cpu == 'native' else cpu
    configured = nb.config.CPU_NAME or llvm.get_host_cpu_name()
    if configured != target:
        raise ValueError(f'NUMBA_CPU_NAME must be {target!r} before importing Numba; got {configured!r}')
    if nb.config.CPU_FEATURES:
        raise ValueError('Unset NUMBA_CPU_FEATURES so dependencies use the declared CPU target')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    samples = sample_arguments()
    signatures = {
        'run': nb.types.Tuple((nb.float64[:, ::1], nb.float64[:, :, ::1],
                              nb.float64[:, :, ::1], nb.float64)),
        'evaluate': nb.void,
    }
    cc = CC(MODULE)
    cc.output_dir = str(folder)
    cc.target_cpu = target
    metadata = dict(format=FORMAT, environment=environment(), cpu=target, signatures={},
                    cpu_features={} if target == 'generic' else dict(llvm.get_host_cpu_features()))
    for name, args in samples.items():
        argtypes = tuple(nb.typeof(arg) for arg in args)
        metadata['signatures'][name] = [str(t) for t in argtypes]
        # pycc's own default error model is Python, whereas these dispatchers
        # use NumPy's division semantics. Exporting .py_func directly changes
        # 1/0 from inf to ZeroDivisionError when helpers are inlined. Calling
        # the dispatcher preserves its flags and links the same native code.
        namespace = dict(callee=getattr(ss.kernels, name))
        parameters = ', '.join(f'a{i}' for i in range(len(args)))
        exec(f'def exported({parameters}):\n    return callee({parameters})\n', namespace)
        cc.export(name, signatures[name](*argtypes))(namespace['exported'])
    start = time.perf_counter()
    # Never overwrite a shared library that another process may have mapped.
    # The manifest switches to the new binary only after a successful build.
    with tempfile.TemporaryDirectory(prefix='build-', dir=folder) as temporary:
        cc.output_dir = temporary
        cc.compile()
        built = Path(temporary)/(MODULE + sysconfig.get_config_var('EXT_SUFFIX'))
        checksum = digest(built)
        binary = folder/(MODULE + '_' + checksum + sysconfig.get_config_var('EXT_SUFFIX'))
        if not binary.exists():
            built.replace(binary)
        metadata.update(binary=binary.name, sha256=checksum, build_seconds=time.perf_counter() - start)
        manifest = Path(temporary)/'manifest.json'
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        manifest.replace(folder/'manifest.json')
    print(json.dumps(dict(binary=str(binary), build_seconds=metadata['build_seconds'])), flush=True)
    return metadata


class Entry:
    """Keep JIT as a safe fallback for any dtype, layout, or signature mismatch."""
    def __init__(self, original, native, types):
        self.original, self.native, self.types = original, native, tuple(types)
        self.signature = inspect.signature(original.py_func)
        self.hits = self.fallbacks = 0
        functools.update_wrapper(self, original, updated=())

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        import numba as nb
        if kwargs or len(args) != len(self.types):
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            args = tuple(bound.arguments.values())
        try:
            matches = tuple(str(nb.typeof(arg)) for arg in args) == self.types
        except ValueError: # Let the original dispatcher report arguments Numba cannot type
            matches = False
        if not matches:
            self.fallbacks += 1
            return self.original(*args)
        self.hits += 1
        return self.native(*args)


def load(folder):
    """Load a compatible artifact and return its entry points (with hit/fallback counts).

    Call before running simulations, in each worker process that needs AOT.
    Different shapes share the binary; unsupported types/layouts fall back to
    the original JIT dispatcher. Incompatible artifacts raise without replacing
    an already loaded artifact. This does not change ``ss.options.compile``.
    """
    import llvmlite.binding as llvm
    import starsim as ss
    folder = Path(folder).expanduser().resolve()
    metadata = json.loads((folder/'manifest.json').read_text())
    if metadata.get('format') != FORMAT or set(metadata.get('signatures', {})) != {'run', 'evaluate'}:
        raise RuntimeError('Unsupported AOT manifest; rebuild with ss.aot.build()')
    if metadata['environment'] != environment():
        raise RuntimeError('AOT artifact does not match this source and Python/NumPy/Numba environment; rebuild it')
    if metadata['cpu'] not in ['generic', llvm.get_host_cpu_name()]:
        raise RuntimeError('AOT artifact targets another CPU; rebuild with NUMBA_CPU_NAME=generic')
    available = llvm.get_host_cpu_features()
    if any(required and not available.get(feature, False)
           for feature, required in metadata.get('cpu_features', {}).items()):
        raise RuntimeError('AOT artifact requires unavailable CPU features; rebuild for generic CPU')
    binary = folder/metadata['binary']
    if digest(binary) != metadata['sha256']:
        raise RuntimeError('AOT binary differs from its build manifest')
    spec = importlib.util.spec_from_file_location(MODULE, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = {}
    for name, types in metadata['signatures'].items():
        entries[name] = Entry(getattr(ss.kernels, name), getattr(module, name), types)
    ss.kernels.native = SimpleNamespace(**entries)
    return entries


def unload():
    """Stop using AOT entry points for subsequent calls, without changing compile options."""
    import starsim as ss
    ss.kernels.native = None
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--cpu', choices=['generic', 'native'], default='generic')
    args = parser.parse_args()
    _build(args.folder, args.cpu)
