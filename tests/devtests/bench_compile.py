"""Measure cold Numba compilation and warm runtime in a fresh process.

Examples (each invocation uses a new cache unless --cache is given)::

    python tests/devtests/bench_compile.py --output /tmp/default.json
    NUMBA_LLVM_REFPRUNE_FLAGS=per_bb python tests/devtests/bench_compile.py
    NUMBA_OPT=0 python tests/devtests/bench_compile.py
    python tests/devtests/bench_compile.py --full

--model-code can also time an exported model file defining ``sim``; --arrays
saves its results for numerical comparison. Keep private model results outside
the repository. Compile and warm-run times are reported separately; compare
variants on an otherwise quiet machine.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--full', action='store_true', help='Time ss.compile() instead of the first ODE run')
    parser.add_argument('--cache', help='Use this cache directory (default: a fresh temporary directory)')
    parser.add_argument('--output', type=Path, help='Write the measurements as JSON')
    parser.add_argument('--model-code', type=Path, help='Also run an exported Python model defining sim')
    parser.add_argument('--arrays', type=Path, help='Save that model\'s result arrays as an NPZ for comparison')
    args = parser.parse_args()
    if args.arrays and not args.model_code:
        parser.error('--arrays requires --model-code')

    # Set these before importing Numba or Starsim. A new directory prevents a
    # previous compiler setting's cached machine code contaminating a trial.
    os.environ['NUMBA_CACHE_DIR'] = args.cache or tempfile.mkdtemp(prefix='starsim-compile-')
    os.environ.setdefault('STARSIM_INSTALL_FONTS', '0')
    os.environ.setdefault('STARSIM_UNCOMPILED_WARNING', '0')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import numpy as np
    import numba as nb
    from numba.core import event

    class Timings(event.Listener):
        def __init__(self):
            self.stack = []
            self.rows = []

        def on_start(self, ev):
            self.stack.append([time.perf_counter(), time.process_time(), 0., 0.])

        def on_end(self, ev):
            start, cpu_start, child_wall, child_cpu = self.stack.pop()
            wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
            if self.stack:
                self.stack[-1][2] += wall
                self.stack[-1][3] += cpu
            self.rows.append(dict(function=ev.data['dispatcher'].py_func.__qualname__,
                                  wall=wall, cpu=cpu, self_wall=wall-child_wall, self_cpu=cpu-child_cpu))

    def timed(func):
        start, cpu_start = time.perf_counter(), time.process_time()
        result = func()
        return result, dict(wall=time.perf_counter()-start, cpu=time.process_time()-cpu_start)

    listener = Timings()
    with event.install_listener('numba:compile', listener):
        import importlib
        ss, import_time = timed(lambda: importlib.import_module('starsim'))
        import_compiles = len(listener.rows)
        ss.options(compile='always', verbose=0)

        def make(stop=3):
            S, I, R = ss.states(3)
            disease = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.1), name='sir')
            sim = ss.Sim(disease, total_pop=100, unit='day', stop=stop, solver='ode', verbose=0)
            sim.jax = False
            return sim

        _, cold_time = timed(lambda: ss.compile(verbose=False) if args.full else make().run())
        cold_compiles = len(listener.rows) - import_compiles
        out = dict(numba=nb.__version__, starsim=ss.__version__, python=sys.version,
                   mode='full' if args.full else 'ode', import_time=import_time, cold=cold_time,
                   import_compiles=import_compiles, cold_compiles=cold_compiles,
                   env={k:v for k,v in os.environ.items() if k.startswith('NUMBA_')})
        print('Cold:', json.dumps(out), flush=True)
        out['warm_sir'] = []
        for _ in range(3):
            sim = make(1000).init()
            _, elapsed = timed(sim.run)
            out['warm_sir'].append(elapsed)

        if args.model_code:
            ns = dict(__name__='ported')
            exec(compile(args.model_code.read_text(), str(args.model_code), 'exec'), ns)
            sim = ns['sim']
            sim.jax = False
            _, out['model_init'] = timed(sim.init)
            _, out['model_run'] = timed(sim.run)
            if args.arrays:
                arrays = {f'{mod.name}.y':np.asarray(mod.engine.y) for mod in sim.diseases.values()}
                arrays.update({k:np.asarray(v) for k,v in sim.results.flatten().items()})
                np.savez(args.arrays, **arrays)
        out['compiler_events'] = sorted(listener.rows, key=lambda r:-r['self_cpu'])

    print('Warm:', json.dumps({k:v for k,v in out.items() if k.startswith(('warm_', 'model_'))}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + '\n')
    return out


if __name__ == '__main__':
    main()
