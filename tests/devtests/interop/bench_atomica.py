"""
Benchmark an imported Atomica model: cat-tb's India model (its framework, databook, and calibration), which is private, so is found through CATTB

Times reading the workbooks, running the code written for them, initializing, and the run, each on its own,
and saves the final compartments and every result to a baseline (kept outside the repo, since cat-tb's data
is private), or checks them against it.

    python bench_atomica.py                  # Time, and check against the baseline
    python bench_atomica.py --save           # Time, and save the baseline
    python bench_atomica.py --stop 2002      # A shorter run (with its own baseline)
    python bench_atomica.py --profile        # And profile the run
"""
import os
import sys
import time
import argparse
import numpy as np
import sciris as sc
import starsim as ss
import starsim.interop
sys.path.insert(0, str(sc.thispath().parent.parent))
import paths

ss.options.compile = 'always' # Timed compiled, as a long run would be
framework = paths.cattb/'framework'/'tb_framework_2025.xlsx'
databook = paths.cattb/'databooks'/'IND.xlsx'
calibration = paths.cattb/'databooks'/'IND_calibration.xlsx'
folder = sc.path(os.getenv('STARSIM_BENCH', paths.root/'tb_ensemble'/'atomica_port'/'results')) # Where the code and baselines are kept
start, stop, dt = 2000, 2051, 1/12 # As cat-tb runs it (tb/constants.py)


def port(): # Read the workbooks, and write the code
    sim = ss.interop.load(framework, format='atomica', databook=databook, calibration=calibration, start=start, stop=stop, dt=dt, verbose=False)
    return sim.imported.code


def final(sim):
    """ The final compartments and every result, as arrays """
    out = sc.objdict()
    for mod in sim.diseases.values():
        out[f'{mod.name}.y'] = np.array(mod.engine.y)
    for key,res in sim.results.flatten().items():
        out[key] = np.array(res)
    return out


def check(new, old):
    """ The largest difference between the final state of two runs, relative to the largest value of each array """
    apart = set(old) ^ set(new)
    if apart:
        print(f'{len(apart)} results are in one run only (e.g. {sc.strjoin(sorted(apart)[:3])}), and are not compared')
    worst = max(((np.max(np.abs(new[k] - old[k]))/max(np.max(np.abs(old[k])), 1e-300), k) for k in old if k in new and old[k].size), default=(0.0, None))
    return worst


def main(save=False, stop=stop, profile=False, reread=False):
    if not framework.exists():
        print(f'Skipped: cat-tb not found at {paths.cattb} (set CATTB to where it is)')
        return
    folder.mkdir(parents=True, exist_ok=True)
    codefile = folder/'bench_IND.py'
    times = sc.objdict()
    if reread or not codefile.exists():
        t = time.perf_counter()
        sc.savetext(codefile, port())
        times.port = time.perf_counter() - t
    t = time.perf_counter()
    code = compile(codefile.read_text(), str(codefile), 'exec')
    ns = dict(__name__='ported')
    exec(code, ns)
    sim = ns['sim']
    sim.pars.stop = float(stop)
    times.code = time.perf_counter() - t
    t = time.perf_counter()
    sim.init()
    times.init = time.perf_counter() - t
    t = time.perf_counter()
    if profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
    sim.run()
    if profile:
        pr.disable()
        pstats.Stats(pr).sort_stats('cumulative').print_stats(40)
        pstats.Stats(pr).sort_stats('tottime').print_stats(25)
    times.run = time.perf_counter() - t
    print(f'Code: {codefile.stat().st_size/1e6:.1f} MB, longest line {max(len(line) for line in codefile.read_text().splitlines()):,} characters')
    print(f'{len(sim.t)} steps to {stop}: ' + ', '.join(f'{k} {v:.2f} s' for k,v in times.items()))
    base = folder/f'bench_baseline_{int(stop)}.obj'
    out = final(sim)
    if save:
        sc.save(base, out)
        print(f'Saved the baseline to {base}')
    elif base.exists():
        diff, key = check(out, sc.load(base))
        print(f'Largest difference from the baseline: {diff:.3g} (relative, in {key})' if diff else 'Identical to the baseline')
    return times


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--reread', action='store_true', help='read the workbooks again (they are read once, and the code kept)')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--stop', type=float, default=stop)
    args = ap.parse_args()
    main(save=args.save, stop=args.stop, profile=args.profile, reread=args.reread)
