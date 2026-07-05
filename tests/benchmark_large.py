#!/usr/bin/env python3
"""
Large benchmark of Starsim performance, saved to YAML with one entry per version.

Each run times a large default sim `repeats` times and keeps the best (fastest)
run to reduce noise. Times are normalized by CPU performance in the same way as
benchmark_tests.py, so results are comparable across machines.

For best results (avoiding thread locking), run from a terminal:
    ./benchmark_large.py            # run and compare against saved versions
    ./benchmark_large.py save       # run and save the entry for the current version
    ./benchmark_large.py repeats=10 # use 10 repeats instead of the default 5
"""
import sciris as sc
import starsim as ss

filename = sc.thisdir(__file__, 'benchmark_large.yaml')

# Define the parameters
default_repeats = 5
pars = sc.objdict(
    n_agents = 100_000,   # Number of agents
    dur      = 100,       # Number of years to simulate
    dt       = 1.0,       # Timestep
    verbose  = 0,         # Don't print details of the run
    networks = 'random',  # Default network
    diseases = 'sis',     # Default disease
)


def run_benchmark(repeats=default_repeats):
    """ Run the large sim `repeats` times and return the best (normalized) result. """
    ref = 270 # Reference benchmark for sc.benchmark(which='numpy') on an Intel i7-12700H (for scaling performance)

    # Test CPU performance before the run
    r1 = sc.benchmark(which='numpy')

    # Do the actual benchmarking, keeping every trial's time
    times = []
    print(f'Benchmarking {repeats} runs of a {pars.n_agents:,}-agent, {pars.dur}-year sim ...')
    for r in range(repeats):
        sim = ss.Sim(pars=pars)
        t0 = sc.tic()
        sim.run()
        t = sc.toc(t0, output=True)
        times.append(t)
        print(f'  Trial {r+1}/{repeats}: {t:0.3f} s')

    # Test CPU performance after the run and compute the performance ratio
    r2 = sc.benchmark(which='numpy')
    perf_ratio = (r1+r2)/2/ref

    # Keep the best (fastest) run and normalize by the performance ratio
    best = min(times)
    out = sc.objdict()
    out.version = ss.__version__
    out.date = sc.getdate(dateformat='%Y-%m-%d')
    out.cpu_performance = round(perf_ratio, 3)
    out.scaled_time = round(best*perf_ratio, 3) # Normalized time, comparable across machines
    out.actual_time = round(best, 3)            # Raw best time on this machine
    return out


def load_benchmarks():
    """ Load the saved per-version benchmarks, or an empty dict if none exist yet. """
    try:
        return sc.objdict(sc.loadyaml(filename))
    except FileNotFoundError:
        return sc.objdict()


def compare_benchmark(out):
    """ Compare the current run against the saved per-version entries. """
    data = load_benchmarks()
    rows = []
    for version, entry in data.items():
        rows.append(dict(version=version, date=entry.get('date'), scaled_time=entry['scaled_time'], actual_time=entry['actual_time'], cpu_performance=entry['cpu_performance']))
    rows.append(dict(version=f'{out.version} (current)', date=out.date, scaled_time=out.scaled_time, actual_time=out.actual_time, cpu_performance=out.cpu_performance))

    df = sc.dataframe(rows)
    df.disp()

    # Report the ratio against the most recent saved entry, if any
    if len(data):
        prev_version = list(data.keys())[-1]
        prev = data[prev_version]['scaled_time']
        ratio = out.scaled_time/prev
        print(f'\nCurrent scaled time: {out.scaled_time:0.3f} s')
        print(f'Previous ({prev_version}): {prev:0.3f} s')
        print(f'Ratio (current/previous): {ratio:0.3f}')
    return df


def save_benchmark(out):
    """ Append (or overwrite) the entry for the current version and save to YAML. """
    data = load_benchmarks()
    entry = dict(out)
    entry.pop('version') # The version is the key, so no need to store it in the body too
    data[out.version] = entry
    print(f'Saving entry for version {out.version} ...')
    sc.saveyaml(filename, dict(data), sort_keys=False)
    return


def benchmark_large(save=False, compare=True, repeats=default_repeats):
    """ Run the large benchmark, then optionally compare and/or save the result. """
    out = run_benchmark(repeats=repeats)

    if compare:
        compare_benchmark(out)

    if save:
        save_benchmark(out)

    sc.heading(f'Best of {repeats}: {out.actual_time*1000:0.0f} ms (scaled: {out.scaled_time*1000:0.0f} ms)')
    return out


if __name__ == '__main__':
    args = sc.argparse(save=None, compare=True, repeats=default_repeats)
    if args.save not in ['save', 'True', None]:
        errormsg = f'Invalid value for save: {args.save}, should just be "save" or "True"; False by default'
        raise ValueError(errormsg)
    out = benchmark_large(save=args.save, compare=args.compare, repeats=args.repeats)
