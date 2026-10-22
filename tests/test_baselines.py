"""
Test that the current version of Starsim exactly matches
the baseline results.
"""
import os
import pytest
import sciris as sc
import starsim as ss
import benchmark_models as bm

baseline_filename   = sc.thisdir(__file__, 'baseline.yaml')
benchmark_filename  = sc.thisdir(__file__, 'benchmark.yaml')
perf_guard_filename = sc.thisdir(__file__, 'perf_guard.yaml')
sc.options(interactive=False) # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

REGRESSION_FACTOR = 1.5 # Performance guard: fail if a normalized time exceeds its baseline by more than this
REF_MOPS = 270 # Reference sc.benchmark(which='numpy') MOPS (Intel i7-12700H), used for CPU normalization

# Define the parameters
pars = sc.objdict(
    n_agents  = 10e3, # Number of agents
    start     = 2000, # Starting year
    dur       = 20,   # Number of years to simulate
    dt        = 0.2,  # Timestep
    verbose   = 0,    # Don't print details of the run
    rand_seed = 2,    # Set a non-default seed
)


def make_sim(run=False, **kwargs):
    """
    Define a default simulation for testing the baseline. If run directly (not
    via pytest), also plot the sim by default.
    """
    diseases = [dict(type='sir', beta=dict(random=ss.peryear(0.1), sexual=ss.peract(0.02), prenatal=ss.peryear(0.1))), # A probability per act on ss.SexualNet, about the default rate's chance in a step of 0.2 years
                dict(type='sis', beta=dict(random=ss.peryear(0.05), sexual=ss.peract(0.01), prenatal=ss.peryear(0.05)))]
    networks = ['random', 'sexual', 'prenatal']
    sim = ss.Sim(pars=pars | kwargs, networks=networks, diseases=diseases, demographics=True)

    # Optionally run and plot
    if run:
        sim.run()
        sim.plot()

    return sim


def save_baseline():
    """
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    """
    sc.heading('Updating baseline values...')

    # Make and run sim
    sim = make_sim()
    sim.run()

    # Export results
    json = sim.to_json(keys='summary')['summary']
    sc.saveyaml(baseline_filename, json)

    print('Done.')
    return


@sc.timer()
def test_baseline():
    """ Compare the current default sim against the saved baseline """

    # Load existing baseline
    old = sc.loadyaml(baseline_filename)

    # Calculate new baseline
    new = make_sim()
    new.run()

    # Compute the comparison
    ss.debugtools.diff_sims(old, new, die=True)

    return new


@sc.timer()
def test_benchmark(do_save=False, repeats=1, verbose=True):
    """ Compare benchmark performance """

    if verbose: print('Running benchmark...')
    try:
        previous = sc.loadyaml(benchmark_filename)
    except FileNotFoundError:
        previous = None

    t_inits = []
    t_runs  = []

    # Time initializing and running, bracketed by CPU benchmarks for normalization
    r1 = sc.benchmark(which='numpy')
    for r in range(repeats):
        t0 = sc.tic()
        sim = make_sim()
        sim.init()
        t_inits.append(sc.toc(t0, output=True))
        t0 = sc.tic()
        sim.run()
        t_runs.append(sc.toc(t0, output=True))
    r2 = sc.benchmark(which='numpy')
    ratio = (r1+r2)/2/REF_MOPS
    t_init = ratio*min(t_inits)
    t_run  = ratio*min(t_runs)

    # Construct json
    n_decimals = 3
    json = {'time': {
                'initialize': round(t_init, n_decimals),
                'run':        round(t_run,  n_decimals),
                },
            'parameters': {
                'n_agents': sim.pars.n_agents,
                'dur':      sim.t.dur,
                'dt':       sim.t.dt,
                },
            'cpu_performance': ratio,
            }

    if verbose:
        if previous:
            print('Previous benchmark:')
            sc.pp(previous)

        print('\nNew benchmark:')
        sc.pp(json)
    else:
        brief = sc.dcp(json['time'])
        brief['cpu_performance'] = json['cpu_performance']
        sc.pp(brief)

    if do_save:
        sc.saveyaml(filename=benchmark_filename, obj=json)

    if verbose:
        print('Done.')

    return json


#%% Performance regression guard

def perf_cases():
    """
    Factories for the performance-guard sims; each guards a distinct cost term

    - bare_init / bare_run: canonical timeline construction, per-tick dispatch, result collection
    - bare_run_no_people: the People-results opt-out (largest removable per-tick cost)
    - het_init: the heterogeneous-dt numeric-sort plan fallback
    - births_run: demographics (birth draw + grow)
    - crn_false_run: the crn=False path (guards the distribution jumping gate; all other
      cases run crn=True and so would not catch a reinstated per-step/per-draw RNG jump)
    - fseis10, counts, coded, by_age (init and run): the models in benchmark_models.py, a model written as code
      beside a coded ODE (malariasim), counts stepped beside agents, a coded disease on a network, and ss.by age tables

    Each case is (factory, which, crn).
    """
    bare = lambda **kw: ss.Sim(start=0, stop=365*10, dt=ss.days(1), verbose=0, **kw)
    het  = lambda: ss.Sim(diseases=ss.SIS(dt=ss.days(1)), networks=ss.RandomNet(dt=ss.weeks(1)),
                          demographics=ss.Births(dt=ss.days(10)), dt=ss.days(2),
                          start='2000-01-01', stop='2002-01-01', n_agents=200, verbose=0)
    births = lambda: ss.Sim(n_agents=5000, dur=20, dt=0.25, demographics=ss.Births(birth_rate=ss.peryear(30)),
                            rand_seed=1, verbose=0)
    crnf = lambda: ss.Sim(diseases=ss.SIS(beta=0.1, init_prev=0.05), networks=ss.RandomNet(),
                          demographics=ss.Births(birth_rate=ss.peryear(20)),
                          n_agents=2000, dur=20, dt=0.2, rand_seed=1, verbose=0)
    return {
        'bare_init':          (bare, 'init', True),
        'bare_run':           (bare, 'run', True),
        'bare_run_no_people': (lambda: bare(people_results=False), 'run', True),
        'het_init':           (het, 'init', True),
        'births_run':         (births, 'run', True),
        'crn_false_run':      (crnf, 'run', False),
        'fseis10_init':       (bm.make_fseis, 'init', False),
        'fseis10_run':        (bm.make_fseis, 'run', False),
        'counts_init':        (bm.make_counts, 'init', True),
        'counts_run':         (bm.make_counts, 'run', True),
        'coded_init':         (bm.make_coded, 'init', True),
        'coded_run':          (bm.make_coded, 'run', True),
        'by_age_init':        (bm.make_by_age, 'init', True),
        'by_age_run':         (bm.make_by_age, 'run', True),
    }


def measure_case(fn, which, repeats, warmup=1, crn=True):
    """ Return the minimum wall-clock time (seconds) to init or run a fresh sim, after warmup """
    with ss.options.context(crn=crn):
        for _ in range(warmup): # Warm up caches/JIT so the first (cold) sample doesn't dominate
            s = fn(); s.init()
            if which == 'run': s.run()
        times = []
        for _ in range(repeats):
            if which == 'init':
                t0 = sc.tic(); s = fn(); s.init(); times.append(sc.toc(t0, output=True))
            else:
                s = fn(); s.init()
                t0 = sc.tic(); s.run(); times.append(sc.toc(t0, output=True))
    return min(times)


@sc.timer()
def test_performance_guard(do_save=False, verbose=True):
    """
    Guard against major performance regressions.

    Measures a small matrix of CPU-normalized wall-clock times (each guarding a distinct cost
    term) and fails if any exceeds its recorded baseline by more than REGRESSION_FACTOR (1.5x).
    Baselines track the *improved* state, so this catches only major regressions (e.g. reinstating
    the timeline deepcopy or the object-key plan sort), not normal drift. Times use min-of-repeats
    with CPU normalization (see test_benchmark) to cancel machine speed. Refresh with
    ./update_benchmarks.py after an intended speedup.

    Note: this test is timing-sensitive, so it is skipped when running under pytest-xdist
    (parallel contention makes per-test wall-clock unreliable). Run it serially to gate, e.g.
    `pytest test_baselines.py::test_performance_guard`, or via ./update_benchmarks.py.
    """
    # Skip under heavy parallel contention (xdist workers), where timing is unreliable
    if not do_save and os.environ.get('PYTEST_XDIST_WORKER') is not None:
        pytest.skip('Performance guard is timing-sensitive; run serially (pytest test_baselines.py::test_performance_guard).')

    repeats = dict(bare_init=8, bare_run=4, bare_run_no_people=4, het_init=8, births_run=4, crn_false_run=4)
    repeats.update({f'{name}_{which}':(8 if which == 'init' else 3) for name in ['fseis10', 'counts', 'coded', 'by_age'] for which in ['init', 'run']})

    # Measure, bracketed by CPU benchmarks for normalization
    r1 = sc.benchmark(which='numpy')
    measured = {name: measure_case(fn, which, repeats[name], crn=crn) for name,(fn,which,crn) in perf_cases().items()}
    r2 = sc.benchmark(which='numpy')
    ratio = (r1 + r2) / 2 / REF_MOPS
    normalized = {k: round(ratio*v, 4) for k,v in measured.items()}

    if do_save:
        sc.saveyaml(perf_guard_filename, dict(cpu_performance=ratio, times=normalized))
        if verbose:
            print('Saved performance-guard baselines (normalized seconds):')
            sc.pp(normalized)
        return normalized

    # Compare against recorded baselines and fail on a major regression
    recorded = sc.loadyaml(perf_guard_filename)['times']
    failures = []
    for name, base in recorded.items():
        new = normalized.get(name)
        if new is None:
            continue
        limit = base * REGRESSION_FACTOR
        if verbose:
            status = 'OK' if new <= limit else 'REGRESSION'
            print(f'  {name:22s}: {new*1e3:8.1f} ms (baseline {base*1e3:7.1f} ms, limit {limit*1e3:7.1f} ms) [{status}]')
        if new > limit:
            failures.append(f'{name}: {new*1e3:.1f} ms > limit {limit*1e3:.1f} ms (baseline {base*1e3:.1f} ms x {REGRESSION_FACTOR})')
    assert not failures, 'Performance regression detected:\n' + '\n'.join(failures)
    return normalized


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    json = test_benchmark() # Run this first so benchmarking is available even if results are different
    new  = test_baseline()
    guard = test_performance_guard()
    sim = make_sim(run=do_plot)

    T.toc()
