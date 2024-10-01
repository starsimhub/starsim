"""
Test that the current version of Starsim exactly matches
the baseline results.
"""

import numpy as np
import sciris as sc
import starsim as ss

baseline_filename  = sc.thisdir(__file__, 'baseline.json')
multisim_filename  = sc.thisdir(__file__, 'baseline_multisim.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(ss.__file__, 'regression', f'pars_v{ss.__version__}.json')
sc.options(interactive=False) # Assume not running interactively

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
    diseases = ['sir', 'sis']
    networks = ['random', 'mf', 'maternal']
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
    sim.to_json(filename=baseline_filename, keys='summary')

    # Save parameters
    sim.to_json(filename=parameters_filename, keys='pars') # If not different from previous version, can safely delete

    print('Done.')
    return


def multisim_baseline(save=False, n_runs=10, **kwargs):
    """
    Check or update the multisim baseline results; not part of the integration tests
    """
    word = 'Saving' if save else 'Checking'
    sc.heading(f'{word} MultiSim baseline...')

    # Make and run sims
    kwargs.setdefault('verbose', ss.options.verbose/n_runs*2)
    sim = make_sim(**kwargs)
    msim = ss.MultiSim(base_sim=sim)
    msim.run(n_runs)
    summary = msim.summarize()

    # Export results
    if save:
        sc.savejson(multisim_filename, summary)
        print('Done.')
    else:
        baseline = sc.loadjson(multisim_filename)
        ss.diff_sims(baseline, summary, die=False)

    return


def test_baseline():
    """ Compare the current default sim against the saved baseline """

    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline['summary']

    # Calculate new baseline
    new = make_sim()
    new.run()

    # Compute the comparison
    ss.diff_sims(old, new, die=True)

    return new


def test_benchmark(do_save=False, repeats=1, verbose=True):
    """ Compare benchmark performance """

    if verbose: print('Running benchmark...')
    try:
        previous = sc.loadjson(benchmark_filename)
    except FileNotFoundError:
        previous = None

    t_inits = []
    t_runs  = []

    def normalize_performance():
        """ Normalize performance across CPUs """
        t_bls = []
        bl_repeats = 3
        n_outer = 10
        n_inner = 1e6
        for r in range(bl_repeats):
            t0 = sc.tic()
            for i in range(n_outer):
                a = np.random.random(int(n_inner))
                b = np.random.random(int(n_inner))
                a*b
            t_bl = sc.toc(t0, output=True)
            t_bls.append(t_bl)
        t_bl = min(t_bls)
        reference = 0.07 # Benchmarked on an Intel i7-12700H CPU @ 2.90GHz
        ratio = reference/t_bl
        return ratio


    # Test CPU performance before the run
    r1 = normalize_performance()

    # Do the actual benchmarking
    for r in range(repeats):

        print(f'Repeat {r}')

        # Time initialization
        t0 = sc.tic()
        sim = make_sim()
        sim.init()
        t_init = sc.toc(t0, output=True)

        # Time running
        t0 = sc.tic()
        sim.run()
        t_run = sc.toc(t0, output=True)

        # Store results
        t_inits.append(t_init)
        t_runs.append(t_run)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1+r2)/2
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
                'dur':      sim.pars.dur,
                'dt':       sim.pars.dt,
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
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    if verbose:
        print('Done.')

    return json


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    json = test_benchmark() # Run this first so benchmarking is available even if results are different
    new  = test_baseline()
    sim = make_sim(run=do_plot)

    T.toc()
