"""
Test that the current version of Starsim exactly matches
the baseline results.
"""

import numpy as np
import sciris as sc
import starsim as ss

do_plot = True
do_save = False
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(ss.__file__, 'regression', f'pars_v{ss.__version__}.json')
sc.options(interactive=False) # Assume not running interactively

# Define the parameters
pars = sc.objdict(
    start         = 2000,       # Starting year
    n_years       = 20,         # Number of years to simulate
    dt            = 0.2,        # Timestep
    verbose       = 0,          # Don't print details of the run
    rand_seed     = 2,          # Set a non-default seed
)

def make_people():
    ss.set_seed(pars.rand_seed)
    n_agents = int(10e3)
    ppl = ss.People(n_agents=n_agents)
    return ppl


def make_sim(ppl=None, do_plot=False, **kwargs):
    '''
    Define a default simulation for testing the baseline, including
    interventions to increase coverage. If run directly (not via pytest), also
    plot the sim by default.
    '''

    if ppl is None:
        ppl = make_people()

    # Make the sim
    hiv = ss.HIV()
    hiv.pars.beta = {'mf': [0.15, 0.10], 'maternal': [0.2, 0]}
    networks = [ss.MFNet(), ss.MaternalNet()]
    sim = ss.Sim(pars=pars, people=ppl, networks=networks, demographics=ss.Pregnancy(), diseases=hiv)

    # Optionally plot
    if do_plot:
        sim.run()
        sim.plot()

    return sim


def save_baseline():
    '''
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    '''

    print('Updating baseline values...')

    # Export default parameters
    s1 = make_sim(use_defaults=True)
    s1.export_pars(filename=parameters_filename) # If not different from previous version, can safely delete

    # Export results
    s2 = make_sim(use_defaults=False)
    s2.run()
    s2.to_json(filename=baseline_filename, keys='summary')

    print('Done.')

    return


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''
    
    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline['summary']

    # Calculate new baseline
    new = make_sim()
    new.run()

    # Compute the comparison
    ss.diff_sims(old, new, die=True)

    return new


def test_benchmark(do_save=do_save, repeats=1, verbose=True):
    ''' Compare benchmark performance '''
    
    if verbose: print('Running benchmark...')
    try:
        previous = sc.loadjson(benchmark_filename)
    except FileNotFoundError:
        previous = None

    t_peoples = []
    t_inits   = []
    t_runs    = []

    def normalize_performance():
        ''' Normalize performance across CPUs '''
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
        
        print("Repeat ", r)
        
        # Time people
        t0 = sc.tic()
        ppl = make_people()
        t_people = sc.toc(t0, output=True)
        
        # Time initialization
        t0 = sc.tic()
        sim = make_sim(ppl, verbose=0)
        sim.initialize()
        t_init = sc.toc(t0, output=True)

        # Time running
        t0 = sc.tic()
        sim.run()
        t_run = sc.toc(t0, output=True)

        # Store results
        t_peoples.append(t_people)
        t_inits.append(t_init)
        t_runs.append(t_run)
        
        # print(t_people, t_init, t_run)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1+r2)/2
    t_people = min(t_peoples)*ratio
    t_init = min(t_inits)*ratio
    t_run  = min(t_runs)*ratio

    # Construct json
    n_decimals = 3
    json = {'time': {
                'people':     round(t_people, n_decimals),
                'initialize': round(t_init, n_decimals),
                'run':        round(t_run,  n_decimals),
                },
            'parameters': {
                'n_agents': sim.pars['n_agents'],
                'n_years':  sim.pars['n_years'],
                'dt':       sim.pars['dt'],
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

    # Start timing and optionally enable interactive plotting
    sc.options(interactive=do_plot)
    T = sc.tic()

    json = test_benchmark(do_save=do_save, repeats=5) # Run this first so benchmarking is available even if results are different
    new  = test_baseline()
    sim = make_sim(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')
