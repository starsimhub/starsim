"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import numpy as np
import pandas as pd
import sciris as sc

debug = False # If true, will run in serial
n_reps = 10 # Per trial
total_trials = 100
n_agents = 2_000
do_plot = 1


#%% Helper functions

def make_sim():
    sir = ss.SIR(
        beta = ss.beta(0.075),
        init_prev = ss.bernoulli(0.02),
    )
    random = ss.RandomNet(n_contacts=ss.poisson(4))

    sim = ss.Sim(
        n_agents = n_agents,
        start = sc.date('2020-01-01'),
        dur = 40,
        dt = 1,
        unit = 'day',
        diseases = sir,
        networks = random,
        verbose = 0,
    )

    return sim


def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_reps', n_reps)

    sir = sim.pars.diseases # There is only one disease in this simulation and it is a SIR
    net = sim.pars.networks # There is only one network in this simulation and it is a RandomNet

    for k, pars in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue

        v = pars['value']
        if k == 'beta':
            sir.pars.beta = ss.beta(v)
        elif k == 'init_prev':
            sir.pars.init_prev = ss.bernoulli(v)
        elif k == 'n_contacts':
            net.pars.n_contacts = ss.poisson(v)
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    if n_reps == 1:
        return sim

    ms = ss.MultiSim(sim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, n_reps)), initialize=True, debug=True, parallel=False) # Run in serial
    return ms


#%% Define the tests
def test_calibration(do_plot=False):
    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True), # Log scale and no "path", will be handled by build_sim (ablve)
        init_prev = dict(low=0.01, high=0.05, guess=0.15, path=('diseases', 'hiv', 'init_prev')), # Default type is suggest_float, no need to re-specify
        n_contacts = dict(low=2, high=10, guess=3, suggest_type='suggest_int', path=('networks', 'randomnet', 'n_contacts')), # Suggest int just for demo
    )

    # Make the sim and data
    sim = make_sim()

    infectious = ss.CalibComponent(
        name = 'Infectious',

        # "expected" actually from a simulation with pars
        #   beta=0.075, init_prev=0.02, n_contacts=4
        expected = pd.DataFrame({
            'n': [200, 197, 195], # Number of individuals sampled
            'x': [30, 30, 10],    # Number of individuals found to be infectious
        }, index=pd.Index([ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'n': sim.results.n_alive,
            'x': sim.results.sir.n_infected,
        }, index=pd.Index(sim.results.timevec, name='t')),

        conform = 'prevalent',
        nll_fn = 'betabinomial',

        weight = 1,
    )

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim, # Use default builder, Calibration.translate_pars
        components = infectious,
        #eval_fn = my_function, # Will call my_function(msim, eval_kwargs)
        #eval_kwargs = dict(expected=TRIAL_DATA),
        total_trials = total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    sc.printcyan('\nChecking fit...')
    calib.check_fit()

    if do_plot:
        calib.plot_sims()
        calib.plot_trend()

    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    # Useful for generating fake "expected" data
    if False:
        sim = make_sim()
        pars = {
            'beta'      : dict(value=0.075),
            'init_prev' : dict(value=0.02),
            'n_contacts': dict(value=4),
        }
        ms = build_sim(sim, pars, n_reps=25)
        ms.run().plot()

        dfs = []
        for sim in ms.sims:
            df = sim.to_df()
            df['prevalence'] = df['sir_n_infected']/df['n_alive']
            df['rand_seed'] = sim.pars.rand_seed
            dfs.append(df)
        df = pd.concat(dfs)
        import seaborn as sns
        sns.relplot(data=df, x='timevec', y='prevalence', hue='rand_seed', kind='line')

    T = sc.timer()
    do_plot = True

    sim, calib = test_calibration(do_plot=do_plot)

    T.toc()

    import matplotlib.pyplot as plt
    plt.show()