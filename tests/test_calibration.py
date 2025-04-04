"""
Test calibration
"""

#%% Imports and settings

import starsim as ss
import numpy as np
import pandas as pd
import sciris as sc
from functools import partial
import pytest

debug = False # If true, will run in serial
total_trials = [10, 2][debug]
n_agents = 1_000
do_plot = True


#%% Helper functions

# Define a socio-economic status (SES) variable
SES = dict(LOW=0, HIGH=1) # Enum not pickling, so using dict instead

class ByAgeSES(ss.Analyzer):
    def __init__(self, age_groups=None, **kwargs):
        """
        Analyzer to count the number of infected by age and SES

        Args:
            age_groups (list): List of tuples with the age groups to analyze. Each tuple should contain the lower and upper bounds of the age group.
        """
        super().__init__(**kwargs)
        self.age_groups = age_groups
        if age_groups is None:
            self.age_groups = [(0, 1000)] # Default to all ages
        return
    
    def mod(self, lbl, ses, age_group):
        """
        Modify the label of the results to include SES and age group.

        Args:
            lbl (str): The label to modify.
            ses (str): The SES value.
            age_group (tuple): The age group as a tuple of lower and upper bounds.
        """
        agl, agh = age_group
        append = f'{ses}_{agl}_{agh}' if len(self.age_groups)>1 else ses
        return lbl + '_' + append

    def init_results(self):
        assert hasattr(self.sim.people, 'SES')
        for ses in SES.keys():
            for ag in self.age_groups:
                agl, agh = ag # age lower and upper bounds
                desc = f'SES={ses} & Age Group {agl}-{agh}'
                self.define_results(
                    ss.Result(self.mod('n_infected', ses, ag), dtype=int, label=f'Number Infected [{desc}]'),
                    ss.Result(self.mod('new_infections', ses, ag), dtype=int, label=f'Newly Infected [{desc}]'),
                    ss.Result(self.mod('n_alive', ses, ag), dtype=int, label=f'Number of Agents [{desc}]'),
                )
        return

    def update_results(self):
        ti = self.sim.t.ti
        for ses, val in SES.items():
            for ag in self.age_groups:
                agl, agh = ag # age lower and upper bounds
                self.results[self.mod('n_infected', ses, ag)][ti] = np.count_nonzero( self.sim.diseases.sir.infected[(self.sim.people.SES == val) & (self.sim.people.age >= agl) & (self.sim.people.age < agh)] )
                self.results[self.mod('new_infections', ses, ag)][ti] = np.count_nonzero( self.sim.diseases.sir.ti_infected[(self.sim.people.SES == val) & (self.sim.people.age >= agl) & (self.sim.people.age < agh)] == self.sim.diseases.sir.t.ti )
                self.results[self.mod('n_alive', ses, ag)][ti] = np.count_nonzero( self.sim.people.alive[(self.sim.people.SES == val) & (self.sim.people.age >= agl) & (self.sim.people.age < agh)] )
        return

    def step(self):
        pass


def make_sim(by_ses=False, analyzers=None, n_reps=1):
    """ Make a simple simulation with a SIR model and a random network """

    people_pars = dict(n_agents=n_agents)
    if by_ses:
        people_pars['extra_states'] = ss.State('SES', default=ss.choice([SES['LOW'], SES['HIGH']], p=[0.7, 0.3]))
    pop = ss.People(**people_pars)

    sir = ss.SIR(
        beta = ss.beta(0.075),
        init_prev = ss.bernoulli(0.02),
    )
    random = ss.RandomNet(n_contacts=ss.poisson(4))

    sim = ss.Sim(
        people = pop,
        start = sc.date('2020-01-01'),
        stop = sc.date('2020-02-12'),
        dt = 1,
        unit = 'day',
        diseases = sir,
        networks = random,
        verbose = 0,
        analyzers = analyzers,
    )

    return sim


def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_reps', 1)

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

    if reps == 1:
        return sim

    # Ignoring the random seed if provided via the reseed=True option in Calibration
    ms = ss.MultiSim(sim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, reps)), initialize=True, debug=True, parallel=False)
    return ms


#%% Define the tests

def test_onepar_normal(do_plot=True):
    sc.heading('Testing a single parameter (beta) with a normally distributed likelihood')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    # Make the sim and data
    sim = make_sim()

    prevalence = ss.Normal(
        name = 'Disease prevalence',
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': [0.13, 0.16, 0.06],    # Prevalence of infection
        }, index=pd.Index([ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']], name='t')), # On these dates

        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.sir.n_infected, # Instead of prevalence, let's compute it from infected and n_alive
            'n': sim.results.n_alive,
        }, index=pd.Index(sim.results.timevec, name='t')),

        # User can specify sigma2 to avoid automatically choosing the value that maximizes likelihood
        #sigma2 = 0.05, # (num_replicates/sigma2_model + 1/sigma2_data)^-1
        sigma2 = np.array([0.05, 0.25, 0.01])
    )

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        build_kw = dict(n_reps=3), # Reps per point
        reseed = False,
        components = [prevalence],
        total_trials = total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(do_plot=False), 'Calibration did not improve the fit'

    # Call plotting to look for exceptions
    if do_plot:
        calib.plot_final()
        calib.plot(bootstrap=False)
        calib.plot(bootstrap=True)
        calib.plot_optuna(['plot_param_importances', 'plot_optimization_history'])

    return


def test_onepar_custom(do_plot=True):
    sc.heading('Testing a single parameter (beta) with a custom likelihood')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    # Make the sim and data
    sim = make_sim()

    def eval(sim, expected):
        # Compute the squared error at one point in time
        date, p = expected
        if not isinstance(sim, ss.MultiSim):
            sim = ss.MultiSim(sims=[sim])

        ret = 0
        for s in sim.sims:
            ind = np.searchsorted(s.results.timevec, date, side='left')
            prev = s.results.sir.prevalence[ind]
            ret += (prev - p)**2
        return ret

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        #build_kw = dict(n_reps=3), # Use default if n_reps = 1
        reseed = True,
        eval_fn = eval, # Will call my_function(msim, eval_kwargs)
        eval_kw = dict(expected=(ss.date('2020-01-12'), 0.13)), # Will call eval(sim, **eval_kw)
        total_trials = total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(), 'Calibration did not improve the fit'
    return


def test_twopar_betabin_gammapois(do_plot=True):
    sc.heading('Testing a two parameters (beta and initial prevalence) with a two likelihoods (BetaBinomial and GammaPoisson)')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True), # Float par with log scale
        init_prev = dict(low=0.01, high=0.25, guess=0.15), # Default type is suggest_float, no need to re-specify
    )

    # Make the sim and data
    azs = [ByAgeSES(name='byses'), ByAgeSES(name='bysesage', age_groups=[(0,15),(15,100)])]
    sim = make_sim(by_ses=True, analyzers=azs) # Included SES and the corresponding analyzer

    num_infectious = ss.BetaBinomial(
        name = 'Number Infectious',
        weight = 0.75,
        conform = 'prevalent',

        n_boot = 25, # Testing bootstrap
        combine_reps = 'mean',

        expected = pd.DataFrame({
            'n': [140, 137, 60, 59], # Number of individuals sampled
            'x': [21, 23, 9, 11],    # Number of individuals found to be infectious
            't': [ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-01-12', '2020-01-25']], # On these dates
            'ses': ['LOW', 'LOW', 'HIGH', 'HIGH'], # By SES
        }).set_index(['t', 'ses']),

        extract_fn = lambda sim: pd.DataFrame({
            #'n': sim.results.n_alive,
            #'x': sim.results.sir.n_infected,
            'x': np.concatenate((sim.results.byses.n_infected_LOW, sim.results.byses.n_infected_HIGH)),
            'n': np.concatenate((sim.results.byses.n_alive_LOW, sim.results.byses.n_alive_HIGH)),
        }, index=pd.MultiIndex.from_product([SES.keys(), sim.results.timevec], names=['ses', 't'])),
    )

    incident_cases = ss.GammaPoisson(
        name = 'Incident Cases',
        weight = 1.5,
        conform = 'incident',

        n_boot = 25, # Testing bootstrap (note n_reps>1)
        combine_reps = 'sum',

        expected = pd.DataFrame({
            'n':  [100, 27, 54, 36],   # Number of person-years
            'x':  [740, 325, 200, 125], # Number of new infections
            't':  [ss.date(d) for d in ['2020-01-14', '2020-01-14', '2020-01-14', '2020-01-14']], # Between t and t1
            't1': [ss.date(d) for d in ['2020-01-15', '2020-01-15', '2020-01-15', '2020-01-15']],
            'ses': ['LOW', 'LOW', 'HIGH', 'HIGH'], # By SES
            'age': ['0-15', '15-100', '0-15', '15-100'], # By age group
        }).set_index(['t', 't1', 'ses', 'age']),

        extract_fn = lambda sim: pd.DataFrame({
            'x': np.concatenate((
                sim.results.bysesage.n_infected_LOW_0_15,
                sim.results.bysesage.n_infected_LOW_15_100,
                sim.results.bysesage.n_infected_HIGH_0_15,
                sim.results.bysesage.n_infected_HIGH_15_100
            )), # Number of events
            'n': np.concatenate((
                sim.results.bysesage.n_alive_LOW_0_15 * sim.t.dt_year,
                sim.results.bysesage.n_alive_LOW_15_100 * sim.t.dt_year,
                sim.results.bysesage.n_alive_HIGH_0_15 * sim.t.dt_year,
                sim.results.bysesage.n_alive_HIGH_15_100 * sim.t.dt_year
            )),# Person-years at risk
        }, index=pd.MultiIndex.from_product([SES.keys(), ['0-15', '15-100'], sim.results.timevec], names=['ses', 'age', 't'])),
    )

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        build_kw = dict(n_reps=3), # 3 reps per point
        reseed = True,
        components = incident_cases, #[num_infectious, incident_cases],
        total_trials = 15, #total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(), 'Calibration did not improve the fit'

    # Call plotting to look for exceptions
    if do_plot:
        calib.plot(bootstrap=False)
        calib.plot(bootstrap=True)
    return


def test_threepar_dirichletmultinomial_10reps(do_plot=True):
    sc.heading('Testing a three parameters (beta, initial prevalence, and number of contacts) with a DirichletMultinomial likelihood')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
        init_prev = dict(low=0.01, high=0.25, guess=0.15),
        n_contacts = dict(low=2, high=10, guess=3, suggest_type='suggest_int'), # Suggest int just for demo
    )

    # Make the sim and data
    sim = make_sim()

    def by_dow(sim):
        # Extract the number of new infections by day of week
        ret = pd.DataFrame({
            'x': sim.results.sir.new_infections, # Events
        }, index=pd.Index(sim.results.timevec, name='t'))
        ret['var'] = [f'x_{d.weekday()}' for d in ret.index]
        ret = ret.reset_index() \
            .pivot(columns='var', index='t', values='x') \
            .fillna(0)
        return ret

    cases_by_dow = ss.DirichletMultinomial(
        name = 'Cases by day of week',
        weight = 1,
        conform = 'incident',

        expected = pd.DataFrame({
            'x_0': [40, 60], # Monday
            'x_1': [40, 60], # Tuesday
            'x_2': [40, 60], # Wednesday
            'x_3': [40, 60], # Thursday
            'x_4': [40, 60], # Friday
            'x_5': [40, 60], # Saturday
            'x_6': [40, 60], # Sunday

            # incident conform will compute different in comulative counts between the
            # end of step t1 and the end of step t
            't': [ss.date(d) for d in ['2020-01-07', '2020-01-21']],
            't1': [ss.date(d) for d in ['2020-01-21', '2020-02-11']],
        }).set_index(['t', 't1']),

        extract_fn = by_dow
    )

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        build_kw = dict(n_reps=10), # Pass n_reps=10 to build_sim
        reseed = False,
        components = [cases_by_dow],
        total_trials = total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(), 'Calibration did not improve the fit'
    return


#%% Run as a script
if __name__ == '__main__':
    import matplotlib.pyplot as plt

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
        plt.show()


    do_plot = True
    #for f in [test_onepar_normal, test_onepar_custom, test_twopar_betabin_gammapois, test_threepar_dirichletmultinomial_10reps]:
    for f in [test_twopar_betabin_gammapois]:
        T = sc.timer()
        f(do_plot=do_plot)
        T.toc()
    plt.show()
