"""
Define the calibration class
"""
import os
import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as sps
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

# Lazy imports (do not import unless actually used, saves 500 ms on load time)
op = sc.importbyname('optuna', lazy=True)
sns = sc.importbyname('seaborn', lazy=True)
vis = sc.importbyname('optuna.visualization.matplotlib', lazy=True)

__all__ = ['Calibration']


class Calibration(sc.prettyobj):
    """
    A class to handle calibration of Starsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Args:
        sim          (Sim)   : the base simulation to calibrate
        calib_pars   (dict)  : a dictionary of the parameters to calibrate of the format `dict(key1=dict(low=1, high=2, guess=1.5, **kwargs), key2=...)`, where kwargs can include "suggest_type" to choose the suggest method of the trial (e.g. suggest_float) and args passed to the trial suggest function like "log" and "step"
        n_workers    (int)   : the number of parallel workers (if None, will use all available CPUs)
        total_trials (int)   : the total number of trials to run, each worker will run approximately n_trials = total_trial / n_workers
        reseed       (bool)  : whether to generate new random seeds for each trial
        build_fn  (callable) : function that takes a sim object and calib_pars dictionary and returns a modified sim
        build_kw      (dict) : a dictionary of options that are passed to build_fn to aid in modifying the base simulation. The API is `self.build_fn(sim, calib_pars=calib_pars, **self.build_kw)`, where sim is a copy of the base simulation to be modified with calib_pars
        components    (list) : CalibComponents independently assess pseudo-likelihood as part of evaluating the quality of input parameters
        prune_fn  (callable) : Function that takes a dictionary of parameters and returns True if the trial should be pruned
        eval_fn   (callable) : Function mapping a sim to a float (e.g. negative log likelihood) to be maximized. If None, the default will use CalibComponents.
        eval_kw       (dict) : Additional keyword arguments to pass to the eval_fn
        label        (str)   : a label for this calibration object
        study_name   (str)   : name of the optuna study
        db_name      (str)   : the name of the database file (default: 'starsim_calibration.db')
        continue_db  (bool)  : whether to continue if the database already exists, removes the database if false (default: false, any existing database will be deleted)
        keep_db      (bool)  : whether to keep the database after calibration (default: false, the database will be deleted)
        storage      (str)   : the location of the database (default: sqlite)
        sampler (BaseSampler): the sampler used by optuna, like optuna.samplers.TPESampler
        die          (bool)  : whether to stop if an exception is encountered (default: false)
        debug        (bool)  : if True, do not run in parallel
        verbose      (bool)  : whether to print details of the calibration
    """
    def __init__(self, sim, calib_pars, n_workers=None, total_trials=None, reseed=True,
                 build_fn=None, build_kw=None, eval_fn=None, eval_kw=None, components=None, prune_fn=None,
                 label=None, study_name=None, db_name=None, keep_db=None, continue_db=None, storage=None,
                 sampler=None, die=False, debug=False, verbose=True):

        # Handle run arguments
        if total_trials is None: total_trials   = 100
        if n_workers    is None: n_workers      = 1 if debug else sc.cpu_count()
        if study_name   is None: study_name     = 'starsim_calibration'
        if db_name      is None: db_name        = f'{study_name}.db'
        if continue_db  is None: continue_db    = False
        if keep_db      is None: keep_db        = False
        if storage      is None: storage        = f'sqlite:///{db_name}'

        self.build_fn       = build_fn
        self.build_kw       = build_kw or dict()
        self.eval_fn        = eval_fn or self._eval_fit
        self.eval_kw        = eval_kw or dict()
        self.components     = sc.tolist(components)
        self.prune_fn       = prune_fn

        n_trials = int(np.ceil(total_trials/n_workers))
        kw = dict(n_trials=n_trials, n_workers=int(n_workers), debug=debug, study_name=study_name,
                  db_name=db_name, continue_db=continue_db, keep_db=keep_db, storage=storage, sampler=sampler)
        self.run_args = sc.objdict(kw)

        # Handle other inputs
        self.label      = label
        self.sim        = sim
        self.calib_pars = calib_pars
        self.reseed     = reseed
        self.die        = die
        self.verbose    = verbose
        self.calibrated = False
        self.before_msim = None
        self.after_msim  = None

        self.study = None

        return

    def run_sim(self, calib_pars=None, label=None):
        """ Create and run a simulation """
        sim = sc.dcp(self.sim)
        if label: sim.label = label

        sim = self.build_fn(sim, calib_pars=calib_pars, **self.build_kw)

        try:
            sim.run() # Run the simulation (or MultiSim)
            return sim
        except Exception as E:
            if self.die:
                raise E
            else:
                print(f'Encountered error running sim!\nParameters:\n{calib_pars}\nTraceback:\n{sc.traceback()}')
                output = None
                return output

    def _sample_from_trial(self, pardict=None, trial=None):
        """
        Take in an optuna trial and sample from pars, after extracting them from
        the structure they're provided in
        """
        pars = sc.dcp(pardict)
        for parname, spec in pars.items():
            if 'value' in spec:
                # Already have a value, likely running initial or final values as part of checking the fit
                continue

            if 'suggest_type' in spec:
                suggest_type = spec.pop('suggest_type')
                sampler_fn = getattr(trial, suggest_type)
            else:
                sampler_fn = trial.suggest_float

            path = spec.pop('path', None) # remove path for the sampler
            guess = spec.pop('guess', None) # remove guess for the sampler
            spec['value'] = sampler_fn(name=parname, **spec) # suggest values!
            spec['path'] = path
            spec['guess'] = guess

        return pars

    def _eval_fit(self, sim, **kwargs):
        """ Evaluate the fit by evaluating the negative log likelihood, used only for components"""
        nll = 0 # Negative log likelihood
        for component in self.components:
            nll += component(sim, **kwargs)
        return nll

    def plot(self, **kwargs):
        """"
        Plot the calibration results. For a component-based likelihood, it only
        makes sense to directly call plot after calling eval_fn.
        """
        assert self.before_msim is not None and self.after_msim is not None, 'Please run check_fit() before plotting'
        figs = []
        for component in self.components:
            component.eval(self.before_msim)
            before_actual = component.actual
            before_actual['calibrated'] = 'Before Calibration'

            component.eval(self.after_msim)
            after_actual = component.actual
            after_actual['calibrated'] = 'After Calibration'

            actual = pd.concat([before_actual, after_actual])
            fig = component.plot(actual, **kwargs)
            figs.append(fig)
        return figs

    def run_trial(self, trial):
        """ Define the objective for Optuna """
        if self.calib_pars is not None:
            pars = self._sample_from_trial(self.calib_pars, trial)
        else:
            pars = None

        if self.reseed:
            pars['rand_seed'] = trial.suggest_int('rand_seed', 0, 1_000_000) # Choose a random rand_seed

        # Prune if the prune_fn returns True
        if self.prune_fn is not None and self.prune_fn(pars):
            raise op.exceptions.TrialPruned()

        sim = self.run_sim(pars)

        # Compute fit
        fit = self.eval_fn(sim, **self.eval_kw)
        return fit

    def worker(self):
        """ Run a single worker """

        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None)
        return output

    def run_workers(self):
        """ Run multiple workers in parallel """
        if self.run_args.n_workers > 1 and not self.run_args.debug: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output

    def remove_db(self):
        """ Remove the database file if keep_db is false and the path exists """
        try:
            if 'sqlite' in self.run_args.storage:
                # Delete the file from disk
                if os.path.exists(self.run_args.db_name):
                    os.remove(self.run_args.db_name)
                if self.verbose: print(f'Removed existing calibration file {self.run_args.db_name}')
            else:
                # Delete the study from the database e.g., mysql
                op.delete_study(study_name=self.run_args.study_name, storage=self.run_args.storage)
                if self.verbose: print(f'Deleted study {self.run_args.study_name} in {self.run_args.storage}')
        except Exception as E:
            if self.verbose:
                print('Could not delete study, skipping...')
                print(str(E))
        return

    def make_study(self):
        """ Make a study, deleting if it already exists and user does not want to continue_db """
        if not self.run_args.continue_db:
            self.remove_db()
        if self.verbose: print(self.run_args.storage)
        try:
            study = op.create_study(storage=self.run_args.storage, study_name=self.run_args.study_name, direction='minimize')
        except op.exceptions.DuplicatedStudyError:
            ss.warn(f'Study named {self.run_args.study_name} already exists in storage {self.run_args.storage}, loading...')
            study = op.create_study(storage=self.run_args.storage, study_name=self.run_args.study_name, direction='minimize', load_if_exists=True)
            try:
                self.best_pars = sc.objdict(study.best_params)
            except Exception as E:
                print(f'Could not get best parameters: {str(E)}')
                self.best_pars = None
        return study

    def calibrate(self, calib_pars=None, **kwargs):
        """
        Perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.study = self.make_study()
        self.run_workers()
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
        self.best_pars = sc.objdict(study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        # Parse the study into a data frame, self.df while also storing the best parameters
        self.parse_study(study)

        if self.verbose: print('Best pars:', self.best_pars)

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()

        return self

    def to_df(self, top_k=None):
        """ Return the top K results as a dataframe, sorted by value """
        if self.study is None:
            raise ValueError('Please run calibrate() before saving results')

        df = sc.dataframe(self.study.trials_dataframe())
        df = df.sort_values(by='value').set_index('number')

        if top_k is not None:
            df = df.head(top_k)

        return df

    def check_fit(self, do_plot=True):
        """ Run before and after simulations to validate the fit """
        if self.verbose: sc.printcyan('\nChecking fit...')

        before_pars = sc.dcp(self.calib_pars)
        for spec in before_pars.values():
            spec['value'] = spec['guess'] # Use guess values

        # Load in case calibration was interrupted
        if self.best_pars is None:
            try:
                study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
                self.best_pars = sc.objdict(study.best_params)
            except:
                raise ValueError('Seems like calibration did not finish successfully and also unable to obtain best parameters from the {self.run_args.storage}:{self.run_args.study_name} as the study was likely automatically deleted, see keep_db.')

        after_pars = sc.dcp(self.calib_pars)
        for parname, spec in after_pars.items():
            spec['value'] = self.best_pars[parname] # Use best parameters from calibration

        self.before_msim = self.build_fn(self.sim.copy(), calib_pars=before_pars, **self.build_kw)
        self.after_msim = self.build_fn(self.sim.copy(), calib_pars=after_pars, **self.build_kw)

        fix_before = isinstance(self.before_msim, ss.Sim)
        fix_after = isinstance(self.after_msim, ss.Sim)
        if fix_after or fix_after:
            if fix_before:
                self.before_msim = ss.MultiSim(self.before_msim, initialize=True, debug=True, parallel=False, n_runs=1)

            if fix_after:
                self.after_msim = ss.MultiSim(self.after_msim, initialize=True, debug=True, parallel=False, n_runs=1)

        msim = ss.MultiSim(self.before_msim.sims + self.after_msim.sims)
        msim.run()

        self.before_fits = self.eval_fn(self.before_msim, **self.eval_kw)
        self.after_fits = self.eval_fn(self.after_msim, **self.eval_kw)

        if do_plot:
            self.plot()

        print(f'Fit with original pars: {self.before_fits}')
        print(f'Fit with best-fit pars: {self.after_fits}')

        before = self.before_fits.mean() if isinstance(self.before_fits, np.ndarray) else self.before_fits
        after = self.after_fits.mean() if isinstance(self.after_fits, np.ndarray) else self.after_fits

        if after <= before:
            print(f'✓ Calibration improved fit {before} --> {after}')
            return True

        print(f'✗ Calibration did not improve fit as the objective got worse ({before} --> {after}), but this sometimes happens stochastically and is not necessarily an error')
        return False

    def parse_study(self, study):
        """Parse the study into a data frame -- called automatically """
        best = study.best_params
        self.best_pars = best

        if self.verbose: print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        for trial in study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        if self.verbose: print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    print(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.study_data = data
        self.df = sc.dataframe.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch']) # Sort
        return

    def to_json(self, filename=None, indent=2, **kwargs):
        """ Convert the results to JSON """
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json

    def plot_final(self, **kwargs):
        """
        Plot sims after calibration

        Args:
            kwargs (dict): passed to MultiSim.plot()
        """
        jup = ss.options.jupyter if 'jupyter' in ss.options else sc.isjupyter()
        ss.options.jupyter = False

        pars = sc.dcp(self.calib_pars)
        for parname, spec in pars.items():
            spec['value'] = self.best_pars[parname] # Use best parameters from calibration
        msim = self.build_fn(self.sim.copy(), calib_pars=pars, **self.build_kw)

        msim.run()
        self.eval_fn(msim, **self.eval_kw)

        if isinstance(msim, ss.MultiSim): # It could be a single simulation
            msim.reduce()
        fig = msim.plot()
        fig.suptitle('After calibration')
        ss.options.jupyter = jup
        return fig

    def plot_optuna(self, methods=None):
        """ Plot Optuna's visualizations """
        figs = []

        methods = sc.tolist(methods)

        if not methods:
            methods = [
                'plot_contour',
                'plot_edf',
                'plot_hypervolume_history',
                'plot_intermediate_values',
                'plot_optimization_history',
                'plot_parallel_coordinate',
                'plot_param_importances',
                'plot_pareto_front',
                'plot_rank',
                'plot_slice',
                'plot_terminator_improvement',
                'plot_timeline',
            ]

        for method in methods:
            try:
                if method == 'plot_contour':
                    # Remove 'rand_seed' if present
                    params = list(self.calib_pars.keys())
                    if 'rand_seed' in params:
                        params.remove('rand_seed')
                    fig = vis.plot_contour(self.study, params=params)
                else:
                    fig = getattr(vis, method)(self.study)
                figs.append(fig)
            except Exception as E:
                print(f'Could not run {method}: {str(E)}')
        return figs


#%% Calibration components

__all__ += ['linear_interp', 'linear_accum', 'step_containing'] # Conformers
__all__ += ['CalibComponent'] # Calib component base class
__all__ += ['BetaBinomial', 'DirichletMultinomial', 'GammaPoisson', 'Normal'] # Specific calib components

def linear_interp(expected, actual):
    """
    Simply interpolate, use for prevalent (stock) data like prevalence

    Args:
        expected (pd.DataFrame): The expected data from field observation, must have 't' in the index and columns corresponding to specific needs of the selected component.
        actual (pd.DataFrame): The actual data from the simulation, must have 't' in the index and columns corresponding to specific needs of the selected component.
    """
    t = expected.index
    conformed = pd.DataFrame(index=expected.index)
    for k in actual:
        conformed[k] = np.interp(x=t, xp=actual.index, fp=actual[k])

    return conformed

def step_containing(expected, actual):
    """
    Find the step containing the the timepoint.  Use for prevalent data like
    prevalence where you want to match a specific time point rather than
    interpolate.

    Args:
        expected (pd.DataFrame): The expected data from field observation, must have 't' in the index and columns corresponding to specific needs of the selected component.
        actual (pd.DataFrame): The actual data from the simulation, must have 't' in the index and columns corresponding to specific needs of the selected component.
    """
    t = expected.index
    inds = np.searchsorted(actual.index, t, side='left')
    conformed = pd.DataFrame(index=expected.index)
    for k in actual:
        conformed[k] = actual[k].values[inds] # .values because indices differ

    return conformed

def linear_accum(expected, actual):
    """
    Interpolate in the cumulative sum, then difference. Use for incident data
    (flows) like incidence or new_deaths. The accumulation is done between 't'
    and 't1', both of which must be present in the index of expected and actual
    dataframes.

    Args:
        expected (pd.DataFrame): The expected data from field observation, must have 't' and 't1' in the index and columns corresponding to specific needs of the selected component.
        actual (pd.DataFrame): The actual data from the simulation, must have 't' and 't1' in the index and columns corresponding to specific needs of the selected component.
    """
    t0 = expected.index.get_level_values('t')
    t1 = expected.index.get_level_values('t1')
    sim_t = actual.index

    fp = actual.cumsum()
    ret = {}
    for c in fp.columns:
        vals = fp[c].values.flatten()
        v0 = np.interp(x=t0, xp=sim_t, fp=vals) # Cum value at t0
        v1 = np.interp(x=t1, xp=sim_t, fp=vals) # Cum value at t1

        # Difference between end of step t1 and end of step t
        ret[c] = v1 - v0

    df = pd.DataFrame(ret, index=expected.index)
    return df


class CalibComponent(sc.prettyobj):
    """
    A class to compare a single channel of observed data with output from a
    simulation. The Calibration class can use several CalibComponent objects to
    form an overall understanding of how will a given simulation reflects
    observed data.

    Args:
        name (str) : the name of this component. Importantly, if
            extract_fn is None, the code will attempt to use the name, like
            "hiv.prevalence" to automatically extract data from the simulation.
        expected (df) : pandas DataFrame containing calibration data. The index should be the time 't' in either floating point years or datetime.
        extract_fn (callable) : a function to extract predicted/actual data in the same
            format and with the same columns as `expected`.
        conform (str | callable): specify how to handle timepoints that don't
            align exactly between the expected and the actual/predicted/simulated
            data so they "conform" to a common time grid. Whether the data represents
            a 'prevalent' or an 'incident' quantity impacts how this alignment is performed.

            If 'prevalent', it means data in expected & actual dataframes represent
            the current state of the system, stocks like the number of currently infected
            individuals. In this case, the data in 'simulated' or 'actual' will be interpolated
            to match the timepoints in 'expected', allowing for pointwise comparisons
            between the expected and actual data.

            If 'incident', it means data in expected & actual dataframes represent the accumulation
            of system states over a period of time, flows like the incidence of new infections. In
            this case, teh data in 'simulated' or 'actual' will be interpolated at the start ('t')
            and the end ('t1') of the period of interest in 'expected'. The difference between these
            two interpolated values will be used for comparison.

            Finally, 'step_containing' is a special case for prevalent data where the actual data is
            interpolated using a "zero order hold" method. This means that the value of the actual (simulated) data
            is matched to the timepoint in the expected data that contains the timepoint of the actual data.

        weight (float): The weight applied to the log likelihood of this component. The total log likelihood is the sum of the log likelihoods of all components, each multiplied by its weight.
        include_fn (callable): A function accepting a single simulation and returning a boolean to determine if the simulation should be included in the current component. If None, all simulations are included.
        n_boot (int): Experimental! Bootstrap sum sim results over seeds before comparing against expected results. Not appropriate for all component types.
        combine_reps (str): How to combine multiple repetitions of the same pars. Options are None, 'mean', 'sum', or other such operation. Default is None, which evaluates replicates independently instead of first combining before likelihood evaluation.
        kwargs: Additional arguments to pass to the likelihood function
    """
    def __init__(self, name, expected, extract_fn, conform, weight=1, include_fn=None, n_boot=None, combine_reps=None):
        self.name = name
        self.expected = expected
        self.extract_fn = extract_fn
        self.weight = weight
        self.include_fn = include_fn
        self.n_boot = n_boot

        self.combine_reps = combine_reps
        self.combine_kwargs = dict()
        if isinstance(self.combine_reps, str) and hasattr(pd.core.groupby.DataFrameGroupBy, self.combine_reps):
            # Most of these methods take numeric_only, which can help with stability
            self.combine_kwargs = dict(numeric_only=True)

        self.avail_conforms = {
            'none': None, # passthrough
            'incident':  linear_accum,  # or self.linear_accum if left as staticmethod
            'prevalent': linear_interp,
            'step_containing': step_containing,
        }
        self.conform = self._validate_conform(conform)
        return

    def _validate_conform(self, conform):
        ''' Validate the conform argument '''
        if not isinstance(conform, str) and not callable(conform):
            raise Exception(f"The conform argument must be a string or a callable function, not {type(conform)}.")
        elif isinstance(conform, str):
            conform_ = self.avail_conforms.get(conform.lower(), 'NOT FOUND')
            if conform_ == 'NOT FOUND':
                avail = self.avail_conforms.keys()
                raise ValueError(f"The conform argument must be one of {avail}, not {conform}.")
        else:
            conform_ = conform
        return conform_

    def _combine_reps_nll(self, expected, actual, **kwargs):
        if self.combine_reps is None:
            nll = self.compute_nll(expected, actual, **kwargs) # Negative log likelihood
        else:
            timecols = [c for c in self.actual.columns if isinstance(self.actual[c].iloc[0], dt.datetime)] # Not robust to data types
            actual_combined = actual.groupby(timecols).aggregate(func=self.combine_reps, **self.combine_kwargs)
            actual_combined['rand_seed'] = 0 # Fake the seed
            actual_combined = actual_combined.reset_index().set_index('rand_seed') # Make it look like self.actual
            nll = self.compute_nll(expected, actual_combined, **kwargs)
        return nll

    def eval(self, sim, **kwargs):
        """ Compute and return the negative log likelihood """
        actuals = []
        if isinstance(sim, ss.MultiSim):
            for s in sim.sims:
                if self.include_fn is not None and not self.include_fn(s):
                    continue # Skip this simulation
                actual = self.extract_fn(s)
                if self.conform is not None:
                    actual = self.conform(self.expected, actual) # Conform
                actual['rand_seed'] = s.pars.rand_seed
                actuals.append(actual)
        else:
            if self.include_fn is None or self.include_fn(sim):
                actual = self.extract_fn(sim) # Extract
                if self.conform is not None:
                    actual = self.conform(self.expected, actual) # Conform
                actual['rand_seed'] = sim.pars.rand_seed
                actuals = [actual]

        if len(actuals) == 0: # No sims met the include criteria
            self.actual = None
            return np.inf

        self.actual = pd.concat(actuals).reset_index().set_index('rand_seed')
        seeds = self.actual.index.unique()

        if self.n_boot is None or self.n_boot == 1 or len(seeds) == 1:
            self.nll = self._combine_reps_nll(self.expected, self.actual, **kwargs)
            if self.weight == 0: # Resolve possible 0 * inf
                return 0
            wnll = self.weight * np.mean(self.nll)
            if np.isnan(wnll):
                return np.inf # Convert nan to inf
            return wnll

        # Bootstrapped aggregation
        boot_size = len(seeds)
        nlls = np.zeros(self.n_boot)
        for bi in range(self.n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)
            actual = self.actual.loc[use_seeds]
            nll = self._combine_reps_nll(self.expected, actual, **kwargs)
            nlls[bi] = np.mean(nll) # Mean across reps
        self.nll = np.mean(nlls) # Mean across bootstraps

        if self.weight == 0:
            return 0 # Resolve possible 0 * inf

        wnll = self.weight * self.nll
        if np.isnan(wnll):
            return np.inf # Convert nan to inf
        return wnll

    def __call__(self, sim, **kwargs):
        return self.eval(sim, **kwargs)

    def __repr__(self):
        return f'Calibration component with name {self.name}'

    def plot(self, actual=None, bootstrap=False, **kwargs):
        actual = self.actual if actual is None else actual
        if actual is None:
            return None # Nothing to do

        if 'calibrated' not in actual.columns:
            actual['calibrated'] = 'Calibration'

        g = sns.FacetGrid(data=actual.reset_index(), col='t', row='calibrated', sharex='col', sharey=False, margin_titles=True, height=3, aspect=1.5)

        if bootstrap:
            g.map_dataframe(self.plot_facet_bootstrap)
        else:
            g.map_dataframe(self.plot_facet)
        g.set_titles(row_template='{row_name}')
        for (row_val, col_val), ax in g.axes_dict.items():
            if row_val == g.row_names[0] and isinstance(col_val, dt.datetime):
                ax.set_title(col_val.strftime('%Y-%m-%d'))

        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle(self.name)
        return g.fig

class BetaBinomial(CalibComponent):
    def __init__(self, name, expected, extract_fn, conform, weight=1, alpha_prior=1, beta_prior=1, include_fn=None, n_boot=None, combine_reps=None):
        """
        Beta-binomial likelihood component with per-timepoint concentration parameters.
        
        This component maps simulated prevalence to a detection probability q(x̃), then
        evaluates observed data using a beta-binomial distribution with concentration
        parameter kappa that reflects survey-level variability (not simulation uncertainty).
        
        Args:
            name (str): Name of this component
            expected (pd.DataFrame): Real observed data with required columns:
                - 'n': sample sizes (positive integers) 
                - 'x': positive cases (non-negative integers ≤ n)
                - EITHER 'kappa': concentration parameters (positive floats)
                - OR 'ci_low' AND 'ci_high': 95% confidence interval bounds (0 ≤ low < high ≤ 1)
            extract_fn (callable): Function to extract simulated data in same format
            conform (str|callable): How to align timepoints ('prevalent', 'incident', etc.)
            weight (float): Weight for this component in overall likelihood
            alpha_prior (float): Prior parameter for prevalence estimation (default=1)
            beta_prior (float): Prior parameter for prevalence estimation (default=1) 
            include_fn (callable): Optional filter for which simulations to include
            n_boot (int): Optional bootstrap samples
            combine_reps (str): How to combine replicates ('mean', 'sum', etc.)
            
        The likelihood uses: s^obs ~ BetaBinomial(n^obs, α=κ*q, β=κ*(1-q))
        where q = (x_sim + alpha_prior) / (n_sim + alpha_prior + beta_prior)
        and κ can vary by timepoint to reflect different survey characteristics.
        """
        super().__init__(name, expected, extract_fn, conform, weight, include_fn, n_boot, combine_reps)
        
        # Validate priors
        if alpha_prior <= 0:
            raise ValueError(f"alpha_prior must be positive, got {alpha_prior}")
        if beta_prior <= 0:
            raise ValueError(f"beta_prior must be positive, got {beta_prior}")
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Validate expected data contains integers for counts
        if not all(expected['x'].apply(lambda x: isinstance(x, (int, np.integer)) and x >= 0)):
            raise ValueError("expected['x'] must contain non-negative integers (observed positive cases)")
        if not all(expected['n'].apply(lambda x: isinstance(x, (int, np.integer)) and x > 0)):
            raise ValueError("expected['n'] must contain positive integers (sample sizes)")
        
        # Validate that x <= n for all observations
        if not all(expected['x'] <= expected['n']):
            raise ValueError("observed positive cases (x) cannot exceed sample size (n)")
            
        # Check for dispersion parameters (kappa or CI bounds)
        has_kappa = 'kappa' in expected.columns
        has_ci = 'ci_low' in expected.columns and 'ci_high' in expected.columns
        
        if has_kappa and has_ci:
            raise ValueError("Provide either 'kappa' column OR 'ci_low'/'ci_high' columns, not both")
        
        # Process expected DataFrame
        self.expected = expected.copy()
        
        # Store dispersion mode flag
        self.use_overdispersion = has_kappa or has_ci
        
        if has_ci:
            # Calculate kappa from confidence intervals
            self.expected['kappa'] = self._calculate_kappa_from_ci()
        elif has_kappa:
            # Validate all kappa values are positive
            if not all(self.expected['kappa'] > 0):
                raise ValueError("All kappa values must be positive")
        # If neither kappa nor CI provided, will use Binomial fallback
            
    def _calculate_kappa_from_ci(self):
        """Calculate kappa values from confidence interval columns using vectorized operations"""
        import warnings
        
        # Single epsilon for all numerical guards
        eps = 1e-10
        
        # Extract arrays for vectorized operations
        n_obs = self.expected['n'].values
        x_obs = self.expected['x'].values
        ci_lower = self.expected['ci_low'].values
        ci_upper = self.expected['ci_high'].values
        
        # Numerical guard: ensure p ∈ (ε, 1-ε) to avoid division by zero in p*(1-p)
        p_hat = x_obs / n_obs
        p_hat = np.clip(p_hat, eps, 1 - eps)

        # Validate CI bounds (vectorized checks)
        invalid_bounds = (ci_lower < 0) | (ci_upper > 1) | (ci_lower >= ci_upper)
        if invalid_bounds.any():
            bad_indices = self.expected.index[invalid_bounds].tolist()
            raise ValueError(f"Invalid confidence intervals at indices {bad_indices}: must have 0 ≤ low < high ≤ 1")

        outside_ci = (p_hat < ci_lower) | (p_hat > ci_upper)
        if outside_ci.any():
            bad_indices = self.expected.index[outside_ci].tolist()
            bad_props = p_hat[outside_ci]
            bad_cis = list(zip(ci_lower[outside_ci], ci_upper[outside_ci]))
            raise ValueError(f"Observed proportions {bad_props} not within confidence intervals {bad_cis} at indices {bad_indices}")

        # Calculate standard error from CI (vectorized)
        se_hat = (ci_upper - ci_lower) / (2 * 1.96)

        # Calculate ICC and convert to kappa
        # For n=1: design effect = 1, equivalent to binomial (rho=0, kappa=large)
        # For n>1: calculate rho from ICC formula
        rho = np.zeros_like(n_obs, dtype=float)
        
        # Calculate rho only for n > 1 cases
        valid = n_obs > 1
        if np.any(valid):
            n_v, se_v, p_v = n_obs[valid], se_hat[valid], p_hat[valid]
            # Ensure denominator > 0 to prevent division by zero
            denom = np.maximum(p_v * (1 - p_v), eps)
            rho[valid] = (n_v * se_v**2) / denom - 1 / (n_v - 1)
        
        # Numerical guard: clip negative rho to 0 (ensures valid ICC ≥ 0)
        rho = np.maximum(rho, 0)

        # Check for problematic cases where rho >= 1 (implies kappa <= 0)
        # This happens when CI is too tight for the given p and n combination
        problematic_rho = rho >= 1
        if problematic_rho.any():
            bad_indices = self.expected.index[problematic_rho].tolist()
            bad_rho_values = rho[problematic_rho]
            warnings.warn(
                f"Confidence intervals at indices {bad_indices} are too tight for the observed data, "
                f"implying ρ ≥ 1 (ρ = {bad_rho_values}), which would give κ ≤ 0. "
                f"This is inconsistent with the beta-binomial model. "
                f"Capping ρ at {1 - eps:.6f} to ensure positive κ.",
                UserWarning
            )
        
        # Cap rho to ensure positive kappa
        # For kappa > 0: (1/rho) - 1 > 0 => 1/rho > 1 => rho < 1
        max_rho = 1 - eps  # This gives kappa ≈ eps/(1-eps) ≈ eps, very small but positive
        rho = np.minimum(rho, max_rho)
        
        # Convert to kappa: κ = (1/ρ) - 1
        # Handle rho=0 case (design effect=1, binomial limit) without division by zero
        kappa = np.full_like(rho, 1e6, dtype=float)  # Default to large kappa
        nonzero = rho > 0
        if np.any(nonzero):
            kappa[nonzero] = (1 / rho[nonzero]) - 1

        # Numerical guard: cap huge kappa values for SciPy numerical stability (can't handle ∞)
        kappa = np.minimum(kappa, 1e6)
        return kappa

    def compute_nll(self, expected, actual, **kwargs):
        """
        Beta-binomial or Binomial negative log-likelihood.
        
        If kappa values are provided (or derived from CIs), uses Beta-binomial distribution.
        Otherwise, falls back to Binomial distribution with warning.
        
        Args:
            expected (pd.DataFrame): Real observed data (not used directly, uses self.expected)
            actual (pd.DataFrame): Simulated data with 'n' and 'x' columns  
            kwargs: Additional evaluation arguments
            
        Returns:
            ndarray: Negative log likelihoods for each data point
        """
        import warnings
        
        logLs = []
        combined = pd.merge(self.expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        eps = 1e-10
        
        # Issue warning once if using fallback
        if not self.use_overdispersion:
            warnings.warn("No kappa provided; using Binomial (no overdispersion).", UserWarning)
        
        for _, rep in combined.iterrows():
            # Real observed data  
            e_n, e_x = rep['n_e'], rep['x_e']
            
            # Simulated data (extract prevalence with Bayesian adjustment)
            a_n, a_x = rep['n_a'], rep['x_a']
            
            # Compute prevalence q from simulation with prior
            q = (a_x + self.alpha_prior) / (a_n + self.alpha_prior + self.beta_prior)
            
            # Numerical guard: ensure q ∈ (ε, 1-ε) for numerical stability
            q = np.clip(q, eps, 1 - eps)
            
            if self.use_overdispersion:
                # Beta-binomial path: use provided/calculated kappa
                e_kappa = rep['kappa']
                alpha = e_kappa * q
                beta = e_kappa * (1 - q)
                
                # Numerical guard: ensure α, β > ε for valid beta-binomial
                alpha = max(alpha, eps)
                beta = max(beta, eps)
                
                logL = sps.betabinom.logpmf(k=e_x, n=e_n, a=alpha, b=beta)
            else:
                # Binomial fallback: exact base distribution
                logL = sps.binom.logpmf(k=e_x, n=e_n, p=q)
                
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        """Plot beta-binomial or binomial likelihood facet."""
        t = data.iloc[0]['t']
        expected_t = self.expected.loc[t]
        e_n, e_x = expected_t['n'], expected_t['x']
        kk = np.arange(0, int(2*e_x))
        
        for idx, row in data.iterrows():
            # Compute prevalence from simulation with prior
            a_n, a_x = row['n'], row['x']
            q = (a_x + self.alpha_prior) / (a_n + self.alpha_prior + self.beta_prior)
            
            # Numerical guard: ensure q ∈ (ε, 1-ε) for numerical stability
            eps = 1e-10
            q = np.clip(q, eps, 1 - eps)
            
            if self.use_overdispersion:
                # Beta-binomial path
                e_kappa = expected_t['kappa']
                alpha = e_kappa * q
                beta = e_kappa * (1 - q)
                
                # Numerical guard: ensure α, β > ε for valid beta-binomial
                alpha = max(alpha, eps)
                beta = max(beta, eps)
                
                dist = sps.betabinom(n=e_n, a=alpha, b=beta)
            else:
                # Binomial path
                dist = sps.binom(n=e_n, p=q)
            
            yy = dist.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy_obs = dist.pmf(e_x)
            plt.plot(e_x, yy_obs, 'x', ms=10, color='k')
            
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        """Plot bootstrap distribution using beta-binomial or binomial."""
        t = data.iloc[0]['t']
        expected_t = self.expected.loc[t]
        e_n, e_x = expected_t['n'], expected_t['x']

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)
            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby('t').aggregate(func=self.combine_reps, **self.combine_kwargs)

            for _, row in actual.iterrows():
                # Compute prevalence from simulation with prior
                a_n, a_x = row['n'], row['x']  
                q = (a_x + self.alpha_prior) / (a_n + self.alpha_prior + self.beta_prior)
                
                # Numerical guard: ensure q ∈ (ε, 1-ε) for numerical stability
                eps = 1e-10
                q = np.clip(q, eps, 1 - eps)
                
                if self.use_overdispersion:
                    # Beta-binomial path
                    e_kappa = expected_t['kappa']
                    alpha = e_kappa * q
                    beta = e_kappa * (1 - q)
                    
                    # Numerical guard: ensure α, β > ε for valid beta-binomial
                    alpha = max(alpha, eps)
                    beta = max(beta, eps)
                    
                    dist = sps.betabinom(n=e_n, a=alpha, b=beta)
                else:
                    # Binomial path
                    dist = sps.binom(n=e_n, p=q)
                    
                means[bi] = dist.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        ax.axvline(e_x, color='k', linestyle='--')
        return


class DirichletMultinomial(CalibComponent):
    def __init__(self, *args, alpha_smooth=1e-3, **kwargs):
        """
        Dirichlet-Multinomial likelihood with user-specified concentration kappa.
        Uses simulation data only to estimate category probabilities, with user-controlled
        dispersion parameter kappa stored in the expected DataFrame.
        
        The expected DataFrame must contain:
        - x_0, x_1, ..., x_k: Integer counts for each category (>=0)
        - kappa: Concentration parameter per survey (>0)
        - Standard time columns (t, t1 if applicable)
        
        Parameters:
        -----------
        alpha_smooth : float, default 1e-3
            Small smoothing parameter added to each category before normalizing
            to prevent degenerate probabilities on the simplex.
        """
        from pandas.api.types import is_integer_dtype, is_numeric_dtype
        super().__init__(*args, **kwargs)
        
        # Find x_vars columns and sort for consistent ordering
        x_vars = sorted([col for col in self.expected.columns if col.startswith('x_')])
        if not x_vars:
            raise ValueError("expected DataFrame must contain columns starting with 'x_' for category counts")
        
        # Validate x_vars are integer counts >= 0
        for xvar in x_vars:
            if not is_integer_dtype(self.expected[xvar]) or (self.expected[xvar] < 0).any():
                raise ValueError(f'expected["{xvar}"] must be integer counts (>=0)')
        
        # Check for dispersion parameters (kappa)
        has_kappa = 'kappa' in self.expected.columns
        
        # Store dispersion mode flag
        self.use_overdispersion = has_kappa
        
        # Process expected DataFrame
        self.expected = self.expected.copy()
        
        if has_kappa:
            # Validate all kappa values are positive
            if not is_numeric_dtype(self.expected['kappa']) or (self.expected['kappa'] <= 0).any():
                raise ValueError("All kappa values must be positive numbers")
            # Cap kappa for numerical stability
            self.expected['kappa'] = np.minimum(self.expected['kappa'], 1e9)
        # If kappa not provided, will use Multinomial fallback
        
        # Store consistent x_vars ordering and parameters
        self.x_vars = x_vars
        self.alpha_smooth = alpha_smooth
        self._eps = 1e-10

    def compute_nll(self, expected, actual, **kwargs):
        """
        Dirichlet-Multinomial or Multinomial negative log-likelihood.
        
        If kappa values are provided, uses Dirichlet-Multinomial distribution.
        Otherwise, falls back to Multinomial distribution with warning.
        
        Args:
            expected (pd.DataFrame): Real observed data (not used directly, uses self.expected)
            actual (pd.DataFrame): Simulated data with x_vars columns
            kwargs: Additional evaluation arguments
            
        Returns:
            ndarray: Negative log likelihoods for each data point
        """
        import warnings
        
        logLs = []
        
        # Issue warning once if using fallback
        if not self.use_overdispersion:
            warnings.warn("No kappa provided; using Multinomial (no overdispersion).", UserWarning)
        
        # Use consistent x_vars ordering from __init__
        x_vars = self.x_vars
        K = len(x_vars)  # Number of categories
        
        # Group by time and compute likelihood for each survey
        for t_idx in expected.index:
            expected_t = expected.loc[[t_idx]]
            
            # Get the time value for filtering actual data
            if isinstance(t_idx, tuple):
                # Multi-index case (t, t1)
                t_val = t_idx[0]  # Use first level (t) for matching
                actual_t = actual[actual['t'] == t_val] if 't' in actual.columns else actual[actual.index == t_val]
            else:
                # Single index case
                t_val = t_idx
                actual_t = actual[actual.index == t_val] if actual.index.name == 't' else actual[actual['t'] == t_val]
            
            # Extract expected data for this timepoint
            e_x = expected_t[x_vars].iloc[0].to_numpy(dtype=int)  # observed counts by category
            n = e_x.sum()  # total observed count
            
            # Process each simulation replicate for this timepoint
            for _, rep in actual_t.iterrows():
                # Sim counts -> probabilities with smoothing on simplex
                c_sim = rep[x_vars].to_numpy(dtype=float)
                total_sim = c_sim.sum() + K * self.alpha_smooth
                
                if total_sim <= self._eps:
                    # Degenerate case: uniform probabilities
                    pi = np.full(K, 1.0 / K, dtype=float)
                else:
                    # Smoothed probabilities on simplex
                    pi = (c_sim + self.alpha_smooth) / total_sim
                
                if self.use_overdispersion:
                    # Dirichlet-Multinomial path: use provided kappa
                    kappa = expected_t['kappa'].iloc[0]
                    alpha = kappa * pi
                    
                    # Ensure alpha values are valid (> eps)
                    alpha = np.maximum(alpha, self._eps)
                    
                    logL = sps.dirichlet_multinomial.logpmf(x=e_x, n=n, alpha=alpha)
                else:
                    # Multinomial fallback: exact base distribution
                    # Ensure probabilities are in valid range
                    pi_clipped = np.clip(pi, self._eps, 1 - self._eps)
                    pi_clipped = pi_clipped / pi_clipped.sum()  # Renormalize
                    
                    logL = sps.multinomial.logpmf(x=e_x, n=n, p=pi_clipped)
                    
                logLs.append(logL)
        
        return -np.nan_to_num(np.array(logLs), nan=-np.inf)

    def plot(self, actual=None, **kwargs):
        if actual is None:
            actual = self.actual

        if 'calibrated' not in actual.columns:
            actual['calibrated'] = 'Calibration'

        x_vars = self.x_vars
        actual = actual \
            .reset_index() \
            [['t', 'calibrated', 'rand_seed']+x_vars] \
            .melt(id_vars=['t', 'calibrated', 'rand_seed'], var_name='var', value_name='x')
        g = sns.FacetGrid(data=actual, col='t', row='var', hue='calibrated', sharex=False, height=2, aspect=1.7, **kwargs)
        g.map_dataframe(self.plot_facet, full_actual=actual)
        g.set_titles(row_template='{row_name}')
        for (row_val, col_val), ax in g.axes_dict.items():
            if row_val == g.row_names[0] and isinstance(col_val, dt.datetime):
                ax.set_title(col_val.strftime('%Y-%m-%d'))

        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(self.name)
        return g.fig

    def plot_facet(self, data, color, **kwargs):
        """Plot marginal beta-binomial approximation for each category"""
        t = data.iloc[0]['t']
        var = data.iloc[0]['var']
        cal = data.iloc[0]['calibrated']

        expected = self.expected.loc[[t]]
        actual = kwargs.get('full_actual', self.actual)
        actual = actual[(actual['t'] == t) & (actual['calibrated'] == cal)]

        # Use consistent x_vars ordering
        x_vars = self.x_vars
        K = len(x_vars)
        e_x = expected[var].iloc[0]  # observed count for this category
        e_n = expected[x_vars].sum(axis=1).iloc[0]  # total observed count  
        kappa = expected['kappa'].iloc[0]  # concentration parameter

        kk = np.arange(0, int(2*e_x + 10))
        for seed, row in actual.groupby('rand_seed'):
            # Get simulation data for this replicate
            a = row.set_index('var')['x']
            c_sim = a[x_vars].to_numpy(dtype=float)
            
            # Apply simplex smoothing
            total_sim = c_sim.sum() + K * self.alpha_smooth
            if total_sim <= self._eps:
                pi = np.full(K, 1.0 / K, dtype=float)
            else:
                pi = (c_sim + self.alpha_smooth) / total_sim
            
            # Get probability for this specific category
            var_idx = x_vars.index(var)
            p_cat = pi[var_idx]
            
            # Beta-binomial approximation using kappa and category probability
            alpha = kappa * p_cat
            beta = kappa * (1 - p_cat)
            
            # Ensure valid parameters
            alpha = max(alpha, self._eps)
            beta = max(beta, self._eps)
            
            q = sps.betabinom(n=e_n, a=alpha, b=beta)
            yy = q.pmf(kk)
            plt.step(kk, yy, color=color, alpha=0.7, label=f"{seed}")
            yy_obs = q.pmf(e_x)
            plt.plot(e_x, yy_obs, 'x', ms=8, color='black')
            
        plt.axvline(e_x, color='k', linestyle='--', alpha=0.8)
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        """Plot bootstrap distribution using marginal beta-binomial approximation"""
        t = data.iloc[0]['t']
        var = data.iloc[0]['var']
        
        expected = self.expected.loc[[t]]
        x_vars = self.x_vars
        K = len(x_vars)
        e_x = expected[var].iloc[0]
        e_n = expected[x_vars].sum(axis=1).iloc[0]
        kappa = expected['kappa'].iloc[0]
        
        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)
            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby(['t']).aggregate(func=self.combine_reps, **self.combine_kwargs)
                
            # Calculate bootstrap mean
            total_mean = 0
            n_reps = 0
            for _, row in actual.iterrows():
                # Extract simulation counts for all categories  
                c_sim = np.array([row[xv] for xv in x_vars], dtype=float)
                
                # Apply simplex smoothing
                total_sim = c_sim.sum() + K * self.alpha_smooth
                if total_sim <= self._eps:
                    pi = np.full(K, 1.0 / K, dtype=float)
                else:
                    pi = (c_sim + self.alpha_smooth) / total_sim
                
                # Get probability for this category and create beta-binomial
                var_idx = x_vars.index(var)
                p_cat = pi[var_idx]
                alpha = max(kappa * p_cat, self._eps)
                beta = max(kappa * (1 - p_cat), self._eps)
                
                q = sps.betabinom(n=e_n, a=alpha, b=beta)
                total_mean += q.mean()
                n_reps += 1
                
            means[bi] = total_mean / n_reps if n_reps > 0 else 0
        
        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        plt.axvline(e_x, color='k', linestyle='--')
        return

class GammaPoisson(CalibComponent):
    def __init__(self, *args, rate_smoothing=(1e-6, 1e-6), **kwargs):
        """
        Gamma-Poisson (NB2) likelihood with user-specified dispersion k_size.

        Uses simulation data only to estimate the mean rate, with user-controlled
        dispersion parameter k_size stored in the expected DataFrame.

        Parameters:
        -----------
        rate_smoothing : tuple of (c0, e0), default (1e-6, 1e-6)
            Small values added to simulated counts (c0) and exposure (e0) 
            to prevent division by zero when computing rates.
            rate = (sim_counts + c0) / (sim_exposure + e0)
        """
        from pandas.api.types import is_integer_dtype, is_numeric_dtype
        super().__init__(*args, **kwargs)
        
        # Validate rate_smoothing parameter
        if not isinstance(rate_smoothing, (tuple, list)) or len(rate_smoothing) != 2:
            raise ValueError("rate_smoothing must be a tuple/list of length 2: (c0, e0)")
        
        # Validation: n numeric (not integer), x integer
        if not is_numeric_dtype(self.expected['n']) or (self.expected['n'] <= 0).any():
            raise ValueError('expected["n"] must be numeric exposure (>0)')
        if not is_integer_dtype(self.expected['x']) or (self.expected['x'] < 0).any():
            raise ValueError('expected["x"] must be integer counts (>=0)')
        
        # Check for dispersion parameters (k_size)
        has_k_size = 'k_size' in self.expected.columns
        
        # Store dispersion mode flag
        self.use_overdispersion = has_k_size
        
        # Process expected DataFrame
        self.expected = self.expected.copy()
        
        if has_k_size:
            # Validate all k_size values are positive
            if not all(self.expected['k_size'] > 0):
                raise ValueError("All k_size values must be positive")
            # Cap k_size values for numerical stability
            self.expected['k_size'] = np.minimum(self.expected['k_size'], 1e6)
        # If k_size not provided, will use Poisson fallback
        
        # Rate smoothing parameters (renamed for clarity)
        self.c0, self.e0 = rate_smoothing
        self._eps = 1e-10

    def compute_nll(self, expected, actual, **kwargs):
        """
        Negative log-likelihood under NB2 or Poisson.
        
        If k_size values are provided, uses NB2 distribution.
        Otherwise, falls back to Poisson distribution with warning.
        
        Args:
            expected (pd.DataFrame): Real observed data (not used directly, uses self.expected)
            actual (pd.DataFrame): Simulated data with 'n' and 'x' columns
            kwargs: Additional evaluation arguments
            
        Returns:
            ndarray: Negative log likelihoods for each data point
        """
        import warnings
        
        # Merge expected/actual on ['t','t1'] 
        combined = pd.merge(self.expected.reset_index(), actual.reset_index(), 
                           on=['t', 't1'], suffixes=('_e', '_a'), 
                           how='inner')
        
        # Extract arrays (vectorized)
        c_obs = combined['x_e'].to_numpy(dtype=int)        # observed counts
        e_obs = combined['n_e'].to_numpy(dtype=float)      # observed exposure  
        c_sim = combined['x_a'].to_numpy(dtype=float)      # simulated counts
        e_sim = combined['n_a'].to_numpy(dtype=float)      # simulated exposure
        
        # Calculate rate from simulation with smoothing (guard against divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_sim = (c_sim + self.c0) / np.maximum(e_sim + self.e0, self._eps)
        rate_sim = np.maximum(rate_sim, 0.0)  # Ensure non-negative
        
        # Compute mean: μ = n_obs × rate_sim
        mu = np.maximum(self._eps, e_obs * rate_sim)  # Clip mu >= eps
        
        # Issue warning once if using fallback
        if not self.use_overdispersion:
            warnings.warn("No k_size provided; using Poisson (no overdispersion).", UserWarning)
        
        if self.use_overdispersion:
            # NB2 path: use provided k_size
            k_size = combined['k_size'].to_numpy(dtype=float)  # per-timepoint k_size
            p = np.clip(k_size / (k_size + mu), self._eps, 1 - self._eps)
            logL = sps.nbinom.logpmf(k=c_obs, n=k_size, p=p)
        else:
            # Poisson fallback: exact base distribution  
            logL = sps.poisson.logpmf(k=c_obs, mu=mu)
        
        return -np.nan_to_num(logL, nan=-np.inf)  # Return NLL per row

    def plot_facet(self, data, color, **kwargs):
        """Plot NB2 or Poisson likelihood distributions for each simulation replicate"""
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_n, e_x = expected['n'].iloc[0], expected['x'].iloc[0]
        
        kk = np.arange(0, int(2*e_x))
        for idx, row in data.iterrows():
            c_sim, e_sim = row['x'], row['n']
            
            # Calculate rate and mean
            rate_sim = (c_sim + self.c0) / max(e_sim + self.e0, self._eps)
            mu = max(self._eps, e_n * rate_sim)
            
            if self.use_overdispersion:
                # NB2 path
                k_size = expected['k_size'].iloc[0]
                p = np.clip(k_size / (k_size + mu), self._eps, 1 - self._eps)
                dist = sps.nbinom(n=k_size, p=p)
            else:
                # Poisson path
                dist = sps.poisson(mu=mu)
            
            yy = dist.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy_obs = dist.pmf(e_x)
            plt.plot(e_x, yy_obs, 'x', ms=10, color='k')
        
        plt.axvline(e_x, color='k', linestyle='--')

    def plot_facet_bootstrap(self, data, color, **kwargs):
        """Plot bootstrap distribution of means using NB2 or Poisson"""
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_n, e_x = expected['n'].iloc[0], expected['x'].iloc[0]

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)

            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby(['t', 't1']).aggregate(func=self.combine_reps, **self.combine_kwargs)

            # Calculate bootstrap mean
            total_mean = 0
            n_reps = 0
            for idx, row in actual.iterrows():
                c_sim, e_sim = row['x'], row['n']
                rate_sim = (c_sim + self.c0) / max(e_sim + self.e0, self._eps)
                mu = max(self._eps, e_n * rate_sim)
                
                if self.use_overdispersion:
                    # NB2 path
                    k_size = expected['k_size'].iloc[0]
                    p = np.clip(k_size / (k_size + mu), self._eps, 1 - self._eps)
                    dist = sps.nbinom(n=k_size, p=p)
                else:
                    # Poisson path
                    dist = sps.poisson(mu=mu)
                    
                total_mean += dist.mean()
                n_reps += 1
            
            means[bi] = total_mean / max(n_reps, 1)

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        plt.axvline(e_x, color='k', linestyle='--')


class Normal(CalibComponent):
    def __init__(self, name, expected, extract_fn, conform, weight=1, sigma2=None, **kwargs):
        super().__init__(name, expected, extract_fn, conform, weight, **kwargs)
        self.sigma2 = sigma2
        return

    def compute_nll(self, expected, actual, **kwargs):
        """
        Normal log-likelihood component.

        Note that minimizing the negative log likelihood of a Gaussian likelihood is
        equivalent to minimizing the Euclidean distance between the expected and
        actual values.

        User-provided variance can be passed as a keyword argument named sigma2 when creating this component.

        Args:
            expected (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from the reference dataset.
            predicted (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from simulated dataset.
            kwargs (dict): contains any eval_kwargs that were specified when instantiating the Calibration

        Returns:
            nll (float): negative Euclidean distance between expected and predicted values.
        """

        logLs = []
        sigma2 = self.sigma2
        compute_var = sigma2 is None

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        for idx, rep in combined.iterrows():
            e_x = rep['x_e']
            a_x = rep['x_a']

            # TEMP TODO calculate rate if 'n' supplied
            if 'n' in rep:
                a_x = rep['x_a'] / rep['n']

            if compute_var:
                sigma2 = self.compute_var(expected['x'], a_x)

            logL = sps.norm.logpdf(x=e_x, loc=a_x, scale=np.sqrt(sigma2))
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def compute_var(self, expected_x, actual_x):
        """
        Compute the maximum-likelihood variance of the residuals between expected and actual values.
        """
        diffs = expected_x - actual_x
        SSE = np.sum(diffs**2)
        N = len(expected_x) if sc.isiterable(expected_x) else 1
        sigma2 = SSE/N
        return sigma2

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_x = expected['x'].values.flatten()[0]
        nll = 0
        for idx, row in data.iterrows():
            a_x = row['x']

            # TEMP TODO calculate rate if 'n' supplied
            if 'n' in row:
                a_x = row['x'] / row['n'] # row[['x']].values[0] / row[['n']].values[0]

            sigma2 = self.sigma2 if self.sigma2 is not None else self.compute_var(e_x, a_x)
            if isinstance(sigma2, (list, np.ndarray)):
                assert len(sigma2) == len(self.expected), 'Length of sigma2 must match the number of timepoints'
                # User provided a vector of variances
                ti = self.expected.index.get_loc(t)
                sigma2 = sigma2[ti]

            sigma = np.sqrt(sigma2)
            kk = np.linspace(a_x - 1.96*sigma, a_x + 1.96*sigma, 1000)
            q = sps.norm(loc=a_x, scale=sigma)

            yy = q.pdf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pdf(e_x)
            nll += -q.logpdf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]] # Gracefully handle Series and DataFrame, if 't1' in index
        e_x = expected['x'].values.flatten()[0] # Due to possible presence of 't1' in the index

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)

            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby('t').aggregate(func=self.combine_reps, **self.combine_kwargs)

            for idx, row in actual.iterrows():
                a_x = row['x']

                # TEMP TODO calculate rate if 'n' supplied
                if 'n' in row:
                    a_x = row['x'] / row['n'] #row['x'].values / row['n'].values

                # No need to form the sps.norm involving sigma2 because the mean will be a_x
                means[bi] = a_x

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        plt.axvline(e_x, color='k', linestyle='--')
        return
