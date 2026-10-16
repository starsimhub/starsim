"""
Define the calibration class
"""
import os
import tempfile
import importlib.util
import functools
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

# Defaults
default_n_boot = 1000 # Default number of bootstrap samples for plot_facet_bootstrap()



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
        db_name      (str)   : the name of the database file (default: temporary file)
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
        use_defaults = study_name is None and db_name is None
        if total_trials is None: total_trials   = 100
        if n_workers    is None: n_workers      = 1 if debug else sc.cpu_count()
        if study_name   is None: study_name     = 'starsim_calibration'
        if db_name      is None: db_name        = sc.path(tempfile.mkdtemp()) / f'{study_name}.db' if use_defaults else f'{study_name}.db'
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
        self.best_pars = None

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
        """
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
        if sim is None: # The sim failed and die=False, so give it the worst possible fit
            return np.inf

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
        for name, spec in before_pars.items():
            if 'guess' not in spec:
                raise KeyError(f"Calibration parameter '{name}' has no 'guess' key; ensure all calib_pars have a 'guess' entry")
            spec['value'] = spec['guess'] # Use guess values

        # Load in case calibration was interrupted
        if self.best_pars is None:
            try:
                study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
                self.best_pars = sc.objdict(study.best_params)
            except Exception as e:
                errormsg = f'Seems like calibration did not finish successfully and also unable to obtain best parameters from the {self.run_args.storage}:{self.run_args.study_name} as the study was likely automatically deleted, see keep_db. Exception:\n{e}'
                raise ValueError(errormsg)

        after_pars = sc.dcp(self.calib_pars)
        for parname, spec in after_pars.items():
            spec['value'] = self.best_pars[parname] # Use best parameters from calibration

        self.before_msim = self.build_fn(sc.dcp(self.sim), calib_pars=before_pars, **self.build_kw)
        self.after_msim = self.build_fn(sc.dcp(self.sim), calib_pars=after_pars, **self.build_kw)

        fix_before = isinstance(self.before_msim, ss.Sim)
        fix_after = isinstance(self.after_msim, ss.Sim)
        if fix_before or fix_after:
            if fix_before:
                self.before_msim = ss.MultiSim(self.before_msim, initialize=True, debug=True, parallel=False, n_runs=1)

            if fix_after:
                self.after_msim = ss.MultiSim(self.after_msim, initialize=True, debug=True, parallel=False, n_runs=1)

        for msim in [self.before_msim, self.after_msim]:
            if msim.sims is None: # The build_fn returned a MultiSim that hasn't been initialized
                msim.init_sims(parallel=False)

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
            data = {'index':trial.number, 'mismatch': trial.value}  # 'mismatch' stores the negative log-likelihood
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
    inds = np.maximum(np.searchsorted(actual.index, t, side='right') - 1, 0) # Last step at or before t (or the first step, if t is before it)
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
            errormsg = f"The conform argument must be a string or a callable function, not {type(conform)}."
            raise TypeError(errormsg)
        elif isinstance(conform, str):
            conform_ = self.avail_conforms.get(conform.lower(), 'NOT FOUND')
            if conform_ == 'NOT FOUND':
                avail = self.avail_conforms.keys()
                raise ValueError(f"The conform argument must be one of {avail}, not {conform}.")
        else:
            conform_ = conform
        return conform_

    def _combine_reps_nll(self, expected, actual, **kwargs):
        """ Aggregate replicates (if combine_reps is set) and compute negative log likelihood """
        if self.combine_reps is None:
            nll = self.compute_nll(expected, actual, **kwargs) # Negative log likelihood
        else:
            timecols = list(expected.index.names) # e.g. ['t'] or ['t', 't1']
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
            use_seeds = np.random.choice(seeds, boot_size, replace=True)  # intentionally unseeded: bootstrap is a statistical tool, not a sim step
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
    """ Beta-binomial negative log-likelihood component for count data with overdispersion """

    def compute_nll(self, expected, actual, **kwargs):
        """
        For the beta-binomial negative log-likelihood, we begin with a Beta(1,1) prior
        and subsequently observe actual['x'] successes (positives) in actual['n'] trials (total observations).
        The result is a Beta(actual['x']+1, actual['n']-actual['x']+1) posterior.
        We then compare this to the real data, which has expected['x'] successes (positives) in expected['n'] trials (total observations).
        To do so, we use a beta-binomial likelihood:
        p(x|n, a, b) = (n choose x) B(x+a, n-x+b) / B(a, b)
        where
          x=expected['x']
          n=expected['n']
          a=actual['x']+1
          b=actual['n']-actual['x']+1
        and B is the beta function, B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)

        We return the log of p(x|n, a, b)

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """

        logLs = []

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        for idx, rep in combined.iterrows():
            e_n, e_x = rep['n_e'], rep['x_e']
            a_n, a_x = rep['n_a'], rep['x_a']

            logL = sps.betabinom.logpmf(k=e_x, n=e_n, a=a_x+1, b=a_n-a_x+1)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']
        kk = np.arange(0, int(2*e_x))
        for idx, row in data.iterrows():
            alpha = row['x'] + 1
            beta = row['n'] - row['x'] + 1
            q = sps.betabinom(n=e_n, a=alpha, b=beta)
            yy = q.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pmf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']

        n_boot = kwargs.get('n_boot', default_n_boot)
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
                alpha = row['x'] + 1
                beta = row['n'] - row['x'] + 1
                q = sps.betabinom(n=e_n, a=alpha, b=beta)
                means[bi] = q.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        ax.axvline(e_x, color='k', linestyle='--')
        return

class Binomial(CalibComponent):
    """ Binomial negative log-likelihood component for count data with a known number of trials """

    @staticmethod
    def get_p(df, x_col='x', n_col='n'):
        if 'p' in df:
            p = df['p']
        else:
            p = df[x_col] / df[n_col] # Switched to MLE x/n from previous "Bayesian" (Laplace +1, Jeffreys) before: (df[x_col]+1) / (df[n_col]+2)
        return p

    def compute_nll(self, expected, actual, **kwargs):
        """
        Binomial log likelihood component.
        We return the log of p(x|n, p)

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """

        logLs = []

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        for idx, rep in combined.iterrows():
            if 'p' in rep:
                # p specified, no collision
                e_n, e_x = rep['n'], rep['x']
                p = self.get_p(rep)
            else:
                assert 'n_e' in rep and 'x_e' in rep, 'Expected columns n_e and x_e not found'
                # Collision in merge, get _e and _a values
                e_n, e_x = rep['n_e'], rep['x_e']
                if rep['n_a'] == 0:
                    return np.inf
                p = self.get_p(rep, 'x_a', 'n_a')

            logL = sps.binom.logpmf(k=e_x, n=e_n, p=p)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']
        kk = np.arange(0, int(2*e_x))
        for idx, row in data.iterrows():
            p = self.get_p(row)
            q = sps.binom(n=e_n, p=p)
            yy = q.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pmf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']

        n_boot = kwargs.get('n_boot', default_n_boot)
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
                p = self.get_p(row)
                q = sps.binom(n=e_n, p=p)
                means[bi] = q.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        ax.axvline(e_x, color='k', linestyle='--')
        return

class DirichletMultinomial(CalibComponent):
    """ Dirichlet-multinomial negative log-likelihood component for compositional count data """

    def compute_nll(self, expected, actual, **kwargs):
        """
        The Dirichlet-multinomial negative log-likelihood is the
        multi-dimensional analog of the beta-binomial likelihood. We begin with
        a Dirichlet(1,1,...,1) prior and subsequently observe:
            actual['x1'], actual['x2'], ..., actual['xk']
        successes (positives). The result is a
            Dirichlet(alpha=actual[x_vars]+1)
        We then compare this to the real data, which has outcomes:
            expected['x1'], expected['x2'], ..., expected['xk']

        The count variables are any keys in the expected dataframe that start with 'x'.

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """

        logLs = []
        x_vars = [xkey for xkey in expected.columns if xkey.startswith('x')]
        for t, rep_t in actual.groupby('t'):
            for idx, rep in rep_t.iterrows():
                e_x = expected.loc[t, x_vars].values.flatten()
                n = e_x.sum()
                a_x = rep[x_vars].astype('float')
                logL = sps.dirichlet_multinomial.logpmf(x=e_x, n=n, alpha=a_x+1)
                logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot(self, actual=None, **kwargs):
        if actual is None:
            actual = self.actual

        if 'calibrated' not in actual.columns:
            actual['calibrated'] = 'Calibration'

        x_vars = [xkey for xkey in self.expected.columns if xkey.startswith('x')]
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
        # It's challenging to plot the Dirichlet-multinomial likelihood, so we use Beta binomial as a stand-in
        t = data.iloc[0]['t']
        var = data.iloc[0]['var']
        cal = data.iloc[0]['calibrated']

        expected = self.expected.loc[[t]]
        actual = kwargs.get('full_actual', self.actual)
        actual = actual[(actual['t'] == t) & (actual['calibrated'] == cal)]

        x_vars = [xkey for xkey in expected.columns if xkey.startswith('x')]

        e_x = expected[var].values.flatten()[0]
        e_n = expected[x_vars].sum(axis=1)

        kk = np.arange(0, int(2*e_x))
        for seed, row in actual.groupby('rand_seed'):
            a = row.set_index('var')['x']
            a_x = a[var]
            a_n = a[x_vars].sum()
            alpha = a_x + 1
            beta = a_n - a_x + 1
            q = sps.betabinom(n=e_n, a=alpha, b=beta)
            yy = q.pmf(kk)
            plt.step(kk, yy, color=color, label=f"{seed}")
            yy = q.pmf(e_x)
            darker = [0.8*c for c in color]
            plt.plot(e_x, yy, 'x', ms=10, color=darker)
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        return

class GammaPoisson(CalibComponent):
    """ Gamma-Poisson (negative binomial) negative log-likelihood component for overdispersed count data """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.expected['n'].dtype == int, 'The expected must have an integer column named "n" for the total number of person-years'
        assert self.expected['x'].dtype == int, 'The expected must have an integer column named "x" for the total number of events'
        return

    def compute_nll(self, expected, actual, **kwargs):
        """
        The gamma-poisson likelihood is a Poisson likelihood with a
        gamma-distributed rate parameter. Through a parameter transformation, we
        end up calling a negative binomial, which is functionally equivalent.

        For the gamma-poisson negative log-likelihood, we begin with a
        gamma(1,1) and subsequently observe actual['x'] events (positives) in
        actual['n'] trials (total person years). The result is a
        gamma(alpha=1+actual['x'], beta=1+actual['n']) posterior.

        To evaluate the gamma-poisson likelihood as a negative binomial, we use
        the following: n = alpha, p = beta / (beta + 1)

        However, the gamma is estimating a rate, and the observation is a count over a some period of time, say T person-years. To account for T other than 1, we scale beta by 1/T to arrive at beta' = beta / T. When passing into the negative binomial, alpha remains the same, but p = beta' / (beta' + 1) or equivalently p = beta / (beta + T).

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """
        logLs = []

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t', 't1'], suffixes=('_e', '_a'))
        for idx, rep in combined.iterrows():
            e_n, e_x = rep['n_e'], rep['x_e']
            a_n, a_x = rep['n_a'], rep['x_a']
            T = e_n
            beta = 1 + a_n
            logL = sps.nbinom.logpmf(k=e_x, n=1+a_x, p=beta/(beta+T))
            logL = np.nan_to_num(logL, nan=-np.inf)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_n, e_x = expected['n'].values.flatten()[0], expected['x'].values.flatten()[0]
        kk = np.arange(0, int(2*e_x))
        nll = 0
        for idx, row in data.iterrows():
            a_n, a_x = row['n'], row['x']
            beta = (a_n+1)
            T = e_n
            q = sps.nbinom(n=1+a_x, p=beta/(beta+T))

            yy = q.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pmf(e_x)
            nll += -q.logpmf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_n, e_x = expected['n'].values.flatten()[0], expected['x'].values.flatten()[0]

        n_boot = kwargs.get('n_boot', default_n_boot)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)

            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby(['t', 't1']).aggregate(func=self.combine_reps, **self.combine_kwargs)

            for idx, row in actual.iterrows():
                a_n, a_x = row['n'], row['x']
                beta = (a_n+1)
                T = e_n
                q = sps.nbinom(n=1+a_x, p=beta/(beta+T))
                means[bi] = q.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        plt.axvline(e_x, color='k', linestyle='--')
        return


class Normal(CalibComponent):
    """ Normal (Gaussian) negative log-likelihood component for continuous data """

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

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        if sigma2 is None: # Use the maximum-likelihood variance of all the residuals
            a_x = combined['x_a'] / combined['n'] if 'n' in combined else combined['x_a']
            sigma2 = max(self.compute_var(combined['x_e'], a_x), np.finfo(float).eps) # Avoid zero variance (and hence a NaN likelihood) for a perfect fit

        for idx, rep in combined.iterrows():
            e_x = rep['x_e']
            a_x = rep['x_a']

            # TODO: calculate rate if 'n' supplied
            if 'n' in rep:
                a_x = rep['x_a'] / rep['n']

            this_sigma2 = sigma2
            if isinstance(sigma2, (list, np.ndarray)): # User provided a vector of variances, one per timepoint
                this_sigma2 = sigma2[expected.index.get_loc(rep['t'])]

            logL = sps.norm.logpdf(x=e_x, loc=a_x, scale=np.sqrt(this_sigma2))
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

            # TODO: calculate rate if 'n' supplied
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

        n_boot = kwargs.get('n_boot', default_n_boot)
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

                # TODO: calculate rate if 'n' supplied
                if 'n' in row:
                    a_x = row['x'] / row['n'] #row['x'].values / row['n'].values

                # No need to form the sps.norm involving sigma2 because the mean will be a_x
                means[bi] = a_x

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        plt.axvline(e_x, color='k', linestyle='--')
        return


#%% Data and fitting for models with declared parameters

class data:
    """
    Data to compare with a sim's results, matched by name: each column is named like the result it corresponds to

    Args:
        source (str/df): a file name or a dataframe, with a time column (`date`, `year`, `t`, or `time`) and one column per data series
        aliases (dict): for headers that do not match: the result (or time-column) name, and the column that holds it
        ignore_extra (bool): if True, drop columns that match no result instead of raising an error

    Examples:
        ```python
        data = ss.data('kenya_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
        sim = ss.Sim(sir, data=data, start='2020-01-01', stop='2020-12-31')
        ```
    """
    def __init__(self, source, aliases=None, ignore_extra=False):
        self.source = source
        self.aliases = aliases or {}
        self.ignore_extra = ignore_extra
        return

    def load(self, sim=None):
        """ Load the data and rename the columns """
        df = sc.dataframe(pd.read_csv(self.source) if isinstance(self.source, (str, os.PathLike)) else self.source)
        for name,col in self.aliases.items():
            if col not in df.columns:
                errormsg = f'The alias {name}="{col}" names a column the data does not have; its columns are {list(df.columns)}'
                raise KeyError(errormsg)
            df = df.rename(columns={col:name})
        if sim is not None and self.ignore_extra:
            dims = [strat.name for strat in getattr(sim, 'strats', [])] # Columns that say which stratum a row is for
            keep = [col for col in df.columns if col in self.time_cols or col in dims or self.match(sim, col) is not None]
            df = df[keep]
        return df

    time_cols = ['t', 'timevec', 'tvec', 'time', 'day', 'date', 'year']

    @staticmethod
    def widen(sim, df):
        """
        Data in long form, with columns that say which stratum each row is for (named after dimensions of the sim, e.g. `age`), as one column for each result and stratum, e.g. `cases[age=0-15]`

        A level is given by its label (or its position).
        """
        dims = [strat.name for strat in getattr(sim, 'strats', []) if strat.name in df.columns]
        if not dims:
            return df
        parts = []
        for key,group in df.groupby(dims, sort=False):
            tag = ', '.join(f'{dim}={level}' for dim,level in zip(dims, key if isinstance(key, tuple) else [key]))
            parts.append(group.drop(columns=dims).rename(columns=lambda col: f'{col}[{tag}]'))
        return sc.dataframe(pd.concat(parts, axis=1))

    @staticmethod
    def parse(col):
        """ A column's result and stratum: "cases[age=0-15]" gives "cases" and {'age': '0-15'} """
        name, _, rest = col.partition('[')
        dims = {}
        for item in rest.rstrip(']').split(', ') if rest else []:
            dim, _, level = item.partition('=')
            dims[dim] = level
        return name, dims

    @staticmethod
    def levels(sim, dims):
        """ The levels of a stratum as `ss.Result.select()` takes them: by label, or by position if no label matches """
        out = {}
        for dim,level in dims.items():
            strat = sim.strats[sim.dim_index(dim)]
            labels = [str(label) for label in strat.labels]
            out[dim] = strat.labels[labels.index(level)] if level in labels else int(level)
        return out

    @staticmethod
    def match(sim, col):
        """ Find the result that a data column corresponds to: "cases", or "sir.cases" if the name is used by more than one module, or the sum over some groups, "cases[age=0-15]" """
        col, dims = data.parse(col)
        res = data.match_name(sim, col)
        if res is not None and dims:
            if getattr(res, 'strata', None) is None:
                return None
            res = res.select(**data.levels(sim, dims))
        return res

    @staticmethod
    def match_name(sim, col):
        """ The result a column corresponds to, by name; see `match()` """
        modname, _, name = col.replace('_', '.', 1).rpartition('.') if '.' not in col and col.split('_')[0] in sim.results else col.rpartition('.')
        matches = []
        for mod in sim.modules:
            if modname in ['', mod.name] and name in mod.results and isinstance(mod.results[name], ss.Result):
                matches.append(mod.results[name])
        if not matches and col in sim.results and isinstance(sim.results[col], ss.Result):
            matches.append(sim.results[col])
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def check(sim, df):
        """ Check that every data column matches a result, and that the data overlaps the simulated period """
        options = [f'{mod.name}.{key}' for mod in sim.modules for key,res in mod.results.items() if isinstance(res, ss.Result)]
        for col in df.columns:
            name, dims = data.parse(col)
            base = data.match_name(sim, name)
            if dims and base is not None:
                for dim,level in dims.items():
                    strat = sim.strats[sim.dim_index(dim)]
                    if level not in [str(label) for label in strat.labels] and not (level.isdigit() and int(level) < strat.n):
                        ss.error('E705', f'the data for {name} has {dim}={level}, which is not a level of "{dim}"; its levels are {sc.strjoin(strat.labels)}.')
                if getattr(base, 'strata', None) is None:
                    ss.error('E705', f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is not kept by group, so it cannot be split by them.')
            if data.match(sim, col) is None:
                ss.error('E705', f'the data column "{col}" matches no result. The closest results are {ss.transitions.closest(col, [o.split(".")[-1] for o in options])}.\n'
                                 f"     Rename it with ss.data(..., aliases=dict(result='{col}')), or drop it with ss.data(..., ignore_extra=True).")
        return


@functools.lru_cache(maxsize=64)
def date_years(dates):
    """ Dates as years, kept for the next call with the same dates (each evaluation of a fit compares with the same data) """
    return np.array([ss.date(t).years for t in dates])


def data_times(sim, df):
    """ The times of the data, on the sim's own time axis (the result timevec as numbers) """
    timevec = sim.t.timevec
    index = df.index
    dated = isinstance(index, pd.DatetimeIndex) or (len(index) and isinstance(index[0], ss.date))
    if sim.t.is_absolute or (dated and len(timevec) and isinstance(timevec[0], ss.date)): # A sim that starts in a year has dates, although its time is not absolute
        sim_t = np.asarray(sim.t.yearvec, dtype=float)
        data_t = date_years(tuple(index)).copy()
    else:
        sim_t = np.asarray(timevec, dtype=float)
        data_t = np.asarray(index, dtype=float)
    return sim_t, data_t


def compare(sim, df=None):
    """
    Line up a sim's results with its data

    A flow (new infections, reported cases) is summed over the interval ending at
    each data time; a stock or ratio (a count, a prevalence) is read at the data's time.

    Returns a dict of `{column: (data values, model values)}`.
    """
    df = sim.data if df is None else df
    sim_t, data_t = data_times(sim, df)
    if data_t.max() < sim_t.min() or data_t.min() > sim_t.max():
        ss.error('E704', f'the data ({df.index[0]} to {df.index[-1]}) lies entirely outside the simulated period ({sim.t.start} to {sim.t.stop}).')
    out = sc.objdict()
    eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9 # Times that are the same up to rounding (e.g. 84.00000015 days from 12 weeks) are the same time point
    inside = (data_t >= sim_t.min() - eps) & (data_t <= sim_t.max() + eps)
    for col in df.columns:
        res = data.match(sim, col)
        values = np.asarray(res.values, dtype=float)
        expected = np.asarray(df[col], dtype=float)
        if res.summarize_by == 'sum': # A flow: sum over the interval ending at each data time
            spacing = np.median(np.diff(data_t)) if len(data_t) > 1 else np.inf
            starts = np.append(data_t[0] - spacing, data_t[:-1])
            model = np.array([values[(sim_t > lo + eps) & (sim_t <= hi + eps)].sum() for lo,hi in zip(starts, data_t)])
        else:
            model = np.interp(data_t, sim_t, values)
        ok = inside & np.isfinite(expected)
        out[col] = (expected[ok], model[ok])
    return out


def lineup(sim, df=None):
    """
    The matrix that lines up each result that has data with its data, as `compare()` does

    Returns a dict of `{column: (W, data values, rows)}`, where the model's values at the data
    (those that are inside the simulated period and not missing, which are at these rows of the
    data) are W @ the result.
    """
    df = sim.data if df is None else df
    sim_t, data_t = data_times(sim, df)
    eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9
    inside = (data_t >= sim_t.min() - eps) & (data_t <= sim_t.max() + eps)
    npts = len(sim_t)
    out = sc.objdict()
    for col in df.columns:
        res = data.match(sim, col)
        expected = np.asarray(df[col], dtype=float)
        ok = inside & np.isfinite(expected)
        if res.summarize_by == 'sum':
            spacing = np.median(np.diff(data_t)) if len(data_t) > 1 else np.inf
            starts = np.append(data_t[0] - spacing, data_t[:-1])
            W = np.array([((sim_t > lo + eps) & (sim_t <= hi + eps)).astype(float) for lo,hi in zip(starts, data_t)])
        else:
            W = np.array([np.interp(data_t, sim_t, row) for row in np.eye(npts)]).T
        out[col] = (W[ok], expected[ok], np.flatnonzero(ok))
    return out


def mismatch(sim, df=None, weights=None, gof=None):
    """
    The goodness of fit between a sim and its data (smaller is better)

    By default, for each data series, the normalized absolute error: the mean of
    |model − data| divided by the mean of the data; the series are then averaged.

    Args:
        sim (`ss.Sim`): a sim that has been run
        df (df): the data, if not the sim's
        weights (dict): the relative weight of each data series (default: equal)
        gof (str/func): 'nae' (default), 'rmse', or a function of the sim returning a number
    """
    if callable(gof):
        return gof(sim)
    total = 0.0
    wsum = 0.0
    for col,(expected,model) in compare(sim, df).items():
        scale = np.abs(expected).mean() or np.ptp(expected) or 1.0
        diff = model - expected
        value = np.sqrt(np.mean(diff**2))/scale if gof == 'rmse' else np.mean(np.abs(diff))/scale
        weight = series_weight(weights, col)
        total += weight*value
        wsum += weight
    return total/wsum if wsum else np.nan


def series_weight(weights, col):
    """ The weight of a data series: given for its column, or for its result (which applies to every stratum of it) """
    weights = weights or {}
    return weights.get(col, weights.get(data.parse(col)[0], 1.0))


def likelihoods(sim, df=None, likelihood=None):
    """ The likelihood of each data series: given at calibration time (by column or result name), else declared on the outcome, else None """
    df = sim.data if df is None else df
    out = sc.objdict()
    for col in df.columns:
        name = data.parse(col)[0] # A stratum of a result has the likelihood of the result
        res = data.match_name(sim, name)
        given = (likelihood or {}).get(col, (likelihood or {}).get(name, (likelihood or {}).get(res.name)))
        if given is None:
            for mod in sim.declared:
                if mod.results.get(res.name) is res and res.name in mod.graph.outcomes: # By the result itself, since its module is a label
                    given = mod.graph.outcomes[res.name].likelihood
        out[col] = given
    return out


def logpdf(dist, observed, model):
    """
    The log-likelihood of each observed value, when the model's value is the mean of the distribution

    The distribution's other parameters are taken as given (a parameter that is an
    `ss.par` takes its current value). Supported: `ss.poisson`, `ss.nbinom` (with
    `n` as the dispersion), `ss.normal` (with `scale`), and `ss.lognorm_ex` (with `std`).
    """
    pars = {key:(val.raw if isinstance(val, ss.par) else val) for key,val in dist.pars.items()}
    mean = np.maximum(model, 1e-9)
    if isinstance(dist, ss.poisson):
        return sps.poisson.logpmf(observed, mean)
    elif isinstance(dist, ss.nbinom):
        n = pars['n']
        return sps.nbinom.logpmf(observed, n, n/(n + mean))
    elif isinstance(dist, ss.normal):
        return sps.norm.logpdf(observed, model, pars['scale'])
    elif isinstance(dist, ss.lognorm_ex):
        sigma2 = np.log1p(pars['std']**2/mean**2)
        return sps.lognorm.logpdf(observed, s=np.sqrt(sigma2), scale=mean*np.exp(-sigma2/2))
    errormsg = f'{type(dist).__name__} cannot be used as a likelihood; use ss.poisson(), ss.nbinom(n=...), ss.normal(scale=...), or ss.lognorm_ex(std=...)'
    raise TypeError(errormsg)


def loglik(sim, df=None, likelihood=None, weights=None, by_series=False):
    """
    The log-likelihood of a sim's data (larger is better): the sum over the series, which are treated as independent

    Args:
        sim (`ss.Sim`): a sim that has been run
        df (df): the data, if not the sim's
        likelihood (dict): the likelihood of each series, if not declared on its outcome
        weights (dict): a multiplier on each series' log-likelihood (default 1)
        by_series (bool): return a dict of the log-likelihood of each series instead of the total
    """
    liks = likelihoods(sim, df, likelihood)
    out = sc.objdict()
    for col,(expected,model) in compare(sim, df).items():
        if liks[col] is None:
            errormsg = f'The data series "{col}" has no likelihood; declare one on its outcome (ss.outcome(..., likelihood=ss.poisson())) or pass likelihood=dict({col}=...)'
            raise ValueError(errormsg)
        out[col] = series_weight(weights, col)*logpdf(liks[col], expected, model).sum()
    return out if by_series else float(sum(out.values()))


def logprior(par, x):
    """ The log-density of a parameter's prior at a value: its `prior=` if it has one, else uniform over its bounds """
    prior = par.prior
    low, high = par.bounds
    if prior is None:
        return -np.log(high - low)
    elif isinstance(prior, ss.uniform):
        return sps.uniform.logpdf(x, prior.pars.low, prior.pars.high - prior.pars.low)
    frozen = sc.dcp(prior).init(force=True).dist
    if frozen is None:
        errormsg = f'The prior of "{par.name}" ({type(prior).__name__}) has no density to use; use e.g. ss.normal(), ss.lognorm_ex(), ss.gamma(), ss.beta_dist(), or ss.uniform()'
        raise TypeError(errormsg)
    return frozen.logpdf(x)


class ParticleFilter:
    """
    An estimate of the likelihood of a stochastic sim's data, from a bootstrap particle filter (see `ss.fit()` with `method='pmcmc'`)

    Many runs of the sim (the particles) go forward together. At each data point, each is
    weighted by the likelihood of the data given its results, and when the weights become
    uneven, the runs are drawn again in proportion to them, so that runs that have strayed
    from the data are dropped and those that follow it are copied. The product over the data
    of the mean weight is an unbiased estimate of the likelihood (Andrieu, Doucet & Holenstein 2010),
    which is what particle MCMC needs.

    Each declared disease is filtered on its own, since the diseases of a sim that the kernels run
    whole do not affect each other. An outcome is compared with the data as its expected number
    (rather than drawn, as a run of the sim draws it), as the likelihood would compare it.

    Args:
        sim (`ss.Sim`): the sim, initialized; it is restarted with each set of values before the filter is called
        likelihood (dict): the likelihood of each data series, if not declared on its outcome
        weights (dict): a multiplier on each series' log-likelihood (default 1)
    """
    def __init__(self, sim, likelihood=None, weights=None):
        liks = likelihoods(sim, likelihood=likelihood)
        missing = [col for col,lik in liks.items() if lik is None]
        why = self.reason(sim) or (f'the data series {sc.strjoin(missing)} have no likelihood' if missing else None)
        if why:
            errormsg = f'The particle filter cannot run this sim, since {why}; use method="optuna" instead'
            raise ValueError(errormsg)
        self.sim = sim
        lined = lineup(sim)
        mods = list(sim.declared)
        self.series = []
        for col in sim.data.columns:
            base, dims = data.parse(col)
            res = data.match_name(sim, base)
            j = [j for j,mod in enumerate(mods) if mod.results.get(res.name) is res][0]
            engine = mods[j].engine
            keep = None
            if dims: # The groups of a stratum
                keep = np.ones(sim.n_groups)
                for dim,level in data.levels(sim, dims).items():
                    d = sim.dim_index(dim)
                    keep = keep*(sim.group_levels[:, d] == sim.strats[d].index(level))
            W, expected, rows = lined[col]
            spans = [(np.flatnonzero(w).min(initial=0), np.flatnonzero(w).max(initial=0) + 1) for w in W] # The steps that each data point is worked out from
            self.series.append(sc.objdict(col=col, j=j, name=res.name, read=self.reader(engine, res.name), keep=keep, W=W, spans=spans, expected=expected, rows=rows,
                                          lik=liks[col], weight=series_weight(weights, col)))
        self.engines = {j:mods[j].engine for j in sorted({s.j for s in self.series})}
        return

    @staticmethod
    def reason(sim):
        """ Why the particle filter cannot run a sim (apart from any data series without a likelihood), if it cannot """
        if sim.solver not in ['sde', 'binomial', 'tau', 'gillespie']:
            return f'it has solver="{sim.solver}", not a stochastic compartmental one'
        elif not sim.kernel_only():
            return f'it is not run in one call to the kernels ({sim.kernel_reason()})'
        elif sim.data is None:
            return 'it has no data'
        for col in sim.data.columns:
            base, dims = data.parse(col)
            res = data.match_name(sim, base)
            mods = [mod for mod in sim.declared if mod.results.get(res.name) is res]
            if not mods:
                return f'the data series "{col}" is not a result of a declared disease'
            if dims and ParticleFilter.reader(mods[0].engine, res.name) is None:
                return f'the data series "{col}" is a stratum of "{res.name}", which is not a state or a transition'
        return None

    @staticmethod
    def reader(engine, name):
        """ Where a result is in the stocks or flows of a run, if it is read straight from them: ('stocks' or 'flows', indices to sum) """
        graph = engine.graph
        if name.startswith('n_') and name[2:] in graph.states:
            return 'stocks', [list(graph.states).index(name[2:])]
        flows = engine.tr_index | engine.transfer_index
        if name in flows:
            return 'flows', [flows[name]]
        if name == 'new_infections' and graph.transmissions:
            return 'flows', [engine.tr_index[tr.name] for tr in graph.transmissions]
        return None

    def __call__(self, n_particles=100, seed=None):
        """ The estimate of the log-likelihood, for the sim as it was last restarted """
        rng = np.random.default_rng(seed)
        return sum(self.filter(j, engine, n_particles, rng) for j,engine in self.engines.items())

    def filter(self, j, engine, n, rng):
        """ The estimate of the log-likelihood of the data of one disease """
        series = [s for s in self.series if s.j == j]
        (y0, prog, dt, npts, solver, n_sub), tables = engine.run_args()
        states = list(engine.graph.states)
        state_of = np.array([states.index(comp.state.name) for comp in engine.comps])
        n_flows = prog.n_trans + len(prog.mov_mat)
        by_group = any(s.keep is not None for s in series)
        nk = engine.n_groups if by_group else 1
        stocks = np.zeros((n, npts, len(states), nk))
        flows = np.zeros((n, npts, n_flows, nk))
        for c,state in enumerate(state_of):
            stocks[:, 0, state] += y0[c] if by_group else y0[c].sum()
        ys = np.ascontiguousarray(np.repeat(y0[None], n, axis=0))

        # The data points in order, with the last step each needs and the first step that any later one needs
        points = sc.ddict(list)
        for s in series:
            for i,row in enumerate(s.rows):
                points[row].append((s, i))
        order = sorted(points)
        last = np.maximum.accumulate([max(s.spans[i][1] - 1 for s,i in points[row]) for row in order])
        first = [min(s.spans[i][0] if s.read is not None else 0 for s,i in points[row]) for row in order]
        needed = np.append(np.minimum.accumulate(first[::-1])[::-1][1:], npts) # For each point, the earliest step of the runs that the points after it read

        ss.kernels.seed(int(rng.integers(2**31)))
        logw = np.full(n, -np.log(n)) # The normalized log-weight of each particle
        total = 0.0
        done = 0
        for k,row in enumerate(order):
            if last[k] > done:
                ss.kernels.advance(ys, prog, dt, done, last[k], solver, n_sub, *tables, engine.register, state_of, stocks, flows)
                done = last[k]
            out = None
            obs = np.zeros(n)
            for s,i in points[row]:
                lo, hi = s.spans[i]
                if s.read is None: # A result worked out from the others, as a run works it out
                    if out is None:
                        out = engine.collect(stocks, flows, expected=True)
                    vals = out[s.name][:, lo:hi]
                else:
                    kind, inds = s.read
                    arr = (stocks if kind == 'stocks' else flows)[:, lo:hi, inds, :].sum(axis=2)
                    vals = (arr*s.keep).sum(axis=-1) if s.keep is not None else arr.sum(axis=-1)
                obs += s.weight*logpdf(s.lik, s.expected[i], vals @ s.W[i, lo:hi])
            obs = np.where(np.isnan(obs), -np.inf, obs)
            joint = logw + obs
            top = joint.max()
            if not np.isfinite(top): # No run can give this data
                return -np.inf
            norm = top + np.log(np.exp(joint - top).sum())
            total += norm
            logw = joint - norm
            weights = np.exp(logw)
            if 1/(weights**2).sum() < n/2 and k < len(order) - 1: # Too uneven: draw the runs again (systematically), and keep the steps that are still needed
                cum = np.cumsum(weights)
                picks = np.minimum(np.searchsorted(cum, (rng.random() + np.arange(n))/n*cum[-1]), n - 1)
                ys = ys[picks]
                lo = needed[k]
                if lo <= done:
                    stocks[:, lo:done+1] = stocks[picks, lo:done+1]
                    flows[:, lo:done+1] = flows[picks, lo:done+1]
                logw = np.full(n, -np.log(n))
        engine.cache.clear()
        return total


def pmmh(logpost, z0, draws=1000, warmup=1000, chains=4, particles=None, seed=None, parallel=True, verbose=True):
    """
    Sample from a posterior by particle marginal Metropolis-Hastings: a random-walk Metropolis sampler on a log-density that is estimated by a particle filter

    Since the estimate of the likelihood is unbiased, the chains sample from the exact
    posterior (Andrieu, Doucet & Holenstein 2010), however noisy it is; more noise only
    makes them move more slowly. In the warm-up, the proposal's covariance is learned from
    the chain so far (Haario et al. 2001), and its scale is tuned toward accepting 15% of
    proposals (about the best when the estimate is noisy; Sherlock et al. 2015). In the first
    half of the warm-up, the estimate where the chain is is made again at each step, so that
    a chain far from the posterior, where the estimate is noisiest, cannot be held by a lucky one.

    Args:
        logpost (func): called with a point (an array, on an unbounded scale), a random number generator (`np.random.Generator`) to draw any seeds from, and the number of particles; returns an estimate of the log of the posterior density there
        z0 (array): the point to start each chain from
        draws (int): the draws to keep from each chain
        warmup (int): the draws for the warm-up of each chain, which are not kept
        chains (int): the number of chains
        particles (int): the number of particles; by default, each chain uses 100 for the first half of its warm-up, and then enough that the variance of the estimate at the middle of where it has been lately is about 1 (Doucet et al. 2015)
        seed (int): the random seed
        parallel (bool): whether to run the chains in parallel processes
        verbose (bool): whether to say what is being done

    Returns:
        The draws (chains × draws × values), the estimate of the log-density at each (chains × draws), the share of proposals accepted after the warm-up in each chain, and the number of particles that each chain used
    """
    z0 = np.asarray(z0, dtype=float)
    d = len(z0)

    def one_chain(chain_seed):
        rng = np.random.default_rng(chain_seed)
        n = particles or 100
        out = np.zeros((draws, d))
        logdens = np.zeros(draws)
        accepted = 0.0
        z = z0.copy()
        lp = logpost(z, rng, n)
        log_scale = np.log(2.38**2/d)
        cov = np.eye(d)*0.01
        history = []
        for i in range(warmup + draws):
            if i < warmup//2: # Early in the warm-up, the estimate where the chain is is made again each time, so a lucky one cannot hold it (the draws are not kept)
                lp = logpost(z, rng, n)
            elif particles is None and i == warmup//2: # Near the posterior by now, so the number of particles it needs there (at the middle of where the chain has been lately) can be found
                center = np.mean(history[len(history)//2:], axis=0)
                spread = np.var([logpost(center, rng, n) for k in range(10)])
                n = int(np.clip(np.ceil(n*spread), 50, 5000)) if np.isfinite(spread) else 5000
                lp = logpost(z, rng, n)
            prop = rng.multivariate_normal(z, np.exp(log_scale)*cov)
            lp_prop = logpost(prop, rng, n)
            accept = np.log(rng.random()) < lp_prop - lp
            if accept:
                z, lp = prop, lp_prop
            if i < warmup: # Learn the shape of the posterior, and tune the scale
                history.append(z)
                log_scale += (float(accept) - 0.15)/np.sqrt(i + 1)
                if i >= 50 and i % 10 == 0:
                    cov = np.cov(np.array(history[len(history)//2:]).T).reshape(d, d) + 1e-8*np.eye(d)
            else:
                out[i - warmup] = z
                logdens[i - warmup] = lp
                accepted += accept/draws
        return out, logdens, accepted, n

    T = sc.timer()
    seeds = np.random.default_rng(seed).integers(2**31, size=chains)
    if parallel and chains > 1:
        results = sc.parallelize(one_chain, iterarg=list(seeds))
    else:
        results = [one_chain(chain_seed) for chain_seed in seeds]
    out, logdens, accepted, n = [np.array(v) for v in zip(*results)]
    if verbose:
        print(f'Particle MCMC: {chains} chain{"s" if chains != 1 else ""}{" in parallel" if parallel and chains > 1 else ""}, each of {warmup} warm-up draws and then {draws} draws, '
              f'in {T.tt(output=True):.1f} s; particles: {sc.strjoin(n)}; proposals accepted: {sc.strjoin([f"{a:.0%}" for a in accepted])}')
    return out, logdens, accepted, n


def nuts(logpost, z0, draws=500, warmup=500, chains=4, seed=None, verbose=True):
    """
    Sample from a posterior with the No-U-Turn sampler (NUTS, from BlackJAX), with Stan's warm-up of the step size and the mass matrix

    The log-density is given as a function of NumPy arrays, so it can be anything (here, a
    run of the sim, differentiated by `ss.gradients.gradient()`); it is called back from JAX.

    Args:
        logpost (func): called with a point (an array, on an unbounded scale); returns the log of the posterior density there and its gradient
        z0 (array): the point to start each chain near
        draws (int): the draws to keep from each chain
        warmup (int): the draws for the warm-up of each chain, which are not kept
        chains (int): the number of chains, which are run one after another
        seed (int): the random seed
        verbose (bool): whether to say what is being done

    Returns:
        The draws (chains × draws × values), the log-density at each (chains × draws), and whether each followed a divergent trajectory (chains × draws)
    """
    jax = ss.gradients.jax()
    try:
        import blackjax
    except ImportError as e:
        errormsg = 'NUTS needs BlackJAX, which is not installed: pip install blackjax (or pip install starsim[jax])'
        raise ImportError(errormsg) from e
    jnp = jax.numpy
    z0 = np.asarray(z0, dtype=float)
    shape = (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct(z0.shape, jnp.float64))

    def host(z):
        value, grad = logpost(np.asarray(z, dtype=float))
        if not np.isfinite(value):
            return np.float64(-np.inf), np.zeros(z0.shape)
        return np.float64(value), np.asarray(grad, dtype=np.float64)

    @jax.custom_vjp
    def logdensity(z):
        return jax.pure_callback(host, shape, z)[0]

    def forward(z):
        value, grad = jax.pure_callback(host, shape, z)
        return value, grad

    def backward(grad, cotangent):
        return (cotangent*grad,)

    logdensity.defvjp(forward, backward)
    out = np.zeros((chains, draws, len(z0)))
    logdens = np.zeros((chains, draws))
    divergent = np.zeros((chains, draws), dtype=bool)
    x64 = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True) # For the whole run, since the callbacks are made from other threads, which jax.enable_x64() does not reach
    try:
        key = jax.random.key(seed if seed is not None else 0)
        for c in range(chains):
            key, start_key, warm_key, run_key = jax.random.split(key, 4)
            start = jnp.asarray(z0) + 0.01*jax.random.normal(start_key, z0.shape) # Near the optimum, each a little apart
            if verbose:
                print(f'NUTS chain {c+1} of {chains}: {warmup} warm-up draws, then {draws} draws')
            adapt = blackjax.window_adaptation(blackjax.nuts, logdensity)
            (state, params), _ = adapt.run(warm_key, start, num_steps=warmup)
            step = jax.jit(blackjax.nuts(logdensity, **params).step)
            for i,draw_key in enumerate(jax.random.split(run_key, draws)):
                state, info = step(draw_key, state)
                out[c, i] = np.asarray(state.position)
                logdens[c, i] = float(state.logdensity)
                divergent[c, i] = bool(info.is_divergent)
    finally:
        jax.config.update('jax_enable_x64', x64)
    return out, logdens, divergent


def nuts_jax(logdensity, z0, draws=500, warmup=500, chains=4, seed=None, verbose=True):
    """
    Sample from a posterior with the No-U-Turn sampler (BlackJAX), as `nuts()`, but from a log-density that is a JAX function, so that the whole of the sampling is compiled, with the chains run together

    The gradient is taken in forward mode (one direction per parameter), which for a few
    parameters is much cheaper than reverse mode through the steps of an ODE.

    Args:
        logdensity (func): a JAX function of a point (on an unbounded scale) giving the log of the posterior density there
        z0 (array): the point to start each chain near
        draws, warmup, chains, seed, verbose: as for `nuts()`

    Returns:
        The draws (chains × draws × values), the log-density at each (chains × draws), and whether each followed a divergent trajectory (chains × draws)
    """
    jax = ss.gradients.jax()
    import blackjax
    jnp = jax.numpy
    z0 = np.asarray(z0, dtype=float)
    d = len(z0)
    logdensity = jax.jit(logdensity) # Traced once, rather than wherever the sampler uses it

    @jax.custom_vjp
    def logp(z):
        return logdensity(z)

    def forward(z): # The value and every partial derivative, in forward mode
        values, slopes = jax.vmap(lambda e: jax.jvp(logdensity, (z,), (e,)))(jnp.eye(d))
        return values[0], slopes

    def backward(slopes, cotangent):
        return (cotangent*slopes,)

    logp.defvjp(forward, backward)

    def one_chain(key, start):
        warm_key, run_key = jax.random.split(key)
        adapt = blackjax.window_adaptation(blackjax.nuts, logp)
        (state, params), _ = adapt.run(warm_key, start, num_steps=warmup)
        step = blackjax.nuts(logp, **params).step
        def body(state, key):
            state, info = step(key, state)
            return state, (state.position, state.logdensity, info.is_divergent)
        _, out = jax.lax.scan(body, state, jax.random.split(run_key, draws))
        return out

    x64 = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        key = jax.random.key(seed if seed is not None else 0)
        start_key, chain_key = jax.random.split(key)
        starts = jnp.asarray(z0) + 0.01*jax.random.normal(start_key, (chains, d)) # Near the optimum, each a little apart
        keys = jax.random.split(chain_key, chains)
        devices = jax.devices('cpu')
        n_dev = max(k for k in range(1, min(len(devices), chains) + 1) if chains % k == 0) # Each device takes the same number of chains
        shape = lambda arr: arr.reshape((n_dev, chains//n_dev) + arr.shape[1:])
        T = sc.timer()
        if n_dev > 1: # Each on cores of its own
            compiled = jax.pmap(jax.vmap(one_chain), devices=devices[:n_dev]).lower(shape(keys), shape(starts)).compile()
            args = (shape(keys), shape(starts))
        else:
            compiled = jax.jit(jax.vmap(one_chain)).lower(keys, starts).compile()
            args = (keys, starts)
        compiling = T.tt(output=True)
        out = [np.asarray(v).reshape((chains,) + v.shape[2 if n_dev > 1 else 1:]) for v in compiled(*args)]
        if verbose:
            print(f'NUTS: {chains} chains on {n_dev} core{"s" if n_dev != 1 else ""}, each of {warmup} warm-up draws and then {draws} draws: compiled in {compiling:.1f} s, sampled in {T.tt(output=True):.1f} s')
    finally:
        jax.config.update('jax_enable_x64', x64)
    return out


def rank_normalize(x):
    """ Draws (chains × draws) replaced by the normal quantiles of their ranks among all of them """
    ranks = sps.rankdata(x, method='average').reshape(x.shape)
    return sps.norm.ppf((ranks - 3/8)/(x.size + 1/4))


def split_chains(x):
    """ Each chain (chains × draws) cut into its first and second halves """
    half = x.shape[1]//2
    return np.concatenate([x[:, :half], x[:, -half:]])


def basic_rhat(x):
    """ The potential scale reduction of draws (chains × draws) """
    n = x.shape[1]
    within = x.var(axis=1, ddof=1).mean()
    between = n*x.mean(axis=1).var(ddof=1)
    return np.sqrt((between/within + n - 1)/n) if within > 0 else np.nan


def basic_ess(x):
    """ The effective sample size of draws (chains × draws), from their autocorrelation, with Geyer's initial monotone sequence (as Stan and ArviZ work it out) """
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    size = 2**int(np.ceil(np.log2(2*n)))
    acov = np.fft.irfft(np.abs(np.fft.rfft(centered, size, axis=1))**2, axis=1)[:, :n]/n # Each chain's autocovariance, by lag
    mean_var = acov[:, 0].mean()*n/(n - 1)
    var_plus = mean_var*(n - 1)/n + (x.mean(axis=1).var(ddof=1) if m > 1 else 0)
    if var_plus <= 0:
        return float(m*n)
    rho = np.zeros(n)
    rho[0] = 1.0
    even = 1.0
    odd = 1 - (mean_var - acov[:, 1].mean())/var_plus
    rho[1] = odd
    t = 1
    while t < n - 3 and even + odd > 0: # Sum pairs of lags while they are positive
        even = 1 - (mean_var - acov[:, t+1].mean())/var_plus
        odd = 1 - (mean_var - acov[:, t+2].mean())/var_plus
        if even + odd >= 0:
            rho[t+1], rho[t+2] = even, odd
        t += 2
    max_t = t - 2
    if even > 0:
        rho[max_t+1] = even
    t = 1
    while t <= max_t - 2: # Make the pairs fall monotonically
        if rho[t+1] + rho[t+2] > rho[t-1] + rho[t]:
            rho[t+1] = rho[t+2] = (rho[t-1] + rho[t])/2
        t += 2
    tau = -1 + 2*rho[:max_t+1].sum() + rho[max_t+1]
    return m*n/max(tau, 1/np.log10(m*n))


def diagnose(draws):
    """
    The R-hat (near 1 if the chains agree) and the effective sample size of each value, from draws (chains × draws × values)

    Both are rank-normalized and from split chains (Vehtari et al. 2021, as Stan, ArviZ, and BlackJAX work them out): the
    R-hat is the larger of that of the draws and that of their distance from the median, and the effective sample size is of the bulk.
    """
    rhat, ess = [], []
    for j in range(draws.shape[2]):
        x = split_chains(draws[:, :, j])
        folded = np.abs(x - np.median(x))
        rhat.append(max(basic_rhat(rank_normalize(x)), basic_rhat(rank_normalize(folded))))
        ess.append(basic_ess(rank_normalize(x)))
    return np.array(rhat), np.array(ess)


class Fit(sc.prettyobj):
    """
    The result of fitting a sim to its data; see `ss.fit()`

    Attributes:
        sim (`ss.Sim`): the calibrated sim: a copy of the original with the best-fit values, already run
        pars (dict): the best-fit value of each fitted parameter; can be passed to `ss.Sim(pars=...)`
        table (df): parameter, initial value, best fit, and bounds
        before (`ss.Sim`): the sim with the initial values, already run
        mismatch (float): the goodness of fit of the calibrated sim, or its negative log-likelihood if the fit used likelihoods
        measure (str): what `mismatch` measures
        trials (df): every evaluation
        samples (df): with `method='nuts'` or `'pmcmc'`, the draws from the posterior: one row per draw, with its chain, its number, and a column for each parameter
        divergences (int): with `method='nuts'`, the number of draws that followed a divergent trajectory (which suggests the sampler missed part of the posterior)
        acceptance (array): with `method='pmcmc'`, the share of proposals that each chain accepted after its warm-up
        band (dict): with `method='nuts'` or `'pmcmc'`, for each data series, the 2.5th and 97.5th percentiles of the model's value over draws from the posterior (with `'pmcmc'`, each from a run with a seed of its own, so the band includes chance)
    """
    def __init__(self, sim, before, pars, table, trials, method, mismatch, mismatch_before, measure='goodness of fit', samples=None, divergences=None, acceptance=None, band=None):
        self.samples = samples
        self.divergences = divergences
        self.acceptance = acceptance
        self.band = band
        self.measure = measure
        self.sim = sim
        self.before = before
        self.pars = pars
        self.table = table
        self.trials = trials
        self.method = method
        self.mismatch = mismatch
        self.mismatch_before = mismatch_before
        return

    def summarize(self):
        """ Print the estimates and the goodness of fit before and after """
        if self.samples is None:
            print(f'Fitted {len(self.pars)} parameter(s) by {self.method}, with {len(self.trials)} evaluations')
        else: # The evaluations in JAX are not recorded
            print(f'Fitted {len(self.pars)} parameter(s) by {self.method}')
        print(self.table.to_string(index=False))
        print(f'{self.measure.capitalize()}: {self.mismatch_before:.4g} before, {self.mismatch:.4g} after')
        if self.samples is not None:
            n_chains = self.samples.chain.nunique()
            print(f'Posterior: {len(self.samples):,} draws from {n_chains} chain{"s" if n_chains != 1 else ""} (best: the draw with the highest posterior density; median, lower, and upper: the median and 95% interval of the posterior)')
            if self.acceptance is not None:
                print(f'Proposals accepted by each chain: {sc.strjoin([f"{a:.0%}" for a in self.acceptance])}')
            unsettled = self.table.par[self.table.rhat > 1.01].tolist()
            if unsettled or self.divergences:
                ss.warn(f'W1304 NotConverged: {f"the chains disagree about {sc.strjoin(unsettled)} (R-hat above 1.01)" if unsettled else ""}{"; " if unsettled and self.divergences else ""}{f"{self.divergences} draws diverged" if self.divergences else ""}; '
                        f'try more warm-up (warmup=), or check that the parameters are identifiable')
        for par in self.table.itertuples():
            span = par.high - par.low
            if min(par.best - par.low, par.high - par.best) < 0.01*span:
                ss.warn(f'W1301 AtBound: the best-fit value of "{par.par}" ({par.best:n}) is at one of its bounds [{par.low:n}, {par.high:n}]')
        return self.table

    def to_df(self):
        """ Every evaluation: the parameter values tried, and the goodness of fit of each """
        return self.trials

    def plot(self, **kwargs):
        """ Plot the results that have data against it, before and after fitting """
        import matplotlib.pyplot as plt
        cols = list(self.sim.data.columns)
        fig, axs = sc.getrowscols(len(cols), make=True, squeeze=False, **kwargs)
        for ax,col in zip(axs.flatten(), cols):
            for sim,label in [(self.before, 'Before'), (self.sim, 'After')]:
                res = data.match(sim, col)
                ax.plot(res.timevec, res.values, label=label, alpha=0.8)
            if self.band is not None:
                ax.fill_between(res.timevec, *self.band[col], alpha=0.3, label='95% of the posterior')
            ax.scatter(self.sim.data.index, self.sim.data[col], c='k', s=15, label='Data', zorder=10)
            ax.set_title(col)
            ax.legend(frameon=False)
        sc.figlayout(fig=fig)
        return ss.return_fig(fig)


def product_pairs(sim, names):
    """
    The pairs of fitted parameters that appear in the model only multiplied together, or only one divided by the other, so that data can determine only their product (or ratio)

    Args:
        sim (`ss.Sim`): the sim, initialized
        names (list): the names of the fitted parameters of the model ("beta", or "sir.beta")

    Returns a list of (name, name, 'product' or 'ratio').
    """
    found = {}
    for name in names:
        modname, _, key = name.rpartition('.')
        found[name] = [mod.graph.pars[key] for mod in sim.declared if key in mod.graph.pars and modname in ['', mod.name]]
    uses = [] # Every expression that a parameter can appear in
    for mod in sim.declared:
        graph = mod.graph
        for tr in graph.transitions.values():
            uses += [tr.rate, tr.dur] + sc.tolist(tr.p)
        for tf in graph.transfers.values():
            uses += [tf.rate, tf.dur]
        for oc in graph.outcomes.values():
            uses += [oc.source, oc.delay, oc.duration]
        uses += list(graph.exprs.values()) + [state.init for state in graph.states.values()]
    uses += [m.amount for mods in sim.modifiers.values() for m in mods]
    uses = [use for use in uses if isinstance(use, ss.Expr)]

    def signs(expr, sign=1, out=None):
        """ The parameters that are factors of an expression: +1 for one it is multiplied by, -1 for one it is divided by """
        out = {} if out is None else out
        if isinstance(expr, ss.transitions.BinOp) and expr.op in ['*', '/']:
            signs(expr.a, sign, out)
            signs(expr.b, sign if expr.op == '*' else -sign, out)
        elif isinstance(expr, ss.par):
            out[id(expr)] = out.get(id(expr), 0) + sign
        return out

    out = []
    for i,a in enumerate(names):
        for b in names[i+1:]:
            relation = None
            for use in uses:
                ids = [id(leaf) for leaf in use.leaves]
                n_a = sum(ids.count(id(p)) for p in found[a])
                n_b = sum(ids.count(id(p)) for p in found[b])
                if n_a == 0 and n_b == 0:
                    continue
                factors = signs(use)
                sa = sum(factors.get(id(p), 0) for p in found[a])
                sb = sum(factors.get(id(p), 0) for p in found[b])
                if n_a != 1 or n_b != 1 or abs(sa) != 1 or abs(sb) != 1 or relation not in [None, sa*sb]: # Each once, as a factor, the same way everywhere
                    relation = 0
                    break
                relation = sa*sb
            if relation:
                out.append((a, b, 'product' if relation > 0 else 'ratio'))
    return out


def correlated_pairs(names, values, threshold):
    """ The pairs of parameters whose values (one row each, e.g. draws from a posterior) are correlated beyond a threshold: (name, name, r) """
    out = []
    if len(names) > 1 and values.shape[1] > 2:
        r = np.corrcoef(values)
        for i in range(len(names)):
            for k in range(i+1, len(names)):
                if np.isfinite(r[i, k]) and abs(r[i, k]) > threshold:
                    out.append((names[i], names[k], r[i, k]))
    return out


def default_method(sim, use_lik, posterior=True):
    """
    The fitting method that suits a sim, and why: see `ss.fit()`

    Args:
        sim (`ss.Sim`): the sim, initialized
        use_lik (bool): whether every data series has a likelihood
        posterior (bool): whether a posterior can be the default (if not, a point estimate)
    """
    no_grad = ss.gradients.reason(sim)
    if sim.solver != 'ode':
        if posterior and use_lik and ParticleFilter.reason(sim) is None:
            return 'pmcmc', 'a stochastic compartmental sim whose data series all have a likelihood, so a posterior, from particle MCMC'
        return 'optuna', 'a stochastic sim'
    elif no_grad:
        return 'nelder-mead', f'no gradients, since {no_grad}'
    elif posterior and use_lik:
        if importlib.util.find_spec('blackjax') is None:
            return 'l-bfgs-b', 'gradients by automatic differentiation; for a posterior from NUTS, pip install blackjax'
        return 'nuts', 'every data series has a likelihood, so a posterior, with gradients by automatic differentiation'
    return 'l-bfgs-b', 'gradients by automatic differentiation'


def fit(sim, pars=None, data=None, weights=None, likelihood=None, gof=None, method=None, n_runs=None, rand_seed=None, verbose=None, **kwargs):
    """
    Fit a sim's parameters to its data; usually called as `sim.calibrate()`

    Fits every parameter that has `bounds` (see `ss.par`) to every result that has
    data (see `ss.Sim(data=...)`). The original sim is not changed.

    Args:
        sim (`ss.Sim`): the sim to fit, with data
        pars (list): the names of the parameters to fit (default: all that have bounds)
        data (df): the data to fit to (default: the sim's data)
        weights (dict): the relative weight of each data series (default: equal)
        likelihood (dict): the likelihood of a data series, by its name (e.g. `dict(prevalence=ss.normal(scale=0.02))`), for series whose outcome does not declare one. If every series has a likelihood, the fit maximizes their sum
        gof (str/func): the goodness of fit: 'nae' (normalized absolute error, the default), 'rmse', or a function of the sim
        method (str): 'nuts' (the default for a deterministic sim with gradients whose data series all have a likelihood: a posterior, from the No-U-Turn sampler, with the chains started from the initial values; the priors are the parameters' `prior=`, or uniform over their bounds), 'l-bfgs-b' (the default for other deterministic sims, with gradients from JAX; see `ss.gradients`), 'nelder-mead' (the default for one that JAX cannot run), 'asd' (adaptive stochastic descent, `sc.asd()`), any other `scipy.optimize.minimize` method, 'pmcmc' (particle MCMC, the default for a stochastic compartmental sim whose data series all have a likelihood: a posterior, from a random-walk Metropolis sampler with the likelihood estimated by a particle filter; see `ss.ParticleFilter`), or 'optuna' (the default for other stochastic sims, including agent-based ones: a best fit, not a posterior)
        scan (int): for the SciPy methods and particle MCMC, the number of points spread over the bounds to try first (default: 10 per parameter); the search (or the chains) starts from the best of them
        n_runs (int): runs per evaluation, for a stochastic sim (default 1)
        rand_seed (int): the random seed for the fitting method
        verbose (bool): whether to print progress
        kwargs (dict): passed to the fitting method, e.g. `trials=200` for Optuna, `maxiter=500` for SciPy, `draws=500`, `warmup=500`, and `chains=4` for NUTS (the defaults; 1,000 draws and 1,000 warm-up draws for particle MCMC, whose draws are more correlated), `particles=` for particle MCMC (default: chosen by each chain in its warm-up; see `ss.calibration.pmmh()`), or `parallel=False` to run its chains one after another

    Returns:
        An `ss.Fit` object

    Examples:
        ```python
        beta = ss.par(0.1, bounds=[0.01, 0.5])
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(10), init_prev=0.01)
        sim = ss.Sim(sir, data='cases.csv', start='2020-01-01', stop='2020-06-01')
        fit = sim.calibrate()
        fit.sim.plot()
        ```
    """
    import scipy.optimize as spo
    base = sim.new(verbose=0)
    if data is not None:
        base.data = data
    probe = sim.new(verbose=0) # To find the parameters and check the data
    if data is not None:
        probe.data = data
    probe.init()
    if probe.data is None:
        ss.error('E1302', 'calibrate() needs data: attach it to the sim with ss.Sim(..., data=...), with one column named like each result to fit.')
    verbose = sc.ifelse(verbose, ss.options.verbose)
    liks = likelihoods(probe, likelihood=likelihood)
    use_lik = all(lik is not None for lik in liks.values()) and gof is None
    used = [id(val) for lik in liks.values() if use_lik and lik is not None for val in lik.pars.values()]
    unused = [id(val) for mod in probe.declared for oc in mod.graph.outcomes.values() if oc.likelihood is not None for val in oc.likelihood.pars.values() if id(val) not in used]
    fitted = sc.objdict()
    for mod in probe.declared:
        for name,p in mod.graph.pars.items():
            if id(p) in unused: # A parameter of a likelihood that is not used in this fit
                continue
            given = mod.pars.get(name)
            if isinstance(given, ss.par) and given.bounds is not None: # Bounds given with the sim's pars, e.g. pars=dict(k=ss.par(3, bounds=[0.1, 100]))
                p = given
            if p.bounds is not None and (pars is None or name in pars or f'{mod.name}.{name}' in pars) and not any(p is q for q in fitted.values()): # A parameter shared by several modules is fitted once
                same = [m.graph.pars[name] for m in probe.declared if name in m.graph.pars]
                key = name if all(q is same[0] for q in same) else f'{mod.name}.{name}'
                fitted[key] = p
    n_model = len(fitted)
    if use_lik:
        for col,lik in liks.items(): # A parameter of a likelihood given at calibration time, with bounds, is fitted too
            for key,val in lik.pars.items():
                if isinstance(val, ss.par) and val.module is None and val.bounds is not None and (pars is None or val.name in sc.tolist(pars)):
                    fitted[val.name or f'{col}_{key}'] = val
        if verbose:
            print(f'Using the likelihood: the sum of the log-likelihoods of {sc.strjoin(liks.keys())}, treated as independent' + (' (weighted)' if weights else ''))
    elif any(lik is not None for lik in liks.values()) and verbose:
        print(f'Note: not every data series has a likelihood, so the likelihoods of {sc.strjoin([k for k,v in liks.items() if v is not None])} are not used')
    if not fitted:
        ss.error('E1301', 'calibrate() found no parameter with bounds; give the parameters to fit bounds, e.g. beta = ss.par(0.1, bounds=[0.01, 0.5]).')
    names = list(fitted.keys())
    products = product_pairs(probe, names[:n_model]) # Parameters that the data cannot tell apart, from the model
    lows = np.array([p.bounds[0] for p in fitted.values()], dtype=float)
    highs = np.array([p.bounds[1] for p in fitted.values()], dtype=float)
    x0 = np.array([p.raw.value if isinstance(p.raw, ss.TimePar) else p.raw for p in fitted.values()], dtype=float)
    types = [type(p.raw) if isinstance(p.raw, ss.TimePar) else None for p in fitted.values()] # Keep the unit of typed values
    logs = (lows > 0) & (highs/np.maximum(lows, 1e-300) >= 10) # Wide positive bounds are searched on a log scale
    if verbose and logs.any():
        print(f'Searching on a log scale: {sc.strjoin(np.array(names)[logs])}')
    stochastic = probe.solver not in ['ode']
    n_runs = sc.ifelse(n_runs, 1)

    external = [i >= n_model for i in range(len(fitted))] # Parameters of a likelihood given at calibration time

    def to_pars(x):
        for p,val,ext in zip(fitted.values(), x, external):
            if ext:
                p.value = float(val)
        return {name:(cls(val) if cls is not None else float(val)) for name,val,cls,ext in zip(names, x, types, external) if not ext}

    def measure(sim):
        return -loglik(sim, likelihood=likelihood, weights=weights) if use_lik else mismatch(sim, weights=weights, gof=gof)

    runner = base.new() # If the sim is nothing but compartments, it is built once and restarted for each evaluation
    runner.data = base.data
    restart = runner.init().kernel_only()
    no_grad = ss.gradients.reason(runner) # Why there are no gradients, if there are not
    if method is None:
        method, why = default_method(runner, use_lik)
        if verbose:
            print(f'Fitting by {method} ({why}; override with method=)')
    sample = method in ['nuts', 'pmcmc']
    if method == 'nuts' and (not use_lik or no_grad):
        why = 'not every data series has a likelihood' if not use_lik else f'it has no gradients, since {no_grad}'
        ss.error('E1303', f'method="nuts" gives a posterior, which needs a likelihood for every data series and the gradients of a sim run in JAX, but {why}.')
    elif method == 'pmcmc' and (not use_lik or ParticleFilter.reason(runner)):
        why = 'not every data series has a likelihood' if not use_lik else ParticleFilter.reason(runner)
        ss.error('E1303', f'method="pmcmc" gives a posterior, which needs a likelihood for every data series and a stochastic compartmental sim that the kernels run whole, but {why}.')
    if sample:
        sampling = {key:kwargs.pop(key) for key in ['draws', 'warmup', 'chains'] + (['parallel'] if method == 'pmcmc' else []) if key in kwargs}
        n_particles = kwargs.pop('particles', None)
    use_grad = method.lower() in ['l-bfgs-b', 'bfgs', 'cg', 'tnc', 'slsqp', 'trust-constr'] and not no_grad

    def run(x, seed=None, fresh=False):
        if restart and not fresh:
            return runner.restart(pars=to_pars(x), rand_seed=seed).run()
        kw = {} if seed is None else dict(rand_seed=seed)
        new = base.new(pars=to_pars(x), **kw)
        new.data = base.data
        return new.run()

    trials = []
    def objective(x):
        values = [measure(run(x, seed=None if n_runs == 1 else i + 1)) for i in range(n_runs)]
        value = float(np.mean(values))
        trials.append(dict(zip(names, x), mismatch=value))
        return value

    def with_gradient(x):
        """ The objective and its gradient, through the run in JAX """
        value, derivs = ss.gradients.gradient(runner, lambda x: runner.restart(pars=to_pars(x)), x, measure)
        value = float(value)
        trials.append(dict(zip(names, x), mismatch=value))
        return value, np.array(derivs, dtype=float)

    def encode(x): return np.where(logs, np.log(np.maximum(x, 1e-300)), x)
    def decode(z): return np.clip(np.where(logs, np.exp(np.minimum(z, 700)), z), lows, highs)

    mismatch_before = objective(x0)
    start = x0
    scan = kwargs.pop('scan', None if method in ['optuna', 'nuts', 'pmcmc'] else 10*len(names))
    if scan and method != 'pmcmc': # The fit can have several local optima, so start from the best of a spread of points
        sampler = sps.qmc.LatinHypercube(d=len(names), seed=rand_seed if rand_seed is not None else 1)
        points = decode(encode(lows) + sampler.random(scan)*(encode(highs) - encode(lows)))
        values = [objective(x) for x in points]
        if min(values) < mismatch_before:
            start = points[int(np.argmin(values))]
    if sample: # The chains start from the initial values, and the warm-up finds the posterior
        best = x0
    elif method == 'optuna':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=rand_seed))
        study.enqueue_trial(dict(zip(names, x0)))
        def trial_objective(trial):
            x = [trial.suggest_float(name, lo, hi, log=bool(log)) for name,lo,hi,log in zip(names, lows, highs, logs)]
            return objective(np.array(x))
        study.optimize(trial_objective, n_trials=kwargs.pop('trials', 100), **kwargs)
        best = np.array([study.best_params[name] for name in names])
    elif method == 'asd': # Adaptive stochastic descent (sc.asd)
        lo, hi = encode(lows), encode(highs)
        result = sc.asd(lambda z: objective(decode(z)), encode(start), xmin=lo, xmax=hi, stepsize=kwargs.pop('stepsize', 0.1), randseed=rand_seed, verbose=0, **kwargs)
        best = decode(result.x)
    else:
        bounds = list(zip(encode(lows), encode(highs)))
        options = sc.mergedicts(dict(xatol=1e-4, fatol=1e-6) if method == 'nelder-mead' else None, kwargs)
        if use_grad: # The gradient with respect to the scale searched on: d/dz = x d/dx where x = exp(z)
            def fun(z):
                x = decode(z)
                value, grad = with_gradient(x)
                return value, grad*np.where(logs, x, 1.0)
            result = spo.minimize(fun, encode(start), method=method, jac=True, bounds=bounds, options=options)
        else:
            result = spo.minimize(lambda z: objective(decode(z)), encode(start), method=method, bounds=bounds, options=options)
        best = decode(result.x)

    samples = divergences = acceptance = band = None
    if sample: # On a scale on which every value is allowed: the logit of each value's place between its bounds (of its log, for one searched on a log scale)
        base_lows = np.where(logs, np.log(np.where(logs, lows, 1.0)), lows)
        span = np.where(logs, np.log(np.where(logs, highs, 1.0)), highs) - base_lows

        def to_x(z):
            share = np.clip(1/(1 + np.exp(-z)), 1e-12, 1 - 1e-12)
            pos = base_lows + span*share
            return np.where(logs, np.exp(pos), pos), share

        def share_of(x):
            return np.clip((np.where(logs, np.log(np.maximum(x, 1e-300)), x) - base_lows)/span, 1e-9, 1 - 1e-9)

        def to_z(x):
            share = share_of(x)
            return np.log(share/(1 - share))

        if method == 'pmcmc': # A random walk, on a likelihood estimated by a particle filter
            filt = ParticleFilter(runner.restart(), likelihood=likelihood, weights=weights)
            rng = np.random.default_rng(rand_seed)

            def logpost_pf(z, rng, n):
                x, share = to_x(z)
                runner.restart(pars=to_pars(x), rand_seed=int(rng.integers(2**31))) # Each with its own start, if the start is drawn
                return filt(n_particles=n, seed=int(rng.integers(2**31))) + sum(logprior(p, val) for p,val in zip(fitted.values(), x)) + np.log(span*share*(1 - share)*np.where(logs, x, 1.0)).sum()

            starts = [x0] # The chains start from the best of the initial values and a spread of points
            sampler = sps.qmc.LatinHypercube(d=len(names), seed=rand_seed if rand_seed is not None else 1)
            starts += list(decode(encode(lows) + sampler.random(scan or 10*len(names))*(encode(highs) - encode(lows))))
            start = starts[int(np.argmax([logpost_pf(to_z(x), rng, 100) for x in starts]))]
            zs, logdens, acceptance, n_particles = pmmh(logpost_pf, to_z(start), particles=n_particles, seed=rand_seed, verbose=verbose, **sampling)
        else: # NUTS
            def logpost(z):
                x, share = to_x(z)
                dxdz = span*share*(1 - share)*np.where(logs, x, 1.0)
                value, derivs = ss.gradients.gradient(runner, lambda x: runner.restart(pars=to_pars(x)), x, measure)
                hs = 1e-6*np.maximum(np.abs(x), 1e-6)
                prior = sum(logprior(p, val) for p,val in zip(fitted.values(), x))
                dprior = np.array([(logprior(p, val + h) - logprior(p, val - h))/(2*h) if p.prior is not None else 0.0 for p,val,h in zip(fitted.values(), x, hs)])
                dlogjac = 1 - 2*share + np.where(logs, span*share*(1 - share), 0.0) # The derivative of log(dx/dz)
                return -float(value) + prior + np.log(dxdz).sum(), (dprior - np.array(derivs, dtype=float))*dxdz + dlogjac

            def exact(z): # The log of the posterior density at a point, from a run of the sim
                x, share = to_x(z)
                return -measure(run(x)) + sum(logprior(p, val) for p,val in zip(fitted.values(), x)) + np.log(span*share*(1 - share)*np.where(logs, x, 1.0)).sum()

            try: # The log-density as one function in JAX, if the model allows; otherwise each gradient is worked out with Python in the loop
                density = ss.gradients.Density(runner, fitted, x0, lambda x: runner.restart(pars=to_pars(x)), likelihood=likelihood, weights=weights, external=external)
            except ss.gradients.Unsupported as E:
                density = None
                if verbose:
                    print(f'Note: sampling with Python in the loop, which is slower, since the log-density in JAX does not cover {E} yet')
            if density is not None:
                jnp = ss.gradients.jax().numpy
                def logpost_jax(z):
                    share = jnp.clip(1/(1 + jnp.exp(-z)), 1e-12, 1 - 1e-12)
                    pos = base_lows + span*share
                    x = jnp.where(logs, jnp.exp(pos), pos)
                    return density(x) + jnp.log(span*share*(1 - share)*jnp.where(logs, x, 1.0)).sum()
                zs, logdens, divergent = nuts_jax(logpost_jax, to_z(x0), seed=rand_seed, verbose=verbose, **sampling)
                flat_z, flat_lp = zs.reshape(-1, len(names)), logdens.reshape(-1)
                for i in np.linspace(0, len(flat_lp) - 1, 3).astype(int): # The JAX log-density is the one the sim gives, at a few of the draws
                    expected = exact(flat_z[i])
                    if not np.isclose(flat_lp[i], expected, rtol=1e-6, atol=1e-6):
                        ss.warn(f'The log-density in JAX ({flat_lp[i]:.10g}) differs from the one from the sim ({expected:.10g}) at {dict(zip(names, to_x(flat_z[i])[0]))}, so the posterior is sampled again with Python in the loop')
                        density = None
                        break
            if density is None:
                zs, logdens, divergent = nuts(logpost, to_z(x0), seed=rand_seed, verbose=verbose, **sampling)
        xs = np.array([[to_x(z)[0] for z in chain] for chain in zs])
        logjac = np.log(span*share_of(xs)*(1 - share_of(xs))*np.where(logs, xs, 1.0)).sum(axis=-1)
        best = xs.reshape(-1, len(names))[np.argmax((logdens - logjac).reshape(-1))] # The draw with the highest posterior density (of the values themselves, not on the sampler's scale)
        n_chains, n_draws = zs.shape[:2]
        rhat, n_eff = diagnose(xs)
        samples = pd.DataFrame(xs.reshape(-1, len(names)), columns=names)
        samples.insert(0, 'chain', np.repeat(np.arange(n_chains), n_draws))
        samples.insert(1, 'draw', np.tile(np.arange(n_draws), n_chains))
        divergences = int(divergent.sum()) if method == 'nuts' else None
        table = pd.DataFrame(dict(par=names, initial=x0, best=best, low=lows, high=highs))
        table['median'] = np.median(samples[names].values, axis=0)
        table['lower'], table['upper'] = np.percentile(samples[names].values, [2.5, 97.5], axis=0)
        table['rhat'], table['ess'] = rhat, n_eff
        picks = samples[names].values[np.linspace(0, len(samples) - 1, min(100, len(samples))).astype(int)] # Spread evenly over the draws
        values = sc.ddict(list)
        for k,x in enumerate(picks):
            sim = run(x, seed=k + 1 if method == 'pmcmc' else None) # A stochastic run takes a seed of its own, so the band includes chance
            for col in sim.data.columns:
                values[col].append(np.array(ss.data.match(sim, col).values, dtype=float)) # ss.data, since data is an argument here
        band = {col:np.percentile(vals, [2.5, 97.5], axis=0) for col,vals in values.items()}

    else:
        table = pd.DataFrame(dict(par=names, initial=x0, best=best, low=lows, high=highs))
    unidentifiable = []
    for a,b,how in products:
        va, vb = best[names.index(a)], best[names.index(b)]
        unidentifiable.append(f'"{a}" and "{b}" appear in the model only {"multiplied together" if how == "product" else "one divided by the other"}, so the data can determine only their {how} '
                              f'({va*vb if how == "product" else va/vb:.3g}), not each of them; fit one and fix the other')
    known = [{a, b} for a,b,how in products]
    if sample: # From the fit: parameters almost perfectly correlated over the posterior (of their logs, for those searched on a log scale)
        draws = np.where(logs[:, None], np.log(np.maximum(samples[names].values.T, 1e-300)), samples[names].values.T)
        for a,b,r in correlated_pairs(names, draws, 0.95):
            if {a, b} not in known:
                unidentifiable.append(f'"{a}" and "{b}" are almost perfectly correlated in the posterior (r = {r:.3f}), so the data determine a combination of them rather than each')
    elif use_grad and len(names) > 1: # Or parameters that change the fitted results in almost the same way at the best fit
        def read(sim):
            parts = [model/max(np.abs(expected).mean(), 1e-12) for expected,model in compare(sim).values()]
            return np.concatenate(parts)
        value, derivs = ss.gradients.gradient(runner, lambda x: runner.restart(pars=to_pars(x)), best, read)
        slopes = np.array([d*x for d,x in zip(derivs, best)]) # Relative to each value, so the scale of a parameter does not matter
        norms = np.linalg.norm(slopes, axis=1)
        for i in range(len(names)):
            for k in range(i+1, len(names)):
                cos = slopes[i] @ slopes[k]/(norms[i]*norms[k]) if norms[i]*norms[k] > 0 else 0.0
                if abs(cos) > 0.999 and {names[i], names[k]} not in known:
                    unidentifiable.append(f'"{names[i]}" and "{names[k]}" change the fitted results in almost the same way at the best fit (r = {cos:.4f}), so the data can hardly tell them apart')

    before = run(x0, fresh=True)
    after = run(best, fresh=True) # Last, so a likelihood's own parameters keep their best values
    for msg in unidentifiable:
        ss.note(msg, code='W1302', sim=after, verbose=verbose)
    best_pars = sc.objdict(zip(names, [cls(val) if cls is not None else float(val) for val,cls in zip(best, types)]))
    out = Fit(sim=after, before=before, pars=best_pars, table=table, trials=pd.DataFrame(trials), method=method,
              mismatch=measure(after), mismatch_before=mismatch_before, measure='negative log-likelihood' if use_lik else 'goodness of fit',
              samples=samples, divergences=divergences, acceptance=acceptance, band=band)
    return out


def split_data(df, holdout=0.25, folds=None):
    """
    Split data into the part to fit to and the part to predict, for each fold

    Each series is split on its own times. With a fraction, the last `holdout` of
    each series' points is held out; with `folds=k`, that part is cut into k blocks,
    and fold j fits to everything before block j and predicts block j. With a date
    or time, everything from it on is held out.

    Returns a list of (fitting data, held-out data) pairs, one per fold.
    """
    k = folds or 1
    out = []
    for j in range(k):
        train = df.copy().astype(float)
        test = df.copy().astype(float)
        for col in df.columns:
            ok = np.flatnonzero(np.isfinite(df[col].values.astype(float)))
            if sc.isnumber(holdout) and 0 < holdout < 1:
                n_out = max(k, int(np.ceil(holdout*len(ok))))
                blocks = np.array_split(ok[len(ok)-n_out:], k)
                predict = blocks[j]
            else:
                if k > 1:
                    errormsg = 'folds= needs holdout= to be a fraction'
                    raise ValueError(errormsg)
                index = df.index.values
                after = (np.array([ss.date(t).years for t in index]) >= ss.date(holdout).years) if not sc.isnumber(index[0]) else (index.astype(float) >= float(holdout))
                predict = ok[after[ok]]
            if not len(predict) or predict[0] == ok[0]:
                errormsg = f'Holding out {holdout} leaves no data to fit to, or none to predict, for "{col}"'
                raise ValueError(errormsg)
            fit_part = ok[ok < predict[0]]
            train.iloc[np.setdiff1d(np.arange(len(df)), fit_part), train.columns.get_loc(col)] = np.nan
            test.iloc[np.setdiff1d(np.arange(len(df)), predict), test.columns.get_loc(col)] = np.nan
        out.append((train, test))
    return out


class ModelComparison(sc.prettyobj):
    """
    The result of comparing models by how well they predict held-out data; see `ss.compare_models()`

    Attributes:
        fits (dict): each model's `ss.Fit` (a list of them, one per fold, if there are several folds)
        table (df): the fitted parameters, the error on the data fitted to, the prediction error, and the rank of each model
        holdout (str): a description of what was held out
        measure (str): how the prediction error is measured
    """
    def __init__(self, fits, table, holdout, measure, splits, clear=True):
        self.fits = fits
        self.table = table
        self.holdout = holdout
        self.measure = measure
        self.splits = splits
        self.clear = clear
        return

    def summarize(self):
        """ Print the table """
        print(f'held out: {self.holdout}')
        print(self.table.to_string())
        print(f'prediction error is {self.measure} on held-out data (lower is better)')
        if not self.clear:
            print('the spread of the prediction error across folds is larger than the difference between the best two models, so there is no clear difference between them')
        return self.table

    def to_df(self):
        """ The table, as a dataframe """
        return self.table

    def plot(self, **kwargs):
        """ Plot each model's fit and prediction against the data, with the held-out data hollow """
        train, test = self.splits[-1]
        cols = list(train.columns)
        fig, axs = sc.getrowscols(len(cols), make=True, squeeze=False, **kwargs)
        for ax,col in zip(axs.flatten(), cols):
            for label,fits in self.fits.items():
                fit = sc.tolist(fits)[-1]
                res = data.match(fit.sim, col)
                ax.plot(res.timevec, res.values, label=label, alpha=0.8)
            ax.scatter(train.index, train[col], c='k', s=15, label='Fitted to', zorder=10)
            ax.scatter(test.index, test[col], facecolors='none', edgecolors='k', s=15, label='Held out', zorder=10)
            ax.set_title(col)
            ax.legend(frameon=False)
        sc.figlayout(fig=fig)
        return ss.return_fig(fig)


def compare_models(sims, data=None, holdout=0.25, folds=None, verbose=None, **kwargs):
    """
    Compare models by how well they predict data they were not fitted to

    Each model is fitted (with `sim.calibrate()`) to all but the last part of each
    data series, run forward over that part, and scored on it: by the default
    goodness of fit, or by the log predictive density if every series has a likelihood.
    A model with more parameters usually fits better; the prediction error needs no
    separate penalty for them.

    Args:
        sims (list/dict): the sims to compare; labeled by their `label`, else by position (or by the keys of a dict)
        data (df/str): the data, attached to every sim (default: each sim's own data, which must be the same for all)
        holdout (float/date): the fraction of each series to hold out (default: the last 25%), or the date (or time) from which to hold it out
        folds (int): if given, repeat the comparison with this many cut points over the held-out part (fit to the first part, predict the next block), and report the mean and spread of the prediction error
        verbose (bool): whether to print the table
        kwargs (dict): passed to `sim.calibrate()`, e.g. `weights=`

    Returns:
        An `ss.ModelComparison`; `comp.fits` holds each model's fit

    Examples:
        ```python
        comp = ss.compare_models([sim_sir, sim_seir], data=df)
        comp.plot()
        ```
    """
    verbose = sc.ifelse(verbose, ss.options.verbose)
    if isinstance(sims, dict):
        labels, sims = list(sims.keys()), list(sims.values())
    else:
        sims = sc.tolist(sims)
        labels = [sim.label or f'model {i+1}' for i,sim in enumerate(sims)]
    if len(set(labels)) < len(labels):
        labels = [f'{label} ({i+1})' for i,label in enumerate(labels)]

    # Find the data
    probes = []
    for sim in sims:
        probe = sim.new(verbose=0)
        if data is not None:
            probe.data = data
        probes.append(probe.init())
    dfs = [probe.data for probe in probes]
    if any(df is None for df in dfs):
        ss.error('E1302', 'compare_models() needs data: pass data=, or attach the same data to every sim.')
    if data is None and not all(df.equals(dfs[0]) for df in dfs[1:]):
        errormsg = 'The sims have different data; pass data= to compare them on the same data'
        raise ValueError(errormsg)
    df = dfs[0]
    splits = split_data(df, holdout=holdout, folds=folds)

    # Describe what was held out
    train, test = splits[-1] if folds is None else (splits[0][0], pd.concat([s[1] for s in splits]).groupby(level=0).first())
    spans = []
    for col in df.columns:
        times = test.index[np.isfinite(test[col].values.astype(float))]
        fmt = lambda t: f'{round(t, 6):g}' if sc.isnumber(t) else str(t)
        spans.append(f'{col}: {fmt(times[0])} to {fmt(times[-1])}')
    what = f'the last {holdout:.0%} of each series' if sc.isnumber(holdout) and 0 < holdout < 1 else f'from {holdout}'
    desc = f'{what}{f", in {folds} folds" if folds else ""} ({"; ".join(spans)})'

    # Fit and predict
    liks = [likelihoods(probe, likelihood=kwargs.get('likelihood')) for probe in probes]
    use_lik = all(lik is not None for lk in liks for lik in lk.values()) and kwargs.get('gof') is None
    fits = sc.objdict()
    rows = []
    for label,sim,probe in zip(labels, sims, probes):
        fold_fits = []
        fit_errors = []
        errors = []
        for train,test in splits:
            fit = ss.fit(sim, data=train, verbose=False, **sc.mergedicts(dict(method=default_method(probe, use_lik, posterior=False)[0]), kwargs)) # A point estimate, for each fold
            fold_fits.append(fit)
            fit_errors.append(fit.mismatch)
            if use_lik:
                errors.append(-loglik(fit.sim, df=test, likelihood=kwargs.get('likelihood'), weights=kwargs.get('weights')))
            else:
                errors.append(mismatch(fit.sim, df=test, weights=kwargs.get('weights'), gof=kwargs.get('gof')))
        fits[label] = fold_fits if folds else fold_fits[0]
        row = dict(model=label, fitted_pars=len(fold_fits[0].pars), fit_error=np.mean(fit_errors), prediction_error=np.mean(errors))
        if folds:
            row['spread'] = np.std(errors)
        rows.append(row)
    table = pd.DataFrame(rows).set_index('model')
    table['rank'] = table['prediction_error'].rank(method='min').astype(int)
    table = table.sort_values('rank')

    # Check whether the difference is larger than the spread
    clear = True
    if folds and len(table) > 1:
        diff = table['prediction_error'].iloc[1] - table['prediction_error'].iloc[0]
        clear = diff > table['spread'].iloc[:2].max()
    measure = 'the negative log predictive density' if use_lik else 'the normalized absolute error' if kwargs.get('gof') is None else f'the goodness of fit ({kwargs["gof"]})'
    comp = ModelComparison(fits=fits, table=table, holdout=desc, measure=measure, splits=splits, clear=clear)
    if verbose:
        comp.summarize()
    return comp
