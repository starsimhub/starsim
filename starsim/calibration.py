"""
Define the calibration class
"""
import os
import inspect
import sys
import tempfile
import importlib.util
import functools
import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as sps
import scipy.special as spsp
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
            output = sc.parallelize(functools.partial(ss.kernels.one_thread, self.worker), iterarg=self.run_args.n_workers) # The workers share the cores, so each takes one thread
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

        fig = msim.plot() # For a MultiSim, the median and an interval
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
        if isinstance(self.source, (str, os.PathLike)): # Separated by commas, tabs, or semicolons, whichever the file uses
            df = sc.dataframe(pd.read_csv(self.source, sep=None, engine='python'))
        else:
            df = sc.dataframe(self.source)
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
        if not matches and name == 'new_infections': # Starsim 3's name, for a declared disease with one transmission: its flow
            for mod in getattr(sim, 'declared', []):
                if modname in ['', mod.name] and len(mod.graph.transmissions) == 1:
                    matches.append(mod.results[mod.graph.transmissions[0].name])
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
                        raise ss.UnknownDataColumn(f'the data for {name} has {dim}={level}, which is not a level of "{dim}"; its levels are {sc.strjoin(strat.labels)}.')
                if getattr(base, 'strata', None) is None:
                    raise ss.UnknownDataColumn(f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is not kept by group, so it cannot be split by them.')
                elif getattr(base, 'group_ti', None) is not None:
                    raise ss.UnknownDataColumn(f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is kept by group only every few steps, to save memory. '
                                               f'Give the data to the sim when it is made (ss.Sim(..., data=...)), or name fewer results to keep by group with results=[...].')
            if data.match(sim, col) is None:
                several = [mod.name for mod in sim.modules if name in mod.results and isinstance(mod.results[name], ss.Result)]
                if len(several) > 1: # The name of a result of more than one module
                    raise ss.UnknownDataColumn(f'the data column "{col}" could be the result {name} of any of {sc.strjoin(several)}; name the column after one of them, e.g. "{several[0]}.{name}", or rename it with ss.data(..., aliases=dict(...)).')
                raise ss.UnknownDataColumn(f'the data column "{col}" matches no result. The closest results are {ss.transitions.closest(col, [o.split(".")[-1] for o in options])}.\n'
                                           f"     Rename it with ss.data(..., aliases=dict(result='{col}')), or drop it with ss.data(..., ignore_extra=True).")
        sim_t, data_t = data_times(sim, df)
        if len(sim_t) > 1 and len(data_t): # Data entirely outside is an error when it is compared (ss.DataMisaligned); partly outside, the rest is left out of the comparison
            eps = np.median(np.diff(sim_t))/2
            outside = (data_t < sim_t.min() - eps) | (data_t > sim_t.max() + eps)
            if outside.any() and not outside.all():
                ss.note(f'{outside.sum()} of the {len(data_t)} times in the data are outside the simulated period ({sim.t.start} to {sim.t.stop}), so are not compared with the results', sim=sim)
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
        raise ss.DataMisaligned(f'the data ({df.index[0]} to {df.index[-1]}) lies entirely outside the simulated period ({sim.t.start} to {sim.t.stop}).')
    out = sc.objdict()
    eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9 # Times that are the same up to rounding (e.g. 84.00000015 days from 12 weeks) are the same time point
    inside = (data_t >= sim_t.min() - eps) & (data_t <= sim_t.max() + eps)
    for col in df.columns:
        res = data.match(sim, col)
        values = np.asarray(res.values, dtype=float)
        expected = np.asarray(df[col], dtype=float)
        if res.kind == 'flow': # A flow: sum over the interval ending at each data time
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
        if res.kind == 'flow':
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
    check_gof(gof)
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
    for col in ([] if df is None else df.columns): # No data, with a loss that is a function of the sim
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
    `n` as the dispersion), `ss.binom` (with `n` the number of trials, and the model's
    value the probability), `ss.bernoulli` (an observation of 0 or 1, with the model's
    value its probability), `ss.normal` (with `scale`), and `ss.lognorm_ex` (with `std`).
    The counts of a Poisson or negative binomial need not be whole numbers (e.g. weekly
    averages): the factorials are taken as gamma functions, which gives the same values
    for whole numbers.
    """
    pars = {key:(val.raw if isinstance(val, ss.par) else val) for key,val in dist.pars.items()}
    mean = np.maximum(model, 1e-9)
    if isinstance(dist, ss.poisson):
        out = observed*np.log(mean) - mean - spsp.gammaln(observed + 1)
        return np.where(observed < 0, -np.inf, out) # A count cannot be negative
    elif isinstance(dist, ss.nbinom):
        n = pars['n']
        out = spsp.gammaln(observed + n) - spsp.gammaln(n) - spsp.gammaln(observed + 1) + n*np.log(n/(n + mean)) + observed*np.log(mean/(n + mean))
        return np.where(observed < 0, -np.inf, out)
    elif isinstance(dist, ss.binom): # The positives among n, with the model's value as the probability
        n = pars['n']
        p = np.clip(model, 1e-12, 1 - 1e-12)
        out = spsp.gammaln(n + 1) - spsp.gammaln(observed + 1) - spsp.gammaln(n - observed + 1) + observed*np.log(p) + (n - observed)*np.log1p(-p)
        return np.where((observed < 0) | (observed > n), -np.inf, out)
    elif isinstance(dist, ss.bernoulli): # Whether something happened, with the model's value as its probability
        p = np.clip(model, 1e-12, 1 - 1e-12)
        return observed*np.log(p) + (1 - observed)*np.log1p(-p)
    elif isinstance(dist, ss.normal):
        return sps.norm.logpdf(observed, model, pars['scale'])
    elif isinstance(dist, ss.lognorm_ex):
        sigma2 = np.log1p(pars['std']**2/mean**2)
        return sps.lognorm.logpdf(observed, s=np.sqrt(sigma2), scale=mean*np.exp(-sigma2/2))
    errormsg = f'{type(dist).__name__} cannot be used as a likelihood; use ss.poisson(), ss.nbinom(n=...), ss.binom(n=...), ss.bernoulli(), ss.normal(scale=...), or ss.lognorm_ex(std=...)'
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
    low, high = [b.value if isinstance(b, ss.TimePar) else b for b in par.bounds] # Uniform over the bounds is a constant, so the unit of typed bounds does not matter
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
    whole do not affect each other, except those tracked jointly (see `ss.solvers.Crossing`), which are
    filtered together, as the runs of their program. An outcome is compared with the data as its expected number
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
        if sim.solver not in ['binomial', 'gillespie', 'tau']:
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
        total = 0.0
        for j,engine in self.engines.items():
            cross = engine.cross
            if cross is None:
                total += self.filter([j], n_particles, rng)
            elif j == min(i for i,e in self.engines.items() if e.cross is cross): # Once for the diseases tracked jointly
                total += self.filter([i for i,e in self.engines.items() if e.cross is cross], n_particles, rng, cross=cross)
        return total

    def filter(self, js, n, rng, cross=None):
        """ The estimate of the log-likelihood of the data of one disease (or of the diseases of a crossing, run by its program) """
        series = [s for s in self.series if s.j in js]
        by_group = any(s.keep is not None for s in series)
        if cross is None:
            engine = self.engines[js[0]]
            (y0, prog, dt, npts, solver, n_sub), tables = engine.run_args()
            states = list(engine.graph.states)
            state_of = np.array([states.index(comp.state.name) for comp in engine.comps])
            n_states, nk = len(states), engine.n_groups if by_group else 1
            reg, shifts, per_step = engine.register, engine.shifts, engine.per_step()
        else: # Each of the program's compartments is counted on its own, and split between the diseases' states when read
            (y0, prog, dt, npts, solver, n_sub), tables, per_step = cross.run_args()
            state_of = np.arange(cross.n_rows)
            n_states, nk = cross.n_rows, cross.n_cols if by_group else 1
            reg, shifts = cross.register, cross.shifts

        def view(j, arrays):
            """ The stocks or flows of disease j, as its engine has them """
            if cross is None:
                return arrays
            k = cross.engines.index(self.engines[j])
            return (cross.engine_stocks(k, arrays[0], by_group), cross.engine_flows(k, arrays[1], by_group))

        n_flows = prog.n_trans + prog.n_movers
        stocks = np.zeros((n, npts, n_states, nk))
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

        krng = np.random.default_rng(rng.integers(2**63)) # The particles' random numbers
        logw = np.full(n, -np.log(n)) # The normalized log-weight of each particle
        total = 0.0
        done = 0
        for k,row in enumerate(order):
            if last[k] > done:
                ss.kernels.advance(krng, ys, prog, dt, done, last[k], solver, n_sub, *tables, reg, shifts, state_of, stocks, flows, per_step)
                done = last[k]
            outs = dict()
            obs = np.zeros(n)
            for s,i in points[row]:
                lo, hi = s.spans[i]
                if s.read is None: # A result worked out from the others, as a run works it out
                    if s.j not in outs:
                        outs[s.j] = self.engines[s.j].collect(*view(s.j, (stocks, flows)), expected=True)
                    vals = outs[s.j][s.name][:, lo:hi]
                else:
                    kind, inds = s.read
                    arr = view(s.j, (stocks[:, lo:hi], flows[:, lo:hi]))[0 if kind == 'stocks' else 1][:, :, inds, :].sum(axis=2)
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
        for j in js:
            self.engines[j].cache.clear()
        return total


def abc_smc(distance, logprior, z0, particles=200, generations=8, quantile=0.5, tolerance=None, min_acceptance=0.02, seed=None, verbose=True):
    """
    Approximate Bayesian computation by sequential Monte Carlo (Beaumont et al. 2009): a posterior without a likelihood

    A population of particles is drawn from the prior, and in each generation a new one is drawn
    from the last (each particle perturbed by a Gaussian kernel twice as wide as the population)
    and kept only if its run is within the tolerance of the data; the tolerance is the median
    distance of the last population, so it shrinks as the population moves toward the data. Weights
    correct for the particles not being drawn from the prior.

    Args:
        distance (func): the distance of a run from the data, given a point (on the sampler's scale) and a seed
        logprior (func): the log of the prior density of a point, on the sampler's scale
        z0 (array): a point on the sampler's scale (only its length is used: the first generation is drawn uniformly on that scale and weighted by the prior)
        particles (int): the size of the population
        generations (int): the most generations; fewer if the tolerance or the acceptance rate says to stop
        quantile (float): the share of the last population's distances below the next tolerance
        tolerance (float): stop once the tolerance is this small
        min_acceptance (float): stop once a generation accepts fewer of its proposals than this
        seed (int): the random seed

    Returns the population (particles × values, on the sampler's scale), its normalized weights and
    distances, the tolerance and acceptance rate of each generation, and the number of runs.
    """
    rng = np.random.default_rng(seed)
    d = len(z0)
    seeds = iter(rng.integers(2**31, size=10**8))
    # Generation 0: from the prior, by importance sampling of uniform draws on the sampler's scale (each value's logit between its bounds)
    z = rng.uniform(-6, 6, size=(particles, d)) # Within 0.25% of each bound
    dist = np.array([distance(zi, next(seeds)) for zi in z])
    logw = np.array([logprior(zi) for zi in z])
    w = np.exp(logw - logw.max())
    w /= w.sum()
    tolerances, acceptances = [np.inf], [1.0]
    n_runs = particles
    for g in range(generations):
        eps = np.quantile(dist[np.isfinite(dist)], quantile) if np.isfinite(dist).any() else np.inf
        cov = 2*np.atleast_2d(np.cov(z.T, aweights=w)) + 1e-12*np.eye(d)
        chol = np.linalg.cholesky(cov)
        inv = np.linalg.inv(cov)
        new_z, new_dist, tried = [], [], 0
        while len(new_z) < particles and tried < 50*particles:
            zi = z[rng.choice(particles, p=w)] + chol @ rng.standard_normal(d)
            tried += 1
            di = distance(zi, next(seeds))
            if di <= eps:
                new_z.append(zi)
                new_dist.append(di)
        n_runs += tried
        if len(new_z) < particles: # Too few accepted: keep the last population
            break
        new_z = np.array(new_z)
        kern = np.exp(-0.5*np.einsum('ijk,kl,ijl->ij', new_z[:, None] - z[None], inv, new_z[:, None] - z[None])) @ w # The density each new particle was proposed with (up to a constant)
        logw = np.array([logprior(zi) for zi in new_z]) - np.log(np.maximum(kern, 1e-300))
        w = np.exp(logw - logw.max())
        w /= w.sum()
        z, dist = new_z, np.array(new_dist)
        tolerances.append(eps)
        acceptances.append(particles/tried)
        if verbose:
            print(f'ABC-SMC generation {g+1}: tolerance {eps:.4g}, {acceptances[-1]:.0%} of proposals accepted, effective sample size {1/np.sum(w**2):.0f}')
        if (tolerance is not None and eps <= tolerance) or acceptances[-1] < min_acceptance:
            break
    return z, w, dist, tolerances[1:], acceptances[1:], n_runs


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
    if parallel and chains > 1 and 'jax' in sys.modules: # Forking a process that has started JAX can deadlock
        parallel = False
        if verbose:
            print('Particle MCMC: running the chains one after another, since JAX has been started in this process')
    if parallel and chains > 1:
        results = sc.parallelize(functools.partial(ss.kernels.one_thread, one_chain), iterarg=list(seeds))
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
    if n < 2:
        return np.nan
    within = x.var(axis=1, ddof=1).mean()
    between = n*x.mean(axis=1).var(ddof=1)
    return np.sqrt((between/within + n - 1)/n) if within > 0 else np.nan


def basic_ess(x):
    """ The effective sample size of draws (chains × draws), from their autocorrelation, with Geyer's initial monotone sequence (as Stan and ArviZ work it out) """
    m, n = x.shape
    if n < 4: # Too few draws to estimate an autocorrelation
        return np.nan
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
        converged (bool): whether the fitting method reached its own stopping rule (for SciPy and adaptive stochastic descent), stopped improving (for Optuna: by less than 1% over the last quarter of the trials), or has chains that agree (for NUTS and particle MCMC: an R-hat of at most 1.01, and no divergent draws)
        why (str): what `converged` is based on
        scores (list): the score of the best-fit values in each run (one for a deterministic sim); `mismatch` is their mean, so it is the score that was minimized
        seeds (list): the random seed of each of those runs; `sim` is the run with the first
        trials (df): every evaluation
        samples (df): with `method='nuts'` or `'pmcmc'`, the draws from the posterior: one row per draw, with its chain, its number, and a column for each parameter
        divergences (int): with `method='nuts'`, the number of draws that followed a divergent trajectory (which suggests the sampler missed part of the posterior)
        acceptance (array): with `method='pmcmc'`, the share of proposals that each chain accepted after its warm-up
        band (dict): with `method='nuts'` or `'pmcmc'`, for each data series, the 2.5th and 97.5th percentiles of the model's value over draws from the posterior (with `'pmcmc'`, each from a run with a seed of its own, so the band includes chance)
    """
    def __init__(self, sim, before, pars, table, trials, method, mismatch, mismatch_before, measure='goodness of fit', samples=None, divergences=None, acceptance=None, band=None,
                 converged=None, why=None, scores=None, seeds=None):
        self.converged = converged
        self.why = why
        self.scores = scores
        self.seeds = seeds
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

    def project(self, scenarios=None, n=100, rand_seed=None, **kwargs):
        """
        Run the fitted model forward under each scenario, with the uncertainty in the fitted values: one run per draw from the posterior (or of the best fit, for a method that gives none), with the same draws in every scenario

        Args:
            scenarios (dict): for each scenario, what changes: None, an intervention or a list of them, or a dict of arguments for `sim.new()` (default: the fitted sim alone)
            n (int): the number of runs of each scenario
            rand_seed (int): the seed of the draws
            kwargs (dict): passed to `sim.new()` for every scenario, e.g. a later `stop=`

        Returns an `ss.MultiSim`, whose results have a run dimension labelled by scenario (see `msim.to_df(quantiles=...)`).

        Examples:
            ```python
            fit = sim.calibrate(method='abc-smc')
            proj = fit.project(dict(baseline=None, vaccine=ss.vaccinate(...)), stop='2021-01-01')
            df = proj.to_df(quantiles=[0.05, 0.5, 0.95])
            ```
        """
        names = list(self.table['par']) # With a parameter by level, one name for each level
        if self.samples is not None:
            rows = self.samples.sample(n=n, replace=len(self.samples) < n, random_state=rand_seed)[names]
            sets = rows.to_dict('records')
        else:
            sets = [dict(self.pars) for i in range(n)]
        sims = []
        for label,change in (scenarios or dict(fit=None)).items():
            if change is None:
                change = {}
            elif not isinstance(change, dict):
                change = dict(interventions=change)
            sims.append(self.sim.new(label=label, **sc.mergedicts(kwargs, change)))
        return ss.parallel(*sims, pars=sets)

    def summarize(self):
        """ Print the estimates and the goodness of fit before and after """
        if self.samples is None:
            print(f'Fitted {len(self.pars)} parameter(s) by {self.method}, with {len(self.trials)} evaluations')
        else: # The evaluations in JAX are not recorded
            print(f'Fitted {len(self.pars)} parameter(s) by {self.method}')
        print(self.table.to_string(index=False))
        print(f'{self.measure.capitalize()}: {self.mismatch_before:.4g} before, {self.mismatch:.4g} after' + (f' (the mean of {len(self.scores)} runs)' if self.scores is not None and len(self.scores) > 1 else ''))
        if self.converged is not None:
            print(f'{"Converged" if self.converged else "Not converged"}: {self.why}')
        if 'spread_lower' in self.table:
            print('spread_lower and spread_upper: the range of the best 10% of the trials, an indication of the uncertainty only, not an interval')
        if self.samples is not None:
            n_chains = self.samples.chain.nunique()
            print(f'Posterior: {len(self.samples):,} draws from {n_chains} chain{"s" if n_chains != 1 else ""} (best: the draw with the highest posterior density; median, lower, and upper: the median and 95% interval of the posterior)')
            if self.acceptance is not None:
                print(f'Proposals accepted by each chain: {sc.strjoin([f"{a:.0%}" for a in self.acceptance])}')
            unsettled = self.table.par[self.table.rhat > 1.01].tolist() if 'rhat' in self.table else []
            if unsettled or self.divergences:
                ss.warn(f'{f"the chains disagree about {sc.strjoin(unsettled)} (R-hat above 1.01)" if unsettled else ""}{"; " if unsettled and self.divergences else ""}{f"{self.divergences} draws diverged" if self.divergences else ""}; '
                        f'try more warm-up (warmup=), or check that the parameters are identifiable', ss.NotConverged)
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


def vector_labels(sim, by, key):
    """ The labels of the levels of a parameter with a value for each level of one stratification """
    if by.dim is None:
        raise ss.CapabilityError(f'the parameter "{key}" has a value for each combination of {sc.strjoin(by.dims)}; a fit takes a value for each level of one stratification, e.g. ss.par(ss.by(patch=[0.2, 0.3]), bounds=[0.1, 1])')
    strat = next((strat for strat in sim.strats if strat.name == by.dim), None)
    return [str(label) for label in strat.labels] if strat is not None else [str(i) for i in range(len(sc.tolist(by.values)))]


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
    uses = [use for use in uses if isinstance(use, ss.transitions.Expr)]

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


gof_names = ['nae', 'rmse'] # The goodness-of-fit measures that mismatch() knows
scipy_methods = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
fit_methods = ['nuts', 'pmcmc', 'abc-smc', 'optuna', 'asd'] + scipy_methods # The methods that ss.fit() knows


class Optimization(sc.prettyobj):
    """
    The result of allocating spending to maximize (or minimize) a result; see `ss.optimize()`

    Attributes:
        alloc (dict): the best allocation, by program
        value (float): the result with it
        alloc_before (dict): the allocation before
        value_before (float): the result with it
        sim (`ss.Sim`): the sim run with the best allocation
        n_runs (int): the number of runs the search took
    """
    def __init__(self, **kwargs):
        for key,val in kwargs.items():
            setattr(self, key, val)
        return

    def summarize(self):
        """ Print the allocation before and after, and the result """
        df = pd.DataFrame(dict(before=self.alloc_before, after=self.alloc))
        print(df.to_string(float_format=lambda v: f'{v:,.0f}'))
        print(f'{self.what}: {self.value_before:,.6g} before, {self.value:,.6g} after ({self.n_runs} runs)')
        return


def optimize(sim, total=None, maximize=None, minimize=None, at=None, start=None, spend=None, bounds=None, maxiter=50, verbose=None):
    """
    Allocate spending between programs to maximize (or minimize) a result, keeping the total fixed

    The spending of each program is a parameter of the model: one per program of an imported program book
    (`sim.imported.programs`), or any parameters, given by name with `spend=`. The search (SciPy's SLSQP) is
    over the shares of the total, and each point is a run (restarted, if the sim is nothing but compartments).

    Args:
        sim (`ss.Sim`): the sim, not yet run
        total (float): the total to allocate (default: the current total)
        maximize (str/func): what to make as large as possible: a result of the disease (e.g. `'all_vs'`, or `'hiv.n_vs'`), or a function of the run sim
        minimize (str/func): or what to make as small as possible
        at (float/date): the time of the result (default: the end); without it, a function of the sim decides
        start (float/date): when the new allocation takes effect (default: the start of the sim)
        spend (dict/list): the spending parameters: names, or {program: name} (default: the imported programs)
        bounds (tuple): the least and most a program can have (default: 0 and the total)
        maxiter (int): the most iterations of the search
        verbose (bool): whether to print progress

    Returns an `ss.Optimization`.

    Examples:
        ```python
        sim = ss.load('hiv_framework.xlsx', format='atomica', databook='hiv_databook.xlsx', progbook='hiv_progbook.xlsx', programs_start=2020)
        best = ss.optimize(sim, start=2020, maximize='all_vs', at=2030)
        best.summarize()
        ```
    """
    import scipy.optimize as spo
    verbose = sc.ifelse(verbose, ss.options.verbose)
    if (maximize is None) == (minimize is None):
        errormsg = 'ss.optimize() takes one of maximize= or minimize=: a result (e.g. maximize="all_vs") or a function of the sim'
        raise ValueError(errormsg)
    if spend is None:
        spend = getattr(sim, 'imported', {}).get('programs') if isinstance(getattr(sim, 'imported', None), dict) else None
        if not spend:
            errormsg = 'ss.optimize() needs the parameters of spending: give spend= (their names, or {program: name}), or import a program book'
            raise ValueError(errormsg)
    spend = dict(spend) if isinstance(spend, dict) else {name:name for name in sc.tolist(spend)}
    labels, names = list(spend.keys()), list(spend.values())
    probe = sim.new(verbose=0).init()
    current = {}
    for name in names:
        found = [mod.pars[name] for mod in probe.declared if name in mod.pars]
        if not found:
            raise ss.UnknownParameter(f'ss.optimize() has spend="{name}", which is not a parameter of the model')
        current[name] = found[0].raw if isinstance(found[0], ss.par) else found[0]
    t0 = float(probe.t.yearvec[0])
    begin = None if start is None else (ss.date(start).years if isinstance(start, (str, ss.date)) else float(start))
    def level(raw):
        """ A spending parameter's value at the start of the new allocation """
        if isinstance(raw, ss.timeseries):
            keys = sorted(raw.values) if isinstance(raw.values, dict) else None
            return float(np.interp(begin if begin is not None else t0, [float(k) for k in keys], [raw.values[k] for k in keys])) if keys else float(np.mean(raw.values))
        return float(raw.value if isinstance(raw, ss.TimePar) else raw)
    before = np.array([level(current[name]) for name in names])
    total = float(sc.ifelse(total, before.sum()))
    low, high = bounds if bounds is not None else (0.0, total)

    def values(x):
        """ The parameters for an allocation: from the start of the new allocation, if given """
        out = {}
        for name,val in zip(names, x*total):
            raw = current[name]
            if begin is None or begin <= t0:
                out[name] = float(val)
            else:
                old = dict(raw.values) if isinstance(raw, ss.timeseries) and isinstance(raw.values, dict) else {t0:level(raw)}
                old = {k:v for k,v in old.items() if float(k) < begin}
                out[name] = ss.timeseries({**old, begin:float(val)})
        return out

    what = maximize if maximize is not None else minimize
    sign = -1.0 if maximize is not None else 1.0
    def read(run):
        if callable(what):
            return float(what(run))
        modname, _, key = str(what).rpartition('.')
        mod = run.diseases[modname] if modname else run.declared[0]
        vals = np.asarray(mod.results[key].values)
        i = len(vals) - 1 if at is None else int(np.argmin(abs(np.asarray(run.t.yearvec) - (ss.date(at).years if isinstance(at, (str, ss.date)) else float(at)))))
        return float(vals[i])

    runner = sim.new(pars=values(before/total), verbose=0).init() # Built with the spending as it will be given, so a restart can change it
    restart = runner.kernel_only()
    n_runs = [0]
    def run(x):
        n_runs[0] += 1
        if restart:
            return runner.restart(pars=values(x)).run()
        return sim.new(pars=values(x), verbose=0).run()
    value_before = read(run(before/total))
    scale = max(abs(value_before), 1e-12)
    objective = lambda x: sign*read(run(x))/scale
    T = sc.timer()
    res = spo.minimize(objective, before/total, method='SLSQP', bounds=[(low/total, high/total)]*len(names),
                       constraints=[dict(type='eq', fun=lambda x: x.sum() - 1)], options=dict(maxiter=maxiter))
    x = np.clip(res.x, low/total, high/total)
    x = x/x.sum()
    best = run(x)
    value = read(best)
    if sign*value > sign*value_before: # The search can end worse than it started
        x, value, best = before/total, value_before, run(before/total)
    if verbose:
        print(f'Optimized {what}: {value_before:,.6g} to {value:,.6g}, in {n_runs[0]} runs ({T.toc(output=True):.1f} s)')
    return Optimization(alloc=dict(zip(labels, x*total)), value=value, alloc_before=dict(zip(labels, before)), value_before=value_before,
                        sim=best, n_runs=n_runs[0], what=str(getattr(what, '__name__', what)), converged=bool(res.success), why=str(res.message))


def bounded_pars(probe, pars=None, unused=()):
    """
    The parameters with bounds in an initialized sim, as fit() and sensitivity() vary them

    Args:
        probe (Sim): an initialized sim
        pars (list): the names to take (default: all with bounds)
        unused (set): the ids of parameters to leave out (those of likelihoods not used)

    Returns the parameters by name, and the names of those of networks that a compartmental run turns into contact rates (which are set when the sim is built).
    """
    fitted = sc.objdict()
    for mod in probe.declared:
        for name,p in mod.graph.pars.items():
            if id(p) in unused: # A parameter of a likelihood that is not used in this fit
                continue
            if p.bounds is not None and (pars is None or name in pars or f'{mod.name}.{name}' in pars) and not any(p is q for q in fitted.values()): # A parameter shared by several modules is fitted once
                same = [m.graph.pars[name] for m in probe.declared if name in m.graph.pars]
                key = name if all(q is same[0] for q in same) else f'{mod.name}.{name}'
                fitted[key] = p
    mixed = getattr(probe, 'mixed', []) if probe.solver != 'abm' else [] # Networks that a compartmental run turns into contact rates
    contacts = [] # Their parameters that are fitted, which set the contact rates when the sim is built
    for mod in list(probe.modules) + mixed: # A parameter of a module written as code, given as ss.par with bounds, e.g. ssl.HIV(beta=ss.par(0.01, bounds=[0.001, 0.1])); see ss.Pars.bounded()
        if getattr(mod, 'graph', None) is None and isinstance(mod.pars, ss.Pars):
            for name,p in mod.pars.bounded().items():
                if p.bounds is not None and (pars is None or name in pars or f'{mod.name}.{name}' in pars):
                    taken = name in fitted or any(name in m.pars for m in probe.declared)
                    fitted[f'{mod.name}.{name}' if taken else name] = p
                    if any(mod is net for net in mixed):
                        contacts.append(f'{mod.name}.{name}')
    return fitted, contacts


def sensitivity(sim, pars=None, results=None, n=100, rand_seed=None, verbose=None):
    """
    Which parameters matter: Latin hypercube samples over the bounds of the parameters, with the partial rank correlation of each with a summary of each result

    Needs no data. A result is summarized by its peak and its final value (a flow by its peak and its
    total). A partial rank correlation near 1 or -1 means the result goes up or down with the parameter,
    with the others held fixed; near 0, that it does not matter, so it cannot be fitted either.

    Args:
        sim (`ss.Sim`): the sim, which is not changed
        pars (list): the names of the parameters (default: all that have bounds)
        results (list): the results, e.g. `['n_I']` (default: every count and flow)
        n (int): the number of samples
        rand_seed (int): the seed of the samples
        verbose (bool): whether to print the table

    Returns a dataframe of the partial rank correlations, with a row for each parameter and a column for each summary of a result.
    """
    probe = sim.new(verbose=0).init()
    bounded = bounded_pars(probe, pars)[0]
    if not bounded:
        raise ss.NothingToFit('sensitivity() needs parameters with bounds, e.g. beta = ss.par(0.3, bounds=[0.1, 0.5]).')
    names = list(bounded.keys())
    lows, highs = [np.array([float(p.bounds[k]) for p in bounded.values()]) for k in [0, 1]]
    logs = (lows > 0) & (highs/np.maximum(lows, 1e-300) >= 10) # Wide positive bounds are sampled on a log scale, as calibrate() searches them
    unit = sps.qmc.LatinHypercube(d=len(names), seed=rand_seed if rand_seed is not None else 1).random(n)
    lo, hi = np.where(logs, np.log(lows), lows), np.where(logs, np.log(highs), highs)
    X = lo + unit*(hi - lo)
    X = np.where(logs, np.exp(X), X)
    msim = ss.parallel(sim.new(verbose=0), pars={name:X[:, j].tolist() for j,name in enumerate(names)})
    summaries = sc.objdict()
    for modname,key,res in msim.results.walk():
        if modname is None or res.columns is not None:
            continue
        if results is not None and key not in results:
            continue
        if results is None and not (key.startswith('n_') or res.kind == 'flow'):
            continue
        vals = np.asarray(res.values)
        summaries[f'{modname}.{key} peak'] = vals.max(axis=1)
        summaries[f'{modname}.{key} {"total" if res.kind == "flow" else "final"}'] = vals.sum(axis=1) if res.kind == 'flow' else vals[:, -1]
    ranks = lambda v: sps.rankdata(v)
    RX = np.column_stack([ranks(X[:, j]) for j in range(len(names))])
    table = sc.dataframe(index=names)
    for label,y in summaries.items():
        ry = ranks(y)
        col = []
        for j in range(len(names)): # The correlation of what is left of the parameter and the result once the other parameters are regressed out
            others = np.column_stack([np.ones(n), np.delete(RX, j, axis=1)])
            res_x = RX[:, j] - others @ np.linalg.lstsq(others, RX[:, j], rcond=None)[0]
            res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
            denom = np.sqrt((res_x**2).sum()*(res_y**2).sum())
            col.append(float(res_x @ res_y/denom) if denom > 0 else 0.0)
        table[label] = col
    if sc.ifelse(verbose, ss.options.verbose):
        print(f'Partial rank correlations, from {n} samples over the bounds:')
        print(table.round(2).to_string())
    return table


def prior_draws(sim, n, seed=None):
    """
    Values of the model's parameters drawn from their priors (`ss.par(prior=...)`, or uniform over the bounds of one with bounds and no prior), for `ss.parallel(sim, priors=True)`

    Returns a dict with a list of `n` values for each parameter that has a prior or bounds;
    a parameter with a value for each level (`ss.par(ss.by(age=[...]), prior=...)`) has a draw
    for each level, as "beta[a]", "beta[b]", ..., as `calibrate()` fits it.
    """
    rng = np.random.default_rng(seed)
    probe = sim.new(verbose=0).init()
    out = sc.objdict()
    for mod in probe.declared:
        for name,p in mod.graph.pars.items():
            if name in out or (p.prior is None and p.bounds is None):
                continue
            keys = [f'{name}[{label}]' for label in vector_labels(probe, p.raw, name)] if isinstance(p.raw, ss.by) else [name]
            for key in keys:
                if p.prior is None:
                    vals = rng.uniform(*p.bounds, size=n)
                else:
                    prior = sc.dcp(p.prior)
                    prior.strict = False
                    vals = prior.init(trace=f'prior_{key}', seed=int(rng.integers(2**31)), force=True).rvs(n)
                    if p.bounds is not None:
                        vals = np.clip(vals, *p.bounds)
                out[key] = [float(v) for v in vals]
    if not out:
        errormsg = 'priors=True needs parameters with a prior or bounds, e.g. ss.par(0.3, prior=ss.normal(0.3, 0.05)) or ss.par(0.3, bounds=[0.1, 0.5])'
        raise ValueError(errormsg)
    return out


def fit_stages(sim, stages, verbose=None, **kwargs):
    """
    Staged fitting: run each method in turn, each starting from the best fit of the one before; see `ss.fit()` with `method=` a list

    Each stage is a method's name, or a dict with `method=` and that method's settings. A stage
    that does not converge stops the sequence (`ss.StageStopped`), and the fit returned is that
    of the last stage that did, or the first stage's best attempt if it was the one that failed.
    """
    verbose = sc.ifelse(verbose, ss.options.verbose)
    stages = [dict(method=stage) if isinstance(stage, str) else dict(stage) for stage in stages]
    fit, good = None, None
    current = sim
    for i,stage in enumerate(stages):
        method = stage.pop('method')
        if verbose:
            print(f'Stage {i+1} of {len(stages)}: {method}' + (f', started from stage {i}' if i else ''))
        fit = ss.fit(current, method=method, verbose=verbose, **sc.mergedicts(kwargs, stage))
        if not fit.converged:
            if i < len(stages) - 1:
                ss.warn(f'stage {i+1} ({method}) did not converge ({fit.why}), so the stages after it were not run; returning {"stage " + str(i) if good is not None else "its best attempt"}', ss.StageStopped)
            return good if good is not None else fit
        good = fit
        current = fit.sim.new() # With the best-fit values
    return fit


def check_gof(gof):
    """ Raise an error for a goodness of fit that is not known (rather than using the default in its place) """
    if gof is not None and not callable(gof) and gof not in gof_names:
        errormsg = f'Unknown goodness of fit gof="{gof}"; the choices are {sc.strjoin(gof_names)}, or a function of the sim that returns a number'
        raise ValueError(errormsg)
    return


def default_method(sim, use_lik, posterior=True, no_grad=None):
    """
    The fitting method that suits a sim, and why: see `ss.fit()`

    Args:
        sim (`ss.Sim`): the sim, initialized
        use_lik (bool): whether every data series has a likelihood
        posterior (bool): whether a posterior can be the default (if not, a point estimate)
        no_grad (str): why there are no gradients, if the fit knows a reason the sim does not (e.g. a parameter that sets contact rates)
    """
    no_grad = no_grad or ss.gradients.reason(sim)
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
        likelihood (dict): the likelihood of a data series, by its name (e.g. `dict(prevalence=ss.normal(scale=0.02))`), for series whose outcome does not declare one. If every series has a likelihood, the fit maximizes their sum (plus the log of the priors of the parameters that have one, for the methods that find a best fit: the most probable values)
        gof (str/func): the goodness of fit: 'nae' (normalized absolute error, the default), 'rmse', or a function of the sim; any other name is an error
        method (str): 'nuts' (the default for a deterministic sim with gradients whose data series all have a likelihood: a posterior, from the No-U-Turn sampler, with the chains started from the initial values; the priors are the parameters' `prior=`, or uniform over their bounds), 'l-bfgs-b' (the default for other deterministic sims, with gradients from JAX; see `ss.gradients`), 'nelder-mead' (the default for one that JAX cannot run), 'asd' (adaptive stochastic descent, `sc.asd()`), any other `scipy.optimize.minimize` method, 'pmcmc' (particle MCMC, the default for a stochastic compartmental sim whose data series all have a likelihood: a posterior, from a random-walk Metropolis sampler with the likelihood estimated by a particle filter; see `ss.ParticleFilter`), or 'optuna' (the default for other stochastic sims, including agent-based ones: a best fit, not a posterior)
        scan (int): for the SciPy methods and particle MCMC, the number of points spread over the bounds to try first (default: 10 per parameter); the search (or the chains) starts from the best of them
        n_runs (int): runs per evaluation, for a stochastic sim (default 4, and 1 for a deterministic one); every point is scored by the mean over the same seeds (common random numbers), and `fit.mismatch` is that score at the best fit
        rand_seed (int): the random seed for the fitting method (default: the sim's, so a fit gives the same answer each time)
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
    if isinstance(method, (list, tuple)): # Staged fitting: each method starts from the best fit of the one before
        return fit_stages(sim, method, pars=pars, data=data, weights=weights, likelihood=likelihood, gof=gof, n_runs=n_runs, rand_seed=rand_seed, verbose=verbose, **kwargs)
    base = sim.new(verbose=0)
    rand_seed = sc.ifelse(rand_seed, base.pars.rand_seed) # A fit is reproducible, as a sim is
    if data is not None:
        base.data = data
    probe = sim.new(verbose=0) # To find the parameters and check the data
    if data is not None:
        probe.data = data
    probe.init()
    if probe.data is None and not callable(gof): # A loss that is a function of the sim brings its own data
        raise ss.NoData('calibrate() needs data: attach it to the sim with ss.Sim(..., data=...), with one column named like each result to fit; or give gof= as a function of the sim that returns the loss.')
    verbose = sc.ifelse(verbose, ss.options.verbose)
    check_gof(gof)
    if method is not None:
        method = method.lower()
        if method not in fit_methods:
            errormsg = f'Unknown fitting method method="{method}"; the choices are {sc.strjoin(fit_methods)}'
            raise ValueError(errormsg)
    liks = likelihoods(probe, likelihood=likelihood)
    use_lik = all(lik is not None for lik in liks.values()) and gof is None
    used = [id(val) for lik in liks.values() if use_lik and lik is not None for val in lik.pars.values()]
    unused = [id(val) for mod in probe.declared for oc in mod.graph.outcomes.values() if oc.likelihood is not None for val in oc.likelihood.pars.values() if id(val) not in used]
    fitted, contacts = bounded_pars(probe, pars, unused)
    levels = {} # For a parameter with a value per level (ss.par(ss.by(patch=[...]), bounds=)): each level is fitted, with the same bounds and prior, as "beta[a]", "beta[b]", ...
    for key,p in list(fitted.items()):
        if isinstance(p.raw, ss.by):
            labels = vector_labels(probe, p.raw, key)
            del fitted[key]
            for i,label in enumerate(labels):
                fitted[f'{key}[{label}]'] = p
                levels[f'{key}[{label}]'] = i
    n_model = len(fitted)
    if use_lik:
        for col,lik in liks.items(): # A parameter of a likelihood given at calibration time, with bounds, is fitted too
            for key,val in lik.pars.items():
                if isinstance(val, ss.par) and val.module is None and val.bounds is not None and (pars is None or val.name in sc.tolist(pars)):
                    fitted[val.name or f'{col}_{key}'] = val
        if verbose:
            print(f'Using the likelihood: the sum of the log-likelihoods of {sc.strjoin(liks.keys())}, treated as independent' + (' (weighted)' if weights else ''))
    elif any(lik is not None for lik in liks.values()) and verbose:
        why = f'gof="{gof}" was given' if gof is not None else 'not every data series has a likelihood'
        print(f'Note: {why}, so the likelihoods of {sc.strjoin([k for k,v in liks.items() if v is not None])} are not used')
    if not fitted:
        raise ss.NothingToFit('calibrate() found no parameter with bounds; give the parameters to fit bounds, e.g. beta = ss.par(0.1, bounds=[0.01, 0.5]).')
    names = list(fitted.keys())
    products = product_pairs(probe, names[:n_model]) # Parameters that the data cannot tell apart, from the model
    raws = [sc.tolist(p.raw.values)[levels[name]] if name in levels else p.raw for name,p in fitted.items()]
    def bound(b, raw): # A typed bound (e.g. ss.days(3)) in the unit of the parameter's value
        return (type(raw)(b) if isinstance(raw, ss.TimePar) else b).value if isinstance(b, ss.TimePar) else b
    lows = np.array([bound(p.bounds[0], raw) for p,raw in zip(fitted.values(), raws)], dtype=float)
    highs = np.array([bound(p.bounds[1], raw) for p,raw in zip(fitted.values(), raws)], dtype=float)
    x0 = np.array([raw.value if isinstance(raw, ss.TimePar) else raw for raw in raws], dtype=float)
    types = [type(raw) if isinstance(raw, ss.TimePar) else None for raw in raws] # Keep the unit of typed values
    logs = (lows > 0) & (highs/np.maximum(lows, 1e-300) >= 10) # Wide positive bounds are searched on a log scale
    if verbose and logs.any():
        print(f'Searching on a log scale: {sc.strjoin(np.array(names)[logs])}')
    stochastic = probe.solver not in ['ode']
    n_runs = sc.ifelse(n_runs, 4 if stochastic else 1) # A stochastic sim is scored over several runs, so the fit is not to one draw of chance
    seeds = [sc.ifelse(base.pars.rand_seed, 1) + i for i in range(n_runs)] # The same seeds at every point (common random numbers), so the score changes only with the values

    external = [i >= n_model for i in range(len(fitted))] # Parameters of a likelihood given at calibration time

    def to_pars(x):
        for p,val,ext in zip(fitted.values(), x, external):
            if ext:
                p.value = float(val)
        return {name:(cls(val) if cls is not None else float(val)) for name,val,cls,ext in zip(names, x, types, external) if not ext}

    def measure(sim):
        return -loglik(sim, likelihood=likelihood, weights=weights) if use_lik else mismatch(sim, weights=weights, gof=gof)

    priors = [p.prior is not None for p in fitted.values()]
    def penalty(x):
        """ Minus the log of the priors, so that with likelihoods an optimizer finds the most probable values (the maximum a posteriori) rather than ignoring the priors """
        return -sum(logprior(p, val) for p,val,has in zip(fitted.values(), x, priors) if has) if use_map else 0.0

    runner = base.new() # If the sim is nothing but compartments, it is built once and restarted for each evaluation
    runner.data = base.data
    restart = runner.init().kernel_only() and not contacts # Contact rates are worked out when the sim is built, so a new value needs a new sim
    no_grad = ss.gradients.reason(runner) or ('a parameter has a value for each level' if levels else None) or (f'{contacts[0]} sets contact rates when the sim is built' if contacts else None) # Why there are no gradients, if there are not
    if method is None:
        method, why = default_method(runner, use_lik, no_grad=no_grad)
        if verbose:
            print(f'Fitting by {method} ({why}; override with method=)')
    sample = method in ['nuts', 'pmcmc', 'abc-smc']
    use_map = use_lik and any(priors) and not sample # A posterior method uses the priors itself
    if verbose and stochastic and method not in ['pmcmc', 'abc-smc']: # Which take a run of their own for each point
        print(f'Scoring each point by the mean of {n_runs} run{"s" if n_runs != 1 else ""}, with the same seeds at every point (override with n_runs=)')
    if verbose and use_map:
        print(f'Minimizing the negative log-posterior, since {sc.strjoin(np.array(list(fitted.keys()))[priors])} {"has a prior" if sum(priors) == 1 else "have priors"}')
    if method == 'nuts' and (not use_lik or no_grad):
        why = 'not every data series has a likelihood' if not use_lik else f'it has no gradients, since {no_grad}'
        raise ss.NoPosterior(f'method="nuts" gives a posterior, which needs a likelihood for every data series and the gradients of a sim run in JAX, but {why}.')
    elif method == 'pmcmc' and (not use_lik or ParticleFilter.reason(runner)):
        why = 'not every data series has a likelihood' if not use_lik else ParticleFilter.reason(runner)
        raise ss.NoPosterior(f'method="pmcmc" gives a posterior, which needs a likelihood for every data series and a stochastic compartmental sim that the kernels run whole, but {why}.')
    if sample:
        keys = ['generations', 'quantile', 'tolerance'] if method == 'abc-smc' else ['draws', 'warmup', 'chains'] + (['parallel'] if method == 'pmcmc' else [])
        sampling = {key:kwargs.pop(key) for key in keys if key in kwargs}
        n_particles = kwargs.pop('particles', None)
    use_grad = method in ['l-bfgs-b', 'bfgs', 'cg', 'tnc', 'slsqp', 'trust-constr'] and not no_grad

    def run(x, seed=None, fresh=False):
        if restart and not fresh:
            return runner.restart(pars=to_pars(x), rand_seed=seed).run()
        kw = {} if seed is None else dict(rand_seed=seed)
        new = base.new(pars=to_pars(x), **kw)
        new.data = base.data
        return new.run()

    trials = []
    def objective(x):
        values = [measure(run(x, seed=seed)) for seed in seeds]
        value = float(np.mean(values)) + penalty(x)
        trials.append(dict(zip(names, x), mismatch=value))
        return value

    def with_gradient(x):
        """ The objective and its gradient, through the run in JAX """
        value, derivs = ss.gradients.gradient(runner, lambda x: runner.restart(pars=to_pars(x)), x, measure)
        derivs = np.array(derivs, dtype=float)
        if use_map: # The slope of the priors, numerically
            hs = 1e-6*np.maximum(np.abs(x), 1e-6)
            derivs += np.array([(penalty(x + h*e) - penalty(x - h*e))/(2*h) for h,e in zip(hs, np.eye(len(x)))])
        value = float(value) + penalty(x)
        trials.append(dict(zip(names, x), mismatch=value))
        return value, derivs

    def encode(x): return np.where(logs, np.log(np.maximum(x, 1e-300)), x)
    def decode(z): return np.clip(np.where(logs, np.exp(np.minimum(z, 700)), z), lows, highs)

    mismatch_before = objective(x0)
    if not np.isfinite(mismatch_before): # Nothing can be compared with it, so the fit would wander
        series = loglik(run(x0), likelihood=likelihood, weights=weights, by_series=True) if use_lik else {}
        bad = [col for col,val in series.items() if not np.isfinite(val)]
        errormsg = f'The fit cannot start: its {"negative log-likelihood" if use_lik else "mismatch"} at the starting values is {mismatch_before}'
        errormsg += f', from the data series {sc.strjoin(bad)}' if bad else ''
        raise ValueError(errormsg + '; check the data (e.g. for negative or missing values) and the likelihoods')
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
        n_trials = kwargs.pop('trials', kwargs.pop('n_trials', 100)) # Optuna's own name works too
        known = [key for key in inspect.signature(study.optimize).parameters if key not in ['func', 'n_trials']]
        unknown = [key for key in kwargs if key not in known]
        if unknown:
            errormsg = f'calibrate(method="optuna") takes trials= (the number of trials) and Optuna\'s own keywords ({sc.strjoin(known)}), not {sc.strjoin(unknown)}'
            raise TypeError(errormsg)
        study.optimize(trial_objective, n_trials=n_trials, **kwargs)
        best = np.array([study.best_params[name] for name in names])
        values = np.array([t.value for t in study.trials if t.value is not None])
        early = values[:max(1, int(np.ceil(0.75*len(values))))].min() # The best before the last quarter of the trials
        gain = (early - values.min())/max(abs(early), 1e-12)
        converged = gain < 0.01
        why = f'the best value improved by {gain:.1%} over the last quarter of the {len(values)} trials' + ('' if converged else '; try more trials (trials=)')
    elif method == 'asd': # Adaptive stochastic descent (sc.asd)
        lo, hi = encode(lows), encode(highs)
        result = sc.asd(lambda z: objective(decode(z)), encode(start), xmin=lo, xmax=hi, stepsize=kwargs.pop('stepsize', 0.1), randseed=rand_seed, verbose=0, **kwargs)
        best = decode(result.x)
        converged = not any(word in result.exitreason for word in ['Maximum', 'limit', 'Unknown'])
        why = result.exitreason
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
        converged = bool(result.success)
        why = f'{result.message} ({result.nfev} evaluations)'

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
        elif method == 'abc-smc': # A posterior without a likelihood: the runs within a shrinking distance of the data
            def logjac_of(z):
                x, share = to_x(z)
                return np.log(span*share*(1 - share)*np.where(logs, x, 1.0)).sum(axis=-1)

            def distance(z, seed):
                return measure(run(to_x(z)[0], seed=seed if stochastic else None)) # A stochastic run takes a seed of its own, so the posterior includes chance

            def logprior_z(z):
                return sum(logprior(p, val) for p,val in zip(fitted.values(), to_x(z)[0])) + logjac_of(z)

            pop, weights_abc, dists, tolerances, acceptance, n_abc = abc_smc(distance, logprior_z, to_z(x0), particles=n_particles or 200, seed=rand_seed, verbose=verbose, **sampling)
            picks = np.random.default_rng(rand_seed).choice(len(pop), size=len(pop), p=weights_abc) # Equal weights, as draws
            zs = pop[picks][None] # One chain
            logdens = (np.array([logjac_of(z) for z in zs[0]]) - dists[picks])[None] # So the best draw is the closest run
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
        converged = bool(np.all(rhat <= 1.01)) and not divergences # NaN (chains too short to tell) is not converged
        why = f'R-hat at most {np.max(rhat):.3f} and effective sample size at least {np.min(n_eff):.0f}'
        why += f', with {divergences} divergent draws' if method == 'nuts' else f', with {sc.strjoin([f"{a:.0%}" for a in acceptance])} of proposals accepted'
        if method == 'abc-smc': # One population, so no R-hat: the tolerance has stopped shrinking
            ess = 1/np.sum(weights_abc**2)
            table = table.drop(columns='rhat')
            table['ess'] = ess
            settled = len(tolerances) >= 2 and tolerances[-1] > 0.9*tolerances[-2]
            converged = bool(settled)
            why = (f'the tolerance {"settled" if settled else "was still shrinking"} at {tolerances[-1]:.4g} after {len(tolerances)} generations ({n_abc} runs), with {acceptance[-1]:.0%} of the last proposals accepted '
                   f'and an effective sample size of {ess:.0f}' + ('' if settled else '; try more generations (generations=)'))
            acceptance = None
        picks = samples[names].values[np.linspace(0, len(samples) - 1, min(100, len(samples))).astype(int)] # Spread evenly over the draws
        values = sc.ddict(list)
        for k,x in enumerate(picks):
            sim = run(x, seed=k + 1 if method == 'pmcmc' else None) # A stochastic run takes a seed of its own, so the band includes chance
            for col in sim.data.columns:
                values[col].append(np.array(ss.data.match(sim, col).values, dtype=float)) # ss.data, since data is an argument here
        band = {col:np.percentile(vals, [2.5, 97.5], axis=0) for col,vals in values.items()}

    else:
        table = pd.DataFrame(dict(par=names, initial=x0, best=best, low=lows, high=highs))
        if method == 'optuna' and len(trials) >= 10: # Not a posterior, but some indication of the uncertainty: the spread of the best trials
            tried = pd.DataFrame(trials)
            top = tried.nsmallest(max(5, len(tried)//10), 'mismatch')
            table['spread_lower'], table['spread_upper'] = top[names].min().values, top[names].max().values
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

    before = run(x0, seed=seeds[0], fresh=True)
    after = run(best, seed=seeds[0], fresh=True) # A run at the best values, for plotting; also the last, so a likelihood's own parameters keep their best values
    scores = [measure(after)] + [measure(run(best, seed=seed)) for seed in seeds[1:]] # The score of the best values, as in the search
    for msg in unidentifiable:
        ss.note(msg, ss.Unidentifiable, sim=after, verbose=verbose)
    zs, zlo, zhi = encode(best), encode(lows), encode(highs) # On the scale searched on, so a value near the low end of a log scale is not at its bound
    for name,z,lo,hi,val,low,high in zip(names, zs, zlo, zhi, best, lows, highs):
        if min(z - lo, hi - z) < 0.01*(hi - lo):
            ss.warn(f'the best-fit value of "{name}" ({val:n}) is at one of its bounds [{low:n}, {high:n}]; consider widening them', ss.AtBound)
    if verbose:
        print(f'{"Converged" if converged else "Not converged"}: {why}')
    best_pars = sc.objdict()
    for name,val,cls,p in zip(names, best, types, fitted.values()):
        val = cls(val) if cls is not None else float(val)
        if name in levels: # One value per level, gathered as the parameter was given: ss.by(patch=[...])
            base = name.rpartition('[')[0]
            vals = best_pars[base].values if base in best_pars else []
            best_pars[base] = ss.by(**{p.raw.dim:list(vals) + [val]})
        else:
            best_pars[name] = val
    out = Fit(sim=after, before=before, pars=best_pars, table=table, trials=pd.DataFrame(trials), method=method,
              mismatch=float(np.mean(scores)) + penalty(best), mismatch_before=mismatch_before,
              measure='negative log-posterior' if use_map else 'negative log-likelihood' if use_lik else 'goodness of fit',
              samples=samples, divergences=divergences, acceptance=acceptance, band=band, converged=converged, why=why, scores=scores, seeds=seeds)
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
        measure (str): how the fit error and the prediction error are measured (the same for every model)
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
        print(f'fit error and prediction error are {self.measure}, on the data fitted to and on the held-out data (lower is better)')
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


def compare_models(sims, data=None, holdout=0.25, folds=None, **kwargs):
    """
    Compare models by how well they predict data they were not fitted to

    Each model is fitted (with `sim.calibrate()`) to all but the last part of each
    data series, run forward over that part, and scored on it: by the default
    goodness of fit, or by the log predictive density if every series of every model
    has a likelihood; the same measure is used for every model. A model with more
    parameters usually fits better; the prediction error needs no separate penalty
    for them.

    Args:
        sims (list/dict): the sims to compare; labeled by their `label`, else by position (or by the keys of a dict)
        data (df/str): the data, attached to every sim (default: each sim's own data, which must be the same for all)
        holdout (float/date): the fraction of each series to hold out (default: the last 25%), or the date (or time) from which to hold it out
        folds (int): if given, repeat the comparison with this many cut points over the held-out part (fit to the first part, predict the next block), and report the mean and spread of the prediction error
        kwargs (dict): passed to `sim.calibrate()`, e.g. `weights=`

    Returns:
        An `ss.ModelComparison`; `comp.fits` holds each model's fit

    Examples:
        ```python
        comp = ss.compare_models([sim_sir, sim_seir], data=df)
        comp.summarize()
        comp.plot()
        ```
    """
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
        raise ss.NoData('compare_models() needs data: pass data=, or attach the same data to every sim.')
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
    if not use_lik: # One measure for every model, even one whose series all have a likelihood
        kwargs['gof'] = kwargs.get('gof') or 'nae'
    fits = sc.objdict()
    rows = []
    for label,sim,probe in zip(labels, sims, probes):
        fold_fits = []
        fit_errors = []
        errors = []
        for train,test in splits:
            fit = ss.fit(sim, data=train, verbose=False, **sc.mergedicts(dict(method=default_method(probe, use_lik, posterior=False)[0]), kwargs)) # A point estimate, for each fold
            fold_fits.append(fit)
            for df_part,out in [(train, fit_errors), (test, errors)]: # Each scored the same way
                if use_lik:
                    out.append(-loglik(fit.sim, df=df_part, likelihood=kwargs.get('likelihood'), weights=kwargs.get('weights')))
                else:
                    out.append(mismatch(fit.sim, df=df_part, weights=kwargs.get('weights'), gof=kwargs['gof']))
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
    measure = 'the negative log predictive density' if use_lik else 'the normalized absolute error' if kwargs['gof'] == 'nae' else f'the goodness of fit ({kwargs["gof"]})'
    comp = ModelComparison(fits=fits, table=table, holdout=desc, measure=measure, splits=splits, clear=clear)
    return comp
