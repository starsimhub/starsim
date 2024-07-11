"""
Define the calibration class
"""

import os
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['Calibration']


def import_optuna():
    """ A helper function to import Optuna, which is an optional dependency """
    try:
        import optuna as op # Import here since it's slow
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op


def compute_gof(actual, predicted, normalize=True, use_frac=False, use_squared=False, as_scalar='none', eps=1e-9, skestimator=None, estimator=None, **kwargs):
    """
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        actual      (arr):   array of actual (data) points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        as_scalar   (str):   return as a scalar instead of a time series: choices are sum, mean, median
        eps         (float): to avoid divide-by-zero
        skestimator (str):   if provided, use this scikit-learn estimator instead
        estimator   (func):  if provided, use this custom estimator instead
        kwargs      (dict):  passed to the scikit-learn or custom estimator

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
        e4 = compute_gof(x1, x2, skestimator='mean_squared_error') # Scikit-learn's MSE method
        e5 = compute_gof(x1, x2, as_scalar='median') # Normalized median absolute error -- highly robust
    """

    # Handle inputs
    actual    = np.array(sc.dcp(actual), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Scikit-learn estimator is supplied: use that
    if skestimator is not None: # pragma: no cover
        try:
            import sklearn.metrics as sm
            sklearn_gof = getattr(sm, skestimator) # Shortcut to e.g. sklearn.metrics.max_error
        except ImportError as E:
            raise ImportError(f'You must have scikit-learn >=0.22.2 installed: {str(E)}')
        except AttributeError:
            raise AttributeError(f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options')
        gof = sklearn_gof(actual, predicted, **kwargs)
        return gof

    # Custom estimator is supplied: use that
    if estimator is not None:
        try:
            gof = estimator(actual, predicted, **kwargs)
        except Exception as E:
            errormsg = f'Custom estimator "{estimator}" must be a callable function that accepts actual and predicted arrays, plus optional kwargs'
            raise RuntimeError(errormsg) from E
        return gof

    # Default case: calculate it manually
    else:
        # Key step -- calculate the mismatch!
        gofs = abs(np.array(actual) - np.array(predicted))

        if normalize and not use_frac:
            actual_max = abs(actual).max()
            if actual_max>0:
                gofs /= actual_max

        if use_frac:
            if (actual<0).any() or (predicted<0).any():
                print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
            else:
                maxvals = np.maximum(actual, predicted) + eps
                gofs /= maxvals

        if use_squared:
            gofs = gofs**2

        if as_scalar == 'sum':
            gofs = np.sum(gofs)
        elif as_scalar == 'mean':
            gofs = np.mean(gofs)
        elif as_scalar == 'median':
            gofs = np.median(gofs)

        return gofs


class Calibration(sc.prettyobj):
    """
    A class to handle calibration of STIsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org), which must be installed separately (via
    pip install optuna).
    Args:
        sim          (Sim)  : the simulation to calibrate
        data         (df)   : pandas dataframe
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        fit_args     (dict) : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit
        par_samplers (dict) : an optional mapping from parameters to the Optuna sampler to use for choosing new points for each; by default, suggest_float
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers)
        name         (str)  : the name of the database (default: 'hpvsim_calibration')
        db_name      (str)  : the name of the database file (default: 'hpvsim_calibration.db')
        keep_db      (bool) : whether to keep the database after calibration (default: false)
        storage      (str)  : the location of the database (default: sqlite)
        rand_seed    (int)  : if provided, use this random seed to initialize Optuna runs (for reproducibility)
        label        (str)  : a label for this calibration object
        die          (bool) : whether to stop if an exception is encountered (default: false)
        verbose      (bool) : whether to print details of the calibration
        kwargs       (dict) : passed to hpv.Calibration()

    Returns:
        A Calibration object
    """
    def __init__(self, sim, data, calib_pars=None, weights=None, fit_args=None, par_samplers=None, n_trials=None, n_workers=None,
                total_trials=None, name=None, db_name=None, estimator=None, keep_db=None, storage=None, rand_seed=None,
                 sampler=None, label=None, die=False, verbose=True):

        import multiprocessing as mp

        # Handle run arguments
        if n_trials  is None: n_trials  = 20
        if n_workers is None: n_workers = mp.cpu_count()
        if name      is None: name      = 'starsim_calibration'
        if db_name   is None: db_name   = f'{name}.db'
        if keep_db   is None: keep_db   = False
        if storage   is None: storage   = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = int(np.ceil(total_trials/n_workers))
        self.run_args   = sc.objdict(n_trials=int(n_trials), n_workers=int(n_workers), name=name, db_name=db_name,
                                     keep_db=keep_db, storage=storage, rand_seed=rand_seed, sampler=sampler)

        # Handle other inputs
        self.label          = label
        self.sim            = sim
        self.calib_pars     = calib_pars
        self.weights        = weights
        self.fit_args       = sc.mergedicts(fit_args)
        self.par_samplers   = sc.mergedicts(par_samplers)
        self.die            = die
        self.verbose        = verbose
        self.calibrated     = False

        # Load data -- this is expecting a dataframe with a column for 'year' and other columns for to sim results
        if not isinstance(data, pd.DataFrame):
            errormsg = 'Please pass data as a pandas dataframe'
            raise ValueError(errormsg)
        self.target_data = data
        self.target_data.set_index('year', inplace=True)

        # Temporarily store a filename
        self.tmp_filename = 'tmp_calibration_%05i.obj'

        # Initialize sim
        if not self.sim.initialized:
            self.sim.initialize()

        # Figure out which sim results to get
        self.sim_result_list = self.target_data.columns.values.tolist()

        return

    def run_sim(self, calib_pars=None, label=None):
        """ Create and run a simulation """
        sim = sc.dcp(self.sim)
        if label: sim.label = label

        sim = self.translate_pars(sim, calib_pars=calib_pars)

        # Run the sim
        try:
            sim.run()
            return sim

        except Exception as E:
            if self.die:
                raise E
            else:
                print(f'Encountered error running sim!\nParameters:\n{calib_pars}\nTraceback:\n{sc.traceback()}')
                output = None
                return output

    @staticmethod
    def translate_pars(sim=None, calib_pars=None):
        """ Take the nested dict of calibration pars and modify the sim """
        for modtype in calib_pars.keys():
            for dkey, dpars in calib_pars[modtype].items():
                for dparkey, dparval in dpars.items():
                    targetpar = sim[modtype][dkey].pars[dparkey]
                    if sc.isnumber(targetpar):
                        sim[modtype][dkey].pars[dparkey] = dparval
                    elif isinstance(targetpar, ss.Dist):
                        sim[modtype][dkey].pars[dparkey].set(dparval)
                    else:
                        errormsg = 'Type not implemented'
                        raise ValueError(errormsg)

        return sim

    def trial_to_sim_pars(self, pardict=None, trial=None):
        """
        Take in an optuna trial and sample from pars, after extracting them from the structure they're provided in
        Different use cases:
            - pardict is self.calib_pars, i.e. {'diseases':{'hiv':{'art_efficacy':[0.96, 0.9, 0.99]}}}, need to sample
            - pardict is self.initial_pars, i.e. {'diseases':{'hiv':{'art_efficacy':[0.96, 0.9, 0.99]}}}, pull 1st vals
            - pardict is self.best_pars, i.e. {'diseases':{'hiv':{'art_efficacy':0.96786}}}, pull single vals
        """
        pars = sc.dcp(pardict)
        flattened_inputs = sc.flattendict(pars)

        structured_outputs = sc.dcp(self.calib_pars)
        flattened_outputs = sc.flattendict(structured_outputs)

        for key in flattened_outputs.keys():
            sampler_key = '_'.join(key)

            if key in flattened_inputs.keys():
                val = flattened_inputs[key]
            else:
                val = pars[sampler_key]

            if sc.isnumber(val) and trial is None:
                sc.setnested(structured_outputs, list(key), val)

            else:
                low, high = val[1], val[2]
                step = val[3] if len(val) > 3 else None

                if trial is not None:
                    if key in self.par_samplers:  # If a custom sampler is used, get it now (Not working properly for now)
                        try:
                            sampler_fn = getattr(trial, self.par_samplers[key])
                        except Exception as E:
                            errormsg = 'The requested sampler function is not found: ensure it is a valid attribute of an Optuna Trial object'
                            raise AttributeError(errormsg) from E
                    else:
                        sampler_fn = trial.suggest_float
                    sc.setnested(structured_outputs, list(key), sampler_fn(sampler_key, low, high, step=step))
                else:
                    sc.setnested(structured_outputs, list(key), val[0])

        return structured_outputs

    def run_trial(self, trial, save=True):
        """ Define the objective for Optuna """
        if self.calib_pars is not None:
            calib_pars = self.trial_to_sim_pars(self.calib_pars, trial)
        else:
            calib_pars = None
        sim = self.run_sim(calib_pars)

        # Export results
        df_res = sim.export_df()
        df_res['year'] = np.floor(np.round(df_res.index, 1)).astype(int)
        sim_results = sc.objdict()

        for skey in self.sim_result_list:
            if 'prevalence' in skey:
                model_output = df_res.groupby(by='year')[skey].mean()
            else:
                model_output = df_res.groupby(by='year')[skey].sum()
            sim_results[skey] = model_output.values

        sim_results['year'] = model_output.index.values
        # Store results in temporary files
        if save:
            filename = self.tmp_filename % trial.number
            sc.save(filename, sim_results)

        # Compute fit
        fit = self.compute_fit(sim)
        return fit

    def compute_fit(self, sim):
        """ Compute goodness-of-fit """
        fit = 0
        df_res = sim.export_df()
        df_res['year'] = np.floor(np.round(df_res.index, 1)).astype(int)
        for skey in self.sim_result_list:
            if 'prevalence' in skey:
                model_output = df_res.groupby(by='year')[skey].mean()
            else:
                model_output = df_res.groupby(by='year')[skey].sum()

            data = self.target_data[skey]
            combined = pd.merge(data, model_output, how='left', on='year')
            combined['diffs'] = combined[skey+'_x'] - combined[skey+'_y']
            gofs = compute_gof(combined.dropna()[skey+'_x'], combined.dropna()[skey+'_y'])

            losses = gofs  #* self.weights[skey]
            mismatch = losses.sum()
            fit += mismatch

        return fit

    def worker(self):
        """ Run a single worker """
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler = self.run_args.sampler)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None)
        return output

    def run_workers(self):
        """ Run multiple workers in parallel """
        if self.run_args.n_workers > 1: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output

    def remove_db(self):
        """
        Remove the database file if keep_db is false and the path exists.
        """
        try:
            if 'sqlite' in self.run_args.storage:
                # Delete the file from disk
                if os.path.exists(self.run_args.db_name):
                    os.remove(self.run_args.db_name)
                if self.verbose:
                    print(f'Removed existing calibration file {self.run_args.db_name}')
            else:
                # Delete the study from the database e.g., mysql
                op = import_optuna()
                op.delete_study(study_name=self.run_args.name, storage=self.run_args.storage)
                if self.verbose:
                    print(f'Deleted study {self.run_args.name} in {self.run_args.storage}')
        except Exception as E:
            print('Could not delete study, skipping...')
            print(str(E))
        return

    def make_study(self):
        """ Make a study, deleting one if it already exists """
        op = import_optuna()
        if not self.run_args.keep_db:
            self.remove_db()
        if self.run_args.rand_seed is not None:
            sampler = op.samplers.RandomSampler(self.run_args.rand_seed)
            sampler.reseed_rng()
            raise NotImplementedError('Implemented but does not work')
        else:
            sampler = None
        print(self.run_args.storage)
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler)
        return output

    def calibrate(self, calib_pars=None, confirm_fit=False, load=True, tidyup=True, **kwargs):
        """
        Perform calibration.
        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            confirm_fit (bool): if True, run simulations with parameters from before and after calibration
            load (bool): whether to load existing trials from the database (if rerunning the same calibration)
            tidyup (bool): whether to delete temporary files from trial runs
            verbose (bool): whether to print output from each trial
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        op = import_optuna()

        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler = self.run_args.sampler)
        self.best_pars = sc.objdict(study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        self.sim_results = []
        if load:
            print('Loading saved results...')
            for trial in study.trials:
                n = trial.number
                try:
                    filename = self.tmp_filename % trial.number
                    results = sc.load(filename)
                    self.sim_results.append(results)
                    if tidyup:
                        try:
                            os.remove(filename)
                            print(f'    Removed temporary file {filename}')
                        except Exception as E:
                            errormsg = f'Could not remove {filename}: {str(E)}'
                            print(errormsg)
                    print(f'  Loaded trial {n}')
                except Exception as E:
                    errormsg = f'Warning, could not load trial {n}: {str(E)}'
                    print(errormsg)

        # Compare the results
        self.initial_pars = self.trial_to_sim_pars(pardict=self.calib_pars)
        self.parse_study(study)

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()
        
        # Optionally compute the sims before and after the fit
        if confirm_fit:
            self.confirm_fit()

        return self
    
    def confirm_fit(self):
        """ Run before and after simulations to validate the fit """
        before_pars = self.initial_pars
        after_pars  = self.trial_to_sim_pars(pardict=self.best_pars)
        self.before_sim = self.run_sim(calib_pars=before_pars, label='Before calibration')
        self.after_sim  = self.run_sim(calib_pars=after_pars, label='After calibration')
        self.before_fit = self.compute_fit(self.before_sim)
        self.after_fit  = self.compute_fit(self.after_sim)
        return self.before_fit, self.after_fit

    def parse_study(self, study):
        """Parse the study into a data frame -- called automatically """
        best = study.best_params
        self.best_pars = best

        print('Making results structure...')
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
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    print(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)
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

