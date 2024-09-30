"""
Define the calibration class
"""
import os
import numpy as np
import pandas as pd
import sciris as sc
import optuna as op
import matplotlib.pyplot as plt
import starsim as ss


__all__ = ['Calibration', 'compute_gof']


def compute_gof(actual, predicted, normalize=True, use_frac=False, use_squared=False,
                as_scalar='none', eps=1e-9, skestimator=None, estimator=None, **kwargs):
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
            errormsg = f'You must have scikit-learn >=0.22.2 installed: {str(E)}'
            raise ImportError(errormsg) from E
        except AttributeError as E:
            errormsg = f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options'
            raise AttributeError(errormsg) from E
        gof = sklearn_gof(actual, predicted, **kwargs)
        return gof

    # Custom estimator is supplied: use that
    if estimator is not None: # pragma: no cover
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
            if actual_max > 0:
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


class Calibration(sc.prettyobj): # pragma: no cover
    """
    A class to handle calibration of Starsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Args:
        sim          (Sim)  : the simulation to calibrate
        data         (df)   : pandas dataframe (or dataframe-compatible dict) of the data to calibrate to
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum number of available CPUs)
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers
        reseed       (bool) : whether to generate new random seeds for each trial
        weights      (dict) : the relative weights of each data source
        fit_args     (dict) : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit
        sep          (str)  : the separate between different types of results, e.g. 'hiv.deaths' vs 'hiv_deaths'
        name         (str)  : the name of the database (default: 'starsim_calibration')
        db_name      (str)  : the name of the database file (default: 'starsim_calibration.db')
        keep_db      (bool) : whether to keep the database after calibration (default: false)
        storage      (str)  : the location of the database (default: sqlite)
        rand_seed    (int)  : if provided, use this random seed to initialize Optuna runs (for reproducibility)
        label        (str)  : a label for this calibration object
        die          (bool) : whether to stop if an exception is encountered (default: false)
        debug        (bool) : if True, do not run in parallel
        verbose      (bool) : whether to print details of the calibration

    Returns:
        A Calibration object
    """
    def __init__(self, sim, data, calib_pars, n_trials=None, n_workers=None, total_trials=None, reseed=True,
                 weights=None, fit_args=None, sep='.', name=None, db_name=None, keep_db=None, storage=None,
                 rand_seed=None, sampler=None, label=None, die=False, debug=False, verbose=True):

        # Handle run arguments
        if n_trials  is None: n_trials  = 20
        if n_workers is None: n_workers = sc.cpu_count()
        if name      is None: name      = 'starsim_calibration'
        if db_name   is None: db_name   = f'{name}.db'
        if keep_db   is None: keep_db   = False
        if storage   is None: storage   = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = int(np.ceil(total_trials/n_workers))
        kw = dict(n_trials=int(n_trials), n_workers=int(n_workers), debug=debug, name=name, db_name=db_name,
                  keep_db=keep_db, storage=storage, rand_seed=rand_seed, sampler=sampler)
        self.run_args = sc.objdict(kw)

        # Handle other inputs
        self.label      = label
        self.sim        = sim
        self.calib_pars = calib_pars
        self.reseed     = reseed
        self.sep        = sep
        self.weights    = sc.mergedicts(weights)
        self.fit_args   = sc.mergedicts(fit_args)
        self.die        = die
        self.verbose    = verbose
        self.calibrated = False
        self.before_sim = None
        self.after_sim  = None

        # Load data -- this is expecting a dataframe with a column for 'time' and other columns for to sim results
        self.data = ss.validate_sim_data(data, die=True)

        # Temporarily store a filename
        self.tmp_filename = 'tmp_calibration_%05i.obj'

        # Initialize sim
        if not self.sim.initialized:
            self.sim.init()

        # Figure out which sim results to get
        self.sim_result_list = self.data.cols

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

        if 'rand_seed' in calib_pars:
            sim.pars['rand_seed'] = calib_pars['rand_seed']

        for parname, spec in calib_pars.items():
            if parname == 'rand_seed':
                continue

            if 'path' not in spec:
                raise ValueError(f'Cannot map {parname} because "path" is missing from the parameter configuration.')

            p = spec['path']

            if len(p) != 3:
                raise ValueError(f'Cannot map {parname} because "path" must be a tuple of length 3.')

            modtype = p[0]
            dkey = p[1]
            dparkey = p[2]
            dparval = spec['value']
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
        for parname, spec in pars.items():

            if 'value' in spec:
                # Already have a value, likely running initial or final values as part of checking the fit
                continue

            if 'sampler' in spec:
                sampler = spec.pop('sampler')
                sampler_fn = getattr(trial, sampler)
            else:
                sampler_fn = trial.suggest_float

            path = spec.pop('path', None) # remove path
            guess = spec.pop('guess', None) # remove guess
            spec['value'] = sampler_fn(name=parname, **spec) # Sample!
            spec['path'] = path
            spec['guess'] = guess

        return pars

    @staticmethod
    def sim_to_df(sim): # TODO: remove this method
        """ Convert a sim to the expected dataframe type """
        df_res = sim.to_df(sep='.')
        df_res['t'] = df_res['timevec']
        df_res = df_res.set_index('t')
        df_res['time'] = np.floor(np.round(df_res.index, 1)).astype(int)
        return df_res

    def run_trial(self, trial, save=False):
        """ Define the objective for Optuna """
        if self.calib_pars is not None:
            calib_pars = self.trial_to_sim_pars(self.calib_pars, trial)
        else:
            calib_pars = None

        if self.reseed:
            calib_pars['rand_seed'] = trial.suggest_int('rand_seed', 0, 1_000_000) # Choose a random rand_seed

        sim = self.run_sim(calib_pars)

        # Export results # TODO: make more robust
        df_res = self.sim_to_df(sim)
        sim_results = sc.objdict()

        for skey in self.sim_result_list:
            if 'prevalence' in skey:
                model_output = df_res.groupby(by='time')[skey].mean()
            else:
                model_output = df_res.groupby(by='time')[skey].sum()
            sim_results[skey] = model_output.values

        sim_results['time'] = model_output.index.values
        # Store results in temporary files
        if save:
            filename = self.tmp_filename % trial.number
            sc.save(filename, sim_results)

        # Compute fit
        fit = self.compute_fit(df_res=df_res)
        return fit

    def compute_fit(self, sim=None, df_res=None):
        """ Compute goodness-of-fit """
        fit = 0

        # TODO: reduce duplication with above
        if df_res is None:
            df_res = self.sim_to_df(sim)
        for skey in self.sim_result_list:
            if 'prevalence' in skey:
                model_output = df_res.groupby(by='time')[skey].mean()
            else:
                model_output = df_res.groupby(by='time')[skey].sum()

            data = self.data[skey]
            combined = pd.merge(data, model_output, how='left', on='time')
            combined['diffs'] = combined[skey+'_x'] - combined[skey+'_y']
            gofs = compute_gof(combined.dropna()[skey+'_x'], combined.dropna()[skey+'_y'])

            losses = gofs  #* self.weights[skey]
            mismatch = losses.sum()
            fit += mismatch

        return fit

    def worker(self):
        """ Run a single worker """
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler = self.run_args.sampler)
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
                op.delete_study(study_name=self.run_args.name, storage=self.run_args.storage)
                if self.verbose: print(f'Deleted study {self.run_args.name} in {self.run_args.storage}')
        except Exception as E:
            if self.verbose:
                print('Could not delete study, skipping...')
                print(str(E))
        return

    def make_study(self):
        """ Make a study, deleting one if it already exists """
        if not self.run_args.keep_db:
            self.remove_db()
        if self.run_args.rand_seed is not None:
            sampler = op.samplers.RandomSampler(self.run_args.rand_seed)
            sampler.reseed_rng()
            raise NotImplementedError('Implemented but does not work')
        else:
            sampler = None
        if self.verbose: print(self.run_args.storage)
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler)
        return output

    def calibrate(self, calib_pars=None, confirm_fit=False, load=False, tidyup=True, **kwargs):
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
            if self.verbose: print('Loading saved results...')
            for trial in study.trials:
                n = trial.number
                try:
                    filename = self.tmp_filename % trial.number
                    results = sc.load(filename)
                    self.sim_results.append(results)
                    if tidyup:
                        try:
                            os.remove(filename)
                            if self.verbose: print(f'    Removed temporary file {filename}')
                        except Exception as E:
                            errormsg = f'Could not remove {filename}: {str(E)}'
                            if self.verbose: print(errormsg)
                    if self.verbose: print(f'  Loaded trial {n}')
                except Exception as E:
                    errormsg = f'Warning, could not load trial {n}: {str(E)}'
                    if self.verbose: print(errormsg)

        # Compare the results
        self.parse_study(study)

        if self.verbose: print('Best pars:', self.best_pars)

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

        if self.verbose: print('\nConfirming fit...')

        before_pars = sc.dcp(self.calib_pars)
        for spec in before_pars.values():
            spec['value'] = spec['guess'] # Use guess values

        after_pars = sc.dcp(self.calib_pars)
        for parname, spec in after_pars.items():
            spec['value'] = self.best_pars[parname]

        self.before_sim = self.run_sim(calib_pars=before_pars, label='Before calibration')
        self.after_sim  = self.run_sim(calib_pars=after_pars, label='After calibration')
        self.before_fit = self.compute_fit(self.before_sim)
        self.after_fit  = self.compute_fit(self.after_sim)

        # Add the data to the sims
        for sim in [self.before_sim, self.after_sim]:
            sim.init_data(self.data)

        print(f'Fit with original pars: {self.before_fit:n}')
        print(f'Fit with best-fit pars: {self.after_fit:n}')
        if self.after_fit <= self.before_fit:
            print('✓ Calibration improved fit')
        else:
            print('✗ Calibration did not improve fit, but this sometimes happens stochastically and is not necessarily an error')

        return self.before_fit, self.after_fit

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

    def plot_sims(self, **kwargs):
        """
        Plot sims, before and after calibration.

        Args:
            kwargs (dict): passed to MultiSim.plot()
        """
        if self.before_sim is None:
            self.comfirm_fit()
        msim = ss.MultiSim([self.before_sim, self.after_sim])
        fig = msim.plot(**kwargs)
        return fig

    def plot_trend(self, best_thresh=None, fig_kw=None):
        """
        Plot the trend in best mismatch over time.

        Args:
            best_thresh (int): Define the threshold for the "best" fits, relative to the lowest mismatch value (if None, show all)
            fig_kw (dict): passed to plt.figure()
        """
        df = self.df.sort_values('index') # Make a copy of the dataframe, sorted by trial number
        mismatch = sc.dcp(df['mismatch'].values)
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i+1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = plt.figure(**sc.mergedicts(fig_kw))

        ax1 = plt.subplot(2,1,1)
        plt.plot(mismatch, alpha=0.2, label='Original')
        plt.plot(smoothed_mismatch, lw=3, label='Smoothed')
        plt.plot(best_mismatch, lw=3, label='Best')

        ax2 = plt.subplot(2,1,2)
        max_mismatch = mismatch.min()*best_thresh if best_thresh is not None else np.inf
        inds = sc.findinds(mismatch<=max_mismatch)
        plt.plot(best_mismatch, lw=3, label='Best')
        plt.scatter(inds, mismatch[inds], c=mismatch[inds], label='Trials')
        for ax in [ax1, ax2]:
            plt.sca(ax)
            plt.grid(True)
            plt.legend()
            sc.setylim()
            sc.setxlim()
            plt.xlabel('Trial number')
            plt.ylabel('Mismatch')
        sc.figlayout()
        return fig