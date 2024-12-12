"""
Define the calibration class
"""
import os
import numpy as np
import optuna as op
import optuna.visualization.matplotlib as vis
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt


__all__ = ['Calibration']


class Calibration(sc.prettyobj):
    """
    A class to handle calibration of Starsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Args:
        sim          (Sim)   : the base simulation to calibrate
        calib_pars   (dict)  : a dictionary of the parameters to calibrate of the format dict(key1=dict(low=1, high=2, guess=1.5, **kwargs), key2=...), where kwargs can include "suggest_type" to choose the suggest method of the trial (e.g. suggest_float) and args passed to the trial suggest function like "log" and "step"
        n_workers    (int)   : the number of parallel workers (if None, will use all available CPUs)
        total_trials (int)   : the total number of trials to run, each worker will run approximately n_trials = total_trial / n_workers
        reseed       (bool)  : whether to generate new random seeds for each trial
        build_fn  (callable) : function that takes a sim object and calib_pars dictionary and returns a modified sim
        build_kw      (dict) : a dictionary of options that are passed to build_fn to aid in modifying the base simulation. The API is self.build_fn(sim, calib_pars=calib_pars, **self.build_kw), where sim is a copy of the base simulation to be modified with calib_pars
        components    (list) : CalibComponents independently assess pseudo-likelihood as part of evaluating the quality of input parameters
        eval_fn   (callable) : Function mapping a sim to a float (e.g. negative log likelihood) to be maximized. If None, the default will use CalibComponents.
        eval_kw       (dict) : Additional keyword arguments to pass to the eval_fn
        label        (str)   : a label for this calibration object
        study_name   (str)   : name of the optuna study
        db_name      (str)   : the name of the database file (default: 'starsim_calibration.db')
        keep_db      (bool)  : whether to keep the database after calibration (default: false)
        storage      (str)   : the location of the database (default: sqlite)
        sampler (BaseSampler): the sampler used by optuna, like optuna.samplers.TPESampler
        die          (bool)  : whether to stop if an exception is encountered (default: false)
        debug        (bool)  : if True, do not run in parallel
        verbose      (bool)  : whether to print details of the calibration

    Returns:
        A Calibration object
    """
    def __init__(self, sim, calib_pars, n_workers=None, total_trials=None, reseed=True,
                 build_fn=None, build_kw=None, eval_fn=None, eval_kw=None, components=None,
                 label=None, study_name=None, db_name=None, keep_db=None, storage=None,
                 sampler=None, die=False, debug=False, verbose=True):

        # Handle run arguments
        if total_trials is None: total_trials   = 100
        if n_workers    is None: n_workers      = 1 if debug else sc.cpu_count()
        if study_name   is None: study_name     = 'starsim_calibration'
        if db_name      is None: db_name        = f'{study_name}.db'
        if keep_db      is None: keep_db        = False
        if storage      is None: storage        = f'sqlite:///{db_name}'
        
        self.build_fn       = build_fn or self.translate_pars
        self.build_kw       = build_kw or dict()
        self.eval_fn        = eval_fn or self._eval_fit
        self.eval_kw        = eval_kw or dict()
        self.components     = sc.tolist(components)

        n_trials = int(np.ceil(total_trials/n_workers))
        kw = dict(n_trials=n_trials, n_workers=int(n_workers), debug=debug, study_name=study_name,
                  db_name=db_name, keep_db=keep_db, storage=storage, sampler=sampler)
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

        # Temporarily store a filename for storing intermediate results
        self.tmp_filename = 'tmp_calibration_%06i.obj'
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
        figs = []
        for component in self.components:
            fig = component.plot(**kwargs)
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
        """ Make a study, deleting one if it already exists """
        if not self.run_args.keep_db:
            self.remove_db()
        if self.verbose: print(self.run_args.storage)
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.study_name, direction='minimize')
        return output

    def calibrate(self, calib_pars=None, load=False, tidyup=True, **kwargs):
        """
        Perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
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
        self.study = self.make_study()
        self.run_workers()
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
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

        return self

    def check_fit(self, n_runs=10, do_plot=True):
        """ Run before and after simulations to validate the fit """
        if self.verbose: sc.printcyan('\nChecking fit...')

        before_pars = sc.dcp(self.calib_pars)
        for spec in before_pars.values():
            spec['value'] = spec['guess'] # Use guess values

        # Load in case calibration was interrupted
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
        self.best_pars = sc.objdict(study.best_params)

        after_pars = sc.dcp(self.calib_pars)
        for parname, spec in after_pars.items():
            spec['value'] = self.best_pars[parname]

        self.before_msim = self.build_fn(self.sim.copy(), calib_pars=before_pars, n_reps=n_runs, **self.build_kw)
        self.after_msim = self.build_fn(self.sim.copy(), calib_pars=after_pars, n_reps=n_runs, **self.build_kw)

        fix_before = isinstance(self.before_msim, ss.Sim)
        fix_after = isinstance(self.after_msim, ss.Sim)
        if fix_after or fix_after:
            ss.warn(f'Calibration was expecting the build function to return a MultiSim, but instead got a single Sim; wrapping it in a MultiSim with {n_runs} runs')

            if fix_before:
                self.before_msim = ss.MultiSim(self.before_msim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, n_runs)), initialize=True, debug=True, parallel=False)

            if fix_after:
                self.after_msim = ss.MultiSim(self.after_msim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, n_runs)), initialize=True, debug=True, parallel=False)

        for sim in self.before_msim.sims: sim.calibrated = False
        for sim in self.after_msim.sims: sim.calibrated = True
        msim = ss.MultiSim(self.before_msim.sims + self.after_msim.sims)
        msim.run()
        self.before_fits = self.eval_fn(self.before_msim, **self.eval_kw)
        self.after_fits = self.eval_fn(self.after_msim, **self.eval_kw)
        fits = self.eval_fn(msim, **self.eval_kw)

        if do_plot:
            figs = self.plot()

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

    def plot_final(self, n_runs=10, **kwargs):
        """
        Plot sims after calibration

        Args:
            kwargs (dict): passed to MultiSim.plot()
        """

        jup = ss.options.jupyter if 'jupyter' in ss.options else sc.isjupyter()
        ss.options.jupyter = False

        pars = sc.dcp(self.calib_pars)
        for parname, spec in pars.items():
            spec['value'] = self.best_pars[parname]
        msim = self.build_fn(self.sim.copy(), calib_pars=pars, n_reps=n_runs, **self.build_kw)

        if isinstance(msim, ss.Sim):
            ss.warn(f'Calibration was expecting the build function to return a MultiSim, but instead got a single Sim; wrapping it in a MultiSim with {n_runs} runs')
            msim = ss.MultiSim(msim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, n_runs)), initialize=True, debug=True, parallel=False)

        #for sim in msim.sims: sim.label = 'Calibration'
        msim.run()
        fits = self.eval_fn(msim, **self.eval_kw)

        msim.reduce()
        fig = msim.plot()
        fig.suptitle('After calibration')

        ss.options.jupyter = jup
        return fig

    def plot_optuna(self, methods=None):
        """ Plot Optuna's visualizations """
        figs = []

        if methods is None:
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
                fig = getattr(vis, method)(self.study)
                figs.append(fig)
            except Exception as E:
                print(f'Could not run {method}: {str(E)}')
        return figs