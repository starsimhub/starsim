"""
Utilities for running in parallel
"""
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

__all__ = ['MultiSim', 'single_run', 'multi_run', 'parallel']


class MultiSim:
    """
    Class for running multiple copies of a simulation.

    Args:
        sims (Sim/list): a single sim or a list of sims
        base_sim (Sim): the sim used for shared properties; if not supplied, the first of the sims provided
        label (str): the name of the multisim
        n_runs (int): if a single sim is provided, the number of replicates (default 4)
        initialize (bool): whether or not to initialize the sims (otherwise, initialize them during run)
        inplace (bool): whether to modify the sims in-place (default True); else return new sims
        debug (bool): if True, run in serial
        kwargs (dict): stored in run_args and passed to run()
    """
    def __init__(self, sims=None, base_sim=None, label=None, n_runs=4, initialize=False,
                 inplace=True, debug=False, **kwargs):
        # Handle inputs
        if base_sim is None:
            if isinstance(sims, ss.Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, list):
                base_sim = sims[0]
            else:
                errormsg = (f'If base_sim is not supplied, sims must be either a single sim'
                            f' (treated as base_sim) or a list of sims, not {type(sims)}')
                raise TypeError(errormsg)

        # Set properties
        self.sims = sims
        self.base_sim = base_sim
        self.label = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args = sc.mergedicts(dict(n_runs=n_runs, inplace=inplace, debug=debug), kwargs)
        self.results = None
        self.summary = None
        self.which = None  # Whether the multisim is to be reduced, combined, etc.
        self.timer = sc.timer() # Create a timer

        # Optionally initialize
        if initialize:
            self.init_sims()

        return

    def __len__(self):
        """ The length of a MultiSim is how many sims it contains """
        try:
            return len(self.sims)
        except:
            return 0

    def __repr__(self):
        """ Return a brief description of a multisim; see multisim.disp() for the more detailed version. """
        try:
            labelstr = f'"{self.label}"; ' if self.label else ''
            string   = f'MultiSim({labelstr}n_sims: {len(self)}; base: {self.base_sim})'
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, multisim appears to be malformed:\n{str(E)}'
        return string

    def brief(self):
        """ A single-line display of the MultiSim; same as print(multisim) """
        print(self)
        return

    def show(self, output=False):
        """
        Print a moderate length summary of the MultiSim. See also multisim.disp()
        (detailed output) and multisim.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = ss.MultiSim(ss.demo(run=False), label='Example multisim')
            msim.run()
            msim.show() # Prints moderate length output
        """
        labelstr = f' "{self.label}"' if self.label else ''
        simlenstr = f'{len(self)}'
        string  = f'MultiSim{labelstr} summary:\n'
        string += f'  Number of sims: {simlenstr}\n'
        string += f'  Reduced/combined: {self.which}\n'
        string += f'  Base: {self.base_sim}\n'
        if self.sims:
            string += '  Sims:\n'
            for s,sim in enumerate(self.sims):
                string += f'    {s}: {sim}\n'
        if not output:
            print(string)
            return
        else:
            return string

    def disp(self):
        """ Display the full object """
        return sc.pr(self)

    def init_sims(self, **kwargs):
        """
        Initialize the sims
        """

        # Handle which sims to use
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

        # Initialize the sims but don't run them
        kwargs = sc.mergedicts(self.run_args, kwargs, {'do_run': False})  # Never run, that's the point!
        kwargs.pop('inplace', None)
        kwargs.pop('debug', None)
        self.sims = multi_run(sims, **kwargs)

        return

    def run(self, **kwargs):
        """
        Run the sims; see ``ss.multi_run()`` for additional arguments

        Args:
            n_runs (int): how many replicates of each sim to run (if a list of sims is not provided)
            inplace (bool): whether to modify the sims in place (otherwise return copies)
            kwargs (dict): passed to multi_run(); use run_args to pass arguments to sim.run()

        Returns:
            None (modifies MultiSim object in place)
        """

        # Handle which sims to use -- same as init_sims()
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

            # Handle missing labels
            for s, sim in enumerate(sims):
                if sim.label is None:
                    sim.label = f'Sim {s}'

        # Run
        self.timer.start()
        kwargs = sc.mergedicts(self.run_args, kwargs)
        inplace = kwargs.pop('inplace', True)
        debug = kwargs.pop('debug', False)
        if debug:
            kwargs.pop('n_runs', None)
            kwargs.pop('iterpars', None)
            kwargs.pop('parallel', None)
            run_sims = [single_run(sim, **kwargs) for sim in sims]
        else: # The next line does all the work!
            run_sims = multi_run(sims, **kwargs) # Output sims are copies due to the pickling during parallelization

        # Handle output
        if inplace and isinstance(self.sims, list) and len(run_sims) == len(self.sims): # Validation
            for old,new in zip(self.sims, run_sims):
                old.__dict__.update(new.__dict__) # Update the same object with the new results
        self.sims = run_sims # Just overwrite references
        self.timer.stop()

        return self

    def _has_orig_sim(self):
        """ Helper method for determining if an original base sim is present """
        return hasattr(self, 'orig_base_sim')

    def _rm_orig_sim(self, reset=False):
        """ Helper method for removing the original base sim, if present """
        if self._has_orig_sim():
            if reset:
                self.base_sim = self.orig_base_sim
            delattr(self, 'orig_base_sim')
        return

    def shrink(self, **kwargs):
        """
        Not to be confused with reduce(), this shrinks each sim in the msim;
        see sim.shrink() for more information.

        Args:
            kwargs (dict): passed to sim.shrink() for each sim
        """
        self.base_sim.shrink(**kwargs)
        self._rm_orig_sim()
        for sim in self.sims:
            sim.shrink(**kwargs)
        return

    def reset(self):
        """ Undo reduce() by resetting the base sim, which, and results """
        self._rm_orig_sim(reset=True)
        self.which = None
        self.results = None
        return

    def reduce(self, quantiles=None, use_mean=False, bounds=None, output=False):
        """
        Combine multiple sims into a single sim statistically: by default, use
        the median value and the 10th and 90th percentiles for the lower and upper
        bounds. If use_mean=True, then use the mean and ±2 standard deviations
        for lower and upper bounds.

        Args:
            quantiles (dict): the quantiles to use, e.g. [0.1, 0.9] or {'low : '0.1, 'high' : 0.9}
            use_mean (bool): whether to use the mean instead of the median
            bounds (float): if use_mean=True, the multiplier on the standard deviation for upper and lower bounds (default 2)
            output (bool): whether to return the "reduced" sim (in any case, modify the multisim in-place)

        **Example**::

            msim = ss.MultiSim(ss.Sim())
            msim.run()
            msim.reduce()
            msim.summarize()
        """
        if use_mean:
            if bounds is None:
                bounds = 2
        else:
            if quantiles is None:
                quantiles = {'low': 0.1, 'high': 0.9}
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low': float(quantiles[0]), 'high': float(quantiles[1])}
                except Exception as E:
                    errormsg = (f'Could not figure out how to convert {quantiles} into a quantiles object:'
                                f' must be a dict with keys low, high or a 2-element array ({str(E)})')
                    raise ValueError(errormsg) from E

        # Store information on the sims
        n_runs = len(self)
        reduced_sim = sc.dcp(self.sims[0])
        reduced_sim.metadata = dict(parallelized=True, combined=False, n_runs=n_runs, quantiles=quantiles,
                                    use_mean=use_mean, bounds=bounds)  # Store how this was parallelized

        # Calculate the statistics
        raw = {}

        rflat = reduced_sim.results.flatten()
        rkeys = list(rflat.keys())
        for rkey in rkeys:
            raw[rkey] = np.zeros((len(rflat[rkey]), len(self.sims)))
            for s, sim in enumerate(self.sims):
                flat = sim.results.flatten()
                raw[rkey][:, s] = flat[rkey]

        for rkey in rkeys:
            res = rflat[rkey]
            if use_mean:
                r_mean = np.mean(raw[rkey], axis=1)
                r_std = np.std(raw[rkey], axis=1)
                res[:] = r_mean
                res.low = r_mean - bounds * r_std
                res.high = r_mean + bounds * r_std
            else:
                res[:] = np.quantile(raw[rkey], q=0.5, axis=1)
                res.low = np.quantile(raw[rkey], q=quantiles['low'], axis=1)
                res.high = np.quantile(raw[rkey], q=quantiles['high'], axis=1)

        # Compute and store final results
        reduced_sim.summarize()
        self.orig_base_sim = self.base_sim
        self.base_sim = reduced_sim
        self.results = ss.Results('MultiSim').merge(rflat) # Create the dictionary and merge it
        self.summary = reduced_sim.summary
        self.which = 'reduced'

        if output:
            return self.base_sim
        else:
            return

    def mean(self, bounds=None, **kwargs):
        """
        Alias for reduce(use_mean=True). See reduce() for full description.

        Args:
            bounds (float): multiplier on the standard deviation for the upper and lower bounds (default, 2)
            kwargs (dict): passed to reduce()
        """
        return self.reduce(use_mean=True, bounds=bounds, **kwargs)

    def median(self, quantiles=None, **kwargs):
        """
        Alias for reduce(use_mean=False). See reduce() for full description.

        Args:
            quantiles (list or dict): upper and lower quantiles (default, 0.1 and 0.9)
            kwargs (dict): passed to reduce()
        """
        return self.reduce(use_mean=False, quantiles=quantiles, **kwargs)

    def summarize(self, method='mean', quantiles=None, how='default'):
        """
        Summarize the simulations statistically.

        Args:
            method (str): one of 'mean' (default: [mean, 2*std]), 'median' ([median, min, max]), or 'all' (all results)
            quantiles (dict): if method='median', use these quantiles
            how (str): passed to sim.summarize()
        """

        # Compute the summaries
        summaries = []
        for sim in self.sims:
            summaries.append(sim.summarize(how=how))

        summary = sc.dcp(summaries[0]) # Use the first one as a template
        for k in summary.keys():
            arr = np.array([s[k] for s in summaries])
            if method == 'all':
                summary[k] = arr
            elif method == 'mean':
                summary[k] = sc.objdict({'mean':arr.mean(), 'std':arr.std(), 'sem':sc.sem(arr)})
            elif method == 'median':
                if quantiles is None:
                    quantiles = sc.objdict({'median':0.5, 'min':0, 'max':1, 'q25':0.25, 'q75':0.75})
                elif isinstance(quantiles, list):
                    quantiles = {q:q for q in quantiles}
                summary[k] = {q:v for q,v in zip(quantiles, np.quantile(arr, quantiles))}

        self.summary = summary # Could reconcile with reduce()'s summary

        return summary

    def plot(self, key=None, fig=None, fig_kw=None, plot_kw=None, fill_kw=None):
        """
        Plot all results in the MultiSim object.

        If the MultiSim object has been reduced (i.e. mean or median), then plot
        the best value and uncertainty bound. Otherwise, plot individual sims.

        Args:
            key (str): the results key to plot (by default, all)
            fig (Figure): if provided, plot results into an existing figure
            fig_kw (dict): passed to ``plt.subplots()``
            plot_kw (dict): passed to ``plt.plot()``
            fill_kw (dict): passed to ``plt.fill_between()``
        """
        # Has not been reduced yet, plot individual sim
        if self.which is None:
            fig = None
            alpha = 0.7 if len(self) < 5 else 0.5
            plot_kw = sc.mergedicts({'alpha':alpha}, plot_kw)
            with ss.options.context(jupyter=False): # Always return the figure
                for sim in self.sims:
                    fig = sim.plot(key=key, fig=fig, fig_kw=fig_kw, plot_kw=plot_kw)
            plt.legend()

        # Has been reduced, plot with uncertainty bounds
        else:
            flat = self.results
            n_cols = np.ceil(np.sqrt(len(flat))) # TODO: remove duplication with sim.plot()
            default_figsize = np.array([8, 6])
            figsize_factor = np.clip((n_cols-3)/6+1, 1, 1.5) # Scale the default figure size based on the number of rows and columns
            figsize = default_figsize*figsize_factor
            fig_kw = sc.mergedicts({'figsize':figsize}, fig_kw)
            fig_kw = sc.mergedicts(fig_kw)
            fill_kw = sc.mergedicts({'alpha':0.2}, fill_kw)
            plot_kw = sc.mergedicts({'lw':2, 'alpha':0.8}, plot_kw)
            with sc.options.with_style('simple'):
                if key is not None:
                    flat = {k:v for k,v in flat.items() if k.startswith(key)}
                if fig is None:
                    fig, axs = sc.getrowscols(len(flat), make=True, **fig_kw)
                else:
                    axs = sc.toarray(fig.axes)

                # Do the plotting
                for ax, (key, res) in zip(axs.flatten(), flat.items()):
                    ax.fill_between(res.timevec, res.low, res.high, **fill_kw)
                    ax.plot(res.timevec, res, **plot_kw)
                    ax.set_title(getattr(res, 'label', key))
                    ax.set_xlabel('Year')

        return ss.return_fig(fig)


def single_run(sim, ind=0, reseed=True, shrink=True, run_args=None, sim_args=None,
               verbose=None, do_run=True, **kwargs):
    """
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly.

    Args:
        sim         (Sim)   : the sim instance to be run
        ind         (int)   : the index of this sim
        reseed      (bool)  : whether to generate a fresh seed for each run
        noise       (float) : the amount of noise to add to each run
        noisepar    (str)   : the name of the parameter to add noise to
        shrink      (bool)  : whether to shrink the sim after the sim run
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim, e.g. 'n_infected'
        verbose     (int)   : detail to print
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        kwargs      (dict)  : also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    **Example**::

        import starsim as ss
        sim = ss.Sim() # Create a default simulation
        sim = ss.single_run(sim) # Run it, equivalent(ish) to sim.run()
    """

    # Set sim and run arguments
    sim_args = sc.mergedicts(sim_args, kwargs)
    run_args = sc.mergedicts({'verbose': verbose}, run_args)
    if verbose is None:
        verbose = sim.pars['verbose']

    if not sim.label:
        sim.label = f'Sim {ind}'

    if reseed:
        sim.pars['rand_seed'] += ind  # Reset the seed, otherwise no point of parallel runs
        ss.set_seed() # Note: may not be needed

    # Handle additional arguments
    for key, val in sim_args.items():
        if key in sim.pars.keys():
            if verbose >= 1:
                print(f'Setting key {key} from {sim[key]} to {val}')
            sim.pars[key] = val
            if key == 'rand_seed':
                ss.set_seed() # Note: may not be needed
        else:
            raise sc.KeyNotFoundError(f'Could not set key {key}: not a valid parameter name')

    # Run
    if do_run:
        sim.run(**run_args)

    # Shrink the sim to save memory
    if shrink:
        sim.shrink()

    return sim


def multi_run(sim, n_runs=4, reseed=None, iterpars=None, shrink=None, run_args=None, sim_args=None,
              par_args=None, do_run=True, parallel=True, n_cpus=None, verbose=None, **kwargs):
    """
    For running multiple sims in parallel. If the first argument is a list of sims
    rather than a single sim, exactly these will be run and most other arguments
    will be ignored.

    Args:
        sim         (Sim/list): the sim instance to be run, or a list of sims.
        n_runs      (int)   : the number of parallel runs
        reseed      (bool)  : whether or not to generate a fresh seed for each run (default: true for single, false for list of sims)
        iterpars    (dict)  : any other parameters to iterate over the runs; see sc.parallelize() for syntax
        shrink      (bool)  : whether to shrink the sim after the sim run
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim
        par_args    (dict)  : arguments passed to sc.parallelize()
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        parallel    (bool)  : whether to run in parallel using multiprocessing (else, just run in a loop)
        n_cpus      (int)   : the number of CPUs to run on (if blank, set automatically; otherwise, passed to par_args)
        verbose     (int)   : detail to print
        kwargs      (dict)  : also passed to the sim

    Returns:
        If combine is True, a single sim object with the combined results from each sim.
        Otherwise, a list of sim objects (default).

    **Example**::

        import starsim as ss
        sim = ss.Sim()
        sims = ss.multi_run(sim, n_runs=6)
    """

    # Handle inputs
    sim_args = sc.mergedicts(sim_args, kwargs)  # Handle blank
    par_args = sc.mergedicts({'ncpus': n_cpus}, par_args)  # Handle blank

    # Handle iterpars
    if iterpars is None:
        iterpars = {}
    else:
        n_runs = None  # Reset and get from length of dict instead
        for key, val in iterpars.items():
            new_n = len(val)
            if n_runs is not None and new_n != n_runs:
                raise ValueError(f'Each entry in iterpars must have the same length, not {n_runs} and {len(val)}')
            else:
                n_runs = new_n

    # Run the sims
    if isinstance(sim, ss.Sim):  # One sim
        if reseed is None: reseed = True
        iterkwargs = dict(ind=np.arange(n_runs))
        iterkwargs.update(iterpars)
        kwargs = dict(sim=sim, reseed=reseed, verbose=verbose, shrink=shrink,
                      sim_args=sim_args, run_args=run_args, do_run=do_run)
    elif isinstance(sim, list):  # List of sims
        if reseed is None: reseed = False
        iterkwargs = dict(sim=sim, ind=np.arange(len(sim)))
        kwargs = dict(reseed=reseed, verbose=verbose, shrink=shrink, sim_args=sim_args, run_args=run_args,
                      do_run=do_run)
    else:
        errormsg = f'Must be Sim object or list, not {type(sim)}'
        raise TypeError(errormsg)

    # Actually run
    if parallel:
        try:
            sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs, **par_args)  # Run in parallel
        except RuntimeError as E:  # Handle if run outside __main__ on Windows
            if 'freeze_support' in E.args[0]:  # For this error, add additional information
                errormsg = '''
 Uh oh! It appears you are trying to run with multiprocessing on Windows outside
 of the __main__ block; please see https://docs.python.org/3/library/multiprocessing.html
 for more information. The correct syntax to use is e.g.

     import starsim as ss
     sim = ss.Sim()
     msim = ss.MultiSim(sim)

     if __name__ == '__main__':
         msim.run()

Alternatively, to run without multiprocessing, set parallel=False.
 '''
                raise RuntimeError(errormsg) from E
            else:  # For all other runtime errors, raise the original exception
                raise E
    else:  # Run in serial, not in parallel
        sims = []
        n_sims = len(list(iterkwargs.values())[0])  # Must have length >=1 and all entries must be the same length
        for s in range(n_sims):
            this_iter = {k: v[s] for k, v in iterkwargs.items()}  # Pull out items specific to this iteration
            this_iter.update(kwargs)  # Merge with the kwargs
            this_iter['sim'] = this_iter[
                'sim'].copy()  # Ensure we have a fresh sim; this happens implicitly on pickling with multiprocessing
            sim = single_run(**this_iter)  # Run in series
            sims.append(sim)

    return sims


def parallel(*args, **kwargs):
    """
    A shortcut to ``ss.MultiSim()``, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        kwargs (dict): passed to multi_run()

    Returns:
        A run MultiSim object.

    **Examples**::

        s1 = ss.Sim(n_agents=1000, label='Small', diseases='sis', networks='random')
        s2 = ss.Sim(n_agents=2000, label='Large', diseases='sis', networks='random')
        ss.parallel(s1, s2).plot()
        msim = ss.parallel([s1, s2], shrink=False)
    """
    sims = sc.mergelists(*args)
    msim = MultiSim(sims=sims, **kwargs)
    msim.run()
    return msim
