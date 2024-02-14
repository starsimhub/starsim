"""
Utilities for running in parallel
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['MultiSim', 'single_run', 'multi_run', 'parallel']


class MultiSim(sc.prettyobj):
    """
    Class for running multiple copies of a simulation.
    """

    def __init__(self, sims=None, base_sim=None, label=None, initialize=False, *args, **kwargs):

        # Handle inputs
        super().__init__(*args, **kwargs)
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
        self.run_args = sc.mergedicts(kwargs)
        self.results = None
        self.which = None  # Whether the multisim is to be reduced, combined, etc.

        # Optionally initialize
        if initialize:
            self.init_sims()

        return

    def __len__(self):
        return len(self.sims)

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
        self.sims = multi_run(sims, **kwargs)

        return

    def run(self, reduce=False, **kwargs):
        """
        Run the sims

        Args:
            reduce  (bool): whether to reduce after running (see reduce())
            kwargs  (dict): passed to multi_run(); use run_args to pass arguments to sim.run()

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
        kwargs = sc.mergedicts(self.run_args, kwargs)
        self.sims = multi_run(sims, **kwargs)

        # Reduce
        if reduce:
            self.reduce()

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
                    raise ValueError(errormsg)

        # Store information on the sims
        n_runs = len(self)
        reduced_sim = sc.dcp(self.sims[0])
        reduced_sim.metadata = dict(parallelized=True, combined=False, n_runs=n_runs, quantiles=quantiles,
                                    use_mean=use_mean, bounds=bounds)  # Store how this was parallelized

        # Perform the statistics
        raw = {}

        rkeys = reduced_sim.results.keys()

        for rkey in rkeys:
            raw[rkey] = np.zeros((reduced_sim.res_npts, len(self.sims)))
            for s, sim in enumerate(self.sims):
                vals = sim.results[rkey].values
                raw[rkey][:, s] = vals

        for rkey in rkeys:
            results = reduced_sim.results
            if use_mean:
                r_mean = np.mean(raw[rkey], axis=1)
                r_std = np.std(raw[rkey], axis=1)
                results[rkey].values[:] = r_mean
                results[rkey].low = r_mean - bounds * r_std
                results[rkey].high = r_mean + bounds * r_std
            else:
                results[rkey].values[:] = np.quantile(raw[rkey], q=0.5, axis=1)
                results[rkey].low = np.quantile(raw[rkey], q=quantiles['low'], axis=1)
                results[rkey].high = np.quantile(raw[rkey], q=quantiles['high'], axis=1)

        # Compute and store final results
        reduced_sim.compute_summary()
        self.orig_base_sim = self.base_sim
        self.base_sim = reduced_sim
        self.results = reduced_sim.results
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


def single_run(sim, ind=0, reseed=True, keep_people=False, run_args=None, sim_args=None,
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
        keep_people (bool)  : whether to keep the people after the sim run
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
        ss.set_seed()

    if verbose >= 1:
        verb = 'Running' if do_run else 'Creating'
        print(f'{verb} a simulation using seed={sim["rand_seed"]}')

    # Handle additional arguments
    for key, val in sim_args.items():
        print(f'Processing {key}:{val}')
        if key in sim.pars.keys():
            if verbose >= 1:
                print(f'Setting key {key} from {sim[key]} to {val}')
                sim[key] = val
        else:
            raise sc.KeyNotFoundError(f'Could not set key {key}: not a valid parameter name')

    # Run
    if do_run:
        sim.run(**run_args)

    # Shrink the sim to save memory
    if not keep_people:
        sim.shrink()

    return sim


def multi_run(sim, n_runs=4, reseed=None, iterpars=None, keep_people=None, run_args=None, sim_args=None,
              par_args=None, do_run=True, parallel=True, n_cpus=None, verbose=None, **kwargs):
    """
    For running multiple runs in parallel. If the first argument is a list of sims,
    exactly these will be run and most other arguments will be ignored.

    Args:
        sim         (Sim)   : the sim instance to be run, or a list of sims.
        n_runs      (int)   : the number of parallel runs
        reseed      (bool)  : whether or not to generate a fresh seed for each run (default: true for single, false for list of sims)
        iterpars    (dict)  : any other parameters to iterate over the runs; see sc.parallelize() for syntax
        keep_people (bool)  : whether to keep the people after the sim run (default false)
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
        kwargs = dict(sim=sim, reseed=reseed, verbose=verbose, keep_people=keep_people,
                      sim_args=sim_args, run_args=run_args, do_run=do_run)
    elif isinstance(sim, list):  # List of sims
        if reseed is None: reseed = False
        iterkwargs = dict(sim=sim, ind=np.arange(len(sim)))
        kwargs = dict(reseed=reseed, verbose=verbose, keep_people=keep_people, sim_args=sim_args, run_args=run_args,
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
    A shortcut to ``hpv.MultiSim()``, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        kwargs (dict): passed to multi_run()

    Returns:
        A run MultiSim object.

    **Examples**::

        s1 = ss.Sim(beta=0.01, label='Low')
        s2 = ss.Sim(beta=0.02, label='High')
        ss.parallel(s1, s2).plot()
        msim = ss.parallel([s1, s2], keep_people=True)
    """
    sims = sc.mergelists(*args)
    return MultiSim(sims=sims).run(**kwargs)

