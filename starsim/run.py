"""
Utilities for running in parallel
"""
import numpy as np
import sciris as sc
import starsim as ss



class MultiSim:
    """
    Class for running multiple copies of a simulation in parallel.
    
    This is the most common way for running multiple simulation: for example, to
    do scenario analysis or to collect statistics

    Args:
        sims (Sim/list): a single sim, a list/tuple of sims, or another MultiSim (or list of MultiSims)
        base_sim (Sim): the sim used for shared properties; if not supplied, the first of the sims provided
        label (str): the name of the multisim
        n_runs (int): if a single sim is provided, the number of replicates (default 4)
        initialize (bool): whether or not to initialize the sims (otherwise, initialize them during run)
        inplace (bool): whether to modify the sims in-place (default True); else return new sims
        debug (bool): if True, run in serial
        kwargs (dict): stored in run_args and passed to run()
    
    Examples:
        ```python
        import starsim as ss
        
        s1 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.02), label='Low transmission')
        s2 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.05), label='Medium transmission')
        s3 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.10), label='High transmission')
        
        msim = ss.MultiSim(sims=[s1, s2, s3])
        msim.run()

        # Plot individual sims
        msim.plot()
        
        # Calculate mean results across sims and plot
        msim.mean()
        msim.plot()
        ```
    """
    def __init__(self, sims=None, base_sim=None, label=None, n_runs=4, initialize=False,
                 inplace=True, debug=False, **kwargs):
        
        # Handle MultiSim input
        if isinstance(sims, MultiSim):
            if base_sim is None: # Extract base sim if not supplied
                base_sim = sims.base_sim
            sims = sims.sims # Extract actual sims
        
        # Handle sims as a list (potentially mixed list of sims and MultiSims)
        elif isinstance(sims, (list, tuple)):
            merged_sims = []
            for entry in sims:
                if isinstance(entry, ss.Sim): # Standard case: use directly
                    merged_sims.append(entry)
                elif isinstance(entry, ss.MultiSim): # Extract sims
                    merged_sims.extend(entry.sims)
                elif isinstance(entry, (list, tuple)): # Merge with current list (rare)
                    merged_sims.extend(list(entry)) # Does not handle doubly nested MultiSim, and that's ok!
                else:
                    errormsg = f'Unable to process sim {entry}: expecting ss.Sim or ss.MultiSim'
                    raise TypeError(errormsg)
            sims = merged_sims
            
        # Handle base_sim being empty
        if base_sim is None:
            if isinstance(sims, ss.Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, (list, tuple)):
                if not sims:
                    errormsg = 'You must supply at least one sim to create a MultiSim'
                    raise ValueError(errormsg)
                base_sim = sims[0]
            else:
                errormsg = 'If base_sim is not supplied, sims must be either a single sim '
                errormsg += f'(treated as base_sim) or a list/tuple of sims, not {type(sims)}'
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

        Examples:
            ```python
            msim = ss.MultiSim(ss.demo(run=False), label='Example multisim')
            msim.run()
            msim.show() # Prints moderate length output
            ```
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
        Run the sims; see `ss.multi_run()` for additional arguments

        Args:
            n_runs (int): how many replicates of each sim to run (if a list of sims is not provided)
            inplace (bool): whether to modify the sims in place (otherwise return copies)
            kwargs (dict): passed to multi_run(); use run_args to pass arguments to sim.run()

        Returns:
            None (modifies MultiSim object in place)
        """

        # Handle which sims to use -- same as init_sims()
        if self.sims is None:
            run_target = self.base_sim # Pass the single sim to multi_run() so n_runs is honored (rather than a 1-element list, which it would run as-is)
        else:
            run_target = self.sims

            # Handle missing labels
            for s, sim in enumerate(self.sims):
                if sim.label is None:
                    sim.label = f'Sim {s}'

        # Run
        self.timer.start()
        kwargs = sc.mergedicts(self.run_args, kwargs)
        inplace = kwargs.pop('inplace', True)
        debug = kwargs.pop('debug', False)
        if debug:
            kwargs['parallel'] = False # Run in serial
        run_sims = multi_run(run_target, **kwargs) # This does all the work! Output sims are copies due to the pickling during parallelization

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
    
    def copy(self, die=True):
        """ Perform a deep copy of the MultiSim (including sims contained within)

        Args:
            die (bool): whether to raise an exception if copy fails (else, try a shallow copy)
        """
        out = sc.dcp(self, die=die)
        return out

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

        Examples:
            ```python
            msim = ss.MultiSim(ss.Sim())
            msim.run()
            msim.reduce()
            msim.summarize()
            ```
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
        length_mismatches = sc.ddict(int)
        for rkey in rkeys:
            raw[rkey] = np.full(rflat[rkey].shape + (len(self.sims),), np.nan) # Shape (npts, nsims), or (npts, ncols, nsims) for 2D results
            for s, sim in enumerate(self.sims):
                flat = sim.results.flatten()
                this_raw = raw[rkey]
                this_flat = flat[rkey]
                l1 = this_raw.shape[0]
                l2 = this_flat.shape[0]
                if l1 == l2:
                    length = l1
                else:
                    length_mismatches[sim.label] += 1
                    length = min(l1, l2)
                this_raw[:length, ..., s] = this_flat[:length]
        if length_mismatches:
            warnmsg = 'Sim results have mismatched lengths; results have been truncated but are not necessarily aligned. Mismatches:\n'
            for k,v in length_mismatches.items():
                warnmsg += f'{k}: {v} mismatched results\n'
            ss.warn(warnmsg)

        for rkey in rkeys:
            res = rflat[rkey]
            if use_mean:
                r_mean = np.mean(raw[rkey], axis=-1)
                r_std = np.std(raw[rkey], axis=-1)
                res[:] = r_mean
                res.low = r_mean - bounds * r_std
                res.high = r_mean + bounds * r_std
            else:
                res[:] = np.quantile(raw[rkey], q=0.5, axis=-1)
                res.low = np.quantile(raw[rkey], q=quantiles['low'], axis=-1)
                res.high = np.quantile(raw[rkey], q=quantiles['high'], axis=-1)

        # Compute and store final results
        reduced_sim.summarize()
        if not self._has_orig_sim(): # Don't overwrite the original if reducing again
            self.orig_base_sim = self.base_sim
        self.base_sim = reduced_sim
        self.results = MultiResults('MultiSim', nested=getattr(self.results, 'nested', None)).merge(rflat) # Create the dictionary and merge it
        self.summary = reduced_sim.summary
        self.which = 'reduced'

        if output:
            return self.base_sim
        else:
            return self

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

    def combine(self, output=False):
        """
        Combine multiple sims into a single sim, e.g. to treat several smaller sims
        as one larger population.

        Results that scale with population size (`res.scale=True`, e.g. counts) are
        summed; other results (e.g. prevalence) are averaged, weighted by each sim's
        `total_pop`. The population sizes (`n_agents` and `total_pop`) are summed, and
        `pop_scale` is updated to match. The people are not combined: the combined
        sim keeps the people (if any) of the first sim.

        Args:
            output (bool): whether to return the combined sim (otherwise, return the MultiSim)

        Examples:
            ```python
            msim = ss.MultiSim(ss.Sim(n_agents=1e3, diseases='sis', networks='random'), n_runs=4)
            msim.run()
            msim.combine() # Equivalent to a single sim with n_agents=4e3
            msim.plot()
            ```
        """
        # Combine the population sizes
        n_runs = len(self)
        combined_sim = sc.dcp(self.sims[0])
        combined_sim.metadata = dict(parallelized=True, combined=True, n_runs=n_runs) # Store how this was parallelized
        pars = combined_sim.pars
        pops = np.array([sim.pars.total_pop for sim in self.sims]) # Used to weight the non-count results
        pars.n_agents = sum(sim.pars.n_agents for sim in self.sims)
        pars.total_pop = pops.sum()
        pars.pop_scale = pars.total_pop/pars.n_agents

        # Combine the results
        flats = [sim.results.flatten() for sim in self.sims]
        cflat = combined_sim.results.flatten()
        for key,res in cflat.items():
            vals = [flat[key].values for flat in flats]
            if any(len(v) != len(res) for v in vals):
                errormsg = f'Cannot combine sims with inconsistent lengths for result "{key}": {[len(v) for v in vals]}'
                raise ValueError(errormsg)
            raw = np.array(vals) # Shape (n_runs, npts), or (n_runs, npts, ncols) for 2D results
            res[:] = raw.sum(axis=0) if res.scale else np.average(raw, axis=0, weights=pops)
            res.low = None # Bounds (e.g. from reduced sims) don't apply to the combined result
            res.high = None

        # Compute and store final results
        combined_sim.summarize()
        if not self._has_orig_sim():
            self.orig_base_sim = self.base_sim
        self.base_sim = combined_sim
        self.results = ss.Results('MultiSim').merge(cflat) # As in reduce()
        self.summary = combined_sim.summary
        self.which = 'combined'

        if output:
            return self.base_sim
        else:
            return self

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
                summary[k] = {q: np.quantile(arr, v) for q, v in quantiles.items()}

        self.summary = summary # Could reconcile with reduce()'s summary

        return summary

    def compare(self, t=None, sim_inds=None, output=False, do_plot=False, **kwargs):
        """
        Create a dataframe comparing the sims, with one column per sim and one row per result.

        Args:
            t (int/str/date): if None, compare the sim summaries (see `sim.summarize()`); else, the timestep index or date to compare the results at
            sim_inds (list): the indices of the sims to include (default: all)
            output (bool): whether to return the dataframe (otherwise, print it)
            do_plot (bool): whether to also plot the comparison (see `msim.plot_compare()`)
            kwargs (dict): passed to `msim.plot_compare()`

        Examples:
            ```python
            s1 = ss.Sim(diseases=ss.SIS(beta=0.05), networks='random', label='Low')
            s2 = ss.Sim(diseases=ss.SIS(beta=0.10), networks='random', label='High')
            msim = ss.MultiSim([s1, s2]).run()
            msim.compare() # Print the summaries
            df = msim.compare(t='2030-01-01', output=True) # Results on that date
            ```
        """
        sim_inds = sc.ifelse(sim_inds, range(len(self)))
        year = ss.date(t).years if (t is not None and not sc.isnumber(t)) else None
        resdict = {}
        for i in sim_inds:
            sim = self.sims[i]
            label = sim.label if sim.label else f'Sim {i}'
            if label in resdict: # Avoid duplicates
                label += f' ({i})'
            if t is None:
                resdict[label] = sim.summarize() if sim.summary is None else sim.summary
            else:
                flat = sim.results.flatten(columns=True)
                resdict[label] = {}
                for key,res in flat.items():
                    if sc.isnumber(t):
                        ti = t
                    else: # Nearest timestep to the date, using the result's own timevec since modules can have different dt
                        ti = sc.findnearest(res.convert_timevec().years, year)
                    val = res[ti]
                    if res.scale and val == np.round(val): # Show whole-number counts as ints (but not e.g. 4.5 from scaling)
                        val = int(val)
                    resdict[label][key] = val

        df = sc.dataframe(resdict, dtype=object) # Object dtype prevents ints being converted to floats
        if do_plot:
            self.plot_compare(df=df, **kwargs)
        if output:
            return df
        else:
            timestr = 'Summary' if t is None else f'Results for t={t}'
            print(f'{timestr} for each sim:')
            print(df)
            return

    def diff(self, baseline):
        """
        The difference of each scenario from a baseline, for every result they share, run by run

        For sims run with `ss.parallel(baseline=sim1, vaccine=sim2)`. Returns a dict by
        scenario, then module, then result; each entry has one row per run.

        Args:
            baseline (str): the label of the baseline scenario

        Examples:
            ```python
            sims = ss.parallel(baseline=base, vaccine=vx, n_runs=10)
            averted = -sims.diff('baseline').vaccine.sir.cum_infection.median()[-1]
            ```
        """
        if not isinstance(self.results, MultiResults):
            errormsg = 'msim.diff() is only available for sims with declared states and transitions, run with ss.parallel()'
            raise ValueError(errormsg)
        return self.results.diff(baseline)

    def extinction(self, output=False):
        """
        How often each declared disease died out: the share of runs in which nobody was infected by the end, and the median time it happened

        Reported for each scenario, if there are several. Runs that start with nobody infected are not counted as dying out.

        Args:
            output (bool): if True, return a dataframe (one row per disease and scenario) instead of printing it

        Examples:
            ```python
            sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.15), recovery=ss.progression('I -> R', dur=10), init=dict(I=2))
            msim = ss.Sim(sir, total_pop=200, solver='gillespie', unit='day', stop=100).run(n_runs=100)
            msim.extinction() # e.g. "sir: died out in 41% of 100 runs (median time 6 days)"
            ```
        """
        if not isinstance(self.results, MultiResults):
            errormsg = 'msim.extinction() is only available for sims with declared states and transitions'
            raise ValueError(errormsg)
        base = self.base_sim if isinstance(self.base_sim, ss.Sim) else self.sims[0]
        rows = []
        for dis in getattr(base, 'declared', []):
            infected = [s for s in dis.graph.infected if s.people]
            if not infected or dis.name not in self.results.nested:
                continue
            res = self.results.nested[dis.name]
            first = res[f'n_{infected[0].name}']
            labels = first.labels if first.labels is not None and len(set(first.labels)) > 1 else [None]*len(first.values)
            n = sum(res[f'n_{s.name}'].values for s in infected) # Runs by steps
            for label in dict.fromkeys(labels):
                runs = n[[i for i,lab in enumerate(labels) if lab == label]]
                over = (runs[:, -1] <= 0) & (runs[:, 0] > 0)
                times = [first.timevec[len(row) - np.argmax(row[::-1] > 0)] for row in runs[over]] if over.any() else [] # The step after the last with anyone infected
                rows.append(dict(disease=dis.name, scenario=label, runs=len(runs), share=over.mean(), median_time=float(np.median(times)) if times else None))
        df = sc.dataframe(rows)
        if output:
            return df
        for row in rows:
            scenario = f' ({row["scenario"]})' if row['scenario'] is not None else ''
            unit = f' {base.unit}s' if getattr(base, 'unit', None) else ''
            when = f' (median time {row["median_time"]:n}{unit})' if row['median_time'] is not None else ''
            print(f'{row["disease"]}{scenario}: died out in {row["share"]:.0%} of {row["runs"]} runs{when}')
        return

    def plot_compare(self, t=None, sim_inds=None, df=None, **kwargs):
        """
        Plot a bar chart comparing the sims, with one panel per result; see `msim.compare()`.

        Args:
            t (int/str/date): passed to `msim.compare()`
            sim_inds (list): passed to `msim.compare()`
            df (dataframe): if supplied, plot this instead of calling `msim.compare()`
            kwargs (dict): see `ss.plot_args()` for all valid options

        Examples:
            ```python
            msim = ss.MultiSim(ss.Sim(diseases='sis', networks='random'), n_runs=3).run()
            msim.plot_compare()
            ```
        """
        if df is None:
            df = self.compare(t=t, sim_inds=sim_inds, output=True)
        df = df[df.map(sc.isnumber).all(axis=1)] # Skip non-numeric rows
        kw = ss.plot_args(kwargs)
        with ss.style(**kw.style):
            fig, axs = sc.getrowscols(len(df), make=True, **kw.fig)
            for ax, (key, row) in zip(sc.toarray(axs).flatten(), df.iterrows()):
                ax.barh(row.index, row.values.astype(float), **kw.plot)
                ax.set_title(key)
        return ss.return_fig(fig, **kw.return_fig)

    def plot(self, key=None, fig=None, legend=True, **kwargs):
        """
        Plot all results in the MultiSim object.

        If the MultiSim object has been reduced (i.e. mean or median), then plot
        the best value and uncertainty bound. Otherwise, plot individual sims.

        Args:
            key (str): the results key to plot (by default, all)
            fig (Figure): if provided, plot results into an existing figure
            fig_kw (dict): passed to `sc.getrowscols()`, then `plt.subplots()` and `plt.figure()`
            plot_kw (dict): passed to `plt.plot()`
            data_kw (dict): passed to `plt.scatter()`, for plotting the data
            style_kw (dict): passed to `sc.options.with_style()`, for controlling the detailed plotting style
            fill_kw (dict): passed to `plt.fill_between()`
            legend_kw (dict): passed to `plt.legend()`
            legend (bool): whether to show the legend
            **kwargs (dict): known arguments (e.g. figsize, font) split between the above dicts; see `ss.plot_args()` for all valid options
        """
        # Has not been reduced yet, plot individual sim
        if self.which is None:
            res_keys = None
            kw = ss.plot_args(kwargs)
            alpha = kwargs.pop('alpha', 0.7 if len(self) < 5 else 0.5) # Set default alpha
            if key is None: # Set keys
                for sim in self.sims:
                    sim_keys = set(sim.results.flatten(only_auto=True).keys()) # Check if keys match for auto-plotting results
                    if res_keys is None:
                        res_keys = sim_keys
                    else:
                        if res_keys != sim_keys: # TODO: would be good to fix and plot all, but hard with sim.plot()
                            missing = res_keys - sim_keys
                            extra = sim_keys - res_keys
                            extratxt = f'\nExtra: {sc.strjoin(extra)}'
                            missingtxt = f'\nMissing: {sc.strjoin(missing)}'
                            warnmsg = f'Sim "{sim.label}" has different results keys:{extratxt}{missingtxt}\nResults may not plot correctly (i.e. axes titles may not be correct for all sims)'
                            ss.warn(warnmsg)
            for sim in self.sims: # Actually plot
                fig = sim.plot(key=key, fig=fig, alpha=alpha, is_jupyter=False, do_show=False, **kwargs)
            if legend:
                leg = None
                shape = getattr(fig, '_subplots_shape', None) # Sciris-generated figure
                if shape: # If we have empty space on the bottom right, put the legend there
                    n = np.prod(shape)
                    if len(fig.axes) != n: # Bottom-right axes is empty
                        ax = fig.add_subplot(shape[0], shape[1], n)
                        leg = sc.movelegend(fig.axes[0], ax, **kw.legend)
                if leg is None: # Otherwise, just put it in the last axes anyway
                    fig.axes[-1].legend(**kw.legend)

        # Has been reduced, plot with uncertainty bounds
        else:
            # Get arguments
            n_ticks = kwargs.pop('n_ticks', None)
            show_module = kwargs.pop('show_module', None)
            show_skipped = kwargs.pop('show_skipped', None)

            # Figure out the flat structure of results to plot
            flat = ss.utils.match_result_keys(self.results, key, show_skipped=show_skipped, flattened=True)

            # Set figure size
            n_cols,_ = sc.getrowscols(len(flat))
            default_figsize = np.array([8, 6])
            figsize_factor = np.clip((n_cols-3)/6+1, 1, 1.5) # Scale the default figure size based on the number of rows and columns
            figsize = default_figsize*figsize_factor
            kw = ss.plot_args(kwargs, figsize=figsize, alpha=0.8, fill_alpha=0.2, lw=2)

            # Get ready to plot
            with ss.style(**kw.style):
                if fig is None:
                    fig, axs = sc.getrowscols(len(flat), make=True, **kw.fig)
                else:
                    axs = fig.axes
                axs = sc.toarray(axs) # Ensure axs is always an array: with a single key, getrowscols() returns a bare Axes

                # Do the plotting
                for ax, (key, res) in zip(axs.flatten(), flat.items()):
                    lines = res.by_column().items() if res.columns is not None else [(None, res)] # One line per column for 2D results
                    for col,gres in lines:
                        if gres.low is not None: # Combined sims don't have bounds
                            ax.fill_between(gres.timevec, gres.low, gres.high, **kw.fill)
                        col_kw = dict(label=col) if col is not None else None
                        ax.plot(gres.timevec, gres, **sc.mergedicts(col_kw, kw.plot))
                    if res.columns is not None:
                        ax.legend(**kw.legend)
                    ss.utils.format_axes(ax, res, n_ticks, show_module)

        return ss.return_fig(fig, **kw.return_fig)

    @classmethod
    def merge(cls, *args, base=False):
        """
        Merge several MultiSims into one; see also `msim.split()`.

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate arguments)
            base (bool): if True, make a new MultiSim from the base sims of each MultiSim (e.g. after `reduce()`); otherwise, merge the lists of sims

        Returns:
            A new MultiSim

        Examples:
            ```python
            m1 = ss.MultiSim(ss.Sim(diseases='sis', networks='random', label='SIS'), n_runs=3).run()
            m2 = ss.MultiSim(ss.Sim(diseases='sir', networks='random', label='SIR'), n_runs=3).run()
            msim = ss.MultiSim.merge(m1, m2) # 6 sims
            m1.mean(); m2.mean()
            mm = ss.MultiSim.merge(m1, m2, base=True) # 2 sims, the means of each
            mm.plot()
            ```
        """
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0] # A single list of MultiSims has been provided

        # Create the MultiSim from the base sim of the first argument
        msim = cls(base_sim=sc.dcp(args[0].base_sim), label=args[0].label)
        msim.sims = []
        msim.chunks = [] # Used to enable automatic splitting later
        for i,ms in enumerate(args):
            if base: # Only keep the base sims
                sim = sc.dcp(ms.base_sim)
                sim.label = ms.label
                msim.chunks.append([i])
                msim.sims.append(sim)
            else: # Keep all the sims
                n = len(msim.sims)
                msim.chunks.append(list(range(n, n+len(ms))))
                msim.sims += sc.dcp(ms.sims)
        return msim

    def split(self, inds=None, chunks=None):
        """
        Split one MultiSim into several; the reverse of `ss.MultiSim.merge()`.

        Specify either the indices of the sims for each new MultiSim (`inds`), or
        consecutive chunks (`chunks`). For a merged MultiSim, neither is needed.

        Args:
            inds (list): a list of lists of indices, with each list turned into a MultiSim
            chunks (int/list): if an int, split the MultiSim into that many equal chunks; if a list, the number of sims in each chunk

        Returns:
            A list of MultiSims

        Examples:
            ```python
            msim = ss.MultiSim(ss.Sim(diseases='sis', networks='random'), n_runs=6).run()
            m1, m2 = msim.split(inds=[[0,2,4], [1,3,5]])
            m1, m2 = msim.split(chunks=[2,4]) # Equivalent to inds=[[0,1], [2,3,4,5]]
            m1, m2 = msim.split(chunks=2) # Equivalent to inds=[[0,1,2], [3,4,5]]
            m1, m2 = ss.MultiSim.merge(m1, m2).split() # Use the chunks from the merge
            ```
        """
        if inds is None:
            if chunks is not None:
                sim_inds = np.arange(len(self))
                split = np.cumsum(chunks)[:-1] if sc.isiterable(chunks) else chunks # e.g. chunks=[2,4] or chunks=2
                inds = np.split(sim_inds, split) # Raises an exception if the chunks don't divide the sims evenly
            elif hasattr(self, 'chunks'): # Created from a merged MultiSim
                inds = self.chunks
            else:
                errormsg = 'If a MultiSim has not been created via merge(), you must supply either inds or chunks to split it'
                raise ValueError(errormsg)
        return [self.__class__(sims=sc.dcp([self.sims[i] for i in indlist])) for indlist in inds]


def single_run(sim, ind=0, reseed=True, shrink=True, run_args=None, sim_args=None,
               verbose=None, do_run=True, copy_sim=False, **kwargs):
    """
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly.

    Args:
        sim         (Sim)   : the sim instance to be run
        ind         (int)   : the index of this sim
        reseed      (bool)  : whether to generate a fresh seed for each run
        shrink      (bool)  : whether to shrink the sim after the sim run
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim, e.g. 'n_infected'
        verbose     (int)   : detail to print
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        copy_sim    (bool)  : whether to explicitly copy the sim before running (default: False)
        kwargs      (dict)  : also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    Examples:
        ```python
        import starsim as ss
        sim = ss.Sim() # Create a default simulation
        sim = ss.single_run(sim) # Run it, equivalent(ish) to sim.run()
        ```
    """

    # Set sim and run arguments
    sim_args = sc.mergedicts(sim_args, kwargs)
    run_args = sc.mergedicts({'verbose': verbose}, run_args)
    if verbose is None:
        verbose = sim.pars['verbose']

    if copy_sim:
        sim = sim.copy() # Make a copy to avoid modifying the original sim (typically done by default via pickling with multi_run)

    if not sim.label:
        sim.label = f'Sim {ind}'

    if reseed:
        sim.pars['rand_seed'] += ind  # Reset the seed, otherwise no point of parallel runs
        if ind and sim.initialized:
            ss.warn(f'Sim "{sim.label}" is already initialized, so changing its seed has no effect; pass an uninitialized sim instead')

    # Handle additional arguments
    for key, val in sim_args.items():
        if key in sim.pars.keys():
            if verbose >= 1:
                print(f'Setting key {key} from {sim[key]} to {val}')
            sim.pars[key] = val
        else:
            raise sc.KeyNotFoundError(f'Could not set key {key}: not a valid parameter name')

    # Run
    if do_run:
        sim.run(**run_args)

    # Shrink the sim to save memory
    if shrink:
        sim.shrink(die=False)

    return sim


def multi_run(sim, n_runs=4, reseed=None, iterpars=None, shrink=None, run_args=None, sim_args=None,
              par_args=None, do_run=True, parallel=True, n_cpus=None, copy_sim=False, verbose=None, **kwargs):
    """
    For running multiple sims in parallel. If the first argument is a list of sims
    rather than a single sim, exactly these will be run and most other arguments
    will be ignored.

    Note: `n_cpus=1` is *not* the same thing as setting `parallel=False`. The former
    still uses the parallelization methods (just on a single core), while the latter
    simply runs in a loop.

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
        n_cpus      (int)   : the number of CPUs to run on (if blank, set automatically; otherwise, passed to par_args, and use all cores)
        copy_sim    (bool)  : whether to explicitly copy the sim before running (default: False)
        verbose     (int)   : detail to print
        kwargs      (dict)  : also passed to the sim

    Returns:
        A list of sim objects (default).

    Examples:
        ```python
        import starsim as ss
        sim = ss.Sim()
        sims = ss.multi_run(sim, n_runs=6)
        ```
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
                      sim_args=sim_args, run_args=run_args, do_run=do_run, copy_sim=copy_sim)
    elif isinstance(sim, (list, tuple)):  # List of sims
        if reseed is None: reseed = False
        iterkwargs = dict(sim=sim, ind=np.arange(len(sim)))
        kwargs = dict(reseed=reseed, verbose=verbose, shrink=shrink, sim_args=sim_args, run_args=run_args,
                      do_run=do_run, copy_sim=copy_sim)
    else:
        errormsg = f'Must be Sim object or list/tuple, not {type(sim)}'
        raise TypeError(errormsg)

    # Actually run
    if parallel:
        sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs, **par_args)  # Run in parallel
    else:  # Run in serial, not in parallel
        sims = []
        n_sims = len(list(iterkwargs.values())[0])  # Must have length >=1 and all entries must be the same length
        for s in range(n_sims):
            this_iter = {k: v[s] for k, v in iterkwargs.items()}  # Pull out items specific to this iteration
            this_iter.update(kwargs)  # Merge with the kwargs
            this_iter['copy_sim'] = True  # Ensure we have a fresh sim; this happens implicitly on pickling with multiprocessing but must be done explicitly here
            sim = single_run(**this_iter)  # Run in series
            sims.append(sim)

    return sims


class MultiResult(sc.prettyobj):
    """
    One result across the runs of a MultiSim: the same result, with a run dimension

    Args:
        name (str): the name of the result
        values (array): the values, with one row per run
        timevec (array): the time points
        labels (list): the label (scenario) of each run
        label (str): a human-readable name
    """
    def __init__(self, name, values, timevec, labels=None, label=None, strata=None, like=None):
        self.name = name
        self.label = label or name
        self.values = np.asarray(values)
        self.timevec = timevec
        self.labels = labels
        self.strata = strata # For a stratified result: the value of each group, as runs by steps by groups
        self.strats = getattr(like, 'strats', None) # The stratifications, and the dimensions this result has, from a result of one run
        self.dims = getattr(like, 'dims', None)
        return

    def __len__(self):
        return len(self.values)

    def quantiles(self, q=(0.025, 0.5, 0.975)):
        """ One series per quantile, over the runs """
        return np.quantile(self.values, q, axis=0)

    def mean(self):
        return self.values.mean(axis=0)

    def median(self):
        return np.median(self.values, axis=0)

    def select(self, run=None, scenario=None, **dims):
        """
        Select one run by position, the runs of one scenario by its label, or groups of a stratified result

        Args:
            run (int): the position of one run; returns its values
            scenario (str): the label of a scenario; returns its runs
            dims (dict): for a stratified result, the dimension and the level (or levels) to keep, summing over the rest, as for `ss.Result.select()`

        Examples:
            ```python
            msim = sim.run(n_runs=100)
            msim.results.sir.n_I.select(age='0-4').quantiles()
            ```
        """
        if dims:
            keep = self._keep(dims)
            out = MultiResult(self.name, self.strata[:, :, keep].sum(axis=-1), self.timevec, labels=self.labels,
                              label=f'{self.label} ({", ".join(f"{k}={v}" for k,v in dims.items())})', strata=self.strata[:, :, keep], like=self)
            if run is not None or scenario is not None:
                return out.select(run=run, scenario=scenario)
            return out
        if run is not None:
            return self.values[run]
        keep = [i for i,label in enumerate(self.labels) if label == scenario]
        if not keep:
            errormsg = f'No runs are labelled "{scenario}"; the labels are {sorted(set(self.labels))}'
            raise KeyError(errormsg)
        return MultiResult(self.name, self.values[keep], self.timevec, labels=[self.labels[i] for i in keep], label=self.label)

    def by(self, dim):
        """ Split a stratified result by one dimension, summing over the rest: a dict of results, one per level """
        strat = self._strat(dim)
        return sc.objdict({str(label):self.select(**{dim:label}) for label in strat.labels})

    def _strat(self, dim):
        if self.strata is None:
            errormsg = f'The result "{self.name}" is not kept by group across the runs, so there is nothing to select'
            raise ValueError(errormsg)
        names = self.dims or [strat.name for strat in self.strats]
        if dim not in names:
            errormsg = f'The result "{self.name}" has no dimension "{dim}"; its dimensions are {names}'
            raise KeyError(errormsg)
        return [strat for strat in self.strats if strat.name == dim][0]

    def _keep(self, dims):
        """ Which groups have the given levels """
        coords = self.strats[0].sim_coords
        names = [strat.name for strat in self.strats]
        keep = np.ones(self.strata.shape[-1], dtype=bool)
        for dim,levels in dims.items():
            strat = self._strat(dim)
            inds = [strat.index(level) for level in sc.tolist(levels)]
            keep &= np.isin(coords[:, names.index(dim)], inds)
        return keep

    def diff(self, baseline):
        """
        The difference of each other scenario from the baseline, run by run (the runs are paired by position)

        Args:
            baseline (str): the label of the baseline scenario

        Returns a dict of MultiResults, one per scenario other than the baseline.
        """
        base = self.select(scenario=baseline).values
        out = sc.objdict()
        for label in dict.fromkeys(self.labels):
            if label != baseline:
                other = self.select(scenario=label)
                if other.values.shape != base.shape:
                    errormsg = f'Cannot compare "{label}" with "{baseline}": they have {len(other.values)} and {len(base)} runs'
                    raise ValueError(errormsg)
                out[label] = MultiResult(self.name, other.values - base, self.timevec, labels=other.labels, label=f'{self.label}: difference from {baseline}')
        return out

    def plot(self, lines=False, ax=None, **kwargs):
        """ Plot the median and a 95% interval, or every run if `lines=True` """
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        if lines:
            ax.plot(self.timevec, self.values.T, alpha=0.3, **kwargs)
        else:
            low, mid, high = self.quantiles()
            ax.fill_between(self.timevec, low, high, alpha=0.3)
            ax.plot(self.timevec, mid, **kwargs)
        ax.set_title(self.label)
        return ax


class MultiResults(ss.Results):
    """ The reduced results of a MultiSim (as `ss.Results`), which also gives each result across the runs: `msim.results.sir.n_I` """
    def __init__(self, *args, nested=None, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'nested', nested or {})
        return

    def __getattribute__(self, attr):
        if not attr.startswith('_'):
            nested = object.__getattribute__(self, '__dict__').get('nested')
            if nested and attr in nested:
                return nested[attr]
        return super().__getattribute__(attr)

    def diff(self, baseline):
        """ The difference of every result from the baseline scenario: a dict by scenario, then module, then result """
        out = sc.objdict()
        for modname,modres in self.nested.items():
            for key,res in modres.items():
                if baseline in res.labels:
                    for label,diff in res.diff(baseline).items():
                        out.setdefault(label, sc.objdict()).setdefault(modname, sc.objdict())[key] = diff
        return out

    @staticmethod
    def from_sims(sims):
        """ Collect each result of each module across the sims, as one MultiResult with a run dimension """
        nested = sc.objdict()
        labels = [sim.label for sim in sims]
        first = sims[0]
        for mod in first.modules:
            modres = sc.objdict()
            for key,res in mod.results.items():
                if isinstance(res, ss.Result) and np.ndim(res.values) == 1:
                    found = [sim.results[mod.name][key] for sim in sims if mod.name in sim.results and key in sim.results[mod.name]]
                    values = [np.asarray(r.values) for r in found]
                    if len(values) == len(sims) and len({len(v) for v in values}) == 1:
                        strata = [getattr(r, 'strata', None) for r in found]
                        strata = np.stack(strata) if all(x is not None for x in strata) and len({x.shape for x in strata}) == 1 else None
                        modres[key] = MultiResult(key, values, res.timevec, labels=labels, label=res.label, strata=strata, like=res)
            if len(modres):
                nested[mod.name] = modres
        return MultiResults('MultiSim', nested=nested)


def par_sets(pars):
    """ A list of sets of parameter values, from a list of dicts or a dict of lists (one value per set) """
    if pars is None:
        return [None]
    if isinstance(pars, dict):
        lens = {len(sc.tolist(v)) for v in pars.values()}
        if len(lens) != 1:
            errormsg = f'Each parameter in pars= needs the same number of values, not {sc.strjoin(sorted(lens))}'
            raise ValueError(errormsg)
        n = lens.pop()
        return [{key:sc.tolist(vals)[i] for key,vals in pars.items()} for i in range(n)]
    return [dict(p) for p in pars]


def restart_runs(sims, n_runs, pars=None):
    """
    Run each sim a number of times, each with its own seed, by building it once and restarting it

    Only for sims that are nothing but compartments (see `ss.Sim.restart()`), where this
    is many times faster than running copies. Returns None if any of the sims is not.
    The MultiSim that comes back has the results of every run, and one sim for each
    that was given, as it was after its last run.

    With `pars=` (see `ss.parallel()`), each set of values is run `n_runs` times. A sim
    run with `integrator='jax'` takes all of its runs in one call to JAX (see `ss.gradients`).
    """
    if not all(hasattr(sim, 'inputs') and not sim.initialized for sim in sims):
        return None
    runners = []
    for sim in sims:
        runner = sim.new(rand_seed=sim.pars.rand_seed).init() # The seed as it is now, which may have been set since the sim was made
        runner.label = sim.label
        if not runner.kernel_only():
            return None
        runners.append(runner)
    sets = par_sets(pars)
    labels = [sim.label for sim in runners for p in sets for i in range(n_runs)]
    values = sc.ddict(list)
    groups = sc.ddict(list)
    for sim in runners:
        seed = sim.pars.rand_seed
        if pars is not None and sim.jax and ss.gradients.reason(sim) is None: # Every run in one call to JAX (an ODE run does not depend on the seed)
            outs = list(ss.gradients.run_sets(sim, [p for p in sets for i in range(n_runs)]).items())
        else:
            outs = []
            for p in sets: # All the runs of each module with each set of values are one call to the kernels
                if p is not None:
                    sim.restart(pars=p)
                outs += [(mod.name, mod.engine.run_many(seed + np.arange(n_runs))) for mod in sim.declared]
        for modname,out in outs:
            for key,vals in out.pop('groups').items():
                groups[modname, key].append(vals)
            for key,vals in out.items():
                values[modname, key].append(vals)
        sim.restart(pars=sets[-1], rand_seed=seed + n_runs - 1).run() # The sim itself is left as its last run
    nested = sc.objdict()
    for (modname,key),vals in values.items():
        res = runners[0].results[modname][key]
        vals = np.concatenate(vals)
        if len(vals) == len(labels): # A result that every sim has
            strata = groups.get((modname, key))
            strata = np.concatenate(strata) if strata is not None else None
            strata = strata if strata is not None and len(strata) == len(labels) else None
            nested.setdefault(modname, sc.objdict())[key] = MultiResult(key, vals, res.timevec, labels=labels, label=res.label, strata=strata, like=res)
    msim = MultiSim(sims=runners)
    msim.results = MultiResults('MultiSim', nested=nested)
    if pars is not None:
        msim.run_pars = sc.dataframe([p for sim in runners for p in sets for i in range(n_runs)]) # The values of each run
    return msim


def parallel(*args, n_runs=None, pars=None, **kwargs):
    """
    A shortcut to `ss.MultiSim()`, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        n_runs (int): if provided, run each sim this many times, each with its own seed
        pars (dict/list): sets of values of model parameters to run each sim with: a dict with a list of values for each parameter (`dict(beta=[0.2, 0.3])`), or a list of dicts; each set is run `n_runs` times (default 1), and `msim.run_pars` has the values of each run
        kwargs (dict): sims passed as keywords are labelled with the keyword, e.g. `ss.parallel(baseline=s1, vaccine=s2)`;
            everything else is passed to multi_run()

    Returns:
        A run MultiSim object. For sims with declared states and transitions, each result
        is also available across the runs, e.g. `msim.results.sir.n_I.quantiles([0.025, 0.5, 0.975])`.

    Examples:
        ```python
        s1 = ss.Sim(n_agents=1000, label='Small', diseases='sis', networks='random')
        s2 = ss.Sim(n_agents=2000, label='Large', diseases='sis', networks='random')
        ss.parallel(s1, s2).plot()
        msim = ss.parallel([s1, s2], shrink=False)
        msim = ss.parallel(s1, n_runs=10) # Ten runs of one sim
        msim = ss.parallel(sim, pars=dict(beta=np.linspace(0.1, 0.5, 100))) # One run with each value of beta (for a sim with declared states and transitions)
        ```
    """
    sims = sc.mergelists(*args)
    for key in list(kwargs.keys()): # Named sims
        if isinstance(kwargs[key], ss.Sim):
            sim = kwargs.pop(key)
            sim.label = key
            sims.append(sim)
    if pars is not None:
        n_runs = sc.ifelse(n_runs, 1)
    serial = {'verbose', 'n_cpus', 'parallel'} # Options of running in parallel, which don't matter to runs restarted in this process
    if n_runs is not None and set(kwargs) <= serial: # Built once and restarted for each run, where the sims allow, which is many times faster
        msim = restart_runs(sims, n_runs, pars=pars)
        if msim is not None:
            return msim
    if pars is not None: # Otherwise, a copy of each sim for each set of values
        sets = par_sets(pars)
        run_pars = sc.dataframe([p for sim in sims for p in sets for i in range(n_runs)])
        sims = [sim.new(pars=p, label=sim.label) for sim in sims for p in sets]
    if n_runs is not None:
        replicates = []
        for sim in sims:
            for i in range(n_runs):
                new = sim.copy() if sim.initialized or not hasattr(sim, 'inputs') else sim.new()
                new.label = sim.label
                new.pars.rand_seed = sim.pars.rand_seed + i
                replicates.append(new)
        sims = replicates
        kwargs.setdefault('reseed', False) # The seeds have just been set
    msim = MultiSim(sims=sims, **kwargs)
    msim.run()
    if pars is not None:
        msim.run_pars = run_pars
    if any(getattr(sim, 'declared', None) for sim in msim.sims):
        msim.results = MultiResults.from_sims(msim.sims)
    return msim
