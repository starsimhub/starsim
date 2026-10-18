"""
Running many sims: ss.parallel() and ss.MultiSim
"""
import functools as ft
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss



hub_quantiles = [0.01, 0.025] + [round(q, 2) for q in np.arange(0.05, 0.951, 0.05)] + [0.975, 0.99] # The 23 quantiles of the US forecast hubs
run_options = ['parallel', 'n_cpus', 'par_args', 'shrink', 'verbose', 'run_args'] # How to run the sims, as opposed to which runs to take


class MultiSim:
    """
    The runs of one or more sims: every result of the sims with a run axis first, and the sims that were run

    Made and run by `ss.parallel()`, the way to run many sims (or by `sim.run(n_runs=...)`,
    a shortcut for running one sim many times). Making a MultiSim and calling `msim.run()`
    does the same.

    Args:
        sims (Sim/list): a single sim, a list/tuple of sims, or another MultiSim (or list of MultiSims)
        base_sim (Sim): the sim used for shared properties; if not supplied, the first of the sims provided
        label (str): the name of the multisim
        n_runs (int): the number of runs of each sim, each with its own seed (by default, 4 for a single sim given as `sims`, otherwise 1)
        pars (dict/list): sets of values of parameters to run each sim with; see `ss.parallel()`
        priors (bool): draw `n_runs` sets of values from the priors of the parameters; see `ss.parallel()`
        initialize (bool): whether to make and initialize the sim of each run now (otherwise, when it is run)
        inplace (bool): whether to update sims that are run as they are (with no `n_runs` or `pars`) with their results (default True)
        debug (bool): if True, run in serial
        kwargs (dict): how to run the sims: `parallel` (default True), `n_cpus`, `par_args` (passed to `sc.parallelize()`), `shrink`, `verbose`, and `run_args` (passed to `sim.run()`)

    Once run, a MultiSim has:

    - `results`: every result with a run axis first (see `ss.Result`), e.g. `msim.results.sir.n_I.quantiles()`;
    - `runs`: one row per run, with its scenario (the label of its sim) and the values of any parameters swept;
    - `sims`: the sims that were run (for runs restarted from one compartmental sim, that sim, as it was after its last run).

    `len(msim)` is the number of runs.

    Examples:
        ```python
        import starsim as ss

        s1 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.02), label='Low transmission')
        s2 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.05), label='High transmission')
        msim = ss.parallel(s1, s2, n_runs=5)
        msim.plot() # The median and a 95% interval of each scenario
        msim.results.sis.prevalence.select(scenario='Low transmission').median()
        ```
    """
    def __init__(self, sims=None, base_sim=None, label=None, n_runs=None, pars=None, priors=False, initialize=False,
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

        unknown = set(kwargs) - set(run_options)
        if unknown:
            errormsg = f'Unknown argument{"s" if len(unknown) > 1 else ""} {sc.strjoin(sorted(unknown))} for running many sims; the options are n_runs, pars, priors, inplace, debug, and {sc.strjoin(run_options)}'
            if 'iterpars' in unknown:
                errormsg += ' (iterpars is now pars, e.g. pars=dict(rand_seed=[1, 2, 3]))'
            raise TypeError(errormsg)

        # Set properties
        self.sims = sims
        self.base_sim = base_sim
        self.label = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args = sc.mergedicts(dict(n_runs=n_runs, pars=pars, priors=priors, inplace=inplace, debug=debug), kwargs)
        self.results = None # Every result, with a run axis first
        self.runs = None # One row per run: its scenario, and the values of any parameters swept
        self.summary = None
        self.ready = False # Whether the sim of each run has been made, by init_sims()
        self.timer = sc.timer() # Create a timer

        # Optionally initialize
        if initialize:
            self.init_sims()

        return

    def __len__(self):
        """ The length of a MultiSim is its number of runs """
        if self.runs is not None:
            return len(self.runs)
        return len(self.sims) if self.sims is not None else 0

    def __repr__(self):
        """ Return a brief description of a multisim; see multisim.disp() for the more detailed version. """
        try:
            labelstr = f'"{self.label}"; ' if self.label else ''
            string   = f'MultiSim({labelstr}n_runs: {len(self)}; base: {self.base_sim})'
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
        string  = f'MultiSim{labelstr} summary:\n'
        string += f'  Number of runs: {len(self)}\n'
        if self.runs is not None:
            string += f'  Scenarios: {sc.strjoin(f"{k} ({v} runs)" for k,v in self.runs.scenario.value_counts(sort=False).items())}\n'
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

    @property
    def run_pars(self):
        """ The values of the parameters swept (or drawn) in each run, as a dataframe (or None if there are none) """
        if self.runs is None or len(self.runs.columns) == 1:
            return None
        return self.runs.drop(columns='scenario')

    def resolve(self, n_runs=None, pars=None, priors=False):
        """ The sims to run, how many times to run each, and the sets of values to run them with """
        sims = [self.base_sim] if self.sims is None else list(self.sims)
        if self.sims is None and n_runs is None and pars is None and not priors:
            n_runs = 4 # A single sim given to a MultiSim is run four times by default
        if len(sims) > 1: # Different sims are different scenarios
            for i,sim in enumerate(sims):
                if not sim.label:
                    sim.label = f'Sim {i}'
        many = n_runs is not None or pars is not None or priors
        sims = [sim.new(rand_seed=sim.pars.rand_seed, label=sim.label) if hasattr(sim, 'inputs') and (sim.results_ready or (many and sim.initialized)) else sim for sim in sims] # A sim that has been run (e.g. fit.sim), or that would be run more than once after init(), is built again
        if priors:
            if pars is not None:
                errormsg = 'ss.parallel() takes pars= or priors=True, not both'
                raise ValueError(errormsg)
            pars = ss.calibration.prior_draws(sims[0], sc.ifelse(n_runs, 100), seed=sims[0].pars.rand_seed)
            n_runs = 1
        if pars is not None:
            n_runs = sc.ifelse(n_runs, 1)
        return sims, n_runs, pars

    @staticmethod
    def expand(sims, n_runs=None, pars=None):
        """ The sim of each run, each with its own seed (seed, seed + 1, ...) and set of values, and a dataframe of the runs """
        sets = par_sets(pars)
        out = []
        rows = []
        for sim in sims:
            if n_runs is not None and sim.initialized and not hasattr(sim, 'inputs'):
                errormsg = f'Sim "{sim.label}" is already initialized, so its runs would all be the same; pass an uninitialized sim to run it with n_runs'
                raise ValueError(errormsg)
            for p in sets:
                for i in range(sc.ifelse(n_runs, 1)):
                    out.append(sim if n_runs is None and p is None else make_run(sim, p, sim.pars.rand_seed + i)) # A sim run once as it is is run itself
                    rows.append(dict(scenario=sim.label or '', **(p or {})))
        return out, pd.DataFrame(rows)

    def init_sims(self, **kwargs):
        """ Make the sim of each run, and initialize them without running them """
        kwargs = sc.mergedicts(self.run_args, kwargs)
        sims, n_runs, pars = self.resolve(kwargs.get('n_runs'), kwargs.get('pars'), kwargs.get('priors'))
        self.sims, self.runs = self.expand(sims, n_runs, pars)
        for sim in self.sims:
            if not sim.initialized:
                sim.init()
        self.ready = True
        return

    def run(self, **kwargs):
        """
        Run the sims, each `n_runs` times with its own seed and with each set of values of `pars`, and collect their results; see `ss.parallel()`

        Args:
            kwargs (dict): change the arguments the MultiSim was made with, e.g. `n_runs=10` or `parallel=False`

        Returns:
            The MultiSim, with its results
        """
        kwargs = sc.mergedicts(self.run_args, kwargs)
        n_runs, pars, priors = [kwargs.pop(key, None) for key in ['n_runs', 'pars', 'priors']]
        inplace = kwargs.pop('inplace', True)
        if kwargs.pop('debug', False):
            kwargs['parallel'] = False # Run in serial
        self.timer.start()
        if self.ready: # The sim of each run was made by init_sims()
            sims = self.sims
        else:
            base, n_runs, pars = self.resolve(n_runs, pars, priors)
            if n_runs is not None and set(kwargs) <= {'verbose', 'n_cpus', 'parallel'}: # Built once and restarted for each run, where the sims allow, which is many times faster
                restarted = restart_runs(base, n_runs, pars=pars)
                if restarted is not None:
                    self.sims, self.results, self.runs = restarted
                    self.ready = False
                    self.timer.stop()
                    return self
            sims, self.runs = self.expand(base, n_runs, pars)

        run_sims = run_each(sims, **kwargs) # This does all the work! Output sims are copies due to the pickling during parallelization
        if inplace:
            for old,new in zip(sims, run_sims):
                if old is not new:
                    old.__dict__.update(new.__dict__) # Update the same object with the new results
        self.sims = run_sims
        self.results = MultiResults.from_sims(run_sims, self.runs)
        self.ready = False
        self.timer.stop()
        return self

    def check_run(self, what):
        if self.results is None:
            errormsg = f'msim.{what}() needs the results of the runs; run the MultiSim first (or use ss.parallel())'
            raise ValueError(errormsg)
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
        Shrink each sim in the msim (the results across runs are kept); see sim.shrink() for more information.

        Args:
            kwargs (dict): passed to sim.shrink() for each sim
        """
        self.base_sim.shrink(**kwargs)
        for sim in self.sims:
            sim.shrink(**kwargs)
        return

    def combine(self):
        """
        Combine the sims into a single sim, e.g. to treat several smaller sims as one larger population

        Results that scale with population size (`res.scale=True`, e.g. counts) are
        summed; other results (e.g. prevalence) are averaged, weighted by each sim's
        `total_pop`. The population sizes (`n_agents` and `total_pop`) are summed, and
        `pop_scale` is updated to match. The people are not combined: the combined
        sim keeps the people (if any) of the first sim.

        Returns:
            The combined sim

        Examples:
            ```python
            msim = ss.MultiSim(ss.Sim(n_agents=1e3, diseases='sis', networks='random'), n_runs=4)
            msim.run()
            sim = msim.combine() # Equivalent to a single sim with n_agents=4e3
            sim.plot()
            ```
        """
        # Combine the population sizes
        n_runs = len(self.sims)
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

        # Compute and store final results
        combined_sim.summarize()
        return combined_sim

    def summarize(self, method='mean', quantiles=None, how=None):
        """
        Summarize the runs statistically: each result over each run, by its kind (see `sim.summarize()`), and then over the runs

        Args:
            method (str): one of 'mean' (default: the mean, standard deviation, and standard error over the runs), 'median' (quantiles over the runs), or 'all' (the value of each run)
            quantiles (dict/list): if method='median', the quantiles (default: median, min, max, q25, and q75)
            how (str/dict): how to summarize each result over each run: as for `sim.summarize()`
        """
        self.check_run('summarize')
        funcs = dict(mean=np.mean, median=np.median, sum=np.sum, last=lambda vals: vals[-1])
        if not isinstance(how, dict):
            how = {} if how is None else {'':how} # '' matches every key
        if method == 'median':
            if quantiles is None:
                quantiles = sc.objdict({'median':0.5, 'min':0, 'max':1, 'q25':0.25, 'q75':0.75})
            elif isinstance(quantiles, list):
                quantiles = {q:q for q in quantiles}

        summary = sc.objdict()
        for key,res in self.results.flatten(sep='_', keep_case=True, columns=True).items():
            func = how.get(key, how.get(''))
            if func is None:
                arr = np.asarray(res.summarize()) # One value per run
            else:
                func = func if callable(func) else funcs[func]
                arr = np.array([func(vals) for vals in np.asarray(res.values)])
            if method == 'all':
                summary[key] = arr
            elif method == 'mean':
                summary[key] = sc.objdict({'mean':arr.mean(), 'std':arr.std(), 'sem':sc.sem(arr)})
            elif method == 'median':
                summary[key] = {q: np.quantile(arr, v) for q, v in quantiles.items()}
            else:
                errormsg = f'Unknown method "{method}"; use "mean", "median", or "all"'
                raise ValueError(errormsg)

        self.summary = summary
        return summary

    def compare(self, t=None, sim_inds=None, output=False, do_plot=False, **kwargs):
        """
        Create a dataframe comparing the runs, with one column per run and one row per result.

        Args:
            t (int/str/date): if None, compare each result over the whole run (see `sim.summarize()`); else, the timestep index or date to compare the results at
            sim_inds (list): the positions of the runs to include (default: all)
            output (bool): whether to return the dataframe (otherwise, print it)
            do_plot (bool): whether to also plot the comparison (see `msim.plot_compare()`)
            kwargs (dict): passed to `msim.plot_compare()`

        Examples:
            ```python
            s1 = ss.Sim(diseases=ss.SIS(beta=0.05), networks='random', label='Low')
            s2 = ss.Sim(diseases=ss.SIS(beta=0.10), networks='random', label='High')
            msim = ss.parallel(s1, s2)
            msim.compare() # Print the summaries
            df = msim.compare(t='2030-01-01', output=True) # Results on that date
            ```
        """
        self.check_run('compare')
        inds = list(sc.ifelse(sim_inds, range(len(self))))
        scenarios = list(self.runs.scenario)
        labels = []
        for i in inds:
            label = scenarios[i] if scenarios[i] else f'Run {i}'
            labels.append(f'{label} ({i})' if label in labels or scenarios.count(scenarios[i]) > 1 else label) # Avoid duplicates
        year = ss.date(t).years if (t is not None and not sc.isnumber(t)) else None
        resdict = {label:{} for label in labels}
        for key,res in self.results.flatten(sep='_', keep_case=True, columns=True).items():
            if t is None:
                vals = np.asarray(res.summarize())
            else:
                ti = t if sc.isnumber(t) else sc.findnearest(res.convert_timevec().years, year) # Nearest timestep to the date, using the result's own timevec since modules can have different dt
                vals = np.asarray(res.values)[:, ti]
            for label,i in zip(labels, inds):
                val = vals[i]
                if t is not None and res.scale and val == np.round(val): # Show whole-number counts as ints (but not e.g. 4.5 from scaling)
                    val = int(val)
                resdict[label][key] = val

        df = sc.dataframe(resdict, dtype=object) # Object dtype prevents ints being converted to floats
        if do_plot:
            self.plot_compare(df=df, **kwargs)
        if output:
            return df
        else:
            timestr = 'Summary' if t is None else f'Results for t={t}'
            print(f'{timestr} for each run:')
            print(df)
            return

    def diff(self, baseline):
        """
        The difference of each scenario from a baseline, for every result they share, run by run

        For sims run with `ss.parallel(baseline=sim1, vaccine=sim2, n_runs=...)`. Returns a dict by
        scenario, then module, then result; each entry has one row per run.

        Args:
            baseline (str): the label of the baseline scenario

        Examples:
            ```python
            sims = ss.parallel(baseline=base, vaccine=vx, n_runs=10)
            averted = -sims.diff('baseline').vaccine.sir.infection.cum.median()[-1]
            ```
        """
        self.check_run('diff')
        return self.results.diff(baseline)

    def to_df(self, quantiles=(0.025, 0.5, 0.975)):
        """ Every result across the runs, as quantiles at each time for each scenario: a long dataframe with columns `module`, `result`, `scenario`, `time`, `quantile`, and `value` (with `quantiles=None`, one row per run and time instead) """
        self.check_run('to_df')
        return self.results.to_df(quantiles=quantiles)

    def to_hub(self, targets, by=None, every='week', quantiles=hub_quantiles, location=None):
        """
        The runs as a submission to a forecast hub: quantiles of each target, by location and period

        Args:
            targets (dict): the hub's name for each result, e.g. `dict(incidHosp='inc hosp')`
            by (str): the dimension whose levels are the locations (default: one location, the total)
            every (str): the period: a flow is totaled over it, and a count is taken at its end: 'day' or 'week' (of the sim's steps, which must then be days)
            quantiles (list): the quantiles (default: the 23 that the US forecast hubs ask for)
            location (str): the name of the single location, if `by` is not given

        Returns a long dataframe with columns `target`, `location`, `target_end_date`, `output_type`, `quantile`, and `value`.
        """
        self.check_run('to_hub')
        steps = dict(day=1, week=7)[every]
        found = {key:res for modname,key,res in self.results.walk() if modname is not None}
        rows = []
        for key,target in targets.items():
            if key not in found:
                errormsg = f'No result called "{key}"; the results are {sc.strjoin(found)}'
                raise KeyError(errormsg)
            res = found[key]
            places = res.by(by) if by is not None else {location or 'total':res}
            for place,sub in places.items():
                vals = sub.values[:, 1:] # The steps, after the start
                n = vals.shape[1]//steps
                if sub.kind == 'flow':
                    periods = vals[:, :n*steps].reshape(len(vals), n, steps).sum(axis=2)
                else:
                    periods = vals[:, steps-1:n*steps:steps]
                ends = np.array(sub.timevec)[1:][steps-1:n*steps:steps]
                for q,qvals in zip(quantiles, np.quantile(periods, quantiles, axis=0)):
                    rows.append(pd.DataFrame(dict(target=target, location=str(place), target_end_date=ends, output_type='quantile', quantile=q, value=qvals)))
        return pd.concat(rows, ignore_index=True)

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
        self.check_run('extinction')
        base = self.sims[0]
        rows = []
        for dis in getattr(base, 'declared', []):
            infected = [s for s in dis.graph.infected if s.people]
            if not infected or dis.name not in self.results:
                continue
            res = self.results[dis.name]
            first = res[f'n_{infected[0].name}']
            n = sum(res[f'n_{s.name}'].values for s in infected) # Runs by steps
            for label,inds in first.scenarios().items():
                runs = n[inds]
                over = (runs[:, -1] <= 0) & (runs[:, 0] > 0)
                times = [first.timevec[len(row) - np.argmax(row[::-1] > 0)] for row in runs[over]] if over.any() else [] # The step after the last with anyone infected
                scenario = label if len(first.scenarios()) > 1 else None
                rows.append(dict(disease=dis.name, scenario=scenario, runs=len(runs), share=over.mean(), median_time=float(np.median(times)) if times else None))
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
        Plot a bar chart comparing the runs, with one panel per result; see `msim.compare()`.

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

    def plot(self, key=None, fig=None, legend=True, lines=False, **kwargs):
        """
        Plot the results across the runs: the median and a 95% interval of each scenario, or every run if `lines=True`

        Args:
            key (str/list): the result (e.g. `'sir.n_I'`) or results to plot; by default, those that `sim.plot()` shows
            fig (Figure): if provided, plot results into an existing figure
            legend (bool): whether to label the scenarios, if there is more than one
            lines (bool): plot every run instead of the median and interval
            **kwargs (dict): known arguments (e.g. figsize, font) split between the plotting dicts; see `ss.plot_args()` for all valid options
        """
        self.check_run('plot')
        return self.results.plot(key=key, fig=fig, legend=legend, lines=lines, **kwargs)

    @classmethod
    def merge(cls, *args):
        """
        Merge several MultiSims into one, with the runs of each; see also `msim.split()`.

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate arguments)

        Returns:
            A new MultiSim

        Examples:
            ```python
            m1 = ss.MultiSim(ss.Sim(diseases='sis', networks='random', label='SIS'), n_runs=3).run()
            m2 = ss.MultiSim(ss.Sim(diseases='sis', networks='random', label='SIS 2'), n_runs=3).run()
            msim = ss.MultiSim.merge(m1, m2) # 6 runs
            ```
        """
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0] # A single list of MultiSims has been provided
        msim = sc.dcp(args[0])
        msim.sims = [sim for ms in args for sim in sc.dcp(ms.sims)]
        msim.results = MultiResults.concat([ms.results for ms in args])
        msim.runs = pd.concat([ms.runs for ms in args], ignore_index=True)
        starts = np.cumsum([0] + [len(ms) for ms in args])
        msim.chunks = [list(range(a, b)) for a,b in zip(starts[:-1], starts[1:])] # Used to enable automatic splitting later
        return msim

    def split(self, inds=None, chunks=None):
        """
        Split one MultiSim into several, by run; the reverse of `ss.MultiSim.merge()`.

        Specify either the positions of the runs for each new MultiSim (`inds`), or
        consecutive chunks (`chunks`). For a merged MultiSim, neither is needed.

        Args:
            inds (list): a list of lists of positions, with each list turned into a MultiSim
            chunks (int/list): if an int, split the MultiSim into that many equal chunks; if a list, the number of runs in each chunk

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
        self.check_run('split')
        if inds is None:
            if chunks is not None:
                run_inds = np.arange(len(self))
                split = np.cumsum(chunks)[:-1] if sc.isiterable(chunks) else chunks # e.g. chunks=[2,4] or chunks=2
                inds = np.split(run_inds, split) # Raises an exception if the chunks don't divide the runs evenly
            elif hasattr(self, 'chunks'): # Created from a merged MultiSim
                inds = self.chunks
            else:
                errormsg = 'If a MultiSim has not been created via merge(), you must supply either inds or chunks to split it'
                raise ValueError(errormsg)
        out = []
        for indlist in inds:
            indlist = list(indlist)
            msim = sc.dcp(self)
            msim.results = self.results.take(indlist)
            msim.runs = self.runs.iloc[indlist].reset_index(drop=True)
            if len(self.sims) == len(self): # One sim per run
                msim.sims = sc.dcp([self.sims[i] for i in indlist])
            if hasattr(msim, 'chunks'):
                del msim.chunks
            out.append(msim)
        return out


def make_run(sim, values, seed):
    """ The sim of one run: a copy of a sim that has not been initialized, otherwise one built again (see `sim.new()`), with a seed and values of model parameters (`rand_seed` among them is the seed) """
    values = dict(values or {})
    seed = values.pop('rand_seed', seed)
    if sim.initialized:
        new = sim.new(pars=values or None, rand_seed=seed)
    else: # A copy, which keeps any changes made to the sim after it was made
        new = sim.copy()
        new.pars.rand_seed = seed
        if values:
            new.par_values = sc.mergedicts(new.par_values, values)
    new.label = sim.label
    return new


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
        sim = sim.copy() # Make a copy to avoid modifying the original sim (typically done by default via pickling with ss.parallel())

    if not sim.label:
        sim.label = f'Sim {ind}'

    if reseed:
        sim.pars['rand_seed'] += ind  # Reset the seed, otherwise no point of parallel runs
        if ind and sim.initialized:
            ss.warn(f'Sim "{sim.label}" is already initialized, so changing its seed has no effect; pass an uninitialized sim instead')

    # Handle additional arguments
    for key, val in sim_args.items():
        if key in sim.pars.keys():
            if verbose >= 2:
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




def run_each(sims, parallel=True, n_cpus=None, par_args=None, shrink=None, verbose=None, run_args=None):
    """
    Run each of a list of sims, in parallel using multiprocessing (or, with `parallel=False`, in a loop)

    Note: `n_cpus=1` is *not* the same thing as setting `parallel=False`. The former
    still uses the parallelization methods (just on a single core), while the latter
    simply runs in a loop.

    Returns the list of run sims (copies, since they are pickled to run in parallel).
    """
    kwargs = dict(reseed=False, verbose=verbose, shrink=shrink, run_args=run_args)
    if parallel:
        par_args = sc.mergedicts({'ncpus': n_cpus}, par_args)
        return sc.parallelize(ft.partial(ss.kernels.one_thread, single_run), iterkwargs=dict(sim=sims, ind=np.arange(len(sims))), kwargs=kwargs, **par_args) # The runs share the cores, so each takes one thread
    return [single_run(sim, ind=i, copy_sim=True, **kwargs) for i,sim in enumerate(sims)] # A fresh sim, as pickling gives when running in parallel


class MultiResults(ss.Results):
    """
    The results of the runs of a MultiSim: the results of a sim, each with a run axis first

    Each result is an `ss.Result` (see there for what one across runs does), e.g. `msim.results.sir.n_I.quantiles()`.
    """
    def walk(self):
        """ Each result, as (module name, key, result), with the module None for the sim's own results """
        for key,val in self.items():
            if isinstance(val, ss.Results):
                for k,res in val.items():
                    if isinstance(res, ss.Result):
                        yield key, k, res
            elif isinstance(val, ss.Result):
                yield None, key, val

    def to_df(self, quantiles=None, **kwargs):
        """ Every result across the runs, as a long dataframe with columns for the module and result: one row per run and time (see `ss.Result.to_df()`), or with `quantiles=`, per scenario, quantile, and time """
        dfs = []
        for modname,key,res in self.walk():
            parts = res.by_column().items() if res.columns is not None else [(None, res)]
            for col,sub in parts:
                df = sub.to_df(quantiles=quantiles, **kwargs)
                df.insert(0, 'module', modname or 'sim')
                df.insert(1, 'result', key if col is None else f'{key}_{col}')
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def diff(self, baseline):
        """ The difference of every result from the baseline scenario: a dict by scenario, then module, then result """
        out = sc.objdict()
        for modname,key,res in self.walk():
            if baseline in res.scenarios():
                for label,diff in res.diff(baseline).items():
                    entry = out.setdefault(label, sc.objdict())
                    if modname is None:
                        entry[key] = diff
                    else:
                        entry.setdefault(modname, sc.objdict())[key] = diff
        return out

    def plot(self, key=None, lines=False, fig=None, legend=True, **kwargs):
        """
        Plot each result across the runs: the median and a 95% interval for each scenario, or every run if `lines=True`

        Args:
            key (str/list): the result (e.g. `'sir.n_I'`) or results to plot; by default, those that `sim.plot()` shows
            lines (bool): plot every run instead of the median and interval
            fig (Figure): if provided, plot into an existing figure
            legend (bool): whether to label the scenarios, if there is more than one
            kwargs (dict): passed to `ss.plot_args()`
        """
        flat = sc.objdict({key if modname is None else f'{modname}.{key}':res for modname,key,res in self.walk()})
        norm = lambda k: k.replace('.', '_').lower()
        flatkeys = {norm(k):k for k in flat} # Also as Starsim 3 wrote them, e.g. 'sis_new_infections'
        if key is None:
            keys = [k for k,res in flat.items() if res.auto_plot]
        else:
            keys = []
            for k in sc.tolist(key):
                if k not in flat and isinstance(self.get(k), ss.Results): # A module: its results that sim.plot() shows
                    keys += [f'{k}.{rk}' for rk,res in self[k].items() if isinstance(res, ss.Result) and res.auto_plot]
                else:
                    keys.append(k if k in flat else flatkeys.get(norm(k), k))
        kw = ss.plot_args(kwargs)
        with ss.style(**kw.style):
            if fig is None:
                fig, axs = sc.getrowscols(len(keys), make=True, **kw.fig)
                axs = np.array(axs).flatten()
            else:
                axs = fig.axes
            many = False
            for ax,k in zip(axs, keys):
                res = flat[k]
                res.plot(ax=ax, lines=lines, plot_kw=kw.plot, fill_kw=kw.fill)
                ax.set_title(res.label)
                many = many or len(res.scenarios()) > 1
            if legend and many:
                axs[0].legend(**kw.legend)
        return ss.return_fig(fig, **kw.return_fig)

    @classmethod
    def build(cls, arrays, runs, timevec=None):
        """
        The results of many runs, each with a run axis first: the one constructor that every way of running many sims ends in

        Args:
            arrays (dict): for each result, by (module, key), with the module None for the sim's own results: the result of one run it is like (`like`, for its name, kind, label, time points, and dimensions), its `values` (runs by time points), and if it is kept by group, `strata` and `denom` (runs by rows by groups) and `group_ti` (see `ss.Sim.group_rows()`)
            runs (dataframe): one row per run: its scenario, and the values of any parameters swept
            timevec (array): the sim's time points
        """
        out = cls('MultiSim', strict=False)
        for (modname,key),arr in arrays.items():
            like = arr['like']
            values = np.asarray(arr['values'])
            res = like.asnew(values=values, _shape=values.shape, runs=runs, low=None, high=None)
            if getattr(like, 'strata', None) is not None or arr.get('strata') is not None:
                res.strata, res.denom = arr.get('strata'), arr.get('denom')
                res.group_ti = arr.get('group_ti', getattr(like, 'group_ti', None))
                if res.strata is None:
                    res.no_groups = 'it was not kept by group in every run'
            if modname is None:
                out[key] = res
            else:
                if modname not in out:
                    out[modname] = ss.Results(modname, strict=False)
                out[modname][key] = res
        if timevec is not None:
            out['timevec'] = timevec
        return out

    @classmethod
    def from_sims(cls, sims, runs):
        """ Every result that every sim has (with the same shape), across the sims: one run per sim """
        def get(sim, modname, key):
            try:
                return sim.results[key] if modname is None else sim.results[modname][key]
            except (KeyError, AttributeError, TypeError):
                return None

        first = sims[0]
        entries = [(None, key, res) for key,res in first.results.items() if isinstance(res, ss.Result)]
        entries += [(modname, key, res) for modname,modres in first.results.items() if isinstance(modres, ss.Results) for key,res in modres.items() if isinstance(res, ss.Result)]
        arrays = {}
        for modname,key,res in entries:
            found = [get(sim, modname, key) for sim in sims]
            if not all(isinstance(r, ss.Result) and r.values is not None for r in found):
                continue
            values = [np.asarray(r.values) for r in found]
            if len({v.shape for v in values}) != 1:
                continue
            arr = dict(like=res, values=np.stack(values))
            strata = [getattr(r, 'strata', None) for r in found]
            if all(x is not None for x in strata) and len({x.shape for x in strata}) == 1:
                arr['strata'] = np.stack(strata)
                denom = [getattr(r, 'denom', None) for r in found]
                arr['denom'] = np.stack(denom) if all(x is not None for x in denom) else None # A ratio's denominator by group, so that selecting groups adds up each part before dividing
            arrays[modname, key] = arr
        return cls.build(arrays, runs, timevec=first.results.get('timevec'))

    def take(self, inds):
        """ The results of some of the runs, by position """
        arrays = {}
        for modname,key,res in self.walk():
            sub = res.take(inds)
            arrays[modname, key] = dict(like=sub, values=sub.values, strata=getattr(sub, 'strata', None), denom=getattr(sub, 'denom', None))
        runs = next(iter(arrays.values()))['like'].runs if arrays else None
        return MultiResults.build(arrays, runs, timevec=self.get('timevec'))

    @classmethod
    def concat(cls, results):
        """ The results of several sets of runs, one after the other (each result that all of them have) """
        arrays = {}
        first = results[0]
        others = [{(m, k):r for m,k,r in res.walk()} for res in results]
        runs = pd.concat([next(iter(o.values())).runs for o in others], ignore_index=True)
        for modname,key,res in first.walk():
            found = [o.get((modname, key)) for o in others]
            if any(r is None for r in found):
                continue
            join = lambda attr: None if any(getattr(r, attr, None) is None for r in found) else np.concatenate([getattr(r, attr) for r in found])
            arrays[modname, key] = dict(like=res, values=join('values'), strata=join('strata'), denom=join('denom'))
        return cls.build(arrays, runs, timevec=first.get('timevec'))


def par_sets(pars):
    """ A list of sets of parameter values, from a list of dicts or a dict of lists (one value per set) """
    if pars is None:
        return [None]
    if isinstance(pars, dict):
        pars = {key:list(vals) if isinstance(vals, (np.ndarray, tuple, range)) else sc.tolist(vals) for key,vals in pars.items()} # E.g. np.linspace()
        lens = {len(vals) for vals in pars.values()}
        if len(lens) != 1:
            errormsg = f'Each parameter in pars= needs the same number of values, not {sc.strjoin(sorted(lens))}'
            raise ValueError(errormsg)
        n = lens.pop()
        return [{key:vals[i] for key,vals in pars.items()} for i in range(n)]
    return [dict(p) for p in pars]


jax_runs = 300 # The number of runs of an ODE with different values from which ss.parallel() takes them in one call to JAX: about 0.5 s to set up, then faster than the kernels (a three-state model crosses over at about 300 runs)

def restart_runs(sims, n_runs, pars=None):
    """
    Run each sim a number of times, each with its own seed, by building it once and restarting it

    Only for sims that are nothing but compartments (see `ss.Sim.restart()`), where this
    is many times faster than running copies. Returns None if any of the sims is not.
    Otherwise returns the sims (one for each that was given, as it was after its last run),
    the results of every run (`ss.run.MultiResults`), and a dataframe of the runs.

    With `pars=` (see `ss.parallel()`), each set of values is run `n_runs` times. An ODE
    with at least `jax_runs` runs (and JAX installed) takes them all in one call to JAX
    (see `ss.gradients`), which is faster than the kernels for that many.
    """
    declared = lambda sim: any(getattr(mod, 'graph', None) is not None for val in sim.inputs.values() for mod in sc.tolist(val)) # Only a sim with a declared module can be run whole
    if not all(hasattr(sim, 'inputs') and declared(sim) for sim in sims): # A sim that has been initialized or run is rebuilt from its inputs too
        return None
    sets = par_sets(pars)
    if any('rand_seed' in p for p in sets if p is not None): # Each run with a seed given is built
        return None
    runners = []
    for sim in sims:
        runner = sim.new(rand_seed=sim.pars.rand_seed).init() # The seed as it is now, which may have been set since the sim was made
        runner.label = sim.label
        if not runner.kernel_only():
            return None
        keys = {key for p in sets if p is not None for key in p}
        for net in getattr(runner, 'mixed', []): # A network's parameter sets the contact rates when the sim is built, so each run with a value for one is built
            if any(key.split('.')[-1] in net.pars and ('.' not in key or key.split('.')[0] == net.name) for key in keys):
                return None
        runners.append(runner)
    runs = pd.DataFrame([dict(scenario=sim.label or '', **(p or {})) for sim in runners for p in sets for i in range(n_runs)])
    values = sc.ddict(list)
    groups = sc.ddict(list)
    denoms = sc.ddict(list)
    group_ti = dict()
    for sim in runners:
        seed = sim.pars.rand_seed
        if pars is not None and len(sets)*n_runs >= jax_runs and ss.gradients.reason(sim) is None: # Every run in one call to JAX (an ODE run does not depend on the seed)
            outs = list(ss.gradients.run_sets(sim, [p for p in sets for i in range(n_runs)]).items())
        else:
            outs = []
            for p in sets: # All the runs of each module with each set of values are one call to the kernels
                if p is not None:
                    sim.restart(pars=p)
                for mod in sim.declared:
                    cross = mod.engine.cross
                    if cross is None:
                        outs.append((mod.name, mod.engine.run_many(seed + np.arange(n_runs))))
                    elif mod.name == cross.names[0]: # Diseases tracked jointly are run together
                        outs += cross.run_many(seed + np.arange(n_runs))
        for modname,out in list(outs):
            outs += list(out.pop('deliveries', {}).items()) # The results of the deliveries the kernels took, as those of modules of their own
        for modname,out in outs:
            group_ti[modname] = out.pop('group_ti', None) # The time point of each row of the values by group, if they are kept every few steps
            for key,vals in out.pop('denoms', {}).items():
                denoms[modname, key].append(vals)
            for key,vals in out.pop('groups').items():
                groups[modname, key].append(vals)
            for key,vals in out.items():
                values[modname, key].append(vals)
        sim.restart(pars=sets[-1], rand_seed=seed + n_runs - 1).run() # The sim itself is left as its last run
    arrays = {}
    join = lambda arrs: arrs[0] if len(arrs) == 1 else np.concatenate(arrs) # Without a copy for one sim, since by group they can be large
    for (modname,key),vals in values.items():
        res = next(sim.results[modname][key] for sim in runners if modname in sim.results) # A delivery may be in some of the sims only
        vals = join(vals)
        if len(vals) == len(runs): # A result that every sim has
            strata, denom = [join(d[modname, key]) if (modname, key) in d else None for d in [groups, denoms]]
            strata = strata if strata is not None and len(strata) == len(runs) else None
            arrays[modname, key] = dict(like=res, values=vals, strata=strata, denom=denom if strata is not None else None, group_ti=group_ti.get(modname))
    return runners, MultiResults.build(arrays, runs), runs


def parallel(*args, n_runs=None, pars=None, priors=False, **kwargs):
    """
    Run many sims: different sims, each sim many times with its own seed, or with sets of values of its parameters

    This is the one way to run many sims; `sim.run(n_runs=...)` is a shortcut for running one sim
    many times. Every way of running ends in the same results: each result with a run axis first.

    Args:
        args (list): The simulations to run
        n_runs (int): if provided, run each sim this many times, each with its own seed (the sim's seed, plus 1, 2, ...)
        pars (dict/list): sets of values of parameters to run each sim with: a dict with a list of values for each parameter (`dict(beta=[0.2, 0.3])`), or a list of dicts; each set is run `n_runs` times (default 1), and the values are in `msim.runs` (and can be selected, e.g. `res.select(beta=0.2)`); `rand_seed` can be given too, e.g. `pars=dict(rand_seed=[1, 5, 9])`
        priors (bool): draw `n_runs` sets of values from the priors of the parameters (`ss.par(prior=...)`, or uniform over its bounds), one run each: a projection that carries the uncertainty in the values
        kwargs (dict): sims passed as keywords are labelled with the keyword, e.g. `ss.parallel(baseline=s1, vaccine=s2)`;
            everything else is passed to `ss.MultiSim()`, e.g. `parallel=False`, `n_cpus`, `shrink`

    Returns:
        A run MultiSim. Each result is available across the runs, with the label of each sim as its scenario,
        e.g. `msim.results.sir.n_I.quantiles([0.025, 0.5, 0.975])`.

    Examples:
        ```python
        s1 = ss.Sim(n_agents=1000, label='Small', diseases='sis', networks='random')
        s2 = ss.Sim(n_agents=2000, label='Large', diseases='sis', networks='random')
        ss.parallel(s1, s2).plot()
        msim = ss.parallel([s1, s2], shrink=False)
        msim = ss.parallel(s1, n_runs=10) # Ten runs of one sim (also sim.run(n_runs=10))
        msim = ss.parallel(baseline=s1, vaccine=s2, n_runs=10) # Two scenarios; see msim.diff('baseline')
        msim = ss.parallel(sim, pars=dict(beta=np.linspace(0.1, 0.5, 100))) # One run with each value of beta
        ```
    """
    sims = sc.mergelists(*args)
    for key in list(kwargs.keys()): # Named sims
        if isinstance(kwargs[key], ss.Sim):
            sim = kwargs.pop(key)
            sim.label = key
            sims.append(sim)
    return MultiSim(sims, n_runs=n_runs, pars=pars, priors=priors, **kwargs).run()
