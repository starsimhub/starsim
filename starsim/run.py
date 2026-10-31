"""
Running many sims: ss.parallel() and ss.MultiSim

These exist so that a model can be run many times, across seeds, parameter values, or scenarios, with the results gathered in one place for comparing and plotting.

Used with: agents and compartments; declared and coded modules
"""
import collections.abc as cabc
import time
import functools as ft
import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss



run_options = ['parallel', 'n_cpus', 'par_args', 'shrink', 'verbose', 'run_args'] # How to run the sims, as opposed to which runs to take
removed_methods = ['reduce', 'mean', 'median'] # Starsim 3's, which reduced the sims in place


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
        priors (bool/dict): draw `n_runs` sets of values from the priors of the parameters, or the draws to run; see `ss.parallel()`
        initialize (bool): whether to make and initialize the sim of each run now (otherwise, when it is run)
        inplace (bool): whether to update sims that are run as they are (with no `n_runs` or `pars`) with their results (default True)
        debug (bool): if True, run in serial
        kwargs (dict): how to run the sims: `parallel` (default True), `n_cpus`, `par_args` (passed to `sc.Parallel()`), `shrink`, `verbose`, and `run_args` (passed to `sim.run()`)

    By default (`verbose` not given: the sims' own `verbose`, if they share one, else `ss.options.verbose`) the runs print
    nothing of their own, and the MultiSim shows their progress: nothing for
    runs that are done within a second, a bar for each run if there are up to 10 that each take more than 10 seconds, and
    otherwise one bar with the number of runs done (to a file, a line every tenth); then a line on what was run.

    Once run, a MultiSim has:

    - `results`: every result with a run axis first (see `ss.Result`), e.g. `msim.results.sir.I.quantiles()`;
    - `runs`: one row per run, with its scenario (the label of its sim), the values of any parameters swept or drawn, and its seed (`rand_seed`);
    - `sims`: the sims that were run (for runs restarted from one compartmental sim, that sim, as it was after its last run).

    `len(msim)` is the number of runs.

    A subclass changes the sim of each run by overriding `make_run()`, and can take options of its own for it by adding
    them to `run_options` (see the last example).

    Examples:
        ```python
        import starsim as ss

        s1 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.02), label='Low transmission')
        s2 = ss.Sim(networks='random', diseases=ss.SIS(beta=0.05), label='High transmission')
        msim = ss.parallel(s1, s2, n_runs=5)
        msim.plot() # The median and a 95% interval of each scenario
        msim.results.sis.prevalence.select(scenario='Low transmission').median()

        class NoisyMultiSim(ss.MultiSim): # Each run with its own beta, drawn around 0.05
            run_options = ss.MultiSim.run_options + ['noise']

            def make_run(self, sim, values, seed, noise=0.1):
                beta = 0.05*np.exp(noise*np.random.default_rng(seed).normal())
                return super().make_run(sim, dict(values or {}, beta=beta), seed)

        msim = NoisyMultiSim(s1, n_runs=10, noise=0.2).run()
        ```
    """
    run_options = run_options # Those of a subclass go to its make_run()

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

        unknown = set(kwargs) - set(self.run_options)
        if unknown:
            errormsg = f'Unknown argument{"s" if len(unknown) > 1 else ""} {sc.strjoin(sorted(unknown))} for running many sims; the options are n_runs, pars, priors, inplace, debug, and {sc.strjoin(self.run_options)}'
            if 'iterpars' in unknown:
                errormsg += ' (iterpars is now pars, e.g. pars=dict(rand_seed=[1, 2, 3]))'
            raise TypeError(errormsg)

        # Set properties
        self.sims = sims
        self.base_sim = base_sim
        self.label = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args = sc.mergedicts(dict(n_runs=n_runs, pars=pars, priors=priors, inplace=inplace, debug=debug), kwargs)
        verbosities = {sim.pars.verbose for sim in sc.tolist(sims) or [base_sim] if isinstance(sim, ss.Sim)}
        self.verbose = verbosities.pop() if len(verbosities) == 1 else None # The sims' verbose, if they share one (e.g. ss.Sim(verbose=0).run(n_runs=10) prints nothing)
        self.results = None # Every result, with a run axis first
        self._same_times = True # Whether the sims have the same time points, so their results can be put together as runs (see collect())
        self.runs = None # One row per run: its scenario, the values of any parameters swept, and its seed
        self.summary = None
        self.ready = False # Whether the sim of each run has been made, by init_sims()
        self.timer = sc.timer() # Create a timer

        # Sims that have already run: their results across runs, without running them again
        if sims is not None and all(sim.results_ready for sim in sims):
            self.sims, self.runs = self.expand(sims)
            self.collect(self.sims)

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
    def sims(self):
        """ The sim of each run, by position (`msim.sims[0]`) or by scenario (`msim.sims['Vaccine']`: its sim, or a list of them if it has several runs; and `msim.sims.items()`), each labeled by its scenario """
        sims = self.__dict__.get('_sims', self.__dict__.get('sims'))
        runs = self.__dict__.get('runs')
        if isinstance(sims, (Sims, RestartedSims)):
            sims.scenarios = list(runs['scenario']) if runs is not None and len(runs) == len(sims) else None
        return sims

    @sims.setter
    def sims(self, sims):
        self.__dict__['_sims'] = Sims(sims) if isinstance(sims, (list, tuple)) and not isinstance(sims, Sims) else sims
        return

    @property
    def branching(self):
        """ For the runs of a branching process (`ss.Sim(branching=True)`): what came of them together, as an `ss.run.Branching`; None if the runs are not branching processes (see `sim.branching` for one run) """
        sims = list(self.sims or [])
        runs = [sim.branching for sim in sims]
        if not runs or any(b is None for b in runs):
            return None
        if 'size' not in runs[0]: # Several declared modules: by module
            return sc.obj({name:Branching([b[name] for b in runs]) for name in runs[0].keys()})
        return Branching(runs)

    @property
    def run_pars(self):
        """ The values of the parameters swept (or drawn) in each run, as a dataframe (or None if there are none) """
        if self.runs is None or not len(self.runs.columns.difference(['scenario', 'rand_seed'])):
            return None
        return self.runs.drop(columns=['scenario', 'rand_seed'], errors='ignore')

    def resolve(self, n_runs=None, pars=None, priors=False):
        """ The sims to run, how many times to run each, the sets of values to run them with, and whether those are draws (runs of each scenario) rather than a sweep (scenarios of their own) """
        sims = [self.base_sim] if self.sims is None else list(getattr(self.sims, 'runners', self.sims)) # Sims restarted for their runs are run again from the sim of each scenario
        draws = priors is not None and priors is not False
        if n_runs is not None and n_runs == int(n_runs): # A whole number given as a float, e.g. 20 from R
            n_runs = int(n_runs)
        if self.sims is None and n_runs is None and pars is None and not draws:
            n_runs = 4 # A single sim given to a MultiSim is run four times by default
        if len(sims) > 1: # Different sims are different scenarios
            for i,sim in enumerate(sims):
                if not sim.label:
                    sim.label = f'Sim {i}'
        many = n_runs is not None or pars is not None or draws
        sims = [sim.new(rand_seed=sim.pars.rand_seed, label=sim.label) if sim.results_ready or (many and sim.initialized) else sim for sim in sims] # A sim that has been run (e.g. fit.sim), or that would be run more than once after init(), is built again
        if draws:
            if pars is not None:
                errormsg = 'ss.parallel() takes pars= or priors=, not both'
                raise ValueError(errormsg)
            if priors is True:
                from .fit import prior_draws
                pars = prior_draws(sims[0], sc.ifelse(n_runs, 100), seed=sims[0].pars.rand_seed)
            else:
                if n_runs is not None:
                    errormsg = 'With the draws given as priors=, each is run once; pass more draws rather than n_runs'
                    raise ValueError(errormsg)
                pars = priors.to_dict('records') if isinstance(priors, pd.DataFrame) else priors
            n_runs = 1
        if pars is not None:
            n_runs = sc.ifelse(n_runs, 1)
        return sims, n_runs, pars, draws

    def expand(self, sims, n_runs=None, pars=None, draws=False, **opts):
        """ The sim of each run, each with its own seed (see `run_seed()`) and set of values (see `make_run()`), labeled by its scenario, and a dataframe of the runs; a set of values swept is a scenario of its own, and a draw (`draws=True`) or a seed swept is a run of its sim's scenario """
        sets = par_sets(pars)
        out = []
        rows = []
        for sim in sims:
            for j,p in enumerate(sets):
                for i in range(sc.ifelse(n_runs, 1)):
                    seed = run_seed(sim.pars.rand_seed, p, j, i, n_runs, draws)
                    row = run_row(sim, p, seed, draws)
                    run = sim if n_runs is None and p is None else self.make_run(sim, p, seed, **opts) # A sim run once as it is is run itself
                    if row['scenario'] and run.label != row['scenario']: # Labeled by its scenario, e.g. with the values swept
                        run.label = row['scenario']
                    out.append(run)
                    rows.append(row)
        return out, pd.DataFrame(rows)

    def make_run(self, sim, values, seed):
        """
        The sim of one run: a new one (see `sim.new()`), with a seed and values of model parameters

        A subclass can override this to change the sim of each run, e.g. to add noise to a parameter; the options it adds to
        `run_options` are passed to it as keywords.

        Args:
            sim (Sim): the sim of the scenario
            values (dict): the values of parameters for this run (from `pars=`), or None; `rand_seed` among them is the seed
            seed (int): the seed of this run
        """
        values = dict(values or {})
        seed = values.pop('rand_seed', seed)
        new = sim.new(pars=values or None, rand_seed=seed)
        new.label = sim.label
        return new

    def own_options(self, kwargs):
        """ Take the options a subclass added to `run_options` out of kwargs, for its `make_run()` """
        return {key:kwargs.pop(key) for key in self.run_options if key not in run_options and key in kwargs}

    def __getattr__(self, attr):
        """ Called only if the attribute isn't found: what replaced Starsim 3's methods that reduced the sims """
        if attr in removed_methods:
            errormsg = f'msim.{attr}() was removed: the results of a MultiSim have a run axis first, e.g. res = msim.results.sis.prevalence, then res.{attr}() (by scenario), or low, mid, high = res.quantiles([0.1, 0.5, 0.9])'
            raise AttributeError(errormsg)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def init_sims(self, **kwargs):
        """ Make the sim of each run, and initialize them without running them

        Args:
            kwargs (dict): the same arguments as `msim.run()`
        """
        kwargs = sc.mergedicts(self.run_args, kwargs)
        sims, n_runs, pars, draws = self.resolve(kwargs.get('n_runs'), kwargs.get('pars'), kwargs.get('priors'))
        self.sims, self.runs = self.expand(sims, n_runs, pars, draws, **self.own_options(kwargs))
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
        opts = self.own_options(kwargs)
        if kwargs.pop('debug', False):
            kwargs['parallel'] = False # Run in serial
        self.timer.start()
        verbose = sc.ifelse(kwargs.get('verbose'), getattr(self, 'verbose', None), ss.options.verbose)
        kwargs.setdefault('verbose', 0) # The runs say nothing, unless asked to; the MultiSim says what it runs
        if self.ready: # The sim of each run was made by init_sims()
            sims = self.sims
        else:
            base, n_runs, pars, draws = self.resolve(n_runs, pars, priors)
            plain = not opts and type(self).make_run is MultiSim.make_run # No changes to the sim of each run
            if n_runs is not None and plain and set(kwargs) <= {'verbose', 'n_cpus', 'parallel'}: # Built once and restarted for each run, where the sims allow, which is many times faster
                restarted = restart_runs(base, n_runs, pars=pars, draws=draws)
                if restarted is not None:
                    self.sims, self.results, self.runs = restarted
                    self.ready = False
                    self.timer.stop()
                    self.say(verbose)
                    return self
            sims, self.runs = self.expand(base, n_runs, pars, draws, **opts)

        progress = ss.display.RunsProgress(sims) if verbose and not kwargs['verbose'] else None # The runs' own progress, if they say nothing
        run_sims = run_each(sims, progress=progress, **kwargs) # This does all the work! Output sims are copies due to the pickling during parallelization
        if inplace:
            for old,new in zip(sims, run_sims):
                if old is not new:
                    old.__dict__.update(new.__dict__) # Update the same object with the new results
        self.sims = run_sims
        self.collect(run_sims)
        self.ready = False
        self.timer.stop()
        self.say(verbose, progress)
        return self

    def say(self, verbose, progress=None):
        """ Say what was run, and how long it took: in place of the runs' progress, if it was shown """
        if verbose:
            scenarios = list(dict.fromkeys(self.runs['scenario'])) if self.runs is not None else []
            n = len(self.runs) if self.runs is not None else len(self.sims)
            which = f' of {len(scenarios)} scenarios ({sc.strjoin(scenarios)})' if len(scenarios) > 1 else f' of "{scenarios[0]}"' if scenarios and scenarios[0] else ''
            took = f'{self.timer.elapsed:.2g}' if self.timer.elapsed < 1 else f'{self.timer.elapsed:.1f}'
            text = f'MultiSim: {n} run{"s" if n != 1 else ""}{which}\n  Done in {took} s'
            if progress is not None:
                progress.finish(text)
            else:
                print(text)
        return

    def collect(self, sims):
        """
        Put the results of the sims together as runs (`msim.results`), if they have the same time points (`msim._same_times`)

        Sims with different time points (e.g. different steps) are kept as they are: each can be plotted on its own time
        axis (`msim.plot()`), but their results can't be put together, so what combines the runs raises an error.
        """
        self._same_times = same_times(sims)
        self.results = MultiResults.from_sims(sims, self.runs) if self._same_times else DifferentTimes(sims)
        return

    def check_run(self, what):
        if not self._same_times:
            self.results.error(f'msim.{what}()')
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
        sims = self.sims.made.values() if isinstance(self.sims, RestartedSims) else self.sims # Of sims restarted for their runs, only those that have been made
        for sim in sims:
            sim.shrink(**kwargs)
        return

    def combine(self):
        """
        Combine the sims into a single sim, whose results are those of one population made of all of them, e.g. to treat several smaller sims as one larger population

        Only the output is combined. Results that scale with population size (`res.scale=True`,
        e.g. counts) are summed; other results (e.g. prevalence) are averaged, weighted by
        each sim's `total_pop`. The population sizes (`n_agents` and `total_pop`) are summed,
        and `pop_scale` is updated to match. The people are not joined: the combined sim keeps
        the people (if any) of the first sim, so it is not a larger population of agents that
        interact, and cannot be run again as one. To keep the runs as runs, see `ss.MultiSim.merge()`.

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
        pars = combined_sim.pars.freeze(False) # Its settings say what it combines (it is not run again)
        pops = np.array([sim.pars.total_pop for sim in self.sims]) # Used to weight the non-count results
        pars.n_agents = sum(sim.pars.n_agents for sim in self.sims)
        pars.total_pop = pops.sum()
        pars.pop_scale = pars.total_pop/pars.n_agents
        pars.freeze()

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

    def summarize(self, key=None, interval=0.95, how=None, baseline=None):
        """
        A table of each result over the whole of each run, across the runs: the median, an interval, the mean, and the standard deviation

        Each result is first summarized over each run by its `summarize_by` (see `sim.summarize()`): a flow (e.g. infections) by its
        total, a stock (e.g. the number recovered) by its final value, and a ratio (e.g. prevalence) by its mean. Results
        that are 0 throughout are kept, so the tables of different runs (and of `sim.summarize()`) line up row for row.

        Args:
            key (str): if given, only this result (e.g. `'sir_infection'` or `'sir.infection'`), as `msim.summarize().loc[key]`: its row (or a row per scenario, if there are several)
            interval (float): the share of runs the interval holds (default 0.95: from the 2.5th to the 97.5th percentile), as `msim.plot()` shows
            how (str/dict): how to summarize each result over each run: 'mean', 'median', 'sum', 'last', 'max', 'min', or a function, or a dict of these by result key, as for `sim.summarize()`
            baseline (str): the label of a scenario to compare with: each other scenario's difference from it, run by run (the runs are paired by position, as in `msim.diff()`), so the interval is of the paired differences

        Returns:
            A dataframe with a row for each result (and scenario, if there are several), and the columns `median`, `low`, `high`, `mean`, and `std`
            (or `value`, if each scenario was run once); with `key`, the rows of that result

        Example:
            ```python
            msim = sim.run(n_runs=20)
            msim.summarize()                     # Every result
            msim.summarize('sir_infection')      # The total infections
            msim.summarize(how=dict(sir_I='max')) # The peak number infectious in each run, over the runs
            msim.summarize(baseline='No intervention') # Each scenario's difference from the baseline, e.g. the infections averted (negative)
            ```
        """
        self.check_run('summarize')
        how = ss.results.summary_funcs(how, self.results)
        low, high = (1 - interval)/2, 1 - (1 - interval)/2
        rows = {}
        n_scenarios = len(set(self.runs['scenario'])) if self.runs is not None else 1 # A result of only some of the runs (e.g. of an intervention one scenario has) has fewer
        if baseline is not None and (self.runs is None or baseline not in set(self.runs['scenario'])):
            errormsg = f'No runs are labeled "{baseline}"; the labels are {list(dict.fromkeys(self.runs["scenario"])) if self.runs is not None else []}'
            raise KeyError(errormsg)
        flat = self.results.flatten(sep='_', columns=True)
        one = key is not None
        if one: # Only the result asked for
            flat = ss.results.match_result_keys(flat, key, show_skipped=True, flattened=True)
        for name,res in flat.items():
            func = how.get(name, how.get(''))
            if func is None and res.summarize_by is None: # Not summarized over time
                continue
            if func is None:
                arr = np.asarray(res._summarize(), dtype=float) # One value per run
            else:
                arr = np.array([func(vals) for vals in np.asarray(res.values)], dtype=float)
            groups = res.scenarios() if res.runs is not None else {None:list(range(len(arr)))}
            if baseline is not None and baseline not in groups: # A result the baseline does not have (e.g. of an intervention)
                continue
            for scenario,inds in groups.items():
                vals = arr[inds]
                if baseline is not None: # The paired differences from the baseline, run by run
                    if scenario == baseline:
                        continue
                    base = arr[groups[baseline]]
                    if len(base) != len(vals):
                        errormsg = f'Cannot compare "{scenario}" with "{baseline}": they have {len(vals)} and {len(base)} runs'
                        raise ValueError(errormsg)
                    vals = vals - base
                stats = dict(median=np.median(vals), low=np.quantile(vals, low), high=np.quantile(vals, high), mean=vals.mean(), std=vals.std()) if len(vals) > 1 else dict(value=vals[0])
                rows[name if n_scenarios == 1 else (name, scenario)] = {k:float(v) for k,v in stats.items()}
        several = any(isinstance(row, tuple) for row in rows)
        index = pd.MultiIndex.from_tuples(list(rows), names=['result', 'scenario']) if several else pd.Index(list(rows), name='result')
        summary = sc.dataframe(list(rows.values()), index=index)
        if one:
            names = list(dict.fromkeys(summary.index.get_level_values(0))) if len(summary) else []
            if not names:
                errormsg = f'"{key}" has no summary: it is not summarized over time (give how=), or the baseline does not have it'
                raise ValueError(errormsg)
            return summary.loc[names[0]] if len(names) == 1 else summary # A 2D result has rows for each column
        self.summary = summary
        return summary

    def compare(self, t=None, sim_inds=None, output=False, do_plot=False, **kwargs):
        """
        Create a dataframe comparing the runs, with one column per run and one row per result.

        Args:
            t (int/str/date): if None, compare each result over the whole run (see `sim.summarize()`); else, the timestep index or date to compare the results at
            sim_inds (list): the positions of the runs to include (default: all)
            output (bool): whether to only return the dataframe (otherwise, also print it in a script; in a notebook, it shows what is returned; see `ss.utils.return_table()`)
            do_plot (bool): whether to also plot the comparison as a bar chart, with one panel per result
            kwargs (dict): for the plot: see `ss.plot_args()`

        Examples:
            ```python
            s1 = ss.Sim(diseases=ss.SIS(beta=0.05), networks='random', label='Low')
            s2 = ss.Sim(diseases=ss.SIS(beta=0.10), networks='random', label='High')
            msim = ss.parallel(s1, s2)
            msim.compare() # Print the summaries
            df = msim.compare(t='2030-01-01', output=True) # Results on that date
            msim.compare(do_plot=True) # Plot the summaries
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
        for key,res in self.results.flatten(sep='_', columns=True).items():
            if t is None:
                if res.summarize_by is None: # Not summarized over time
                    continue
                vals = np.asarray(res._summarize()) # One value per run
            else:
                ti = t if sc.isnumber(t) else sc.findnearest(res.convert_timevec().years, year) # Nearest timestep to the date, using the result's own timevec since modules can have different dt
                vals = np.asarray(res.values)[:, ti]
            pos = getattr(res, 'run_pos', None) # A result of only some of the runs is NaN in the others
            for label,i in zip(labels, inds):
                if pos is not None and i not in pos:
                    resdict[label][key] = np.nan
                    continue
                val = vals[i if pos is None else pos.index(i)]
                if t is not None and res.scale and val == np.round(val): # Show whole-number counts as ints (but not e.g. 4.5 from scaling)
                    val = int(val)
                resdict[label][key] = val

        df = sc.dataframe(resdict, dtype=object) # Object dtype prevents ints being converted to floats
        if do_plot:
            numeric = df[df.map(sc.isnumber).all(axis=1)] # Skip non-numeric rows
            kw = ss.plot_args(kwargs)
            with ss.style(**kw.style):
                fig, axs = sc.getrowscols(len(numeric), make=True, **kw.fig)
                for ax, (key, row) in zip(sc.toarray(axs).flatten(), numeric.iterrows()):
                    ax.barh(row.index, row.values.astype(float), **kw.plot)
                    ax.set_title(key)
            ss.return_fig(fig, **kw.return_fig)
        if output:
            return df
        timestr = 'Summary' if t is None else f'Results for t={t}'
        print(f'{timestr} for each run:')
        return ss.utils.return_table(df) # Printed in a script; in a notebook, shown once, as what is returned

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
        """ Every result across the runs, as quantiles at each time for each scenario: a long dataframe with columns `module`, `result`, `scenario`, `time`, `quantile`, and `value` (with `quantiles=None`, one row per run and time instead)

        Args:
            quantiles (tuple): the quantiles to compute, from 0 to 1
        """
        self.check_run('to_df')
        return self.results.to_df(quantiles=quantiles)

    def plot(self, key=None, fig=None, legend=True, lines=False, show_data=True, **kwargs):
        """
        Plot the results across the runs: the median and a 95% interval of each scenario, or every run if `lines=True`

        Args:
            key (str/list): the result (e.g. `'sir.I'`) or results to plot; by default, those that `sim.plot()` shows
            fig (Figure): if provided, plot results into an existing figure
            legend (bool): whether to label the scenarios, if there is more than one
            lines (bool): plot every run instead of the median and interval
            show_data (bool): plot the data of the first sim that has any, as `sim.plot()` does (a flow with data at a longer period than the step is shown as its totals over the data's periods)
            **kwargs (dict): known arguments (e.g. figsize, font) split between the plotting dicts; see `ss.plot_args()` for all valid options
        """
        if not self._same_times and (lines or self.runs['scenario'].is_unique): # Each run on its own time axis, which needs no combining of the runs
            return self.plot_each(key=key, fig=fig, legend=legend, show_data=show_data, **kwargs)
        self.check_run('plot')
        sims = list(getattr(self.sims, 'runners', self.sims) or []) # Of sims restarted for their runs, the sim of each scenario, which was run
        withdata = next((sim for sim in sims if getattr(sim, 'data', None) is not None and sim.results_ready), None) if show_data else None
        data = ss.results.plot_data(withdata) if withdata is not None else None
        return self.results.plot(key=key, fig=fig, legend=legend, lines=lines, data=data, **kwargs)

    def plot_each(self, key=None, fig=None, legend=True, show_data=True, log=False, **kwargs):
        """ Plot each run on its own time axis, for sims with different time points (see `plot()`): each scenario in a color of its own

        Args:
            key (str/list): the result or results to plot (default: as for `plot()`)
            fig (Figure): the figure to plot on
            legend (bool): whether to show a legend
            show_data (bool): whether to show the sim's data, if any
            log (bool): whether to use a log scale for the y-axis
            kwargs (dict): passed to `ss.plot_args()`
        """
        sims = list(self.sims)
        scenarios = list(self.runs['scenario'])
        def get(sim, res, origin=sims[0]): # The same result of another sim, or of its one disease if the origin's was named differently (see own_disease())
            mod = sim.results.get(res.module)
            if not isinstance(mod, ss.Results) and res.module is not None and res.module == own_disease(origin):
                mod = sim.results.get(own_disease(sim))
            return mod[res.name] if isinstance(mod, ss.Results) and res.name in mod else sim.results.get(res.name)
        keys = list(ss.results.match_result_keys(sims[0].results, key, show_skipped=bool(key)).values())
        if key is None: # Also the results that only later sims have, e.g. of an intervention only some scenarios have
            for sim in sims[1:]:
                seen = {(k.module, k.name) for k in keys}
                keys += [res for res in ss.results.match_result_keys(sim.results, None).values() if get(sims[0], res, sim) is None and (res.module, res.name) not in seen]
        n_cols,_ = sc.getrowscols(len(keys))
        kw = ss.plot_args(kwargs, figsize=np.array([8, 6])*np.clip((n_cols-3)/6+1, 1, 1.5), alpha=0.8, data_alpha=0.3, data_color='k')
        given = fig is not None
        withdata = next((sim for sim in sims if getattr(sim, 'data', None) is not None), None) if show_data else None
        data = ss.results.plot_data(withdata) if withdata is not None else {}
        several = not self.runs['scenario'].is_unique # Several runs of a scenario, drawn faintly, as with lines=True
        with ss.style(**kw.style):
            if fig is None:
                fig, axs = sc.getrowscols(len(keys), make=True, **kw.fig)
                axs = axs.flatten() if isinstance(axs, np.ndarray) else axs
            else:
                axs = fig.axes
            axs = axs if sc.isiterable(axs) else [axs]
            cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['k'])
            colors = {scenario:cycle[i % len(cycle)] for i,scenario in enumerate(dict.fromkeys(scenarios))}
            for ax,first in zip(axs, keys):
                entry = data.get((first.module, first.name))
                if entry is not None:
                    ax.scatter(entry[0], entry[1], **kw.data)
                labeled = set()
                for sim,scenario in zip(sims, scenarios):
                    res = get(sim, first)
                    if res is None: # E.g. of an intervention only some scenarios have
                        continue
                    plot_kw = sc.mergedicts(dict(color=colors[scenario]), kw.plot, dict(alpha=0.3, lw=1) if several else {}, dict(label=None if scenario in labeled else scenario))
                    labeled.add(scenario)
                    ax.plot(res.timevec, res.values, **plot_kw)
                if log:
                    ss.results.log_scale(ax)
                else:
                    first._floor(ax)
                ss.results.format_axes(ax, first)
            if legend and len(colors) > 1:
                axs[0].legend(**kw.legend)
        sc.figlayout(fig=fig)
        return ss.return_fig(fig, given=given, **kw.return_fig)

    @classmethod
    def merge(cls, *args):
        """
        Merge several MultiSims into one, with the runs of each, one after the other; `msim.split()` separates them again.

        Unlike `msim.combine()`, which adds up the sims as one population, the runs stay separate runs.

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
        msim.chunk_labels = [ms.label for ms in args] # So each part of the split has its own label again
        return msim

    def split(self, inds=None, chunks=None):
        """
        Split one MultiSim into several, each with some of its runs; the reverse of `ss.MultiSim.merge()`.

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
        labels = getattr(self, 'chunk_labels', None) if inds is getattr(self, 'chunks', None) else None
        for i,indlist in enumerate(inds):
            indlist = list(indlist)
            msim = sc.dcp(self)
            if labels is not None:
                msim.label = labels[i]
                del msim.chunk_labels
            msim.results = self.results.take(indlist)
            msim.runs = self.runs.iloc[indlist].reset_index(drop=True)
            if len(self.sims) == len(self): # One sim per run
                msim.sims = sc.dcp([self.sims[i] for i in indlist])
            if hasattr(msim, 'chunks'):
                del msim.chunks
            out.append(msim)
        return out


class Branching(sc.prettyobj):
    """
    What came of the runs of a branching process together (`msim.branching`): see `sim.branching` for one run

    Attributes:
        sizes (array): the people infected in each run, the seeds included
        outcomes (array): each run's outcome: `'extinct'`, `'stopped'` (by `until=`), or `'ongoing'`
        p_extinct (float): the share of the runs that went extinct, and `p_extinct_ci` its 95% interval (Wilson's)
        lengths (array): as agents, the most generations of each run's chain
        offspring (array): as agents, the number each person infected, of everyone infected long enough before the end that their offspring are all counted (see `sim.branching`); only if no run was stopped by `until=`, since where a run stops depends on its offspring
        R, k (float): as agents, the mean of the offspring and their dispersion, as a negative binomial fitted by maximum likelihood (k is infinite for offspring no more variable than a Poisson's); None if a run was stopped by `until=`, so to estimate them, run for a fixed `dur` instead

    Example:
        ```python
        sir = ss.SIR(beta=ss.perday(0.3), dur_inf=ss.days(10), init_n=dict(I=1))
        sim = ss.Sim(diseases=sir, total_pop=1e6, branching=True, solver='gillespie', unit='day', dur=365)
        msim = sim.run(n_runs=200, until=sir.I.n + sir.R.n >= 200)
        msim.branching.p_extinct # About 1/R0 = 1/3
        msim.branching.plot()
        ```
    """
    def __init__(self, runs):
        self.sizes = np.array([b.size for b in runs], dtype=float)
        self.outcomes = np.array([b.outcome for b in runs], dtype=object)
        n = len(runs)
        x = int(np.sum(self.outcomes == 'extinct'))
        self.p_extinct = x/n if n else np.nan
        z = 1.96
        mid = (x + z**2/2)/(n + z**2) if n else np.nan
        half = z*np.sqrt(x*(n - x)/n + z**2/4)/(n + z**2) if n else np.nan
        self.p_extinct_ci = (max(mid - half, 0.0), min(mid + half, 1.0))
        agents = all('offspring' in b for b in runs)
        self.lengths = np.array([b.length for b in runs]) if agents else None
        fixed = not any(b.outcome == 'stopped' for b in runs) # A run stopped by until= stops when its chain is large, so the people infected before then had more offspring than most: R and k would be biased
        self.offspring = np.concatenate([b.offspring[b.complete] for b in runs]) if agents and fixed else None
        self.R, self.k = self.fit_offspring(self.offspring) if self.offspring is not None and len(self.offspring) else (None, None)
        return

    @staticmethod
    def fit_offspring(counts):
        """ The mean of some offspring counts, and the dispersion k of a negative binomial with that mean fitted to them by maximum likelihood (infinite if they are no more variable than a Poisson's)

        Args:
            counts (array): the number of offspring of each case
        """
        import scipy.optimize as spo
        import scipy.special as sps
        counts = np.asarray(counts, dtype=float)
        R = counts.mean()
        if R == 0 or counts.var() <= R:
            return float(R), np.inf
        def nll(logk):
            k = np.exp(logk)
            return -np.sum(sps.gammaln(counts + k) - sps.gammaln(k) - sps.gammaln(counts + 1) + k*np.log(k/(k + R)) + counts*np.log(R/(k + R)))
        res = spo.minimize_scalar(nll, bounds=(-10, 10), method='bounded')
        return float(R), float(np.exp(res.x))

    def plot(self, **kwargs):
        """ The distribution of the chains' sizes (and, as agents, of their lengths in generations)

        Args:
            kwargs (dict): passed to `ss.plot_args()`
        """
        agents = self.lengths is not None
        kw = ss.plot_args(kwargs)
        with ss.style(**kw.style):
            fig,axs = sc.getrowscols(2 if agents else 1, make=True, **kw.fig)
            axs = sc.toarray(axs).flatten()
            ax = axs[0]
            sizes = self.sizes[self.sizes > 0]
            bins = np.unique(np.round(np.logspace(0, np.log10(max(sizes.max(), 2)), 30))) if len(sizes) else 10
            ax.hist(sizes, bins=bins)
            ax.set_xscale('log')
            ax.set_xlabel('People infected')
            ax.set_ylabel('Runs')
            ax.set_title(f'Chain sizes (extinct in {self.p_extinct:.0%} of {len(self.sizes)} runs)')
            if agents:
                ax = axs[1]
                ax.hist(self.lengths, bins=np.arange(self.lengths.max() + 2) - 0.5)
                ax.set_xlabel('Generations')
                ax.set_ylabel('Runs')
                ax.set_title(f'Chain lengths (R = {self.R:.2f}, k = {self.k:.2g})' if self.R is not None else 'Chain lengths')
        return ss.return_fig(fig, **kw.return_fig)


def single_run(sim, ind=0, shrink=True, run_args=None, verbose=None, copy_sim=False, track=False, globaldict=None, show=None):
    """
    Run one sim, as `run_each()` does: labeled by its position if it has no label, and shrunk afterwards if `shrink`

    Args:
        sim (Sim): the sim to run
        ind (int): its position among the sims
        shrink (bool): whether to shrink the sim after it has run
        run_args (dict): arguments passed to `sim.run()`
        verbose (int): detail to print
        copy_sim (bool): whether to run a copy of the sim (as pickling gives when running in parallel)
        track (bool): whether to put how far the run has got in `globaldict`, by `ind` (see `ss.display.RunsProgress`)
        globaldict (dict): a dict shared with the other runs (given by `sc.parallelize()`, or by `run_each()` when run in a loop)
        show (func): what shows the progress, when run in this process
    """
    if copy_sim:
        sim = sim.copy()
    if not sim.label:
        sim.label = f'Sim {ind}'
    if track:
        sim.progress = ss.display.Share(sim, globaldict, ind, show)
    sim.run(**sc.mergedicts({'verbose': verbose}, run_args))
    if track:
        globaldict[ind] = 1.0
        if show is not None:
            show()
    if shrink:
        sim.shrink(die=False)
    return sim


def run_each(sims, parallel=True, n_cpus=None, par_args=None, shrink=None, verbose=None, run_args=None, progress=None):
    """
    Run each of a list of sims, in parallel using multiprocessing (or, with `parallel=False`, in a loop)

    Note: `n_cpus=1` is *not* the same thing as setting `parallel=False`. The former
    still uses the parallelization methods (just on a single core), while the latter
    simply runs in a loop.

    With `progress` (an `ss.display.RunsProgress`), each run puts how far it has got in a dict shared with
    this process (Sciris' `globaldict`), from which the progress is drawn ten times a second.

    Args:
        sims (list): the sims to run
        parallel (bool): if True (default), run in parallel; if False, run in a loop
        n_cpus (int): the number of CPUs to use (default: all of them)
        par_args (dict): passed to `sc.parallelize()`
        shrink (bool): whether to shrink each sim after it runs
        verbose (float): how much to print
        run_args (dict): passed to `sim.run()`
        progress (`ss.display.RunsProgress`): the progress to draw

    Returns the list of run sims (copies, since they are pickled to run in parallel).
    """
    kwargs = dict(verbose=verbose, shrink=shrink, run_args=run_args, track=progress is not None)
    if parallel:
        par_args = sc.mergedicts({'ncpus': n_cpus, 'start_method': ss.kernels.start_method(), 'parallelizer': 'multiprocess-async', 'globaldict': {}}, par_args)
        with ss.kernels.starting_workers(par_args['start_method']):
            P = sc.Parallel(ss.kernels.worker(single_run), iterkwargs=dict(sim=sims, ind=np.arange(len(sims))), kwargs=kwargs, **par_args) # The runs share the cores, so each takes one thread
            try:
                P.run_async()
                if progress is not None:
                    progress.store = P.globaldict
                    while P.running:
                        progress.update()
                        time.sleep(0.05)
                P.finalize()
            except BaseException as E:
                P.close() # Without leaving the workers running, as sc.parallelize() does
                ss.kernels.workers_failed(E, par_args['start_method']) # A clear error if they ran the main script again
                raise
        return P.results
    show = None if progress is None else progress.update
    store = {} if progress is None else progress.store
    return [single_run(sim, ind=i, copy_sim=True, globaldict=store, show=show, **kwargs) for i,sim in enumerate(sims)] # A fresh sim, as pickling gives when running in parallel


def same_times(sims):
    """ Whether the sims have the same time points, so that their results can be put together as runs """
    times = [sim.t.tvec for sim in sims]
    return all(len(t) == len(times[0]) and np.all(t == times[0]) for t in times[1:])


class DifferentTimes:
    """ In place of `msim.results`, for sims with different time points (e.g. different steps): any use of it raises an error saying why, and what to do instead """
    def __init__(self, sims):
        self.steps = sc.strjoin(dict.fromkeys(f'"{sim.label}": {len(sim.t.tvec)} steps of {sim.t.dt}' for sim in sims)) # Once for each scenario

    def error(self, what='msim.results'):
        errormsg = f'{what} needs the results of the runs put together, but the sims have different time vectors ({self.steps}). Plot each on its own time axis with msim.plot(lines=True), or use each sim\'s results (msim.sims[0].results); or give the sims the same start, stop, and dt.'
        raise ValueError(errormsg)

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        self.error()

    def __getitem__(self, key):
        self.error()

    def __repr__(self):
        return f'<No results across runs: the sims have different time vectors ({self.steps})>'


class MultiResults(ss.Results):
    """
    The results of the runs of a MultiSim: the results of a sim, each with a run axis first

    Each result is an `ss.Result` (see there for what one across runs does), e.g. `msim.results.sir.I.quantiles()`.
    """
    def to_df(self, quantiles=None, **kwargs):
        """ Every result across the runs, as a long dataframe with columns for the module and result: one row per run and time (see `ss.Result.to_df()`), or with `quantiles=`, per scenario, quantile, and time

        Args:
            quantiles (tuple): the quantiles to compute, from 0 to 1 (default: one row per run)
            kwargs (dict): passed to `ss.Result.to_df()`
        """
        dfs = []
        for modname,key,res in self.walk():
            parts = res.by_column().items() if res.columns is not None else [(None, res)]
            for col,sub in parts:
                df = sub.to_df(quantiles=quantiles, **kwargs)
                df.insert(0, 'module', modname or 'sim')
                df.insert(1, 'result', key if col is None else f'{key}_{col}')
                dfs.append(df)
        return sc.dataframe(pd.concat(dfs, ignore_index=True))

    def diff(self, baseline):
        """ The difference of every result from the baseline scenario: a dict by scenario, then module, then result

        Args:
            baseline (str): the name of the baseline scenario
        """
        out = sc.obj()
        for modname,key,res in self.walk():
            if baseline in res.scenarios():
                for label,diff in res.diff(baseline).items():
                    entry = out.setdefault(label, sc.obj())
                    if modname is None:
                        entry[key] = diff
                    else:
                        entry.setdefault(modname, sc.obj())[key] = diff
        return out

    @classmethod
    def build(cls, arrays, runs, timevec=None, order=None):
        """
        The results of many runs, each with a run axis first: the one constructor that every way of running many sims ends in

        Args:
            arrays (dict): for each result, by (module, key), with the module None for the sim's own results: the result of one run it is like (`like`, for its name, kind, label, time points, and dimensions), its `values` (runs by time points), and if it is kept by group, `strata` and `denom` (runs by rows by groups) and `group_ti` (see `ss.solvers.group_rows()`)
            runs (dataframe): one row per run: its scenario, and the values of any parameters swept
            timevec (array): the sim's time points
            order (Results): the results of a sim, whose order the results keep (e.g. the modules' results before the sim's own, as in `sim.summarize()`); results it doesn't have go last
        """
        out = cls('MultiSim')
        if order is not None:
            rank = {(modname, key):i for i,(modname,key,_) in enumerate(order.walk())}
            arrays = dict(sorted(arrays.items(), key=lambda item: rank.get(item[0], len(rank)))) # Stable, so the others keep their order
        for (modname,key),arr in arrays.items():
            like = arr['like']
            values = np.asarray(arr['values'])
            res = like.asnew(values=values, _shape=values.shape, runs=arr['runs'] if 'runs' in arr else runs, low=None, high=None)
            res.run_pos = arr.get('run_pos') # For a result of only some of the runs, their positions among all the runs
            if getattr(like, 'strata', None) is not None or arr.get('strata') is not None:
                res.strata, res.denom = arr.get('strata'), arr.get('denom')
                res.group_ti = arr.get('group_ti', getattr(like, 'group_ti', None))
                if res.strata is None:
                    res.no_groups = 'it was not kept by group in every run'
            if modname is None:
                out[key] = res
            else:
                if modname not in out:
                    out[modname] = ss.Results(modname)
                out[modname][key] = res
        if timevec is not None:
            out['timevec'] = timevec
        if order is not None: # The modules and the sim's own results in the sim's order
            keys = [k for k in order.keys() if k in out] + [k for k in out.keys() if k not in order]
            ordered = cls('MultiSim')
            for k in keys:
                ordered[k] = out[k]
            out = ordered
        return out

    @classmethod
    def from_sims(cls, sims, runs):
        """ Every result that every sim has (with the same shape), across the sims: one run per sim """
        def get(sim, modname, key):
            try:
                return sim.results[key] if modname is None else sim.results[modname][key]
            except (KeyError, AttributeError, TypeError):
                return None

        def stack(found):
            """ The same result of each run that has it, as one array across those runs """
            pos = [i for i,r in enumerate(found) if isinstance(r, ss.Result) and r.values is not None]
            if not pos:
                return None
            found = [found[i] for i in pos]
            values = [np.asarray(r.values) for r in found]
            if len({v.shape for v in values}) != 1:
                return None
            arr = dict(like=found[0], values=np.stack(values))
            if len(pos) < len(sims): # A result of only some of the runs is across those runs
                arr['run_pos'] = pos
                arr['runs'] = runs.iloc[pos].reset_index(drop=True) if runs is not None else None
            strata = [getattr(r, 'strata', None) for r in found]
            if all(x is not None for x in strata) and len({x.shape for x in strata}) == 1:
                arr['strata'] = np.stack(strata)
                denom = [getattr(r, 'denom', None) for r in found]
                arr['denom'] = np.stack(denom) if all(x is not None for x in denom) else None # A ratio's denominator by group, so that selecting groups adds up each part before dividing
            return arr

        first = sims[0]
        times = [sim.t.tvec for sim in sims]
        if any(len(t) != len(times[0]) or np.any(t != times[0]) for t in times[1:]): # Results at different times cannot be stacked as runs of one thing
            steps = sc.strjoin(f'"{sim.label}": {len(t)} steps of {sim.t.dt}' for sim,t in zip(sims, times))
            raise ValueError(f'the sims have results at different times ({steps}), so they cannot be put together as runs of one MultiSim. Give them the same start, stop, and dt.')
        entries = {}
        for sim in sims: # Every result of any run, e.g. of an intervention only one scenario has
            entries |= {(None, key):res for key,res in sim.results.items() if isinstance(res, ss.Result) and (None, key) not in entries}
            entries |= {(modname, key):res for modname,modres in sim.results.items() if isinstance(modres, ss.Results) for key,res in modres.items() if isinstance(res, ss.Result) and (modname, key) not in entries}
        arrays = {(modname, key):arr for modname,key in entries if (arr := stack([get(sim, modname, key) for sim in sims])) is not None}
        out = cls.build(arrays, runs, timevec=first.results.get('timevec'), order=first.results)

        # The results of each sim's one disease are also at the top, e.g. msim.results.I, as sim.results.I is for a sim's (see ss.Results._implicit_get())
        names = [own_disease(sim) for sim in sims]
        if None not in names and len(set(names)) == 1: # The same disease in each, e.g. msim.results.sir.I
            out.setattribute('_implicit', names[0])
        elif None not in names: # Diseases named differently, e.g. sir and seir: their results put together
            keys = dict.fromkeys(key for sim,name in zip(sims, names) for key,res in sim.results[name].items() if isinstance(res, ss.Result))
            own = {(None, key):arr for key in keys if (arr := stack([get(sim, name, key) for sim,name in zip(sims, names)])) is not None}
            label = '/'.join(dict.fromkeys(names))
            own = cls.build(own, runs, timevec=first.results.get('timevec'))
            for res in own.values():
                if isinstance(res, ss.Result):
                    res.module = label # E.g. "sir/seir: I"
            out.setattribute('_implicit', own)
        return out

    def take(self, inds):
        """ The results of some of the runs, by position

        Args:
            inds (array): the positions of the runs
        """
        arrays = {}
        runs = None
        inds = list(sc.toarray(inds))
        for modname,key,res in self.walk():
            pos = getattr(res, 'run_pos', None)
            if pos is None:
                sub = res.take(inds)
                runs = sub.runs if runs is None else runs
                arrays[modname, key] = dict(like=sub, values=sub.values, strata=getattr(sub, 'strata', None), denom=getattr(sub, 'denom', None))
            else: # A result of only some of the runs: those of them taken
                keep = [(j, inds.index(p)) for j,p in enumerate(pos) if p in inds]
                if keep:
                    sub = res.take([j for j,_ in keep])
                    arrays[modname, key] = dict(like=sub, values=sub.values, strata=getattr(sub, 'strata', None), denom=getattr(sub, 'denom', None), runs=sub.runs, run_pos=[i for _,i in keep])
        out = MultiResults.build(arrays, runs, timevec=self.get('timevec'))
        implicit = self._implicit # The results of each sim's one disease (see from_sims())
        if implicit is not None:
            out.setattribute('_implicit', implicit if isinstance(implicit, str) else implicit.take(inds))
        return out

    @classmethod
    def concat(cls, results):
        """ The results of several sets of runs, one after the other (each result that all of them have) """
        arrays = {}
        first = results[0]
        others = [{(m, k):r for m,k,r in res.walk()} for res in results]
        runs = pd.concat([next(r for r in o.values() if getattr(r, 'run_pos', None) is None).runs for o in others], ignore_index=True)
        for modname,key,res in first.walk():
            found = [o.get((modname, key)) for o in others]
            if any(r is None or getattr(r, 'run_pos', None) is not None for r in found): # Only results of every run
                continue
            join = lambda attr: None if any(getattr(r, attr, None) is None for r in found) else np.concatenate([getattr(r, attr) for r in found])
            arrays[modname, key] = dict(like=res, values=join('values'), strata=join('strata'), denom=join('denom'))
        return cls.build(arrays, runs, timevec=first.get('timevec'))


def own_disease(sim):
    """ The name of a sim's one disease: the one it made or named, else its only disease (see `ss.Sim.init_results()`); None if it has several, or none """
    return sim.results._implicit


def run_row(sim, p, seed, draws=False):
    """ The row of a run in `msim.runs`: its scenario, its values, and its seed; a set of swept values is a scenario of its own, and a draw or a seed swept is a run of its sim's scenario """
    tag = '' if draws else ', '.join(f'{k}={v:n}' if sc.isnumber(v) else f'{k}={v}' for k,v in (p or {}).items() if k != 'rand_seed')
    return dict(scenario=f'{sim.label}: {tag}' if sim.label and tag else (sim.label or tag), **(p or {})) | dict(rand_seed=seed)


def run_seed(seed, p, j, i, n_runs, draws=False):
    """
    The seed of run `i` of set of values `j` (`p`) of a sim with this seed: the seed given in the set (plus `i`), if it has one; otherwise
    `seed + i`, the same for every set swept (so they are compared with the same chance events), or for draws, a seed of each draw's own,
    since the draws of a posterior are runs of a scenario, and so differ by chance as well as by their values
    """
    if p is not None and 'rand_seed' in p:
        return p['rand_seed'] + i
    return seed + (j*sc.ifelse(n_runs, 1) if draws else 0) + i


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

def by_scenario(sims, key):
    """ For `msim.sims['Vaccine']`: the sim of the scenario, or a list of them if it has several runs """
    scenarios = getattr(sims, 'scenarios', None) or []
    inds = [i for i,label in enumerate(scenarios) if label == key]
    if not inds:
        errormsg = f'msim.sims has no scenario "{key}"; the scenarios are {sc.strjoin(dict.fromkeys(scenarios))}, or give the number of a run'
        raise KeyError(errormsg)
    out = [sims[i] for i in inds]
    return out[0] if len(out) == 1 else out


class ByScenario:
    """ For the sims of a MultiSim: `keys()` and `items()` by scenario, as a dict of them would have (each scenario's sim, or a list of them if it has several runs) """
    def keys(self):
        """ The scenarios, in order """
        return list(dict.fromkeys(getattr(self, 'scenarios', None) or []))

    def items(self):
        """ Each scenario, and its sim (or a list of them, if it has several runs) """
        return [(key, by_scenario(self, key)) for key in self.keys()]


class Sims(ByScenario, list):
    """ The sims of the runs of a MultiSim: a list, which can also be indexed by scenario (`msim.sims['Vaccine']`, `msim.sims.items()`; see `MultiSim.sims`) """
    scenarios = None # The scenario of each run, set by the MultiSim

    def __getitem__(self, ind):
        if isinstance(ind, str):
            return by_scenario(self, ind)
        return super().__getitem__(ind)


class RestartedSims(ByScenario, cabc.Sequence):
    """
    The sim of each run of sims that were restarted for their runs (see `restart_runs()`), made when it is first asked for

    The runs are kept as results, so the sim of a run is made again only when it is
    asked for: a copy of its scenario's sim, restarted with the run's seed and values
    and run. `runners` are the sims (one per scenario) that were restarted.
    """
    def __init__(self, runners, sets, n_runs, seeds):
        self.runners = runners
        self.sets = sets # The values of each set, for each runner
        self.n_runs = n_runs
        self.seeds = seeds # The seed of each run
        self.made = {}
        return

    def __len__(self):
        return len(self.runners)*len(self.sets[0])*self.n_runs

    def __getitem__(self, ind):
        if isinstance(ind, str):
            return by_scenario(self, ind)
        if isinstance(ind, slice):
            return [self[i] for i in range(*ind.indices(len(self)))]
        ind = range(len(self))[ind] # Raises an IndexError if out of range, and handles negative indices
        if ind not in self.made:
            r = ind // (len(self.sets[0])*self.n_runs)
            runner = self.runners[r]
            p = self.sets[r][(ind // self.n_runs) % len(self.sets[r])]
            sim = sc.dcp(runner)
            if getattr(self, 'scenarios', None): # Labeled by its scenario, e.g. with the values swept
                sim.label = self.scenarios[ind] or sim.label
            self.made[ind] = sim.restart(pars=p, rand_seed=self.seeds[ind]).run()
        return self.made[ind]

    def __add__(self, other):
        return list(self) + list(other)

    def __radd__(self, other):
        return list(other) + list(self)

    def __repr__(self):
        return f'<{len(self)} sims, one per run, made when asked for>'


def base_values(sim, keys):
    """ The values of the parameters named in `keys` that the sim was built with, or None if one can't be found this way (e.g. a value for one level) """
    if not keys:
        return {}
    try:
        found = ss.gradients.find_pars(sim, keys)
    except ValueError:
        return None
    return {key:sc.dcp(p.raw) for key,p in found.items()}


def restart_runs(sims, n_runs, pars=None, draws=False):
    """
    Run each sim a number of times, each with its own seed, by building it once and restarting it

    Only for sims that are nothing but compartments (see `ss.Sim.restart()`), where this
    is many times faster than running copies. Returns None if any of the sims is not.
    Otherwise returns the sim of each run (`RestartedSims`, which makes each one when it is
    asked for), the results of every run (`ss.run.MultiResults`), and a dataframe of the runs.

    With `pars=` (see `ss.parallel()`), each set of values is run `n_runs` times (as runs of their sim's scenario if they are `draws`). An ODE
    with at least `jax_runs` runs (and JAX installed) takes them all in one call to JAX
    (see `ss.gradients`), which is faster than the kernels for that many.
    """
    has_graph = lambda sim: any(getattr(mod, 'graph', None) is not None for mod in ss.utils.given_modules(sim.pars)) # Only a sim with a declared module can be run whole
    if not all(has_graph(sim) for sim in sims):
        return None
    sets = par_sets(pars)
    runners = []
    for sim in sims:
        runner = sim.new(rand_seed=sim.pars.rand_seed, verbose=0).init() # The seed as it is now, which may have been set since the sim was made; the MultiSim says what it runs
        runner.label = sim.label
        if not runner.kernel_only() or not same_times([runners[0] if runners else runner, runner]): # Runs at different times can't be put together as results across runs
            return None
        keys = {key for p in sets if p is not None for key in p}
        for net in getattr(runner, 'mixed', []): # A network's parameter sets the contact rates when the sim is built, so each run with a value for one is built
            if any(key.split('.')[-1] in net.pars and ('.' not in key or key.split('.')[0] == net.name) for key in keys):
                return None
        runners.append(runner)
    runs = pd.DataFrame([run_row(sim, p, run_seed(sim.pars.rand_seed, p, j, i, n_runs, draws), draws) for sim in runners for j,p in enumerate(sets) for i in range(n_runs)])
    seeds = list(runs['rand_seed'])
    sets = [None if p is None else {k:v for k,v in p.items() if k != 'rand_seed'} or None for p in sets] # A seed given is the seed of its runs, not a parameter
    keys = sorted({key for p in sets if p is not None for key in p})
    bases = [base_values(sim, keys) for sim in runners] # Each set starts from the values the sim was built with, not from the set before
    if any(base is None for base in bases):
        return None
    sets = [[sc.mergedicts(base, p) or None for p in sets] for base in bases]
    values = sc.ddict(list)
    groups = sc.ddict(list)
    denoms = sc.ddict(list)
    group_ti = dict()
    for r,sim in enumerate(runners):
        first = r*len(sets[r])*n_runs # The position of the sim's first run
        if pars is not None and None not in sets[r] and len(sets[r])*n_runs >= jax_runs and ss.gradients.reason(sim) is None: # Every run in one call to JAX (an ODE run does not depend on the seed)
            outs = list(ss.gradients.run_sets(sim, [p for p in sets[r] for i in range(n_runs)]).items())
        else:
            outs = []
            for j,p in enumerate(sets[r]): # All the runs of each module with each set of values are one call to the kernels
                if p is not None:
                    sim.restart(pars=p)
                run_seeds = np.array(seeds[first + j*n_runs:first + (j+1)*n_runs])
                for mod in sim.declared:
                    cross = mod.engine.cross
                    if cross is None:
                        outs.append((mod.name, mod.engine.run_many(run_seeds)))
                    elif mod.name == cross.names[0]: # Diseases tracked jointly are run together
                        outs += cross.run_many(run_seeds)
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
        sim.restart(pars=sets[r][-1], rand_seed=seeds[first + len(sets[r])*n_runs - 1]).run() # The sim itself is left as its last run
    arrays = {}
    join = lambda arrs: arrs[0] if len(arrs) == 1 else np.concatenate(arrs) if len({np.shape(a)[1:] for a in arrs}) == 1 else None # Without a copy for one sim, since by group they can be large; None if the sims have different groups (e.g. a delivery that adds a group), as in MultiResults.from_sims()
    for (modname,key),vals in values.items():
        res = next(sim.results[modname][key] for sim in runners if modname in sim.results) # A delivery may be in some of the sims only
        vals = join(vals)
        if len(vals) == len(runs): # A result that every sim has
            strata, denom = [join(d[modname, key]) if (modname, key) in d else None for d in [groups, denoms]]
            strata = strata if strata is not None and len(strata) == len(runs) else None
            arrays[modname, key] = dict(like=res, values=vals, strata=strata, denom=denom if strata is not None else None, group_ti=group_ti.get(modname))
    return RestartedSims(runners, sets, n_runs, seeds), MultiResults.build(arrays, runs, order=runners[0].results), runs


def parallel(*args, n_runs=None, pars=None, priors=False, **kwargs):
    """
    Run many sims: different sims, each sim many times with its own seed, or with sets of values of its parameters

    This is the one way to run many sims; `sim.run(n_runs=...)` is a shortcut for running one sim
    many times. Every way of running ends in the same results: each result with a run axis first.

    Args:
        args (list): the sims to run, or a dict of them by label (e.g. `{'No intervention': s1, 'Vaccine': s2}`)
        n_runs (int): if provided, run each sim this many times, each with its own seed (the sim's seed, plus 1, 2, ...)
        pars (dict/list): sets of values of parameters to run each sim with: a dict with a list of values for each parameter (`dict(beta=[0.2, 0.3])`), or a list of dicts; each set is run `n_runs` times (default 1), with the same seeds for every set, and each set is a scenario of its own; the values are in `msim.runs` (and can be selected, e.g. `res.select(beta=0.2)`), with each run's `rand_seed`. Seeds can be given too, e.g. `pars=dict(rand_seed=[1, 5, 9])`, as runs of the scenario rather than scenarios
        priors (bool/dict): draw `n_runs` sets of values from the priors of the parameters (`ss.par(prior=...)`, or uniform over its bounds), one run each: a projection that carries the uncertainty in the values; or the draws themselves, as for `pars=` (e.g. from a posterior, as `fit.project()` passes them). Unlike a sweep with `pars=`, the draws are runs of their sim's scenario, each with a seed of its own (the sim's seed, plus 1, 2, ...), so quantiles over a scenario's runs include both the draws and chance
        kwargs (dict): sims passed as keywords are labeled with the keyword, e.g. `ss.parallel(baseline=s1, vaccine=s2)`;
            everything else is passed to `ss.MultiSim()`, e.g. `parallel=False`, `n_cpus`, `shrink`

    Returns:
        A run MultiSim. Each result is available across the runs, with the label of each sim as its scenario,
        e.g. `msim.results.sir.I.quantiles([0.025, 0.5, 0.975])`.

    Examples:
        ```python
        s1 = ss.Sim(n_agents=1000, label='Small', diseases='sis', networks='random')
        s2 = ss.Sim(n_agents=2000, label='Large', diseases='sis', networks='random')
        ss.parallel(s1, s2).plot()
        msim = ss.parallel([s1, s2], shrink=False)
        msim = ss.parallel(s1, n_runs=10) # Ten runs of one sim (also sim.run(n_runs=10))
        msim = ss.parallel({'Baseline': s1, 'With vaccine': s2}, n_runs=10) # Two scenarios; see msim.diff('Baseline')
        msim = ss.parallel(sim, pars=dict(beta=np.linspace(0.1, 0.5, 100))) # One run with each value of beta
        ```
    """
    sims = []
    for arg in args: # A dict is sims by their labels, e.g. {'No intervention': s1, 'Vaccine': s2}
        if isinstance(arg, dict):
            for label,sim in arg.items():
                sim.label = label
                sims.append(sim)
        else:
            sims += sc.tolist(arg)
    for key in list(kwargs.keys()): # Named sims
        if isinstance(kwargs[key], ss.Sim):
            sim = kwargs.pop(key)
            sim.label = key
            sims.append(sim)
    return MultiSim(sims, n_runs=n_runs, pars=pars, priors=priors, **kwargs).run()
