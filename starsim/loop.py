"""
Parent class for the integration loop.
"""
import time
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt


@dataclass(slots=True)
class LoopEntry:
    """
    One executable event in the integration loop

    Note: `slots=True` is used primarily to reduce the memory footprint across the
    10⁴–10⁵ instances created for a typical plan (no per-instance `__dict__`).
    """
    time: object
    ti: int
    func_order: int | None
    func: Callable
    label: str
    module: str | None
    func_name: str


#%% Loop class

class Loop:
    """
    Define the integration loop

    The Loop handles the order in which each function is called in the sim. The
    order is defined in `Loop.collect_funcs()`, which searches through the sim
    and collects all methods to call, in order, in the integration loop.

    Each type of module is called at a different time. Within each module type,
    they are called in the order listed. The default loop order is:

        1. sim:               start_step()     # Initialize the sim, including plotting progress
        2. all modules:       start_step()     # Initialize the modules, including the random number distribution
        3. sim.custom:        step()           # Run any custom modules
        4. sim.demographics:  step()           # Update the demographics, including adding new agents
        5. sim.diseases:      step_state()     # Update the disease states, e.g. exposed -> infected
        6. sim.connectors:    step()           # Run the connectors
        7. sim.networks:      step()           # Run the networks, including adding/removing edges
        8. sim.interventions: step()           # Run the interventions
        9. sim.diseases:      step()           # Run the diseases, including transmission
        10. people:           step_die()       # Figure out who died on this timestep
        11. people:           update_results() # Update basic state results
        12. all modules:      update_results() # Update any results
        13. sim.analyzers:    step()           # Run the analyzers
        14. all modules:      finish_step()    # Do any final tidying
        15. people:           finish_step()    # Clean up dead agents
        16. sim:              finish_step()    # Increment the timestep
    """
    def __init__(self, sim):
        self.sim = sim
        self.funcs = None
        self.func_list = []
        self.abs_tvecs = None
        self.abs_numvecs = None # Canonical numeric (float-year) time arrays, parallel to abs_tvecs, used for sorting/uniformity
        self.plan = None
        self.insertions = []
        self.index = 0 # The next function to execute
        self.profile = False # If True, record per-entry CPU timing during run() (see Loop.run)
        self.cpu_time = [] # Store the CPU time of execution of each function (only if profiling)
        self.df = None # User-friendly version of the plan
        self.cpu_df = None # User-friendly time analysis
        self.initialized = False
        return

    def init(self):
        """ Parse the sim into the integration plan """
        self.collect_funcs()
        self.collect_abs_tvecs()
        self.make_plan()
        self.initialized = True
        return

    def __len__(self):
        if self.initialized:
            return len(self.plan)
        else:
            return 0 # Or None?

    def __iadd__(self, func):
        """ Allow functions to be added to the function list """
        parent = func.__self__
        func_name = func.__name__

        # Get the name if it's defined, the class otherwise; these must match abs_tvecs
        if isinstance(parent, ss.Module):
            module = parent.name
        elif isinstance(parent, ss.Sim):
            module = 'sim'
        else:
            module = parent.__class__.__name__.lower()

        # Create the row and append it to the function list
        func_path = f'{parent.__class__.__module__}.{func_name}'
        row = dict(
            func_order = len(self.funcs),
            module = module,
            func_name = func_name,
            func_path = func_path,
            func = func,
        )
        self.funcs.append(row)
        return self

    def __repr__(self):
        if self.initialized:
            arrs = list({len(arr) for arr in self.abs_tvecs.values()})
            if len(arrs) == 1: arrs = arrs[0] # If all are the same, just use that
            string = f'Loop(n={len(self)}, funcs={len(self.funcs)}, npts={arrs}, index={self.index})'
        else:
            string = 'Loop(initialized=False)'
        return string

    def disp(self):
        return sc.pr(self)

    def collect_funcs(self):
        """ Collect all the callable functions (methods) that comprise the step """

        # Run the simulation step first (updates the distributions)
        self.funcs = [] # Reset, just in case
        sim = self.sim

        # Collect the start_steps
        self += sim.start_step # Note special __iadd__() method above, which appends these to the funcs list
        for mod in sim.modules:
            self += mod.start_step

        # Update any nonspecific modules
        for mod in sim.custom():
            self += mod.step

        # Update demographic modules (create new agents from births/immigration, schedule non-disease deaths and emigration)
        for dem in sim.demographics():
            self += dem.step

        # Carry out autonomous state changes in the disease modules. This allows autonomous state changes/initializations
        # to be applied to newly created agents
        for disease in sim.diseases():
            if isinstance(disease, ss.Disease):
                self += disease.step_state

        # Update connectors
        for connector in sim.connectors():
            self += connector.step

        # Update networks - this takes place here in case autonomous state changes at this timestep
        # affect eligibility for contacts
        for network in sim.networks():
            self += network.step

        # Apply interventions - new changes to contacts will be visible and so the final networks can be customized by
        # interventions, by running them at this point
        for intv in sim.interventions():
            self += intv.step

        # Carry out autonomous state changes in the disease modules, including transmission (but excluding deaths)
        for disease in sim.diseases():
            self += disease.step

        # Update people who died -- calls disease.step_die() internally
        self += sim.people.step_die

        # Update results. A module's update_results is skipped only when it is the inherited
        # base method (not overridden anywhere in the MRO) and the module has no auto-generated
        # boolean-state results to count -- i.e. the base loop is provably a no-op. Overrides
        # (including those that call super() or gain behavior via a mixin) are always kept.
        if sim.pars.get('people_results', True): # People results can be opted out for lightweight sims
            self += sim.people.update_results
        for mod in sim.modules:
            if not self._null_update_results(mod):
                self += mod.update_results

        # Apply analyzers
        for ana in sim.analyzers():
            self += ana.step

        # Clean up dead agents, increment the time index, and perform other housekeeping tasks
        for mod in sim.modules:
            self += mod.finish_step
        self += sim.people.finish_step
        self += sim.finish_step

        return self.funcs

    @staticmethod
    def _null_update_results(mod):
        """
        Return True if a module's `update_results` is a guaranteed no-op and can be skipped

        This holds only when the module uses the inherited base `Module.update_results`
        (not overridden anywhere below `ss.Module` in the class hierarchy, so no `super()`
        call or mixin behavior is missed) and has no auto-generated boolean-state or
        derived-state results for the base method to count. Such a call would iterate an empty list and change
        nothing, so omitting it from the plan is result-identical.
        """
        not_overridden = type(mod).update_results is ss.Module.update_results
        return not_overridden and not len(mod.auto_state_list) and not len(mod.derived_state_names)

    def collect_abs_tvecs(self):
        """
        Collect time arrays for each module

        Two parallel arrays are stored per module: `abs_tvecs` holds the ground-truth
        time objects (`ss.date`/`ss.dur`, used verbatim as each plan entry's time), and
        `abs_numvecs` holds the canonical numeric representation (float years). The
        numeric arrays are what `make_plan()` uses to decide uniformity and to sort,
        so that date/dur objects are never compared during plan construction.
        """
        self.abs_tvecs = sc.objdict()
        self.abs_numvecs = sc.objdict()

        # Handle the sim and people first
        sim = self.sim
        for key in ['sim', sim.people.__class__.__name__.lower()]: # To handle subclassing of People -- TODO, make more elegant!
            self.abs_tvecs[key] = sim.t.tvec
            self.abs_numvecs[key] = sim.t.yearvec

        # Handle all other modules
        for mod in sim.modules:
            self.abs_tvecs[mod.name] = mod.t.tvec
            self.abs_numvecs[mod.name] = mod.t.yearvec

        return self.abs_tvecs

    def make_plan(self):
        """
        Combine the module ordering and the time vectors into the integration plan

        The plan is the list of `LoopEntry` events in execution order: tick-major, and
        within each tick ordered by function order. When every module shares the sim's
        canonical timeline (the common case) this order can be generated directly, with
        no sort. Otherwise a numeric sort is used. In neither case are date/dur objects
        compared (which is slow); the canonical numeric time vectors (`abs_numvecs`) are
        used instead.
        """
        if self._timelines_uniform():
            self.plan = self._make_plan_uniform()
        else:
            self.plan = self._make_plan_sorted()

        # Replay any user insertions tied to the current loop definition
        for insertion in self.insertions:
            self._insert_into_plan(**insertion)

        return

    def _timelines_uniform(self):
        """ Check whether every module shares the sim's canonical timeline (numeric values, length, and unit) """
        sim_num = self.abs_numvecs['sim']
        sim_unit = self.abs_tvecs['sim'].unit
        npts = len(sim_num)
        for key, num in self.abs_numvecs.items():
            # Exact numeric equality plus matching unit: only then is the schedule provably identical
            if len(num) != npts or self.abs_tvecs[key].unit is not sim_unit or not np.array_equal(num, sim_num):
                return False
        return True

    def _make_plan_uniform(self):
        """
        Fast path: all modules share the sim's timeline, so emit entries directly in
        execution order (each tick, every function in func order) without sorting.

        Because all entries at a given tick share one time and the times are strictly
        increasing, this is identical to sorting by (time, func_order); and the tick
        index equals the sim time index, so no separate ti pass is needed.
        """
        tvec = self.abs_tvecs['sim']
        # Precompute the per-function fields once, rather than rebuilding them per entry
        specs = [(fr['func_order'], fr['func'], f"{fr['module']}.{fr['func_name']}", fr['module'], fr['func_name']) for fr in self.funcs]
        plan = []
        append = plan.append
        for ti in range(len(tvec)):
            t = tvec[ti]
            for func_order, func, label, module, func_name in specs:
                append(LoopEntry(time=t, ti=ti, func_order=func_order, func=func, label=label, module=module, func_name=func_name))
        return plan

    def _make_plan_sorted(self):
        """
        General path: modules have heterogeneous timelines, so build the full list of
        entries and sort into execution order. The sort uses `np.lexsort` on the
        canonical numeric times (and function order), never on date/dur objects.
        """
        raw = []
        times_num = []
        func_orders = []
        r_append = raw.append
        t_append = times_num.append
        f_append = func_orders.append
        for func_row in self.funcs:
            module = func_row['module']
            func_name = func_row['func_name']
            func_order = func_row['func_order']
            func = func_row['func']
            label = f'{module}.{func_name}'
            tvec = self.abs_tvecs[module]
            numvec = self.abs_numvecs[module]
            for j in range(len(tvec)):
                r_append(LoopEntry(time=tvec[j], ti=0, func_order=func_order, func=func, label=label, module=module, func_name=func_name))
                t_append(numvec[j])
                f_append(func_order)

        # Sort by (time, func_order), using numeric keys so date/dur objects are never compared
        times_num = np.asarray(times_num, dtype=float)
        func_orders = np.asarray(func_orders)
        order = np.lexsort((func_orders, times_num)) # Last key (times_num) is primary
        plan = [raw[i] for i in order]

        # Calculate the sim time index (ti) by counting sim.start_step boundaries
        start_step = 'sim.start_step'
        ti = -1
        for entry in plan:
            if entry.label == start_step:
                ti += 1
            entry.ti = ti

        # Warn if any consecutive time values are close but not identical (likely a floating-point issue)
        self._warn_near_identical(times_num[order])
        return plan

    @staticmethod
    def _warn_near_identical(sorted_times):
        """ Warn if any consecutive plan times are close but not identical (a floating-point issue) """
        eps = 1e-9
        diffs = np.diff(sorted_times)
        small_diffs = diffs[(diffs > 0) & (diffs < eps)]
        if len(small_diffs):
            warnmsg = f'{len(small_diffs)} integration loop entries have near-identical times, indicating a floating-point issue:\n{small_diffs}\nCheck your time units across the sim and modules!'
            ss.warn(warnmsg)
        return

    def store_time(self):
        """ Store the current time in as high resolution as possible (only called when profiling) """
        self.cpu_time.append(time.perf_counter())
        return

    def run_one_step(self):
        """
        Take a single step, i.e. call a single function; only used for debugging purposes.

        Compare sim.run_one_step(), which runs a full timestep (which involves multiple function calls).
        """
        self._check_initialized()
        entry = self.plan[self.index] # Get the next entry
        entry.func() # Call it
        self.index += 1 # Increment the time
        return

    def _check_initialized(self):
        """ Check that the Loop has been initialized """
        if not self.initialized:
            errormsg = 'Please initialize the loop (typically sim.init()) before calling this method.'
            raise RuntimeError(errormsg)
        return

    def run(self, until=None, verbose=None, profile=None):
        """
        Actually run the integration loop; usually called by sim.run()

        By default, per-entry CPU timing is *not* recorded and the plan DataFrame is *not*
        built at the end of the run -- both are pure overhead (~15–25 ms for a bare 10-year
        sim) that most runs do not use. Pass `profile=True` to record per-entry timing (which
        `to_df()`/`plot_cpu()` then expose as the `cpu_time` column); otherwise `cpu_time` is
        `NaN`. The plan metadata (via `to_df()`) is always available on demand regardless.

        Args:
            until (str/date): if supplied, stop after this date (used by sim.run_one_step)
            verbose (bool): if True, print each function call as it runs
            profile (bool): if True, record per-entry CPU timing (default: keep the current setting)
        """
        self._check_initialized()
        if profile is not None:
            self.profile = profile

        # Convert e.g. '2020-01-01' to an actual date
        if isinstance(until, str):
            until = ss.date(until)

        # Loop over every function in the integration loop, e.g. disease.step()
        if self.profile:
            self.store_time()
        while self.index < len(self.plan):
            entry = self.plan[self.index]
            if verbose:
                print(f'Running t={entry.time:n}, step={self.index}, {entry.label}()')

            entry.func() # Execute the function -- this is where all of Starsim happens!!

            # Tidy up
            self.index += 1 # Increment the count
            if self.profile:
                self.store_time()
            if until is not None and self.sim.now > until: # Terminate if asked to
                break

        # Note: the plan DataFrame is now built lazily by to_df() when requested, not here
        return

    def plan_metadata(self):
        """ Return the dataframe view of the plan used for matching and display """
        cols = ['time', 'ti', 'func_order', 'label', 'module', 'func_name']
        rows = [{col:getattr(entry, col) for col in cols} for entry in self.plan]
        return sc.dataframe(rows, columns=cols)

    def _insert_into_plan(self, func, label=None, match_fn=None, before=False):
        """ Insert into `self.plan` without recording the insertion for replay """
        if label:
            match_fn = lambda plan: plan.label == label

        # Compute the matches against a dataframe metadata view
        metadata = self.plan_metadata()
        matches = match_fn(metadata)
        matches = np.asarray(matches)
        if matches.dtype == bool:
            matches = sc.findinds(matches)

        # Perform the insertion in reverse order
        name = func.__name__
        sim_func = lambda: func(self.sim) # Construct a partial function
        for m in sorted(matches, reverse=True):
            current = self.plan[m]
            row = LoopEntry(
                time = current.time,
                ti = current.ti,
                func_order = None,
                func = sim_func,
                label = name,
                module = None,
                func_name = name,
            )
            ind = m if before else m+1
            self.plan.insert(ind, row)

        self.df = None
        self.cpu_df = None
        return

    def insert(self, func, label=None, match_fn=None, before=False):
        """
        Insert a function into the loop plan at the specified location.

        The loop plan metadata view is a dataframe with columns including time (e.g. `date('2025-05-05')`),
        label (e.g. `'randomnet.step'`), module ('`randomnet'`), and function name (`'step'`).
        By default, this method will match the conditions in the plan based on
        the criteria specified.

        This functionality is similar to an analyzer or an intervention, but gives
        additional flexibility since can be inserted at (almost) any point in a sim.

        Note: the loop must be initialized (`sim.init()`) before you can call this.

        Args:
            func (func): the function to insert; must take a single argument, `sim`
            label (str): the label (module.name) of the function to match; see `sim.loop.to_df().label.unique() for choices`
            match_fn (func): if supplied, use this function to perform the matching on the plan dataframe, returning a boolean array or list of indices of matching rows (see example below)
            before (bool): if true, insert the function before rather than after the match

        Examples:
            ```python
            # Simple label matching with analyzer-like functionality
            def check_pop_size(sim):
                print(f'Population size is {len(sim.people)}')

            sim = ss.Sim(diseases='sir', networks='random', demographics=True)
            sim.init()
            sim.loop.insert(check_pop_size, label='people.finish_step')
            sim.run()

            # Function-based matching with intervention-like functionality
            def match_fn(plan):
                past_2010 = plan.time > ss.date(2010)
                is_step = (plan.label == 'sir.step') | (plan.label == 'randomnet.step')
                return past_2010 * is_step

            def update_betas(sim):
                if not sim.metadata.get('updated'):
                    print(f'Updating beta values on {sim.now}')
                    sim.diseases.sis.beta = 0.1
                    sim.networks.randomnet.edges.beta[:] = 0.5
                    sim.metadata.updated = True
                return

            sim = ss.Sim(diseases='sis', networks='random')
            sim.init()
            sim.loop.insert(update_betas, match_fn=match_fn, before=True)
            sim.run()
            ```
        """
        self._check_initialized()

        if label and match_fn:
            errormsg = "You can supply label or match, but not both; 'label' is equivalent to 'plan.label == label', please include this in your match function"
            raise ValueError(errormsg)

        insertion = dict(func=func, label=label, match_fn=match_fn, before=before)
        self.insertions.append(insertion)
        self._insert_into_plan(**insertion)
        return

    def to_df(self):
        """
        Return a user-friendly version of the plan, omitting object columns

        The `cpu_time` column is populated only if the run recorded per-entry timing
        (i.e. `sim.run(profile=True)`); otherwise it is `NaN`. This is built lazily and
        cached in `self.df`/`self.cpu_df`.
        """
        # Compute the main dataframe
        if self.plan is not None:
            df = self.plan_metadata()
        else:
            errormsg = f'Simulation "{self.sim}" needs to be initialized before exporting the Loop dataframe'
            raise RuntimeError(errormsg)
        times = np.diff(self.cpu_time)
        if len(times) == len(df):
            df['cpu_time'] = times
        else:
            df['cpu_time'] = np.nan
        self.df = df

        # Compute the CPU dataframe
        by_func = df.groupby('label')
        method = dict(func_order='first', module='first', func_name='first', cpu_time='sum')
        cdf = sc.dataframe(by_func.agg(method))
        cdf['percent'] = cdf.cpu_time / cdf.cpu_time.sum()*100
        cdf.insert(cdf.cols.index('cpu_time'), 'calls', by_func.size())
        cdf.sort_values('cpu_time', inplace=True, ascending=False)
        self.cpu_df = cdf
        return df

    def shrink(self):
        """ Shrink the size of the loop for saving to disk """
        to_shrink = ['sim', 'funcs', 'plan']
        ss.shrink(self, to_shrink)
        return

    def plot(self, simplify=False, max_len=100, fig_kw=None, plot_kw=None, scatter_kw=None):
        """
        Plot a diagram of all the events

        Args:
            simplify (bool): if True, skip update_results and finish_step events, which are automatically applied
            max_len (int): maximum number of entries to plot
            fig_kw (dict): passed to `plt.figure()`
            plot_kw (dict): passed to `plt.plot()`
            scatter_kw (dict): passed to `plt.scatter()`
        """

        # Assemble data
        df = self.df
        if df is None:
            df = self.to_df()
        if simplify:
            filter_out = ['update_results', 'finish_step']
            df = df[~df.func_name.isin(filter_out)]
        if max_len:
            df = df[:max_len]
        yticks = df.func_order.unique()
        ylabels = df.label.unique()
        x = df.time
        y = df.func_order

        # Convert module names to integers for plotting colors
        mod_int, _ = pd.factorize(df.module)
        colors = sc.gridcolors(np.unique(mod_int), asarray=True)

        # Do the plotting
        plot_kw = sc.mergedicts(dict(lw=2, alpha=0.2, c='k'), plot_kw)
        scatter_kw = sc.mergedicts(dict(s=200, alpha=0.6), scatter_kw)
        fig = plt.figure(**sc.mergedicts(fig_kw))
        plt.plot(x, y, **plot_kw)
        plt.scatter(x, y, c=colors[mod_int], **scatter_kw)
        plt.yticks(yticks, ylabels)
        plt.title(f'Integration plan ({len(df)} events)')
        plt.xlabel('Time since simulation start')
        plt.grid(True)
        sc.figlayout()
        sc.boxoff()
        return ss.return_fig(fig)

    def plot_cpu(self, bytime=True, max_entries=10, fig_kw=None, bar_kw=None):
        """
        Plot the CPU time spent on each event; visualization of Loop.cpu_df.

        Args:
            bytime (bool): if True, order events by total time rather than actual order
            fig_kw (dict): passed to `plt.figure()`
            bar_kw (dict): passed to `plt.bar()`
        """
        # Assemble data
        if self.cpu_df is None:
            self.to_df()
        if not len(self.cpu_time): # No per-entry timing was recorded
            ss.warn('No CPU timing was recorded: run with sim.run(profile=True) (or loop.run(profile=True)) to populate cpu_time.')
        df = self.cpu_df
        ylabels = df.index.values.copy() # Copy to avoid mutating the cached cpu_df when labels are assembled below
        if bytime:
            y = np.arange(len(ylabels))
        else:
            y = df.func_order.values
        y = y[::-1] # Reverse order so plots from top to bottom

        x = df.cpu_time.values
        pcts = df.percent.values

        if x.max() < 1:
            x = x*1e3
            unit = 'ms'
        else:
            unit = 's'

        # Assemble labels
        for i in range(len(df)):
            timestr = sc.sigfig(x[i], 3) + f' {unit}'
            pctstr = sc.sigfig(pcts[i], 3) + '%'
            ylabels[i] += f'()\n{timestr}, {pctstr}'

        # Trim if needed
        if max_entries:
            x = x[:max_entries]
            y = y[:max_entries]
            ylabels = ylabels[:max_entries]

        # Do the plotting
        bar_kw = sc.mergedicts(bar_kw)
        fig = plt.figure(**sc.mergedicts(fig_kw))
        plt.barh(y, width=x, **bar_kw)
        plt.yticks(y, ylabels)
        plt.xlabel(f'CPU time ({unit})')
        plt.ylabel('Function call')
        plt.grid(True)
        sc.figlayout()
        sc.boxoff()
        return ss.return_fig(fig)

    def plot_step_order(self, which='default', max_len=500, plot_kw=None, scatter_kw=None, fig_kw=None, legend_kw=None):
        """
        Plot the order of the module steps across timesteps -- useful for debugging
        when using different time units.

        Note: generates a lot of data, best to debug with a small number of timesteps first!

        Args:
            which (dict): columns and values to filter to (default: {'func_name':'step'}; if None, do not filter)
            max_len (int): maximum number of entries to plot
            plot_kw (dict): passed to `plt.plot()`
            scatter_kw (dict): passed to `plt.scatter()`
            fig_kw (dict): passed to `plt.figure()`
            legend_kw (dict): passed to `plt.legend()`

        Examples:
            ```python
            sis = ss.SIS(dt=0.1)
            net = ss.RandomNet(dt=0.5)
            births = ss.Births(dt=1)
            sim = ss.Sim(dt=0.1, dur=5, diseases=sis, networks=net, demographics=births)
            sim.init()
            sim.loop.plot_step_order()
            ```
        """
        self._check_initialized()
        df = self.df
        if df is None:
            df = self.to_df()
        if which == 'default':
            which = dict(func_name='step')
        if which:
            for col,value in which.items():
                df = df[df[col] == value]
        if max_len and len(df) > max_len:
            print(f'Note: truncating from {len(df)} to {max_len} entries')
            df = df[:max_len]

        # Construct data
        unique = df.label.unique()
        n_unique = len(unique)
        colors = sc.gridcolors(n_unique)
        colormap = {k:v for k,v in zip(unique, colors)}
        d = sc.dictobj()
        for key in ['x', 'y', 'z', 'label']:
            d[key] = sc.autolist()
        for ti in df.ti.unique():
            this = df[df.ti==ti]
            d.x += list(range(len(this))) # Convert [0,0,0,...] to [0,1,2,...]
            d.y += list(this.func_order)
            d.z += list(this.ti)
            d.label += list(this.label)

        dd = sc.dataframe(d)

        fig = plt.figure(**sc.mergedicts(fig_kw))

        plot_kw = sc.mergedicts(dict(alpha=0.5, lw=2), plot_kw)
        scatter_kw = sc.mergedicts(dict(s=100, alpha=0.5), scatter_kw)
        ax = plt.axes(projection='3d')
        sc.plot3d(dd.x, dd.y, dd.z, ax=ax, **plot_kw)
        for label in unique:
            this = dd[dd.label==label]
            ax.scatter(this.x, this.y, this.z, color=colormap[label], label=label, **scatter_kw)
        ax.set_xlabel('Position within timestep')
        ax.set_ylabel('Original function order')
        ax.set_zlabel('Timestep')
        legend_kw = sc.mergedicts(dict(loc='upper left', bbox_to_anchor=(1.05, 1)), legend_kw)
        ax.legend(**legend_kw)
        return ss.return_fig(fig)

    def __deepcopy__(self, memo):
        """ Deep-copy the loop """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in vars(self).items():
            setattr(new, k, sc.dcp(v, memo=memo, die=False))
        return new
