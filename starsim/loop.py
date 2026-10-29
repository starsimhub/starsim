"""
Parent class for the integration loop.
"""
import time
import functools as ft
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

        1. sim:               advance()        # Move the time to this step's time point
        2. sim:               start_step()     # Initialize the sim, including plotting progress
        3. all modules:       start_step()     # Initialize the modules, including the random number distribution
        4. sim.custom:        step()           # Run any custom modules
        5. sim.demographics:  step()           # Update the demographics, including adding new agents
        6. sim.diseases:      step_state()     # Update the disease states, e.g. exposed -> infected
        7. sim.connectors:    step()           # Run the connectors
        8. sim.networks:      step()           # Run the networks, including adding/removing edges
        9. sim.interventions: step()           # Run the interventions
        10. sim.diseases:     step()           # Run the diseases, including transmission
        11. people:           step_die()       # Figure out who died on this timestep
        12. people:           update_results() # Update basic state results
        13. all modules:      update_results() # Update any results
        14. sim.analyzers:    step()           # Run the analyzers
        15. all modules:      finish_step()    # Do any final tidying
        16. people:           finish_step()    # Clean up dead agents
        17. sim:              finish_step()    # Finish the step

    One time index holds for every function of a step, in every module: the step from time
    point k-1 to k is step k, and throughout it `sim.ti == k`, where its results go (so the
    first step has `ti == 1`): `self.results.x[self.ti]` is this step's value, in `step()` as
    in `update_results()`, and a time recorded as `self.ti` in one is `self.ti` in the other.
    `sim.now` is when the step starts, time point k-1, which it reads time-varying inputs and
    dates at. At the first time point (`ti == 0`, `now` the start) there is no step: only the
    results of the start are recorded (people's and every module's but the interventions',
    which have done nothing yet), and the analyzers run (`at_start`).

    Args:
        sim (Sim): the sim to run (usually made by the sim itself)
    """
    def __init__(self, sim):
        self.sim = sim
        self.funcs = None
        self.at_start = None # The functions that also run at the first time point, by function order: those that record the start
        self.shared_timelines = [] # The timelines of the modules on the sim's timeline, which the sim's advance() moves
        self.abs_tvecs = None
        self.abs_numvecs = None # Canonical numeric (float-year) time arrays, parallel to abs_tvecs, used for sorting/uniformity
        self._plan = None # Made when it is first needed: see the plan property
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
        self._plan = None
        self.index = 0
        self.initialized = True
        return self

    @property
    def plan(self):
        """ The list of `LoopEntry` events in execution order; made the first time it is needed, since a sim run by the kernels in one call never needs it """
        if self._plan is None and self.initialized:
            self.make_plan()
        return self._plan

    @plan.setter
    def plan(self, value):
        self._plan = value

    def __len__(self):
        if self.initialized:
            return len(self.plan)
        else:
            return 0

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

        # Move the time to the step about to be taken, so that it is the same in every function of the step: the sim's time,
        # with that of the modules on its timeline; a module with a timeline of its own moves its own
        self.shared_timelines = [mod.t for mod in sim.modules if self._on_sim_time(mod)] # Their timelines, not the modules, so a search of the sim does not find the modules again here
        self += sim.advance # Note special __iadd__() method above, which appends these to the funcs list
        for mod in sim.modules:
            if not self._on_sim_time(mod):
                self += mod.advance

        # Collect the start_steps
        self += sim.start_step
        for mod in sim.modules:
            if self._keep(mod.start_step):
                self += mod.start_step

        # Rescale the population if needed
        if sim.pars.rescale:
            self += sim.rescale

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
        if self._keep(sim.people.step_die):
            self += sim.people.step_die

        first = len(self.funcs) # From here to the analyzers, the functions also run at the first time point, to record the start (except the interventions')

        # Update results. A module's update_results is skipped only when it is the inherited
        # base method (not overridden anywhere in the MRO) and the module has no auto-generated
        # boolean-state results to count -- i.e. the base loop is provably a no-op. Overrides
        # (including those that call super() or gain behavior via a mixin) are always kept.
        if sim.pars.get('people_results', True): # People results can be opted out for lightweight sims
            self += sim.people.update_results
        for mod in sim.modules:
            if self._keep(mod.update_results):
                self += mod.update_results

        # Apply analyzers
        for ana in sim.analyzers():
            self += ana.step
        intvs = {intv.name for intv in sim.interventions()} # Which have done nothing before the first step, so record nothing at the start
        self.at_start = {i for i in range(first, len(self.funcs)) if self.funcs[i]['module'] not in intvs}

        # Clean up dead agents, and perform other housekeeping tasks
        for func in [mod.finish_step for mod in sim.modules] + [sim.people.finish_step]:
            if self._keep(func):
                self += func
        self += sim.finish_step

        return self.funcs

    def _on_sim_time(self, mod):
        """ Whether a module shares the sim's timeline (the usual case), so its time moves with the sim's """
        t, sim_t = mod.t, self.sim.t
        return len(t.yearvec) == len(sim_t.yearvec) and t.tvec.unit is sim_t.tvec.unit and np.array_equal(t.yearvec, sim_t.yearvec)

    def _keep(self, func):
        """ Whether to put a call in the plan (see `_noop()`); one left out counts as called, for `ss.Sim.check_method_calls()` """
        if not self._noop(func):
            return True
        required = getattr(func.__self__, '_call_required', {})
        for key in required:
            if key.endswith(f'.{func.__name__}'):
                required[key] += 1
        return False

    def _noop(self, func):
        """
        Whether a call of the step is sure to do nothing in this sim, so is left out of the plan

        This is so only for Starsim's own methods, when nothing has been changed that they
        would act on: a module's `start_step()` with no random numbers to move on, its
        `finish_step()` and `update_results()` (see `_null_update_results()`) when inherited
        unchanged, and the people's deaths and removals in a run without agents (in which nobody
        dies as an agent). The sim's own calls are always kept, as places to insert others. A method that a
        module overrides is always kept, so a module written as code costs the loop only the
        methods it defines.
        """
        obj, name = func.__self__, func.__name__
        sim = self.sim
        if isinstance(obj, ss.Module):
            inherited = getattr(type(obj), name) is getattr(ss.Module, name)
            if name == 'start_step':
                return inherited and (obj.dists is None or not ss.options.crn) # Moving the streams on is all it does
            elif name == 'finish_step':
                return inherited
            elif name == 'update_results':
                return self._null_update_results(obj)
        elif obj is sim.people and name in ['step_die', 'finish_step']:
            people = type(obj)
            same = people.step_die is ss.People.step_die and people.finish_step is ss.People.finish_step and people.update_post is ss.People.update_post and people.remove_dead is ss.People.remove_dead
            return same and sim.solver != 'abm' # One row per group, which never die or leave
        return False

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
        sim_unit = getattr(self.abs_tvecs['sim'], 'unit', None) # Plain numbers have no unit
        npts = len(sim_num)
        for key, num in self.abs_numvecs.items():
            # Exact numeric equality plus matching unit: only then is the schedule provably identical
            if len(num) != npts or getattr(self.abs_tvecs[key], 'unit', None) is not sim_unit or not np.array_equal(num, sim_num):
                return False
        return True

    def _make_plan_uniform(self):
        """
        Fast path: all modules share the sim's timeline, so emit entries directly in
        execution order (each tick, every function in func order) without sorting.

        Because all entries at a given tick share one time and the times are strictly
        increasing, this is identical to sorting by (time, func_order); and the tick
        index equals the sim time index, so no separate ti pass is needed. At the first
        tick there is no step, so only the functions that record the start are emitted.
        """
        tvec = self.abs_tvecs['sim']
        # Precompute the per-function fields once, rather than rebuilding them per entry
        specs = [(fr['func_order'], fr['func'], f"{fr['module']}.{fr['func_name']}", fr['module'], fr['func_name']) for fr in self.funcs]
        plan = []
        append = plan.append
        for ti in range(len(tvec)):
            t = tvec[ti]
            for func_order, func, label, module, func_name in specs:
                if ti or func_order in self.at_start:
                    append(LoopEntry(t, ti, func_order, func, label, module, func_name)) # By position, which is quicker for the many entries
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
            for j in range(0 if func_order in self.at_start else 1, len(tvec)): # At its first time point, a module has no step, and only records the start
                r_append(LoopEntry(time=tvec[j], ti=0, func_order=func_order, func=func, label=label, module=module, func_name=func_name))
                t_append(numvec[j])
                f_append(func_order)

        # Sort by (time, func_order), using numeric keys so date/dur objects are never compared
        times_num = np.asarray(times_num, dtype=float)
        func_orders = np.asarray(func_orders)
        order = np.lexsort((func_orders, times_num)) # Last key (times_num) is primary
        plan = [raw[i] for i in order]

        # Calculate the sim time index (ti): that of the last sim time point at or before each entry
        tis = np.searchsorted(self.abs_numvecs['sim'], times_num[order], side='right') - 1
        for entry,ti in zip(plan, tis):
            entry.ti = int(max(ti, 0))

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

    def store_time(self, start):
        """ Store the time elapsed since start in as high resolution as possible (only called when profiling) """
        self.cpu_time.append(time.perf_counter() - start)
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
            until (str/date/Expr/func): if a time, stop after the step that starts at it (used by sim.run_one_step); if a condition (see `sim.run()`), stop once it holds at the end of a step
            verbose (bool): if True, print each function call as it runs
            profile (bool): if True, record per-entry CPU timing (default: keep the current setting)
        """
        self._check_initialized()
        if profile is not None:
            self.profile = profile

        # Convert e.g. '2020-01-01' to an actual date
        cond = None
        if isinstance(until, str):
            until = ss.date(until)
        elif isinstance(until, ss.expressions.Expr) or callable(until): # A condition, e.g. sir.I.n == 0
            cond, until = until, None
        self.stopped = None

        # Loop over every function in the integration loop, e.g. disease.step()
        plan = self.plan
        while self.index < len(plan):
            entry = plan[self.index]
            if until is not None and entry.label == 'sim.advance' and self.sim.t.tvec[entry.ti - 1] > until: # Stop before a step that starts after the time asked for
                break
            if cond is not None and entry.label == 'sim.advance' and self.sim.holds(cond): # It holds at the end of the step just taken
                self.stopped = self.sim.t.ti
                break
            if verbose:
                now = entry.time if isinstance(entry.time, ss.date) else f'{entry.time:n}' # Dates don't support number formatting
                print(f'Running t={now}, step={self.index}, {entry.label}()')

            if self.profile:
                start = time.perf_counter()
            entry.func() # Execute the function -- this is where all of Starsim happens!!

            # Tidy up
            self.index += 1 # Increment the count
            if self.profile:
                self.store_time(start)

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

        # A module is moved: its step() runs at the matching place, on each timestep where there is one
        if isinstance(func, ss.Module):
            step_label = f'{func.name}.step'
            own = {repr(entry.time):entry for entry in self.plan if entry.label == step_label}
            if not own:
                errormsg = f'Cannot move "{func.name}" in the loop, since its step() is not in the plan; the labels are {sc.strjoin(metadata.label.unique())}'
                raise ValueError(errormsg)
            moved = {repr(self.plan[m].time) for m in matches} & set(own.keys())
            matches = set(matches.tolist())
            plan = []
            for i,entry in enumerate(self.plan):
                key = repr(entry.time)
                if entry.label == step_label and key in moved:
                    continue
                elif i in matches and key in moved:
                    plan += [own[key], entry] if before else [entry, own[key]]
                else:
                    plan.append(entry)
            self.plan = plan
            self.df = None
            self.cpu_df = None
            return

        # Perform the insertion in reverse order
        name = func.__name__
        sim_func = ft.partial(func, self.sim) # Construct a partial function (not a lambda, so the sim is updated when deepcopied)
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
            func (func/Module): the function to insert, which must take a single argument, `sim`; or a module of the sim, whose `step()` is moved there
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
        self._insert_into_plan(**insertion) # First, since making the plan (it is made when first needed) replays the insertions recorded
        self.insertions.append(insertion)
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
            df['time'] = [float(t.value) if isinstance(t, ss.dur) else t for t in df['time']] # Plain numbers (on the module's own axis), or dates, rather than durations
        else:
            errormsg = f'Simulation "{self.sim}" needs to be initialized before exporting the Loop dataframe'
            raise RuntimeError(errormsg)
        times = np.array(self.cpu_time)
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
        to_shrink = ['sim', 'funcs', '_plan'] # Not the plan property, which would make the plan
        ss.utils.shrink(self, to_shrink)
        return

    def plot(self, simplify=False, max_len=100, fig_kw=None, plot_kw=None, scatter_kw=None):
        """
        Plot a diagram of all the events

        Args:
            simplify (bool): if True, skip advance, update_results, and finish_step events, which are automatically applied
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
            filter_out = ['advance', 'update_results', 'finish_step']
            df = df[~df.func_name.isin(filter_out)]
        if max_len:
            df = df[:max_len]
        ticks = df[['func_order', 'label']].dropna().drop_duplicates() # Inserted functions have no func_order, so skip them
        yticks = ticks.func_order
        ylabels = ticks.label
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
            y = np.arange(len(ylabels))[::-1] # Reverse order so plots from top to bottom
        else:
            y = -df.func_order.values # Negative so plots from top to bottom

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

    def __deepcopy__(self, memo):
        """ Deep-copy the loop """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in vars(self).items():
            setattr(new, k, sc.dcp(v, memo=memo, die=False))
        return new
