"""
The integration loop: the order in which each module's code runs in each timestep

It exists so that modules with different timesteps run together in one fixed order, which can be inspected with `sim.loop`.

Used with: agents and compartments; declared and coded modules
"""
import dis
import time
import functools as ft
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
plt = sc.lazyimport('matplotlib.pyplot') # Imported when first used, since most runs never plot


def empty_function(func):
    """ Whether a function does nothing: `pass`, with or without a docstring """
    return [(i.opname, i.argval) for i in dis.get_instructions(func) if i.opname not in ['RESUME', 'NOP']] in [[('RETURN_CONST', None)], [('LOAD_CONST', None), ('RETURN_VALUE', None)]]


@dataclass(slots=True)
class LoopEntry:
    """
    One executable event in the integration loop: its time, module, and function (with slots, since a plan has 10⁴–10⁵ of them)
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
    Define the integration loop: the order in which each function is called in the sim (see `collect_funcs()`)

    Each type of module is called at a different time; within each type, in the order listed. The default order is:

        1. sim:               advance()        # Move the time to this step's time point (from the second step on)
        2. sim:               start_step()     # Initialize the sim, including plotting progress
        3. all modules:       start_step()     # Initialize the modules, including the random number distribution
        4. people:            record_states()  # Record the basic states at the start of the step
        5. all modules:       record_states()  # Record each module's states at the start of the step
        6. sim.custom:        step()           # Run any custom modules
        7. sim.demographics:  step()           # Update the demographics, including adding new agents
        8. sim.diseases:      step_state()     # Update the disease states, e.g. exposed -> infected
        9. sim.connectors:    step()           # Run the connectors
        10. sim.networks:     step()           # Run the networks, including adding/removing edges
        11. sim.interventions: step()          # Run the interventions
        12. sim.diseases:     step()           # Run the diseases, including transmission
        13. people:           step_die()       # Figure out who died on this timestep
        14. people:           record_events()   # Record who died or left in the step
        15. all modules:      record_events()   # Record what happened in the step
        16. sim.analyzers:    step()           # Run the analyzers
        17. all modules:      finish_step()    # Do any final tidying
        18. people:           finish_step()    # Clean up dead agents
        19. sim:              finish_step()    # Finish the step

    With `ss.Sim(step_order='given')`, steps 4 to 10 are instead each module's step in the
    order the modules were given (a disease's `step_state()` and then its `step()`).

    A sim that is nothing but compartments the kernels can run from start to finish goes
    through them in one call instead, not this order step by step (see `sim.describe()`).

    One time index holds for every function of a step, in every module: the step from time
    point k to k+1 is step k, and throughout it `sim.ti == k` and `sim.now` is time point k,
    which is when the step happens, what it reads time-varying inputs and dates at, and where
    its results go. A step is taken from every time point, the last included, so each time
    point is a period: its stocks are the state at its start (recorded by `record_states()`,
    before any module's step), and its flows the totals over the step (recorded by
    `record_events()`, after them). The state after the last step is not recorded.

    Args:
        sim (Sim): the sim to run (usually made by the sim itself)
    """
    def __init__(self, sim):
        self.sim = sim
        self.funcs = None
        self.shared_timelines = [] # The timelines of the modules on the sim's timeline, which the sim's advance() moves
        self.abs_tvecs = None
        self.abs_numvecs = None # Canonical numeric (float-year) time arrays, parallel to abs_tvecs, used for sorting/uniformity
        self._plan = None # Made when it is first needed: see the plan property
        self.insertions = []
        self.index = 0 # The next function to execute
        self.entry = None # The entry of the plan running now, if any
        self.profile = False # If True, record per-entry CPU timing during run() (see Loop.run)
        self.cpu_time = [] # Store the CPU time of execution of each function (only if profiling)
        self.stopped = None # The step at which a run until a condition stopped, if it did: see run()
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
        return len(self.plan) if self.initialized else 0

    def add(self, func):
        """ Add a function (a method of the sim, the people, or a module) to the end of the list of functions of a step, unless it would do nothing (see `_keep()`) """
        if not self._keep(func):
            return self
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
        row = dict(
            func_order = len(self.funcs),
            module = module,
            func_name = func_name,
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

        self.funcs = [] # Reset, just in case
        sim = self.sim

        # Move the time to the step about to be taken, so that it is the same in every function of the step: the sim's time,
        # with that of the modules on its timeline; a module with a timeline of its own moves its own. The first step is at the
        # time the sim starts at, so these are not called then (see make_plan())
        self.shared_timelines = [mod.t for mod in sim.modules if self._on_sim_time(mod)] # Their timelines, not the modules, so a search of the sim does not find the modules again here
        self.add(sim.advance)
        for mod in sim.modules:
            if not self._on_sim_time(mod):
                self.add(mod.advance)

        # Collect the start_steps
        self.add(sim.start_step)
        for mod in sim.modules:
            self.add(mod.start_step)

        # Record the state at the start of the step. A module's record_states() is skipped only when it is the inherited
        # base method and the module has no auto-generated boolean-state results to count (see _noop())
        if sim.solver.record_people: # People results can be opted out for lightweight sims
            self.add(sim.people.record_states)
        for mod in sim.modules:
            self.add(mod.record_states)

        # Rescale the population if needed
        if sim.solver.rescale:
            self.add(sim.rescale)

        if sim.pars.step_order == 'given': # Each module steps in the order it was given in (analyzers still last, after the results)
            self.collect_given(sim)
        else:
            self.collect_by_type(sim)

        # Update people who died -- calls each module's on_death() internally
        self.add(sim.people.step_die)

        # Record what happened in the step
        if sim.solver.record_people:
            self.add(sim.people.record_events)
        for mod in sim.modules:
            self.add(mod.record_events)

        # Apply analyzers
        for ana in sim.analyzers():
            self.add(ana.step)

        # Clean up dead agents, and perform other housekeeping tasks
        for func in [mod.finish_step for mod in sim.modules] + [sim.people.finish_step]:
            self.add(func)
        self.add(sim.finish_step)

        return self.funcs

    def _on_sim_time(self, mod):
        """ Whether a module shares the sim's timeline (the usual case: the same time points in the same unit), so its time moves with the sim's """
        t, sim_t = mod.t, self.sim.t
        return len(t.yearvec) == len(sim_t.yearvec) and getattr(t.tvec, 'unit', None) is getattr(sim_t.tvec, 'unit', None) and np.array_equal(t.yearvec, sim_t.yearvec)

    def collect_by_type(self, sim):
        """ The steps of the modules, each kind at its place in the loop (the default; see `ss.Loop`) """

        # Update any nonspecific modules
        for mod in sim.custom():
            self.add(mod.step)

        # Update demographic modules (create new agents from births/immigration, schedule non-disease deaths and emigration)
        for dem in sim.demographics():
            self.add(dem.step)

        # Carry out autonomous state changes in the disease modules. This allows autonomous state changes/initializations
        # to be applied to newly created agents
        for disease in sim.diseases():
            if isinstance(disease, ss.Disease):
                self.add(disease.step_state)

        # Update connectors
        for connector in sim.connectors():
            self.add(connector.step)

        # Update networks - this takes place here in case autonomous state changes at this timestep
        # affect eligibility for contacts
        for network in sim.networks():
            self.add(network.step)

        # Apply interventions - new changes to contacts will be visible and so the final networks can be customized by
        # interventions, by running them at this point
        for intv in sim.interventions():
            self.add(intv.step)

        # Carry out autonomous state changes in the disease modules, including transmission (but excluding deaths)
        for disease in sim.diseases():
            self.add(disease.step)
        return

    def collect_given(self, sim):
        """ The steps of the modules in the order they were given in, with `ss.Sim(step_order='given')`; a disease takes its state changes and then its step together, and analyzers still step last """
        mods = [mod for mod in sim.modules if not isinstance(mod, (ss.Analyzer, ss.Product))]
        mods.sort(key=lambda mod: getattr(mod, '_given_index', np.inf)) # Modules added later (e.g. by sim.add_module()) after those given, by kind
        for mod in mods:
            if isinstance(mod, ss.Disease):
                self.add(mod.step_state)
            self.add(mod.step)
        return

    def _keep(self, func):
        """ Whether to put a call in the plan (see `_noop()`); one left out counts as called, for `ss.Sim._check_method_calls()` """
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
        would act on, and for a `step()` with nothing in it: a module's `start_step()` with no random numbers to move on, its
        `finish_step()`, `record_states()` and `record_events()` when inherited
        unchanged, and the people's deaths and removals in a run without agents (in which nobody
        dies as an agent). The sim's own calls are always kept, as places to insert others. A method that a
        module overrides is always kept, so a module written as code costs the loop only the
        methods it defines.
        """
        obj, name = func.__self__, func.__name__
        sim = self.sim
        if isinstance(obj, ss.Module):
            if name == 'step':
                return empty_function(func) # E.g. an analyzer that only has finalize(), which needs one
            elif name not in ['start_step', 'finish_step', 'record_states', 'record_events']: # An intervention given as a function
                return False
            inherited = getattr(type(obj), name) is getattr(ss.Module, name)
            if name == 'start_step':
                return inherited and (obj.dists is None or not ss.options.crn) # Moving the streams on is all it does
            elif name == 'record_states': # Counting its states is all it does
                return inherited and not obj.auto_state_list and not obj.derived_state_names
            return inherited # finish_step() and record_events()
        elif obj is sim.people and name in ['step_die', 'finish_step']:
            people = type(obj)
            same = all(getattr(people, m) is getattr(ss.People, m) for m in ['step_die', 'finish_step', 'update_post', 'remove_dead'])
            return same and not sim.solver.agents # One row per group, which never die or leave
        return False

    def collect_abs_tvecs(self):
        """
        Collect the time arrays of each module: its times (`abs_tvecs`, each plan entry's time), and the same as numbers
        (`abs_numvecs`, in years), which `make_plan()` sorts by, since comparing dates and durations is slow
        """
        self.abs_tvecs = sc.obj()
        self.abs_numvecs = sc.obj()

        # Handle the sim and people first
        sim = self.sim
        for key in ['sim', sim.people.__class__.__name__.lower()]: # The people by the name of its class, which may be a subclass (see add())
            self.abs_tvecs[key] = sim.t.tvec
            self.abs_numvecs[key] = sim.t.yearvec

        # Handle all other modules
        for mod in sim.modules:
            self.abs_tvecs[mod.name] = mod.t.tvec
            self.abs_numvecs[mod.name] = mod.t.yearvec

        return self.abs_tvecs

    def make_plan(self):
        """
        Combine the module ordering and the time vectors into the integration plan: the `LoopEntry` events by time, then by
        function order; made directly if every module shares the sim's timeline (the usual case), else by sorting
        """
        if all(self._on_sim_time(mod) for mod in self.sim.modules):
            self.plan = self._make_plan_uniform()
        else:
            self.plan = self._make_plan_sorted()

        # Replay any user insertions tied to the current loop definition
        for insertion in self.insertions:
            self._insert_into_plan(**insertion)

        return

    def _make_plan_uniform(self):
        """
        The plan when every module shares the sim's timeline: at each time point, every function in order (the first with
        no advance(), since the timelines start there)
        """
        tvec = self.abs_tvecs['sim']
        # Precompute the per-function fields once, rather than rebuilding them per entry
        specs = [(fr['func_order'], fr['func'], f"{fr['module']}.{fr['func_name']}", fr['module'], fr['func_name']) for fr in self.funcs]
        plan = []
        append = plan.append
        for ti in range(len(tvec)):
            t = tvec[ti]
            for func_order, func, label, module, func_name in specs:
                if ti or func_name != 'advance':
                    append(LoopEntry(t, ti, func_order, func, label, module, func_name)) # By position, which is quicker for the many entries
        return plan

    def _make_plan_sorted(self):
        """
        The plan when modules have timelines of their own: every entry, sorted by numeric time and then function order
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
            for j in range(1 if func_name == 'advance' else 0, len(tvec)): # A timeline starts at its first time point, so is not moved there
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

    def run_one_step(self):
        """
        Call a single function of the plan, for debugging; `sim.run_one_step()` instead runs a full timestep
        """
        self._check_initialized()
        entry = self.plan[self.index] # Get the next entry
        self.entry = entry
        try:
            entry.func() # Call it
        finally:
            self.entry = None
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

        With `profile=True`, the CPU time of each entry is recorded (the `cpu_time` column of `to_df()`, and `plot_cpu()`).

        Args:
            until (str/date/Expr/func): if a time, stop after the step that starts at it (used by sim.run_one_step); if a condition (see `sim.run()`), stop once it holds at the end of a step
            verbose (bool): if True, print each function call as it runs
            profile (bool): if True, record per-entry CPU timing (default: keep the current setting)
        """
        self._check_initialized()
        if profile is not None:
            self.profile = profile

        # Convert e.g. '2020-01-01' or 2020 to a time on the timeline, as interventions read them (see ss.connectors.to_time()); a duration is read by sim.run(), as ss.Timeline.to_time() reads one (that long after the start with a calendar)
        cond = None
        if isinstance(until, (str, ss.date)) or sc.isnumber(until):
            until = ss.connectors.to_time(self.sim, until)
        elif isinstance(until, ss.expressions.Expr) or callable(until): # A condition, e.g. sir.I.n == 0
            cond, until = until, None
        self.stopped = None

        # Loop over every function in the integration loop, e.g. disease.step()
        plan = self.plan
        while self.index < len(plan):
            entry = plan[self.index]
            if until is not None and entry.label == 'sim.advance' and self.sim.t.tvec[entry.ti] > until: # Stop before a step that starts after the time asked for
                break
            if cond is not None and entry.label == 'sim.advance' and self.sim._condition_holds(cond): # It holds at the end of the step just taken
                self.stopped = self.sim.t.ti
                break
            if verbose:
                now = entry.time if isinstance(entry.time, ss.date) else f'{entry.time:n}' # Dates don't support number formatting
                print(f'Running t={now}, step={self.index}, {entry.label}()')

            if self.profile:
                start = time.perf_counter()
            self.entry = entry
            try:
                entry.func() # Execute the function -- this is where all of Starsim happens!!
            finally:
                self.entry = None

            # Tidy up
            self.index += 1 # Increment the count
            if self.profile:
                self.cpu_time.append(time.perf_counter() - start)

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

        else: # Perform the insertion in reverse order
            name = func.__name__
            sim_func = ft.partial(func, self.sim) # Construct a partial function (not a lambda, so the sim is updated when deepcopied)
            for m in sorted(matches, reverse=True):
                current = self.plan[m]
                row = LoopEntry(time=current.time, ti=current.ti, func_order=None, func=sim_func, label=name, module=None, func_name=name)
                self.plan.insert(m if before else m+1, row)

        self.df = None
        self.cpu_df = None
        return

    def insert(self, func, label=None, match_fn=None, before=False):
        """
        Insert a function into the loop plan at the specified location, like an analyzer or an intervention but at (almost) any point; after `sim.init()`

        The plan is matched as a dataframe with columns including time (e.g. `date('2025-05-05')`), label (e.g. `'randomnet.step'`),
        module (`'randomnet'`), and function name (`'step'`).

        Args:
            func (func/Module): the function to insert, which must take a single argument, `sim`; or a module of the sim, whose `step()` is moved there
            label (str): the label (module.name) of the function to match; see `sim.loop.to_df().label.unique()` for choices
            match_fn (func): if supplied, use this function to perform the matching on the plan dataframe, returning a boolean array or list of indices of matching rows (see example below)
            before (bool): if true, insert the function before rather than after the match

        Examples:
            ```python
            # Simple label matching with analyzer-like functionality
            def check_pop_size(sim):
                print(f'Population size is {sim.people.n:n}')

            sim = ss.Sim(diseases='sir', networks='random', demographics=True)
            sim.init()
            sim.loop.insert(check_pop_size, label='people.finish_step')
            sim.run()

            # Function-based matching with intervention-like functionality
            def match_fn(plan):
                past_2010 = plan.time > ss.date(2010)
                is_step = (plan.label == 'sis.step') | (plan.label == 'randomnet.step')
                return past_2010 * is_step

            def update_betas(sim):
                if not sim.metadata.get('updated'):
                    print(f'Updating beta values on {sim.now}')
                    sim.diseases.sis.pars.beta = 0.1
                    sim.networks.randomnet.edges.weight[:] = 0.5
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
        Return a user-friendly version of the plan, omitting object columns, also kept as `self.df` (and the time by function as `self.cpu_df`)

        The `cpu_time` column is filled only if the run was profiled (`sim.perf_profile(kind="loop")`); otherwise it is `NaN`.
        """
        # Compute the main dataframe
        if self.plan is not None:
            df = self.plan_metadata()
            df['time'] = [float(t.value) if isinstance(t, ss.Duration) else t for t in df['time']] # Plain numbers (on the module's own axis), or dates, rather than durations
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
            simplify (bool): if True, skip advance, record_states, record_events, and finish_step events, which are automatically applied
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
            filter_out = ['advance', 'record_states', 'record_events', 'finish_step']
            df = df[~df.func_name.isin(filter_out)]
        if max_len:
            df = df[:max_len]
        ticks = df[['func_order', 'label']].dropna().drop_duplicates() # Inserted functions have no func_order, so skip them
        yticks = ticks.func_order
        ylabels = ticks.label
        x = ss.DateArray(df.time) if pd.api.types.is_datetime64_any_dtype(df.time) else df.time # Dates on Starsim's date axis, rather than Matplotlib's, whose labels crowd
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
        plt.xlabel('Time')
        plt.grid(True)
        sc.figlayout()
        sc.boxoff()
        return ss.return_fig(fig)

    def plot_cpu(self, bytime=True, max_entries=10, fig_kw=None, bar_kw=None):
        """
        Plot the CPU time spent on each event; visualization of Loop.cpu_df.

        Args:
            bytime (bool): if True, order events by total time rather than actual order
            max_entries (int): the number of events to show
            fig_kw (dict): passed to `plt.figure()`
            bar_kw (dict): passed to `plt.bar()`
        """
        # Assemble data
        if not len(self.cpu_time): # No per-entry timing was recorded, so there is nothing to plot
            errormsg = 'No CPU timing was recorded: profile the sim with sim.perf_profile(kind="loop") (or loop.run(profile=True)) first'
            raise ValueError(errormsg)
        if self.cpu_df is None:
            self.to_df()
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
