"""
Parent class for the integration loop.
"""
import time
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

# What classes are externally visible
__all__ = ['Loop']



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
        3. sim.modules:       step()           # Run any custom modules
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
        self.plan = None
        self.index = 0 # The next function to execute
        self.cpu_time = [] # Store the CPU time of execution of each function
        self.df = None # User-friendly verison of the plan
        self.cpu_df = None # User-friendly time analysis
        self.initialized = False
        return

    def init(self):
        """ Parse the sim into the integration plan """
        self.collect_funcs()
        self.collect_abs_tvecs()
        self.make_plan()
        self.to_df()
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
        module = parent.name if isinstance(parent, ss.Module) else parent.__class__.__name__.lower()

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
        for mod in sim.module_list:
            self += mod.start_step

        # Update any nonspecific modules
        for mod in sim.modules():
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

        # Update results
        self += sim.people.update_results
        for mod in sim.module_list:
            self += mod.update_results

        # Apply analyzers
        for ana in sim.analyzers():
            self += ana.step

        # Clean up dead agents, increment the time index, and perform other housekeeping tasks
        for mod in sim.module_list:
            self += mod.finish_step
        self += sim.people.finish_step
        self += sim.finish_step

        return self.funcs

    def collect_abs_tvecs(self):
        """ Collect numerical time arrays for each module """
        self.abs_tvecs = sc.objdict()

        # Handle the sim and people first
        sim = self.sim
        for key in ['sim', sim.people.__class__.__name__.lower()]: # To handle subclassing of People -- TODO, make more elegant!
            self.abs_tvecs[key] = sim.t.tvec

        # Handle all other modules
        for mod in sim.module_list:
            self.abs_tvecs[mod.name] = mod.t.tvec

        return self.abs_tvecs

    def make_plan(self):
        """ Combine the module ordering and the time vectors into the integration plan """
        # Assemble the list of dicts
        raw = []
        ti = -1
        for func_row in self.funcs:
            for t in self.abs_tvecs[func_row['module']]:
                row = func_row.copy()
                row['time'] = t # Add time column
                raw.append(row)

        # Turn it into a dataframe
        self.plan = sc.dataframe(raw)

        # Sort it by step_order, a combination of time and function order
        self.plan['ti'] = 0
        self.plan['label'] = self.plan.module + '.' + self.plan.func_name
        col_order = ['time', 'ti', 'func_order', 'func', 'label', 'module', 'func_name'] # Func in the middle to hide it
        self.plan = self.plan.sort_values(['time','func_order']).reset_index(drop=True)[col_order]

        # Calculate the sim time index (ti)
        start_step = 'sim.start_step'
        ti = -1
        ti_vals = []
        for i,label in enumerate(self.plan.label):
            if label == start_step:
                ti += 1
            ti_vals.append(ti)
        self.plan.loc[:, 'ti'] = ti_vals
        return

    def store_time(self):
        """ Store the current time in as high resolution as possible """
        self.cpu_time.append(time.perf_counter())
        return

    def run_one_step(self):
        """
        Take a single step, i.e. call a single function; only used for debugging purposes.

        Compare sim.run_one_step(), which runs a full timestep (which involves multiple function calls).
        """
        f = self.plan.func[self.index] # Get the next function
        f() # Call it
        self.index += 1 # Increment the time
        return

    def _check_initialized(self):
        """ Check that the Loop has been initialized """
        if not self.initialized:
            errormsg = 'Please initialize the loop (typically sim.init()) before calling insert().'
            raise RuntimeError(errormsg)
        return

    def run(self, until=None, verbose=None):
        """ Actually run the integration loop; usually called by sim.run() """
        self._check_initialized()

        # Convert e.g. '2020-01-01' to an actual date
        if isinstance(until, str):
            until = ss.date(until)

        # Loop over every function in the integration loop, e.g. disease.step()
        self.store_time()
        for f,label in zip(self.plan.func[self.index:], self.plan.label[self.index:]):
            if verbose:
                row = self.plan[self.index]
                print(f'Running t={row.time:n}, step={row.name}, {label}()')

            f() # Execute the function -- this is where all of Starsim happens!!

            # Tidy up
            self.index += 1 # Increment the count
            self.store_time()
            if until is not None and self.sim.now > until: # Terminate if asked to
                break

        self.to_df() # Store results as a dataframe
        return

    def insert(self, func, label=None, match_fn=None, before=False, verbose=True, die=True):
        """
        Insert a function into the loop plan at the specified location.

        The loop plan is a dataframe with columns including time (e.g. `date('2025-05-05')`),
        label (e.g. `'randomnet.step'`), module ('`randomnet'`), and function name (`'step'`).
        By default, this method will match the conditions in the plan based on
        the criteria specified.

        This functionality is similar to an analyzer or an intervention, but gives
        additional flexibility since can be inserted at (almost) any point in a sim.

        Note: the loop must be initialized (`sim.init()`) before you can call this.

        Args:
            func (func): the function to insert; must take a single argument, `sim`
            label (str): the label (module.name) of the function to match; see `sim.loop.plan.label.unique() for choices`
            match_fn (func): if supplied, use this function to perform the matching on the plan dataframe, returning a boolean array or list of indices of matching rows (see example below)
            before (bool): if true, insert the function before rather than after the match
            die (bool): whether to raise an exception if no matches found

        **Examples**:

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
        """
        self._check_initialized()

        if label and match_fn:
            errormsg = "You can supply label or match, but not both; 'label' is equivalent to 'plan.label == label', please include this in your match function"
            raise ValueError(errormsg)

        if label:
            match_fn = lambda plan: plan.label == label

        # Compute the matches
        matches = match_fn(self.plan)
        if matches.dtype == bool:
            matches = sc.findinds(matches)

        # Perform the insertion in reverse order
        name = func.__name__
        sim_func = lambda: func(self.sim) # Construct a partial function
        for m in matches[::-1]:
            ind = m-1 if before else m
            current = self.plan[ind]
            row = dict(
                time = current.time,
                ti = current.ti,
                func_order = None,
                func = sim_func,
                label = name,
                module = None,
                func_name = name,
            )
            self.plan.insertrow(ind, row)

        return

    def to_df(self):
        """ Return a user-friendly version of the plan, omitting object columns """
        # Compute the main dataframe
        cols = ['time', 'ti', 'func_order', 'label', 'module', 'func_name']
        if self.plan is not None:
            df = self.plan[cols].copy() # Need to copy, otherwise it's messed up
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
        shrunk = ss.utils.shrink()
        self.sim = shrunk
        self.funcs = shrunk
        self.plan = shrunk
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
        df = self.cpu_df
        ylabels = df.index.values
        if bytime:
            y = np.arange(len(ylabels))
        else:
            y = df.func_order.values
        y = y[::-1] # Reverse order so plots from top to bottom

        x = df.cpu_time.values
        pcts = df.percent.values

        if x.max() < 1:
            x *= 1e3
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

        **Example**:

            sis = ss.SIS(dt=0.1)
            net = ss.RandomNet(dt=0.5)
            births = ss.Births(dt=1)
            sim = ss.Sim(dt=0.1, dur=5, diseases=sis, networks=net, demographics=births)
            sim.init()
            sim.loop.plot_step_order()
        """
        self._check_initialized()
        df = self.plan
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
        """ A dataframe that has functions in it doesn't copy well; convert to a dict first """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in vars(self).items():
            if k == 'plan' and isinstance(v, sc.dataframe):
                origdict = v.to_dict() # Convert to a dictionary
                newdict = sc.dcp(origdict, memo=memo, die=False) # Copy the dict
                newdf = sc.dataframe(newdict)
                setattr(new, k, newdf)
            else:
                setattr(new, k, sc.dcp(v, memo=memo, die=False)) # Regular deepcopy
        return new

