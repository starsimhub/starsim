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
    """ Base class for integration loop """
    def __init__(self, sim): # TODO: consider eps=1e-6 and round times to this value
        self.sim = sim
        self.funcs = None
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
        row = dict(func_order=len(self.funcs), module=module, func_name=func_name, func=func)
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

        # Update demographic modules (create new agents from births/immigration, schedule non-disease deaths and emigration)
        for dem in sim.demographics():
            self += dem.step

        # Carry out autonomous state changes in the disease modules. This allows autonomous state changes/initializations
        # to be applied to newly created agents
        for disease in sim.diseases():
            if isinstance(disease, ss.Disease): # Could be a connector instead -- TODO, rethink this
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
        for mod in sim.modules:
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

    def collect_abs_tvecs(self):
        """ Collect numerical time arrays for each module """
        self.abs_tvecs = sc.objdict()

        # Handle the sim and people first
        sim = self.sim
        for key in ['sim', 'people']:
            self.abs_tvecs[key] = sim.t.abstvec

        # Handle all other modules
        for mod in sim.modules:
            self.abs_tvecs[mod.name] = mod.t.abstvec

        return self.abs_tvecs

    def make_plan(self):
        """ Combine the module ordering and the time vectors into the integration plan """

        # Assemble the list of dicts
        raw = []
        for func_row in self.funcs:
            for t in self.abs_tvecs[func_row['module']]:
                row = func_row.copy()
                row['time'] = t # Add time column
                raw.append(row)

        # Turn it into a dataframe
        self.plan = sc.dataframe(raw)

        # Sort it by step_order, a combination of time and function order
        self.plan['step_order'] = self.plan.time + ss.options.time_eps*self.plan.func_order
        self.plan['func_label'] = self.plan.module + '.' + self.plan.func_name
        col_order = ['time', 'func_order', 'step_order', 'func', 'func_label', 'module', 'func_name'] # Func in the middle to hide it
        self.plan = self.plan.sort_values('step_order').reset_index(drop=True)[col_order]
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

    def run(self, until=None, verbose=None):
        """ Actually run the integration loop; usually called by sim.run() """

        # Convert e.g. '2020-01-01' to an actual date
        if isinstance(until, str):
            until = ss.date(until)

        # Loop over every function in the integration loop, e.g. disease.step()
        self.store_time()
        for f,label in zip(self.plan.func[self.index:], self.plan.func_label[self.index:]):
            if verbose:
                row = self.plan[self.index]
                print(f'Running t={row.time:n}, step={row.step_order}, {label}()')

            f() # Execute the function -- this is where all of Starsim happens!!

            # Tidy up
            self.index += 1 # Increment the count
            self.store_time()
            if until is not None and self.sim.now > until: # Terminate if asked to
                break
        return

    def manual_reset(self): # TODO: do we need this? I feel if we don't have it, people will be tempted to manually set loop.index = 0.
        """
        Reset the loop to run again. Note, does not reset sim quantities so should
        only be used for debugging.
        """
        self.index = 0
        self.sim.complete = False
        return

    def to_df(self):
        """ Return a user-friendly version of the plan, omitting object columns """
        # Compute the main dataframe
        cols = ['time', 'func_order', 'module', 'func_name', 'func_label']
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
        by_func = df.groupby('func_label')
        method = dict(func_order='first', module='first', func_name='first', cpu_time='sum')
        cdf = sc.dataframe(by_func.agg(method))
        cdf['percent'] = cdf.cpu_time / cdf.cpu_time.sum()*100
        cdf.insert(cdf.cols.index('cpu_time'), 'calls', by_func.size())
        cdf.sort_values('cpu_time', inplace=True, ascending=False)
        self.cpu_df = cdf
        return df

    def plot(self, simplify=False, fig_kw=None, plot_kw=None, scatter_kw=None):
        """
        Plot a diagram of all the events

        Args:
            simplify (bool): if True, skip update_results and finish_step events, which are automatically applied
            fig_kw (dict): passed to ``plt.figure()``
            plot_kw (dict): passed to ``plt.plot()``
            scatter_kw (dict): passed to ``plt.scatter()``
        """

        # Assemble data
        df = self.to_df()
        if simplify:
            filter_out = ['update_results', 'finish_step']
            df = df[~df.func_name.isin(filter_out)]
        yticks = df.func_order.unique()
        ylabels = df.func_label.unique()
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
        plt.xlabel(f'Time since simulation start (in {self.sim.pars.unit}s)')
        plt.grid(True)
        sc.figlayout()
        sc.boxoff()
        return ss.return_fig(fig)

    def plot_cpu(self, bytime=True, fig_kw=None, bar_kw=None):
        """
        Plot the CPU time spent on each event; visualization of Loop.cpu_df.

        Args:
            bytime (bool): if True, order events by total time rather than actual order
            fig_kw (dict): passed to ``plt.figure()``
            plot_kw (dict): passed to ``plt.bar()``
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
        """ A dataframe that has functions in it doesn't copy well; convert to a dict first """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == 'plan' and isinstance(v, sc.dataframe):
                origdict = v.to_dict() # Convert to a dictionary
                newdict = sc.dcp(origdict, memo=memo) # Copy the dict
                newdf = sc.dataframe(newdict)
                setattr(new, k, newdf)
            else:
                setattr(new, k, sc.dcp(v, memo=memo)) # Regular deepcopy
        return new

