"""
Parent class for the integration loop.
"""

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
        self.timearrays = None
        self.plan = None
        self.index = 0 # The next function to execute
        self.initialized = False
        return
    
    def init(self):
        """ Parse the sim into the integration plan """
        self.collect_funcs()
        self.collect_timearrays()
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
        
        # Get the name if it's defined, the class otherwise; these must match timearrays
        module = parent.name if isinstance(parent, ss.Module) else parent.__class__.__name__.lower()
        
        # Create the row and append it to the function list
        row = dict(func_order=len(self.funcs), module=module, func_name=func_name, func=func)
        self.funcs.append(row)
        return self
    
    def __repr__(self):
        if self.initialized:
            arrs = list({len(arr) for arr in self.timearrays.values()})
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
        self += sim.start_step # Note special __iadd__() method above, which appends these to the funcs list
        
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
    
    def collect_timearrays(self):
        """ Collect numerical time arrays for each module """
        self.timearrays = sc.objdict()
        
        # Handle the sim and people first
        sim = self.sim
        for key in ['sim', 'people']:
            self.timearrays[key] = sim.timearray
        
        # Handle all other modules
        for mod in sim.modules:
            timearray = ss.make_timearray(mod.timevec, mod.unit, sim.pars.unit)
            self.timearrays[mod.name] = timearray
            
        return self.timearrays
    
    def make_plan(self):
        """ Combine the module ordering and the time vectors into the integration plan """
        
        # Assemble the list of dicts
        raw = []
        for func_row in self.funcs:
            for time in self.timearrays[func_row['module']]:
                row = func_row.copy()
                row['time'] = time
                row['key'] = (time, row['func_order'])
                row['func_label'] = f"{row['module']}.{row['func_name']}"
                raw.append(row)
        
        # Turn it into a dataframe and sort it
        col_order = ['time', 'func_order', 'func', 'key', 'func_label', 'module', 'func_name'] # Func in the middle to hide it
        self.plan = sc.dataframe(raw).sort_values('key').reset_index(drop=True)[col_order]
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
        # Loop over every function in the integration loop, e.g. disease.step()
        for f in self.plan.func[self.index:]:
            if verbose:
                row = self.plan[self.index]
                print(f'Running t={row.time:n}, {row.func_label}()')
            
            f() # Execute the function -- this is where all of Starsim happens!!

            # Tidy up
            self.index += 1 # Increment the count
            if until and self.sim.now > until: # Terminate if asked to
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
        cols = ['time', 'func_order', 'module', 'func_name', 'func_label']
        df = self.plan[cols]
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
        return fig
    
    def __deepcopy__(self, memo):
        """ A dataframe that has functions in it doesn't copy well; convert to a dict first """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == 'plan':
                origdict = v.to_dict() # Convert to a dictionary
                newdict = sc.dcp(origdict, memo=memo) # Copy the dict
                newdf = sc.dataframe(newdict)
                setattr(new, k, newdf)
            else:
                setattr(new, k, sc.dcp(v, memo=memo)) # Regular deepcopy
        return new
        
    