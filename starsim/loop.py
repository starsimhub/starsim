"""
Parent class for the integration loop.
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

# What classes are externally visible
__all__ = ['Loop']



#%% Loop class

class Loop:
    """ Base class for integration loop """
    def __init__(self, sim):
        self.sim = sim
        self.funcs = []
        self.timearrays = sc.objdict()
        self.plan = sc.dataframe(columns=['order', 'time', 'module', 'func_name', 'func'])
        return
    
    def init(self):
        """ Parse the sim into the integration plan """
        self.collect_funcs()
        self.collect_timearrays()
        self.make_plan()
        return
    
    def __iadd__(self, func):
        """ Allow functions to be added to the function list """
        parent = func.__self__
        func_name = func.__name__
        
        # Get the name if it's defined, the class otherwise; these must match timearrays
        if isinstance(parent, ss.Sim):
            module = 'sim'
        elif isinstance(parent, ss.People):
            module = 'people'
        else:
            module = parent.name
        
        # Create the row and append it to the function list
        row = dict(func_order=len(self.funcs), module=module, func_name=func_name, func=func)
        self.funcs.append(row)
        return self

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
            
        # Clean up dead agents and perform other housekeeping tasks
        self += sim.people.finish_step
        self += sim.finish_step
        return self.funcs
    
    def collect_timearrays(self):
        """ Collect numerical time arrays for each module """
        
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
    
    def run(self, until=np.nan): # TODO: process until into absolute units
        """ Actually run the integration loop """
        for t,f in zip(self.plan.time, self.plan.func):
            if t > until: break # Terminate if asked to
            f() # Actually execute the step -- this is where all of Starsim happens!!
        return
    
    def plot(self, fig_kw=None, **kwargs):
        """ Plot a diagram of all the events """
        # Assemble data
        yticks = self.plan.func_order.unique()
        ylabels = self.plan.func_label.unique()
        x = self.plan.time
        y = self.plan.func_order
        
        # Do the plotting
        kw = sc.mergedicts(lw=2, markersize=4, alpha=0.8)
        fig = plt.figure(**sc.mergedicts(fig_kw))
        plt.plot(x, y, 'o-', **kw)
        plt.yticks(yticks, ylabels)
        plt.title(f'Integration plan ({len(self.plan)} events)')
        plt.xlabel(f'Time since simulation start (in {self.sim.pars.unit}s)')
        plt.grid(axis='y')
        sc.figlayout()
        sc.boxoff()
        return fig
        
    