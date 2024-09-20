"""
Parent class for the integration loop.
"""

import sciris as sc

# What classes are externally visible
__all__ = ['Loop']



#%% Loop class

class Loop:
    """ Base class for integration loop """
    
    def __init__(self, sim):
        self.sim = sim
        self.plan = sc.dataframe(columns=['time', 'module', 'funcname', 'call'])
        return
    
    def initialize(self):
        """ Parse the sim modules into the integration plan """
    
    def run(self):
        """ Actually run the integration loop """
        for i,event in self.plan.enumrows():
            event.call() # This is all that's needed to execute the step
        return
    
    def plot(self):
        """ Plot a diagram of all the events """
        raise NotImplementedError
        
    