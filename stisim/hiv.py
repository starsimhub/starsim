"""
Defne HIV
"""

import numpy as np
import sciris as sc
import stisim as ss

__all__ = ['HIV', 'SimpleDiagnosticTest', 'ART', 'CD4_analyzer']


class HIV(ss.Disease):

    def __init__(self, pars=None):
        super().__init__(pars)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', float, 0)
        self.on_art = ss.State('on_art', bool, False)
        self.cd4 = ss.State('cd4', float, 500)
        
        # SimpleDiagnosticTest states (should this be done as part of SimpleDiagnosticTest.initialize()? )
        self.diagnosed = ss.State('diagnosed', bool, False)
        self.ti_diagnosed = ss.State('ti_diagnosed', float, np.nan)

        # Default parameters
        self.pars = ss.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'initial': 30,
            'eff_condoms': 0.7,
        }, self.pars)
        
        return

    def update_states(self, sim):
        """ Update CD4 """
        self.cd4[sim.people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[sim.people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[sim.people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[sim.people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate
        return

    def init_results(self, sim):
        super().init_results(sim)
        return

    def update_results(self, sim):
        super(HIV, self).update_results(sim)
        return

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids):
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti


# %% Interventions

class SimpleDiagnosticTest(ss.Intervention):
    """ Symtomatic testing intervention."""

    # Initial and final simulation time for diagnostic test execution
    start = None
    end   = None
    ti_start = None
    ti_end   = None
    
    # Threshold for symptomatic cases
    cd4_threshold = None
    
    # Test performance
    coverage = None
    sensitivity = None
    

    def __init__(self, start=None, end=None, coverage=0.8, 
                 sensitivity=0.99, specificity=0.99, 
                 cd4_threshold=500 ):
        """ Initializes the intervention instance.
        
        Args:
          start: A float indicating the beginning time of the intervention.
          end: A float indicating the ending time of the intervention.
          coverage: A float indicating the fraction of symptomatic individuals covered by the intervention.
          sensitivity: Proportion of infected individuals accurately receiving a positive diagnosis.
          cd4_threshold: Level of CD4 that defines symptomatic cases; individuals whose CD4 levels are below this value are considered symptomatic.
        """
        self.requires = HIV
        
        self.start = 0 if start is None else start
        self.end   = 3000 if end is None else end
        
        self.cd4_threshold = cd4_threshold

        self.coverage = coverage
        self.sensitivity = sensitivity
        
        return

    
    def initialize(self, sim):
        """ Configures the intervention in a specific sim instance."""
        
        # Find the time steps where the intervention is active
        self.ti_start = np.clip( int( sim.dt*( self.start - sim.pars['start'] ) ), 0, sim.npts )
        self.ti_end   = np.clip( int( sim.dt*( self.end   - sim.pars['start'] ) ), 0, sim.npts )
        
        # Create rrom for simulation results
        sim.results.hiv += ss.Result('hiv', 'n_diagnosed'  , sim.npts, dtype=int)
        sim.results.hiv += ss.Result('hiv', 'new_diagnoses', sim.npts, dtype=int)

        return


    def apply(self, sim):
        """ Executes the intervention at time step sim.ti."""

        inds = []
        if self.ti_start <= sim.ti <= self.ti_end: 
        
            # Find eligible subjects
            symptomatic = sim.people.hiv.cd4 <= self.cd4_threshold
            eligible = sim.people.alive & sim.people.hiv.infected & symptomatic & ~sim.people.hiv.diagnosed # Remove 'infected' as a condition once we get symptomatic working well (because cd4 levels are not going down now)
            n_eligible = np.count_nonzero(eligible)
            
            # Create new diagnoses
            if n_eligible:
                n_new_diagnoses = int( n_eligible * self.coverage * self.sensitivity )
                inds = np.random.choice(ss.true(eligible), n_new_diagnoses, replace=False)
                sim.people.hiv.diagnosed[inds] = True
                sim.people.hiv.ti_diagnosed[inds] = sim.ti
                    
        # Add results
        sim.results.hiv.n_diagnosed[sim.ti] = np.count_nonzero(sim.people.hiv.diagnosed)
        sim.results.hiv.new_diagnoses[sim.ti] = len(inds)
        
        return




class ART(ss.Intervention):

    def __init__(self, t: np.array, capacity: np.array):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.capacity = sc.promotetoarray(capacity)

    def initialize(self, sim):
        sim.hiv.results += ss.Result(self.name, 'n_art', sim.npts, dtype=int)

    def apply(self, sim):
        if sim.t < self.t[0]:
            return

        capacity = self.capacity[np.where(self.t <= sim.t)[0][-1]]
        on_art = sim.people.alive & sim.people.hiv.on_art

        n_change = capacity - np.count_nonzero(on_art)
        if n_change > 0:
            # Add more ART
            eligible = sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art
            n_eligible = np.count_nonzero(eligible)
            if n_eligible:
                inds = np.random.choice(ss.true(eligible), min(n_eligible, n_change), replace=False)
                sim.people.hiv.on_art[inds] = True
        elif n_change < 0:
            # Take some people off ART
            eligible = sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art
            inds = np.random.choice(ss.true(eligible), min(n_change), replace=False)
            sim.people.hiv.on_art[inds] = False

        # Add result
        sim.results.hiv.n_art = np.count_nonzero(sim.people.alive & sim.people.hiv.on_art)

        return


#%% Analyzers

class CD4_analyzer(ss.Analyzer):

    def __init__(self):
        self.requires = HIV
        self.cd4 = None
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.cd4 = np.zeros((sim.npts, sim.people.n), dtype=int)

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
