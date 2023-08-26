"""
Defne HIV
"""

import numpy as np
import sciris as sc
from . import utils as ssu
from . import people as ssp
from . import modules as ssm
from . import results as ssr
from . import interventions as ssi
from . import analyzers as ssa


class HIV(ssm.Disease):
    
    def __init__(self, pars=None):
        super().__init__(pars)
        self.states = ssu.NDict(
            ssp.State('susceptible', bool, True),
            ssp.State('infected', bool, False),
            ssp.State('ti_infected', float, 0),
            ssp.State('on_art', bool, False),
            ssp.State('cd4', float, 500),
            self.states,
        )
    
        self.pars = ssu.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'initial': 30,
            'eff_condoms': 0.7,
        }, self.pars)
        return


    def update_states(self, sim):
        """ Update CD4 """
        hivppl = sim.people.hiv
        hivppl.cd4[sim.people.alive & hivppl.infected & hivppl.on_art] += (sim.pars.hiv.cd4_max - hivppl.cd4[sim.people.alive & hivppl.infected & hivppl.on_art])/sim.pars.hiv.cd4_rate
        hivppl.cd4[sim.people.alive & hivppl.infected & ~hivppl.on_art] += (sim.pars.hiv.cd4_min - hivppl.cd4[sim.people.alive & hivppl.infected & ~hivppl.on_art])/sim.pars.hiv.cd4_rate
        return
    

    def init_results(self, sim):
        super().init_results(sim)
        self.results['n_art'] = ssr.Result('n_art', self.name, sim.npts, dtype=int)
        return
    

    def update_results(self, sim):
        super(HIV, self).update_results(sim)
        sim.results[self.name]['n_art'] = np.count_nonzero(sim.people.alive & sim.people[self.name].on_art)
        return
    

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return
    
    def set_prognoses(self, sim, uids):
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.ti


#%% Interventions

class ART(ssi.Intervention):

    def __init__(self,t:np.array,capacity: np.array):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.capacity = sc.promotetoarray(capacity)

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
                inds = np.random.choice(ssu.true(eligible), min(n_eligible, n_change), replace=False)
                sim.people.hiv.on_art[inds] = True
        elif n_change < 0:
            # Take some people off ART
            eligible = sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art
            inds = np.random.choice(ssu.true(eligible), min(n_change), replace=False)
            sim.people.hiv.on_art[inds] = False
        
        return
    
    
#%% Analyzers
    
class CD4_analyzer(ssa.Analyzer):

    def __init__(self):
        self.requires = HIV
        self.cd4 = None
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.cd4 = np.zeros((sim.npts, sim.people.n), dtype=int)

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
