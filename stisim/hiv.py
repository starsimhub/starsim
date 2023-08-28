"""
Defne HIV
"""

import numpy as np
import sciris as sc
import stisim as ss

__all__ = ['HIV', 'ART', 'CD4_analyzer']


class HIV(ss.Disease):

    def __init__(self, pars=None):
        super().__init__(pars)
        self.states = ss.ndict(
            ss.State('susceptible', bool, True),
            ss.State('infected', bool, False),
            ss.State('ti_infected', float, 0),
            ss.State('on_art', bool, False),
            ss.State('cd4', float, 500),
            self.states,
        )

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
        self.results['n_art'] = ss.Result('n_art', self.name, sim.npts, dtype=int)
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
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti


# %% Interventions

class ART(ss.Intervention):

    def __init__(self, t: np.array, capacity: np.array):
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
                inds = np.random.choice(ss.true(eligible), min(n_eligible, n_change), replace=False)
                sim.people.hiv.on_art[inds] = True
        elif n_change < 0:
            # Take some people off ART
            eligible = sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art
            inds = np.random.choice(ss.true(eligible), min(n_change), replace=False)
            sim.people.hiv.on_art[inds] = False

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
