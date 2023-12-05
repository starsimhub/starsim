"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import stisim as ss
from .disease import STI

__all__ = ['HIV', 'ART', 'CD4_analyzer']


class HIV(STI):

    def __init__(self, pars=None):
        super().__init__(pars)

        # States additional to the default disease states (see base class)
        self.on_art = ss.State('on_art', bool, False)
        self.cd4 = ss.State('cd4', float, 500)

        self.pars = ss.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'init_prev': 0.05,
            'eff_condoms': 0.7,
        }, self.pars)

        return

    def update_states_pre(self, sim):
        """ Update CD4 """
        self.cd4[sim.people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[sim.people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[sim.people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[sim.people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        return super().init_results(sim)

    def update_results(self, sim):
        return super(HIV, self).update_results(sim)

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids):
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        return

    def set_congenital(self, sim, uids):
        return self.set_prognoses(sim, uids)  # Pass back?


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, t: np.array, capacity: np.array):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.capacity = sc.promotetoarray(capacity)
        return

    def initialize(self, sim):
        sim.hiv.results += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        return

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
        return

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
        return
