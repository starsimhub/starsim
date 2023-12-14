"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import stisim as ss
from .disease import STI
import scipy.stats as sps

__all__ = ['HIV', 'ART', 'CD4_analyzer']


class HIV(STI):

    def __init__(self, pars=None):
        super().__init__(pars)

        self.on_art      = ss.State('on_art', bool, False)
        self.ti_art      = ss.State('ti_art', int, ss.INT_NAN)
        self.cd4         = ss.State('cd4', float, 500)
        self.ti_dead     = ss.State('ti_dead', int, ss.INT_NAN) # Time of HIV-cause death

        self.death_prob_per_dt = sps.bernoulli(p=self.death_prob)

        self.pars = ss.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'seed_infections': sps.bernoulli(p=0.05),
            'eff_condoms': 0.7,
            'art_efficacy': 0.96,
        }, self.pars)

        return

    @staticmethod
    def death_prob(self, sim, uids):
        return 0.05 / (self.pars.cd4_min - self.pars.cd4_max)**2 *  (self.cd4[uids] - self.pars.cd4_max)**2

    def update_pre(self, sim):
        """ Update CD4 """
        self.cd4[sim.people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[sim.people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[sim.people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[sim.people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[sim.people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.death_prob_per_dt.filter(can_die)
        
        sim.people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = sim.ti
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        super(HIV, self).update_results(sim)
        self.results['new_deaths'][sim.ti] = np.count_nonzero(self.ti_dead == sim.ti)
        return 

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        return

    def set_congenital(self, sim, target_uids, source_uids):
        return self.set_prognoses(sim, target_uids)  # Pass back?


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, t: np.array, coverage: np.array, **kwargs):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.coverage = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.prob_art_at_infection = sps.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.t, self.coverage))
        return

    def initialize(self, sim):
        super().initialize(sim)
        sim.results.hiv += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.ti < self.t[0]:
            return

        ti_delay = 1 # 1 time step delay TODO
        recently_infected = ss.true((sim.people.hiv.ti_infected == sim.ti-ti_delay) & sim.people.alive)

        n_added = 0
        if len(recently_infected) > 0:
            inds = self.prob_art_at_infection.filter(recently_infected)
            sim.people.hiv.on_art[inds] = True
            sim.people.hiv.ti_art[inds] = sim.ti
            n_added = len(inds)

        # Add result
        sim.results.hiv.n_art = np.count_nonzero(sim.people.alive & sim.people.hiv.on_art)

        return n_added


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
