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

        self.on_art      = ss.State('on_art', bool, False)
        self.ti_art      = ss.State('ti_art', int, ss.INT_NAN)
        self.cd4         = ss.State('cd4', float, 500)
        self.ti_dead     = ss.State('ti_dead', int, ss.INT_NAN) # Time of HIV-cause death

        self.rng_dead = ss.RNG(f'dead_{self.name}')

        self.pars = ss.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'init_prev': 0.05,
            'eff_condoms': 0.7,
            'art_efficacy': 0.96,
        }, self.pars)

        return

    def update_pre(self, sim):
        """ Update CD4 """
        self.cd4[sim.people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[sim.people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[sim.people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[sim.people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[sim.people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        hiv_death_prob = 0.05 / (self.pars.cd4_min - self.pars.cd4_max)**2 *  (self.cd4 - self.pars.cd4_max)**2
        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.rng_dead.bernoulli_filter(hiv_death_prob[can_die], can_die)
        
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

    def set_prognoses(self, sim, uids, from_uids=None):
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        return

    def set_congenital(self, sim, uids):
        return self.set_prognoses(sim, uids)  # Pass back?


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, t: np.array, coverage: np.array, **kwargs):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.coverage = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.rng_add_ART = ss.RNG('add_ART')

        return

    def initialize(self, sim):
        sim.results.hiv += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.ti < self.t[0]:
            return

        coverage = self.coverage[np.where(self.t <= sim.ti)[0][-1]]
        ti_delay = 1 # 1 time step delay
        recently_infected = ss.true((sim.people.hiv.ti_infected == sim.ti-ti_delay) & sim.people.alive)

        n_added = 0
        if len(recently_infected) > 0:
            inds = self.rng_add_ART.bernoulli_filter(recently_infected, prob=coverage)
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