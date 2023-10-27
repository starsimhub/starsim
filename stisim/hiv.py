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

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected    = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', int, ss.INT_NAN)
        self.on_art      = ss.State('on_art', bool, False)
        self.ti_art      = ss.State('ti_art', int, ss.INT_NAN)
        self.cd4         = ss.State('cd4', float, 500)

        self.rng_dead = ss.Stream(f'dead_{self.name}')

        self.pars = ss.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'init_prev': 0.05,
            'eff_condoms': 0.7,
            'art_efficacy': 0.96,
        }, self.pars)

        return

    def update_states_pre(self, sim):
        """ Update CD4 """
        self.cd4[sim.people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[sim.people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[sim.people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[sim.people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[sim.people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        hiv_death_prob = 0.05 / (self.pars.cd4_min - self.pars.cd4_max)**2 *  (self.cd4 - self.pars.cd4_max)**2
        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.rng_dead.bernoulli_filter(uids=can_die, prob=hiv_death_prob[can_die])
        sim.people.alive[hiv_deaths] = False
        sim.people.ti_dead[hiv_deaths] = sim.ti
        self.results['new_deaths'][sim.ti] = len(hiv_deaths)

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

    def set_prognoses(self, sim, uids, from_uids=None):
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        return

    def set_congenital(self, sim, uids):
        return self.set_prognoses(sim, uids)  # Pass back?


# %% Interventions

class ART(ss.Intervention):

    def __init__(self, t: np.array, coverage: np.array, **kwargs):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.coverage = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.rng_add_ART = ss.Stream('add_ART')

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
            inds = self.rng_add_ART.bernoulli_filter(uids=recently_infected, prob=coverage)
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