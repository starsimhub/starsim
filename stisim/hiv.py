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
        self.ti_infected = ss.State('ti_infected', float, 0)
        self.on_art      = ss.State('on_art', bool, False)
        self.on_prep     = ss.State('on_prep', bool, False)
        self.cd4         = ss.State('cd4', float, 500)

        self.rng_dead = ss.Stream('dead')

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

        self.rel_sus[sim.people.alive & self.on_prep] = 0.1

        hiv_death_prob = 0.1 / (self.pars.cd4_min - self.pars.cd4_max)**2 *  (self.pars.cd4_max - self.cd4)**2
        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.rng_dead.bernoulli_filter(prob=hiv_death_prob[can_die], arr = can_die)
        
        sim.people.alive[hiv_deaths] = False
        sim.people.ti_dead[hiv_deaths] = sim.ti
        self.results['new_deaths'][sim.ti] = len(hiv_deaths)

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

    def set_prognoses(self, sim, to_uids, from_uids=None):
        self.susceptible[to_uids] = False
        self.infected[to_uids] = True
        self.ti_infected[to_uids] = sim.ti


# %% Interventions

class ART(ss.Intervention):

    def __init__(self, t: np.array, capacity: np.array):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.capacity = sc.promotetoarray(capacity)

        self.rng_add_ART = ss.Stream('add_ART', seed_offset=100)
        self.rng_remove_ART = ss.Stream('remove_ART', seed_offset=101)

        return

    def initialize(self, sim):
        sim.results.hiv += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.ti < self.t[0]:
            return

        capacity = self.capacity[np.where(self.t <= sim.ti)[0][-1]]
        on_art = sim.people.alive & sim.people.hiv.on_art

        n_change = capacity - np.count_nonzero(on_art)
        if n_change > 0:
            # Add more ART
            eligible = ss.true(sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art)
            n_eligible = len(eligible) #np.count_nonzero(eligible)
            if n_eligible:
                inds = self.rng_add_ART.bernoulli_filter(prob=min(n_eligible, n_change)/n_eligible, arr=eligible)
                sim.people.hiv.on_art[inds] = True
        elif n_change < 0:
            # Take some people off ART
            eligible = sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art
            n_eligible = np.count_nonzero(eligible)
            inds = self.rng_remove_ART.bernoulli_filter(prob=-n_change/n_eligible, arr=eligible)
            sim.people.hiv.on_art[inds] = False

        # Add result
        sim.results.hiv.n_art = np.count_nonzero(sim.people.alive & sim.people.hiv.on_art)

        return

class PrEP(ss.Intervention):

    def __init__(self, t: np.array, capacity: np.array):
        self.requires = HIV
        self.t = sc.promotetoarray(t)
        self.capacity = sc.promotetoarray(capacity)

        self.rng_add_PrEP = ss.Stream('add_PrEP', seed_offset=102)
        self.rng_remove_PrEP = ss.Stream('remove_PrEP', seed_offset=103)

        return

    def initialize(self, sim):
        sim.results.hiv += ss.Result(self.name, 'n_prep', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.ti < self.t[0]:
            return

        capacity = self.capacity[np.where(self.t <= sim.ti)[0][-1]]
        on_prep = sim.people.alive & sim.people.hiv.on_prep

        n_change = capacity - np.count_nonzero(on_prep)
        if n_change > 0:
            # Add more PrEP
            eligible = ss.true(sim.people.alive & ~sim.people.hiv.infected & ~sim.people.hiv.on_prep)
            n_eligible = len(eligible) #np.count_nonzero(eligible)
            if n_eligible:
                inds = self.rng_add_PrEP.bernoulli_filter(prob=min(n_eligible, n_change)/n_eligible, arr=eligible)
                sim.people.hiv.on_prep[inds] = True
        elif n_change < 0:
            # Take some people off PrEP
            eligible = sim.people.alive & sim.people.hiv.on_prep
            n_eligible = np.count_nonzero(eligible)
            inds = self.rng_remove_PrEP.bernoulli_filter(prob=-n_change/n_eligible, arr=eligible)
            sim.people.hiv.on_prep[inds] = False

        # Add result
        sim.results.hiv.n_prep = np.count_nonzero(sim.people.alive & sim.people.hiv.on_prep)

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
