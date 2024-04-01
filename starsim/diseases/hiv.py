"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['HIV', 'ART', 'CD4_analyzer']

class HIV(ss.Infection):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):

        # States
        self.add_states(
            ss.State('on_art', bool, False),
            ss.State('ti_art', int, ss.INT_NAN),
            ss.State('cd4', float, 500),
            ss.State('ti_dead', int, ss.INT_NAN), # Time of HIV-cause death
        )

        pars = ss.omergeleft(pars,
            cd4_min = 100,
            cd4_max = 500,
            cd4_rate = 5,
            init_prev = 0.05,
            eff_condoms = 0.7,
            art_efficacy = 0.96,
            death_prob = 0.05,
        )

        par_dists = ss.omergeleft(par_dists,
            init_prev  = ss.bernoulli,
            death_prob = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self.death_prob_data = sc.dcp(self.pars.death_prob)
        self.pars.death_prob = self.make_death_prob

        return

    @staticmethod
    def make_death_prob(module, sim, uids):
        p = module.pars
        out = sim.dt * module.death_prob_data / (p.cd4_min - p.cd4_max)**2 *  (module.cd4[uids] - p.cd4_max)**2
        out = np.array(out)
        return out

    def update_pre(self, sim):
        """ Update CD4 """
        self.cd4[sim.people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[sim.people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[sim.people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[sim.people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[sim.people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.pars.death_prob.filter(can_die)
        
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
        super().update_results(sim)
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
        return self.set_prognoses(sim, target_uids, source_uids)


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, year: np.array, coverage: np.array, **kwargs):
        self.requires = HIV
        self.year = sc.promotetoarray(year)
        self.coverage = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.prob_art_at_infection = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage))
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
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
        self.results['n_art'][sim.ti] = np.count_nonzero(sim.people.alive & sim.people.hiv.on_art)

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
