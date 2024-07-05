"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['HIV', 'ART', 'CD4_analyzer']

class HIV(ss.Infection):

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(
            beta = 1.0, # Placeholder value
            cd4_min = 100,
            cd4_max = 500,
            cd4_rate = 5,
            eff_condoms = 0.7,
            art_efficacy = 0.96,
            init_prev = ss.bernoulli(p=0.05),
            death_dist = ss.bernoulli(p=self.death_prob_func), # Uses p_death by default, modulated by CD4
            p_death = 0.05,
        )
        self.update_pars(pars=pars, **kwargs)

        # States
        self.add_states(
            ss.BoolArr('on_art', label='On ART'),
            ss.FloatArr('ti_art', label='Time of ART initiation'),
            ss.FloatArr('ti_dead', label='Time of death'), # Time of HIV-caused death
            ss.FloatArr('cd4', default=500, label='CD4 count'),
        )
        return

    @staticmethod
    def death_prob_func(module, sim, uids):
        p = module.pars
        out = sim.dt * p.p_death / (p.cd4_min - p.cd4_max)**2 *  (module.cd4[uids] - p.cd4_max)**2
        out = np.array(out)
        return out

    def update_pre(self):
        """ Update CD4 """
        people = self.sim.people
        self.cd4[people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        can_die = people.hiv.infected.uids
        hiv_deaths = self.pars.death_dist.filter(can_die)
        
        people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = self.sim.ti
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.results += ss.Result(self.name, 'new_deaths', self.sim.npts, dtype=int, label='Deaths')
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        self.results['new_deaths'][ti] = np.count_nonzero(self.ti_dead == ti)
        return 

    def set_prognoses(self, uids, source_uids=None):
        super().set_prognoses(uids, source_uids)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.sim.ti
        return

    def set_congenital(self, uids, source_uids):
        return self.set_prognoses(uids, source_uids)


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, year: np.array, coverage: np.array, **kwargs):
        self.requires = HIV
        self.year = sc.promotetoarray(year)
        self.coverage = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.prob_art_at_infection = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage))
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        ti_delay = 1 # 1 time step delay TODO
        recently_infected = (sim.people.hiv.ti_infected == sim.ti-ti_delay).uids

        n_added = 0
        if len(recently_infected) > 0:
            inds = self.prob_art_at_infection.filter(recently_infected)
            sim.people.hiv.on_art[inds] = True
            sim.people.hiv.ti_art[inds] = sim.ti
            n_added = len(inds)

        # Add result
        self.results['n_art'][sim.ti] = np.count_nonzero(sim.people.hiv.on_art)

        return n_added


#%% Analyzers

class CD4_analyzer(ss.Analyzer):

    def __init__(self):
        self.requires = HIV
        self.cd4 = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.cd4 = np.zeros((sim.npts, sim.people.n), dtype=int)
        return

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
        return
