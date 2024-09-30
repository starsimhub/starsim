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
        self.define_pars(
            unit = 'year',
            beta = ss.beta(1.0), # Placeholder value
            cd4_min = 100,
            cd4_max = 500,
            cd4_rate = 5,
            eff_condoms = 0.7,
            art_efficacy = 0.96,
            init_prev = ss.bernoulli(p=0.05),
            death_dist = ss.bernoulli(p=self.death_prob_func), # Uses p_death by default, modulated by CD4
            p_death = ss.rate(0.05), # NB: this is death per unit time, not death per infection
        )
        self.update_pars(pars=pars, **kwargs)

        # States
        self.define_states(
            ss.State('on_art', label='On ART'),
            ss.FloatArr('ti_art', label='Time of ART initiation'),
            ss.FloatArr('ti_dead', label='Time of death'), # Time of HIV-caused death
            ss.FloatArr('cd4', default=500, label='CD4 count'),
        )
        return

    @staticmethod
    def death_prob_func(module, sim, uids):
        p = module.pars
        out = p.p_death / (p.cd4_min - p.cd4_max)**2 *  (module.cd4[uids] - p.cd4_max)**2
        out = np.array(out)
        return out

    def step_state(self):
        """ Update CD4 """
        people = self.sim.people
        self.cd4[people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        can_die = people.hiv.infected.uids
        hiv_deaths = self.pars.death_dist.filter(can_die)

        people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = self.ti
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            ss.Result('new_deaths', dtype=int, label='Deaths')
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results['new_deaths'][ti] = np.count_nonzero(self.ti_dead == ti)
        return

    def set_prognoses(self, uids, source_uids=None):
        super().set_prognoses(uids, source_uids)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        return

    def set_congenital(self, uids, source_uids):
        return self.set_prognoses(uids, source_uids)


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, year, coverage, pars=None, **kwargs):
        self.requires = HIV
        self.year = sc.toarray(year)
        self.coverage = sc.toarray(coverage)
        super().__init__()
        self.define_pars(
            art_delay = ss.constant(v=ss.years(1.0)) # Value in years
        )
        self.update_pars(pars=pars, **kwargs)

        prob_art = lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage)
        self.prob_art_at_infection = ss.bernoulli(p=prob_art)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.initialized = True
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('n_art', dtype=int, label='Number on ART'))
        return

    def step(self):
        sim = self.sim
        if sim.year < self.year[0]:
            return

        hiv = sim.people.hiv
        infected = hiv.infected.uids
        ti_delay = np.round(self.pars.art_delay.rvs(infected)).astype(int)
        recently_infected = infected[hiv.ti_infected[infected] == sim.ti-ti_delay]

        n_added = 0
        if len(recently_infected) > 0:
            inds = self.prob_art_at_infection.filter(recently_infected)
            hiv.on_art[inds] = True
            hiv.ti_art[inds] = sim.ti
            n_added = len(inds)

        # Add result
        self.results['n_art'][sim.ti] = np.count_nonzero(hiv.on_art)

        return n_added


#%% Analyzers

class CD4_analyzer(ss.Analyzer):

    def __init__(self):
        self.requires = HIV
        self.cd4 = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.cd4 = np.zeros((self.npts, sim.people.n), dtype=int)
        return

    def step(self):
        sim = self.sim
        self.cd4[sim.t] = sim.people.hiv.cd4
        return
