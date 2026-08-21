"""
HIV, with CD4 dynamics, plus an ART intervention and a CD4 analyzer.
"""

import numpy as np
import sciris as sc
import starsim as ss


class HIV(ss.Infection):
    """
    Simple HIV model with CD4 count dynamics and ART.

    Infected agents have a CD4 count that declines towards `cd4_min` while
    untreated and recovers towards `cd4_max` while on ART, at a rate set by
    `cd4_rate`. The per-timestep probability of death scales with how far the
    CD4 count has fallen, so agents with low CD4 counts die soonest. Agents on
    ART have their transmissibility reduced by `art_efficacy`. Vertical
    transmission is supported: `set_congenital()` infects the newborn.

    Use with `ART` to scale up treatment over time, and `CD4_analyzer` to record
    CD4 counts. Since `beta` defaults to 0, it must be set for transmission to
    occur.

    Args:
        beta (float):           per-contact transmission probability (0 by default)
        cd4_min (float):        CD4 count approached by untreated agents
        cd4_max (float):        CD4 count approached by agents on ART
        cd4_rate (float):       number of timesteps to close the CD4 gap
        eff_condoms (float):    efficacy of condoms (not currently used internally)
        art_efficacy (float):   proportional reduction in transmission on ART
        init_prev (Dist):       initial prevalence
        death_dist (Dist):      death probability, by default CD4-modulated `p_death`
        p_death (rate):         baseline death rate per unit time (not per infection)

    Attributes:
        on_art (BoolState):     currently on ART
        ti_art (FloatArr):      timestep of ART initiation
        ti_dead (FloatArr):     timestep of HIV-caused death
        cd4 (FloatArr):         current CD4 count (default 500)

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(
            diseases = ssl.HIV(beta=0.02, init_prev=0.05),
            networks = 'random',
            interventions = ssl.ART(year=[2000, 2020], coverage=[0, 0.8]),
        )
        sim.run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta = 0.0, # Placeholder value; replaced with a dict for acts
            cd4_min = 100,
            cd4_max = 500,
            cd4_rate = 5,
            eff_condoms = 0.7,
            art_efficacy = 0.96,
            init_prev = ss.bernoulli(p=0.05),
            death_dist = ss.bernoulli(p=self.death_prob_func), # Uses p_death by default, modulated by CD4
            p_death = ss.peryear(0.05), # NB: this is death per unit time, not death per infection
        )
        self.update_pars(pars, **kwargs)

        # States
        self.define_states(
            ss.BoolState('on_art', label='On ART'),
            ss.FloatArr('ti_art', label='Time of ART initiation'),
            ss.FloatArr('ti_dead', label='Time of death'), # Time of HIV-caused death
            ss.FloatArr('cd4', default=500, label='CD4 count'),
        )
        return

    @staticmethod
    def death_prob_func(module, sim, uids):
        p = module.pars
        dur = module.t.dt
        scale = (module.cd4[uids] - p.cd4_max)**2 / (p.cd4_min - p.cd4_max)**2 # Scale by cd4
        out = p.p_death.to_prob(dur, scale=scale)
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

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        return

    def set_congenital(self, uids, sources):
        return self.set_prognoses(uids, sources)


# %% HIV-related interventions

class ART(ss.Intervention):
    """
    Scale up antiretroviral therapy over time.

    Each timestep, agents infected `art_delay` ago are offered ART, and are
    treated with a probability interpolated from `coverage` at the corresponding
    `year`. Requires the `HIV` module.

    Args:
        year (float/array):     year(s) at which coverage is specified
        coverage (float/array): probability of ART initiation at each year
        art_delay (Dist):       par: delay from infection to ART eligibility

    Examples:
        ```python
        import starsim.library as ssl

        art = ssl.ART(year=[2000, 2010, 2020], coverage=[0, 0.4, 0.8])
        ```
    """
    def __init__(self, year, coverage, pars=None, **kwargs):
        self.requires = HIV
        self.year = sc.toarray(year)
        self.coverage = sc.toarray(coverage)
        super().__init__()
        self.define_pars(
            art_delay = ss.constant(v=ss.years(1.0)) # Value in years
        )
        self.update_pars(pars, **kwargs)

        prob_art = lambda self, sim, uids: np.interp(self.t.now('year'), self.year, self.coverage)
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
        ti = self.ti
        if self.t.now('year') < self.year[0]:
            return 0

        hiv = self.sim.diseases.hiv
        infected = hiv.infected.uids
        ti_delay = np.round(self.pars.art_delay.rvs(infected)).astype(int)
        recently_infected = infected[hiv.ti_infected[infected] == ti - ti_delay]

        n_added = 0
        if len(recently_infected) > 0:
            inds = self.prob_art_at_infection.filter(recently_infected)
            hiv.on_art[inds] = True
            hiv.ti_art[inds] = ti
            n_added = len(inds)

        # Add result
        self.results['n_art'][ti] = np.count_nonzero(hiv.on_art)

        return n_added


#%% Analyzers

class CD4_analyzer(ss.Analyzer):
    """
    Record the CD4 count of every agent at every timestep.

    Results are stored in `self.cd4`, a (timesteps × agents) array. Requires the
    `HIV` module. Note that this analyzer allocates a full dense array, so it is
    best suited to small simulations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires = HIV
        self.cd4 = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.cd4 = np.zeros((self.t.npts, sim.people.n_uids), dtype=int)
        return

    def step(self):
        cd4 = self.sim.diseases.hiv.cd4.raw
        n = min(self.cd4.shape[1], len(cd4)) # Truncate in case the population has grown
        self.cd4[self.ti, :n] = cd4[:n]
        return
