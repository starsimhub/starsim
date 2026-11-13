"""
HIV, with CD4 dynamics, plus an ART intervention and a CD4 analyzer.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['HIV', 'ART', 'ARTDrug', 'CD4_analyzer']


class HIV(ss.AgentDisease):
    """
    Simple HIV model with CD4 count dynamics and ART.

    Infected agents have a CD4 count that declines towards `cd4_min` while
    untreated and recovers towards `cd4_max` while on ART, at a rate set by
    `cd4_rate`. The per-timestep probability of death scales with how far the
    CD4 count has fallen, so agents with low CD4 counts die soonest. Agents on
    ART have their transmissibility reduced by `art_efficacy`. Vertical
    transmission is supported: a fetus infected over `ss.PrenatalNet` is infected as anyone else is.

    It is written as code, as agents (see `ss.AgentDisease`), which gives it its seeding, transmission, and the
    `infection` and `prevalence` results; it adds the CD4 count, ART, and death.

    Use with `ART` to scale up treatment over time, and `CD4_analyzer` to record
    CD4 counts. Since `beta` defaults to 0, it must be set for transmission to
    occur.

    Note: CRN safe.

    Args:
        beta (float):           per-contact transmission probability (0 by default)
        cd4_min (float):        CD4 count approached by untreated agents
        cd4_max (float):        CD4 count approached by agents on ART
        cd4_rate (float):       number of timesteps to close the CD4 gap
        art_efficacy (float):   proportional reduction in transmission on ART
        init_prev (Dist):       initial prevalence
        death_dist (Dist):      death probability, by default from `death_rate`, modulated by CD4
        death_rate (rate):      baseline rate of death of the infected, per unit time (not a probability per infection)

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
            interventions = ssl.ART(coverage=ss.timeseries({2000: 0, 2020: 0.8}, 'linear')),
        )
        sim.run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', list(kwargs) + list(pars or {}))
        super().__init__()
        self.define_pars(
            beta = 0.0, # Placeholder value: set it for transmission to occur
            cd4_min = 100,
            cd4_max = 500,
            cd4_rate = 5,
            art_efficacy = 0.96,
            init_prev = ss.bernoulli(p=0.05),
            death_dist = ss.bernoulli(p=self.death_prob_func), # Uses death_rate by default, modulated by CD4
            death_rate = ss.peryear(0.05), # A rate per unit time, not a probability per infection
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
        out = p.death_rate.to_prob(dur, scale=scale)
        return out

    def step_state(self):
        """ Update CD4 """
        people = self.sim.people
        self.cd4[people.alive & self.infected & self.on_art] += (self.pars.cd4_max - self.cd4[people.alive & self.infected & self.on_art])/self.pars.cd4_rate
        self.cd4[people.alive & self.infected & ~self.on_art] += (self.pars.cd4_min - self.cd4[people.alive & self.infected & ~self.on_art])/self.pars.cd4_rate

        self.rel_trans[people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        can_die = self.infected.uids
        hiv_deaths = self.pars.death_dist.filter(can_die)

        people.request_death(hiv_deaths, cause=self)
        self.ti_dead[hiv_deaths] = self.ti
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            ss.Result('death', dtype=int, label='Deaths', kind='flow')
        )
        return

    def record_events(self):
        super().record_events()
        ti = self.ti
        self.results.death[ti] = np.count_nonzero(self.ti_dead == ti) # Died in this step
        return


# %% HIV-related interventions

class ART(ss.Delivery):
    """
    Antiretroviral therapy, delivered as `ss.treat()` delivers a treatment

    By default, ART is offered once to each person who has been infected with `HIV` for `art_delay`
    and is not on it, with the probability `coverage`; `who=` restricts who among them is offered it,
    and `when=` when, as for any delivery (see `ss.Delivery`). The people treated are on ART for good:
    their CD4 count recovers, and they are less infectious (see `HIV`). Requires the `HIV` module.

    Note: CRN safe.

    Args:
        coverage (float/timeseries): the probability that each person eligible starts ART, e.g. scaled up over time with `ss.timeseries({2000: 0, 2020: 0.8}, 'linear')`
        who (Cond/func): who among the people eligible is offered it (e.g. `ss.person.female`)
        when (various): when it is offered, as for any delivery (default: on every step, once to each person when they become eligible)
        art_delay (dur): the time from infection to being eligible
        kwargs (dict): passed to `ss.Delivery` (e.g. `capacity=`, `name=`)

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        art = ssl.ART(coverage=ss.timeseries({2000: 0, 2010: 0.4, 2020: 0.8}, 'linear'))
        ```
    """
    def __init__(self, coverage=1.0, who=None, when=None, art_delay=ss.years(1), name=None, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', kwargs)
        self.art_delay = art_delay
        eligible = self.eligible if who is None else ss.conditions.Cond('&', who, self.eligible)
        super().__init__(ARTDrug(name='art_drug'), who=eligible, when=when, coverage=coverage, disease='hiv', name=name, **kwargs)
        return

    def eligible(self, sim):
        """ The people infected for at least art_delay who are not on ART """
        hiv = sim.diseases.hiv
        delay = self.art_delay/sim.t.dt if isinstance(self.art_delay, ss.Duration) else self.art_delay
        return hiv.infected & ~hiv.on_art & (sim.ti - hiv.ti_infected >= delay)

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('n_art', dtype=int, label='Number on ART', kind='stock'))
        return

    def record_states(self):
        super().record_states()
        self.results['n_art'][self.ti] = np.count_nonzero(self.sim.diseases.hiv.on_art)
        return


class ARTDrug(ss.Product):
    """ What `ART` delivers: the people given it are on ART from then on """
    def administer(self, uids):
        hiv = self.sim.diseases.hiv
        hiv.on_art[uids] = True
        hiv.ti_art[uids] = self.sim.ti
        return


#%% Analyzers

class CD4_analyzer(ss.Analyzer):
    """
    Record the CD4 count of every agent at every timestep.

    Results are stored in `self.cd4`, a (timesteps × agents) array. Requires the
    `HIV` module. Note that this analyzer allocates a full dense array, so it is
    best suited to small simulations.

    Note: CRN safe.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
