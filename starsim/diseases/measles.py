"""
Define measles model.
Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import numpy as np
import starsim as ss

__all__ = ['Measles']


class Measles(ss.diseases.SIR):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """

        pars = ss.omerge({
            # Natural history parameters, all specified in days
            'dur_exp': 8,       # (days) - source: US CDC
            'dur_inf': 11,      # (days) - source: US CDC
            'p_death': 0.005,   # Probability of death

            # Initial conditions and beta
            'init_prev': 0.005,
            'beta': None,
        }, pars)

        par_dists = ss.omerge({
            'dur_exp': ss.norm,
            'dur_inf': ss.norm,
            'init_prev': ss.bernoulli,
            'p_death': ss.bernoulli,
        }, par_dists)

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        # Boolean states
        # SIR are added automatically, here we add E
        self.exposed = ss.State('exposed', bool, False)

        # Timepoint states
        self.ti_exposed = ss.State('ti_exposed', float, np.nan)

        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    def update_pre(self, sim):

        # Progress exposed -> infected
        infected = ss.true(self.exposed & (self.ti_infected <= sim.ti))
        self.exposed[infected] = False
        self.infected[infected] = True

        # Progress infected -> recovered
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.year)
        if len(deaths):
            sim.people.request_death(deaths)

    def set_prognoses(self, sim, uids, from_uids=None):
        """ Set prognoses for those who get infected """
        super().set_prognoses(sim, uids, from_uids)

        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = sim.ti

        p = self.pars

        # Determine when exposed become infected
        self.ti_infected[uids] = sim.ti + p.dur_exp.rvs(uids)

        # Determine who dies and who recovers and when
        dead_uids = p.p_death.filter(uids)
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + p.dur_inf.rvs(dead_uids)
        rec_uids = np.setdiff1d(uids, dead_uids)
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + p.dur_inf.rvs(rec_uids)

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        self.susceptible[uids] = False
        self.exposed[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return

