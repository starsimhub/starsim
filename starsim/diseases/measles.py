"""
Define measles model.
Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import numpy as np
import starsim as ss
from starsim.diseases.sir import SIR

__all__ = ['Measles']


class Measles(SIR):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """

        pars = ss.omergeleft(pars,
            # Natural history parameters, all specified in days
            dur_exp = 8,       # (days) - source: US CDC
            dur_inf = 11,      # (days) - source: US CDC
            p_death = 0.005,   # Probability of death

            # Initial conditions and beta
            init_prev = 0.005,
            beta = None,
        )

        par_dists = ss.omergeleft(par_dists,
            dur_exp   = ss.normal,
            dur_inf   = ss.normal,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        # SIR are added automatically, here we add E
        self.add_states(
            ss.State('exposed', bool, False),
            ss.State('ti_exposed', float, np.nan),
        )

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
        deaths = ss.true(self.ti_dead <= sim.ti)
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        # Do not call set_prognosis on parent
        # super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = sim.ti

        p = self.pars

        # Determine when exposed become infected
        self.ti_infected[uids] = sim.ti + p.dur_exp.rvs(uids) / sim.dt

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die] / sim.dt
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die] / sim.dt

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'recovered']:
            self.statesdict[state][uids] = False
        return

