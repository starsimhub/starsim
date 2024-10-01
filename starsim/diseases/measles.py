"""
Define measles model.
Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import starsim as ss
from starsim.diseases.sir import SIR

__all__ = ['Measles']


class Measles(SIR):

    def __init__(self, pars=None, *args, **kwargs):
        """ Initialize with parameters """
        super().__init__()
        self.define_pars(
            # Initial conditions and beta
            beta = 1.0, # Placeholder value
            init_prev = ss.bernoulli(p=0.005),

            # Natural history parameters, all specified in days
            dur_exp = ss.normal(loc=ss.days(8)),        # (days) - source: US CDC
            dur_inf = ss.normal(loc=ss.days(11)),       # (days) - source: US CDC
            p_death = ss.bernoulli(p=0.005), # Probability of death
        )
        self.update_pars(pars=pars, **kwargs)

        # SIR are added automatically, here we add E
        self.define_states(
            ss.State('exposed', label='Exposed'),
            ss.FloatArr('ti_exposed', label='Time of exposure'),
        )

        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    def step_state(self):
        # Progress exposed -> infected
        ti = self.ti
        infected = (self.exposed & (self.ti_infected <= ti)).uids
        self.exposed[infected] = False
        self.infected[infected] = True

        # Progress infected -> recovered
        recovered = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)
        return

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        super().set_prognoses(uids, source_uids)
        ti = self.ti

        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti

        p = self.pars

        # Determine when exposed become infected
        self.ti_infected[uids] = ti + p.dur_exp.rvs(uids)

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die]
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die]

        return

    def step_die(self, uids):
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'recovered']:
            self.statesdict[state][uids] = False
        return

