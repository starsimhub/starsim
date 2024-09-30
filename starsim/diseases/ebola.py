"""
Define Ebola model.
Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/ebola.py
Original version by@domdelport and @RomeshA
"""

import numpy as np
import starsim as ss
from starsim.diseases.sir import SIR

__all__ = ['Ebola']


class Ebola(SIR):

    def __init__(self, pars=None, *args, **kwargs):
        """ Initialize with parameters """
        super().__init__()
        self.define_pars(
            # Initial conditions and beta
            init_prev       = ss.bernoulli(p=0.005),
            beta            = ss.beta(1.0), # Placeholder value
            sev_factor      = 2.2,
            unburied_factor = 2.1,

            # Natural history parameters, all specified in days
            dur_exp2symp    = ss.lognorm_ex(mean=ss.days(12.7)), # Add source
            dur_symp2sev    = ss.lognorm_ex(mean=ss.days(6)), # Add source
            dur_sev2dead    = ss.lognorm_ex(mean=ss.days(1.5)), # Add source
            dur_dead2buried = ss.lognorm_ex(mean=ss.days(2)), # Add source
            dur_symp2rec    = ss.lognorm_ex(mean=ss.days(10)), # Add source
            dur_sev2rec     = ss.lognorm_ex(mean=ss.days(10.4)), # Add source
            p_sev           = ss.bernoulli(p=0.7), # Add source
            p_death         = ss.bernoulli(p=0.55), # Add source
            p_safe_bury     = ss.bernoulli(p=0.25), # Probability of a safe burial - should be linked to diagnoses
        )
        self.update_pars(pars=pars, **kwargs)

        # Boolean states
        self.define_states(
            # SIR are added automatically, here we add E
            ss.State('exposed', label='Exposed'),
            ss.State('severe', label='Severe'),
            ss.State('recovered', label='Recovered'),
            ss.State('buried', label='Buried'),

            # Timepoint states
            ss.FloatArr('ti_exposed', label='Time of exposure'),
            ss.FloatArr('ti_severe', label='Time of severe symptoms'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('ti_buried', label='Time of burial'),
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

        # Progress infectious -> severe
        severe = (self.infected & (self.ti_severe <= ti)).uids
        self.severe[severe] = True

        # Progress infected -> recovered
        recovered = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Progress severe -> recovered
        recovered_sev = (self.severe & (self.ti_recovered <= ti)).uids
        self.severe[recovered_sev] = False
        self.recovered[recovered_sev] = True

        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)

        # Progress dead -> buried
        buried = (self.ti_buried <= ti).uids
        self.buried[buried] = True

        return

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        # Do not call set_prognoses on the parent
        #super().set_prognoses(sim, uids, source_uids)

        ti = self.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti

        p = self.pars

        # Determine when exposed become infected
        self.ti_infected[uids] = ti + p.dur_exp2symp.rvs(uids)

        # Determine who progresses to sever and when
        sev_uids = p.p_sev.filter(uids)
        self.ti_severe[sev_uids] = self.ti_infected[sev_uids] + p.dur_symp2sev.rvs(sev_uids)

        # Determine who dies and who recovers and when
        dead_uids = p.p_death.filter(sev_uids)
        self.ti_dead[dead_uids] = self.ti_severe[dead_uids] + p.dur_sev2dead.rvs(dead_uids)
        rec_sev_uids = np.setdiff1d(sev_uids, dead_uids)
        self.ti_recovered[rec_sev_uids] = self.ti_severe[rec_sev_uids] + p.dur_sev2rec.rvs(rec_sev_uids)
        rec_symp_uids = np.setdiff1d(uids, sev_uids)
        self.ti_recovered[rec_symp_uids] = self.ti_infected[rec_symp_uids] + p.dur_symp2rec.rvs(rec_symp_uids)

        # Determine time of burial - either immediate (safe burials) or after a delay (unsafe)
        safe_buried = p.p_safe_bury.filter(dead_uids)
        unsafe_buried = np.setdiff1d(dead_uids, safe_buried)
        self.ti_buried[safe_buried] = self.ti_dead[safe_buried]
        self.ti_buried[unsafe_buried] = self.ti_dead[unsafe_buried] + p.dur_dead2buried.rvs(unsafe_buried)

        # Change rel_trans values
        self.rel_trans[self.infectious] = 1
        self.rel_trans[self.severe] = self.pars['sev_factor']  # Change for severe
        unburied_uids = ((self.ti_dead <= ti) & (self.ti_buried > ti)).uids
        self.rel_trans[unburied_uids] = self.pars['unburied_factor']  # Change for unburied
        return

    def step_die(self, uids):
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'severe', 'recovered']:
            self.statesdict[state][uids] = False
        return
