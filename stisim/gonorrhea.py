"""
Defne gonorrhea
"""

import numpy as np
import stisim as ss


__all__ = ['Gonorrhea']


class Gonorrhea(ss.Disease):

    def __init__(self, pars=None):
        super().__init__(pars)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', float, np.nan)
        self.ti_recovered = ss.State('ti_recovered', float, np.nan)
        self.ti_dead = ss.State('ti_dead', int, ss.INT_NAN)  # Death due to gonorrhea

        self.pars = ss.omerge({
            'dur_inf': 3,  # not modelling diagnosis or treatment explicitly here
            'p_death': 0,
            'initial': 3,
            'eff_condoms': 0.7,
        }, self.pars)
        return
    
    def update_states(self, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something

        # Recovery
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.susceptible[recovered] = True

        # Schedule death for anyone that is due to die
        gonorrhea_deaths = ss.true(self.ti_dead <= sim.ti)
        sim.people.request_death(gonorrhea_deaths)

        return
    
    def update_results(self, sim):
        super(Gonorrhea, self).update_results(sim)
        return
    
    def make_new_cases(self, sim):
        super(Gonorrhea, self).make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids):
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        dur = sim.ti + np.random.poisson(self.pars['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids)) < self.pars.p_death
        self.ti_recovered[uids[~dead]] = dur[~dead]
        self.ti_dead[uids[dead]] = dur[dead]
        return
