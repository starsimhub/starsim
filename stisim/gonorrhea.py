"""
Defne gonorrhea
"""

import numpy as np
import stisim as ss


__all__ = ['Gonorrhea']


class Gonorrhea(ss.Disease):

    def __init__(self, pars=None):
        super().__init__(pars)

        self.states = ss.ndict(
            ss.State('susceptible', bool, True),
            ss.State('infected', bool, False),
            ss.State('ti_infected', float, 0),
            ss.State('ti_recovered', float, 0),
            ss.State('ti_dead', float, np.nan), # Death due to gonorrhea
            self.states,
        )

        for state in self.states.values():
            self.__setattr__(state.name, state)

        self.pars = ss.omerge({
            'dur_inf': 3, # not modelling diagnosis or treatment explicitly here
            'p_death': 0.2,
            'initial': 3,
            'eff_condoms': 0.7,
        }, self.pars)
        return
    
    def update_states(self, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        gonorrhea_deaths = self.ti_dead <= sim.ti
        sim.people.alive[gonorrhea_deaths] = False
        sim.people.ti_dead[gonorrhea_deaths] = sim.ti
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
