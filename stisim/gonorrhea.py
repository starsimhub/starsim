"""
Defne gonorrhea
"""

import numpy as np
from . import utils as ssu
from . import people as ssp
from . import modules as ssm


class Gonorrhea(ssm.Disease):

    def __init__(self, pars=None):
        super().__init__(pars)
        self.states = ssu.NDict(
            ssp.State('susceptible', bool, True),
            ssp.State('infected', bool, False),
            ssp.State('ti_infected', float, 0),
            ssp.State('ti_recovered', float, 0),
            ssp.State('ti_dead', float, np.nan), # Death due to gonorrhea
            self.states,
        )

        self.pars = ssu.omerge({
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
        gonorrhea_deaths = sim.people.gonorrhea.ti_dead <= sim.ti
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
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.ti

        dur = sim.ti + np.random.poisson(sim.pars[self.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids)) < sim.pars[self.name].p_death
        sim.people[self.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[self.name].ti_dead[uids[dead]] = dur[dead]
        return