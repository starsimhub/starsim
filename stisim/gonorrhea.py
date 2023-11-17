"""
Defne gonorrhea
"""

import numpy as np
import stisim as ss


__all__ = ['Gonorrhea']


class Gonorrhea(ss.Disease):

    def __init__(self, pars=None):
        super().__init__(pars)

        # States additional to the default disease states (see base class)
        self.symptomatic = ss.State('symptomatic', float, False)
        self.ti_clearance = ss.State('ti_clearance', int, ss.INT_NAN)
        self.p_symp = ss.State('p_symp', float, 1)

        self.rng_clear   = ss.RNG(f'clear_{self.name}')
        self.rng_symp    = ss.RNG(f'symp_{self.name}')
        self.rng_dur_inf = ss.RNG(f'dur_inf_{self.name}')

        # Parameters
        self.pars = ss.omerge({
            'dur_inf': 10/365,  # Median for those who spontaneously clear: https://sti.bmj.com/content/96/8/556
            'p_symp': 0.5,  # Share of infections that are symptomatic. Placeholder value
            'p_clear': 0.2,  # Share of infections that spontaneously clear: https://sti.bmj.com/content/96/8/556
            'init_prev': 0.03,
        }, self.pars)

        # Additional states dependent on parameter values, e.g. self.p_symp?
        # These might be useful for connectors to target, e.g. if HIV reduces p_clear

        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'n_sympotmatic', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_clearances', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        super(Gonorrhea, self).update_results(sim)
        self.results['n_sympotmatic'][sim.ti] = np.count_nonzero(self.symptomatic)
        self.results['new_clearances'][sim.ti] = np.count_nonzero(self.ti_clearance == sim.ti)
        return

    def update_states_pre(self, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        # Natural clearance
        clearances = self.ti_clearance <= sim.ti
        self.susceptible[clearances] = True
        self.infected[clearances] = False
        self.symptomatic[clearances] = False
        self.ti_clearance[clearances] = sim.ti

        return
    
    def make_new_cases(self, sim):
        super(Gonorrhea, self).make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids, from_uids=None):
        """
        Natural history of gonorrhea for adult infection
        """

        # Set infection status
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        # Set infection status
        symptomatic_uids = self.rng_symp.bernoulli(uids, prob=self.pars.p_symp)
        self.symptomatic[symptomatic_uids] = True

        # Set natural clearance
        clear_uids = self.rng_clear.bernoulli_filter(uids, prob=self.pars.p_clear)
        dur = sim.ti + self.rng_dur_inf.poisson(clear_uids, lam=self.pars['dur_inf']/sim.pars.dt)
        self.ti_clearance[clear_uids] = dur

        return

    def set_congenital(self, sim, uids):
        pass