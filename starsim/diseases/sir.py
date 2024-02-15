"""
Define example disease modules
"""

import numpy as np
import starsim as ss

__all__ = ['SIR']

class SIR(ss.Infection):
    """
    Example SIR model

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.
    """

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omergeleft(pars,
            dur_inf = 1,
            init_prev = 0.1,
            p_death = 0.2,
            beta = None,
        )

        par_dists = ss.omergeleft(par_dists,
            dur_inf   = ss.lognorm,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            ss.State('susceptible', bool, True),
            ss.State('infected', bool, False),
            ss.State('recovered', bool, False),
            ss.State('ti_infected', float, np.nan),
            ss.State('ti_recovered', float, np.nan),
            ss.State('ti_dead', float, np.nan),
        )
        return

    def update_pre(self, sim):
        # Progress infectious -> recovered
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.year))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.year)
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        self.infected[uids] = False
        self.recovered[uids] = False
        return
