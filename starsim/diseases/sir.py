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
            dur_inf = 10,
            init_prev = 0.01,
            p_death = 0.1,
            beta = 0.1,
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

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        p = self.pars

        # Determine who dies and who recovers and when
        dead_uids = p.p_death.filter(uids)
        rec_uids = np.setdiff1d(uids, dead_uids)
        self.ti_dead[dead_uids] = sim.ti + p.dur_inf.rvs(dead_uids)
        self.ti_recovered[rec_uids] = sim.ti + p.dur_inf.rvs(rec_uids)

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return


# %% Interventions
__all__ += ['sir_vaccine']


class sir_vaccine(ss.vx):
    """
    Create a vaccine product that changes susceptible people to recovered (i.e., perfect immunity)
    """
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omerge({
            'efficacy': 0.9,
        }, pars)

        par_dists = ss.omerge({
            'efficacy': ss.bernoulli
        }, par_dists)

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        return

    def administer(self, people, uids):
        eff_vacc_uids = self.pars.efficacy.filter(uids)
        people.sir.susceptible[eff_vacc_uids] = False
        people.sir.recovered[eff_vacc_uids] = True
        return

