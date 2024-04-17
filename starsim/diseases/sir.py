"""
Define SIR and SIS disease modules
"""

import numpy as np
import matplotlib.pyplot as pl
import starsim as ss

__all__ = ['SIR', 'SIS']

class SIR(ss.Infection):
    """
    Example SIR model

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.
    """

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omergeleft(pars,
            dur_inf = 6,
            init_prev = 0.01,
            p_death = 0.01,
            beta = 0.5,
        )

        par_dists = ss.omergeleft(par_dists,
            dur_inf   = ss.lognorm_ex,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            ss.State('recovered', bool, False),
            ss.State('ti_recovered', int, ss.INT_NAN),
            ss.State('ti_dead', int, ss.INT_NAN),
        )
        return

    def update_pre(self, sim):
        # Progress infectious -> recovered
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.ti)
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        p = self.pars

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_inf[will_die] / sim.dt # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = sim.ti + dur_inf[~will_die] / sim.dt

        return

    def update_death(self, sim, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return

    def plot(self):
        """ Default plot for SIR model """
        fig = pl.figure()
        for rkey in ['susceptible', 'infected', 'recovered']:
            pl.plot(self.results['n_'+rkey], label=rkey.title())
        pl.legend()
        return fig
    

class SIS(ss.Infection):
    """
    Example SIS model

    This class implements a basic SIS model with states for susceptible,
    infected/infectious, and back to susceptible based on waning immunity. There
    is no death in this case.
    """
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omergeleft(pars,
            dur_inf = 10,
            init_prev = 0.01,
            beta = 0.05,
            waning = 0.05,
            imm_boost = 1.0,
        )

        par_dists = ss.omergeleft(par_dists,
            dur_inf   = ss.lognorm_ex,
            init_prev = ss.bernoulli,
        )
        
        self.add_states(
            ss.State('ti_recovered', int, ss.INT_NAN),
            ss.State('immunity', float, 0.0),
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        return

    def update_pre(self, sim):
        """ Progress infectious -> recovered """
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity(sim)
        return
    
    def update_immunity(self, sim):
        uids = ss.true(self.immunity > 0)
        self.immunity[uids] = (self.immunity[uids])*(1 - self.pars.waning*sim.dt)
        self.rel_sus[uids] = np.maximum(0, 1 - self.immunity[uids])
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        self.immunity[uids] += self.pars.imm_boost

        # Sample duration of infection
        dur_inf = self.pars.dur_inf.rvs(uids)

        # Determine when people recover
        self.ti_recovered[uids] = sim.ti + dur_inf / sim.dt

        return
    
    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'rel_sus', sim.npts, dtype=float)
        return

    def update_results(self, sim):
        """ Store the population immunity (susceptibility) """
        super().update_results(sim)
        self.results['rel_sus'][sim.ti] = self.rel_sus.mean()
        return 

    def plot(self):
        """ Default plot for SIS model """
        fig = pl.figure()
        for rkey in ['susceptible', 'infected']:
            pl.plot(self.results['n_'+rkey], label=rkey.title())
        pl.legend()
        return fig


# %% Interventions
__all__ += ['sir_vaccine']


class sir_vaccine(ss.Vx):
    """
    Create a vaccine product that changes susceptible people to recovered (i.e., perfect immunity)
    """
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omerge({
            'efficacy': 0.9,
        }, pars)

        super().__init__(pars=pars, *args, **kwargs)

        return

    def administer(self, people, uids):
        people.sir.rel_sus[uids] *= 1-self.pars.efficacy
        return
