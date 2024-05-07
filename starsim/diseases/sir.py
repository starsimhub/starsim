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
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta = 0.5,
            init_prev = ss.bernoulli(0.01),
            dur_inf = ss.lognorm_ex(6),
            p_death = ss.bernoulli(0.01),
        )
        self.update_pars(pars, **kwargs)
        
        self.define_states(
            'susceptible',
            'infected',
            'recovered',
            'dead',
        )
        
        self.define_transitions(
            ss.Transition('susceptible -> infected', self.infect),
            ss.Transition('infected -> recovered', self.recover),
            ss.Transition('infected -> dead', self.die),
        )

        self.add_states(
            ss.BoolArr('recovered'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
        )
        return

    def update_pre(self, sim):
        # Progress infectious -> recovered
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids
        if len(deaths):
            sim.people.request_death(sim, deaths)
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
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(
            beta = 0.05,
            init_prev = ss.bernoulli(0.01),
            dur_inf = ss.lognorm_ex(10),
            waning = 0.05,
            imm_boost = 1.0,
        )
        self.update_pars(pars=pars, *args, **kwargs)

        self.add_states(
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('immunity', default=0.0),
        )
        return

    def update_pre(self, sim):
        """ Progress infectious -> recovered """
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity(sim)
        return
    
    def update_immunity(self, sim):
        has_imm = (self.immunity > 0).uids
        self.immunity[has_imm] = (self.immunity[has_imm])*(1 - self.pars.waning*sim.dt)
        self.rel_sus[has_imm] = np.maximum(0, 1 - self.immunity[has_imm])
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
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(efficacy=0.9)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        people.sir.rel_sus[uids] *= 1-self.pars.efficacy
        return
