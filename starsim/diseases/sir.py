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
        self.default_pars(
            beta = 0.1,
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=6),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.BoolArr('recovered', label='Recovered'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
        )
        return

    def update_pre(self):
        # Progress infectious -> recovered
        sim = self.sim
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses """
        super().set_prognoses(uids, source_uids)
        ti = self.sim.ti
        dt = self.sim.dt
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti

        p = self.pars

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = ti + dur_inf[will_die] / dt # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = ti + dur_inf[~will_die] / dt
        return

    def update_death(self, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return

    def plot(self):
        """ Default plot for SIR model """
        fig = pl.figure()
        for rkey in ['n_susceptible', 'n_infected', 'n_recovered']:
            pl.plot(self.results[rkey], label=self.results[rkey].label)
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
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=10),
            waning = 0.05,
            imm_boost = 1.0,
        )
        self.update_pars(pars=pars, *args, **kwargs)

        self.add_states(
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('immunity', default=0.0),
        )
        return

    def update_pre(self):
        """ Progress infectious -> recovered """
        recovered = (self.infected & (self.ti_recovered <= self.sim.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity()
        return
    
    def update_immunity(self):
        has_imm = (self.immunity > 0).uids
        self.immunity[has_imm] = (self.immunity[has_imm])*(1 - self.pars.waning*self.sim.dt)
        self.rel_sus[has_imm] = np.maximum(0, 1 - self.immunity[has_imm])
        return

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses """
        super().set_prognoses(uids, source_uids)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.sim.ti
        self.immunity[uids] += self.pars.imm_boost

        # Sample duration of infection
        dur_inf = self.pars.dur_inf.rvs(uids)

        # Determine when people recover
        self.ti_recovered[uids] = self.sim.ti + dur_inf / self.sim.dt

        return
    
    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.results += ss.Result(self.name, 'rel_sus', self.sim.npts, dtype=float, label='Relative susceptibility')
        return

    def update_results(self):
        """ Store the population immunity (susceptibility) """
        super().update_results()
        self.results['rel_sus'][self.sim.ti] = self.rel_sus.mean()
        return 

    def plot(self):
        """ Default plot for SIS model """
        fig = pl.figure()
        for rkey in ['n_susceptible', 'n_infected']:
            pl.plot(self.results[rkey], label=self.results[rkey].label)
        pl.legend()
        return fig


# %% Interventions

__all__ += ['sir_vaccine']

class sir_vaccine(ss.Vx):
    """
    Create a vaccine product that affects the probability of infection.
    
    The vaccine can be either "leaky", in which everyone who receives the vaccine 
    receives the same amount of protection (specified by the efficacy parameter) 
    each time they are exposed to an infection. The alternative (leaky=False) is
    that the efficacy is the probability that the vaccine "takes", in which case
    that person is 100% protected (and the remaining people are 0% protected).
    
    Args:
        efficacy (float): efficacy of the vaccine (0<=efficacy<=1)
        leaky (bool): see above
    """
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(
            efficacy = 0.9,
            leaky = True
        )
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):        
        if self.pars.leaky:
            people.sir.rel_sus[uids] *= 1-self.pars.efficacy
        else:
            people.sir.rel_sus[uids] *= np.random.binomial(1, 1-self.pars.efficacy, len(uids))
        return
