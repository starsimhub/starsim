"""
Define non-communicable disease (NCD) model
"""

import numpy as np
import starsim as ss
import sciris as sc


__all__ = ['NCD']

class NCD(ss.Disease):
    """
    Example non-communicable disease

    This class implements a basic NCD model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having the condition, and associated
    mortality.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            initial_risk = ss.bernoulli(p=0.3), # Initial prevalence of risk factors
            dur_risk = ss.expon(scale=10),
            prognosis = ss.weibull(c=2, scale=5), # Time in years between first becoming affected and death
        )
        self.update_pars(pars=pars, **kwargs)
        
        self.add_states(
            ss.BoolArr('at_risk', label='At risk'),
            ss.BoolArr('affected', label='Affected'),
            ss.FloatArr('ti_affected', label='Time of becoming affected'),
            ss.FloatArr('ti_dead', label='Time of death'),
        )
        return

    @property
    def not_at_risk(self):
        return ~self.at_risk

    def init_post(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        super().init_post()
        initial_risk = self.pars['initial_risk'].filter()
        self.at_risk[initial_risk] = True
        self.ti_affected[initial_risk] = self.sim.ti + sc.randround(self.pars['dur_risk'].rvs(initial_risk) / self.sim.dt)
        return initial_risk

    def update_pre(self):
        ti = self.sim.ti
        deaths = (self.ti_dead == ti).uids
        self.sim.people.request_death(deaths)
        self.log.add_data(deaths, died=True)
        self.results.new_deaths[ti] = len(deaths) # Log deaths attributable to this module
        return

    def make_new_cases(self):
        ti = self.sim.ti
        new_cases = (self.ti_affected == ti).uids
        self.affected[new_cases] = True
        prog_years = self.pars.prognosis.rvs(new_cases)
        self.ti_dead[new_cases] = ti + sc.randround(prog_years / self.sim.dt)
        super().set_prognoses(new_cases)
        return new_cases

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'n_not_at_risk', npts, dtype=int, label='Not at risk'),
            ss.Result(self.name, 'prevalence', npts, dtype=float, label='Prevalence'),
            ss.Result(self.name, 'new_deaths', npts, dtype=int, label='Deaths'),
        ]
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        self.results.n_not_at_risk[ti] = np.count_nonzero(self.not_at_risk)
        self.results.prevalence[ti]    = np.count_nonzero(self.affected)/len(self.sim.people)
        self.results.new_deaths[ti]    = np.count_nonzero(self.ti_dead == ti)
        return
