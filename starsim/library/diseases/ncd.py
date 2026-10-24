"""
An example non-communicable disease, written as code
"""
import numpy as np
import starsim as ss

_ = None # Sentinel for "not provided" default arguments


class NCD(ss.Disease):
    """
    Example non-communicable disease

    This class implements a basic NCD model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having the condition, and associated
    mortality. It is written as code (see `ss.Disease`), and is not transmitted.

    Args:
        initial_risk (float/`ss.bernoulli`): initial prevalence of risk factors
        dur_risk (float/`ss.dur`/`ss.Dist`): how long a person is at risk for
        prognosis (float/`ss.dur`/`ss.Dist`): time in years between first becoming affected and death
    """
    def __init__(self, pars=None, initial_risk=_, dur_risk=_, prognosis=_, **kwargs):
        super().__init__()
        self.define_pars(
            initial_risk = ss.bernoulli(p=0.3), # Initial prevalence of risk factors
            dur_risk = ss.expon(scale=ss.years(10)),
            prognosis = ss.weibull(c=2, scale=ss.years(5)), # Time between first becoming affected and death; c is the (dimensionless) shape parameter
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('at_risk', label='At risk'),
            ss.BoolState('affected', label='Affected'),
            ss.FloatArr('ti_affected', label='Time of becoming affected'),
            ss.FloatArr('ti_dead', label='Time of death'),
        )
        return

    @property
    def not_at_risk(self):
        """ Boolean array of agents not currently at risk. """
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
        self.ti_affected[initial_risk] = self.ti + self.pars['dur_risk'].rvs(initial_risk, round=True)
        return initial_risk

    def step_state(self):
        ti = self.ti
        deaths = (self.ti_dead <= ti).uids # <= since a prognosis that rounds to 0 is only caught on the next step
        self.sim.people.request_death(deaths, cause=self) # Counted once resolved: see update_results()
        if self.infection_log:
            self.infection_log.add_data(deaths, died=True)
        return

    def step(self):
        ti = self.ti
        new_cases = (self.ti_affected == ti).uids
        self.affected[new_cases] = True
        dur_prog = self.pars.prognosis.rvs(new_cases, round=True)
        self.ti_dead[new_cases] = ti + dur_prog
        if self.infection_log:
            self.infection_log.add_entries(new_cases, None, self.now)
        return new_cases

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        self.define_results(
            ss.Result('n_not_at_risk', dtype=int,   label='Not at risk'),
            ss.Result('prevalence',    dtype=float, label='Prevalence'),
            ss.Result('new_deaths',    dtype=int,   label='Deaths'),
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results.new_deaths[ti] = len(self.sim.people.died_of(self)) # Deaths attributable to this module in this step, once each has one cause
        self.results.n_not_at_risk[ti] = np.count_nonzero(self.not_at_risk)
        self.results.prevalence[ti]    = np.count_nonzero(self.affected)/self.sim.people.n_alive
        return
