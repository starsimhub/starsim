"""
Define example disease modules
"""

import numpy as np
import starsim as ss
<<<<<<< HEAD:starsim/diseases/examples.py
import sciris as sc
from .disease import Disease, HHT
import scipy.stats as sps


class SIR(HHT):
=======
import scipy.stats as sps


class SIR(ss.Infection):
>>>>>>> apis:starsim/diseases/sir.py
    """
    Example SIR model

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.

    Note that this class is not fully compatible with common random numbers.
    """

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omerge({
            'dur_inf': 1,
            'init_prev': 0.1,
            'p_death': 0.2,
            'beta': None,
        }, pars)

        par_dists = ss.omerge({
            'dur_inf': sps.lognorm,
            'init_prev': sps.bernoulli,
            'p_death': sps.bernoulli,
        }, par_dists)

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.recovered = ss.State('recovered', bool, False)
        self.ti_infected = ss.State('ti_infected', float, np.nan)
        self.ti_recovered = ss.State('ti_recovered', float, np.nan)
        self.ti_dead = ss.State('ti_dead', float, np.nan)

        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_infections', sim.npts, dtype=int)
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
        return len(deaths)

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        # This is an optional step. Implementing this function means that in `SIR.update_results()` the prevalence
        # calculation does not need to filter the infected agents by the alive agents. An alternative would be
        # to omit implementing this function, and instead filter by the alive agents when calculating prevalence
        super().update_death(sim, uids)
        self.infected[uids] = False
        self.recovered[uids] = False
        return

<<<<<<< HEAD:starsim/diseases/examples.py

class NCD(Disease):
    """
    Example non-communicable disease

    This class implements a basic NCD model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having the condition, and associated
    mortality.
    """
    def __init__(self, pars=None):
        default_pars = {
            'initial_risk': sps.bernoulli(p=0.3), # Initial prevalence of risk factors
            #'affection_rate': ss.rate(p=0.1), # Instantaneous rate of acquisition applied to those at risk (units are acquisitions / year)
            'dur_risk': sps.expon(scale=10),
            'prognosis': sps.weibull_min(c=2, scale=5), # Time in years between first becoming affected and death
        }

        super().__init__(ss.omerge(default_pars, pars))
        self.at_risk      = ss.State('at_risk', bool, False)
        self.affected     = ss.State('affected', bool, False)
        self.ti_affected  = ss.State('ti_affected', int, ss.INT_NAN)
        self.ti_dead      = ss.State('ti_dead', int, ss.INT_NAN)
        return

    @property
    def not_at_risk(self):
        return ~self.at_risk

=======
>>>>>>> apis:starsim/diseases/sir.py
    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        alive_uids = ss.true(sim.people.alive)
<<<<<<< HEAD:starsim/diseases/examples.py
        initial_risk = self.pars['initial_risk'].filter(alive_uids)
        self.at_risk[initial_risk] = True
        self.ti_affected[initial_risk] = sim.ti + sc.randround(self.pars['dur_risk'].rvs(initial_risk) / sim.dt)

        return initial_risk

    def update_pre(self, sim):
        deaths = ss.true(self.ti_dead == sim.ti)
        sim.people.request_death(deaths)
        self.log.add_data(deaths, died=True)
        self.results.new_deaths[sim.ti] = len(deaths) # Log deaths attributable to this module
        return

    def make_new_cases(self, sim):
        #atrisk_uids = ss.true(self.at_risk)
        #new_cases = self.pars['affection_rate'].filter(atrisk_uids)
        new_cases = ss.true(self.ti_affected == sim.ti)
        self.affected[new_cases] = True
        prog_years = self.pars['prognosis'].rvs(new_cases)
        self.ti_dead[new_cases] = sim.ti + sc.randround(prog_years / sim.dt)
        super().set_prognoses(sim, new_cases)
        return new_cases

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'n_not_at_risk', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
=======
        initial_cases = self.pars['init_prev'].filter(alive_uids)
        self.infect(sim, initial_cases, None)
        return

    def infect(self, sim, uids, from_uids):
        super().set_prognoses(sim, uids, from_uids)

        # Carry out state changes associated with infection
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.t_infected[uids] = sim.year

        # Calculate and schedule future outcomes for recovery/death
        dur_inf = self.pars['dur_inf'].rvs(uids)
        will_die = self.pars['p_death'].rvs(uids)
        self.t_recovered[uids[~will_die]] = sim.year + dur_inf[~will_die]
        self.t_dead[uids[will_die]] = sim.year + dur_inf[will_die]

        # Update result count of new infections - important to use += because
        # infect() may be called multiple times per timestep
        self.results['new_infections'][sim.ti] += len(uids)
        return

    def make_new_cases(self, sim): # TODO: Use function from STI
        for k, layer in sim.people.networks.items():
            if k in self.pars['beta']:
                contacts = layer.contacts
                rel_trans = (self.infected & sim.people.alive).astype(ss.dtypes.float)
                rel_sus = (self.susceptible & sim.people.alive).astype(ss.dtypes.float)
                for a, b, beta in [[contacts.p1, contacts.p2, self.pars.beta[k][0]],
                                    [contacts.p2, contacts.p1, self.pars.beta[k][1]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.contacts['beta'] * beta * sim.dt
                    new_cases = np.random.random(len(a)) < p_transmit # As this class is not common-random-number safe anyway, calling np.random is perfectly fine!
                    if new_cases.any():
                        self.infect(sim, b[new_cases], a[new_cases])
>>>>>>> apis:starsim/diseases/sir.py
        return

    def update_results(self, sim):
        super().update_results(sim)
<<<<<<< HEAD:starsim/diseases/examples.py
        self.results['n_not_at_risk'][sim.ti] = np.count_nonzero(self.not_at_risk & sim.people.alive)
        self.results['prevalence'][sim.ti] = np.count_nonzero(self.affected & sim.people.alive)/np.count_nonzero(sim.people.alive)
        self.results['new_deaths'][sim.ti] = np.count_nonzero(self.ti_dead == sim.ti)
=======
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
>>>>>>> apis:starsim/diseases/sir.py
        return
