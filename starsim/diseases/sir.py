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
            'dur_inf': ss.lognorm,
            'init_prev': ss.bernoulli,
            'p_death': ss.bernoulli,
        })

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.recovered = ss.State('recovered', bool, False)
        self.t_infected = ss.State('t_infected', float, np.nan)
        self.t_recovered = ss.State('t_recovered', float, np.nan)
        self.t_dead = ss.State('t_dead', float, np.nan)

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
        recovered = ss.true(self.infected & (self.t_recovered <= sim.year))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ss.true(self.t_dead <= sim.year)
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

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        alive_uids = ss.true(sim.people.alive)
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
        return

    def update_results(self, sim):
        super().update_results(sim)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        return
