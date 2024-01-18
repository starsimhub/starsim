"""
Define example disease modules
"""

import numpy as np
import stisim as ss
import sciris as sc
# from stisim.diseases.super.disease import Disease
import stisim.utils.ndict as ssu
import stisim.diseases.super.disease as sp
import stisim.results as ssr 
import stisim.states.states as sst
import stisim.utils.ndict as ssu

import scipy.stats as sps

class SIR(sp.Disease):
    """
    Example SIR model

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.

    Note that this class is not fully compatible with common random numbers.
    """

    def __init__(self, pars=None, *args, **kwargs):
        default_pars = {
            #'dur_inf': sps.weibull_min(c=lambda self, sim, uids: sim.people.age[uids], scale=10),#, seed='Duration of SIR Infection'),
            #'dur_inf': sps.norm(loc=lambda self, sim, uids: sim.people.age[uids], scale=2),
            'dur_inf': sps.lognorm(s=1, loc=10),
            'seed_infections': sps.bernoulli(p=0.1),
            'death_given_infection': sps.bernoulli(p=0.2),
            'beta': None,
        }

        super().__init__(pars=ssu.omerge(default_pars, pars), *args, **kwargs)

        self.susceptible = sst.State('susceptible', bool, True)
        self.infected = sst.State('infected', bool, False)
        self.recovered = sst.State('recovered', bool, False)
        self.t_infected = sst.State('t_infected', float, np.nan)
        self.t_recovered = sst.State('t_recovered', float, np.nan)
        self.t_dead = sst.State('t_dead', float, np.nan)

        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ssr.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ssr.Result(self.name, 'new_infections', sim.npts, dtype=int)
        return

    def update_pre(self, sim):
        # Progress infectious -> recovered
        recovered = ssu.true(self.infected & (self.t_recovered <= sim.year))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ssu.true(self.t_dead <= sim.year)
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

    def validate_pars(self, sim):
        if self.pars.beta is None:
            self.pars.beta = sc.objdict({k: 1 for k in sim.people.networks})
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        alive_uids = ssu.true(sim.people.alive)
        initial_cases = self.pars['seed_infections'].filter(alive_uids)
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
        will_die = self.pars['death_given_infection'].rvs(uids)
        self.t_recovered[uids[~will_die]] = sim.year + dur_inf[~will_die]
        self.t_dead[uids[will_die]] = sim.year + dur_inf[will_die]

        # Update result count of new infections - important to use += because
        # infect() may be called multiple times per timestep
        self.results['new_infections'][sim.ti] += len(uids)
        return

    def make_new_cases(self, sim): # TODO: Use function from STI
        for k, layer in sim.people.networks.items():
            if k in self.pars['beta']:
                rel_trans = (self.infected & sim.people.alive).astype(float)
                rel_sus = (self.susceptible & sim.people.alive).astype(float)
                for a, b, beta in [[layer.contacts['p1'], layer.contacts['p2'], self.pars['beta'][k]],
                                   [layer.contacts['p2'], layer.contacts['p1'], self.pars['beta'][k]]]:
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

