"""
Define default rsv disease module
"""

import numpy as np
import sciris as sc
from sciris import randround as rr

import stisim as ss
from .disease import Disease


__all__ = ['RSV']



class RSV(Disease):

    def __init__(self, pars=None):
        super().__init__(pars)

        # General syphilis states

        # RSV states
        self.susceptible = ss.State('susceptible', bool, True)
        self.exposed = ss.State('exposed', bool, False)  # AKA incubating. Free of symptoms, not transmissible
        self.infected = ss.State('infected', bool, False)
        self.symptomatic = ss.State('symptomatic', bool, False)
        self.severe = ss.State('severe', bool, False)
        self.critical = ss.State('critical', bool, False)
        self.recovered = ss.State('recovered', bool, False)

        self.t_exposed = ss.State('t_exposed', float, np.nan)
        self.t_infected = ss.State('t_infected', float, np.nan)
        self.t_symptomatic = ss.State('t_symptomatic', float, np.nan)
        self.t_severe = ss.State('t_severe', float, np.nan)
        self.t_critical = ss.State('t_critical', float, np.nan)
        self.t_recovered = ss.State('t_recovered', float, np.nan)
        self.t_dead = ss.State('t_dead', float, np.nan)

        # Duration of stages
        self.dur_exposed = ss.State('dur_exposed', float, np.nan)
        self.dur_symptomatic = ss.State('dur_symptomatic', float, np.nan)
        self.dur_infection = ss.State('dur_infection', float, np.nan)  # Sum of all the stages

        # Timestep of state changes
        self.ti_exposed = ss.State('ti_exposed', int, ss.INT_NAN)
        self.ti_infected = ss.State('ti_infected', int, ss.INT_NAN)
        self.ti_symptomatic = ss.State('ti_symptomatic', int, ss.INT_NAN)
        self.ti_severe = ss.State('ti_severe', int, ss.INT_NAN)
        self.ti_critical = ss.State('ti_critical', int, ss.INT_NAN)
        self.ti_recovered = ss.State('ti_recovered', int, ss.INT_NAN)
        self.ti_dead = ss.State('ti_dead', int, ss.INT_NAN)

        # Parameters
        default_pars = dict(
            # RSV natural history
            dur_exposed=ss.lognormal(.05/12, 1/36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_symptomatic=ss.lognormal(1.5/12, 1/36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/

            p_symptom=0.2,
            p_critical=0.2,
            p_severe=0.2,

            # Initial conditions
            init_prev=0.03,
        )
        self.pars = ss.omerge(default_pars, self.pars)

        return

    @property
    def infectious(self):
        """ Infectious """
        return self.symptomatic

    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        n_init_cases = int(self.pars['init_prev'] * len(sim.people))
        initial_cases = np.random.choice(sim.people.uid, n_init_cases, replace=False)
        self.set_prognoses(sim, initial_cases)
        return

    def make_new_cases(self, sim):
        for k, layer in sim.people.networks.items():
            if k in self.pars['beta']:
                rel_trans = (self.infected & sim.people.alive).astype(float)
                rel_sus = (self.susceptible & sim.people.alive).astype(float)
                for a, b, beta in [[layer.contacts['p1'], layer.contacts['p2'], self.pars['beta'][k]],
                                   [layer.contacts['p2'], layer.contacts['p1'], self.pars['beta'][k]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.contacts['beta'] * beta * sim.dt
                    new_cases = np.random.random(len(a)) < p_transmit
                    if new_cases.any():
                        self.set_prognoses(sim, b[new_cases], source_uids=a[new_cases])
        return
    def update_results(self, sim):
        """ Update results """
        super().update_results(sim)
        return

    def update_pre(self, sim):
        """ Updates prior to interventions """

        # Symptomatic
        symptomatic = self.ti_symptomatic == sim.ti
        self.symptomatic[symptomatic] = True
        self.exposed[symptomatic] = False

        # Severe
        severe = self.ti_severe == sim.ti
        self.severe[severe] = True

        # Critical
        critical = self.ti_critical == sim.ti
        self.critical[critical] = True

        # Recovered
        recovered = self.ti_recovered == sim.ti
        self.recovered[recovered] = True
        self.exposed[recovered] = False
        self.infected[recovered] = False
        self.symptomatic[recovered] = False
        self.severe[recovered] = False
        return

    def record_exposure(self, sim, uids):
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.infected[uids] = True
        self.ti_exposed[uids] = sim.ti
        self.ti_infected[uids] = sim.ti

    def set_prognoses(self, sim, target_uids, source_uids=None):
        """
        Natural history of RSV
        """

        # Subset target_uids to only include ones who are infectious
        if source_uids is not None:
            infectious_sources = self.infectious[source_uids].values.nonzero()[-1]
            uids = target_uids[infectious_sources]
        else:
            uids = target_uids

        n_uids = len(uids)
        self.record_exposure(sim, uids)

        # Set future dates and probabilities

        # Determine which infections will become symptomatic
        symptomatic_uids = ss.binomial_filter(self.pars.p_symptom, uids)
        never_symptomatic_uids = np.setdiff1d(uids, symptomatic_uids)
        severe_uids = ss.binomial_filter(self.pars.p_severe, symptomatic_uids)
        never_severe_uids = np.setdiff1d(symptomatic_uids, severe_uids)
        critical_uids = ss.binomial_filter(self.pars.p_critical, severe_uids)

        # Exposed to symptomatic
        dur_exposed = self.pars.dur_exposed(n_uids)
        self.dur_exposed[uids] = dur_exposed
        self.ti_recovered[never_symptomatic_uids] = sim.ti + rr(dur_exposed/sim.dt) # TODO: Need to subset dur_exposed
        self.ti_symptomatic[symptomatic_uids] = sim.ti + rr(dur_exposed/sim.dt)
        self.dur_infection[uids] = dur_exposed

        dur_symptomatic = self.pars.dur_symptomatic(len(symptomatic_uids))
        self.ti_recovered[never_severe_uids] = sim.ti + rr(dur_symptomatic/sim.dt)
        self.dur_infection[symptomatic_uids] += dur_symptomatic


        return

