"""
Base classes for diseases
"""

import numpy as np
import sciris as sc
import stisim as ss

__all__ = ['Disease', 'STI']

class Disease(ss.Module):
    """ Base module class for diseases """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = ss.ndict(type=ss.Result)

    @property
    def _boolean_states(self):
        """
        Iterator over states with boolean type

        For diseases, these states typically represent attributes like 'susceptible',
        'infectious', 'diagnosed' etc. These variables are typically useful to

        :return:
        """
        for state in self.states:
            if state.dtype == bool:
                yield state

    def initialize(self, sim):
        super().initialize(sim)
        self.validate_pars(sim)
        self.init_results(sim)
        self.set_initial_states(sim)
        return

    def finalize(self, sim):
        super().finalize(sim)
        self.finalize_results(sim)

    def validate_pars(self, sim):
        """
        Perform any parameter validation

        :return: None if parameters are all valid
        :raises: Exception if there are any invalid parameters (or if the initialization is otherwise invalid in some way)
        """
        pass

    def set_initial_states(self, sim):
        """
        Set initial values for states

        This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called. This method is about supplying initial values
        for the states (e.g., seeding initial infections)
        """
        pass

    def init_results(self, sim):
        """
        Initialize results

        By default, diseases all report on counts for any boolean states e.g., if
        a disease contains a boolean state 'susceptible' it will automatically contain a
        Result for 'n_susceptible'
        """
        for state in self._boolean_states:
            self.results += ss.Result(self.name, f'n_{state.name}', sim.npts, dtype=int)
        return

    def finalize_results(self, sim):
        """
        Finalize results
        """
        # TODO - will probably need to account for rescaling outputs for the default results here
        pass

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
        """
        pass

    def update_death(self, sim, uids):
        """
        Carry out state changes upon death

        This function is triggered after deaths are resolved, and before analyzers are run.
        See the SIR example model for a typical use case - deaths are requested as an autonomous
        update, to take effect after transmission on the same timestep. State changes that occur
        upon death (e.g., clearing an `infected` flag) are executed in this function. That also
        allows an intervention to avert a death scheduled on the same timestep, without having
        to undo any state changes that have already been applied (because they only run via this
        function if the death actually occurs).

        Depending on the module and the results it produces, it may or may not be necessary
        to implement this.

        :param sim:
        :param uids:
        :return:
        """
        pass

    def make_new_cases(self, sim):
        """
        Add new cases of the disease

        This method is agnostic as to the mechanism by which new cases occur. This
        could be through transmission (parametrized in different ways, which may or
        may not use the contact networks) or it may be based on risk factors/seeding,
        as may be the case for non-communicable diseases.
        """
        pass

    def set_prognoses(self, sim, uids):
        pass

    def update_results(self, sim):
        """
        Update results

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
        for state in self._boolean_states:
            self.results[f'n_{state.name}'][sim.ti] = np.count_nonzero(state & sim.people.alive)
        return



class STI(Disease):
    """
    Base class for STIs used in STIsim

    This class contains specializations for STI transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that STIsim connectors
    operate on to capture co-infection
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rel_sus = ss.State('rel_sus', float, 1)
        self.rel_sev = ss.State('rel_sev', float, 1)
        self.rel_trans = ss.State('rel_trans', float, 1)
        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', int, ss.INT_NAN)

        # Random number generators
        self.rng_init_cases = ss.RNG(f'initial_cases_{self.name}')
        self.rng_trans = ss.RNG(f'trans_{self.name}')
        self.rng_choose_infector = ss.RNG(f'choose_infector_{self.name}')
        return

    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
        super().validate_pars(sim)
        if 'beta' not in self.pars:
            self.pars.beta = sc.objdict({k: [1, 1] for k in sim.people.networks})
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        if self.pars['init_prev'] <= 0:
            return

        initial_cases = self.rng_init_cases.bernoulli_filter(ss.bernoulli(self.pars['init_prev']), uids=ss.true(sim.people.alive))

        self.set_prognoses(sim, initial_cases, from_uids=None)  # TODO: sentinel value to indicate seeds?
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
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
        """
        pass

    def _make_new_cases_singlerng(self, sim):
        # Not common-random-number-safe, but more efficient for when not using the multirng feature
        new_cases = []
        sources = []
        for k, layer in sim.people.networks.items():
            if k in self.pars['beta']:
                contacts = layer.contacts
                rel_trans = (self.infected & sim.people.alive) * self.rel_trans
                rel_sus = (self.susceptible & sim.people.alive) * self.rel_sus
                for a, b, beta in [[contacts.p1, contacts.p2, self.pars.beta[k][0]],
                                   [contacts.p2, contacts.p1, self.pars.beta[k][1]]]:
                    if beta == 0:
                        continue
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta  # TODO: Needs DT
                    # new_cases.append(ss.true(np.random.random(len(a)) < p_transmit))
                    new_cases_bool = np.random.random(len(a)) < p_transmit
                    new_cases.append(b[new_cases_bool])
                    sources.append(a[new_cases_bool])
        return np.concatenate(new_cases), np.concatenate(sources)

    def _choose_new_cases_multirng(self, people):
        '''
        Common-random-number-safe transmission code works by computing the
        probability of each _node_ acquiring a case rather than checking if each
        _edge_ transmits.
        '''
        n = len(people.uid)  # TODO: possibly could be shortened to just the people who are alive
        p_acq_node = np.zeros(n)

        for lkey, layer in people.networks.items():
            if lkey in self.pars['beta']:
                contacts = layer.contacts
                rel_trans = self.rel_trans * (self.infected & people.alive)
                rel_sus = self.rel_sus * (self.susceptible & people.alive)

                a_to_b = [contacts.p1, contacts.p2, self.pars.beta[lkey][0]]
                b_to_a = [contacts.p2, contacts.p1, self.pars.beta[lkey][1]]
                for a, b, beta in [a_to_b, b_to_a]:  # Transmission from a --> b
                    if beta == 0:
                        continue

                    # Accumulate acquisition probability for each person (node)
                    node_from_edge = np.ones((n, len(a)))
                    bi = people._uid_map[b]  # Indices of b (rather than uid)
                    node_from_edge[bi, np.arange(len(a))] = 1 - rel_trans[a] * rel_sus[b] * contacts.beta * beta  # TODO: Needs DT
                    p_acq_node = 1 - (1 - p_acq_node) * node_from_edge.prod(axis=1)  # (1-p1)*(1-p2)*...

        new_cases = self.rng_trans.bernoulli_filter(p_acq_node, people.uid)
        return new_cases

    def _determine_case_source_multirng(self, people, new_cases):
        '''
        Given the uids of new cases, determine which agents are the source of each case
        '''
        sources = np.zeros_like(new_cases)
        r = self.rng_choose_infector.random(new_cases)  # Random number slotted to each new case
        for i, uid in enumerate(new_cases):
            p_acqs = []
            possible_sources = []

            for lkey, layer in people.networks.items():
                if lkey in self.pars['beta']:
                    contacts = layer.contacts
                    a_to_b = [contacts.p1, contacts.p2, self.pars.beta[lkey][0]]
                    b_to_a = [contacts.p2, contacts.p1, self.pars.beta[lkey][1]]
                    for a, b, beta in [a_to_b, b_to_a]:  # Transmission from a --> b
                        if beta == 0:
                            continue

                        inds = np.where(b == uid)[0]
                        if len(inds) == 0:
                            continue
                        neighbors = a[inds]

                        rel_trans = self.rel_trans[neighbors] * (self.infected[neighbors] & people.alive[neighbors])
                        rel_sus = self.rel_sus[uid] * (self.susceptible[uid] & people.alive[uid])
                        beta_combined = contacts.beta[inds] * beta

                        # Compute acquisition probabilities from neighbors --> uid
                        # TODO: Could remove zeros
                        p_acqs.append((rel_trans * rel_sus * beta_combined).__array__())  # Needs DT
                        possible_sources.append(neighbors.__array__())

            # Concatenate across layers and directions (p1->p2 vs p2->p1)
            p_acqs = np.concatenate(p_acqs)
            possible_sources = np.concatenate(possible_sources)

            if len(possible_sources) == 1:  # Easy if only one possible source
                sources[i] = possible_sources[0]
            else:
                # Roulette selection using slotted draw r associated with this new case
                cumsum = p_acqs / p_acqs.sum()
                source_idx = np.argmax(cumsum >= r[i])
                sources[i] = possible_sources[source_idx]
        return sources

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """
        if not ss.options.multirng:
            # Determine new cases for singlerng
            new_cases, sources = self._make_new_cases_singlerng(sim)
        else:
            # Determine new cases for multirng
            new_cases = self._choose_new_cases_multirng(sim.people)
            if len(new_cases):
                # Now determine whom infected each case
                sources = self._determine_case_source_multirng(sim.people, new_cases)

        if len(new_cases):
            self.set_prognoses(sim, new_cases, sources)

        return len(new_cases)  # number of new cases made

    def set_prognoses(self, sim, uids, from_uids):
        pass

    def set_congenital(self, sim, uids):
        # Need to figure out whether we would have a methods like this here or make it
        # part of a pregnancy/STI connector
        pass

    def update_results(self, sim):
        super().update_results(sim)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
