'''
Disease modules
'''

import numpy as np
import sciris as sc
import stisim as ss

__all__ = ['Module', 'Disease']


class Module(sc.prettyobj):

    def __init__(self, pars=None, label=None, requires=None, *args, **kwargs):

        self.pars = ss.omerge(pars)
        self.label = label if label else ''
        self.requires = sc.mergelists(requires)
        self.results = ss.ndict(type=ss.Result)
        self.initialized = False
        self.finalized = False

        return

    def __call__(self, *args, **kwargs):
        """ Makes module(sim) equivalent to module.apply(sim) """
        if not self.initialized:  # pragma: no cover
            errormsg = f'{self.name} (label={self.label}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)

    def check_requires(self, sim):
        for req in self.requires:
            if req not in sim.modules:
                raise Exception(f'{self.__name__} requires module {req} but the Sim did not contain this module')
        return

    def initialize(self, sim):
        self.check_requires(sim)

        # Connect the states to the sim
        for state in self.states:
            state.initialize(sim.people)

        # Connect the random number generators to the sim
        for rng in self.rngs:
            if not rng.initialized:
                rng.initialize(sim.rng_container, sim.people.slot)

        self.initialized = True
        return

    def finalize(self, sim):
        self.finalized = True
        return

    @property
    def name(self):
        """ The module name is a lower-case version of its class name """
        return self.__class__.__name__.lower()

    @property
    def states(self):
        """
        Return a flat collection of all states

        The base class returns all states that are contained in top-level attributes
        of the Module. If a Module stores states in a non-standard location (e.g.,
        within a list of states, or otherwise in some other nested structure - perhaps
        due to supporting features like multiple genotypes) then the Module should
        overload this attribute to ensure that all states appear in here.

        :return:
        """
        return [x for x in self.__dict__.values() if isinstance(x, ss.State)]

    @property
    def rngs(self):
        """
        Return a flat collection of all random number generators, as with states above

        :return:
        """
        return [x for x in self.__dict__.values() if isinstance(x, (ss.MultiRNG, ss.SingleRNG))]


class Disease(Module):
    """ Base module contains states/attributes that all modules have """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(pars, *args, **kwargs)
        self.rel_sus = ss.State('rel_sus', float, 1)
        self.rel_sev = ss.State('rel_sev', float, 1)
        self.rel_trans = ss.State('rel_trans', float, 1)
        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', int, ss.INT_NAN)

        # Random number generators
        self.rng_init_cases      = ss.RNG(f'initial_cases_{self.name}')
        self.rng_trans           = ss.RNG(f'trans_{self.name}')
        self.rng_choose_infector = ss.RNG(f'choose_infector_{self.name}')

        return

    def initialize(self, sim):
        super().initialize(sim)

        # Initialization steps
        self.validate_pars(sim)
        self.set_initial_states(sim)
        self.init_results(sim)
        return

    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
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

        self.set_prognoses(sim, initial_cases, from_uids=None) # TODO: sentinel value to indicate seeds?
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        self.results += ss.Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_infected', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_infections', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
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
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta # TODO: Needs DT
                    #new_cases.append(ss.true(np.random.random(len(a)) < p_transmit))
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
        n = len(people.uid) # TODO: possibly could be shortened to just the people who are alive
        p_acq_node = np.zeros(n)

        for lkey, layer in people.networks.items():
            if lkey in self.pars['beta']:
                contacts = layer.contacts
                rel_trans = self.rel_trans * (self.infected & people.alive)
                rel_sus = self.rel_sus * (self.susceptible & people.alive)

                a_to_b = [contacts.p1, contacts.p2, self.pars.beta[lkey][0]]
                b_to_a = [contacts.p2, contacts.p1, self.pars.beta[lkey][1]]
                for a, b, beta in [a_to_b, b_to_a]: # Transmission from a --> b
                    if beta == 0:
                        continue
                    
                    # Accumulate acquisition probability for each person (node)
                    node_from_edge = np.ones( (n, len(a)) )
                    bi = people._uid_map[b] # Indices of b (rather than uid)
                    node_from_edge[bi, np.arange(len(a))] = 1 - rel_trans[a] * rel_sus[b] * contacts.beta * beta # TODO: Needs DT
                    p_acq_node = 1 - (1-p_acq_node) * node_from_edge.prod(axis=1) # (1-p1)*(1-p2)*...

        new_cases = self.rng_trans.bernoulli_filter(p_acq_node, people.uid)
        return new_cases

    def _determine_case_source_multirng(self, people, new_cases):
        '''
        Given the uids of new cases, determine which agents are the source of each case
        '''
        sources = np.zeros_like(new_cases)
        r = self.rng_choose_infector.random(new_cases) # Random number slotted to each new case
        for i, uid in enumerate(new_cases):
            p_acqs = []
            possible_sources = []

            for lkey, layer in people.networks.items():
                if lkey in self.pars['beta']:
                    contacts = layer.contacts
                    a_to_b = [contacts.p1, contacts.p2, self.pars.beta[lkey][0]]
                    b_to_a = [contacts.p2, contacts.p1, self.pars.beta[lkey][1]]
                    for a, b, beta in [a_to_b, b_to_a]: # Transmission from a --> b
                        if beta == 0:
                            continue
                    
                        inds = np.where(b==uid)[0]
                        if len(inds) == 0:
                            continue
                        neighbors = a[inds]

                        rel_trans = self.rel_trans[neighbors] * (self.infected[neighbors] & people.alive[neighbors])
                        rel_sus = self.rel_sus[uid] * (self.susceptible[uid] & people.alive[uid])
                        beta_combined = contacts.beta[inds] * beta

                        # Compute acquisition probabilities from neighbors --> uid
                        # TODO: Could remove zeros
                        p_acqs.append((rel_trans * rel_sus * beta_combined).__array__()) # Needs DT
                        possible_sources.append(neighbors.__array__())

            # Concatenate across layers and directions (p1->p2 vs p2->p1)
            p_acqs = np.concatenate(p_acqs)
            possible_sources = np.concatenate(possible_sources)

            if len(possible_sources) == 1: # Easy if only one possible source
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
        
        return len(new_cases) # number of new cases made

    def set_prognoses(self, sim, uids, from_uids):
        pass

    def set_congenital(self, sim, uids):
        # Need to figure out whether we would have a methods like this here or make it
        # part of a pregnancy/STI connector
        pass

    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti]  = np.count_nonzero(self.susceptible & sim.people.alive)
        self.results['n_infected'][sim.ti]     = np.count_nonzero(self.infected & sim.people.alive)
        self.results['prevalence'][sim.ti]     = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)

    def finalize_results(self, sim):
        pass