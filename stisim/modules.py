'''
Disease modules
'''

import numpy as np
import sciris as sc
import stisim as ss

__all__ = ['Module', 'Disease']


class Module(sc.prettyobj):

    def __init__(self, pars=None, label=None, requires=None, *args, **kwargs):

        self.pars = pars
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

        # Connect the streams to the sim
        for stream in self.streams:
            if not stream.initialized:
                stream.initialize(sim.streams, sim.people.slot)

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
    def streams(self):
        """
        Return a flat collection of all streams, as with states above

        :return:
        """
        return [x for x in self.__dict__.values() if isinstance(x, (ss.MultiStream, ss.CentralizedStream))]


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

        # Random number streams
        self.rng_init_cases      = ss.Stream(f'initial_cases_{self.name}')
        self.rng_trans           = ss.Stream(f'trans_{self.name}')
        self.rng_choose_infector = ss.Stream(f'choose_infector_{self.name}')

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

        initial_cases = self.rng_init_cases.bernoulli_filter(uids=ss.true(sim.people.alive), prob=self.pars['init_prev'])

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

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """
        pars = self.pars

        if not ss.options.multistream:
            # Not stream-safe, but more efficient for when not using multistream feature
            for k, layer in sim.people.networks.items():
                if k in pars['beta']:
                    contacts = layer.contacts
                    rel_trans = (self.infected & sim.people.alive) * self.rel_trans
                    rel_sus = (self.susceptible & sim.people.alive) * self.rel_sus
                    for a, b, beta in [[contacts.p1, contacts.p2, pars.beta[k][0]],
                                    [contacts.p2, contacts.p1, pars.beta[k][1]]]:
                        # probability of a->b transmission
                        p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta
                        new_cases = np.random.random(len(a)) < p_transmit
                        if np.any(new_cases):
                            self.set_prognoses(sim, b[new_cases])
            return len(new_cases)

        # Multistream-safe transmission code below here

        # Probability of each node acquiring a case
        n = len(sim.people.uid) # TODO: possibly could be shortened to just the people who are alive
        p_acq_node = np.zeros( n )

        for lkey, layer in sim.people.networks.items():
            if lkey in pars['beta']:
                contacts = layer.contacts
                rel_trans = self.rel_trans * (self.infected & sim.people.alive)
                rel_sus = self.rel_sus * (self.susceptible & sim.people.alive)

                a_to_b = [contacts.p1, contacts.p2, pars.beta[lkey][0]]
                b_to_a = [contacts.p2, contacts.p1, pars.beta[lkey][1]]
                for a, b, beta in [a_to_b, b_to_a]:
                    if beta == 0:
                        continue
                    
                    # Check for new transmission from a --> b
                    node_from_edge = np.ones( (n, len(a)) )
                    bi = sim.people._uid_map[b] # Indices of b (rather than uid)
                    p_not_acq = 1 - rel_trans[a] * rel_sus[b] * contacts.beta * beta # Needs DT

                    node_from_edge[bi, np.arange(len(a))] = p_not_acq
                    p_not_acq_by_node_this_layer_b_from_a = node_from_edge.prod(axis=1) # (1-p1)*(1-p2)*...
                    p_acq_node = 1 - (1-p_acq_node) * p_not_acq_by_node_this_layer_b_from_a

        new_cases_bool = self.rng_trans.bernoulli(sim.people.uid, prob=p_acq_node)
        new_cases = sim.people.uid[new_cases_bool]

        if not len(new_cases):
            return 0

        # Now determine who infected each case
        frm = np.zeros_like(new_cases)
        r = self.rng_choose_infector.random(new_cases)
        for i, uid in enumerate(new_cases):
            p_acqs = []
            sources = []

            for lkey, layer in sim.people.networks.items():
                if lkey in pars['beta']:
                    contacts = layer.contacts
                    a_to_b = [contacts.p1, contacts.p2, pars.beta[lkey][0]]
                    b_to_a = [contacts.p2, contacts.p1, pars.beta[lkey][1]]
                    for a, b, beta in [a_to_b, b_to_a]:
                        if beta == 0:
                            continue
                    
                        inds = np.where(b==uid)[0]
                        if len(inds) == 0:
                            continue
                        frms = a[inds]

                        # TODO: Likely no longer need alive here, at least not if dead people are removed
                        rel_trans = self.rel_trans[frms] * (self.infected[frms] & sim.people.alive[frms])
                        rel_sus = self.rel_sus[uid] * (self.susceptible[uid] & sim.people.alive[uid])
                        beta_combined = contacts.beta[inds] * beta

                        # Check for new transmission from a --> b
                        # TODO: Remove zeros from this...
                        p_acqs.append((rel_trans * rel_sus * beta_combined).__array__()) # Needs DT
                        sources.append(frms.__array__())

            p_acqs = np.concatenate(p_acqs)
            sources = np.concatenate(sources)

            if len(sources) == 1:
                frm[i] = sources[0]
            else:
                # Choose using draw r from above
                cumsum = p_acqs / p_acqs.sum()
                frm_idx = np.argmax( cumsum >= r[i])
                frm[i] = sources[frm_idx]

        self.set_prognoses(sim, new_cases, frm)
        
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