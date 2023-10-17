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
        self.ti_infected = ss.State('ti_infected', float, np.nan)

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
        if self.pars['initial'] <= 0:
            return

        initial_cases = self.rng_init_cases.bernoulli_filter(size=sim.people.uid, prob=self.pars['initial']/len(sim.people))

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
        pars = sim.pars[self.name]

        # Probability of each node acquiring a case
        # TODO: Just people that who are alive?
        n = len(sim.people.uid)
        p_acq_node = np.zeros( n )
        node_from_node = np.ones( (n,n) ) 

        for lkey, layer in sim.people.networks.items():
            if lkey in pars['beta']:
                # TODO: Likely no longer need alive here, at least not if dead people are removed
                rel_trans = self.rel_trans * (self.infected & sim.people.alive)
                rel_sus = self.rel_sus * (self.susceptible & sim.people.alive)

                a_to_b = [layer['p1'], layer['p2'], pars['beta'][lkey][0]]
                b_to_a = [layer['p2'], layer['p1'], pars['beta'][lkey][1]]
                for a, b, beta in [a_to_b, b_to_a]:
                    if beta == 0:
                        continue
                    
                    # Check for new transmission from a --> b
                    # TODO: Will need to be more efficient here - can maintain edge to node matrix
                    node_from_edge = np.ones( (n, len(a)) )
                    ai = sim.people._uid_map[a] # Indices of a and b (rather than uid)
                    bi = sim.people._uid_map[b]
                    p_not_acq = 1 - rel_trans[a] * rel_sus[b] * layer['beta'] * beta # Needs DT

                    node_from_edge[bi, np.arange(len(a))] = p_not_acq
                    p_not_acq_by_node_this_layer_b_from_a = node_from_edge.prod(axis=1) # (1-p1)*(1-p2)*...
                    p_acq_node = 1 - (1-p_acq_node) * p_not_acq_by_node_this_layer_b_from_a

                    node_from_node_this_layer_b_from_a = np.ones( (n,n) ) 
                    node_from_node_this_layer_b_from_a[bi, ai] = p_not_acq
                    node_from_node *= node_from_node_this_layer_b_from_a

        new_cases_bool = self.rng_trans.bernoulli(size=sim.people.uid, prob=p_acq_node)
        new_cases = sim.people.uid[new_cases_bool]

        if not len(new_cases):
            return 0

        # Decide whom the infection came from using one random number for each b (aligned by block size)
        frm = np.zeros_like(new_cases)
        r = self.rng_choose_infector.random( new_cases )
        new_cases_idx = new_cases_bool.nonzero()[0]
        prob = (1-node_from_node[new_cases_idx]) # Prob of acquiring from each node | can constrain to just neighbors?
        cumsum = (prob / ((prob.sum(axis=1)[:,np.newaxis]))).cumsum(axis=1)
        frm_idx = np.argmax( cumsum >= r[:,np.newaxis], axis=1)
        frm = sim.people.uid[frm_idx]
        self.set_prognoses(sim, new_cases, frm)
        
        return len(new_cases) # number of new cases made


    def set_prognoses(self, sim, to_uids, from_uids):
        pass

    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti]  = np.count_nonzero(self.susceptible & sim.people.alive)
        self.results['n_infected'][sim.ti]     = np.count_nonzero(self.infected & sim.people.alive)
        self.results['prevalence'][sim.ti]     = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)

    def finalize_results(self, sim):
        pass