'''
Disease modules
'''

import numpy as np
import sciris as sc
import stisim as ss


__all__ = ['Module', 'Modules', 'Disease']


class Module(sc.prettyobj):
    
    def __init__(self, pars=None, label=None, requires=None, *args, **kwargs):
        self.pars = ss.omerge(pars)
        self.label = label if label else ''
        self.requires = sc.mergelists(requires)
        self.results = ss.Results()
        self.initialized = False
        self.finalized = False

        # Random number streams
        self.rng_init_cases         = ss.Stream('initial_cases')
        # The following random streams are dicts from layer key to rng
        self.rng_trans_ab           = {}
        self.rng_trans_ba           = {}
        self.rng_choose_infector_ab = {}
        self.rng_choose_infector_ba = {}

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

        # Random number streams per network layer
        for lkey in sim.people.networks.keys():
            self.rng_trans_ab[lkey]           = ss.Stream(f'trans_ab: {lkey}')
            self.rng_trans_ba[lkey]           = ss.Stream(f'trans_ba: {lkey}')
            self.rng_choose_infector_ab[lkey] = ss.Stream(f'choose_infector_ab: {lkey}')
            self.rng_choose_infector_ba[lkey] = ss.Stream(f'choose_infector_ba: {lkey}')

        # Connect the states to the people
        for state in self.states.values():
            state.initialize(sim.people)

        # Connect the streams to the sim
        for stream in self.streams.values():
            stream.initialize(sim.streams, sim.people._uid_map)

        self.initialized = True
        return

    def apply(self, sim):
        pass

    def finalize(self, sim):
        self.finalized = True

    @property
    def name(self):
        """ The module name is a lower-case version of its class name """
        return self.__class__.__name__.lower()

    @property
    def states(self):
        return ss.ndict({k:v for k,v in self.__dict__.items() if isinstance(v, ss.State)})

    @property
    def streams(self):
        return ss.ndict({k:v for k,v in self.__dict__.items() if isinstance(v, ss.Stream)})


class Modules(ss.ndict):
    def __init__(self, *args, type=Module, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)


class Disease(Module):
    """ Base module contains states/attributes that all modules have """
    
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(pars, *args, **kwargs)
        self.rel_sus = ss.State('rel_sus', float, 1)
        self.rel_sev = ss.State('rel_sev', float, 1)
        self.rel_trans = ss.State('rel_trans', float, 1)
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

        #initial_cases = np.random.choice(sim.people.uid, self.pars['initial'])
        #rng = sim.rngs.get(f'initial_cases_{self.name}')
        #initial_cases = ss.binomial_filter_rng(prob=self.pars['initial']/len(sim.people), arr=sim.people.uid, rng=rng, block_size=len(sim.people._uid_map))
        initial_cases = self.rng_init_cases.bernoulli_filter(prob=self.pars['initial']/len(sim.people), arr=sim.people.uid)

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

    def update(self, sim):
        """
        Perform all updates
        """
        self.update_states(sim)
        self.make_new_cases(sim)
        self.update_results(sim)
        return

    def update_states(self, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """
        n_new_cases = 0 # number of new cases made
        pars = sim.pars[self.name]
        for lkey, layer in sim.people.networks.items():
            if lkey in pars['beta']:
                rel_trans = self.rel_trans * (self.infected & sim.people.alive)
                rel_sus = self.rel_sus * (self.susceptible & sim.people.alive)

                a_to_b = [layer['p1'], layer['p2'], pars['beta'][lkey][0], self.rng_trans_ab[lkey], self.rng_choose_infector_ab[lkey]]
                b_to_a = [layer['p2'], layer['p1'], pars['beta'][lkey][1], self.rng_trans_ba[lkey], self.rng_choose_infector_ba[lkey]]
                for a, b, beta, rng_trans, rng_chs_inf in [a_to_b, b_to_a]:
                    if beta == 0:
                        continue
                    
                    # Check for new transmission from a --> b
                    # TODO: Will need to be more efficient here - can maintain edge to node matrix
                    node_from_edge = np.ones( (len(sim.people._uid_map), len(a)) )
                    node_from_edge[b, np.arange(len(b))] = 1 - rel_trans[a] * rel_sus[b] * layer['beta'] * beta
                    p_acq_node = 1 - node_from_edge.prod(axis=1) # 1 - (1-p1)*(1-p2)*...
                    new_cases_bool = rng_trans.bernoulli(p_acq_node[b], b)
                    new_cases = b[new_cases_bool]

                    if not len(new_cases):
                        continue

                    n_new_cases += len(new_cases)

                    # Decide whom the infection came from using one random number for each b (aligned by block size)
                    frm = np.zeros_like(new_cases)
                    r = rng_chs_inf.random(new_cases)
                    prob = (1-node_from_edge[new_cases])
                    cumsum = (prob / ((prob.sum(axis=1)[:,np.newaxis]))).cumsum(axis=1)
                    frm_idx = np.argmax( cumsum >= r[:,np.newaxis], axis=1)
                    frm = a[frm_idx]
                    self.set_prognoses(sim, new_cases, frm)

        return n_new_cases

    def set_prognoses(self, sim, to_uids, from_uids):
        pass

    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti]  = np.count_nonzero(self.susceptible)
        self.results['n_infected'][sim.ti]     = np.count_nonzero(self.infected)
        self.results['prevalence'][sim.ti]     = self.results.n_infected[sim.ti] / len(sim.people)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)

    def finalize_results(self, sim):
        pass