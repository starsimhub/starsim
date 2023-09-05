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
        self.rng_init_cases = ss.Stream('initial_cases')
        self.rng_trans_ab = ss.Stream('trans_ab')
        self.rng_trans_ba = ss.Stream('trans_ba')

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

        # Connect the states to the people
        for state in self.states.values():
            state.initialize(sim.people)

        # Connect the streams to the sim
        for stream in self.streams.values():
            stream.initialize(sim)

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

        self.set_prognoses(sim, initial_cases)
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        self.results += ss.Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_infected', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_infections', sim.npts, dtype=int)
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
        for k, layer in sim.people.networks.items():
            if k in pars['beta']:
                rel_trans = (self.infected & sim.people.alive).astype(float)
                rel_sus = (self.susceptible & sim.people.alive).astype(float)
                for a, b, beta, rng in [[layer['p1'], layer['p2'], pars['beta'][k][0], self.rng_trans_ab], [layer['p2'], layer['p1'], pars['beta'][k][1], self.rng_trans_ba]]:
                    # probability of a->b transmission
                    p_tran_edge = rel_trans[a] * rel_sus[b] * layer['beta'] * beta
                    
                    # Will need to be more efficient here - can maintain edge to node matrix
                    node_from_edge = np.zeros( (len(sim.people._uid_map), len(p_tran_edge)) )
                    node_from_edge[b, np.arange(len(b))] = 1
                    p_acq_node = np.dot(node_from_edge, p_tran_edge) # calculated for all nodes, including those outside b
                    new_cases = rng.bernoulli_filter(p_acq_node[b], b)

                    n_new_cases += len(new_cases)
                    if len(new_cases):
                        self.set_prognoses(sim, new_cases)
        return n_new_cases

    def set_prognoses(self, sim, uids):
        pass

    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti]  = np.count_nonzero(self.susceptible)
        self.results['n_infected'][sim.ti]     = np.count_nonzero(self.infected)
        self.results['prevalence'][sim.ti]     = self.results.n_infected[sim.ti] / len(sim.people)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)

    def finalize_results(self, sim):
        pass
