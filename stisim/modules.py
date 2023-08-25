'''
Disease modules
'''

import numpy as np
import sciris as sc
from . import people as ssp
from . import results as ssr
from . import utils as ssu


class Modules(ssu.NDict):
    pass



class Module(sc.prettyobj):
    
    def __init__(self, pars=None, requires=None, *args, **kwargs):
        self.pars = ssu.omerge(pars)
        self.requires = sc.mergelists(requires)
        self.states = ssu.NDict()
        self.results = ssr.Results()
        return
    
    def initialize(self, sim):
        pass
    
    def apply(self, sim):
        pass
    
    def finalize(self, sim):
        pass
    
    @property
    def name(self):
        """ The module name is a lower-case version of its class name """
        return self.__class__.__name__.lower()


class Disease(Module):
    """ Base module contains states/attributes that all modules have """
    
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(pars, *args, **kwargs)
        self.states = ssu.NDict(
            ssp.State('rel_sus', float, 1),
            ssp.State('rel_sev', float, 1),
            ssp.State('rel_trans', float, 1),
        )
        return


    def initialize(self, sim):
        # Merge parameters
        sim.pars[self.name] = self.pars
        sim.results[self.name] = self.results

        # Add this module to a People instance. This would always involve calling People.add_module
        # but subsequently modules could have their own logic for initializing the default values
        # and initializing any outputs that are required
        sim.people.add_module(self)

        # Initialization steps
        self.validate_pars(sim)
        self.init_states(sim)
        self.init_results(sim)


    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
        if 'beta' not in sim.pars[self.name]:
            sim.pars[self.name].beta = sc.objdict({k: [1, 1] for k in sim.people.networks})


    def init_states(self, sim):
        """
        Initialize states. This could involve passing in a full set of initial conditions, or using init_prev, or other
        """
        initial_cases = np.random.choice(sim.people.uid, sim.pars[self.name]['initial'])
        self.set_prognoses(sim, initial_cases)
        return


    def init_results(self, sim):
        """
        Initialize results. TODO, should these be stored in the module or just added directly to the sim?
        """
        self.results['n_susceptible']  = ssr.Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        self.results['n_infected']     = ssr.Result(self.name, 'n_infected', sim.npts, dtype=int)
        self.results['prevalence']     = ssr.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results['new_infections'] = ssr.Result(self.name, 'n_infected', sim.npts, dtype=int)
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
        pars = sim.pars[self.name]
        for k, layer in sim.people.networks.items():
            if k in pars['beta']:
                rel_trans = (sim.people[self.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[self.name].susceptible & sim.people.alive).astype(float)
                for a, b, beta in [[layer['p1'], layer['p2'], pars['beta'][k][0]], [layer['p2'], layer['p1'], pars['beta'][k][1]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer['beta'] * beta
                    new_cases = np.random.random(len(a)) < p_transmit
                    if new_cases.any():
                        self.set_prognoses(sim, b[new_cases])


    def set_prognoses(self, sim, uids):
        pass


    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti]  = np.count_nonzero(sim.people[self.name].susceptible)
        self.results['n_infected'][sim.ti]     = np.count_nonzero(sim.people[self.name].infected)
        self.results['prevalence'][sim.ti]     = sim.results[self.name].n_infected[sim.ti] / len(sim.people)
        self.results['new_infections'][sim.ti] = np.count_nonzero(sim.people[self.name].ti_infected == sim.ti)


    def finalize_results(self, sim):
        pass