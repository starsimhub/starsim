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
        n_init_cases = int(self.pars['init_prev'] * len(sim.people))
        initial_cases = np.random.choice(sim.people.uid, n_init_cases, replace=False)
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
        for k, layer in sim.people.networks.items():
            if k in pars['beta']:
                contacts = layer.contacts
                rel_trans = (self.infected & sim.people.alive).astype(float) * self.rel_trans
                rel_sus = (self.susceptible & sim.people.alive).astype(float) * self.rel_sus
                for a, b, beta in [[contacts.p1, contacts.p2, pars.beta[k][0]],
                                   [contacts.p2, contacts.p1, pars.beta[k][1]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta
                    new_cases = np.random.random(len(a)) < p_transmit
                    if np.any(new_cases):
                        self.set_prognoses(sim, b[new_cases])

    def set_prognoses(self, sim, uids):
        pass

    def set_congenital(self, sim, uids):
        # Need to figure out whether we would have a methods like this here or make it
        # part of a pregnancy/STI connector
        pass

    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti] = np.count_nonzero(self.susceptible)
        self.results['n_infected'][sim.ti] = np.count_nonzero(self.infected)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)

    def finalize_results(self, sim):
        pass
