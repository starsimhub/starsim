'''
Disease modules
'''

import sciris as sc
import starsim as ss
from scipy.stats._distn_infrastructure import rv_frozen

__all__ = ['Module']


class Module(sc.prettyobj):

    def __init__(self, pars=None, name=None, label=None, requires=None, *args, **kwargs):
        self.pars = ss.omerge(pars)
        self.name = name if name else self.__class__.__name__.lower() # Default name is the class name
        self.label = label if label else ''
        self.requires = sc.mergelists(requires)
        self.results = ss.ndict(type=ss.Result)
        self.initialized = False
        self.finalized = False

        return

    def check_requires(self, sim):
        for req in sc.tolist(self.requires):
            if req not in [m.__class__ for m in sim.modules]:
                raise Exception(f'{self.name} (label={self.label}) requires module {req} but the Sim did not contain a module of this type.')
        return

    def initialize(self, sim):
        """
        Perform initialization steps

        This method is called once, as part of initializing a Sim

        :param sim:
        :return:
        """
        self.check_requires(sim)

        # Connect the random number generators to the sim. The RNGs for this module should be initialized
        # first as some of the module's State instances may require them to generate initial values
        for rng in self.rngs:
            if not rng.initialized :
                rng.initialize(sim.rng_container, sim.people.slot)

        # Initialize distributions in pars
        for key, value in self.pars.items():
            if isinstance(value, rv_frozen):
                self.pars[key] = ss.ScipyDistribution(value, f'{self.name}_{self.label}_{key}')
                self.pars[key].initialize(sim, self)
            #elif isinstance(value, ss.rate):
            #    self.pars[key].initialize(sim, f'{self.name}_{self.label}_{key}')

        for key, value in self.__dict__.items():
            if isinstance(value, rv_frozen):
                setattr(self, key, ss.ScipyDistribution(value, f'{self.name}_{self.label}_{key}'))
                getattr(self, key).initialize(sim, self)

        # Connect the states to the sim
        # Will use random numbers, so do after distribution initialization
        for state in self.states:
            state.initialize(sim)

        self.initialized = True
        return

    def finalize(self, sim):
        self.finalize_results(sim)
        self.finalized = True
        return

    def finalize_results(self, sim):
        """
        Finalize results
        """
        # Scale results
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result) and res.scale:
                self.results[reskey] = self.results[reskey]*sim.pars.pop_scale
        return

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

    @property
    def scipy_dbns(self):
        """
        Return a flat collection of all ScipyDistributions

        :return:
        """
        return [x for x in self.__dict__.values() if isinstance(x, ss.ScipyDistribution)] \
             + [x for x in self.pars.values()     if isinstance(x, ss.ScipyDistribution)]
