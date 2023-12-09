'''
Disease modules
'''

import numpy as np
import sciris as sc
import stisim as ss

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
        for req in self.requires:
            if req not in sim.modules:
                raise Exception(f'{self.name} (label={self.label}) requires module {req} but the Sim did not contain this module')
        return

    def initialize(self, sim):
        """
        Perform initialization steps

        This method is called once, as part of initializing a Sim

        :param sim:
        :return:
        """
        self.check_requires(sim)

        # Connect the states to the sim
        for state in self.states:
            state.initialize(sim.people)

        self.initialized = True
        return

    def finalize(self, sim):
        self.finalized = True
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

