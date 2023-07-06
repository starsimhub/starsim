from .modules import *


class Analyzer:
    requires = []

    # TODO - what if the analyzer depends on a specific variable? How does the analyzer know which modules are required?
    def initialize(self, sim):
        for req in self.requires:
            if req not in sim.modules:
                raise Exception(f'{self.__name__} requires module {req} but the Sim did not contain this module')
        pass

    def apply(self, sim):
        pass

    def finalize(self, sim):
        pass


class CD4_analyzer(Analyzer):
    requires = [HIV]

    def __init__(self):
        self.cd4 = None

    def initialize(self, sim):
        super().initialize(sim)
        self.cd4 = np.zeros((sim.npts, sim.people.n), dtype=int)

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
