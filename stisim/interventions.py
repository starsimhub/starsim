import numpy as np
import sciris as sc
from . import utils as ssu
from . import hiv as ssh


class Interventions(ssu.NDict):
    pass


class Intervention():

    requires = [] # Optionally list required modules here

    def initialize(self, sim):
        for req in self.requires:
            if req not in sim.modules:
                raise Exception(f'{self.__name__} requires module {req} but the Sim did not contain this module')

    def apply(self, sim):
        pass

    def finalize(self, sim):
        pass



class ART(Intervention):

    requires = [ssh.HIV]

    def __init__(self,t:np.array,capacity: np.array):
        self.t = sc.promotetoarray(t)
        self.capacity = sc.promotetoarray(capacity)

    def apply(self, sim):
        if sim.t < self.t[0]:
            return

        capacity = self.capacity[np.where(self.t <= sim.t)[0][-1]]
        on_art = sim.people.alive & sim.people.hiv.on_art

        n_change = capacity - np.count_nonzero(on_art)
        if n_change > 0:
            # Add more ART
            eligible = sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art
            n_eligible = np.count_nonzero(eligible)
            if n_eligible:
                inds = np.random.choice(ssu.true(eligible), min(n_eligible, n_change), replace=False)
                sim.people.hiv.on_art[inds] = True
        elif n_change < 0:
            # Take some people off ART
            eligible = sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art
            inds = np.random.choice(ssu.true(eligible), min(n_change), replace=False)
            sim.people.hiv.on_art[inds] = False
