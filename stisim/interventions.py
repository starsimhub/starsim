import numpy as np
import sciris as sc
from . import utils as ssu
from . import modules as ssm
from . import hiv as ssh


class Interventions(ssu.NDict):
    pass


class Intervention(ssm.Module):
    pass


class ART(Intervention):

    def __init__(self,t:np.array,capacity: np.array):
        self.requires = ssh.HIV
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
        
        return
