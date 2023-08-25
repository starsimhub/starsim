"""
Define analyzers
"""

import numpy as np
from . import utils as ssu
from . import modules as ssm
from . import hiv as ssh


class Analyzers(ssu.NDict):
    pass


class Analyzer(ssm.Module):
    pass


class CD4_analyzer(Analyzer):

    def __init__(self):
        self.requires = ssh.HIV
        self.cd4 = None
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.cd4 = np.zeros((sim.npts, sim.people.n), dtype=int)
        return

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
        return