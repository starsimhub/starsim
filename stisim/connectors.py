'''
Define connections between disease modules
'''

from . import utils as ssu
from . import modules as ssm

class Connectors(ssu.NDict):
    pass


class Connector(ssm.Module):
    def __init__(self, pars=None, *args, **kwargs):
        self.pars = ssu.omerge(pars)
        self.states = ssu.NDict()
        self.results = ssu.NDict()
        return