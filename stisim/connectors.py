'''
Define connections between disease modules
'''

from . import utils as ssu
from . import modules as ssm


class Connector(ssm.Module):
    def __init__(self, pars=None, *args, **kwargs):
        self.pars = ssu.omerge(pars)
        self.states = ssu.ndict()
        self.results = ssu.ndict()
        return


class Connectors(ssu.ndict):
    def __init__(self, *args, type=Connector, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)