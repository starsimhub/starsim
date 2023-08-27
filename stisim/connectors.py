'''
Define connections between disease modules
'''

import stisim as ss


__all__ = ['Connector', 'Connectors']


class Connector(ss.Module):
    def __init__(self, pars=None, *args, **kwargs):
        self.pars    = ss.omerge(pars)
        self.states  = ss.ndict()
        self.results = ss.ndict()
        return


class Connectors(ss.ndict):
    def __init__(self, *args, type=Connector, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)