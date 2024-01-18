"""
Define analyzers
"""

#from .modules import Module
from stisim.modules import Module

__all__ = ['Analyzer']


class Analyzer(Module):

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def update_results(self, sim):
        raise NotImplementedError