"""
Define analyzers
"""

from . import utils as ssu
from . import modules as ssm


class Analyzer(ssm.Module):
    pass


class Analyzers(ssu.ndict):
    def __init__(self, *args, type=Analyzer, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)