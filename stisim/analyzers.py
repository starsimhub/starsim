"""
Define analyzers
"""

import stisim as ss


__all__ = ['Analyzer', 'Analyzers']


class Analyzer(ss.Module):
    pass

class Analyzers(ss.ndict):
    def __init__(self, *args, type=Analyzer, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)