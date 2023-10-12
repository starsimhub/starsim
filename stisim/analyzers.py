"""
Define analyzers
"""

import stisim as ss


__all__ = ['Analyzer']


class Analyzer(ss.Module):

    def update_results(self, sim):
        raise NotImplementedError

