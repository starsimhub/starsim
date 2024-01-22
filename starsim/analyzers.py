"""
Define analyzers
"""

import starsim as ss


__all__ = ['Analyzer']


class Analyzer(ss.Module):

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def update_results(self, sim):
        raise NotImplementedError