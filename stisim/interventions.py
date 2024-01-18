"""
Define interventions
"""

from stisim.modules import Module


__all__ = ['Intervention']


class Intervention(Module):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, sim, *args, **kwargs):
        raise NotImplementedError