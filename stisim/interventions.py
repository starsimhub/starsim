"""
Define interventions
"""

import stisim as ss


__all__ = ['Intervention']


class Intervention(ss.Module):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, sim, *args, **kwargs):
        raise NotImplementedError