"""
Define interventions
"""

import stisim as ss


__all__ = ['Intervention', 'Interventions']


class Intervention(ss.Module):
    pass


class Interventions(ss.ndict):
    def __init__(self, *args, type=Intervention, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)
