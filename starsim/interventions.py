"""
Define interventions
"""

import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Intervention']


class Intervention(ss.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, sim, *args, **kwargs):
        raise NotImplementedError

    def finalize(self, sim):
        super().finalize(sim)
        self.finalize_results(sim)

