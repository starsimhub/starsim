"""
Vignette 08: Custom results
"""

import stisim as sts
import starsim as ss
import matplotlib.pyplot as plt


##################
# @RomeshA
##################

# Scalar results

# How would you add an item to the simulations summary?
class MyModule(Module):
    def summary(self, sim):
        # Adds _extra_ summary items - augments/updates the auto summary computed by sim.compute_summary()
        s = super().summary(sim)
        s['new_quantity'] = <something>
        return s

# Results with dimension t

class MyModule(Module):
    def init_results(self, sim):
        Module.init_results(self, sim)
        sim.results[self.name]['n_art'] = sts.Result('n_art', self.name, sim.npts, dtype=int)

    def update_results(self, sim):
        sim.results[self.name]['n_art'] = np.count_nonzero(sim.people.alive & sim.people[self.name].on_art)

# OR

class MyAnalyzer(Analyzer):
    def initialize(self, sim):
        super().initialize()
        sim.results[self.name]['n_art'] = sts.Result('n_art', self.name, sim.npts, dtype=int)

    def update(self, sim):
        sim.results[self.name]['n_art'] = np.count_nonzero(sim.people.alive & sim.people[self.name].on_art)

# Results with other dimension e.g., age bins

class MyAnalyzer(Analyzer):
    def __init__(self, age_bin_width):
        self.bins = np.arange(0,120,age_bin_width)

    def initialize(self, sim):
        super().initialize()
        self.binned_result = np.zeros((sim.npts, len(self.bins)))

    def update(self, sim):
        self.binned_result[sim.ti,:] = np.histogram(sim.people.hiv.on_art, self.bins)

