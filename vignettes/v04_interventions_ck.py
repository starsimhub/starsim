"""
Vignette 04: Default interventions
The individual examples are quite different so these are being kept as separate files

Features of CK vignette:
    - interventions can be imported from other *sims
"""

import numpy as np
import sciris as sc
import stisim as ss
import hpvsim as hpv  # Eventually, we should have full compatibility between sims
import hivsim as hiv

# %% 1. Default interventions

pars = sc.objdict(
    start=1980,
    end=2040,
    dt=0.2,
)

routine_hpv_vx = hpv.routine_vx(
    prob=0.9,
    years=[2025, 2030],
    product='bivalent',
    age_range=[9, 10],
    label='routine vx'
)

art_start = 2000
art = ss.interventions.ART(
    start=art_start,
    coverage=np.linspace(0, 1, (pars.end - art_start) / pars.dt)
)

interventions = [
    routine_hpv_vx,
    art,
]

hpv = hpv.Sim()  # do we want to make this work?
hiv = hiv.Sim()

sim = ss.Sim(modules=[hpv, hiv], interventions=interventions)
sim.run()
sim.plot()


# %% 2. Custom intervention

class anc_visit(ss.Intervention):

    def __init__(self, start, end, freq=1.0, link=True):
        self.start = start
        self.end = end
        self.freq = freq
        self.link = link
        self.known_stis = ['hiv', 'gonorrhea', 'syphilis', 'chlamydia']
        self.present = []
        self.results = dict()
        return

    def initialize(self, sim):
        # Check that right modules are present
        if not 'hiv' in sim.modules:
            raise ValueError('Must have HIV to use an ANC visit intervention')
        for sti in self.known_stis:
            if sti in sim.modules:
                self.present.append(sti)
        return

    def apply(self, sim):
        if (self.start <= sim.t < self.end) and (not sim.year % self.freq):

            # Initialize results of agents who are positive
            pos = sc.objdict()

            # Then check other STIs
            for sti in self.present:
                pos[sti] = sim.modules[sti].test()

            # Optionally immediately link HIV cases to care
            if self.link:
                sim.hiv.link_to_care(pos.hiv)

            self.results[sim.year] = pos


# Run
anc = anc_visit(start=2010, end=2030)
sim = ss.Sim(modules=['hiv', 'hpv', 'gonorrhea'], interventions=anc)
sim.run()
sim.plot()

