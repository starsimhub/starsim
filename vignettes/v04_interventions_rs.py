"""
Vignette 04: Default interventions
The individual examples are quite different so these are being kept as separate files

Features of RS vignette:
    - separation between Products and Interventions (as in HPVsim)
"""

import pandas as pd

import stisim as ss
import numpy as np
import pandas as pd
import sciris as sc

# Default interventions
#   - screening / testing
#   - antibiotics
#   - ART
#   - vaccination
#   - custom
#   - prevention (condoms, PrEP, VMMC, etc)


#############
# SCREENING/TESTING
#############

# Example 1: a dual HIV/syphilis test, defined by using a combination of two tests
test_pos = pd.read_csv('products_dx.csv')
hiv_test = ss.Product(test_positivity=test_pos[test_pos.name == 'hiv_test'])
syph_test = ss.Product(test_positivity=test_pos[test_pos.name == 'syph_test'])

# Delivery via ANC and MSM outreach
anc_screening = ss.screening(
    eligibility='pregnant',  # Pass in people attribute or indices or function that generates indices
    products=[hiv_test, syph_test],  # Pass in ss.Product() or a string to reference a pre-defined one
    prob=0.3,  # 30% coverage of all pregnant women in each year from start_year
    start_year=2015,
)

msm_screening = ss.screening(
    eligibility='msm',  # I think this is common enough that we will want to be able to target it this way
    products=[hiv_test, syph_test],
    years=np.arange(2015, 2021),  # Run a 5-year-long program
    prob=np.linspace(0, 0.3, 6),  # Linear scale-up
)

# Make a sim
pars = dict(
    n_agents=100,
    birth_rate=0.028,  # Add birth rates so that we know to capture pregnancies
    death_rate=0.008,  # Add death rate
    networks=['mf', 'msm'],  # Add an MSM network - somehow this will also add MSM as a people attribute??
)
sim = ss.Sim(pars=pars, interventions=[anc_screening, msm_screening])

# Example 2: HIV tests are offered to people after they test positive for syphilis
syph_screening = ss.screening(products=syph_test, prob=0.3, start_year=2015)
to_hiv_test = lambda sim: sim.get_intervention('syph_screening').outcomes['positive']
hiv_testing = ss.screening(eligibility=to_hiv_test, products=hiv_test, prob=0.3, start_year=2015)
sim = ss.Sim(interventions=[syph_screening, hiv_testing], modules=[ss.syphilis(), ss.hiv()])

#############
# ANTIBIOTICS
#############
products_tx = pd.read_csv('products_tx.csv')
syph_tx = ss.Product(efficacy=products_tx[products_tx.name == 'syph_tx'])
syph_treatment = ss.treatment(eligibility='syph.diagnosed', products=syph_tx, prob=0.8, start_year=2020)
sim = ss.Sim(interventions=[syph_screening, syph_tx], modules=ss.syphilis())

# Cross-cutting antibiotics
antibiotics = ss.Product(efficacy=products_tx[products_tx.name == 'antibiotics'])
gon_test = ss.Product(test_positivity=test_pos[test_pos.name == 'gon_test'])
chlam_test = ss.Product(test_positivity=test_pos[test_pos.name == 'chlam_test'])

sti_screening = ss.screening(products=[gon_test, syph_test, chlam_test], prob=0.3, start_year=2015)
sti_treatment = ss.treatment(
    eligibility=['syph.diagnosed', 'gon.diagnosed', 'chlam.diagnosed'],
    products=antibiotics, prob=0.8, start_year=2020)
sim = ss.Sim(
    interventions=[sti_screening, sti_treatment],
    modules=[ss.syphilis(), ss.gonorrhea(), ss.chlamydia()]
)

#############
# ART
#############
# Super simple example
art = ss.treatment(eligibility='hiv.diagnosed', products='art', prob=0.8, start_year=2000)

# Example where data is provided with number of people on ART per year
art_coverage = pd.read_csv('art_coverage.csv')
art = ss.treatment(eligibility='hiv.diagnosed', products='art', data=art_coverage)
# Ideally I think this should try to match the numbers


#############
# Vaccination
#############
bexsero = ss.Product()  # define a vaccine product somehow
routine_vx = ss.routine_vx(age=[9,10], sex='female', product=bexsero, prob=0.2, start_year=2025)


#############
# Custom interventions
#############
# Realistically this might be a default intervention, but let's imagine it isn't.
class condoms(ss.Intervention):
    def __init__(self, coverage=None, efficacy=None):
        self.coverage=coverage  # Coverage by network?
        self.efficacy=efficacy  # Efficacy by disease??

    def initialize(self, sim):
        if sc.checktype(self.coverage,'num'):
            self.coverage = {nkey:self.coverage for nkey in sim.networks}
        if sc.checktype(self.efficacy, 'num'):
            self.efficacy = {mkey:self.efficacy for mkey in sim.modules}

    def apply(self, sim):
        for mkey, eff in self.efficacy.items():
            for nkey, cov in self.coverage.items():
                prev_beta = sim.pars[mkey]['beta'][nkey]
                sim.pars[mkey]['beta'][nkey] = prev_beta*(1-cov) + prev_beta*cov*(1-eff)


def quarantine_syph(sim):
    """ Hypotehtical example where people with syphilis are quarantined from HIV """
    syph = sim.people.syph.infected
    sim.people.hiv.rel_sus[syph] = 0.0


class caesarian(ss.Intervention):
    """
    Give women with gonorrhea caesarians to prevent transmission from the birth canal.
    Note, this is NOT a recommended intervention!!!
    """
    def __init__(self, start_year=None, *args, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        return

    def initialize(self, sim):
        super().initialize()
        self.caesareans = np.zeros(sim.npts)
        return

    def apply(self, sim):
        if sim.year >= self.start_year:
            sim.pars.gonorrhea.beta['maternal'] = [0, 0]
            self.caesareans[sim.ti] = np.count_nonzero((sim.people.ti_delivery==sim.ti) & sim.people.gonorrhea.infected)
        return
