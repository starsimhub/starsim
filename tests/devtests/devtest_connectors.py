"""
Experiment with connectors
"""

# %% Imports and settings
import starsim as ss
import pylab as pl
import sciris as sc
import numpy as np


class hiv_syph(ss.Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Syphilis', requires=[ss.HIV, ss.Syphilis])
        self.default_pars(**{
            'rel_sus_syph_hiv': 2, # People with HIV are 2x more likely to acquire syphilis
            'rel_sus_syph_aids': 5, # People with AIDS are 5x more likely to acquire syphilis
            'rel_trans_syph_hiv': 1.5, # People with HIV are 1.5x more likely to transmit syphilis
            'rel_trans_syph_aids': 3, # People with AIDS are 3x more likely to transmit syphilis
            'rel_sus_hiv_syph': 2.7, # People with syphilis are 2.7x more likely to acquire HIV
            'rel_trans_hiv_syph': 2.7, # People with syphilis are 2.7x more likely to transmit HIV
        })
        self.update_pars(pars, **kwargs)
        return

    def update(self, sim):
        """ Specify HIV-syphilis interactions """
        # People with HIV are more likely to acquire syphilis
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_syph_hiv
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_syph_aids

        # People with HIV are more likely to transmit syphilis
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_syph_hiv
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_syph_aids

        # People with syphilis are more likely to acquire HIV
        sim.diseases.hiv.rel_sus[sim.diseases.syphilis.active] = self.pars.rel_sus_hiv_syph

        # People with syphilis are more likely to transmit HIV
        sim.diseases.hiv.rel_trans[sim.diseases.syphilis.active] = self.pars.rel_trans_hiv_syph

        return


class BPG(ss.Intervention):  # Create a BPG intervention

    def __init__(self, year=2020, prob=0.5):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of treatment
        self.year = year

    def apply(self, sim):
        if sim.year > self.year:
            syphilis = sim.diseases.syphilis

            # Define who is eligible for treatment
            eligible_ids = syphilis.infected.uids  # People are eligible for treatment if they have just started exhibiting symptoms
            n_eligible = len(eligible_ids) # Number of people who are eligible

            # Define who receives treatment
            is_treated = np.random.rand(n_eligible) < self.prob  # Define which of the n_eligible people get treated by comparing np.random.rand() to self.p
            treat_ids = eligible_ids[is_treated]  # Pull out the IDs for the people receiving the treatment
            syphilis.infected[treat_ids] = False
            syphilis.susceptible[treat_ids] = True
            sim.diseases.hiv.rel_sus[treat_ids] = 1
            sim.diseases.hiv.rel_trans[treat_ids] = 1


# Make people, HIV, syphilis, and network
def make_args():
    # Marital
    mf = ss.MFNet(
        pars = dict(
            duration = ss.lognorm_ex(mean=5, stdev=0.5),
        )
    )
    hiv = ss.HIV()
    hiv.pars['beta'] = {'mf': [0.0008, 0.0004]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.2)
    syph = ss.Syphilis()
    syph.pars['beta'] = {'mf': [0.1, 0.05]}
    syph.pars['init_prev'] = ss.bernoulli(p=0.05)
    ppl = ss.People(10000)

    return hiv, syph, mf, ppl

# Make a sim with a connector, and run
hiv, syph, mf, ppl = make_args()
sim_connect = ss.Sim(people=ppl, networks=mf, diseases=[hiv, syph], connectors=hiv_syph())
sim_connect.run()

# Make a sim without a connector, and run
hiv, syph, mf, ppl = make_args()
sim_noconnect = ss.Sim(people=ppl, networks=mf, diseases=[hiv, syph])
sim_noconnect.run()

# Make a sim with a connector and syph treatment, and run
hiv, syph, mf, ppl = make_args()
sim_treat = ss.Sim(people=ppl, networks=mf, diseases=[hiv, syph], connectors=hiv_syph(), interventions=BPG())
sim_treat.run()


# Plotting
pl.figure()

pl.subplot(2,1,1)
pl.plot(sim_treat.yearvec, sim_treat.results.syphilis.n_infected, label='With syphilis treatment')
pl.plot(sim_connect.yearvec, sim_connect.results.syphilis.n_infected, label='Baseline')
pl.title('Syphilis infections')
pl.xlabel('Year')
pl.ylabel('Count')
pl.axvline(2020)
pl.legend()

pl.subplot(2,1,2)
pl.plot(sim_treat.yearvec, sim_treat.results.hiv.n_infected, label='With syphilis treatment')
pl.plot(sim_connect.yearvec, sim_connect.results.hiv.n_infected, label='Baseline')
pl.title('HIV infections')
pl.xlabel('Year')
pl.ylabel('Count')
pl.axvline(2020)
pl.legend()

sc.figlayout()
pl.show()
