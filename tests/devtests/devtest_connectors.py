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
        self.default_pars(
            rel_sus_syph_hiv    = 2,   # People with HIV are 2x more likely to acquire syphilis
            rel_sus_syph_aids   = 5,   # People with AIDS are 5x more likely to acquire syphilis
            rel_trans_syph_hiv  = 1.5, # People with HIV are 1.5x more likely to transmit syphilis
            rel_trans_syph_aids = 3,   # People with AIDS are 3x more likely to transmit syphilis
            rel_sus_hiv_syph    = 2.7, # People with syphilis are 2.7x more likely to acquire HIV
            rel_trans_hiv_syph  = 2.7, # People with syphilis are 2.7x more likely to transmit HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        """ Specify HIV-syphilis interactions """
        diseases = self.sim.diseases
        syph = diseases.syphilis
        hiv = diseases.hiv
        cd4 = self.sim.people.hiv.cd4
        
        # People with HIV are more likely to acquire syphilis
        syph.rel_sus[cd4 < 500] = self.pars.rel_sus_syph_hiv
        syph.rel_sus[cd4 < 200] = self.pars.rel_sus_syph_aids

        # People with HIV are more likely to transmit syphilis
        syph.rel_trans[cd4 < 500] = self.pars.rel_trans_syph_hiv
        syph.rel_trans[cd4 < 200] = self.pars.rel_trans_syph_aids

        # People with syphilis are more likely to acquire HIV
        hiv.rel_sus[syph.active] = self.pars.rel_sus_hiv_syph

        # People with syphilis are more likely to transmit HIV
        hiv.rel_trans[syph.active] = self.pars.rel_trans_hiv_syph
        return


class Penicillin(ss.Intervention):  # Create a penicillin (BPG) intervention

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
    pars = dict(n_agents=1000, verbose=0)
    mf = ss.MFNet(duration=ss.lognorm_ex(mean=5, stdev=0.5))
    hiv = ss.HIV(beta={'mf': [0.0008, 0.0004]}, init_prev=0.2)
    syph = ss.Syphilis(beta={'mf': [0.1, 0.05]}, init_prev=0.05)
    args = dict(pars=pars, networks=mf, diseases=[hiv, syph])
    return args


if __name__ == '__main__':
    
    # Make arguments
    args = make_args()
    
    # Make a sim with a connector, and run
    sim_connect = ss.Sim(label='With connector', connectors=hiv_syph(), **args)
    sim_connect.run()
    
    # Make a sim without a connector, and run
    sim_noconnect = ss.Sim(label='Without connector', **args)
    sim_noconnect.run()
    
    # Make a sim with a connector and syph treatment, and run
    sim_treat = ss.Sim(label='With treatment', connectors=hiv_syph(), interventions=Penicillin(), **args)
    sim_treat.run()
    
    # Or in parallel:
    # sim_connect, sim_noconnect, sim_treat = ss.parallel(sim_connect, sim_noconnect, sim_treat).sims
    
    
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
