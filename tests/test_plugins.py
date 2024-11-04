"""
Test connectors and custom interventions
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss

sc.options(interactive=False) # Assume not running interactively


class hiv_syph(ss.Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            label = 'HIV-Syphilis',
            rel_sus_syph_hiv    = 2,   # People with HIV are 2x more likely to acquire syphilis
            rel_sus_syph_aids   = 5,   # People with AIDS are 5x more likely to acquire syphilis
            rel_trans_syph_hiv  = 1.5, # People with HIV are 1.5x more likely to transmit syphilis
            rel_trans_syph_aids = 3,   # People with AIDS are 3x more likely to transmit syphilis
            rel_sus_hiv_syph    = 2.7, # People with syphilis are 2.7x more likely to acquire HIV
            rel_trans_hiv_syph  = 2.7, # People with syphilis are 2.7x more likely to transmit HIV
        )
        self.update_pars(pars, **kwargs)
        return

    def step(self):
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


class Penicillin(ss.Intervention):
    """ Create a penicillin (BPG) intervention for treating syphilis """
    def __init__(self, year=2020, prob=0.8):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of treatment
        self.year = year
        return

    def step(self):
        sim = self.sim
        if sim.now > self.year:
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
        return


def make_args():
    """ Make people, HIV, syphilis, and network """
    pars = dict(n_agents=2000, verbose=0)
    mf = ss.MFNet(duration=ss.lognorm_ex(mean=5, std=0.5)) # TODO: think about whether these should be ss.dur(); currently they are not since stored in natural units with -self.dt
    hiv = ss.HIV(beta={'mf': [0.0008, 0.0004]}, init_prev=0.2) # TODO: beta should wrap the other way
    syph = ss.Syphilis(beta={'mf': [0.1, 0.05]}, init_prev=0.05)
    args = dict(pars=pars, networks=mf, diseases=[hiv, syph])
    return args


def test_connectors(do_plot=False):
    """ Test connector example """
    sc.heading('Testing connectors')

    # Make arguments
    args = make_args()
    sims = sc.objdict() # List of sims

    # Make a sim with a connector, and run
    sims.con = ss.Sim(label='With connector', connectors=hiv_syph(), **args)
    sims.con.run()

    # Make a sim without a connector, and run
    sims.nocon = ss.Sim(label='Without connector', **args)
    sims.nocon.run()

    # Make a sim with a connector and syph treatment, and run
    sims.treat = ss.Sim(label='With treatment', connectors=hiv_syph(), interventions=Penicillin(), **args)
    sims.treat.run()

    # Parse results
    results = sc.odict()
    diseases = ['syphilis', 'hiv']
    for sim in sims.values():
        results[sim.label] = sc.objdict()
        for disease in diseases:
            results[sim.label][disease] = sim.results[disease].n_infected

    # Plot
    if do_plot:
        plt.figure()

        plt.subplot(2,1,1)
        x = sims.con.timevec
        for label,res in results.items():
            plt.plot(x, res.syphilis, label=label)
        plt.title('Syphilis infections')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.axvline(2020)
        plt.legend()

        plt.subplot(2,1,2)
        for label,res in results.items():
            plt.plot(x, res.hiv, label=label)
        plt.title('HIV infections')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.axvline(2020)
        plt.legend()

        sc.figlayout()
        plt.show()

    # Check results
    for disease in diseases:
        assert results[0][disease].sum() > results[1][disease].sum(), f'{disease.title()} infections should be higher with connector'
        assert results[0][disease].sum() > results[2][disease].sum(), f'{disease.title()} infections should be lower with treatment'

    return sims


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    sims = test_connectors(do_plot=do_plot)

    T.toc()

