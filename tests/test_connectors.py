"""
Test connectors and custom interventions

Demonstrates a connector mediating a bidirectional interaction between two diseases
(HIV and SIS), plus a custom intervention that treats one of them. HIV provides a
``cd4`` state (lower CD4 = more advanced disease); SIS provides a simple infected/
susceptible cycle that stands in for a co-infection.
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl

sc.options(interactive=False) # Assume not running interactively


class hiv_sis(ss.Connector):
    """ Connector for a bidirectional HIV<->SIS interaction """
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            label = 'HIV-SIS',
            rel_sus_sis_hiv    = 2,   # People with HIV are 2x more likely to acquire SIS
            rel_sus_sis_aids   = 5,   # People with AIDS (low CD4) are 5x more likely to acquire SIS
            rel_trans_sis_hiv  = 1.5, # People with HIV are 1.5x more likely to transmit SIS
            rel_trans_sis_aids = 3,   # People with AIDS are 3x more likely to transmit SIS
            rel_sus_hiv_sis    = 2.7, # People with SIS are 2.7x more likely to acquire HIV
            rel_trans_hiv_sis  = 2.7, # People with SIS are 2.7x more likely to transmit HIV
        )
        self.update_pars(**kwargs)
        return

    def step(self):
        """ Specify HIV-SIS interactions """

        diseases = self.sim.diseases
        sis = diseases.sis
        hiv = diseases.hiv
        cd4 = self.sim.people.hiv.cd4

        # People with HIV are more likely to acquire SIS
        sis.rel_sus[cd4 < 500] = self.pars.rel_sus_sis_hiv
        sis.rel_sus[cd4 < 200] = self.pars.rel_sus_sis_aids

        # People with HIV are more likely to transmit SIS
        sis.rel_trans[cd4 < 500] = self.pars.rel_trans_sis_hiv
        sis.rel_trans[cd4 < 200] = self.pars.rel_trans_sis_aids

        # People with SIS are more likely to acquire HIV
        hiv.rel_sus[sis.infected] = self.pars.rel_sus_hiv_sis

        # People with SIS are more likely to transmit HIV
        hiv.rel_trans[sis.infected] = self.pars.rel_trans_hiv_sis
        return


class TreatSIS(ss.Intervention):
    """ Treat (cure) SIS-infected people from a given year onwards """
    def __init__(self, year=2020, prob=0.8):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of treatment
        self.year = ss.date(year)
        return

    def step(self):
        sim = self.sim
        if sim.now > self.year:
            sis = sim.diseases.sis

            # Define who is eligible for treatment
            eligible_ids = sis.infected.uids
            n_eligible = len(eligible_ids) # Number of people who are eligible

            # Define who receives treatment
            is_treated = np.random.rand(n_eligible) < self.prob  # Compare np.random.rand() to self.prob
            treat_ids = eligible_ids[is_treated]  # Pull out the IDs for the people receiving the treatment
            sis.infected[treat_ids] = False
            sis.susceptible[treat_ids] = True
            sim.diseases.hiv.rel_sus[treat_ids] = 1
            sim.diseases.hiv.rel_trans[treat_ids] = 1
        return


def make_args():
    """ Make people, HIV, SIS, and network """
    pars = dict(n_agents=2000, verbose=0)
    mf = ss.MFNet(duration=ss.lognorm_ex(mean=5, std=0.5, unit=ss.years)) # TODO: think about whether these should be ss.dur(); currently they are not since stored in natural units with -self.dt
    hiv = ssl.diseases.HIV(beta={'mf': [0.0008, 0.0004]}, init_prev=0.2) # TODO: beta should wrap the other way
    sis = ss.SIS(beta={'mf': [0.1, 0.05]}, init_prev=0.05)
    args = dict(pars=pars, networks=mf, diseases=[hiv, sis])
    return args


@sc.timer()
def test_connectors(do_plot=False):
    """ Test connector example """
    sc.heading('Testing connectors')

    # Make arguments
    args = make_args()
    sims = sc.objdict() # List of sims

    # Make a sim with a connector, and run
    sims.con = ss.Sim(label='With connector', connectors=hiv_sis(), **args)
    sims.con.run()

    # Make a sim without a connector, and run
    sims.nocon = ss.Sim(label='Without connector', **args)
    sims.nocon.run()

    # Make a sim with a connector and SIS treatment, and run
    sims.treat = ss.Sim(label='With treatment', connectors=hiv_sis(), interventions=TreatSIS(), **args)
    sims.treat.run()

    # Parse results
    results = sc.odict()
    diseases = ['sis', 'hiv']
    for sim in sims.values():
        results[sim.label] = sc.objdict()
        for disease in diseases:
            results[sim.label][disease] = sim.results[disease].n_infected

    # Plot
    if do_plot:
        plt.figure()

        plt.subplot(2,1,1)
        x = sims.con.t.yearvec
        for label,res in results.items():
            plt.plot(x, res.sis, label=label)
        plt.title('SIS infections')
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
