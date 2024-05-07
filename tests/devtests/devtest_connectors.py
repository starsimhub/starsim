"""
Experiment with connectors
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt

class simple_hiv_ng(Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Gonorrhea', requires=[ss.HIV, ss.Gonorrhea])
        self.default_pars(
            rel_trans_hiv  = 2,
            rel_trans_aids = 5,
            rel_sus_hiv    = 2,
            rel_sus_aids   = 5,
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """

        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.people.gonorrhea.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids

        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.people.gonorrhea.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids

        return

ng = ss.Gonorrhea()
ng.pars['beta'] = {'mf': [0.05, 0.025]}
ng.pars['init_prev'] = 0.025

ppl1 = ss.People(10000)
sim_nohiv = ss.Sim(people=ppl1, networks=ss.MFNet(), diseases=ng)
sim_nohiv.run()

hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.0008, 0.0004]}
hiv.pars['init_prev'] = 0.05
ng = ss.Gonorrhea()
ng.pars['beta'] = {'mf': [0.05, 0.025]}
ng.pars['init_prev'] = 0.025
ppl2 = ss.People(10000)
sim_hiv = ss.Sim(people=ppl2, networks=ss.MFNet(), diseases=[hiv, ng], connectors=simple_hiv_ng())
sim_hiv.run()

plt.figure()
plt.plot(sim_hiv.yearvec, sim_hiv.results.gonorrhea.n_infected, label='With HIV')
plt.plot(sim_nohiv.yearvec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
plt.title('Gonorrhea infections')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()