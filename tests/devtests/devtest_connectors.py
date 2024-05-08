"""
Experiment with connectors
"""

# %% Imports and settings
import starsim as ss
import pylab as pl
import sciris as sc

class simple_hiv_syph(ss.Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, label='HIV-Gonorrhea', diseases=[ss.HIV, ss.Syphilis])
        self.pars = ss.dictmerge({
            'rel_trans_hiv': 20,
            'rel_trans_aids': 50,
            'rel_sus_hiv': 20,
            'rel_sus_aids': 50,
        }, self.pars)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids
        return


# Make HIV and syphilis with a connector, and run
hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.0008, 0.0004]}
hiv.pars['init_prev'] = ss.bernoulli(p=0.2)
syph = ss.Syphilis()
syph.pars['beta'] = {'mf': [0.1, 0.05]}
syph.pars['init_prev'] = ss.bernoulli(p=0.05)
ppl2 = ss.People(10000)
sim_hiv = ss.Sim(people=ppl2, networks=ss.MFNet(), diseases=[hiv, syph], connectors=simple_hiv_syph())
sim_hiv.run()

# Make syphilis sim
syph = ss.Syphilis()
syph.pars['beta'] = {'mf': [0.1, 0.05]}
syph.pars['init_prev'] = ss.bernoulli(p=0.05)

ppl1 = ss.People(10000)
sim_nohiv = ss.Sim(people=ppl1, networks=ss.MFNet(), diseases=syph)
sim_nohiv.run()

# Plotting
pl.figure()
pl.subplot(2,1,1)
pl.plot(sim_hiv.yearvec, sim_hiv.results.syphilis.n_infected, label='With HIV')
pl.plot(sim_nohiv.yearvec, sim_nohiv.results.syphilis.n_infected, label='Without HIV')
pl.title('Syphilis infections')
pl.xlabel('Year')
pl.ylabel('Count')
pl.legend()

pl.subplot(2,1,2)
pl.plot(sim_hiv.yearvec, sim_hiv.results.hiv.n_infected)
pl.title('HIV infections')
pl.xlabel('Year')
pl.ylabel('Count')

sc.figlayout()
plt.show()