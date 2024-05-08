"""
Experiment with connectors
"""

# %% Imports and settings
import starsim as ss
import pylab as pl
import sciris as sc

class hiv_syph(ss.Connector):
    """ Simple connector whereby rel_sus to NG doubles if CD4 count is <200"""
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, label='HIV-Syphilis', diseases=[ss.HIV, ss.Syphilis])
        self.pars = ss.dictmerge({
            'rel_trans_hiv': 20,
            'rel_trans_aids': 50,
            'rel_sus_hiv': 20,
            'rel_sus_aids': 50,
            'rel_trans_syph': 20,
            'rel_sus_syph': 20,
        }, self.pars)
        return

    def update(self, sim):
        """ Specify how HIV increases NG rel_sus and rel_trans """
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_hiv
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_aids
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_hiv
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_aids

        sim.diseases.hiv.rel_sus[sim.diseases.syphilis.active] = self.pars.rel_sus_syph
        sim.diseases.hiv.rel_trans[sim.diseases.syphilis.active] = self.pars.rel_trans_syph

        return


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

# Plotting
pl.figure()
pl.subplot(2,1,1)
pl.plot(sim_connect.yearvec, sim_connect.results.syphilis.n_infected, label='Accounting for relative risks')
pl.plot(sim_noconnect.yearvec, sim_noconnect.results.syphilis.n_infected, label='No relative risks')
pl.title('Syphilis infections')
pl.xlabel('Year')
pl.ylabel('Count')
pl.legend()

pl.subplot(2,1,2)
pl.plot(sim_connect.yearvec, sim_connect.results.hiv.n_infected, label='Accounting for relative risks')
pl.plot(sim_noconnect.yearvec, sim_noconnect.results.hiv.n_infected, label='No relative risks')
pl.title('HIV infections')
pl.xlabel('Year')
pl.ylabel('Count')

sc.figlayout()
pl.show()