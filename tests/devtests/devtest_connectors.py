"""
Experiment with connectors; see tut_diseases.ipynb
"""

# %% Imports and settings
import sciris as sc
import starsim as ss
import pylab as pl

class simple_hiv_ng(ss.Connector):
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

# Make HIV
hiv = ss.HIV(
    beta = {'mf': [0.0008, 0.0004]},  # Specify transmissibility over the MF network
    init_prev = 0.05,
)

# Make gonorrhea
ng = ss.Gonorrhea(
    beta = {'mf': [0.05, 0.025]},  # Specify transmissibility over the MF network
    init_prev = 0.025,
)

# Make the sim, including a connector betweeh HIV and gonorrhea:
n_agents = 5_000
sim = ss.Sim(n_agents=n_agents, networks='mf', diseases=[hiv, ng], connectors=simple_hiv_ng())
sim.run()
sim_nohiv = ss.Sim(n_agents=n_agents, networks='mf', diseases=ng)
sim_nohiv.run()

pl.figure()
pl.plot(sim.yearvec, sim.results.gonorrhea.n_infected, label='With HIV')
pl.plot(sim_nohiv.yearvec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
pl.title('Gonorrhea infections')
pl.xlabel('Year')
pl.ylabel('Count')
pl.legend()
pl.show()