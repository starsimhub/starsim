"""
Experiment with connectors
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt

ng = ss.Gonorrhea()
ng.pars['beta'] = {'simple_sexual': [0.05, 0.025]}
ng.pars['init_prev'] = 0.025

ppl1 = ss.People(10000)
ppl1.networks = ss.ndict(ss.mf())
sim_nohiv = ss.Sim(people=ppl1, diseases=ng)
sim_nohiv.run()

hiv = ss.HIV()
hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004]}
hiv.pars['init_prev'] = 0.05
ng = ss.Gonorrhea()
ng.pars['beta'] = {'simple_sexual': [0.05, 0.025]}
ng.pars['init_prev'] = 0.025
ppl2 = ss.People(10000)
ppl2.networks = ss.ndict(ss.mf())
sim_hiv = ss.Sim(people=ppl2, diseases=[hiv, ng], connectors=ss.simple_hiv_ng())
sim_hiv.run()

plt.figure()
plt.plot(sim_hiv.tivec, sim_hiv.results.gonorrhea.n_infected, label='With HIV')
plt.plot(sim_nohiv.tivec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
plt.title('Gonorrhea infections')
plt.legend()
plt.show()
