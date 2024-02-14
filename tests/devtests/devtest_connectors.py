"""
Experiment with connectors
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt

ng = ss.Gonorrhea()
ng.pars['beta'] = {'mf': [0.05, 0.025]}
ng.pars['init_prev'] = 0.025

ppl1 = ss.People(10000)
sim_nohiv = ss.Sim(people=ppl1, networks=ss.mf(), diseases=ng)
sim_nohiv.run()

hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.0008, 0.0004]}
hiv.pars['init_prev'] = 0.05
ng = ss.Gonorrhea()
ng.pars['beta'] = {'mf': [0.05, 0.025]}
ng.pars['init_prev'] = 0.025
ppl2 = ss.People(10000)
sim_hiv = ss.Sim(people=ppl2, networks=ss.mf(), diseases=[hiv, ng], connectors=ss.simple_hiv_ng())
sim_hiv.run()

plt.figure()
plt.plot(sim_hiv.yearvec, sim_hiv.results.gonorrhea.n_infected, label='With HIV')
plt.plot(sim_nohiv.yearvec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
plt.title('Gonorrhea infections')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()