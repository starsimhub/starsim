"""
Experiment with connectors
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt

syph = ss.Syphilis()
syph.pars['beta'] = {'mf': [0.5, 0.3]}
syph.pars['init_prev'] = 0.05

ppl1 = ss.People(10000)
sim_nohiv = ss.Sim(people=ppl1, networks=ss.MFNet(), diseases=ng)
sim_nohiv.run()

hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.0008, 0.0004]}
hiv.pars['init_prev'] = 0.05
syph = ss.Syphilis()
syph.pars['beta'] = {'mf': [0.5, 0.3]}
syph.pars['init_prev'] = 0.05
ppl2 = ss.People(10000)
sim_hiv = ss.Sim(people=ppl2, networks=ss.MFNet(), diseases=[hiv, syph], connectors=ss.simple_hiv_ng())
sim_hiv.run()

plt.figure()
plt.plot(sim_hiv.yearvec, sim_hiv.results.gonorrhea.n_infected, label='With HIV')
plt.plot(sim_nohiv.yearvec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
plt.title('Gonorrhea infections')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()