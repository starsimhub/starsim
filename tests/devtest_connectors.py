"""
Experiment with connectors
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt

hiv = ss.HIV()
hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004]}
hiv.pars['init_prev'] = 0.8
ng = ss.Gonorrhea()
ng.pars['beta'] = {'simple_sexual': [0.08, 0.04]}


ppl1 = ss.People(10000)
ppl1.networks = ss.ndict(ss.simple_sexual())

sim_nohiv = ss.Sim(people=ppl1, modules=ng)
sim_nohiv.initialize()
# sim_nohiv.run()

ppl2 = ss.People(10000)
ppl2.networks = ss.ndict(ss.simple_sexual())
# sim_hiv = ss.Sim(people=ppl2, modules=[ss.Gonorrhea(), ss.HIV()])
sim_hiv = ss.Sim(people=ppl2, modules=[hiv, ng], connectors=ss.simple_hiv_ng(pars={'rel_trans_hiv':2, 'rel_sus_hiv':2, 'rel_sus_aids':10, 'rel_trans_aids':10}))
sim_hiv.initialize()
sim_hiv.run()

plt.figure()
plt.plot(sim_nohiv.tivec, sim_hiv.results.gonorrhea.n_infected, label='With HIV')
plt.plot(sim_nohiv.tivec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
plt.title('Gonorrhea infections')
plt.legend()
plt.show()
