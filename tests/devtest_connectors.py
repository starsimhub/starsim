"""
Experiment with connectors
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt


ppl1 = ss.People(10000)
ppl1.networks = ss.ndict(ss.simple_sexual())

sim_nohiv = ss.Sim(people=ppl1, modules=ss.Gonorrhea())
sim_nohiv.initialize()
sim_nohiv.run()

ppl2 = ss.People(10000)
ppl2.networks = ss.ndict(ss.simple_sexual())
sim_hiv = ss.Sim(people=ppl2, modules=[ss.HIV(), ss.Gonorrhea()], connectors=ss.simple_hiv_ng())
sim_hiv.initialize()
sim_hiv.run()

plt.figure()
plt.plot(sim_nohiv.tivec, sim_hiv.results.gonorrhea.n_infected, label='With HIV')
plt.plot(sim_nohiv.tivec, sim_nohiv.results.gonorrhea.n_infected, label='Without HIV')
plt.title('Gonorrhea infections')
plt.legend()
plt.show()
