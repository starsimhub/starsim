"""
Run simplest tests
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt


ppl = ss.People(10000)
ppl.networks = ss.ndict(ss.mf(), ss.maternal())

hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.08, 0.04], 'maternal': [0.2, 0]}

ng = ss.Gonorrhea()

sim = ss.Sim(people=ppl, demographics=[ss.Pregnancy()], diseases=[hiv, ng])
sim.initialize()
sim.run()

plt.figure()
plt.plot(sim.tivec, sim.results.hiv.n_infected)
plt.title('HIV number of infections')