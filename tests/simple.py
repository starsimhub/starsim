"""
Run simplest tests
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt


ppl = ss.People(10_000)
ppl.networks = ss.ndict(ss.simple_embedding(), ss.maternal())

hiv = ss.HIV()
#hiv.pars['beta'] = {'simple_embedding': [0.0008, 0.0004], 'maternal': [0.2, 0]}
hiv.pars['beta'] = {'simple_embedding': [0.02, 0.01], 'maternal': [0.2, 0]}

sim1 = ss.Sim(people=ppl, modules=[hiv, ss.Gonorrhea(), ss.Pregnancy()])
sim1.initialize()
sim1.run()

'''
ppl = ss.People(10_000)
ppl.networks = ss.ndict(ss.simple_embedding(), ss.maternal())
hiv = ss.HIV()
#hiv.pars['beta'] = {'simple_embedding': [0.0008, 0.0004], 'maternal': [0.2, 0]}
hiv.pars['beta'] = {'simple_embedding': [0.02, 0.01], 'maternal': [0.2, 0]}
hiv.pars['interventions'] = ss.hiv.ART(0, 10_000)
sim2 = ss.Sim(people=ppl, modules=[hiv, ss.Gonorrhea(), ss.Pregnancy()])
sim2.initialize()
sim2.run()
'''

fig, axv = plt.subplots(2,1, sharex=True)
axv[0].plot(sim1.tivec, sim1.results.hiv.n_infected)
#axv[0].plot(sim2.tivec, sim2.results.hiv.n_infected)
axv[0].set_title('HIV number of infections')
axv[1].plot(sim1.tivec, sim1.results.gonorrhea.n_infected)
#axv[1].plot(sim2.tivec, sim2.results.gonorrhea.n_infected)
axv[1].set_title('Gonorrhea number of infections')
print('Done')