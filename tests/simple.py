"""
Run simplest tests
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import sciris as sc

def run_sim(n=2_000, intervention=False):
    ppl = ss.People(n)
    ppl.networks = ss.ndict(ss.simple_embedding(), ss.maternal())

    hiv = ss.HIV()
    #hiv.pars['beta'] = {'simple_embedding': [0.0008, 0.0004], 'maternal': [0.2, 0]}
    hiv.pars['beta'] = {'simple_embedding': [0.02, 0.01], 'maternal': [0.2, 0]}

    gon = ss.Gonorrhea()
    gon.pars['beta'] = 0.6

    pars = {
        'start': 1980,
        'end': 2010,
        'interventions': [ss.hiv.PrEP(0, 10_000)] if intervention else []
        #'interventions': [ss.hiv.ART(0, 10_000)] if intervention else []
    }
    sim = ss.Sim(people=ppl, modules=[hiv, ss.Gonorrhea(), ss.Pregnancy()], pars=pars)
    sim.initialize()
    sim.run()

    return sim

sim1, sim2 = sc.parallelize(run_sim, iterkwargs=[{'intervention':False}, {'intervention':True}])
#sim1 = run_sim(intervention=False)
#sim2 = run_sim(intervention=True)

# Plot
fig, axv = plt.subplots(3,1, sharex=True)
axv[0].plot(sim1.tivec, sim1.results.hiv.n_infected, label='Baseline')
axv[0].plot(sim2.tivec, sim2.results.hiv.n_infected, ls=':', label='Intervention')
axv[0].set_title('HIV number of infections')

axv[1].plot(sim1.tivec, sim1.results.gonorrhea.n_infected, label='Baseline')
axv[1].plot(sim2.tivec, sim2.results.gonorrhea.n_infected, ls=':', label='Intervention')
axv[1].set_title('Gonorrhea number of infections')

axv[2].plot(sim1.tivec, sim1.results.hiv.new_deaths, label='Baseline')
axv[2].plot(sim2.tivec, sim2.results.hiv.new_deaths, ls=':', label='Intervention')
axv[2].set_title('HIV Deaths')


plt.legend()
plt.show()
print('Done')