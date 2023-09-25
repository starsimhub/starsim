"""
Run a simple HIV simulation with random number coherence
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import sciris as sc

n = 1_000 # Agents

def run_sim(n=25, intervention=False, analyze=False):
    ppl = ss.People(n)
    ppl.networks = ss.ndict(ss.simple_embedding())#, ss.maternal())

    hiv = ss.HIV()
    hiv.pars['beta'] = {'simple_embedding': [0.10, 0.08]}
    hiv.pars['initial'] = 10

    pars = {
        'start': 1980,
        'end': 2010,
        'interventions': [ss.hiv.ART(t=[0, 1], coverage=[0, 0.9**3])] if intervention else [],
        'rand_seed': 0,
    }
    sim = ss.Sim(people=ppl, modules=[hiv], pars=pars, label=f'Sim with {n} agents and intv={intervention}')
    sim.initialize()
    sim.run()

    return sim

sim1, sim2 = sc.parallelize(run_sim, kwargs={'n': n}, iterkwargs=[{'intervention':False}, {'intervention':True}], die=True)

# Plot
fig, axv = plt.subplots(2,1, sharex=True)
axv[0].plot(sim1.tivec, sim1.results.hiv.n_infected, label='Baseline')
axv[0].plot(sim2.tivec, sim2.results.hiv.n_infected, ls=':', label='Intervention')
axv[0].set_title('HIV number of infections')

axv[1].plot(sim1.tivec, sim1.results.hiv.new_deaths, label='Baseline')
axv[1].plot(sim2.tivec, sim2.results.hiv.new_deaths, ls=':', label='Intervention')
axv[1].set_title('HIV Deaths')

plt.legend()
plt.show()
print('Done')