"""
Run simplest tests
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt


ppl = ss.People(9000)
ppl.networks = ss.ndict(ss.MFNet(), ss.MaternalNet())

hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.08, 0.04], 'maternal': [0.2, 0]}

ng = ss.Gonorrhea()

sim = ss.Sim(people=ppl, demographics=[ss.Pregnancy()], diseases=[hiv, ng])
sim.initialize()
sim.run()

# Plotting
fig, axv = plt.subplots(3,1, figsize=(10,8))
for ax, module in zip(axv[:2], ['HIV', 'Gonorrhea']):
    results = sim.results[module.lower()]
    for ch in results.keys():
        if ch[:2] == 'n_':
            ax.plot(sim.yearvec, results[ch], label=ch)
    ax.legend()
    ax.set_title(module)

results = sim.results.pregnancy
for ch in results.keys():
    axv[2].plot(sim.yearvec, sim.results.pregnancy[ch], label=ch)
axv[2].set_title('Pregnancy')
axv[2].legend()
fig.supylabel('Count')
fig.supxlabel('Year')
fig.tight_layout()
plt.show()