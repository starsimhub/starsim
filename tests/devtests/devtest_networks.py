"""
Network connections proof of concept
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt


ppl = ss.People(10000)

mf_pars = {'part_rates': 0.85}
msm_pars = {'part_rates': 0.1}
ppl.networks = ss.ndict(
    ss.mf_msm(msm_pars=msm_pars, mf_pars=mf_pars),
    ss.maternal()
)

hiv = ss.HIV()
hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'msm': [0.004, 0.004], 'maternal': [0.2, 0]}
 
sim = ss.Sim(people=ppl, demographics=ss.Pregnancy(), diseases=[hiv, ss.Gonorrhea()])
sim.initialize()
sim.run()

plt.figure()
plt.plot(sim.tivec, sim.results.hiv.n_infected)
plt.title('HIV number of infections')
plt.show()
