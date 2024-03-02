"""
Network connections proof of concept
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt


ppl = ss.People(10000)

# This example uses a connector between the MF and MSM networks.
# The connector overrides participation rates so that a given
# percentage of males (prop_bi) in the MSM network are also in the
# MF network.
# This particular connector doesn't do anything with concurrency,
# i.e. a male who participates in both networks and is currently in
# a relationship in the MSM network is still considered available for
# partnerships in the MF network.
mf_pars = {'part_rates': 0.85}
msm_pars = {'part_rates': 0.1}
ppl.networks = ss.Networks(
    ss.MSMNet(pars=msm_pars), ss.MFNet(pars=mf_pars), ss.MaternalNet(),
    connectors=ss.MF_MSM(pars={'prop_bi': 0.4})
)

hiv = ss.HIV()
hiv.pars['beta'] = {'mf': [0.0008, 0.0004], 'msm': [0.004, 0.004], 'maternal': [0.2, 0]}
 
sim = ss.Sim(people=ppl, demographics=ss.Pregnancy(), diseases=[hiv, ss.Gonorrhea()])
sim.initialize()
sim.run()

plt.figure()
plt.plot(sim.yearvec, sim.results.hiv.n_infected)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('HIV number of infections')
plt.show()