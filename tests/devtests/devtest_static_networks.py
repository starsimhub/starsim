"""
Static networks from networkx
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import networkx as nx

ppl = ss.People(10000)

# This example runs on one static networks + the maternal network
g1 = nx.scale_free_graph(1000)
ppl.networks = ss.Networks(
    ss.static(graph=g1), ss.maternal()
)

hiv = ss.HIV()
hiv.pars['beta'] = {'static': [0.0008, 0.0008], 'maternal': [0.2, 0]}

sim = ss.Sim(people=ppl, demographics=ss.Pregnancy(), diseases=[hiv, ss.Gonorrhea()])
sim.initialize()
sim.run()

plt.figure()
plt.plot(sim.tivec, sim.results.hiv.n_infected)
plt.title('HIV number of infections')
plt.show()
