"""
Static networks from networkx or .csv files
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import networkx as nx

ppl = ss.People(10000)

# This example
G = nx.erdos_renyi_graph(10000, 0.0001)
ppl.networks = ss.Networks(
    ss.static(graph=G), ss.maternal()
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
