"""
Run test with stable_monogamy network
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sys

plot_graph = True

class Graph():
    def __init__(self, nodes, edges):
        self.graph = nx.from_pandas_edgelist(df=edges, source='p1', target='p2', edge_attr=True)
        self.graph.add_nodes_from(nodes.index)
        nx.set_node_attributes(self.graph, nodes.transpose().to_dict())
        return

    def draw_nodes(self, filter, ax, **kwargs):
        inds = [i for i,n in self.graph.nodes.data() if filter(n)]
        nc = ['red' if nd['hiv'] else 'lightgray' for i, nd in self.graph.nodes.data() if i in inds]
        ec = ['green' if nd['on_art'] or nd['on_prep'] else 'black' for i, nd in self.graph.nodes.data() if i in inds]
        if inds:
            nx.draw_networkx_nodes(self.graph, nodelist=inds, pos=pos, ax=ax, node_color=nc, edgecolors=ec, **kwargs)
        return

    def plot(self, pos, ax=None):
        kwargs = dict(node_shape='x', node_size=250, linewidths=2, ax=ax)
        self.draw_nodes(lambda n: n['dead'], **kwargs)

        kwargs['node_shape'] = 'o'
        self.draw_nodes(lambda n: not n['dead'] and n['female'], **kwargs)
        
        kwargs['node_shape'] = 's'
        self.draw_nodes(lambda n: not n['dead'] and not n['female'], **kwargs)

        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
        nx.draw_networkx_labels(self.graph, labels={i:int(a['cd4']) for i,a in self.graph.nodes.data()}, font_size=8, pos=pos, ax=ax)
        nx.draw_networkx_edge_labels(self.graph, edge_labels={(i,j): int(a['dur']) for i,j,a in self.graph.edges.data()}, font_size=8, pos=pos, ax=ax)
        return


class rng_analyzer(ss.Analyzer):
    ''' Simple analyzer to assess if random streams are working '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object

        self.graphs = {}
        return

    def initialize(self, sim):
        self.initialized = True
        return

    def apply(self, sim):
        nodes = pd.DataFrame({
            'female': sim.people.female.values,
            'dead': sim.people.dead.values,
            'hiv': sim.people.hiv.infected.values,
            'on_art': sim.people.hiv.on_art.values,
            'cd4': sim.people.hiv.cd4.values,
            'on_prep': sim.people.hiv.on_prep.values,
            #'ng': sim.people.gonorrhea.infected.values,
        })

        edges = pd.DataFrame(sim.people.networks[0].to_dict()) #sim.people.networks['simple_embedding'].to_df() #TODO: repr issues

        self.graphs[sim.ti] = Graph(nodes, edges)
        return

    def finalize(self, sim):
        super().finalize(sim)
        return


def run_sim(n=25, intervention=False, analyze=False):
    ppl = ss.People(n)
    ppl.networks = ss.ndict(ss.stable_monogamy())#, ss.maternal())

    hiv = ss.HIV()
    hiv.pars['beta'] = {'stable_monogamy': [0.06, 0.04]}
    hiv.pars['initial'] = 0

    pars = {
        'start': 1980,
        'end': 2010,
        'interventions': [ss.hiv.ART(0, 0.5)] if intervention else [], # ss.hiv.PrEP(0, 0.2), 
        'rand_seed': 0,
        'analyzers': [rng_analyzer()] if analyze else [],
    }
    sim = ss.Sim(people=ppl, modules=[hiv], pars=pars, label=f'Sim with {n} agents and intv={intervention}')
    sim.initialize()

    sim.modules['hiv'].set_prognoses(sim, np.arange(0,n,2), from_uids=None)

    sim.run()

    return sim

n = 10
sim2 = run_sim(n, intervention=True, analyze=plot_graph)
sim1 = run_sim(n, intervention=False, analyze=plot_graph)

if plot_graph:
    g1 = sim1.analyzers[0].graphs
    g2 = sim2.analyzers[0].graphs

    pos = {i:(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}

    fig, axv = plt.subplots(1, 2, figsize=(10,5))
    global ti
    ti = 0
    timax = sim1.tivec[-1]
    def on_press(event):
        print('press', event.key)
        sys.stdout.flush()
        global ti
        if event.key == 'right':
            ti = min(ti+1, timax)
        elif event.key == 'left':
            ti = max(ti-1, 0)

        # Clear
        axv[0].clear()
        axv[1].clear()

        g1[ti].plot(pos, ax=axv[0])
        g2[ti].plot(pos, ax=axv[1])
        fig.suptitle(f'Time is {ti}')
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)

    g1[ti].plot(pos, ax=axv[0])
    g2[ti].plot(pos, ax=axv[1])
    plt.show()

# Plot
fig, axv = plt.subplots(2,1, sharex=True)
axv[0].plot(sim1.tivec, sim1.results.hiv.n_infected, label='Baseline')
axv[0].plot(sim2.tivec, sim2.results.hiv.n_infected, ls=':', label='Intervention')
axv[0].set_title('HIV number of infections')

axv[1].plot(sim1.tivec, sim1.results.hiv.new_deaths, label='Baseline')
axv[1].plot(sim2.tivec, sim2.results.hiv.new_deaths, ls=':', label='Intervention')
axv[1].set_title('HIV Deaths')

''' Gonorrhea removed for now
axv[2].plot(sim1.tivec, sim1.results.gonorrhea.n_infected, label='Baseline')
axv[2].plot(sim2.tivec, sim2.results.gonorrhea.n_infected, ls=':', label='Intervention')
axv[2].set_title('Gonorrhea number of infections')
'''
plt.legend()
plt.show()
print('Done')