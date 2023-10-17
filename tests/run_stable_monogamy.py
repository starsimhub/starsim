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
import os
import argparse
import sciris as sc

ss.options(multistream = True) # Can set multistream to False for comparison
plot_graph = True

figdir = os.path.join(os.getcwd(), 'figs', 'stable_monogamy')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

class Graph():
    def __init__(self, nodes, edges):
        self.graph = nx.from_pandas_edgelist(df=edges, source='p1', target='p2', edge_attr=True)
        self.graph.add_nodes_from(nodes.index)
        nx.set_node_attributes(self.graph, nodes.transpose().to_dict())
        return

    def draw_nodes(self, filter, pos, ax, **kwargs):
        inds = [i for i,n in self.graph.nodes.data() if filter(n)]
        nc = ['red' if nd['hiv'] else 'lightgray' for i, nd in self.graph.nodes.data() if i in inds]
        ec = ['green' if nd['on_art'] or nd['on_prep'] else 'black' for i, nd in self.graph.nodes.data() if i in inds]
        if inds:
            nx.draw_networkx_nodes(self.graph, nodelist=inds, pos=pos, ax=ax, node_color=nc, edgecolors=ec, **kwargs)
        return

    def plot(self, pos, edge_labels=False, ax=None):
        kwargs = dict(node_shape='x', node_size=250, linewidths=2, ax=ax, pos=pos)
        self.draw_nodes(lambda n: n['dead'], **kwargs)

        kwargs['node_shape'] = 'o'
        self.draw_nodes(lambda n: not n['dead'] and n['female'], **kwargs)
        
        kwargs['node_shape'] = 's'
        self.draw_nodes(lambda n: not n['dead'] and not n['female'], **kwargs)

        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax)
        nx.draw_networkx_labels(self.graph, labels={i:int(a['cd4']) for i,a in self.graph.nodes.data()}, font_size=8, pos=pos, ax=ax)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, edge_labels={(i,j): int(a['dur']) for i,j,a in self.graph.edges.data()}, font_size=8, pos=pos, ax=ax)
        return


class GraphAnalyzer(ss.Analyzer):
    ''' Simple analyzer to assess if random streams are working '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object

        self.graphs = {}
        return

    def initialize(self, sim):
        self.initialized = True
        self.update_results(sim, init=True)
        return

    def update_results(self, sim, init=False):
        nodes = pd.DataFrame({
            'age': sim.people.age.values,
            'female': sim.people.female.values,
            'dead': sim.people.dead.values,
            'hiv': sim.people.hiv.infected.values,
            'on_art': sim.people.hiv.on_art.values,
            'cd4': sim.people.hiv.cd4.values,
            'on_prep': sim.people.hiv.on_prep.values,
            #'ng': sim.people.gonorrhea.infected.values,
        })

        edges = pd.DataFrame(sim.people.networks[0].to_dict()) #sim.people.networks['simple_embedding'].to_df() #TODO: repr issues

        idx = sim.ti if not init else -1
        self.graphs[idx] = Graph(nodes, edges)
        return

    def finalize(self, sim):
        super().finalize(sim)
        return


def run_sim(n=25, rand_seed=0, intervention=False, analyze=False):
    ppl = ss.People(n)

    ppl.networks = ss.ndict(ss.simple_embedding(mean_dur=5))#, ss.maternal())

    hiv_pars = {
        #'beta': {'simple_embedding': [0.06, 0.04]},
        'beta': {'simple_embedding': [0.3, 0.25]},
        'initial': 0.25 * n,
    }
    hiv = ss.HIV(hiv_pars)

    art = ss.hiv.ART(0, 0.5)

    pars = {
        'start': 1980,
        'end': 2020,
        'remove_dead': False, # So we can see who dies, sim results should not change with True
        'interventions': [art] if intervention else [],
        'rand_seed': rand_seed,
        'analyzers': [GraphAnalyzer()] if analyze else [],
        'n_agents': len(ppl), # TODO: Build into Sim
    }
    sim = ss.Sim(people=ppl, diseases=[hiv], demographics=[ss.Pregnancy()], pars=pars, label=f'Sim with {n} agents and intv={intervention}')
    sim.initialize()

    #sim.diseases['hiv'].set_prognoses(sim, np.arange(0,n,2), from_uids=None)

    sim.run()

    return sim


def run_scenario(n=10, rand_seed=0, analyze=True):
    sims = sc.parallelize(run_sim,
                          kwargs={'n':n, 'analyze': analyze, 'rand_seed': rand_seed},
                          iterkwargs=[{'intervention':True}, {'intervention':False}], die=True)

    for i, sim in enumerate(sims):
        sim.save(os.path.join(figdir, f'sim{i}.obj'))

    return sims


def plot_graph(sim1, sim2):
    g1 = sim1.analyzers[0].graphs
    g2 = sim2.analyzers[0].graphs

    n = len(g1[0].graph)
    el = n <= 10 # Edge labels
    #pos = {i:(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}
    pos = {i:(nd['age'], 2*nd['female']-1 + np.random.uniform(-0.3, 0.3)) for i, nd in g1[0].graph.nodes.data()}
    #pos = nx.spring_layout(g1[0].graph, k=None, pos=None, fixed=None, iterations=50, threshold=0.0001, weight=None, scale=1, center=None, dim=2, seed=None)
    #pos = nx.multipartite_layout(g1[0].graph, subset_key='female', align='vertical', scale=10, center=None)

    fig, axv = plt.subplots(1, 2, figsize=(10,5))
    global ti
    ti = -1 # Initial state is -1
    timax = sim1.tivec[-1]
    def on_press(event):
        print('press', event.key)
        sys.stdout.flush()
        global ti
        if event.key == 'right':
            ti = min(ti+1, timax)
        elif event.key == 'left':
            ti = max(ti-1, -1)

        # Clear
        axv[0].clear()
        axv[1].clear()

        g1[ti].plot(pos, edge_labels=el, ax=axv[0])
        g2[ti].plot(pos, edge_labels=el, ax=axv[1])
        fig.suptitle(f'Time is {ti}')
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)

    g1[ti].plot(pos, ax=axv[0])
    g2[ti].plot(pos, ax=axv[1])
    fig.suptitle(f'Time is {ti}')

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

def analyze_people(sim):
    p = sim.people
    ever_alive = ss.false(np.isnan(p.age))
    years_lived = np.full(len(p), sim.ti+1) # Actually +1 dt here, I think
    years_lived[p.dead] = p.ti_dead[p.dead]
    years_lived = years_lived[ever_alive] # Trim, could be more efficient
    age_initial = p.age[ever_alive].values - years_lived
    age_initial = age_initial.astype(np.float32) # For better hash comparability, there are small differences at float64

    df = pd.DataFrame({
        #'uid': p.uid[ever_alive], # if slicing, don't need ._view,
        'id': [hash((p.slot[i], age_initial[i], p.female[i])) for i in ever_alive], # if slicing, don't need ._view,
        'age_initial': age_initial,
        'years_lived': years_lived,
        'ti_infected': p.hiv.ti_infected[ever_alive].values,
        'ti_art': p.hiv.ti_art[ever_alive].values,
        'ti_dead': p.ti_dead[ever_alive].values,

        # Useful for debugging, but not needed for plotting
        'slot': p.slot[ever_alive].values,
        'female': p.female[ever_alive].values,
    })
    df.replace(to_replace=ss.INT_NAN, value=np.nan, inplace=True)
    df['age_infected'] = df['age_initial'] + df['ti_infected']
    df['age_art']      = df['age_initial'] + df['ti_art']
    df['age_dead']     = df['age_initial'] + df['ti_dead']
    return df
        
def life_bars(data, **kwargs):
    age_final = data['age_initial'] + data['years_lived']
    plt.barh(y=data.index, left=data['age_initial'], width=age_final-data['age_initial'], color='k')

    # Define bools
    infected = ~data['ti_infected'].isna()
    art = ~data['ti_art'].isna()
    dead = ~data['ti_dead'].isna()

    # Infected
    plt.barh(y=data.index[infected], left=data.loc[infected]['age_infected'], width=age_final[infected]-data.loc[infected]['age_infected'], color='r')

    # ART
    plt.barh(y=data.index[art], left=data.loc[art]['age_art'], width=age_final[art]-data.loc[art]['age_art'], color='g')

    # Dead
    #plt.barh(y=data.index[dead], left=data.loc[dead]['age_dead'], width=age_final[dead]-data.loc[dead]['age_dead'], color='k')
    plt.scatter(y=data.index[dead], x=data.loc[dead]['age_dead'], color='k', marker='|')

    return

def plot_longitudinal(sim1, sim2):

    df1 = analyze_people(sim1)
    df1['sim'] = 'Baseline'
    df2 = analyze_people(sim2)
    df2['sim'] = 'With ART'

    df = pd.concat([df1, df2]).set_index('id')
    #f = ti_bars_nested(df)

    df['ypos'] = pd.factorize(df.index.values)[0]
    N = df['sim'].nunique()
    height = 0.5/N

    fig, ax = plt.subplots(figsize=(10,6))

    for n, (lbl, data) in enumerate(df.groupby('sim')):
        yp = data['ypos'] + n/(N+1) # Leave space

        ti_initial = np.maximum(-data['age_initial'], 0)
        ti_final = data['ti_dead'].fillna(40)
        plt.barh(y=yp, left=ti_initial, width=ti_final - ti_initial, color='k', height=height)

        # Infected before birth
        vertical = data['age_infected']<0
        plt.barh(y=yp[vertical], left=data.loc[vertical]['ti_infected'], width=ti_final[vertical]-data.loc[vertical]['ti_infected'], color='m', height=height)

        # Infected
        infected = ~data['ti_infected'].isna()
        ai = data.loc[infected]['age_infected'].values # Adjust for vertical transmission
        ai[~(ai<0)] = 0
        plt.barh(y=yp[infected], left=data.loc[infected]['ti_infected']-ai, width=ti_final[infected]-data.loc[infected]['ti_infected']+ai, color='r', height=height)

        # ART
        art = ~data['ti_art'].isna()
        plt.barh(y=yp[art], left=data.loc[art]['ti_art'], width=ti_final[art]-data.loc[art]['ti_art'], color='g', height=height)

        # Dead
        dead = ~data['ti_dead'].isna()
        plt.scatter(y=yp[dead], x=data.loc[dead]['ti_dead'], color='k', marker='|')

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='Plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=100)
    parser.add_argument('-s', help='Rand seed', type=int, default=2)
    args = parser.parse_args()

    if args.plot:
        print('Reading files', args.plot)
        sim1 = sc.load(os.path.join(args.plot, 'sim1.obj'))
        sim2 = sc.load(os.path.join(args.plot, 'sim2.obj'))
    else:
        print('Running scenarios')
        [sim1, sim2] = run_scenario(n=args.n, rand_seed=args.s)

    #plot_graph(sim1, sim2)
    plot_longitudinal(sim1, sim2)

    plt.show()
    print('Done')