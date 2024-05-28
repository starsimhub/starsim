"""
Compare two HIV simulations, one baseline and the other with ART
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import argparse
import sciris as sc

default_n_agents = 25
# Three choices for network here, note that only the first two are common-random-number safe
network = ['stable_monogamy', 'EmbeddingNet'][1]

do_plot_graph = True
# Several choices for how to layout the graph when plotting
kind = ['radial', 'bipartite', 'spring', 'multipartite'][1]

do_plot_longitudinal = True
do_plot_timeseries = True

ss.options(multirng = True) # Can set multirng to False for comparison

figdir = os.path.join(os.getcwd(), 'figs', network)
sc.path(figdir).mkdir(parents=True, exist_ok=True)


class stable_monogamy(ss.SexualNetwork):
    """
    Very simple network for debugging in which edges are:
    1-2, 3-4, 5-6, ...
    """
    def __init__(self, **kwargs):
        # Call init for the base class, which sets all the keys
        super().__init__(**kwargs)
        return

    def init_pre(self, sim):
        n = len(sim.people._uid_map)
        n_edges = n//2
        self.edges.p1 = np.arange(0, 2*n_edges, 2) # EVEN
        self.edges.p2 = np.arange(1, 2*n_edges, 2) # ODD
        self.edges.beta = np.ones(n_edges)
        return


class Graph():
    def __init__(self, nodes, edges):
        self.graph = nx.from_pandas_edgelist(df=edges, source='p1', target='p2', edge_attr=True)
        self.graph.add_nodes_from(nodes.index)
        nx.set_node_attributes(self.graph, nodes.transpose().to_dict())
        return

    def draw_nodes(self, filter, pos, ax, **kwargs):
        inds = [i for i,n in self.graph.nodes.data() if filter(n)]
        nc = ['red' if nd['hiv'] else 'lightgray' for i, nd in self.graph.nodes.data() if i in inds]
        ec = ['green' if nd['on_art'] else 'black' for i, nd in self.graph.nodes.data() if i in inds]
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
            nx.draw_networkx_edge_labels(self.graph, edge_labels={(i,j): int(a['dur']) if 'dur' in a else np.nan for i,j,a in self.graph.edges.data()}, font_size=8, pos=pos, ax=ax)
        return


class GraphAnalyzer(ss.Analyzer):
    ''' Simple analyzer to assess if common random numbers are working '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object

        self.graphs = {}
        return

    def init_pre(self, sim):
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
        })

        edges = sim.people.networks[network.lower()].to_df()

        idx = sim.ti if not init else -1
        self.graphs[idx] = Graph(nodes, edges)
        return

    def finalize(self, sim):
        super().finalize(sim)
        return


def run_sim(n=25, rand_seed=0, intervention=False, analyze=False, lbl=None):
    ppl = ss.People(n)

    try:
        net_class = eval(network)
    except Exception as e:
        net_class = getattr(ss, network)
    ppl.networks = ss.ndict(net_class(), ss.MaternalNet())

    hiv_pars = {
        'beta': {network: [0.3, 0.25], 'maternal': [0.2, 0]},
        'init_prev': 0.25,
        'art_efficacy': 0.96,
    }
    hiv = ss.HIV(hiv_pars)

    art = ss.hiv.ART(0, 0.6) # 60% coverage from day 0

    pars = {
        'start': 1980,
        'end': 2020,
        'remove_dead': False, # So we can see who dies, sim results should not change with True
        'interventions': [art] if intervention else [],
        'rand_seed': rand_seed,
        'analyzers': [GraphAnalyzer()] if analyze else [],
    }
    sim = ss.Sim(people=ppl, diseases=[hiv], demographics=[ss.Pregnancy(), ss.Deaths()], pars=pars, label=lbl)
    sim.initialize()

    # Infect every other person, useful for exploration in conjunction with init_prev=0
    #sim.diseases['hiv'].set_prognoses(sim, np.arange(0,n,2), from_uids=None)

    sim.run()

    return sim


def run_scenario(n=10, rand_seed=0, analyze=True):
    sims = sc.parallelize(run_sim,
                          kwargs={'n':n, 'analyze': analyze, 'rand_seed': rand_seed},
                          iterkwargs=[{'intervention':False, 'lbl':'Baseline'}, {'intervention':True, 'lbl':'Intervention'}], die=True)

    for i, sim in enumerate(sims):
        sim.save(os.path.join(figdir, f'sim{i}.obj'))

    return sims


def getpos(ti, g1, g2, guess=None, kind='bipartite'):

    n1 = dict(g1[ti].graph.nodes.data())
    n2 = dict(g2[ti].graph.nodes.data())
    nodes = sc.mergedicts(n2, n1)
    n = len(nodes)

    if kind == 'radial':
        pos = {i:(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}
        if guess:
            if len(guess) < n:
                pos = {i:(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}

    elif kind == 'spring':
        pos = nx.spring_layout(g1[ti].graph, k=None, pos=guess, fixed=None, iterations=50, threshold=0.0001, weight=None, scale=1, center=None, dim=2, seed=None)
        if guess:
            pos = sc.mergedicts(pos, guess)

    elif kind == 'multipartite':
        pos = nx.multipartite_layout(g1[ti].graph, subset_key='female', align='vertical', scale=10, center=None)
        if guess:
            pos = sc.mergedicts(pos, guess)

        if guess:
            for i in guess.keys():
                pos[i] = (pos[i][0], guess[i][1]) # Keep new x but carry over y

    elif kind == 'bipartite':
        pos = {i:(nd['age'], 2*nd['female']-1 + np.random.uniform(-0.3, 0.3)) for i, nd in nodes.items()}

        if guess:
            for i in guess.keys():
                pos[i] = (pos[i][0], guess[i][1]) # Keep new x but carry over y

    return pos


def plot_graph(sim1, sim2):
    g1 = sim1.analyzers[0].graphs
    g2 = sim2.analyzers[0].graphs

    n = len(g1[-1].graph)
    el = n <= 25 # Draw edge labels

    fig, axv = plt.subplots(1, 2, figsize=(10,5))
    global ti
    timax = sim1.tivec[-1]

    global pos
    pos = {}
    pos[-1] = getpos(0, g1, g2, kind=kind)
    for ti in range(timax+1):
        pos[ti] = getpos(ti, g1, g2, guess=pos[ti-1], kind=kind)

    ti = -1 # Initial state is -1, representing the state before the first step

    def on_press(event):
        print('press', event.key)
        sys.stdout.flush()
        global ti, pos
        if event.key == 'right':
            ti = min(ti+1, timax)
        elif event.key == 'left':
            ti = max(ti-1, -1)

        # Clear
        axv[0].clear()
        axv[1].clear()

        g1[ti].plot(pos[ti], edge_labels=el, ax=axv[0])
        g2[ti].plot(pos[ti], edge_labels=el, ax=axv[1])
        fig.suptitle(f'Time is {ti} (use the arrow keys to change)')
        axv[0].set_title(sim1.label)
        axv[1].set_title(sim2.label)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)

    g1[ti].plot(pos[ti], edge_labels=el, ax=axv[0])
    g2[ti].plot(pos[ti], edge_labels=el, ax=axv[1])
    fig.suptitle(f'Time is {ti} (use the arrow keys to change)')
    axv[0].set_title(sim1.label)
    axv[1].set_title(sim2.label)
    return fig


def plot_ts():
    # Plot timeseries summary
    fig, axv = plt.subplots(2,1, sharex=True)
    axv[0].plot(sim1.tivec, sim1.results.hiv.n_infected, label=sim1.label)
    axv[0].plot(sim2.tivec, sim2.results.hiv.n_infected, ls=':', label=sim2.label)
    axv[0].set_title('HIV number of infections')

    axv[1].plot(sim1.tivec, sim1.results.hiv.new_deaths, label=sim1.label)
    axv[1].plot(sim2.tivec, sim2.results.hiv.new_deaths, ls=':', label=sim2.label)
    axv[1].set_title('HIV Deaths')

    plt.legend()
    return fig


def analyze_people(sim):
    p = sim.people
    ever_alive = ss.false(np.isnan(p.age))
    years_lived = np.full(len(p), sim.ti+1) # Actually +1 dt here, I think
    years_lived[p.dead] = p.ti_dead[p.dead]
    years_lived = years_lived[ever_alive] # Trim, could be more efficient
    age_initial = p.age[ever_alive].values - years_lived
    age_initial = age_initial.astype(np.float32) # For better hash comparability, there are small differences at float64

    df = pd.DataFrame({
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


def plot_longitudinal(sim1, sim2):

    df1 = analyze_people(sim1)
    df1['sim'] = 'Baseline'
    df2 = analyze_people(sim2)
    df2['sim'] = 'With ART'

    df = pd.concat([df1, df2]).set_index('id')

    df['ypos'] = pd.factorize(df.index.values)[0]
    N = df['sim'].nunique()
    height = 0.5/N

    fig, ax = plt.subplots(figsize=(10,6))

    # For the legend:
    plt.barh(y=0, left=0, width=1e-6, color='k', height=height, label='Alive')
    plt.barh(y=0, left=0, width=1e-6, color='m', height=height, label='Infected before birth')
    plt.barh(y=0, left=0, width=1e-6, color='r', height=height, label='Infected')
    plt.barh(y=0, left=0, width=1e-6, color='g', height=height, label='ART')
    plt.scatter(y=0, x=0, color='c', marker='|', label='Death')

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
        plt.scatter(y=yp[dead], x=data.loc[dead]['ti_dead'], color='c', marker='|')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('UID')
    ax.legend()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='Plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Rand seed', type=int, default=0)
    args = parser.parse_args()

    if args.plot:
        print('Reading files', args.plot)
        sim1 = sc.load(os.path.join(args.plot, 'sim1.obj'))
        sim2 = sc.load(os.path.join(args.plot, 'sim2.obj'))
    else:
        print('Running scenarios')
        [sim1, sim2] = run_scenario(n=args.n, rand_seed=args.s)

    if do_plot_longitudinal:
        plot_longitudinal(sim1, sim2)

    if do_plot_graph:
        plot_graph(sim1, sim2)

    if do_plot_timeseries:
        plot_ts()

    plt.show()
    print('Done')