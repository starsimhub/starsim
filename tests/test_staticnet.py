#!/usr/bin/env python3
"""
Larger test of sim performance.
"""

import sciris as sc
import starsim as ss
import networkx as nx
import numpy as np
import pandas as pd 


# Define the parameters
#ss.options(multirng=True) #MultiRNG

repeats = 10
n_agents = 1_000

# list of probabilities of edge connection for the networkx graph
prob = [0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1
    ]

def make_run_sim(pr,r):
    """ Make and run a decent-sized default simulation """
    # Make the people
    ppl = ss.People(n_agents=n_agents)
    demographics = [
    ss.Births(pars={'birth_rate': 20}),
    ss.Deaths(pars={'death_rate': 15})
    ]

    g=nx.erdos_renyi_graph(n=1000, p=pr[r], directed=True)
    sir = ss.SIR()
    ss.SIR(pars=dict( dur_inf=10, beta=0.2, init_prev=0.4, p_death=0.2))
    networks=ss.StaticNet(graph=g)


    # Make the sim
    sim = ss.Sim(people=ppl, networks=networks, demographics=demographics, diseases=sir)

    # Run the sim
    sim.run()
    
    # Static NetworkX Graph Check that number of nodes (agents) <= population/number of agents
    assert len(g.nodes)<=n_agents, "Error: Please ensure the number of nodes in graph is smaller or equal to the population size"
    
    # Check that the networkx number of edges <= mathematical maximum 2n(n-1)/2
    assert len(g.edges)<=((2*len(g.nodes))*(len(g.nodes)-1))/2, "EdgeCountError: More edges created than expected"

    print("Edges "+ str(len(g.edges)))

    return sim


if __name__ == '__main__':
    
    edge_count=[]    
    T = sc.timer()
    cpr = sc.cprofile()
    cpr.start()
    for r in range(repeats):    
        make_run_sim(prob, r)
        T.tt(f'Trial {r+1}/{repeats}')      

    sc.heading(f'Average: {T.mean()*1000:0.0f} ± {T.std()/len(T)**0.5*1000:0.0f} ms')
    cpr.stop()
    df=cpr.to_df()
    df.to_csv('StaticNet Test Profile.csv')
    print("Run "+str(repeats)+" Mem "+str(sc.memload()))rint("RAM "+str(repeats)+" "+str(sc.checkram()))
