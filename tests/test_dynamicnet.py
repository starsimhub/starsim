#!/usr/bin/env python3
"""
Larger test of sim performance.
"""

import sciris as sc
import starsim as ss
import networkx as nx
import numpy as np
import pandas as pd 


repeats = 10
n_agents = 1_000

# Vary the infection duration from 1 to 100 days
duration = [1,
    2,
    5,
    10,
    15,
    25,
    50,
    75,
    90,
    100
    ]

def make_run_sim(duration,r):
    """ Make and run a decent-sized default simulation """
    # Make the people
    ppl = ss.People(n_agents=n_agents)
    demographics = [
    ss.Births(pars={'birth_rate': 20}),
    ss.Deaths(pars={'death_rate': 15})
    ]

    sir = ss.SIR()
    ss.SIR(pars=dict( dur_inf=duration[r], beta=0.2, init_prev=0.4, p_death=0.2))
    networks=ss.DynamicNetwork()

    # Make the sim
    sim = ss.Sim(people=ppl, networks=networks, demographics=demographics, diseases=sir)

    # Run the sim
    sim.run()
    
    return sim


if __name__ == '__main__':
    
    edge_count=[]    
    T = sc.timer()
    cpr = sc.cprofile()
    cpr.start()
    for r in range(repeats):    
        make_run_sim(duration, r)
        T.tt(f'Trial {r+1}/{repeats}')      

    sc.heading(f'Average: {T.mean()*1000:0.0f} ± {T.std()/len(T)**0.5*1000:0.0f} ms')
    cpr.stop()
    df=cpr.to_df()
    df.to_csv('DynamicNet Test Profile.csv', index=False)
    print("Run "+str(repeats)+" Mem "+str(sc.memload()))
    print("RAM "+str(repeats)+" "+str(sc.checkram()))
