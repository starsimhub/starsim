#!/usr/bin/env python3
"""
Larger test of sim performance.
"""

import sciris as sc
import starsim as ss

# Define the parameters
#ss.options(multirng=True) #MultiRNG

repeats = 10
n_agents = 10_000

def make_run_sim():
    """ Make and run a decent-sized default simulation """
    # Make the people
    ppl = ss.People(n_agents=n_agents)
    demographics = [
    ss.Births(pars={'birth_rate': 20}),
    ss.Deaths(pars={'death_rate': 15})
    ]

    sir = ss.SIR()
    ss.SIR(pars=dict( dur_inf=10, beta=0.2, init_prev=0.4, p_death=0.2))
    networks=ss.DynamicNetwork()

    # Make the sim
    sim = ss.Sim(people=ppl, networks=networks, demographics=demographics, diseases=sir)

    # Run the sim
    sim.run()
    
    return sim


if __name__ == '__main__':
    
    T = sc.timer()
    for r in range(repeats):
        make_run_sim()
        T.tt(f'Trial {r+1}/{repeats}')      

    sc.heading(f'Average: {T.mean()*1000:0.0f} ± {T.std()/len(T)**0.5*1000:0.0f} ms')
