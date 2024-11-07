#!/usr/bin/env python3
"""
Larger test of sim performance.
"""

import sciris as sc
import starsim as ss

# Define the parameters
repeats = 10
n_agents = 100_000
pars = sc.objdict(
    dur     = 100, # Number of years to simulate
    dt      = 0.5, # Timestep
    verbose = 0,   # Don't print details of the run
)

def make_run_sim():
    """ Make and run a decent-sized default simulation """
    # Make the people
    ppl = ss.People(n_agents=n_agents)

    # Make the components
    sir = ss.SIS()
    hiv = ss.HIV()
    hiv.pars['beta'] = {'mf': [0.15, 0.10], 'maternal': [0.2, 0], 'random': [0,0]}
    networks = [ss.RandomNet(), ss.MFNet(), ss.MaternalNet()]

    # Make the sim
    sim = ss.Sim(pars=pars, people=ppl, networks=networks, demographics=ss.Pregnancy(), diseases=[sir, hiv])

    # Run the sim
    sim.run()

    return sim


if __name__ == '__main__':

    T = sc.timer()
    for r in range(repeats):
        make_run_sim()
        T.tt(f'Trial {r+1}/{repeats}')

    sc.heading(f'Average: {T.mean()*1000:0.0f} ± {T.std()/len(T)**0.5*1000:0.0f} ms')