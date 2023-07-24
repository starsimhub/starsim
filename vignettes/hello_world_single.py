'''
"Hello world" example of creating a single STI
'''

import stisim as sti

#%% 1. One-liner
sti.Sim(modules='hpv').run().plot()


#%% 2. Supply parameters
pars = dict(
    n_agents = 10e3,
    start = '2022-01-01',
    end = '2024-01-01',
)

sim = sti.Sim(pars, modules='HPV') # Case-insensitive
sim.run()
sim.plot('hpv') # Plot HPV-specific results


#%% 3. More complex example (from hpvsim/tests/test_sim.py)

# Create and run the simulation
pars = {
    'n_agents': 5e3,
    'start': 1970,
    'burnin': 30,
    'end': 2030,
    'ms_agent_ratio': 100
}

# Create some genotype pars
genotype_pars = {
    16: {
        'sev_fn': dict(form='logf2', k=0.25, x_infl=0, ttc=30)
    }
}

hpv = sti.HPV(genotype_pars=genotype_pars)
sim = sti.Sim(pars=pars, modules=hpv)

sim.run(verbose=False)
sim.plot(do_save=True)