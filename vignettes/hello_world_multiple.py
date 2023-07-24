'''
"Hello world" example of creating multiple STI + results output
'''

import stisim as sti

#%% 1. One-liner
sti.Sim(modules=['hiv', 'gonorrhea']).run().plot()


#%% 2. Supply parameters as a single nested dict
pars = dict(
    n_agents = 10e3,
    start = 1970,
    end = 2020,
    hiv = dict(
        cd4_pars = dict(
            initial = 500,
            rate = 50,
        ),
    ),
    gonorrhea = dict(
        transm2f = 4.0,
    ),
)

sim = sti.Sim(pars, modules=['hiv', 'gonorrhea']) # Case-insensitive
sim.run()
sim.plot('default', 'hiv', 'gonorrhea') # Plot default as well as disease-specific plots
print(sim.results.hiv) # All HIV results
print(sim.results['hiv']['incidence']) # A single result, can also be accessed as dict keys
print(sim.results.hiv_gonorrhea_coinfections) # Result populated by the HIV-gonorrhea connector


#%% 3. Create modules first, then combine

sti_pars = dict(
    n_agents = 10e3,
    start = 1970,
    end = 2020,
)

hiv_pars = dict(
    cd4_pars = dict(
        initial = 500,
        rate = 50,
    ),
)

# Create modules first
hiv = sti.HIV(hiv_pars)
gon = sti.Gonorrhea(transm2f=4.0)

# Create and run the sim
sim = sti.Sim(sti_pars, modules=[hiv, gon]) # Case-insensitive
sim.initialize() # Happens automatically as part of sim.run(), but can call explicitly
assert id(sim.pars.hiv) == id(sim.hiv.pars) # These are the same object after initialization
sim.run() # Run

# Plot by module
sim.plot() # Equivalent to sim.plot('default')
sim.hiv.plot() # Equivalent to sim.plot('hiv')
sim.gonorrhea.plot() # Equivalent to sim.plot('gonorrhea')

# Results by module
print(sim.hiv.results) # Reference to sim.results.hiv
