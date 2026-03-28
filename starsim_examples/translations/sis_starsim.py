"""
Test simulation performance
"""
import starsim as ss
import sciris as sc

# Define the parameters
pars = sc.dictobj(
    n_agents = 100_000,
    dur = 100,
)

# Create the sim
sim = ss.Sim(
    n_agents=pars.n_agents,
    dur=pars.dur,
    diseases = 'sis',
    networks = 'random',
    verbose = 0,
)

# Run and time
T = sc.timer()
sim.run()
T.toc(f'Time for SIS-Starsim, n_agents={pars.n_agents}, dur={pars.dur}')