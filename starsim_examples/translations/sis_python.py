"""
Test simulation performance
"""
import sciris as sc

# Define the parameters
n_agents = 100_000
dur = 100

class Sim:
    pass

class Random:
    pass

class SIS:
    pass


# Create the sim
sim = Sim(
    n_agents=n_agents,
    dur=dur,
    diseases = SIS(),
    networks = Random(),
    verbose = 0,
)

# Run and time
T = sc.timer()
sim.run()
T.toc(f'Time for SIS-Python, n_agents={n_agents}, dur={dur}')