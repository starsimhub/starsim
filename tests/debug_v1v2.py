"""
More debugging ...
"""

import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

# Import the Starsims
root = sc.path('/home/cliffk/idm/')
ss1 = sc.importbypath(root / 'starsim2/starsim') # Starsim v1.0
ss2 = sc.importbypath(root / 'starsim/starsim') # Starsim v2.0

# Define the parameters
v2pars = sc.objdict(
    n_agents  = 10e3, # Number of agents
    start     = 2000, # Starting year
    dur       = 20,   # Number of years to simulate
    dt        = 0.2,  # Timestep
    verbose   = 0.05,    # Don't print details of the run
    rand_seed = 2,    # Set a non-default seed
    diseases = ['sir', 'sis'],
    networks = ['random', 'mf', 'maternal'],
    demographics = True,
)

v1pars = v2pars.copy()
v1pars.n_years = v1pars.pop('dur')

s1 = ss1.Sim(v1pars)
s2 = ss2.Sim(v2pars)

s1.run()
s2.run()

df = ss.diff_sims(s1.summary, s2.summary, output=True)
df.disp()

s1.plot()
s2.plot()

plt.show()