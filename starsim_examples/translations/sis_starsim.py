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

# Tests
def test_inf(sim):
    n_inf = sim.results.sis.n_infected
    inf0 = n_inf[0]
    infm = n_inf.max()
    inf1 = n_inf[-1]
    tests = {
        f'Initial infections are nonzero: {inf0}' : inf0 > 0.01*pars.n_agents,
        f'Initial infections start low: {inf0}'   : inf0 < 0.05*pars.n_agents,
        f'Infections peak high: {infm}'           : infm > 0.5*pars.n_agents,
        f'Infections stabilize: {inf1}'           : inf1 < n_inf.max(),
    }
    for k,tf in tests.items():
        print(f'✓ {k}') if tf else print(f'× {k}')
    assert all(tests.values())
    return tests

test_inf(sim)