import sciris as sc
import starsim as ss

debug = 1

# Define the parameters
pars = sc.objdict(
    dur        = 20,
    n_agents   = 10e3,
    dt         = 0.2,
    rand_seed  = 2,
    verbose    = 0.02,
    diseases   = ['sir', 'sis'],
    networks   = ['random', 'mf', 'maternal'],
)

s1 = ss.Sim(pars)

if debug:
    s1.run()
    s1.plot()
    
else:
    with sc.timer():
        n = 10
        m1 = ss.MultiSim(s1).run(n_runs=n)
        new = m1.summarize()
        folder = '/home/cliffk/idm/starsim2/tests/'
        old = sc.loadjson(f'{folder}tmp_random.json')
        df = ss.diff_sims(old, new, output=True)
        df.disp()