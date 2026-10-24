import os; os.environ['MPLBACKEND']='agg'
import numpy as np, sciris as sc, starsim as ss, json
b = ss.par(0.3, bounds=[0.05, 1.0], name='beta', prior=ss.lognorm_ex(mean=0.3, std=0.1))
S, I, R = ss.states(3)
infection = S >> I @ b * I
cases = ss.outcome(0.5 * infection, every=ss.weeks(1), likelihood=ss.nbinom(n=ss.par(10, bounds=[0.1, 100], name='k')))
sir = ss.Disease(infection, I >> R @ ss.days(10), cases, init_prev=0.01, name='sir')
sim = ss.Sim(sir, total_pop=10_000, unit='day', stop=100, verbose=0).init()
spec = sim.to_spec()
mod = spec['modules'][0]
print(json.dumps(mod['pars'], default=str)[:600])
print(json.dumps([o for o in mod.get('outcomes', mod.get('results', []))], default=str)[:600])
print('fit field in spec:', spec.get('fit', 'ABSENT'), '| config keys:', list(spec['config'].keys())[:20])
