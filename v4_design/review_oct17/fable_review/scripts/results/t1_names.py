"""Flow result naming, and ABM vs ODE result objects"""
import numpy as np
import starsim as ss

def make():
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=10),
        init_prev = 0.05,
    )

for solver in ['ode', 'abm', 'binomial']:
    sim = ss.Sim(make(), solver=solver, total_pop=1000, n_agents=1000, unit='day', stop=30, verbose=0).run()
    res = sim.results.disease if 'disease' in sim.results else sim.results[list(sim.results.keys())[-1]]
    mods = [k for k,v in sim.results.items() if isinstance(v, ss.Results)]
    print(f'\n=== solver={solver}: module result containers: {mods}')
    for m in mods:
        r = sim.results[m]
        keys = [k for k,v in r.items() if isinstance(v, ss.Result)]
        print(m, 'keys:', keys)
        for k in keys:
            v = r[k]
            print(f'   {k:16s} dtype={v.values.dtype} scale={v.scale} auto_plot={v.auto_plot} summarize_by={v.summarize_by!r} summary_method={v.summary_method()} label={v.label!r}')
    print('sim-level keys:', [k for k,v in sim.results.items() if isinstance(v, ss.Result)])
    r = sim.results[mods[0]]
    if 'infection' in r and 'new_infections' in r:
        print('infection == new_infections:', np.allclose(r.infection.values, r.new_infections.values))
        print('cum_infection == cum_infections:', np.allclose(r.cum_infection.values, r.cum_infections.values))
    print('n_R[1]-n_R[0], recovery[1]:', r.n_R[1]-r.n_R[0], r.recovery[1], ' recovery[0]:', r.recovery[0])
    print('len timevec', len(r.n_S), 'stop', sim.t.stop, 'npts', sim.t.npts)
