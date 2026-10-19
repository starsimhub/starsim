"""Steady-state timings of the four paths, small and medium declared models."""
import time
import numpy as np
import sciris as sc
import starsim as ss

def sir():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=8), init=dict(I=0.01), name='sir')

def ages():
    d = ss.Disease(infection=ss.transmission('S -> E', beta=ss.seasonal(0.4, amplitude=0.3, peak=1)), latency=ss.progression('E -> I', dur=ss.erlang(mean=5, k=3)),
                   outcome=ss.progression('I -> (R, D)', dur=8, p=[None, 0.1]), init=dict(I=0.01), name='seir')
    return d.stratify(age=[0, 5, 15, 30, 50, 65], split=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05])

cases = {
    'sir 100 days': (sir, dict(unit='day', stop=100, total_pop=10_000)),
    'sir 10 years daily': (sir, dict(unit='day', stop=3650, total_pop=10_000)),
    'seir x 6 ages, 365 days': (ages, dict(unit='day', stop=365, total_pop=100_000, networks=ss.contacts(age=np.ones((6, 6)) + 5*np.eye(6)))),
}

def timed(make, pars, how, solver='ode', n=5):
    best = np.inf
    for i in range(n):
        kw = dict(pars, verbose=0, solver=solver)
        if how == 'jax':
            kw['integrator'] = 'jax'
        sim = ss.Sim(make(), **kw).init()
        for mod in sim.declared:
            mod.engine.use_kernels = how != 'python'
        t = time.perf_counter()
        sim.run(profile=(how == 'step'))
        best = min(best, time.perf_counter() - t)
    return best*1e3, sim

for name, (make, pars) in cases.items():
    print(f'--- {name} ---')
    ref = None
    for solver in ['ode', 'binomial']:
        for how in (['python', 'step', 'all', 'jax'] if solver == 'ode' else ['python', 'all']):
            ms, sim = timed(make, pars, how, solver, n=3 if 'years' in name else 5)
            mod = sim.declared[0].name
            print(f'  {solver:9s} {how:7s} {ms:9.2f} ms   n_I max {sim.results[mod].n_I.max():12.4f}  npts={sim.t.npts} groups={sim.declared[0].engine.n_groups} comps={len(sim.declared[0].engine.comps)}')
