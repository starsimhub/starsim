"""Counts in rates across dimensions: is I.n aligned on shared dimensions, as the spec says, or only on space?"""
import numpy as np
import sciris as sc
import starsim as ss

def make(kind_kwargs, dim='risk'):
    S, I, R = ss.states(3)
    N = (S + I + R).n
    dis = ss.Disease(
        infection = S >> I @ 0.5 * I.n / N,  # written out: spec 06 says "the same thing written out for homogeneous mixing"
        recovery  = I >> R @ ss.days(10),
        init      = dict(I=ss.by(**{dim: [0.01, 0.0]})),  # infected only in the first level
        name      = 'dis',
    )
    dis.stratify(**{dim: ['a', 'b']}, **kind_kwargs)
    return dis

for label, kw, dim in [('plain (risk)', {}, 'risk'), ('space (patch, kind=space)', dict(kind='space'), 'patch'), ('age labels', {}, 'age')]:
    for solver in ['ode', 'abm']:
        extra = dict(n_agents=2000) if solver == 'abm' else {}
        sim = ss.Sim(make(kw, dim), solver=solver, unit='day', stop=60, total_pop=20_000, verbose=0, **extra).run()
        cum = sim.results.dis.cum_infection.by(dim).values[-1]
        print(f'{label:28s} {solver:4s} cum_infection by level: {np.round(cum)}  -> second level reached: {cum[1] > 0}')

print()
print('== the same with a bare I (exposure through the routes)')
def make_bare(kind_kwargs, dim):
    S, I, R = ss.states(3)
    dis = ss.Disease(infection=S >> I @ 0.5 * I, recovery=I >> R @ ss.days(10), init=dict(I=ss.by(**{dim: [0.01, 0.0]})), name='dis')
    return dis.stratify(**{dim: ['a', 'b']}, **kind_kwargs)
for label, kw, dim in [('plain (risk)', {}, 'risk'), ('space (patch, kind=space)', dict(kind='space'), 'patch')]:
    sim = ss.Sim(make_bare(kw, dim), unit='day', stop=60, total_pop=20_000, verbose=0).run()
    cum = sim.results.dis.cum_infection.by(dim).values[-1]
    print(f'{label:28s} ode  cum_infection by level: {np.round(cum)}  -> second level reached: {cum[1] > 0}')

print()
print('== describe() of the written-out form under a contact matrix: is the user told the matrix is ignored?')
S, I, R = ss.states(3)
N = (S + I + R).n
dis = ss.Disease(infection=S >> I @ 0.5 * I.n / N, recovery=I >> R @ ss.days(10), init_prev=0.01, name='dis')
sim = ss.Sim(dis, networks=ss.contacts([[10, 2], [2, 5]], age=[0, 15]), unit='day', stop=30, verbose=0).init()
print(sim.describe(output=True))
