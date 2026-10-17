"""States that are not people: cholera and friends."""
import starsim as ss, numpy as np, sciris as sc

beta, K, xi = 0.5, 1e5, 10
S, I, R, W = ss.states(4)
cholera = ss.Disease(
    infection = S >> I @ beta * W.n / (W.n + K),
    recovery  = I >> R @ ss.days(5),
    shedding  = ss.null >> W @ xi * I.n,
    decay     = W >> ss.null @ ss.days(30),
    init      = dict(I=0.001),
)
cholera.describe()
cholera.equations()
sim = ss.Sim(cholera, total_pop=100_000, unit='day', stop=200, verbose=0).run()
print('solver', sim.pars.solver, 'peak I', sim.results.cholera.n_I.max(), 'peak W', sim.results.cholera.n_W.max())
sim2 = ss.Sim(sc.dcp(cholera), total_pop=100_000, n_agents=2000, unit='day', stop=200, verbose=0).run()
print('agents: solver', sim2.pars.solver, 'peak I', sim2.results.cholera.n_I.max(), 'peak W', sim2.results.cholera.n_W.max())

print('\n--- bare W in a rate')
S, I, R, W = ss.states(4)
try:
    ss.Disease(infection=S >> I @ beta * W, recovery=I >> R @ ss.days(5), shedding=ss.null >> W @ xi*I.n, decay=W >> ss.null @ ss.days(30))
except Exception as e: print('ERROR', e)

print('\n--- maternal immunity M >> S: M is people')
M, S, I, R = ss.states(4)
d = ss.Disease(waning=M >> S @ ss.days(180), infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), birth=ss.birth('-> M', rate=ss.peryear(0.03)), init=dict(I=0.01, M=0.1))
d.describe()
print('first state:', d.graph.first.name, '| people:', [s.name for s in d.graph.people_states])

print('\n--- order matters: the first state created is the people anchor. Create W first:')
W, S, I, R = ss.states('W S I R')
d = ss.Disease(infection=S >> I @ beta * W.n / (W.n + K), recovery=I >> R @ ss.days(5), shedding=ss.null >> W @ xi*I.n, decay=W >> ss.null @ ss.days(30), init=dict(I=0.001))
d.describe()
print('people:', [s.name for s in d.graph.people_states], 'counts:', [s.name for s in d.graph.count_states])

print('\n--- a state only reached by an inflow and leaving by an outflow, but meant as people (immigrants holding pen)')
S, I, R, Q = ss.states('S I R Q')
d = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), arrivals=ss.immigration('-> Q', rate=ss.perday(5)), leaving=Q >> ss.null @ ss.days(14), init_prev=0.01)
d.describe()

print('\n--- vector model where mosquitoes infect via bare state')
Sh, Ih, Sv, Iv = ss.states('Sh Ih Sv Iv')
try:
    ss.Disease(inf_h=Sh >> Ih @ 0.3*Iv, inf_v=Sv >> Iv @ 0.3*Ih, rec=Ih >> Sh @ ss.days(100), init=dict(Ih=0.01, Sv=1000))
except Exception as e: print('ERROR', str(e)[:300])

print('\n--- init semantics: 1 vs 1.0 vs counts')
for init in [dict(I=1), dict(I=1.0), dict(I=50), dict(I=0.5)]:
    S, I, R = ss.states('S I R')
    d = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init=init, name='sir')
    sim = ss.Sim(d, total_pop=1000, unit='day', stop=1, verbose=0).run()
    print(init, '-> n_I[0] =', sim.results.sir.n_I[0], 'n_S[0] =', sim.results.sir.n_S[0])
print('agents, init=dict(I=1):')
S, I, R = ss.states('S I R')
d = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init=dict(I=1), name='sir')
sim = ss.Sim(d, n_agents=1000, unit='day', stop=1, verbose=0).run()
print('n_I[0] =', sim.results.sir.n_I[0])

print('\n--- init_prev vs init(I=): the same?')
S, I, R = ss.states('S I R')
d1 = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')
S, I, R = ss.states('S I R')
d2 = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name='sir')
r1 = ss.Sim(d1, unit='day', stop=30, verbose=0).run().results.sir.n_I[:]
r2 = ss.Sim(d2, unit='day', stop=30, verbose=0).run().results.sir.n_I[:]
print('identical:', np.allclose(r1, r2))

print('\n--- init_prev on a disease with no transmission (cholera)')
try:
    S, I, R, W = ss.states(4)
    c = ss.Disease(infection=S >> I @ beta*W.n/(W.n+K), recovery=I >> R @ ss.days(5), shedding=ss.null >> W @ xi*I.n, decay=W >> ss.null @ ss.days(30), init_prev=0.01)
    ss.Sim(c, unit='day', stop=5, verbose=0).run()
except Exception as e: print('ERROR', str(e)[:300])

print('\n--- both init_prev and init=dict(I=...)')
S, I, R = ss.states('S I R')
d = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, init=dict(I=0.05, R=0.2), name='sir')
sim = ss.Sim(d, total_pop=1000, unit='day', stop=1, verbose=0).run()
print('n_I[0]:', sim.results.sir.n_I[0], 'n_R[0]:', sim.results.sir.n_R[0])
