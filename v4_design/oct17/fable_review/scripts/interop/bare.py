import numpy as np, starsim as ss
def make(unit, typed=False):
    S, E, I, R = ss.states(4)
    R0 = ss.par(2.5, name='R0'); gamma = ss.par(ss.perday(0.2) if typed else 0.2, name='gamma')
    seir = ss.Disease(infection=S >> E @ R0*gamma*I, onset=E >> I @ (ss.perday(0.25) if typed else 0.25), recovery=I >> R @ gamma, init=dict(E=5), name='seir')
    sim = ss.Sim(seir, total_pop=10000, unit=unit, start='2020-02-01', stop='2020-08-31', solver='ode', verbose=0).run()
    return sim
for typed in [False, True]:
    d = make('day', typed); w = make('week', typed)
    print(f'typed={typed}: peak n_I day={d.results.seir.n_I.values.max():.1f} week={w.results.seir.n_I.values.max():.1f}')
    print('   R0 in spec:', [p['value'] for p in d.to_spec()['modules'][0]['pars'] if p['name']=='R0'])
