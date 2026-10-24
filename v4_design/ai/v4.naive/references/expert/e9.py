import numpy as np, starsim as ss
# H1
def mk():
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ (2.5/7) * I, I >> R @ ss.days(7), init=dict(I=100), name='sir')
    sir.stratify(age=[0, 5, 15], split=[0.1, 0.2, 0.7])
    return sir
kw = dict(total_pop=1e6, start='2025-03-01', stop='2025-12-31', unit='day', verbose=0)
base = ss.Sim(mk(), **kw).run().results.sir.S_to_I.sum()
sir = mk()
vx = ss.vaccinate(ss.Vaccine(efficacy=1.0, leaky=False), where=sir.age < 5, coverage=0.8, at=['2025-06-01'])
s = ss.Sim(sir, interventions=vx, **kw).run()
print('H1 averted', base - s.results.sir.S_to_I.sum(), 'base', base)
# H2
S, I, R, D = ss.states(4)
inf = S >> I @ (2.5/7) * I
rec = I >> R @ ss.days(7)
sir = ss.Disease(inf, I >> [R, D] @ ss.days(7) if False else rec, init=dict(I=10), name='sir')
