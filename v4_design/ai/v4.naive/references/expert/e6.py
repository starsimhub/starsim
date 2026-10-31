import numpy as np, starsim as ss
def mk():
    S, I, R = ss.states(3)
    return ss.Disease(S >> I @ (2/7) * I, I >> R @ ss.days(7), init=dict(I=100), name='sir')
kw = dict(total_pop=1e6, stop=ss.days(365), verbose=0)
base = ss.Sim(mk(), **kw).run().results.sir.S_to_I.sum()
for leaky in [False, True]:
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.9, leaky=leaky), coverage=0.3, at=[0])
    s = ss.Sim(mk(), interventions=vx, **kw).run()
    print('T6 leaky', leaky, 'base', base, 'averted', base - s.results.sir.S_to_I.sum())
# T7
S, I, R = ss.states(3)
beta = ss.par(2.5/7)
sir = ss.Disease(S >> I @ beta * I, I >> R @ ss.days(7), init=dict(I=100), name='sir')
cl = ss.modify(sir.pars.beta, multiply=0.6, start=ss.days(30), stop=ss.days(90))
s = ss.Sim(sir, interventions=cl, **kw).run()
print('T7', s.results.sir.S_to_I.sum())
