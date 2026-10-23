import numpy as np, starsim as ss
out = {}
# T2 with init=dict(I=100)
S, E, I, R = ss.states(4)
seir = ss.Disease(S >> E @ (3/7) * I, E >> I @ ss.days(5), I >> R @ ss.days(7), init=dict(I=100), name='seir')
sim = ss.Sim(seir, total_pop=1e6, stop=ss.days(365), verbose=0).run()
out['T2'] = int(sim.results.seir.n_I.argmax())
# T4 births and deaths
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ (10/14) * I, I >> R @ ss.days(14), init_prev=0.001, name='sir')
try:
    sim = ss.Sim(sir, total_pop=1e5, stop=ss.years(200), demographics=[ss.Births(birth_rate=20), ss.Deaths(death_rate=20)], verbose=1).run()
    r = sim.results.sir
    print(r.keys() if hasattr(r,'keys') else r)
except Exception as e:
    print('T4 err', type(e).__name__, e)
print(out)
