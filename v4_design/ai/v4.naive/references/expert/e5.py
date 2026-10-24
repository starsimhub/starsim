import numpy as np, starsim as ss
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ 0.02 * I, I >> R @ ss.days(7), init_prev=1e-4, name='sir')
sir.stratify(age=[0, 15], split=[0.3, 0.7])
contacts = ss.contacts([[10, 3], [2, 6]], bands=[0, 15])
sim = ss.Sim(sir, networks=contacts, total_pop=1e6, stop=ss.days(365)).run()
r = sim.results.sir
for g in ['0-15', '15+']:
    try:
        S0 = r.n_S.select(age=g)[0]; S1 = r.n_S.select(age=g)[-1]
        Nt = (r.n_S+r.n_I+r.n_R).select(age=g)[0]
        print(g, 'attack', (Nt - S1)/Nt, 'N', Nt)
    except Exception as e:
        print('err', type(e).__name__, e)
print(r.n_S.by('age'))
