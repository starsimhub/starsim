import numpy as np, starsim as ss
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ (10/14) * I, I >> R @ ss.days(14), init_prev=0.001, name='sir')
mu = ss.peryear(1/50)
sim = ss.Sim(sir, total_pop=1e5, stop=ss.years(200), unit='day', demographics=[ss.birth(rate=mu), ss.death(rate=mu)]).run()
r = sim.results.sir
N = r.n_S + r.n_I + r.n_R
f = np.asarray(r.n_S/N)
n = len(f); print('n', n, 'T4 mean S frac years 180-200:', f[int(n*0.9):].mean(), 'N end', np.asarray(N)[-1])
