import numpy as np, starsim as ss
# H2
S, I, R, D = ss.states(4)
infection = S >> I @ (2.5/7) * I
recover = I >> 0.98 * R @ ss.days(7)
die = I >> 0.02 * D @ ss.days(7)
sir = ss.Disease(infection, recover, die, init=dict(I=10), name='sir')
s = ss.Sim(sir, total_pop=1e5, stop=ss.days(365)).run()
print('H2 deaths', s.results.sir.die.sum(), 'n_D end', s.results.sir.n_D[-1])
# H3
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ (2/5) * I, I >> R @ ss.days(5), init=dict(I=10), name='sir')
s = ss.Sim(sir, total_pop=1e6, start='2025-01-01', stop='2025-12-31', verbose=0).run()
inf = s.results.sir.S_to_I
ser = inf.to_series()
print('H3 peak date', ser.idxmax(), 'value', ser.max())
# H4
s = ss.Sim(sir, total_pop=1e4, stop=ss.days(365), solver='gillespie', verbose=0)
sir2 = ss.Disease(S >> I @ (2/7) * I, I >> R @ ss.days(7), init=dict(I=10), name='sir')
msim = ss.Sim(sir2, total_pop=1e4, stop=ss.days(730), solver='gillespie', verbose=0).run(n_runs=100)
fs = 1 - np.asarray(msim.results.sir.n_S.values)[:, -1]/1e4
print('H4', np.percentile(fs, [5, 50, 95]))
