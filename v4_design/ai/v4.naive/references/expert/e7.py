import numpy as np, starsim as ss
# T10
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ ss.perweek(1.5) * I, I >> R @ ss.weeks(2), init=dict(I=10), name='sir')
s = ss.Sim(sir, total_pop=1e4, stop=ss.years(1), dt=ss.days(1), verbose=1).run()
print('T10', 1 - s.results.sir.n_S[-1]/1e4)
# T9
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ (2/7) * I, I >> R @ ss.days(7), init=dict(I=1), name='sir')
sim = ss.Sim(sir, total_pop=1000, stop=ss.days(365), solver='gillespie', verbose=0)
msim = sim.run(n_runs=200)
fs = [1 - m.results.sir.n_S[-1]/1000 for m in msim.sims]
print('T9', np.mean(np.array(fs) > 0.1))
