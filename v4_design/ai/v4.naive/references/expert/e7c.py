import numpy as np, starsim as ss
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ (2/7) * I, I >> R @ ss.days(7), init=dict(I=1), name='sir')
for solver in ['gillespie', 'binomial', 'abm']:
    sim = ss.Sim(sir, total_pop=1000, stop=ss.days(365), solver=solver, verbose=0)
    out = []
    for seed in range(20):
        s = sim.new(rand_seed=seed, verbose=0).run()
        out.append(s.results.sir.S_to_I.sum())
    print(solver, np.round(out))
msim = ss.Sim(sir, total_pop=1000, stop=ss.days(365), solver='gillespie', verbose=0).run(n_runs=20)
print([a for a in dir(msim) if not a.startswith('_')][:60])
