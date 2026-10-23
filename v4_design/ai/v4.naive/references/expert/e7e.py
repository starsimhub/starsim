import numpy as np, starsim as ss
S, I, R = ss.states(3)
sir = ss.Disease(S >> I @ (2/7) * I, I >> R @ ss.days(7), init=dict(I=1), name='sir')
for solver in ['gillespie', 'binomial']:
    sim = ss.Sim(sir, total_pop=1000, stop=ss.days(365), solver=solver, verbose=0)
    msim = sim.run(n_runs=20)
    v = np.asarray(msim.results.sir.S_to_I.values)
    print(solver, 'total infections per run:', v.sum(axis=1).round())
    print(msim.runs.head(3).to_string())
    msim2 = ss.parallel(sim, n_runs=20)
    print(solver, 'parallel:', np.asarray(msim2.results.sir.S_to_I.values).sum(axis=1).round())
