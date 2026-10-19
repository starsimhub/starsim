"""Run the same SIR under every solver; print the notes and compare outputs."""
import numpy as np
import sciris as sc
import starsim as ss

def make():
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=0.3),
        recovery  = ss.progression('I -> R', dur=10),
        init_prev = 0.01,
        name = 'sir',
    )

results = {}
for solver in ['auto', 'ode', 'gillespie', 'ctmc', 'tau', 'binomial', 'sde', 'abm', 'stochastic']:
    print(f'\n===== solver={solver!r} =====')
    sim = ss.Sim(make(), total_pop=10_000, unit='day', stop=100, solver=solver, rand_seed=1, verbose=0.1)
    T = sc.timer()
    sim.run()
    t = T.toc(output=True)
    r = sim.results.sir
    results[solver] = r
    print(f'  chosen={sim.solver} integrator={getattr(sim, "integrator", None)} time={t:.2f}s')
    print(f'  notes={sim.notes}')
    print(f'  peak I={r.n_I.values.max():.1f} at t={r.n_I.timevec[r.n_I.values.argmax()]} ; final R={r.n_R.values[-1]:.1f} ; cum_inf={r.cum_infections.values[-1]:.1f}')

print('\n===== Summary table =====')
ode = results['ode']
for k,r in results.items():
    print(f'{k:11s} final R={r.n_R.values[-1]:9.1f}  diff vs ode={r.n_R.values[-1]-ode.n_R.values[-1]:+8.1f} ({100*(r.n_R.values[-1]/ode.n_R.values[-1]-1):+.1f}%)')
