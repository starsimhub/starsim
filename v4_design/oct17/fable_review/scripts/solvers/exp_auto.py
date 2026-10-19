"""When does auto pick agents, and silently-meaned distributions; the 'stochastic' heuristic."""
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(total_pop=10_000, unit='day', stop=100, rand_seed=1, verbose=0.1)

def sir(beta=0.3, **extra):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir', **extra)

def run(label, *args, **kwargs):
    print(f'\n--- {label} ---')
    try:
        sim = ss.Sim(*args, **kwargs).init()
        print(f'  solver={sim.solver} reason={sim.solver_reason!r} notes={sim.notes}')
        return sim
    except Exception as e:
        print(f'  ERROR {type(e).__name__}: {str(e)[:300]}')

run('people=ss.People(1000)', sir(), people=ss.People(1000), unit='day', stop=100, verbose=0.1)
run('n_agents=1000', sir(), n_agents=1000, unit='day', stop=100, verbose=0.1)

print('\n=== 3. The stochastic heuristic: SIS endemic, pop 50k, 1 year in days ===')
sis = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> S', dur=10), init_prev=0.1, name='sis')
s = ss.Sim(sis, total_pop=50_000, unit='day', stop=365, solver='stochastic', verbose=0.1).init()
T = sc.timer(); s.run(); t = T.toc(output=True)
print(f'  chosen={s.solver} ({s.solver_reason}); actual events={s.results.sis.cum_infections.values[-1] + s.results.sis.cum_recovery.values[-1]:,.0f}; time={t:.2f}s')
s = ss.Sim(sis, total_pop=50_000, unit='day', stop=365, solver='tau', verbose=0).init()
T = sc.timer(); s.run(); t = T.toc(output=True)
print(f'  tau: time={t:.2f}s')
