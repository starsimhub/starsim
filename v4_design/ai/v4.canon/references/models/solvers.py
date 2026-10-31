"""
One model, many solvers: run each of a set of canon models (chapters 2 to 4 and Vynnycky and White) under the ODE, chain binomial, Gillespie, and agents (1e4),
and compare the peak and final size of the first result with the ODE's. Stochastic solvers: the median over a few runs.
"""
import sys
import warnings
import numpy as np
import sciris as sc
import starsim as ss
from common import capture, series
import kr_ch2, kr_ch3, kr_ch4, vw
warnings.simplefilter('ignore')

cases = [ # (module, model name, result, overrides for the sim)
    (kr_ch2, 'KR 2.1 SIR', 'sir.n_I', dict(total_pop=1e4)),
    (kr_ch2, 'KR 2.6 SEIR demography', 'sir.n_I', dict(total_pop=1e5, stop=3*365)),
    (kr_ch3, 'KR 3.1 SIS risk groups', 'sis.n_I', dict(total_pop=1e4)),
    (kr_ch3, 'KR 3.3 SIR two ages (stratify)', 'sir.n_I', dict(total_pop=1e5, stop=2*365)),
    (kr_ch4, 'KR 4.3 two hosts (stratify)', 'hosts.n_I', dict()),
    (kr_ch4, 'KR 4.4 vector-borne', 'vb.n_IH', dict()),
    (vw, 'VW 2.1 SEIR closed', 'seir.n_I', dict()),
    (vw, 'VW 7.1 WAIFW two ages', 'sir.n_I', dict()),
    (vw, 'VW 8.1 STI core group', 'sti.n_I', dict(stop=5*365)),
]

def run(mod, name, key, over, solver, seeds=3):
    vals = []
    for seed in range(seeds if solver != 'ode' else 1):
        kw = dict(solver=solver, verbose=0, rand_seed=seed) | over
        if solver == 'abm':
            kw['n_agents'] = 10_000
        T = sc.timer()
        sim, _, err = capture(lambda: mod.models[name]['build']().new(**kw).run())
        if err:
            return f'error: {err[:160]}'
        y = series(sim, key)
        vals.append((y.max(), y.argmax(), y[-1], T.toc(output=True)))
    v = np.median(np.array(vals), axis=0)
    return v

if __name__ == '__main__':
    for mod, name, key, over in cases:
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        print(name)
        base = run(mod, name, key, over, 'ode')
        for solver in ['ode', 'binomial', 'gillespie', 'abm']:
            v = base if solver == 'ode' else run(mod, name, key, over, solver)
            if isinstance(v, str):
                print(f'  {solver:9s} {v}')
            else:
                print(f'  {solver:9s} peak {v[0]:9.1f} ({v[0]/base[0]-1:+6.1%}) at step {v[1]:5.0f}; final {v[2]:9.1f}; {v[3]:.1f} s')
