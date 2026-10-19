"""Chain-binomial stages fix, substeps, use_kernels equivalence/timing, try_functions cost, run-time failure of agent-only function under ode."""
import numpy as np
import sciris as sc
import starsim as ss

def seir(k=4, **extra):
    return ss.Disease(
        infection   = ss.transmission('S -> E', beta=0.5),
        progression = ss.progression('E -> I', dur=ss.gamma(a=k, scale=ss.days(4/k))),
        recovery    = ss.progression('I -> R', dur=ss.gamma(a=k, scale=ss.days(8/k))),
        init_prev = 0.01, name='seir', **extra)

base = dict(total_pop=100_000, unit='day', stop=120, rand_seed=1, verbose=0)

print('=== A. SEIR with Erlang(4) stages: ODE vs binomial at dt=1 (one multinomial per state per step) vs dt=0.25 vs substeps=4 ===')
ode = ss.Sim(seir(), solver='ode', **base).run()
ref = ode.results.seir.cum_infections.values[-1]
print(f'ode dt=1          cum_inf={ref:9.1f}  peak I={ode.results.seir.n_I.values.max():8.1f}')
for label,kw in [('binomial dt=1', dict(solver='binomial')), ('binomial dt=1 substeps=4', dict(solver='binomial', substeps=4)), ('binomial dt=0.25', dict(solver='binomial', dt=0.25)),
                 ('tau dt=1', dict(solver='tau')), ('sde dt=1', dict(solver='sde')), ('sde dt=1 substeps=4', dict(solver='sde', substeps=4)), ('ode euler substeps=1', dict(solver='ode', integrator='euler', substeps=1)), ('ode substeps=1 (rk4)', dict(solver='ode', substeps=1))]:
    s = ss.Sim(seir(), **(base | kw)).run()
    r = s.results.seir
    print(f'{label:26s} cum_inf={r.cum_infections.values[-1]:9.1f} ({100*(r.cum_infections.values[-1]/ref-1):+.1f}%) peak I={r.n_I.values.max():8.1f}  notes={s.notes}')

print('\n=== B. use_kernels True vs False: same numbers? timing ===')
for solver in ['ode', 'binomial', 'gillespie', 'tau', 'sde']:
    outs, times = [], []
    for kernels in [True, False]:
        s = ss.Sim(seir(), solver=solver, **(base | dict(total_pop=20_000))).init()
        s.diseases.seir.engine.use_kernels = kernels
        T = sc.timer(); s.run(); times.append(T.toc(output=True))
        outs.append(s.results.seir.cum_infections.values[-1])
    print(f'{solver:10s} kernels={outs[0]:9.1f} ({times[0]:.2f}s)  python={outs[1]:9.1f} ({times[1]:.2f}s)  identical={np.isclose(outs[0], outs[1])}')

print('\n=== C. try_functions cost: init time with and without a function in a rate ===')
def fn(sim):
    return 0.1
for label,dis in [('plain', seir()), ('with function', ss.Disease(infection=ss.transmission('S -> E', beta=0.5), progression=ss.progression('E -> I', dur=4), recovery=ss.progression('I -> R', rate=fn), init_prev=0.01, name='seir'))]:
    T = sc.timer(); s = ss.Sim(dis, **base).init(); t = T.toc(output=True)
    print(f'{label:15s} init {t:.3f}s solver={s.solver}')

print('\n=== D. agent-only function under requested ode: when does it fail? ===')
def partners(sim):
    return 0.1*sim.people.n_partners
s = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=partners), init_prev=0.01, name='sir'), solver='ode', **base)
try:
    s.run()
    print('ran without error!')
except Exception as e:
    print(f'ERROR at run: {type(e).__name__}: {str(e)[:300]}')

print('\n=== E. sim.run(solver=) after init ===')
s = ss.Sim(seir(), **base).init()
try:
    s.run(solver='binomial')
except Exception as e:
    print(f'ERROR: {e}')
