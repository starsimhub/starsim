import sciris as sc, starsim as ss, numpy as np
def fn(sim):
    return 0.1 + 0.0*sim.t.now('year')
kw = dict(total_pop=10_000, unit='day', stop=200, rand_seed=1, verbose=0)
for label, rate in [('plain number', 0.1), ('python function', fn), ('seasonal', ss.seasonal(0.1, amplitude=0.3, peak=1))]:
    for solver in ['ode', 'binomial', 'gillespie']:
        d = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', rate=rate), init_prev=0.01, name='sir')
        s = ss.Sim(d, solver=solver, **kw).init()
        T = sc.timer(); s.run(); t = T.toc(output=True)
        e = s.diseases.sir.engine
        print(f'{label:16s} {solver:10s} program={"built" if e.program else e.program!r} kernel_only={s.kernel_only()} time={t:.3f}s cum={s.results.sir.cum_infections.values[-1]:.0f}')
