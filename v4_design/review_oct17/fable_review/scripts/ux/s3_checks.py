import starsim as ss, sciris as sc, traceback
print('=== A. name inference: module vs function')
sir_mod = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
def make():
    sir_fn = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
    return sir_fn
sir_fn = make()
sim = ss.Sim(sir_mod, stop=10, verbose=0).init(); print('module-level name:', sir_mod.name)
sim = ss.Sim(sir_fn, stop=10, verbose=0).init(); print('function-local name:', sir_fn.name)
print('=== B. Sim repr n=')
print(ss.Sim(sir_mod, stop=10, verbose=0).init())
print(ss.Sim(sir_mod, stop=10, verbose=0, stratify=dict(age=[0,15,65])).init())
print('=== C. msim.plot small')
try:
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
    msim = ss.parallel(ss.Sim(sir, stop=50, solver='stochastic', verbose=0), n_runs=3)
    print(msim); print(msim.results)
    msim.plot()
    print('plot ok')
except Exception:
    print('\n'.join(traceback.format_exc().splitlines()[-3:]))
print('=== C2. msim.plot abm')
try:
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
    msim = ss.parallel(ss.Sim(sir, stop=50, solver='abm', n_agents=500, verbose=0), n_runs=3)
    msim.plot()
    print('plot ok')
except Exception:
    print('\n'.join(traceback.format_exc().splitlines()[-3:]))
print('=== D. reuse of run sim / fit.sim')
sim = ss.Sim(sir_mod, stop=10, verbose=0).run()
for label, s in [('same', sim), ('copy', sim.copy()), ('sc.dcp', sc.dcp(sim))]:
    try:
        ss.parallel(baseline=s, other=ss.Sim(sir_mod, stop=10, verbose=0), verbose=0); print(label, 'ok')
    except Exception as e:
        print(label, 'FAIL', type(e).__name__, str(e).splitlines()[0])
print('=== E. reuse of disease object in two sims')
try:
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
    a = ss.Sim(sir, stop=10, verbose=0).run()
    b = ss.Sim(sir, stop=10, verbose=0).run()
    print('two sims same disease ok', a.results.sir.n_I[-1], b.results.sir.n_I[-1])
except Exception as e:
    print('FAIL', type(e).__name__, str(e)[:300])
