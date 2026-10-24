import starsim as ss, sciris as sc, traceback, pylab as pl
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
msim = ss.parallel(ss.Sim(sir, stop=50, solver='stochastic', verbose=0), n_runs=3)
print(msim.sims, len(msim.sims))
r = msim.results.sir.n_I
print(type(r), getattr(r,'shape',None), r.values.shape if hasattr(r,'values') else None)
msim.plot(); pl.savefig('s4_msim.png')
print('=== naming via alias')
x = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
y = x
ss.Sim(y, stop=5, verbose=0).init(); print('alias name:', x.name)
z = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='flu')
ss.Sim(z, stop=5, verbose=0).init(); print('explicit name:', z.name)
d = dict(sir=ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01))
ss.Sim(d['sir'], stop=5, verbose=0).init(); print('dict name:', d['sir'].name)
print('=== rerun a run sim')
sim = ss.Sim(sir, stop=10, verbose=0).run()
for how in ['sim.init(force=True)', 'sim.init(reset=True)', 'sim.reset()', 'sim.run(reset=True)']:
    try:
        exec(how); print(how, 'ok')
    except Exception as e:
        print(how, 'FAIL', type(e).__name__, str(e).splitlines()[0][:120])
