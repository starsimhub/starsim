"""Crossing: two diseases coupled by a modifier with where= on the other's state."""
import numpy as np
import sciris as sc
import starsim as ss

def make(k=1, couple=True, **kw):
    hiv = ss.Disease(infection=ss.transmission('S -> I', beta=0.1), death=ss.progression('I -> D', dur=ss.years(10)), init_prev=0.05, name='hiv')
    dur = ss.months(6) if k == 1 else ss.gamma(a=k, scale=ss.months(6/k))
    syph = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=dur), waning=ss.progression('R -> S', dur=ss.years(2)), init_prev=0.02, name='syph')
    conn = [ss.modify(hiv.rel_trans, multiply=3, where=syph.I, name='hiv_trans')] if couple else []
    base = dict(total_pop=10_000, unit='month', stop=ss.years(5), rand_seed=1, verbose=0.1)
    return ss.Sim([hiv, syph], connectors=conn, **(base | kw))

for label,kw in [('no coupling, ode', dict(couple=False)), ('coupled, ode', dict()),
                 ('coupled, ode, syph with 4-stage Erlang', dict(k=4)),
                 ('coupled, ode, 4 stages, stratified by age x3 and sex', dict(k=4, stratify=dict(age=[0, 15, 50], sex=['f', 'm']))),
                 ('coupled, binomial', dict(solver='binomial')),
                 ('coupled, gillespie', dict(solver='gillespie')),
                 ('coupled, abm', dict(solver='abm'))]:
    print(f'\n--- {label} ---')
    try:
        sim = make(**kw).init()
        T = sc.timer(); sim.run(); t = T.toc(output=True)
        eng = sim.diseases.hiv.engine
        ncomp = getattr(eng, 'n_comps', None)
        print(f'  solver={sim.solver} crossings={sim.crossings} n_groups={sim.n_groups} hiv.n_comps={ncomp} syph.n_comps={getattr(sim.diseases.syph.engine, "n_comps", None)} strats={[s.name for s in sim.strats]} time={t:.2f}s kernel_only={sim.kernel_only() if sim.solver != "abm" else None}')
        print(f'  hiv cum_inf={sim.results.hiv.cum_infections.values[-1]:.1f}  syph cum_inf={sim.results.syph.cum_infections.values[-1]:.1f}  notes={sim.notes}')
    except Exception as e:
        print(f'  ERROR {type(e).__name__}: {str(e)[:500]}')

print('\n--- describe() of the coupled ode sim ---')
sim = make(verbose=0).init()
sim.describe()
print('\n--- string where= ---')
try:
    hiv = ss.Disease(infection=ss.transmission('S -> I', beta=0.1), death=ss.progression('I -> D', dur=ss.years(10)), init_prev=0.05, name='hiv')
    ss.Sim([hiv], connectors=ss.modify('hiv.rel_trans', multiply=3, where='hiv.I'), total_pop=1000, unit='month', stop=12, verbose=0).run()
except Exception as e:
    print(f'  ERROR {type(e).__name__}: {str(e)[:300]}')
