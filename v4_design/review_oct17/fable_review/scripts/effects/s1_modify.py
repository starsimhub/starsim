""" ss.modify: object vs path targets, base vs effective, where= conditions, what_modifies """
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=20_000, stop=60)

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')

print('=== 1. object vs string path target give the same result?')
for solver in ['ode', 'abm']:
    sir = make()
    a = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, start=20), solver=solver, n_agents=2000, **kw).run()
    sir = make()
    b = ss.Sim(sir, interventions=ss.modify('sir.pars.beta', multiply=0.5, start=20), solver=solver, n_agents=2000, **kw).run()
    print(solver, 'object:', a.results.sir.cum_infection[-1], 'path:', b.results.sir.cum_infection[-1])

print('\n=== 2. network parameter: object vs path')
net = ss.RandomNet(n_contacts=5, name='school')
try:
    m = ss.modify(net.pars.n_contacts, multiply=0.2)
    print('object OK', m)
except Exception as E:
    print('object FAILS:', type(E).__name__, str(E)[:200])
m = ss.modify('school.pars.n_contacts', multiply=0.2)
print('path OK:', m.target)

print('\n=== 3. reading a modified array: base vs .effective (ABM)')
sir = make()
protect = ss.modify(sir.rel_sus, multiply=0.4, where=sir.R, name='protect')
sim = ss.Sim(sir, interventions=protect, solver='abm', n_agents=2000, **kw).run()
arr = sim.diseases.sir.rel_sus
R = sim.diseases.sir.R.uids[:3]
print('type:', type(arr).__name__)
print('rel_sus[R]       =', arr[R])
print('rel_sus.effective[R] =', arr.effective[R])
print('has .base?', hasattr(arr, 'base'))

print('\n=== 3b. the same in a compartmental run')
sir = make()
protect = ss.modify(sir.rel_sus, multiply=0.4, where=sir.R, name='protect')
sim = ss.Sim(sir, interventions=protect, solver='ode', **kw).run()
arr = sim.diseases.sir.rel_sus
print('type:', type(arr).__name__, '| has effective?', hasattr(arr, 'effective'))
try:
    print('effective:', arr.effective)
except Exception as E:
    print('effective FAILS:', type(E).__name__, str(E)[:200])

print('\n=== 4. what_modifies')
sim.what_modifies(sir.rel_sus)
sim.what_modifies('sir.pars.beta')

print('\n=== 5. bare numbers for start= in a dated sim vs unitless sim')
sir = make()
m = ss.modify(sir.pars.beta, multiply=0.5, start=2020.5)
sim = ss.Sim(sir, interventions=m, start='2020-01-01', stop='2021-01-01', solver='ode', verbose=0, total_pop=20_000).run()
print('dated sim, start=2020.5 ->', ss.connectors.to_time(sim, 2020.5), ' ti=', ss.connectors.to_ti(sim, 2020.5))
sir = make()
m = ss.modify(sir.pars.beta, multiply=0.5, start=30)
sim = ss.Sim(sir, interventions=m, solver='ode', **kw).run()
print('unitless sim, start=30 ->', ss.connectors.to_time(sim, 30), ' ti=', ss.connectors.to_ti(sim, 30))

print('\n=== 6. does the modifier\'s where= see state changes within a step? (cache cleared per step)')
sir = make()
mods = [ss.modify(sir.rel_sus, multiply=0.0, where=sir.R, name='a'), ss.modify(sir.rel_sus, multiply=0.5, where=sir.R, name='b')]
sim = ss.Sim(sir, interventions=mods, solver='abm', n_agents=2000, **kw).run()
print('two multipliers on same target, R:', sim.diseases.sir.rel_sus.effective[sim.diseases.sir.R.uids[:2]])

print('\n=== 7. two set= modifiers -> W502')
sir = make()
mods = [ss.modify(sir.pars.beta, set=0.1, start=10, name='a'), ss.modify(sir.pars.beta, set=0.2, start=10, name='b')]
sim = ss.Sim(sir, interventions=mods, solver='ode', **kw).run()

print('\n=== 8. a modifier on a quantity in connectors= vs interventions=: any difference?')
for key in ['connectors', 'interventions']:
    sir = make()
    sim = ss.Sim(sir, solver='abm', n_agents=2000, **{key: ss.modify(sir.rel_sus, multiply=0.3, where=sir.R)}, **kw).run()
    print(key, sim.results.sir.cum_infection[-1])

print('\n=== 9. a modifier with when= on an ABM, in-step: what does when= see?')
seen = []
sir = make()
def when(sim):
    seen.append(sim.ti)
    return True
sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, when=when), solver='abm', n_agents=2000, **kw).run()
print('when() called', len(seen), 'times over', sim.t.npts, 'steps; per step:', sc.objdict(sc.odict({k:seen.count(k) for k in set(seen)}).sorted()).__repr__()[:120])

print('\n=== 10. error for an unknown target')
sir = make()
try:
    ss.Sim(sir, interventions=ss.modify('sir.pars.betta', multiply=0.5), **kw).init()
except Exception as E:
    print(type(E).__name__, str(E)[:400])
