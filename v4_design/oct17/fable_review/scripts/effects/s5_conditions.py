""" Conditions: lazy expressions vs functions of the sim """
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day', total_pop=20_000, stop=60)

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')

print('=== 1. a lazy condition on age needs a People object')
sir = make()
cond = ss.People(1).age < 5
print('cond:', repr(cond), type(cond).__name__)
sir = make().stratify(age=[0, 5, 65])
sim = ss.Sim(sir, interventions=ss.modify(sir.rel_sus, multiply=0.0, where=ss.People(1).age < 5), solver='ode', **kw).run()
print('ode with age bin edge at 5: ran')
sir = make().stratify(age=[0, 15, 65])
try:
    ss.Sim(sir, interventions=ss.modify(sir.rel_sus, multiply=0.0, where=ss.People(1).age < 5), solver='ode', **kw).run()
except Exception as E:
    print('ode with bin edge not at 5:', type(E).__name__, str(E)[:300])

print('\n=== 2. lambda vs lazy: same answer?')
for form, cond in [('lazy', None), ('lambda', lambda sim: sim.diseases.sir.R)]:
    sir = make()
    where = sir.R if cond is None else cond
    sim = ss.Sim(sir, interventions=ss.modify(sir.rel_sus, multiply=0.3, where=where), solver='abm', n_agents=2000, **kw).run()
    print(form, sim.results.sir.cum_infection[-1])
for form, cond in [('lazy', None), ('lambda', lambda sim: sim.diseases.sir.R)]:
    sir = make()
    where = sir.R if cond is None else cond
    try:
        sim = ss.Sim(sir, interventions=ss.modify(sir.rel_sus, multiply=0.3, where=where), solver='ode', **kw).run()
        print(form, 'ode', sim.results.sir.cum_infection[-1])
    except Exception as E:
        print(form, 'ode FAILS:', type(E).__name__, str(E)[:300])

print('\n=== 3. things that look lazy but are not')
sir = make()
print('sir.S:', type(sir.S).__name__, '| sir.S & sir.R:', repr(sir.S & sir.R) if hasattr(sir.S, '__and__') else 'n/a')
print('~sir.R:', repr(~sir.R))
print('sir.R == True?', repr(sir.R == True))
try:
    if sir.R: pass
except TypeError as E:
    print('if cond: ->', str(E)[:120])
try:
    print('arith People(1):', repr(ss.People(1).age - 5 > 60))
except Exception as E: print('arith People(1) FAILS:', type(E).__name__, E)
try:
    print('arith People(100):', repr(ss.People(100).age - 5 > 60))
except Exception as E: print('arith People(100) FAILS:', type(E).__name__, E)
try:
    print('arith sir.rel_sus*2:', repr(sir.rel_sus*2 > 1))
except Exception as E: print('arith rel_sus FAILS:', type(E).__name__, E)
print('sir.infection as a condition (an event):', repr(ss.conditions.as_cond(sir.infection)))

print('\n=== 4. an initialized array in step(): immediate')
sir = make()
sim = ss.Sim(sir, solver='abm', n_agents=200, **kw).init()
print(type(sim.diseases.sir.R & sim.diseases.sir.S).__name__, type(sim.people.age > 5).__name__)

print('\n=== 5. a condition on one disease in another\'s modifier, compartmental: crossing')
hiv = ss.Disease(infection=ss.transmission('S -> I', beta=0.1), name='hiv', init_prev=0.05)
syph = ss.Disease(infection=ss.transmission('S -> I', beta=0.2), recovery=ss.progression('I -> S', dur=30), name='syph', init_prev=0.05)
mods = [ss.modify(hiv.rel_trans, multiply=2.0, where=syph.I), ss.modify(syph.rel_sus, multiply=1.5, where=hiv.I)]
for solver in ['ode', 'abm']:
    sim = ss.Sim([hiv, syph], connectors=mods, solver=solver, n_agents=2000, **kw).run()
    print(solver, 'hiv n_I', sim.results.hiv.n_I[-1], 'syph n_I', sim.results.syph.n_I[-1], ' groups:', sim.n_groups)
