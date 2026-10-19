"""Sim-level stratify: adjust across diseases, strain at the sim level, where= at the sim level, and the written-out /N form with a contact matrix"""
import numpy as np
import sciris as sc
import starsim as ss

def sir(name='sir'):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name=name)
def sis(name='sis'):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), clearance=ss.progression('I -> S', dur=10), init_prev=0.01, name=name)

print('== A. sim.stratify(age=..., adjust={recovery: ...}) with two diseases, one of which has no "recovery"')
try:
    sim = ss.Sim([sir(), sis()], stratify=dict(age=[0, 15, 65], adjust={'recovery': [2, 1, 0.5]}), stop=5, verbose=0).init()
    print('no error; sir adjust:', {k: v for k, v in sim.diseases.sir.engine.adjust.items() if v is not None}.keys(), '| sis adjust:', {k: v for k, v in sim.diseases.sis.engine.adjust.items() if v is not None}.keys())
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])
print('-- and a typo at the sim level')
try:
    sim = ss.Sim(sir(), stratify=dict(age=[0, 15, 65], adjust={'recovry': [2, 1, 0.5]}), stop=5, verbose=0).init()
    print('no error for a misspelled transition at the sim level')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])
print('-- the same typo on the disease')
try:
    sir().stratify(age=[0, 15, 65], adjust={'recovry': [2, 1, 0.5]})
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:200])

print('== B. strain at the sim level: which states get it?')
sim = ss.Sim(sir(), stratify=dict(strain=['A', 'B']), stop=5, verbose=0).init()
print('sim strat states:', sim.strats[0].states, 'kind', sim.strats[0].kind)
print('n_S has strata?', getattr(sim.results.sir.n_S, 'strata', None) is not None)
d = sir(); d.stratify(strain=['A', 'B'])
print('disease-level strat states:', d.graph.strats[0].states)

print('== C. describe() of the written-out /N form under a contact matrix')
S, I, R = ss.states(3)
N = (S + I + R).n
dis = ss.Disease(infection=S >> I @ 0.5 * I.n / N, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name='dis')
sim = ss.Sim(dis, networks=ss.contacts([[10, 2], [2, 5]], age=[0, 15]), unit='day', stop=30, verbose=0).init()
print(sim.describe(output=True))

print('== D. the bare form, same network, for comparison')
dis = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='dis')
sim = ss.Sim(dis, networks=ss.contacts([[10, 2], [2, 5]], age=[0, 15]), unit='day', stop=30, verbose=0).init()
print(sim.describe(output=True))

print('== E. ss.contacts keyword overload: age= as edges vs age= as matrix')
m = ss.contacts([[10, 2], [2, 5]], age=[0, 15])
print('contacts(matrix, age=edges):', type(m).__name__, list(m.pars.src.keys()), getattr(m, 'dims', None))
m = ss.contacts(age=[[10, 2], [2, 5]])
print('contacts(age=matrix):      ', type(m).__name__, list(m.pars.src.keys()), list(m.dims.keys()))
try:
    ss.contacts(age=[0, 15])
except Exception as e:
    print('contacts(age=[0, 15]) ERR:', type(e).__name__, str(e)[:160])

print('== F. ss.by(age=..., sex=...) product form: arithmetic refused')
b = ss.by(age=[1, 2], sex=[1, 0.5])
try:
    print(2*b)
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:200])

print('== G. ss.transfer as intervention returns a Delivery')
t = ss.transfer(vaccinated=True, coverage=0.7, years=[2021])
print(type(t).__name__, t)
t2 = ss.transfer(vaccinated=True, rate=0.1)
print(type(t2).__name__, t2)
