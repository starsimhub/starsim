"""A two-dimension model (age x place) and a strain dimension: what does the user see?"""
import numpy as np
import sciris as sc
import starsim as ss

print('== named data offline?')
for call in ["ss.geo('Kenya', level=1)", "ss.contacts('Kenya')"]:
    try:
        obj = eval(call)
        print(call, '->', repr(obj)[:120])
    except Exception as e:
        print(call, 'ERR', type(e).__name__, str(e)[:200])

print()
print('== age x place SIR, aging, gravity commuting, Prem contacts aggregated')
towns = ss.geo(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init=dict(I=ss.by(patch=[20, 0, 0])), name='sir')
sir.stratify(patch=towns, mobility=ss.gravity(k=0.01))
sim = ss.Sim(sir, networks=ss.contacts('Kenya'), stratify=dict(age=[0, 15, 65]), unit='day', stop=120, total_pop=100_000, verbose=0)
sim.init()
print(sim.describe(output=True))
sim.run()
res = sim.results.sir
print('cum_infection by patch:', np.round(res.cum_infection.by('patch').values[-1]))
print('cum_infection by age:', np.round(res.cum_infection.by('age').values[-1]))
print('select(age=0, patch="south"):', round(float(res.cum_infection.select(age=0, patch='south').values[-1])))
print('n_groups', sim.n_groups, 'group_shape', sim.group_shape)

print()
print('== the same but with the split as ss.by over age x patch and adjust on the disease by age')
sir2 = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init=dict(I=ss.by(patch=[20, 0, 0])), name='sir')
pop = ss.by([[20_000, 10_000, 5_000], [25_000, 15_000, 10_000], [5_000, 5_000, 5_000]], dims=['age', 'patch'])
sir2.stratify(patch=towns, split=pop, mobility=ss.gravity(k=0.01))
try:
    sir2.stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1.0, 0.5]})
    sim2 = ss.Sim(sir2, networks=ss.contacts('Kenya'), stratify=dict(age=[0, 15, 65]), unit='day', stop=20, total_pop=100_000, verbose=0).init()
    print('OK: disease and sim both by age; splits:', [np.round(s.split, 3) for s in sim2.strats])
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])

print()
print('== order dependence: stratify patch with split=ss.by(age x patch) BEFORE the model knows age')
sir3 = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), init_prev=0.01, name='sir')
sir3.stratify(patch=towns, split=pop)
try:
    ss.Sim(sir3, unit='day', stop=5, verbose=0).init()
    print('no error although age is not a dimension')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])

print()
print('== strain dimension on an SEIR')
seir = ss.Disease(infection=ss.transmission('S -> E', beta=0.3, source='I'), onset=ss.progression('E -> I', dur=3), recovery=ss.progression('I -> R', dur=7), init_prev=0.01, name='flu')
seir.stratify(strain=['H1N1', 'H3N2'], immunity=[[1.0, 0.3], [0.3, 1.0]], adjust={'infection': [1.0, 1.3]})
sim = ss.Sim(seir, unit='day', stop=100, total_pop=50_000, verbose=0).init()
print(sim.describe(output=True))
sim.run()
print('n_I by strain at end:', np.round(sim.results.flu.n_I.by('strain').values[-1]))
print()
print('== a strain-like name not in the list')
try:
    d = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=7), init_prev=0.01, name='x').stratify(lineage=['A', 'B'])
    print('lineage kind:', d.graph.strats[0].kind, 'states:', d.graph.strats[0].states)
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])
try:
    d = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=7), init_prev=0.01, name='x').stratify(lineage=['A', 'B'], immunity=np.eye(2))
except Exception as e:
    print('lineage + immunity ERR:', type(e).__name__, str(e)[:300])
print()
print('== MixingPools() with no arguments (docstring says default <15/15+)')
try:
    ss.MixingPools()
    print('OK')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:200])
