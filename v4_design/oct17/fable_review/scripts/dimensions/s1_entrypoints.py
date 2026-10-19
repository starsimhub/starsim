"""Entry points and option handling for stratify"""
import numpy as np
import sciris as sc
import starsim as ss

def sir():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')

print('== A. ss.Sim(stratify=dict(age=..., states=[...])): is states= honored at the sim level?')
try:
    sim = ss.Sim(sir(), stratify=dict(age=[0, 15, 65], states=['S', 'I']), stop=5, verbose=0).init()
    print('dims:', [(s.name, s.labels, s.states) for s in sim.strats])
except Exception as e:
    print('ERR', type(e).__name__, e)

print('== A2. ss.Sim(stratify=dict(strain=[...], immunity=...))')
try:
    sim = ss.Sim(sir(), stratify=dict(strain=['A', 'B'], immunity=[[1, 0.3], [0.3, 1]]), stop=5, verbose=0).init()
    print('dims:', [(s.name, s.labels, s.kind) for s in sim.strats])
except Exception as e:
    print('ERR', type(e).__name__, e)

print('== B. place= without kind: what kind?')
d = sir().stratify(place=['a', 'b'])
print('place kind:', d.graph.strats[0].kind)
d = sir().stratify(herd=['a', 'b'])
print('herd kind:', d.graph.strats[0].kind)
d = sir().stratify(location=['a', 'b'])
print('location kind:', d.graph.strats[0].kind)
try:
    sir().stratify(place=['a', 'b'], mobility=ss.Mobility(matrix=[[0, 0.1], [0.1, 0]]))
except Exception as e:
    print('place + mobility ERR:', type(e).__name__, e)
d = sir().stratify(place=ss.geo(names=['a', 'b']))
print('place=ss.geo(names=...) kind:', d.graph.strats[0].kind, 'labels', d.graph.strats[0].labels)

print('== C. age given as labels: kind and attr')
d = sir().stratify(age=['young', 'old'])
print('age labels kind:', d.graph.strats[0].kind, 'attr:', d.graph.strats[0].attr, 'guessed:', dict(d.graph.strats[0].guessed))

print('== D. sim.stratify after sim exists; disease.stratify same dim same levels')
s = sir()
s.stratify(age=[0, 15, 65], adjust={'recovery': [1.5, 1, 0.5]})
sim = ss.Sim(s, stop=5, verbose=0)
sim.stratify(age=[0, 15, 65])
sim.init()
print('sim dims:', [(st.name, st.labels) for st in sim.strats], 'module dims:', [(st.name, dict(st.adjust)) for st in s.graph.strats])

print('== E. stratify twice in place -> E305; inplace=False copy')
s = sir()
s.stratify(age=[0, 15, 65])
try:
    s.stratify(age=[0, 15, 65])
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:160])
s2 = sir()
s3 = s2.stratify(age=[0, 15, 65], inplace=False)
print('s2 strats', len(s2.graph.strats), 's3 strats', len(s3.graph.strats))

print('== F. four spellings')
a = sir(); a.stratify(age=[0, 15, 65])
b = ss.stratify(sir(), age=[0, 15, 65])
c = ss.Sim(sir(), stratify=dict(age=[0, 15, 65]), stop=5, verbose=0).init()
d = ss.Sim(sir(), stop=5, verbose=0).stratify(age=[0, 15, 65]).init()
print('module:', [s.name for s in a.graph.strats], [s.name for s in b.graph.strats], 'sim:', [s.name for s in c.strats], [s.name for s in d.strats])
print('sim-level strat visible on the module graph?', [s.name for s in c.diseases.sir.graph.strats])

print('== G. multiple dims in one call with adjust -> error message')
try:
    sir().stratify(age=[0, 15], sex=['f', 'm'], adjust={'recovery': [1, 2]})
except Exception as e:
    print('ERR', type(e).__name__, e)

print('== H. dependence= on MixingPool')
try:
    ss.Sim(sir(), networks=ss.MixingPool(n_contacts=10, dependence='density'), stop=5, verbose=0).init()
    print('accepted')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:200])

print('== I. MFNet under solver=ode')
try:
    ss.Sim(sir(), networks=ss.MFNet(), solver='ode', stop=5, verbose=0).init()
    print('accepted')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])

print('== J. MixingPool with src under ode')
try:
    ss.Sim(sir(), networks=ss.MixingPool(n_contacts=10, src=lambda sim: sim.people.female), solver='ode', stop=5, verbose=0).init()
    print('accepted')
except Exception as e:
    print('ERR', type(e).__name__, str(e)[:300])
