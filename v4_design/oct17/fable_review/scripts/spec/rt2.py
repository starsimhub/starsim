""" Spec shape, network dump size, hash stability, label/unit invariance, spellings """
import json, copy
import numpy as np, sciris as sc, starsim as ss
kw = dict(verbose=0)

beta = ss.par(0.05, bounds=[0.01, 0.5]); dur = ss.days(10)
S, I, R = ss.states(3)
infection = S >> I @ beta * I; recovery = I >> R @ dur
cases = ss.outcome(0.5 * infection, delay=ss.days(5), every=ss.weeks(1))
sir = ss.Disease(infection, recovery, cases, init_prev=0.01)
masks = ss.modify('sir.pars.beta', multiply=0.6, start='2020-03-15')
simkw = dict(stratify=dict(age=[0, 15, 65]), networks=ss.contacts('Kenya'), total_pop=50e6, start='2020-01-01', stop='2021-01-01')
sim = ss.Sim(sir, interventions=masks, **simkw, **kw); sim.init()
spec = sim.to_spec()
for k in ['modules', 'stratify', 'modifiers', 'interventions', 'config']:
    print(f'--- {k} ---'); print(json.dumps(spec[k], indent=1))
net = json.dumps(spec['networks'])
print('--- networks entry: chars', len(net), 'keys', list(spec['networks'][0].keys()))
print('   data field:', spec['networks'][0].get('data'))
print('   total spec chars', len(json.dumps(spec)), 'of which networks', len(net))
print('   python_class occurrences:', net.count('python_class'))

# Hash before and after run
h0 = sim.hash(); sim.run(); h1 = sim.hash()
print('--- hash before/after run equal:', h0 == h1); print(h0); print(h1)

# Label and doc do not change the hash; days(7) vs weeks(1)
def mk(label=None, doc=None, d=ss.days(7)):
    S, I, R = ss.states(3)
    S.label = label
    b = ss.par(0.05, doc=doc)
    sir = ss.Disease(S >> I @ b * I, I >> R @ d, name='sir')
    s = ss.Sim(sir, total_pop=1000, start='2020-01-01', stop='2020-07-01', **kw); s.init(); return s
a, b, c = mk(), mk(label='Susceptible', doc='from X'), mk(d=ss.weeks(1))
print('--- label/doc invariant:', a.hash() == b.hash(), '| days(7)==weeks(1):', a.hash() == c.hash())
print(a.hash().pars, c.hash().pars)

# Three spellings give the same spec?
s1 = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), name='sir')
S, I, R = ss.states(3)
s2 = ss.Disease(infection=S >> I @ 0.05 * I, recovery=I >> R @ 10, name='sir')
S, I, R = ss.states(3)
s3 = ss.Disease(infection=ss.Transition(S, I, rate=0.05 * I), recovery=ss.Transition(I, R, dur=10), name='sir')
specs = [ {k:v for k,v in ss.Sim(x, stop=100, total_pop=1000, **kw).to_spec().items() if k != 'meta'} for x in [s1, s2, s3]]
print('--- spellings equal:', specs[0] == specs[1], specs[1] == specs[2])
if specs[0] != specs[1]:
    for k in specs[0]:
        if specs[0][k] != specs[1][k]: print(k); print(json.dumps(specs[0][k])[:1500]); print(json.dumps(specs[1][k])[:1500])
print(json.dumps(specs[1]['modules'][0]['pars']))
print(json.dumps(specs[1]['modules'][0]['transitions']))
print('guessed:', specs[0]['modules'][0]['guessed'], specs[1]['modules'][0]['guessed'])
