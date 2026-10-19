import re
import warnings, numpy as np, sciris as sc, starsim as ss
out = sc.path('.')
S, I, R, D = ss.states('S I R D')
sir = ss.Disease(
    infection = S >> I @ ss.par(ss.perday(0.3), name='beta') * I,
    recovery  = ss.Transition(I, [R, D], dur=ss.par(ss.lognorm_ex(mean=ss.days(10), std=ss.days(3)), name='dur'), p=[0.95, None]),
    births    = ss.birth('-> S', rate=ss.peryear(0.02)),
    cases     = ss.outcome('infection', delay=ss.days(3)),
    init      = dict(I=10), name='sir')
sir.stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2])
sim = ss.Sim(sir, total_pop=10000, stop=ss.days(120), unit='day', networks=ss.contacts(age=[[3,1,0.5],[1,2,0.5],[0.5,0.5,1]]), verbose=0)
for fmt, ext in [('mermaid', '.md'), ('sbml', '.xml'), ('odin', '.R')]:
    print(f'\n===== export {fmt}')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        text = ss.save(out/f'seird{ext}', sim, format=fmt)
    for x in w: print(str(x.message))
    txt = sc.loadtext(out/f'seird{ext}')
    print(f'--- {len(txt.splitlines())} lines, {len(txt)} chars')
    if fmt != 'sbml': print(txt[:3000])
# Round trip
print('\n===== round trip odin')
back = ss.load(out/'seird.R', unit='day', stop=120, verbose=0)
sim.run(); back.run()
a = sim.results.sir.n_I.values; rb = back.results[back.diseases[0].name]; b = sum(rb[k].values for k in rb.keys() if re.match(r'n_I(_\d+)?$', k))
print('n_I max |diff| (stratified+network original vs flat reimport):', np.abs(a-b).max(), 'peak', a.max())
print('\n===== round trip sbml')
back2 = ss.load(out/'seird.xml', stop=120, verbose=0); back2.run()
rc = back2.results[back2.diseases[0].name]; c = sum(rc[k].values for k in rc.keys() if re.match(r'n_I(_\d+)?$', k))
print('n_I max |diff| odin vs sbml reimport:', np.abs(b-c).max())
# A homogeneous unstratified version: should round-trip exactly
print('\n===== exact round trip, no network/strat')
sir2 = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', rate=ss.perday(0.1)), init=dict(I=10), name='sir')
s2 = ss.Sim(sir2, total_pop=1000, stop=ss.days(100), unit='day', verbose=0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    ss.save(out/'sir.R', s2, format='odin'); ss.save(out/'sir.xml', s2, format='sbml')
for x in w: print('WARN', str(x.message)[:300])
print(sc.loadtext(out/'sir.R'))
b2 = ss.load(out/'sir.R', stop=100, verbose=0); s2.run(); b2.run()
print('exact round trip diff:', np.abs(s2.results.sir.n_I.values - b2.results[b2.diseases[0].name].n_I.values).max())
print(sim.diseases.sir.diagram() if hasattr(sim.diseases.sir, 'diagram') else 'no diagram()')
