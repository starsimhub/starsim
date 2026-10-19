""" Spelling equality, precision, functions in spec, schema check, misc """
import json, numpy as np, sciris as sc, starsim as ss
kw = dict(verbose=0, total_pop=1000, stop=100)
def spec_of(dis): return {k:v for k,v in ss.Sim(dis, **kw).to_spec().items() if k != 'meta'}
s1 = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), name='sir')
S, I, R = ss.states(3)
s3 = ss.Disease(infection=ss.Transition(S, I, rate=0.05 * I), recovery=ss.Transition(I, R, dur=10), name='sir')
S, I, R = ss.states(3)
s2 = ss.Disease(infection=S >> I @ 0.05 * I, recovery=I >> R @ ss.par(10, name='dur'), name='sir')  # hmm: bare 10 after @ is a rate
a, b, c = spec_of(s1), spec_of(s3), spec_of(s2)
print('arrow vs explicit identical:', a == b)
for k in a:
    if a[k] != b[k]:
        for m1, m2 in zip(a[k], b[k]):
            for kk in m1:
                if m1[kk] != m2[kk]: print('  ', kk, '|', json.dumps(m1[kk])[:300], '|', json.dumps(m2[kk])[:300])
print('operator (@ par(10)) : recovery dur =', json.dumps(c['modules'][0]['transitions'][1]['dur']), c['modules'][0]['pars'])
import starsim.sim
# Hash equality across spellings (guessed is skipped)
print('hash arrow == explicit:', ss.Sim(s1, **kw).hash() == ss.Sim(s3, **kw).hash())

print('\n--- precision ---')
S, I, R = ss.states(3)
d = ss.Disease(S >> I @ 0.1 * np.cos(np.pi * I.frac) * I, I >> R @ ss.days(1/3), name='x')
sp = ss.Sim(d, **kw).to_spec()
print('rate:', sp['modules'][0]['transitions'][0]['rate'])
print('dur par:', [p['value'] for p in sp['modules'][0]['pars']])
print('float(1/3) repr:', repr(1/3))

print('\n--- functions in spec ---')
def clearance(sim): return 0.2 + 0.0*sim.t.ti
S, I, R = ss.states(3)
d = ss.Disease(S >> I @ 0.1 * I, I >> R @ clearance, name='x')
sim = ss.Sim(d, **kw)
try:
    sim.init(); sp = sim.to_spec()
    print('functions:', json.dumps(sp['modules'][0]['functions'])[:300]); print('rate/dur of recovery:', sp['modules'][0]['transitions'][1]['rate'], sp['modules'][0]['transitions'][1]['dur'])
    print('pars:', json.dumps(sp['modules'][0]['pars'])[:300])
    try: print(ss.to_code(sp)[:200])
    except Exception as e: print('to_code with function:', type(e).__name__, e)
    try: ss.from_spec(sp)
    except Exception as e: print('from_spec with function:', type(e).__name__, e)
except Exception as e:
    import traceback; traceback.print_exc()

print('\n--- schema version check (E1401) ---')
sp = ss.Sim(s1, **kw).to_spec(); sp['schema'] = 'lingua-franca/2.0'
try: ss.from_spec(sp); print('loaded a 2.0 spec without complaint')
except Exception as e: print(type(e).__name__, e)

print('\n--- sir.to_json() vs sim.to_json() ---')
j = s1.to_json(); print('Disease.to_json keys:', list(j.keys()) if isinstance(j, dict) else type(j))
sj = ss.Sim(s1, **kw).to_json(); print('Sim.to_json keys:', list(sj.keys()))

print('\n--- to_code for a sim with a matrix contacts network and a modify in connectors ---')
S, I, R = ss.states(3)
d = ss.Disease(S >> I @ ss.par(0.05, name='beta') * I, I >> R @ ss.days(10), name='sir')
d.stratify(age=[0, 20])
sim = ss.Sim(d, networks=ss.contacts(age=np.array([[10, 3], [3, 5]])), connectors=ss.modify('sir.pars.beta', multiply=0.5), total_pop=1000, start='2020-01-01', stop='2020-07-01', verbose=0)
sim.init()
print(sim.to_code())
print('modifiers in spec:', json.dumps(sim.to_spec()['modifiers']))

print('\n--- an outcome with a likelihood, par with prior ---')
S, I, R = ss.states(3)
inf = S >> I @ ss.par(0.05, bounds=[0.01, 0.1], prior=ss.lognorm_ex(0.05, 0.02)) * I
d = ss.Disease(inf, I >> R @ ss.days(10), cases=ss.outcome(0.5*inf, likelihood=ss.poisson()), name='sir')
sim = ss.Sim(d, **kw); sim.init(); sp = sim.to_spec()
print('prior:', sp['modules'][0]['pars'][0]['prior'], '| likelihood:', sp['modules'][0]['results'][0]['likelihood'])
print(sim.to_code().split('# Not written')[1] if '# Not written' in sim.to_code() else 'no "not written" footer')
