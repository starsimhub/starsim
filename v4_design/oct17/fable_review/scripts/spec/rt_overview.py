""" Round trip of the overview model (minus data/calibration), to_code, and hash checks """
import json, sys, traceback
import numpy as np, sciris as sc, starsim as ss

kw = dict(verbose=1)
beta = ss.par(0.05, bounds=[0.01, 0.5])
dur  = ss.days(10)
S, I, R = ss.states(3)
infection = S >> I @ beta * I
recovery  = I >> R @ dur
cases     = ss.outcome(0.5 * infection, delay=ss.days(5), every=ss.weeks(1))
sir = ss.Disease(infection, recovery, cases, init_prev=0.01)
masks = ss.modify('sir.pars.beta', multiply=0.6, start='2020-03-15')
simkw = dict(stratify=dict(age=[0, 15, 65]), networks=ss.contacts('Kenya'), total_pop=50e6, start='2020-01-01', stop='2021-01-01')
sim = ss.Sim(sir, interventions=masks, **simkw, **kw)
print('--- init ---')
sim.init()
spec = sim.to_spec()
print('--- spec (no meta) ---')
print(json.dumps({k:v for k,v in spec.items() if k!='meta'}, indent=1)[:6000])
print('--- meta ---'); print(spec['meta'])
print('--- hash ---'); print(sim.hash())
print('--- from_spec ---')
try:
    sim2 = ss.from_spec(spec)
    spec2 = sim2.to_spec()
    a = {k:v for k,v in spec.items() if k!='meta'}; b = {k:v for k,v in spec2.items() if k!='meta'}
    print('round trip identical:', a == b)
    if a != b:
        for k in a:
            if a[k] != b[k]: print('  differs:', k); print('   A', json.dumps(a[k])[:800]); print('   B', json.dumps(b[k])[:800])
    print(sim2.hash())
except Exception as e:
    print('from_spec FAILED:', type(e).__name__, e)
print('--- without interventions ---')
sim_noiv = ss.Sim(sir, **simkw, **kw); sim_noiv.init()
try:
    s3 = ss.from_spec(sim_noiv.to_spec())
    print('hash equal:', s3.hash() == sim_noiv.hash(), s3.hash(), sim_noiv.hash())
except Exception as e:
    print('from_spec FAILED:', type(e).__name__, e)
print('--- to_code ---')
code = sim.to_code()
print(code)
sc.savetext('/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/spec/overview_code.py', code)
