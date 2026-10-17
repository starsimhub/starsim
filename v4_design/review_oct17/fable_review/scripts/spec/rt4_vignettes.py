""" Round trip of two vignette models: summer2 seir-age-clinical, memilio secir_ageres """
import json, runpy, traceback
import numpy as np, sciris as sc, starsim as ss
out = '/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/spec/'
for name, path in [('summer2', '/home/cliffk/idm/lingua_franca_design/vignettes/summer2/seir-age-clinical/model.py'),
                   ('memilio', '/home/cliffk/idm/lingua_franca_design/vignettes/memilio/secir_ageres/model.py')]:
    print(f'\n################ {name}')
    ns = runpy.run_path(path, run_name='x')
    sim = ns['sim']; sim.pars.verbose = 0
    sim.init()
    spec = sim.to_spec()
    print(json.dumps({k:v for k,v in spec.items() if k not in ['meta', 'networks']}, indent=None)[:5000])
    print('networks:', json.dumps(spec['networks'])[:600])
    print('hash:', dict(sim.hash()))
    try:
        sim2 = ss.from_spec(spec)
        spec2 = sim2.to_spec()
        a = {k:v for k,v in spec.items() if k!='meta'}; b = {k:v for k,v in spec2.items() if k!='meta'}
        print('round trip identical:', a == b, '| hash equal:', sim2.hash() == sim.hash())
        for k in a:
            if a[k] != b[k]:
                print('  differs:', k); print('   A', json.dumps(a[k])[:1500]); print('   B', json.dumps(b[k])[:1500])
    except Exception as e:
        print('from_spec FAILED:', type(e).__name__, e)
    try:
        code = sim.to_code()
        print('--- to_code ---'); print(code)
        sc.savetext(out + f'{name}_code.py', code)
    except Exception as e:
        print('to_code FAILED:', type(e).__name__, e); traceback.print_exc()
