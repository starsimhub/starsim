import runpy, sys, traceback
import numpy as np, starsim as ss
out = '/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/spec/'
for name, path in [('summer2', '/home/cliffk/idm/lingua_franca_design/vignettes/summer2/seir-age-clinical/model.py'),
                   ('memilio', '/home/cliffk/idm/lingua_franca_design/vignettes/memilio/secir_ageres/model.py')]:
    print(f'\n######## {name}')
    try:
        orig = runpy.run_path(path, run_name='x')['sim']; orig.pars.verbose = 0
        gen = runpy.run_path(out + f'{name}_code.py', run_name='x')['sim']; gen.pars.verbose = 0
        orig.init(); gen.init()
        print('hash equal:', orig.hash() == gen.hash()); print(dict(orig.hash())); print(dict(gen.hash()))
        if orig.hash() != gen.hash():
            ss.diff(orig, gen)
        orig.run(); gen.run()
        for res in ['n_I', 'n_R']:
            try:
                a = orig.results[orig.diseases()[0].name][res].values; b = gen.results[gen.diseases()[0].name][res].values
                print(res, 'max abs diff:', np.max(np.abs(a - b)))
            except Exception as e:
                print(res, 'compare failed:', e)
    except Exception as e:
        print('FAILED:', type(e).__name__, e); traceback.print_exc()
