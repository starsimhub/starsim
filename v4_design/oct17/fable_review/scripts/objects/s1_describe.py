"""describe() on the overview example (without named data) and on vignettes"""
import os, sys
os.environ['MPLBACKEND'] = 'agg'
import starsim as ss

print('=== overview example (operator form, no Kenya data) ===')
beta = ss.par(0.05, bounds=[0.01, 0.5])
dur  = ss.days(10)
S, I, R = ss.states(3)
infection = S >> I @ beta * I
recovery  = I >> R @ dur
cases     = ss.outcome(0.5 * infection, delay=ss.days(5), every=ss.weeks(1))
sir = ss.Disease(infection, recovery, cases, init_prev=0.01)
sim = ss.Sim(sir, stratify=dict(age=[0, 15, 65]), total_pop=50e3, start='2020-01-01', stop='2020-07-01')
sim.describe()
print()
print('sir.pars:', sir.pars)
print('types:', {k: type(v).__name__ for k, v in sir.pars.items()})

print()
print('=== short form, stop=100 (spec 01 example) ===')
sir2 = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
ss.Sim(sir2, stop=100).describe()
print('sir2.pars:', sir2.pars)

for path in ['memilio/secir', 'summer2/seir-age-clinical', 'atomica/sir']:
    print()
    print(f'=== vignette {path} ===')
    sys.path.insert(0, f'/home/cliffk/idm/lingua_franca_design/vignettes/{path}')
    import importlib
    if 'model' in sys.modules: del sys.modules['model']
    m = importlib.import_module('model')
    m.sim.describe()
    sys.path.pop(0)
