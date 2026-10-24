import os; os.environ['MPLBACKEND']='agg'
import numpy as np, pandas as pd, sciris as sc, starsim as ss

def make(beta=0.3, dur=10, durname='dur', **kw):
    b = ss.par(beta, bounds=[0.05, 1.0], name='beta')
    d = ss.par(ss.days(dur), bounds=[3, 30], name=durname)
    S, I, R = ss.states(3)
    infection = S >> I @ b * I
    cases = ss.outcome(0.5 * infection, every=ss.weeks(1))
    sir = ss.Disease(infection, I >> R @ d, cases, init_prev=0.01, name='sir')
    return ss.Sim(sir, **sc.mergedicts(dict(total_pop=10_000, start='2020-01-01', stop='2020-05-01', unit='day', verbose=0), kw))

truth = make().run()
df = sc.dataframe(date=truth.t.timevec[7::7], cases=truth.results.sir.cases.values[7::7])
for durname in ['dur', 'recovery_time']:
    print(f'\n=== fitted par named "{durname}" ===')
    fit = make(0.15, 6, durname=durname, data=df).calibrate(method='nelder-mead')
    print('fit.pars:', dict(fit.pars))
    try:
        new = make(0.15, 6, durname=durname, pars=fit.pars, data=df).run()
        print('round trip OK; max diff n_I:', np.abs(new.results.sir.n_I.values - fit.sim.results.sir.n_I.values).max())
    except Exception as e:
        print('round trip FAILED:', type(e).__name__, str(e)[:120])
    try:
        new = fit.sim.new().run()  # does fit.sim.new() keep the fitted values?
        print('fit.sim.new() keeps fit:', np.abs(new.results.sir.n_I.values - fit.sim.results.sir.n_I.values).max() == 0)
    except Exception as e:
        print('fit.sim.new() FAILED:', type(e).__name__, str(e)[:120])

print('\n=== spec contents for fitted things ===')
spec = fit.sim.to_spec()
s = str(spec)
print('keys:', list(spec.keys()))
print('has "bounds":', 'bounds' in s, '| has "likelihood":', 'likelihood' in s, '| has "prior":', 'prior' in s)
print(sc.pp(spec['modules']['sir']['pars']) if 'modules' in spec else '')
print('\n=== ss.par with no value, only bounds ===')
p = ss.par(bounds=[1, 3]); print('value =', p.value)
