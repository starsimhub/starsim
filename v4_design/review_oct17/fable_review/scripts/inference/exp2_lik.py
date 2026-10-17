import os; os.environ['MPLBACKEND']='agg'
import numpy as np, pandas as pd, sciris as sc, starsim as ss, sys

def make(beta=0.3, dur=10, lik=True, prior=False, **kw):
    b = ss.par(beta, bounds=[0.05, 1.0], name='beta', prior=ss.lognorm_ex(mean=0.3, std=0.2) if prior else None)
    d = ss.par(ss.days(dur), bounds=[3, 30], name='dur')
    S, I, R = ss.states(3)
    infection = S >> I @ b * I
    recovery = I >> R @ d
    cases = ss.outcome(0.5 * infection, delay=ss.days(3), every=ss.weeks(1), likelihood=ss.poisson() if lik else None)
    sir = ss.Disease(infection, recovery, cases, init_prev=0.01, name='sir')
    args = dict(total_pop=10_000, start='2020-01-01', stop='2020-05-01', unit='day', verbose=0)
    return ss.Sim(sir, **sc.mergedicts(args, kw))

truth = make().run()
res = truth.results.sir
rng = np.random.default_rng(1)
df = sc.dataframe(date=truth.t.timevec[7::7], cases=rng.poisson(res.cases.values[7::7]))
section = sys.argv[1] if len(sys.argv) > 1 else 'nuts'

if section == 'nuts':
    print('=== ODE + likelihood: default (NUTS) ===')
    T = sc.timer()
    fit = make(0.15, 6, data=df).calibrate(verbose=True, draws=200, warmup=200, chains=2, rand_seed=1)
    T.toc('NUTS calibrate')
    fit.summarize()
    print('pars:', fit.pars, '| dur type:', type(fit.pars.dur))
    new = make(0.15, 6, pars=fit.pars, data=df).run()
    print('round trip max diff n_I:', np.abs(new.results.sir.n_I.values - fit.sim.results.sir.n_I.values).max())
    print('spec has likelihood?', 'likelihood' in str(fit.sim.to_spec()), '| spec has bounds?', 'bounds' in str(fit.sim.to_spec()), '| prior?', 'prior' in str(fit.sim.to_spec()))
    print(sc.pp(fit.sim.to_spec()['modules']['sir']['pars'] if 'modules' in fit.sim.to_spec() else list(fit.sim.to_spec().keys())))

elif section == 'prior':
    print('=== prior used? (point estimate ignores it) ===')
    f1 = make(0.15, 6, data=df, prior=False).calibrate(method='l-bfgs-b', verbose=True)
    f2 = make(0.15, 6, data=df, prior=True).calibrate(method='l-bfgs-b', verbose=True)
    print('no prior:', f1.pars, '\nwith prior:', f2.pars, '(same => prior ignored by l-bfgs-b)')
    print('=== convergence surfaced? maxiter=1 ===')
    f3 = make(0.15, 6, data=df, lik=False).calibrate(method='l-bfgs-b', maxiter=1, verbose=True)
    f3.summarize()
    print('attrs:', [k for k in vars(f3)])
    print('=== par inside a distribution ===')
    try:
        d = ss.par(10, bounds=[3, 30], name='dur')
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3 * I, I >> R @ ss.gamma(a=3, scale=d), init_prev=0.01, name='sir')
        ss.Sim(sir, total_pop=10_000, unit='day', stop=100, data=df.rename(columns=dict(cases='infection')), verbose=0).calibrate()
    except Exception as e:
        print('error:', type(e).__name__, str(e)[:300])
    print('=== a bare number in bounds? (bounds without ss.par) ===')
    print('=== NAE with zeros in data ===')
    df0 = df.copy(); df0['cases'] = 0
    sim = make(0.3, data=df0).run()
    print('all-zero data: mismatch =', ss.mismatch(sim), ' (scale falls back to range then 1.0)')
    print('=== summarize table of a nelder-mead fit with likelihood: measure label ===')
    f4 = make(0.15, 6, data=df).calibrate(method='nelder-mead', verbose=True)
    print(f4.measure, f4.mismatch)
