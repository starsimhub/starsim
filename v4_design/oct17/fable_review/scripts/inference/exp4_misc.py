import os; os.environ['MPLBACKEND']='agg'
import numpy as np, pandas as pd, sciris as sc, starsim as ss
b = ss.par(0.3, bounds=[0.05, 1.0], name='beta')
S, I, R = ss.states(3)
infection = S >> I @ b * I
cases = ss.outcome(0.5 * infection, every=ss.weeks(1))
sir = ss.Disease(infection, I >> R @ ss.days(10), cases, init_prev=0.01, name='sir')
kw = dict(total_pop=10_000, start='2020-01-01', stop='2020-05-01', unit='day', verbose=0)
truth = ss.Sim(sir, **kw).run()
df = sc.dataframe(date=truth.t.timevec[7::7], cases=truth.results.sir.cases.values[7::7])
print('=== CRN pairing in optuna: is the enqueued x0 trial equal to mismatch_before? ===')
fit = ss.Sim(sir, data=df, solver='sde', **sc.mergedicts(kw, dict(pars=dict(beta=0.15)))).calibrate(trials=5, verbose=False)
print('mismatch_before', fit.mismatch_before, '| trials:'); print(fit.trials.head(4).to_string())
print('\n=== ss.Calibration (Starsim 3) without build_fn on a declared sim: error text ===')
try:
    ss.Calibration(ss.Sim(sir, **kw), calib_pars=dict(beta=dict(low=0.05, high=1, guess=0.15)), total_trials=2, n_workers=1, die=True).calibrate()
except Exception as e:
    print(type(e).__name__, str(e)[:200])
print('\n=== what does summarize print for optuna (convergence?) ===')
fit.summarize()
print('\n=== sim.sensitivity exists?', hasattr(ss.Sim, 'sensitivity'))
print('=== time_cols accepted:', ss.data.time_cols)
print('=== data= as a path to csv works?')
df.to_csv('/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/inference/cases.csv', index=False)
sim = ss.Sim(sir, data='/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/inference/cases.csv', **kw).init()
print('data from path:', type(sim.data), sim.data.shape, 'index type', type(sim.data.index[0]))
print('=== the 1174-line Starsim 3 part: is CalibComponent usable as gof= for a declared sim? ===')
comp = ss.Normal(name='cases', conform='incident', expected=df.rename(columns=dict(date='t1', cases='x')).assign(t=lambda d: pd.to_datetime(d.t1)-pd.Timedelta(days=7)).set_index('t').drop(columns='t1').assign(t1=lambda d: d.index + pd.Timedelta(days=7)), extract_fn=lambda sim: pd.DataFrame({'x': sim.results.sir.cases.values}, index=pd.Index(pd.to_datetime([str(t) for t in sim.results.timevec]), name='t')), sigma2=100)
try:
    f = ss.Sim(sir, data=df, **sc.mergedicts(kw, dict(pars=dict(beta=0.15)))).calibrate(gof=comp, method='nelder-mead')
    print('CalibComponent as gof works; beta =', f.pars.beta)
except Exception as e:
    print('failed:', type(e).__name__, str(e)[:200])
